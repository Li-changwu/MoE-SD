"""
MoE-SD Runtime CLI

This CLI launches vLLM with MoE-SD controller integration.

Usage:
    python -m vllm_moe_sd_scheduler.cli --config-json '{"model": "...", ...}'
    python -m vllm_moe_sd_scheduler.cli --config-json '...' --launch-server
"""

import argparse
import json
import logging
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional

from .config import SchedulerConfig
from .entrypoints import build_runtime
from .feature_flags import FeatureFlags

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default vLLM serve parameters
# ---------------------------------------------------------------------------
_VLLM_DEFAULTS: Dict[str, Any] = {
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.90,
    "max_model_len": 4096,
    "port": 8000,
    "host": "0.0.0.0",
    "speculative_model": None,
    "num_speculative_tokens": 3,
}


def load_controller(policy_name: str) -> Any:
    """Load controller by policy name."""
    try:
        if policy_name == "static":
            from controllers.static_governor import StaticGovernor
            return StaticGovernor()
        elif policy_name == "phase_aware":
            from controllers.phase_aware_governor import PhaseAwareGovernor
            return PhaseAwareGovernor()
        elif policy_name == "noop":
            from controllers.interface import NoOpController
            return NoOpController()
        else:
            logger.warning(f"Unknown policy '{policy_name}', using NoOpController")
            from controllers.interface import NoOpController
            return NoOpController()
    except ImportError as e:
        logger.error(f"Failed to load controller '{policy_name}': {e}")
        from controllers.interface import NoOpController
        return NoOpController()


def _build_vllm_serve_cmd(config: SchedulerConfig) -> List[str]:
    """
    Build the ``vllm serve`` command line from SchedulerConfig.

    Extra vLLM arguments can be passed in ``config.vllm_args`` (a dict).
    """
    vllm_args: Dict[str, Any] = {}
    vllm_args.update(_VLLM_DEFAULTS)
    # Merge user-provided overrides
    extra = getattr(config, "vllm_args", None) or {}
    vllm_args.update(extra)

    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
    cmd += ["--model", config.model]

    # Speculative decoding
    spec_model = vllm_args.pop("speculative_model", None)
    if spec_model:
        cmd += ["--speculative-model", str(spec_model)]
        cmd += [
            "--num-speculative-tokens",
            str(vllm_args.pop("num_speculative_tokens", 3)),
        ]
    else:
        vllm_args.pop("num_speculative_tokens", None)

    # Standard args
    for key in ("tensor_parallel_size", "gpu_memory_utilization", "max_model_len", "port", "host"):
        val = vllm_args.pop(key, None)
        if val is not None:
            cli_key = "--" + key.replace("_", "-")
            cmd += [cli_key, str(val)]

    # Pass remaining keys as --key value
    for key, val in vllm_args.items():
        if val is None or val is False:
            continue
        cli_key = "--" + key.replace("_", "-")
        if val is True:
            cmd.append(cli_key)
        else:
            cmd += [cli_key, str(val)]

    return cmd


def _start_sidecar(port: int, controller_name: str, default_k: int = 3) -> Optional[subprocess.Popen]:
    """
    Start the metrics-sidecar process in the background.

    Returns the Popen object or None if sidecar script is not available.
    """
    sidecar_script = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "adapters", "metrics_sidecar.py"
    )
    if not os.path.isfile(sidecar_script):
        logger.warning("Metrics sidecar script not found at %s", sidecar_script)
        return None

    cmd = [
        sys.executable,
        sidecar_script,
        "--port", str(port),
        "--controller", controller_name,
        "--default-k", str(default_k),
    ]
    logger.info("Starting metrics sidecar: %s", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc


def launch_server(config: SchedulerConfig) -> None:
    """
    Launch vLLM server with MoE-SD controller.

    1. Build the ``vllm serve`` command.
    2. Optionally start the metrics sidecar.
    3. Spawn the vLLM server process and wait for it.
    4. On SIGINT/SIGTERM, gracefully shut down both.
    """
    # Load controller and build runtime for pre-flight checks
    controller = load_controller(config.policy_name)
    logger.info("Loaded controller: %s", type(controller).__name__)

    runtime = build_runtime(
        flags=config.feature_flags,
        controller=controller,
    )

    logger.info("Runtime mode: %s", runtime.mode)
    logger.info("Runtime reason: %s", runtime.reason)

    # --- Build vLLM command ---
    cmd = _build_vllm_serve_cmd(config)
    logger.info("vLLM command: %s", " ".join(shlex.quote(c) for c in cmd))

    # --- Environment variables for MoE-SD ---
    env = os.environ.copy()
    env["MOE_SD_POLICY"] = config.policy_name
    env["MOE_SD_MODE"] = runtime.mode
    if config.feature_flags.observation_only:
        env["MOE_SD_OBSERVE_ONLY"] = "1"

    # --- Print config summary ---
    summary = {
        "mode": runtime.mode,
        "reason": runtime.reason,
        "controller": type(controller).__name__,
        "policy": config.policy_name,
        "model": config.model,
        "workload": config.workload_profile,
        "command": " ".join(shlex.quote(c) for c in cmd),
    }
    print(json.dumps(summary, indent=2))

    # --- Start vLLM server ---
    vllm_proc: Optional[subprocess.Popen] = None
    sidecar_proc: Optional[subprocess.Popen] = None

    def _shutdown(signum, frame):
        logger.info("Received signal %d, shutting down...", signum)
        if sidecar_proc and sidecar_proc.poll() is None:
            sidecar_proc.terminate()
        if vllm_proc and vllm_proc.poll() is None:
            vllm_proc.terminate()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        vllm_proc = subprocess.Popen(cmd, env=env)
        logger.info("vLLM server started (PID %d)", vllm_proc.pid)

        # Give vLLM a moment to initialize before starting sidecar
        if config.feature_flags.enable_controller:
            port = _VLLM_DEFAULTS["port"]
            extra = getattr(config, "vllm_args", None) or {}
            port = extra.get("port", port)
            default_k = extra.get("num_speculative_tokens", 3)

            # Wait for server to be ready (poll health endpoint)
            _wait_for_server(port, timeout=300)

            sidecar_proc = _start_sidecar(
                port=port,
                controller_name=config.policy_name,
                default_k=default_k,
            )
            if sidecar_proc:
                logger.info("Sidecar started (PID %d)", sidecar_proc.pid)

        # Wait for vLLM to exit
        rc = vllm_proc.wait()
        logger.info("vLLM server exited with code %d", rc)
        sys.exit(rc)

    finally:
        if sidecar_proc and sidecar_proc.poll() is None:
            sidecar_proc.terminate()
            sidecar_proc.wait(timeout=5)
        if vllm_proc and vllm_proc.poll() is None:
            vllm_proc.terminate()
            vllm_proc.wait(timeout=10)


def _wait_for_server(port: int, timeout: float = 300) -> bool:
    """
    Poll the vLLM health endpoint until it responds or timeout elapses.
    """
    import urllib.request

    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    logger.info("Waiting for vLLM server at %s ...", url)

    while time.time() < deadline:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    logger.info("vLLM server is ready")
                    return True
        except Exception:
            pass
        time.sleep(2)

    logger.warning("vLLM server did not become ready within %ds", timeout)
    return False


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="MoE-SD Runtime CLI")
    parser.add_argument(
        "--config-json",
        required=True,
        help="Scheduler config as JSON string"
    )
    parser.add_argument(
        "--launch-server",
        action="store_true",
        help="Actually launch vLLM server (not just print config)"
    )
    
    args = parser.parse_args()

    try:
        payload = json.loads(args.config_json)
        cfg = SchedulerConfig.from_dict(payload)
        
        if args.launch_server:
            launch_server(cfg)
        else:
            # Just build runtime and print info
            controller = load_controller(cfg.policy_name)
            runtime = build_runtime(cfg.feature_flags, controller=controller)
            
            output = {
                "mode": runtime.mode,
                "reason": runtime.reason,
                "controller": type(controller).__name__,
                "policy": cfg.policy_name,
            }
            print(json.dumps(output, indent=2))
            
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON config: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Runtime failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
