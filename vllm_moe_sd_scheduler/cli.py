"""
MoE-SD Runtime CLI

This CLI launches vLLM with MoE-SD controller integration.

Usage:
    python -m vllm_moe_sd_scheduler.cli --config-json '{"model": "...", ...}'
"""

import argparse
import json
import logging
import sys
from typing import Any, Dict

from .config import SchedulerConfig
from .entrypoints import build_runtime
from .feature_flags import FeatureFlags

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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


def launch_server(config: SchedulerConfig) -> None:
    """Launch vLLM server with MoE-SD controller."""
    logger.info(f"Launching server with config: {config}")
    
    # Load controller
    controller = load_controller(config.policy_name)
    logger.info(f"Loaded controller: {type(controller).__name__}")
    
    # Build runtime
    runtime = build_runtime(
        flags=config.feature_flags,
        controller=controller,
    )
    
    logger.info(f"Runtime mode: {runtime.mode}")
    logger.info(f"Runtime reason: {runtime.reason}")
    
    # Print runtime info as JSON
    output = {
        "mode": runtime.mode,
        "reason": runtime.reason,
        "controller": type(controller).__name__,
        "policy": config.policy_name,
        "model": config.model,
        "workload": config.workload_profile,
    }
    
    print(json.dumps(output, indent=2))
    
    # TODO: Actually launch vLLM server here
    # This requires importing vllm and starting the server with the runtime hooks
    logger.info("Server launch placeholder - integrate with vllm serve command")


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
