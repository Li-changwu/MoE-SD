#!/usr/bin/env python3
"""
MoE-SD Metrics Sidecar Controller
===================================
Polls vLLM's /metrics endpoint for KV cache usage and acceptance rate,
feeds RuntimeState to the StaticGovernor controller, and writes the
decided K to /dev/shm/moe_sd_k for the patched eagle.py proposer to read.

Usage:
    python metrics_sidecar.py [--port 8000] [--interval 0.5] [--controller static]

The sidecar runs until killed (SIGINT/SIGTERM) and logs decisions to stderr.
It also writes a JSON trace to /dev/shm/moe_sd_trace.jsonl for analysis.
"""

import argparse
import json
import logging
import os
import re
import signal
import sys
import time
import urllib.request
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from controllers.interface import Phase, RequestState, RuntimeState
from controllers.static_governor import StaticGovernor, StaticGovernorConfig

logging.basicConfig(
    level=logging.INFO,
    format="[sidecar %(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("moe_sd_sidecar")

# Shared memory paths
K_FILE = "/dev/shm/moe_sd_k"
TRACE_FILE = "/dev/shm/moe_sd_trace.jsonl"

# Prometheus metric patterns
RE_KV_USAGE = re.compile(
    r'^vllm:kv_cache_usage_perc\{[^}]*\}\s+([\d.eE+-]+)', re.MULTILINE
)
RE_NUM_REQUESTS = re.compile(
    r'^vllm:num_requests_running\{[^}]*\}\s+([\d.eE+-]+)', re.MULTILINE
)
RE_SPEC_ACCEPTANCE = re.compile(
    r'^vllm:spec_decode_draft_acceptance_rate\{[^}]*\}\s+([\d.eE+-]+)',
    re.MULTILINE,
)
# Fallback: extract from log-style metrics
RE_SPEC_ACCEPTED = re.compile(
    r'^vllm:spec_decode_num_accepted_tokens_total\{[^}]*\}\s+([\d.eE+-]+)',
    re.MULTILINE,
)
RE_SPEC_DRAFTED = re.compile(
    r'^vllm:spec_decode_num_draft_tokens_total\{[^}]*\}\s+([\d.eE+-]+)',
    re.MULTILINE,
)


class MetricsSidecar:
    def __init__(self, port: int, interval: float, default_k: int = 3):
        self.url = f"http://localhost:{port}/metrics"
        self.interval = interval
        self.default_k = default_k
        self.running = True
        self.current_k = default_k
        self.step = 0
        self.trace_fd = None

        # State tracking for acceptance rate (cumulative counters)
        self._prev_accepted = 0.0
        self._prev_drafted = 0.0
        self._acceptance_rate = 0.5  # Start with neutral estimate

        # Initialize controller
        config = StaticGovernorConfig(default_k=default_k)
        self.controller = StaticGovernor(config)

        # Set up signal handling
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _shutdown(self, signum, frame):
        logger.info("Shutting down sidecar (signal %d)", signum)
        self.running = False

    def _fetch_metrics(self):
        """Fetch prometheus metrics from vLLM."""
        try:
            req = urllib.request.Request(self.url, method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                return resp.read().decode("utf-8")
        except Exception:
            return None

    def _parse_metrics(self, text):
        """Extract KV usage and acceptance rate from prometheus text."""
        kv_usage = 0.0
        num_requests = 0
        acceptance_rate = self._acceptance_rate

        m = RE_KV_USAGE.search(text)
        if m:
            kv_usage = float(m.group(1))

        m = RE_NUM_REQUESTS.search(text)
        if m:
            num_requests = int(float(m.group(1)))

        # Try direct acceptance rate metric first
        m = RE_SPEC_ACCEPTANCE.search(text)
        if m:
            acceptance_rate = float(m.group(1))
        else:
            # Compute from cumulative counters
            accepted = 0.0
            drafted = 0.0
            m = RE_SPEC_ACCEPTED.search(text)
            if m:
                accepted = float(m.group(1))
            m = RE_SPEC_DRAFTED.search(text)
            if m:
                drafted = float(m.group(1))

            delta_accepted = accepted - self._prev_accepted
            delta_drafted = drafted - self._prev_drafted
            if delta_drafted > 0:
                acceptance_rate = delta_accepted / delta_drafted
            self._prev_accepted = accepted
            self._prev_drafted = drafted

        # Adjust acceptance rate for padding: when effective K < original K,
        # padded tokens inflate rejection rate. Scale to compensate.
        if self.current_k < self.default_k and self.current_k > 0:
            acceptance_rate = min(1.0, acceptance_rate * self.default_k / self.current_k)


        self._acceptance_rate = acceptance_rate
        return kv_usage, num_requests, acceptance_rate

    def _build_runtime_state(self, kv_usage, num_requests, acceptance_rate):
        """Build a RuntimeState from scraped metrics."""
        # Map KV usage (0.0-1.0) to GPU memory pressure
        # KV cache is the primary memory consumer in our setup
        gpu_total = 48000.0  # A6000 48GB
        # Estimate GPU used from KV usage as a proxy for memory pressure
        # In our setup, KV cache is the main variable consumer
        gpu_used = gpu_total * (0.6 + 0.4 * kv_usage)  # Base 60% (model) + 40% variable

        request = RequestState(
            request_id="aggregate",
            prompt_len=0,
            output_len=0,
            request_rate=float(num_requests),
            phase=Phase.DECODE if num_requests > 0 else Phase.UNKNOWN,
        )
        return RuntimeState(
            request=request,
            step_id=self.step,
            gpu_mem_used_mb=gpu_used,
            gpu_mem_total_mb=gpu_total,
            kv_cache_mb=kv_usage * gpu_total * 0.4,  # KV portion estimate
            acceptance_rate=acceptance_rate,
        )

    def _write_k(self, k: int):
        """Atomically write K to shared memory."""
        tmp = K_FILE + ".tmp"
        with open(tmp, "w") as f:
            f.write(str(k))
        os.replace(tmp, K_FILE)  # Atomic on Linux

    def _log_trace(self, kv_usage, num_requests, acceptance_rate, k, reason):
        """Append decision to trace file."""
        entry = {
            "t": time.time(),
            "step": self.step,
            "kv_usage": round(kv_usage, 4),
            "num_requests": num_requests,
            "acceptance_rate": round(acceptance_rate, 4),
            "k": k,
            "reason": reason,
        }
        if self.trace_fd:
            self.trace_fd.write(json.dumps(entry) + "\n")
            self.trace_fd.flush()

    def run(self):
        """Main control loop."""
        logger.info("Starting MoE-SD sidecar controller")
        logger.info("  Metrics URL: %s", self.url)
        logger.info("  Poll interval: %.1fs", self.interval)
        logger.info("  Default K: %d", self.default_k)
        logger.info("  K file: %s", K_FILE)

        # Write initial K
        self._write_k(self.default_k)

        # Open trace file
        self.trace_fd = open(TRACE_FILE, "w")

        # Wait for vLLM to be ready
        logger.info("Waiting for vLLM metrics endpoint...")
        while self.running:
            metrics = self._fetch_metrics()
            if metrics is not None:
                logger.info("vLLM metrics endpoint is ready")
                break
            time.sleep(1.0)

        # Main control loop
        while self.running:
            try:
                metrics_text = self._fetch_metrics()
                if metrics_text is None:
                    time.sleep(self.interval)
                    continue

                kv_usage, num_requests, acceptance_rate = self._parse_metrics(
                    metrics_text
                )

                state = self._build_runtime_state(
                    kv_usage, num_requests, acceptance_rate
                )
                decision = self.controller.decide_speculation_k(state)

                new_k = decision.get("k", self.default_k)
                reason = decision.get("reason", "unknown")

                if new_k != self.current_k:
                    logger.info(
                        "K: %d -> %d  (kv=%.1f%% acc=%.1f%% reqs=%d reason=%s)",
                        self.current_k,
                        new_k,
                        kv_usage * 100,
                        acceptance_rate * 100,
                        num_requests,
                        reason,
                    )
                    self._write_k(new_k)
                    self.current_k = new_k

                self._log_trace(
                    kv_usage, num_requests, acceptance_rate, new_k, reason
                )
                self.step += 1

            except Exception as e:
                logger.error("Sidecar error: %s", e)

            time.sleep(self.interval)

        # Cleanup
        logger.info("Sidecar stopped. Final K=%d, steps=%d", self.current_k, self.step)
        if self.trace_fd:
            self.trace_fd.close()
        # Remove K file so next vLLM run uses default
        try:
            os.unlink(K_FILE)
        except FileNotFoundError:
            pass


def main():
    parser = argparse.ArgumentParser(description="MoE-SD Metrics Sidecar")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--interval", type=float, default=0.5,
                        help="Polling interval in seconds")
    parser.add_argument("--default-k", type=int, default=3,
                        help="Default speculation depth (K)")
    args = parser.parse_args()

    sidecar = MetricsSidecar(
        port=args.port,
        interval=args.interval,
        default_k=args.default_k,
    )
    sidecar.run()


if __name__ == "__main__":
    main()
