from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class FallbackConfig:
    force_mode: str = ""  # allowed: native_eagle3, no_sd, observe_only


class FallbackManager:
    """Handles safe fallback and hot-switch decisions."""

    def __init__(self, config: FallbackConfig | None = None):
        self.cfg = config or FallbackConfig()

    def resolve_mode(self, has_error: bool, requested_mode: str) -> Dict[str, Any]:
        if self.cfg.force_mode:
            return {
                "mode": self.cfg.force_mode,
                "fallback_applied": True,
                "reason": f"force_mode:{self.cfg.force_mode}",
            }

        if not has_error:
            return {"mode": requested_mode, "fallback_applied": False, "reason": "none"}

        if requested_mode == "controller":
            return {
                "mode": "native_eagle3",
                "fallback_applied": True,
                "reason": "controller_error",
            }

        if requested_mode == "native_eagle3":
            return {
                "mode": "no_sd",
                "fallback_applied": True,
                "reason": "native_eagle3_error",
            }

        return {
            "mode": "observe_only",
            "fallback_applied": True,
            "reason": "unknown_mode_error",
        }

    def trace(self, request_id: str, requested_mode: str, resolved: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "request_id": request_id,
            "requested_mode": requested_mode,
            "resolved_mode": resolved.get("mode"),
            "fallback_applied": bool(resolved.get("fallback_applied")),
            "reason": resolved.get("reason", ""),
        }
