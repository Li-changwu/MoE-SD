from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureFlags:
    """Feature switches for controlled rollout.

    All flags default to False so that installation never changes native behavior.
    """

    enable_controller: bool = False
    enable_prefetch: bool = False
    enable_memory_partition: bool = False
    observation_only: bool = True

    @staticmethod
    def from_dict(payload: dict | None) -> "FeatureFlags":
        payload = payload or {}
        return FeatureFlags(
            enable_controller=bool(payload.get("enable_controller", False)),
            enable_prefetch=bool(payload.get("enable_prefetch", False)),
            enable_memory_partition=bool(payload.get("enable_memory_partition", False)),
            observation_only=bool(payload.get("observation_only", True)),
        )
