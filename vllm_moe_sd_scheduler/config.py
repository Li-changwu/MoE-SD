from dataclasses import dataclass, field

from .feature_flags import FeatureFlags


@dataclass
class SchedulerConfig:
    model: str
    workload_profile: str
    policy_name: str = "native"
    feature_flags: FeatureFlags = field(default_factory=FeatureFlags)

    @staticmethod
    def from_dict(payload: dict) -> "SchedulerConfig":
        return SchedulerConfig(
            model=str(payload["model"]),
            workload_profile=str(payload["workload_profile"]),
            policy_name=str(payload.get("policy_name", "native")),
            feature_flags=FeatureFlags.from_dict(payload.get("feature_flags")),
        )
