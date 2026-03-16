from dataclasses import dataclass

from .feature_flags import FeatureFlags


@dataclass
class RuntimeBinding:
    mode: str
    reason: str


def build_runtime(flags: FeatureFlags) -> RuntimeBinding:
    """Build runtime binding based on feature flags.

    This function is intentionally conservative: when controller is disabled,
    runtime remains native and no intrusive patch is applied.
    """

    if not flags.enable_controller:
        return RuntimeBinding(mode="native", reason="controller disabled")

    if flags.observation_only:
        return RuntimeBinding(mode="observe-only", reason="observation_only=true")

    return RuntimeBinding(mode="controller-enabled", reason="controller enabled")
