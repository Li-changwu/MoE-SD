"""
Runtime Entrypoints for MoE-SD Scheduler

This module builds runtime bindings that integrate controllers with vLLM.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .feature_flags import FeatureFlags

logger = logging.getLogger(__name__)


@dataclass
class RuntimeBinding:
    """Runtime binding result."""
    mode: str
    reason: str
    controller: Optional[Any] = None
    hooks: Optional[Any] = None
    spec_integrator: Optional[Any] = None
    memory_manager: Optional[Any] = None
    

def build_runtime(
    flags: FeatureFlags,
    controller: Optional[Any] = None,
    vllm_scheduler: Optional[Any] = None,
    vllm_worker: Optional[Any] = None,
) -> RuntimeBinding:
    """
    Build runtime binding based on feature flags and components.
    
    This function sets up the complete MoE-SD integration stack:
    1. Controller (if enabled)
    2. Scheduler hooks (if enabled)
    3. Spec integrator (if SD enabled)
    4. Memory manager (if dynamic partition enabled)
    
    Args:
        flags: Feature flags
        controller: Scheduler controller instance
        vllm_scheduler: vLLM scheduler instance
        vllm_worker: vLLM worker instance
        
    Returns:
        Runtime binding with all initialized components
    """
    if not flags.enable_controller:
        logger.info("Controller disabled, using native vLLM runtime")
        return RuntimeBinding(mode="native", reason="controller disabled")

    if flags.observation_only:
        logger.info("Observation-only mode enabled")
        return RuntimeBinding(
            mode="observe-only",
            reason="observation_only=true",
            controller=controller,
        )

    # Full controller mode - initialize all adapters
    logger.info("Building full controller runtime stack")
    
    binding = RuntimeBinding(
        mode="controller-enabled",
        reason="full controller integration",
        controller=controller,
    )
    
    # Install scheduler hooks if vLLM scheduler provided
    if vllm_scheduler is not None and controller is not None:
        try:
            from adapters.vllm_hooks import SchedulerHookManager
            hooks = SchedulerHookManager(controller)
            hooks.install(vllm_scheduler)
            binding.hooks = hooks
            logger.info("Scheduler hooks installed")
        except ImportError as e:
            logger.warning(f"Failed to import scheduler hooks: {e}")
        except Exception as e:
            logger.error(f"Failed to install scheduler hooks: {e}")
    
    # Initialize spec integrator if vLLM worker provided
    if vllm_worker is not None and controller is not None:
        try:
            from adapters.spec_integrator import SpecIntegrator, SpeculationConfig
            spec_cfg = SpeculationConfig(
                spec_method=flags.spec_method if hasattr(flags, 'spec_method') else "eagle3",
                adaptive=flags.adaptive_spec if hasattr(flags, 'adaptive_spec') else True,
            )
            spec_integrator = SpecIntegrator(controller, vllm_worker, spec_cfg)
            binding.spec_integrator = spec_integrator
            logger.info("Spec integrator initialized")
        except ImportError as e:
            logger.warning(f"Failed to import spec integrator: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize spec integrator: {e}")
    
    # Initialize memory manager if enabled
    if flags.dynamic_memory if hasattr(flags, 'dynamic_memory') else False:
        try:
            from adapters.memory_manager import MemoryManager
            gpu_id = flags.gpu_id if hasattr(flags, 'gpu_id') else 0
            memory_manager = MemoryManager(controller, gpu_id=gpu_id)
            binding.memory_manager = memory_manager
            logger.info("Memory manager initialized")
        except ImportError as e:
            logger.warning(f"Failed to import memory manager: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize memory manager: {e}")
    
    return binding
