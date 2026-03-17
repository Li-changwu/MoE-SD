"""Memory Manager for Dynamic Memory Partition Enforcement"""
import gc
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

@dataclass
class MemoryPartition:
    expert_budget_mb: float
    speculative_budget_mb: float
    kv_reserve_mb: float
    total_gpu_mb: float
    
    @property
    def allocated_mb(self) -> float:
        return self.expert_budget_mb + self.speculative_budget_mb + self.kv_reserve_mb
        
    @property
    def utilization(self) -> float:
        return self.allocated_mb / self.total_gpu_mb if self.total_gpu_mb > 0 else 0.0

class MemoryManager:
    """Enforces controller memory partitions on GPU."""
    
    def __init__(self, controller, gpu_id=0, safety_margin_mb=512.0):
        self.controller = controller
        self.gpu_id = gpu_id
        self.safety_margin_mb = safety_margin_mb
        self._torch_device = torch.device(f"cuda:{gpu_id}") if TORCH_AVAILABLE else None
        self._total_gpu_mb = self._get_total_gpu_memory()
        self._current_partition = None
        self._oom_events = 0
        
    def _get_total_gpu_memory(self) -> float:
        if not TORCH_AVAILABLE:
            return 80 * 1024
        try:
            total_bytes = torch.cuda.get_device_properties(self.gpu_id).total_memory
            return total_bytes / (1024 ** 2)
        except Exception:
            return 80 * 1024
            
    def enforce_partition(self, state) -> MemoryPartition:
        try:
            decision = self.controller.decide_memory_partition(state)
            if not decision.get("apply", False):
                return self._create_default_partition()
            expert_mb = decision.get("expert_budget_mb", self._total_gpu_mb * 0.3)
            spec_mb = decision.get("speculative_budget_mb", self._total_gpu_mb * 0.2)
            kv_mb = decision.get("kv_reserve_mb", self._total_gpu_mb * 0.4)
            partition = MemoryPartition(expert_mb, spec_mb, kv_mb, self._total_gpu_mb)
            self._apply_partition(partition)
            return partition
        except Exception as e:
            if TORCH_AVAILABLE:
                try:
                    if isinstance(e, torch.cuda.OutOfMemoryError):
                        self._oom_events += 1
                        return self._emergency_shrink()
                except:
                    pass
            logger.error(f"Memory partition failed: {e}")
            return self._create_default_partition()
            
    def _apply_partition(self, partition: MemoryPartition) -> None:
        if TORCH_AVAILABLE:
            gc.collect()
            torch.cuda.empty_cache()
        self._current_partition = partition
        
    def _create_default_partition(self) -> MemoryPartition:
        thirds = self._total_gpu_mb / 3.0
        return MemoryPartition(thirds, thirds, thirds, self._total_gpu_mb)
        
    def _emergency_shrink(self) -> MemoryPartition:
        return MemoryPartition(self._total_gpu_mb * 0.2, self._total_gpu_mb * 0.1, self._total_gpu_mb * 0.5, self._total_gpu_mb)
        
    def get_statistics(self) -> Dict[str, Any]:
        return {"total_gpu_mb": self._total_gpu_mb, "oom_events": self._oom_events, "partition_utilization": self._current_partition.utilization if self._current_partition else 0.0}
