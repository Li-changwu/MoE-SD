# MoE-SD 优化全链路总结

> **硬件环境**：NVIDIA RTX A6000 48 GB, PCIe Gen4 x16 (~25 GB/s), HBM 768 GB/s  
> **模型**：Qwen3-30B-A3B-Instruct-2507 (128 experts, top-8, 48 layers, ~54 GB bf16)  
> **推测解码**：EAGLE-3, K=3  
> **Offload 配置**：`--cpu-offload-gb 30`（26/48 层卸载至 CPU）  
> **最终成果**：2.01 tok/s → **19.88 tok/s**（**9.89× 加速**）

---

## 性能演进总览

| 阶段 | 吞吐量 | 相对基线 | 增量 |
|------|--------|---------|------|
| UVA 基线 (EAGLE-3 + vLLM offload) | 2.01 tok/s | 1.00× | — |
| + ELMM v1 LRU Expert Cache | 5.89 tok/s | 2.93× | +193% |
| + Pool-Direct (消除冗余 HBM 拷贝) | 10.23 tok/s | 5.09× | +73.7% |
| + Direct Dispatch (绕过 Python 调用链) | 10.31 tok/s | 5.13× | +0.8% |
| + TASER (自适应 Stale Expert Routing) | 14.63 tok/s | 7.28× | +41.9% |
| + Oracle 跨层预取 | 16.85 tok/s | 8.38× | +15.2% |
| + Triton Tile Config 调优 | **19.88 tok/s** | **9.89×** | **+18.0%** |

---

## 优化一：ELMM v1 — Expert-Level LRU GPU 缓存

### 关键瓶颈

推测解码（SD）与 MoE offloading 结合时，PCIe 带宽成为绝对瓶颈。每个 decode step 中 K+1 个 token 各激活 top-8 个 expert，所有 expert 权重（~9 MB/个）通过 UVA 从 CPU pinned memory 经 PCIe (~25 GB/s) 按需传输。MAF（Memory Amplification Factor）理论预测 SD 造成 3.64× 的 PCIe 流量放大，导致 EAGLE-3 几乎无法加速（2.01 vs 1.99 tok/s）。

**根源**：vLLM 原生 UVA offload 无跨步缓存机制 —— 即使连续 step 激活相同 expert，每次仍重新通过 PCIe 传输。

### 详细解决方案

**核心架构**：三组件协同

1. **Per-Layer LRU Expert Cache**：为每个 offloaded 层预分配 GPU HBM 缓存池（共 4 GB，~17 slots/层 × 26 层），用 `OrderedDict` 实现 LRU 淘汰策略。Cache hit 走 HBM（768 GB/s），miss 才走 PCIe（25 GB/s）—— **30× 带宽差异**。

2. **Shared GPU Scratchpad**：所有层共享一个 1.15 GB 的 GPU buffer（层顺序执行互不冲突），作为 FusedMoE kernel 的输入。每层 forward 前将 cached/uncached expert 权重填入 scratchpad。

3. **Scratchpad-Swap 协议**：通过指针交换（`module.weight.data = scratchpad`）实现零拷贝切换，FusedMoE kernel 直接读 GPU 数据；kernel 完成后恢复原始 UVA 指针。**关键优化**：移除 `torch.cuda.synchronize()` —— 同 CUDA stream 内操作天然有序，消除 pipeline stall 后吞吐翻倍。

**实现方式**：vLLM `general_plugins` 机制注入，**不修改 vLLM 源码**。

### 具体结果

| 指标 | 值 |
|------|-----|
| 吞吐 | 2.01 → **5.89 tok/s** |
| 加速 | **2.93×** |
| Cache 命中率 | >99%（warmup 后稳态） |
| GPU 显存占用 | +4 GB（cache pool）+ 1.15 GB（scratchpad） |

---

## 优化二：Pool-Direct — 消除冗余 HBM 拷贝

### 关键瓶颈

ELMM v1 的 Scratchpad-Swap 流程存在冗余数据搬运：每步每层需将 cache pool 中的 expert 权重**复制**到 scratchpad（HBM→HBM, ~6 GB/step），即使 cache 已100%命中。Scratchpad 作为中间层引入了不必要的 HBM 带宽消耗。

Phase Profiling 量化显示：**P3_cache（缓存管理） ≈ P4_kernel（MoE 计算）≈ 48-50%**，cache 搬运开销与实际计算对等。

### 详细解决方案

**核心思想**：将每层分散的 cache slots 合并为一个**连续权重池** `[max_slots, D1, D2]`，让 FusedMoE kernel 直接在 pool 张量上运行，彻底消除 scratchpad 中间层。

**关键改造**：
1. 将 expert ID 重映射到 pool slot 位置：`topk_ids` 中的原始 expert ID（0-127）替换为 pool slot ID（0-16），FusedMoE kernel 将 "pool slot" 视为 "expert"
2. 参数修改：`fused_moe` 调用中 `num_experts` 从 128 改为 `max_slots`（17），`w1` / `w2` 直接指向 pool 张量
3. Triton 兼容性验证：确认 E=17 的非 2 的幂次不影响 Triton kernel 正确性（实测通过）

**数据路径对比**：
```
v1:  cache_pool[slot] → copy → scratchpad[eid] → FusedMoE → output
v2:  cache_pool[slot] → (remap topk_ids) → FusedMoE(pool) → output  (零拷贝)
```

### 具体结果

| 指标 | 值 |
|------|-----|
| 吞吐 | 9.99 → **10.23 tok/s** |
| 增量加速 | **+2.4%** |
| 消除的冗余拷贝 | ~6 GB/step HBM→HBM |
| 释放的 GPU 显存 | 1.15 GB（scratchpad 被移除） |

> **注**：+2.4% 看似不大，但 Pool-Direct 是后续 Direct Dispatch、TASER 等优化的**架构基础** —— 所有后续优化均建立在 "FusedMoE 直接操作 pool" 的前提上。

---

## 优化三：TASER — 自适应 Stale Expert Routing

### 关键瓶颈

Pool-Direct + Direct Dispatch 后，Phase Profiling 再次显示 **P3_cache（48.8%）≈ P4_kernel（49.7%）**。P3 的热点在于每层每步的 `unique().tolist()`（GPU→CPU 同步）+ Python dict lookup + scatter/remap 链路，26 层共 26 次同步屏障。

初版 Stale-Remap（warmup=200, 固定 interval=16）实验暴露两个问题：
1. **warmup 过长**：200 步 ≈ 400-600 tokens ≈ 3-4 个请求才激活 stale path，导致前 3 个测试请求完全未受益（平均 +4.7%，但稳态 +27.4%）
2. **固定间隔的缓存抖动（Cache Churn）**：每 16 步的强制验证触发 LRU eviction，扰乱已稳定的缓存状态，导致后续 stale remap 映射不精确

### 详细解决方案

**TASER（Temporal-Adaptive Stale Expert Routing）** —— 借鉴 CPU 分支预测器的自适应间隔调度，三相设计：

**Phase I（Warmup, step < 32）**：每步执行完整 Phase 3，建立精确的 `_layer_remap` 映射表。warmup 从 200 降至 32（17-slot LRU 在 ~2 步内填满，32 步后缓存完全稳定）。

**Phase II（Fast Path, 95%+ 步骤）**：单次 GPU gather 操作 `remapped_ids = remap_table[topk_ids]`，**零 CPU 同步、零 dict lookup、零 eviction**。

**Phase III（自适应验证）**：执行完整 Phase 3 + 对比 expert set 变化：
- 路由集合不变 → 间隔指数退避：`interval = min(interval × 2, 128)`
- 路由集合变化 → 重置为初始间隔：`interval = 16`

**核心数学**：

$$f_{val}^{TASER} \approx 1/I_{max} = 1/128 = 0.78\%$$

相比固定间隔的 $1/16 = 6.25\%$，验证频率降低 **8×**。

更重要的是**缓存抖动消除效应**：减少验证 → 减少 eviction → 缓存状态更稳定 → stale remap 更准确。固定 interval=16 在某些请求中产生 6.81 tok/s 的严重跌落，TASER 自适应间隔将最低值提升至 9.83 tok/s。

### 具体结果

| 指标 | 值 |
|------|-----|
| 吞吐 | 10.33 → **14.63 tok/s** |
| 增量加速 | **+41.6%** |
| 累计 vs v1 | **1.93×** |
| vs 旧 Stale-Remap (warmup=200) | +34.8% |
| 最小 tok/s | 6.81（旧固定间隔）→ **9.83**（TASER） |
| 核心代码量 | ~25 行核心逻辑 |

**Per-Request 详情**：R2 从 7.31 → 18.10 tok/s（**+147.6%**），因 warmup=32 使 TASER 在 warmup 请求期间即完成收敛，所有测试请求均享受 fast path。

---

## 优化四：Oracle 跨层预取

### 关键瓶颈

TASER 将 MoE forward 时间构成从 P3≈P4 变为 **P4 独占 >97%**，Phase 3 开销近乎消除。但在 TASER validation/warmup 过渡期（adaptive interval 从 16→128 逐步增长），cache miss 引发的 PCIe 传输仍然阻塞关键路径：单次 miss 的 CPU→GPU 拷贝（~9.44 MB, ~0.38 ms）直接叠加到步骤延迟上。

同时，benchmark 发现 TASER 冻结的 remap 表具有一个独特属性：每层的 `_layer_remap[layer_name]` 在 stale path 中是**完全冻结的 GPU 张量**，精确记录了所有 expert→pool_slot 映射 —— 这是一个**零成本跨层 oracle**。

### 详细解决方案

**核心方法 `_oracle_prefetch_next_layer()`**：在 Layer N 的 MoE kernel（HBM-bound）执行期间，通过独立 `_prefetch_stream` 利用 PCIe 带宽（与 HBM 不竞争）异步预取 Layer N+1 的 miss expert。

**预取逻辑**：
1. 读取 Layer N+1 的 remap 表（frozen, 零读取成本）
2. 对比 Layer N+1 的 cache slot_map，判断缓存是否已满
3. 若未满：利用**层间 expert 相关性**（当前层 cache 的 expert 集合作为下一层的预测），通过 `non_blocking=True` 的 `copy_()` 发起异步 H2D 传输

**硬件资源并行性**：MoE kernel 消耗 GPU HBM 带宽，Oracle 预取消耗 PCIe 带宽，两者使用**不同硬件通道**，完全重叠。

**RTFA（运行时可行性分析）**：MoE kernel 时间窗口 0.353 ms × PCIe 25 GB/s = 可预取 8.83 MB ≈ 0.94 个 expert，而 decode 场景平均 miss < 1 个/step → **完全覆盖**。

**Race Condition 修复**：因 Oracle 在 `_prefetch_stream` 上分配 slot 并启动异步拷贝，下一层可能看到已分配但未完成拷贝的 slot。在 Phase 3 前添加 `wait_stream()` 同步保证正确性。

### 具体结果

| 指标 | 值 |
|------|-----|
| 吞吐 | 15.45 → **16.85 tok/s** |
| 增量加速 | **+9.1%** |
| 累计 vs v1 | **2.23×** |
| 前中段请求改善 (R1-R3) | -12.7% ~ -18.9% 延迟 |
| Stale path 开销 | 零（cache 已满时 early return） |

> Oracle 的 +9.1% 高于理论预期，原因在于 validation 级联效应：Layer N 更新 remap 后，Layer N+1-N+25 的 Oracle 提前为它们加载新 expert，减少后续层的同步等待。

---

## 优化五：Triton Tile Config 调优

### 关键瓶颈

ALPS 实施后瓶颈再分析揭示：**HBM 带宽利用率仅 59.2%**，远低于理论峰值。Roofline 分析确认 M=4 decode 场景下的 MoE kernel 是极端 memory-bound（AI = 1.882 FLOP/byte << ridge point 50.4），理论最小延迟 208.9 μs/层，实际 353 μs/层 —— 存在 ~40% 优化空间。

**根本原因**：A6000 不在 vLLM 预置 Triton 配置列表中，生产服务器始终使用保守的默认回退配置（BSM=16, BSN=32, BSK=64），且 `VLLM_TUNED_CONFIG_FOLDER` 环境变量从未被设置。

### 详细解决方案

**微基准测试**：自建 Triton FusedMoE 微基准（`scripts/triton_config_focused.py`），对 200+ 配置组合进行扫描（200 次预热 + 500 次测量）。

**最优配置**（E=17, M=4, A6000）：
```json
{"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128,
 "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 3}
```
Shared memory 验证：`2 × (16×128 + 128×64) × 2 = 40 KB < 48 KB（A6000 限制）` ✓

**双路注入策略**：
1. **ELMM Direct Dispatch 硬编码覆盖**：当 `try_get_optimal_moe_config()` 返回的配置缺少 `num_warps` 时，用最优配置替换，覆盖 26 层卸载路径
2. **VLLM_TUNED_CONFIG_FOLDER 自动注入**：在 plugin `register()` 中设置环境变量指向 A6000 专用 JSON 配置，覆盖 22 层非卸载路径

### 具体结果

| 指标 | 值 |
|------|-----|
| 吞吐 | 16.87 → **19.88 tok/s** |
| 增量加速 | **+17.8%** |
| 累计 vs 基线 | **9.89×** |
| HBM 利用率 | 59.2% → **72.6%** |
| 内核级延迟改善 | 294.9 → 287.6 μs/层 (-2.5%) |

> **E2E 17.8% >> 内核级 2.5%**：额外提升来自三个放大效应：(1) 调优内核被 CUDA Graph 捕获后消除 Python 调度开销；(2) TASER 在更多步骤后收敛到更优的 expert 映射；(3) 更快的单步延迟提升了 speculative decode 有效吞吐量。

---

## 失败的优化尝试

以下优化经过完整实现和 A/B 测试，结果证明无效或产生回退，但提供了重要的工程洞察：

| 优化 | 结果 | 关键教训 |
|------|------|---------|
| **Draft-Guided Prefetch** | 中性 | LRU 命中率已 >99%，预取无触发空间 |
| **自适应 Cache Budget** | 中性 | 26 层访问模式高度均匀，重分配几乎不变 |
| **GPU-Side Cache Lookup (E1)** | **-18.5%** | 19 个元素的 tensor 操作不适合 GPU；`tolist()` vs `item()` 的同步开销相同，瓶颈是 pipeline stall 而非数据量 |
| **Per-Layer CUDA Graph** | **-32.4%** | 每层仅 5 个 kernel 但需 3 次 `copy_()` 填充 placeholder，额外 launch 几乎完全抵消 Graph 节省 |
| **SharedExpert-RoutedMoE 并行** | **+0.06%** | SharedExpert 仅 0.035 ms（MoE 的 1/10），stream 同步开销 ≈ 重叠收益 |
| **Prefill 级联预热 (routing-based)** | **-5.9%** | `topk_ids.tolist()` 触发 GPU→CPU sync，且预测集过大（>80 experts vs 17 slots），Oracle cache-based 路径已足够 |

---

## 核心工程洞察

1. **"不要优化算子本身，优化算子之间的空泡"**：真正的瓶颈往往是 kernel 间的调度间隙（launch overhead、CPU-GPU 同步、内存传输等待），而非 kernel 本身

2. **"利用异构硬件的并行维度"**：GPU HBM 带宽、PCIe 带宽、CPU 计算是三个独立资源 —— Oracle 预取正是利用 MoE kernel（HBM-bound）期间空闲的 PCIe 带宽

3. **"小数据量反模式"**：对 ~19 个元素的 cache lookup，Python dict 比 GPU sort+gather+compare 更快 —— GPU 优势在于大规模并行，小 tensor 反而受 launch/alloc 开销拖累

4. **"频繁验证 ≠ 高正确性"**：TASER 证明减少 cache 验证频率反而提升性能**和**正确性（更少 eviction → 更稳定映射），与 CPU 分支预测的经典教训一致

5. **"默认配置税"**：A6000 因不在 vLLM 预置列表中，始终使用保守默认 Triton 配置，仅调优 tile 参数即带来 +17.8% E2E 提升 —— 永远检查你的运行时是否真的在用调优配置
