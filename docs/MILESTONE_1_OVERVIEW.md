# 里程碑一：ELMM — 首次成功优化 vLLM MoE 推理性能

> **日期**：2025-03-20  
> **成果**：在 CPU offload 场景下，通过 ELMM（Expert-Level Memory Management）将 EAGLE-3 推测解码的 MoE 推理吞吐从 **2.01 tok/s 提升至 5.89 tok/s（2.93× 加速）**。

---

## 一、问题定义

### 1.1 背景

大规模 MoE 模型（如 Qwen3-30B-A3B，128 experts × 48 layers，~54 GB bf16）在单卡
GPU（RTX A6000, 48 GB）上无法全部放入显存，必须将部分专家权重 **offload 到 CPU**。
vLLM 使用 **UVA（Unified Virtual Addressing）** 机制：通过 `get_accelerator_view_from_cpu_tensor()` 
创建 CPU pinned memory 的 CUDA 视图，GPU 内核直接访问时触发 PCIe 传输。

### 1.2 核心矛盾

当 **推测解码（Speculative Decoding, SD）** 与 **MoE offloading** 结合时：

```
每个 decode step:
  verify 阶段: K+1 个 token → 每个 token 激活 top-k 个专家
  每个激活的专家: ~9 MB bf16 通过 PCIe (~25 GB/s) 从 CPU 传入
  48 层 × top-8 × (K+1) tokens = 大量 PCIe 传输
```

**MAF（Memory Amplification Factor）理论**预测：

$$\text{MAF}(K) = \frac{N}{k}\left(1 - \left(1 - \frac{k}{N}\right)^{K+1}\right)$$

当 N=128, k=8, K=3 时，MAF(3) = 3.64×。即 SD 比普通 AR 解码多造成 **3.64 倍的 PCIe 流量**。

实验验证：C2（EAGLE-3 + UVA）= 2.009 tok/s vs C1（纯 AR + UVA）= 1.988 tok/s，SD 几乎没有带来加速——**PCIe 瓶颈完全吞噬了 SD 收益**。

### 1.3 关键洞察

SD 的 K+1 个 token 之间存在 **强专家时间局部性**：
- 连续 decode step 的专家激活集合高度重叠
- K+1 个 token 的 router 决策来自同一上下文，倾向激活相同专家

→ **如果在 GPU 缓存中保留最近使用的专家，大部分访问可以走 HBM（768 GB/s）而非 PCIe（25 GB/s）——30× 带宽提升**。

---

## 二、ELMM 解决方案

### 2.1 架构概览

```
┌─────────────────────────────────────────────────────┐
│                    vLLM V1 Engine                     │
│                                                       │
│  ┌─────────────┐   ┌──────────────┐   ┌───────────┐ │
│  │ EAGLE-3     │   │ Qwen3-30B    │   │ KV Cache  │ │
│  │ Draft Model │   │ Target Model │   │ (8.6 GiB) │ │
│  └─────────────┘   └──────┬───────┘   └───────────┘ │
│                            │                          │
│              ┌─────────────▼─────────────┐            │
│              │     ELMM Plugin (hook)     │            │
│              │  ┌───────────────────────┐ │            │
│              │  │ Per-Layer LRU Cache   │ │            │
│              │  │ (4 GB pre-alloc pool) │ │            │
│              │  │ ~17 experts × 26 层   │ │            │
│              │  └───────────────────────┘ │            │
│              │  ┌───────────────────────┐ │            │
│              │  │ Shared GPU Scratchpad │ │            │
│              │  │ (1.15 GB, [E,D1,D2]) │ │            │
│              │  └───────────────────────┘ │            │
│              └───────────────────────────┘            │
│                            │                          │
│              ┌─────────────▼─────────────┐            │
│              │    FusedMoE Kernel Call    │            │
│              │  (读取 scratchpad 而非    │            │
│              │   UVA CPU pinned memory)  │            │
│              └───────────────────────────┘            │
└─────────────────────────────────────────────────────┘
```

### 2.2 三大核心组件

| 组件 | 功能 | 实现 |
|------|------|------|
| **Per-Layer LRU Expert Cache** | GPU 上为每层维护一个 LRU 缓存池 | 预分配 `torch.empty` 固定大小 slot 数组，OrderedDict 跟踪 LRU |
| **Shared GPU Scratchpad** | 所有层共享的临时 GPU buffer（层顺序执行） | 1 个 [E, 2N, D] + 1 个 [E, D, N] tensor，每层复用 |
| **Scratchpad-Swap Protocol** | 在 kernel 调用前将专家数据填入 scratchpad | swap `param.data` → scratchpad → run kernel → restore |

### 2.3 Scratchpad-Swap 协议详解

```python
# 每次 forward_impl 调用:
for eid in unique_experts:
    if cache.hit(eid):
        # HIT: HBM → scratchpad (768 GB/s, ~12μs per expert)
        scratch[eid].copy_(cache_pool[slot])
    else:
        # MISS: UVA/PCIe → scratchpad (25 GB/s, ~360μs per expert)
        scratch[eid].copy_(module.weight[eid])
        # 同时填入 cache pool
        cache_pool[new_slot].copy_(scratch[eid])

# Swap & Run
module.w13_weight.data = scratch_w13   # 指针交换，零开销
module.w2_weight.data  = scratch_w2
quant_method.apply(...)                 # FusedMoE kernel 读 GPU 数据
module.w13_weight.data = orig_data      # 恢复 UVA 指针
```

**关键**：同一 CUDA stream 上的操作天然 FIFO 有序，**无需显式 synchronize()**——
这是性能从 ~3 tok/s 提升到 ~6 tok/s 的关键优化。

---

## 三、实现为 vLLM Plugin

### 3.1 插件机制

ELMM **不修改 vLLM 源码**，而是通过 vLLM 的 `general_plugins` 机制注入：

```toml
# pyproject.toml
[project.entry-points."vllm.general_plugins"]
elmm = "adapters.vllm_elmm_plugin:register"
```

启动方式：`VLLM_PLUGINS=elmm python -m vllm.entrypoints.openai.api_server ...`

### 3.2 注入时序

```
vLLM 启动
 → load_general_plugins() 调用 register()
   → Monkey-patch Worker.load_model
     → 原始 load_model() 完成（权重加载 + offload）
       → activate_elmm(model) 
         → 扫描 FusedMoE 层，检测 offloaded 层
         → 预分配 GPU cache pool + scratchpad（在 KV profiler 之前！）
         → Monkey-patch 每个 offloaded FusedMoE.forward_impl
```

### 3.3 关键技术难点及解决方案

| 难点 | 问题 | 解决方案 |
|------|------|---------|
| **Offload 检测** | `process_weights_after_loading` 中 `replace_parameter()` 创建新 Parameter，丢失 `_vllm_offloaded_cpu_data` 属性 | 检测 module 内 **任意** param 是否有该属性（gate weight 保留了它） |
| **OOM** | Cache 懒分配导致 vLLM KV profiler 占满剩余 GPU | 在 `install()` 中**预分配** cache pool（KV profiler 之前执行） |
| **Pipeline Stall** | `torch.cuda.synchronize()` 导致 GPU pipeline 停顿 | 移除 sync——同 stream 内操作天然有序 |
| **Logger 被抑制** | vLLM 抑制非 vllm logger | 使用 `print(file=sys.stderr, flush=True)` |

---

## 四、实验结果

### 4.1 硬件环境
- GPU: NVIDIA RTX A6000 (48 GB, 44.55 GB usable)
- PCIe: Gen4 x16 (~25 GB/s)
- CPU: Pinned memory via UVA

### 4.2 模型配置
- 目标模型: Qwen3-30B-A3B-Instruct-2507
- 草稿模型: EAGLE-3 (0.6B params, K=3)
- Offload: ~30 GB 到 CPU (26/48 层 offloaded)

### 4.3 吞吐量对比

| 配置 | 说明 | 吞吐 (tok/s) | 相对加速 |
|------|------|:------------:|:--------:|
| C1 | 纯 AR 解码 + UVA offload | 1.99 | 0.99× |
| C2 | EAGLE-3 (K=3) + UVA offload | **2.01** | 1.00× (baseline) |
| C3 | C2 + SpecFusedMoE 专家去重 | 2.12 | 1.06× |
| **ELMM** | **C2 + ELMM (4GB cache)** | **5.89** | **2.93×** |

### 4.4 ELMM 详细测量

| 请求 | 输出 Tokens | 耗时 (s) | tok/s |
|:----:|:-----------:|:--------:|:-----:|
| 0 | 128 | 23.94 | 5.35 |
| 1 | 128 | 19.56 | 6.54 |
| 2 | 128 | 19.22 | 6.66 |
| 3 | 128 | 24.79 | 5.16 |
| 4 | 128 | 21.18 | 6.04 |
| **平均** | **128** | **21.74** | **5.89** |

### 4.5 关键发现

1. **C2 vs C1 几乎无差异**（2.01 vs 1.99）→ PCIe 完全瓶颈，SD 的批次计算优势被 offload 开销吞噬
2. **C3 的 5.6% 提升来自去重**，但去重只减少了重复的 PCIe 传输，未解决跨步缓存
3. **ELMM 的 2.93× 加速**证明了**跨步专家缓存**是解决 SD + MoE offload 性能问题的关键
4. **ELMM 预分配 pool 意外地改善了 GPU 碎片化**，使得相同 `gpu_memory_utilization=0.85` 下 ELMM 能分配更多 KV cache

---

## 五、代码文件结构

### 5.1 核心代码

```
adapters/
  elmm_plugin.py          # ELMM 核心实现 (643 行)
                           #   - ELMMConfig: 配置
                           #   - _LayerExpertCache: 预分配 LRU 缓存池
                           #   - ELMMManager: 安装/forward/统计/清理
                           #   - activate_elmm() / deactivate_elmm(): 全局单例
  vllm_elmm_plugin.py     # vLLM 插件入口 (66 行)
                           #   - register(): Worker.load_model monkey-patch
```

### 5.2 实验脚本

```
scripts/
  run_eagle3_experiment.py  # 5 配置消融实验自动化
  elmm_bench_worker.py      # ELMM 基准测试 worker
  run_elmm_ablation.sh      # ELMM 消融 shell 脚本
  specmoe_server.py         # SpecMoE 服务端启动器
  bench_serving_sharegpt.py # ShareGPT 格式 benchmark
  measure_routing_overlap.py # 专家路由重叠分析
  plot_routing_overlap*.py  # 可视化脚本
```

### 5.3 实验结果

```
results/
  eagle3_ablation/          # C1-C5 消融数据
    final_results.json      # 完整结果
  elmm_ablation_results.md  # ELMM 实验结果报告
  routing_overlap_*.json/csv/png  # 路由重叠分析
```

---

## 六、启动命令参考

```bash
# ELMM 模式启动（推荐）
cd /root/MoE-SD
VLLM_PLUGINS=elmm ELMM_CACHE_GB=4 ELMM_LOG_INTERVAL=0 \
  python -m vllm.entrypoints.openai.api_server \
    --model /root/models/Qwen3-30B-A3B-Instruct-2507 \
    --speculative-config '{"model": "/root/models/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3", "method": "eagle3", "num_speculative_tokens": 3, "draft_tensor_parallel_size": 1}' \
    --tensor-parallel-size 1 \
    --cpu-offload-gb 30 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --port 8000
```

---

## 七、后续计划

1. **Temporal Locality 数据采集**：开启 ELMM 日志采集 overlap_history，量化专家时间局部性
2. **Draft-Guided Prefetch**：利用 draft model 的路由决策预取专家到 GPU cache
3. **自适应 Cache Budget**：根据运行时命中率动态调整每层缓存大小
4. **论文撰写**：整理成 CCF-A 会议论文，对标 SP-MoE / MoE-SpeQ / SpecMoEOff
