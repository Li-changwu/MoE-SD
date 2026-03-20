# SpecMoE：面向内存受限 MoE 推理的推测解码优化框架

## 一、研究问题

大规模 Mixture-of-Experts (MoE) 模型（如 Qwen3-30B-A3B，128 experts × 48 layers）在推理时面临两个核心矛盾：

1. **Speculative Decoding (SD) 与 MoE 的乘性内存放大**：SD 的 verify 阶段需要同时处理 K+1 个 token，每个 token 激活 top-k 个专家。原生 `fused_moe` 对每个 token 独立加载专家，导致内存带宽放大 (K+1)×top_k 倍——我们称为 **MAF (Memory Amplification Factor)**。
2. **GPU 显存无法容纳全部专家权重**：128 experts × 48 layers = 57.6 GB (bf16)，超过单卡 48 GB。需要 CPU offload，但 offload 开销与专家访问次数成正比。

**SpecMoE** 从三个维度解决该问题：
- **Expert Deduplication**：verify batch 内跨 token 去重，将 MAF 从 (K+1)×top_k 降至 ≈ top_k
- **Early Termination (SDD)**：层间 router logits 监控，提前冻结已发散的 draft token
- **Expert Caching + Prefetch**：GPU/CPU 分层缓存 + draft-guided 预取，降低 offload 延迟

## 二、系统架构

```
                          ┌──────────────────────────────────┐
                          │   vLLM Server + EAGLE-3 SD       │
                          │   (Qwen3-30B-A3B-Instruct-2507)  │
                          └───────────────┬──────────────────┘
                                          │
                    ┌─────────────────────▼─────────────────────┐
                    │      SchedulerHookManager (vllm_hooks)    │
                    │      拦截 engine.step() + spec_execute()  │
                    └─────────────────────┬─────────────────────┘
                                          │
              ┌───────────────────────────▼───────────────────────────┐
              │              SchedulerController (interface)          │
              │  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  │
              │  │StaticGovernor│  │PhaseAware    │  │ NoOp       │  │
              │  │  (v0 规则)   │  │Governor(v1)  │  │ Controller │  │
              │  └──────────────┘  └──────────────┘  └────────────┘  │
              └───────────────────────────┬──────────────────────────┘
                                          │ decide_speculation_k()
                                          │ decide_memory_partition()
                                          │ decide_prefetch()
                                          ▼
          ┌─────────────────────────────────────────────────────────┐
          │               SpecMoE Engine (specmoe_engine)           │
          │     统一编排所有优化组件的生命周期                         │
          └────┬──────────┬──────────────┬──────────────┬───────────┘
               │          │              │              │
       ┌───────▼───┐  ┌──▼────────┐  ┌──▼──────────┐  ┌▼────────────┐
       │FusedMoE   │  │Expert     │  │Speculation  │  │Trace        │
       │Hook       │  │Weight     │  │Divergence   │  │Collectors   │
       │(去重分发) │  │Cache      │  │Detector     │  │(数据采集)   │
       └───────────┘  │(分层缓存) │  │(早停检测)   │  └─────────────┘
                      └───────────┘  └─────────────┘

          ┌──────────────────────────────────────────────────┐
          │          MetricsSidecar (指标反馈闭环)            │
          │  ← Prometheus /metrics → K 决策 → /dev/shm      │
          └──────────────────────────────────────────────────┘
```

## 三、模块详解

### 3.1 适配器层 (`adapters/`)

核心优化组件，以 monkey-patch 方式无侵入地集成到 vLLM 运行时。

#### 3.1.1 `fused_moe_hook.py` — FusedMoE 运行时拦截器

| 项目 | 说明 |
|------|------|
| **文件** | `adapters/fused_moe_hook.py` (~350 行) |
| **作用** | 运行时 monkey-patch vLLM 的 `fused_moe` 函数，在 SD verify 阶段将调用重定向到 SpecFusedMoE 去重路径 |
| **核心类** | `FusedMoEHook` |
| **关键方法** | `install()` / `uninstall()` — 安装/卸载 hook |
| | `set_verify_mode(on)` — 手动切换 verify 模式 |
| | `verify_context()` — 上下文管理器自动管理 verify 状态 |
| | `_wrapped_fused_moe()` — 包装后的 fused_moe，根据 batch size 自动检测 verify 阶段 |
| **设计要点** | batch_size ≥ threshold (默认 4，对应 K=3) 时自动判定为 verify batch；可通过 REST API 运行时重配置 |

#### 3.1.2 `triton_spec_moe.py` — Triton 去重 MoE 内核

| 项目 | 说明 |
|------|------|
| **文件** | `adapters/triton_spec_moe.py` (~400 行) |
| **作用** | 生产级 Triton JIT 内核，实现跨 token 专家去重、fused SiLU-gate-up 投影、frozen-token 跳过 |
| **核心类** | `SpecFusedMoEDispatcher` — 高层调度器，自动选择 Triton / PyTorch fallback |
| | `SpecFusedMoEFunction` — PyTorch 备选实现（CPU 兼容） |
| **关键指标** | 实时跟踪 dedup ratio、MAF、hit rate |

#### 3.1.3 `spec_fused_moe.py` — SD 感知的 FusedMoE 算子

| 项目 | 说明 |
|------|------|
| **文件** | `adapters/spec_fused_moe.py` (~250 行) |
| **作用** | 将 verify batch 的专家加载分解为唯一专家集合，按 acceptance probability 排序执行 |
| **核心类** | `SpecFusedMoE` (nn.Module) — 去重调度核心 |
| | `DedupAnalyzer` — 离线分析器，从 routing trace 估算去重收益 |
| **核心优化** | 唯一专家提取 → acceptance-order 执行 → early-abort 跳过低概率专家 |

#### 3.1.4 `expert_cache.py` — GPU/CPU 分层专家缓存

| 项目 | 说明 |
|------|------|
| **文件** | `adapters/expert_cache.py` (~400 行) |
| **作用** | LRU GPU 缓存 + CPU pinned memory 冷存储 + 异步预取 |
| **核心类** | `ExpertWeightCache` — 缓存管理器 |
| | `ExpertCacheConfig` — 配置：GPU 预算 (如 8GB)、淘汰策略、预取深度 |
| | `CacheStats` — 命中率、预取准确率、传输带宽统计 |
| **设计要点** | 128×48 = 6144 个专家，GPU 放不下 → LRU 淘汰 + draft phase 路由预测驱动预取 |

#### 3.1.5 `layer_early_terminator.py` — 推测发散检测器 (SDD)

| 项目 | 说明 |
|------|------|
| **文件** | `adapters/layer_early_terminator.py` (~300 行) |
| **作用** | 层间 router logits 监控，检测 draft token 是否已发散，提前冻结避免无效计算 |
| **核心类** | `SpeculationDivergenceDetector` — 支持 KL 散度、overlap、entropy 三种检测方法 |
| | `TokenDivergenceState` — 每 token 的发散状态跟踪 |
| | `SDDConfig` — 阈值、连续发散层数门限 |
| **核心逻辑** | 逐层监控 router logits → KL/overlap 超阈值 → 连续 N 层发散 → 标记 frozen → 跳过后续层计算 |

#### 3.1.6 `specmoe_engine.py` — 端到端编排引擎

| 项目 | 说明 |
|------|------|
| **文件** | `adapters/specmoe_engine.py` (~300 行) |
| **作用** | 统一编排 SpecFusedMoE + SDD + ExpertCache + TraceCollector 的生命周期 |
| **核心类** | `SpecMoEEngine` — 顶层引擎 |
| | `SpecMoEConfig` — 全局配置 |
| **生命周期** | `initialize()` → `on_draft_step()` → `begin_verify()` → `dispatch_moe()` → `end_verify()` → `shutdown()` |

#### 3.1.7 `spec_integrator.py` — SD 集成器

| 项目 | 说明 |
|------|------|
| **文件** | `adapters/spec_integrator.py` (~350 行) |
| **作用** | 桥接 Controller 的 K 决策与 vLLM SpecDecodeWorker，追踪 acceptance rate |
| **核心类** | `SpecIntegrator` — monkey-patch worker.execute_model |
| | `AcceptanceTracker` — EMA + 滑动窗口 acceptance 追踪器 |
| **关键功能** | 自适应推测调优、预取信号分发、per-request 状态追踪 |

#### 3.1.8 `vllm_hooks.py` — vLLM 调度器钩子

| 项目 | 说明 |
|------|------|
| **文件** | `adapters/vllm_hooks.py` (~300 行) |
| **作用** | 包装 `engine.step()` 和 `SpecDecodeWorker.execute_model`，在每个 decode step 注入 controller 决策 |
| **核心类** | `SchedulerHookManager` — 钩子安装/卸载管理 |
| | `StepTrace` — 每步 trace：K 值、acceptance rate、KV usage |
| | `AcceptanceWindow` — 滑动窗口 acceptance 估计 |

#### 3.1.9 `memory_manager.py` — 内存分区管理

| 项目 | 说明 |
|------|------|
| **文件** | `adapters/memory_manager.py` (~150 行) |
| **作用** | 执行 controller 的内存分区决策，动态分配 expert / speculative / KV-cache 预算 |
| **核心类** | `MemoryManager` — 分区执行器 |
| | `MemoryPartition` — expert_budget_mb / speculative_budget_mb / kv_reserve_mb |

#### 3.1.10 `metrics_sidecar.py` — 指标采集 Sidecar

| 项目 | 说明 |
|------|------|
| **文件** | `adapters/metrics_sidecar.py` (~250 行) |
| **作用** | 独立进程轮询 vLLM `/metrics` 端点，解析 KV usage + acceptance rate，生成 K 决策写入共享内存 |
| **核心类** | `MetricsSidecar` — Prometheus 指标消费者 |
| **数据流** | vLLM /metrics → parse → RuntimeState → controller.decide_k() → `/dev/shm/moe_sd_k` → EAGLE proposer |

#### 3.1.11 `patch_eagle_dynamic_k.py` — EAGLE 动态 K 补丁

| 项目 | 说明 |
|------|------|
| **文件** | `adapters/patch_eagle_dynamic_k.py` (~100 行) |
| **作用** | 补丁 vLLM 的 eagle.py，支持从 `/dev/shm/moe_sd_k` 读取动态 K 值 |
| **函数** | `apply_patch()` / `revert_patch()` / `verify_patch()` |
| **机制** | propose() 时读 effective_k → 提前退出循环 → 输出 padding |

---

### 3.2 数据采集层 (`collectors/`)

用于实验数据收集和离线分析。

#### 3.2.1 `acceptance_collector.py` — Acceptance Rate 分析器

| 项目 | 说明 |
|------|------|
| **文件** | `collectors/acceptance_collector.py` (~150 行) |
| **作用** | 按 prompt 长度、输出长度、QPS、temperature 分桶分析 acceptance rate |
| **输出** | parquet + PNG 分桶可视化图 |

#### 3.2.2 `moe_trace_collector.py` — MoE 路由追踪分析

| 项目 | 说明 |
|------|------|
| **文件** | `collectors/moe_trace_collector.py` (~150 行) |
| **作用** | 专家访问热力图、token 间 Jaccard overlap、专家复用距离分析 |
| **输出** | expert_heat.parquet / overlap.parquet / reuse_distance.parquet + 可视化 |

#### 3.2.3 `expert_locality_analyzer.py` — 时间局部性分析器

| 项目 | 说明 |
|------|------|
| **文件** | `collectors/expert_locality_analyzer.py` (~200 行) |
| **作用** | 跨 SD verify round 的专家时间局部性量化：inter-round overlap、reuse distance、draft-target 相关性 |
| **核心类** | `ExpertTemporalLocalityAnalyzer` |
| **输出** | 局部性统计 + 缓存设计建议 |

#### 3.2.4 `expert_trace_hook.py` — 路由追踪钩子

| 项目 | 说明 |
|------|------|
| **文件** | `collectors/expert_trace_hook.py` (~350 行) |
| **作用** | 在 MoE router 层注册 forward hook，捕获每 token 每层的专家路由决策 |
| **核心类** | `ExpertTraceCollector` — hook 管理 + JSONL 输出 |
| | `TraceEvent` — 单条路由记录 |
| **核心函数** | `compute_maf_from_trace()` — 从滑动窗口计算 MAF(K) |

---

### 3.3 调度策略层 (`controllers/`)

抽象调度接口 + 多种策略实现。

#### 3.3.1 `interface.py` — 调度器抽象接口

| 项目 | 说明 |
|------|------|
| **文件** | `controllers/interface.py` (~80 行) |
| **核心抽象** | `SchedulerController` — 三大决策方法 |
| | `decide_speculation_k(state) → int` — 决定推测深度 |
| | `decide_memory_partition(state) → MemoryPartition` — 决定内存分区 |
| | `decide_prefetch(state) → List[ExpertId]` — 决定预取专家 |
| **数据模型** | `Phase` (PREFILL/DECODE)、`RequestState`、`RuntimeState` |

#### 3.3.2 `static_governor.py` — 静态规则调度器 (v0)

| 项目 | 说明 |
|------|------|
| **文件** | `controllers/static_governor.py` (~140 行) |
| **策略** | 基于内存压力 + acceptance rate 的规则：高压力 → K=1，中压力 → K=2，否则 K=4 |
| **内存分区** | 默认 55% KV / 25% expert / 20% speculative，高压力下 KV↑ expert↓ |

#### 3.3.3 `phase_aware_governor.py` — 阶段感知调度器 (v1)

| 项目 | 说明 |
|------|------|
| **文件** | `controllers/phase_aware_governor.py` (~100 行) |
| **策略** | prefill 阶段保守 (K=1, 优先 TTFT)，decode 阶段激进 (K=4, 优先 TPOT) |
| **内存** | prefill 时 KV 预算倾斜，decode 时 expert 预算增加 |

#### 3.3.4 `memory_partition_controller.py` — 动态内存分区 (v2)

| 项目 | 说明 |
|------|------|
| **文件** | `controllers/memory_partition_controller.py` (~100 行) |
| **策略** | 指数平滑追踪压力变化，动态调整 expert/spec/KV 预算比例 |

#### 3.3.5 `prefetch_policy.py` — 预取策略

| 项目 | 说明 |
|------|------|
| **文件** | `controllers/prefetch_policy.py` (~250 行) |
| **v1** | `AcceptanceAwarePrefetchPolicy` — score = p_need × p_accept × benefit − cost |
| **v2** | `FrontierAwarePrefetchPolicy` — 扩展 v1，加入共享分数、深度比率、浪费估计 |

#### 3.3.6 `fallbacks.py` — 降级策略

| 项目 | 说明 |
|------|------|
| **文件** | `controllers/fallbacks.py` (~60 行) |
| **降级链** | controller → native_eagle3 → no_sd → observe_only |

---

### 3.4 运行时入口 (`vllm_moe_sd_scheduler/`)

#### 3.4.1 `cli.py` — 命令行启动器

| 项目 | 说明 |
|------|------|
| **文件** | `vllm_moe_sd_scheduler/cli.py` (~200 行) |
| **功能** | 构建 `vllm serve` 命令 + 启动 MetricsSidecar + 信号管理 |
| **入口** | `launch_server()` → build_runtime() → start vLLM + sidecar |

#### 3.4.2 `config.py` — 调度器配置

| 项目 | 说明 |
|------|------|
| **文件** | `vllm_moe_sd_scheduler/config.py` (~50 行) |
| **核心类** | `SchedulerConfig` — model_path, workload_profile, policy_name, feature_flags |

#### 3.4.3 `entrypoints.py` — 运行时绑定

| 项目 | 说明 |
|------|------|
| **文件** | `vllm_moe_sd_scheduler/entrypoints.py` (~140 行) |
| **功能** | 根据 FeatureFlags 有条件地初始化各适配器，组装 RuntimeBinding |

#### 3.4.4 `feature_flags.py` — 功能开关

| 项目 | 说明 |
|------|------|
| **文件** | `vllm_moe_sd_scheduler/feature_flags.py` (~30 行) |
| **开关** | enable_controller / enable_prefetch / enable_memory_partition / observation_only |

---

### 3.5 实验脚本 (`scripts/`)

#### 3.5.1 `specmoe_server_v2.py` — 可重配置实验服务器

| 项目 | 说明 |
|------|------|
| **文件** | `scripts/specmoe_server_v2.py` (~200 行) |
| **功能** | 单次模型加载，通过 REST API 运行时切换 SpecMoE 配置 |
| **配置 ID** | 1=passthrough, 2=vanilla SD, 3=+dedup, 4=+SDD, 5=+cache |
| **API** | `POST /specmoe/configure` — 切换配置 |
| | `GET /specmoe/status` — 查询 hook 统计 |

#### 3.5.2 `bench_serving_sharegpt.py` — ShareGPT 顺序单请求基准测试

| 项目 | 说明 |
|------|------|
| **文件** | `scripts/bench_serving_sharegpt.py` (~200 行) |
| **功能** | 用真实 ShareGPT 对话数据做**顺序单请求**延迟基准（每请求完成后再发下一个，无并发积压） |
| **指标** | Output TPS (tok/s)、TTFT、E2E latency (mean/p50/p90/p99)、acceptance length τ̄ |
| **注意** | 不使用 Poisson 到达率（serving QPS 模式），原因：系统最大 QPS ≈ 0.05 req/s，高 QPS 只会触发排队，导致 TTFT 包含大量等待时间，与 MAF 无关 |
| **依赖** | aiohttp — 异步 HTTP 客户端 |

#### 3.5.3 `run_full_experiment.py` — 全实验编排器

| 项目 | 说明 |
|------|------|
| **文件** | `scripts/run_full_experiment.py` (~300 行) |
| **功能** | 双进程架构：Process A (Config 1 No-SD baseline) / Process B (Configs 2-5，复用同一 EAGLE-3 服务进程) |
| **数据集** | ShareGPT 真实对话，50 prompts（第 1 个作为 warmup），max_tokens=128/256/512 |
| **发送方式** | 顺序单请求（sequential），不设置 request_rate，每请求完成后立即发下一个 |
| **切换方式** | 通过 `POST /specmoe/configure` REST API 运行时切换配置，无需重启服务 |

---

### 3.6 验证实验 (`scripts/validation/`)

| 文件 | 用途 |
|------|------|
| `exp0_collect_expert_trace.py` | 收集专家路由 trace |
| `exp1_maf_analysis.py` | MAF(K) 分析 |
| `exp2_dedup_savings.py` | 去重节省量估算 |
| `exp3_sdd_feasibility.py` | SDD 可行性验证 |
| `exp4_temporal_locality.py` | 时间局部性分析 |
| `plot_fig1_moe_tax.py` | Figure 1 MoE Tax 可视化 |

---

### 3.7 测试 (`tests/`)

| 文件 | 覆盖模块 | 测试数 |
|------|---------|--------|
| `test_fused_moe_hook.py` | FusedMoEHook | 8 |
| `test_triton_spec_moe.py` | TritonSpecMoE | 6 |
| `test_spec_fused_moe.py` | SpecFusedMoE | 5 |
| `test_expert_cache.py` | ExpertWeightCache | 8 |
| `test_sdd.py` | SDD detector | 7 |
| `test_specmoe_engine.py` | SpecMoEEngine | 8 |
| `test_spec_integrator_real.py` | SpecIntegrator | 7 |
| `test_vllm_hooks_real.py` | SchedulerHookManager | 4 |
| `test_cli.py` | CLI launcher | 7 |
| `test_static_governor.py` | StaticGovernor | 4 |
| `test_phase_aware_governor.py` | PhaseAwareGovernor | 4 |
| `test_prefetch_policy.py` | PrefetchPolicy | 5 |
| `test_memory_partition_controller.py` | MemoryPartition | 3 |
| `test_controller_interface.py` | Interface | 3 |
| `test_fallbacks.py` | Fallbacks | 3 |
| `test_maf_computation.py` | MAF 计算 | 4 |
| **合计** | | **~86 tests** |

---

## 四、核心数据流

### 4.1 推理时数据流 (单个 decode step)

```
1. EAGLE-3 Draft Phase
   └─ Proposer 生成 K 个 draft tokens (通过 /dev/shm 读取动态 K)
   └─ ExpertTraceCollector 记录 draft 路由
   └─ PrefetchPolicy 根据 draft 路由预取专家到 GPU cache

2. Verify Phase (K+1 tokens 并行)
   └─ FusedMoEHook 检测 batch_size ≥ threshold → 启动 verify 模式
   └─ 逐层执行:
       ├─ SDD 监控 router logits → 冻结已发散 token
       ├─ SpecFusedMoE 去重 active tokens 的专家集合
       ├─ ExpertWeightCache 命中 → GPU 直取；miss → CPU→GPU 传输
       └─ Triton kernel 执行 fused_moe (去重后)

3. Acceptance & Feedback
   └─ AcceptanceTracker 更新 EMA acceptance rate
   └─ MetricsSidecar 刷新 RuntimeState
   └─ Controller 决定下一步 K + 内存分区 + 预取列表
```

### 4.2 实验数据流

```
ShareGPT Dataset (50 prompts)
        │
        ▼  顺序单请求（sequential，无并发积压）
bench_serving_sharegpt.py
        │                              │
        ▼                              ▼
vLLM Server (specmoe_server_v2)    REST API /specmoe/configure
        │                          在 C1→C2→C3→C4→C5 间切换
        ▼
每请求指标: TTFT, E2E latency, output tokens
        │
        ├─ 聚合: Output TPS (tok/s), mean TTFT, E2E p50/p99
        │
        └─ 内部指标: FusedMoEHook stats
                    (dedup ratio, 实测 MAF, SDD frozen tokens, cache hit rate)
                    写入 results/{config_id}/metrics.json
```

---

## 五、实验设计

### 5.1 适用场景边界

我们的问题成立需要以下三个条件**同时满足**：

| 条件 | 说明 | 本机情况 |
|------|------|----------|
| **C1：模型超 VRAM** | 专家权重 57 GB > A6000 48 GB | ✅ 约 9 GB 卸载到 CPU |
| **C2：PCIe 是瓶颈（β 大）** | A6000 PCIe 25 GB/s vs HBM 768 GB/s，β ≈ 0.6 | ✅ 实测专家加载占 ~60% 延迟 |
| **C3：SD verify 放大专家加载量** | K+1 个 token 各自路由不同 expert → MAF(3)=2.985 | ✅ 实测 EAGLE-3 退化 17.1% |

**不适用场景**：A100/H100（VRAM 足够，无 offload，β≈0，SD 必然加速）；Mixtral-8x7B 在 250 GB CPU 内存的大 batch 高吞吐场景（SpecMoEOff 的目标）。

### 5.2 实验场景：顺序单请求（Latency Mode）

**为什么不用 serving（高并发 QPS）**：

系统物理约束决定了稳态并发 ≈ 1。单请求完成时间约 18s（128 output tokens），最大承载 QPS ≈ 0.05 req/s。以 QPS=1.0 运行只会造成排队积压，TTFT 数字包含大量等待时间，与 MAF 无关，无法说明任何问题。

MAF 问题来源于 **SD Verify Batch（K+1 tokens）**，与 Serving Batch Size 无关——即使系统里只有 1 个用户请求，每次 verify pass 依然处理 K+1 个 token，依然触发 MAF 放大。

**实验协议**：
- 数据集：ShareGPT 真实对话（`data/combined_sharegpt.json`），50 个 prompts
- 发送方式：**顺序发送**——每个请求完全完成后再发下一个（无并发积压）
- max_tokens：128（Short）/ 256（Medium）/ 512（Long）
- temperature：0（确定性，acceptance rate 更稳定）
- 重复次数：每个配置跑 3 遍取均值，第 1 个 prompt 作为 warmup 排除

**主实验完整参数表**：

| 参数 | 值 | 说明 |
|------|----|------|
| 服务器脚本 | `scripts/specmoe_server_v2.py` | vLLM 0.8.5 V1 engine，单次加载复用 |
| 模型 | `Qwen3-30B-A3B-Instruct-2507` | |
| 推测器 | `Qwen3-30B-A3B-Instruct-2507-speculator.eagle3` | EAGLE-3，仅 C2-C5 启用 |
| `--gpu-memory-utilization` | `0.85` | 主实验固定值；留 ~7 GB 给 expert cache |
| `--max-model-len` | `4096` | 覆盖所有 ShareGPT prompts |
| `--num-speculative-tokens` | `3`（K=3） | C2-C5 默认；C1 不设此参数 |
| `--speculative-disable-by-batch-size` | `1` | 禁用 vLLM 内置的动态关闭逻辑，由我们接管 |
| `max_tokens`（client） | **256**（主要）/ 128 / 512 | Short/Medium/Long 三档 |
| `temperature` | `0.0` | 确定性输出，acceptance rate 可复现 |
| `top_p` | `1.0` | |
| `num_prompts` | `50`（第 1 个 warmup 排除 → **49 有效**） | |
| 每配置重复次数 | `3` 遍，取 mean ± std | |
| 并发度 | `1`（顺序） | `asyncio.gather` 去掉，改为 `await send()` 逐个 |

**与相关工作对齐**：

| 工作 | Batch | 协议 |
|------|-------|------|
| MoE-SpAc（arXiv 2603.09983） | 1 | 顺序单请求，TPS + Latency |
| MoE-Spec（arXiv 2602.16052） | 1 | 顺序单请求，Speedup + Accuracy |
| SpecMoEOff（arXiv 2508.21706） | 数百 | 最大化 batch，Throughput |
| **我们（SpecMoE）** | **1** | **顺序单请求，TPS + TTFT + τ̄** |

### 5.3 指标体系

| 指标 | 含义 | 获取方式 |
|------|------|----------|
| **Output TPS (tok/s)** | 每秒生成的 output token 数，主要吞吐指标 | total_output_tokens / total_time |
| **Avg TTFT (ms)** | Time To First Token，感知延迟 | 每请求首 token 时刻 - 发送时刻 |
| **Avg E2E Latency (s)** | 单请求总耗时 | 最后一个 token 时刻 - 发送时刻 |
| **Acceptance Length τ̄** | 每次 SD round 平均接受的 draft token 数 | vLLM metrics / hook 统计 |
| **MAF (实测)** | 实际每 verify step 加载的唯一专家数 / 单 token 专家数 | FusedMoEHook 统计 |
| **Speedup over No-SD** | 相对 AR baseline 的加速比 | TPS_system / TPS_nosd |

### 5.4 对比配置（5 个，增量消融）

| Config ID | 名称 | 启用组件 | 说明 |
|-----------|------|----------|------|
| **C1** | **No-SD Baseline** | 纯 AR | 无推测解码，每步 1 token，基准线 |
| **C2** | **EAGLE-3 Vanilla** | EAGLE-3 SD (K=3) | 未经优化的标准 SD，**预期退化 ~17%** |
| **C3** | **+ SpecFusedMoE** | C2 + Expert Dedup | 加入 verify batch 内跨 token 去重 |
| **C4** | **+ SDD** | C3 + SDD | 加入层间早停，冻结已发散 draft token |
| **C5** | **Full SpecMoE** | C4 + Expert Cache + Prefetch | 完整系统，预期超过 C1 baseline |

**各配置的具体开关参数（通过 `POST /specmoe/configure` 切换，无需重启）**：

| Config | `config_id` | `enable_dedup` | `enable_sdd` | `enable_cache` | `enable_prefetch` | K | max_tokens |
|--------|-------------|----------------|--------------|----------------|-------------------|---|------------|
| C1 | `1` | — | — | — | — | N/A | 256 |
| C2 | `2` | `false` | `false` | `false` | `false` | **3** | 256 |
| C3 | `3` | `true` | `false` | `false` | `false` | 3 | 256 |
| C4 | `4` | `true` | `true` | `false` | `false` | 3 | 256 |
| C5 | `5` | `true` | `true` | `true` | `true` | 3 | 256 |

> C1 以独立服务进程运行（无 EAGLE-3 推测器），C2-C5 共用同一 EAGLE-3 服务进程，通过 API 运行时切换，模型权重只加载一次。

**C2 必须出现退化**，这是论文的核心 claim（MAF 导致 SD 在 offloaded MoE 上默认有害），C3→C5 展示我们逐步修复它。

### 5.5 补充实验

| 实验 | 目的 | 具体参数 |
|------|------|----------|
| **K 值敏感性** | 展示无 SpecMoE 时 K 越大越差 | Config=C2，K∈{1,2,3,4,6}，max_tokens=256，50 prompts，temp=0，重复 3 次 |
| **output 长度敏感性** | 展示不同输出长度下的 speedup 变化 | Config=C1 vs C5，max_tokens∈{128,256,512}，50 prompts，temp=0 |
| **内存压力敏感性** | 展示 SpecMoE 跨 GPU 利用率稳定 | Config=C1 vs C5，`--gpu-memory-utilization`∈{0.70,0.80,0.90}，max_tokens=256 |
| **MAF 实测验证** | 对比理论 MAF(K) 曲线与实测值 | Config=C2，K∈{1,2,3,4,5}，读取 FusedMoEHook `unique_experts / single_token_experts` |
| **SDD 精度** | 报告 precision/recall（冻结正确率）| Config=C4，SDD 统计字段 `frozen_correct / frozen_total` |
| **Expert Cache 命中率** | cold / steady-state / with-prefetch 三阶段 | Config=C5，按请求序号分桶（前 5 = cold，后 45 = steady） |

---

## 六、环境依赖

| 组件 | 版本 |
|------|------|
| vLLM | 0.8.5 (V1 engine) |
| PyTorch | 2.6.0+cu124 |
| Triton | 3.2.0 |
| transformers | 4.51.3+ |
| speculators | 0.3.0 (EAGLE-3 训练/加载) |
| 模型 | Qwen3-30B-A3B-Instruct-2507 |
| 推测器 | Qwen3-30B-A3B-Instruct-2507-speculator.eagle3 |
| GPU | NVIDIA RTX A6000 (48 GB) |

---

## 七、统计概览

| 类别 | 文件数 | 总行数 (约) |
|------|--------|------------|
| adapters/ | 11 | ~2,900 |
| collectors/ | 4 | ~850 |
| controllers/ | 6 | ~800 |
| vllm_moe_sd_scheduler/ | 4 | ~420 |
| scripts/ (核心) | 3 | ~700 |
| scripts/validation/ | 6 | ~600 |
| tests/ | 16 | ~1,500 |
| tools/ | 14 | ~1,200 |
| **合计** | **~64** | **~8,970** |

---

## 八、设计原则

1. **插件化 (Plugin Architecture)**：所有优化通过 monkey-patch + hook 实现，不修改 vLLM 源码（除少量兼容性补丁），可随 vLLM 版本升级解耦
2. **渐进式启用 (Feature Flags)**：每个组件独立开关，支持 A/B 消融实验
3. **零开销观察 (Observation Mode)**：`observation_only=True` 时仅采集数据不改变行为
4. **安全降级 (Graceful Fallback)**：controller 异常时自动降级到原生 EAGLE-3 或关闭 SD
5. **数据驱动 (Metrics-Driven)**：Sidecar 持续采集 → Controller 基于实时指标决策 → 形成闭环
