# BriskMoE ATC 2026 论文撰写规划

---

## 1. 论文全称

> **BriskMoE: Restoring Speculative Speedups for Offloaded Mixture-of-Experts Inference**

- "Accelerating MoE Inference" — 核心贡献是加速 MoE 推理，不限定 offload 场景
- "with Speculative Decoding" — SD 是手段/工具，而非被改进的对象

---

## 2. SC → ATC 迁移：关键差异

| 维度 | SC '26 | ATC '26 |
|------|--------|---------|
| 格式 | ACM sigconf (acmart) | **USENIX** (usenix-2e) |
| 页数 | 10 pages + refs | **12 pages** + refs + appendix |
| 评审 | Double-blind | **Double-blind** |
| 侧重 | HPC / 计算密集 | **系统实现 / 实用性 / 真实部署** |
| 审稿偏好 | 模型分析 + 大规模实验 | **实现深度 + 可复现性 + 实际影响** |

**核心调整方向：**
1. **新增 Implementation 章节** — ATC 审稿人期望看到系统实现细节（vLLM 插件架构、代码行数、侵入性评估）
2. **扩展 Cross-Architecture 实验** — GPT-OSS-120B (MXFP4) 结果作为泛化性证据，这是 SC 版本没有的
3. **新增 Artifact 描述** — ATC 有 Artifact Evaluation track，需准备可复现的实验包
4. **强化 "实用性" 论述** — 从 "能否部署" 角度论述，非纯理论分析

---

## 3. 章节结构规划

### 整体架构（12 页正文 + references + appendix）

```
§1  Introduction                          ~2.0 pages
§2  Background and Problem Setting        ~1.0 page
§3  Analytical Model                      ~1.5 pages
§4  System Design                         ~3.0 pages
§5  Implementation                        ~1.0 page   ← 新增
§6  Evaluation                            ~3.0 pages  ← 扩展 cross-arch
§7  Related Work                          ~0.5 page
§8  Conclusion                            ~0.25 page
    References                            (不计页数)
    Appendix                              (不计页数, 可选)
```

---

### §1 Introduction（~2.0 pages）

**目标：** 建立问题、展示失败现象、提出方案、列出贡献。

| 段落 | 内容 | 写作要点 |
|------|------|----------|
| ¶1 宏观背景 | MoE 模型 + 单 GPU 服务场景 + SD 对 dense 模型的成功 | 1-2句定位：latency-sensitive, single-request serving。引出 MoE + commodity GPU 的矛盾 |
| ¶2 问题展示 | SD 在 offloaded MoE 上的失败现象 + motivation figure | **Figure 1**: (a) working set 扩展 $W_{AR}=8 → W_{SD}≈19.3$; (b) 4-way 对比柱状图。核心数据：SD 仅 1.17× |
| ¶3 现有工作局限 | 三个 gap：batch-centric 分析不适用、uncached offloading 无缓存层、AR-optimized cache 不适应 SD | 逐条点名批判，每条 1-2 句 |
| ¶4 Key insight | 缓存是必要但远不充分的：即使 η→1 也只有 29% compute efficiency。两个隐藏瓶颈：(1) cache-hit path overhead 2.4×, (2) SD-induced transient overflow → cascading eviction | 先承认缓存是已有技术，再揭示"缓存+SD"特有的两个新瓶颈，引出 Unblock + Protect |
| ¶5 贡献列表 | 3 条 bullet points | (1) 首个 bandwidth-regime 分析 + $M^*$ 阈值; (2) 两阶段系统设计 (Unblock + Protect) + 非侵入 vLLM 插件; (3) 两个模型 × 三个 workload 的实验 |
| ¶6 论文组织 | 各章导航 | 一句话 |

**与 SC 版本差异：**
- 贡献 (3) 增加 cross-architecture 实验描述
- ¶5 增加 "implemented as a non-intrusive vLLM plugin with <800 LoC of core logic"

---

### §2 Background and Problem Setting（~1.0 page）

**目标：** 形式化服务场景、定义符号、建立 cached offloading baseline。

| 小节 | 内容 |
|------|------|
| 2.1 Serving Setting | 定义：single-request, single-GPU, 延迟优先。MoE 模型参数 ($L$, $N$, $k$, $m_e$)。CPU offloading 约束 |
| 2.2 Cached Offloading Baseline | 定义 scratchpad execution model 的四阶段：$T_{sync}$, $T_{copy}$, $T_{kernel}$, $T_{load}$。命名 notation |
| 2.3 AR vs SD Under Offloaded MoE | AR: $W_{AR}=k$。SD: $W_{SD} = |\bigcup top\text{-}k_i|$，推导期望公式 |

**与 SC 版本差异：** 基本复用，适配 USENIX 格式。

---

### §3 Analytical Model（~1.5 pages）

**目标：** 推导 $M^*$ 阈值 + 两个 Observation，建立设计动机。

| 小节 | 内容 |
|------|------|
| 3.1 Memory Threshold $M^*$ | 推导 $S \ge \bar{W}_{SD}$ → $M^* = M_{base} + L_{off} \cdot \bar{W}_{SD} \cdot m_e$ |
| 3.2 Observation 1: Hit-Path Overhead | cache hit ≠ HBM-speed 执行；分解 $(T_{copy}+T_{sync})/T_{kernel} ≈ 2.4×$ |
| 3.3 Observation 2: Transient Overflow | SD working set 波动 → cascading eviction → $\eta$ 下降 |
| 3.4 Design Implications | 映射表：Finding → Bottleneck → Mechanism |

**与 SC 版本差异：** 基本复用。可考虑增加 GPT-OSS-120B 的 $M^*$ 计算作为泛化示例。

---

### §4 System Design（~3.0 pages）

**目标：** 详述两阶段设计 (Unblock + Protect) + 架构总览图。$M^*$ 作为前提条件在 §3 中已建立。

| 小节 | 内容 | 页数 |
|------|------|------|
| 4.0 Overview | 架构图（Figure 2）+ Table: Finding→Mechanism 映射 | 0.3 |
| 4.1 Unblock: Pool-Direct | 消除 scratchpad copy；remap topk_ids → slot IDs | 0.5 |
| 4.2 Unblock: TASER | 冻结 expert-to-slot mapping + 指数回退验证 | 0.5 |
| 4.3 Unblock: Oracle Cross-Layer Prefetch | 利用 TASER frozen mapping 预取下一层 experts | 0.4 |
| 4.4 Unblock: Kernel Tuning | Triton block size / warp 调优 | 0.3 |
| 4.5 Protect: Draft-Guided Preloading (DIPP) | 利用 draft 生成窗口预取 verify experts | 0.5 |
| 4.6 Assumptions and Boundaries | $M^*$ 前提、mapping 稳定性假设、适用范围 | 0.5 |

**与 SC 版本差异：**
- 4.1 增加 Scale/Bias Pool 子段（量化模型的 pool-direct 需要同时 swap scales/biases）
- 4.5 增加 DIPP 与 SACR/ELP 的对比讨论（实验证明 LRU + DIPP 优于复杂策略）

---

### §5 Implementation（~1.0 page）★ 新增

**目标：** 展示系统工程深度，这是 ATC 审稿人最关注的章节之一。

| 小节 | 内容 |
|------|------|
| 5.1 vLLM Plugin Architecture | 非侵入集成：`general_plugins` 入口 → `Worker.load_model` monkey-patch → `forward_impl` 替换。零修改 vLLM 核心代码 |
| 5.2 Memory Management | (1) Per-layer expert cache pool (GPU pinned); (2) Scratch tensors 共享; (3) 量化模型的 scale/bias pool; (4) Memory accounting: 手动注入 `model_memory_usage` 防止 KV cache 过分配导致 OOM |
| 5.3 Quantized Model Support | MXFP4 Marlin 路径：将 Marlin-repacked weights 移至 CPU-pinned（GPU 上仅保留 scales/biases）。Scale/bias pool-direct swap 确保量化模型正确性 |
| 5.4 Code Complexity | 核心逻辑 ~800 LoC (`elmm_plugin.py`)，辅助模块 ~400 LoC（plugin entry, draft hook, config）。完全 Python 层，无 C++/CUDA 修改 |

**价值：** ATC 审稿人重视实现深度和可复现性。此章节展示 BriskMoE 不仅是理论设计，而是一个可部署的生产级系统。

---

### §6 Evaluation（~3.0 pages）

**目标：** 端到端对比 + bandwidth-regime 验证 + 消融 + cross-architecture 泛化 + 敏感性分析。

| 小节 | 内容 | 数据来源 |
|------|------|----------|
| 6.1 Setup | 硬件、模型、baseline 定义 (B1-B3 + Ours) | — |
| 6.2 End-to-End (Qwen3-30B) | **Table 1**: B1=2.08, B2=2.44, B3=6.13, Ours=9.98 (4.80×) | `results/ar_vs_sd/`, humaneval |
| 6.3 Cross-Workload | **Table 2**: ShareGPT/GSM8K/HumanEval 三 workload 对比 | `results/main_table/` |
| 6.4 Bandwidth-Regime Validation | **Table 3 + Figure 3**: 24-44 GiB sweep，三 regime 验证 | `results/memory_sweep/` |
| 6.5 Ablation | **Table 4**: 优化链 UVA→cache→PD→TASER→prefetch→tune + acceptance rate | `results/ablation/` |
| 6.6 Protect Evaluation | DIPP micro-results: 78.3% accuracy, 72μs overhead, +15.6% η | `results/elp_dipp_ablation/` |
| **6.7 Cross-Architecture** ★ 新增 | **Table 5**: GPT-OSS-120B (MXFP4, 36L, 128E, top-4) 结果 | `results/gptoss_cross_arch/` |
| 6.8 Sensitivity | 内存、speculation depth、prompt shape 敏感性 | 各 sweep 数据 |
| 6.9 Summary of Findings | 4-5 条总结 | — |

**§6.7 Cross-Architecture 详细规划：**

这是 ATC 版本相对 SC 版本最重要的新增实验。

```
GPT-OSS-120B (117B params, 5.1B active)
- 36 layers, 128 experts/layer, top-4 routing
- MXFP4 quantization (Marlin backend), BF16 attention
- EAGLE-3 speculator, K=3

| Config                     | TPS Mean | vs AR  |
|---------------------------|----------|--------|
| AR + ELMM (baseline)      | 2.40     | —      |
| SD + ELMM (scratch)       | 11.52    | +380%  |
| SD + DIPP + TASER (PD)    | 12.01    | +400%  |
```

**叙事重点：**
1. BriskMoE 跨架构泛化：不仅适用于 Qwen3 (BF16, top-8)，也适用于 GPT-OSS (MXFP4, top-4)
2. 量化模型场景的工程挑战和解决方案（Marlin CPU-pinned patch, scale/bias pool）
3. 不同 top-k 路由策略下 working set 扩展比例不同，但 bandwidth-regime 现象一致

---

### §7 Related Work（~0.5 page）

| 小节 | 对比论文 |
|------|---------|
| 7.1 Speculative Decoding | Leviathan+Chen (2023), EAGLE/EAGLE-2/EAGLE-3, SpecInfer, Medusa |
| 7.2 MoE Inference & Offloading | MoE-Infinity, Pre-gated MoE, Mixtral offloading, DeepSpeed-MoE, Tutel |
| 7.3 MoE + SD | SpecMoE-Offload, SP-MoE, MoE-SpAc, Speculating Experts |
| 7.4 Operator-Level Optimization | FlashAttention, PagedAttention, vLLM |

**与 SC 版本差异：** 增加 7.3 中最新的 MoE+SD 工作（MoE-SpAc ICML'25, Speculating Experts 等）。

---

### §8 Conclusion（~0.25 page）

复用 SC 版本核心内容，新增：
- "We further validate BriskMoE on GPT-OSS-120B with MXFP4 quantization, demonstrating cross-architecture generality."
- Future work: multi-GPU distributed serving, other quantization formats (GPTQ, AWQ), longer context scenarios.

---

## 4. Figure / Table 清单

| 编号 | 类型 | 内容 | 状态 |
|------|------|------|------|
| Fig 1 | Motivation | (a) Working set bar chart (b) 4-way throughput bar | ✅ 已有 |
| Fig 2 | Architecture | BriskMoE 系统架构图 | ✅ 已有 |
| Fig 3 | Phase Transition | 24-44 GiB memory sweep + regime 标注 | ✅ 已有 |
| Fig 4 | Ablation Chain | 优化链可视化（optional: 可用 table 替代） | ⬜ 可选 |
| Fig 5 | Cross-Arch | GPT-OSS-120B vs Qwen3-30B 对比柱状图 | ⬜ **新增** |
| Tab 1 | Main Result | B1/B2/B3/Ours 端到端对比 | ✅ 已有 |
| Tab 2 | Cross-Workload | ShareGPT/GSM8K/HumanEval | ✅ 已有 |
| Tab 3 | Memory Sweep | 24-44 GiB sweep results | ✅ 已有 |
| Tab 4 | Ablation | UVA→cache→PD→TASER→prefetch→tune | ✅ 已有 |
| Tab 5 | Protect | DIPP micro-results | ✅ 已有 |
| Tab 6 | Cross-Arch | GPT-OSS-120B 结果表 | ⬜ **新增** |
| Tab 7 | Notation | 符号表 | ✅ 已有 |
| Tab 8 | Design Map | Finding→Mechanism 映射 | ✅ 已有 |

---

## 5. 写作优先级与时间线

### Phase 1: 框架迁移（1-2 天）
- [ ] 创建 USENIX 格式模板 (`usenix-2e.sty`)
- [ ] 迁移各章节 .tex 文件到 ATC 目录
- [ ] 调整页面格式、字体、margin

### Phase 2: 新增内容撰写（3-5 天）
- [ ] **§5 Implementation** — 全新章节，约 1 page
- [ ] **§6.7 Cross-Architecture** — 新增实验+表格+分析，约 0.5 page
- [ ] **§1 Introduction** — 更新贡献列表，增加 cross-arch 描述
- [ ] **§4.1 Pool-Direct** — 增加 scale/bias pool 子段
- [ ] **§8 Conclusion** — 更新 cross-arch 描述

### Phase 3: 内容打磨（2-3 天）
- [ ] 统一符号表记（USENIX 风格）
- [ ] 调整 Figure 尺寸适配新格式
- [ ] 精简各章节以满足 12 页限制
- [ ] Cross-reference 检查
- [ ] 参考文献补全（新增最新 MoE+SD 工作）

### Phase 4: Artifact 准备（1-2 天）
- [ ] 整理可复现脚本（benchmark runner + config）
- [ ] 编写 Artifact Appendix
- [ ] README: 环境配置 + 一键运行指令

---

## 6. 核心论点回顾

为确保所有章节围绕统一论点，记录核心叙事链：

```
问题：SD 对 offloaded MoE 几乎无效 (仅 1.17×)
  ↓
原因：SD 扩大 working set (8→19.3)，加剧 PCIe 瓶颈
  ↓
洞察：关键杠杆不是提升算术强度，而是切换带宽 regime (PCIe→HBM, 30.7×)
  ↓
前提：$M^* $ 阈值 → per-layer expert cache 保证缓存空间充足（已有技术，非贡献）
  ↓
方案：BriskMoE 两阶段
  ├── Unblock: Pool-Direct + TASER + Oracle Prefetch → 消除 cache-path 开销
  └── Protect: DIPP draft-guided preload → 维持 SD 下的 cache 覆盖率
  ↓
结果：Qwen3-30B 4.80× (2.08→9.98), GPT-OSS-120B 5.00× (2.40→12.01)
```
