#!/usr/bin/env python3
"""
Create SpecMoE paper project issues on GitHub.
Uses GitHub REST API with git credential token.

Usage:
    python scripts/create_specmoe_issues.py [--dry-run]
"""
import json
import subprocess
import sys
import time
import urllib.request
import urllib.error

REPO = "Li-changwu/MoE-SD"
API = f"https://api.github.com/repos/{REPO}/issues"


def get_token():
    proc = subprocess.run(
        ["git", "credential", "fill"],
        input="protocol=https\nhost=github.com\n",
        capture_output=True, text=True,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("password="):
            return line.split("=", 1)[1]
    raise RuntimeError("Cannot extract GitHub token from git credential")


TOKEN = get_token()

# ============================================================================
# Issue definitions — SpecMoE paper task breakdown
# ============================================================================
ISSUES = [
    # ── Phase 0: Bench 审查与 MAF 实证 ──────────────────────────────────────
    {
        "title": "[SpecMoE] P0-1: 现有 Bench 审查与实验评估",
        "labels": ["specmoe", "P0-foundation"],
        "body": """## 🎯 目标

审查现有实验的正确性，确认数据可用于论文。

## 📋 审查清单

### bench_runner.py
- [ ] 确认 `vllm bench` 命令参数正确传递
- [ ] 确认 `--speculative-config` JSON 序列化正确
- [ ] 确认 dataset mode 和 synthetic mode 的分支逻辑

### 实验数据审查
- [ ] `pilot_pressure_sweep`: K=3, P∈{0.90,0.80,0.70} → speedup=1.307x, acc=52.8%
  - 数据来源: 5 条硬编码 prompt × 2 repeats = 10 次推理
  - ⚠️ **问题**: 样本量过小（10条），且 prompt 过于简单和同质
- [ ] `concurrent_sweep`: B∈{8000..600}, 50 concurrent prompts → speedup=0.904-0.948x
  - 数据来源: combined_sharegpt.json 前 50 条（Alpaca+Dolly, ShareGPT格式）
  - ✅ 更合理的负载
- [ ] `baseline_bench`: no-SD 23.56 tok/s vs EAGLE-3 19.53 tok/s
  - ⚠️ K=2 (SPEC_TOKENS=2), 仅 8 prompts
  
### 数据源审查
- [ ] combined_sharegpt.json: 200 条，Alpaca(100)+Dolly(100)
- [ ] alpaca_subset.jsonl / dolly_subset.jsonl: instruction-following pairs
- [ ] pilot 实验的 5 条硬编码 prompt 需要替换为 dataset-based

## 🔍 已发现的问题

1. **pilot_pressure_sweep 样本量不足**: 5 条 prompt × 2 = 10 次推理不具统计意义
2. **baseline_bench K=2**: 与 concurrent_sweep K=3 不一致，需要统一
3. **缺少 K sweep**: 没有 K=1,2,3,4,5 的系统对比
4. **缺少 expert trace 数据**: moe_trace_collector 有代码但无实际收集数据
5. **缺少 MAF 实证数据**: 无法直接计算每层 expert union size

## ✅ 验收标准

- [ ] 完成审查报告
- [ ] 标记可直接用于论文的数据
- [ ] 列出需要补充/重跑的实验
""",
    },
    {
        "title": "[SpecMoE] P0-2: MAF 实证测量——Expert Trace 数据采集",
        "labels": ["specmoe", "P0-foundation", "experiment"],
        "body": """## 🎯 目标

实际测量 Qwen3-30B MoE Expert routing 数据，计算 MAF(K) 的真实值。

## 📐 理论背景

$$\\text{MAF}(K) = \\frac{\\sum_{l=1}^{L}|\\cup_{i=0}^{K} E_i^{(l)}|}{L \\cdot k}$$

其中 L=48 层, k=8 (top-8 routing), N=128 experts/layer

## 🔬 实验方案

### Step 1: 修改 vLLM 捕获 Router Logits
- 在 `Qwen3MoeForCausalLM` 的 forward 中 hook MoE router
- 每个 token × 每层输出 top-k expert indices  
- 写入 JSONL trace 文件

### Step 2: 运行 Trace 收集
- 使用 combined_sharegpt.json 200 条 prompt
- 收集 no-SD decode 阶段所有 token 的 routing decision
- 输出格式: `{token_idx, layer_id, top_k_experts: [int×8]}`

### Step 3: 计算 MAF
- 对于连续 K+1 个 token 的滑动窗口，计算每层 expert union size
- 统计 MAF(K) for K=1,2,3,4,5
- 同时计算 token 间 expert overlap 分布
- 计算 per-token marginal MAF (mMAF)

### Step 4: 与理论对比
- 理论随机值: MAF_random(K=2)=2.82 (i.i.d. uniform)
- 实验反推值: MAF≈2.93 (from speedup=0.906)
- 实际测量值: ?

## 📦 输出

- `results/moe_trace/expert_routing_trace.jsonl`
- `results/moe_trace/maf_by_k.csv` (K, mean_MAF, std_MAF, p25, p75)
- `results/moe_trace/maf_per_layer.csv` (layer, K, mean_union_size)
- `results/moe_trace/mmaf_distribution.csv` (per-token marginal MAF)

## ✅ 验收标准

- [ ] 完成 router hook 实现
- [ ] 收集 ≥1000 tokens 的 routing trace
- [ ] MAF(K=2) 实测值与理论值误差 <20%
- [ ] 生成 MAF vs K 曲线图
""",
    },
    {
        "title": "[SpecMoE] P0-3: 统一 K-sweep 对比实验",
        "labels": ["specmoe", "P0-foundation", "experiment"],
        "body": """## 🎯 目标

在统一实验条件下跑 K=0,1,2,3,4,5 的系统对比，为 MAF 公式提供 ground truth。

## 🔬 实验设计

### 固定条件
- Model: Qwen3-30B-A3B-Instruct-2507
- Speculator: EAGLE-3
- GPU: RTX A6000 48GB
- CPU offload: 30 GB
- Dataset: combined_sharegpt.json, 100 prompts
- max_model_len: 2048
- max_tokens: 256

### 变量
- K ∈ {0(no-SD), 1, 2, 3, 4, 5}
- Memory Pressure: gpu_blocks ∈ {8000(low), 1500(medium), 800(high)}

### 指标
- Throughput (tok/s)
- Avg latency (s)
- Acceptance rate (%)
- Mean acceptance length
- Peak KV usage (%)

## 📐 验证 MAF 公式

对每个 (K, blocks) 组合:
1. 测量 speedup = throughput_SD / throughput_noSD
2. 用公式反推 MAF: `MAF = 1 + (α̅K+1)/S - 1 - γ) / β`
3. 对比 P0-2 中直接测量的 MAF

## 📦 输出

- `results/k_sweep_unified/results.json`
- `results/k_sweep_unified/speedup.csv`
- `results/k_sweep_unified/maf_empirical.csv`
- Plot: Speedup vs K (分 pressure level)
- Plot: MAF_measured vs MAF_formula

## ✅ 验收标准

- [ ] 完成 6×3 = 18 组实验
- [ ] Speedup 曲线体现非单调特性
- [ ] MAF 公式预测误差 <15%
- [ ] 确定 breakeven α̅ 条件
""",
    },
    # ── Phase 1: SpecFusedMoE 算子 ──────────────────────────────────────────
    {
        "title": "[SpecMoE] P1-1: vLLM fused_moe 算子分析——识别 SD Multi-token 去重缺口",
        "labels": ["specmoe", "P1-operator", "analysis"],
        "body": """## 🎯 目标

分析 vLLM 0.17.1 中 `fused_moe` Triton kernel 的实现，确认是否存在 SD 场景下
多 token expert 重复加载的问题（MAF_naive = K+1 vs MAF_dedup）。

## 🔬 分析步骤

### Step 1: 定位 vLLM fused_moe 代码
- `vllm/model_executor/layers/fused_moe/` 
- 找到 Triton kernel 入口: `fused_moe()`, `invoke_fused_moe_kernel()`
- 理解 token dispatch → expert compute → output gather 流程

### Step 2: 分析 SD verify 阶段的调用路径
- EAGLE-3 verify: target model forward 对 K+1 个 token 做 parallel forward
- 这 K+1 个 token 是怎么传入 fused_moe 的？
  - 是 batched 一起还是逐 token？
  - 如果 batched，kernel 内部是否做了 cross-token expert dedup？

### Step 3: Profile 实际行为
- 在 verify 阶段对 fused_moe 做 CUDA profiling
- 测量: 实际加载了多少唯一 expert（vs 理论最小值 |∪E_i|）
- 计算: MAF_actual vs MAF_dedup vs MAF_naive

## 📐 理论分析

若 vLLM 没做 dedup:
```
MAF_naive(K=2) = 3  (每token独立加载8个expert = 24次)
MAF_dedup(K=2) ≈ 2.93 (去重后约23.4次)
节省: (24-23.4)/24 ≈ 2.5% (K=2时dedup收益不大)
```

但在 CPU offload 下 PCIe 是瓶颈:
```
MAF_naive(K=4) = 5 → 40 expert loads
MAF_dedup(K=4) = 4.40 → 35.2 expert loads  
节省: 12% PCIe 带宽
```

## 📦 输出

- 分析报告: `docs/fused_moe_analysis.md`
- Profile 数据: expert load count per verify round
- 确认: vLLM 是否已有 dedup（如有则调整研究方向）

## ✅ 验收标准

- [ ] 完成 vLLM fused_moe 代码 walkthrough
- [ ] 确认 SD verify 的 token dispatch 机制
- [ ] Profile 实际 expert load 次数
- [ ] 确定 SpecFusedMoE 的优化空间大小
""",
    },
    {
        "title": "[SpecMoE] P1-2: SpecFusedMoE Kernel 原型实现",
        "labels": ["specmoe", "P1-operator", "implementation"],
        "body": """## 🎯 目标

基于 P1-1 的分析结果，实现 SD-aware 的 fused_moe kernel 原型。

## 依赖

- #P1-1 (确认优化空间存在)
- #P0-2 (MAF 实测数据)

## 🏗️ 设计方案

### 核心改动: Token-aware Expert Dispatch

```python
def spec_fused_moe(
    hidden_states,     # [K+1, hidden_size]  -- verify batch
    router_logits,     # [K+1, num_experts]
    expert_weights,    # expert param pointers
    acceptance_probs,  # [K] -- draft token 预估接受概率 (可选)
):
    # 1. Token-aware routing: 收集所有 token 的 top-k experts
    all_experts = topk(router_logits, k=8)  # [K+1, 8]
    
    # 2. Expert dedup: 计算唯一 expert 集合
    unique_experts = union(all_experts)  # size ≤ (K+1)*k
    
    # 3. Priority ordering (可选): 按 acceptance_prob 排序
    # 高概率 accept 的 token 的 expert 优先加载
    
    # 4. Batch expert loading: 一次性加载唯一集合
    loaded = load_experts(unique_experts)
    
    # 5. Dispatched computation: 每个 token 用自己的 experts
    output = dispatch_and_compute(hidden_states, all_experts, loaded)
    
    return output
```

### 实现路径

**Option A: Triton Kernel 修改**
- 修改 vLLM 的 `fused_moe_kernel` Triton template
- 在 dispatch 前添加 dedup 逻辑
- 优点: 性能最优; 缺点: 工程量大

**Option B: Python Wrapper**  
- 在调用 fused_moe 前做 dedup pre-processing
- 用 torch operations 实现 expert set union
- 优点: 快速原型; 缺点: 有额外开销

**推荐: 先 Option B 验证可行性，再 Option A 优化性能**

## 📦 输出

- `adapters/spec_fused_moe.py` — Python wrapper 原型
- `adapters/spec_fused_moe_triton.py` — Triton kernel (if Option A)
- Unit test: `tests/test_spec_fused_moe.py`

## ✅ 验收标准

- [ ] 原型通过正确性测试（output 与标准 fused_moe 一致）
- [ ] 测量 dedup 带来的 expert load 减少量
- [ ] 性能对比: SpecFusedMoE vs vanilla fused_moe 在 K=3 下的延迟
""",
    },
    # ── Phase 2: Layer-wise Early Termination ───────────────────────────────
    {
        "title": "[SpecMoE] P2-1: Router Logits 分歧检测可行性研究",
        "labels": ["specmoe", "P2-early-term", "analysis"],
        "body": """## 🎯 目标

验证是否可以通过 router logits 的层间分歧来预测 draft token 的拒绝。

## 📐 理论基础

对于 draft token t_j，如果 target model 在第 l 层的 router logits 与 draft model
预期的差异很大（KL散度高），则该 token 大概率会被拒绝。

Speculation Divergence Detector (SDD):
```
divergence(t_j, l) = KL(router_target^(l)(t_j) || router_draft^(l)(t_j))
if divergence > θ for N consecutive layers → early terminate t_j
```

## 🔬 实验方案

### Step 1: 收集 Draft vs Target Router Logits
- 在 EAGLE-3 verify 过程中 hook 每层 router
- 对 K 个 draft token，记录:
  - target router logits (48 layers × 128 experts)
  - 对应的 acceptance/rejection 结果

### Step 2: 分析分歧信号
- 计算每层的 KL 散度 / L2 距离 / top-k overlap
- 画出: accepted tokens vs rejected tokens 的分歧分布
- 找到: 最早可区分的层 (layer l*)

### Step 3: 量化 SDD 指标
- Precision: 判定为 reject 的 token 中实际被 reject 的比例
- Recall: 实际被 reject 的 token 中被 SDD 检测到的比例
- 最优 (θ, N) 参数
- 平均检测层 l*（越早越好）

## 📦 输出

- `results/sdd_analysis/divergence_trace.jsonl`
- `results/sdd_analysis/sdd_precision_recall.csv`
- `results/sdd_analysis/optimal_layer_threshold.json`
- Plot: divergence distribution (accept vs reject, per layer)
- Plot: precision-recall curve vs threshold θ

## ✅ 验收标准

- [ ] precision >80%, recall >60% 则 SDD 可行
- [ ] 平均检测层 l* < L/2 = 24 则有显著收益
- [ ] 若不满足 → 关闭此研究方向，在 issue 中记录 negative result
""",
    },
    {
        "title": "[SpecMoE] P2-2: Layer-wise Early Termination 实现与评估",
        "labels": ["specmoe", "P2-early-term", "implementation"],
        "body": """## 🎯 目标

实现基于 SDD 的 layer-wise early termination，评估 MAF 降低效果。

## 依赖

- #P2-1 (SDD 可行性确认，precision/recall 达标)
- #P0-2 (MAF 实测数据，per-token mMAF)

## 🏗️ 设计方案

### Early Termination 逻辑

在 target model 逐层执行时:
```python
for layer_idx in range(48):
    # 正常执行 attention (成本低)
    hidden = self_attn(hidden)
    
    # MoE 前检查: 对每个 draft token 判断是否终止
    if layer_idx >= min_check_layer:
        router_logits = compute_router(hidden)
        for j in draft_tokens:
            if sdd.should_terminate(j, layer_idx, router_logits):
                # 将 token j 的 hidden state 标记为 frozen
                # 后续层只执行 attn，不执行 MoE FFN
                frozen_tokens.add(j)
    
    # 只对 non-frozen tokens 执行 MoE
    active_hidden = hidden[~frozen_mask]
    moe_out = fused_moe(active_hidden, router_logits[~frozen_mask])
    # ... merge back
```

### 节省量化

每 freeze 1 个 token 在第 l* 层:
```
ΔC = Σ_{l=l*+1}^{48} |E_j^(l) \\ ∪_{i≠j} E_i^(l)| × C_load
   = (48 - l*) × mMAF(t_j) × k × C_load
```

## 📦 输出

- `adapters/layer_early_terminator.py`
- `tests/test_early_terminator.py`
- Evaluation results:
  - MAF_effective (with early termination) vs MAF_original
  - Output quality degradation (if any)
  - Wall-clock speedup delta

## ✅ 验收标准

- [ ] MAF 降低 ≥15% (from ~2.93 to ≤2.49 at K=2)
- [ ] Output 质量: perplexity 增加 <1%
- [ ] 若 MAF 降低 <10% 或质量损失 >3% → 记录 negative result 并关闭
""",
    },
    # ── Phase 3: Cross-Phase Expert Cache ───────────────────────────────────
    {
        "title": "[SpecMoE] P3-1: Expert Temporal Locality 实证分析",
        "labels": ["specmoe", "P3-expert-cache", "analysis"],
        "body": """## 🎯 目标

量化 SD draft→verify→draft 循环中 expert 的时间局部性。

## 📐 关键指标

### Inter-round Expert Overlap
相邻 verify round 的 expert 集合重叠率:
```
overlap(r, r-1) = |experts(r) ∩ experts(r-1)| / |experts(r)|
```

### Expert Reuse Distance
某 expert 被使用后，到下次被使用经过的 round 数。

### Draft-Target Routing Correlation
Draft model 预测的 expert 与 target model 实际使用的 expert 的重合度:
```
correlation = |E_draft ∩ E_target| / |E_target|
```

## 🔬 实验方案

1. 使用 P0-2 收集的 routing trace
2. 模拟 K=2,3,4 的 SD round 划分
3. 计算上述三个指标的分布

## 📦 输出

- `results/expert_locality/inter_round_overlap.csv`
- `results/expert_locality/reuse_distance_dist.csv`
- `results/expert_locality/draft_target_correlation.csv`
- Plot: overlap rate distribution
- Plot: reuse distance CDF

## ✅ 验收标准

- [ ] inter-round overlap > 40% → expert cache 有价值
- [ ] draft-target correlation > 50% → draft-guided prefetch 有价值
- [ ] 若 overlap < 30% → expert cache 收益有限，降低优先级
""",
    },
    # ── Phase 4: 论文集成 ───────────────────────────────────────────────────
    {
        "title": "[SpecMoE] P4-1: 端到端集成与对比实验",
        "labels": ["specmoe", "P4-integration", "experiment"],
        "body": """## 🎯 目标

将可行的创新点集成到统一系统中，与 baseline 和竞品方法做端到端对比。

## 依赖

- P0 全部完成
- P1 / P2 / P3 中可行的部分

## 🔬 对比方案

### Baselines
1. **no-SD**: 原生 Qwen3-30B 推理
2. **vLLM-EAGLE3**: 原生 vLLM + EAGLE-3 (当前 benchmark)
3. **Cascade** (if reproducible): Utility-driven K selection
4. **MoE-Spec** (if reproducible): Expert budgeting

### Our Methods (组合消融)
5. **SpecMoE-dedup**: SpecFusedMoE kernel only
6. **SpecMoE-SDD**: + layer early termination
7. **SpecMoE-cache**: + expert temporal cache
8. **SpecMoE-full**: all combined

### Metrics
- Throughput (tok/s)
- TPOT p50/p95
- TTFT p50/p95
- Effective MAF
- Output quality (perplexity, match rate vs no-SD)

### Workloads
- Short: prompt=128, output=64
- Medium: prompt=512, output=128
- Long: prompt=2048, output=256

### Memory Pressure
- Low: gpu_blocks=8000
- Medium: gpu_blocks=1500
- High: gpu_blocks=800

## 📦 输出

- Full comparison table
- Speedup bar chart
- MAF reduction waterfall chart
- Ablation table (which component contributes how much)

## ✅ 验收标准

- [ ] SpecMoE-full 相对 vLLM-EAGLE3 有 ≥20% throughput 提升
- [ ] 或在 MAF 降低方面有显著改善
- [ ] 所有实验可复现
""",
    },
    {
        "title": "[SpecMoE] P4-2: 论文撰写——Motivation & Background",
        "labels": ["specmoe", "P4-integration", "paper"],
        "body": """## 🎯 目标

撰写论文 Section 1 (Introduction) + Section 2 (Background & Motivation)。

## 📄 大纲

### Section 1: Introduction
- MoE 模型兴起 → 参数规模暴涨 → 需要 CPU offload
- SD 是 dense LLM 推理加速的标准手段
- **但 SD on MoE 反而可能减速** ← 核心 observation
- 现有工作都在调度层面 → 我们提出算子层面的 co-design
- 贡献列表

### Section 2: Background & Motivation
- 2.1 MoE Architecture (Qwen3-30B 参数)
- 2.2 Speculative Decoding 原理
- 2.3 **The MoE Amplification Problem** ← MAF 理论
  - 公式推导
  - Breakeven condition
  - 与实验数据吻合
- 2.4 Why Scheduling-Level Solutions Are Insufficient

## 📦 输出

- `docs/paper/01_introduction.md`
- `docs/paper/02_background.md`
- Key figures: MAF vs K curve, speedup vs acceptance rate contour

## ✅ 验收标准

- [ ] MAF 理论推导完整、可验证
- [ ] Motivation 有实验数据支撑
- [ ] 与 Related Work 清晰区隔
""",
    },
    {
        "title": "[SpecMoE] P4-3: 论文撰写——System Design & Evaluation",
        "labels": ["specmoe", "P4-integration", "paper"],
        "body": """## 🎯 目标

撰写论文 Section 3 (Design) + Section 4 (Evaluation) + Section 5 (Related Work)。

## 📄 大纲

### Section 3: SpecMoE Design
- 3.1 System Overview (architecture diagram)
- 3.2 SpecFusedMoE Kernel (dedup + priority loading)
- 3.3 Speculation Divergence Detector (SDD)  
- 3.4 Expert Temporal Cache (if applicable)
- 3.5 Integration with vLLM

### Section 4: Evaluation
- 4.1 Experimental Setup
- 4.2 End-to-end Comparison
- 4.3 MAF Reduction Analysis
- 4.4 Ablation Study
- 4.5 Sensitivity Analysis (K, memory pressure, model size)

### Section 5: Related Work
- Speculative Decoding: EAGLE, Medusa, SpecInfer
- MoE Inference: SiDA, COMET, Samoyeds
- MoE + SD: MoESD, SP-MoE, Cascade, MoE-Spec, SpecMoEOff, MoE-SpAc
- Operator Co-design: SpecAttn

## ✅ 验收标准

- [ ] 论文完整初稿
- [ ] 所有图表可复现
- [ ] Target: OSDI'26 / ATC'26
""",
    },
]


def create_issue(issue_data, dry_run=False):
    """Create a single GitHub issue via REST API."""
    if dry_run:
        print(f"  [DRY-RUN] Would create: {issue_data['title']}")
        return {"number": 0, "title": issue_data["title"]}

    payload = json.dumps({
        "title": issue_data["title"],
        "body": issue_data["body"],
        "labels": issue_data.get("labels", []),
    }).encode()

    req = urllib.request.Request(
        API,
        data=payload,
        headers={
            "Authorization": f"token {TOKEN}",
            "Content-Type": "application/json",
            "Accept": "application/vnd.github.v3+json",
        },
        method="POST",
    )

    try:
        resp = urllib.request.urlopen(req, timeout=30)
        data = json.loads(resp.read())
        return data
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"  ERROR: {e.code} {body}")
        return None


def update_issue_state(issue_number, state, dry_run=False):
    """Update issue state (open/closed)."""
    if dry_run:
        print(f"  [DRY-RUN] Would update #{issue_number} → {state}")
        return

    url = f"https://api.github.com/repos/{REPO}/issues/{issue_number}"
    payload = json.dumps({"state": state}).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"token {TOKEN}",
            "Content-Type": "application/json",
        },
        method="PATCH",
    )
    resp = urllib.request.urlopen(req, timeout=30)
    return json.loads(resp.read())


def main():
    dry_run = "--dry-run" in sys.argv

    print(f"SpecMoE Issue Creator — {len(ISSUES)} issues")
    print(f"Repo: {REPO}")
    print(f"Mode: {'DRY-RUN' if dry_run else 'LIVE'}")
    print()

    # Ensure labels exist
    if not dry_run:
        for label_name in ["specmoe", "P0-foundation", "P1-operator",
                           "P2-early-term", "P3-expert-cache",
                           "P4-integration", "experiment", "analysis",
                           "implementation", "paper"]:
            try:
                payload = json.dumps({
                    "name": label_name,
                    "color": {
                        "specmoe": "0052CC",
                        "P0-foundation": "D4C5F9",
                        "P1-operator": "FBCA04",
                        "P2-early-term": "B60205",
                        "P3-expert-cache": "0E8A16",
                        "P4-integration": "006B75",
                        "experiment": "C2E0C6",
                        "analysis": "E4E669",
                        "implementation": "BFD4F2",
                        "paper": "F9D0C4",
                    }.get(label_name, "EDEDED"),
                }).encode()
                req = urllib.request.Request(
                    f"https://api.github.com/repos/{REPO}/labels",
                    data=payload,
                    headers={
                        "Authorization": f"token {TOKEN}",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                urllib.request.urlopen(req, timeout=15)
                print(f"  Created label: {label_name}")
            except urllib.error.HTTPError:
                pass  # label already exists
            time.sleep(0.3)

    created = []
    for i, issue in enumerate(ISSUES):
        print(f"\n[{i+1}/{len(ISSUES)}] Creating: {issue['title']}")
        result = create_issue(issue, dry_run=dry_run)
        if result:
            num = result.get("number", "?")
            print(f"  ✓ Created #{num}")
            created.append({"number": num, "title": issue["title"]})
        else:
            print(f"  ✗ Failed")
        time.sleep(1)  # rate limit

    print(f"\n{'='*60}")
    print(f"Created {len(created)}/{len(ISSUES)} issues:")
    for c in created:
        print(f"  #{c['number']}: {c['title']}")

    # Save mapping for reference
    mapping_path = "results/specmoe_issues.json"
    with open(mapping_path, "w") as f:
        json.dump(created, f, indent=2, ensure_ascii=False)
    print(f"\nIssue mapping saved to {mapping_path}")


if __name__ == "__main__":
    main()
