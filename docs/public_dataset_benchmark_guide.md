# 公共数据集基准测试指南

本指南介绍如何使用固定的公开数据集进行权威、可复现的基准测试，替代合成的随机工作负载。

## 为什么使用公共数据集？

- ✅ **权威性**：ShareGPT、Alpaca、Dolly 等是业界标准评测集
- ✅ **可复现性**：固定数据集确保实验结果可重复验证
- ✅ **真实性**：真实用户查询比合成数据更能反映实际性能
- ✅ **可比性**：便于与已发表研究结果对比

## 支持的数据集

| 数据集 | 来源 | 领域 | 推荐样本数 |
|--------|------|------|-----------|
| ShareGPT | anon8231489123/ShareGPT_Vicuna_unfiltered | 对话 | 50-500 |
| Alpaca | tatsu-lab/alpaca | 指令跟随 | 100-1000 |
| Dolly-15k | databricks/databricks-dolly-15k | 多任务指令 | 100-1000 |

## 快速开始

### 方法 1：使用 Make 命令（推荐）

```bash
# 下载 ShareGPT 数据集（默认 100 条样本）
make download-sharegpt

# 自定义样本数量
make download-sharegpt DATASET_SAMPLES=50

# 运行基准测试
make bench-sharegpt

# 一键完成下载 + 测试
make download-and-bench DATASET=sharegpt SAMPLES=100
```

### 方法 2：直接调用工具

```bash
# 下载数据集
python3 tools/dataset_downloader.py sharegpt --samples 100 --output data/sharegpt_sample.jsonl

# 运行基准测试
python3 tools/bench_runner.py configs/experiments/baseline_sharegpt.yaml
```

### 方法 3：使用快捷脚本

```bash
./scripts/download_and_bench.sh sharegpt 100
```

## 详细用法

### 下载数据集

```bash
# ShareGPT - 真实对话数据
python3 tools/dataset_downloader.py sharegpt \
    --samples 100 \
    --output data/sharegpt_sample.jsonl

# Alpaca - 指令跟随数据
python3 tools/dataset_downloader.py alpaca \
    --samples 200 \
    --output data/alpaca_sample.jsonl

# Dolly-15k - 多任务指令
python3 tools/dataset_downloader.py dolly \
    --samples 150 \
    --output data/dolly_sample.jsonl

# 自定义数据集（已有 JSONL 文件）
python3 tools/dataset_downloader.py custom \
    --input my_data.jsonl \
    --output data/custom_sample.jsonl
```

### 运行基准测试

```bash
# 使用预定义配置
python3 tools/bench_runner.py configs/experiments/baseline_sharegpt.yaml

# 或指定参数覆盖
python3 tools/bench_runner.py \
    --config configs/experiments/baseline_sharegpt.yaml \
    --num-prompts 50 \
    --mode offline_throughput
```

## 输出格式

数据集文件采用统一的 JSONL 格式：

```json
{
  "prompt": "用户输入的问题或指令",
  "completion": "期望的回答（可选）",
  "source": "数据集来源标识",
  "length_input": 输入 token 数（估计）,
  "length_output": 输出 token 数（估计）
}
```

## 配置说明

每个数据集对应一个 YAML 配置文件，位于 `configs/experiments/`：

- `baseline_sharegpt.yaml` - ShareGPT 配置
- `baseline_alpaca.yaml` - Alpaca 配置
- `baseline_dolly.yaml` - Dolly-15k 配置

关键参数：
- `dataset_file`: 数据集路径
- `num_prompts`: 使用的样本数量
- `gpu_memory_utilization`: GPU 显存占用率
- `max_model_len`: 最大序列长度
- `modes`: 基准测试模式（offline_throughput / online_latency）

## 故障排查

### 问题：`ModuleNotFoundError: No module named 'datasets'`

**解决**：安装 HuggingFace datasets 库
```bash
pip install datasets
```

### 问题：下载速度慢或失败

**解决**：使用镜像源
```bash
HF_ENDPOINT=https://hf-mirror.com python3 tools/dataset_downloader.py sharegpt
```

### 问题：显存不足

**解决**：降低 `gpu_memory_utilization` 或减少 `num_prompts`
```yaml
engine_args:
  gpu_memory_utilization: 0.7  # 从 0.9 降至 0.7
```

### 问题：提示词过长导致 OOM

**解决**：减小 `max_model_len` 或使用更短的样本
```yaml
engine_args:
  max_model_len: 2048  # 从 4096 降至 2048
```

## 最佳实践

1. **样本选择**：
   - 初步测试：10-50 条样本
   - 正式实验：100-500 条样本
   - 完整评测：1000+ 条样本

2. **可复现性**：
   - 固定随机种子（默认 42）
   - 记录数据集版本和采样数量
   - 保存完整的配置文件

3. **性能调优**：
   - 先用小样本（10-20 条）验证配置
   - 逐步增加样本量
   - 监控 GPU 利用率和显存占用

## 引用

如果使用这些数据集进行研究，请引用原始论文：

- **ShareGPT**: Vicuna 项目数据，参考 Vicuna 技术报告
- **Alpaca**: Taori et al., "Alpaca: A Strong, Affordable Alternative to GPT-3.5", 2023
- **Dolly**: Conover et al., "Free Dolly: Introducing the World's First Truly Open Instruction-Tuned LLM", 2023

## 相关文件

- [`tools/dataset_downloader.py`](../tools/dataset_downloader.py) - 数据集下载工具
- [`tools/bench_runner.py`](../tools/bench_runner.py) - 基准测试执行器
- [`scripts/download_and_bench.sh`](../scripts/download_and_bench.sh) - 一键脚本
- [`configs/experiments/`](../configs/experiments/) - 实验配置目录
