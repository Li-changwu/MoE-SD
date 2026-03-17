# 公共数据集基准测试 - 快速开始

## 🎯 解决的问题

之前的基线测试使用合成随机工作负载，存在以下问题：
- ❌ 难以理解：只看 "baseline" 不知道测的是什么
- ❌ 不可复现：每次运行的随机数据不同
- ❌ 缺乏权威性：合成数据无法代表真实场景

现在使用**固定的公开数据集**（ShareGPT、Alpaca、Dolly-15k），实现：
- ✅ 清晰明确：清楚标注数据来源和类型
- ✅ 完全可复现：固定数据集和随机种子
- ✅ 权威可信：业界标准评测集，可与论文结果对比

## 📁 新增文件清单

```
MoE-SD/
├── tools/
│   ├── dataset_downloader.py          # 数据集下载工具（新增）
│   └── bench_runner.py                # 基准测试执行器（已增强）
├── configs/experiments/
│   ├── baseline_sharegpt.yaml         # ShareGPT 配置（新增）
│   ├── baseline_alpaca.yaml           # Alpaca 配置（新增）
│   └── baseline_dolly.yaml            # Dolly-15k 配置（新增）
├── scripts/
│   ├── download_and_bench.sh          # 一键下载 + 测试脚本（新增）
│   └── install_deps.sh                # 依赖安装脚本（新增）
├── docs/
│   └── public_dataset_benchmark_guide.md  # 详细使用指南（新增）
└── PUBLIC_DATASETS_README.md          # 本文件（新增）
```

## 🚀 三步快速开始

### 步骤 1：安装依赖

```bash
# 使用 conda 环境（推荐）
conda activate your_env

# 安装 HuggingFace datasets
pip install datasets

# 或使用提供的安装脚本
./scripts/install_deps.sh
```

### 步骤 2：下载数据集

```bash
# 下载 ShareGPT（100 条样本用于测试）
make download-sharegpt DATASET_SAMPLES=100

# 或下载其他数据集
make download-alpaca DATASET_SAMPLES=200
make download-dolly DATASET_SAMPLES=150
```

### 步骤 3：运行基准测试

```bash
# 使用预定义配置运行
make bench-sharegpt

# 或一键完成下载 + 测试
make download-and-bench DATASET=sharegpt SAMPLES=100
```

## 💡 使用示例

### 示例 1：快速测试（10 条样本）

```bash
# 下载少量样本快速验证
./scripts/download_and_bench.sh sharegpt 10
```

### 示例 2：正式实验（500 条样本）

```bash
# 下载完整测试集
python3 tools/dataset_downloader.py sharegpt \
    --samples 500 \
    --output data/sharegpt_500.jsonl

# 修改配置文件指向新数据集
# 编辑 configs/experiments/baseline_sharegpt.yaml
# 将 dataset_file 改为：data/sharegpt_500.jsonl

# 运行基准测试
python3 tools/bench_runner.py configs/experiments/baseline_sharegpt.yaml
```

### 示例 3：使用自定义数据集

```bash
# 准备自己的 JSONL 文件（每行包含 prompt 和 completion 字段）
# 例如：my_custom_data.jsonl

# 转换为标准格式
python3 tools/dataset_downloader.py custom \
    --input my_custom_data.jsonl \
    --output data/custom_sample.jsonl

# 创建对应的配置文件
cp configs/experiments/baseline_sharegpt.yaml \
   configs/experiments/baseline_custom.yaml

# 编辑 baseline_custom.yaml，修改 dataset_file 路径
# 然后运行基准测试
python3 tools/bench_runner.py configs/experiments/baseline_custom.yaml
```

## 📊 输出结果

基准测试结果保存在 `results/` 目录：

```
results/
├── sharegpt/
│   ├── benchmark_results.json       # JSON 格式结果
│   └── benchmark_summary.txt        # 文本摘要
├── alpaca/
└── dolly/
```

结果包含：
- 吞吐量指标（tokens/s, requests/s）
- 延迟指标（P50, P90, P99）
- 模型配置信息
- 数据集元数据

## 🔧 常用命令

```bash
# 查看所有可用的 make 目标
make list-datasets

# 查看帮助
python3 tools/dataset_downloader.py --help
python3 tools/bench_runner.py --help

# 清理临时文件
rm -rf data/*.jsonl.tmp
```

## 🛠️ 故障排查

### 问题 1：缺少 `datasets` 模块

```bash
pip install datasets
```

### 问题 2：下载速度慢

使用国内镜像：
```bash
HF_ENDPOINT=https://hf-mirror.com make download-sharegpt
```

### 问题 3：显存不足

编辑配置文件，降低显存占用：
```yaml
engine_args:
  gpu_memory_utilization: 0.7  # 默认 0.9，可调低至 0.5
  max_model_len: 2048          # 默认 4096，可调低
```

### 问题 4：找不到配置文件

确认当前目录在 MoE-SD 根目录：
```bash
pwd  # 应该显示 /home/sage3/workspace/MoE-SD
ls configs/experiments/baseline_*.yaml
```

## 📖 详细文档

完整的使用指南、数据集介绍、最佳实践和引用格式请参考：

👉 [`docs/public_dataset_benchmark_guide.md`](docs/public_dataset_benchmark_guide.md)

## 🤝 贡献

如需添加新的数据集支持，请参考：
1. 在 `tools/dataset_downloader.py` 中添加下载函数
2. 创建对应的配置文件 `configs/experiments/baseline_<name>.yaml`
3. 更新本文档和详细指南

## 📄 许可证

继承主项目的许可证。注意各数据集可能有各自的使用条款：
- ShareGPT: CC BY 4.0
- Alpaca: CC BY-NC 4.0（非商业用途）
- Dolly-15k: MIT License
