#!/bin/bash
# 安装公共数据集基准测试所需的依赖

set -e

echo "=== 安装公共数据集基准测试依赖 ==="
echo ""

# 检查是否在虚拟环境中
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  警告：未检测到 Python 虚拟环境"
    echo "建议使用 conda 或 venv 创建隔离环境后再安装"
    echo ""
    read -p "是否继续安装到系统环境？(y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 安装 HuggingFace datasets
echo "📦 安装 HuggingFace datasets..."
pip install datasets

# 验证安装
echo ""
echo "✅ 验证安装..."
python3 -c "import datasets; print(f'datasets 版本：{datasets.__version__}')"

echo ""
echo "=== 安装完成 ==="
echo ""
echo "使用方法："
echo "  make download-sharegpt DATASET_SAMPLES=10"
echo "  make bench-sharegpt"
echo ""
echo "或直接运行："
echo "  ./scripts/download_and_bench.sh sharegpt 10"
