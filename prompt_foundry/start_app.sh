#!/bin/bash
# Prompt Foundry 启动脚本

# 激活 conda 环境

SERVER_URL="http://10.249.42.141:7878/v1"
MODEL="/data_all/share/models/Qwen3-VL-32B-Instruct"

# 设置 Python 路径
export PYTHONPATH="/home/intern10/InSpec/prompt_foundry:$PYTHONPATH"

# 启动 Web UI（增强版）
python web/app_enhanced.py \
    --server $SERVER_URL \
    --model $MODEL \
    --port 7860 \
    --host 0.0.0.0

echo "🎉 Prompt Foundry 已启动！"
echo "🌐 访问地址: http://localhost:7860"
