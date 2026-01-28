#!/bin/bash
# DINO + 多 MLP 推理服务启动脚本

set -euo pipefail

# ==================== 按需修改以下参数 ====================
DINO_MODEL_PATH="/data_all/share/models/dinov3-vit7b16-pretrain-lvd1689m"
MLP_CHECKPOINT_DIR="/data_all/lyh/LLaMA-Factory_1124/dino_pipeline/my_dino/checkpoints"
# 如果有额外的 checkpoint 不在目录中，可在此处追加，格式 name=/path/to/model.pth
declare -a EXTRA_MLP_CHECKPOINTS=(
    # "best=/data_all/lyh/LLaMA-Factory_1124/my_dino/best_model.pth"
)
# 针对特定 checkpoint 设置独立阈值，格式 name=0.6
declare -a CHECKPOINT_THRESHOLDS=(
    # "best=0.55"
)

DEFAULT_CHECKPOINT="best"
DEVICE="cuda:0"
THRESHOLD=0.5
HOST="0.0.0.0"
PORT=8808
LOG_LEVEL="INFO"
RUN_BACKGROUND="n"  # y: 后台运行, n: 前台运行
LOG_FILE="dino_multi_mlp_server_$(date +%Y%m%d_%H%M%S).log"

# ==================== 工具函数 ====================

check_path() {
    local path="$1"
    local desc="$2"
    if [ ! -e "$path" ]; then
        echo "❌ $desc 不存在: $path"
        exit 1
    fi
    echo "✅ $desc: $path"
}

show_config() {
    printf '=%.0s' {1..80}; echo
    echo "🚀 DINO 多 MLP 服务配置"
    printf '=%.0s' {1..80}; echo
    echo "📁 DINO 模型: $DINO_MODEL_PATH"
    echo "📁 MLP 目录: $MLP_CHECKPOINT_DIR"
    if [ "${#EXTRA_MLP_CHECKPOINTS[@]}" -gt 0 ]; then
        echo "➕ 额外 MLP:"
        for ckpt in "${EXTRA_MLP_CHECKPOINTS[@]}"; do
            echo "   - $ckpt"
        done
    fi
    if [ "${#CHECKPOINT_THRESHOLDS[@]}" -gt 0 ]; then
        echo "🎚️  特定阈值:"
        for item in "${CHECKPOINT_THRESHOLDS[@]}"; do
            echo "   - $item"
        done
    fi
    echo "⭐ 默认 checkpoint: $DEFAULT_CHECKPOINT"
    echo "🎯 阈值: $THRESHOLD"
    echo "🖥️  设备: $DEVICE"
    echo "🌐 服务: http://$HOST:$PORT"
    echo "📝 日志: $LOG_FILE"
    printf '=%.0s' {1..80}; echo
}

build_command() {
    CMD=(python dino_multi_mlp_server.py
        --dino_model_path "$DINO_MODEL_PATH"
        --device "$DEVICE"
        --threshold "$THRESHOLD"
        --host "$HOST"
        --port "$PORT"
        --log_level "$LOG_LEVEL"
    )

    if [ -d "$MLP_CHECKPOINT_DIR" ]; then
        CMD+=(--mlp_checkpoint_dir "$MLP_CHECKPOINT_DIR")
    fi

    if [ -n "$DEFAULT_CHECKPOINT" ]; then
        CMD+=(--default_checkpoint "$DEFAULT_CHECKPOINT")
    fi

    if [ "${#EXTRA_MLP_CHECKPOINTS[@]}" -gt 0 ]; then
        for ckpt in "${EXTRA_MLP_CHECKPOINTS[@]}"; do
            CMD+=(--mlp_checkpoint "$ckpt")
        done
    fi

    if [ "${#CHECKPOINT_THRESHOLDS[@]}" -gt 0 ]; then
        for item in "${CHECKPOINT_THRESHOLDS[@]}"; do
            CMD+=(--mlp_threshold "$item")
        done
    fi
}

start_server() {
    build_command
    echo "启动命令:"
    printf ' %q' "${CMD[@]}"; echo

    if [ "$RUN_BACKGROUND" = "y" ]; then
        echo "🚀 后台运行，日志输出到 $LOG_FILE"
        nohup "${CMD[@]}" > "$LOG_FILE" 2>&1 &
        echo "✅ 进程 PID: $!"
        echo "查看日志: tail -f $LOG_FILE"
    else
        echo "🚀 前台运行 (Ctrl+C 停止)"
        "${CMD[@]}"
    fi
}

# ==================== 主流程 ====================

check_path "$DINO_MODEL_PATH" "DINO 模型路径"
[ -d "$MLP_CHECKPOINT_DIR" ] && check_path "$MLP_CHECKPOINT_DIR" "MLP 目录"
show_config
start_server
