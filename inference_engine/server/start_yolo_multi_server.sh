#!/bin/bash
# YOLOv10 多权重推理服务启动脚本

set -euo pipefail

# ==================== 按需修改以下参数 ====================
# 权重目录: 自动读取目录下的 .pt/.pth 文件，文件名作为权重名
WEIGHTS_DIR="/data_all/lyh/LLaMA-Factory_1124/yolov10-main/runs/detect/train24_BBU/weights"

# 如果有额外的权重不在目录中，可在此处追加，格式 name=/path/to/model.pt
declare -a EXTRA_WEIGHTS=(
    # "bbu=/data_all/lyh/LLaMA-Factory_1124/yolov10-main/runs/detect/train24_BBU/weights/best.pt"
)

DEFAULT_WEIGHT="best"   # 默认使用的权重名，留空则自动取扫描到的第一个
DEVICE="cuda:0"         # 推理设备
CONF=0.35               # 置信度阈值
IOU=0.45                # NMS IoU 阈值
MAX_DET=300             # 最大检测框数量
HOST="0.0.0.0"          # 服务监听地址
PORT=8810               # 服务监听端口
LOG_LEVEL="INFO"        # 日志等级
RUN_BACKGROUND="n"      # y: 后台运行, n: 前台运行
LOG_FILE="yolo_multi_server_$(date +%Y%m%d_%H%M%S).log"  # 后台运行时输出的日志文件

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

check_extra_weights() {
    if [ "${#EXTRA_WEIGHTS[@]}" -eq 0 ]; then
        return
    fi
    for item in "${EXTRA_WEIGHTS[@]}"; do
        if [[ "$item" != *=* ]]; then
            echo "❌ 额外权重格式错误，应为 name=/path/to/model.pt: $item"
            exit 1
        fi
        local path="${item#*=}"
        check_path "$path" "额外权重"
    done
}

show_config() {
    printf '=%.0s' {1..80}; echo
    echo "🚀 YOLOv10 多权重服务配置"
    printf '=%.0s' {1..80}; echo
    if [ -n "$WEIGHTS_DIR" ]; then
        echo "📁 权重目录: $WEIGHTS_DIR"
    fi
    if [ "${#EXTRA_WEIGHTS[@]}" -gt 0 ]; then
        echo "➕ 额外权重:"
        for wt in "${EXTRA_WEIGHTS[@]}"; do
            echo "   - $wt"
        done
    fi
    echo "⭐ 默认权重: ${DEFAULT_WEIGHT:-<自动选择>}"
    echo "🎯 conf: $CONF | iou: $IOU | max_det: $MAX_DET"
    echo "🖥️  设备: $DEVICE"
    echo "🌐 服务: http://$HOST:$PORT"
    echo "📝 日志: $LOG_FILE"
    printf '=%.0s' {1..80}; echo
}

build_command() {
    CMD=(python yolo_multi_server.py
        --device "$DEVICE"
        --conf "$CONF"
        --iou "$IOU"
        --max_det "$MAX_DET"
        --host "$HOST"
        --port "$PORT"
        --log_level "$LOG_LEVEL"
    )

    if [ -n "$WEIGHTS_DIR" ]; then
        CMD+=(--weights_dir "$WEIGHTS_DIR")
    fi

    if [ -n "$DEFAULT_WEIGHT" ]; then
        CMD+=(--default_weight "$DEFAULT_WEIGHT")
    fi

    if [ "${#EXTRA_WEIGHTS[@]}" -gt 0 ]; then
        for wt in "${EXTRA_WEIGHTS[@]}"; do
            CMD+=(--weight "$wt")
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

if [ -n "$WEIGHTS_DIR" ]; then
    check_path "$WEIGHTS_DIR" "权重目录"
fi
check_extra_weights
show_config
start_server
