#!/bin/bash
# RT-DETR 多权重推理服务启动脚本

set -euo pipefail

# ==================== 按需修改以下参数 ====================
# 权重目录: 自动读取目录下的 .jit/.pt/.pth 文件，文件名作为权重名
WEIGHTS_DIR=""

# 如果有额外的权重不在目录中，可在此处追加，格式 name=/path/to/model.jit
declare -a EXTRA_WEIGHTS=(
    "rtdetr=/data_all/share/Huawei/src/RT-DETR/rtdetr_pytorch/output/rtdetr_r50vd_6x_coco/eval/rtdetr_r50vd_100.jit"
)

# 类别名称文件，格式 weight_name=/path/to/class_names.txt
declare -a CLASS_NAMES=(
    # "rtdetr=/path/to/class_names.txt"
)

DEFAULT_WEIGHT="rtdetr"  # 默认使用的权重名，留空则自动取扫描到的第一个
DEVICE="cuda:0"          # 推理设备
INPUT_SIZE="640 640"     # 输入图像尺寸 (H W)
NUM_CLASSES=80           # 类别数量
CONF=0.35                # 置信度阈值
IOU=0.45                 # NMS IoU 阈值
MAX_DET=300              # 最大检测框数量
HOST="0.0.0.0"           # 服务监听地址
PORT=8811                # 服务监听端口
LOG_LEVEL="INFO"         # 日志等级
RUN_BACKGROUND="n"       # y: 后台运行, n: 前台运行
LOG_FILE="rtdetr_multi_server_$(date +%Y%m%d_%H%M%S).log"  # 后台运行时输出的日志文件

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
            echo "❌ 额外权重格式错误，应为 name=/path/to/model.jit: $item"
            exit 1
        fi
        local path="${item#*=}"
        check_path "$path" "额外权重"
    done
}

show_config() {
    printf '=%.0s' {1..80}; echo
    echo "🚀 RT-DETR 多权重服务配置"
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
    if [ "${#CLASS_NAMES[@]}" -gt 0 ]; then
        echo "🏷️  类别名称:"
        for cn in "${CLASS_NAMES[@]}"; do
            echo "   - $cn"
        done
    fi
    echo "⭐ 默认权重: ${DEFAULT_WEIGHT:-<自动选择>}"
    echo "📐 输入尺寸: $INPUT_SIZE | 类别数: $NUM_CLASSES"
    echo "🎯 conf: $CONF | iou: $IOU | max_det: $MAX_DET"
    echo "🖥️  设备: $DEVICE"
    echo "🌐 服务: http://$HOST:$PORT"
    echo "📝 日志: $LOG_FILE"
    printf '=%.0s' {1..80}; echo
}

build_command() {
    CMD=(python rtdetr_multi_server.py
        --device "$DEVICE"
        --input_size $INPUT_SIZE
        --num_classes "$NUM_CLASSES"
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

    if [ "${#CLASS_NAMES[@]}" -gt 0 ]; then
        for cn in "${CLASS_NAMES[@]}"; do
            CMD+=(--class_names "$cn")
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
