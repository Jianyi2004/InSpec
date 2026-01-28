#!/bin/bash
# 多图客户端脚本（支持单图逐推 & 多图一次性推理）
# 基于 vllm_client_multi_mode.py，支持两轮推理及不同的 Server / Model

# 默认配置
SERVER_URL="http://localhost:9191"                # 第一轮推理服务器
SUMMARY_SERVER_URL="http://localhost:9191"        # 总结服务器（可选，默认跟随 SERVER_URL）
MODEL="/data_all/share/models/Qwen3-VL-32B-Instruct"
SUMMARY_MODEL="/data_all/share/models/Qwen3-VL-32B-Instruct"
PROMPT_FILE="/data_all/lyh/LLaMA-Factory_1124/prompts/BBU_split/prompt_v6_2.txt,/data_all/lyh/LLaMA-Factory_1124/prompts/BBU_danban/test.txt"
SYSTEM_PROMPT="/data_all/lyh/LLaMA-Factory_1124/prompts/BBU/system_prompt.txt"
SUMMARY_SYSTEM_PROMPT=""
SUMMARY_PROMPT="/data_all/lyh/LLaMA-Factory_1124/prompts/BBU_split/summary_prompt.txt"
INFERENCE_MODE="single"                            # 可选：single | multi
DINO_SERVER_URL="http://127.0.0.1:8808"
YOLO_SERVER_URL="http://127.0.0.1:8810"

if [ $# -eq 0 ]; then
    echo "使用方法："
    echo "  $0 <image_path|image_dir|image1,image2,...> [prompt_file] [system_prompt] [summary_prompt] [summary_system_prompt] [inference_mode]"
    echo ""
    echo "示例："
    echo "  $0 /path/to/img1.jpg,/path/to/img2.jpg                 # 多图推理（默认 multi）"
    echo "  $0 /path/to/images/                                    # 指定文件夹"
    echo "  $0 /path/to/img.jpg prompts/prompt.txt                 # 指定单个 prompt（单图可结合 --inference_mode single）"
    echo "  $0 /path/to/imgs/ prompt1.txt,prompt2.txt              # 多个 prompt，首轮分别推理后合并"
    echo "可通过设置 DINO_SERVER_URL 调整 DINO 服务地址 (默认: $DINO_SERVER_URL)"
    echo "  $0 /path/to/imgs/ prompts/prompt.txt \"system\" \"sum\" \"sum_system\" single"
    exit 1
fi

IMAGE_INPUT=$1
if [ $# -ge 2 ]; then
    PROMPT_FILE=$2
fi
if [ $# -ge 3 ]; then
    SYSTEM_PROMPT=$3
fi
if [ $# -ge 4 ]; then
    SUMMARY_PROMPT=$4
fi
if [ $# -ge 5 ]; then
    SUMMARY_SYSTEM_PROMPT=$5
fi
if [ $# -ge 6 ]; then
    INFERENCE_MODE=$6
fi

if [ -d "$IMAGE_INPUT" ]; then
    echo "📁 检测到文件夹，使用其中所有图片..."
    IMAGE_ARGS="--image_dir \"$IMAGE_INPUT\""
elif [[ "$IMAGE_INPUT" == *","* ]]; then
    echo "📷 检测到逗号分隔的图片列表..."
    IMAGE_ARGS="--image_paths \"$IMAGE_INPUT\""
elif [ -f "$IMAGE_INPUT" ]; then
    echo "🖼️  检测到单张图片，包装为多图输入..."
    IMAGE_ARGS="--image_paths \"$IMAGE_INPUT\""
else
    echo "❌ 输入无效: $IMAGE_INPUT"
    exit 1
fi

echo "🚀 vLLM Multi Mode 推理"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🌐 推理 Server: $SERVER_URL"
echo "🤖 推理 Model: $MODEL"
echo "📝 Prompt: $PROMPT_FILE"
echo "⚙️  推理模式: $INFERENCE_MODE"
if [ -n "$DINO_SERVER_URL" ]; then
    echo "🦖 DINO Server: $DINO_SERVER_URL"
fi
if [ -n "$YOLO_SERVER_URL" ]; then
    echo "🟦 YOLO Server: $YOLO_SERVER_URL"
fi
if [ -n "$SYSTEM_PROMPT" ]; then
    echo "🔧 System Prompt: $SYSTEM_PROMPT"
fi
if [ -n "$SUMMARY_PROMPT" ]; then
    echo "🔄 启用总结"
    echo "📝 Summary Prompt: $SUMMARY_PROMPT"
    if [ -n "$SUMMARY_SYSTEM_PROMPT" ]; then
        echo "🔧 Summary System Prompt: $SUMMARY_SYSTEM_PROMPT"
    fi
    if [ -n "$SUMMARY_SERVER_URL" ]; then
        echo "🌐 Summary Server: $SUMMARY_SERVER_URL"
    fi
    if [ -n "$SUMMARY_MODEL" ]; then
        echo "🤖 Summary Model: $SUMMARY_MODEL"
    fi
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

CMD="python vllm_client_multi_mode.py \
    --server_url \"$SERVER_URL\" \
    --system_prompt \"$SYSTEM_PROMPT\" \
    --prompt_file \"$PROMPT_FILE\" \
    --dino_server_url \"$DINO_SERVER_URL\" \
    --yolo_server_url \"$YOLO_SERVER_URL\" \
    --temperature 0.1 \
    --model \"$MODEL\" \
    --max_tokens 28000 \
    --inference_mode \"$INFERENCE_MODE\" \
    $IMAGE_ARGS"

if [ -n "$SUMMARY_PROMPT" ]; then
    CMD="$CMD --summary_prompt \"$SUMMARY_PROMPT\""
fi
if [ -n "$SUMMARY_SYSTEM_PROMPT" ]; then
    CMD="$CMD --summary_system_prompt \"$SUMMARY_SYSTEM_PROMPT\""
fi
if [ -n "$SUMMARY_SERVER_URL" ]; then
    CMD="$CMD --summary_server_url \"$SUMMARY_SERVER_URL\""
fi
if [ -n "$SUMMARY_MODEL" ]; then
    CMD="$CMD --summary_model \"$SUMMARY_MODEL\""
fi

eval $CMD
