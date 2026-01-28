#!/bin/bash
# 多图异步批量推理脚本（single/multi 模式）

# 基础配置（按需修改）
SERVER_URL="http://10.249.42.141:9191"                # 第一轮推理服务器
# SERVER_URL="http://localhost:3822"
SUMMARY_SERVER_URL="http://10.249.42.141:9191"        # 总结服务器（可选，默认跟随 SERVER_URL）
MODEL="/data_all/share/models/Qwen3-VL-32B-Instruct"
SUMMARY_MODEL="/data_all/share/models/Qwen3-VL-32B-Instruct"
# TEST_DATA="/data_all/lyh/LLaMA-Factory_1124/data/8-BBU/12_24.json"
# TEST_DATA="/data_all/lyh/LLaMA-Factory_1124/extracted_outputs/multi.json"
TEST_DATA="/data_all/share/Huawei/tmp_data/LLaMA-Factory_1124/unify_final_3/eval_screw.json"
# PROMPT_FILE="/data_all/lyh/LLaMA-Factory_1124/prompts/DCDU/prompt_jiedi_v5.txt"
PROMPT_FILE="/data_all/lyh/LLaMA-Factory_1124/prompts/BBU_danban/test.txt"
SYSTEM_PROMPT="/data_all/lyh/LLaMA-Factory_1124/prompts/BBU/system_prompt.txt"
# SUMMARY_PROMPT="/data_all/lyh/LLaMA-Factory_1124/prompts/DCDU/summary_v2.txt"
SUMMARY_SYSTEM_PROMPT=""
INFERENCE_MODE="multi"   # single | multi
MAX_CONCURRENT=5
TEMPERATURE=0.1
MAX_TOKENS=26000
YOLO_SERVER_URL="http://127.0.0.1:8810"
RTDETR_SERVER_URL="http://127.0.0.1:8811"

TEST_NAME=$(basename "$TEST_DATA" | sed 's/\.[^.]*$//')
if [[ "$PROMPT_FILE" == *","* ]]; then
    PROMPT_NAME=""
    IFS=',' read -ra PROMPTS <<< "$PROMPT_FILE"
    for PROMPT in "${PROMPTS[@]}"; do
        BASE_NAME=$(basename "$PROMPT")
        BASE_NAME="${BASE_NAME%.*}"
        PROMPT_NAME+="${PROMPT_NAME:+_}${BASE_NAME}"
    done
else
    PROMPT_NAME=$(basename "$PROMPT_FILE")
    PROMPT_NAME="${PROMPT_NAME%.*}"
fi
OUTPUT_DIR="server_inference_results/${SYSTEM_PROMPT##*/}/${PROMPT_NAME}/${TEST_DATA##*/}"

echo "🚀 vLLM 多图异步批量推理"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🌐 推理 Server: $SERVER_URL"
echo "🤖 推理模型: $MODEL"
echo "🌐 总结 Server: $SUMMARY_SERVER_URL"
echo "🤖 总结模型: $SUMMARY_MODEL"
echo "🧪 测试数据: $TEST_DATA"
echo "📝 Prompt: $PROMPT_FILE"
echo "⚙️  模式: $INFERENCE_MODE"
echo "⚡ 并发数: $MAX_CONCURRENT"
echo "💾 输出目录: $OUTPUT_DIR"
echo "💾 实时输出结果: $OUTPUT_DIR/detailed_results.json"
if [ -n "$YOLO_SERVER_URL" ]; then
    echo "🟦 YOLO Server: $YOLO_SERVER_URL"
fi
if [ -n "$RTDETR_SERVER_URL" ]; then
    echo "🔷 RTDETR Server: $RTDETR_SERVER_URL"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

CMD=(python vllm_client_multi_mode_async.py
    --server_url "$SERVER_URL"
    --test_data_path "$TEST_DATA"
    --prompt_file "$PROMPT_FILE"
    --output_dir "$OUTPUT_DIR"
    --temperature "$TEMPERATURE"
    --max_tokens "$MAX_TOKENS"
    --model "$MODEL"
    --max_concurrent "$MAX_CONCURRENT"
    --inference_mode "$INFERENCE_MODE"
    --yolo_server_url "$YOLO_SERVER_URL"
    --rtdetr_server_url "$RTDETR_SERVER_URL"
)

if [ -n "$SYSTEM_PROMPT" ]; then
    CMD+=(--system_prompt "$SYSTEM_PROMPT")
fi
if [ -n "$SUMMARY_PROMPT" ]; then
    CMD+=(--summary_prompt "$SUMMARY_PROMPT")
fi
if [ -n "$SUMMARY_SYSTEM_PROMPT" ]; then
    CMD+=(--summary_system_prompt "$SUMMARY_SYSTEM_PROMPT")
fi
if [ -n "$SUMMARY_SERVER_URL" ]; then
    CMD+=(--summary_server_url "$SUMMARY_SERVER_URL")
fi
if [ -n "$SUMMARY_MODEL" ]; then
    CMD+=(--summary_model "$SUMMARY_MODEL")
fi

"${CMD[@]}"
