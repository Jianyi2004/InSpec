#!/bin/bash
# 多图异步批量推理脚本（single/multi 模式）

# 基础配置（按需修改）
SERVER_URL="http://localhost:8009"                # 第一轮推理服务器
SUMMARY_SERVER_URL="http://localhost:8010"        # 总结服务器（可选，默认跟随 SERVER_URL）
MODEL="/data_all/share/models/Qwen3-VL-32B-Thinking"
SUMMARY_MODEL="/data_all/share/models/Qwen3-4B-Instruct-2507"
TEST_DATA="/home/intern10/LLaMA-Factory/data/2-guding/fail.json"
# TEST_DATA="/home/intern10/LLaMA-Factory/data/3-jiedi/kongjie.json"
PROMPT_FILE="/home/intern10/LLaMA-Factory/prompts/2-guding/prompt_v1.txt"
SYSTEM_PROMPT="/home/intern10/LLaMA-Factory/prompts/2-guding/system_prompt_v1.txt"
SUMMARY_PROMPT="/home/intern10/LLaMA-Factory/prompts/2-guding/prompt_sumup_v1.txt"
SUMMARY_SYSTEM_PROMPT=""
INFERENCE_MODE="multi"   # single | multi
MAX_CONCURRENT=10
TEMPERATURE=0.1
MAX_TOKENS=28000
# DINO_SERVER_URL="http://127.0.0.1:8808"

TEST_NAME=${TEST_DATA##*/}
if [[ "$PROMPT_FILE" == *","* ]]; then
    FIRST_PROMPT="${PROMPT_FILE%%,*}"
    PROMPT_NAME="$(basename "$FIRST_PROMPT" | sed 's/\.[^.]*$//')_multi"
else
    PROMPT_NAME=$(basename "$PROMPT_FILE" | sed 's/\.[^.]*$//')
fi
OUTPUT_DIR="server_inference_results/${SYSTEM_PROMPT##*/}/${PROMPT_NAME}_${INFERENCE_MODE}/${TEST_DATA##*/}"

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

if [ -n "$DINO_SERVER_URL" ]; then
    echo "🦖 DINO Server: $DINO_SERVER_URL"
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
    --dino_server_url "$DINO_SERVER_URL"
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
