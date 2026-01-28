#!/bin/bash
# MODEL_PATH="/data_all/share/models/qwen3-vl32b-thinking/"
MODEL_PATH="/data_all/share/models/Qwen3-VL-32B-Instruct"

export CUDA_VISIBLE_DEVICES=2,3

vllm serve $MODEL_PATH \
    --host 0.0.0.0 \
    --port 7878 \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code

