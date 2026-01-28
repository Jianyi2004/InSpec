#!/bin/bash

# 准确率计算脚本

# 标注数据路径
ANNOTATION_FILE="data/6-outside/output_annotated.json"

# 推理结果路径
INFERENCE_FILE="server_inference_results/outside/system_prompt_v5_2.txt/cabinet_inside_prompt_v13_2_multi/new_summary_v3_2.txt/output.json/detailed_results.json"

# 输出报告路径
OUTPUT_FILE="accuracy_report.json"

# 更新后的推理结果路径
UPDATED_INFERENCE_FILE="server_inference_results/outside/system_prompt_v5_2.txt/cabinet_inside_prompt_v13_2_multi/new_summary_v3_2.txt/output.json/detailed_results.json"

echo "📊 准确率统计工具"
echo "==============================================="
echo "标注数据: $ANNOTATION_FILE"
echo "推理结果: $INFERENCE_FILE"
echo "输出报告: $OUTPUT_FILE"
echo "更新后推理结果: $UPDATED_INFERENCE_FILE"
echo "==============================================="
echo ""

python calculate_accuracy.py \
    --annotation "$ANNOTATION_FILE" \
    --inference "$INFERENCE_FILE" \
    --output "$OUTPUT_FILE" \
    --update-inference "$UPDATED_INFERENCE_FILE"
