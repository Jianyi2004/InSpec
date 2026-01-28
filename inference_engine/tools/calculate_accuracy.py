#!/usr/bin/env python3
"""
准确率统计工具
对比标注数据和推理结果，计算单张图和按组的准确率
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


class AccuracyCalculator:
    def __init__(self, annotation_file: str, inference_file: str, update_labels: bool = True):
        self.annotation_file = annotation_file
        self.inference_file = inference_file
        self.update_labels = update_labels
        
        self.annotations = []
        self.inferences = []
        self.updated_inferences = []  # 更新后的推理结果
        
        self.load_data()
    
    def load_data(self):
        """加载标注数据和推理结果"""
        with open(self.annotation_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        
        with open(self.inference_file, 'r', encoding='utf-8') as f:
            self.inferences = json.load(f)
        
        print(f"✅ 加载标注数据: {len(self.annotations)} 个样本")
        print(f"✅ 加载推理结果: {len(self.inferences)} 个样本")
    
    def parse_annotation(self, content: str) -> Tuple[List[str], str]:
        """
        解析标注数据的content字段
        格式: {"analysis": ["image1:PASS", "image2:NOT_INVOLVED"], "result": "PASS"}
        返回: (per_image_labels, group_result)
        """
        try:
            data = json.loads(content)
            analysis = data.get("analysis", [])
            result = data.get("result", "UNKNOWN")
            
            # 提取每张图片的标签
            per_image_labels = []
            for item in analysis:
                # 匹配格式: "image1:PASS" 或 "image2:NOT_INVOLVED"
                match = re.search(r'image\d+:(\w+)', item)
                if match:
                    per_image_labels.append(match.group(1))
            
            return per_image_labels, result
        except Exception as e:
            print(f"⚠️ 解析标注失败: {content[:100]}... 错误: {e}")
            return [], "UNKNOWN"
    
    def parse_inference(self, raw_response: str) -> Tuple[List[str], str]:
        """
        解析推理结果的raw_response字段
        从analysis的最后一项提取每张图片的结果
        格式: "image1:NOT_INVOLVED, image2:NOT_INVOLVED, image3:NOT_INVOLVED"
        返回: (per_image_labels, group_result)
        """
        try:
            data = json.loads(raw_response)
            analysis = data.get("analysis", [])
            result = data.get("result", "UNKNOWN")
            
            # 最后一项包含所有图片的结果
            if analysis:
                last_item = analysis[-1]
                # 匹配所有 "imageX:LABEL" 格式
                matches = re.findall(r'image\d+:(\w+)', last_item)
                if matches:
                    return matches, result
            
            return [], result
        except Exception as e:
            print(f"⚠️ 解析推理结果失败: {raw_response[:100]}... 错误: {e}")
            return [], "UNKNOWN"
    
    def match_samples(self) -> List[Tuple[int, int]]:
        """
        匹配标注数据和推理结果
        通过图片路径进行匹配
        返回: [(annotation_idx, inference_idx), ...]
        """
        matches = []
        
        # 为推理结果建立索引
        inference_index = {}
        for idx, inf in enumerate(self.inferences):
            # 使用第一张图片路径作为key
            if inf.get("image_paths"):
                first_image = inf["image_paths"][0]
                inference_index[first_image] = idx
        
        # 匹配标注数据
        for ann_idx, ann in enumerate(self.annotations):
            if ann.get("images"):
                first_image = ann["images"][0]
                if first_image in inference_index:
                    inf_idx = inference_index[first_image]
                    matches.append((ann_idx, inf_idx))
        
        print(f"✅ 成功匹配 {len(matches)} 个样本")
        return matches
    
    def calculate_accuracy(self):
        """计算准确率"""
        matches = self.match_samples()
        
        if not matches:
            print("❌ 没有匹配的样本，无法计算准确率")
            return
        
        # 复制推理结果用于更新
        if self.update_labels:
            import copy
            self.updated_inferences = copy.deepcopy(self.inferences)
        
        # 统计数据
        per_image_stats = {
            "total": 0,
            "correct": 0,
            "by_label": defaultdict(lambda: {"total": 0, "correct": 0})
        }
        
        group_stats = {
            "total": 0,
            "correct": 0,
            "by_label": defaultdict(lambda: {"total": 0, "correct": 0})
        }
        
        # 详细错误记录
        per_image_errors = []
        group_errors = []
        
        for ann_idx, inf_idx in matches:
            ann = self.annotations[ann_idx]
            inf = self.inferences[inf_idx]
            
            # 解析标注
            ann_content = ann["messages"][1]["content"]  # assistant的回复
            ann_per_image, ann_group = self.parse_annotation(ann_content)
            
            # 解析推理结果
            inf_raw = inf.get("raw_response", "")
            inf_per_image, inf_group = self.parse_inference(inf_raw)
            
            # 更新推理结果中的true_label
            if self.update_labels:
                self.updated_inferences[inf_idx]["true_label"] = ann_group
                
                # 更新per_image_details中每张图片的true_label
                if "per_image_details" in self.updated_inferences[inf_idx]:
                    per_image_details = self.updated_inferences[inf_idx]["per_image_details"]
                    for i, detail in enumerate(per_image_details):
                        if i < len(ann_per_image):
                            detail["true_label"] = ann_per_image[i]
                
                # 重新计算correct字段
                self.updated_inferences[inf_idx]["correct"] = (ann_group == inf_group)
            
            # 统计按组准确率
            group_stats["total"] += 1
            group_stats["by_label"][ann_group]["total"] += 1
            
            if ann_group == inf_group:
                group_stats["correct"] += 1
                group_stats["by_label"][ann_group]["correct"] += 1
            else:
                group_errors.append({
                    "sample_index": ann_idx,
                    "images": ann.get("images", []),
                    "true_label": ann_group,
                    "predicted_label": inf_group
                })
            
            # 统计单张图片准确率
            num_images = min(len(ann_per_image), len(inf_per_image))
            for i in range(num_images):
                ann_label = ann_per_image[i]
                inf_label = inf_per_image[i]
                
                per_image_stats["total"] += 1
                per_image_stats["by_label"][ann_label]["total"] += 1
                
                if ann_label == inf_label:
                    per_image_stats["correct"] += 1
                    per_image_stats["by_label"][ann_label]["correct"] += 1
                else:
                    per_image_errors.append({
                        "sample_index": ann_idx,
                        "image_index": i,
                        "image_path": ann["images"][i] if i < len(ann["images"]) else "unknown",
                        "true_label": ann_label,
                        "predicted_label": inf_label
                    })
        
        # 打印结果
        self.print_results(per_image_stats, group_stats, per_image_errors, group_errors)
        
        # 提示更新信息
        if self.update_labels:
            print(f"\n✅ 已更新 {len(matches)} 个样本的 true_label")
        
        return per_image_stats, group_stats, per_image_errors, group_errors
    
    def print_results(self, per_image_stats, group_stats, per_image_errors, group_errors):
        """打印统计结果"""
        print("\n" + "="*80)
        print("📊 准确率统计结果")
        print("="*80)
        
        # 按组统计
        print("\n【按组统计】")
        print(f"总样本数: {group_stats['total']}")
        print(f"正确数量: {group_stats['correct']}")
        print(f"准确率: {group_stats['correct']/group_stats['total']*100:.2f}%")
        
        print("\n各类别准确率:")
        for label in sorted(group_stats['by_label'].keys()):
            stats = group_stats['by_label'][label]
            acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
            print(f"  {label:15s}: {stats['correct']:3d}/{stats['total']:3d} = {acc:6.2f}%")
        
        # 单张图片统计
        print("\n【单张图片统计】")
        print(f"总图片数: {per_image_stats['total']}")
        print(f"正确数量: {per_image_stats['correct']}")
        print(f"准确率: {per_image_stats['correct']/per_image_stats['total']*100:.2f}%")
        
        print("\n各类别准确率:")
        for label in sorted(per_image_stats['by_label'].keys()):
            stats = per_image_stats['by_label'][label]
            acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
            print(f"  {label:15s}: {stats['correct']:3d}/{stats['total']:3d} = {acc:6.2f}%")
        
        # 错误样本
        if group_errors:
            print(f"\n【按组错误样本】({len(group_errors)} 个)")
            for i, err in enumerate(group_errors[:10], 1):  # 只显示前10个
                print(f"\n  错误 {i}:")
                print(f"    样本索引: {err['sample_index']}")
                print(f"    真实标签: {err['true_label']}")
                print(f"    预测标签: {err['predicted_label']}")
                print(f"    图片数量: {len(err['images'])}")
            
            if len(group_errors) > 10:
                print(f"\n  ... 还有 {len(group_errors) - 10} 个错误样本")
        
        if per_image_errors:
            print(f"\n【单张图片错误】({len(per_image_errors)} 个)")
            for i, err in enumerate(per_image_errors[:10], 1):  # 只显示前10个
                print(f"\n  错误 {i}:")
                print(f"    样本索引: {err['sample_index']}, 图片索引: {err['image_index']}")
                print(f"    真实标签: {err['true_label']}")
                print(f"    预测标签: {err['predicted_label']}")
            
            if len(per_image_errors) > 10:
                print(f"\n  ... 还有 {len(per_image_errors) - 10} 个错误图片")
        
        print("\n" + "="*80)
    
    def save_detailed_report(self, output_path: str, per_image_stats, group_stats, 
                            per_image_errors, group_errors):
        """保存详细报告到JSON文件"""
        report = {
            "summary": {
                "group_accuracy": {
                    "total": group_stats["total"],
                    "correct": group_stats["correct"],
                    "accuracy": group_stats["correct"] / group_stats["total"] if group_stats["total"] > 0 else 0,
                    "by_label": {
                        label: {
                            "total": stats["total"],
                            "correct": stats["correct"],
                            "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0
                        }
                        for label, stats in group_stats["by_label"].items()
                    }
                },
                "per_image_accuracy": {
                    "total": per_image_stats["total"],
                    "correct": per_image_stats["correct"],
                    "accuracy": per_image_stats["correct"] / per_image_stats["total"] if per_image_stats["total"] > 0 else 0,
                    "by_label": {
                        label: {
                            "total": stats["total"],
                            "correct": stats["correct"],
                            "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0
                        }
                        for label, stats in per_image_stats["by_label"].items()
                    }
                }
            },
            "errors": {
                "group_errors": group_errors,
                "per_image_errors": per_image_errors
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 详细报告已保存到: {output_path}")
    
    def save_updated_inference(self, output_path: str):
        """保存更新后的推理结果"""
        if not self.updated_inferences:
            print("⚠️ 没有更新的推理结果可保存")
            return
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.updated_inferences, f, ensure_ascii=False, indent=2)
        
        print(f"💾 更新后的推理结果已保存到: {output_path}")
        
        # 统计更新情况
        updated_count = sum(1 for inf in self.updated_inferences if "true_label" in inf)
        print(f"   - 总样本数: {len(self.updated_inferences)}")
        print(f"   - 已更新: {updated_count}")
        print(f"   - 正确预测: {sum(1 for inf in self.updated_inferences if inf.get('correct', False))}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="准确率统计工具")
    parser.add_argument(
        "--annotation",
        type=str,
        required=True,
        help="标注数据JSON文件路径"
    )
    parser.add_argument(
        "--inference",
        type=str,
        required=True,
        help="推理结果JSON文件路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="详细报告输出路径（可选）"
    )
    parser.add_argument(
        "--update-inference",
        type=str,
        default=None,
        help="更新后的推理结果输出路径（可选）"
    )
    parser.add_argument(
        "--no-update",
        action="store_true",
        help="不更新推理结果中的true_label"
    )
    
    args = parser.parse_args()
    
    # 创建计算器
    update_labels = not args.no_update
    calculator = AccuracyCalculator(args.annotation, args.inference, update_labels=update_labels)
    
    # 计算准确率
    per_image_stats, group_stats, per_image_errors, group_errors = calculator.calculate_accuracy()
    
    # 保存详细报告
    if args.output:
        calculator.save_detailed_report(
            args.output,
            per_image_stats,
            group_stats,
            per_image_errors,
            group_errors
        )
    
    # 保存更新后的推理结果
    if args.update_inference and update_labels:
        calculator.save_updated_inference(args.update_inference)
    elif update_labels and not args.update_inference:
        # 默认保存到原文件同目录，文件名加_updated后缀
        inference_path = Path(args.inference)
        default_output = inference_path.parent / f"{inference_path.stem}_updated{inference_path.suffix}"
        calculator.save_updated_inference(str(default_output))


if __name__ == "__main__":
    main()
