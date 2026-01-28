#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多图/单图标注工具 - 单文件版本
使用 Gradio 框架，只需要一个 Python 文件就能运行（保留多图样本展示逻辑）
"""

import json
import os
import argparse
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import tempfile

# 设置 Gradio 临时目录为当前用户有权限的目录
os.environ['GRADIO_TEMP_DIR'] = os.path.join(tempfile.gettempdir(), f'gradio_{os.getuid()}')

VALID_LABELS = {"PASS", "FAIL", "NOT_INVOLVED"}

class SimpleErrorViewer:
    def __init__(self):
        self.error_data = None
        self.current_samples = []
        self.current_index = 0
        self.data_format = None  # 'error_analysis' 或 'new_format'
        self.current_image_index = 0  # 多图样本中当前显示的图片索引
        self.extracted_raw_by_id = {}  # 抽取数据: {unique_id: raw_item}
        self.extracted_order = []  # 抽取顺序（用于稳定导出）
        self.extracted_filter_name = "已抽取"
        # 实时监控相关
        self.json_file_path = None  # 当前监控的JSON文件路径
        self.last_modified_time = None  # 文件最后修改时间
        # 标注相关
        self.label_mode = None  # "multi" 或 "single"
        self.default_output_path = None
        self.single_image_labels_by_uid: Dict[str, List[Optional[str]]] = {}  # uid -> labels

    # ==================== 标注功能 ====================

    @staticmethod
    def compute_default_output_path(input_path: str) -> str:
        """默认输出路径：与输入同目录，文件名加 _labeld"""
        if not input_path:
            return ""
        directory = os.path.dirname(str(input_path))
        base = os.path.basename(str(input_path))
        stem, ext = os.path.splitext(base)
        if not ext:
            ext = ".json"
        if stem.endswith("_labeld"):
            return os.path.join(directory, f"{stem}{ext}")
        return os.path.join(directory, f"{stem}_labeld{ext}")

    def set_label_mode(self, mode: str) -> str:
        """设置标注模式：multi(整组) / single(逐图+合并)"""
        if mode not in {"multi", "single"}:
            return "❌ 标注模式无效，请选择 multi 或 single"
        self.label_mode = mode
        return f"✅ 已选择标注模式：{'多图逻辑(整组标注)' if mode == 'multi' else '单图逻辑(逐图标注+合并)'}"

    def _get_image_paths_from_raw_item(self, raw_item: Dict[str, Any]) -> List[str]:
        """从原始条目中提取图片路径列表（兼容 images/image_paths/image_path）"""
        if not isinstance(raw_item, dict):
            return []
        images = raw_item.get("images")
        if isinstance(images, list):
            return [str(p) for p in images if p]
        if isinstance(images, str) and images:
            return [images]

        image_paths = raw_item.get("image_paths")
        if isinstance(image_paths, list):
            return [str(p) for p in image_paths if p]
        if isinstance(image_paths, str) and image_paths:
            return [image_paths]

        single = raw_item.get("image_path") or raw_item.get("original_image_path") or ""
        return [str(single)] if single else []

    def _get_assistant_message_ref(self, raw_item: Dict[str, Any]) -> Dict[str, Any]:
        """获取（或创建）assistant message 字典引用，便于原地更新 content"""
        messages = raw_item.get("messages")
        if not isinstance(messages, list):
            messages = []
            raw_item["messages"] = messages

        assistant_msg = None
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                assistant_msg = msg
                break

        if assistant_msg is None:
            assistant_msg = {"role": "assistant", "content": ""}
            # 若没有 user 消息，也补一个，保持训练格式一致
            if not any(isinstance(m, dict) and m.get("role") == "user" for m in messages):
                messages.append({"role": "user", "content": "<image>"})
            messages.append(assistant_msg)
        return assistant_msg

    def _parse_payload_from_assistant_content(self, content: Any) -> Optional[Dict[str, Any]]:
        """解析 assistant content（可能为纯标签字符串或JSON字符串）为 dict"""
        if content is None:
            return None
        if isinstance(content, dict):
            return content
        if isinstance(content, list):
            return {"analysis": content}
        if not isinstance(content, str):
            content = str(content)
        text = content.strip()
        if not text:
            return None
        if text.upper() in VALID_LABELS:
            return {"analysis": [], "result": text.upper()}
        if text.startswith("{") and text.endswith("}"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                return None
        return None

    def _extract_single_labels_from_payload(
        self, payload: Dict[str, Any], total_images: int
    ) -> Optional[List[Optional[str]]]:
        """从 payload['analysis'] 中解析逐图标签（形如 image1:PASS）"""
        if total_images <= 0:
            return []
        analysis = payload.get("analysis")
        if analysis is None:
            return None

        labels: List[Optional[str]] = [None] * total_images

        import re

        def assign_from_text(text: str) -> None:
            match = re.search(r"image\s*(\d+)\s*[:：]\s*([A-Za-z_]+)", text, re.IGNORECASE)
            if not match:
                return
            idx = int(match.group(1)) - 1
            if idx < 0 or idx >= total_images:
                return
            label = match.group(2).upper()
            if label in VALID_LABELS:
                labels[idx] = label

        if isinstance(analysis, list):
            for item in analysis:
                if not isinstance(item, str):
                    continue
                assign_from_text(item)
        elif isinstance(analysis, str):
            assign_from_text(analysis)
            for part in analysis.split(","):
                assign_from_text(part)
        else:
            return None

        if all(v is None for v in labels):
            return None
        return labels

    @staticmethod
    def merge_single_labels(labels: List[str]) -> str:
        """合并逻辑：有FAIL->FAIL；无FAIL且有PASS->PASS；全NOT_INVOLVED->FAIL"""
        normalized = [str(x).upper() for x in labels if x]
        if any(x == "FAIL" for x in normalized):
            return "FAIL"
        if normalized and all(x == "NOT_INVOLVED" for x in normalized):
            return "FAIL"
        if any(x == "PASS" for x in normalized):
            return "PASS"
        return "FAIL"

    def _build_multi_payload(self, result: str) -> Dict[str, Any]:
        result_norm = str(result).upper()
        if result_norm not in VALID_LABELS:
            result_norm = "FAIL"
        return {"analysis": [], "result": result_norm}

    def _build_single_payload(self, per_image_labels: List[str]) -> Dict[str, Any]:
        normalized = [str(x).upper() for x in per_image_labels]
        analysis = [f"image{i + 1}:{label}" for i, label in enumerate(normalized)]
        merged = self.merge_single_labels(normalized)
        return {"analysis": analysis, "result": merged}

    def _set_raw_item_payload(self, raw_item: Dict[str, Any], payload: Dict[str, Any]) -> None:
        """将 payload 写回 raw_item 的 assistant content（JSON字符串）"""
        assistant_msg = self._get_assistant_message_ref(raw_item)
        assistant_msg["content"] = json.dumps(payload, ensure_ascii=False)

        # 确保 images 字段存在，便于直接作为训练数据使用
        if "images" not in raw_item or not raw_item.get("images"):
            raw_item["images"] = self._get_image_paths_from_raw_item(raw_item)

    def _reset_sample_to_first_image(self, sample: Dict[str, Any]) -> None:
        """将样本的当前图片重置为组内第一张（用于跳转到下一组时从第一张开始）"""
        if not isinstance(sample, dict):
            return
        image_paths = sample.get("image_paths")
        if not isinstance(image_paths, list):
            image_paths = []

        if not image_paths:
            fallback = sample.get("original_image_path") or sample.get("image_path") or ""
            if fallback:
                image_paths = [str(fallback)]
                sample["image_paths"] = image_paths

        sample["current_image_index"] = 0
        if image_paths:
            sample["original_image_path"] = image_paths[0]
            sample["copied_image_path"] = image_paths[0]

    def _goto_next_group_first_image(self) -> bool:
        """跳转到下一组的第一张图；返回是否成功跳转"""
        if not self.current_samples:
            return False
        if self.current_index >= len(self.current_samples) - 1:
            return False
        self.current_index += 1
        next_sample = self.current_samples[self.current_index]
        if isinstance(next_sample, dict):
            self._reset_sample_to_first_image(next_sample)
        return True

    def _get_or_init_single_labels(self, raw_item: Dict[str, Any]) -> List[Optional[str]]:
        """获取当前 raw_item 的逐图标注缓存；如无则从已存在 content 中解析或初始化"""
        uid = self._get_raw_unique_id(raw_item)
        image_paths = self._get_image_paths_from_raw_item(raw_item)
        total = len(image_paths)
        if uid in self.single_image_labels_by_uid and len(self.single_image_labels_by_uid[uid]) == total:
            return self.single_image_labels_by_uid[uid]

        labels: List[Optional[str]] = [None] * total

        payload = None
        messages = raw_item.get("messages")
        if isinstance(messages, list):
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    payload = self._parse_payload_from_assistant_content(msg.get("content"))
                    break
        if isinstance(payload, dict):
            parsed_labels = self._extract_single_labels_from_payload(payload, total)
            if parsed_labels is not None:
                labels = parsed_labels

        self.single_image_labels_by_uid[uid] = labels
        return labels

    def get_current_annotation_text(self) -> str:
        """生成当前样本的标注状态文本"""
        raw_item, err = self._get_current_raw_item()
        if err:
            return err
        image_paths = self._get_image_paths_from_raw_item(raw_item)
        total = len(image_paths)
        uid = self._get_raw_unique_id(raw_item)

        # 读取当前 assistant 的已写入结果（用于多图模式显示）
        payload = None
        messages = raw_item.get("messages")
        if isinstance(messages, list):
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    payload = self._parse_payload_from_assistant_content(msg.get("content"))
                    break

        if self.label_mode == "single":
            labels = self._get_or_init_single_labels(raw_item)
            parts = []
            for i in range(total):
                label = labels[i] or "未标注"
                marker = "👉 " if i == self.current_samples[self.current_index].get("current_image_index", 0) else "   "
                parts.append(f"{marker}image{i + 1}:{label}")
            labeled_count = sum(1 for x in labels if x in VALID_LABELS)
            merged = ""
            if labeled_count == total and total > 0:
                merged = f"\n合并结果: {self.merge_single_labels([x for x in labels if x])}"
            return (
                f"📝 单图逻辑标注进度: {labeled_count}/{total}\n"
                + "\n".join(parts)
                + merged
            )

        # 默认/多图模式
        result = None
        if isinstance(payload, dict) and payload.get("result") is not None:
            result = str(payload.get("result")).upper()
        elif isinstance(payload, dict) and payload.get("analysis") is None and payload.get("result") is None:
            result = None
        if result and result in VALID_LABELS:
            return f"📝 多图逻辑(整组)当前标注: {result}"
        return "📝 多图逻辑(整组)当前标注: 未标注"

    def apply_label(
        self,
        label: str,
        label_mode: str,
        output_path: str,
        auto_save: bool,
        show_bbox: bool,
    ) -> Tuple[str, str, str, str, str, str, str, List, str, str]:
        """
        对当前样本应用标注，并返回用于刷新界面的数据。
        返回: (status_msg, *navigate_sample_outputs)
        """
        mode_msg = self.set_label_mode(label_mode)
        if mode_msg.startswith("❌"):
            return (mode_msg, *self.navigate_sample(0, show_bbox))

        raw_item, err = self._get_current_raw_item()
        if err:
            return (err, *self.navigate_sample(0, show_bbox))

        label_norm = str(label).upper()
        if label_norm not in VALID_LABELS:
            return (f"❌ 标签无效: {label}", *self.navigate_sample(0, show_bbox))

        if not output_path:
            output_path = self.default_output_path or self.compute_default_output_path(self.json_file_path or "")

        sample = self.current_samples[self.current_index]
        image_paths = self._get_image_paths_from_raw_item(raw_item)
        total_images = len(image_paths)

        if self.label_mode == "multi":
            payload = self._build_multi_payload(label_norm)
            self._set_raw_item_payload(raw_item, payload)
            status = f"✅ 已标注(多图整组): {label_norm}"

            if auto_save:
                save_msg = self.save_labeled_json(output_path)
                status = f"{status}\n{save_msg}"

            # 当前组已标注完成：自动跳转到下一组的第一张
            self._reset_sample_to_first_image(sample)
            self._goto_next_group_first_image()
            # 组间切换后刷新操作结果，避免停留在上一组信息
            try:
                status = f"{status}\n\n{self.get_current_annotation_text()}"
            except Exception:
                pass
            return (status, *self.navigate_sample(0, show_bbox))

        # 单图逻辑：逐张标注
        labels = self._get_or_init_single_labels(raw_item)
        if len(labels) != total_images:
            labels = [None] * total_images
        current_img_idx = sample.get("current_image_index", 0)
        if total_images == 0:
            return (f"❌ 当前样本没有图片，无法标注", *self.navigate_sample(0, show_bbox))
        if current_img_idx < 0 or current_img_idx >= total_images:
            current_img_idx = 0
            sample["current_image_index"] = 0

        labels[current_img_idx] = label_norm
        uid = self._get_raw_unique_id(raw_item)
        self.single_image_labels_by_uid[uid] = labels

        labeled_count = sum(1 for x in labels if x in VALID_LABELS)
        all_labeled = total_images > 0 and all(x in VALID_LABELS for x in labels)

        # 单图标注后自动跳转到下一张照片（顺序；若最后一张则兜底找未标注）
        if not all_labeled:
            next_idx = current_img_idx + 1 if current_img_idx < total_images - 1 else None
            if next_idx is None:
                for i in range(total_images):
                    if labels[i] not in VALID_LABELS:
                        next_idx = i
                        break
            if next_idx is not None:
                sample["current_image_index"] = next_idx
                if sample.get("image_paths") and next_idx < len(sample["image_paths"]):
                    sample["original_image_path"] = sample["image_paths"][next_idx]
                    sample["copied_image_path"] = sample["image_paths"][next_idx]

        def format_group_labels(display_idx: int) -> str:
            lines = []
            for i in range(total_images):
                v = labels[i] if labels[i] in VALID_LABELS else "未标注"
                prefix = "👉 " if i == display_idx else "   "
                lines.append(f"{prefix}image{i + 1}:{v}")
            return "\n".join(lines)

        display_idx = sample.get("current_image_index", current_img_idx)
        status = (
            f"✅ 已标注 image{current_img_idx + 1}: {label_norm} ({labeled_count}/{total_images})\n"
            f"{format_group_labels(display_idx)}"
        )

        # 若完成整组标注，则合并并写回
        if all_labeled:
            payload = self._build_single_payload([x for x in labels if x])
            self._set_raw_item_payload(raw_item, payload)
            status += f"\n✅ 已自动合并结果: {payload.get('result')}"

            if auto_save:
                save_msg = self.save_labeled_json(output_path)
                status = f"{status}\n{save_msg}"

            # 完成后，重置当前组图片索引，便于下一组查看
            sample["current_image_index"] = 0
            if sample.get("image_paths"):
                sample["original_image_path"] = sample["image_paths"][0]
                sample["copied_image_path"] = sample["image_paths"][0]

            # 当前组已标注完成：自动跳转到下一组的第一张
            self._goto_next_group_first_image()
            try:
                status = f"{status}\n\n{self.get_current_annotation_text()}"
            except Exception:
                pass
            return (status, *self.navigate_sample(0, show_bbox))

        return (status, *self.navigate_sample(0, show_bbox))

    def save_labeled_json(self, output_path: str) -> str:
        """保存当前数据到 JSON 文件（原地覆盖写入到 output_path）"""
        if not output_path:
            return "❌ 输出路径为空，无法保存"
        if self.error_data is None:
            return "❌ 尚未加载数据，无法保存"

        output_path = str(output_path).strip()
        if not output_path:
            return "❌ 输出路径为空，无法保存"

        try:
            parent = os.path.dirname(output_path)
            if parent:
                os.makedirs(parent, exist_ok=True)

            tmp_path = f"{output_path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self.error_data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, output_path)
            return f"💾 已保存到: {output_path}"
        except Exception as e:
            return f"❌ 保存失败: {str(e)}"
        
    def detect_data_format(self, data) -> str:
        """检测数据格式"""
        if isinstance(data, dict) and 'error_samples' in data:
            return 'error_analysis'
        elif isinstance(data, list) and len(data) > 0:
            # 检查是否是vLLM推理结果格式
            # 检查前几个样本，因为第一个可能是ERROR
            for item in data[:min(10, len(data))]:
                # vLLM格式：有 prediction 和 image_paths 字段
                if 'prediction' in item and 'image_paths' in item:
                    return 'vllm_format'
                # 新格式：有 messages 字段
                elif 'messages' in item:
                    return 'new_format'
        return 'unknown'
    
    def convert_new_format_to_samples(self, data) -> List[Dict]:
        """将新格式数据转换为样本格式（支持多图）"""
        samples = []
        for i, item in enumerate(data):
            # 提取对话内容
            messages = item.get('messages', [])
            user_message = ""
            assistant_message = ""
            
            for msg in messages:
                if msg.get('role') == 'user':
                    user_message = msg.get('content', '')
                elif msg.get('role') == 'assistant':
                    assistant_message = msg.get('content', '')
            
            # 提取图片路径（支持多图）
            images = item.get('images', [])
            
            # 解析助手响应中的result字段和单图详情
            predicted_result = "未知"
            single_image_details = []
            try:
                if assistant_message.strip().startswith('{'):
                    parsed_response = json.loads(assistant_message)
                    predicted_result = parsed_response.get('result', '未知')
                    
                    # 提取单图详情
                    details = parsed_response.get('details', [])
                    for detail in details:
                        single_image_details.append({
                            'image_index': detail.get('image_index', 0),
                            'result': detail.get('result', '未知'),
                            'reason': detail.get('reason_cn', ''),
                            'bboxes': detail.get('bboxes', [])
                        })
            except:
                pass
            
            # 为每张图片创建一个样本条目，但保持多图信息
            if images:
                # 主样本（包含所有图片信息）
                sample = {
                    'sample_index': i,
                    'image_names': [os.path.basename(img) for img in images],
                    'image_paths': images,  # 多图路径列表
                    'current_image_index': 0,  # 当前显示的图片索引
                    'original_image_path': images[0],  # 兼容性：第一张图片
                    'copied_image_path': images[0],
                    'user_prompt': user_message,
                    'model_response': assistant_message,
                    'summary_response': item.get('summary_response', assistant_message),
                    'model_prediction': predicted_result,
                    'single_image_details': single_image_details,
                    'true_label': self.infer_true_label_from_path(images[0]),
                    'true_label_raw': predicted_result,
                    'data_format': 'multi_image_format',
                    'total_images': len(images),
                    '_raw_item': item,
                }
                samples.append(sample)
            else:
                # 无图片的样本
                sample = {
                    'sample_index': i,
                    'image_names': [],
                    'image_paths': [],
                    'current_image_index': 0,
                    'original_image_path': "",
                    'copied_image_path': "",
                    'user_prompt': user_message,
                    'model_response': assistant_message,
                    'summary_response': item.get('summary_response', assistant_message),
                    'model_prediction': predicted_result,
                    'single_image_details': single_image_details,
                    'true_label': "未知",
                    'true_label_raw': predicted_result,
                    'data_format': 'multi_image_format',
                    'total_images': 0,
                    '_raw_item': item,
                }
                samples.append(sample)
        
        return samples
    
    def convert_vllm_format_to_samples(self, data) -> List[Dict]:
        """将vLLM格式数据转换为样本格式（支持单图/多图，自动跳过预测失败的样本）"""
        samples = []
        skipped_count = 0
        
        print(f"\n🔄 开始转换 vLLM 格式数据，总样本数: {len(data)}")
        
        for i, item in enumerate(data):
            # 确定错误类型
            true_label = item.get('true_label', 'UNKNOWN')
            model_prediction = item.get('prediction', 'UNKNOWN')
            
            # 跳过预测失败的样本（prediction == "ERROR"）
            if model_prediction == 'ERROR':
                skipped_count += 1
                if skipped_count <= 5:  # 只打印前5个
                    print(f"⚠️  跳过预测失败的样本 {i}: {item.get('image_name', 'unknown')}")
                continue
            
            if true_label == 'PASS' and model_prediction == 'FAIL':
                error_type = "假负例 (False Negative)"
            elif true_label == 'FAIL' and model_prediction == 'PASS':
                error_type = "假正例 (False Positive)"
            elif true_label == model_prediction:
                error_type = "正确预测"
            else:
                error_type = "未知错误类型"
            
            # 提取图片路径（支持单图/多图）
            image_paths = item.get('image_paths', [])
            if not image_paths:
                # 兼容旧格式的单图路径
                image_path = item.get('image_path', '')
                image_paths = [image_path] if image_path else []
            
            # 处理单图推理的逐图详情
            per_image_details = item.get('per_image_details', [])
            per_image_summary_map = {}
            per_image_summary_by_label = {}
            per_image_prediction_map = {}
            per_image_label_by_path = {}
            for detail in per_image_details:
                detail_path = detail.get('image_path')
                detail_label = detail.get('image_label')
                detail_summary = detail.get('summary_response')
                detail_prediction = detail.get('prediction')
                if detail_path:
                    per_image_summary_map[detail_path] = detail_summary
                    per_image_prediction_map[detail_path] = detail_prediction
                    if detail_label:
                        per_image_label_by_path[detail_path] = detail_label
                if detail_label:
                    per_image_summary_by_label[detail_label] = detail_summary
            
            # 转换为标准格式
            sample = {
                'error_id': len(samples) + 1,  # 使用实际样本数量作为ID
                'sample_index': item.get('sample_index', i),
                'image_name': item.get('image_name', f"sample_{i}"),
                'image_paths': image_paths,  # 多图路径列表
                'image_names': [os.path.basename(img) for img in image_paths],
                'current_image_index': 0,  # 当前显示的图片索引
                'original_image_path': image_paths[0] if image_paths else '',  # 兼容性：第一张图片
                'copied_image_path': image_paths[0] if image_paths else '',
                'user_prompt': "工业质检任务",  # vLLM格式中没有用户提示
                'model_response': item.get('raw_response', ''),
                'summary_response': item.get('summary_response', item.get('raw_response', '')),
                'model_prediction': model_prediction,
                'true_label': true_label,
                'true_label_raw': item.get('true_label_raw', true_label),
                'error_type': error_type,
                'data_format': 'vllm_format',
                'total_images': len(image_paths),
                'inference_mode': item.get('inference_mode', 'unknown'),
                'per_image_details': per_image_details,
                'per_image_summary_map': per_image_summary_map,
                'per_image_summary_by_label': per_image_summary_by_label,
                'per_image_prediction_map': per_image_prediction_map,
                'per_image_label_by_path': per_image_label_by_path,
                '_raw_item': item,
                '_raw_item_index': i,
            }
            samples.append(sample)
        
        print(f"\n📊 转换完成:")
        print(f"   - 跳过样本: {skipped_count} 个 (ERROR)")
        print(f"   - 有效样本: {len(samples)} 个")
        
        if skipped_count > 5:
            print(f"   - (省略了 {skipped_count - 5} 个跳过样本的详细信息)")
        
        return samples
    
    def auto_reload(self) -> Tuple[bool, int]:
        """
        自动检查文件变化并重新加载
        返回: (是否有更新, 新增样本数)
        """
        if not self.json_file_path or not os.path.exists(self.json_file_path):
            return False, 0
        
        try:
            current_mtime = os.path.getmtime(self.json_file_path)
            
            # 检查文件是否有变化
            if self.last_modified_time is None or current_mtime > self.last_modified_time:
                # 记录当前样本数和索引
                old_sample_count = len(self.current_samples)
                old_index = self.current_index
                
                # 文件已更新，静默重新加载
                self.load_json_file(self.json_file_path)
                
                # 计算新增样本数
                new_sample_count = len(self.current_samples)
                new_samples = new_sample_count - old_sample_count
                
                # 保持用户当前的浏览位置（如果索引仍然有效）
                if old_index < new_sample_count:
                    self.current_index = old_index
                
                return True, new_samples
            else:
                # 文件未变化
                return False, 0
        except Exception as e:
            return False, 0
    
    def infer_true_label_from_path(self, image_path: str) -> str:
        """从图片路径推断真实标签（用于新格式数据）"""
        if not image_path:
            return "未知"
        
        # 根据路径中的关键词推断标签
        path_lower = image_path.lower()
        if '合格' in image_path or 'pass' in path_lower:
            return "PASS"
        elif '不合格' in image_path or 'fail' in path_lower:
            return "FAIL"
        else:
            return "未知"
    
    def load_json_file(self, file_path: str) -> Tuple[str, str]:
        """加载JSON文件"""
        try:
            if not file_path:
                return "❌ 请选择JSON文件", ""
            
            # 记录文件路径和修改时间
            self.json_file_path = file_path
            self.default_output_path = self.compute_default_output_path(file_path)
            # 切换文件时清空逐图标注缓存（避免不同文件UID冲突）
            self.single_image_labels_by_uid = {}
            if os.path.exists(file_path):
                self.last_modified_time = os.path.getmtime(file_path)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 检测数据格式
            self.data_format = self.detect_data_format(data)
            
            if self.data_format == 'error_analysis':
                # 原有的错误分析格式
                if 'error_samples' not in data or not data['error_samples']:
                    return "❌ JSON文件格式错误：缺少error_samples字段", ""
                
                self.error_data = data
                self.current_samples = data['error_samples']
                self.current_index = 0
                
                total_errors = len(self.current_samples)
                timestamp = data.get('timestamp', '未知')
                
                # 统计错误类型
                error_types = {}
                for sample in self.current_samples:
                    error_type = sample.get('error_type', '未知')
                    error_types[error_type] = error_types.get(error_type, 0) + 1
                
                stats_info = f"""
✅ 错误分析数据加载成功！

📊 统计信息：
• 总错误样本：{total_errors}
• 生成时间：{timestamp[:19] if timestamp else '未知'}

📈 错误类型分布：
"""
                for error_type, count in error_types.items():
                    stats_info += f"• {error_type}：{count} 个\n"
                
                return stats_info, "已加载错误分析数据，可以开始查看样本"
                
            elif self.data_format == 'new_format':
                # 新格式数据
                self.error_data = data
                self.current_samples = self.convert_new_format_to_samples(data)
                self.current_index = 0
                
                total_samples = len(self.current_samples)
                
                # 统计预测结果分布
                prediction_stats = {}
                for sample in self.current_samples:
                    pred = sample.get('model_prediction', '未知')
                    prediction_stats[pred] = prediction_stats.get(pred, 0) + 1
                
                stats_info = f"""
✅ 对话数据加载成功！

📊 统计信息：
• 总样本数：{total_samples}
• 数据格式：对话格式（用于检查模型输出与图片匹配度）

📈 模型预测分布：
"""
                for pred, count in prediction_stats.items():
                    stats_info += f"• {pred}：{count} 个\n"
                
                return stats_info, "已加载对话数据，可以开始检查模型输出与图片的匹配度"
                
            elif self.data_format == 'vllm_format':
                # vLLM推理结果格式
                self.error_data = data
                original_count = len(data)
                self.current_samples = self.convert_vllm_format_to_samples(data)
                self.current_index = 0
                
                total_samples = len(self.current_samples)
                skipped_samples = original_count - total_samples
                
                # 检查是否所有样本都被跳过
                if total_samples == 0:
                    return f"""
⚠️ 警告：所有样本都被跳过！

📊 统计信息：
• 原始样本数：{original_count}
• 跳过样本数：{skipped_samples}（全部为预测失败的 ERROR 样本）
• 有效样本数：0

💡 建议：
1. 检查推理是否正常运行
2. 查看推理日志中的错误信息
3. 尝试重新运行推理
""", "所有样本都是预测失败（ERROR），无法展示"
                
                # 统计错误类型分布
                error_type_stats = {}
                prediction_stats = {}
                for sample in self.current_samples:
                    error_type = sample.get('error_type', '未知')
                    pred = sample.get('model_prediction', '未知')
                    error_type_stats[error_type] = error_type_stats.get(error_type, 0) + 1
                    prediction_stats[pred] = prediction_stats.get(pred, 0) + 1
                
                stats_info = f"""
✅ vLLM推理结果数据加载成功！

📊 统计信息：
• 原始样本数：{original_count}
• 跳过样本数：{skipped_samples}（预测失败的 ERROR 样本）
• 有效样本数：{total_samples}
• 数据格式：vLLM推理结果格式

📈 错误类型分布：
"""
                for error_type, count in error_type_stats.items():
                    stats_info += f"• {error_type}：{count} 个\n"
                
                stats_info += "\n📈 模型预测分布：\n"
                for pred, count in prediction_stats.items():
                    stats_info += f"• {pred}：{count} 个\n"
                
                return stats_info, "已加载vLLM推理结果，可以开始查看样本分析"
            
            else:
                return "❌ 不支持的数据格式，请选择error_analysis.json、new_format_data.json或vLLM推理结果格式的文件", ""
            
        except Exception as e:
            return f"❌ 加载失败：{str(e)}", ""
    
    def filter_samples(self, filter_value: str) -> str:
        """筛选样本"""
        if not self.error_data:
            return "请先加载JSON文件"

        if filter_value == self.extracted_filter_name:
            self.current_samples = self._get_extracted_samples_for_view()
            self.current_index = 0
            return f"筛选结果：{len(self.current_samples)} 个样本"
        
        if filter_value == "全部":
            if self.data_format == 'error_analysis':
                self.current_samples = self.error_data['error_samples']
            elif self.data_format == 'vllm_format':
                self.current_samples = self.convert_vllm_format_to_samples(self.error_data)
            else:  # new_format
                self.current_samples = self.convert_new_format_to_samples(self.error_data)
        else:
            if self.data_format == 'error_analysis':
                # 按错误类型筛选
                self.current_samples = [
                    sample for sample in self.error_data['error_samples']
                    if sample.get('error_type', '') == filter_value
                ]
            elif self.data_format == 'vllm_format':
                # vLLM格式：按错误类型筛选
                all_samples = self.convert_vllm_format_to_samples(self.error_data)
                self.current_samples = [
                    sample for sample in all_samples
                    if sample.get('error_type', '') == filter_value
                ]
            else:  # new_format
                # 按模型预测结果筛选
                all_samples = self.convert_new_format_to_samples(self.error_data)
                self.current_samples = [
                    sample for sample in all_samples
                    if sample.get('model_prediction', '') == filter_value
                ]
        
        self.current_index = 0
        return f"筛选结果：{len(self.current_samples)} 个样本"
    
    def get_sample_info(self, index: int) -> Tuple[str, str, str, str, str, str, str, str]:
        """获取样本信息（支持多图）"""
        if not self.current_samples or index < 0 or index >= len(self.current_samples):
            return "无数据", "", "", "", "", "", "", ""
        
        sample = self.current_samples[index]
        
        if sample.get('data_format') == 'multi_image_format':
            # 多图格式的显示
            total_images = sample.get('total_images', 0)
            current_img_idx = sample.get('current_image_index', 0)
            image_names = sample.get('image_names', [])
            single_details = sample.get('single_image_details', [])
            
            info = f"""
📋 多图样本信息：
• 样本索引：{sample.get('sample_index', 'N/A')}
• 图片总数：{total_images} 张
• 当前图片：第 {current_img_idx + 1} 张 / 共 {total_images} 张
• 图片名称：{image_names[current_img_idx] if current_img_idx < len(image_names) else 'N/A'}
• 模型预测：{sample.get('model_prediction', 'N/A')}

样本进度：第 {index + 1} 个 / 共 {len(self.current_samples)} 个
            """
            
            # 多图标签对比和单图详情
            model_prediction = sample.get('model_prediction', 'N/A')
            
            label_info = f"""
🖼️ 多图分析结果：
• 最终预测：{model_prediction}
• 图片列表：{', '.join(image_names)}

🔍 单图详情：
"""
            
            # 显示每张图片的分析结果
            for i, detail in enumerate(single_details):
                img_result = detail.get('result', '未知')
                img_reason = detail.get('reason', '')
                bbox_count = len(detail.get('bboxes', []))
                
                # 当前图片高亮显示
                highlight = "👉 " if i == current_img_idx else "   "
                label_info += f"{highlight}图片{i}: {img_result}"
                if bbox_count > 0:
                    label_info += f" (检测区域: {bbox_count}个)"
                label_info += "\n"
                
                # 显示原因（简化版）
                if img_reason and len(img_reason) > 100:
                    label_info += f"      {img_reason[:100]}...\n"
                elif img_reason:
                    label_info += f"      {img_reason}\n"
            
            # 图片切换提示
            multi_image_nav = f"""
🎮 图片导航：
• 使用 "切换图片" 按钮浏览 {total_images} 张图片
• 当前显示：{image_names[current_img_idx] if current_img_idx < len(image_names) else 'N/A'}
            """ if total_images > 1 else ""
            
        elif self.data_format == 'error_analysis' or sample.get('data_format') == 'vllm_format':
            # 错误分析格式或vLLM格式的显示（支持多图）
            total_images = sample.get('total_images', 1)
            current_img_idx = sample.get('current_image_index', 0)
            image_names = sample.get('image_names', [sample.get('image_name', 'N/A')])
            
            info = f"""
📋 样本信息：
• 错误ID：{sample.get('error_id', 'N/A')}
• 原始索引：{sample.get('sample_index', 'N/A')}
• 图片名称：{sample.get('image_name', 'N/A')}
• 错误类型：{sample.get('error_type', 'N/A')}
• 图片总数：{total_images} 张
"""
            if total_images > 1:
                info += f"• 当前图片：第 {current_img_idx + 1} 张 / 共 {total_images} 张\n"
                info += f"• 当前文件：{image_names[current_img_idx] if current_img_idx < len(image_names) else 'N/A'}\n"
            
            info += f"\n当前：第 {index + 1} 个 / 共 {len(self.current_samples)} 个"
            
            # 标签对比
            true_label = sample.get('true_label', 'N/A')
            model_prediction = sample.get('model_prediction', 'N/A')
            true_label_raw = sample.get('true_label_raw', 'N/A')
            
            label_info = f"""
🏷️ 标签对比：
• 真实标签：{true_label} (原始: {true_label_raw})
• 模型预测：{model_prediction}
• 判断结果：{'✅ 正确' if true_label == model_prediction else '❌ 错误'}
            """
            
            # 多图导航提示
            if total_images > 1:
                multi_image_nav = f"""
🎮 图片导航：
• 使用 "切换图片" 按钮浏览 {total_images} 张图片
• 当前显示：{image_names[current_img_idx] if current_img_idx < len(image_names) else 'N/A'}
                """
            else:
                multi_image_nav = ""
        
        else:  # 单图new_format
            # 新格式的显示
            info = f"""
📋 样本信息：
• 样本索引：{sample.get('sample_index', 'N/A')}
• 图片名称：{sample.get('image_name', 'N/A')}
• 数据格式：对话格式
• 模型预测：{sample.get('model_prediction', 'N/A')}

当前：第 {index + 1} 个 / 共 {len(self.current_samples)} 个
            """
            
            # 标签对比（新格式主要用于检查匹配度）
            true_label = sample.get('true_label', 'N/A')
            model_prediction = sample.get('model_prediction', 'N/A')
            
            label_info = f"""
🔍 匹配度检查：
• 推断标签：{true_label} (根据路径推断)
• 模型预测：{model_prediction}
• 匹配状态：{'✅ 匹配' if true_label == model_prediction else '❌ 不匹配' if true_label != '未知' else '⚠️ 需要人工检查'}
• 用途：检查模型输出是否与图片内容匹配
            """
            multi_image_nav = ""
        
        # 模型响应（优先展示 summary_response）
        response_content = sample.get('summary_response')
        if not response_content:
            response_content = sample.get('model_response')
        if not response_content:
            response_content = sample.get('raw_response')
        
        def format_response_text(content: Any) -> Tuple[str, str]:
            """返回(原始字符串, 格式化字符串)"""
            if isinstance(content, (dict, list)):
                raw_text = json.dumps(content, ensure_ascii=False)
                return raw_text, json.dumps(content, ensure_ascii=False, indent=2)
            content_str = str(content) if content is not None else ""
            if not content_str:
                return "N/A", "N/A"
            try:
                parsed = json.loads(content_str)
                if isinstance(parsed, (dict, list)):
                    raw_text = json.dumps(parsed, ensure_ascii=False)
                    return raw_text, json.dumps(parsed, ensure_ascii=False, indent=2)
            except Exception:
                pass
            return content_str, content_str
        
        raw_response, formatted_response = format_response_text(response_content)
        
        # 图片路径
        image_path = sample.get('original_image_path', '')
        
        # 单图结果：单图推理模式下展示当前预览图片对应的 summary_response
        single_image_result = ""
        single_image_raw_response = ""
        if sample.get('inference_mode') == 'single':
            current_img_idx = sample.get('current_image_index', 0)
            image_paths = sample.get('image_paths', [])
            current_image_path = image_paths[current_img_idx] if current_img_idx < len(image_paths) else ""
            summary_text = None
            raw_text = None
            per_image_details = sample.get('per_image_details', [])

            # 优先找到当前图片对应的详情，以便同时取 summary 与 raw
            detail_for_image = None
            for detail in per_image_details:
                if current_image_path and detail.get('image_path') == current_image_path:
                    detail_for_image = detail
                    break
            if detail_for_image is None and per_image_details and current_img_idx < len(per_image_details):
                detail_for_image = per_image_details[current_img_idx]
            if detail_for_image:
                summary_text = detail_for_image.get('summary_response')
                raw_text = detail_for_image.get('raw_response') or detail_for_image.get('raw') or detail_for_image.get('response')
            
            per_image_map = sample.get('per_image_summary_map') or {}
            if summary_text is None and current_image_path:
                summary_text = per_image_map.get(current_image_path)
            
            if not summary_text:
                # 尝试通过 image_label 匹配
                per_image_label_map = sample.get('per_image_label_by_path') or {}
                label = per_image_label_map.get(current_image_path)
                if label:
                    summary_text = (sample.get('per_image_summary_by_label') or {}).get(label)
            
            if not summary_text:
                # 遍历详情作为最后兜底
                for detail in sample.get('per_image_details', []):
                    if current_image_path and detail.get('image_path') == current_image_path:
                        summary_text = detail.get('summary_response')
                        raw_text = raw_text or detail.get('raw_response') or detail.get('raw') or detail.get('response')
                        break
                else:
                    details = sample.get('per_image_details', [])
                    if details and current_img_idx < len(details):
                        summary_text = details[current_img_idx].get('summary_response')
                        raw_text = raw_text or details[current_img_idx].get('raw_response') or details[current_img_idx].get('raw') or details[current_img_idx].get('response')
            
            if raw_text is None:
                raw_text = sample.get('raw_response') or sample.get('model_response')
            
            if summary_text:
                if isinstance(summary_text, (dict, list)):
                    formatted_summary = json.dumps(summary_text, ensure_ascii=False, indent=2)
                else:
                    summary_str = str(summary_text)
                    try:
                        parsed_summary = json.loads(summary_str)
                        if isinstance(parsed_summary, (dict, list)):
                            formatted_summary = json.dumps(parsed_summary, ensure_ascii=False, indent=2)
                        else:
                            formatted_summary = summary_str
                    except:
                        formatted_summary = summary_str
                header = ""
                if current_image_path:
                    header = f"当前图片：{os.path.basename(current_image_path)}\n"
                single_image_result = f"{header}{formatted_summary}"
            else:
                single_image_result = "未找到对应的单图结果"
            
            single_image_raw_response, _ = format_response_text(raw_text)
        else:
            # multi 模式或无单图数据时，展示整体的原始响应
            raw_source = sample.get('raw_response') or sample.get('model_response') or sample.get('summary_response')
            single_image_raw_response, _ = format_response_text(raw_source)

        # 追加手工标注信息（标注工具核心信息）
        try:
            manual_annotation_text = self.get_current_annotation_text()
        except Exception as e:
            manual_annotation_text = f"⚠️ 标注信息生成失败: {e}"

        if manual_annotation_text:
            label_info = f"{manual_annotation_text}\n\n{label_info}".strip()

        return (
            info,
            label_info,
            raw_response,
            formatted_response,
            single_image_result,
            single_image_raw_response,
            image_path,
            multi_image_nav,
        )
    
    def switch_image(self, direction: int, show_bbox: bool = True) -> Tuple[str, str, str, str, str, str, List, str, str]:
        """切换多图样本中的图片"""
        if not self.current_samples:
            return "无数据", "", "", "", "", "", [], "", ""
        
        sample = self.current_samples[self.current_index]
        
        total_images = sample.get('total_images', len(sample.get('image_paths', [])))
        if total_images <= 1:
            return self.navigate_sample(0, show_bbox)
        
        # 更新图片索引
        current_img_idx = sample.get('current_image_index', 0)
        current_img_idx += direction
        current_img_idx = max(0, min(current_img_idx, total_images - 1))
        
        # 更新样本中的当前图片索引
        sample['current_image_index'] = current_img_idx
        
        # 更新图片路径
        image_paths = sample.get('image_paths', [])
        if current_img_idx < len(image_paths):
            sample['original_image_path'] = image_paths[current_img_idx]
            sample['copied_image_path'] = image_paths[current_img_idx]
        
        return self.navigate_sample(0, show_bbox)
    
    def navigate_sample(self, direction: int, show_bbox: bool = True) -> Tuple[str, str, str, str, str, str, List, str, str]:
        """导航样本（支持多图同页展示）"""
        if not self.current_samples:
            return "无数据", "", "", "", "", "", [], "", ""

        self.current_index += direction
        self.current_index = max(0, min(self.current_index, len(self.current_samples) - 1))

        (
            info,
            label_info,
            raw_response,
            formatted_response,
            single_image_result,
            single_image_raw_response,
            image_path,
            multi_image_nav,
        ) = self.get_sample_info(self.current_index)
        
        # 获取当前样本的所有图片路径
        sample = self.current_samples[self.current_index]
        image_paths = sample.get('image_paths', [])
        if not image_paths and image_path:
            image_paths = [image_path]
        
        # 加载所有图片（使用路径而不是PIL对象，以便点击放大时显示原图）
        images = []

        draw_bbox_fn = None
        if show_bbox:
            try:
                from bbox_utils import draw_bounding_boxes_with_cache as draw_bbox_fn
            except Exception as e:
                draw_bbox_fn = None
                print(f"⚠️ BBox 绘制不可用，将展示原图：{e}")
        
        for img_path in image_paths:
            if not img_path:
                continue
                
            # 正确提取JSON路径
            json_path = os.path.splitext(img_path)[0] + ".json"
            
            # 根据show_bbox参数决定是否绘制边界框
            if draw_bbox_fn and json_path and os.path.exists(json_path):
                image_path_to_load = draw_bbox_fn(img_path, json_path, draw_text=True, save_to_cache=True)
            else:
                image_path_to_load = img_path
            
            # 直接使用图片路径，Gradio会自动处理缩略图和原图
            if os.path.exists(image_path_to_load) and os.path.isfile(image_path_to_load):
                images.append(image_path_to_load)
            else:
                print(f"❌ 图片不存在：{image_path_to_load}")
        
        bbox_status = "✅ 显示边界框" if show_bbox else "❌ 隐藏边界框"
        total = len(image_paths)
        current_img_idx = sample.get("current_image_index", 0)
        if not isinstance(current_img_idx, int):
            current_img_idx = 0
        if total > 0:
            current_img_idx = max(0, min(current_img_idx, total - 1))
            current_name = os.path.basename(image_paths[current_img_idx]) if current_img_idx < len(image_paths) else ""
            image_info = f"当前: {current_img_idx + 1}/{total} {current_name}\n{bbox_status}"
        else:
            image_info = f"共 {len(images)} 张图片\n{bbox_status}"
        
        return (
            info,
            label_info,
            raw_response,
            formatted_response,
            single_image_result,
            single_image_raw_response,
            images,
            image_info,
            multi_image_nav,
        )
    
    def jump_to_sample(self, target_index: int, show_bbox: bool = True) -> Tuple[str, str, str, str, str, str, List, str, str]:
        """跳转到指定样本"""
        if not self.current_samples:
            return "无数据", "", "", "", "", "", [], "", ""

        target_index = max(1, min(target_index, len(self.current_samples))) - 1
        self.current_index = target_index

        return self.navigate_sample(0, show_bbox)
    
    def get_filter_options(self) -> List[str]:
        """获取筛选选项列表"""
        if not self.error_data:
            return ["全部"]
        
        if self.data_format == 'error_analysis':
            # 错误分析格式：按错误类型筛选
            error_types = set()
            for sample in self.error_data['error_samples']:
                error_types.add(sample.get('error_type', '未知'))
            options = ["全部"] + sorted(list(error_types))
        
        elif self.data_format == 'vllm_format':
            # vLLM格式：按错误类型筛选
            error_types = set()
            all_samples = self.convert_vllm_format_to_samples(self.error_data)
            for sample in all_samples:
                error_types.add(sample.get('error_type', '未知'))
            options = ["全部"] + sorted(list(error_types))
        
        else:  # new_format
            # 新格式：按模型预测结果筛选
            predictions = set()
            all_samples = self.convert_new_format_to_samples(self.error_data)
            for sample in all_samples:
                predictions.add(sample.get('model_prediction', '未知'))
            options = ["全部"] + sorted(list(predictions))

        if self.extracted_order and self.extracted_filter_name not in options:
            options.insert(1, self.extracted_filter_name)
        return options

    # ==================== 抽取功能 ====================

    def _get_raw_unique_id(self, raw_item: Dict[str, Any]) -> str:
        """生成原始样本唯一ID（用于抽取去重）"""
        image_paths = raw_item.get("image_paths")
        if isinstance(image_paths, list) and image_paths:
            return "||".join([str(p) for p in image_paths])
        if isinstance(image_paths, str) and image_paths:
            return image_paths
        images = raw_item.get("images")
        if isinstance(images, list) and images:
            return "||".join([str(p) for p in images])
        if isinstance(images, str) and images:
            return images
        image_path = raw_item.get("image_path") or raw_item.get("original_image_path") or ""
        if image_path:
            return str(image_path)
        return f"{raw_item.get('sample_index', '')}_{raw_item.get('image_name', '')}"

    def _iter_extracted_raw_items(self) -> List[Dict[str, Any]]:
        return [self.extracted_raw_by_id[uid] for uid in self.extracted_order if uid in self.extracted_raw_by_id]

    def _get_extracted_samples_for_view(self) -> List[Dict[str, Any]]:
        """将抽取到的原始样本转换为当前可展示的 samples"""
        raw_items = self._iter_extracted_raw_items()
        if not raw_items:
            return []

        if self.data_format == "vllm_format":
            return self.convert_vllm_format_to_samples(raw_items)
        if self.data_format == "new_format":
            return self.convert_new_format_to_samples(raw_items)
        if self.data_format == "error_analysis":
            return raw_items

        detected_format = self.detect_data_format(raw_items)
        if detected_format == "vllm_format":
            return self.convert_vllm_format_to_samples(raw_items)
        if detected_format == "new_format":
            return self.convert_new_format_to_samples(raw_items)
        return raw_items

    def _get_prediction_from_raw_item(self, raw_item: Dict[str, Any]) -> str:
        """从原始样本中提取预测结果（尽量兼容不同格式）"""
        for key in ("prediction", "model_prediction", "pred", "result"):
            if key in raw_item and raw_item.get(key) is not None:
                return str(raw_item.get(key))

        messages = raw_item.get("messages")
        if isinstance(messages, list):
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                if msg.get("role") != "assistant":
                    continue
                content = msg.get("content")
                if content is None:
                    continue
                if isinstance(content, (dict, list)):
                    if isinstance(content, dict) and content.get("result") is not None:
                        return str(content.get("result"))
                    return json.dumps(content, ensure_ascii=False)
                content_str = str(content)
                try:
                    parsed = json.loads(content_str)
                    if isinstance(parsed, dict) and parsed.get("result") is not None:
                        return str(parsed.get("result"))
                except Exception:
                    pass
                return content_str

        return ""

    def get_extraction_stats(self) -> Tuple[int, int, int, int]:
        """返回(总数, 预测FAIL, 预测PASS, 其他)"""
        total = 0
        pred_fail = 0
        pred_pass = 0
        other = 0
        for raw_item in self._iter_extracted_raw_items():
            total += 1
            pred = self._get_prediction_from_raw_item(raw_item).upper()
            if pred == "FAIL":
                pred_fail += 1
            elif pred == "PASS":
                pred_pass += 1
            else:
                other += 1
        return total, pred_fail, pred_pass, other

    def get_extraction_stats_text(self) -> str:
        total, pred_fail, pred_pass, other = self.get_extraction_stats()
        return f"已抽取: {total} | 预测(FAIL): {pred_fail} | 预测(PASS): {pred_pass} | 其他: {other}"

    def clear_extraction(self) -> str:
        """清空抽取结果"""
        self.extracted_raw_by_id = {}
        self.extracted_order = []
        return "✅ 已清空抽取结果"

    def _get_current_raw_item(self) -> Tuple[Optional[Dict[str, Any]], str]:
        """获取当前样本对应的原始数据条目"""
        if not self.current_samples:
            return None, "❌ 当前没有样本"
        if self.current_index < 0 or self.current_index >= len(self.current_samples):
            return None, "❌ 当前样本索引无效"
        sample = self.current_samples[self.current_index]
        if not isinstance(sample, dict):
            return None, "❌ 当前样本数据异常"
        raw_item = sample.get("_raw_item")
        if not isinstance(raw_item, dict):
            # 兼容 error_analysis 等：sample 本身就是原始条目
            return sample, ""
        return raw_item, ""

    def extract_current_sample(self) -> str:
        """抽取当前样本（以一组多图为单位）"""
        raw_item, err = self._get_current_raw_item()
        if err:
            return err

        uid = self._get_raw_unique_id(raw_item)
        if uid in self.extracted_raw_by_id:
            return f"⚠️ 当前样本已在抽取列表中；当前总计 {len(self.extracted_order)} 条"

        self.extracted_raw_by_id[uid] = raw_item
        self.extracted_order.append(uid)
        return f"✅ 已抽取当前样本；当前总计 {len(self.extracted_order)} 条"

    def remove_current_sample_from_extraction(self) -> str:
        """从抽取结果中删除当前样本（以一组多图为单位）"""
        raw_item, err = self._get_current_raw_item()
        if err:
            return err

        uid = self._get_raw_unique_id(raw_item)
        if uid not in self.extracted_raw_by_id:
            return f"❌ 当前样本不在抽取列表中；当前总计 {len(self.extracted_order)} 条"

        self.extracted_raw_by_id.pop(uid, None)
        self.extracted_order = [x for x in self.extracted_order if x != uid]
        return f"✅ 已从抽取中删除当前样本；当前总计 {len(self.extracted_order)} 条"

    def _raw_item_to_multi_json(self, raw_item: Dict[str, Any]) -> Dict[str, Any]:
        """将 detailed_results 单条转为 multi.json 单条"""
        label = raw_item.get("true_label", "")
        label_str = str(label) if label is not None else ""

        image_paths = raw_item.get("image_paths")
        if isinstance(image_paths, str):
            images = [image_paths] if image_paths else []
        elif isinstance(image_paths, list):
            images = [str(p) for p in image_paths if p]
        else:
            raw_images = raw_item.get("images")
            if isinstance(raw_images, str):
                images = [raw_images] if raw_images else []
            elif isinstance(raw_images, list):
                images = [str(p) for p in raw_images if p]
            else:
                single = raw_item.get("image_path") or ""
                images = [str(single)] if single else []

        return {
            "messages": [
                {"role": "user", "content": "<image>"},
                {"role": "assistant", "content": label_str},
            ],
            "images": images,
        }

    def export_extraction(self, detailed_results_path: str, multi_json_path: str) -> str:
        """导出抽取结果到 detailed_results.json 与 multi.json"""
        if not self.extracted_order:
            return "❌ 还没有抽取任何数据"

        if not detailed_results_path or not str(detailed_results_path).strip():
            return "❌ 请填写 detailed_results.json 导出路径"
        if not multi_json_path or not str(multi_json_path).strip():
            return "❌ 请填写 multi.json 导出路径"

        detailed_results_path = str(detailed_results_path).strip()
        multi_json_path = str(multi_json_path).strip()

        raw_items = self._iter_extracted_raw_items()
        exported_raw_items = []
        for new_index, raw_item in enumerate(raw_items):
            if isinstance(raw_item, dict):
                raw_item_copy = raw_item.copy()
                raw_item_copy["sample_index"] = new_index
                exported_raw_items.append(raw_item_copy)
            else:
                exported_raw_items.append(raw_item)

        multi_items = [self._raw_item_to_multi_json(item) for item in exported_raw_items]

        try:
            detailed_parent = os.path.dirname(detailed_results_path)
            if detailed_parent:
                os.makedirs(detailed_parent, exist_ok=True)
            multi_parent = os.path.dirname(multi_json_path)
            if multi_parent:
                os.makedirs(multi_parent, exist_ok=True)

            with open(detailed_results_path, "w", encoding="utf-8") as f:
                json.dump(exported_raw_items, f, ensure_ascii=False, indent=2)

            with open(multi_json_path, "w", encoding="utf-8") as f:
                json.dump(multi_items, f, ensure_ascii=False, indent=2)

            total, pred_fail, pred_pass, other = self.get_extraction_stats()
            return (
                "✅ 导出完成\n"
                f"- detailed_results.json: {detailed_results_path} ({len(exported_raw_items)} 条)\n"
                f"- multi.json: {multi_json_path} ({len(multi_items)} 条)\n"
                f"- 统计: 预测(FAIL)={pred_fail}, 预测(PASS)={pred_pass}, 其他={other}, 总计={total}"
            )
        except Exception as e:
            return f"❌ 导出失败: {str(e)}"

    def export_extraction_to_dir(self, output_dir: str) -> str:
        """导出抽取结果到指定文件夹下（生成 detailed_results.json 与 multi.json）"""
        if not self.extracted_order:
            return "❌ 还没有抽取任何数据"

        if not output_dir or not str(output_dir).strip():
            return "❌ 请填写输出文件夹路径"

        output_dir = str(output_dir).strip()

        try:
            if os.path.exists(output_dir) and not os.path.isdir(output_dir):
                return f"❌ 输出路径不是文件夹: {output_dir}"
            os.makedirs(output_dir, exist_ok=True)

            detailed_results_path = os.path.join(output_dir, "detailed_results.json")
            multi_json_path = os.path.join(output_dir, "multi.json")
            return self.export_extraction(detailed_results_path, multi_json_path)
        except Exception as e:
            return f"❌ 创建输出文件夹失败: {str(e)}"

def create_interface(json_file_path=None):
    """创建Gradio界面"""
    import gradio as gr
    viewer = SimpleErrorViewer()
    
    # 如果指定了JSON文件路径，预加载数据
    initial_load_status = "请选择JSON文件"
    initial_filter_options = ["全部"]
    initial_sample_data = ["无数据", "", "", "", "", "", [], "", ""]
    
    if json_file_path:
        load_msg, _ = viewer.load_json_file(json_file_path)
        initial_load_status = load_msg
        initial_filter_options = viewer.get_filter_options()
        # 获取第一个样本的数据
        if viewer.current_samples:
            initial_sample_data = list(viewer.navigate_sample(0))
    
    initial_output_path = viewer.default_output_path or ""
    initial_selected_index = 0
    try:
        if viewer.current_samples:
            sample0 = viewer.current_samples[viewer.current_index]
            idx0 = sample0.get("current_image_index", 0) if isinstance(sample0, dict) else 0
            if isinstance(idx0, int) and idx0 >= 0:
                initial_selected_index = idx0
    except Exception:
        initial_selected_index = 0
    
    initial_group_progress = "📦 当前组: 0/0"
    if viewer.current_samples:
        initial_group_progress = f"📦 当前组: {viewer.current_index + 1}/{len(viewer.current_samples)}"
    initial_label_comparison_value = initial_sample_data[1] or ""
    if initial_group_progress:
        initial_label_comparison_value = f"{initial_group_progress}\n\n{initial_label_comparison_value}".strip()

    with gr.Blocks(title="多图/单图标注工具", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🏷️ 多图/单图标注工具")
        if json_file_path:
            gr.Markdown(f"**已加载文件：** `{os.path.basename(json_file_path)}`")
        else:
            gr.Markdown("选择JSON文件开始标注（支持多图样本展示）")
        
        # 文件上传（如果预加载了文件则隐藏）
        if not json_file_path:
            file_input = gr.File(
                label="📁 选择JSON文件（标注数据）",
                file_types=[".json"],
                type="filepath"
            )
        
        # 主要布局：左边图片，右边内容
        with gr.Row():
            gallery_columns = 8  # 缩略图列数，供事件计算索引用
            # 左侧：大图 + 下方缩略图（Gallery 预览模式）
            with gr.Column(scale=1):
                image_gallery = gr.Gallery(
                    label="🖼️ 当前组图片（点击缩略图选择当前图片）",
                    value=initial_sample_data[6],
                    columns=gallery_columns,
                    rows=1,
                    height=720,
                    object_fit="contain",
                    show_label=True,
                    interactive=True,
                    allow_preview=True,
                    preview=True,
                    selected_index=initial_selected_index,
                    show_download_button=True,
                )
                
                label_comparison = gr.Textbox(
                    label="📝 标注信息（含合并结果/进度）",
                    value=initial_label_comparison_value,
                    interactive=False,
                    lines=10,
                )
                
                # 边界框显示控制
                show_bbox_checkbox = gr.Checkbox(
                    label="🔲 显示边界框 (BBox)",
                    value=True,
                    info="勾选后显示标注的边界框，取消勾选显示原始图片"
                )
                
                # 多图导航控件（保留用于显示信息）
                multi_image_nav_display = gr.Textbox(
                    label="📊 多图统计",
                    value=initial_sample_data[8],
                    interactive=False,
                    lines=2
                )
            
            # 右侧：模型输出和控件
            with gr.Column(scale=1):
                # 上半部分：标注控件
                gr.Markdown("### 📝 标注")

                label_mode_dropdown = gr.Dropdown(
                    label="标注模式（开始标注前选择）",
                    choices=[
                        ("多图逻辑（整组只标注一个结果）", "multi"),
                        ("单图逻辑（组内每张图标注并自动合并）", "single"),
                    ],
                    value="multi",
                    interactive=True,
                )

                output_path_box = gr.Textbox(
                    label="💾 输出标注结果 JSON 路径（默认：与输入同目录 + _labeld）",
                    value=initial_output_path,
                    placeholder="例如：/data_all/lyh/LLaMA-Factory_1124/tmp/a_labeld.json",
                    lines=1,
                )

                with gr.Row():
                    auto_save_checkbox = gr.Checkbox(
                        label="自动保存（每完成一组标注）",
                        value=True,
                    )

                with gr.Row():
                    label_pass_btn = gr.Button("✅ PASS", variant="primary", size="sm")
                    label_fail_btn = gr.Button("❌ FAIL", variant="stop", size="sm")
                    label_ni_btn = gr.Button("➖ NOT_INVOLVED", variant="secondary", size="sm")

                with gr.Row():
                    save_btn = gr.Button("💾 手动保存", variant="secondary", size="sm")

                annotation_status = gr.Textbox(
                    label="操作结果",
                    value="",
                    interactive=False,
                    lines=10,
                )

                gr.Markdown("---")
                
                # 下半部分：导航和筛选控件
                with gr.Row():
                    # 样本导航按钮
                    prev_btn = gr.Button("⬅️ 上一个样本", variant="secondary", size="sm")
                    next_btn = gr.Button("下一个样本 ➡️", variant="secondary", size="sm")
                
                with gr.Row():
                    # 图片切换按钮（仅多图时有效）
                    prev_img_btn = gr.Button("🖼️ ⬅️ 上一张图", variant="primary", size="sm")
                    next_img_btn = gr.Button("下一张图 ➡️ 🖼️", variant="primary", size="sm")
                
                with gr.Row():
                    # 跳转控件
                    jump_input = gr.Number(
                        label="跳转到第几个",
                        value=1,
                        minimum=1,
                        step=1,
                        precision=0,
                        scale=2
                    )
                    jump_btn = gr.Button("🎯 跳转", variant="primary", size="sm", scale=1)
                
                with gr.Row():
                    # 筛选控件
                    filter_dropdown = gr.Dropdown(
                        label="🔽 筛选选项",
                        choices=initial_filter_options,
                        value="全部",
                        interactive=True
                    )
                
                # 筛选和加载状态
                filter_status = gr.Textbox(
                    label="筛选结果",
                    value="未筛选",
                    interactive=False,
                    lines=1
                )
                
                # 加载状态（预加载文件时也需要创建，用于实时更新）
                load_status = gr.Textbox(
                    label="📊 加载状态",
                    value=initial_load_status,
                    interactive=False,
                    lines=4
                )
        
        # 事件绑定
        def pack_view(sample_data):
            """将 navigate_sample 的9元组裁剪为界面需要的部分（Gallery 预览+缩略图）"""
            # (info, label_info, raw_response, formatted_response, single_image_result,
            #  single_image_raw_response, images, image_info, multi_image_nav)
            thumbs = sample_data[6] or []
            group_progress = "📦 当前组: 0/0"
            if viewer.current_samples:
                group_progress = f"📦 当前组: {viewer.current_index + 1}/{len(viewer.current_samples)}"
            label_text = sample_data[1] or ""
            if group_progress:
                label_text = f"{group_progress}\n\n{label_text}".strip()

            selected_index = 0
            if viewer.current_samples:
                sample = viewer.current_samples[viewer.current_index]
                idx = sample.get("current_image_index", 0) if isinstance(sample, dict) else 0
                if isinstance(idx, int):
                    selected_index = idx

            if not thumbs:
                gallery_update = gr.update(value=[], selected_index=None)
            else:
                if selected_index < 0 or selected_index >= len(thumbs):
                    selected_index = 0
                gallery_update = gr.update(value=thumbs, selected_index=selected_index)

            return [label_text, gallery_update, sample_data[8]]

        def on_file_change(file_path):
            load_msg, _ = viewer.load_json_file(file_path)
            filter_options = viewer.get_filter_options()
            sample_data = viewer.navigate_sample(0)
            
            # 根据数据格式更新下拉框标签
            if viewer.data_format == 'error_analysis':
                dropdown_label = "🔽 筛选错误类型"
            elif viewer.data_format == 'vllm_format':
                dropdown_label = "🔽 筛选错误类型"
            elif viewer.data_format == 'new_format':
                dropdown_label = "🔽 筛选预测结果"
            else:
                dropdown_label = "🔽 筛选选项"
            
            return [
                load_msg,
                gr.Dropdown(choices=filter_options, value="全部", label=dropdown_label),
                viewer.default_output_path or "",
                *pack_view(sample_data),
                viewer.get_current_annotation_text() if viewer.current_samples else "",
            ]
        
        def on_filter_change(filter_value, show_bbox):
            filter_msg = viewer.filter_samples(filter_value)
            return [
                filter_msg,
                *pack_view(viewer.navigate_sample(0, show_bbox)),
                viewer.get_current_annotation_text() if viewer.current_samples else "",
            ]
        
        def on_prev_click(show_bbox):
            return [
                *pack_view(viewer.navigate_sample(-1, show_bbox)),
                viewer.get_current_annotation_text() if viewer.current_samples else "",
            ]
        
        def on_next_click(show_bbox):
            return [
                *pack_view(viewer.navigate_sample(1, show_bbox)),
                viewer.get_current_annotation_text() if viewer.current_samples else "",
            ]
        
        def on_prev_img_click(show_bbox):
            return pack_view(viewer.switch_image(-1, show_bbox))
        
        def on_next_img_click(show_bbox):
            return pack_view(viewer.switch_image(1, show_bbox))
        
        def on_jump_click(target_index, show_bbox):
            return [
                *pack_view(viewer.jump_to_sample(target_index, show_bbox)),
                viewer.get_current_annotation_text() if viewer.current_samples else "",
            ]
        
        def on_bbox_toggle(show_bbox):
            """当复选框状态改变时，重新加载当前样本"""
            return pack_view(viewer.navigate_sample(0, show_bbox))
        
        def on_gallery_select(show_bbox, evt: gr.SelectData):
            """点击图库缩略图时切换当前图片索引"""
            if not viewer.current_samples:
                return pack_view(viewer.navigate_sample(0, show_bbox))
            
            sample = viewer.current_samples[viewer.current_index]
            image_paths = sample.get('image_paths', [])
            
            idx = None
            # Gradio Gallery select event提供 index（可能为int或(row, col)）
            event_index = getattr(evt, 'index', None) if evt is not None else None
            if isinstance(event_index, int):
                idx = event_index
            elif isinstance(event_index, (list, tuple)) and len(event_index) >= 2:
                # Gallery按行列返回索引，例如(row, col)
                try:
                    idx = event_index[0] * gallery_columns + event_index[1]
                except Exception:
                    idx = None
            
            if idx is None:
                selected_value = getattr(evt, 'value', None) if evt is not None else None
                if selected_value is None:
                    selected_value = evt
                target_path = None
                if isinstance(selected_value, str):
                    target_path = selected_value
                elif isinstance(selected_value, (list, tuple)) and selected_value:
                    target_path = selected_value[0]
                elif isinstance(selected_value, dict):
                    target_path = selected_value.get('name') or selected_value.get('image') or selected_value.get('value')
                if target_path:
                    if target_path in image_paths:
                        idx = image_paths.index(target_path)
                    else:
                        # 兼容 BBox 缓存图路径：尝试在当前展示的 Gallery 值里匹配
                        try:
                            displayed = viewer.navigate_sample(0, show_bbox)[6] or []
                        except Exception:
                            displayed = []
                        if target_path in displayed:
                            idx = displayed.index(target_path)
            
            if idx is not None and 0 <= idx < len(image_paths):
                sample['current_image_index'] = idx
                sample['original_image_path'] = image_paths[idx]
                sample['copied_image_path'] = image_paths[idx]
            
            return pack_view(viewer.navigate_sample(0, show_bbox))

        def on_mode_change(label_mode, show_bbox):
            """切换标注模式并刷新展示"""
            msg = viewer.set_label_mode(label_mode)
            sample_data = viewer.navigate_sample(0, show_bbox)
            return [msg, *pack_view(sample_data)]

        def on_label_click(label, label_mode, output_path, auto_save, show_bbox):
            """对当前样本应用标注并刷新展示"""
            status, *sample_data = viewer.apply_label(label, label_mode, output_path, auto_save, show_bbox)
            return [status, *pack_view(sample_data)]

        def on_save_click(output_path):
            """手动保存当前数据"""
            if not output_path:
                output_path = viewer.default_output_path or ""
            return viewer.save_labeled_json(output_path)
        
        # 绑定事件（只有在没有预加载文件时才绑定文件上传事件）
        if not json_file_path:
            file_input.change(
                fn=on_file_change,
                inputs=[file_input],
                outputs=[
                    load_status,
                    filter_dropdown,
                    output_path_box,
                    label_comparison,
                    image_gallery,
                    multi_image_nav_display,
                    annotation_status,
                ]
            )
        
        filter_dropdown.change(
            fn=on_filter_change,
            inputs=[filter_dropdown, show_bbox_checkbox],
            outputs=[
                filter_status,
                label_comparison,
                image_gallery,
                multi_image_nav_display,
                annotation_status,
            ]
        )
        
        prev_btn.click(
            fn=on_prev_click,
            inputs=[show_bbox_checkbox],
            outputs=[
                label_comparison,
                image_gallery,
                multi_image_nav_display,
                annotation_status,
            ]
        )
        
        next_btn.click(
            fn=on_next_click,
            inputs=[show_bbox_checkbox],
            outputs=[
                label_comparison,
                image_gallery,
                multi_image_nav_display,
                annotation_status,
            ]
        )
        
        # 图片切换按钮事件
        prev_img_btn.click(
            fn=on_prev_img_click,
            inputs=[show_bbox_checkbox],
            outputs=[
                label_comparison,
                image_gallery,
                multi_image_nav_display,
            ]
        )
        
        next_img_btn.click(
            fn=on_next_img_click,
            inputs=[show_bbox_checkbox],
            outputs=[
                label_comparison,
                image_gallery,
                multi_image_nav_display,
            ]
        )
        
        jump_btn.click(
            fn=on_jump_click,
            inputs=[jump_input, show_bbox_checkbox],
            outputs=[
                label_comparison,
                image_gallery,
                multi_image_nav_display,
                annotation_status,
            ]
        )
        
        # 边界框复选框事件
        show_bbox_checkbox.change(
            fn=on_bbox_toggle,
            inputs=[show_bbox_checkbox],
            outputs=[
                label_comparison,
                image_gallery,
                multi_image_nav_display,
            ]
        )

        image_gallery.select(
            fn=on_gallery_select,
            inputs=[show_bbox_checkbox],
            outputs=[
                label_comparison,
                image_gallery,
                multi_image_nav_display,
            ]
        )

        # 标注模式切换
        label_mode_dropdown.change(
            fn=on_mode_change,
            inputs=[label_mode_dropdown, show_bbox_checkbox],
            outputs=[
                annotation_status,
                label_comparison,
                image_gallery,
                multi_image_nav_display,
            ],
        )

        # 标注按钮
        label_pass_btn.click(
            fn=lambda label_mode, output_path, auto_save, show_bbox: on_label_click(
                "PASS", label_mode, output_path, auto_save, show_bbox
            ),
            inputs=[label_mode_dropdown, output_path_box, auto_save_checkbox, show_bbox_checkbox],
            outputs=[
                annotation_status,
                label_comparison,
                image_gallery,
                multi_image_nav_display,
            ],
        )

        label_fail_btn.click(
            fn=lambda label_mode, output_path, auto_save, show_bbox: on_label_click(
                "FAIL", label_mode, output_path, auto_save, show_bbox
            ),
            inputs=[label_mode_dropdown, output_path_box, auto_save_checkbox, show_bbox_checkbox],
            outputs=[
                annotation_status,
                label_comparison,
                image_gallery,
                multi_image_nav_display,
            ],
        )

        label_ni_btn.click(
            fn=lambda label_mode, output_path, auto_save, show_bbox: on_label_click(
                "NOT_INVOLVED", label_mode, output_path, auto_save, show_bbox
            ),
            inputs=[label_mode_dropdown, output_path_box, auto_save_checkbox, show_bbox_checkbox],
            outputs=[
                annotation_status,
                label_comparison,
                image_gallery,
                multi_image_nav_display,
            ],
        )

        # 手动保存
        save_btn.click(
            fn=on_save_click,
            inputs=[output_path_box],
            outputs=[annotation_status],
        )
        
        # 使用说明
        gr.Markdown("""
### 📖 使用说明：
1. **选择文件**：加载需要标注的 JSON（每条样本包含 `images` 列表）
2. **选择模式**（开始标注前）：
   - **多图逻辑**：一组图片只标注一个结果（PASS/FAIL/NOT_INVOLVED），写入 `{"analysis": [], "result": "..."}`  
   - **单图逻辑**：对组内每张图片标注（PASS/FAIL/NOT_INVOLVED），全部完成后自动合并：  
     - 存在 FAIL → 最终 FAIL  
     - 不存在 FAIL 且存在 PASS → 最终 PASS  
     - 全为 NOT_INVOLVED → 最终 FAIL  
3. **标注**：点击 **PASS/FAIL/NOT_INVOLVED** 按钮（单图逻辑会自动跳到下一张；当前组完成后两种模式都会自动跳到下一组第一张）
4. **保存**：默认输出到输入文件同目录、文件名加 `_labeld`，也可手动修改输出路径并点击 **手动保存**
5. **浏览**：使用上一/下一样本、上一/下一图、跳转；点击下方缩略图可切换“当前图片”（用于显示与单图标注）
        """)
    
    return demo

def main():
    parser = argparse.ArgumentParser(description='多图/单图标注工具（Gradio）')
    parser.add_argument('--port', type=int, default=7870, help='服务器端口 (默认: 7870)')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='服务器地址 (默认: 127.0.0.1)')
    parser.add_argument('--share', action='store_true', help='生成公共分享链接')
    parser.add_argument('--json-file', type=str, default='', help='预加载的JSON文件路径（可选）')
    
    args = parser.parse_args()
    
    # 检查依赖
    try:
        import gradio as gr
        from PIL import Image
    except ImportError as e:
        print(f"❌ 缺少依赖库: {e}")
        print("请安装依赖: pip install gradio pillow")
        return
    
    # 设置 Gradio 临时目录为当前用户有权限的目录
    import tempfile
    gradio_temp_dir = os.path.join(os.path.expanduser("~"), ".gradio", "temp")
    os.makedirs(gradio_temp_dir, exist_ok=True)
    os.environ["GRADIO_TEMP_DIR"] = gradio_temp_dir
    
    print("🚀 启动多图/单图标注工具...")
    print(f"📍 地址: http://{args.host}:{args.port}")
    if args.share:
        print("🌐 公共分享链接将在启动后显示")
    
    # 创建并启动界面
    demo = create_interface(json_file_path=args.json_file or None)
    
    try:
        demo.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            show_error=True,
            quiet=False,
            allowed_paths=[
                "/data_all/share",
                "/data_all/lyh",
                os.getcwd(),
                os.path.expanduser("~"),
            ]
        )
    except Exception as e:
        print(f"❌ 启动失败: {e}")

if __name__ == "__main__":
    main()
