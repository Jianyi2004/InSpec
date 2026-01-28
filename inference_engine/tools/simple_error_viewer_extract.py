#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单错误样本查看器 - 单文件版本
使用 Gradio 框架，只需要一个 Python 文件就能运行
"""

import json
import os
import argparse
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import tempfile

# 设置 Gradio 临时目录为当前用户有权限的目录
os.environ['GRADIO_TEMP_DIR'] = os.path.join(tempfile.gettempdir(), f'gradio_{os.getuid()}')

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
        # 正确性判断功能
        self.correctness_judgments = {}  # 正确性判断: {sample_index: True/False} True=正确, False=错误
        
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
            return None, "❌ 当前样本缺少原始数据，无法抽取/删除"
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
    
    # ==================== 正确性判断功能 ====================
    
    def judge_current_sample(self, is_correct: bool) -> str:
        """判断当前样本的模型预测是否正确"""
        if not self.current_samples:
            return "❌ 没有数据"
        
        if self.current_index < 0 or self.current_index >= len(self.current_samples):
            return "❌ 索引无效"
        
        sample = self.current_samples[self.current_index]
        sample_idx = sample.get('_raw_item_index', sample.get('sample_index', self.current_index))
        
        self.correctness_judgments[sample_idx] = is_correct
        
        # 自动跳转到下一个未判断的样本
        next_index = self.find_next_unjudged()
        if next_index != -1:
            self.current_index = next_index
            status = "正确" if is_correct else "错误"
            return f"✅ 已标记为{status}，自动跳转到下一个未判断样本"
        else:
            status = "正确" if is_correct else "错误"
            return f"✅ 已标记为{status}，所有样本已判断完成！"
    
    def find_next_unjudged(self) -> int:
        """查找下一个未判断的样本索引"""
        # 从当前位置往后找
        for i in range(self.current_index + 1, len(self.current_samples)):
            sample = self.current_samples[i]
            sample_idx = sample.get('_raw_item_index', sample.get('sample_index', i))
            if sample_idx not in self.correctness_judgments:
                return i
        
        # 从头开始找
        for i in range(0, self.current_index):
            sample = self.current_samples[i]
            sample_idx = sample.get('_raw_item_index', sample.get('sample_index', i))
            if sample_idx not in self.correctness_judgments:
                return i
        
        return -1  # 全部判断完成
    
    def get_judgment_stats(self) -> Tuple[int, int, int, int]:
        """返回(总数, 已判断, 判断为正确, 判断为错误)"""
        total = len(self.current_samples)
        judged = len(self.correctness_judgments)
        correct_count = sum(1 for v in self.correctness_judgments.values() if v)
        incorrect_count = sum(1 for v in self.correctness_judgments.values() if not v)
        return total, judged, correct_count, incorrect_count
    
    def get_judgment_stats_text(self) -> str:
        """获取判断统计文本"""
        total, judged, correct_count, incorrect_count = self.get_judgment_stats()
        unjudged = total - judged
        accuracy = (correct_count / judged * 100) if judged > 0 else 0
        return f"总数: {total} | 已判断: {judged} | 正确: {correct_count} | 错误: {incorrect_count} | 未判断: {unjudged} | 准确率: {accuracy:.1f}%"
    
    def export_judgment_results(self, output_dir: str = None) -> str:
        """导出判断结果"""
        if not self.correctness_judgments:
            return "❌ 还没有判断任何数据"
        
        # 确定输出目录
        if not output_dir or not output_dir.strip():
            if self.json_file_path:
                output_dir = os.path.dirname(self.json_file_path)
            else:
                return "❌ 请指定输出目录"
        
        output_dir = output_dir.strip()
        os.makedirs(output_dir, exist_ok=True)
        
        # 分类样本
        correct_samples = []  # 模型预测正确的
        incorrect_samples = []  # 模型预测错误的
        all_samples = []  # 全部数据
        
        for sample_idx, is_correct in self.correctness_judgments.items():
            # 查找对应的原始样本
            raw_item = None
            for sample in self.current_samples:
                if sample.get('_raw_item_index', sample.get('sample_index')) == sample_idx:
                    raw_item = sample.get('_raw_item')
                    break
            
            if not raw_item:
                continue
            
            # 提取图片路径
            image_paths = raw_item.get('image_paths', [])
            if not image_paths:
                image_path = raw_item.get('image_path', '')
                image_paths = [image_path] if image_path else []
            
            # 提取模型预测结果
            prediction = raw_item.get('prediction', raw_item.get('result', 'UNKNOWN'))
            
            # 推断ground truth
            # 如果用户判断"正确" → ground truth = 模型预测
            # 如果用户判断"错误" → ground truth = 模型预测的反面
            if is_correct:
                ground_truth = str(prediction)
            else:
                # 预测错误，ground truth是反面
                if str(prediction).upper() == "PASS":
                    ground_truth = "FAIL"
                elif str(prediction).upper() == "FAIL":
                    ground_truth = "PASS"
                else:
                    ground_truth = "UNKNOWN"
            
            # 构造multi.json格式 - assistant的content是ground truth
            multi_item = {
                "messages": [
                    {"role": "user", "content": "<image>"},
                    {"role": "assistant", "content": ground_truth}
                ],
                "images": image_paths
            }
            
            # 添加到对应列表
            all_samples.append(multi_item)
            if is_correct:
                correct_samples.append(multi_item)
            else:
                incorrect_samples.append(multi_item)
        
        # 保存文件
        try:
            # 正确数据 (multi.json格式)
            correct_path = os.path.join(output_dir, "correct_predictions.json")
            with open(correct_path, 'w', encoding='utf-8') as f:
                json.dump(correct_samples, f, ensure_ascii=False, indent=2)
            
            # 错误数据 (multi.json格式)
            incorrect_path = os.path.join(output_dir, "incorrect_predictions.json")
            with open(incorrect_path, 'w', encoding='utf-8') as f:
                json.dump(incorrect_samples, f, ensure_ascii=False, indent=2)
            
            # 全部数据 (multi.json格式)
            all_path = os.path.join(output_dir, "all_predictions.json")
            with open(all_path, 'w', encoding='utf-8') as f:
                json.dump(all_samples, f, ensure_ascii=False, indent=2)
            
            # 统计正负样本数量
            positive_samples = sum(1 for item in all_samples if item["messages"][1]["content"].upper() == "PASS")
            negative_samples = sum(1 for item in all_samples if item["messages"][1]["content"].upper() == "FAIL")
            
            # 统计正负样本准确率
            positive_correct = sum(1 for item in correct_samples if item["messages"][1]["content"].upper() == "PASS")
            negative_correct = sum(1 for item in correct_samples if item["messages"][1]["content"].upper() == "FAIL")
            
            positive_accuracy = (positive_correct / positive_samples * 100) if positive_samples > 0 else 0
            negative_accuracy = (negative_correct / negative_samples * 100) if negative_samples > 0 else 0
            
            # 统计文件
            total, judged, correct_count, incorrect_count = self.get_judgment_stats()
            stats = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source_file": self.json_file_path,
                "total_samples": total,
                "judged_samples": judged,
                "exported_samples": len(all_samples),
                "model_accuracy": (correct_count / judged * 100) if judged > 0 else 0,
                "correct_predictions": correct_count,
                "incorrect_predictions": incorrect_count,
                "positive_samples": {
                    "total": positive_samples,
                    "correct": positive_correct,
                    "incorrect": positive_samples - positive_correct,
                    "accuracy": round(positive_accuracy, 2)
                },
                "negative_samples": {
                    "total": negative_samples,
                    "correct": negative_correct,
                    "incorrect": negative_samples - negative_correct,
                    "accuracy": round(negative_accuracy, 2)
                }
            }
            
            stats_path = os.path.join(output_dir, "judgment_stats.json")
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            
            result = f"""
✅ 导出完成！

📁 输出目录：{output_dir}

📊 导出统计：
• 已判断样本：{len(all_samples)} 个
• 模型预测正确：{correct_count} 个 → correct_predictions.json
• 模型预测错误：{incorrect_count} 个 → incorrect_predictions.json
• 模型整体准确率：{stats['model_accuracy']:.2f}%

📈 正负样本分布（Ground Truth）：
• 正样本（PASS）：{positive_samples} 个
  - 模型预测正确：{positive_correct} 个
  - 模型预测错误：{positive_samples - positive_correct} 个
  - 正样本准确率：{positive_accuracy:.2f}%

• 负样本（FAIL）：{negative_samples} 个
  - 模型预测正确：{negative_correct} 个
  - 模型预测错误：{negative_samples - negative_correct} 个
  - 负样本准确率：{negative_accuracy:.2f}%

📄 文件路径：
• {correct_path} ({correct_count} 个)
• {incorrect_path} ({incorrect_count} 个)
• {all_path} ({len(all_samples)} 个)
• {stats_path}

💡 说明：
- 所有导出数据的assistant content都是ground truth（真实标签）
- 只导出已经判断的样本
- 正样本准确率 = 模型对PASS样本的预测准确率
- 负样本准确率 = 模型对FAIL样本的预测准确率
"""
            
            return result
            
        except Exception as e:
            return f"❌ 导出失败：{str(e)}"
    
    def clear_judgments(self) -> str:
        """清空所有判断"""
        self.correctness_judgments = {}
        return "✅ 已清空所有判断"

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
    
    with gr.Blocks(title="错误样本分析器", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🔍 Qwen3-VL 错误样本分析器")
        if json_file_path:
            gr.Markdown(f"**已加载文件：** `{os.path.basename(json_file_path)}`")
        else:
            gr.Markdown("上传error_analysis.json或new_format_data.json文件，查看模型推理结果")
        
        # 文件上传（如果预加载了文件则隐藏）
        if not json_file_path:
            file_input = gr.File(
                label="📁 选择JSON文件 (error_analysis.json 或 new_format_data.json)",
                file_types=[".json"],
                type="filepath"
            )
        
        # 主要布局：左边图片，右边内容
        with gr.Row():
            gallery_columns = 2  # Gallery列数，供事件计算索引用
            # 左侧：多图片显示区域（Gallery）
            with gr.Column(scale=1):
                image_gallery = gr.Gallery(
                    label="🖼️ 样本图片（点击可放大查看原图）",
                    value=initial_sample_data[6],
                    columns=gallery_columns,  # 每行显示2张图片（适合多图场景）
                    rows=1,  # 最多显示1行
                    height=600,  # 固定高度，方便查看
                    object_fit="contain",  # 保持宽高比
                    show_label=True,
                    interactive=True,
                    preview=True,  # 启用预览模式，点击可放大
                    show_download_button=True  # 显示下载按钮
                )
                
                # 图片信息
                image_path_display = gr.Textbox(
                    label="📍 图片信息",
                    value=initial_sample_data[7],
                    interactive=False,
                    lines=2
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
                # 1. 模型响应区域（最上面）
                gr.Markdown("### � 模型输出")
                with gr.Tabs():
                    with gr.TabItem("🤖 总结响应"):
                        raw_response = gr.Textbox(
                            label="模型总结响应",
                            value=initial_sample_data[2],
                            interactive=False,
                            lines=6
                        )
                    
                    with gr.TabItem("📄 格式化JSON"):
                        formatted_response = gr.Textbox(
                            label="格式化JSON响应",
                            value=initial_sample_data[3],
                            interactive=False,
                            lines=6
                        )
                    
                    with gr.TabItem("🖼️ 单图结果"):
                        single_image_result_box = gr.Textbox(
                            label="单图结果",
                            value=initial_sample_data[4],
                            interactive=False,
                            lines=6,
                            placeholder="单图结果将在后续版本中填充"
                        )
                    
                    with gr.TabItem("🧾 原始响应"):
                        single_image_raw_response_box = gr.Textbox(
                            label="模型原始响应",
                            value=initial_sample_data[5],
                            interactive=False,
                            lines=6
                        )
                
                gr.Markdown("---")
                
                # 2. 正确性判断功能
                gr.Markdown("### ✅ 模型预测正确性判断")
                
                judgment_stats = gr.Textbox(
                    label="判断统计",
                    value=viewer.get_judgment_stats_text(),
                    interactive=False,
                    lines=1
                )
                
                with gr.Row():
                    judge_correct_btn = gr.Button("✅ 预测正确", variant="primary", size="lg")
                    judge_incorrect_btn = gr.Button("❌ 预测错误", variant="stop", size="lg")
                
                judgment_status = gr.Textbox(
                    label="判断状态",
                    value="",
                    interactive=False,
                    lines=2
                )
                
                gr.Markdown("---")
                
                # 3. 抽取与导出
                gr.Markdown("### 🧲 抽取与导出")

                extraction_stats = gr.Textbox(
                    label="抽取统计（按预测）",
                    value=viewer.get_extraction_stats_text(),
                    interactive=False,
                    lines=1
                )

                with gr.Row():
                    extract_current_btn = gr.Button("🧲 抽取当前样本", variant="primary", size="sm")
                    remove_current_btn = gr.Button("�️ 从抽取删除", variant="stop", size="sm")
                    clear_extraction_btn = gr.Button("🧹 清空抽取", variant="secondary", size="sm")

                extraction_status = gr.Textbox(
                    label="操作结果",
                    value="",
                    interactive=False,
                    lines=2
                )

                gr.Markdown("---")
                
                # 4. 导出判断结果
                gr.Markdown("### 📤 导出判断结果")
                
                judgment_output_dir = gr.Textbox(
                    label="输出路径（留空则保存到源文件同目录）",
                    value="",
                    placeholder="留空则保存到源文件同目录",
                    lines=1
                )
                
                with gr.Row():
                    export_judgment_btn = gr.Button("📤 导出判断结果", variant="primary", size="sm")
                    clear_judgment_btn = gr.Button("🧹 清空判断", variant="secondary", size="sm")
                
                judgment_export_status = gr.Textbox(
                    label="导出状态",
                    value="",
                    interactive=False,
                    lines=6
                )

                gr.Markdown("---")
                
                # 5. 导出抽取结果
                gr.Markdown("### 📤 导出抽取结果")

                export_output_dir = gr.Textbox(
                    label="输出路径（自动生成 detailed_results.json + multi.json）",
                    value="/data_all/lyh/LLaMA-Factory_1124/extracted_outputs",
                    placeholder="输入输出文件夹路径...",
                    lines=1
                )

                export_extraction_btn = gr.Button("📤 导出抽取结果", variant="primary", size="sm")
                
                gr.Markdown("---")
                
                # 6. 导航和筛选控件（最下面）
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
                
                gr.Markdown("---")

                # 样本信息与标签对比
                with gr.Row():
                    with gr.Column():
                        sample_info = gr.Textbox(
                            label="📋 样本信息",
                            value=initial_sample_data[0],
                            interactive=False,
                            lines=3
                        )

                        label_comparison = gr.Textbox(
                            label="🏷️ 标签对比",
                            value=initial_sample_data[1],
                            interactive=False,
                            lines=3
                        )
                
                # 加载状态（预加载文件时也需要创建，用于实时更新）
                load_status = gr.Textbox(
                    label="📊 加载状态" if not json_file_path else "🔄 实时监控状态",
                    value=initial_load_status,
                    interactive=False,
                    lines=4
                )
        
        # 事件绑定
        def on_file_change(file_path):
            load_msg, _ = viewer.load_json_file(file_path)
            viewer.clear_extraction()
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
                *sample_data,
                "",
                viewer.get_extraction_stats_text(),
            ]
        
        def on_filter_change(filter_value, show_bbox):
            filter_msg = viewer.filter_samples(filter_value)
            return [filter_msg, *viewer.navigate_sample(0, show_bbox)]
        
        def on_prev_click(show_bbox):
            return viewer.navigate_sample(-1, show_bbox)
        
        def on_next_click(show_bbox):
            return viewer.navigate_sample(1, show_bbox)
        
        def on_prev_img_click(show_bbox):
            return viewer.switch_image(-1, show_bbox)
        
        def on_next_img_click(show_bbox):
            return viewer.switch_image(1, show_bbox)
        
        def on_jump_click(target_index, show_bbox):
            return viewer.jump_to_sample(target_index, show_bbox)
        
        def on_bbox_toggle(show_bbox):
            """当复选框状态改变时，重新加载当前样本"""
            return viewer.navigate_sample(0, show_bbox)
        
        def on_gallery_select(show_bbox, evt):
            """点击图库缩略图时切换当前图片索引"""
            if not viewer.current_samples:
                return viewer.navigate_sample(0, show_bbox)
            
            sample = viewer.current_samples[viewer.current_index]
            image_paths = sample.get('image_paths', [])
            
            idx = None
            if evt is not None:
                # Gradio Gallery select event提供 index（可能为int或(row, col)）
                event_index = evt.index
                if isinstance(event_index, int):
                    idx = event_index
                elif isinstance(event_index, (list, tuple)) and event_index:
                    # Gallery按行列返回索引，例如(row, col)
                    idx = event_index[0] * gallery_columns + event_index[1]
            
            if idx is None:
                selected_value = evt.value if evt else None
                target_path = None
                if isinstance(selected_value, str):
                    target_path = selected_value
                elif isinstance(selected_value, (list, tuple)) and selected_value:
                    target_path = selected_value[0]
                elif isinstance(selected_value, dict):
                    target_path = selected_value.get('name') or selected_value.get('image') or selected_value.get('value')
                if target_path and target_path in image_paths:
                    idx = image_paths.index(target_path)
            
            if idx is not None and 0 <= idx < len(image_paths):
                sample['current_image_index'] = idx
                sample['original_image_path'] = image_paths[idx]
                sample['copied_image_path'] = image_paths[idx]
            
            return viewer.navigate_sample(0, show_bbox)

        def on_extract_current(current_filter_value):
            """抽取当前样本（以一组多图为单位）"""
            result = viewer.extract_current_sample()
            filter_options = viewer.get_filter_options()
            new_value = current_filter_value if current_filter_value in filter_options else "全部"
            dropdown_update = gr.update(choices=filter_options, value=new_value)
            return result, viewer.get_extraction_stats_text(), dropdown_update

        def on_remove_current_from_extraction(current_filter_value, show_bbox):
            """从抽取中删除当前样本（以一组多图为单位）"""
            result = viewer.remove_current_sample_from_extraction()
            filter_options = viewer.get_filter_options()
            new_value = current_filter_value if current_filter_value in filter_options else "全部"
            dropdown_update = gr.update(choices=filter_options, value=new_value)

            if current_filter_value == viewer.extracted_filter_name:
                filter_msg = viewer.filter_samples(new_value)
                sample_data = viewer.navigate_sample(0, show_bbox)
                return result, viewer.get_extraction_stats_text(), dropdown_update, filter_msg, *sample_data

            return (
                result,
                viewer.get_extraction_stats_text(),
                dropdown_update,
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
            )
        
        def on_clear_extraction(current_filter_value, show_bbox):
            """清空抽取结果"""
            result = viewer.clear_extraction()
            filter_options = viewer.get_filter_options()
            new_value = current_filter_value if current_filter_value in filter_options else "全部"
            dropdown_update = gr.update(choices=filter_options, value=new_value)

            if current_filter_value == viewer.extracted_filter_name:
                filter_msg = viewer.filter_samples(new_value)
                sample_data = viewer.navigate_sample(0, show_bbox)
                return result, viewer.get_extraction_stats_text(), dropdown_update, filter_msg, *sample_data

            return (
                result,
                viewer.get_extraction_stats_text(),
                dropdown_update,
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
            )

        def on_export_extraction(output_dir):
            """导出抽取结果到指定文件夹（生成 detailed_results.json 与 multi.json）"""
            result = viewer.export_extraction_to_dir(output_dir)
            return result, viewer.get_extraction_stats_text()
        
        # 绑定事件（只有在没有预加载文件时才绑定文件上传事件）
        if not json_file_path:
            file_input.change(
                fn=on_file_change,
                inputs=[file_input],
                outputs=[
                    load_status, filter_dropdown,
                    sample_info, label_comparison, raw_response, 
                    formatted_response, single_image_result_box, single_image_raw_response_box, image_gallery,
                    image_path_display, multi_image_nav_display, extraction_status, extraction_stats
                ]
            )
        
        filter_dropdown.change(
            fn=on_filter_change,
            inputs=[filter_dropdown, show_bbox_checkbox],
            outputs=[
                filter_status, sample_info, label_comparison, 
                raw_response, formatted_response, single_image_result_box, single_image_raw_response_box,
                image_gallery, image_path_display, multi_image_nav_display
            ]
        )
        
        prev_btn.click(
            fn=on_prev_click,
            inputs=[show_bbox_checkbox],
            outputs=[
                sample_info, label_comparison, raw_response, 
                formatted_response, single_image_result_box, single_image_raw_response_box,
                image_gallery, image_path_display, multi_image_nav_display
            ]
        )
        
        next_btn.click(
            fn=on_next_click,
            inputs=[show_bbox_checkbox],
            outputs=[
                sample_info, label_comparison, raw_response, 
                formatted_response, single_image_result_box, single_image_raw_response_box,
                image_gallery, image_path_display, multi_image_nav_display
            ]
        )
        
        # 图片切换按钮事件
        prev_img_btn.click(
            fn=on_prev_img_click,
            inputs=[show_bbox_checkbox],
            outputs=[
                sample_info, label_comparison, raw_response, 
                formatted_response, single_image_result_box, single_image_raw_response_box,
                image_gallery, image_path_display, multi_image_nav_display
            ]
        )
        
        next_img_btn.click(
            fn=on_next_img_click,
            inputs=[show_bbox_checkbox],
            outputs=[
                sample_info, label_comparison, raw_response, 
                formatted_response, single_image_result_box, single_image_raw_response_box,
                image_gallery, image_path_display, multi_image_nav_display
            ]
        )
        
        jump_btn.click(
            fn=on_jump_click,
            inputs=[jump_input, show_bbox_checkbox],
            outputs=[
                sample_info, label_comparison, raw_response, 
                formatted_response, single_image_result_box, single_image_raw_response_box,
                image_gallery, image_path_display, multi_image_nav_display
            ]
        )
        
        # 边界框复选框事件
        show_bbox_checkbox.change(
            fn=on_bbox_toggle,
            inputs=[show_bbox_checkbox],
            outputs=[
                sample_info, label_comparison, raw_response, 
                formatted_response, single_image_result_box, single_image_raw_response_box,
                image_gallery, image_path_display, multi_image_nav_display
            ]
        )

        image_gallery.select(
            fn=on_gallery_select,
            inputs=[show_bbox_checkbox],
            outputs=[
                sample_info, label_comparison, raw_response,
                formatted_response, single_image_result_box, single_image_raw_response_box,
                image_gallery, image_path_display, multi_image_nav_display
            ]
        )

        extract_current_btn.click(
            fn=on_extract_current,
            inputs=[filter_dropdown],
            outputs=[extraction_status, extraction_stats, filter_dropdown]
        )

        remove_current_btn.click(
            fn=on_remove_current_from_extraction,
            inputs=[filter_dropdown, show_bbox_checkbox],
            outputs=[
                extraction_status,
                extraction_stats,
                filter_dropdown,
                filter_status,
                sample_info,
                label_comparison,
                raw_response,
                formatted_response,
                single_image_result_box,
                single_image_raw_response_box,
                image_gallery,
                image_path_display,
                multi_image_nav_display,
            ]
        )

        clear_extraction_btn.click(
            fn=on_clear_extraction,
            inputs=[filter_dropdown, show_bbox_checkbox],
            outputs=[
                extraction_status,
                extraction_stats,
                filter_dropdown,
                filter_status,
                sample_info,
                label_comparison,
                raw_response,
                formatted_response,
                single_image_result_box,
                single_image_raw_response_box,
                image_gallery,
                image_path_display,
                multi_image_nav_display,
            ]
        )

        export_extraction_btn.click(
            fn=on_export_extraction,
            inputs=[export_output_dir],
            outputs=[extraction_status, extraction_stats]
        )
        
        # 正确性判断功能事件绑定
        def on_judge(is_correct, show_bbox):
            """判断当前样本并自动跳转"""
            result = viewer.judge_current_sample(is_correct)
            sample_data = viewer.navigate_sample(0, show_bbox)
            stats_text = viewer.get_judgment_stats_text()
            return result, stats_text, *sample_data
        
        judge_correct_btn.click(
            fn=lambda show_bbox: on_judge(True, show_bbox),
            inputs=[show_bbox_checkbox],
            outputs=[
                judgment_status,
                judgment_stats,
                sample_info,
                label_comparison,
                raw_response,
                formatted_response,
                single_image_result_box,
                single_image_raw_response_box,
                image_gallery,
                image_path_display,
                multi_image_nav_display
            ]
        )
        
        judge_incorrect_btn.click(
            fn=lambda show_bbox: on_judge(False, show_bbox),
            inputs=[show_bbox_checkbox],
            outputs=[
                judgment_status,
                judgment_stats,
                sample_info,
                label_comparison,
                raw_response,
                formatted_response,
                single_image_result_box,
                single_image_raw_response_box,
                image_gallery,
                image_path_display,
                multi_image_nav_display
            ]
        )
        
        export_judgment_btn.click(
            fn=viewer.export_judgment_results,
            inputs=[judgment_output_dir],
            outputs=[judgment_export_status]
        )
        
        clear_judgment_btn.click(
            fn=lambda: (viewer.clear_judgments(), viewer.get_judgment_stats_text()),
            inputs=[],
            outputs=[judgment_export_status, judgment_stats]
        )
        
        # 实时监控：添加定时器（每2秒检查一次文件变化）
        if json_file_path:
            def auto_refresh():
                """定时自动刷新 - 热重载模式"""
                has_update, new_samples = viewer.auto_reload()
                
                if has_update:
                    # 文件已更新，只更新状态文本，不刷新整个界面
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    total_samples = len(viewer.current_samples)
                    
                    if new_samples > 0:
                        status_msg = f"🔄 [{timestamp}] 新增 {new_samples} 个样本 (总计: {total_samples})"
                    else:
                        status_msg = f"🔄 [{timestamp}] 数据已更新 (总计: {total_samples})"
                    
                    # 只返回状态文本的更新，其他组件不变
                    return status_msg
                else:
                    # 无变化，不更新任何内容
                    return gr.update()
            
            # 创建定时器组件（每2秒触发一次）
            timer = gr.Timer(value=2, active=True)
            timer.tick(
                fn=auto_refresh,
                outputs=[load_status]  # 只更新状态文本
            )
        
        # 使用说明
        if not json_file_path:
            gr.Markdown("""
### 📖 使用说明：
1. **上传文件**：选择JSON文件进行上传
2. **筛选样本**：使用下拉框筛选特定类型的样本
3. **浏览样本**：使用"上一个样本/下一个样本"按钮浏览不同样本
4. **切换图片**：对于多图样本，使用"上一张图/下一张图"按钮浏览同一样本的不同图片
5. **显示边界框**：勾选/取消勾选"显示边界框"复选框来切换是否显示标注框
6. **抽取样本**：点击"🧲 抽取当前样本"将当前样本加入抽取
7. **删除样本**：点击"🗑️ 从抽取删除当前样本"将当前样本从抽取中移除
8. **查看统计**：在"抽取统计"查看当前抽取的预测结果统计
9. **导出抽取结果**：填写输出文件夹后点击"📤 导出"，自动生成 detailed_results.json 与 multi.json
10. **清空抽取**：点击"🧹 清空抽取"重置抽取列表（可选）；在各标签页查看模型输出
11. **查看已抽取**：当有抽取样本时，筛选选项会出现"已抽取"，可查看所有抽取样本

### 📋 支持的文件格式：
- **error_analysis.json**: 传统错误分析数据格式
- **vLLM推理结果**: vLLM推理生成的JSON数组格式（包含prediction、raw_response等字段）
- **new_format_data.json**: 对话数据格式（支持多图）
- **多图测评结果**: 支持包含多张图片的测评数据展示
            """)
        else:
            gr.Markdown("""
### 💡 操作提示：
- 🔄 **热重载已启用**：每2秒自动检查JSON文件变化，新增样本时静默加载，不影响当前浏览
- 使用 **⬅️ 上一个样本** / **下一个样本 ➡️** 按钮浏览不同样本
- 使用 **🖼️ ⬅️ 上一张图** / **下一张图 ➡️ 🖼️** 按钮浏览多图样本中的不同图片
- 在 **跳转到第几个** 输入框输入数字快速跳转到指定样本
- 使用 **筛选选项** 下拉框过滤特定类型的数据
- 当有抽取样本时，筛选选项会出现 **已抽取**，可查看抽取样本列表
- 勾选/取消勾选 **🔲 显示边界框** 来切换是否显示标注的边界框
- 点击 **🧲 抽取当前样本** 将当前样本加入抽取
- 点击 **🗑️ 从抽取删除当前样本** 将当前样本从抽取中移除
- 在 **抽取统计** 查看预测(FAIL)/预测(PASS)
- 点击 **📤 导出** 在输出文件夹生成 `detailed_results.json` 与 `multi.json`
- 点击 **🧹 清空抽取** 重置抽取列表
- 点击 **原始响应** / **格式化JSON** 标签查看不同格式的模型输出
- **多图导航** 区域显示当前图片在多图样本中的位置和导航信息
- 对于多图数据，可以查看每张图片的单独分析结果和检测区域
            """)
    
    return demo

def main():
    parser = argparse.ArgumentParser(description='简单错误样本查看器')
    parser.add_argument('--port', type=int, default=7870, help='服务器端口 (默认: 7860)')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='服务器地址 (默认: 127.0.0.1)')
    parser.add_argument('--share', action='store_true', help='生成公共分享链接')
    parser.add_argument('--json-file', type=str, default='/data_all/share/datasets/Huawei/HuaweiDefeactDetection/data/for_training/categorized_data/weihuqiang_balanced/sample.json', help='预加载的JSON文件路径')
    
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
    
    print("🚀 启动错误样本分析器...")
    print(f"📍 地址: http://{args.host}:{args.port}")
    if args.share:
        print("🌐 公共分享链接将在启动后显示")
    
    # 创建并启动界面
    demo = create_interface(json_file_path=args.json_file)
    
    try:
        demo.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            show_error=True,
            quiet=False,
            allowed_paths=[
                "/data_all/share",  # 允许访问数据集目录
                "/home/intern10/LLaMA-Factory",  # 允许访问工作目录
                os.path.expanduser("~")  # 允许访问用户主目录
            ]
        )
    except Exception as e:
        print(f"❌ 启动失败: {e}")

if __name__ == "__main__":
    main()
