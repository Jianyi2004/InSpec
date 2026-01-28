#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vLLM 官方 Server 客户端 - 多图输入 + 可选单图/多图推理逻辑

基于 vllm_client_single_sumup.py，输入统一为多图（列表或文件夹）。
新增 --inference_mode，single 模式下逐图推理并拼接结果，multi 模式下沿用原始多图逻辑。
"""

import argparse
import base64
import json
import logging
import os
import re
from glob import glob
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import requests

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("run.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

VALID_RESULTS = {"PASS", "FAIL", "NOT_INVOLVED"}


class VLLMClient:
    """vLLM 官方 Server 客户端"""

    def __init__(
        self,
        server_url: str = "http://localhost:8006",
        yolo_server_url: Optional[str] = None,
        rtdetr_server_url: Optional[str] = None
    ):
        self.server_url = server_url.rstrip('/')
        self.api_url = f"{self.server_url}/v1/chat/completions"
        env_yolo_url = yolo_server_url or os.environ.get("YOLO_SERVER_URL")
        self.yolo_server_url = env_yolo_url.rstrip('/') if env_yolo_url else None
        env_rtdetr_url = rtdetr_server_url or os.environ.get("RTDETR_SERVER_URL")
        self.rtdetr_server_url = env_rtdetr_url.rstrip('/') if env_rtdetr_url else None
        self.current_yolo_config: Optional[Dict[str, Any]] = None
        self.current_rtdetr_config: Optional[Dict[str, Any]] = None

    def health_check(self) -> bool:
        """健康检查"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info("✅ Server 健康检查通过")
                return True
            logger.error("❌ Server 健康检查失败: HTTP %s", response.status_code)
            return False
        except Exception as exc:
            logger.error("❌ 无法连接到 Server: %s", exc)
            return False

    def encode_image_to_base64(self, image_path: str) -> str:
        """将图片编码为 base64"""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def _parse_key_value_lines(self, section: str) -> Dict[str, str]:
        result: Dict[str, str] = {}
        if not section:
            return result
        for raw_line in section.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            for sep in (":", "：", "="):
                if sep in line:
                    key, value = line.split(sep, 1)
                    result[key.strip()] = value.strip()
                    break
        return result

    @staticmethod
    def _strip_quotes(text: str) -> str:
        if len(text) >= 2 and ((text[0] == '"' and text[-1] == '"') or (text[0] == "'" and text[-1] == "'")):
            return text[1:-1]
        return text

    def _load_prompt_text(self, prompt: Optional[str], prompt_type: str = "System") -> Optional[str]:
        """如果参数是文件路径则读取内容，否则原样返回"""
        if not prompt:
            return None
        if os.path.isfile(prompt):
            try:
                logger.info("📄 载入 %s Prompt 文件: %s", prompt_type, prompt)
                return Path(prompt).read_text(encoding="utf-8")
            except Exception as exc:
                logger.warning("⚠️  无法读取 %s Prompt 文件 %s: %s", prompt_type, prompt, exc)
                return prompt
        return prompt

    def _parse_mapping_block(self, section: str) -> Dict[str, Any]:
        """
        解析简单的 key: value 配置块，支持带引号、逗号结尾，自动转数值/布尔/JSON数组

        支持的格式:
        - 字符串: "key": "value" 或 "key": value
        - 数值: "key": 123 或 "key": 3.14
        - 布尔: "key": true
        - JSON数组: "key": ["item1", "item2"]
        - JSON对象: "key": {"nested": "value"}
        """
        mapping: Dict[str, Any] = {}
        if not section:
            return mapping
        for raw_line in section.splitlines():
            line = raw_line.strip().rstrip(",")
            if not line or line.startswith("#"):
                continue
            for sep in (":", "：", "="):
                if sep in line:
                    key, value = line.split(sep, 1)
                    key = self._strip_quotes(key.strip())
                    value_raw = value.strip()

                    # 尝试 JSON 解析（支持数组和对象）
                    if value_raw.startswith("[") or value_raw.startswith("{"):
                        try:
                            mapping[key] = json.loads(value_raw)
                            break
                        except json.JSONDecodeError:
                            # JSON 解析失败，继续按字符串处理
                            pass

                    # 去除引号
                    value = self._strip_quotes(value_raw)
                    lower = value.lower()

                    # 布尔值
                    if lower in {"true", "false"}:
                        mapping[key] = lower == "true"
                    else:
                        # 数值解析
                        try:
                            if "." in value:
                                mapping[key] = float(value)
                            else:
                                mapping[key] = int(value)
                            continue
                        except ValueError:
                            # 字符串
                            mapping[key] = value
                    break
        return mapping

    def parse_prompt_with_icl(self, prompt_text: str) -> Tuple[str, str, List[Tuple[str, str]]]:
        """
        解析 prompt 文件，提取基础规则、任务说明和 ICL 示例

        Returns:
            (base_prompt, task_instruction, icl_examples)
            icl_examples: [(image_path, description), ...]
        """
        self.current_yolo_config = None
        self.current_rtdetr_config = None
        icl_examples: List[Tuple[str, str]] = []
        yolo_start = '=== YOLO配置开始 ==='
        yolo_end = '=== YOLO配置结束 ==='
        rtdetr_start = '=== RTDETR配置开始 ==='
        rtdetr_end = '=== RTDETR配置结束 ==='

        # YOLO 配置块处理：截掉配置，后续文本继续正常解析
        if yolo_start in prompt_text and yolo_end in prompt_text:
            before, after_start = prompt_text.split(yolo_start, 1)
            config_section, after_end = after_start.split(yolo_end, 1)
            config_section = config_section.strip()
            yolo_config = self._parse_mapping_block(config_section)
            self.current_yolo_config = {
                "config": yolo_config,
                "raw_section": config_section
            }
            prompt_text = (before + after_end).strip()

        # RTDETR 配置块处理：截掉配置，后续文本继续正常解析
        if rtdetr_start in prompt_text and rtdetr_end in prompt_text:
            before, after_start = prompt_text.split(rtdetr_start, 1)
            config_section, after_end = after_start.split(rtdetr_end, 1)
            config_section = config_section.strip()
            rtdetr_config = self._parse_mapping_block(config_section)
            self.current_rtdetr_config = {
                "config": rtdetr_config,
                "raw_section": config_section
            }
            prompt_text = (before + after_end).strip()

        if '=== ICL示例开始 ===' not in prompt_text:
            return prompt_text.strip(), "", icl_examples

        parts = prompt_text.split('=== ICL示例开始 ===')
        base_prompt = parts[0].strip()
        if len(parts) < 2:
            return base_prompt, "", icl_examples

        remaining = parts[1]
        if '=== ICL示例结束 ===' in remaining:
            icl_section, task_instruction = remaining.split('=== ICL示例结束 ===')
            icl_section = icl_section.strip()
            task_instruction = task_instruction.strip()
        else:
            icl_section = remaining.strip()
            task_instruction = ""

        example_pattern = r'\[示例\d+\]\s*图片:\s*([^\n]+)\s*说明:\s*(.+?)(?=\[示例\d+\]|$)'
        matches = re.findall(example_pattern, icl_section, re.DOTALL)

        for image_path, description in matches:
            image_path = image_path.strip()
            description = description.strip()

            if os.path.exists(image_path):
                icl_examples.append((image_path, description))
                logger.info("✓ 加载 ICL 示例: %s", os.path.basename(image_path))
            else:
                logger.warning("⚠️  ICL 示例图片不存在: %s", image_path)

        logger.info("📋 解析完成: %d 个 ICL 示例", len(icl_examples))
        return base_prompt, task_instruction, icl_examples

    def extract_prompt_name(self, prompt_text: str, default_name: Optional[str] = None) -> Tuple[str, str]:
        """
        解析并移除 prompt 名称块，格式：
        === PROMPT名字开始 ===
        <名称内容>
        === PROMPT名字结束 ===
        """
        start_marker = "=== PROMPT名字开始 ==="
        end_marker = "=== PROMPT名字结束 ==="
        prompt_name = default_name or ""

        if start_marker in prompt_text and end_marker in prompt_text:
            before, after_start = prompt_text.split(start_marker, 1)
            name_section, after_end = after_start.split(end_marker, 1)
            prompt_name = name_section.strip() or prompt_name
            prompt_text = (before + after_end).strip()

        return prompt_text, prompt_name

    def build_messages(
        self,
        base_prompt: str,
        task_instruction: str,
        icl_examples: List[Tuple[str, str]],
        test_image_paths: List[str],
        system_prompt: Optional[str] = None,
        test_images_base64: Optional[List[str]] = None
    ) -> List[dict]:
        """
        构建 OpenAI 格式的消息列表
        """
        messages: List[Dict[str, Any]] = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        content: List[Dict[str, Any]] = []

        if base_prompt:
            content.append({
                "type": "text",
                "text": base_prompt
            })

        for idx, (example_image_path, example_description) in enumerate(icl_examples):
            content.append({
                "type": "text",
                "text": f"\n[示例{idx+1}]\n说明: {example_description}\n"
            })
            try:
                example_base64 = self.encode_image_to_base64(example_image_path)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{example_base64}"
                    }
                })
            except Exception as exc:
                logger.warning("⚠️  无法加载 ICL 示例图片 %s: %s", example_image_path, exc)

        if task_instruction:
            content.append({
                "type": "text",
                "text": f"\n{task_instruction}"
            })

        for idx, test_image_path in enumerate(test_image_paths):
            if test_images_base64 and idx < len(test_images_base64):
                test_image_base64 = test_images_base64[idx]
            else:
                test_image_base64 = self.encode_image_to_base64(test_image_path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{test_image_base64}"
                }
            })

        messages.append({"role": "user", "content": content})
        return messages

    def _run_prompt_once(
        self,
        image_paths: List[str],
        prompt_path: str,
        temperature: float,
        max_tokens: int,
        model: str,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """使用单个 prompt 执行一次推理"""
        try:
            logger.info("📂 读取 prompt: %s", prompt_path)
            prompt_text_raw = Path(prompt_path).read_text(encoding='utf-8')
        except Exception as exc:
            error_msg = f"无法读取 prompt 文件 {prompt_path}: {exc}"
            logger.error("❌ %s", error_msg)
            return {'success': False, 'error': error_msg}

        prompt_text, prompt_name = self.extract_prompt_name(
            prompt_text_raw,
            default_name=Path(prompt_path).stem
        )

        base_prompt, task_instruction, icl_examples = self.parse_prompt_with_icl(prompt_text)

        # 用于存储检测后的图片 base64 (YOLO 或 RTDETR)
        detection_images_base64: Optional[List[str]] = None

        # YOLO 推理
        if self.current_yolo_config:
            yolo_result = self._run_yolo_inference(image_paths)
            if not yolo_result.get("success"):
                return yolo_result
            detection_images_base64 = yolo_result.get("images_base64")

            save_dir = Path("yolo_outputs")
            save_dir.mkdir(parents=True, exist_ok=True)
            for idx, img_b64 in enumerate(detection_images_base64 or [], start=1):
                base_name = Path(image_paths[idx - 1]).stem
                target_path = save_dir / f"{base_name}_yolo.jpg"
                counter = 1
                while target_path.exists():
                    counter += 1
                    target_path = save_dir / f"{base_name}_yolo_{counter}.jpg"
                target_path.write_bytes(base64.b64decode(img_b64))
            if detection_images_base64:
                logger.info("💾 YOLO 可视化已保存到 %s", save_dir.resolve())

            logger.info("🟢 YOLO 可视化完成，使用带框图送入大模型")

        # RTDETR 推理
        if self.current_rtdetr_config:
            rtdetr_result = self._run_rtdetr_inference(image_paths)
            if not rtdetr_result.get("success"):
                return rtdetr_result
            detection_images_base64 = rtdetr_result.get("images_base64")

            save_dir = Path("rtdetr_outputs")
            save_dir.mkdir(parents=True, exist_ok=True)
            for idx, img_b64 in enumerate(detection_images_base64 or [], start=1):
                base_name = Path(image_paths[idx - 1]).stem
                target_path = save_dir / f"{base_name}_rtdetr.jpg"
                counter = 1
                while target_path.exists():
                    counter += 1
                    target_path = save_dir / f"{base_name}_rtdetr_{counter}.jpg"
                target_path.write_bytes(base64.b64decode(img_b64))
            if detection_images_base64:
                logger.info("💾 RTDETR 可视化已保存到 %s", save_dir.resolve())

            logger.info("🔷 RTDETR 可视化完成，使用带框图送入大模型")

        logger.info("📷 处理 %d 张图片: %s", len(image_paths), [p for p in image_paths])
        system_prompt_text = self._load_prompt_text(system_prompt, "System")
        # print(system_prompt_text)
        if system_prompt_text:
            logger.info("🔧 使用 System Prompt: %s...", system_prompt_text[:100])
        messages = self.build_messages(
            base_prompt,
            task_instruction,
            icl_examples,
            image_paths,
            system_prompt_text,
            test_images_base64=detection_images_base64
        )

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        response = requests.post(self.api_url, json=payload, timeout=1200)
        if response.status_code != 200:
            error_msg = f"HTTP {response.status_code}: {response.text}"
            logger.error("❌ 请求失败: %s", error_msg)
            return {'success': False, 'error': error_msg}

        result = response.json()
        raw_response = result['choices'][0]['message']['content']

        logger.info("📥 收到响应")
        logger.info("原始响应:\n%s\n", raw_response)

        prediction = self.parse_response(raw_response)
        if prediction:
            logger.info("✅ 预测结果: %s", prediction)
        else:
            logger.warning("⚠️  无法解析预测结果")

        return {
            'success': True,
            'prompt_file': prompt_path,
            'prompt_name': prompt_name,
            'raw_response': raw_response,
            'prediction': prediction or 'UNKNOWN',
            'normalized_prediction': normalize_result(prediction)
        }

    def _run_yolo_inference(self, image_paths: List[str]) -> Dict[str, Any]:
        if not self.current_yolo_config:
            return {'success': False, 'error': "YOLO 配置缺失"}

        config = dict(self.current_yolo_config.get("config") or {})
        server_url = (
            config.pop("server_url", None)
            or config.pop("server", None)
            or self.yolo_server_url
        )
        if not server_url:
            return {'success': False, 'error': "未配置 YOLO Server 地址 (server_url 或 YOLO_SERVER_URL)"}

        endpoint = f"{server_url.rstrip('/')}/inference"
        images_base64: List[str] = []
        logger.info("🎯 使用 YOLO server: %s", endpoint)

        for image_path in image_paths:
            payload = dict(config)
            payload["image_path"] = image_path
            payload.setdefault("return_image", True)
            payload.setdefault("image_format", "JPEG")
            payload.setdefault("image_quality", 90)

            try:
                response = requests.post(endpoint, json=payload, timeout=600)
            except Exception as exc:
                return {'success': False, 'error': f"YOLO 请求失败: {exc}"}

            if response.status_code != 200:
                return {'success': False, 'error': f"YOLO HTTP {response.status_code}: {response.text}"}

            data = response.json()
            if not data.get("success"):
                return {'success': False, 'error': data.get("error", "YOLO 返回失败")}

            result_payload = data.get("result", {})
            b64 = result_payload.get("image_base64")
            if not b64:
                return {'success': False, 'error': "YOLO 返回缺少 image_base64，请确认服务端开启 return_image"}
            images_base64.append(b64)
            logger.info("📥 YOLO 返回 detections=%s", len(result_payload.get("detections", [])))

        return {'success': True, 'images_base64': images_base64}

    def _run_rtdetr_inference(self, image_paths: List[str]) -> Dict[str, Any]:
        """执行 RT-DETR 推理，返回带可视化框的图片 base64"""
        if not self.current_rtdetr_config:
            return {'success': False, 'error': "RTDETR 配置缺失"}

        config = dict(self.current_rtdetr_config.get("config") or {})
        server_url = (
            config.pop("server_url", None)
            or config.pop("server", None)
            or self.rtdetr_server_url
        )
        if not server_url:
            return {'success': False, 'error': "未配置 RTDETR Server 地址 (server_url 或 RTDETR_SERVER_URL)"}

        endpoint = f"{server_url.rstrip('/')}/inference"
        images_base64: List[str] = []
        logger.info("🔷 使用 RTDETR server: %s", endpoint)

        # 处理配置中的特殊字段
        # label_names: 可以是逗号分隔的字符串，需要转换为列表
        if "label_names" in config and isinstance(config["label_names"], str):
            config["label_names"] = [name.strip() for name in config["label_names"].split(",") if name.strip()]

        # conflict_suppress: 可以是逗号分隔的字符串，需要转换为列表
        if "conflict_suppress" in config and isinstance(config["conflict_suppress"], str):
            config["conflict_suppress"] = [name.strip() for name in config["conflict_suppress"].split(",") if name.strip()]

        # class_id_to_name: 可以是 JSON 字符串，需要解析为字典
        if "class_id_to_name" in config:
            id_to_name = config["class_id_to_name"]
            if isinstance(id_to_name, str):
                try:
                    config["class_id_to_name"] = json.loads(id_to_name)
                except json.JSONDecodeError as exc:
                    logger.warning("⚠️  无法解析 class_id_to_name JSON: %s", exc)
                    del config["class_id_to_name"]

        for image_path in image_paths:
            # 读取图片并编码为 base64
            try:
                with open(image_path, "rb") as f:
                    image_bytes = f.read()
                image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            except Exception as exc:
                return {'success': False, 'error': f"无法读取图片 {image_path}: {exc}"}

            payload = dict(config)
            payload["image_base64"] = image_base64
            payload.setdefault("return_image", True)
            payload.setdefault("image_format", "JPEG")
            payload.setdefault("image_quality", 90)

            try:
                response = requests.post(endpoint, json=payload, timeout=600)
            except Exception as exc:
                return {'success': False, 'error': f"RTDETR 请求失败: {exc}"}

            if response.status_code != 200:
                return {'success': False, 'error': f"RTDETR HTTP {response.status_code}: {response.text}"}

            data = response.json()
            if not data.get("success"):
                return {'success': False, 'error': data.get("error", "RTDETR 返回失败")}

            result_payload = data.get("result", {})
            b64 = result_payload.get("image_base64")
            if not b64:
                return {'success': False, 'error': "RTDETR 返回缺少 image_base64，请确认服务端开启 return_image"}
            images_base64.append(b64)
            logger.info("📥 RTDETR 返回 detections=%s", len(result_payload.get("detections", [])))

        return {'success': True, 'images_base64': images_base64}

    def parse_response(self, response_text: str) -> Optional[str]:
        """
        解析模型响应，提取结果（PASS/FAIL/NOT_INVOLVED）
        支持普通模型和 Thinking 模型（<think>...</think> 格式）
        """
        if '</think>' in response_text:
            clean_text = response_text.split('</think>', 1)[1].strip()
        else:
            clean_text = response_text

        for text in [clean_text, response_text]:
            try:
                result = json.loads(text)
                if isinstance(result, dict) and 'result' in result:
                    return result['result'].replace(' ', '_')
            except Exception:
                pass

            json_matches = re.findall(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
            for match in json_matches:
                try:
                    result = json.loads(match)
                    if isinstance(result, dict) and 'result' in result:
                        return result['result'].replace(' ', '_')
                except Exception:
                    pass

        for text in [clean_text, response_text]:
            if 'NOT_INVOLVED' in text or 'NOT INVOLVED' in text:
                return 'NOT_INVOLVED'
            if 'FAIL' in text:
                return 'FAIL'
            if 'PASS' in text:
                return 'PASS'

        logger.warning("⚠️  无法解析结果，原始响应: %s", response_text[:200])
        return None

    def inference_single(
        self,
        image_paths: List[str],
        prompt_file: str,
        temperature: float = 0.1,
        max_tokens: int = 512,
        model: str = "/data_all/share/models/Qwen3-VL-32B-Instruct",
        system_prompt: Optional[str] = None,
        summary_system_prompt: Optional[str] = None,
        summary_prompt: Optional[str] = None,
        summary_server_url: Optional[str] = None,
        summary_model: Optional[str] = None
    ) -> Dict[str, Any]:
        try:
            prompt_paths = [p.strip() for p in prompt_file.split(",") if p.strip()]
            if not prompt_paths:
                error_msg = "Prompt 文件列表为空"
                logger.error("❌ %s", error_msg)
                return {'success': False, 'error': error_msg}

            per_prompt_outputs: List[Dict[str, Any]] = []
            combined_analysis: List[str] = []
            prompt_results: List[str] = []
            named_prompt_payloads: List[Tuple[str, Dict[str, Any]]] = []

            for idx, prompt_path in enumerate(prompt_paths, start=1):
                logger.info("\n%s", "=" * 80)
                logger.info("🧠 Prompt %d/%d: %s", idx, len(prompt_paths), prompt_path)
                single_result = self._run_prompt_once(
                    image_paths=image_paths,
                    prompt_path=prompt_path,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    model=model,
                    system_prompt=system_prompt
                )
                if not single_result.get('success'):
                    return single_result

                prompt_name = single_result.get('prompt_name') or Path(prompt_path).stem or f"prompt{idx}"
                raw_response = single_result.get('raw_response', '')
                json_payload = extract_json_payload(raw_response) if raw_response else None
                prompt_analysis: List[str] = []
                if json_payload and isinstance(json_payload.get("analysis"), list):
                    prompt_analysis = [
                        text.strip()
                        for text in json_payload["analysis"]
                        if isinstance(text, str) and text.strip()
                    ]
                    combined_analysis.extend(prompt_analysis)
                    prompt_result = normalize_result(json_payload.get("result"))
                else:
                    fallback_text = raw_response.strip() if raw_response else ""
                    if raw_response:
                        combined_analysis.append(fallback_text)
                        prompt_analysis = [fallback_text]
                    prompt_result = single_result.get('normalized_prediction', 'NOT_INVOLVED')

                named_prompt_payloads.append((
                    prompt_name,
                    {
                        "analysis": prompt_analysis,
                        "result": prompt_result
                    }
                ))

                single_result['parsed_payload'] = json_payload
                single_result['final_prompt_result'] = prompt_result
                per_prompt_outputs.append(single_result)
                prompt_results.append(prompt_result)

            if any(res == "FAIL" for res in prompt_results):
                final_result = "FAIL"
            elif any(res == "PASS" for res in prompt_results):
                final_result = "PASS"
            else:
                final_result = "NOT_INVOLVED"

            combined_payload = {
                "analysis": combined_analysis,
                "result": final_result
            }
            aggregated_raw_response = json.dumps(combined_payload, ensure_ascii=False)
            summary_input_text = None
            if named_prompt_payloads:
                parts: List[str] = []
                for name, payload in named_prompt_payloads:
                    payload_json = json.dumps(payload, ensure_ascii=False)
                    parts.append(f"{name}:{payload_json}")
                summary_input_text = ", ".join(parts)

            result_dict: Dict[str, Any] = {
                'success': True,
                'prediction': final_result,
                'raw_response': aggregated_raw_response,
                'combined_payload': combined_payload,
                'summary_input': summary_input_text,
                'per_prompt_responses': per_prompt_outputs
            }

            summary_prompt_text: Optional[str] = None
            if summary_prompt:
                prompt_path = Path(summary_prompt)
                if prompt_path.is_file():
                    try:
                        summary_prompt_text = prompt_path.read_text(encoding='utf-8').strip()
                        logger.info("📄 使用总结 Prompt 文件: %s", summary_prompt)
                    except Exception as exc:
                        logger.error("❌ 无法读取总结 Prompt 文件 %s: %s", summary_prompt, exc)
                        summary_prompt_text = None
                else:
                    summary_prompt_text = summary_prompt.strip()
                    if not summary_prompt_text:
                        summary_prompt_text = None

            if summary_prompt_text:
                logger.info("\n%s", '=' * 80)
                logger.info("🔄 开始第二轮推理（总结）")
                logger.info("%s", '=' * 80)

                summary_result = self.inference_summary(
                    original_response=summary_input_text or aggregated_raw_response,
                    summary_prompt=summary_prompt_text,
                    summary_system_prompt=summary_system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    model=summary_model if summary_model else model,
                    server_url=summary_server_url if summary_server_url else self.server_url
                )

                if summary_result['success']:
                    result_dict['summary_response'] = summary_result['summary_response']
                    logger.info("✅ 总结完成")
                else:
                    logger.warning("⚠️  总结失败: %s", summary_result.get('error', 'Unknown error'))
                    result_dict['summary_response'] = None

            return result_dict

        except Exception as exc:
            logger.error("❌ 推理失败: %s", exc)
            return {'success': False, 'error': str(exc)}

    def inference_summary(
        self,
        original_response: str,
        summary_prompt: str,
        summary_system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
        model: str = "/data_all/share/models/Qwen3-VL-32B-Instruct",
        server_url: Optional[str] = None
    ) -> Dict[str, Any]:
        try:
            messages: List[Dict[str, Any]] = []

            summary_system_prompt_text = self._load_prompt_text(summary_system_prompt, "总结 System")
            if summary_system_prompt_text:
                messages.append({
                    "role": "system",
                    "content": summary_system_prompt_text
                })
                logger.info("🔧 使用总结 System Prompt: %s...", summary_system_prompt_text[:100])

            user_content = f"{summary_prompt}\n\n原始回答：\n{original_response}"
            messages.append({
                "role": "user",
                "content": user_content
            })

            api_url = f"{server_url.rstrip('/')}/v1/chat/completions" if server_url else self.api_url

            logger.info("📤 发送总结请求...")
            if server_url:
                logger.info("🌐 使用总结 Server: %s", server_url)
            logger.info("🤖 使用总结 Model: %s", model)

            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }

            response = requests.post(api_url, json=payload, timeout=1200)

            if response.status_code != 200:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                logger.error("❌ 总结请求失败: %s", error_msg)
                return {'success': False, 'error': error_msg}

            result = response.json()
            summary_response = result['choices'][0]['message']['content']

            logger.info("📥 收到总结响应")
            logger.info("总结后的回答:\n%s\n", summary_response)

            return {
                'success': True,
                'summary_response': summary_response
            }

        except Exception as exc:
            logger.error("❌ 总结失败: %s", exc)
            return {'success': False, 'error': str(exc)}


def normalize_result(raw_result: Optional[str]) -> str:
    """统一结果标签"""
    if not raw_result:
        return "NOT_INVOLVED"
    value = raw_result.strip().upper().replace(" ", "_")
    if value in VALID_RESULTS:
        return value
    if "FAIL" in value:
        return "FAIL"
    if "PASS" in value:
        return "PASS"
    if "NOT" in value:
        return "NOT_INVOLVED"
    return "NOT_INVOLVED"


def extract_json_payload(text: str) -> Optional[Dict[str, Any]]:
    """
    从模型输出中解析 JSON（支持 <think> 块和 Markdown 代码块）
    """
    candidate_texts = []
    clean_text = text
    if "</think>" in text:
        clean_text = text.split("</think>", 1)[1]
    candidate_texts.append(clean_text.strip())
    candidate_texts.append(text.strip())

    for candidate in candidate_texts:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        matches = re.findall(r'```(?:json)?\s*(\{.*?\})\s*```', candidate, re.DOTALL)
        for block in matches:
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                continue
    return None


def collect_image_paths(image_dir: Optional[str], image_paths: Optional[str]) -> List[str]:
    """根据参数收集图片路径"""
    collected: List[str] = []
    if image_paths:
        collected = [p.strip() for p in image_paths.split(",") if p.strip()]
    elif image_dir:
        exts = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        for ext in exts:
            collected.extend(glob(os.path.join(image_dir, ext)))
        collected.sort()
    return [p for p in collected if Path(p).exists()]


def run_single_logic(
    client: VLLMClient,
    image_paths: List[str],
    args: argparse.Namespace
) -> Dict[str, Any]:
    """逐图推理并拼接结果"""
    aggregate_analysis: List[str] = []
    per_image_results: List[Tuple[str, str]] = []

    for idx, image_path in enumerate(image_paths, start=1):
        image_label = f"image{idx}"
        logger.info("=" * 80)
        logger.info("🖼️  逐图推理: %s", image_path)
        result = client.inference_single(
            image_paths=[image_path],
            prompt_file=args.prompt_file,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            model=args.model,
            system_prompt=args.system_prompt,
            summary_system_prompt=args.summary_system_prompt,
            summary_prompt=args.summary_prompt,
            summary_server_url=args.summary_server_url,
            summary_model=args.summary_model
        )
        if not result.get("success"):
            raise RuntimeError(result.get("error", "Unknown error"))

        summary_text = result.get("summary_response")
        final_text = summary_text if summary_text else result.get("raw_response", "")
        json_payload = extract_json_payload(final_text)

        if json_payload and isinstance(json_payload.get("analysis"), list):
            analysis_list = [item.strip() for item in json_payload["analysis"] if isinstance(item, str)]
            image_result = normalize_result(json_payload.get("result"))
        else:
            logger.warning("⚠️  无法解析分析内容，使用原始文本")
            analysis_list = [final_text.strip()] if final_text else []
            image_result = normalize_result(result.get("prediction"))

        formatted_analysis = f"{image_label}: " + ", ".join(f"\"{text}\"" for text in analysis_list if text)
        aggregate_analysis.append(formatted_analysis)
        per_image_results.append((image_label, image_result))
        logger.info("✅ %s -> %s", image_label, image_result)

    result_summary = ", ".join(f"{label}:{res}" for label, res in per_image_results)
    aggregate_analysis.append(result_summary)

    normalized_results = [res for _, res in per_image_results]
    if any(res == "FAIL" for res in normalized_results):
        final_result = "FAIL"
    elif normalized_results and all(res == "NOT_INVOLVED" for res in normalized_results):
        final_result = "FAIL"
    else:
        final_result = "PASS"

    aggregated_payload = {
        "analysis": aggregate_analysis,
        "result": final_result
    }
    return aggregated_payload


def run_multi_logic(
    client: VLLMClient,
    image_paths: List[str],
    args: argparse.Namespace
) -> Dict[str, Any]:
    """按旧逻辑执行多图推理"""
    result = client.inference_single(
        image_paths=image_paths,
        prompt_file=args.prompt_file,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        model=args.model,
        system_prompt=args.system_prompt,
        summary_system_prompt=args.summary_system_prompt,
        summary_prompt=args.summary_prompt,
        summary_server_url=args.summary_server_url,
        summary_model=args.summary_model
    )
    if not result.get("success"):
        raise RuntimeError(result.get("error", "Unknown error"))
    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="vLLM 官方 Server 客户端 - 多图输入 + 单/多逻辑推理"
    )
    parser.add_argument("--server_url", type=str, default="http://localhost:8006")
    parser.add_argument("--yolo_server_url", type=str, help="YOLO 多权重服务地址 (如 http://localhost:8810)")
    parser.add_argument("--rtdetr_server_url", type=str, help="RTDETR 服务地址 (如 http://localhost:8811)")
    parser.add_argument("--image_paths", type=str, help="多张图片路径，逗号分隔")
    parser.add_argument("--image_dir", type=str, help="图片文件夹路径")
    parser.add_argument("--prompt_file", type=str, required=True, help="Prompt 文件路径")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--model", type=str, default="/data_all/share/models/Qwen3-VL-32B-Instruct")
    parser.add_argument("--system_prompt", type=str, help="第一轮推理 System prompt")
    parser.add_argument("--summary_system_prompt", type=str, help="总结 System prompt")
    parser.add_argument("--summary_prompt", type=str, help="总结 Prompt（可选）")
    parser.add_argument("--summary_server_url", type=str, help="总结 Server（可选）")
    parser.add_argument("--summary_model", type=str, help="总结模型（可选）")
    parser.add_argument(
        "--inference_mode",
        type=str,
        choices=["single", "multi"],
        default="multi",
        help="single：逐图推理并拼接；multi：一次性多图推理"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    client = VLLMClient(
        server_url=args.server_url,
        yolo_server_url=args.yolo_server_url,
        rtdetr_server_url=args.rtdetr_server_url
    )

    # 健康检查
    if not client.health_check():
        logger.error("❌ Server 不可用")
        return

    # 输入必须为多图（列表或文件夹）
    image_paths = collect_image_paths(args.image_dir, args.image_paths)
    if not image_paths:
        logger.error("❌ 请通过 --image_paths 或 --image_dir 提供至少一张图片")
        return
    logger.info("🖼️  收集到 %d 张图片", len(image_paths))

    prompt_paths = [p.strip() for p in args.prompt_file.split(",") if p.strip()]
    if not prompt_paths:
        logger.error("❌ 请提供至少一个 Prompt 文件")
        return
    missing_prompts = [p for p in prompt_paths if not Path(p).exists()]
    if missing_prompts:
        logger.error("❌ 下列 Prompt 文件不存在: %s", ", ".join(missing_prompts))
        return

    try:
        if args.inference_mode == "single":
            aggregated_payload = run_single_logic(client, image_paths, args)
            logger.info("\n📦 拼接结果：\n%s", json.dumps(aggregated_payload, ensure_ascii=False, indent=2))
        else:
            result = run_multi_logic(client, image_paths, args)
            logger.info("\n📦 多图推理结果：\n%s", json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as exc:
        logger.error("❌ 推理失败: %s", exc)


if __name__ == "__main__":
    main()
