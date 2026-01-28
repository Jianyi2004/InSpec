#!/usr/bin/env python3
"""
vLLM 官方 Server 客户端 - 多图异步批量推理

基于 vllm_client_multi_mode.py 的多图逻辑，仿照 vllm_client_batch_async.py
实现异步并发请求，可选逐图(single)或多图(multi)推理，并支持总结二轮推理。
"""

import os
import json
import argparse
import asyncio
import logging
import requests
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from dataclasses import dataclass
from tqdm import tqdm

from vllm_client_multi_mode import (
    VLLMClient,
    extract_json_payload,
    normalize_result,
)

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class InferenceSettings:
    prompt_file: str
    temperature: float
    max_tokens: int
    model: str
    system_prompt: Optional[str] = None
    summary_prompt: Optional[str] = None
    summary_system_prompt: Optional[str] = None
    summary_server_url: Optional[str] = None
    summary_model: Optional[str] = None


def _create_client(server_url: str, yolo_server_url: Optional[str], rtdetr_server_url: Optional[str] = None) -> VLLMClient:
    return VLLMClient(server_url=server_url, yolo_server_url=yolo_server_url, rtdetr_server_url=rtdetr_server_url)


def _load_prompt_text(prompt: Optional[str], prompt_type: str) -> Optional[str]:
    """如果入参是文件路径则读取内容，否则原样返回"""
    if not prompt:
        return None
    if os.path.isfile(prompt):
        try:
            logger.info("📄 载入 %s: %s", prompt_type, prompt)
            return Path(prompt).read_text(encoding="utf-8")
        except Exception as exc:
            logger.warning("⚠️  无法读取 %s %s: %s", prompt_type, prompt, exc)
            return prompt
    return prompt


def _build_inference_kwargs(settings: InferenceSettings) -> Dict[str, Any]:
    return {
        "prompt_file": settings.prompt_file,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
        "model": settings.model,
        "system_prompt": settings.system_prompt,
        "summary_system_prompt": settings.summary_system_prompt,
        "summary_prompt": settings.summary_prompt,
        "summary_server_url": settings.summary_server_url,
        "summary_model": settings.summary_model,
    }


def run_multi_mode_sync(
    server_url: str,
    yolo_server_url: Optional[str],
    rtdetr_server_url: Optional[str],
    image_paths: List[str],
    settings: InferenceSettings
) -> Dict[str, Any]:
    client = _create_client(server_url, yolo_server_url, rtdetr_server_url)
    kwargs = _build_inference_kwargs(settings)
    result = client.inference_single(
        image_paths=image_paths,
        **kwargs
    )
    if result.get("success"):
        if result.get("summary_input") is None and result.get("combined_payload"):
            result["summary_input"] = json.dumps(result["combined_payload"], ensure_ascii=False)
        combined = result.get("combined_payload")
        if combined and not result.get("analysis"):
            result["analysis"] = combined.get("analysis", [])
        if combined and not result.get("summary_response"):
            result["summary_response"] = result.get("raw_response")
    return result


def run_single_mode_sync(
    server_url: str,
    yolo_server_url: Optional[str],
    rtdetr_server_url: Optional[str],
    image_paths: List[str],
    settings: InferenceSettings
) -> Dict[str, Any]:
    client = _create_client(server_url, yolo_server_url, rtdetr_server_url)
    kwargs = _build_inference_kwargs(settings)

    aggregate_analysis: List[str] = []
    per_image_results: List[Tuple[str, str]] = []
    per_image_details: List[Dict[str, Any]] = []

    for idx, image_path in enumerate(image_paths, start=1):
        result = client.inference_single(
            image_paths=[image_path],
            **kwargs
        )
        if not result.get("success"):
            return result

        summary_text = (
            result.get("summary_response")
            or result.get("raw_response")
            or result.get("summary_input")
            or ""
        )
        json_payload = extract_json_payload(summary_text) if summary_text else None

        if json_payload and isinstance(json_payload.get("analysis"), list):
            analysis_list = [
                item.strip()
                for item in json_payload["analysis"]
                if isinstance(item, str) and item.strip()
            ]
            image_result = normalize_result(json_payload.get("result"))
        else:
            analysis_list = [summary_text.strip()] if summary_text else []
            image_result = normalize_result(result.get("prediction"))

        image_label = f"image{idx}"
        formatted_analysis = f"{image_label}: " + ", ".join(
            f"\"{text}\"" for text in analysis_list if text
        )
        aggregate_analysis.append(formatted_analysis)
        per_image_results.append((image_label, image_result))
        per_image_details.append({
            "image_label": image_label,
            "image_path": image_path,
            "prediction": image_result,
            "analysis": analysis_list,
            "raw_response": result.get("raw_response"),
            "summary_response": result.get("summary_response"),
            "summary_input": result.get("summary_input")
        })

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
    summary_text = json.dumps(aggregated_payload, ensure_ascii=False)

    return {
        'success': True,
        'prediction': final_result,
        'analysis': aggregate_analysis,
        'per_image_details': per_image_details,
        'summary_response': summary_text,
        'raw_response': summary_text
    }


def extract_true_label(messages: List[Dict[str, Any]]) -> str:
    """
    从消息中解析真实标签
    """
    for msg in messages:
        if msg.get('role') != 'assistant':
            continue
        content = msg.get('content', '')
        try:
            payload = json.loads(content)
            if isinstance(payload, dict) and 'result' in payload:
                return str(payload['result']).upper()
        except Exception:
            pass

        content_upper = content.upper()
        if 'NOT_INVOLVED' in content_upper or 'NOT INVOLVED' in content_upper:
            return 'NOT_INVOLVED'
        if 'FAIL' in content_upper:
            return 'FAIL'
        if 'PASS' in content_upper:
            return 'PASS'
    return 'UNKNOWN'



def build_result_entry(
    task_info: Dict[str, Any],
    inference_result: Dict[str, Any],
    inference_mode: str
) -> Dict[str, Any]:
    """
    根据推理结果构建统一的输出条目
    """
    entry = {
        'sample_index': task_info['index'],
        'image_name': task_info['image_name'],
        'image_paths': task_info['image_paths'],
        'true_label': task_info['true_label'],
        'inference_mode': inference_mode
    }

    if inference_result.get('success'):
        prediction = inference_result.get('prediction', 'UNKNOWN') or 'UNKNOWN'
        entry.update({
            'prediction': prediction,
            'correct': prediction == task_info['true_label'],
            'raw_response': inference_result.get('raw_response'),
            'summary_response': inference_result.get('summary_response'),
            'summary_input': inference_result.get('summary_input'),
            'analysis': inference_result.get('analysis') or [],
            'per_image_details': inference_result.get('per_image_details') or [],
            'inference_time': inference_result.get('inference_time')
        })
    else:
        entry.update({
            'prediction': 'ERROR',
            'correct': False,
            'error': inference_result.get('error'),
            'inference_time': inference_result.get('inference_time')
        })
    return entry


def _save_realtime_results(output_file: str, results_list: List[Optional[Dict[str, Any]]]):
    """
    实时写入已经完成的结果
    """
    try:
        finished = [res for res in results_list if res]
        if not finished:
            return
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(finished, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.warning("⚠️  实时保存失败: %s", exc)


async def process_sample_async(
    server_url: str,
    yolo_server_url: Optional[str],
    rtdetr_server_url: Optional[str],
    task_info: Dict[str, Any],
    settings: InferenceSettings,
    semaphore: asyncio.Semaphore,
    inference_mode: str,
    loop: asyncio.AbstractEventLoop
) -> Dict[str, Any]:
    """
    针对单个样本执行推理
    """
    start_time = datetime.now()
    async with semaphore:
        if inference_mode == "single":
            result = await loop.run_in_executor(
                None,
                run_single_mode_sync,
                server_url,
                yolo_server_url,
                rtdetr_server_url,
                task_info['image_paths'],
                settings
            )
        else:
            result = await loop.run_in_executor(
                None,
                run_multi_mode_sync,
                server_url,
                yolo_server_url,
                rtdetr_server_url,
                task_info['image_paths'],
                settings
            )

    result['inference_time'] = (datetime.now() - start_time).total_seconds()
    return result


async def batch_inference_async(
    server_url: str,
    yolo_server_url: Optional[str],
    rtdetr_server_url: Optional[str],
    test_data: List[Dict[str, Any]],
    settings: InferenceSettings,
    max_concurrent: int,
    inference_mode: str,
    realtime_output: Optional[str]
) -> List[Dict[str, Any]]:
    """
    异步批量推理入口
    """
    logger.info("=" * 80)
    logger.info("🚀 vLLM 多图异步批量推理（模式：%s，并发：%d）", inference_mode, max_concurrent)
    logger.info("=" * 80)

    semaphore = asyncio.Semaphore(max_concurrent)
    tasks: List[Dict[str, Any]] = []
    loop = asyncio.get_running_loop()

    for idx, item in enumerate(test_data):
        images = item.get('images') or []
        if not images:
            logger.warning("⚠️  样本 %d 缺少图片，跳过", idx)
            continue

        image_paths = images
        if len(image_paths) == 1:
            image_name = os.path.basename(image_paths[0])
        else:
            folder_name = os.path.basename(os.path.dirname(image_paths[0]))
            image_name = f"{folder_name} ({len(image_paths)} images)"

        true_label = extract_true_label(item.get('messages', []))

        task_info = {
            'index': idx,
            'image_paths': image_paths,
            'image_name': image_name,
            'true_label': true_label
        }

        coro = process_sample_async(
            server_url=server_url,
            yolo_server_url=yolo_server_url,
            rtdetr_server_url=rtdetr_server_url,
            task_info=task_info,
            settings=settings,
            semaphore=semaphore,
            inference_mode=inference_mode,
            loop=loop
        )

        tasks.append({
            'task': coro,
            'info': task_info
        })

    coroutines = [item['task'] for item in tasks]
    results_snapshot: List[Optional[Dict[str, Any]]] = [None] * len(coroutines)

    with tqdm(total=len(coroutines), desc="推理进度", ncols=100) as pbar:
        task_to_index = {}
        pending = set()
        for i, coro in enumerate(coroutines):
            async_task = asyncio.create_task(coro)
            task_to_index[async_task] = i
            pending.add(async_task)

        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                idx = task_to_index[task]
                task_info = tasks[idx]['info']
                try:
                    inference_result = task.result()
                except Exception as exc:
                    inference_result = {
                        'success': False,
                        'error': str(exc)
                    }
                entry = build_result_entry(task_info, inference_result, inference_mode)
                results_snapshot[idx] = entry
                pbar.update(1)

                if realtime_output:
                    _save_realtime_results(realtime_output, results_snapshot)

    finished_entries = [entry for entry in results_snapshot if entry]
    return finished_entries


def calculate_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    计算统计信息
    """
    total = len(results)
    correct = sum(1 for r in results if r.get('correct'))
    accuracy = (correct / total * 100) if total else 0.0

    label_stats: Dict[str, Dict[str, Any]] = {}
    for result in results:
        label = result.get('true_label', 'UNKNOWN')
        label_stats.setdefault(label, {'total': 0, 'correct': 0})
        label_stats[label]['total'] += 1
        if result.get('correct'):
            label_stats[label]['correct'] += 1

    for stats in label_stats.values():
        stats['accuracy'] = (stats['correct'] / stats['total'] * 100) if stats['total'] else 0.0

    return {
        'total': total,
        'correct': correct,
        'accuracy': accuracy,
        'label_stats': label_stats,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }


async def main_async():
    parser = argparse.ArgumentParser(
        description="vLLM 官方 Server 客户端 - 多图异步批量推理"
    )
    parser.add_argument("--server_url", type=str, default="http://localhost:8006")
    parser.add_argument("--test_data_path", type=str, required=True, help="测试数据 JSON")
    parser.add_argument("--prompt_file", type=str, required=True, help="Prompt 文件")
    parser.add_argument("--output_dir", type=str, default="vllm_multi_mode_async_results")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--model", type=str, default="/data_all/share/models/Qwen3-VL-32B-Instruct")
    parser.add_argument("--max_concurrent", type=int, default=5)
    parser.add_argument("--system_prompt", type=str, help="系统 Prompt")
    parser.add_argument("--inference_mode", type=str, choices=["single", "multi"], default="multi")
    parser.add_argument("--summary_prompt", type=str, help="总结 Prompt（文本或文件路径）")
    parser.add_argument("--summary_system_prompt", type=str, help="总结 System Prompt")
    parser.add_argument("--summary_server_url", type=str, help="总结 Server URL")
    parser.add_argument("--summary_model", type=str, help="总结模型")
    parser.add_argument(
        "--yolo_server_url",
        type=str,
        help="YOLO 多权重服务地址 (如 http://127.0.0.1:8810)"
    )
    parser.add_argument(
        "--rtdetr_server_url",
        type=str,
        help="RTDETR 服务地址 (如 http://127.0.0.1:8811)"
    )

    args = parser.parse_args()

    if not Path(args.test_data_path).exists():
        logger.error("❌ 测试数据不存在: %s", args.test_data_path)
        return
    prompt_paths = [p.strip() for p in args.prompt_file.split(",") if p.strip()]
    if not prompt_paths:
        logger.error("❌ 请提供至少一个 Prompt 文件")
        return
    missing_prompts = [p for p in prompt_paths if not Path(p).exists()]
    if missing_prompts:
        logger.error("❌ 下列 Prompt 文件不存在: %s", ", ".join(missing_prompts))
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    yolo_server_url = args.yolo_server_url.strip() if args.yolo_server_url else None
    rtdetr_server_url = args.rtdetr_server_url.strip() if args.rtdetr_server_url else None

    system_prompt_text = _load_prompt_text(args.system_prompt, "System Prompt") if args.system_prompt else None
    summary_system_prompt_text = _load_prompt_text(args.summary_system_prompt, "总结 System Prompt") if args.summary_system_prompt else None

    client = VLLMClient(
        server_url=args.server_url,
        yolo_server_url=yolo_server_url,
        rtdetr_server_url=rtdetr_server_url
    )
    try:
        response = requests.get(f"{args.server_url.rstrip('/')}/health", timeout=5)
        if response.status_code != 200:
            print(f"❌ Server 不可用: {args.server_url}")
            return
    except Exception as e:
        print(f"❌ 无法连接到 Server: {e}")
        return

    # 验证 Prompt 文件的 ICL 解析
    print("\n" + "=" * 80)
    print("🔍 验证 Prompt 文件 ICL 解析")
    print("=" * 80)
    for prompt_path in prompt_paths:
        print(f"\n📄 解析文件: {prompt_path}")
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt_text = f.read()
            
            base_prompt, task_instruction, icl_examples = client.parse_prompt_with_icl(prompt_text)
            
            print(f"✅ 解析成功:")
            print(f"   - 基础规则长度: {len(base_prompt)} 字符")
            print(f"   - 任务说明长度: {len(task_instruction)} 字符")
            print(f"   - ICL 示例数量: {len(icl_examples)} 个")

            if icl_examples:
                print(f"\n   📋 ICL 示例列表:")
                for idx, (img_path, desc) in enumerate(icl_examples, 1):
                    img_name = os.path.basename(img_path)
                    desc_preview = desc[:80] + "..." if len(desc) > 80 else desc
                    print(f"      {idx}. {img_name}")
                    print(f"         说明: {desc_preview}")
            else:
                print(f"   ⚠️  警告: 未解析到任何 ICL 示例")
                print(f"   💡 请检查 Prompt 文件格式:")
                print(f"      - 是否包含 '=== ICL示例开始 ===' 和 '=== ICL示例结束 ==='")
                print(f"      - 示例格式是否为: [示例N]\\n图片: <路径>\\n说明: <内容>")
                
        except Exception as e:
            print(f"❌ 解析失败: {e}")
            return
    
    print("\n" + "=" * 80 + "\n")
    
    logger.info("📂 加载测试数据: %s", args.test_data_path)
    with open(args.test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    logger.info("✓ 载入 %d 个样本", len(test_data))

    realtime_output = output_dir / "detailed_results.json"

    settings = InferenceSettings(
        prompt_file=args.prompt_file,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        model=args.model,
        system_prompt=system_prompt_text,
        summary_prompt=args.summary_prompt,
        summary_system_prompt=summary_system_prompt_text,
        summary_server_url=args.summary_server_url,
        summary_model=args.summary_model
    )

    start_time = datetime.now()
    results = await batch_inference_async(
        server_url=client.server_url,
        yolo_server_url=client.yolo_server_url,
        rtdetr_server_url=client.rtdetr_server_url,
        test_data=test_data,
        settings=settings,
        max_concurrent=args.max_concurrent,
        inference_mode=args.inference_mode,
        realtime_output=str(realtime_output)
    )
    end_time = datetime.now()

    stats = calculate_statistics(results)

    detailed_path = realtime_output
    stats_path = output_dir / "statistics.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("📊 推理完成")
    print("=" * 80)
    print(f"⏱️  总耗时: {(end_time - start_time).total_seconds():.2f} 秒")
    print(f"📈 样本总数: {stats['total']}")
    print(f"✅ 正确样本: {stats['correct']}")
    print(f"🎯 总体准确率: {stats['accuracy']:.2f}%")
    print(f"📂 详细结果: {detailed_path}")
    print(f"📂 统计结果: {stats_path}")
    print("=" * 80)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
