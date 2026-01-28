#!/usr/bin/env python3
"""
YOLO 多权重常驻推理服务

特性:
1. 启动时一次性挂载多个 YOLO 权重，通过名称选择
2. 支持 image_path 或 image_base64 输入
3. 提供 /health /weights /inference /batch_inference /reload 接口

示例启动:
python YOLO-main/yolo_multi_server.py \
    --weights_dir /path/to/weights_dir \
    --weight bbu=/path/to/best.pt \
    --default_weight bbu \
    --host 0.0.0.0 --port 8810
"""

import argparse
import base64
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from flask import Flask, jsonify, request
from ultralytics import YOLO
from ultralytics.utils.plotting import colors

# 重载 YOLO 可视化色表
hexs = (
    "FF3838",
    "FF9D97",
    "FF701F",
    "FFB21D",
    "CFD231",
    "48F90A",
    "92CC17",
    "3DDB86",
    "1A9334",
    "00D4BB",
    "2C99A8",
    "00C2FF",
    "344593",
    "6473FF",
    "0018EC",
    "8438FF",
    "520085",
    "CB38FF",
    "FF95C8",
    "FF37C7",
)
colors.palette = [colors.hex2rgb(f"#{c}") for c in hexs]

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger("yolo-server")

app = Flask(__name__)
service_instance = None


class YOLORegistry:
    """管理多个 YOLO 权重"""

    def __init__(self, weights: Dict[str, str], device: str = "cuda:0"):
        self.device = device
        self.weight_paths = weights
        self.models: Dict[str, YOLO] = {}
        self.load_all()

    def load_single(self, name: str, path: str) -> YOLO:
        logger.info("加载 YOLO 权重: %s -> %s", name, path)
        model = YOLO(path)
        model.to(self.device)
        return model

    def load_all(self):
        self.models.clear()
        for name, path in self.weight_paths.items():
            if not os.path.exists(path):
                logger.warning("权重 %s 不存在: %s", name, path)
                continue
            try:
                self.models[name] = self.load_single(name, path)
            except Exception as exc:
                logger.exception("加载权重 %s 失败: %s", name, exc)
        logger.info("已加载 %d 个权重: %s", len(self.models), list(self.models.keys()))

    def get(self, name: str) -> Optional[YOLO]:
        return self.models.get(name)

    def reload(self):
        logger.info("重新加载所有 YOLO 权重...")
        self.load_all()

    def list_names(self) -> List[str]:
        return sorted(self.models.keys())


class YoloService:
    """封装 YOLO 推理逻辑"""

    def __init__(
        self,
        weights: Dict[str, str],
        default_weight: str,
        device: str = "cuda:0",
        conf: float = 0.35,
        iou: float = 0.45,
        max_det: int = 300
    ):
        if not weights:
            raise ValueError("至少提供一个 YOLO 权重")

        self.device = device
        self.default_weight = default_weight
        self.default_conf = conf
        self.default_iou = iou
        self.default_max_det = max_det

        self.registry = YOLORegistry(weights, device=device)
        if self.default_weight not in self.registry.models:
            available = self.registry.list_names()
            if not available:
                raise RuntimeError("没有可用的 YOLO 权重，无法启动服务")
            logger.warning("默认权重 %s 不可用，切换到 %s", self.default_weight, available[0])
            self.default_weight = available[0]

    @staticmethod
    def _load_image_from_base64(image_base64: str) -> Image.Image:
        if "," in image_base64:
            image_base64 = image_base64.split(",", 1)[1]
        image_bytes = base64.b64decode(image_base64)
        return Image.open(BytesIO(image_bytes))

    @staticmethod
    def _load_image_from_path(image_path: str) -> Image.Image:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片不存在: {image_path}")
        return Image.open(image_path)

    def _prepare_image(self, payload: Dict[str, Any]) -> Image.Image:
        if "image_base64" in payload and payload["image_base64"]:
            return self._load_image_from_base64(payload["image_base64"])
        if "image_path" in payload and payload["image_path"]:
            return self._load_image_from_path(payload["image_path"])
        raise ValueError("需要提供 image_path 或 image_base64")

    def _resolve_weight_name(self, payload: Dict[str, Any]) -> str:
        return payload.get("weight") or payload.get("model") or payload.get("checkpoint") or self.default_weight

    @staticmethod
    def _normalize_to_list(val: Any) -> List[Any]:
        if val is None:
            return []
        if isinstance(val, (list, tuple, set)):
            return list(val)
        return [val]

    def _resolve_classes(self, payload: Dict[str, Any], model: YOLO) -> Optional[List[int]]:
        """
        解析用户的类选择:
        - classes: list/int，直接指定 class_id
        - label_name / label_names: 通过类别名称映射到 id
        """
        target_ids: List[int] = []

        # 直接给定 class id
        for cls_val in self._normalize_to_list(payload.get("classes")):
            try:
                target_ids.append(int(cls_val))
            except (TypeError, ValueError):
                logger.warning("无法解析 classes 条目为 int: %s", cls_val)

        # 通过类别名称解析
        names = getattr(model.model, "names", {}) or {}
        if isinstance(names, dict):
            name_to_id = {str(v): int(k) for k, v in names.items()}
        else:
            # names 可能是 list/tuple
            name_to_id = {str(name): idx for idx, name in enumerate(list(names))}
        label_inputs = self._normalize_to_list(payload.get("label_name")) + self._normalize_to_list(payload.get("label_names"))
        for name in label_inputs:
            if name is None:
                continue
            name_str = str(name)
            if name_str in name_to_id:
                target_ids.append(name_to_id[name_str])
            else:
                logger.warning("未找到类别名称: %s，可选: %s", name_str, list(name_to_id.keys()))

        if not target_ids:
            return None
        # 去重排序
        return sorted(set(target_ids))

    @staticmethod
    def _encode_image_to_base64(image: Image.Image, fmt: str = "JPEG", quality: int = 90) -> str:
        """将 PIL Image 编码为 base64 文本，便于传给后续大模型。"""
        buffer = BytesIO()
        image.save(buffer, format=fmt, quality=quality)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def infer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        weight_name = self._resolve_weight_name(payload)
        model = self.registry.get(weight_name)
        if model is None:
            raise ValueError(f"weight '{weight_name}' 未加载或不存在")

        image = self._prepare_image(payload)
        conf = float(payload.get("conf", self.default_conf))
        iou = float(payload.get("iou", self.default_iou))
        max_det = int(payload.get("max_det", self.default_max_det))
        target_classes = self._resolve_classes(payload, model)

        with torch.inference_mode():
            results = model.predict(
                source=image,
                conf=conf,
                iou=iou,
                max_det=max_det,
                classes=target_classes,
                verbose=False,
            )

        result = results[0]
        boxes = getattr(result, "boxes", None)
        detections: List[Dict[str, Any]] = []
        if boxes is not None and boxes.xyxy is not None:
            xyxy = boxes.xyxy.cpu().tolist()
            cls_list = boxes.cls.cpu().tolist()
            conf_list = boxes.conf.cpu().tolist()
            names = getattr(model.model, "names", {})
            for bbox, cls_id, score in zip(xyxy, cls_list, conf_list):
                cls_int = int(cls_id)
                # 如果 classes 是 predict 过滤后的，一般无需再过滤；此处再按需过滤以防手动传错
                if target_classes and cls_int not in target_classes:
                    continue
                detections.append({
                    "bbox": [float(x) for x in bbox],
                    "class_id": cls_int,
                    "class_name": names.get(cls_int, str(cls_int)) if isinstance(names, dict) else str(cls_int),
                    "confidence": float(score)
                })

        response: Dict[str, Any] = {
            "model": weight_name,
            "num_detections": len(detections),
            "detections": detections,
        }

        # 按需返回可视化后的图片 base64，用于直接送给大模型
        if payload.get("return_image") or payload.get("return_visual"):
            annotated = result.plot()  # BGR ndarray
            if annotated is not None:
                vis_image = Image.fromarray(annotated[:, :, ::-1])  # 转成 RGB
                # vis_image = Image.fromarray(annotated)
                image_format = payload.get("image_format", "JPEG")
                image_quality = int(payload.get("image_quality", 90))
                b64 = self._encode_image_to_base64(vis_image, fmt=image_format, quality=image_quality)
                response["image_base64"] = b64
                response["image_format"] = image_format

        return response

    def batch_infer(self, payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        outputs = []
        for idx, item in enumerate(payloads):
            try:
                result = self.infer(item)
                result["index"] = idx
                if "image_path" in item:
                    result["image_path"] = item["image_path"]
                outputs.append({"success": True, "result": result})
            except Exception as exc:
                outputs.append({
                    "success": False,
                    "error": str(exc),
                    "index": idx,
                    "image_path": item.get("image_path"),
                })
        return outputs

    def health(self) -> Dict[str, Any]:
        return {
            "weights": self.registry.list_names(),
            "default_weight": self.default_weight,
            "device": self.device,
            "conf": self.default_conf,
            "iou": self.default_iou,
            "max_det": self.default_max_det,
        }


# ======================== Flask API ======================== #


@app.route("/health", methods=["GET"])
def health():
    if service_instance is None:
        return jsonify({"status": "initializing"}), 503
    return jsonify({"status": "ok", **service_instance.health()})


@app.route("/weights", methods=["GET"])
def weights():
    if service_instance is None:
        return jsonify({"error": "service not ready"}), 503
    return jsonify({"weights": service_instance.registry.list_names()})


@app.route("/reload", methods=["POST"])
def reload_weights():
    if service_instance is None:
        return jsonify({"error": "service not ready"}), 503
    service_instance.registry.reload()
    return jsonify({"status": "ok", "weights": service_instance.registry.list_names()})


@app.route("/inference", methods=["POST"])
def inference():
    if service_instance is None:
        return jsonify({"error": "service not ready"}), 503

    payload = request.get_json(force=True, silent=True) or {}
    try:
        result = service_instance.infer(payload)
        return jsonify({"success": True, "result": result})
    except Exception as exc:
        logger.exception("Inference failed")
        return jsonify({"success": False, "error": str(exc)}), 400


@app.route("/batch_inference", methods=["POST"])
def batch_inference():
    if service_instance is None:
        return jsonify({"error": "service not ready"}), 503

    payload = request.get_json(force=True, silent=True) or {}
    items = payload.get("items", [])
    if not isinstance(items, list) or not items:
        return jsonify({"error": "items 字段需要为非空数组"}), 400

    results = service_instance.batch_infer(items)
    return jsonify({"success": True, "results": results})


def discover_weights_from_dir(directory: Optional[str]) -> Dict[str, str]:
    if not directory:
        return {}
    weight_dir = Path(directory)
    if not weight_dir.exists():
        logger.warning("权重目录不存在: %s", directory)
        return {}
    weights = {}
    for file in sorted(weight_dir.glob("*.pt")):
        weights[file.stem] = str(file.resolve())
    for file in sorted(weight_dir.glob("*.pth")):
        weights[file.stem] = str(file.resolve())
    return weights


def parse_weight_args(args: argparse.Namespace) -> Dict[str, str]:
    weights = discover_weights_from_dir(args.weights_dir)

    for item in args.weights or []:
        if "=" not in item:
            raise ValueError(f"weight 参数需要 name=path 格式: {item}")
        name, path = item.split("=", 1)
        weights[name.strip()] = os.path.abspath(path.strip())

    return weights


def main():
    parser = argparse.ArgumentParser(description="YOLO 多权重推理服务")
    parser.add_argument("--weights_dir", type=str, help="包含若干 .pt/.pth 权重的目录，文件名作为权重名")
    parser.add_argument("--weight", action="append", dest="weights", help="额外指定的权重，格式 name=/path/to/best.pt，可多次指定")
    parser.add_argument("--default_weight", type=str, help="默认使用的权重名")
    parser.add_argument("--device", type=str, default="cuda:0", help="用于推理的设备")
    parser.add_argument("--conf", type=float, default=0.35, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU 阈值")
    parser.add_argument("--max_det", type=int, default=300, help="最大检测数量")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8810)
    parser.add_argument("--log_level", type=str, default="INFO")
    args = parser.parse_args()

    logging.getLogger().setLevel(args.log_level.upper())

    weights = parse_weight_args(args)
    if not weights:
        raise ValueError("没有发现任何权重，请通过 --weights_dir 或 --weight 指定")

    default_weight = args.default_weight or next(iter(weights.keys()))

    global service_instance
    service_instance = YoloService(
        weights=weights,
        default_weight=default_weight,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
    )

    logger.info("服务启动成功，监听 %s:%s", args.host, args.port)
    app.run(host=args.host, port=args.port, threaded=False)


if __name__ == "__main__":
    main()
