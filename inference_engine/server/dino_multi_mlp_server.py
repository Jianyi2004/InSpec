#!/usr/bin/env python3
"""
DINO + 多 MLP 常驻推理服务

特性:
1. 启动时常驻加载 DINO 特征提取器
2. 可以一次性挂载多个 MLP checkpoint，并通过名称进行选择
3. 提供 HTTP API（/health, /checkpoints, /inference, /batch_inference, /reload）
4. 请求格式尽量贴近 vLLM server，便于与现有 pipeline 对接

示例启动:
python dino_pipeline/dino_multi_mlp_server.py \
    --dino_model_path /path/to/dinov3 \
    --mlp_checkpoint_dir /path/to/checkpoints \
    --mlp_checkpoint day=/path/to/day.pth \
    --host 0.0.0.0 --port 8808

HTTP 调用示例:
curl -X POST http://localhost:8808/inference \
  -H "Content-Type: application/json" \
  -d '{"image_path": "/path/to/img.jpg", "checkpoint": "best"}'
"""

import argparse
import base64
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn
from PIL import Image
from flask import Flask, jsonify, request
from transformers import AutoImageProcessor, AutoModel

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger("dino-server")

app = Flask(__name__)
service_instance = None  # 全局服务实例


class DINOClassifier(nn.Module):
    """DINO 特征 -> MLP 分类器"""

    def __init__(self, feature_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


class DinoFeatureExtractor:
    """只负责管理 DINO 模型和特征抽取"""

    def __init__(self, model_path: str, device: str = "cuda:0"):
        logger.info("加载 DINO 模型: %s", model_path)
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_path)
        # trust_remote_code 避免一些定制模型报错
        self.model = AutoModel.from_pretrained(model_path, device_map=device, trust_remote_code=True)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()
        logger.info("DINO 模型加载完成")

    def extract(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            features = outputs.pooler_output.squeeze(0)
        return features


class MLPRegistry:
    """管理多个 MLP checkpoint，按名称加载"""

    def __init__(self, checkpoints: Dict[str, str], device: str = "cuda:0"):
        self.device = device
        self.checkpoint_paths = checkpoints
        self.models: Dict[str, DINOClassifier] = {}
        self.load_all()

    @staticmethod
    def _resolve_dims(state_dict: Dict[str, torch.Tensor]) -> (int, int):
        first_key = "classifier.0.weight"
        if first_key in state_dict:
            hidden_dim, feature_dim = state_dict[first_key].shape
            return feature_dim, hidden_dim
        # 回退默认值
        return 1024, 512

    def load_single(self, name: str, path: str) -> DINOClassifier:
        logger.info("加载 MLP checkpoint: %s -> %s", name, path)
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        feature_dim, hidden_dim = self._resolve_dims(state_dict)
        model = DINOClassifier(feature_dim, hidden_dim).to(self.device)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def load_all(self):
        self.models.clear()
        for name, path in self.checkpoint_paths.items():
            if not os.path.exists(path):
                logger.warning("checkpoint %s 不存在: %s", name, path)
                continue
            try:
                self.models[name] = self.load_single(name, path)
            except Exception as exc:
                logger.exception("加载 checkpoint %s 失败: %s", name, exc)

        logger.info("已加载 %d 个 MLP checkpoint: %s", len(self.models), list(self.models.keys()))

    def get(self, name: str) -> Optional[DINOClassifier]:
        return self.models.get(name)

    def reload(self):
        logger.info("重新加载所有 MLP checkpoint...")
        self.load_all()

    def list_names(self) -> List[str]:
        return sorted(self.models.keys())


class DinoMLPService:
    """封装图片推理逻辑"""

    def __init__(
        self,
        dino_model_path: str,
        mlp_checkpoints: Dict[str, str],
        default_checkpoint: str,
        checkpoint_thresholds: Dict[str, float],
        device: str = "cuda:0",
        threshold: float = 0.5
    ):
        if not mlp_checkpoints:
            raise ValueError("至少提供一个 MLP checkpoint")

        self.device = device
        self.default_checkpoint = default_checkpoint
        self.default_threshold = threshold
        self.checkpoint_thresholds = checkpoint_thresholds or {}

        self.extractor = DinoFeatureExtractor(dino_model_path, device=device)
        self.registry = MLPRegistry(mlp_checkpoints, device=device)

        if self.default_checkpoint not in self.registry.models:
            # 如果默认 checkpoint 没有成功加载，则选第一个
            available = self.registry.list_names()
            if not available:
                raise RuntimeError("没有可用的 MLP checkpoint，无法启动服务")
            logger.warning("默认 checkpoint %s 不可用，切换到 %s", self.default_checkpoint, available[0])
            self.default_checkpoint = available[0]

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

    def _resolve_threshold(self, checkpoint_name: str, payload_threshold: Optional[float]) -> float:
        if payload_threshold is not None:
            return float(payload_threshold)
        if checkpoint_name in self.checkpoint_thresholds:
            return float(self.checkpoint_thresholds[checkpoint_name])
        return self.default_threshold

    def infer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        checkpoint_name = payload.get("checkpoint") or self.default_checkpoint
        threshold = self._resolve_threshold(checkpoint_name, payload.get("threshold"))

        classifier = self.registry.get(checkpoint_name)
        if classifier is None:
            raise ValueError(f"checkpoint '{checkpoint_name}' 未加载或不存在")

        image = self._prepare_image(payload)
        features = self.extractor.extract(image)

        with torch.no_grad():
            probs = classifier(features.to(self.device).unsqueeze(0))
            prob = float(probs.squeeze().item())

        label = 1 if prob >= threshold else 0
        positive_text = payload.get("positive_text") or "positive"
        negative_text = payload.get("negative_text") or "negative"
        mapped_text = positive_text if label == 1 else negative_text
        result_label = "PASS" if label == 1 else "FAIL"

        return {
            "analysis": [mapped_text],
            "result": result_label
        }

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
                    "image_path": item.get("image_path")
                })
        return outputs

    def health(self) -> Dict[str, Any]:
        return {
            "dino_model_loaded": True,
            "mlp_checkpoints": self.registry.list_names(),
            "default_checkpoint": self.default_checkpoint,
            "checkpoint_thresholds": self.checkpoint_thresholds
        }


# ======================== Flask API ======================== #


@app.route("/health", methods=["GET"])
def health():
    if service_instance is None:
        return jsonify({"status": "initializing"}), 503
    return jsonify({"status": "ok", **service_instance.health()})


@app.route("/checkpoints", methods=["GET"])
def checkpoints():
    if service_instance is None:
        return jsonify({"error": "service not ready"}), 503
    return jsonify({"checkpoints": service_instance.registry.list_names()})


@app.route("/reload", methods=["POST"])
def reload_checkpoints():
    if service_instance is None:
        return jsonify({"error": "service not ready"}), 503
    service_instance.registry.reload()
    return jsonify({"status": "ok", "checkpoints": service_instance.registry.list_names()})


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


def discover_checkpoints_from_dir(directory: Optional[str]) -> Dict[str, str]:
    if not directory:
        return {}
    ckpt_dir = Path(directory)
    if not ckpt_dir.exists():
        logger.warning("checkpoint 目录不存在: %s", directory)
        return {}
    checkpoints = {}
    for file in sorted(ckpt_dir.glob("*.pth")):
        checkpoints[file.stem] = str(file.resolve())
    return checkpoints


def parse_threshold_args(args: argparse.Namespace) -> Dict[str, float]:
    thresholds: Dict[str, float] = {}
    for item in args.mlp_thresholds or []:
        if "=" not in item:
            raise ValueError(f"mlp_threshold 参数需要 name=value 格式: {item}")
        name, value = item.split("=", 1)
        thresholds[name.strip()] = float(value.strip())
    return thresholds


def parse_checkpoint_args(args: argparse.Namespace) -> Dict[str, str]:
    checkpoints = discover_checkpoints_from_dir(args.mlp_checkpoint_dir)

    for item in args.mlp_checkpoints or []:
        if "=" not in item:
            raise ValueError(f"mlp_checkpoint 参数需要 name=path 格式: {item}")
        name, path = item.split("=", 1)
        checkpoints[name.strip()] = os.path.abspath(path.strip())

    return checkpoints


def main():
    parser = argparse.ArgumentParser(description="DINO 多 MLP 推理服务")
    parser.add_argument("--dino_model_path", type=str, required=True, help="DINO 模型路径")
    parser.add_argument("--mlp_checkpoint_dir", type=str, help="包含若干 .pth 的目录，文件名作为 checkpoint 名")
    parser.add_argument("--mlp_checkpoint", action="append", dest="mlp_checkpoints",
                        help="额外的 checkpoint，格式 name=/path/to/model.pth，可以多次指定")
    parser.add_argument("--mlp_threshold", action="append", dest="mlp_thresholds",
                        help="针对特定 checkpoint 的阈值，格式 name=0.6，可多次指定")
    parser.add_argument("--default_checkpoint", type=str, help="默认使用的 checkpoint 名")
    parser.add_argument("--device", type=str, default="cuda:0", help="用于推理的设备")
    parser.add_argument("--threshold", type=float, default=0.5, help="默认判定阈值")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8808)
    parser.add_argument("--log_level", type=str, default="INFO")
    args = parser.parse_args()

    logging.getLogger().setLevel(args.log_level.upper())

    checkpoints = parse_checkpoint_args(args)
    checkpoint_thresholds = parse_threshold_args(args)
    if not checkpoints:
        raise ValueError("没有发现任何 checkpoint，请通过 --mlp_checkpoint_dir 或 --mlp_checkpoint 指定")

    default_checkpoint = args.default_checkpoint or next(iter(checkpoints.keys()))

    global service_instance
    service_instance = DinoMLPService(
        dino_model_path=args.dino_model_path,
        mlp_checkpoints=checkpoints,
        default_checkpoint=default_checkpoint,
        checkpoint_thresholds=checkpoint_thresholds,
        device=args.device,
        threshold=args.threshold
    )

    logger.info("服务启动成功，监听 %s:%s", args.host, args.port)
    app.run(host=args.host, port=args.port, threaded=False)


if __name__ == "__main__":
    main()
