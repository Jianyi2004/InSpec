#!/usr/bin/env python3
"""
RT-DETR 多权重常驻推理服务 (基于 JIT 模型)

特性:
1. 启动时一次性挂载多个 RT-DETR JIT 权重，通过名称选择
2. 支持 image_path 或 image_base64 输入
3. 提供 /health /weights /inference /batch_inference /reload 接口
4. 自实现预处理、后处理、NMS（不依赖 ultralytics）

示例启动:
python rtdetr_multi_server.py \
    --weights_dir /path/to/weights_dir \
    --weight bbu=/path/to/model.jit \
    --default_weight bbu \
    --host 0.0.0.0 --port 8811
"""

import argparse
import base64
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.ops as ops
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, jsonify, request

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger("rtdetr-server")

app = Flask(__name__)
service_instance = None

# 默认颜色表 (用于可视化)
COLOR_PALETTE = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
    (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
    (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
]


class RTDETRPreProcessor:
    """RT-DETR 图像预处理器"""

    def __init__(
        self,
        input_size: Tuple[int, int] = (640, 640),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        self.input_size = input_size  # (H, W)
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def preprocess(self, image: Image.Image, device: str = "cpu") -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        预处理图像

        Args:
            image: PIL Image (RGB)
            device: 目标设备

        Returns:
            tensor: [1, 3, H, W] 归一化后的张量
            meta: 包含原始尺寸等信息的字典
        """
        orig_w, orig_h = image.size

        # 转换为 RGB (如果是 RGBA 或其他模式)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize 到目标尺寸
        target_h, target_w = self.input_size
        resized = image.resize((target_w, target_h), Image.BILINEAR)

        # 转换为 tensor [3, H, W]，范围 [0, 1]
        img_array = np.array(resized, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # HWC -> CHW

        # 标准化
        mean = self.mean.to(img_tensor.device)
        std = self.std.to(img_tensor.device)
        img_tensor = (img_tensor - mean) / std

        # 添加 batch 维度
        img_tensor = img_tensor.unsqueeze(0).to(device)

        meta = {
            "orig_size": (orig_w, orig_h),
            "input_size": (target_w, target_h),
            "scale_w": target_w / orig_w,
            "scale_h": target_h / orig_h,
        }

        return img_tensor, meta


class RTDETRPostProcessor:
    """RT-DETR 后处理器"""

    def __init__(
        self,
        num_classes: int = 80,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        max_det: int = 300,
    ):
        self.num_classes = num_classes
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det

    @staticmethod
    def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        """
        将 (cx, cy, w, h) 格式转换为 (x1, y1, x2, y2) 格式

        Args:
            boxes: [N, 4] tensor in (cx, cy, w, h) format

        Returns:
            boxes: [N, 4] tensor in (x1, y1, x2, y2) format
        """
        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def nms(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        class_ids: torch.Tensor,
        iou_threshold: float,
    ) -> torch.Tensor:
        """
        执行类别感知的 NMS

        Args:
            boxes: [N, 4] tensor in (x1, y1, x2, y2) format
            scores: [N] tensor
            class_ids: [N] tensor
            iou_threshold: IoU 阈值

        Returns:
            keep: 保留的索引
        """
        if boxes.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=boxes.device)

        # 使用 torchvision 的 batched_nms (类别感知)
        keep = ops.batched_nms(boxes, scores, class_ids, iou_threshold)
        return keep

    def postprocess(
        self,
        outputs: torch.Tensor,
        meta: Dict[str, Any],
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        max_det: Optional[int] = None,
        target_classes: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        后处理 RT-DETR 输出

        Args:
            outputs: 模型输出，通常为 [B, num_queries, 4 + num_classes] 或元组
            meta: 预处理元信息
            conf_threshold: 置信度阈值
            iou_threshold: NMS IoU 阈值
            max_det: 最大检测数量
            target_classes: 目标类别 ID 列表 (可选)

        Returns:
            detections: 检测结果列表
        """
        conf_threshold = conf_threshold or self.conf_threshold
        iou_threshold = iou_threshold or self.iou_threshold
        max_det = max_det or self.max_det

        # 处理不同的输出格式
        if isinstance(outputs, (tuple, list)):
            # 某些模型返回 (logits, boxes) 或 (boxes, logits)
            if len(outputs) == 2:
                # 尝试判断哪个是 boxes，哪个是 logits
                out0, out1 = outputs
                if out0.shape[-1] == 4:
                    pred_boxes, pred_logits = out0, out1
                else:
                    pred_logits, pred_boxes = out0, out1
            else:
                # 假设第一个是主输出
                outputs = outputs[0]
                pred_logits = outputs[..., 4:]
                pred_boxes = outputs[..., :4]
        elif isinstance(outputs, dict):
            # 字典格式输出
            pred_logits = outputs.get("pred_logits", outputs.get("logits"))
            pred_boxes = outputs.get("pred_boxes", outputs.get("boxes"))
        else:
            # 单一张量 [B, num_queries, 4 + num_classes]
            pred_boxes = outputs[..., :4]
            pred_logits = outputs[..., 4:]

        # 确保是 3D tensor [B, N, ...]
        if pred_boxes.dim() == 2:
            pred_boxes = pred_boxes.unsqueeze(0)
        if pred_logits.dim() == 2:
            pred_logits = pred_logits.unsqueeze(0)

        batch_size = pred_boxes.shape[0]
        results = []

        for b in range(batch_size):
            boxes = pred_boxes[b]  # [N, 4]
            logits = pred_logits[b]  # [N, num_classes]

            # 获取置信度和类别
            if logits.shape[-1] > 1:
                # 多类别：取最大值
                scores, class_ids = logits.sigmoid().max(dim=-1)
            else:
                # 单类别或已经是 sigmoid 后的值
                scores = logits.sigmoid().squeeze(-1)
                class_ids = torch.zeros_like(scores, dtype=torch.long)

            # 置信度过滤
            mask = scores >= conf_threshold

            # 类别过滤
            if target_classes is not None:
                class_mask = torch.zeros_like(mask, dtype=torch.bool)
                for cls_id in target_classes:
                    class_mask |= (class_ids == cls_id)
                mask &= class_mask

            boxes = boxes[mask]
            scores = scores[mask]
            class_ids = class_ids[mask]

            if boxes.numel() == 0:
                results.append([])
                continue

            # 转换 box 格式 (如果是 cxcywh)
            # RT-DETR 通常输出归一化的 cxcywh
            if boxes.max() <= 1.0:
                # 归一化坐标，先转换到 xyxy，再缩放
                boxes = self.box_cxcywh_to_xyxy(boxes)
                # 缩放到输入尺寸
                input_w, input_h = meta["input_size"]
                boxes[:, [0, 2]] *= input_w
                boxes[:, [1, 3]] *= input_h
            else:
                # 已经是绝对坐标的 cxcywh
                boxes = self.box_cxcywh_to_xyxy(boxes)

            # NMS
            keep = self.nms(boxes, scores, class_ids, iou_threshold)
            keep = keep[:max_det]

            boxes = boxes[keep]
            scores = scores[keep]
            class_ids = class_ids[keep]

            # 缩放回原始图像尺寸
            orig_w, orig_h = meta["orig_size"]
            input_w, input_h = meta["input_size"]
            scale_x = orig_w / input_w
            scale_y = orig_h / input_h

            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y

            # 裁剪到图像边界
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, orig_w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, orig_h)

            # 转换为列表
            detections = []
            for i in range(boxes.shape[0]):
                detections.append({
                    "bbox": boxes[i].cpu().tolist(),
                    "class_id": int(class_ids[i].item()),
                    "confidence": float(scores[i].item()),
                })

            results.append(detections)

        return results


class RTDETRVisualizer:
    """RT-DETR 检测结果可视化器 (与 YOLO 风格一致)"""

    # 与 yolo_multi_server.py 相同的颜色表 (HEX -> RGB)
    YOLO_COLORS = [
        (255, 56, 56),   # FF3838 - 0
        (255, 157, 151), # FF9D97 - 1
        (255, 112, 31),  # FF701F - 2
        (255, 178, 29),  # FFB21D - 3
        (207, 210, 49),  # CFD231 - 4
        (72, 249, 10),   # 48F90A - 5
        (146, 204, 23),  # 92CC17 - 6
        (61, 219, 134),  # 3DDB86 - 7
        (26, 147, 52),   # 1A9334 - 8
        (0, 212, 187),   # 00D4BB - 9
        (44, 153, 168),  # 2C99A8 - 10
        (0, 194, 255),   # 00C2FF - 11
        (52, 69, 147),   # 344593 - 12
        (100, 115, 255), # 6473FF - 13
        (0, 24, 236),    # 0018EC - 14
        (132, 56, 255),  # 8438FF - 15
        (82, 0, 133),    # 520085 - 16
        (203, 56, 255),  # CB38FF - 17
        (255, 149, 200), # FF95C8 - 18
        (255, 55, 199),  # FF37C7 - 19
    ]

    # 浅色背景需要深色文字
    LIGHT_COLORS = {
        (255, 157, 151), (255, 178, 29), (207, 210, 49), (72, 249, 10),
        (146, 204, 23), (61, 219, 134), (0, 212, 187), (0, 194, 255),
        (255, 149, 200),
    }

    def __init__(
        self,
        class_names: Optional[Dict[int, str]] = None,
        colors: Optional[List[Tuple[int, int, int]]] = None,
        line_width: Optional[int] = None,
        font_size: Optional[int] = None,
    ):
        self.class_names = class_names or {}
        self.colors = colors or self.YOLO_COLORS
        self.line_width = line_width
        self.font_size = font_size

    def _get_color(self, class_id: int) -> Tuple[int, int, int]:
        """获取类别对应的颜色 (RGB)"""
        return self.colors[int(class_id) % len(self.colors)]

    def _get_txt_color(self, bg_color: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """根据背景色自动选择文字颜色"""
        if bg_color in self.LIGHT_COLORS:
            return (0, 0, 0)  # 黑色文字
        return (255, 255, 255)  # 白色文字

    def _load_font(self, size: int) -> ImageFont.FreeTypeFont:
        """加载字体"""
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "C:/Windows/Fonts/Arial.ttf",
        ]
        for path in font_paths:
            try:
                return ImageFont.truetype(path, size)
            except (OSError, IOError):
                continue
        return ImageFont.load_default()

    def draw(
        self,
        image: Image.Image,
        detections: List[Dict[str, Any]],
        line_width: Optional[int] = None,
        font_size: Optional[int] = None,
    ) -> Image.Image:
        """
        在图像上绘制检测结果 (与 YOLO/Ultralytics 风格一致)

        Args:
            image: PIL Image
            detections: 检测结果列表
            line_width: 线宽 (可选，自动计算)
            font_size: 字体大小 (可选，自动计算)

        Returns:
            annotated: 绘制后的图像
        """
        image = image.copy()
        if image.mode != "RGB":
            image = image.convert("RGB")

        # 自动计算线宽和字体大小 (与 Ultralytics Annotator 一致)
        # 参考: lw = max(round(sum(im.shape[:2]) / 2 * 0.003), 2)
        # 参考: sf = lw / 3  (scale factor for font)
        img_w, img_h = image.size
        lw = line_width or self.line_width or max(round((img_w + img_h) / 2 * 0.003), 2)
        # 字体大小: Ultralytics 使用 sf = lw / 3，cv2.putText 的 fontScale
        # PIL 字体大小约为 fontScale * 30-40，这里使用更小的系数
        sf = lw / 3  # scale factor
        fs = font_size or self.font_size or max(int(sf * 25), 10)

        font = self._load_font(fs)
        draw = ImageDraw.Draw(image)

        for det in detections:
            bbox = det["bbox"]
            class_id = det["class_id"]
            confidence = det["confidence"]
            class_name = det.get("class_name") or self.class_names.get(class_id, str(class_id))

            x1, y1, x2, y2 = [int(v) for v in bbox]
            color = self._get_color(class_id)
            txt_color = self._get_txt_color(color)

            # 绘制边框
            draw.rectangle([x1, y1, x2, y2], outline=color, width=lw)

            # 绘制标签 (格式: "class conf")
            label = f"{class_name} {confidence:.2f}"

            # 获取文本尺寸
            try:
                text_bbox = draw.textbbox((0, 0), label, font=font)
                text_w = text_bbox[2] - text_bbox[0]
                text_h = text_bbox[3] - text_bbox[1]
            except AttributeError:
                text_w, text_h = draw.textsize(label, font=font)

            # 判断标签位置: 优先放在框上方
            outside = y1 >= text_h + 3

            # 如果标签超出右边界，调整位置
            label_x = x1
            if x1 + text_w + 1 > img_w:
                label_x = img_w - text_w - 1

            if outside:
                # 标签在框上方
                label_y1 = y1 - text_h - 3
                label_y2 = y1
            else:
                # 标签在框内上方
                label_y1 = y1
                label_y2 = y1 + text_h + 3

            # 绘制标签背景 (填充矩形)
            draw.rectangle([label_x, label_y1, label_x + text_w + 2, label_y2], fill=color)

            # 绘制标签文字
            draw.text((label_x + 1, label_y1), label, fill=txt_color, font=font)

        return image


class RTDETRRegistry:
    """管理多个 RT-DETR JIT 权重"""

    def __init__(
        self,
        weights: Dict[str, str],
        device: str = "cuda:0",
        class_names: Optional[Dict[str, Dict[int, str]]] = None,
    ):
        self.device = device
        self.weight_paths = weights
        self.models: Dict[str, torch.jit.ScriptModule] = {}
        self.class_names: Dict[str, Dict[int, str]] = class_names or {}
        self.load_all()

    def load_single(self, name: str, path: str) -> torch.jit.ScriptModule:
        logger.info("加载 RT-DETR JIT 权重: %s -> %s", name, path)
        model = torch.jit.load(path, map_location=self.device)
        model.eval()
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

    def get(self, name: str) -> Optional[torch.jit.ScriptModule]:
        return self.models.get(name)

    def get_class_names(self, name: str) -> Dict[int, str]:
        return self.class_names.get(name, {})

    def reload(self):
        logger.info("重新加载所有 RT-DETR 权重...")
        self.load_all()

    def list_names(self) -> List[str]:
        return sorted(self.models.keys())


class RTDETRService:
    """封装 RT-DETR 推理逻辑"""

    def __init__(
        self,
        weights: Dict[str, str],
        default_weight: str,
        device: str = "cuda:0",
        input_size: Tuple[int, int] = (640, 640),
        num_classes: int = 80,
        conf: float = 0.35,
        iou: float = 0.45,
        max_det: int = 300,
        class_names: Optional[Dict[str, Dict[int, str]]] = None,
    ):
        if not weights:
            raise ValueError("至少提供一个 RT-DETR 权重")

        self.device = device
        self.default_weight = default_weight
        self.input_size = input_size
        self.num_classes = num_classes
        self.default_conf = conf
        self.default_iou = iou
        self.default_max_det = max_det

        self.preprocessor = RTDETRPreProcessor(input_size=input_size)
        self.postprocessor = RTDETRPostProcessor(
            num_classes=num_classes,
            conf_threshold=conf,
            iou_threshold=iou,
            max_det=max_det,
        )
        self.visualizer = RTDETRVisualizer()

        self.registry = RTDETRRegistry(weights, device=device, class_names=class_names)

        if self.default_weight not in self.registry.models:
            available = self.registry.list_names()
            if not available:
                raise RuntimeError("没有可用的 RT-DETR 权重，无法启动服务")
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

    def _resolve_classes(
        self,
        payload: Dict[str, Any],
        weight_name: str,
        client_class_names: Optional[Dict[int, str]] = None,
    ) -> Optional[List[int]]:
        """
        解析用户的类选择:
        - classes: list/int，直接指定 class_id
        - label_name / label_names: 通过类别名称映射到 id

        Args:
            payload: 请求参数
            weight_name: 权重名称
            client_class_names: 客户端传入的 class_id -> name 映射 (优先使用)
        """
        target_ids: List[int] = []

        # 直接给定 class id
        for cls_val in self._normalize_to_list(payload.get("classes")):
            try:
                target_ids.append(int(cls_val))
            except (TypeError, ValueError):
                logger.warning("无法解析 classes 条目为 int: %s", cls_val)

        # 通过类别名称解析
        # 优先使用客户端传入的 class_id_to_name，否则使用服务端配置
        if client_class_names:
            names = client_class_names
        else:
            names = self.registry.get_class_names(weight_name)

        if names:
            name_to_id = {str(v): int(k) for k, v in names.items()}
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
        return sorted(set(target_ids))

    @staticmethod
    def _encode_image_to_base64(image: Image.Image, fmt: str = "JPEG", quality: int = 90) -> str:
        """将 PIL Image 编码为 base64 文本"""
        buffer = BytesIO()
        image.save(buffer, format=fmt, quality=quality)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _parse_rtdetr_outputs(
        self,
        outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        meta: Dict[str, Any],
        conf_threshold: float,
        iou_threshold: float,
        max_det: int,
        target_classes: Optional[List[int]],
    ) -> List[Dict[str, Any]]:
        """
        解析 RT-DETR 模型输出 (模型已内置后处理器)

        Args:
            outputs: (labels, boxes, scores) 三元组
                - labels: [B, num_queries] 类别 ID
                - boxes: [B, num_queries, 4] 边界框 (基于 INPUT_SIZE 的 xyxy 格式)
                - scores: [B, num_queries] 置信度
            meta: 预处理元信息
            conf_threshold: 置信度阈值
            iou_threshold: NMS IoU 阈值 (模型可能已内置 NMS)
            max_det: 最大检测数量
            target_classes: 目标类别 ID 列表

        Returns:
            检测结果列表
        """
        labels, boxes, scores = outputs

        # 取第一个 batch
        labels = labels[0]  # [num_queries]
        boxes = boxes[0]    # [num_queries, 4]
        scores = scores[0]  # [num_queries]

        # 置信度过滤
        mask = scores >= conf_threshold

        # 类别过滤
        if target_classes is not None:
            class_mask = torch.zeros_like(mask, dtype=torch.bool)
            for cls_id in target_classes:
                class_mask |= (labels == cls_id)
            mask &= class_mask

        labels = labels[mask]
        boxes = boxes[mask]
        scores = scores[mask]

        # 限制最大检测数量
        if len(scores) > max_det:
            topk_indices = scores.topk(max_det).indices
            labels = labels[topk_indices]
            boxes = boxes[topk_indices]
            scores = scores[topk_indices]

        # 将 boxes 从 INPUT_SIZE 坐标缩放到原图坐标
        orig_w, orig_h = meta["orig_size"]
        input_w, input_h = meta["input_size"]

        # boxes 格式: [x1, y1, x2, y2]，基于 input_size
        boxes[:, 0] = boxes[:, 0] / input_w * orig_w  # x1
        boxes[:, 1] = boxes[:, 1] / input_h * orig_h  # y1
        boxes[:, 2] = boxes[:, 2] / input_w * orig_w  # x2
        boxes[:, 3] = boxes[:, 3] / input_h * orig_h  # y2

        # 裁剪到图像边界
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, orig_w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, orig_h)

        # 转换为列表
        detections = []
        for i in range(boxes.shape[0]):
            detections.append({
                "bbox": [int(x) for x in boxes[i].cpu().tolist()],
                "class_id": int(labels[i].item()),
                "confidence": float(scores[i].item()),
            })

        return detections


    @staticmethod
    def _compute_iou(box1: List[float], box2: List[float]) -> float:
        """
        计算两个边界框的 IoU

        Args:
            box1: [x1, y1, x2, y2]
            box2: [x1, y1, x2, y2]

        Returns:
            IoU 值
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union = area1 + area2 - inter + 1e-6
        return inter / union

    @staticmethod
    def _boxes_overlap(box1: List[float], box2: List[float]) -> bool:
        """
        判断两个边界框是否有重叠

        Args:
            box1: [x1, y1, x2, y2]
            box2: [x1, y1, x2, y2]

        Returns:
            是否重叠
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        return x2 > x1 and y2 > y1

    def _suppress_conflicts(
        self,
        detections: List[Dict[str, Any]],
        conflict_classes: List[str],
        iou_threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        冲突抑制: 对于指定类别在同一位置出现时，保留置信度更高的框

        Args:
            detections: 检测结果列表
            conflict_classes: 需要进行冲突抑制的类别名称列表
            iou_threshold: IoU 阈值，默认 0.5

        Returns:
            抑制后的检测结果列表
        """
        if not conflict_classes or not detections:
            return detections

        # 分离需要抑制的框和其他框
        conflict_dets = []
        other_dets = []

        for det in detections:
            class_name = det.get("class_name", str(det["class_id"]))
            if class_name in conflict_classes:
                conflict_dets.append(det)
            else:
                other_dets.append(det)

        if len(conflict_dets) <= 1:
            return detections

        # 对冲突类别的框进行抑制
        suppressed = set()
        for i in range(len(conflict_dets)):
            if i in suppressed:
                continue

            for j in range(i + 1, len(conflict_dets)):
                if j in suppressed:
                    continue

                det_i = conflict_dets[i]
                det_j = conflict_dets[j]

                # 只有不同类别之间才进行冲突抑制
                if det_i.get("class_name") == det_j.get("class_name"):
                    continue

                iou = self._compute_iou(det_i["bbox"], det_j["bbox"])
                if iou >= iou_threshold:
                    # 保留置信度更高的
                    if det_i["confidence"] >= det_j["confidence"]:
                        suppressed.add(j)
                    else:
                        suppressed.add(i)

        # 过滤被抑制的框
        filtered_conflict = [det for idx, det in enumerate(conflict_dets) if idx not in suppressed]

        return other_dets + filtered_conflict

    def _filter_by_parent(
        self,
        detections: List[Dict[str, Any]],
        parent_class: str,
    ) -> List[Dict[str, Any]]:
        """
        根据父类别过滤检出框: 仅保留与父类别框有重叠的检出框

        Args:
            detections: 检测结果列表
            parent_class: 父类别名称

        Returns:
            过滤后的检测结果列表
        """
        if not parent_class or not detections:
            return detections

        # 找到所有父类别的框
        parent_boxes = []
        for det in detections:
            class_name = det.get("class_name", str(det["class_id"]))
            if class_name == parent_class:
                parent_boxes.append(det["bbox"])

        # 如果没有检出父类别，返回所有框
        if not parent_boxes:
            return detections

        # 过滤: 保留父类别框 + 与父类别有重叠的其他框
        filtered = []
        for det in detections:
            class_name = det.get("class_name", str(det["class_id"]))
            if class_name == parent_class:
                # 父类别框直接保留
                filtered.append(det)
            else:
                # 其他框需要与至少一个父类别框重叠
                for parent_box in parent_boxes:
                    if self._boxes_overlap(det["bbox"], parent_box):
                        filtered.append(det)
                        break

        return filtered

    def infer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        weight_name = self._resolve_weight_name(payload)
        model = self.registry.get(weight_name)
        if model is None:
            raise ValueError(f"weight '{weight_name}' 未加载或不存在")

        image = self._prepare_image(payload)
        conf = float(payload.get("conf", self.default_conf))
        iou = float(payload.get("iou", self.default_iou))
        max_det = int(payload.get("max_det", self.default_max_det))

        # 解析客户端传入的 class_id_to_name (优先使用)
        client_class_names_raw = payload.get("class_id_to_name")
        if client_class_names_raw:
            # 支持字符串格式的 JSON (从 prompt 配置传入时可能是字符串)
            if isinstance(client_class_names_raw, str):
                try:
                    client_class_names_raw = json.loads(client_class_names_raw)
                except json.JSONDecodeError:
                    logger.warning("无法解析 class_id_to_name JSON 字符串: %s", client_class_names_raw)
                    client_class_names_raw = None
            if client_class_names_raw and isinstance(client_class_names_raw, dict):
                class_names = {int(k): str(v) for k, v in client_class_names_raw.items()}
            else:
                class_names = self.registry.get_class_names(weight_name)
        else:
            class_names = self.registry.get_class_names(weight_name)

        # 解析目标类别 (传入 class_names 以支持 label_names 参数)
        target_classes = self._resolve_classes(payload, weight_name, class_names)

        # 预处理
        input_tensor, meta = self.preprocessor.preprocess(image, device=self.device)

        # 准备 orig_target_sizes 参数 (传入 INPUT_SIZE，模型返回基于此尺寸的坐标)
        input_h, input_w = self.input_size
        orig_target_sizes = torch.tensor([[input_h, input_w]], dtype=torch.float32, device=self.device)

        # 推理 (模型已内置后处理器，返回 labels, boxes, scores)
        with torch.inference_mode():
            outputs = model(input_tensor, orig_target_sizes)

        # 解析模型输出 (labels, boxes, scores)
        detections = self._parse_rtdetr_outputs(
            outputs,
            meta,
            conf_threshold=conf,
            iou_threshold=iou,
            max_det=max_det,
            target_classes=target_classes,
        )

        # 添加类别名称
        for det in detections:
            cls_id = det["class_id"]
            det["class_name"] = class_names.get(cls_id, str(cls_id)) if class_names else str(cls_id)

        # 后处理: 冲突抑制 (conflict_suppress)
        conflict_suppress = payload.get("conflict_suppress")
        if conflict_suppress:
            if isinstance(conflict_suppress, str):
                conflict_suppress = [conflict_suppress]
            conflict_iou = float(payload.get("conflict_iou", 0.5))
            detections = self._suppress_conflicts(detections, conflict_suppress, conflict_iou)

        # 后处理: 父类别过滤 (parent_filter)
        parent_filter = payload.get("parent_filter")
        if parent_filter:
            detections = self._filter_by_parent(detections, parent_filter)

        response: Dict[str, Any] = {
            "model": weight_name,
            "num_detections": len(detections),
            "detections": detections,
        }

        # 按需返回可视化后的图片 base64
        if payload.get("return_image") or payload.get("return_visual"):
            self.visualizer.class_names = class_names
            annotated = self.visualizer.draw(image, detections)
            image_format = payload.get("image_format", "JPEG")
            image_quality = int(payload.get("image_quality", 90))
            b64 = self._encode_image_to_base64(annotated, fmt=image_format, quality=image_quality)
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
            "input_size": self.input_size,
            "num_classes": self.num_classes,
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
    # 支持 .jit, .pt, .pth 格式
    for ext in ["*.jit", "*.pt", "*.pth"]:
        for file in sorted(weight_dir.glob(ext)):
            weights[file.stem] = str(file.resolve())
    return weights


def load_class_names_from_file(path: str) -> Dict[int, str]:
    """
    从文件加载类别名称
    支持格式:
    - 每行一个类别名称 (行号作为 class_id)
    - JSON 格式 {"0": "class0", "1": "class1", ...}
    """
    if not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # 尝试 JSON 格式
    if content.startswith("{"):
        import json
        data = json.loads(content)
        return {int(k): str(v) for k, v in data.items()}

    # 每行一个类别
    lines = content.split("\n")
    return {i: line.strip() for i, line in enumerate(lines) if line.strip()}


def parse_weight_args(args: argparse.Namespace) -> Tuple[Dict[str, str], Dict[str, Dict[int, str]]]:
    weights = discover_weights_from_dir(args.weights_dir)
    class_names: Dict[str, Dict[int, str]] = {}

    for item in args.weights or []:
        if "=" not in item:
            raise ValueError(f"weight 参数需要 name=path 格式: {item}")
        name, path = item.split("=", 1)
        weights[name.strip()] = os.path.abspath(path.strip())

    # 解析类别名称
    for item in args.class_names or []:
        if "=" not in item:
            continue
        name, path = item.split("=", 1)
        names = load_class_names_from_file(path.strip())
        if names:
            class_names[name.strip()] = names

    return weights, class_names


def main():
    parser = argparse.ArgumentParser(description="RT-DETR 多权重推理服务")
    parser.add_argument("--weights_dir", type=str, help="包含若干 .jit/.pt/.pth 权重的目录，文件名作为权重名")
    parser.add_argument("--weight", action="append", dest="weights", help="额外指定的权重，格式 name=/path/to/model.jit，可多次指定")
    parser.add_argument("--class_names", action="append", dest="class_names", help="类别名称文件，格式 weight_name=/path/to/names.txt，可多次指定")
    parser.add_argument("--default_weight", type=str, help="默认使用的权重名")
    parser.add_argument("--device", type=str, default="cuda:0", help="用于推理的设备")
    parser.add_argument("--input_size", type=int, nargs=2, default=[640, 640], help="输入图像尺寸 (H W)")
    parser.add_argument("--num_classes", type=int, default=80, help="类别数量")
    parser.add_argument("--conf", type=float, default=0.35, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU 阈值")
    parser.add_argument("--max_det", type=int, default=300, help="最大检测数量")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8811)
    parser.add_argument("--log_level", type=str, default="INFO")
    args = parser.parse_args()

    logging.getLogger().setLevel(args.log_level.upper())

    weights, class_names = parse_weight_args(args)
    if not weights:
        raise ValueError("没有发现任何权重，请通过 --weights_dir 或 --weight 指定")

    default_weight = args.default_weight or next(iter(weights.keys()))

    global service_instance
    service_instance = RTDETRService(
        weights=weights,
        default_weight=default_weight,
        device=args.device,
        input_size=tuple(args.input_size),
        num_classes=args.num_classes,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        class_names=class_names,
    )

    logger.info("服务启动成功，监听 %s:%s", args.host, args.port)
    app.run(host=args.host, port=args.port, threaded=False)


if __name__ == "__main__":
    main()
