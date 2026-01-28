# Server 服务集

本目录包含工业质检推理引擎的各类模型推理服务，包括 vLLM 视觉语言模型服务和多种目标检测模型服务（DINO、YOLO、RT-DETR）。

## 目录

- [vllm_serve.sh](#vllm_servesh) - vLLM 视觉语言模型服务
- [dino_multi_mlp_server.py](#dino_multi_mlp_serverpy) - DINO + MLP 分类服务
- [yolo_multi_server.py](#yolo_multi_serverpy) - YOLO 目标检测服务
- [rtdetr_multi_server.py](#rtdetr_multi_serverpy) - RT-DETR 目标检测服务
- [启动脚本](#启动脚本) - 各服务的便捷启动脚本

---

## vllm_serve.sh

vLLM 视觉语言模型服务启动脚本，用于启动 Qwen3-VL-32B-Instruct 模型的推理服务。

### 功能特性

- 基于 vLLM 框架的高性能推理
- 支持张量并行（Tensor Parallelism）
- 自动 GPU 内存管理
- 支持长上下文（32K tokens）

### 配置说明

```bash
MODEL_PATH="/data_all/share/models/Qwen3-VL-32B-Instruct"  # 模型路径
export CUDA_VISIBLE_DEVICES=2,3                             # 使用的 GPU
--host 0.0.0.0                                              # 监听地址
--port 7878                                                 # 服务端口
--tensor-parallel-size 2                                    # 张量并行大小
--max-model-len 32768                                       # 最大上下文长度
--gpu-memory-utilization 0.9                                # GPU 内存利用率
```

### 使用方法

```bash
# 直接启动
bash vllm_serve.sh

# 或赋予执行权限后启动
chmod +x vllm_serve.sh
./vllm_serve.sh
```

### API 调用示例

```bash
# 健康检查
curl http://localhost:7878/health

# 推理请求
curl -X POST http://localhost:7878/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-VL-32B-Instruct",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "image_url", "image_url": {"url": "file:///path/to/image.jpg"}},
          {"type": "text", "text": "请描述这张图片"}
        ]
      }
    ]
  }'
```

---

## dino_multi_mlp_server.py

DINO（自监督视觉 Transformer）+ 多 MLP 分类器推理服务，支持一次性挂载多个 MLP checkpoint 并通过名称选择。

### 功能特性

1. **常驻加载** - 启动时加载 DINO 特征提取器，避免重复加载
2. **多 checkpoint 管理** - 支持同时挂载多个 MLP 分类器
3. **灵活阈值配置** - 支持为不同 checkpoint 设置独立阈值
4. **HTTP API** - 提供 RESTful API 接口
5. **热重载** - 支持运行时重新加载 checkpoint

### HTTP API 端点

- `GET /health` - 健康检查和服务状态
- `GET /checkpoints` - 列出所有已加载的 checkpoint
- `POST /reload` - 重新加载所有 checkpoint
- `POST /inference` - 单张图片推理
- `POST /batch_inference` - 批量图片推理

### 启动方法

```bash
# 使用启动脚本（推荐）
bash start_dino_multi_mlp_server.sh

# 或直接使用 Python
python dino_multi_mlp_server.py \
    --dino_model_path /path/to/dinov3 \
    --mlp_checkpoint_dir /path/to/checkpoints \
    --mlp_checkpoint day=/path/to/day.pth \
    --mlp_threshold best=0.55 \
    --default_checkpoint best \
    --device cuda:0 \
    --threshold 0.5 \
    --host 0.0.0.0 \
    --port 8808
```

### 参数说明

- `--dino_model_path`: DINO 模型路径（必需）
- `--mlp_checkpoint_dir`: MLP checkpoint 目录（自动扫描 .pth 文件）
- `--mlp_checkpoint`: 额外的 checkpoint，格式 `name=/path/to/model.pth`（可多次指定）
- `--mlp_threshold`: 特定 checkpoint 的阈值，格式 `name=0.6`（可多次指定）
- `--default_checkpoint`: 默认使用的 checkpoint 名
- `--device`: 推理设备，默认 `cuda:0`
- `--threshold`: 默认判定阈值，默认 `0.5`
- `--host`: 监听地址，默认 `0.0.0.0`
- `--port`: 监听端口，默认 `8808`

### API 调用示例

**单张图片推理：**
```bash
curl -X POST http://localhost:8808/inference \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/path/to/image.jpg",
    "checkpoint": "best",
    "threshold": 0.5,
    "positive_text": "合格",
    "negative_text": "不合格"
  }'
```

**批量推理：**
```bash
curl -X POST http://localhost:8808/batch_inference \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {"image_path": "/path/to/image1.jpg", "checkpoint": "best"},
      {"image_path": "/path/to/image2.jpg", "checkpoint": "day"}
    ]
  }'
```

**响应格式：**
```json
{
  "success": true,
  "result": {
    "analysis": ["合格"],
    "result": "PASS"
  }
}
```

---

## yolo_multi_server.py

YOLO（You Only Look Once）目标检测服务，支持多权重管理和灵活的类别过滤。

### 功能特性

1. **多权重管理** - 同时挂载多个 YOLO 权重，按需选择
2. **类别过滤** - 支持通过 class_id 或 class_name 过滤检测结果
3. **可视化输出** - 可返回标注后的图片 base64 编码
4. **灵活输入** - 支持图片路径或 base64 编码输入
5. **热重载** - 支持运行时重新加载权重

### HTTP API 端点

- `GET /health` - 健康检查和服务状态
- `GET /weights` - 列出所有已加载的权重
- `POST /reload` - 重新加载所有权重
- `POST /inference` - 单张图片推理
- `POST /batch_inference` - 批量图片推理

### 启动方法

```bash
# 使用启动脚本（推荐）
bash start_yolo_multi_server.sh

# 或直接使用 Python
python yolo_multi_server.py \
    --weights_dir /path/to/weights \
    --weight bbu=/path/to/best.pt \
    --default_weight bbu \
    --device cuda:0 \
    --conf 0.35 \
    --iou 0.45 \
    --max_det 300 \
    --host 0.0.0.0 \
    --port 8810
```

### 参数说明

- `--weights_dir`: 权重目录（自动扫描 .pt/.pth 文件）
- `--weight`: 额外的权重，格式 `name=/path/to/best.pt`（可多次指定）
- `--default_weight`: 默认使用的权重名
- `--device`: 推理设备，默认 `cuda:0`
- `--conf`: 置信度阈值，默认 `0.35`
- `--iou`: NMS IoU 阈值，默认 `0.45`
- `--max_det`: 最大检测数量，默认 `300`
- `--host`: 监听地址，默认 `0.0.0.0`
- `--port`: 监听端口，默认 `8810`

### API 调用示例

**基本推理：**
```bash
curl -X POST http://localhost:8810/inference \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/path/to/image.jpg",
    "weight": "bbu",
    "conf": 0.4,
    "iou": 0.5
  }'
```

**类别过滤（通过 class_id）：**
```bash
curl -X POST http://localhost:8810/inference \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/path/to/image.jpg",
    "classes": [0, 1, 2]
  }'
```

**类别过滤（通过 class_name）：**
```bash
curl -X POST http://localhost:8810/inference \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/path/to/image.jpg",
    "label_names": ["defect", "crack"]
  }'
```

**返回可视化图片：**
```bash
curl -X POST http://localhost:8810/inference \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/path/to/image.jpg",
    "return_visual": true,
    "image_format": "JPEG",
    "image_quality": 90
  }'
```

**响应格式：**
```json
{
  "success": true,
  "result": {
    "model": "bbu",
    "num_detections": 3,
    "detections": [
      {
        "bbox": [100.5, 200.3, 300.7, 400.2],
        "class_id": 0,
        "class_name": "defect",
        "confidence": 0.85
      }
    ],
    "image_base64": "base64_encoded_image..."
  }
}
```

---

## rtdetr_multi_server.py

RT-DETR（Real-Time Detection Transformer）目标检测服务，基于 JIT 模型的高性能推理。

### 功能特性

1. **JIT 模型支持** - 使用 TorchScript JIT 模型，推理速度更快
2. **自实现后处理** - 不依赖 ultralytics，完全自主实现预处理和后处理
3. **冲突抑制** - 支持对指定类别进行冲突抑制（同位置不同类别保留高置信度）
4. **父类别过滤** - 支持根据父类别框过滤子类别检测
5. **YOLO 风格可视化** - 与 YOLO 一致的可视化风格
6. **多权重管理** - 支持同时挂载多个模型权重

### HTTP API 端点

- `GET /health` - 健康检查和服务状态
- `GET /weights` - 列出所有已加载的权重
- `POST /reload` - 重新加载所有权重
- `POST /inference` - 单张图片推理
- `POST /batch_inference` - 批量图片推理

### 启动方法

```bash
# 使用启动脚本（推荐）
bash start_rtdetr_multi_server.sh

# 或直接使用 Python
python rtdetr_multi_server.py \
    --weights_dir /path/to/weights \
    --weight rtdetr=/path/to/model.jit \
    --class_names rtdetr=/path/to/class_names.txt \
    --default_weight rtdetr \
    --device cuda:0 \
    --input_size 640 640 \
    --num_classes 80 \
    --conf 0.35 \
    --iou 0.45 \
    --max_det 300 \
    --host 0.0.0.0 \
    --port 8811
```

### 参数说明

- `--weights_dir`: 权重目录（自动扫描 .jit/.pt/.pth 文件）
- `--weight`: 额外的权重，格式 `name=/path/to/model.jit`（可多次指定）
- `--class_names`: 类别名称文件，格式 `weight_name=/path/to/class_names.txt`
- `--default_weight`: 默认使用的权重名
- `--device`: 推理设备，默认 `cuda:0`
- `--input_size`: 输入图像尺寸（H W），默认 `640 640`
- `--num_classes`: 类别数量，默认 `80`
- `--conf`: 置信度阈值，默认 `0.35`
- `--iou`: NMS IoU 阈值，默认 `0.45`
- `--max_det`: 最大检测数量，默认 `300`
- `--host`: 监听地址，默认 `0.0.0.0`
- `--port`: 监听端口，默认 `8811`

### API 调用示例

**基本推理：**
```bash
curl -X POST http://localhost:8811/inference \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/path/to/image.jpg",
    "weight": "rtdetr",
    "conf": 0.4
  }'
```

**类别过滤 + 可视化：**
```bash
curl -X POST http://localhost:8811/inference \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/path/to/image.jpg",
    "label_names": ["defect", "crack"],
    "return_visual": true,
    "class_id_to_name": {"0": "defect", "1": "crack"}
  }'
```

**冲突抑制：**
```bash
curl -X POST http://localhost:8811/inference \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/path/to/image.jpg",
    "conflict_classes": ["class_a", "class_b"],
    "conflict_iou_threshold": 0.5
  }'
```

**父类别过滤：**
```bash
curl -X POST http://localhost:8811/inference \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/path/to/image.jpg",
    "parent_class": "parent_object"
  }'
```

**响应格式：**
```json
{
  "success": true,
  "result": {
    "model": "rtdetr",
    "num_detections": 2,
    "detections": [
      {
        "bbox": [100, 200, 300, 400],
        "class_id": 0,
        "class_name": "defect",
        "confidence": 0.92
      }
    ],
    "image_base64": "base64_encoded_image..."
  }
}
```

---

## 启动脚本

每个服务都配有便捷的启动脚本，简化配置和启动流程。

### start_dino_multi_mlp_server.sh

DINO + MLP 服务启动脚本。

**主要配置项：**
```bash
DINO_MODEL_PATH="/path/to/dinov3"                    # DINO 模型路径
MLP_CHECKPOINT_DIR="/path/to/checkpoints"            # MLP checkpoint 目录
EXTRA_MLP_CHECKPOINTS=("best=/path/to/best.pth")    # 额外 checkpoint
CHECKPOINT_THRESHOLDS=("best=0.55")                  # 特定阈值
DEFAULT_CHECKPOINT="best"                            # 默认 checkpoint
DEVICE="cuda:0"                                      # 推理设备
THRESHOLD=0.5                                        # 默认阈值
PORT=8808                                            # 服务端口
RUN_BACKGROUND="n"                                   # 是否后台运行
```

**使用方法：**
```bash
# 1. 编辑脚本中的配置项
vim start_dino_multi_mlp_server.sh

# 2. 启动服务
bash start_dino_multi_mlp_server.sh

# 3. 后台运行（修改 RUN_BACKGROUND="y"）
bash start_dino_multi_mlp_server.sh
# 查看日志
tail -f dino_multi_mlp_server_*.log
```

### start_yolo_multi_server.sh

YOLO 服务启动脚本。

**主要配置项：**
```bash
WEIGHTS_DIR="/path/to/weights"                       # 权重目录
EXTRA_WEIGHTS=("bbu=/path/to/best.pt")              # 额外权重
DEFAULT_WEIGHT="best"                                # 默认权重
DEVICE="cuda:0"                                      # 推理设备
CONF=0.35                                            # 置信度阈值
IOU=0.45                                             # NMS IoU 阈值
MAX_DET=300                                          # 最大检测数
PORT=8810                                            # 服务端口
RUN_BACKGROUND="n"                                   # 是否后台运行
```

### start_rtdetr_multi_server.sh

RT-DETR 服务启动脚本。

**主要配置项：**
```bash
WEIGHTS_DIR=""                                       # 权重目录
EXTRA_WEIGHTS=("rtdetr=/path/to/model.jit")         # 额外权重
CLASS_NAMES=("rtdetr=/path/to/class_names.txt")     # 类别名称文件
DEFAULT_WEIGHT="rtdetr"                              # 默认权重
DEVICE="cuda:0"                                      # 推理设备
INPUT_SIZE="640 640"                                 # 输入尺寸
NUM_CLASSES=80                                       # 类别数
CONF=0.35                                            # 置信度阈值
IOU=0.45                                             # NMS IoU 阈值
MAX_DET=300                                          # 最大检测数
PORT=8811                                            # 服务端口
RUN_BACKGROUND="n"                                   # 是否后台运行
```

---

## 服务端口分配

| 服务 | 默认端口 | 说明 |
|------|---------|------|
| vLLM | 7878 | 视觉语言模型服务 |
| DINO + MLP | 8808 | 特征提取 + 分类服务 |
| YOLO | 8810 | 目标检测服务 |
| RT-DETR | 8811 | 目标检测服务（Transformer） |

---

## 常见使用场景

### 场景1：启动所有服务

```bash
# 1. 启动 vLLM 服务
bash vllm_serve.sh &

# 2. 启动 DINO 服务
bash start_dino_multi_mlp_server.sh &

# 3. 启动 YOLO 服务
bash start_yolo_multi_server.sh &

# 4. 启动 RT-DETR 服务
bash start_rtdetr_multi_server.sh &

# 5. 检查所有服务状态
curl http://localhost:7878/health
curl http://localhost:8808/health
curl http://localhost:8810/health
curl http://localhost:8811/health
```

### 场景2：多模型联合推理

```bash
# 1. 使用 YOLO 进行目标检测
YOLO_RESULT=$(curl -s -X POST http://localhost:8810/inference \
  -H "Content-Type: application/json" \
  -d '{"image_path": "/path/to/image.jpg", "return_visual": true}')

# 2. 提取可视化图片并送给 vLLM 分析
IMAGE_BASE64=$(echo $YOLO_RESULT | jq -r '.result.image_base64')

# 3. vLLM 分析检测结果
curl -X POST http://localhost:7878/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"Qwen3-VL-32B-Instruct\",
    \"messages\": [{
      \"role\": \"user\",
      \"content\": [
        {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/jpeg;base64,$IMAGE_BASE64\"}},
        {\"type\": \"text\", \"text\": \"请分析图片中的检测结果\"}
      ]
    }]
  }"
```

### 场景3：批量推理

```bash
# 准备批量推理数据
cat > batch_request.json <<EOF
{
  "items": [
    {"image_path": "/path/to/image1.jpg"},
    {"image_path": "/path/to/image2.jpg"},
    {"image_path": "/path/to/image3.jpg"}
  ]
}
EOF

# 批量推理
curl -X POST http://localhost:8810/batch_inference \
  -H "Content-Type: application/json" \
  -d @batch_request.json
```

---

## 性能优化建议

### GPU 内存优化

1. **vLLM 服务**：调整 `--gpu-memory-utilization` 参数（默认 0.9）
2. **目标检测服务**：根据 GPU 显存调整 `--max_det` 和批量大小
3. **多服务部署**：使用 `CUDA_VISIBLE_DEVICES` 分配不同 GPU

### 推理速度优化

1. **使用 JIT 模型**：RT-DETR 使用 JIT 模型，推理速度更快
2. **批量推理**：使用 `/batch_inference` 端点进行批量处理
3. **降低分辨率**：适当降低输入图像分辨率（如 640x640）

### 并发优化

1. **Flask 多线程**：目前服务使用 `threaded=False`，可改为 `True` 支持并发
2. **使用 Gunicorn**：生产环境建议使用 Gunicorn 部署
3. **负载均衡**：多实例部署 + Nginx 负载均衡

---

## 故障排查

### 服务无法启动

**问题：** 端口被占用
```bash
# 查看端口占用
lsof -i :8808
# 或
netstat -tulpn | grep 8808

# 解决：修改启动脚本中的 PORT 参数
```

**问题：** 模型路径不存在
```bash
# 检查路径是否正确
ls -la /path/to/model

# 解决：修改启动脚本中的路径配置
```

**问题：** GPU 内存不足
```bash
# 查看 GPU 使用情况
nvidia-smi

# 解决：
# 1. 减少 --gpu-memory-utilization
# 2. 使用更小的模型
# 3. 减少 --max-model-len 或 --max_det
```

### 推理失败

**问题：** 图片路径不存在
```bash
# 检查图片路径
ls -la /path/to/image.jpg

# 解决：使用正确的绝对路径
```

**问题：** 类别名称不匹配
```bash
# 查看可用类别
curl http://localhost:8810/health

# 解决：使用正确的类别名称或 class_id
```

---

## 依赖要求

### vLLM 服务
- Python 3.8+
- vLLM
- CUDA 11.8+
- PyTorch 2.0+

### 目标检测服务
- Python 3.8+
- PyTorch 1.13+
- torchvision
- Flask
- Pillow
- NumPy

### YOLO 服务额外依赖
- ultralytics

### DINO 服务额外依赖
- transformers

安装依赖：
```bash
pip install -r ../requirements.txt
```

---

## 开发和扩展

### 添加新的检测模型

1. 创建新的服务文件（如 `new_model_server.py`）
2. 实现以下组件：
   - 模型注册器（Registry）
   - 预处理器（PreProcessor）
   - 后处理器（PostProcessor）
   - 服务类（Service）
   - Flask API 端点
3. 创建对应的启动脚本
4. 更新本 README

### 自定义后处理逻辑

修改对应服务的 `PostProcessor` 类，实现自定义的：
- NMS 策略
- 置信度过滤
- 类别映射
- 结果格式化

---

## 技术支持

如有问题或建议，请联系开发团队。
