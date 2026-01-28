# InSpec 推理引擎

工业质检智能推理引擎，基于视觉语言模型（VLM）和目标检测模型的多模态质检系统。

## 📋 目录

- [系统概述](#系统概述)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [核心组件](#核心组件)
- [使用指南](#使用指南)
- [配置说明](#配置说明)
- [开发指南](#开发指南)

---

## 系统概述

InSpec 推理引擎是一个完整的工业质检解决方案，集成了多种 AI 模型和工具，支持：

- **多模态推理** - 结合视觉语言模型（Qwen3-VL）和目标检测模型（YOLO、RT-DETR、DINO）
- **多图推理** - 支持单张图片或多张图片组合推理
- **异步批量处理** - 高效的异步并发推理能力
- **灵活的 Prompt 系统** - 可配置的系统提示和推理提示
- **完整的工具链** - 从数据构建、标注到准确率评估的全流程工具

### 核心特性

- ✅ **多模型集成** - VLM + 目标检测模型协同工作
- ✅ **两轮推理** - 支持初步分析 + 总结推理的两轮机制
- ✅ **异步高并发** - 基于 asyncio 的高性能批量推理
- ✅ **灵活配置** - 通过配置文件和脚本快速调整推理参数
- ✅ **完整工具链** - 数据构建、标注、评估一站式工具
- ✅ **服务化部署** - 所有模型均可作为独立服务部署

---

## 项目结构

```
inference_engine/
├── README.md                           # 本文档
├── requirements.txt                    # Python 依赖
│
├── vllm_client_multi_mode.py          # 多图推理客户端（同步）
├── vllm_client_multi_mode_async.py    # 多图推理客户端（异步）
│
├── server/                             # 模型服务
│   ├── README.md                       # 服务文档
│   ├── vllm_serve.sh                   # vLLM 服务启动脚本
│   ├── dino_multi_mlp_server.py        # DINO + MLP 分类服务
│   ├── yolo_multi_server.py            # YOLO 目标检测服务
│   ├── rtdetr_multi_server.py          # RT-DETR 目标检测服务
│   ├── start_dino_multi_mlp_server.sh  # DINO 服务启动脚本
│   ├── start_yolo_multi_server.sh      # YOLO 服务启动脚本
│   └── start_rtdetr_multi_server.sh    # RT-DETR 服务启动脚本
│
├── tools/                              # 工具集
│   ├── TOOLS_README.md                 # 工具文档
│   ├── bbox_utils.py                   # 边界框可视化工具
│   ├── build_multi_image_json.py       # 多图数据集构建工具
│   ├── calculate_accuracy.py           # 准确率统计工具
│   ├── filter_by_images.py             # 图片数量过滤工具
│   ├── label.py                        # 多图/单图标注工具
│   ├── simple_error_viewer_extract.py  # 错误样本查看器
│   ├── check_vllm_servers.sh           # vLLM 服务器状态检查
│   ├── run_build_image_json.sh         # 构建数据集便捷脚本
│   └── run_calculate_accuracy.sh       # 计算准确率便捷脚本
│
├── scripts/                            # 推理脚本
│   ├── run_vllm_client_multi_mode.sh                      # 同步推理脚本
│   ├── run_vllm_client_multi_mode_async.sh                # 异步推理脚本
│   ├── run_vllm_client_multi_mode_async_dcdu_install.sh   # DCDU 安装场景
│   ├── run_vllm_client_multi_mode_async_dcdu_power.sh     # DCDU 电源场景
│   ├── run_vllm_client_multi_mode_async_inside.sh         # 机柜内部场景
│   ├── run_vllm_client_multi_mode_async_outside.sh        # 机柜外部场景
│   └── ...                                                 # 其他场景脚本
│
├── prompts/                            # Prompt 配置
│   ├── system_prompts/                 # 系统提示
│   ├── task_prompts/                   # 任务提示
│   └── summary_prompts/                # 总结提示
│
└── data/                               # 数据目录
    ├── raw/                            # 原始数据
    ├── processed/                      # 处理后数据
    └── results/                        # 推理结果
```

---

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
cd /path/to/InSpec/inference_engine

# 安装依赖
pip install -r requirements.txt

# 检查 CUDA 环境
nvidia-smi
```

### 2. 启动服务

```bash
# 启动 vLLM 服务（视觉语言模型）
cd server
bash vllm_serve.sh

# 启动目标检测服务（可选）
bash start_yolo_multi_server.sh      # YOLO 服务
bash start_rtdetr_multi_server.sh    # RT-DETR 服务
bash start_dino_multi_mlp_server.sh  # DINO 服务

# 检查服务状态
cd ../tools
bash check_vllm_servers.sh
```

### 3. 准备数据

```bash
cd tools

# 方式1：使用关键字搜索构建数据集
bash run_build_image_json.sh \
    --search-dir /data/raw_dataset \
    --keyword "DCDU,BBU" \
    -o ../data/test_data.json \
    --mode multi

# 方式2：手动指定目录（编辑脚本后运行）
vim run_build_image_json.sh
bash run_build_image_json.sh
```

### 4. 运行推理

```bash
cd ../scripts

# 编辑推理脚本配置
vim run_vllm_client_multi_mode_async.sh

# 运行异步推理
bash run_vllm_client_multi_mode_async.sh
```

### 5. 评估结果

```bash
cd ../tools

# 计算准确率
bash run_calculate_accuracy.sh

# 查看错误样本
python simple_error_viewer_extract.py
```

---

## 核心组件

### 1. 推理客户端

#### vllm_client_multi_mode.py

同步多图推理客户端，支持：
- 单图推理（single mode）
- 多图推理（multi mode）
- 两轮推理（初步分析 + 总结）
- 目标检测模型集成（YOLO/RT-DETR）

**使用示例：**
```python
from vllm_client_multi_mode import VLLMClient

client = VLLMClient(
    server_url="http://localhost:7878",
    yolo_server_url="http://localhost:8810"
)

result = client.inference_single(
    image_paths=["/path/to/image1.jpg", "/path/to/image2.jpg"],
    prompt_file="prompts/task_prompts/dcdu_install.txt",
    temperature=0.1,
    max_tokens=2048,
    model="Qwen3-VL-32B-Instruct"
)
```

#### vllm_client_multi_mode_async.py

异步多图推理客户端，支持：
- 高并发批量推理
- 进度条显示
- 错误重试机制
- 结果自动保存

**使用示例：**
```bash
python vllm_client_multi_mode_async.py \
    --input data/test_data.json \
    --output results/output.json \
    --server-url http://localhost:7878 \
    --prompt-file prompts/task_prompts/dcdu_install.txt \
    --concurrency 10 \
    --mode multi
```

### 2. 模型服务

详见 [`server/README.md`](server/README.md)

- **vLLM 服务** (端口 7878) - Qwen3-VL-32B-Instruct 视觉语言模型
- **YOLO 服务** (端口 8810) - YOLOv10 目标检测
- **RT-DETR 服务** (端口 8811) - RT-DETR 目标检测
- **DINO 服务** (端口 8808) - DINO + MLP 分类

### 3. 工具集

详见 [`tools/TOOLS_README.md`](tools/TOOLS_README.md)

**Python 工具：**
- `build_multi_image_json.py` - 构建多图数据集
- `calculate_accuracy.py` - 计算准确率
- `filter_by_images.py` - 过滤图片数量
- `label.py` - 数据标注工具（Gradio Web 界面）
- `simple_error_viewer_extract.py` - 错误样本查看器

**Shell 脚本：**
- `check_vllm_servers.sh` - 检查 vLLM 服务状态
- `run_build_image_json.sh` - 构建数据集便捷脚本
- `run_calculate_accuracy.sh` - 计算准确率便捷脚本

---

## 使用指南

### 场景1：DCDU 安装质检

```bash
# 1. 准备数据
cd tools
bash run_build_image_json.sh \
    --search-dir /data/raw_dataset \
    --keyword "DCDU安装" \
    -o ../data/dcdu_install.json

# 2. 运行推理
cd ../scripts
bash run_vllm_client_multi_mode_async_dcdu_install.sh

# 3. 评估结果
cd ../tools
bash run_calculate_accuracy.sh
```

### 场景2：机柜外部质检

```bash
# 1. 准备数据
cd tools
bash run_build_image_json.sh \
    --search-dir /data/raw_dataset \
    --keyword "机柜外部,防水,接地" \
    -o ../data/outside.json

# 2. 运行推理
cd ../scripts
bash run_vllm_client_multi_mode_async_outside.sh

# 3. 查看错误样本
cd ../tools
python simple_error_viewer_extract.py
```

### 场景3：批量标注

```bash
# 1. 启动标注工具
cd tools
python label.py

# 2. 在浏览器中打开 http://localhost:7860
# 3. 上传 JSON 文件
# 4. 选择标注模式（multi/single）
# 5. 开始标注
# 6. 保存结果
```

### 场景4：模型评估

```bash
# 1. 检查服务状态
cd tools
bash check_vllm_servers.sh

# 2. 运行推理
cd ../scripts
bash run_vllm_client_multi_mode_async.sh

# 3. 计算准确率
cd ../tools
bash run_calculate_accuracy.sh

# 4. 查看详细报告
python simple_error_viewer_extract.py
```

---

## 配置说明

### Prompt 配置

Prompt 文件位于 `prompts/` 目录，分为三类：

**1. 系统提示（system_prompts/）**
- 定义模型的角色和行为规范
- 示例：`system_prompt_v5_2.txt`

**2. 任务提示（task_prompts/）**
- 定义具体的质检任务和判断标准
- 示例：`dcdu_install_prompt.txt`、`outside_prompt.txt`

**3. 总结提示（summary_prompts/）**
- 定义二轮推理的总结逻辑
- 示例：`summary_v3_2.txt`

### 推理参数

在推理脚本中可配置以下参数：

```bash
# 服务地址
SERVER_URL="http://localhost:7878"
YOLO_SERVER_URL="http://localhost:8810"
RTDETR_SERVER_URL="http://localhost:8811"

# 推理参数
TEMPERATURE=0.1          # 温度参数（0-1，越小越确定）
MAX_TOKENS=2048          # 最大生成 token 数
CONCURRENCY=10           # 并发数

# 推理模式
MODE="multi"             # single: 逐图推理, multi: 多图推理

# Prompt 配置
PROMPT_FILE="prompts/task_prompts/dcdu_install.txt"
SYSTEM_PROMPT="prompts/system_prompts/system_prompt_v5_2.txt"
SUMMARY_PROMPT="prompts/summary_prompts/summary_v3_2.txt"
```

### 数据格式

**输入数据格式：**
```json
[
  {
    "messages": [
      {"content": "<image>", "role": "user"},
      {"content": "PASS", "role": "assistant"}
    ],
    "images": [
      "/path/to/image1.jpg",
      "/path/to/image2.jpg"
    ]
  }
]
```

**输出结果格式：**
```json
[
  {
    "sample_index": 0,
    "image_paths": ["/path/to/image1.jpg", "/path/to/image2.jpg"],
    "prediction": "PASS",
    "true_label": "PASS",
    "correct": true,
    "raw_response": "{\"analysis\": [...], \"result\": \"PASS\"}",
    "summary_response": "综合分析结果...",
    "per_image_details": [
      {
        "image_path": "/path/to/image1.jpg",
        "image_label": "PASS",
        "prediction": "PASS"
      }
    ]
  }
]
```

---

## 开发指南

### 添加新的质检场景

1. **创建 Prompt 文件**
```bash
# 创建任务提示
vim prompts/task_prompts/new_scenario.txt

# 内容示例：
# 你是一个工业质检专家，负责检查...
# 判断标准：
# 1. ...
# 2. ...
```

2. **创建推理脚本**
```bash
# 复制现有脚本
cp scripts/run_vllm_client_multi_mode_async.sh \
   scripts/run_vllm_client_multi_mode_async_new_scenario.sh

# 编辑配置
vim scripts/run_vllm_client_multi_mode_async_new_scenario.sh
```

3. **准备数据**
```bash
cd tools
bash run_build_image_json.sh \
    --search-dir /data/raw_dataset \
    --keyword "新场景关键字" \
    -o ../data/new_scenario.json
```

4. **运行推理**
```bash
cd ../scripts
bash run_vllm_client_multi_mode_async_new_scenario.sh
```

### 集成新的检测模型

1. **创建服务文件**
```python
# server/new_model_server.py
class NewModelService:
    def __init__(self, weights, device):
        # 初始化模型
        pass
    
    def infer(self, payload):
        # 推理逻辑
        pass
```

2. **创建启动脚本**
```bash
# server/start_new_model_server.sh
python new_model_server.py \
    --weights /path/to/weights \
    --device cuda:0 \
    --port 8812
```

3. **在客户端中集成**
```python
# vllm_client_multi_mode.py
def __init__(self, ..., new_model_server_url=None):
    self.new_model_server_url = new_model_server_url
```

### 自定义后处理逻辑

编辑 `vllm_client_multi_mode.py` 中的后处理函数：

```python
def extract_json_payload(raw_response: str) -> dict:
    """自定义 JSON 提取逻辑"""
    # 实现自定义逻辑
    pass

def normalize_result(result: str) -> str:
    """自定义结果标准化逻辑"""
    # 实现自定义逻辑
    pass
```

---

## 性能优化

### 推理性能

1. **调整并发数**
```bash
# 根据 GPU 显存调整并发数
CONCURRENCY=10  # 32GB 显存推荐 10-20
```

2. **使用批量推理**
```bash
# 使用异步客户端进行批量推理
python vllm_client_multi_mode_async.py \
    --concurrency 20 \
    --batch-size 100
```

3. **优化 Prompt**
```bash
# 减少 Prompt 长度
# 使用更精简的系统提示
# 减少示例数量
```

### 服务性能

1. **vLLM 服务优化**
```bash
# 调整 GPU 内存利用率
--gpu-memory-utilization 0.95

# 调整最大模型长度
--max-model-len 32768

# 增加 Tensor Parallel 大小
--tensor-parallel-size 4
```

2. **目标检测服务优化**
```bash
# 降低置信度阈值
--conf 0.25

# 减少最大检测数
--max_det 100
```

---

## 常见问题

### Q1: vLLM 服务启动失败

**问题：** GPU 内存不足

**解决：**
```bash
# 减少 GPU 内存利用率
--gpu-memory-utilization 0.8

# 减少最大模型长度
--max-model-len 16384

# 使用更多 GPU
--tensor-parallel-size 4
```

### Q2: 推理速度慢

**问题：** 并发数设置不合理

**解决：**
```bash
# 增加并发数（根据 GPU 显存调整）
CONCURRENCY=20

# 使用异步客户端
python vllm_client_multi_mode_async.py
```

### Q3: 准确率低

**问题：** Prompt 不够精确

**解决：**
1. 优化系统提示，明确角色定位
2. 细化任务提示，增加判断标准
3. 添加更多示例
4. 调整温度参数（降低到 0.05-0.1）

### Q4: 端口冲突

**问题：** 服务端口被占用

**解决：**
```bash
# 查看端口占用
lsof -i :7878

# 修改服务端口
vim server/vllm_serve.sh
# 修改 --port 参数
```

---

## 依赖要求

### 硬件要求

- **GPU**: NVIDIA GPU with CUDA 12.x
  - 推荐：A100 (80GB) / H100
  - 最低：RTX 3090 (24GB)
- **内存**: 64GB+ RAM
- **存储**: 500GB+ SSD

### 软件要求

- **操作系统**: Linux (Ubuntu 20.04+)
- **Python**: 3.8+
- **CUDA**: 12.0+
- **PyTorch**: 2.0+

### Python 依赖

详见 `requirements.txt`，主要包括：
- vllm==0.12.0
- transformers==4.57.3
- torch==2.9.0
- ultralytics==8.3.248
- gradio==6.0.2
- flask==3.1.2

安装依赖：
```bash
pip install -r requirements.txt
```

---

## 许可证

本项目仅供内部使用。

---

## 联系方式

如有问题或建议，请联系开发团队。

---

## 更新日志

### v1.0.0 (2026-01-28)
- ✅ 初始版本发布
- ✅ 支持多图推理
- ✅ 集成 YOLO/RT-DETR/DINO 模型
- ✅ 完整的工具链
- ✅ 异步批量推理
- ✅ Gradio 标注工具
