# Tools 工具集

本目录包含用于工业质检推理引擎的各种辅助工具，涵盖数据构建、准确率计算、数据过滤、标注和可视化等功能。

## 目录

### Python 工具
- [bbox_utils.py](#bbox_utilspy) - 边界框可视化工具
- [build_multi_image_json.py](#build_multi_image_jsonpy) - 多图数据集构建工具
- [calculate_accuracy.py](#calculate_accuracypy) - 准确率统计工具
- [filter_by_images.py](#filter_by_imagespy) - 图片数量过滤工具
- [label.py](#labelpy) - 多图/单图标注工具
- [simple_error_viewer_extract.py](#simple_error_viewer_extractpy) - 错误样本查看器

### Shell 脚本工具
- [check_vllm_servers.sh](#check_vllm_serverssh) - vLLM 服务器状态检查脚本
- [run_build_image_json.sh](#run_build_image_jsonsh) - 构建图片 JSON 的便捷脚本
- [run_calculate_accuracy.sh](#run_calculate_accuracysh) - 计算准确率的便捷脚本

---

## bbox_utils.py

边界框可视化工具，用于在图片上绘制标注框（矩形/多边形）并支持中文标签显示。

### 主要功能

1. **中文文本绘制** - 支持在图片上绘制中文标签
2. **边界框可视化** - 绘制矩形框和多边形框
3. **缓存支持** - 可将绘制结果保存到缓存目录

### 核心函数

#### `draw_chinese_text(img, text, position, font_size=20, color=(255, 255, 255), bg_color=(255, 0, 0))`

在图片上绘制中文文本。

**参数：**
- `img`: numpy数组格式的图片
- `text`: 要绘制的文本
- `position`: 文本位置 (x, y)
- `font_size`: 字体大小，默认20
- `color`: 文字颜色 (R, G, B)，默认白色
- `bg_color`: 背景颜色 (R, G, B)，默认红色

**返回：** 绘制文本后的图片

#### `draw_bounding_boxes_with_cache(img_path, json_path, draw_text=False, save_to_cache=True)`

在图片上绘制边界框，支持缓存到临时文件夹。

**参数：**
- `img_path`: 图片路径
- `json_path`: JSON标注文件路径
- `draw_text`: 是否绘制标签文本，默认False
- `save_to_cache`: 是否保存到缓存文件夹，默认True

**返回：**
- 如果 `save_to_cache=True`: 返回缓存图片的路径
- 如果 `save_to_cache=False`: 返回带有边界框的图片数组

### 使用示例

```python
from bbox_utils import draw_bounding_boxes_with_cache

# 绘制边界框并保存到缓存
cache_path = draw_bounding_boxes_with_cache(
    img_path="/path/to/image.jpg",
    json_path="/path/to/annotation.json",
    draw_text=True,
    save_to_cache=True
)

# 或直接获取图片数组
img_array = draw_bounding_boxes_with_cache(
    img_path="/path/to/image.jpg",
    json_path="/path/to/annotation.json",
    save_to_cache=False
)
```

---

## build_multi_image_json.py

遍历指定目录，将最末级文件夹内的图片打包为统一的JSON结构，支持单图模式和多图模式。

### 主要功能

1. **多目录支持** - 可同时处理多个根目录
2. **双模式** - 支持单图模式（每张图一个样本）和多图模式（每个文件夹一个样本）
3. **自动统计** - 生成详细的数据说明文件

### 使用方法

```bash
# 多图模式（默认）：每个文件夹生成一个样本
python build_multi_image_json.py \
    --root_dir /path/to/data1 /path/to/data2 \
    -o output.json \
    --mode multi

# 单图模式：每张图片生成一个样本
python build_multi_image_json.py \
    --root_dir /path/to/data \
    -o output.json \
    --mode single

# 自定义说明文件路径
python build_multi_image_json.py \
    --root_dir /path/to/data \
    -o output.json \
    --summary custom_summary.txt
```

### 参数说明

- `--root_dir`: 根目录路径（可传入多个路径，用空格分隔）
- `-o, --output`: 输出JSON文件路径（必需）
- `--summary`: 生成数据说明文件路径（可选，默认为输出文件名_summary.txt）
- `-m, --mode`: 数据模式，可选 `single` 或 `multi`（默认multi）

### 输出格式

**多图模式：**
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

**单图模式：**
```json
[
  {
    "messages": [
      {"content": "<image>", "role": "user"},
      {"content": "PASS", "role": "assistant"}
    ],
    "images": ["/path/to/image1.jpg"]
  },
  {
    "messages": [
      {"content": "<image>", "role": "user"},
      {"content": "PASS", "role": "assistant"}
    ],
    "images": ["/path/to/image2.jpg"]
  }
]
```

---

## calculate_accuracy.py

准确率统计工具，对比标注数据和推理结果，计算单张图和按组的准确率。

### 主要功能

1. **双层准确率计算** - 计算按组准确率和单张图准确率
2. **标签更新** - 自动更新推理结果中的真实标签
3. **详细报告** - 生成包含错误样本的详细报告
4. **分类统计** - 按标签类别统计准确率

### 使用方法

```bash
# 基本用法：计算准确率
python calculate_accuracy.py \
    --annotation /path/to/annotation.json \
    --inference /path/to/inference.json

# 保存详细报告
python calculate_accuracy.py \
    --annotation /path/to/annotation.json \
    --inference /path/to/inference.json \
    --output report.json

# 保存更新后的推理结果
python calculate_accuracy.py \
    --annotation /path/to/annotation.json \
    --inference /path/to/inference.json \
    --update-inference inference_updated.json

# 不更新推理结果中的true_label
python calculate_accuracy.py \
    --annotation /path/to/annotation.json \
    --inference /path/to/inference.json \
    --no-update
```

### 参数说明

- `--annotation`: 标注数据JSON文件路径（必需）
- `--inference`: 推理结果JSON文件路径（必需）
- `--output`: 详细报告输出路径（可选）
- `--update-inference`: 更新后的推理结果输出路径（可选，默认为原文件名_updated.json）
- `--no-update`: 不更新推理结果中的true_label

### 输出示例

```
================================================================================
准确率统计报告
================================================================================

【按组统计】
总样本数: 100
正确数量: 85
准确率: 85.00%

各类别准确率:
  PASS           :  70/ 80 =  87.50%
  FAIL           :  15/ 20 =  75.00%

【单张图片统计】
总图片数: 300
正确数量: 270
准确率: 90.00%

各类别准确率:
  PASS           : 200/220 =  90.91%
  FAIL           :  50/ 60 =  83.33%
  NOT_INVOLVED   :  20/ 20 = 100.00%
```

---

## filter_by_images.py

根据图片数量过滤JSON数据集中的样本。

### 主要功能

1. **灵活过滤** - 支持最小值和最大值过滤
2. **自动识别** - 自动识别JSON结构（顶层list或dict）
3. **统计输出** - 显示保留和丢弃的样本数量

### 使用方法

```bash
# 过滤图片数量在1-6之间的样本
python filter_by_images.py \
    --in input.json \
    --out output.json \
    --min_images 1 \
    --max_images 6

# 只设置最大值
python filter_by_images.py \
    --in input.json \
    --out output.json \
    --max_images 10

# 指定JSON中的样本容器key
python filter_by_images.py \
    --in input.json \
    --out output.json \
    --max_images 6 \
    --key data
```

### 参数说明

- `--in`: 输入JSON文件路径（必需）
- `--out`: 输出JSON文件路径（必需）
- `--max_images`: 每个样本允许的最大图片数，默认6
- `--min_images`: 每个样本要求的最小图片数，默认1
- `--key`: 如果顶层JSON是dict，指定哪个key包含样本列表（如data/samples/items）

### 输出示例

```
Done. min_images=1, max_images=6
Kept: 850
Dropped: 150
```

---

## label.py

多图/单图标注工具，使用Gradio框架提供Web界面进行数据标注。

### 主要功能

1. **双模式标注** - 支持多图逻辑（整组标注）和单图逻辑（逐图标注+合并）
2. **可视化界面** - 基于Gradio的友好Web界面
3. **实时保存** - 支持实时保存标注结果
4. **样本抽取** - 支持抽取特定样本到新文件
5. **自动重载** - 支持监控JSON文件变化并自动重载

### 使用方法

```bash
# 启动标注工具（默认端口7860）
python label.py

# 指定端口
python label.py --port 8080

# 指定服务器地址
python label.py --server_name 0.0.0.0 --port 7860
```

### 参数说明

- `--port`: 服务器端口，默认7860
- `--server_name`: 服务器地址，默认127.0.0.1

### 标注模式

**多图模式（multi）：**
- 对整组图片进行统一标注
- 适用于需要综合判断多张图片的场景

**单图模式（single）：**
- 逐张图片进行标注
- 自动合并逻辑：有FAIL→FAIL；无FAIL且有PASS→PASS；全NOT_INVOLVED→NOT_INVOLVED

### 标签类型

- `PASS`: 合格
- `FAIL`: 不合格
- `NOT_INVOLVED`: 不相关/无关

### 输出格式

```json
[
  {
    "messages": [
      {"content": "<image>", "role": "user"},
      {
        "content": "{\"analysis\": [\"image1:PASS\", \"image2:FAIL\"], \"result\": \"FAIL\"}",
        "role": "assistant"
      }
    ],
    "images": ["/path/to/image1.jpg", "/path/to/image2.jpg"]
  }
]
```

### 使用流程

1. 启动工具：`python label.py`
2. 在浏览器中打开显示的URL（如 http://127.0.0.1:7860）
3. 上传JSON文件
4. 选择标注模式（multi或single）
5. 浏览样本并进行标注
6. 保存标注结果

---

## simple_error_viewer_extract.py

错误样本查看器，用于查看和分析推理结果中的错误样本。

### 主要功能

1. **多格式支持** - 支持错误分析格式、vLLM格式和新格式数据
2. **样本浏览** - 提供友好的样本浏览界面
3. **样本抽取** - 支持抽取特定样本到新文件
4. **实时监控** - 支持监控JSON文件变化并自动重载
5. **多图支持** - 支持查看多图样本
6. **正确性判断** - 支持对样本进行正确性判断

### 使用方法

```bash
# 启动查看器（默认端口7860）
python simple_error_viewer_extract.py

# 指定端口
python simple_error_viewer_extract.py --port 8080

# 指定服务器地址
python simple_error_viewer_extract.py --server_name 0.0.0.0 --port 7860
```

### 参数说明

- `--port`: 服务器端口，默认7860
- `--server_name`: 服务器地址，默认127.0.0.1

### 支持的数据格式

**1. 错误分析格式：**
```json
{
  "error_samples": [
    {
      "sample_index": 0,
      "error_type": "假阳性",
      "model_prediction": "FAIL",
      "true_label": "PASS",
      ...
    }
  ]
}
```

**2. vLLM格式：**
```json
[
  {
    "prediction": "PASS",
    "true_label": "FAIL",
    "image_paths": ["/path/to/image.jpg"],
    "raw_response": "{...}",
    ...
  }
]
```

**3. 新格式：**
```json
[
  {
    "messages": [...],
    "images": ["/path/to/image.jpg"],
    ...
  }
]
```

### 功能特性

- **样本导航**：上一个/下一个样本
- **多图浏览**：对于多图样本，可切换查看不同图片
- **样本抽取**：选择特定样本抽取到新文件
- **过滤功能**：按错误类型、标签等过滤样本
- **实时监控**：自动检测文件变化并重载

### 使用流程

1. 启动工具：`python simple_error_viewer_extract.py`
2. 在浏览器中打开显示的URL
3. 上传JSON文件
4. 浏览样本并进行分析
5. 可选：抽取特定样本到新文件

---

## check_vllm_servers.sh

vLLM 服务器状态检查脚本，用于快速查看系统中运行的所有 vLLM 服务器状态。

### 主要功能

1. **自动发现** - 自动发现所有运行中的 vLLM 服务器进程
2. **状态汇总** - 显示端口、用户、模型路径、GPU 使用等信息
3. **连接监控** - 显示每个服务的活跃连接数
4. **详细信息** - 提供 GPU 内存使用、运行时长等详细信息

### 使用方法

```bash
# 直接运行
bash check_vllm_servers.sh

# 或赋予执行权限后运行
chmod +x check_vllm_servers.sh
./check_vllm_servers.sh
```

### 输出示例

```
==========================================
VLLM Server 状态汇总
==========================================

端口   用户               模型路径                                      GPU             TP大小   连接数    
--------------------------------------------------------------------------------------------------------
7878   intern10           Qwen3-VL-32B-Instruct                        2,3             2        3         

==========================================
详细信息：
==========================================

端口 7878 (intern10):
  模型: /data_all/share/models/Qwen3-VL-32B-Instruct
  GPU内存: 45.2 GB
  连接状态: 2 活跃 / 3 总计
  运行时长: 2-14:35:22

==========================================
提示: 使用 'ss -tn | grep :<端口>' 查看具体连接详情
==========================================
```

### 显示信息说明

- **端口**: vLLM 服务监听的端口号
- **用户**: 运行服务的用户名
- **模型路径**: 加载的模型路径（简化显示）
- **GPU**: 使用的 GPU 编号
- **TP大小**: Tensor Parallel 大小
- **连接数**: 当前活跃的网络连接数
- **GPU内存**: 服务占用的 GPU 显存总量
- **运行时长**: 服务已运行的时间

### 使用场景

- 检查 vLLM 服务是否正常运行
- 查看服务占用的端口和 GPU
- 监控服务的连接状态
- 排查端口冲突问题
- 查看多个服务的资源使用情况

---

## run_build_image_json.sh

构建多图推理 JSON 数据集的便捷脚本，封装了 `build_multi_image_json.py` 的常用功能。

### 主要功能

1. **三种使用模式** - 手动模式、关键字搜索模式、混合模式
2. **关键字搜索** - 自动搜索包含指定关键字的文件夹
3. **多关键字支持** - 支持多个关键字（逗号或空格分隔）
4. **自动去重** - 搜索结果自动去重
5. **灵活配置** - 支持命令行参数和脚本内配置

### 使用方法

**方式1：手动模式**
```bash
# 编辑脚本中的 MANUAL_DIRS 数组
vim run_build_image_json.sh

# 运行脚本
bash run_build_image_json.sh
```

**方式2：关键字搜索模式**
```bash
# 单个关键字
bash run_build_image_json.sh \
    --search-dir /data/raw_dataset \
    --keyword "DCDU" \
    -o data/dcdu.json \
    --mode multi

# 多个关键字（逗号分隔）
bash run_build_image_json.sh \
    --search-dir /data \
    --keyword "BBU安装,防水" \
    -o data/output.json

# 多个关键字（空格分隔）
bash run_build_image_json.sh \
    --search-dir /data \
    --keyword "BBU安装 防水" \
    -o data/output.json
```

**方式3：混合模式**
```bash
# 同时使用手动指定和关键字搜索
# 编辑脚本中的 MANUAL_DIRS，然后运行：
bash run_build_image_json.sh \
    --search-dir /data \
    --keyword "DCDU" \
    -o data/output.json
```

### 参数说明

- `--search-dir`: 要搜索的父目录
- `--keyword`: 文件夹名称中包含的关键字（支持多个，用逗号或空格分隔）
- `-o, --output`: 输出 JSON 文件路径
- `--summary`: 输出摘要文件路径（可选）
- `--mode`: 处理模式，`single` 或 `multi`（默认 `multi`）
- `-h, --help`: 显示帮助信息

### 脚本内配置

编辑脚本可修改以下默认配置：

```bash
# 手动指定的目录列表
MANUAL_DIRS=(
    /path/to/dir1
    /path/to/dir2
)

# 默认参数
DEFAULT_OUTPUT="data/output.json"
DEFAULT_SUMMARY="data/summary.txt"
DEFAULT_MODE="multi"
```

### 输出示例

```
正在搜索目录: /data/raw_dataset
关键字: DCDU,BBU
-----------------------------------
解析到 2 个关键字: DCDU BBU

搜索关键字: 'DCDU'
  找到: /data/raw_dataset/DCDU_install/pass
  找到: /data/raw_dataset/DCDU_power/fail
  关键字 'DCDU' 找到 2 个目录

搜索关键字: 'BBU'
  找到: /data/raw_dataset/BBU_install/pass
  关键字 'BBU' 找到 1 个目录

-----------------------------------
共找到 3 个匹配的目录（已去重）

执行命令：
python build_multi_image_json.py --root_dir "/data/raw_dataset/DCDU_install/pass" "/data/raw_dataset/DCDU_power/fail" "/data/raw_dataset/BBU_install/pass" -o "data/output.json" --mode "multi"

✓ 共生成 150 条记录，包含 450 张图片
✓ JSON 数据已保存至: data/output.json
✓ 数据说明已保存至: data/output_summary.txt
```

### 使用场景

- 快速构建特定场景的数据集（如 "DCDU安装"、"防水检查"）
- 从大型数据集中提取特定类别的数据
- 批量处理多个相关文件夹
- 自动化数据集构建流程

---

## run_calculate_accuracy.sh

计算准确率的便捷脚本，封装了 `calculate_accuracy.py` 的常用功能。

### 主要功能

1. **快速配置** - 通过编辑脚本快速配置文件路径
2. **一键运行** - 简化准确率计算流程
3. **自动更新** - 自动更新推理结果中的真实标签
4. **清晰输出** - 显示配置信息和执行状态

### 使用方法

```bash
# 1. 编辑脚本中的文件路径
vim run_calculate_accuracy.sh

# 2. 运行脚本
bash run_calculate_accuracy.sh
```

### 脚本配置

编辑脚本可修改以下配置：

```bash
# 标注数据路径
ANNOTATION_FILE="data/6-outside/output_annotated.json"

# 推理结果路径
INFERENCE_FILE="server_inference_results/outside/detailed_results.json"

# 输出报告路径
OUTPUT_FILE="accuracy_report.json"

# 更新后的推理结果路径
UPDATED_INFERENCE_FILE="detailed_results_updated.json"
```

### 输出示例

```
📊 准确率统计工具
===============================================
标注数据: data/6-outside/output_annotated.json
推理结果: server_inference_results/outside/detailed_results.json
输出报告: accuracy_report.json
更新后推理结果: detailed_results_updated.json
===============================================

✅ 加载标注数据: 100 个样本
✅ 加载推理结果: 100 个样本
✅ 成功匹配 100 个样本

================================================================================
📊 准确率统计结果
================================================================================

【按组统计】
总样本数: 100
正确数量: 85
准确率: 85.00%

各类别准确率:
  PASS           :  70/ 80 =  87.50%
  FAIL           :  15/ 20 =  75.00%

【单张图片统计】
总图片数: 300
正确数量: 270
准确率: 90.00%

各类别准确率:
  PASS           : 200/220 =  90.91%
  FAIL           :  50/ 60 =  83.33%
  NOT_INVOLVED   :  20/ 20 = 100.00%

✅ 已更新 100 个样本的 true_label
💾 详细报告已保存到: accuracy_report.json
💾 更新后的推理结果已保存到: detailed_results_updated.json
```

### 使用场景

- 快速评估模型性能
- 批量处理多个推理结果
- 自动化准确率统计流程
- 生成标准化的评估报告

---

## 常见使用场景

### 场景1：构建训练数据集

```bash
# 1. 使用关键字搜索构建多图数据集
bash run_build_image_json.sh \
    --search-dir /data/raw_dataset \
    --keyword "BBU安装,防水检查" \
    -o dataset.json \
    --mode multi

# 2. 过滤图片数量
python filter_by_images.py \
    --in dataset.json \
    --out dataset_filtered.json \
    --min_images 2 \
    --max_images 6

# 3. 使用标注工具进行标注
python label.py
```

### 场景2：评估模型性能

```bash
# 1. 检查 vLLM 服务状态
bash check_vllm_servers.sh

# 2. 计算准确率并生成报告（使用便捷脚本）
bash run_calculate_accuracy.sh

# 3. 查看错误样本
python simple_error_viewer_extract.py
# 然后在Web界面中加载更新后的推理结果
```

### 场景3：数据清洗和标注

```bash
# 1. 查看推理结果
python simple_error_viewer_extract.py

# 2. 在Web界面中抽取需要重新标注的样本

# 3. 使用标注工具进行标注
python label.py
```

### 场景4：完整的模型评估工作流

```bash
# 1. 检查服务状态
bash check_vllm_servers.sh

# 2. 使用关键字搜索构建测试数据集
bash run_build_image_json.sh \
    --search-dir /data/test_dataset \
    --keyword "DCDU,BBU,RRU" \
    -o test_data.json

# 3. 运行推理（假设已有推理脚本）
# python run_inference.py --input test_data.json --output results.json

# 4. 计算准确率
bash run_calculate_accuracy.sh

# 5. 查看错误样本
python simple_error_viewer_extract.py

# 6. 标注错误样本
python label.py
```

---

## 依赖要求

所有工具的依赖已在 `requirements.txt` 中定义，主要包括：

- Python 3.7+
- OpenCV (cv2)
- Pillow (PIL)
- NumPy
- Gradio（用于label.py和simple_error_viewer_extract.py）

安装依赖：
```bash
pip install -r ../requirements.txt
```

---

## 注意事项

1. **路径问题**：所有工具都支持绝对路径和相对路径，建议使用绝对路径以避免问题
2. **中文支持**：bbox_utils.py 会自动寻找系统中的中文字体，如果找不到会使用默认字体
3. **内存占用**：处理大量图片时注意内存占用，可以分批处理
4. **数据备份**：在使用过滤和标注工具前，建议备份原始数据
5. **端口冲突**：启动Gradio工具时，如果端口被占用，可以使用 `--port` 参数指定其他端口

---

## 技术支持

如有问题或建议，请联系开发团队。
