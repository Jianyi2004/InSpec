# Prompt Foundry - Prompt 工程系统

基于 LLM 的工业质检 Prompt 自动生成和优化系统。

## 功能概述

### 1. 新 Prompt 生成
- 用户提供简单需求描述
- 系统参考已有 prompt 生成专业的判断 prompt
- 支持用户上传示例图片和说明

### 2. Prompt 微调修改
- 用户指定修改需求
- 系统自动修改并保持专业性
- 生成修改对比

### 3. 示例管理
- 上传示例图片
- 添加示例说明
- 自动整合到 prompt 中

## 项目结构

```
prompt_foundry/
├── README.md                    # 本文档
├── DESIGN.md                    # 详细设计文档
├── requirements.txt             # 依赖
│
├── prompts/                     # Prompt 库
│   ├── index.json              # Prompt 索引
│   ├── DCDU安装/
│   │   ├── prompt.txt          # 统一格式的完整 prompt
│   │   ├── icl/                # ICL 示例图片
│   │   └── examples_metadata.json  # 示例元数据
│   ├── DCDU_输入电源/
│   └── ...
│
├── core/                        # 核心模块
│   ├── prompt_parser.py         # Prompt 解析器
│   ├── prompt_generator.py      # Prompt 生成器
│   ├── prompt_modifier.py       # Prompt 修改器
│   ├── example_manager.py       # 示例管理器 ✅
│   └── llm_client.py            # LLM 客户端
│
├── scripts/                     # 工具脚本
│   └── unify_prompts.py        # 统一 prompt 格式 ✅
│
└── web/                         # Web 界面（待开发）
    ├── app.py                   # 主应用
    ├── generator_ui.py          # 生成界面
    └── modifier_ui.py           # 修改界面
```

## 快速开始

### 1. 统一现有 Prompt 格式

```bash
cd /home/intern10/InSpec/prompt_foundry
python3 scripts/unify_prompts.py
```

这会将分散的 `system_prompt.txt`、`prompt.txt`、`summary.txt` 整合成统一格式的 `prompt.txt`。

### 2. 使用示例管理器

```python
from core.example_manager import ExampleManager

# 初始化
manager = ExampleManager()

# 添加示例
result = manager.add_example(
    task_folder="DCDU安装",
    image_path="/path/to/image.png",
    description="这是一个正确安装的示例，显示了保护罩完整覆盖..."
)

# 批量添加示例
examples = [
    {"image_path": "img1.png", "description": "示例1说明"},
    {"image_path": "img2.png", "description": "示例2说明"}
]
results = manager.add_multiple_examples("DCDU安装", examples)

# 获取格式化的 ICL 文本
icl_text = manager.format_examples_for_prompt("DCDU安装")
```

### 3. 使用 Prompt 解析器

```python
from core.prompt_parser import PromptParser

parser = PromptParser()

# 解析 prompt 文件
structure = parser.parse_file("prompts/DCDU安装/prompt.txt")

print(f"任务名称: {structure.task_name}")
print(f"System Prompt: {structure.system_prompt[:100]}...")
print(f"ICL 图片: {structure.icl_images}")
```

## Prompt 统一格式

所有 prompt 现在使用统一格式：

```
================================================================================
TASK: [任务名称]
================================================================================

[SYSTEM PROMPT]
系统提示词内容...

[MAIN PROMPT]
主要提示词内容...
=== PROMPT名字开始 ===
任务标题
=== PROMPT名字结束 ===
...
=== ICL示例开始 ===
[示例1]
图片: icl/example_1.png
说明: ...
=== ICL示例结束 ===
...

[SUMMARY PROMPT]
总结提示词内容...

[ICL IMAGES]
- icl/example_1.png
- icl/example_2.png
================================================================================
```

## 示例上传工作流

1. **用户上传图片和说明**
   - 通过 Web 界面或 API 上传
   - 提供示例说明文本

2. **系统保存示例**
   - 图片保存到 `prompts/{task}/icl/` 目录
   - 元数据保存到 `examples_metadata.json`

3. **自动更新 Prompt**
   - 将新示例整合到 prompt 的 ICL 部分
   - 更新 `[ICL IMAGES]` 列表

4. **生成新版本**
   - 保存更新后的 prompt
   - 记录版本历史

## 已完成功能

- ✅ Prompt 格式统一脚本
- ✅ 示例管理器（ExampleManager）
- ✅ Prompt 解析器（PromptParser）
- ✅ 统一格式的 prompt 文件

## 待开发功能

- ⏳ Prompt 生成器（PromptGenerator）
- ⏳ Prompt 修改器（PromptModifier）
- ⏳ LLM 客户端（LLMClient）
- ⏳ Web 界面（Gradio）
- ⏳ 版本管理系统

## 依赖

```bash
pip install gradio requests pydantic
```

## 使用场景

### 场景1: 添加新的质检任务

1. 创建任务文件夹
2. 上传示例图片和说明
3. 提供简单需求描述
4. 系统生成专业 prompt

### 场景2: 优化现有 Prompt

1. 选择要修改的 prompt
2. 描述修改需求（如"放宽判断标准"）
3. 系统自动修改并保持专业性
4. 查看修改对比并确认

### 场景3: 补充示例

1. 选择任务
2. 上传新的示例图片
3. 添加说明
4. 系统自动更新 prompt

## 技术特点

- **结构化处理**: 统一的 prompt 格式，便于解析和生成
- **示例管理**: 自动管理 ICL 示例图片和元数据
- **版本控制**: 记录每次修改的历史
- **专业性保证**: 通过 meta-prompt 让 LLM 学习现有优秀案例

## 详细文档

查看 [DESIGN.md](DESIGN.md) 了解完整的系统设计方案。
