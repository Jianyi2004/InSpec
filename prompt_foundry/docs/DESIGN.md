# Prompt Foundry 系统设计方案

## 一、需求分析

### 核心需求
1. **新 Prompt 生成**：用户提供简单需求描述 → 系统参考已有 prompt 生成专业的判断 prompt
2. **Prompt 微调修改**：用户指定修改点 → 系统自动修改并保持专业性

### 现有 Prompt 结构分析

通过分析 `prompts/` 目录下的现有 prompt，发现标准结构：

```
=== PROMPT名字开始 ===
[任务名称]
=== PROMPT名字结束 ===

[角色定位与任务描述]

# 核心概念
[关键术语定义]

=== ICL示例开始 ===
[示例1] 图片: [路径] 说明: [描述]
[示例2] ...
=== ICL示例结束 ===

# 判断流程
## 第一步: [步骤名]
[详细规则]
## 第二步: [步骤名]
...

# 输出格式
[JSON 格式要求]
```

**关键特点：**
- 结构化强：有明确的分隔符和章节
- 专业性高：术语精确、逻辑严密
- 规则详细：包含大量边界情况处理
- ICL 示例：配合图片的实例说明
- 多层判断：前置判断 → 分步检查 → 结果输出

---

## 二、系统架构设计

### 2.1 整体架构

```
prompt_foundry/
├── DESIGN.md                    # 本设计文档
├── README.md                    # 使用说明
├── requirements.txt             # 依赖
│
├── prompts/                     # Prompt 库（已有）
│   ├── DCDU安装/
│   ├── DCDU_输入电源/
│   ├── GPS避雷器安装/
│   └── ...
│
├── core/                        # 核心模块
│   ├── __init__.py
│   ├── prompt_parser.py         # Prompt 解析器
│   ├── prompt_generator.py      # Prompt 生成器
│   ├── prompt_modifier.py       # Prompt 修改器
│   ├── template_manager.py      # 模板管理器
│   └── llm_client.py            # LLM 客户端（调用 VLM）
│
├── templates/                   # Prompt 模板
│   ├── base_template.txt        # 基础模板
│   ├── generation_prompt.txt    # 生成任务的 meta-prompt
│   └── modification_prompt.txt  # 修改任务的 meta-prompt
│
├── utils/                       # 工具函数
│   ├── __init__.py
│   ├── text_utils.py            # 文本处理工具
│   └── validation.py            # Prompt 验证工具
│
├── web/                         # Web 界面（Gradio）
│   ├── __init__.py
│   ├── app.py                   # 主应用
│   ├── generator_ui.py          # 生成界面
│   └── modifier_ui.py           # 修改界面
│
└── examples/                    # 示例和测试
    ├── example_requirements.json
    └── example_modifications.json
```

---

## 三、核心功能设计

### 3.1 功能一：新 Prompt 生成

#### 输入
```json
{
  "task_name": "RRU天线接地检查",
  "simple_requirement": "检查RRU设备的天线接地线是否正确连接，接地线颜色应为黄绿色，OT端子应压紧无漏铜",
  "reference_prompts": ["DCDU安装", "GPS避雷器安装"],  // 可选，指定参考哪些
  "key_concepts": {  // 可选，用户提供的关键概念
    "RRU设备": "射频拉远单元，用于信号放大",
    "OT端子": "圆形压接端子"
  },
  "check_points": [  // 可选，明确的检查点
    "接地线存在性",
    "接地线颜色",
    "端子连接质量"
  ]
}
```

#### 处理流程

```python
1. 需求理解与分析
   - 提取关键实体（设备、部件、材料）
   - 识别检查维度（存在性、连接性、质量、数量等）
   - 确定判断逻辑（串行、并行、条件分支）

2. 参考 Prompt 检索
   - 基于语义相似度检索相关 prompt
   - 提取可复用的结构模式
   - 识别类似的判断逻辑

3. Prompt 结构生成
   - 生成角色定位
   - 定义核心概念
   - 构建判断流程
   - 设计输出格式

4. 专业化增强
   - 补充边界情况处理
   - 添加严格性约束
   - 优化术语表达
   - 生成 ICL 示例占位符

5. 质量验证
   - 结构完整性检查
   - 逻辑一致性检查
   - 术语准确性检查
```

#### 输出
```
生成的完整 prompt 文本 + 元数据
{
  "prompt_text": "...",
  "metadata": {
    "task_name": "...",
    "version": "v1",
    "created_at": "2026-01-28",
    "reference_sources": ["DCDU安装", "GPS避雷器安装"],
    "icl_placeholders": [
      {"id": 1, "description": "正常接地示例"},
      {"id": 2, "description": "漏铜错误示例"}
    ]
  }
}
```

---

### 3.2 功能二：Prompt 微调修改

#### 输入
```json
{
  "source_prompt": "DCDU安装/prompts/prompt_other_v18.txt",
  "modification_request": "将浮动螺丝的判断标准放宽，允许最多1个螺丝缺失仍判PASS",
  "modification_type": "rule_relaxation",  // rule_relaxation | rule_tightening | concept_addition | flow_modification
  "affected_sections": ["判断流程/浮动螺丝检查"]  // 可选，指定影响范围
}
```

#### 处理流程

```python
1. Prompt 解析
   - 解析现有 prompt 结构
   - 识别各个章节和规则
   - 提取判断逻辑树

2. 修改点定位
   - 理解修改需求
   - 定位受影响的规则
   - 分析修改影响范围

3. 规则修改
   - 修改目标规则
   - 调整相关逻辑
   - 更新示例说明

4. 一致性维护
   - 检查逻辑冲突
   - 更新相关章节
   - 保持术语一致

5. 专业性保持
   - 保持原有风格
   - 维持严谨性
   - 补充必要说明

6. 版本管理
   - 生成新版本
   - 记录修改历史
   - 对比差异
```

#### 输出
```
{
  "modified_prompt": "...",
  "diff": {
    "original": "若浮动螺丝缺失 → FAIL",
    "modified": "若浮动螺丝缺失数量 > 1 → FAIL；缺失1个 → 提醒但PASS"
  },
  "metadata": {
    "base_version": "v18",
    "new_version": "v19",
    "modification_type": "rule_relaxation",
    "affected_sections": ["判断流程/第三步/浮动螺丝检查"],
    "modified_at": "2026-01-28"
  }
}
```

---

## 四、技术实现方案

### 4.1 Prompt 解析器（PromptParser）

**功能：** 将结构化 prompt 解析为可操作的数据结构

```python
class PromptParser:
    def parse(self, prompt_text: str) -> PromptStructure:
        """
        解析 prompt 为结构化对象
        """
        return PromptStructure(
            task_name=self._extract_task_name(prompt_text),
            role_description=self._extract_role(prompt_text),
            core_concepts=self._extract_concepts(prompt_text),
            icl_examples=self._extract_icl(prompt_text),
            judgment_flow=self._extract_flow(prompt_text),
            output_format=self._extract_output(prompt_text)
        )
    
    def _extract_task_name(self, text: str) -> str:
        """提取 === PROMPT名字开始 === 之间的内容"""
        
    def _extract_flow(self, text: str) -> List[JudgmentStep]:
        """解析判断流程为步骤列表"""
```

### 4.2 Prompt 生成器（PromptGenerator）

**功能：** 基于需求和参考生成新 prompt

```python
class PromptGenerator:
    def __init__(self, llm_client: LLMClient, template_manager: TemplateManager):
        self.llm = llm_client
        self.templates = template_manager
    
    def generate(self, requirement: dict, reference_prompts: List[str]) -> str:
        """
        生成新 prompt
        
        流程：
        1. 加载参考 prompt
        2. 构建 meta-prompt
        3. 调用 LLM 生成
        4. 后处理和验证
        """
        # 加载参考
        references = self._load_references(reference_prompts)
        
        # 构建 meta-prompt
        meta_prompt = self._build_generation_prompt(
            requirement=requirement,
            references=references,
            template=self.templates.get("generation")
        )
        
        # 调用 LLM
        generated = self.llm.generate(meta_prompt)
        
        # 后处理
        validated = self._validate_and_fix(generated)
        
        return validated
    
    def _build_generation_prompt(self, requirement, references, template) -> str:
        """
        构建用于生成的 meta-prompt
        
        Meta-prompt 结构：
        - 任务说明：你是 prompt 工程专家
        - 参考示例：展示 2-3 个完整的参考 prompt
        - 用户需求：展示用户的简单需求
        - 生成要求：结构、风格、专业性要求
        - 输出格式：要求输出完整 prompt
        """
```

### 4.3 Prompt 修改器（PromptModifier）

**功能：** 基于修改需求修改现有 prompt

```python
class PromptModifier:
    def __init__(self, llm_client: LLMClient, parser: PromptParser):
        self.llm = llm_client
        self.parser = parser
    
    def modify(self, source_prompt: str, modification: dict) -> dict:
        """
        修改 prompt
        
        流程：
        1. 解析原 prompt
        2. 定位修改点
        3. 生成修改后的版本
        4. 验证一致性
        5. 生成 diff
        """
        # 解析
        structure = self.parser.parse(source_prompt)
        
        # 构建修改 meta-prompt
        meta_prompt = self._build_modification_prompt(
            structure=structure,
            modification=modification
        )
        
        # 调用 LLM
        modified = self.llm.generate(meta_prompt)
        
        # 生成 diff
        diff = self._generate_diff(source_prompt, modified)
        
        return {
            "modified_prompt": modified,
            "diff": diff,
            "metadata": self._generate_metadata(modification)
        }
    
    def _build_modification_prompt(self, structure, modification) -> str:
        """
        构建用于修改的 meta-prompt
        
        Meta-prompt 结构：
        - 任务说明：你是 prompt 修改专家
        - 原始 prompt：展示完整原文
        - 修改需求：明确的修改要求
        - 约束条件：保持结构、风格、专业性
        - 输出格式：要求输出完整修改后的 prompt
        """
```

### 4.4 LLM 客户端（LLMClient）

**功能：** 调用 VLM 进行 prompt 生成/修改

```python
class LLMClient:
    def __init__(self, server_url: str, model: str):
        self.server_url = server_url
        self.model = model
    
    def generate(self, prompt: str, temperature: float = 0.1) -> str:
        """
        调用 VLM 生成
        
        使用较低的 temperature 保证输出稳定性
        """
        response = requests.post(
            f"{self.server_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "你是专业的工业质检 Prompt 工程专家。"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "max_tokens": 8192
            }
        )
        return response.json()["choices"][0]["message"]["content"]
```

---

## 五、Meta-Prompt 设计

### 5.1 生成任务的 Meta-Prompt

```
你是一名资深的工业质检 Prompt 工程专家，擅长编写结构化、专业化的质检判断 prompt。

【任务】
用户提供了一个新的质检需求，你需要参考已有的优秀 prompt 案例，生成一个专业、严谨的判断 prompt。

【参考案例】
以下是 2 个相关领域的优秀 prompt 案例，请学习它们的结构、风格和专业性：

案例1: DCDU安装合规性判断
```
[完整的参考 prompt 1]
```

案例2: GPS避雷器安装质量检查
```
[完整的参考 prompt 2]
```

【用户需求】
任务名称：{task_name}
简单描述：{simple_requirement}
关键概念：{key_concepts}
检查点：{check_points}

【生成要求】
1. **结构要求**：
   - 必须包含：PROMPT名字、角色定位、核心概念、ICL示例占位符、判断流程、输出格式
   - 使用标准分隔符：=== PROMPT名字开始 ===、=== ICL示例开始 === 等
   - 判断流程使用多级标题：# 判断流程、## 第一步、### 检查点1

2. **专业性要求**：
   - 术语精确：定义所有关键术语
   - 逻辑严密：考虑边界情况、异常情况
   - 规则明确：使用"若...则..."、"必须..."、"禁止..."等明确表述
   - 零容错：对于关键检查项，要求"必须可见"、"明确呈现"

3. **风格要求**：
   - 保持客观、严谨的工程师语气
   - 使用分点列举，避免长段落
   - 重要约束使用加粗或特殊标记
   - 示例说明要具体、可操作

4. **ICL 示例**：
   - 为每个关键概念生成示例占位符
   - 包含正例和反例
   - 说明部分要详细解释判断要点

【输出格式】
直接输出完整的 prompt 文本，不要任何额外说明。
```

### 5.2 修改任务的 Meta-Prompt

```
你是一名资深的工业质检 Prompt 工程专家，擅长精确修改和优化质检判断 prompt。

【任务】
用户需要修改一个现有的 prompt，你需要根据修改需求进行精确修改，同时保持 prompt 的结构、风格和专业性。

【原始 Prompt】
```
[完整的原始 prompt]
```

【修改需求】
修改类型：{modification_type}
具体要求：{modification_request}
影响范围：{affected_sections}

【修改约束】
1. **保持不变**：
   - 整体结构和分隔符
   - 未涉及部分的内容
   - 术语定义和风格
   - ICL 示例（除非明确要求修改）

2. **必须修改**：
   - 受影响的规则和逻辑
   - 相关的判断流程
   - 可能冲突的其他部分

3. **一致性维护**：
   - 修改后的规则要与其他规则协调
   - 术语使用保持一致
   - 逻辑链条完整无矛盾

4. **专业性保持**：
   - 保持原有的严谨性
   - 使用相同的表述风格
   - 补充必要的边界说明

【输出格式】
直接输出完整的修改后 prompt 文本，不要任何额外说明。
```

---

## 六、Web 界面设计（Gradio）

### 6.1 生成界面

```python
# 界面布局
with gr.Blocks() as generator_ui:
    gr.Markdown("# Prompt 生成器")
    
    with gr.Row():
        with gr.Column():
            task_name = gr.Textbox(label="任务名称", placeholder="例如：RRU天线接地检查")
            requirement = gr.Textbox(label="需求描述", lines=5, 
                placeholder="简单描述检查内容和判断标准...")
            
            with gr.Accordion("高级选项", open=False):
                reference_prompts = gr.CheckboxGroup(
                    label="参考 Prompt",
                    choices=["DCDU安装", "DCDU_输入电源", "GPS避雷器安装", "OutdoorLineGrounding"],
                    value=[]
                )
                key_concepts = gr.Textbox(label="关键概念（JSON格式）", lines=3)
                check_points = gr.Textbox(label="检查点（每行一个）", lines=3)
            
            generate_btn = gr.Button("生成 Prompt", variant="primary")
        
        with gr.Column():
            output_prompt = gr.Textbox(label="生成的 Prompt", lines=30)
            
            with gr.Row():
                download_btn = gr.Button("下载")
                copy_btn = gr.Button("复制")
            
            metadata_display = gr.JSON(label="元数据")
    
    generate_btn.click(
        fn=generate_prompt_handler,
        inputs=[task_name, requirement, reference_prompts, key_concepts, check_points],
        outputs=[output_prompt, metadata_display]
    )
```

### 6.2 修改界面

```python
# 界面布局
with gr.Blocks() as modifier_ui:
    gr.Markdown("# Prompt 修改器")
    
    with gr.Row():
        with gr.Column():
            prompt_selector = gr.Dropdown(
                label="选择要修改的 Prompt",
                choices=list_available_prompts(),
                interactive=True
            )
            
            current_prompt = gr.Textbox(label="当前 Prompt", lines=20, interactive=False)
            
            modification_request = gr.Textbox(
                label="修改需求",
                lines=5,
                placeholder="描述你想要修改的地方..."
            )
            
            modification_type = gr.Radio(
                label="修改类型",
                choices=["规则放宽", "规则收紧", "概念补充", "流程调整"],
                value="规则放宽"
            )
            
            modify_btn = gr.Button("修改 Prompt", variant="primary")
        
        with gr.Column():
            modified_prompt = gr.Textbox(label="修改后的 Prompt", lines=20)
            
            diff_display = gr.HighlightedText(label="修改对比")
            
            with gr.Row():
                save_btn = gr.Button("保存新版本")
                revert_btn = gr.Button("撤销")
    
    prompt_selector.change(
        fn=load_prompt_handler,
        inputs=[prompt_selector],
        outputs=[current_prompt]
    )
    
    modify_btn.click(
        fn=modify_prompt_handler,
        inputs=[current_prompt, modification_request, modification_type],
        outputs=[modified_prompt, diff_display]
    )
```

---

## 七、实现优先级

### Phase 1: 核心功能（2-3天）
1. ✅ Prompt 解析器（PromptParser）
2. ✅ LLM 客户端（LLMClient）
3. ✅ Meta-Prompt 模板设计
4. ✅ 基础生成功能（PromptGenerator）

### Phase 2: 修改功能（2天）
1. ✅ Prompt 修改器（PromptModifier）
2. ✅ Diff 生成工具
3. ✅ 版本管理

### Phase 3: Web 界面（2天）
1. ✅ Gradio 生成界面
2. ✅ Gradio 修改界面
3. ✅ 文件管理功能

### Phase 4: 优化增强（1-2天）
1. ✅ 质量验证工具
2. ✅ 批量处理
3. ✅ 历史记录
4. ✅ 导出功能

---

## 八、技术栈

- **Python**: 3.8+
- **Web 框架**: Gradio 6.0+
- **LLM 调用**: requests / openai
- **文本处理**: re, difflib
- **数据处理**: json, pydantic
- **版本控制**: git（可选）

---

## 九、预期效果

### 生成功能
- **输入**：简单需求描述（2-3句话）
- **输出**：完整的专业 prompt（包含结构、逻辑、示例占位符）
- **质量**：与人工编写的 prompt 相当，结构完整、逻辑严密

### 修改功能
- **输入**：修改需求（1-2句话）
- **输出**：精确修改后的 prompt + diff
- **质量**：保持原有风格和专业性，修改精准无副作用

---

## 十、后续扩展

1. **Prompt 评估器**：自动评估生成的 prompt 质量
2. **A/B 测试**：对比不同版本 prompt 的效果
3. **Prompt 库管理**：版本控制、标签分类、搜索
4. **协作功能**：多人编辑、评论、审核
5. **自动优化**：基于推理结果反馈自动优化 prompt

---

## 十一、风险与挑战

1. **LLM 输出稳定性**：需要精心设计 meta-prompt 和后处理逻辑
2. **专业性保证**：需要大量测试和人工审核
3. **一致性维护**：修改时需要检查全局一致性
4. **边界情况**：需要覆盖各种异常输入

---

**总结**：这是一个基于 LLM 的 Prompt 工程辅助系统，核心是通过精心设计的 meta-prompt 让 LLM 学习和模仿现有优秀 prompt 的风格和结构，从而实现高质量的 prompt 生成和修改。
