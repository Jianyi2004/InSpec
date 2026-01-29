# Qwen-Agent 集成方案 - 快速实现 Prompt Foundry

## 📊 Qwen-Agent 库分析

### 核心组件

#### 1. **LLM 模块** (`qwen_agent/llm/`)
- `base.py` - 基础 LLM 类，支持流式输出、function calling
- `oai.py` - OpenAI API 兼容接口（支持 vLLM）
- `qwen_dashscope.py` - DashScope API 接口
- `function_calling.py` - Function calling 实现
- `schema.py` - 消息格式定义

**可用于 Prompt Foundry：**
- ✅ 直接调用 vLLM 服务生成 prompt
- ✅ 支持流式输出，实时显示生成过程
- ✅ Function calling 可用于结构化输出

#### 2. **Agent 模块** (`qwen_agent/agents/`)
- `assistant.py` - 助手 Agent，支持工具调用和文件读取
- `react_chat.py` - ReAct 模式 Agent
- `fncall_agent.py` - Function calling Agent
- `group_chat.py` - 多 Agent 协作

**可用于 Prompt Foundry：**
- ✅ `Assistant` 可作为 Prompt 生成器的基础
- ✅ 支持读取参考 prompt 文件
- ✅ 可以调用自定义工具

#### 3. **工具模块** (`qwen_agent/tools/`)
- `base.py` - 工具基类和注册机制
- `code_interpreter.py` - 代码执行工具
- `doc_qa/` - 文档问答工具
- 支持自定义工具注册

**可用于 Prompt Foundry：**
- ✅ 可以注册自定义工具（如 prompt 解析、diff 生成）
- ✅ 文档问答可用于检索相似 prompt

#### 4. **GUI 模块** (`qwen_agent/gui/`)
- Gradio 5 界面支持
- 快速部署 Web UI

**可用于 Prompt Foundry：**
- ✅ 直接使用 Gradio 界面框架
- ✅ 快速搭建 Web UI

---

## 🚀 集成方案

### 方案 1: 基于 Qwen-Agent 的完整实现（推荐）

**架构：**
```
Prompt Foundry
├── 使用 Qwen-Agent 的 LLM 模块
│   └── 调用 vLLM 服务生成 prompt
├── 使用 Qwen-Agent 的 Agent 框架
│   ├── PromptGeneratorAgent (继承 Assistant)
│   └── PromptModifierAgent (继承 Assistant)
├── 注册自定义工具
│   ├── PromptParserTool
│   ├── ExampleManagerTool
│   └── DiffGeneratorTool
└── 使用 Qwen-Agent 的 GUI
    └── 快速部署 Gradio 界面
```

**优势：**
- ✅ 开箱即用的 LLM 调用
- ✅ 完善的 Agent 框架
- ✅ 自动处理消息历史
- ✅ 内置 Gradio 支持
- ✅ 工具注册机制完善

**实现步骤：**

#### Step 1: 创建自定义工具

```python
from qwen_agent.tools.base import BaseTool, register_tool
import json5

@register_tool('prompt_parser')
class PromptParserTool(BaseTool):
    description = '解析 prompt 文件，提取任务名称、system prompt、main prompt 等结构化信息'
    parameters = [{
        'name': 'prompt_path',
        'type': 'string',
        'description': 'Prompt 文件路径',
        'required': True
    }]
    
    def call(self, params: str, **kwargs) -> str:
        from core.prompt_parser import PromptParser
        parser = PromptParser()
        prompt_path = json5.loads(params)['prompt_path']
        structure = parser.parse_file(prompt_path)
        return json5.dumps({
            'task_name': structure.task_name,
            'system_prompt': structure.system_prompt[:200],
            'main_prompt': structure.main_prompt[:500],
            'icl_images': structure.icl_images
        }, ensure_ascii=False)

@register_tool('example_manager')
class ExampleManagerTool(BaseTool):
    description = '管理 ICL 示例图片，支持添加、删除、查询示例'
    parameters = [{
        'name': 'action',
        'type': 'string',
        'description': '操作类型：add, delete, list',
        'required': True
    }, {
        'name': 'task_folder',
        'type': 'string',
        'description': '任务文件夹名称',
        'required': True
    }, {
        'name': 'image_path',
        'type': 'string',
        'description': '图片路径（add 操作需要）',
        'required': False
    }, {
        'name': 'description',
        'type': 'string',
        'description': '示例说明（add 操作需要）',
        'required': False
    }]
    
    def call(self, params: str, **kwargs) -> str:
        from core.example_manager import ExampleManager
        manager = ExampleManager()
        params_dict = json5.loads(params)
        
        action = params_dict['action']
        task_folder = params_dict['task_folder']
        
        if action == 'add':
            result = manager.add_example(
                task_folder=task_folder,
                image_path=params_dict['image_path'],
                description=params_dict['description']
            )
            return json5.dumps(result, ensure_ascii=False)
        
        elif action == 'list':
            examples = manager.get_examples(task_folder)
            return json5.dumps(examples, ensure_ascii=False)
        
        return json5.dumps({'error': 'Unknown action'})
```

#### Step 2: 创建 Prompt 生成 Agent

```python
from qwen_agent.agents import Assistant

class PromptGeneratorAgent(Assistant):
    """Prompt 生成 Agent"""
    
    def __init__(self, llm_cfg: dict):
        # 系统提示词
        system_instruction = '''你是一名资深的工业质检 Prompt 工程专家。
你的任务是根据用户的简单需求描述，参考已有的优秀 prompt 案例，生成专业、严谨的质检判断 prompt。

生成要求：
1. 结构完整：包含任务名称、角色定位、核心概念、ICL示例、判断流程、输出格式
2. 专业严谨：术语精确、逻辑严密、考虑边界情况
3. 风格统一：保持客观、严谨的工程师语气
4. 示例清晰：为关键概念生成示例占位符

你可以使用以下工具：
- prompt_parser: 解析参考 prompt 文件
- example_manager: 管理示例图片
'''
        
        # 注册工具
        tools = ['prompt_parser', 'example_manager']
        
        super().__init__(
            llm=llm_cfg,
            system_message=system_instruction,
            function_list=tools
        )
    
    def generate_prompt(self, requirement: dict, reference_prompts: list) -> str:
        """生成新的 prompt"""
        # 构建用户消息
        user_message = f"""
请帮我生成一个新的质检 prompt。

【需求信息】
任务名称：{requirement['task_name']}
需求描述：{requirement['description']}

【参考 Prompt】
请先使用 prompt_parser 工具读取以下参考 prompt：
{', '.join(reference_prompts)}

然后参考它们的结构和风格，生成新的 prompt。
"""
        
        messages = [{'role': 'user', 'content': user_message}]
        
        # 调用 Agent
        response = self.run_nonstream(messages)
        return response
```

#### Step 3: 创建 Prompt 修改 Agent

```python
class PromptModifierAgent(Assistant):
    """Prompt 修改 Agent"""
    
    def __init__(self, llm_cfg: dict):
        system_instruction = '''你是一名资深的工业质检 Prompt 工程专家，擅长精确修改和优化 prompt。

你的任务是根据用户的修改需求，对现有 prompt 进行精确修改，同时保持其结构、风格和专业性。

修改约束：
1. 保持整体结构和分隔符不变
2. 只修改受影响的规则和逻辑
3. 维护术语使用的一致性
4. 保持原有的严谨性和风格
'''
        
        tools = ['prompt_parser']
        
        super().__init__(
            llm=llm_cfg,
            system_message=system_instruction,
            function_list=tools
        )
    
    def modify_prompt(self, prompt_path: str, modification_request: str) -> str:
        """修改现有 prompt"""
        user_message = f"""
请帮我修改一个 prompt。

【原始 Prompt】
请使用 prompt_parser 工具读取：{prompt_path}

【修改需求】
{modification_request}

请精确修改受影响的部分，保持其他部分不变。
"""
        
        messages = [{'role': 'user', 'content': user_message}]
        response = self.run_nonstream(messages)
        return response
```

#### Step 4: 创建 Gradio 界面

```python
from qwen_agent.gui import WebUI
import gradio as gr

def create_prompt_foundry_ui():
    """创建 Prompt Foundry Web UI"""
    
    # LLM 配置
    llm_cfg = {
        'model': 'Qwen2.5-72B-Instruct',
        'model_server': 'http://localhost:8000/v1',
        'api_key': 'EMPTY',
        'generate_cfg': {
            'temperature': 0.1,
            'top_p': 0.8
        }
    }
    
    # 创建 Agent
    generator_agent = PromptGeneratorAgent(llm_cfg)
    modifier_agent = PromptModifierAgent(llm_cfg)
    
    with gr.Blocks(title="Prompt Foundry") as demo:
        gr.Markdown("# 🏭 Prompt Foundry - Prompt 工程系统")
        
        with gr.Tabs():
            # Tab 1: 生成新 Prompt
            with gr.Tab("生成 Prompt"):
                with gr.Row():
                    with gr.Column():
                        task_name = gr.Textbox(label="任务名称")
                        description = gr.Textbox(label="需求描述", lines=5)
                        reference_prompts = gr.CheckboxGroup(
                            label="参考 Prompt",
                            choices=["DCDU安装", "DCDU_输入电源", "GPS避雷器安装"]
                        )
                        generate_btn = gr.Button("生成", variant="primary")
                    
                    with gr.Column():
                        output = gr.Textbox(label="生成的 Prompt", lines=30)
                
                generate_btn.click(
                    fn=lambda name, desc, refs: generator_agent.generate_prompt(
                        {'task_name': name, 'description': desc},
                        refs
                    ),
                    inputs=[task_name, description, reference_prompts],
                    outputs=[output]
                )
            
            # Tab 2: 修改 Prompt
            with gr.Tab("修改 Prompt"):
                with gr.Row():
                    with gr.Column():
                        prompt_selector = gr.Dropdown(
                            label="选择 Prompt",
                            choices=["DCDU安装", "DCDU_输入电源", "GPS避雷器安装"]
                        )
                        modification = gr.Textbox(label="修改需求", lines=5)
                        modify_btn = gr.Button("修改", variant="primary")
                    
                    with gr.Column():
                        modified_output = gr.Textbox(label="修改后的 Prompt", lines=30)
                
                modify_btn.click(
                    fn=lambda prompt, mod: modifier_agent.modify_prompt(
                        f"prompts/{prompt}/prompt.txt",
                        mod
                    ),
                    inputs=[prompt_selector, modification],
                    outputs=[modified_output]
                )
            
            # Tab 3: 示例管理
            with gr.Tab("示例管理"):
                with gr.Row():
                    with gr.Column():
                        task_folder = gr.Textbox(label="任务文件夹")
                        image_upload = gr.File(label="上传图片", type="filepath")
                        example_desc = gr.Textbox(label="示例说明", lines=3)
                        add_btn = gr.Button("添加示例", variant="primary")
                    
                    with gr.Column():
                        examples_display = gr.JSON(label="已有示例")
                
                # 添加示例的逻辑
                def add_example(task, img_path, desc):
                    from core.example_manager import ExampleManager
                    manager = ExampleManager()
                    result = manager.add_example(task, img_path, desc)
                    examples = manager.get_examples(task)
                    return examples
                
                add_btn.click(
                    fn=add_example,
                    inputs=[task_folder, image_upload, example_desc],
                    outputs=[examples_display]
                )
    
    return demo

# 启动
if __name__ == '__main__':
    demo = create_prompt_foundry_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860)
```

---

## 📝 完整实现清单

### 已完成 ✅
- [x] Prompt 格式统一
- [x] 示例管理器（ExampleManager）
- [x] Prompt 解析器（PromptParser）

### 使用 Qwen-Agent 快速实现 🚀

#### Phase 1: 工具注册（1天）
- [ ] 注册 PromptParserTool
- [ ] 注册 ExampleManagerTool
- [ ] 注册 DiffGeneratorTool

#### Phase 2: Agent 实现（2天）
- [ ] 实现 PromptGeneratorAgent
- [ ] 实现 PromptModifierAgent
- [ ] 测试 Agent 功能

#### Phase 3: Web 界面（1天）
- [ ] 创建 Gradio 界面
- [ ] 集成 Agent
- [ ] 添加示例管理界面

#### Phase 4: 优化（1天）
- [ ] 优化 meta-prompt
- [ ] 添加版本管理
- [ ] 完善错误处理

**总计：5天完成**

---

## 💡 关键优势

### 使用 Qwen-Agent 的好处

1. **开箱即用的 LLM 调用**
   - 无需自己实现 vLLM 客户端
   - 支持流式输出
   - 自动处理 API 错误

2. **完善的 Agent 框架**
   - 自动管理对话历史
   - 内置工具调用机制
   - 支持多轮对话

3. **工具注册机制**
   - 简单的装饰器注册
   - 自动生成工具描述
   - 参数验证

4. **Gradio 集成**
   - 快速搭建 Web UI
   - 支持文件上传
   - 实时交互

5. **文档和示例丰富**
   - 32 个示例代码
   - 详细文档
   - 活跃社区

---

## 🔧 配置示例

### vLLM 服务配置

```bash
# 启动 vLLM 服务
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/Qwen2.5-72B-Instruct \
    --served-model-name Qwen2.5-72B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2
```

### Qwen-Agent LLM 配置

```python
llm_cfg = {
    'model': 'Qwen2.5-72B-Instruct',
    'model_server': 'http://localhost:8000/v1',
    'api_key': 'EMPTY',
    'generate_cfg': {
        'temperature': 0.1,  # 低温度保证稳定性
        'top_p': 0.8,
        'max_tokens': 8192
    }
}
```

---

## 📊 对比：自己实现 vs 使用 Qwen-Agent

| 功能 | 自己实现 | 使用 Qwen-Agent | 节省时间 |
|------|---------|----------------|---------|
| LLM 调用 | 2天 | ✅ 开箱即用 | 2天 |
| Agent 框架 | 3天 | ✅ 开箱即用 | 3天 |
| 工具系统 | 2天 | ✅ 装饰器注册 | 1.5天 |
| Web 界面 | 2天 | ✅ Gradio 集成 | 1天 |
| 对话管理 | 1天 | ✅ 自动处理 | 1天 |
| 错误处理 | 1天 | ✅ 内置处理 | 0.5天 |
| **总计** | **11天** | **5天** | **节省 6天** |

---

## 🎯 推荐方案

**强烈推荐使用 Qwen-Agent！**

理由：
1. ✅ 节省 50% 以上开发时间
2. ✅ 代码质量更高（经过大规模测试）
3. ✅ 维护成本低（官方持续更新）
4. ✅ 功能更完善（支持更多特性）
5. ✅ 文档和社区支持好

---

## 📦 下一步行动

1. **安装 Qwen-Agent**
   ```bash
   cd /home/intern10/InSpec/Qwen-Agent
   pip install -e ./"[gui]"
   ```

2. **创建工具注册文件**
   ```bash
   touch /home/intern10/InSpec/prompt_foundry/core/qwen_tools.py
   ```

3. **创建 Agent 文件**
   ```bash
   touch /home/intern10/InSpec/prompt_foundry/core/qwen_agents.py
   ```

4. **创建 Web UI 文件**
   ```bash
   touch /home/intern10/InSpec/prompt_foundry/web/app_qwen.py
   ```

5. **开始实现**
   - 按照上面的代码示例实现
   - 测试每个组件
   - 集成到完整系统

---

## 📚 参考资源

- Qwen-Agent 文档: https://qwenlm.github.io/Qwen-Agent/en/
- Qwen-Agent GitHub: https://github.com/QwenLM/Qwen-Agent
- 示例代码: `/home/intern10/InSpec/Qwen-Agent/examples/`
