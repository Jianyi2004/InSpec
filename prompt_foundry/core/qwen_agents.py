"""
基于 Qwen-Agent 的 Prompt 生成和修改 Agent
"""

from typing import List, Dict, Optional
from qwen_agent.agents import Assistant
from qwen_agent.llm.schema import Message


class PromptGeneratorAgent(Assistant):
    """Prompt 生成 Agent - 根据用户需求生成新的 prompt"""
    
    def __init__(self, llm_cfg: dict):
        """
        初始化 Prompt 生成 Agent
        
        Args:
            llm_cfg: LLM 配置，例如:
                {
                    'model': 'Qwen2.5-72B-Instruct',
                    'model_server': 'http://localhost:7878/v1',
                    'api_key': 'EMPTY'
                }
        """
        # 系统提示词
        system_instruction = '''你是一名资深的工业质检 Prompt 工程专家，擅长编写结构化、专业化的质检判断 prompt。

【核心能力】
1. 理解用户的简单需求描述，提取关键信息
2. 参考已有的优秀 prompt 案例，学习其结构和风格
3. 生成专业、严谨、结构完整的质检判断 prompt

【生成要求】
1. **结构完整**：
   - 必须包含：任务名称、角色定位、核心概念、ICL示例占位符、判断流程、输出格式
   - 使用标准分隔符：=== PROMPT名字开始 ===、=== ICL示例开始 === 等
   - 判断流程使用多级标题：# 判断流程、## 第一步、### 检查点1

2. **专业严谨**：
   - 术语精确：定义所有关键术语
   - 逻辑严密：考虑边界情况、异常情况
   - 规则明确：使用"若...则..."、"必须..."、"禁止..."等明确表述
   - 零容错：对于关键检查项，要求"必须可见"、"明确呈现"

3. **风格统一**：
   - 保持客观、严谨的工程师语气
   - 使用分点列举，避免长段落
   - 重要约束使用加粗或特殊标记
   - 示例说明要具体、可操作

4. **ICL 示例**：
   - 为每个关键概念生成示例占位符
   - 包含正例和反例
   - 说明部分要详细解释判断要点

【工作流程】
1. 使用 list_prompts 工具查看可用的参考 prompt
2. 使用 prompt_parser 工具读取参考 prompt，学习其结构
3. 根据用户需求和参考案例，生成新的 prompt
4. 确保生成的 prompt 符合统一格式

【可用工具】
- list_prompts: 列出所有可用的 prompt 任务
- prompt_parser: 解析参考 prompt 文件
- example_manager: 管理示例图片（如需要）

【输出格式】
直接输出完整的 prompt 文本，使用统一格式：
================================================================================
TASK: [任务名称]
================================================================================

[SYSTEM PROMPT]
系统提示词内容...

[MAIN PROMPT]
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
================================================================================
'''
        
        # 注册工具
        tools = ['list_prompts', 'prompt_parser', 'example_manager']
        
        super().__init__(
            llm=llm_cfg,
            system_message=system_instruction,
            function_list=tools,
            name='PromptGenerator',
            description='生成新的工业质检 Prompt'
        )


class PromptModifierAgent(Assistant):
    """Prompt 修改 Agent - 根据用户需求修改现有 prompt"""
    
    def __init__(self, llm_cfg: dict):
        """
        初始化 Prompt 修改 Agent
        
        Args:
            llm_cfg: LLM 配置
        """
        # 系统提示词
        system_instruction = '''你是一名资深的工业质检 Prompt 工程专家，擅长精确修改和优化质检判断 prompt。

【核心能力】
1. 理解用户的修改需求，定位需要修改的部分
2. 精确修改受影响的规则和逻辑
3. 保持 prompt 的结构、风格和专业性

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

【修改类型】
- **规则放宽**：降低判断标准，允许更多情况通过
- **规则收紧**：提高判断标准，更严格的检查
- **概念补充**：添加新的术语定义或检查点
- **流程调整**：修改判断流程的顺序或逻辑

【工作流程】
1. 使用 prompt_parser 工具读取原始 prompt
2. 理解用户的修改需求
3. 定位需要修改的具体部分
4. 进行精确修改，保持其他部分不变
5. 输出完整的修改后 prompt

【可用工具】
- prompt_parser: 解析原始 prompt 文件
- list_prompts: 列出所有可用的 prompt 任务

【输出格式】
直接输出完整的修改后 prompt 文本，保持统一格式。
在输出前，简要说明修改了哪些部分。
'''
        
        # 注册工具
        tools = ['prompt_parser', 'list_prompts']
        
        super().__init__(
            llm=llm_cfg,
            system_message=system_instruction,
            function_list=tools,
            name='PromptModifier',
            description='修改和优化现有的工业质检 Prompt'
        )


def create_llm_config(
    model_server: str = 'http://localhost:7878/v1',
    model: str = 'Qwen3-VL-32B-Instruct',
    temperature: float = 0.1,
    top_p: float = 0.8,
    max_tokens: int = 8192
) -> dict:
    """
    创建 LLM 配置
    
    Args:
        model_server: vLLM 服务地址
        model: 模型名称
        temperature: 温度参数（越低越稳定）
        top_p: top_p 参数
        max_tokens: 最大生成 token 数
    
    Returns:
        LLM 配置字典
    """
    return {
        'model': model,
        'model_server': model_server,
        'api_key': 'EMPTY',
        'generate_cfg': {
            'temperature': temperature,
            'top_p': top_p,
            'max_tokens': max_tokens
        }
    }
