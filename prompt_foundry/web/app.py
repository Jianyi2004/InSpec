"""
Prompt Foundry Web 界面 - 基于 Gradio 和 Qwen-Agent
"""

import sys
import os
from pathlib import Path

# 添加项目路径到 sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import gradio as gr
from typing import List, Dict, Optional
import json

# 导入自定义工具和 Agent
from core.qwen_tools import PromptParserTool, ExampleManagerTool, ListPromptsTool
from core.qwen_agents import PromptGeneratorAgent, PromptModifierAgent, create_llm_config
from core.example_manager import ExampleManager


class PromptFoundryUI:
    """Prompt Foundry Web 界面"""
    
    def __init__(
        self,
        model_server: str = 'http://localhost:7878/v1',
        model: str = 'Qwen3-VL-32B-Instruct'
    ):
        """
        初始化 UI
        
        Args:
            model_server: vLLM 服务地址
            model: 模型名称
        """
        # 创建 LLM 配置
        self.llm_cfg = create_llm_config(
            model_server=model_server,
            model=model,
            temperature=0.1,
            top_p=0.8,
            max_tokens=8192
        )
        
        # 创建 Agent
        self.generator_agent = PromptGeneratorAgent(self.llm_cfg)
        self.modifier_agent = PromptModifierAgent(self.llm_cfg)
        
        # 创建示例管理器
        self.example_manager = ExampleManager()
    
    def generate_prompt(
        self,
        task_name: str,
        description: str,
        reference_prompts: List[str],
        key_concepts: str = "",
        check_points: str = ""
    ) -> str:
        """生成新的 prompt"""
        if not task_name or not description:
            return "❌ 请填写任务名称和需求描述"
        
        # 构建用户消息
        user_message = f"""请帮我生成一个新的质检 prompt。

【需求信息】
任务名称：{task_name}
需求描述：{description}
"""
        
        if key_concepts:
            user_message += f"\n关键概念：\n{key_concepts}\n"
        
        if check_points:
            user_message += f"\n检查点：\n{check_points}\n"
        
        if reference_prompts:
            user_message += f"""
【参考 Prompt】
请先使用 list_prompts 工具查看可用的 prompt，然后使用 prompt_parser 工具读取以下参考 prompt：
{', '.join(reference_prompts)}

参考它们的结构和风格，生成新的 prompt。
"""
        else:
            user_message += "\n请参考已有的优秀 prompt 案例生成。"
        
        # 调用 Agent
        messages = [{'role': 'user', 'content': user_message}]
        
        try:
            # 流式输出
            full_response = ""
            for response in self.generator_agent.run(messages):
                if response and len(response) > 0:
                    # 获取最后一条消息
                    last_msg = response[-1]
                    if isinstance(last_msg, dict):
                        content = last_msg.get('content', '')
                    else:
                        content = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
                    
                    if content:
                        full_response = content
                        yield full_response
            
            return full_response if full_response else "生成失败，请检查服务状态"
            
        except Exception as e:
            return f"❌ 生成失败: {str(e)}"
    
    def modify_prompt(
        self,
        prompt_name: str,
        modification_request: str,
        modification_type: str
    ) -> str:
        """修改现有 prompt"""
        if not prompt_name or not modification_request:
            return "❌ 请选择 prompt 并填写修改需求"
        
        # 构建用户消息
        user_message = f"""请帮我修改一个 prompt。

【原始 Prompt】
请使用 prompt_parser 工具读取：{prompt_name}/prompt.txt

【修改类型】
{modification_type}

【修改需求】
{modification_request}

请精确修改受影响的部分，保持其他部分不变。在输出前，简要说明修改了哪些部分。
"""
        
        # 调用 Agent
        messages = [{'role': 'user', 'content': user_message}]
        
        try:
            # 流式输出
            full_response = ""
            for response in self.modifier_agent.run(messages):
                if response and len(response) > 0:
                    last_msg = response[-1]
                    if isinstance(last_msg, dict):
                        content = last_msg.get('content', '')
                    else:
                        content = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
                    
                    if content:
                        full_response = content
                        yield full_response
            
            return full_response if full_response else "修改失败，请检查服务状态"
            
        except Exception as e:
            return f"❌ 修改失败: {str(e)}"
    
    def add_example(
        self,
        task_folder: str,
        image_file,
        description: str
    ) -> str:
        """添加示例"""
        if not task_folder or not image_file or not description:
            return "❌ 请填写完整信息"
        
        try:
            result = self.example_manager.add_example(
                task_folder=task_folder,
                image_path=image_file.name if hasattr(image_file, 'name') else str(image_file),
                description=description
            )
            
            if result.get('success'):
                return f"✅ 添加成功！\n图片: {result['image_name']}\n路径: {result['image_path']}"
            else:
                return f"❌ 添加失败: {result.get('error', '未知错误')}"
                
        except Exception as e:
            return f"❌ 添加失败: {str(e)}"
    
    def list_examples(self, task_folder: str) -> str:
        """列出示例"""
        if not task_folder:
            return "❌ 请填写任务文件夹名称"
        
        try:
            examples = self.example_manager.get_examples(task_folder)
            
            if not examples:
                return "📭 暂无示例"
            
            result = f"📋 共 {len(examples)} 个示例：\n\n"
            for ex in examples:
                result += f"**示例 {ex['example_id']}**\n"
                result += f"- 图片: {ex['image_name']}\n"
                result += f"- 说明: {ex['description'][:100]}...\n"
                result += f"- 添加时间: {ex['added_at']}\n\n"
            
            return result
            
        except Exception as e:
            return f"❌ 获取失败: {str(e)}"
    
    def get_available_prompts(self) -> List[str]:
        """获取可用的 prompt 列表"""
        try:
            return self.example_manager.list_all_tasks()
        except:
            return ["DCDU安装", "DCDU_输入电源", "GPS避雷器安装", "OutdoorLineGrounding", "rrugrounding"]
    
    def create_ui(self):
        """创建 Gradio 界面"""
        
        # 获取可用的 prompt 列表
        available_prompts = self.get_available_prompts()
        
        with gr.Blocks(
            title="Prompt Foundry",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {max-width: 1400px !important;}
            .output-text {font-family: monospace; font-size: 12px;}
            """
        ) as demo:
            
            gr.Markdown("""
            # 🏭 Prompt Foundry - Prompt 工程系统
            
            基于 Qwen-Agent 的工业质检 Prompt 自动生成和优化系统
            """)
            
            with gr.Tabs():
                # Tab 1: 生成新 Prompt
                with gr.Tab("📝 生成 Prompt"):
                    gr.Markdown("### 根据需求描述生成新的专业 Prompt")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            task_name_input = gr.Textbox(
                                label="任务名称",
                                placeholder="例如：RRU天线接地检查",
                                lines=1
                            )
                            
                            description_input = gr.Textbox(
                                label="需求描述",
                                placeholder="简单描述检查内容和判断标准...",
                                lines=5
                            )
                            
                            with gr.Accordion("高级选项", open=False):
                                reference_prompts_input = gr.CheckboxGroup(
                                    label="参考 Prompt（可选）",
                                    choices=available_prompts,
                                    value=[]
                                )
                                
                                key_concepts_input = gr.Textbox(
                                    label="关键概念（可选）",
                                    placeholder="每行一个概念定义...",
                                    lines=3
                                )
                                
                                check_points_input = gr.Textbox(
                                    label="检查点（可选）",
                                    placeholder="每行一个检查点...",
                                    lines=3
                                )
                            
                            generate_btn = gr.Button("🚀 生成 Prompt", variant="primary", size="lg")
                        
                        with gr.Column(scale=2):
                            output_prompt = gr.Textbox(
                                label="生成的 Prompt",
                                lines=30,
                                elem_classes="output-text"
                            )
                            
                            with gr.Row():
                                copy_btn = gr.Button("📋 复制", size="sm")
                                download_btn = gr.DownloadButton("💾 下载", size="sm")
                    
                    # 绑定生成按钮
                    generate_btn.click(
                        fn=self.generate_prompt,
                        inputs=[
                            task_name_input,
                            description_input,
                            reference_prompts_input,
                            key_concepts_input,
                            check_points_input
                        ],
                        outputs=[output_prompt]
                    )
                
                # Tab 2: 修改 Prompt
                with gr.Tab("✏️ 修改 Prompt"):
                    gr.Markdown("### 修改现有 Prompt 的规则和逻辑")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            prompt_selector = gr.Dropdown(
                                label="选择 Prompt",
                                choices=available_prompts,
                                value=available_prompts[0] if available_prompts else None
                            )
                            
                            modification_type_input = gr.Radio(
                                label="修改类型",
                                choices=["规则放宽", "规则收紧", "概念补充", "流程调整"],
                                value="规则放宽"
                            )
                            
                            modification_input = gr.Textbox(
                                label="修改需求",
                                placeholder="描述你想要修改的地方...",
                                lines=5
                            )
                            
                            modify_btn = gr.Button("🔧 修改 Prompt", variant="primary", size="lg")
                        
                        with gr.Column(scale=2):
                            modified_output = gr.Textbox(
                                label="修改后的 Prompt",
                                lines=30,
                                elem_classes="output-text"
                            )
                            
                            with gr.Row():
                                copy_modified_btn = gr.Button("📋 复制", size="sm")
                                download_modified_btn = gr.DownloadButton("💾 下载", size="sm")
                    
                    # 绑定修改按钮
                    modify_btn.click(
                        fn=self.modify_prompt,
                        inputs=[prompt_selector, modification_input, modification_type_input],
                        outputs=[modified_output]
                    )
                
                # Tab 3: 示例管理
                with gr.Tab("🖼️ 示例管理"):
                    gr.Markdown("### 管理 ICL 示例图片和说明")
                    
                    with gr.Row():
                        with gr.Column():
                            task_folder_input = gr.Dropdown(
                                label="任务文件夹",
                                choices=available_prompts,
                                value=available_prompts[0] if available_prompts else None
                            )
                            
                            image_upload = gr.File(
                                label="上传图片",
                                file_types=["image"],
                                type="filepath"
                            )
                            
                            example_desc_input = gr.Textbox(
                                label="示例说明",
                                placeholder="详细描述这个示例展示了什么...",
                                lines=5
                            )
                            
                            add_example_btn = gr.Button("➕ 添加示例", variant="primary")
                            
                            add_result = gr.Textbox(
                                label="操作结果",
                                lines=3
                            )
                        
                        with gr.Column():
                            list_examples_btn = gr.Button("📋 查看已有示例")
                            
                            examples_display = gr.Textbox(
                                label="已有示例",
                                lines=25
                            )
                    
                    # 绑定按钮
                    add_example_btn.click(
                        fn=self.add_example,
                        inputs=[task_folder_input, image_upload, example_desc_input],
                        outputs=[add_result]
                    )
                    
                    list_examples_btn.click(
                        fn=self.list_examples,
                        inputs=[task_folder_input],
                        outputs=[examples_display]
                    )
            
            gr.Markdown("""
            ---
            💡 **使用提示**:
            - 生成 Prompt: 提供简单需求描述，系统会参考已有案例生成专业 prompt
            - 修改 Prompt: 选择要修改的 prompt，描述修改需求，系统会精确修改
            - 示例管理: 上传示例图片并添加说明，系统会自动整合到 prompt 中
            
            🔗 **vLLM 服务**: {model_server}  
            🤖 **模型**: {model}
            """.format(
                model_server=self.llm_cfg['model_server'],
                model=self.llm_cfg['model']
            ))
        
        return demo


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Prompt Foundry Web UI')
    parser.add_argument('--server', type=str, default='http://localhost:7878/v1',
                        help='vLLM 服务地址')
    parser.add_argument('--model', type=str, default='Qwen3-VL-32B-Instruct',
                        help='模型名称')
    parser.add_argument('--port', type=int, default=7860,
                        help='Web UI 端口')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Web UI 主机')
    
    args = parser.parse_args()
    
    # 创建 UI
    print(f"🚀 启动 Prompt Foundry...")
    print(f"📡 vLLM 服务: {args.server}")
    print(f"🤖 模型: {args.model}")
    print(f"🌐 Web UI: http://{args.host}:{args.port}")
    
    ui = PromptFoundryUI(
        model_server=args.server,
        model=args.model
    )
    
    demo = ui.create_ui()
    
    # 启动
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=False
    )


if __name__ == '__main__':
    main()
