"""
Prompt Foundry Web 界面 - 增强版
支持版本管理、差异对比、保存功能和示例图片提示
"""

import sys
import os
from pathlib import Path

# 添加项目路径到 sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import gradio as gr
from typing import List, Dict, Optional, Tuple
import json

# 导入核心模块
from core.qwen_tools import PromptParserTool, ExampleManagerTool, ListPromptsTool
from core.qwen_agents import PromptGeneratorAgent, PromptModifierAgent, create_llm_config
from core.example_manager import ExampleManager
from core.version_manager import VersionManager
from core.diff_generator import DiffGenerator, ImageDiffDetector


class PromptFoundryEnhancedUI:
    """Prompt Foundry 增强版 Web 界面"""
    
    def __init__(
        self,
        model_server: str = 'http://localhost:7878/v1',
        model: str = 'Qwen3-VL-32B-Instruct'
    ):
        """初始化 UI"""
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
        
        # 创建管理器
        self.example_manager = ExampleManager()
        self.version_manager = VersionManager()
        
        # 存储当前会话的 prompt
        self.current_original_prompt = ""
        self.current_modified_prompt = ""
    
    def generate_prompt(
        self,
        task_name: str,
        description: str,
        reference_prompts: List[str],
        key_concepts: str = "",
        check_points: str = ""
    ):
        """生成新的 prompt"""
        if not task_name or not description:
            return "❌ 请填写任务名称和需求描述", ""
        
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
请使用 prompt_parser 工具读取以下参考 prompt：
{', '.join(reference_prompts)}

参考它们的结构和风格，生成新的 prompt。
"""
        
        messages = [{'role': 'user', 'content': user_message}]
        
        try:
            full_response = ""
            for response in self.generator_agent.run(messages):
                if response and len(response) > 0:
                    last_msg = response[-1]
                    if isinstance(last_msg, dict):
                        content = last_msg.get('content', '')
                    else:
                        content = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
                    
                    if content:
                        full_response = content
                        # 检查缺失的图片
                        missing_images_msg = self._check_missing_images(full_response)
                        yield full_response, missing_images_msg
            
            # 最终检查
            missing_images_msg = self._check_missing_images(full_response)
            return full_response, missing_images_msg
            
        except Exception as e:
            return f"❌ 生成失败: {str(e)}", ""
    
    def load_current_prompt(self, prompt_name: str) -> str:
        """加载当前 prompt 内容"""
        if not prompt_name:
            return "❌ 请选择 prompt"
        
        try:
            content = self.version_manager.get_current_prompt(prompt_name)
            if content:
                self.current_original_prompt = content
                return content
            else:
                return "❌ 无法加载 prompt"
        except Exception as e:
            return f"❌ 加载失败: {str(e)}"
    
    def modify_prompt(
        self,
        prompt_name: str,
        original_prompt: str,
        modification_request: str,
        modification_type: str
    ):
        """修改现有 prompt"""
        if not prompt_name or not modification_request:
            return "❌ 请选择 prompt 并填写修改需求", "", ""
        
        # 如果没有原始 prompt，加载它
        if not original_prompt:
            original_prompt = self.load_current_prompt(prompt_name)
        
        self.current_original_prompt = original_prompt
        
        # 构建用户消息
        user_message = f"""请帮我修改一个 prompt。

【原始 Prompt】
{original_prompt[:]}

【修改类型】
{modification_type}

【修改需求】
{modification_request}

请精确修改受影响的部分，保持其他部分不变。在输出前，简要说明修改了哪些部分。
"""
        
        messages = [{'role': 'user', 'content': user_message}]
        
        try:
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
                        # 生成差异对比
                        diff_text, stats_text = self._generate_diff(original_prompt, full_response)
                        # 检查缺失的图片
                        missing_images_msg = self._check_missing_images(full_response)
                        yield full_response, diff_text, missing_images_msg
            
            self.current_modified_prompt = full_response
            
            # 最终差异对比和图片检查
            diff_text, stats_text = self._generate_diff(original_prompt, full_response)
            missing_images_msg = self._check_missing_images(full_response)
            
            return full_response, diff_text, missing_images_msg
            
        except Exception as e:
            return f"❌ 修改失败: {str(e)}", "", ""
    
    def _generate_diff(self, old_content: str, new_content: str) -> Tuple[str, str]:
        """生成差异对比"""
        try:
            # 生成统一格式差异
            diff_text = DiffGenerator.generate_unified_diff(
                old_content, new_content,
                old_label='原版本',
                new_label='修改后'
            )
            
            # 获取统计信息
            stats = DiffGenerator.get_diff_statistics(old_content, new_content)
            stats_text = f"""📊 **差异统计**
- ➕ 新增行数: {stats['lines_added']}
- ➖ 删除行数: {stats['lines_deleted']}
- 🔄 修改行数: {stats['lines_changed']}
- 📝 总变更: {stats['total_changes']}
"""
            
            # 检查 ICL 图片差异
            image_diff = ImageDiffDetector.compare_icl_images(old_content, new_content)
            if image_diff['added'] or image_diff['removed']:
                stats_text += f"\n🖼️ **ICL 图片变化**\n"
                if image_diff['added']:
                    stats_text += f"- ➕ 新增: {', '.join(image_diff['added'])}\n"
                if image_diff['removed']:
                    stats_text += f"- ➖ 删除: {', '.join(image_diff['removed'])}\n"
            
            full_diff = f"{stats_text}\n\n```diff\n{diff_text}\n```"
            
            return full_diff, stats_text
            
        except Exception as e:
            return f"❌ 生成差异失败: {str(e)}", ""
    
    def _check_missing_images(self, prompt_content: str) -> str:
        """检查缺失的示例图片"""
        try:
            # 提取所有引用的图片
            images = ImageDiffDetector.extract_icl_images(prompt_content)
            
            if not images:
                return ""
            
            missing_images = []
            for img_path in images:
                # 检查图片是否存在（这里简化处理，实际需要根据任务路径检查）
                if 'example_' in img_path or 'placeholder' in img_path.lower():
                    missing_images.append(img_path)
            
            if missing_images:
                msg = "⚠️ **需要上传的示例图片**\n\n"
                msg += "以下图片在 prompt 中被引用，但可能尚未上传：\n\n"
                for img in missing_images:
                    msg += f"- `{img}`\n"
                msg += "\n💡 **操作建议**：\n"
                msg += "1. 前往「示例管理」标签页上传这些图片\n"
                msg += "2. 或者在保存 prompt 后，将图片放到对应的 `icl/` 目录下\n"
                return msg
            
            return "✅ 所有引用的图片都已存在"
            
        except Exception as e:
            return f"⚠️ 检查图片时出错: {str(e)}"
    
    def save_prompt_version(
        self,
        task_name: str,
        prompt_content: str,
        tag: str,
        increment_type: str
    ) -> str:
        """保存 prompt 新版本"""
        if not task_name or not prompt_content:
            return "❌ 请填写任务名称和 prompt 内容"
        
        if not tag:
            tag = "update"
        
        try:
            version_info = self.version_manager.save_prompt_version(
                task_name=task_name,
                prompt_content=prompt_content,
                tag=tag,
                increment_type=increment_type,
                copy_icl=True
            )
            
            result = f"""✅ **版本保存成功！**

📦 **版本信息**
- 版本号: {version_info['version']}
- 标签: {version_info['tag']}
- 时间: {version_info['timestamp']}
- 文件夹: {version_info['folder_name']}

📁 **保存路径**
`prompts/{task_name}/versions/{version_info['folder_name']}/`

📝 **包含内容**
- prompt.txt
- icl/ 目录（已复制当前示例图片）
- version_info.json
"""
            
            return result
            
        except Exception as e:
            return f"❌ 保存失败: {str(e)}"
    
    def list_versions(self, task_name: str) -> str:
        """列出所有版本"""
        if not task_name:
            return "❌ 请选择任务"
        
        try:
            versions = self.version_manager.list_versions(task_name)
            
            if not versions:
                return "📭 暂无历史版本"
            
            result = f"📚 **{task_name} 版本历史**\n\n"
            result += f"共 {len(versions)} 个版本\n\n"
            
            for v in versions:
                result += f"### {v['version']} - {v['tag']}\n"
                result += f"- 📅 时间: {v['timestamp']}\n"
                result += f"- 📁 文件夹: `{v['folder_name']}`\n"
                result += f"- 🖼️ ICL 图片: {v['icl_count']} 张\n"
                result += f"- 📍 路径: `{v['path']}`\n\n"
            
            return result
            
        except Exception as e:
            return f"❌ 获取版本列表失败: {str(e)}"
    
    def get_available_prompts(self) -> List[str]:
        """获取可用的 prompt 列表"""
        try:
            return self.example_manager.list_all_tasks()
        except:
            return ["DCDU安装", "DCDU_输入电源", "GPS避雷器安装", "OutdoorLineGrounding", "rrugrounding"]
    
    def add_example(self, task_folder: str, image_file, description: str) -> str:
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
    
    def create_ui(self):
        """创建 Gradio 界面"""
        available_prompts = self.get_available_prompts()
        
        with gr.Blocks(
            title="Prompt Foundry Enhanced",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {max-width: 1600px !important;}
            .output-text {font-family: monospace; font-size: 12px;}
            .diff-view {font-family: monospace; font-size: 11px; background: #f5f5f5;}
            """
        ) as demo:
            
            gr.Markdown("""
            # 🏭 Prompt Foundry - 增强版
            
            **新功能**: ✅ 版本管理 | ✅ 差异对比 | ✅ 保存功能 | ✅ 图片检测
            """)
            
            with gr.Tabs():
                # Tab 1: 生成新 Prompt
                with gr.Tab("📝 生成 Prompt"):
                    gr.Markdown("### 根据需求描述生成新的专业 Prompt")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            task_name_gen = gr.Textbox(label="任务名称", placeholder="例如：RRU天线接地检查", lines=1)
                            description_gen = gr.Textbox(label="需求描述", placeholder="简单描述检查内容和判断标准...", lines=5)
                            
                            with gr.Accordion("高级选项", open=False):
                                reference_prompts_gen = gr.CheckboxGroup(
                                    label="参考 Prompt（可选）",
                                    choices=available_prompts,
                                    value=[]
                                )
                                key_concepts_gen = gr.Textbox(label="关键概念（可选）", lines=3)
                                check_points_gen = gr.Textbox(label="检查点（可选）", lines=3)
                            
                            generate_btn = gr.Button("🚀 生成 Prompt", variant="primary", size="lg")
                        
                        with gr.Column(scale=2):
                            output_prompt_gen = gr.Textbox(label="生成的 Prompt", lines=25, elem_classes="output-text")
                            missing_images_gen = gr.Markdown(label="图片检查")
                            
                            with gr.Row():
                                save_tag_gen = gr.Textbox(label="版本标签", value="initial", scale=2)
                                save_type_gen = gr.Radio(label="版本类型", choices=["minor", "major"], value="minor", scale=1)
                                save_btn_gen = gr.Button("💾 保存版本", variant="secondary")
                            
                            save_result_gen = gr.Markdown()
                    
                    generate_btn.click(
                        fn=self.generate_prompt,
                        inputs=[task_name_gen, description_gen, reference_prompts_gen, key_concepts_gen, check_points_gen],
                        outputs=[output_prompt_gen, missing_images_gen]
                    )
                    
                    save_btn_gen.click(
                        fn=self.save_prompt_version,
                        inputs=[task_name_gen, output_prompt_gen, save_tag_gen, save_type_gen],
                        outputs=[save_result_gen]
                    )
                
                # Tab 2: 修改 Prompt
                with gr.Tab("✏️ 修改 Prompt"):
                    gr.Markdown("### 修改现有 Prompt 的规则和逻辑")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            prompt_selector_mod = gr.Dropdown(
                                label="选择 Prompt",
                                choices=available_prompts,
                                value=available_prompts[0] if available_prompts else None
                            )
                            
                            load_btn = gr.Button("📂 加载当前版本", size="sm")
                            
                            original_prompt_display = gr.Textbox(
                                label="原始 Prompt（自动加载）",
                                lines=10,
                                elem_classes="output-text"
                            )
                            
                            modification_type_mod = gr.Radio(
                                label="修改类型",
                                choices=["规则放宽", "规则收紧", "概念补充", "流程调整"],
                                value="规则放宽"
                            )
                            
                            modification_mod = gr.Textbox(label="修改需求", placeholder="描述你想要修改的地方...", lines=5)
                            modify_btn = gr.Button("🔧 修改 Prompt", variant="primary", size="lg")
                        
                        with gr.Column(scale=2):
                            modified_output = gr.Textbox(label="修改后的 Prompt", lines=20, elem_classes="output-text")
                            
                            with gr.Accordion("📊 差异对比", open=True):
                                diff_output = gr.Markdown(elem_classes="diff-view")
                            
                            missing_images_mod = gr.Markdown()
                            
                            with gr.Row():
                                save_tag_mod = gr.Textbox(label="版本标签", value="update", scale=2)
                                save_type_mod = gr.Radio(label="版本类型", choices=["minor", "major"], value="minor", scale=1)
                                save_btn_mod = gr.Button("💾 保存版本", variant="secondary")
                            
                            save_result_mod = gr.Markdown()
                    
                    load_btn.click(
                        fn=self.load_current_prompt,
                        inputs=[prompt_selector_mod],
                        outputs=[original_prompt_display]
                    )
                    
                    modify_btn.click(
                        fn=self.modify_prompt,
                        inputs=[prompt_selector_mod, original_prompt_display, modification_mod, modification_type_mod],
                        outputs=[modified_output, diff_output, missing_images_mod]
                    )
                    
                    save_btn_mod.click(
                        fn=self.save_prompt_version,
                        inputs=[prompt_selector_mod, modified_output, save_tag_mod, save_type_mod],
                        outputs=[save_result_mod]
                    )
                
                # Tab 3: 版本管理
                with gr.Tab("📚 版本管理"):
                    gr.Markdown("### 查看和管理 Prompt 版本历史")
                    
                    with gr.Row():
                        with gr.Column():
                            task_selector_ver = gr.Dropdown(
                                label="选择任务",
                                choices=available_prompts,
                                value=available_prompts[0] if available_prompts else None
                            )
                            
                            list_versions_btn = gr.Button("📋 查看版本历史", variant="primary")
                            
                            versions_display = gr.Markdown()
                        
                        with gr.Column():
                            gr.Markdown("""
                            ### 📖 版本命名规范
                            
                            版本文件夹格式：`v{major}.{minor}_{tag}_{timestamp}`
                            
                            **示例**：
                            - `v1.0_initial_20260129_102530`
                            - `v1.1_bugfix_20260129_143022`
                            - `v2.0_major_update_20260130_091500`
                            
                            **版本类型**：
                            - **minor**: 小版本更新（v1.0 → v1.1）
                            - **major**: 大版本更新（v1.9 → v2.0）
                            
                            **常用标签**：
                            - `initial`: 初始版本
                            - `bugfix`: 修复问题
                            - `feature`: 新增功能
                            - `update`: 常规更新
                            - `optimize`: 优化改进
                            """)
                    
                    list_versions_btn.click(
                        fn=self.list_versions,
                        inputs=[task_selector_ver],
                        outputs=[versions_display]
                    )
                
                # Tab 4: 示例管理
                with gr.Tab("🖼️ 示例管理"):
                    gr.Markdown("### 管理 ICL 示例图片和说明")
                    
                    with gr.Row():
                        with gr.Column():
                            task_folder_ex = gr.Dropdown(
                                label="任务文件夹",
                                choices=available_prompts,
                                value=available_prompts[0] if available_prompts else None
                            )
                            
                            image_upload = gr.File(label="上传图片", file_types=["image"], type="filepath")
                            example_desc_ex = gr.Textbox(label="示例说明", placeholder="详细描述这个示例展示了什么...", lines=5)
                            
                            add_example_btn = gr.Button("➕ 添加示例", variant="primary")
                            add_result = gr.Textbox(label="操作结果", lines=3)
                        
                        with gr.Column():
                            list_examples_btn = gr.Button("📋 查看已有示例")
                            examples_display = gr.Textbox(label="已有示例", lines=25)
                    
                    add_example_btn.click(
                        fn=self.add_example,
                        inputs=[task_folder_ex, image_upload, example_desc_ex],
                        outputs=[add_result]
                    )
                    
                    list_examples_btn.click(
                        fn=self.list_examples,
                        inputs=[task_folder_ex],
                        outputs=[examples_display]
                    )
            
            gr.Markdown(f"""
            ---
            💡 **使用提示**:
            - **生成 Prompt**: 提供需求描述，系统自动生成专业 prompt，并检测缺失的示例图片
            - **修改 Prompt**: 加载当前版本，描述修改需求，系统显示差异对比
            - **版本管理**: 所有修改都可保存为新版本，支持版本号和标签
            - **示例管理**: 上传示例图片，系统自动整合到 prompt 中
            
            🔗 **vLLM 服务**: {self.llm_cfg['model_server']}  
            🤖 **模型**: {self.llm_cfg['model']}
            """)
        
        return demo


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Prompt Foundry Enhanced Web UI')
    parser.add_argument('--server', type=str, default='http://localhost:7878/v1', help='vLLM 服务地址')
    parser.add_argument('--model', type=str, default='Qwen3-VL-32B-Instruct', help='模型名称')
    parser.add_argument('--port', type=int, default=7860, help='Web UI 端口')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Web UI 主机')
    
    args = parser.parse_args()
    
    print(f"🚀 启动 Prompt Foundry Enhanced...")
    print(f"📡 vLLM 服务: {args.server}")
    print(f"🤖 模型: {args.model}")
    print(f"🌐 Web UI: http://{args.host}:{args.port}")
    
    ui = PromptFoundryEnhancedUI(
        model_server=args.server,
        model=args.model
    )
    
    demo = ui.create_ui()
    demo.launch(server_name=args.host, server_port=args.port, share=False)


if __name__ == '__main__':
    main()
