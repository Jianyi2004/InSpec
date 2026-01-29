"""
Qwen-Agent 自定义工具 - 用于 Prompt Foundry
"""

import json
import json5
from pathlib import Path
from typing import Dict, List
from qwen_agent.tools.base import BaseTool, register_tool


@register_tool('prompt_parser')
class PromptParserTool(BaseTool):
    """解析 prompt 文件的工具"""
    
    description = '解析 prompt 文件，提取任务名称、system prompt、main prompt、summary prompt 等结构化信息。返回 JSON 格式的解析结果。'
    
    parameters = [{
        'name': 'prompt_path',
        'type': 'string',
        'description': 'Prompt 文件的相对路径，例如 "DCDU安装/prompt.txt"',
        'required': True
    }]
    
    def call(self, params: str, **kwargs) -> str:
        """调用工具"""
        try:
            from core.prompt_parser import PromptParser
            
            params_dict = json5.loads(params)
            prompt_path = params_dict['prompt_path']
            
            # 构建完整路径
            base_dir = Path('/home/intern10/InSpec/prompt_foundry/prompts')
            full_path = base_dir / prompt_path
            
            if not full_path.exists():
                return json.dumps({
                    'success': False,
                    'error': f'文件不存在: {prompt_path}'
                }, ensure_ascii=False)
            
            # 解析 prompt
            parser = PromptParser()
            structure = parser.parse_file(str(full_path))
            
            # 返回结构化信息（截断过长的内容）
            result = {
                'success': True,
                'task_name': structure.task_name,
                'system_prompt': structure.system_prompt[:500] + '...' if len(structure.system_prompt) > 500 else structure.system_prompt,
                'main_prompt_preview': structure.main_prompt[:1000] + '...' if len(structure.main_prompt) > 1000 else structure.main_prompt,
                'summary_prompt': structure.summary_prompt[:500] + '...' if len(structure.summary_prompt) > 500 else structure.summary_prompt,
                'icl_images_count': len(structure.icl_images),
                'icl_images': structure.icl_images[:5]  # 只返回前5个
            }
            
            return json.dumps(result, ensure_ascii=False, indent=2)
            
        except Exception as e:
            return json.dumps({
                'success': False,
                'error': f'解析失败: {str(e)}'
            }, ensure_ascii=False)


@register_tool('example_manager')
class ExampleManagerTool(BaseTool):
    """管理 ICL 示例图片的工具"""
    
    description = '管理 ICL 示例图片，支持添加、删除、查询示例。可以上传新的示例图片并添加说明。'
    
    parameters = [{
        'name': 'action',
        'type': 'string',
        'description': '操作类型：add（添加示例）, delete（删除示例）, list（列出所有示例）',
        'required': True
    }, {
        'name': 'task_folder',
        'type': 'string',
        'description': '任务文件夹名称，例如 "DCDU安装"',
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
    }, {
        'name': 'example_id',
        'type': 'integer',
        'description': '示例编号（delete 操作需要）',
        'required': False
    }]
    
    def call(self, params: str, **kwargs) -> str:
        """调用工具"""
        try:
            from core.example_manager import ExampleManager
            
            params_dict = json5.loads(params)
            manager = ExampleManager()
            
            action = params_dict['action']
            task_folder = params_dict['task_folder']
            
            if action == 'add':
                if 'image_path' not in params_dict or 'description' not in params_dict:
                    return json.dumps({
                        'success': False,
                        'error': 'add 操作需要 image_path 和 description 参数'
                    }, ensure_ascii=False)
                
                result = manager.add_example(
                    task_folder=task_folder,
                    image_path=params_dict['image_path'],
                    description=params_dict['description']
                )
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            elif action == 'delete':
                if 'example_id' not in params_dict:
                    return json.dumps({
                        'success': False,
                        'error': 'delete 操作需要 example_id 参数'
                    }, ensure_ascii=False)
                
                success = manager.delete_example(
                    task_folder=task_folder,
                    example_id=params_dict['example_id']
                )
                return json.dumps({
                    'success': success,
                    'message': '删除成功' if success else '删除失败'
                }, ensure_ascii=False)
            
            elif action == 'list':
                examples = manager.get_examples(task_folder)
                return json.dumps({
                    'success': True,
                    'count': len(examples),
                    'examples': examples
                }, ensure_ascii=False, indent=2)
            
            else:
                return json.dumps({
                    'success': False,
                    'error': f'未知操作: {action}'
                }, ensure_ascii=False)
                
        except Exception as e:
            return json.dumps({
                'success': False,
                'error': f'操作失败: {str(e)}'
            }, ensure_ascii=False)


@register_tool('list_prompts')
class ListPromptsTool(BaseTool):
    """列出所有可用的 prompt 任务"""
    
    description = '列出 prompts 目录下所有可用的 prompt 任务文件夹，返回任务列表。'
    
    parameters = []
    
    def call(self, params: str, **kwargs) -> str:
        """调用工具"""
        try:
            from core.example_manager import ExampleManager
            
            manager = ExampleManager()
            tasks = manager.list_all_tasks()
            
            return json.dumps({
                'success': True,
                'count': len(tasks),
                'tasks': tasks
            }, ensure_ascii=False, indent=2)
            
        except Exception as e:
            return json.dumps({
                'success': False,
                'error': f'获取任务列表失败: {str(e)}'
            }, ensure_ascii=False)


@register_tool('version_manager')
class VersionManagerTool(BaseTool):
    """版本管理工具"""
    
    description = 'Prompt 版本管理工具，支持列出版本、保存新版本、加载历史版本等操作。'
    
    parameters = [{
        'name': 'action',
        'type': 'string',
        'description': '操作类型：list（列出所有版本）, save（保存新版本）, load（加载历史版本）, latest（获取最新版本）',
        'required': True
    }, {
        'name': 'task_name',
        'type': 'string',
        'description': '任务名称',
        'required': True
    }, {
        'name': 'prompt_content',
        'type': 'string',
        'description': 'Prompt 内容（save 操作需要）',
        'required': False
    }, {
        'name': 'tag',
        'type': 'string',
        'description': '版本标签（save 操作需要），例如：initial, bugfix, feature',
        'required': False
    }, {
        'name': 'increment_type',
        'type': 'string',
        'description': '版本递增类型（save 操作需要）：major 或 minor',
        'required': False
    }, {
        'name': 'version_folder',
        'type': 'string',
        'description': '版本文件夹名称（load 操作需要）',
        'required': False
    }]
    
    def call(self, params: str, **kwargs) -> str:
        """调用工具"""
        try:
            from core.version_manager import VersionManager
            
            params_dict = json5.loads(params)
            manager = VersionManager()
            
            action = params_dict['action']
            task_name = params_dict['task_name']
            
            if action == 'list':
                versions = manager.list_versions(task_name)
                return json.dumps({
                    'success': True,
                    'count': len(versions),
                    'versions': versions
                }, ensure_ascii=False, indent=2)
            
            elif action == 'save':
                if 'prompt_content' not in params_dict:
                    return json.dumps({
                        'success': False,
                        'error': 'save 操作需要 prompt_content 参数'
                    }, ensure_ascii=False)
                
                tag = params_dict.get('tag', 'update')
                increment_type = params_dict.get('increment_type', 'minor')
                
                version_info = manager.save_prompt_version(
                    task_name=task_name,
                    prompt_content=params_dict['prompt_content'],
                    tag=tag,
                    increment_type=increment_type
                )
                
                return json.dumps({
                    'success': True,
                    'message': f"版本 {version_info['version']} 保存成功",
                    'version_info': version_info
                }, ensure_ascii=False, indent=2)
            
            elif action == 'load':
                if 'version_folder' not in params_dict:
                    return json.dumps({
                        'success': False,
                        'error': 'load 操作需要 version_folder 参数'
                    }, ensure_ascii=False)
                
                content = manager.load_prompt_version(
                    task_name=task_name,
                    version_folder=params_dict['version_folder']
                )
                
                if content:
                    return json.dumps({
                        'success': True,
                        'content': content[:1000] + '...' if len(content) > 1000 else content
                    }, ensure_ascii=False)
                else:
                    return json.dumps({
                        'success': False,
                        'error': '版本不存在'
                    }, ensure_ascii=False)
            
            elif action == 'latest':
                latest = manager.get_latest_version(task_name)
                if latest:
                    return json.dumps({
                        'success': True,
                        'latest_version': latest
                    }, ensure_ascii=False, indent=2)
                else:
                    return json.dumps({
                        'success': True,
                        'message': '暂无历史版本'
                    }, ensure_ascii=False)
            
            else:
                return json.dumps({
                    'success': False,
                    'error': f'未知操作: {action}'
                }, ensure_ascii=False)
                
        except Exception as e:
            return json.dumps({
                'success': False,
                'error': f'操作失败: {str(e)}'
            }, ensure_ascii=False)


@register_tool('diff_generator')
class DiffGeneratorTool(BaseTool):
    """差异对比工具"""
    
    description = '比较两个 Prompt 版本的差异，类似 git diff，返回差异统计和详细对比。'
    
    parameters = [{
        'name': 'old_content',
        'type': 'string',
        'description': '旧版本内容',
        'required': True
    }, {
        'name': 'new_content',
        'type': 'string',
        'description': '新版本内容',
        'required': True
    }, {
        'name': 'format_type',
        'type': 'string',
        'description': '差异格式：unified（统一格式）或 inline（行内格式）',
        'required': False
    }]
    
    def call(self, params: str, **kwargs) -> str:
        """调用工具"""
        try:
            from core.diff_generator import DiffGenerator, ImageDiffDetector
            
            params_dict = json5.loads(params)
            old_content = params_dict['old_content']
            new_content = params_dict['new_content']
            format_type = params_dict.get('format_type', 'unified')
            
            # 生成差异
            diff_text = DiffGenerator.format_diff_for_display(
                old_content, new_content, format_type
            )
            
            # 获取统计信息
            stats = DiffGenerator.get_diff_statistics(old_content, new_content)
            
            # 检查 ICL 图片差异
            image_diff = ImageDiffDetector.compare_icl_images(old_content, new_content)
            
            return json.dumps({
                'success': True,
                'statistics': stats,
                'image_changes': image_diff,
                'diff_preview': diff_text[:500] + '...' if len(diff_text) > 500 else diff_text
            }, ensure_ascii=False, indent=2)
            
        except Exception as e:
            return json.dumps({
                'success': False,
                'error': f'生成差异失败: {str(e)}'
            }, ensure_ascii=False)
