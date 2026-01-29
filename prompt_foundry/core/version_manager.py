"""
Prompt 版本管理系统
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import re


class VersionManager:
    """Prompt 版本管理器"""
    
    def __init__(self, base_dir: str = '/home/intern10/InSpec/prompt_foundry/prompts'):
        """
        初始化版本管理器
        
        Args:
            base_dir: prompts 基础目录
        """
        self.base_dir = Path(base_dir)
    
    def get_task_dir(self, task_name: str) -> Path:
        """获取任务目录"""
        return self.base_dir / task_name
    
    def get_versions_dir(self, task_name: str) -> Path:
        """获取版本目录"""
        versions_dir = self.get_task_dir(task_name) / 'versions'
        versions_dir.mkdir(parents=True, exist_ok=True)
        return versions_dir
    
    def parse_version_folder(self, folder_name: str) -> Optional[Dict]:
        """
        解析版本文件夹名称
        
        格式: v{major}.{minor}_{tag}_{timestamp}
        例如: v1.0_initial_20260129_102530
        
        Returns:
            {
                'version': 'v1.0',
                'major': 1,
                'minor': 0,
                'tag': 'initial',
                'timestamp': '20260129_102530',
                'folder_name': 'v1.0_initial_20260129_102530'
            }
        """
        pattern = r'v(\d+)\.(\d+)_([^_]+)_(\d{8}_\d{6})'
        match = re.match(pattern, folder_name)
        
        if match:
            major, minor, tag, timestamp = match.groups()
            return {
                'version': f'v{major}.{minor}',
                'major': int(major),
                'minor': int(minor),
                'tag': tag,
                'timestamp': timestamp,
                'folder_name': folder_name
            }
        return None
    
    def list_versions(self, task_name: str) -> List[Dict]:
        """
        列出所有版本
        
        Returns:
            版本列表，按版本号降序排列
        """
        versions_dir = self.get_versions_dir(task_name)
        versions = []
        
        for folder in versions_dir.iterdir():
            if folder.is_dir():
                version_info = self.parse_version_folder(folder.name)
                if version_info:
                    # 添加完整路径
                    version_info['path'] = str(folder)
                    # 检查是否有 prompt.txt
                    version_info['has_prompt'] = (folder / 'prompt.txt').exists()
                    # 检查 ICL 图片数量
                    icl_dir = folder / 'icl'
                    if icl_dir.exists():
                        version_info['icl_count'] = len(list(icl_dir.glob('*.*')))
                    else:
                        version_info['icl_count'] = 0
                    versions.append(version_info)
        
        # 按版本号降序排列
        versions.sort(key=lambda x: (x['major'], x['minor']), reverse=True)
        return versions
    
    def get_latest_version(self, task_name: str) -> Optional[Dict]:
        """获取最新版本"""
        versions = self.list_versions(task_name)
        return versions[0] if versions else None
    
    def get_next_version(self, task_name: str, increment_type: str = 'minor') -> str:
        """
        获取下一个版本号
        
        Args:
            task_name: 任务名称
            increment_type: 'major' 或 'minor'
        
        Returns:
            下一个版本号，例如 'v1.1' 或 'v2.0'
        """
        latest = self.get_latest_version(task_name)
        
        if not latest:
            return 'v1.0'
        
        if increment_type == 'major':
            return f"v{latest['major'] + 1}.0"
        else:  # minor
            return f"v{latest['major']}.{latest['minor'] + 1}"
    
    def create_version_folder(
        self,
        task_name: str,
        version: Optional[str] = None,
        tag: str = 'update',
        increment_type: str = 'minor'
    ) -> Path:
        """
        创建新版本文件夹
        
        Args:
            task_name: 任务名称
            version: 版本号（如果为 None，自动生成）
            tag: 版本标签
            increment_type: 版本递增类型
        
        Returns:
            新版本文件夹路径
        """
        if version is None:
            version = self.get_next_version(task_name, increment_type)
        
        # 生成时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 生成文件夹名称
        folder_name = f"{version}_{tag}_{timestamp}"
        
        # 创建文件夹
        version_dir = self.get_versions_dir(task_name) / folder_name
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建 icl 子目录
        (version_dir / 'icl').mkdir(exist_ok=True)
        
        return version_dir
    
    def save_prompt_version(
        self,
        task_name: str,
        prompt_content: str,
        tag: str = 'update',
        increment_type: str = 'minor',
        copy_icl: bool = True
    ) -> Dict:
        """
        保存 prompt 新版本
        
        Args:
            task_name: 任务名称
            prompt_content: prompt 内容
            tag: 版本标签
            increment_type: 版本递增类型
            copy_icl: 是否复制当前的 ICL 图片
        
        Returns:
            版本信息字典
        """
        # 创建版本文件夹
        version_dir = self.create_version_folder(task_name, tag=tag, increment_type=increment_type)
        
        # 保存完整的 prompt.txt
        prompt_file = version_dir / 'prompt.txt'
        prompt_file.write_text(prompt_content, encoding='utf-8')
        
        # 拆分并保存各部分 prompt
        self._save_split_prompts(version_dir, prompt_content)
        
        # 复制 ICL 图片（如果需要）
        if copy_icl:
            task_dir = self.get_task_dir(task_name)
            src_icl_dir = task_dir / 'icl'
            dst_icl_dir = version_dir / 'icl'
            
            if src_icl_dir.exists():
                # 复制所有图片
                for img_file in src_icl_dir.glob('*.*'):
                    if img_file.is_file():
                        shutil.copy2(img_file, dst_icl_dir / img_file.name)
            
            # 复制 examples_metadata.json（如果存在）
            metadata_file = task_dir / 'examples_metadata.json'
            if metadata_file.exists():
                shutil.copy2(metadata_file, version_dir / 'examples_metadata.json')
        
        # 创建版本信息文件
        version_info = self.parse_version_folder(version_dir.name)
        if version_info is None:
            # 如果解析失败，创建基本的版本信息
            version_info = {
                'version': 'v1.0',
                'major': 1,
                'minor': 0,
                'tag': tag,
                'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
                'folder_name': version_dir.name
            }
        
        version_info['created_at'] = datetime.now().isoformat()
        version_info['prompt_length'] = len(prompt_content)
        
        info_file = version_dir / 'version_info.json'
        info_file.write_text(json.dumps(version_info, ensure_ascii=False, indent=2), encoding='utf-8')
        
        return version_info
    
    def load_prompt_version(self, task_name: str, version_folder: str) -> Optional[str]:
        """
        加载指定版本的 prompt
        
        Args:
            task_name: 任务名称
            version_folder: 版本文件夹名称
        
        Returns:
            prompt 内容
        """
        version_dir = self.get_versions_dir(task_name) / version_folder
        prompt_file = version_dir / 'prompt.txt'
        
        if prompt_file.exists():
            return prompt_file.read_text(encoding='utf-8')
        return None
    
    def get_current_prompt(self, task_name: str) -> Optional[str]:
        """获取当前（最新）版本的 prompt"""
        task_dir = self.get_task_dir(task_name)
        prompt_file = task_dir / 'prompt.txt'
        
        if prompt_file.exists():
            return prompt_file.read_text(encoding='utf-8')
        return None
    
    def _save_split_prompts(self, version_dir: Path, prompt_content: str):
        """
        拆分并保存各部分 prompt
        
        Args:
            version_dir: 版本目录
            prompt_content: 完整的 prompt 内容
        """
        import re
        
        # 提取 SYSTEM PROMPT - 从 [SYSTEM PROMPT] 到下一个 [XXX PROMPT] 或结束
        system_match = re.search(r'\[SYSTEM PROMPT\]\s*\n(.*?)(?=\n\[(?:MAIN|SUMMARY|ICL) PROMPT\]|={80}|\Z)', prompt_content, re.DOTALL)
        if system_match:
            system_prompt = system_match.group(1).strip()
            system_file = version_dir / 'system_prompt.txt'
            system_file.write_text(system_prompt, encoding='utf-8')
        
        # 提取 MAIN PROMPT - 从 [MAIN PROMPT] 到下一个 [XXX PROMPT] 或结束
        main_match = re.search(r'\[MAIN PROMPT\]\s*\n(.*?)(?=\n\[(?:SUMMARY|ICL) PROMPT\]|={80}|\Z)', prompt_content, re.DOTALL)
        if main_match:
            main_prompt = main_match.group(1).strip()
            main_file = version_dir / 'main_prompt.txt'
            main_file.write_text(main_prompt, encoding='utf-8')
        
        # 提取 SUMMARY PROMPT - 从 [SUMMARY PROMPT] 到 [ICL IMAGES] 或结束
        summary_match = re.search(r'\[SUMMARY PROMPT\]\s*\n(.*?)(?=\n\[ICL (?:IMAGES|PROMPT)\]|={80}|\Z)', prompt_content, re.DOTALL)
        if summary_match:
            summary_prompt = summary_match.group(1).strip()
            summary_file = version_dir / 'summary_prompt.txt'
            summary_file.write_text(summary_prompt, encoding='utf-8')
    
    def compare_versions(
        self,
        task_name: str,
        version1_folder: str,
        version2_folder: str
    ) -> Dict:
        """
        比较两个版本的差异
        
        Args:
            task_name: 任务名称
            version1_folder: 版本1文件夹名称
            version2_folder: 版本2文件夹名称
            version1: 版本1文件夹名称（如果为 None，使用当前版本）
            version2: 版本2文件夹名称（如果为 None，使用最新历史版本）
        
        Returns:
            差异信息
        """
        # 获取版本1内容
        if version1 is None:
            content1 = self.get_current_prompt(task_name)
            version1_name = 'current'
        else:
            content1 = self.load_prompt_version(task_name, version1)
            version1_name = version1
        
        # 获取版本2内容
        if version2 is None:
            latest = self.get_latest_version(task_name)
            if latest:
                content2 = self.load_prompt_version(task_name, latest['folder_name'])
                version2_name = latest['folder_name']
            else:
                content2 = None
                version2_name = 'none'
        else:
            content2 = self.load_prompt_version(task_name, version2)
            version2_name = version2
        
        if content1 is None or content2 is None:
            return {
                'success': False,
                'error': '无法加载版本内容'
            }
        
        # 使用 difflib 生成差异
        import difflib
        
        lines1 = content1.splitlines(keepends=True)
        lines2 = content2.splitlines(keepends=True)
        
        diff = list(difflib.unified_diff(
            lines2, lines1,
            fromfile=version2_name,
            tofile=version1_name,
            lineterm=''
        ))
        
        return {
            'success': True,
            'version1': version1_name,
            'version2': version2_name,
            'diff': ''.join(diff),
            'diff_html': self._generate_diff_html(lines2, lines1)
        }
    
    def _generate_diff_html(self, lines1: List[str], lines2: List[str]) -> str:
        """生成 HTML 格式的差异对比"""
        import difflib
        
        differ = difflib.HtmlDiff(wrapcolumn=80)
        html = differ.make_table(
            lines1, lines2,
            fromdesc='原版本',
            todesc='新版本',
            context=True,
            numlines=3
        )
        return html
    
    def export_version_summary(self, task_name: str) -> str:
        """导出版本摘要"""
        versions = self.list_versions(task_name)
        
        summary = f"# {task_name} 版本历史\n\n"
        summary += f"总版本数: {len(versions)}\n\n"
        
        for v in versions:
            summary += f"## {v['version']} - {v['tag']}\n"
            summary += f"- 时间: {v['timestamp']}\n"
            summary += f"- 文件夹: {v['folder_name']}\n"
            summary += f"- ICL 图片: {v['icl_count']} 张\n"
            summary += f"- 路径: {v['path']}\n\n"
        
        return summary
