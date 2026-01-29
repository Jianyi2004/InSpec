"""
Prompt 差异对比工具 - 类似 git diff
"""

import difflib
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class DiffLine:
    """差异行"""
    line_type: str  # 'add', 'delete', 'context', 'info'
    content: str
    old_line_num: int = 0
    new_line_num: int = 0


class DiffGenerator:
    """差异生成器"""
    
    @staticmethod
    def generate_unified_diff(
        old_content: str,
        new_content: str,
        old_label: str = '原版本',
        new_label: str = '新版本',
        context_lines: int = 3
    ) -> str:
        """
        生成统一格式的差异（类似 git diff）
        
        Args:
            old_content: 旧内容
            new_content: 新内容
            old_label: 旧版本标签
            new_label: 新版本标签
            context_lines: 上下文行数
        
        Returns:
            统一格式的差异文本
        """
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=old_label,
            tofile=new_label,
            lineterm='',
            n=context_lines
        )
        
        return ''.join(diff)
    
    @staticmethod
    def generate_side_by_side_diff(
        old_content: str,
        new_content: str,
        old_label: str = '原版本',
        new_label: str = '新版本'
    ) -> str:
        """
        生成并排对比格式的差异（HTML）
        
        Returns:
            HTML 格式的并排对比
        """
        old_lines = old_content.splitlines()
        new_lines = new_content.splitlines()
        
        differ = difflib.HtmlDiff(wrapcolumn=80, tabsize=4)
        html = differ.make_file(
            old_lines,
            new_lines,
            fromdesc=old_label,
            todesc=new_label,
            context=True,
            numlines=3
        )
        
        return html
    
    @staticmethod
    def generate_inline_diff(
        old_content: str,
        new_content: str
    ) -> List[DiffLine]:
        """
        生成行内差异（带行号）
        
        Returns:
            差异行列表
        """
        old_lines = old_content.splitlines()
        new_lines = new_content.splitlines()
        
        diff_lines = []
        old_line_num = 1
        new_line_num = 1
        
        # 使用 SequenceMatcher 生成差异
        matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                # 相同的行
                for i, line in enumerate(old_lines[i1:i2]):
                    diff_lines.append(DiffLine(
                        line_type='context',
                        content=line,
                        old_line_num=old_line_num + i,
                        new_line_num=new_line_num + i
                    ))
                old_line_num += (i2 - i1)
                new_line_num += (j2 - j1)
            
            elif tag == 'delete':
                # 删除的行
                for i, line in enumerate(old_lines[i1:i2]):
                    diff_lines.append(DiffLine(
                        line_type='delete',
                        content=line,
                        old_line_num=old_line_num + i,
                        new_line_num=0
                    ))
                old_line_num += (i2 - i1)
            
            elif tag == 'insert':
                # 新增的行
                for i, line in enumerate(new_lines[j1:j2]):
                    diff_lines.append(DiffLine(
                        line_type='add',
                        content=line,
                        old_line_num=0,
                        new_line_num=new_line_num + i
                    ))
                new_line_num += (j2 - j1)
            
            elif tag == 'replace':
                # 替换的行（先删除再添加）
                for i, line in enumerate(old_lines[i1:i2]):
                    diff_lines.append(DiffLine(
                        line_type='delete',
                        content=line,
                        old_line_num=old_line_num + i,
                        new_line_num=0
                    ))
                old_line_num += (i2 - i1)
                
                for i, line in enumerate(new_lines[j1:j2]):
                    diff_lines.append(DiffLine(
                        line_type='add',
                        content=line,
                        old_line_num=0,
                        new_line_num=new_line_num + i
                    ))
                new_line_num += (j2 - j1)
        
        return diff_lines
    
    @staticmethod
    def format_diff_for_display(
        old_content: str,
        new_content: str,
        format_type: str = 'unified'
    ) -> str:
        """
        格式化差异用于显示
        
        Args:
            old_content: 旧内容
            new_content: 新内容
            format_type: 'unified', 'side_by_side', 或 'inline'
        
        Returns:
            格式化后的差异文本
        """
        if format_type == 'unified':
            return DiffGenerator.generate_unified_diff(old_content, new_content)
        
        elif format_type == 'side_by_side':
            return DiffGenerator.generate_side_by_side_diff(old_content, new_content)
        
        elif format_type == 'inline':
            diff_lines = DiffGenerator.generate_inline_diff(old_content, new_content)
            return DiffGenerator._format_inline_diff(diff_lines)
        
        else:
            raise ValueError(f"未知的格式类型: {format_type}")
    
    @staticmethod
    def _format_inline_diff(diff_lines: List[DiffLine]) -> str:
        """格式化行内差异为文本"""
        result = []
        
        for line in diff_lines:
            if line.line_type == 'add':
                prefix = '+ '
                line_num = f"{line.new_line_num:4d}"
            elif line.line_type == 'delete':
                prefix = '- '
                line_num = f"{line.old_line_num:4d}"
            else:  # context
                prefix = '  '
                line_num = f"{line.old_line_num:4d}"
            
            result.append(f"{line_num} {prefix}{line.content}")
        
        return '\n'.join(result)
    
    @staticmethod
    def get_diff_statistics(old_content: str, new_content: str) -> Dict:
        """
        获取差异统计信息
        
        Returns:
            {
                'lines_added': int,
                'lines_deleted': int,
                'lines_changed': int,
                'total_changes': int
            }
        """
        old_lines = old_content.splitlines()
        new_lines = new_content.splitlines()
        
        matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
        
        lines_added = 0
        lines_deleted = 0
        lines_changed = 0
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'insert':
                lines_added += (j2 - j1)
            elif tag == 'delete':
                lines_deleted += (i2 - i1)
            elif tag == 'replace':
                lines_changed += max(i2 - i1, j2 - j1)
        
        return {
            'lines_added': lines_added,
            'lines_deleted': lines_deleted,
            'lines_changed': lines_changed,
            'total_changes': lines_added + lines_deleted + lines_changed
        }
    
    @staticmethod
    def highlight_changes_markdown(old_content: str, new_content: str) -> str:
        """
        生成 Markdown 格式的高亮差异
        
        Returns:
            Markdown 格式的差异文本
        """
        diff_lines = DiffGenerator.generate_inline_diff(old_content, new_content)
        
        result = ["```diff"]
        
        for line in diff_lines:
            if line.line_type == 'add':
                result.append(f"+ {line.content}")
            elif line.line_type == 'delete':
                result.append(f"- {line.content}")
            else:
                result.append(f"  {line.content}")
        
        result.append("```")
        
        return '\n'.join(result)


class ImageDiffDetector:
    """ICL 图片差异检测器"""
    
    @staticmethod
    def extract_icl_images(prompt_content: str) -> List[str]:
        """
        从 prompt 内容中提取 ICL 图片路径
        
        Returns:
            图片路径列表
        """
        import re
        
        images = []
        
        # 匹配 [ICL IMAGES] 部分
        icl_section_pattern = r'\[ICL IMAGES\](.*?)(?:={80}|$)'
        match = re.search(icl_section_pattern, prompt_content, re.DOTALL)
        
        if match:
            icl_section = match.group(1)
            # 提取所有图片路径
            image_pattern = r'[-*]\s*(.+?\.(?:png|jpg|jpeg|gif))'
            images = re.findall(image_pattern, icl_section, re.IGNORECASE)
        
        # 也匹配 ICL 示例中的图片
        example_pattern = r'图片:\s*(.+?\.(?:png|jpg|jpeg|gif))'
        example_images = re.findall(example_pattern, prompt_content, re.IGNORECASE)
        images.extend(example_images)
        
        # 去重并返回
        return list(set(images))
    
    @staticmethod
    def compare_icl_images(old_content: str, new_content: str) -> Dict:
        """
        比较两个版本的 ICL 图片差异
        
        Returns:
            {
                'added': List[str],  # 新增的图片
                'removed': List[str],  # 删除的图片
                'unchanged': List[str]  # 未改变的图片
            }
        """
        old_images = set(ImageDiffDetector.extract_icl_images(old_content))
        new_images = set(ImageDiffDetector.extract_icl_images(new_content))
        
        return {
            'added': list(new_images - old_images),
            'removed': list(old_images - new_images),
            'unchanged': list(old_images & new_images)
        }
