"""
Prompt 解析器 - 解析统一格式的 prompt 文件
"""

import re
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class PromptStructure:
    """Prompt 结构化数据"""
    task_name: str
    system_prompt: str
    main_prompt: str
    summary_prompt: str
    icl_images: List[str]
    raw_text: str


class PromptParser:
    """解析统一格式的 prompt 文件"""
    
    def parse_file(self, file_path: str) -> PromptStructure:
        """解析 prompt 文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return self.parse(content)
    
    def parse(self, prompt_text: str) -> PromptStructure:
        """
        解析 prompt 文本
        
        格式：
        ================================================================================
        TASK: [任务名称]
        ================================================================================
        
        [SYSTEM PROMPT]
        ...
        
        [MAIN PROMPT]
        ...
        
        [SUMMARY PROMPT]
        ...
        
        [ICL IMAGES]
        - icl/image1.png
        - icl/image2.jpg
        ================================================================================
        """
        # 提取任务名称
        task_match = re.search(r'TASK:\s*(.+)', prompt_text)
        task_name = task_match.group(1).strip() if task_match else "Unknown"
        
        # 提取各部分
        system_prompt = self._extract_section(prompt_text, "SYSTEM PROMPT")
        main_prompt = self._extract_section(prompt_text, "MAIN PROMPT")
        summary_prompt = self._extract_section(prompt_text, "SUMMARY PROMPT")
        
        # 提取 ICL 图片列表
        icl_images = self._extract_icl_images(prompt_text)
        
        return PromptStructure(
            task_name=task_name,
            system_prompt=system_prompt,
            main_prompt=main_prompt,
            summary_prompt=summary_prompt,
            icl_images=icl_images,
            raw_text=prompt_text
        )
    
    def _extract_section(self, text: str, section_name: str) -> str:
        """提取指定章节的内容"""
        pattern = rf'\[{section_name}\]\s*\n(.*?)(?=\n\[|={80}|\Z)'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""
    
    def _extract_icl_images(self, text: str) -> List[str]:
        """提取 ICL 图片列表"""
        images = []
        icl_section = self._extract_section(text, "ICL IMAGES")
        if icl_section:
            for line in icl_section.split('\n'):
                line = line.strip()
                if line.startswith('- '):
                    images.append(line[2:].strip())
        return images
    
    def extract_main_prompt_sections(self, main_prompt: str) -> Dict[str, str]:
        """
        解析 main prompt 的各个部分
        
        Returns:
            {
                "task_title": "任务标题",
                "role_description": "角色描述",
                "core_concepts": "核心概念",
                "icl_examples": "ICL示例",
                "judgment_flow": "判断流程",
                "output_format": "输出格式"
            }
        """
        sections = {}
        
        # 提取任务标题
        title_match = re.search(r'=== PROMPT名字开始 ===\s*\n(.+?)\n=== PROMPT名字结束 ===', main_prompt)
        if title_match:
            sections["task_title"] = title_match.group(1).strip()
        
        # 提取 ICL 示例
        icl_match = re.search(r'=== ICL示例开始 ===(.*?)=== ICL示例结束 ===', main_prompt, re.DOTALL)
        if icl_match:
            sections["icl_examples"] = icl_match.group(1).strip()
        
        # 提取判断流程
        flow_match = re.search(r'#+\s*判断流程(.*?)(?=#+\s*输出格式|\Z)', main_prompt, re.DOTALL)
        if flow_match:
            sections["judgment_flow"] = flow_match.group(1).strip()
        
        # 提取输出格式
        output_match = re.search(r'#+\s*输出格式(.*)', main_prompt, re.DOTALL)
        if output_match:
            sections["output_format"] = output_match.group(1).strip()
        
        return sections
