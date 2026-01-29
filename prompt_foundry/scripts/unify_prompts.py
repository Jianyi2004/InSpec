#!/usr/bin/env python3
"""
统一 Prompt 格式脚本

将分散的 system_prompt.txt、prompt.txt、summary.txt 整合成一个完整的 prompt 文件
新格式：
- 每个条目一个文件夹
- 文件夹下包含：
  - prompt.txt (完整的 prompt，包含 system、main、summary)
  - icl/ (示例图片文件夹)
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional

class PromptUnifier:
    def __init__(self, prompts_dir: str):
        self.prompts_dir = Path(prompts_dir)
        
    def unify_prompt_folder(self, folder_path: Path) -> Optional[Dict]:
        """
        统一一个 prompt 文件夹
        
        返回：
        {
            "task_name": "任务名称",
            "system_prompt": "系统提示词",
            "main_prompt": "主要提示词",
            "summary_prompt": "总结提示词",
            "icl_images": ["image1.png", "image2.jpg"]
        }
        """
        prompts_subdir = folder_path / "prompts"
        if not prompts_subdir.exists():
            print(f"⚠️  {folder_path.name}: 没有 prompts 子目录")
            return None
        
        # 查找文件
        system_prompt_file = None
        main_prompt_file = None
        summary_prompt_file = None
        
        for file in prompts_subdir.glob("*.txt"):
            filename = file.name.lower()
            if "system" in filename:
                system_prompt_file = file
            elif "summary" in filename:
                summary_prompt_file = file
            elif "prompt" in filename:
                main_prompt_file = file
        
        # 读取内容
        system_prompt = self._read_file(system_prompt_file) if system_prompt_file else ""
        main_prompt = self._read_file(main_prompt_file) if main_prompt_file else ""
        summary_prompt = self._read_file(summary_prompt_file) if summary_prompt_file else ""
        
        # 提取任务名称
        task_name = self._extract_task_name(main_prompt) or folder_path.name
        
        # 查找 ICL 图片
        icl_dir = folder_path / "icl"
        icl_images = []
        if icl_dir.exists():
            icl_images = [f.name for f in icl_dir.glob("*") if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
        
        return {
            "task_name": task_name,
            "folder_name": folder_path.name,
            "system_prompt": system_prompt,
            "main_prompt": main_prompt,
            "summary_prompt": summary_prompt,
            "icl_images": sorted(icl_images)
        }
    
    def _read_file(self, file_path: Path) -> str:
        """读取文件内容"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            print(f"⚠️  读取文件失败 {file_path}: {e}")
            return ""
    
    def _extract_task_name(self, prompt_text: str) -> Optional[str]:
        """从 prompt 中提取任务名称"""
        import re
        match = re.search(r'=== PROMPT名字开始 ===\s*\n(.+?)\n=== PROMPT名字结束 ===', prompt_text)
        if match:
            return match.group(1).strip()
        return None
    
    def create_unified_prompt(self, data: Dict) -> str:
        """
        创建统一格式的 prompt
        
        格式：
        =============================================================================
        TASK: [任务名称]
        =============================================================================
        
        [SYSTEM PROMPT]
        系统提示词内容...
        
        [MAIN PROMPT]
        主要提示词内容...
        
        [SUMMARY PROMPT]
        总结提示词内容...
        
        [ICL IMAGES]
        - icl/image1.png
        - icl/image2.jpg
        =============================================================================
        """
        lines = []
        lines.append("=" * 80)
        lines.append(f"TASK: {data['task_name']}")
        lines.append("=" * 80)
        lines.append("")
        
        if data['system_prompt']:
            lines.append("[SYSTEM PROMPT]")
            lines.append(data['system_prompt'])
            lines.append("")
        
        if data['main_prompt']:
            lines.append("[MAIN PROMPT]")
            lines.append(data['main_prompt'])
            lines.append("")
        
        if data['summary_prompt']:
            lines.append("[SUMMARY PROMPT]")
            lines.append(data['summary_prompt'])
            lines.append("")
        
        if data['icl_images']:
            lines.append("[ICL IMAGES]")
            for img in data['icl_images']:
                lines.append(f"- icl/{img}")
            lines.append("")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def process_all(self, output_dir: Optional[str] = None):
        """处理所有 prompt 文件夹"""
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = self.prompts_dir
        
        results = []
        
        for folder in self.prompts_dir.iterdir():
            if not folder.is_dir():
                continue
            
            print(f"\n📁 处理: {folder.name}")
            
            data = self.unify_prompt_folder(folder)
            if not data:
                continue
            
            # 生成统一的 prompt
            unified_prompt = self.create_unified_prompt(data)
            
            # 保存到原文件夹
            output_file = folder / "prompt.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(unified_prompt)
            
            print(f"✅ 已生成: {output_file}")
            print(f"   - System Prompt: {'✓' if data['system_prompt'] else '✗'}")
            print(f"   - Main Prompt: {'✓' if data['main_prompt'] else '✗'}")
            print(f"   - Summary Prompt: {'✓' if data['summary_prompt'] else '✗'}")
            print(f"   - ICL Images: {len(data['icl_images'])} 张")
            
            results.append(data)
        
        # 生成索引文件
        index_file = output_path / "index.json"
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n📋 索引文件已生成: {index_file}")
        print(f"✅ 共处理 {len(results)} 个 prompt")
        
        return results


def main():
    import sys
    
    if len(sys.argv) > 1:
        prompts_dir = sys.argv[1]
    else:
        prompts_dir = "/home/intern10/InSpec/prompt_foundry/prompts"
    
    print(f"🚀 开始统一 Prompt 格式...")
    print(f"📂 目录: {prompts_dir}\n")
    
    unifier = PromptUnifier(prompts_dir)
    results = unifier.process_all()
    
    print(f"\n🎉 完成！")


if __name__ == "__main__":
    main()
