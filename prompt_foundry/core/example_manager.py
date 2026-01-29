"""
示例管理器 - 处理用户上传的示例图片和说明
"""

import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime


class ExampleManager:
    """管理 ICL 示例图片和说明"""
    
    def __init__(self, prompts_dir: str = "/home/intern10/InSpec/prompt_foundry/prompts"):
        self.prompts_dir = Path(prompts_dir)
    
    def add_example(
        self,
        task_folder: str,
        image_path: str,
        description: str,
        example_id: Optional[int] = None
    ) -> Dict:
        """
        添加一个新的 ICL 示例
        
        Args:
            task_folder: 任务文件夹名称（如 "DCDU安装"）
            image_path: 图片路径（临时路径或本地路径）
            description: 示例说明
            example_id: 示例编号（可选，自动递增）
        
        Returns:
            {
                "success": True/False,
                "image_name": "保存后的图片名称",
                "image_path": "相对路径",
                "example_id": 示例编号
            }
        """
        task_path = self.prompts_dir / task_folder
        icl_dir = task_path / "icl"
        
        # 创建 icl 目录
        icl_dir.mkdir(parents=True, exist_ok=True)
        
        # 确定示例编号
        if example_id is None:
            existing_examples = self._get_existing_examples(task_folder)
            example_id = len(existing_examples) + 1
        
        # 复制图片到 icl 目录
        image_source = Path(image_path)
        if not image_source.exists():
            return {
                "success": False,
                "error": f"图片不存在: {image_path}"
            }
        
        # 生成新的图片名称
        ext = image_source.suffix
        image_name = f"example_{example_id}{ext}"
        image_dest = icl_dir / image_name
        
        # 复制图片
        shutil.copy2(image_source, image_dest)
        
        # 保存示例元数据
        metadata = {
            "example_id": example_id,
            "image_name": image_name,
            "image_path": f"icl/{image_name}",
            "description": description,
            "added_at": datetime.now().isoformat()
        }
        
        self._save_example_metadata(task_folder, metadata)
        
        return {
            "success": True,
            "image_name": image_name,
            "image_path": f"icl/{image_name}",
            "example_id": example_id
        }
    
    def add_multiple_examples(
        self,
        task_folder: str,
        examples: List[Dict[str, str]]
    ) -> List[Dict]:
        """
        批量添加示例
        
        Args:
            task_folder: 任务文件夹名称
            examples: [{"image_path": "...", "description": "..."}, ...]
        
        Returns:
            添加结果列表
        """
        results = []
        for i, example in enumerate(examples, 1):
            result = self.add_example(
                task_folder=task_folder,
                image_path=example["image_path"],
                description=example["description"],
                example_id=i
            )
            results.append(result)
        
        return results
    
    def _get_existing_examples(self, task_folder: str) -> List[Dict]:
        """获取已有的示例"""
        metadata_file = self.prompts_dir / task_folder / "examples_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def _save_example_metadata(self, task_folder: str, metadata: Dict):
        """保存示例元数据"""
        metadata_file = self.prompts_dir / task_folder / "examples_metadata.json"
        
        # 读取现有元数据
        existing = self._get_existing_examples(task_folder)
        
        # 添加新元数据
        existing.append(metadata)
        
        # 保存
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
    
    def get_examples(self, task_folder: str) -> List[Dict]:
        """获取任务的所有示例"""
        return self._get_existing_examples(task_folder)
    
    def format_examples_for_prompt(self, task_folder: str) -> str:
        """
        将示例格式化为 prompt 中的 ICL 部分
        
        Returns:
            格式化后的 ICL 示例文本
        """
        examples = self.get_examples(task_folder)
        if not examples:
            return ""
        
        lines = ["=== ICL示例开始 ==="]
        
        for ex in examples:
            lines.append(f"[示例{ex['example_id']}]")
            lines.append(f"图片: {ex['image_path']}")
            lines.append(f"说明: {ex['description']}")
            lines.append("")
        
        lines.append("=== ICL示例结束 ===")
        
        return "\n".join(lines)
    
    def delete_example(self, task_folder: str, example_id: int) -> bool:
        """删除指定示例"""
        task_path = self.prompts_dir / task_folder
        metadata_file = task_path / "examples_metadata.json"
        
        if not metadata_file.exists():
            return False
        
        # 读取元数据
        examples = self._get_existing_examples(task_folder)
        
        # 查找并删除
        example_to_delete = None
        for i, ex in enumerate(examples):
            if ex['example_id'] == example_id:
                example_to_delete = ex
                examples.pop(i)
                break
        
        if example_to_delete is None:
            return False
        
        # 删除图片文件
        image_path = task_path / example_to_delete['image_path']
        if image_path.exists():
            image_path.unlink()
        
        # 保存更新后的元数据
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(examples, f, ensure_ascii=False, indent=2)
        
        return True
    
    def list_all_tasks(self) -> List[str]:
        """列出所有任务文件夹"""
        tasks = []
        for item in self.prompts_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                tasks.append(item.name)
        return sorted(tasks)
