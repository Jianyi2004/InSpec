"""
为现有版本补充拆分的 prompt 文件
将已有版本的 prompt.txt 拆分成 system_prompt.txt、main_prompt.txt、summary_prompt.txt
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import re


def split_prompt_content(prompt_content: str, version_dir: Path):
    """
    拆分 prompt 内容并保存到各个文件
    
    Args:
        prompt_content: 完整的 prompt 内容
        version_dir: 版本目录
    """
    # 提取 SYSTEM PROMPT - 从 [SYSTEM PROMPT] 到下一个 [XXX PROMPT] 或结束
    system_match = re.search(r'\[SYSTEM PROMPT\]\s*\n(.*?)(?=\n\[(?:MAIN|SUMMARY|ICL) PROMPT\]|={80}|\Z)', prompt_content, re.DOTALL)
    if system_match:
        system_prompt = system_match.group(1).strip()
        system_file = version_dir / 'system_prompt.txt'
        system_file.write_text(system_prompt, encoding='utf-8')
        print(f"    ✅ 创建 system_prompt.txt ({len(system_prompt)} 字符)")
    
    # 提取 MAIN PROMPT - 从 [MAIN PROMPT] 到下一个 [XXX PROMPT] 或结束
    main_match = re.search(r'\[MAIN PROMPT\]\s*\n(.*?)(?=\n\[(?:SUMMARY|ICL) PROMPT\]|={80}|\Z)', prompt_content, re.DOTALL)
    if main_match:
        main_prompt = main_match.group(1).strip()
        main_file = version_dir / 'main_prompt.txt'
        main_file.write_text(main_prompt, encoding='utf-8')
        print(f"    ✅ 创建 main_prompt.txt ({len(main_prompt)} 字符)")
    
    # 提取 SUMMARY PROMPT - 从 [SUMMARY PROMPT] 到 [ICL IMAGES] 或结束
    summary_match = re.search(r'\[SUMMARY PROMPT\]\s*\n(.*?)(?=\n\[ICL (?:IMAGES|PROMPT)\]|={80}|\Z)', prompt_content, re.DOTALL)
    if summary_match:
        summary_prompt = summary_match.group(1).strip()
        summary_file = version_dir / 'summary_prompt.txt'
        summary_file.write_text(summary_prompt, encoding='utf-8')
        print(f"    ✅ 创建 summary_prompt.txt ({len(summary_prompt)} 字符)")


def migrate_task_versions(task_dir: Path):
    """
    迁移某个任务的所有版本
    
    Args:
        task_dir: 任务目录
    """
    task_name = task_dir.name
    versions_dir = task_dir / 'versions'
    
    if not versions_dir.exists():
        return
    
    print(f"\n处理任务: {task_name}")
    
    # 遍历所有版本文件夹
    version_folders = sorted([d for d in versions_dir.iterdir() if d.is_dir()])
    
    if not version_folders:
        print("  暂无版本")
        return
    
    for version_dir in version_folders:
        print(f"  版本: {version_dir.name}")
        
        # 检查是否已经有拆分的文件
        if (version_dir / 'system_prompt.txt').exists():
            print("    ⏭️  已存在拆分文件，跳过")
            continue
        
        # 读取 prompt.txt
        prompt_file = version_dir / 'prompt.txt'
        if not prompt_file.exists():
            print("    ❌ prompt.txt 不存在，跳过")
            continue
        
        try:
            prompt_content = prompt_file.read_text(encoding='utf-8')
            split_prompt_content(prompt_content, version_dir)
        except Exception as e:
            print(f"    ❌ 处理失败: {e}")


def main():
    """主函数"""
    print("=" * 60)
    print("为现有版本补充拆分的 prompt 文件")
    print("=" * 60)
    
    prompts_dir = Path(__file__).parent.parent / 'prompts'
    
    if not prompts_dir.exists():
        print("❌ prompts 目录不存在")
        return
    
    # 遍历所有任务
    task_dirs = sorted([d for d in prompts_dir.iterdir() if d.is_dir() and not d.name.startswith('_')])
    
    print(f"\n找到 {len(task_dirs)} 个任务")
    
    for task_dir in task_dirs:
        migrate_task_versions(task_dir)
    
    print("\n" + "=" * 60)
    print("✅ 迁移完成！")
    print("=" * 60)
    print("\n现在每个版本都包含以下文件：")
    print("1. prompt.txt - 完整的 prompt")
    print("2. system_prompt.txt - 系统提示词")
    print("3. main_prompt.txt - 主提示词")
    print("4. summary_prompt.txt - 总结提示词")
    print("5. version_info.json - 版本信息")
    print("6. icl/ - 示例图片目录")
    print("\n")


if __name__ == '__main__':
    main()
