#!/usr/bin/env python3
"""
遍历指定目录，将最末级文件夹内的图片打包为统一 JSON 结构。
"""

import argparse
import json
import os
from pathlib import Path
from typing import List

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}


def gather_leaf_dirs(root: Path) -> List[Path]:
    """返回所有最末级目录路径"""
    leaf_dirs: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        if not dirnames:
            leaf_dirs.append(Path(dirpath))
    return leaf_dirs


def collect_images(directory: Path) -> List[str]:
    """在目录下收集图片文件"""
    files: List[str] = []
    for file in directory.iterdir():
        if file.is_file() and file.suffix.lower() in IMAGE_EXTS:
            files.append(str(file.resolve()))
    return sorted(files)


def build_entries(root: Path, mode: str = "multi") -> List[dict]:
    """构建最终 JSON 数组
    
    Args:
        root: 根目录路径
        mode: 'single' 为单图模式(每张图一个样本), 'multi' 为多图模式(每个文件夹一个样本)
    """
    entries: List[dict] = []
    for leaf_dir in gather_leaf_dirs(root):
        image_paths = collect_images(leaf_dir)
        if not image_paths:
            continue
        
        if mode == "single":
            # 单图模式: 每张图片生成一个样本
            for image_path in image_paths:
                entry = {
                    "messages": [
                        {"content": "<image>", "role": "user"},
                        {"content": "PASS", "role": "assistant"}
                    ],
                    "images": [image_path]
                }
                entries.append(entry)
        else:
            # 多图模式: 每个文件夹生成一个样本
            entry = {
                "messages": [
                    {"content": "<image>", "role": "user"},
                    {"content": "PASS", "role": "assistant"}
                ],
                "images": image_paths
            }
            entries.append(entry)
    return entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将最末级目录内图片转换为统一 JSON 结构"
    )
    parser.add_argument("--root_dir", nargs="+", required=True, type=str, help="根目录路径（可传入多个路径，用空格分隔）")
    parser.add_argument(
        "-o", "--output", required=True, type=str, help="输出 JSON 文件路径"
    )
    parser.add_argument(
        "--summary", type=str, default=None, help="生成数据说明文件路径（默认为输出文件名_summary.txt）"
    )
    parser.add_argument(
        "-m", "--mode", default="multi", choices=["single", "multi"],
        help="数据模式: single=单图模式(每张图一个样本), multi=多图模式(每个文件夹一个样本)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # 验证所有根目录
    root_paths = []
    for root_dir in args.root_dir:
        root = Path(root_dir).expanduser().resolve()
        if not root.exists() or not root.is_dir():
            raise SystemExit(f"目录无效: {root}")
        root_paths.append(root)
    
    # 从所有根目录收集数据
    all_entries = []
    root_stats = []  # 记录每个根目录的统计信息
    
    for root in root_paths:
        entries = build_entries(root, mode=args.mode)
        all_entries.extend(entries)
        
        # 统计该根目录的图片数量
        image_count = sum(len(entry["images"]) for entry in entries)
        root_stats.append({
            "path": str(root),
            "entries": len(entries),
            "images": image_count
        })
    
    # 保存 JSON 文件
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, ensure_ascii=False, indent=2)
    
    # 计算总体统计
    total_entries = len(all_entries)
    total_images = sum(len(entry["images"]) for entry in all_entries)
    
    # 生成说明文件
    summary_path = args.summary
    if summary_path is None:
        summary_path = output_path.with_name(output_path.stem + "_summary.txt")
    else:
        summary_path = Path(summary_path).expanduser().resolve()
    
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("数据集构建说明\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"输出文件: {output_path}\n")
        f.write(f"数据模式: {args.mode} ({'单图模式' if args.mode == 'single' else '多图模式'})\n")
        f.write(f"生成时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("数据来源:\n")
        f.write("-" * 60 + "\n")
        for i, stat in enumerate(root_stats, 1):
            f.write(f"{i}. {stat['path']}\n")
            f.write(f"   - 数据项数: {stat['entries']}\n")
            f.write(f"   - 图片数: {stat['images']}\n\n")
        
        f.write("总体统计:\n")
        f.write("-" * 60 + "\n")
        f.write(f"总数据项数（每项包含多张图片）: {total_entries}\n")
        f.write(f"总图片数: {total_images}\n")
        f.write(f"平均每项图片数: {total_images / total_entries:.2f}\n" if total_entries > 0 else "平均每项图片数: 0\n")
    
    print(f"✓ 共生成 {total_entries} 条记录，包含 {total_images} 张图片")
    print(f"✓ JSON 数据已保存至: {output_path}")
    print(f"✓ 数据说明已保存至: {summary_path}")


if __name__ == "__main__":
    main()
