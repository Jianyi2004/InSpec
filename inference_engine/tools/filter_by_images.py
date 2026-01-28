#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from typing import Any, Dict, List, Tuple, Optional


CANDIDATE_KEYS = ["data", "samples", "items", "train", "records", "examples"]


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def find_samples_container(obj: Any, key: Optional[str]) -> Tuple[List[Dict], Optional[str]]:
    """
    返回 (samples_list, container_key)
    - 如果顶层是 list：container_key=None
    - 如果顶层是 dict：container_key=实际key
    """
    if isinstance(obj, list):
        # 假设 list 中每个元素是 sample(dict)
        return obj, None

    if not isinstance(obj, dict):
        raise ValueError("Unsupported JSON top-level type (expected list or dict).")

    if key is not None:
        if key not in obj or not isinstance(obj[key], list):
            raise ValueError(f'Key "{key}" not found or not a list in JSON.')
        return obj[key], key

    # 自动猜一个常见 key
    for k in CANDIDATE_KEYS:
        if k in obj and isinstance(obj[k], list):
            return obj[k], k

    # 找不到就报错，让用户指定
    raise ValueError(
        "Top-level is a dict but cannot find a list container automatically. "
        "Please specify --key (e.g., --key data)."
    )


def count_images(sample: Dict) -> int:
    imgs = sample.get("images", [])
    if imgs is None:
        return 0
    if isinstance(imgs, list):
        return len(imgs)
    # 万一有人把 images 写成字符串/其他类型，视为 0（也可改为报错）
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Filter samples by image count (min/max thresholds)."
    )
    parser.add_argument("--in", dest="in_path", required=True, help="Input JSON path")
    parser.add_argument("--out", dest="out_path", required=True, help="Output JSON path")
    parser.add_argument("--max_images", type=int, default=6, help="Max allowed images per sample")
    parser.add_argument("--min_images", type=int, default=1, help="Min required images per sample")
    parser.add_argument(
        "--key",
        type=str,
        default=None,
        help="If top-level JSON is a dict, specify which key holds the samples list (e.g., data/samples/items).",
    )

    args = parser.parse_args()

    if args.max_images is None and args.min_images is None:
        parser.error("At least one of --max_images or --min_images must be specified.")

    obj = load_json(args.in_path)
    samples, container_key = find_samples_container(obj, args.key)

    kept: List[Dict] = []
    dropped: List[Dict] = []

    for s in samples:
        if not isinstance(s, dict):
            # 非法样本直接丢弃（也可以改成 raise）
            dropped.append({"_raw": s, "_reason": "sample_not_dict"})
            continue

        n_img = count_images(s)
        keep = True
        if args.max_images is not None and n_img > args.max_images:
            keep = False
        if args.min_images is not None and n_img < args.min_images:
            keep = False

        if keep:
            kept.append(s)
        else:
            dropped.append(s)

    # 写回原结构
    if container_key is None:
        out_obj: Any = kept
    else:
        out_obj = obj
        out_obj[container_key] = kept

    save_json(out_obj, args.out_path)

    min_str = f"min_images={args.min_images}" if args.min_images is not None else ""
    max_str = f"max_images={args.max_images}" if args.max_images is not None else ""
    filter_str = ", ".join(filter(None, [min_str, max_str]))
    print(f"Done. {filter_str}")
    print(f"Kept: {len(kept)}")
    print(f"Dropped: {len(dropped)}")


if __name__ == "__main__":
    main()
