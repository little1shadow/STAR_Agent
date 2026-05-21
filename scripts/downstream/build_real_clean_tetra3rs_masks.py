#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""批量为真实 clean-data 生成 tetra3rs star pseudo mask。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml


def add_repo_path() -> Path:
    """把 STAR_Agent 根目录加入 `sys.path`。

    输出:
    - STAR_Agent 根目录。
    """

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    return root


def load_yaml(path: str | Path) -> dict:
    """读取 YAML 配置。

    输入:
    - `path`: YAML 文件路径。

    输出:
    - 配置字典。
    """

    p = Path(path)
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Build tetra3rs star pseudo masks for real selected clean images.")
    parser.add_argument("--clean_root", default="data/clean/real_selected_v001")
    parser.add_argument("--recursive", action="store_true", help="Scan nested images, e.g. images/clean_data/*.png.")
    parser.add_argument("--cfg", default="configs/downstream/star_matching/tetra3rs.yaml")
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    """脚本入口。

    功能:
    - 遍历 `clean_root/images` 下的真实 clean 图像。
    - 用 tetra3rs 生成 star pseudo mask、background pseudo mask 和 valid mask。
    - 写入 manifest，供后续真实域 degradation 引用。
    - 真实 clean 阶段不生成 target/proxy；target 会在后续像 synthetic clean 一样单独注入。
    """

    root = add_repo_path()
    from star_agent.downstream.common.image_ops import append_jsonl, ensure_dir, image_files, save_mask, write_json
    from star_agent.downstream.star_matching.tetra3rs_adapter.star_mask import detect_stars_with_tetra3rs

    args = parse_args()
    clean_root = Path(args.clean_root)
    if not clean_root.is_absolute():
        clean_root = (root / clean_root).resolve()
    image_root = clean_root / "images"
    files = image_files(image_root, recursive=args.recursive)
    if args.max_images is not None:
        files = files[: args.max_images]
    if not files:
        raise RuntimeError(f"No images found in {image_root}")

    cfg_path = Path(args.cfg)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    cfg = load_yaml(cfg_path)

    out_dirs = {
        "star_mask": clean_root / "masks" / "star_pseudo_tetra3rs",
        "background_mask": clean_root / "masks" / "background_pseudo_tetra3rs",
        "valid_mask": clean_root / "masks" / "valid_pseudo_tetra3rs",
        "stars_label": clean_root / "labels" / "stars_tetra3rs",
        "metrics": clean_root / "labels" / "tetra3rs_metrics",
    }
    for p in out_dirs.values():
        ensure_dir(p)

    manifest = clean_root / "manifest_tetra3rs_pseudo.jsonl"
    if args.overwrite and manifest.exists():
        manifest.unlink()

    for idx, image_path in enumerate(files, start=1):
        stem = image_path.stem
        star_mask_path = out_dirs["star_mask"] / f"{stem}.png"
        if star_mask_path.exists() and not args.overwrite:
            print(f"[SKIP] {idx}/{len(files)} {stem}: tetra3rs pseudo mask exists")
            continue

        result = detect_stars_with_tetra3rs(image_path, cfg)
        valid_mask = np.ones_like(result.star_mask, dtype=bool)
        background_mask = valid_mask & (~result.star_mask)

        bg_mask_path = out_dirs["background_mask"] / f"{stem}.png"
        valid_mask_path = out_dirs["valid_mask"] / f"{stem}.png"
        stars_label_path = out_dirs["stars_label"] / f"{stem}.json"
        metrics_path = out_dirs["metrics"] / f"{stem}.json"

        save_mask(star_mask_path, result.star_mask)
        save_mask(bg_mask_path, background_mask)
        save_mask(valid_mask_path, valid_mask)
        write_json(stars_label_path, {"mask_source": result.metrics["mask_source"], "stars": result.centroids})
        write_json(metrics_path, result.metrics)

        record = {
            "image_id": stem,
            "image_path": str(image_path),
            "star_mask_path": str(star_mask_path),
            "background_mask_path": str(bg_mask_path),
            "valid_mask_path": str(valid_mask_path),
            "stars_label_path": str(stars_label_path),
            "tetra3rs_metrics_path": str(metrics_path),
            "num_stars": len(result.centroids),
            "has_target": False,
            "target_policy": "not_injected_yet",
            "mask_source": result.metrics["mask_source"],
            "mask_confidence": result.metrics["mask_confidence"],
        }
        append_jsonl(manifest, record)
        print(
            f"[OK] {idx}/{len(files)} {stem} | stars={len(result.centroids)} "
            f"mask_area={result.metrics.get('star_mask_area_ratio', 0.0):.5f}"
        )

    print(f"[DONE] tetra3rs pseudo manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
