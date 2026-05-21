#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""批量为真实 clean-data 生成 star/target/background/valid 信息。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image


def add_repo_path() -> Path:
    """把 STAR_Agent 根目录加入 `sys.path`。

    输入:
    - 无。

    输出:
    - STAR_Agent 根目录。
    """

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    return root


def load_yaml(path: str | Path) -> dict[str, Any]:
    """读取 YAML 配置。

    输入:
    - `path`: YAML 文件路径。

    输出:
    - 配置字典。
    """

    p = Path(path)
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    输入:
    - 命令行参数。

    输出:
    - argparse Namespace。
    """

    parser = argparse.ArgumentParser(
        description="Build tetra3rs star masks and inject targets for real selected clean images."
    )
    parser.add_argument("--clean_root", default="data/clean/real_selected_v001")
    parser.add_argument("--recursive", action="store_true", help="Scan nested images, e.g. images/clean_data/*.png.")
    parser.add_argument("--cfg", default="configs/downstream/star_matching/tetra3rs.yaml")
    parser.add_argument("--target_cfg", default="configs/downstream/real_clean_target_injection.yaml")
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument(
        "--target_image_count",
        type=int,
        default=None,
        help="Override absolute target image count. If omitted, use --target_image_ratio or target_cfg selection.target_image_ratio.",
    )
    parser.add_argument(
        "--target_image_ratio",
        type=float,
        default=None,
        help="Override target image ratio, default from target_cfg. Example: 0.8 means 80%% images have targets.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override target selection seed, default from target_cfg.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def save_float_image_like_reference(path: str | Path, image: np.ndarray, reference_path: str | Path) -> None:
    """按参考图动态范围保存注入目标后的真实 clean 图。

    输入:
    - `path`: 输出图像路径。
    - `image`: HxW float 图像，已经叠加目标。
    - `reference_path`: 原始真实 clean 图，用于判断保存为 uint8 还是 uint16。

    输出:
    - 无。

    设计目的:
    - 按用户要求直接覆盖真实 clean 原图；
    - 尽量保持真实图原本的数据位深，避免后续 degradation 因动态范围变化而失真。
    """

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ref_img = Image.open(reference_path)
    ref = np.asarray(ref_img)
    if ref.dtype == np.uint16 or "16" in ref_img.mode:
        arr = np.clip(image, 0, 65535).astype(np.uint16)
        Image.fromarray(arr).save(out_path)
    else:
        arr = np.clip(image, 0, 255).astype(np.uint8)
        Image.fromarray(arr, mode="L").save(out_path)


def choose_target_image_indices(num_images: int, target_count: int, seed: int) -> set[int]:
    """确定哪些真实 clean 图需要注入目标。

    输入:
    - `num_images`: 当前参与处理的图像数量。
    - `target_count`: 需要注入目标的图像数量。
    - `seed`: 随机种子。

    输出:
    - 需要注入目标的 0-based 图像索引集合。

    设计目的:
    - 用户要求真实 clean 按比例选择含目标图，当前默认 80%；
    - 这里用固定 seed 保证多次运行选中的图一致，便于复现实验。
    """

    count = max(0, min(int(target_count), int(num_images)))
    if count == 0:
        return set()
    rng = np.random.default_rng(seed)
    selected = rng.choice(num_images, size=count, replace=False)
    return {int(i) for i in selected.tolist()}


def resolve_target_image_count(
    num_images: int,
    explicit_count: int | None,
    explicit_ratio: float | None,
    selection_cfg: dict[str, Any],
) -> tuple[int, float]:
    """根据显式参数或配置计算含目标图像数量。

    输入:
    - `num_images`: 当前参与处理的图像数量。
    - `explicit_count`: 命令行指定的绝对数量，优先级最高。
    - `explicit_ratio`: 命令行指定比例。
    - `selection_cfg`: YAML selection 配置。

    输出:
    - `(target_count, target_ratio)`。

    说明:
    - 默认使用 `selection.target_image_ratio=0.8`；
    - 如果旧配置里仍有 `target_image_count`，只有在没有 ratio 时才使用。
    """

    if explicit_count is not None:
        count = int(explicit_count)
        ratio = count / max(1, num_images)
    else:
        ratio_value = explicit_ratio
        if ratio_value is None:
            ratio_value = selection_cfg.get("target_image_ratio", None)
        if ratio_value is not None:
            ratio = float(ratio_value)
            count = int(round(num_images * ratio))
        else:
            count = int(selection_cfg.get("target_image_count", round(num_images * 0.8)))
            ratio = count / max(1, num_images)
    count = max(0, min(int(count), int(num_images)))
    ratio = float(count / max(1, num_images))
    return count, ratio


def centroid_arrays(centroids: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """从 tetra3rs 质心记录中取出星点 x/y 数组。

    输入:
    - `centroids`: `detect_stars_with_tetra3rs` 输出的星点记录。

    输出:
    - `(star_x, star_y)` 两个 float32 数组。
    """

    if not centroids:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)
    star_x = np.asarray([item.get("x", 0.0) for item in centroids], dtype=np.float32)
    star_y = np.asarray([item.get("y", 0.0) for item in centroids], dtype=np.float32)
    return star_x, star_y


def target_type_counts(targets: list[dict[str, Any]]) -> dict[str, int]:
    """统计目标类型数量。

    输入:
    - `targets`: 注入目标标签列表。

    输出:
    - `point_blob` 和 `short_streak` 的数量统计。
    """

    return {
        "point_blob": sum(1 for item in targets if item.get("target_type") == "point_blob"),
        "short_streak": sum(1 for item in targets if item.get("target_type") == "short_streak"),
    }


def main() -> int:
    """脚本入口。

    功能:
    - 遍历 `clean_root/images` 下的真实 clean 图像。
    - 用 tetra3rs 生成真实星点 pseudo mask。
    - 固定选择 80% 图像注入 1-5 个目标，目标类型比例为短条纹:点状 = 7:3。
    - 保存带目标真实 clean 图、star mask、target mask、background mask、valid mask 和标签。

    输出设计:
    - 按用户要求直接覆盖 `images/` 下原始真实 clean 图；
    - masks/labels 目录尽量和 synthetic clean 保持一致；
    - `background_mask = valid_mask - star_mask - target_mask`；
    - manifest 记录所有 mask/label 路径，方便后续 degradation 溯源。
    """

    root = add_repo_path()
    from star_agent.data_generation.clean_simulation.target_injector import build_target_config, inject_targets
    from star_agent.downstream.common.image_ops import append_jsonl, ensure_dir, image_files, save_mask, write_json
    from star_agent.downstream.star_matching.tetra3rs_adapter.star_mask import detect_stars_with_tetra3rs, read_image_for_tetra3rs

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

    target_cfg_path = Path(args.target_cfg)
    if not target_cfg_path.is_absolute():
        target_cfg_path = root / target_cfg_path
    target_raw_cfg = load_yaml(target_cfg_path)
    selection_cfg = target_raw_cfg.get("selection", {}) or {}
    output_cfg = target_raw_cfg.get("outputs", {}) or {}
    seed = int(args.seed if args.seed is not None else selection_cfg.get("seed", 131))
    target_image_count, target_image_ratio = resolve_target_image_count(
        num_images=len(files),
        explicit_count=args.target_image_count,
        explicit_ratio=args.target_image_ratio,
        selection_cfg=selection_cfg,
    )
    selected_target_indices = choose_target_image_indices(len(files), target_image_count, seed)
    injector_cfg = build_target_config(target_raw_cfg.get("targets", {}) or {})

    manifest_name = str(output_cfg.get("manifest_name", "manifest.jsonl"))
    summary_name = str(output_cfg.get("summary_name", "real_clean_summary.json"))
    out_dirs = {
        "star_mask": clean_root / "masks" / "star",
        "target_mask": clean_root / "masks" / "target",
        "background_mask": clean_root / "masks" / "background",
        "valid_mask": clean_root / "masks" / "valid",
        "stars_label": clean_root / "labels" / "stars",
        "targets_label": clean_root / "labels" / "targets",
        "metrics": clean_root / "labels" / "tetra3rs_metrics",
    }
    for p in out_dirs.values():
        ensure_dir(p)

    manifest = clean_root / manifest_name
    if args.overwrite and manifest.exists():
        manifest.unlink()

    total_targets = 0
    total_target_images = 0
    type_counts_total = {"point_blob": 0, "short_streak": 0}
    print(
        f"[INFO] images={len(files)} | target_images={len(selected_target_indices)} "
        f"| target_ratio={target_image_ratio:.3f} "
        f"| seed={seed} | target_cfg={target_cfg_path}"
    )

    for idx, image_path in enumerate(files, start=1):
        zero_idx = idx - 1
        stem = image_path.stem
        star_mask_path = out_dirs["star_mask"] / f"{stem}.png"
        target_mask_path = out_dirs["target_mask"] / f"{stem}.png"
        if star_mask_path.exists() and target_mask_path.exists() and not args.overwrite:
            print(f"[SKIP] {idx}/{len(files)} {stem}: outputs exist")
            continue

        source_image_path_before_overwrite = str(image_path)
        result = detect_stars_with_tetra3rs(image_path, cfg)
        valid_mask = np.ones_like(result.star_mask, dtype=bool)
        star_x, star_y = centroid_arrays(result.centroids)
        original_image = read_image_for_tetra3rs(image_path)

        has_target = zero_idx in selected_target_indices
        if has_target:
            rng = np.random.default_rng(seed + zero_idx * 1009)
            image_with_targets, target_mask_u8, target_records = inject_targets(
                original_image,
                star_x=star_x,
                star_y=star_y,
                config=injector_cfg,
                rng=rng,
            )
        else:
            image_with_targets = original_image.astype(np.float32, copy=True)
            target_mask_u8 = np.zeros_like(result.star_mask, dtype=np.uint8)
            target_records = []

        target_mask = target_mask_u8 > 0
        background_mask = valid_mask & (~result.star_mask) & (~target_mask)
        bg_mask_path = out_dirs["background_mask"] / f"{stem}.png"
        valid_mask_path = out_dirs["valid_mask"] / f"{stem}.png"
        stars_label_path = out_dirs["stars_label"] / f"{stem}.json"
        targets_label_path = out_dirs["targets_label"] / f"{stem}.json"
        metrics_path = out_dirs["metrics"] / f"{stem}.json"

        # 用户明确要求直接在真实 clean 原图上添加目标，因此这里覆盖 image_path。
        save_float_image_like_reference(image_path, image_with_targets, image_path)
        save_mask(star_mask_path, result.star_mask)
        save_mask(target_mask_path, target_mask)
        save_mask(bg_mask_path, background_mask)
        save_mask(valid_mask_path, valid_mask)
        write_json(stars_label_path, {"mask_source": result.metrics["mask_source"], "stars": result.centroids})
        write_json(
            targets_label_path,
            {
                "mask_source": "real_clean_target_injection_v001",
                "source_image_path_before_overwrite": source_image_path_before_overwrite,
                "image_path": str(image_path),
                "has_target": bool(has_target),
                "num_targets": len(target_records),
                "target_type_counts": target_type_counts(target_records),
                "targets": target_records,
            },
        )
        metrics = {
            **result.metrics,
            "source_image_path_before_overwrite": source_image_path_before_overwrite,
            "image_path": str(image_path),
            "target_mask_area_px": int(target_mask.sum()),
            "target_mask_area_ratio": float(target_mask.sum() / max(1, target_mask.size)),
            "background_mask_area_px": int(background_mask.sum()),
            "background_mask_area_ratio": float(background_mask.sum() / max(1, background_mask.size)),
            "has_target": bool(has_target),
            "num_targets": len(target_records),
            "target_type_counts": target_type_counts(target_records),
            "target_policy": "injected_real_clean_ratio_0.8_short_streak_0.7_point_blob_0.3",
        }
        write_json(metrics_path, metrics)

        type_counts = target_type_counts(target_records)
        total_targets += len(target_records)
        total_target_images += int(has_target)
        type_counts_total["point_blob"] += type_counts["point_blob"]
        type_counts_total["short_streak"] += type_counts["short_streak"]

        record = {
            "image_id": stem,
            "source_image_path_before_overwrite": source_image_path_before_overwrite,
            "image_path": str(image_path),
            "star_mask_path": str(star_mask_path),
            "target_mask_path": str(target_mask_path),
            "background_mask_path": str(bg_mask_path),
            "valid_mask_path": str(valid_mask_path),
            "stars_label_path": str(stars_label_path),
            "targets_label_path": str(targets_label_path),
            "tetra3rs_metrics_path": str(metrics_path),
            "num_stars": len(result.centroids),
            "has_target": bool(has_target),
            "num_targets": len(target_records),
            "target_type_counts": type_counts,
            "target_policy": "injected_real_clean_ratio_0.8_short_streak_0.7_point_blob_0.3",
            "mask_source": result.metrics["mask_source"],
            "mask_confidence": result.metrics["mask_confidence"],
        }
        append_jsonl(manifest, record)
        print(
            f"[OK] {idx}/{len(files)} {stem} | stars={len(result.centroids)} "
            f"targets={len(target_records)} types={type_counts} "
            f"star_area={result.metrics.get('star_mask_area_ratio', 0.0):.5f} "
            f"target_area={metrics['target_mask_area_ratio']:.5f}"
        )

    summary_path = clean_root / summary_name
    write_json(
        summary_path,
        {
            "num_images": len(files),
            "target_image_count_requested": target_image_count,
            "target_image_count_actual": total_target_images,
            "target_image_ratio_requested": target_image_ratio,
            "target_image_ratio_actual": float(total_target_images / max(1, len(files))),
            "total_targets": total_targets,
            "target_type_counts": type_counts_total,
            "manifest_path": str(manifest),
            "target_cfg_path": str(target_cfg_path),
            "seed": seed,
        },
    )
    print(f"[DONE] manifest: {manifest}")
    print(f"[DONE] summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
