#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""在已有 clean_data 上追加空间小目标。

该脚本不重新生成星表星场，而是读取已有 clean 图、星点标签和 manifest，
在原始 clean 图基础上额外注入 point_blob / short_streak 目标，输出为新的 clean dataset。

使用场景:
- 已经生成了无目标或空 target mask 的 clean_data；
- 希望保持星场、背景和星点标签完全不变；
- 只为下游小目标检测反馈补充 target mask 和 target label。
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from ..common.image_ops import save_gray_png, save_rgb_png
from ..common.io import append_jsonl, ensure_dir, write_json
from ..common.progress import ProgressPrinter
from .build_clean_data import load_config
from .target_injector import build_target_config, inject_targets


DATA_SUBDIRS = {
    "images": ("images",),
    "mask_star": ("masks", "star"),
    "mask_target": ("masks", "target"),
    "mask_background": ("masks", "background"),
    "mask_valid": ("masks", "valid"),
    "labels_stars": ("labels", "stars"),
    "labels_targets": ("labels", "targets"),
    "labels_camera": ("labels", "camera"),
    "splits": ("splits",),
}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """读取 JSONL manifest。

    输入:
    - `path`: manifest 路径。

    输出:
    - 每一行 JSON 记录组成的列表。
    """

    records: list[dict[str, Any]] = []
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def make_output_dirs(root_dir: Path) -> dict[str, Path]:
    """创建目标注入后的数据集目录。

    输入:
    - `root_dir`: 输出数据集根目录。

    输出:
    - 目录名到 Path 的映射。
    """

    dirs: dict[str, Path] = {}
    for name, parts in DATA_SUBDIRS.items():
        dirs[name] = ensure_dir(root_dir.joinpath(*parts))
    return dirs


def resolve_existing_path(value: str | None, source_root: Path, repo_root: Path) -> Path | None:
    """把 manifest 中的路径解析为真实存在的路径。

    输入:
    - `value`: manifest 字段中的路径字符串。
    - `source_root`: 源 clean_data 根目录。
    - `repo_root`: 当前项目根目录。

    输出:
    - 存在的 Path；若字段为空则返回 None。

    说明:
    - 旧 manifest 里通常是相对仓库根目录的路径；
    - 这里兼容绝对路径、相对仓库路径，以及相对 source_root 的路径。
    """

    if not value:
        return None
    raw = Path(value)
    candidates = [raw]
    if not raw.is_absolute():
        candidates.append(repo_root / raw)
        candidates.append(source_root / raw)
        candidates.append(source_root / raw.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def copy_if_exists(src: Path | None, dst: Path) -> str | None:
    """复制文件并返回输出路径。

    输入:
    - `src`: 源文件路径，允许为空。
    - `dst`: 目标文件路径。

    输出:
    - 若复制成功，返回目标路径字符串；否则返回 None。
    """

    if src is None or not src.exists():
        return None
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return str(dst)


def load_clean_image(path: Path) -> np.ndarray:
    """读取 clean 图为 float32 单通道。

    输入:
    - `path`: clean 图路径。

    输出:
    - HxW float32 图像。

    说明:
    - 当前 pipeline 保存的是 RGB PNG；三通道内容相同；
    - 目标注入只需要单通道亮度，保存时再复制为 RGB。
    """

    image = Image.open(path).convert("L")
    return np.asarray(image, dtype=np.float32)


def load_star_positions(stars_label_path: Path | None) -> tuple[np.ndarray, np.ndarray]:
    """从星点标签中读取像素坐标。

    输入:
    - `stars_label_path`: `labels/stars/*.json` 路径。

    输出:
    - `(x_array, y_array)`，均为 float32。

    说明:
    - 如果没有星点标签，则返回空数组；
    - 目标注入仍可进行，但无法避开已有星点。
    """

    if stars_label_path is None or not stars_label_path.exists():
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)
    data = json.loads(stars_label_path.read_text(encoding="utf-8"))
    stars = data.get("stars", [])
    xs = [float(item.get("x_px", 0.0)) for item in stars]
    ys = [float(item.get("y_px", 0.0)) for item in stars]
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def write_splits_from_source(source_root: Path, output_root: Path, selected_ids: set[str]) -> None:
    """根据源数据集 split 写出新数据集 split。

    输入:
    - `source_root`: 源 clean_data 根目录。
    - `output_root`: 输出数据集根目录。
    - `selected_ids`: 本次实际处理的 image_id 集合。

    输出:
    - 无返回值。

    说明:
    - 如果源数据集已有 train/val/test，则保留交集；
    - 如果没有 split，则默认全部写入 train。
    """

    src_split = source_root / "splits"
    dst_split = ensure_dir(output_root / "splits")
    found = False
    for split_name in ["train", "val", "test"]:
        src_file = src_split / f"{split_name}.txt"
        dst_file = dst_split / f"{split_name}.txt"
        if src_file.exists():
            found = True
            ids = [line.strip() for line in src_file.read_text(encoding="utf-8").splitlines()]
            kept = [item for item in ids if item in selected_ids]
            dst_file.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
        else:
            dst_file.write_text("", encoding="utf-8")
    if not found:
        ids = sorted(selected_ids)
        (dst_split / "train.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
        (dst_split / "val.txt").write_text("", encoding="utf-8")
        (dst_split / "test.txt").write_text("", encoding="utf-8")


def add_targets_to_clean_dataset(
    cfg: dict[str, Any],
    source_dir: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path | None = None,
    max_images: int | None = None,
    seed_override: int | None = None,
) -> dict[str, Any]:
    """在已有 clean_data 上批量注入目标。

    输入:
    - `cfg`: YAML 配置字典，使用其中 `targets` 和 `generation.seed`。
    - `source_dir`: 源 clean_data 根目录。
    - `output_dir`: 输出 clean_data 根目录。
    - `manifest_path`: 可选源 manifest 路径，默认 `source_dir/manifest.jsonl`。
    - `max_images`: 可选，只处理前 N 张，方便预览。
    - `seed_override`: 可选随机种子。

    输出:
    - 生成摘要字典。
    """

    repo_root = Path.cwd()
    source_root = Path(source_dir)
    output_root = Path(output_dir)
    source_manifest = Path(manifest_path) if manifest_path else source_root / "manifest.jsonl"
    records = read_jsonl(source_manifest)
    if max_images is not None:
        records = records[: max(0, int(max_images))]

    target_cfg = build_target_config(cfg.get("targets", {}))
    seed = int(seed_override if seed_override is not None else cfg.get("generation", {}).get("seed", 131))
    rng = np.random.default_rng(seed)
    dirs = make_output_dirs(output_root)

    # 数据集级别精确控制无目标比例。这样 2000 张图时可以稳定得到约 400 张无目标图，
    # 避免每张图独立 Bernoulli 采样造成比例波动。
    no_target_total = int(round(max(0.0, 1.0 - target_cfg.image_probability) * len(records)))
    no_target_indices = set()
    if no_target_total > 0:
        no_target_indices = set(
            int(item) for item in rng.choice(len(records), size=no_target_total, replace=False)
        )
    target_cfg_with_target = replace(target_cfg, image_probability=1.0)
    target_cfg_without_target = replace(target_cfg, image_probability=0.0)

    output_manifest = output_root / "manifest.jsonl"
    output_manifest.write_text("", encoding="utf-8")

    progress = ProgressPrinter(len(records), stage="add_targets_to_clean_data")
    selected_ids: set[str] = set()
    target_type_counts = {"point_blob": 0, "short_streak": 0}
    target_count_hist: dict[str, int] = {}
    no_target_count = 0
    total_targets = 0

    for idx, record in enumerate(records):
        image_id = str(record.get("image_id") or f"clean_{idx:06d}")
        selected_ids.add(image_id)
        src_image = resolve_existing_path(record.get("image_path"), source_root, repo_root)
        if src_image is None:
            raise FileNotFoundError(f"image_path not found for {image_id}: {record.get('image_path')}")

        src_star_label = resolve_existing_path(record.get("stars_label_path"), source_root, repo_root)
        src_camera_label = resolve_existing_path(record.get("camera_label_path"), source_root, repo_root)
        src_star_mask = resolve_existing_path(record.get("star_mask_path"), source_root, repo_root)
        src_background_mask = resolve_existing_path(record.get("background_mask_path"), source_root, repo_root)
        src_valid_mask = resolve_existing_path(record.get("valid_mask_path"), source_root, repo_root)

        image = load_clean_image(src_image)
        star_x, star_y = load_star_positions(src_star_label)
        current_target_cfg = (
            target_cfg_without_target if idx in no_target_indices else target_cfg_with_target
        )
        image_with_targets, target_mask, target_records = inject_targets(
            image=image,
            star_x=star_x,
            star_y=star_y,
            config=current_target_cfg,
            rng=rng,
        )

        out_image_path = dirs["images"] / f"{image_id}.png"
        out_target_mask_path = dirs["mask_target"] / f"{image_id}.png"
        out_star_mask_path = dirs["mask_star"] / f"{image_id}.png"
        out_background_mask_path = dirs["mask_background"] / f"{image_id}.png"
        out_valid_mask_path = dirs["mask_valid"] / f"{image_id}.png"
        out_stars_label_path = dirs["labels_stars"] / f"{image_id}.json"
        out_camera_label_path = dirs["labels_camera"] / f"{image_id}.json"
        out_targets_label_path = dirs["labels_targets"] / f"{image_id}.json"

        save_rgb_png(out_image_path, image_with_targets)
        save_gray_png(out_target_mask_path, target_mask)
        copied_star_mask = copy_if_exists(src_star_mask, out_star_mask_path)
        copied_background_mask = copy_if_exists(src_background_mask, out_background_mask_path)
        copied_valid_mask = copy_if_exists(src_valid_mask, out_valid_mask_path)
        copied_stars_label = copy_if_exists(src_star_label, out_stars_label_path)
        copied_camera_label = copy_if_exists(src_camera_label, out_camera_label_path)

        type_counts = {
            "point_blob": sum(1 for item in target_records if item.get("target_type") == "point_blob"),
            "short_streak": sum(1 for item in target_records if item.get("target_type") == "short_streak"),
        }
        targets_json = {
            "image_id": image_id,
            "source_image_path": str(src_image),
            "num_targets": len(target_records),
            "target_type_counts": type_counts,
            "targets": target_records,
        }
        write_json(out_targets_label_path, targets_json)

        total_targets += len(target_records)
        if len(target_records) == 0:
            no_target_count += 1
        target_count_hist[str(len(target_records))] = target_count_hist.get(str(len(target_records)), 0) + 1
        for key in target_type_counts:
            target_type_counts[key] += type_counts[key]

        new_record = dict(record)
        new_record.update(
            {
                "image_id": image_id,
                "source_image_path": str(src_image),
                "image_path": str(out_image_path),
                "star_mask_path": copied_star_mask,
                "target_mask_path": str(out_target_mask_path),
                "background_mask_path": copied_background_mask,
                "valid_mask_path": copied_valid_mask,
                "stars_label_path": copied_stars_label,
                "camera_label_path": copied_camera_label,
                "targets_label_path": str(out_targets_label_path),
                "num_targets": len(target_records),
                "target_types": [item.get("target_type", "unknown") for item in target_records],
            }
        )
        append_jsonl(output_manifest, new_record)
        progress.update(
            idx + 1,
            extra=f"{image_id} targets={len(target_records)} types={new_record['target_types']}",
        )

    write_splits_from_source(source_root, output_root, selected_ids)
    config_used_path = output_root / "config_used.yaml"
    config_used_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

    summary = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "source_manifest": str(source_manifest),
        "output_manifest": str(output_manifest),
        "num_images": len(records),
        "target_policy": {
            "image_probability": target_cfg.image_probability,
            "targets_per_image_min": target_cfg.targets_per_image_min,
            "targets_per_image_max": target_cfg.targets_per_image_max,
            "point_blob_probability": target_cfg.point_blob.probability,
            "short_streak_probability": target_cfg.short_streak.probability,
            "exact_no_target_count": no_target_total,
        },
        "no_target_count": no_target_count,
        "no_target_ratio": float(no_target_count / len(records)) if records else 0.0,
        "target_count_total": total_targets,
        "target_count_mean": float(total_targets / len(records)) if records else 0.0,
        "target_count_hist": target_count_hist,
        "target_type_counts": target_type_counts,
    }
    write_json(output_root / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    输入:
    - 无，读取命令行。

    输出:
    - argparse Namespace。
    """

    parser = argparse.ArgumentParser(description="Add simulated targets to an existing clean_data dataset.")
    parser.add_argument(
        "--config",
        default="STAR_Agent/configs/data_generation/clean_simulation.yaml",
        help="YAML config containing the targets section.",
    )
    parser.add_argument(
        "--source_dir",
        default="STAR_Agent/data/clean/synthetic_v001",
        help="Existing clean_data root directory.",
    )
    parser.add_argument(
        "--output_dir",
        default="STAR_Agent/data/clean/synthetic_v002_targets",
        help="Output clean_data root with injected targets.",
    )
    parser.add_argument("--manifest", default=None, help="Optional source manifest path.")
    parser.add_argument("--max_images", type=int, default=None, help="Only process the first N images.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    return parser.parse_args()


def main() -> int:
    """命令行主入口。

    输入:
    - 无，读取命令行参数。

    输出:
    - 进程退出码。
    """

    args = parse_args()
    cfg = load_config(args.config)
    summary = add_targets_to_clean_dataset(
        cfg=cfg,
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        max_images=args.max_images,
        seed_override=args.seed,
    )
    print("[OK] targets added to clean dataset")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
