#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""single degradation 生成器通用工具。

本文件只放通用 I/O、manifest、预览拼图和批量调度逻辑。
每一种退化的机理仍然放在独立的 degradation `.py` 文件中，避免把不同退化混在一起。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw
import yaml

from ..common.io import ensure_dir, write_json


def load_rgb_float(path: str | Path) -> np.ndarray:
    """读取 RGB 图像到 `[0, 1]` float32。

    输入:
    - `path`: 图像路径。

    输出:
    - HxWx3 float32 图像。
    """

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def save_rgb_float(path: str | Path, image: np.ndarray) -> None:
    """保存 RGB float 图像。

    输入:
    - `path`: 输出路径。
    - `image`: `[0, 1]` 图像。

    输出:
    - 无返回值。
    """

    p = Path(path)
    ensure_dir(p.parent)
    arr = np.clip(image, 0.0, 1.0)
    Image.fromarray(np.rint(arr * 255.0).astype(np.uint8), mode="RGB").save(p)


def save_gray_float(path: str | Path, image: np.ndarray) -> None:
    """保存单通道 float/uint8 图像。

    输入:
    - `path`: 输出路径。
    - `image`: 单通道图像。

    输出:
    - 无返回值。
    """

    p = Path(path)
    ensure_dir(p.parent)
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.rint(np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(p)


def normalize01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """把数组归一化到 `[0, 1]`。

    输入:
    - `x`: 任意数组。
    - `eps`: 防止除零。

    输出:
    - float32 归一化数组。
    """

    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mn) / (mx - mn + eps)).astype(np.float32)


def set_seed(seed: int | None) -> None:
    """设置 numpy 随机种子。

    输入:
    - `seed`: 随机种子。

    输出:
    - 无返回值。
    """

    if seed is None:
        return
    np.random.seed(seed)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """读取 JSONL。

    输入:
    - `path`: JSONL 文件路径。

    输出:
    - 记录列表。
    """

    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def resolve_path(path_value: str | Path, repo_root: Path) -> Path:
    """解析 manifest 中的相对/绝对路径。

    输入:
    - `path_value`: manifest 中的路径字符串。
    - `repo_root`: 仓库根目录。

    输出:
    - 存在的 Path。
    """

    p = Path(path_value)
    candidates = [p] if p.is_absolute() else [repo_root / p, p]
    for item in candidates:
        if item.exists():
            return item
    raise FileNotFoundError(f"Path not found: {path_value}")


def load_config(path: str | Path | None) -> dict[str, Any]:
    """读取 YAML 配置。

    输入:
    - `path`: YAML 路径。

    输出:
    - 配置字典；文件不存在则返回空字典。
    """

    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def output_paths(output_root: Path, degradation: str, mode: str, level: int, image_id: str) -> dict[str, Path]:
    """构造标准 single degradation 输出路径。

    输入:
    - `output_root`: `single/<degradation>` 根目录。
    - `degradation`: 退化名称。
    - `mode`: 子模式。
    - `level`: 等级。
    - `image_id`: clean 图像 ID。

    输出:
    - image/mask/field/meta 路径字典。
    """

    base = output_root / mode / f"level_{level}"
    return {
        "image": base / "images" / f"{image_id}@{degradation}@{mode}@l{level}.png",
        "mask": base / "masks" / f"{image_id}@{degradation}@{mode}@l{level}.png",
        "field": base / "fields" / f"{image_id}@{degradation}@{mode}@l{level}.png",
        "meta": base / "meta" / f"{image_id}@{degradation}@{mode}@l{level}.json",
    }


def build_preview_grid(records: list[dict[str, Any]], output_path: Path, draw_mask_box: bool = False) -> None:
    """生成预览网格。

    输入:
    - `records`: 生成记录。
    - `output_path`: 输出预览路径。
    - `draw_mask_box`: 是否画 mask 外接框。

    输出:
    - 无返回值。
    """

    if not records:
        return
    tiles: list[Image.Image] = []
    for record in records:
        image = Image.open(record["image_path"]).convert("RGB")
        image.thumbnail((256, 256))
        draw = ImageDraw.Draw(image)
        draw.text((6, 6), f"L{record['level']} {record['mode']}", fill=(255, 80, 80))
        if draw_mask_box:
            mask = Image.open(record["mask_path"]).convert("L").resize(image.size)
            bbox = mask.getbbox()
            if bbox:
                draw.rectangle(bbox, outline=(255, 60, 60), width=2)
        tiles.append(image.copy())

    cols = 5
    rows = int(math.ceil(len(tiles) / cols))
    canvas = Image.new("RGB", (cols * 256, rows * 256), (0, 0, 0))
    for idx, tile in enumerate(tiles):
        canvas.paste(tile, ((idx % cols) * 256, (idx // cols) * 256))
    ensure_dir(output_path.parent)
    canvas.save(output_path)


def run_standard_batch(
    args: Any,
    degradation: str,
    modes: list[str],
    add_func: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]],
    default_thresholds: dict[int, float],
) -> dict[str, Any]:
    """执行标准 single degradation 批量生成。

    输入:
    - `args`: argparse Namespace。
    - `degradation`: 退化名称。
    - `modes`: 支持的模式列表。
    - `add_func`: 单张图退化函数。
    - `default_thresholds`: 默认 mask 阈值。

    输出:
    - summary 字典。
    """

    repo_root = Path.cwd()
    cfg = load_config(args.config)
    deg_cfg = cfg.get(degradation, {}) if isinstance(cfg, dict) else {}
    threshold_cfg = deg_cfg.get("mask_threshold_by_level", default_thresholds)
    threshold_by_level = {int(k): float(v) for k, v in threshold_cfg.items()}

    records = read_jsonl(args.manifest)
    if args.num_images is not None:
        records = records[: max(0, int(args.num_images))]
    output_root = Path(args.output_root)
    manifest_out = output_root / "_manifests" / f"{degradation}_preview_{len(records)}.jsonl"
    ensure_dir(manifest_out.parent)
    manifest_out.write_text("", encoding="utf-8")

    generated: list[dict[str, Any]] = []
    level_cycle = [1, 2, 3, 4, 5]
    for idx, record in enumerate(records):
        image_id = str(record.get("image_id") or f"sample_{idx:06d}")
        image_path = resolve_path(record["image_path"], repo_root)
        level = int(args.level) if args.level is not None else level_cycle[idx % len(level_cycle)]
        mode = str(args.mode) if args.mode else modes[idx % len(modes)]
        seed = int(args.seed + idx) if args.seed is not None else None

        image = load_rgb_float(image_path)
        degraded, mask, field, meta = add_func(
            image=image,
            level=level,
            seed=seed,
            mode=mode,
            mask_threshold_by_level=threshold_by_level,
        )
        paths = output_paths(output_root, degradation, mode, level, image_id)
        save_rgb_float(paths["image"], degraded)
        save_gray_float(paths["mask"], mask)
        save_gray_float(paths["field"], field)
        meta.update(
            {
                "image_id": image_id,
                "source_image_path": str(image_path),
                "image_path": str(paths["image"]),
                "mask_path": str(paths["mask"]),
                "field_path": str(paths["field"]),
                "meta_path": str(paths["meta"]),
            }
        )
        write_json(paths["meta"], meta)
        with manifest_out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        generated.append(meta)
        print(f"[PROGRESS] {degradation} {idx + 1}/{len(records)} | {image_id} | level={level} mode={mode}")

    preview_path = output_root / "_preview" / f"{degradation}_preview_{len(records)}.png"
    raw_preview_path = output_root / "_preview" / f"{degradation}_preview_{len(records)}_raw.png"
    build_preview_grid(generated, preview_path, draw_mask_box=True)
    build_preview_grid(generated, raw_preview_path, draw_mask_box=False)

    summary = {
        "num_images": len(generated),
        "output_root": str(output_root),
        "manifest_path": str(manifest_out),
        "preview_path": str(preview_path),
        "raw_preview_path": str(raw_preview_path),
        "modes": sorted({item["mode"] for item in generated}),
        "levels": sorted({int(item["level"]) for item in generated}),
    }
    write_json(output_root / "_manifests" / f"{degradation}_preview_{len(records)}_summary.json", summary)
    return summary
