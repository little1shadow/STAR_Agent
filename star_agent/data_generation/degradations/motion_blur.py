#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""物理约束版 motion blur 前向仿真。


1. exposure integration
   曝光期间，平台姿态漂移、角速度或抖动会让星场在探测器像面产生连续位移。
2. trajectory PSF
   把曝光时间离散成多个子时刻，采样像面轨迹，并用光学 PSF 宽度把轨迹栅格化成卷积核。
3. signal/background separation
   运动模糊主要作用于星点和空间目标等光学场景信号；背景低频部分基本保持，只把高频天体信号按轨迹积分。

输出：
- degraded image
- degradation mask
- difference/blur field
- meta JSON
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

from ..common.io import ensure_dir, write_json


MOTION_BLUR_MODES = ["linear", "curved", "non_uniform", "jitter"]

LEVEL_PARAMS: dict[int, dict[str, float]] = {
    # length 是曝光期间星点中心在像面扫过的典型距离，单位 pixel。
    # level 1 接近 clean；level 4/5 必须明显拉长星点和目标，便于判别与训练 deblur executor。
    1: {"length_min": 0.45, "length_max": 1.15, "psf_sigma": 0.34, "background_mix": 0.08},
    2: {"length_min": 1.30, "length_max": 2.60, "psf_sigma": 0.42, "background_mix": 0.10},
    3: {"length_min": 3.20, "length_max": 5.80, "psf_sigma": 0.55, "background_mix": 0.13},
    4: {"length_min": 7.20, "length_max": 12.50, "psf_sigma": 0.72, "background_mix": 0.17},
    5: {"length_min": 13.50, "length_max": 23.00, "psf_sigma": 0.92, "background_mix": 0.22},
}

DEFAULT_MASK_THRESHOLD_BY_LEVEL = {1: 0.006, 2: 0.009, 3: 0.013, 4: 0.018, 5: 0.024}
MASK_AREA_CAP_BY_LEVEL = {1: 0.10, 2: 0.16, 3: 0.25, 4: 0.34, 5: 0.44}


def set_seed(seed: int | None) -> None:
    """设置随机种子。

    输入:
    - `seed`: 随机种子。为 None 时保持随机。

    输出:
    - 无返回值。
    """

    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


def load_rgb_float(path: str | Path) -> np.ndarray:
    """读取 RGB 图像到 `[0, 1]` float32。

    输入:
    - `path`: 图像路径。

    输出:
    - HxWx3 float32 图像。
    """

    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


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
    - float32 归一化结果。
    """

    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mn) / (mx - mn + eps)).astype(np.float32)


def sample_motion_trajectory(mode: str, level: int, num_samples: int = 241) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """采样曝光期间的像面运动轨迹。

    输入:
    - `mode`: motion blur 模式。
    - `level`: 退化等级。
    - `num_samples`: 曝光时间采样点数。

    输出:
    - `x`, `y`: 每个曝光子时刻的像面偏移，单位 pixel。
    - `weights`: 每个子时刻的曝光权重。
    - `meta`: 轨迹参数。

    物理含义:
    - `linear`: 平台角速度近似恒定，星点形成直线拖尾。
    - `curved`: 曝光期间角速度或姿态有缓慢加速度，轨迹轻微弯曲。
    - `non_uniform`: 角速度不恒定，拖尾不同位置亮度不同。
    - `jitter`: 高频姿态抖动叠加小漂移，拖尾较短但星点变胖、边缘毛糙。
    """

    if mode not in MOTION_BLUR_MODES:
        raise ValueError(f"Unknown motion blur mode: {mode}")
    if level not in LEVEL_PARAMS:
        raise ValueError("level must be in [1, 5]")

    p = LEVEL_PARAMS[level]
    length = random.uniform(p["length_min"], p["length_max"])
    angle = random.uniform(0.0, 2.0 * math.pi)
    t = np.linspace(-0.5, 0.5, num_samples, dtype=np.float32)

    if mode == "linear":
        u = t
        v = np.zeros_like(t)
        weights = np.ones_like(t)
        curvature = 0.0
        jitter_amp = 0.0
    elif mode == "curved":
        curvature = random.uniform(-0.26, 0.26) * length * (0.70 + 0.10 * level)
        u = t
        v = curvature * (t * t - np.mean(t * t))
        weights = np.ones_like(t)
        jitter_amp = 0.0
    elif mode == "non_uniform":
        gamma = random.uniform(0.58, 1.65)
        u = np.sign(t) * (np.abs(t) ** gamma)
        v = random.uniform(-0.08, 0.08) * length * np.sin(2.0 * math.pi * (t + 0.5))
        phase = random.uniform(0.0, 2.0 * math.pi)
        weights = 0.70 + 0.30 * np.sin(2.0 * math.pi * (t + 0.5) + phase)
        weights *= np.linspace(random.uniform(0.75, 1.05), random.uniform(0.75, 1.05), num_samples)
        curvature = 0.0
        jitter_amp = 0.0
    else:
        drift_scale = random.uniform(0.30, 0.75)
        jitter_amp = random.uniform(0.20, 0.42) * max(length, 1.0)
        freq_x = random.uniform(1.5, 4.2)
        freq_y = random.uniform(1.8, 5.0)
        phase_x = random.uniform(0.0, 2.0 * math.pi)
        phase_y = random.uniform(0.0, 2.0 * math.pi)
        u = drift_scale * t + (jitter_amp / max(length, 1e-6)) * 0.35 * np.sin(2.0 * math.pi * freq_x * (t + 0.5) + phase_x)
        v = (jitter_amp / max(length, 1e-6)) * 0.35 * np.sin(2.0 * math.pi * freq_y * (t + 0.5) + phase_y)
        weights = np.ones_like(t)
        curvature = 0.0

    x_local = length * u
    y_local = length * v
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    x = cos_a * x_local - sin_a * y_local
    y = sin_a * x_local + cos_a * y_local

    # 把核中心移到曝光轨迹质心，避免整体平移图像，只保留模糊效应。
    weights = np.clip(weights.astype(np.float32), 0.05, None)
    x = x.astype(np.float32) - float(np.average(x, weights=weights))
    y = y.astype(np.float32) - float(np.average(y, weights=weights))
    weights = weights / float(np.sum(weights))

    meta = {
        "trajectory_length_px": float(np.max(np.sqrt((x - x[0]) ** 2 + (y - y[0]) ** 2))),
        "nominal_length_px": float(length),
        "angle_deg": float(math.degrees(angle) % 360.0),
        "num_exposure_samples": int(num_samples),
        "curvature_px": float(curvature),
        "jitter_amp_px": float(jitter_amp),
    }
    return x, y, weights.astype(np.float32), meta


def rasterize_trajectory_kernel(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    psf_sigma: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """把连续曝光轨迹栅格化成 motion PSF kernel。

    输入:
    - `x`, `y`: 轨迹偏移，单位 pixel。
    - `weights`: 曝光权重。
    - `psf_sigma`: 单个子曝光星点 PSF 宽度。

    输出:
    - `kernel`: 归一化卷积核。
    - `meta`: kernel 尺寸和能量统计。
    """

    extent = float(max(np.max(np.abs(x)), np.max(np.abs(y)), 1.0))
    radius = int(math.ceil(extent + 4.0 * psf_sigma + 3.0))
    radius = max(radius, 5)
    size = 2 * radius + 1
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    center = float(radius)
    kernel = np.zeros((size, size), dtype=np.float32)
    sigma2 = max(float(psf_sigma) ** 2, 1e-6)

    for xi, yi, wi in zip(x, y, weights):
        dx = xx - (center + float(xi))
        dy = yy - (center + float(yi))
        kernel += float(wi) * np.exp(-0.5 * (dx * dx + dy * dy) / sigma2).astype(np.float32)

    kernel_sum = float(np.sum(kernel))
    if kernel_sum <= 0:
        kernel[radius, radius] = 1.0
    else:
        kernel /= kernel_sum

    meta = {
        "kernel_size": int(size),
        "kernel_radius": int(radius),
        "kernel_peak": float(kernel.max()),
        "psf_sigma_px": float(psf_sigma),
    }
    return kernel.astype(np.float32), meta


def convolve_rgb(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """对 RGB 图像逐通道执行同尺寸卷积。

    输入:
    - `image`: HxWx3 float32。
    - `kernel`: motion PSF kernel。

    输出:
    - HxWx3 float32 卷积结果。
    """

    out = np.empty_like(image, dtype=np.float32)
    for c in range(image.shape[2]):
        out[:, :, c] = fftconvolve(image[:, :, c], kernel, mode="same").astype(np.float32)
    return out


def add_motion_blur(
    image: np.ndarray,
    level: int,
    seed: int | None = None,
    mode: str | None = None,
    mask_threshold_by_level: dict[int, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """给图像添加物理约束版 motion blur。

    输入:
    - `image`: `[0, 1]` RGB 图像。
    - `level`: 退化等级 1-5。
    - `seed`: 随机种子。
    - `mode`: 运动模式。
    - `mask_threshold_by_level`: mask 阈值。

    输出:
    - `degraded`: 退化图。
    - `mask`: 退化 mask。
    - `field`: 连续差异场。
    - `meta`: 参数记录。
    """

    set_seed(seed)
    if level not in LEVEL_PARAMS:
        raise ValueError("level must be in [1, 5]")
    if mode is None:
        mode = random.choice(MOTION_BLUR_MODES)
    if mode not in MOTION_BLUR_MODES:
        raise ValueError(f"Unknown motion blur mode: {mode}")

    p = LEVEL_PARAMS[level]
    x, y, weights, traj_meta = sample_motion_trajectory(mode=mode, level=level)
    psf_sigma = p["psf_sigma"] * random.uniform(0.86, 1.18)
    kernel, kernel_meta = rasterize_trajectory_kernel(x=x, y=y, weights=weights, psf_sigma=psf_sigma)

    # 背景低频保持，天体高频信号随曝光轨迹积分；这样更接近星空成像而不是把底噪也拖成线。
    background = gaussian_filter(image, sigma=(3.0, 3.0, 0.0)).astype(np.float32)
    detail = image - background
    blurred_detail = convolve_rgb(detail, kernel)
    blurred_full = convolve_rgb(image, kernel)
    degraded = background + blurred_detail

    # 对 level 4/5 混入少量全图卷积，模拟长曝光下背景纹理与弱星云状细颗粒也被轻微平均。
    degraded = (1.0 - p["background_mix"]) * degraded + p["background_mix"] * blurred_full
    degraded = np.clip(degraded, 0.0, 1.0).astype(np.float32)

    diff = np.mean(np.abs(degraded - image), axis=2).astype(np.float32)
    field = normalize01(diff)
    thresholds = mask_threshold_by_level or DEFAULT_MASK_THRESHOLD_BY_LEVEL
    threshold = float(thresholds.get(level, DEFAULT_MASK_THRESHOLD_BY_LEVEL[level]))
    mask_bool = diff >= threshold
    area_cap = MASK_AREA_CAP_BY_LEVEL[level]
    if float(np.mean(mask_bool)) > area_cap:
        threshold = max(threshold, float(np.quantile(diff, 1.0 - area_cap)))
        mask_bool = diff >= threshold
    mask = mask_bool.astype(np.uint8) * 255

    meta = {
        "degradation": "motion_blur",
        "physical_model": "exposure_integrated_image_motion_psf",
        "mode": mode,
        "level": int(level),
        "seed": seed,
        "mask_threshold": float(threshold),
        "mask_area_ratio": float(np.mean(mask > 0)),
        "diff_max": float(diff.max()),
        "diff_mean": float(diff.mean()),
        **traj_meta,
        **kernel_meta,
    }
    return degraded, mask, field, meta


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
    - `path_value`: manifest 中记录的路径。
    - `repo_root`: 当前仓库根目录。

    输出:
    - 已存在的 Path。
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
    - 配置字典；文件不存在时返回空字典。
    """

    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def output_paths(output_root: Path, mode: str, level: int, image_id: str) -> dict[str, Path]:
    """构造标准输出路径。

    输入:
    - `output_root`: `single/motion_blur` 根目录。
    - `mode`: 模式。
    - `level`: 等级。
    - `image_id`: 图像 ID。

    输出:
    - 路径字典。
    """

    base = output_root / mode / f"level_{level}"
    return {
        "image": base / "images" / f"{image_id}@motion_blur@{mode}@l{level}.png",
        "mask": base / "masks" / f"{image_id}@motion_blur@{mode}@l{level}.png",
        "field": base / "fields" / f"{image_id}@motion_blur@{mode}@l{level}.png",
        "meta": base / "meta" / f"{image_id}@motion_blur@{mode}@l{level}.json",
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


def run_batch(args: argparse.Namespace) -> dict[str, Any]:
    """批量生成 motion blur。

    输入:
    - `args`: 命令行参数。

    输出:
    - summary 字典。
    """

    repo_root = Path.cwd()
    cfg = load_config(args.config)
    motion_cfg = cfg.get("motion_blur", {}) if isinstance(cfg, dict) else {}
    threshold_cfg = motion_cfg.get("mask_threshold_by_level", DEFAULT_MASK_THRESHOLD_BY_LEVEL)
    threshold_by_level = {int(k): float(v) for k, v in threshold_cfg.items()}

    records = read_jsonl(args.manifest)
    if args.num_images is not None:
        records = records[: max(0, int(args.num_images))]
    output_root = Path(args.output_root)
    manifest_out = output_root / "_manifests" / f"motion_blur_preview_{len(records)}.jsonl"
    ensure_dir(manifest_out.parent)
    manifest_out.write_text("", encoding="utf-8")

    generated: list[dict[str, Any]] = []
    level_cycle = [1, 2, 3, 4, 5]
    mode_cycle = MOTION_BLUR_MODES
    for idx, record in enumerate(records):
        image_id = str(record.get("image_id") or f"sample_{idx:06d}")
        image_path = resolve_path(record["image_path"], repo_root)
        level = int(args.level) if args.level is not None else level_cycle[idx % len(level_cycle)]
        mode = str(args.mode) if args.mode else mode_cycle[idx % len(mode_cycle)]
        seed = int(args.seed + idx) if args.seed is not None else None

        image = load_rgb_float(image_path)
        degraded, mask, field, meta = add_motion_blur(
            image=image,
            level=level,
            seed=seed,
            mode=mode,
            mask_threshold_by_level=threshold_by_level,
        )
        paths = output_paths(output_root, mode, level, image_id)
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
        print(
            "[PROGRESS] motion_blur "
            f"{idx + 1}/{len(records)} | {image_id} | level={level} mode={mode} "
            f"length={meta['trajectory_length_px']:.2f}px kernel={meta['kernel_size']}"
        )

    preview_path = output_root / "_preview" / f"motion_blur_preview_{len(records)}.png"
    raw_preview_path = output_root / "_preview" / f"motion_blur_preview_{len(records)}_raw.png"
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
    write_json(output_root / "_manifests" / f"motion_blur_preview_{len(records)}_summary.json", summary)
    return summary


def run_single(args: argparse.Namespace) -> dict[str, Any]:
    """单张图生成。

    输入:
    - `args`: 命令行参数。

    输出:
    - meta 字典。
    """

    image = load_rgb_float(args.input)
    level = int(args.level or 3)
    degraded, mask, field, meta = add_motion_blur(image, level=level, seed=args.seed, mode=args.mode)
    save_rgb_float(args.output, degraded)
    if args.mask:
        save_gray_float(args.mask, mask)
    if args.field:
        save_gray_float(args.field, field)
    if args.meta:
        write_json(args.meta, meta)
    return meta


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    输入:
    - 无，读取命令行。

    输出:
    - argparse Namespace。
    """

    parser = argparse.ArgumentParser(description="Generate physical motion blur degradation for STAR-Agent.")
    parser.add_argument("--config", default="STAR_Agent/configs/data_generation/degradation_single.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--mask", default=None)
    parser.add_argument("--field", default=None)
    parser.add_argument("--meta", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output_root", default="STAR_Agent/data/degraded/single/motion_blur")
    parser.add_argument("--num_images", type=int, default=None)
    parser.add_argument("--level", type=int, default=None, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--mode", default=None, choices=MOTION_BLUR_MODES)
    parser.add_argument("--seed", type=int, default=131)
    return parser.parse_args()


def main() -> int:
    """命令行入口。

    输入:
    - 无。

    输出:
    - 进程退出码。
    """

    args = parse_args()
    if args.manifest:
        summary = run_batch(args)
        print("[OK] motion blur batch generated")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if not args.input or not args.output:
        raise SystemExit("Either --manifest or both --input/--output are required.")
    meta = run_single(args)
    print("[OK] motion blur generated")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
