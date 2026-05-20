#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""物理约束版 cosmic ray 退化。

宇宙线在探测器上通常表现为随机高亮点、短线段、电荷扩散 blob 或混合形态。
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from ._single_common import normalize01, run_standard_batch, set_seed


COSMIC_RAY_MODES = ["point", "line", "blob", "mixed"]

LEVEL_PARAMS = {
    1: {"count": 2, "amp": 0.18, "line_len": 4, "sigma": 0.35},
    2: {"count": 5, "amp": 0.32, "line_len": 8, "sigma": 0.45},
    3: {"count": 12, "amp": 0.52, "line_len": 15, "sigma": 0.60},
    4: {"count": 25, "amp": 0.75, "line_len": 26, "sigma": 0.80},
    5: {"count": 46, "amp": 0.95, "line_len": 42, "sigma": 1.05},
}

DEFAULT_MASK_THRESHOLD_BY_LEVEL = {1: 0.020, 2: 0.035, 3: 0.055, 4: 0.075, 5: 0.090}


def _draw_point(field: np.ndarray, amp: float, sigma: float) -> None:
    """绘制一个 cosmic ray 点事件。"""

    h, w = field.shape
    y = np.random.randint(1, h - 1)
    x = np.random.randint(1, w - 1)
    field[y, x] += amp * np.random.uniform(0.65, 1.25)
    if sigma > 0.4:
        field[max(0, y - 1): min(h, y + 2), max(0, x - 1): min(w, x + 2)] += amp * 0.12


def _draw_line(field: np.ndarray, amp: float, length: int, sigma: float) -> None:
    """绘制一个 cosmic ray 线状事件。"""

    h, w = field.shape
    y0 = np.random.uniform(2, h - 3)
    x0 = np.random.uniform(2, w - 3)
    angle = np.random.uniform(0.0, 2.0 * math.pi)
    steps = max(2, int(np.random.uniform(0.45, 1.20) * length))
    for i in range(steps):
        t = i / max(steps - 1, 1) - 0.5
        x = int(round(x0 + t * steps * math.cos(angle)))
        y = int(round(y0 + t * steps * math.sin(angle)))
        if 0 <= x < w and 0 <= y < h:
            field[y, x] += amp * np.random.uniform(0.65, 1.10)
    if sigma > 0.3:
        field[:] = np.maximum(field, gaussian_filter(field, sigma=max(0.18, sigma * 0.45)))


def _draw_blob(field: np.ndarray, amp: float, sigma: float) -> None:
    """绘制一个电荷扩散 blob 事件。"""

    h, w = field.shape
    y = np.random.randint(2, h - 2)
    x = np.random.randint(2, w - 2)
    yy, xx = np.mgrid[0:h, 0:w]
    sx = sigma * np.random.uniform(1.0, 2.4)
    sy = sigma * np.random.uniform(1.0, 2.4)
    blob = np.exp(-0.5 * (((xx - x) / sx) ** 2 + ((yy - y) / sy) ** 2))
    field += amp * np.random.uniform(0.45, 1.05) * blob.astype(np.float32)


def add_cosmic_ray(
    image: np.ndarray,
    level: int,
    seed: int | None = None,
    mode: str | None = None,
    mask_threshold_by_level: dict[int, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """添加 cosmic ray 退化。"""

    set_seed(seed)
    if mode is None:
        mode = COSMIC_RAY_MODES[np.random.randint(len(COSMIC_RAY_MODES))]
    if mode not in COSMIC_RAY_MODES:
        raise ValueError(f"Unknown cosmic ray mode: {mode}")

    p = LEVEL_PARAMS[level]
    h, w = image.shape[:2]
    field = np.zeros((h, w), dtype=np.float32)
    count = max(1, np.random.poisson(p["count"]))
    for _ in range(count):
        event_mode = mode if mode != "mixed" else np.random.choice(["point", "line", "blob"], p=[0.50, 0.35, 0.15])
        amp = p["amp"] * np.random.uniform(0.70, 1.30)
        if event_mode == "point":
            _draw_point(field, amp, p["sigma"])
        elif event_mode == "line":
            _draw_line(field, amp, p["line_len"], p["sigma"])
        else:
            _draw_blob(field, amp, p["sigma"])

    degraded = np.clip(image + field[:, :, None], 0.0, 1.0).astype(np.float32)
    thresholds = mask_threshold_by_level or DEFAULT_MASK_THRESHOLD_BY_LEVEL
    thr = float(thresholds.get(level, DEFAULT_MASK_THRESHOLD_BY_LEVEL[level]))
    mask = (field >= thr).astype(np.uint8) * 255
    meta = {
        "degradation": "cosmic_ray",
        "physical_model": "charged_particle_sensor_events",
        "mode": mode,
        "level": int(level),
        "seed": seed,
        "event_count": int(count),
        "field_max": float(field.max()),
        "mask_threshold": thr,
        "mask_area_ratio": float(np.mean(mask > 0)),
    }
    return degraded, mask, normalize01(field), meta


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Generate cosmic ray degradation.")
    parser.add_argument("--config", default="STAR_Agent/configs/data_generation/degradation_single.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--num_images", type=int, default=None)
    parser.add_argument("--level", type=int, default=None, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--mode", default=None, choices=COSMIC_RAY_MODES)
    parser.add_argument("--seed", type=int, default=131)
    return parser.parse_args()


def main() -> int:
    """命令行入口。"""

    summary = run_standard_batch(parse_args(), "cosmic_ray", COSMIC_RAY_MODES, add_cosmic_ray, DEFAULT_MASK_THRESHOLD_BY_LEVEL)
    print("[OK] cosmic ray batch generated")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
