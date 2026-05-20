#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""轻量星点检测和质心估计器。"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage as ndi

from ...common.image_ops import connected_components, robust_background, weighted_centroid


def detect_stars(image: np.ndarray, cfg: dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], np.ndarray]:
    """检测星点候选并生成 star pseudo mask。

    输入:
    - `image`: HxW `[0, 1]` 灰度图。
    - `cfg`: 检测配置。

    输出:
    - `(candidates, mask)`。

    设计动机:
    - 作为在线 proxy，不依赖 GT。
    - 真实 clean 上生成的是 pseudo mask，不能等同于严格真值。
    """

    cfg = cfg or {}
    bg_sigma = float(cfg.get("background_sigma", 9.0))
    smooth_sigma = float(cfg.get("smooth_sigma", 0.8))
    threshold_sigma = float(cfg.get("threshold_sigma", 4.5))
    min_area = int(cfg.get("min_area", 2))
    max_area = int(cfg.get("max_area", 80))
    min_snr = float(cfg.get("min_snr", 4.0))
    dilation = int(cfg.get("mask_dilation", 2))

    background_map = ndi.gaussian_filter(image, sigma=bg_sigma)
    residual = np.clip(image - background_map, 0.0, None)
    smoothed = ndi.gaussian_filter(residual, sigma=smooth_sigma)
    med, sigma = robust_background(smoothed)
    binary = smoothed > (med + threshold_sigma * sigma)

    candidates: list[dict[str, Any]] = []
    mask = np.zeros_like(image, dtype=bool)
    for comp in connected_components(binary):
        coords = comp["coords"]
        area = int(coords.shape[0])
        if area < min_area or area > max_area:
            continue
        yy = coords[:, 0]
        xx = coords[:, 1]
        peak = float(residual[yy, xx].max())
        snr = peak / max(sigma, 1e-6)
        if snr < min_snr:
            continue
        cx, cy, flux = weighted_centroid(image, coords, float(np.median(background_map[yy, xx])))
        radius = float(np.sqrt(area / np.pi))
        candidates.append(
            {
                "id": len(candidates),
                "x": cx,
                "y": cy,
                "area": area,
                "snr": float(snr),
                "flux": flux,
                "fwhm_px_est": max(1.0, 2.355 * radius / 2.0),
                "peak": float(image[yy, xx].max()),
                "detector": "star_blob_v001",
            }
        )
        mask[yy, xx] = True

    if dilation > 0 and mask.any():
        mask = ndi.binary_dilation(mask, iterations=dilation)
    return candidates, mask


def summarize_star_candidates(candidates: list[dict[str, Any]]) -> dict[str, float | int]:
    """把星点候选压缩为 policy 可用 proxy 特征。

    输入:
    - `candidates`: 星点候选列表。

    输出:
    - proxy 指标字典。
    """

    if not candidates:
        return {"detected_star_count": 0, "star_snr_mean": 0.0, "star_fwhm_mean": 0.0}
    return {
        "detected_star_count": len(candidates),
        "star_snr_mean": float(np.mean([c["snr"] for c in candidates])),
        "star_fwhm_mean": float(np.mean([c["fwhm_px_est"] for c in candidates])),
    }
