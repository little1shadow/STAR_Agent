#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""短条纹目标检测器。"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage as ndi

from ...common.image_ops import connected_components, robust_background, weighted_centroid, component_shape


def detect_streak_targets(
    image: np.ndarray,
    star_mask: np.ndarray | None = None,
    cfg: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """检测短条纹目标。

    输入:
    - `image`: HxW `[0, 1]` 灰度图。
    - `star_mask`: 可选星点 mask，用于减少星点误检。
    - `cfg`: 检测配置。

    输出:
    - `(target_candidates, target_mask)`。
    """

    cfg = cfg or {}
    background_sigma = float(cfg.get("background_sigma", 7.0))
    threshold_sigma = float(cfg.get("threshold_sigma", 3.8))
    min_area = int(cfg.get("min_area", 8))
    max_area = int(cfg.get("max_area", 600))
    min_length = float(cfg.get("min_length_px", 8.0))
    min_aspect = float(cfg.get("min_aspect_ratio", 2.0))
    min_snr = float(cfg.get("min_snr", 4.0))
    star_exclusion_dilation = int(cfg.get("star_exclusion_dilation", 1))

    background = ndi.gaussian_filter(image, sigma=background_sigma)
    residual = np.clip(image - background, 0.0, None)
    med, sigma = robust_background(residual)
    binary = residual > (med + threshold_sigma * sigma)
    binary = ndi.binary_closing(binary, iterations=1)
    if star_mask is not None and star_mask.any():
        excluded = ndi.binary_dilation(star_mask > 0, iterations=star_exclusion_dilation)
        binary = binary & (~excluded)

    candidates: list[dict[str, Any]] = []
    mask = np.zeros_like(image, dtype=bool)
    for comp in connected_components(binary):
        coords = comp["coords"]
        area = int(coords.shape[0])
        if area < min_area or area > max_area:
            continue
        shape = component_shape(coords)
        if shape["length_px"] < min_length or shape["aspect_ratio"] < min_aspect:
            continue
        yy = coords[:, 0]
        xx = coords[:, 1]
        peak = float(residual[yy, xx].max())
        snr = peak / max(sigma, 1e-6)
        if snr < min_snr:
            continue
        cx, cy, flux = weighted_centroid(image, coords, float(np.median(background[yy, xx])))
        confidence = float(min(1.0, 0.10 * snr + 0.08 * shape["aspect_ratio"]))
        candidates.append(
            {
                "id": len(candidates),
                "target_type": "short_streak",
                "detector_family": "streak",
                "x": cx,
                "y": cy,
                "snr": float(snr),
                "flux": flux,
                "confidence": confidence,
                **shape,
            }
        )
        mask[yy, xx] = True
    if mask.any():
        mask = ndi.binary_dilation(mask, iterations=1)
    return candidates, mask
