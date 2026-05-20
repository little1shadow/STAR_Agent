#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""从下游检测结果提取 Policy Net 可用 proxy features。"""

from __future__ import annotations

from typing import Any

import numpy as np


def summarize_targets(candidates: list[dict[str, Any]]) -> dict[str, float | int]:
    """汇总目标检测候选。

    输入:
    - `candidates`: 点状和条纹目标候选。

    输出:
    - 目标 proxy 特征。
    """

    if not candidates:
        return {
            "target_candidate_count": 0,
            "target_candidate_snr": 0.0,
            "target_confidence": 0.0,
            "point_blob_count": 0,
            "short_streak_count": 0,
        }
    return {
        "target_candidate_count": len(candidates),
        "target_candidate_snr": float(max(c.get("snr", 0.0) for c in candidates)),
        "target_confidence": float(max(c.get("confidence", 0.0) for c in candidates)),
        "point_blob_count": sum(1 for c in candidates if c.get("target_type") == "point_blob"),
        "short_streak_count": sum(1 for c in candidates if c.get("target_type") == "short_streak"),
    }


def build_proxy_features(
    star_summary: dict[str, Any],
    target_candidates: list[dict[str, Any]],
    image_shape: tuple[int, int],
) -> dict[str, float | int]:
    """构建在线下游 proxy feature。

    输入:
    - `star_summary`: 星点检测摘要。
    - `target_candidates`: 目标候选列表。
    - `image_shape`: 图像高宽。

    输出:
    - policy 可直接使用的 proxy 字典。
    """

    h, w = image_shape
    image_area = max(1, h * w)
    detected_star_count = int(star_summary.get("detected_star_count", 0))
    target_summary = summarize_targets(target_candidates)
    features: dict[str, float | int] = {
        "plate_solve_success": 0,
        "solver_confidence": 0.0,
        "matched_star_count_norm": 0.0,
        "detected_star_count_norm": float(detected_star_count / max(1.0, image_area / 1024.0 / 1024.0)),
        "unmatched_candidate_ratio": 1.0,
        "reprojection_error_norm": 1.0,
        "centroid_fit_residual_norm": 0.0,
        "star_snr_mean": float(star_summary.get("star_snr_mean", 0.0)),
        "star_fwhm_mean": float(star_summary.get("star_fwhm_mean", 0.0)),
    }
    features.update(target_summary)
    return features
