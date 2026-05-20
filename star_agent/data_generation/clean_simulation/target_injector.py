#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""空间小目标注入器。

本文件负责把“下游目标检测器能够检测到”的空间目标注入 clean 星空图。
目标是正常场景内容，不是 degradation；因此它会写入 target mask 和 target label，
后续 restoration 是否伤害目标，就可以用这些标签做离线评价。

当前支持两类目标:
- `point_blob`: 点状/弱小 blob 目标，对应轻量 LoG/blob detector；
- `short_streak`: 短条纹/拖尾目标，对应 ASTRiDE 或类 ASTRiDE streak detector。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PointBlobConfig:
    """点状目标配置。

    输入字段来自 YAML 的 `targets.point_blob`。

    字段:
    - `probability`: 在所有目标中的采样概率。
    - `snr_min`, `snr_max`: 目标峰值相对局部噪声的信噪比范围。
    - `fwhm_min_px`, `fwhm_max_px`: 点状目标 FWHM 范围。
    - `axis_ratio_min`, `axis_ratio_max`: 椭圆长短轴比例范围，1 表示圆形。
    """

    probability: float
    snr_min: float
    snr_max: float
    fwhm_min_px: float
    fwhm_max_px: float
    axis_ratio_min: float
    axis_ratio_max: float


@dataclass(frozen=True)
class ShortStreakConfig:
    """短条纹目标配置。

    输入字段来自 YAML 的 `targets.short_streak`。

    字段:
    - `probability`: 在所有目标中的采样概率。
    - `snr_min`, `snr_max`: 条纹中心线峰值信噪比范围。
    - `length_min_px`, `length_max_px`: 条纹长度范围。
    - `width_min_px`, `width_max_px`: 条纹横向 FWHM 范围。
    - `end_taper_strength`: 端点渐隐强度，避免生成生硬矩形线段。
    """

    probability: float
    snr_min: float
    snr_max: float
    length_min_px: float
    length_max_px: float
    width_min_px: float
    width_max_px: float
    end_taper_strength: float


@dataclass(frozen=True)
class TargetInjectorConfig:
    """目标注入总配置。

    字段:
    - `enabled`: 是否启用目标注入。
    - `image_probability`: 每张图含目标的概率。
    - `targets_per_image_min`, `targets_per_image_max`: 每张含目标图的目标数量范围。
    - `min_edge_distance_px`: 目标中心距离边界的最小距离。
    - `min_distance_from_star_px`: 目标中心距离已有星点的最小距离。
    - `min_distance_between_targets_px`: 同一张图中不同目标中心的最小距离。
    - `local_noise_sigma`: 用于把 SNR 转成图像峰值亮度的局部噪声估计。
    - `mask_threshold_ratio`: target mask 阈值，相对目标峰值。
    - `max_position_attempts`: 采样合法目标位置的最大尝试次数。
    - `point_blob`, `short_streak`: 两类目标的专属配置。
    """

    enabled: bool
    image_probability: float
    targets_per_image_min: int
    targets_per_image_max: int
    min_edge_distance_px: float
    min_distance_from_star_px: float
    min_distance_between_targets_px: float
    local_noise_sigma: float
    mask_threshold_ratio: float
    max_position_attempts: int
    point_blob: PointBlobConfig
    short_streak: ShortStreakConfig


def _range_pair(data: dict[str, Any], key: str, default: tuple[float, float]) -> tuple[float, float]:
    """从配置中读取 `[min, max]` 数值范围。

    输入:
    - `data`: 配置字典。
    - `key`: 字段名。
    - `default`: 缺省范围。

    输出:
    - `(min_value, max_value)`。
    """

    value = data.get(key, list(default))
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return default
    a, b = float(value[0]), float(value[1])
    return (min(a, b), max(a, b))


def build_target_config(cfg: dict[str, Any] | None) -> TargetInjectorConfig:
    """从 YAML 字典构建目标注入配置。

    输入:
    - `cfg`: `targets` 配置段。若为空，则默认禁用。

    输出:
    - `TargetInjectorConfig`。
    """

    cfg = cfg or {}
    point_cfg = cfg.get("point_blob", {}) or {}
    streak_cfg = cfg.get("short_streak", {}) or {}

    point_snr = _range_pair(point_cfg, "snr_range", (6.0, 18.0))
    point_fwhm = _range_pair(point_cfg, "fwhm_px_range", (1.4, 2.4))
    point_axis = _range_pair(point_cfg, "axis_ratio_range", (1.0, 1.45))

    streak_snr = _range_pair(streak_cfg, "snr_range", (7.0, 22.0))
    streak_length = _range_pair(streak_cfg, "length_px_range", (10.0, 32.0))
    streak_width = _range_pair(streak_cfg, "width_px_range", (1.2, 2.8))

    return TargetInjectorConfig(
        enabled=bool(cfg.get("enabled", False)),
        image_probability=float(cfg.get("image_probability", 1.0)),
        targets_per_image_min=int(cfg.get("targets_per_image_min", 1)),
        targets_per_image_max=int(cfg.get("targets_per_image_max", 1)),
        min_edge_distance_px=float(cfg.get("min_edge_distance_px", 36.0)),
        min_distance_from_star_px=float(cfg.get("min_distance_from_star_px", 8.0)),
        min_distance_between_targets_px=float(cfg.get("min_distance_between_targets_px", 28.0)),
        local_noise_sigma=float(cfg.get("local_noise_sigma", 7.0)),
        mask_threshold_ratio=float(cfg.get("mask_threshold_ratio", 0.08)),
        max_position_attempts=int(cfg.get("max_position_attempts", 80)),
        point_blob=PointBlobConfig(
            probability=float(point_cfg.get("probability", 0.3)),
            snr_min=point_snr[0],
            snr_max=point_snr[1],
            fwhm_min_px=point_fwhm[0],
            fwhm_max_px=point_fwhm[1],
            axis_ratio_min=point_axis[0],
            axis_ratio_max=point_axis[1],
        ),
        short_streak=ShortStreakConfig(
            probability=float(streak_cfg.get("probability", 0.7)),
            snr_min=streak_snr[0],
            snr_max=streak_snr[1],
            length_min_px=streak_length[0],
            length_max_px=streak_length[1],
            width_min_px=streak_width[0],
            width_max_px=streak_width[1],
            end_taper_strength=float(streak_cfg.get("end_taper_strength", 0.35)),
        ),
    )


def _sample_position(
    height: int,
    width: int,
    star_x: np.ndarray,
    star_y: np.ndarray,
    min_edge: float,
    min_star_distance: float,
    rng: np.random.Generator,
    max_attempts: int,
) -> tuple[float, float]:
    """采样一个尽量不贴边、不压到已有星点的目标中心。

    输入:
    - `height`, `width`: 图像尺寸。
    - `star_x`, `star_y`: 已投影星点像素坐标。
    - `min_edge`: 距离图像边界的最小距离。
    - `min_star_distance`: 距离星点中心的最小距离。
    - `rng`: 随机数生成器。
    - `max_attempts`: 最大尝试次数。

    输出:
    - `(x, y)` 目标中心。

    说明:
    - 如果多次采样仍失败，会返回最后一次位置；
    - 这避免在星点极密区域生成时死循环。
    """

    x_low = min_edge
    x_high = max(min_edge + 1.0, width - min_edge)
    y_low = min_edge
    y_high = max(min_edge + 1.0, height - min_edge)
    last = (float(width) / 2.0, float(height) / 2.0)

    for _ in range(max(1, max_attempts)):
        x = float(rng.uniform(x_low, x_high))
        y = float(rng.uniform(y_low, y_high))
        last = (x, y)
        if len(star_x) == 0:
            return x, y
        dist2 = (star_x - x) * (star_x - x) + (star_y - y) * (star_y - y)
        if float(np.min(dist2)) >= min_star_distance * min_star_distance:
            return x, y
    return last


def _bbox_from_mask(mask: np.ndarray) -> list[int]:
    """根据目标 mask 计算 bbox。

    输入:
    - `mask`: 单个目标的 uint8 mask。

    输出:
    - `[x1, y1, x2, y2]`，若为空则返回全 0。
    """

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _render_point_blob(
    image: np.ndarray,
    target_mask: np.ndarray,
    x0: float,
    y0: float,
    target_id: int,
    config: TargetInjectorConfig,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """渲染点状弱目标。

    输入:
    - `image`: 待叠加目标的 float 图像，会原地修改。
    - `target_mask`: 全图目标 mask，会原地更新。
    - `x0`, `y0`: 目标中心。
    - `target_id`: 当前目标编号。
    - `config`: 目标注入配置。
    - `rng`: 随机数生成器。

    输出:
    - 目标标签字典。
    """

    h, w = image.shape[:2]
    pcfg = config.point_blob
    snr = float(rng.uniform(pcfg.snr_min, pcfg.snr_max))
    peak = snr * config.local_noise_sigma
    fwhm_major = float(rng.uniform(pcfg.fwhm_min_px, pcfg.fwhm_max_px))
    axis_ratio = float(rng.uniform(pcfg.axis_ratio_min, pcfg.axis_ratio_max))
    fwhm_minor = fwhm_major / max(axis_ratio, 1.0)
    theta = float(rng.uniform(0.0, np.pi))

    sigma_x = fwhm_major / 2.355
    sigma_y = fwhm_minor / 2.355
    radius = int(np.ceil(4.0 * max(sigma_x, sigma_y)))
    cx, cy = int(round(x0)), int(round(y0))
    x1, x2 = max(0, cx - radius), min(w, cx + radius + 1)
    y1, y2 = max(0, cy - radius), min(h, cy + radius + 1)

    yy, xx = np.mgrid[y1:y2, x1:x2].astype(np.float32)
    dx = xx - x0
    dy = yy - y0
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    u = cos_t * dx + sin_t * dy
    v = -sin_t * dx + cos_t * dy
    patch = peak * np.exp(-0.5 * ((u / sigma_x) ** 2 + (v / sigma_y) ** 2))

    image[y1:y2, x1:x2] += patch.astype(np.float32)
    one_mask = np.zeros_like(target_mask)
    one_mask[y1:y2, x1:x2][patch >= peak * config.mask_threshold_ratio] = 255
    target_mask[one_mask > 0] = 255

    return {
        "target_id": int(target_id),
        "target_type": "point_blob",
        "detector_family": "log_blob_detector",
        "x_center_px": float(x0),
        "y_center_px": float(y0),
        "bbox_xyxy": _bbox_from_mask(one_mask),
        "snr": float(snr),
        "peak_adu": float(peak),
        "fwhm_major_px": float(fwhm_major),
        "fwhm_minor_px": float(fwhm_minor),
        "axis_ratio": float(axis_ratio),
        "angle_deg": float(np.degrees(theta)),
        "mask_area_px": int(np.sum(one_mask > 0)),
    }


def _render_short_streak(
    image: np.ndarray,
    target_mask: np.ndarray,
    x0: float,
    y0: float,
    target_id: int,
    config: TargetInjectorConfig,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """渲染短条纹目标。

    输入:
    - `image`: 待叠加目标的 float 图像，会原地修改。
    - `target_mask`: 全图目标 mask，会原地更新。
    - `x0`, `y0`: 条纹中心。
    - `target_id`: 当前目标编号。
    - `config`: 目标注入配置。
    - `rng`: 随机数生成器。

    输出:
    - 目标标签字典。

    设计:
    - 横向使用 Gaussian profile，符合成像 PSF；
    - 纵向加入轻微不均匀和端点渐隐，避免像人工画线；
    - 标签中显式标注 `astride_streak_detector`，提醒后续用条纹检测器评价。
    """

    h, w = image.shape[:2]
    scfg = config.short_streak
    snr = float(rng.uniform(scfg.snr_min, scfg.snr_max))
    peak = snr * config.local_noise_sigma
    length = float(rng.uniform(scfg.length_min_px, scfg.length_max_px))
    width_fwhm = float(rng.uniform(scfg.width_min_px, scfg.width_max_px))
    theta = float(rng.uniform(0.0, np.pi))
    sigma_w = width_fwhm / 2.355

    radius_x = int(np.ceil(abs(np.cos(theta)) * length / 2.0 + 4.0 * sigma_w + 2.0))
    radius_y = int(np.ceil(abs(np.sin(theta)) * length / 2.0 + 4.0 * sigma_w + 2.0))
    cx, cy = int(round(x0)), int(round(y0))
    x1, x2 = max(0, cx - radius_x), min(w, cx + radius_x + 1)
    y1, y2 = max(0, cy - radius_y), min(h, cy + radius_y + 1)

    yy, xx = np.mgrid[y1:y2, x1:x2].astype(np.float32)
    dx = xx - x0
    dy = yy - y0
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    u = cos_t * dx + sin_t * dy
    v = -sin_t * dx + cos_t * dy

    inside = np.abs(u) <= (length / 2.0)
    cross_profile = np.exp(-0.5 * (v / sigma_w) ** 2)
    u_norm = np.clip(np.abs(u) / max(length / 2.0, 1.0), 0.0, 1.0)
    taper = (1.0 - scfg.end_taper_strength) + scfg.end_taper_strength * (0.5 + 0.5 * np.cos(np.pi * u_norm))
    ripple = 1.0 + 0.12 * np.sin(2.0 * np.pi * (u / max(length, 1.0)) + rng.uniform(0.0, 2.0 * np.pi))
    patch = peak * cross_profile * taper * ripple * inside
    patch = np.maximum(patch, 0.0)

    image[y1:y2, x1:x2] += patch.astype(np.float32)
    one_mask = np.zeros_like(target_mask)
    one_mask[y1:y2, x1:x2][patch >= peak * config.mask_threshold_ratio] = 255
    target_mask[one_mask > 0] = 255

    x_start = x0 - np.cos(theta) * length / 2.0
    y_start = y0 - np.sin(theta) * length / 2.0
    x_end = x0 + np.cos(theta) * length / 2.0
    y_end = y0 + np.sin(theta) * length / 2.0

    return {
        "target_id": int(target_id),
        "target_type": "short_streak",
        "detector_family": "astride_streak_detector",
        "x_center_px": float(x0),
        "y_center_px": float(y0),
        "bbox_xyxy": _bbox_from_mask(one_mask),
        "snr": float(snr),
        "peak_adu": float(peak),
        "length_px": float(length),
        "width_fwhm_px": float(width_fwhm),
        "angle_deg": float(np.degrees(theta)),
        "line_start_xy": [float(x_start), float(y_start)],
        "line_end_xy": [float(x_end), float(y_end)],
        "mask_area_px": int(np.sum(one_mask > 0)),
    }


def inject_targets(
    image: np.ndarray,
    star_x: np.ndarray,
    star_y: np.ndarray,
    config: TargetInjectorConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """向 clean 星空图中注入空间目标。

    输入:
    - `image`: float32 clean 图像，形状为 HxW。
    - `star_x`, `star_y`: 已投影星点坐标，用于避开已有星点。
    - `config`: 目标注入配置。
    - `rng`: 随机数生成器。

    输出:
    - `image_with_targets`: 注入目标后的图像。
    - `target_mask`: uint8 目标 mask。
    - `target_records`: 目标标签列表。

    说明:
    - `point_blob` 和 `short_streak` 的比例由配置控制；
    - target 是后续下游检测和 task-aware restoration 评价的真实内容标签。
    """

    h, w = image.shape[:2]
    target_mask = np.zeros((h, w), dtype=np.uint8)
    target_records: list[dict[str, Any]] = []
    if not config.enabled or rng.random() > config.image_probability:
        return image.astype(np.float32), target_mask, target_records

    min_count = max(0, int(config.targets_per_image_min))
    max_count = max(min_count, int(config.targets_per_image_max))
    num_targets = int(rng.integers(min_count, max_count + 1)) if max_count > 0 else 0

    point_p = max(0.0, config.point_blob.probability)
    streak_p = max(0.0, config.short_streak.probability)
    total_p = point_p + streak_p
    point_threshold = point_p / total_p if total_p > 0 else 0.3

    out = image.astype(np.float32, copy=True)
    previous_target_x: list[float] = []
    previous_target_y: list[float] = []

    for target_id in range(num_targets):
        x0, y0 = 0.0, 0.0
        for _ in range(max(1, config.max_position_attempts)):
            x0, y0 = _sample_position(
                height=h,
                width=w,
                star_x=star_x.astype(np.float32, copy=False),
                star_y=star_y.astype(np.float32, copy=False),
                min_edge=config.min_edge_distance_px,
                min_star_distance=config.min_distance_from_star_px,
                rng=rng,
                max_attempts=1,
            )
            if not previous_target_x:
                break
            prev_x = np.asarray(previous_target_x, dtype=np.float32)
            prev_y = np.asarray(previous_target_y, dtype=np.float32)
            dist2 = (prev_x - x0) * (prev_x - x0) + (prev_y - y0) * (prev_y - y0)
            if float(np.min(dist2)) >= config.min_distance_between_targets_px**2:
                break
        if rng.random() < point_threshold:
            record = _render_point_blob(out, target_mask, x0, y0, target_id, config, rng)
        else:
            record = _render_short_streak(out, target_mask, x0, y0, target_id, config, rng)
        target_records.append(record)
        previous_target_x.append(float(x0))
        previous_target_y.append(float(y0))

    return out.astype(np.float32), target_mask, target_records
