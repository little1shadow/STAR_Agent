#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""tetra3rs 星点质心到像素级 star pseudo mask 的转换工具。

本模块运行在已经安装 `tetra3rs` 的 Python 环境中。它不负责完整 plate solving，
而是先利用 tetra3rs 的 centroid extractor 得到真实 clean 图中的星点中心，
再根据局部 PSF/亮度阈值生成可供 restoration loss、real clean 伪标签和
Policy Net proxy 使用的 star mask。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from star_agent.downstream.common.image_ops import robust_background


@dataclass
class Tetra3rsMaskResult:
    """tetra3rs 星点检测与 mask 生成结果。

    输入字段:
    - `centroids`: 统一后的星点质心列表，坐标为像素坐标 x/y。
    - `star_mask`: bool 类型星点 mask。
    - `metrics`: 检测摘要，用于日志、manifest 和 Policy Net proxy。

    输出用途:
    - 真实 clean 的 `masks/star_pseudo_tetra3rs/*.png`。
    - 真实 clean 的 `labels/stars_tetra3rs/*.json`。
    - 下游 star matching / restoration policy 的 proxy feature。
    """

    centroids: list[dict[str, Any]]
    star_mask: np.ndarray
    metrics: dict[str, Any]


def read_image_for_tetra3rs(path: str | Path) -> np.ndarray:
    """读取图像并保留尽可能多的灰度动态范围。

    输入:
    - `path`: PNG/JPG/TIF 等图像路径。

    输出:
    - HxW float32 图像。若原图是 RGB，则转灰度；若是 uint16，则保留 0-65535 范围。

    设计目的:
    - tetra3rs 的阈值和局部背景估计依赖实际灰度范围，因此这里不强制归一化到 `[0,1]`。
    """

    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., :3].astype(np.float32)
        arr = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    else:
        arr = arr.astype(np.float32)
    return arr.astype(np.float32)


def normalize_for_mask(image: np.ndarray) -> np.ndarray:
    """把任意动态范围图像压到 `[0,1]`，仅用于局部 mask 阈值和保存预览。

    输入:
    - `image`: HxW float32 图像。

    输出:
    - HxW float32，范围 `[0,1]`。
    """

    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image, dtype=np.float32)
    lo, hi = np.percentile(finite, [0.1, 99.9])
    if hi <= lo:
        hi = float(finite.max())
        lo = float(finite.min())
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _value_from_object(obj: Any, names: list[str], default: Any = None) -> Any:
    """从 dict/object/tuple 中尽量稳健地取字段。

    输入:
    - `obj`: tetra3rs 返回的 centroid 对象。
    - `names`: 候选字段名。
    - `default`: 取不到时的默认值。

    输出:
    - 字段值或默认值。
    """

    if isinstance(obj, dict):
        for name in names:
            if name in obj:
                return obj[name]
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def normalize_centroid_record(item: Any, width: int, height: int, origin: str = "center") -> dict[str, Any] | None:
    """把 tetra3rs centroid 转成统一像素坐标记录。

    输入:
    - `item`: tetra3rs 返回的单个 centroid，可为对象、dict 或 tuple/list。
    - `width`, `height`: 图像宽高。
    - `origin`: `center` 表示输入坐标以图像中心为原点；`pixel` 表示已经是像素坐标。

    输出:
    - 统一记录 `{x, y, flux, source}`，若坐标非法则返回 None。
    """

    if isinstance(item, (tuple, list)) and len(item) >= 2:
        raw_x, raw_y = float(item[0]), float(item[1])
        flux = float(item[2]) if len(item) >= 3 and item[2] is not None else 0.0
    else:
        raw_x = _value_from_object(item, ["x", "cx", "x_centroid", "center_x"])
        raw_y = _value_from_object(item, ["y", "cy", "y_centroid", "center_y"])
        if raw_x is None or raw_y is None:
            return None
        raw_x, raw_y = float(raw_x), float(raw_y)
        flux = float(_value_from_object(item, ["flux", "brightness", "sum", "peak"], 0.0) or 0.0)

    if origin == "center":
        x = raw_x + width / 2.0
        y = raw_y + height / 2.0
    else:
        x = raw_x
        y = raw_y

    if not np.isfinite(x) or not np.isfinite(y):
        return None
    if x < 0 or x >= width or y < 0 or y >= height:
        return None

    return {
        "x": float(x),
        "y": float(y),
        "raw_x": float(raw_x),
        "raw_y": float(raw_y),
        "flux": flux,
        "source": "tetra3rs_centroid",
    }


def extract_tetra3rs_centroids(image: np.ndarray, cfg: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
    """调用 tetra3rs 提取星点质心。

    输入:
    - `image`: HxW float32 图像。
    - `cfg`: `configs/downstream/star_matching/tetra3rs.yaml` 中的配置。

    输出:
    - `(raw_centroids, raw_metrics)`。

    关键说明:
    - tetra3rs 不同版本 Python API 可能略有差异，因此这里优先调用常见
      `extract_centroids`，并在参数不兼容时自动降级到最小参数。
    """

    import tetra3rs  # type: ignore

    params = cfg.get("centroid_extraction", {})
    kwargs = {
        "sigma_threshold": float(params.get("sigma_threshold", 5.0)),
        "min_pixels": int(params.get("min_pixels", 2)),
        "max_pixels": int(params.get("max_pixels", 300)),
        "local_bg_block_size": int(params.get("local_bg_block_size", 32)),
        "max_elongation": float(params.get("max_elongation", 3.5)),
    }

    if not hasattr(tetra3rs, "extract_centroids"):
        raise RuntimeError("tetra3rs.extract_centroids is not available in this tetra3rs version")

    try:
        result = tetra3rs.extract_centroids(image, **kwargs)
    except TypeError:
        # 兼容旧/新版本参数名变化：保底只传图像和阈值。
        result = tetra3rs.extract_centroids(image, sigma_threshold=kwargs["sigma_threshold"])

    raw_metrics: dict[str, Any] = {"extractor": "tetra3rs.extract_centroids"}
    if hasattr(result, "centroids"):
        # tetra3rs 0.7.x 返回 ExtractionResult；官方用法是 extraction.centroids。
        # 同时把背景和 raw blob 数量写入 metrics，方便后续判断阈值是否合适。
        raw_centroids = getattr(result, "centroids")
        for key in ["num_blobs_raw", "background_mean", "background_sigma"]:
            if hasattr(result, key):
                value = getattr(result, key)
                if isinstance(value, np.generic):
                    value = value.item()
                raw_metrics[key] = value
    elif isinstance(result, tuple):
        raw_centroids = result[0]
        if len(result) > 1:
            raw_metrics["raw_aux"] = str(result[1])[:1000]
    else:
        raw_centroids = result

    if raw_centroids is None:
        raw_centroids = []
    try:
        raw_centroids = list(raw_centroids)
    except TypeError as exc:
        raise TypeError(
            "Unsupported tetra3rs centroid result. Expected ExtractionResult.centroids, tuple/list, or iterable."
        ) from exc
    max_returned = int(params.get("max_returned", 5000))
    if max_returned > 0:
        raw_centroids = raw_centroids[:max_returned]
    raw_metrics["raw_count"] = len(raw_centroids)
    return raw_centroids, raw_metrics


def build_mask_from_centroids(
    image_norm: np.ndarray,
    centroids: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> np.ndarray:
    """根据星点质心和局部亮度生成像素级 star mask。

    输入:
    - `image_norm`: `[0,1]` 灰度图。
    - `centroids`: 已转换为像素坐标的星点列表。
    - `cfg`: mask 配置。

    输出:
    - HxW bool 星点 mask。

    计算逻辑:
    - 在 centroid 周围取局部窗口估计背景和噪声。
    - 在小半径圆盘内保留高于 `median + k*sigma` 的像素。
    - 如果局部阈值过严导致空 mask，则回退为小圆盘，避免漏掉弱星点。
    """

    h, w = image_norm.shape
    mask_cfg = cfg.get("mask", {})
    radius = float(mask_cfg.get("radius_px", 4.0))
    local_radius = int(mask_cfg.get("local_radius_px", max(6, int(radius * 2))))
    threshold_sigma = float(mask_cfg.get("local_threshold_sigma", 2.0))
    fallback_disk = bool(mask_cfg.get("fallback_disk_if_empty", True))

    out = np.zeros((h, w), dtype=bool)
    for item in centroids:
        cx, cy = float(item["x"]), float(item["y"])
        x0 = max(0, int(np.floor(cx - local_radius)))
        x1 = min(w, int(np.ceil(cx + local_radius + 1)))
        y0 = max(0, int(np.floor(cy - local_radius)))
        y1 = min(h, int(np.ceil(cy + local_radius + 1)))
        if x1 <= x0 or y1 <= y0:
            continue

        yy, xx = np.mgrid[y0:y1, x0:x1]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        disk = dist <= radius
        local = image_norm[y0:y1, x0:x1]
        med, sigma = robust_background(local)
        bright = local >= med + threshold_sigma * sigma
        local_mask = disk & bright
        if not np.any(local_mask) and fallback_disk:
            local_mask = disk
        out[y0:y1, x0:x1] |= local_mask

    dilate_px = int(mask_cfg.get("dilate_px", 0))
    if dilate_px > 0 and np.any(out):
        out = ndi.binary_dilation(out, iterations=dilate_px)
    return out


def detect_stars_with_tetra3rs(image_path: str | Path, cfg: dict[str, Any]) -> Tetra3rsMaskResult:
    """用 tetra3rs 为单张图生成 star pseudo mask。

    输入:
    - `image_path`: 真实 clean 图像路径。
    - `cfg`: tetra3rs 配置。

    输出:
    - `Tetra3rsMaskResult`，包含 centroids、star_mask、metrics。
    """

    image = read_image_for_tetra3rs(image_path)
    h, w = image.shape
    raw_centroids, raw_metrics = extract_tetra3rs_centroids(image, cfg)
    origin = str(cfg.get("coordinates", {}).get("origin", "center"))

    centroids: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_centroids):
        record = normalize_centroid_record(item, width=w, height=h, origin=origin)
        if record is None:
            continue
        record["star_id"] = idx
        centroids.append(record)

    image_norm = normalize_for_mask(image)
    star_mask = build_mask_from_centroids(image_norm, centroids, cfg)
    med, sigma = robust_background(image_norm)
    area = int(star_mask.sum())
    metrics = {
        **raw_metrics,
        "image_path": str(image_path),
        "image_height": int(h),
        "image_width": int(w),
        "num_centroids": int(len(centroids)),
        "star_mask_area_px": area,
        "star_mask_area_ratio": float(area / max(1, h * w)),
        "background_median_norm": float(med),
        "background_sigma_norm": float(sigma),
        "mask_source": cfg.get("quality", {}).get("mask_source", "tetra3rs_centroid_pseudo_v001"),
        "mask_confidence": float(cfg.get("quality", {}).get("mask_confidence", 0.85)),
    }
    return Tetra3rsMaskResult(centroids=centroids, star_mask=star_mask, metrics=metrics)
