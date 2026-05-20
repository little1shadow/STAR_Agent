#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""天区筛选工具。

这里提供基于球面角距离的 cone search，用于从 Gaia 子集中取出相机视场附近的星。
"""

from __future__ import annotations

import numpy as np

from .load_catalog import GaiaCatalog


def angular_distance_deg(
    ra1_deg: np.ndarray,
    dec1_deg: np.ndarray,
    ra2_deg: float,
    dec2_deg: float,
) -> np.ndarray:
    """计算球面角距离。

    输入:
    - `ra1_deg`, `dec1_deg`: 多个星点的赤经赤纬，单位 degree。
    - `ra2_deg`, `dec2_deg`: 中心点赤经赤纬，单位 degree。

    输出:
    - 每颗星到中心点的角距离，单位 degree。

    说明:
    - 使用 haversine 形式，避免小角度时数值不稳定。
    """

    ra1 = np.deg2rad(ra1_deg)
    dec1 = np.deg2rad(dec1_deg)
    ra2 = np.deg2rad(float(ra2_deg))
    dec2 = np.deg2rad(float(dec2_deg))

    dra = (ra1 - ra2 + np.pi) % (2.0 * np.pi) - np.pi
    ddec = dec1 - dec2
    a = np.sin(ddec / 2.0) ** 2 + np.cos(dec1) * np.cos(dec2) * np.sin(dra / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    return np.rad2deg(2.0 * np.arcsin(np.sqrt(a)))


def query_cone(
    catalog: GaiaCatalog,
    ra_center_deg: float,
    dec_center_deg: float,
    radius_deg: float,
) -> GaiaCatalog:
    """从星表中筛选中心附近的星。

    输入:
    - `catalog`: Gaia 星表。
    - `ra_center_deg`, `dec_center_deg`: 查询中心，单位 degree。
    - `radius_deg`: 查询半径，单位 degree。

    输出:
    - GaiaCatalog 子集。

    用途:
    - 先粗筛天区，再做精确投影到图像范围。
    """

    dist = angular_distance_deg(
        catalog.ra_deg,
        catalog.dec_deg,
        ra_center_deg,
        dec_center_deg,
    )
    return catalog.take(dist <= float(radius_deg))
