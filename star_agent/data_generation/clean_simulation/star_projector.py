#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""星表到图像像素的投影工具。

核心功能是把 Gaia 星表中的 RA/Dec 坐标通过切平面投影转换成图像上的 x/y 像素坐标。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..catalog.load_catalog import GaiaCatalog
from .camera_model import CameraConfig, Pointing


@dataclass
class ProjectedStars:
    """投影后的星点集合。

    字段:
    - `source_id`: Gaia source 编号。
    - `ra_deg`, `dec_deg`: 原始天球坐标。
    - `g_mag`: Gaia G 星等。
    - `x_px`, `y_px`: 图像像素坐标，浮点数。
    """

    source_id: np.ndarray
    ra_deg: np.ndarray
    dec_deg: np.ndarray
    g_mag: np.ndarray
    x_px: np.ndarray
    y_px: np.ndarray

    def __len__(self) -> int:
        """返回投影到图像内的星点数量。

        输入:
        - 无。

        输出:
        - 星点数量。
        """

        return int(self.x_px.shape[0])


def project_to_image(
    catalog: GaiaCatalog,
    pointing: Pointing,
    camera: CameraConfig,
    margin_px: float = 8.0,
) -> ProjectedStars:
    """将 Gaia 星表投影到图像像素坐标。

    输入:
    - `catalog`: 已经粗筛到当前视场附近的 GaiaCatalog。
    - `pointing`: 当前图像指向。
    - `camera`: 相机配置。
    - `margin_px`: 边缘额外保留范围，避免 PSF 在边缘被过早丢弃。

    输出:
    - `ProjectedStars`，只包含落在图像范围附近的星。

    投影方式:
    - 使用 gnomonic projection，即把天球投影到相机中心切平面；
    - 再按 roll 角旋转；
    - 最后根据 pixel scale 转换到像素坐标。
    """

    if len(catalog) == 0:
        return ProjectedStars(
            source_id=np.asarray([], dtype=str),
            ra_deg=np.asarray([], dtype=np.float64),
            dec_deg=np.asarray([], dtype=np.float64),
            g_mag=np.asarray([], dtype=np.float32),
            x_px=np.asarray([], dtype=np.float32),
            y_px=np.asarray([], dtype=np.float32),
        )

    ra = np.deg2rad(catalog.ra_deg)
    dec = np.deg2rad(catalog.dec_deg)
    ra0 = np.deg2rad(pointing.ra_center_deg)
    dec0 = np.deg2rad(pointing.dec_center_deg)

    dra = (ra - ra0 + np.pi) % (2.0 * np.pi) - np.pi
    sin_dec = np.sin(dec)
    cos_dec = np.cos(dec)
    sin_dec0 = np.sin(dec0)
    cos_dec0 = np.cos(dec0)

    cosc = sin_dec0 * sin_dec + cos_dec0 * cos_dec * np.cos(dra)
    valid = cosc > 1e-8

    xi = np.zeros_like(ra)
    eta = np.zeros_like(ra)
    xi[valid] = cos_dec[valid] * np.sin(dra[valid]) / cosc[valid]
    eta[valid] = (
        cos_dec0 * sin_dec[valid]
        - sin_dec0 * cos_dec[valid] * np.cos(dra[valid])
    ) / cosc[valid]

    # gnomonic 平面坐标是 tan(theta) 量纲，转成 degree 方便和 pixel scale 对齐。
    xi_deg = np.rad2deg(xi)
    eta_deg = np.rad2deg(eta)

    roll = np.deg2rad(pointing.roll_deg)
    cos_r = np.cos(roll)
    sin_r = np.sin(roll)
    x_rot = xi_deg * cos_r - eta_deg * sin_r
    y_rot = xi_deg * sin_r + eta_deg * cos_r

    x_px = camera.width / 2.0 + x_rot / camera.pixel_scale_deg
    y_px = camera.height / 2.0 - y_rot / camera.pixel_scale_deg

    inside = (
        valid
        & (x_px >= -margin_px)
        & (x_px < camera.width + margin_px)
        & (y_px >= -margin_px)
        & (y_px < camera.height + margin_px)
    )

    return ProjectedStars(
        source_id=catalog.source_id[inside],
        ra_deg=catalog.ra_deg[inside],
        dec_deg=catalog.dec_deg[inside],
        g_mag=catalog.g_mag[inside],
        x_px=x_px[inside].astype(np.float32),
        y_px=y_px[inside].astype(np.float32),
    )
