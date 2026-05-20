#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""PSF 渲染和光度转换。

本文件把投影后的星点转换为图像中的光斑。
第一版使用归一化 Moffat PSF，它比单纯 Gaussian 更容易产生真实星点的翼部。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .star_projector import ProjectedStars


@dataclass(frozen=True)
class PhotometryConfig:
    """星等到 flux 的转换配置。

    字段:
    - `mag_zero_point`: 参考星等。
    - `flux_at_zero_point`: 参考星等对应的总 flux。
    - `min_flux`: 最小 flux，避免极暗星完全消失。
    - `max_flux`: 最大 flux，避免极亮星导致整图动态范围崩溃。
    """

    mag_zero_point: float
    flux_at_zero_point: float
    min_flux: float
    max_flux: float


@dataclass(frozen=True)
class PsfConfig:
    """PSF 参数。

    字段:
    - `psf_type`: PSF 类型，V1 支持 `moffat` 和 `gaussian`。
    - `fwhm_px_mean`: 星点 FWHM 均值，单位 pixel。
    - `fwhm_px_std`: 每张图 FWHM 随机扰动。
    - `beta`: Moffat beta 参数。
    - `kernel_radius_factor`: kernel 半径与 FWHM 的比例。
    - `halo_flux_threshold`: 超过该 flux 的亮星添加弱 halo。
    - `halo_fraction`: halo 占星点 flux 的比例。
    - `halo_fwhm_factor`: halo FWHM 相对主 PSF 的放大倍数。
    """

    psf_type: str
    fwhm_px_mean: float
    fwhm_px_std: float
    beta: float
    kernel_radius_factor: float
    halo_flux_threshold: float
    halo_fraction: float
    halo_fwhm_factor: float


@dataclass(frozen=True)
class BackgroundConfig:
    """clean 背景配置。

    字段:
    - `level_mean`: 背景均值。
    - `level_std`: 每张图背景均值扰动。
    - `gradient_amplitude`: 极弱背景梯度幅度。
    - `vignette_amplitude`: 极弱渐晕幅度。
    - `read_noise_sigma`: 正常传感器读出颗粒强度。
    - `shot_noise_scale`: 与亮度相关的散粒纹理强度。
    - `unresolved_speckle_density`: 未解析极弱星点/颗粒密度。
    - `unresolved_speckle_min`: 未解析 speckle 最小亮度。
    - `unresolved_speckle_max`: 未解析 speckle 最大亮度。
    """

    level_mean: float
    level_std: float
    gradient_amplitude: float
    vignette_amplitude: float
    read_noise_sigma: float
    shot_noise_scale: float
    unresolved_speckle_density: float
    unresolved_speckle_min: float
    unresolved_speckle_max: float


def flux_from_mag(mag: np.ndarray, config: PhotometryConfig) -> np.ndarray:
    """将 Gaia G 星等转换为图像总 flux。

    输入:
    - `mag`: G-band 星等数组。
    - `config`: 光度配置。

    输出:
    - 每颗星的总 flux。

    公式:
    - `flux = flux0 * 10^(-0.4 * (mag - mag0))`
    """

    flux = config.flux_at_zero_point * np.power(10.0, -0.4 * (mag - config.mag_zero_point))
    return np.clip(flux, config.min_flux, config.max_flux).astype(np.float32)


def sample_fwhm(config: PsfConfig, rng: np.random.Generator) -> float:
    """采样当前图像的 PSF FWHM。

    输入:
    - `config`: PSF 配置。
    - `rng`: numpy 随机数生成器。

    输出:
    - 当前图像使用的 FWHM，单位 pixel。
    """

    fwhm = rng.normal(config.fwhm_px_mean, config.fwhm_px_std)
    return float(max(0.8, fwhm))


def _moffat_kernel(fwhm_px: float, beta: float, radius: int) -> np.ndarray:
    """生成归一化 Moffat PSF kernel。

    输入:
    - `fwhm_px`: FWHM，单位 pixel。
    - `beta`: Moffat beta 参数。
    - `radius`: kernel 半径，单位 pixel。

    输出:
    - 归一化 2D kernel，sum 为 1。
    """

    y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    alpha = fwhm_px / (2.0 * np.sqrt(np.power(2.0, 1.0 / beta) - 1.0))
    kernel = np.power(1.0 + (x * x + y * y) / (alpha * alpha), -beta)
    kernel /= np.sum(kernel)
    return kernel.astype(np.float32)


def _gaussian_kernel(fwhm_px: float, radius: int) -> np.ndarray:
    """生成归一化 Gaussian PSF kernel。

    输入:
    - `fwhm_px`: FWHM，单位 pixel。
    - `radius`: kernel 半径，单位 pixel。

    输出:
    - 归一化 2D kernel。
    """

    y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    sigma = fwhm_px / 2.355
    kernel = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    kernel /= np.sum(kernel)
    return kernel.astype(np.float32)


def build_clean_background(
    height: int,
    width: int,
    config: BackgroundConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """生成干净背景。

    输入:
    - `height`, `width`: 图像尺寸。
    - `config`: 背景配置。
    - `rng`: 随机数生成器。

    输出:
    - float32 背景图。

    说明:
    - 这里的背景模拟“真实 clean 帧”的正常底色和传感器纹理；
    - 它不是 degradation 中的强噪声，而是目标域本身存在的基线颗粒；
    - 后续 noise degradation 应该在这个基线之上继续增强，而不是从纯黑理想图开始。
    """

    level = rng.normal(config.level_mean, config.level_std)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    x_norm = (xx - width / 2.0) / max(width / 2.0, 1.0)
    y_norm = (yy - height / 2.0) / max(height / 2.0, 1.0)

    grad_angle = rng.uniform(0.0, 2.0 * np.pi)
    gradient = config.gradient_amplitude * (
        np.cos(grad_angle) * x_norm + np.sin(grad_angle) * y_norm
    )
    radius2 = x_norm * x_norm + y_norm * y_norm
    vignette = -config.vignette_amplitude * level * radius2
    background = level + gradient + vignette

    if config.read_noise_sigma > 0:
        background += rng.normal(0.0, config.read_noise_sigma, size=(height, width))

    if config.unresolved_speckle_density > 0:
        num_speckles = int(height * width * config.unresolved_speckle_density)
        if num_speckles > 0:
            ys = rng.integers(0, height, size=num_speckles)
            xs = rng.integers(0, width, size=num_speckles)
            vals = rng.uniform(
                config.unresolved_speckle_min,
                config.unresolved_speckle_max,
                size=num_speckles,
            )
            background[ys, xs] += vals

    return np.maximum(background, 0.0).astype(np.float32)


def add_shot_texture(
    image: np.ndarray,
    config: BackgroundConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """给图像加入与亮度相关的正常散粒纹理。

    输入:
    - `image`: 已经包含背景和星点的图像。
    - `config`: 背景/传感器纹理配置。
    - `rng`: 随机数生成器。

    输出:
    - 加入轻微散粒纹理后的图像。

    注意:
    - 这是为了让 synthetic clean 更接近真实相机帧；
    - 不能设置太强，否则会和后续 noise degradation 混淆。
    """

    if config.shot_noise_scale <= 0:
        return image
    sigma = np.sqrt(np.maximum(image, 0.0)) * config.shot_noise_scale
    return image + rng.normal(0.0, sigma).astype(np.float32)


def render_star_field(
    stars: ProjectedStars,
    height: int,
    width: int,
    photometry: PhotometryConfig,
    psf: PsfConfig,
    background: BackgroundConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float]:
    """渲染星空 clean 图。

    输入:
    - `stars`: 投影后的星点集合。
    - `height`, `width`: 图像尺寸。
    - `photometry`: 星等到 flux 配置。
    - `psf`: PSF 配置。
    - `background`: 背景配置。
    - `rng`: 随机数生成器。

    输出:
    - `image`: 含背景和星点的 float32 图像。
    - `star_layer`: 只含星点贡献的 float32 图像。
    - `fwhm_px`: 当前图像使用的 PSF FWHM。
    """

    image = build_clean_background(height, width, background, rng)
    star_layer = np.zeros((height, width), dtype=np.float32)
    fwhm_px = sample_fwhm(psf, rng)
    radius = max(2, int(np.ceil(psf.kernel_radius_factor * fwhm_px)))

    if psf.psf_type.lower() == "gaussian":
        kernel = _gaussian_kernel(fwhm_px, radius)
    else:
        kernel = _moffat_kernel(fwhm_px, psf.beta, radius)

    halo_radius = max(radius + 1, int(np.ceil(psf.kernel_radius_factor * fwhm_px * psf.halo_fwhm_factor)))
    halo_kernel = _gaussian_kernel(fwhm_px * psf.halo_fwhm_factor, halo_radius)
    halo_k_size = 2 * halo_radius + 1

    fluxes = flux_from_mag(stars.g_mag, photometry)
    k_size = 2 * radius + 1

    for x0, y0, flux in zip(stars.x_px, stars.y_px, fluxes, strict=False):
        cx = int(round(float(x0)))
        cy = int(round(float(y0)))
        x1 = max(0, cx - radius)
        y1 = max(0, cy - radius)
        x2 = min(width, cx + radius + 1)
        y2 = min(height, cy + radius + 1)
        if x1 >= x2 or y1 >= y2:
            continue

        kx1 = x1 - (cx - radius)
        ky1 = y1 - (cy - radius)
        kx2 = kx1 + (x2 - x1)
        ky2 = ky1 + (y2 - y1)
        if kx1 >= k_size or ky1 >= k_size:
            continue

        patch = flux * kernel[ky1:ky2, kx1:kx2]
        star_layer[y1:y2, x1:x2] += patch

        # 亮星在真实星空图中通常有更宽的微弱 halo。这里单独加一层宽 Gaussian，
        # 让亮星不再像“动画点”，但强度保持较低，避免变成杂散光退化。
        if flux >= psf.halo_flux_threshold and psf.halo_fraction > 0:
            hx1 = max(0, cx - halo_radius)
            hy1 = max(0, cy - halo_radius)
            hx2 = min(width, cx + halo_radius + 1)
            hy2 = min(height, cy + halo_radius + 1)
            if hx1 < hx2 and hy1 < hy2:
                hkx1 = hx1 - (cx - halo_radius)
                hky1 = hy1 - (cy - halo_radius)
                hkx2 = hkx1 + (hx2 - hx1)
                hky2 = hky1 + (hy2 - hy1)
                if hkx1 < halo_k_size and hky1 < halo_k_size:
                    halo_patch = flux * psf.halo_fraction * halo_kernel[hky1:hky2, hkx1:hkx2]
                    star_layer[hy1:hy2, hx1:hx2] += halo_patch

    image += star_layer
    image = add_shot_texture(image, background, rng)
    return image.astype(np.float32), star_layer.astype(np.float32), fwhm_px
