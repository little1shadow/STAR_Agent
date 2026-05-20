#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""基于 Gaia DR3 星表生成 clean_data。

输出内容:
- clean 图像: `images/*.png`
- 星点 mask: `masks/star/*.png`
- 背景 mask: `masks/background/*.png`
- 有效区域 mask: `masks/valid/*.png`
- 星点标签: `labels/stars/*.json`
- 相机标签: `labels/camera/*.json`
- 总 manifest: `manifest.jsonl`

设计原则:
- clean 图只包含正常星点、正常 PSF 和非常轻微的干净背景；
- 不加入明显退化，例如噪声、杂散光、坏点、宇宙射线、运动模糊；
- 每颗星都保留 Gaia source_id、RA/Dec、星等、像素坐标和 flux，方便后续计算下游指标。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..catalog.load_catalog import GaiaCatalog, load_gaia_csv, summarize_catalog
from ..catalog.sky_query import query_cone
from ..common.image_ops import save_gray_png, save_rgb_png
from ..common.io import append_jsonl, ensure_dir, write_json
from ..common.masks import (
    build_background_mask,
    build_star_mask,
    build_valid_mask,
)
from ..common.progress import ProgressPrinter
from .camera_model import CameraConfig, PointingConfig, sample_pointing
from .psf_renderer import (
    BackgroundConfig,
    PhotometryConfig,
    PsfConfig,
    flux_from_mag,
    render_star_field,
)
from .star_projector import ProjectedStars, project_to_image
from .target_injector import TargetInjectorConfig, build_target_config, inject_targets


def load_config(config_path: str | Path) -> dict[str, Any]:
    """读取 YAML 配置。

    输入:
    - `config_path`: YAML 配置路径。

    输出:
    - 配置字典。
    """

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def make_output_dirs(root_dir: Path) -> dict[str, Path]:
    """创建 clean_data 输出目录。

    输入:
    - `root_dir`: clean 数据集根目录。

    输出:
    - 字典，包含 images/masks/labels 等目录路径。
    """

    dirs = {
        "images": root_dir / "images",
        "mask_star": root_dir / "masks" / "star",
        "mask_target": root_dir / "masks" / "target",
        "mask_background": root_dir / "masks" / "background",
        "mask_valid": root_dir / "masks" / "valid",
        "labels_stars": root_dir / "labels" / "stars",
        "labels_targets": root_dir / "labels" / "targets",
        "labels_camera": root_dir / "labels" / "camera",
        "wcs": root_dir / "wcs",
        "splits": root_dir / "splits",
    }
    for path in dirs.values():
        ensure_dir(path)
    return dirs


def build_configs(cfg: dict[str, Any]) -> tuple[
    CameraConfig,
    PointingConfig,
    PhotometryConfig,
    PsfConfig,
    BackgroundConfig,
    TargetInjectorConfig,
]:
    """从 YAML 字典构建各类配置对象。

    输入:
    - `cfg`: 配置字典。

    输出:
    - 相机、指向、光度、PSF、背景、目标注入配置。
    """

    camera_cfg = cfg.get("camera", {})
    pointing_cfg = cfg.get("pointing", {})
    photometry_cfg = cfg.get("photometry", {})
    psf_cfg = cfg.get("psf", {})
    background_cfg = cfg.get("background", {})
    target_cfg = cfg.get("targets", {})

    camera = CameraConfig(
        width=int(camera_cfg.get("width", 1024)),
        height=int(camera_cfg.get("height", 1024)),
        fov_deg=float(camera_cfg.get("fov_deg", 1.5)),
        bit_depth=int(camera_cfg.get("bit_depth", 8)),
        max_adu=float(camera_cfg.get("max_adu", 255.0)),
    )
    pointing = PointingConfig(
        ra_center_deg=float(pointing_cfg.get("ra_center_deg", 180.0)),
        dec_center_deg=float(pointing_cfg.get("dec_center_deg", 0.0)),
        ra_jitter_deg=float(pointing_cfg.get("ra_jitter_deg", 0.2)),
        dec_jitter_deg=float(pointing_cfg.get("dec_jitter_deg", 0.2)),
        roll_min_deg=float(pointing_cfg.get("roll_min_deg", 0.0)),
        roll_max_deg=float(pointing_cfg.get("roll_max_deg", 360.0)),
    )
    photometry = PhotometryConfig(
        mag_zero_point=float(photometry_cfg.get("mag_zero_point", 14.0)),
        flux_at_zero_point=float(photometry_cfg.get("flux_at_zero_point", 1800.0)),
        min_flux=float(photometry_cfg.get("min_flux", 0.8)),
        max_flux=float(photometry_cfg.get("max_flux", 180000.0)),
    )
    psf = PsfConfig(
        psf_type=str(psf_cfg.get("type", "moffat")),
        fwhm_px_mean=float(psf_cfg.get("fwhm_px_mean", 2.4)),
        fwhm_px_std=float(psf_cfg.get("fwhm_px_std", 0.25)),
        beta=float(psf_cfg.get("beta", 3.2)),
        kernel_radius_factor=float(psf_cfg.get("kernel_radius_factor", 5.0)),
        halo_flux_threshold=float(psf_cfg.get("halo_flux_threshold", 8000.0)),
        halo_fraction=float(psf_cfg.get("halo_fraction", 0.03)),
        halo_fwhm_factor=float(psf_cfg.get("halo_fwhm_factor", 5.0)),
    )
    background = BackgroundConfig(
        level_mean=float(background_cfg.get("level_mean", 8.0)),
        level_std=float(background_cfg.get("level_std", 1.0)),
        gradient_amplitude=float(background_cfg.get("gradient_amplitude", 1.2)),
        vignette_amplitude=float(background_cfg.get("vignette_amplitude", 0.015)),
        read_noise_sigma=float(background_cfg.get("read_noise_sigma", 0.0)),
        shot_noise_scale=float(background_cfg.get("shot_noise_scale", 0.0)),
        unresolved_speckle_density=float(background_cfg.get("unresolved_speckle_density", 0.0)),
        unresolved_speckle_min=float(background_cfg.get("unresolved_speckle_min", 0.0)),
        unresolved_speckle_max=float(background_cfg.get("unresolved_speckle_max", 0.0)),
    )
    targets = build_target_config(target_cfg)
    return camera, pointing, photometry, psf, background, targets


def choose_projected_stars(
    catalog: GaiaCatalog,
    camera: CameraConfig,
    pointing_cfg: PointingConfig,
    rng: np.random.Generator,
    min_stars: int,
    max_attempts: int,
) -> tuple[ProjectedStars, Any, GaiaCatalog]:
    """采样指向并投影星点，直到星点数量满足要求。

    输入:
    - `catalog`: GaiaCatalog。
    - `camera`: 相机配置。
    - `pointing_cfg`: 指向采样配置。
    - `rng`: 随机数生成器。
    - `min_stars`: 每张图最少星点数。
    - `max_attempts`: 最大重采样次数。

    输出:
    - 投影星点、最终指向、视场附近星表子集。

    说明:
    - 当前星表只是局部 cone subset，所以采样中心不应偏离太大；
    - 如果多次采样仍不足，则返回最后一次结果，避免死循环。
    """

    query_radius = camera.fov_deg * 0.75 + 0.15
    best: tuple[ProjectedStars, Any, GaiaCatalog] | None = None
    best_count = -1

    for _ in range(max(1, max_attempts)):
        pointing = sample_pointing(pointing_cfg, rng)
        nearby = query_cone(catalog, pointing.ra_center_deg, pointing.dec_center_deg, query_radius)
        projected = project_to_image(nearby, pointing, camera, margin_px=10.0)
        if len(projected) > best_count:
            best = (projected, pointing, nearby)
            best_count = len(projected)
        if len(projected) >= min_stars:
            return projected, pointing, nearby

    assert best is not None
    return best


def star_records(stars: ProjectedStars, photometry: PhotometryConfig, fwhm_px: float) -> list[dict]:
    """生成星点 JSON 标签记录。

    输入:
    - `stars`: 投影后的星点。
    - `photometry`: 光度配置。
    - `fwhm_px`: 当前图像 PSF FWHM。

    输出:
    - 每颗星的字典列表。
    """

    fluxes = flux_from_mag(stars.g_mag, photometry)
    records: list[dict] = []
    for sid, ra, dec, mag, x, y, flux in zip(
        stars.source_id,
        stars.ra_deg,
        stars.dec_deg,
        stars.g_mag,
        stars.x_px,
        stars.y_px,
        fluxes,
        strict=False,
    ):
        records.append(
            {
                "source_id": str(sid),
                "ra_deg": float(ra),
                "dec_deg": float(dec),
                "g_mag": float(mag),
                "x_px": float(x),
                "y_px": float(y),
                "flux": float(flux),
                "psf_fwhm_px": float(fwhm_px),
            }
        )
    return records


def write_splits(root_dir: Path, image_ids: list[str]) -> None:
    """写 train/val/test split 占位文件。

    输入:
    - `root_dir`: clean 数据集根目录。
    - `image_ids`: 本次生成的图像 id。

    输出:
    - 无返回值。

    说明:
    - 当前 10 张预览图全部写入 train；
    - 后续正式生成时再按比例拆分。
    """

    split_dir = ensure_dir(root_dir / "splits")
    (split_dir / "train.txt").write_text("\n".join(image_ids) + "\n", encoding="utf-8")
    (split_dir / "val.txt").write_text("", encoding="utf-8")
    (split_dir / "test.txt").write_text("", encoding="utf-8")


def generate_clean_dataset(
    cfg: dict[str, Any],
    num_images_override: int | None = None,
    catalog_path_override: str | None = None,
    output_dir_override: str | None = None,
) -> dict[str, Any]:
    """生成 clean dataset。

    输入:
    - `cfg`: YAML 配置字典。
    - `num_images_override`: 可选，覆盖生成数量。
    - `catalog_path_override`: 可选，覆盖星表路径。
    - `output_dir_override`: 可选，覆盖输出目录。

    输出:
    - 生成摘要字典。
    """

    catalog_cfg = cfg.get("catalog", {})
    output_cfg = cfg.get("output", {})
    generation_cfg = cfg.get("generation", {})
    mask_cfg = cfg.get("masks", {})

    catalog_path = Path(catalog_path_override or catalog_cfg.get("path"))
    output_root = Path(output_dir_override or output_cfg.get("root_dir", "STAR_Agent/data/clean/synthetic_v001"))
    image_prefix = str(output_cfg.get("image_prefix", "synthetic_clean"))
    num_images = int(num_images_override or generation_cfg.get("num_images", 10))
    seed = int(generation_cfg.get("seed", 131))
    min_stars = int(generation_cfg.get("min_stars_per_image", 80))
    max_attempts = int(generation_cfg.get("max_resample_attempts", 30))
    star_radius_px = int(mask_cfg.get("star_radius_px", 4))

    camera, pointing_cfg, photometry, psf, background, targets_cfg = build_configs(cfg)
    dirs = make_output_dirs(output_root)

    manifest_path = output_root / "manifest.jsonl"
    if bool(output_cfg.get("overwrite_manifest", True)):
        manifest_path.write_text("", encoding="utf-8")

    catalog = load_gaia_csv(
        catalog_path,
        mag_limit=catalog_cfg.get("mag_limit"),
        ruwe_max=catalog_cfg.get("ruwe_max"),
    )
    catalog_summary = summarize_catalog(catalog)

    rng = np.random.default_rng(seed)
    progress = ProgressPrinter(num_images, stage="build_clean_data")
    image_ids: list[str] = []
    star_counts: list[int] = []
    target_counts: list[int] = []
    target_type_counts = {"point_blob": 0, "short_streak": 0}

    for idx in range(num_images):
        image_id = f"{image_prefix}_{idx:06d}"
        image_ids.append(image_id)

        stars, pointing, nearby = choose_projected_stars(
            catalog=catalog,
            camera=camera,
            pointing_cfg=pointing_cfg,
            rng=rng,
            min_stars=min_stars,
            max_attempts=max_attempts,
        )
        image, _star_layer, fwhm_px = render_star_field(
            stars=stars,
            height=camera.height,
            width=camera.width,
            photometry=photometry,
            psf=psf,
            background=background,
            rng=rng,
        )
        image, target_mask, target_records = inject_targets(
            image=image,
            star_x=stars.x_px,
            star_y=stars.y_px,
            config=targets_cfg,
            rng=rng,
        )
        star_mask = build_star_mask(stars, camera.height, camera.width, star_radius_px)
        background_mask = build_background_mask(star_mask)
        valid_mask = build_valid_mask(camera.height, camera.width)

        image_path = dirs["images"] / f"{image_id}.png"
        star_mask_path = dirs["mask_star"] / f"{image_id}.png"
        target_mask_path = dirs["mask_target"] / f"{image_id}.png"
        background_mask_path = dirs["mask_background"] / f"{image_id}.png"
        valid_mask_path = dirs["mask_valid"] / f"{image_id}.png"
        stars_label_path = dirs["labels_stars"] / f"{image_id}.json"
        targets_label_path = dirs["labels_targets"] / f"{image_id}.json"
        camera_label_path = dirs["labels_camera"] / f"{image_id}.json"

        save_rgb_png(image_path, image, max_adu=camera.max_adu)
        save_gray_png(star_mask_path, star_mask)
        save_gray_png(target_mask_path, target_mask)
        save_gray_png(background_mask_path, background_mask)
        save_gray_png(valid_mask_path, valid_mask)

        stars_json = {
            "image_id": image_id,
            "num_stars": len(stars),
            "stars": star_records(stars, photometry, fwhm_px),
        }
        targets_json = {
            "image_id": image_id,
            "num_targets": len(target_records),
            "target_type_counts": {
                "point_blob": sum(
                    1 for item in target_records if item.get("target_type") == "point_blob"
                ),
                "short_streak": sum(
                    1 for item in target_records if item.get("target_type") == "short_streak"
                ),
            },
            "targets": target_records,
        }
        camera_json = {
            "image_id": image_id,
            "width": camera.width,
            "height": camera.height,
            "fov_deg": camera.fov_deg,
            "pixel_scale_arcsec": camera.pixel_scale_arcsec,
            "pointing": {
                "ra_center_deg": pointing.ra_center_deg,
                "dec_center_deg": pointing.dec_center_deg,
                "roll_deg": pointing.roll_deg,
            },
            "psf": {
                "type": psf.psf_type,
                "fwhm_px": fwhm_px,
                "beta": psf.beta,
            },
            "catalog": {
                "path": str(catalog_path),
                "nearby_sources_before_projection": len(nearby),
            },
        }
        write_json(stars_label_path, stars_json)
        write_json(targets_label_path, targets_json)
        write_json(camera_label_path, camera_json)

        record = {
            "image_id": image_id,
            "image_path": str(image_path),
            "star_mask_path": str(star_mask_path),
            "target_mask_path": str(target_mask_path),
            "background_mask_path": str(background_mask_path),
            "valid_mask_path": str(valid_mask_path),
            "stars_label_path": str(stars_label_path),
            "targets_label_path": str(targets_label_path),
            "camera_label_path": str(camera_label_path),
            "num_stars": len(stars),
            "num_targets": len(target_records),
            "target_types": [item.get("target_type", "unknown") for item in target_records],
            "ra_center_deg": pointing.ra_center_deg,
            "dec_center_deg": pointing.dec_center_deg,
            "roll_deg": pointing.roll_deg,
            "psf_fwhm_px": fwhm_px,
        }
        append_jsonl(manifest_path, record)
        star_counts.append(len(stars))
        target_counts.append(len(target_records))
        for item in target_records:
            target_type = str(item.get("target_type", "unknown"))
            if target_type in target_type_counts:
                target_type_counts[target_type] += 1
        progress.update(
            idx + 1,
            extra=f"{image_id} stars={len(stars)} targets={len(target_records)}",
        )

    write_splits(output_root, image_ids)
    config_used_path = output_root / "config_used.yaml"
    config_used_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

    summary = {
        "output_root": str(output_root),
        "num_images": num_images,
        "catalog_summary": catalog_summary,
        "star_count_min": int(np.min(star_counts)) if star_counts else 0,
        "star_count_max": int(np.max(star_counts)) if star_counts else 0,
        "star_count_mean": float(np.mean(star_counts)) if star_counts else 0.0,
        "target_count_total": int(np.sum(target_counts)) if target_counts else 0,
        "target_count_mean": float(np.mean(target_counts)) if target_counts else 0.0,
        "target_type_counts": target_type_counts,
        "manifest_path": str(manifest_path),
    }
    write_json(output_root / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    输入:
    - 无，读取命令行。

    输出:
    - argparse Namespace。
    """

    parser = argparse.ArgumentParser(description="Build synthetic clean star images from Gaia DR3.")
    parser.add_argument(
        "--config",
        default="STAR_Agent/configs/data_generation/clean_simulation.yaml",
        help="clean simulation YAML config path.",
    )
    parser.add_argument("--num_images", type=int, default=None, help="Override number of images.")
    parser.add_argument("--catalog_path", default=None, help="Override Gaia catalog CSV path.")
    parser.add_argument("--output_dir", default=None, help="Override output root dir.")
    return parser.parse_args()


def main() -> int:
    """命令行主入口。

    输入:
    - 无，读取命令行。

    输出:
    - 进程退出码。
    """

    args = parse_args()
    cfg = load_config(args.config)
    summary = generate_clean_dataset(
        cfg,
        num_images_override=args.num_images,
        catalog_path_override=args.catalog_path,
        output_dir_override=args.output_dir,
    )
    print("[OK] clean dataset generated")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
