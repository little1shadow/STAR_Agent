#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared helpers for multi-degradation generation."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from PIL import Image

from ..common.io import ensure_dir, write_json


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class DegradationSample:
    """A generated degradation sample discovered from a meta JSON file."""

    stage: str
    sample_id: str
    clean_image_id: str
    image_path: Path
    mask_path: Path | None
    field_path: Path | None
    meta_path: Path
    lineage_path: Path | None
    degradations: tuple[str, ...]
    ordered_degradations: tuple[str, ...]
    mode: str | None
    level: int | None
    seed: int | None
    clean_source: dict[str, Any]
    clean: dict[str, Any]
    steps: tuple[dict[str, Any], ...]
    meta: dict[str, Any]

    @property
    def combo_key(self) -> str:
        return combo_key(self.degradations)

    @property
    def primary_degradation(self) -> str:
        if len(self.degradations) != 1:
            raise ValueError(f"Sample is not a single-degradation support sample: {self.sample_id}")
        return self.degradations[0]


def project_root() -> Path:
    """Return the STAR_Agent repository root."""

    return Path(__file__).resolve().parents[3]


def resolve_path(path_value: str | Path | None, root: Path | None = None, must_exist: bool = False) -> Path | None:
    """Resolve paths written as absolute, repo-relative, or STAR_Agent-prefixed strings."""

    if path_value is None or str(path_value) == "":
        return None
    root = root or project_root()
    raw = Path(path_value)
    if raw.is_absolute():
        resolved = raw
    else:
        candidates = [Path.cwd() / raw, root / raw, root.parent / raw]
        if raw.parts and raw.parts[0] == root.name:
            candidates.append(root / Path(*raw.parts[1:]))
        resolved = candidates[-1]
        for item in candidates:
            if item.exists():
                resolved = item
                break
            if not must_exist and item.parent.exists():
                resolved = item
                break
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Path not found: {path_value}")
    return resolved.resolve()


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping."""

    p = resolve_path(path, must_exist=True)
    assert p is not None
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping: {p}")
    return data


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON object."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """Append one JSONL record."""

    p = Path(path)
    ensure_dir(p.parent)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_effect_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Load the single-degradation config used for replaying support parameters."""

    effect_cfg: dict[str, Any] = {}
    cfg_path = cfg.get("single_degradation_config")
    if cfg_path:
        p = resolve_path(cfg_path)
        if p is not None and p.exists():
            effect_cfg = load_yaml(p)
    merged = dict(effect_cfg)
    for key, value in cfg.items():
        if key not in merged:
            merged[key] = value
    return merged


def read_gray_float(path: str | Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    """Read a grayscale file as [0, 1] float32."""

    p = Path(path)
    arr = np.asarray(Image.open(p).convert("L"), dtype=np.float32)
    max_value = float(arr.max()) if arr.size else 0.0
    if max_value > 1.0:
        arr = arr / 255.0
    if shape is not None and arr.shape != shape:
        raise ValueError(f"Shape mismatch for {p}: expected {shape}, got {arr.shape}")
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def load_optional_gray(path: Path | None, shape: tuple[int, int]) -> np.ndarray:
    """Read an optional grayscale image, returning zeros when absent."""

    if path is None or not path.exists():
        return np.zeros(shape, dtype=np.float32)
    return read_gray_float(path, shape=shape)


def stage_dirs(output_dir: str | Path) -> dict[str, Path]:
    """Create and return flat stage output directories."""

    root = Path(output_dir)
    dirs = {
        "root": root,
        "images": root / "images",
        "masks": root / "masks",
        "fields": root / "fields",
        "meta": root / "meta",
        "lineage": root / "lineage",
        "manifests": root / "_manifests",
    }
    for path in dirs.values():
        ensure_dir(path)
    return dirs


def safe_name(value: str, max_len: int = 80) -> str:
    """Make a compact filesystem-safe token."""

    text = re.sub(r"[^A-Za-z0-9_.@+-]+", "_", str(value)).strip("_")
    if not text:
        text = "sample"
    return text[:max_len]


def combo_key(degradations: Iterable[str]) -> str:
    """Canonical combo key for unordered degradation sets."""

    return "__".join(sorted(dict.fromkeys(str(item) for item in degradations)))


def ordered_key(degradations: Iterable[str]) -> str:
    """Order-aware degradation key."""

    return "then".join(str(item) for item in degradations)


def parse_combo_filters(raw: list[str] | None) -> set[str] | None:
    """Parse CLI combo filters like noise+dqg or noise__dqg."""

    if not raw:
        return None
    parsed = set()
    for item in raw:
        parts = [p for p in re.split(r"[+,/]|__", item) if p]
        parsed.add(combo_key(parts))
    return parsed


def _lineage_candidate(meta_path: Path) -> Path:
    """Infer the lineage path adjacent to a meta path."""

    return meta_path.parent.parent / "lineage" / meta_path.name


def _load_lineage(meta: dict[str, Any], meta_path: Path) -> dict[str, Any]:
    """Load lineage if available, otherwise return an empty mapping."""

    raw = meta.get("lineage_path")
    lineage_path = resolve_path(raw) if raw else None
    if lineage_path is None:
        candidate = _lineage_candidate(meta_path)
        lineage_path = candidate if candidate.exists() else None
    if lineage_path is not None and lineage_path.exists():
        return load_json(lineage_path)
    return {}


def sample_from_meta(meta_path: Path, expected_stage: str | None = None) -> DegradationSample | None:
    """Build a DegradationSample from a meta JSON file."""

    meta = load_json(meta_path)
    stage = str(meta.get("stage") or expected_stage or "")
    if expected_stage and stage and stage != expected_stage:
        return None
    image_path = resolve_path(meta.get("image_path"), must_exist=True)
    if image_path is None:
        return None

    lineage = _load_lineage(meta, meta_path)
    lineage_path = resolve_path(meta.get("lineage_path")) if meta.get("lineage_path") else None
    if lineage_path is None:
        candidate = _lineage_candidate(meta_path)
        lineage_path = candidate.resolve() if candidate.exists() else None

    raw_degradations = meta.get("degradations")
    if raw_degradations is None:
        raw_degradations = meta.get("ordered_degradations")
    if raw_degradations is None and meta.get("degradation"):
        raw_degradations = [meta["degradation"]]
    if not raw_degradations:
        return None

    ordered = meta.get("ordered_degradations") or [str(item) for item in raw_degradations]
    degradations = tuple(str(item) for item in raw_degradations)
    ordered_degradations = tuple(str(item) for item in ordered)
    clean = lineage.get("clean") or {
        "image_id": meta.get("clean_image_id"),
        "image_path": meta.get("clean_image_path"),
        "star_mask_path": meta.get("clean_star_mask_path"),
        "target_mask_path": meta.get("clean_target_mask_path"),
        "background_mask_path": meta.get("clean_background_mask_path"),
        "valid_mask_path": meta.get("clean_valid_mask_path"),
        "stars_label_path": meta.get("clean_stars_label_path"),
        "targets_label_path": meta.get("clean_targets_label_path"),
        "camera_label_path": meta.get("clean_camera_label_path"),
    }
    clean_image_id = str(meta.get("clean_image_id") or clean.get("image_id") or "unknown_clean")
    clean_source = lineage.get("clean_source") or {
        "domain": meta.get("clean_source_domain"),
        "name": meta.get("clean_source_name"),
    }
    steps = tuple(copy.deepcopy(lineage.get("steps") or meta.get("steps") or []))
    mask_path = resolve_path(meta.get("mask_path")) if meta.get("mask_path") else None
    field_path = resolve_path(meta.get("field_path")) if meta.get("field_path") else None
    sample_id = str(meta.get("sample_id") or image_path.stem)
    level = meta.get("level")
    seed = meta.get("seed")

    return DegradationSample(
        stage=stage or expected_stage or "unknown",
        sample_id=sample_id,
        clean_image_id=clean_image_id,
        image_path=image_path,
        mask_path=mask_path,
        field_path=field_path,
        meta_path=meta_path.resolve(),
        lineage_path=lineage_path,
        degradations=tuple(sorted(dict.fromkeys(degradations))),
        ordered_degradations=ordered_degradations,
        mode=str(meta["mode"]) if meta.get("mode") is not None else None,
        level=int(level) if level is not None else None,
        seed=int(seed) if seed is not None else None,
        clean_source=clean_source,
        clean=clean,
        steps=steps,
        meta=meta,
    )


def scan_stage_samples(root: str | Path, expected_stage: str | None = None) -> list[DegradationSample]:
    """Scan a stage directory and return samples backed by meta JSON files."""

    base = resolve_path(root, must_exist=True)
    assert base is not None
    samples: list[DegradationSample] = []
    for meta_path in sorted(base.rglob("*.json")):
        if meta_path.parent.name != "meta":
            continue
        try:
            sample = sample_from_meta(meta_path, expected_stage=expected_stage)
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            continue
        if sample is not None:
            samples.append(sample)
    return samples


def group_single_by_clean_and_degradation(
    samples: Iterable[DegradationSample],
) -> dict[str, dict[str, list[DegradationSample]]]:
    """Group single samples as clean_id -> degradation -> samples."""

    grouped: dict[str, dict[str, list[DegradationSample]]] = {}
    for sample in samples:
        if len(sample.degradations) != 1:
            continue
        grouped.setdefault(sample.clean_image_id, {}).setdefault(sample.primary_degradation, []).append(sample)
    return grouped


def group_by_clean(samples: Iterable[DegradationSample]) -> dict[str, list[DegradationSample]]:
    """Group samples by clean image id."""

    grouped: dict[str, list[DegradationSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.clean_image_id, []).append(sample)
    return grouped


def existing_signatures(samples: Iterable[DegradationSample]) -> set[str]:
    """Collect lineage signatures from existing multi samples."""

    out: set[str] = set()
    for sample in samples:
        sig = sample.meta.get("lineage_signature")
        if sig:
            out.add(str(sig))
    return out


def count_by_combo(samples: Iterable[DegradationSample]) -> dict[str, int]:
    """Count samples per canonical combo."""

    counts: dict[str, int] = {}
    for sample in samples:
        counts[sample.combo_key] = counts.get(sample.combo_key, 0) + 1
    return counts


def combine_parent_and_step_masks(
    parent: DegradationSample,
    step_mask: np.ndarray,
    step_field: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Union parent masks and max-combine continuous fields."""

    shape = step_mask.shape
    parent_mask = load_optional_gray(parent.mask_path, shape)
    parent_field = load_optional_gray(parent.field_path, shape)
    step_mask_float = (step_mask > 0).astype(np.float32)
    combined_mask = ((parent_mask > 0) | (step_mask_float > 0)).astype(np.uint8) * 255
    combined_field = np.maximum(parent_field, np.clip(step_field, 0.0, 1.0)).astype(np.float32)
    return combined_mask, combined_field


def compact_sample_record(sample: DegradationSample) -> dict[str, Any]:
    """Return a compact serializable sample summary."""

    return {
        "stage": sample.stage,
        "sample_id": sample.sample_id,
        "clean_image_id": sample.clean_image_id,
        "degradations": list(sample.degradations),
        "ordered_degradations": list(sample.ordered_degradations),
        "mode": sample.mode,
        "level": sample.level,
        "seed": sample.seed,
        "image_path": str(sample.image_path),
        "mask_path": str(sample.mask_path) if sample.mask_path else None,
        "field_path": str(sample.field_path) if sample.field_path else None,
        "meta_path": str(sample.meta_path),
        "lineage_path": str(sample.lineage_path) if sample.lineage_path else None,
    }


def build_multi_lineage(
    *,
    stage: str,
    sample_id: str,
    parent: DegradationSample,
    support: DegradationSample,
    output_image_path: Path,
    output_mask_path: Path,
    output_field_path: Path,
    output_meta_path: Path,
    output_lineage_path: Path,
    step_meta: dict[str, Any],
) -> dict[str, Any]:
    """Build a lineage record for double/triple outputs."""

    steps = [copy.deepcopy(item) for item in parent.steps]
    step_index = len(steps) + 1
    steps.append(
        {
            "step_index": step_index,
            "input_stage": parent.stage,
            "input_image_path": str(parent.image_path),
            "output_stage": stage,
            "output_image_path": str(output_image_path),
            "degradation": support.primary_degradation,
            "mode": support.mode,
            "level": support.level,
            "seed": support.seed,
            "mask_path": str(output_mask_path),
            "field_path": str(output_field_path),
            "meta_path": str(output_meta_path),
            "support_single": compact_sample_record(support),
            "step_meta": step_meta,
        }
    )
    clean_source = parent.clean_source
    clean = parent.clean
    return {
        "stage": stage,
        "sample_id": sample_id,
        "clean_source": clean_source,
        "clean": clean,
        "parent": compact_sample_record(parent),
        "support_single": compact_sample_record(support),
        "steps": steps,
        "current": {
            "image_path": str(output_image_path),
            "degradation_mask_path": str(output_mask_path),
            "degradation_field_path": str(output_field_path),
            "meta_path": str(output_meta_path),
            "lineage_path": str(output_lineage_path),
        },
    }


def write_summary(path: str | Path, summary: dict[str, Any]) -> None:
    """Write a summary JSON."""

    write_json(path, summary)
