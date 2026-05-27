#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build double-degradation samples from single-degradation lineage."""

from __future__ import annotations

import argparse
import json
import random
import uuid
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from ..common.io import write_json
from ..degradations._single_common import load_rgb_float, save_gray_float, save_rgb_float
from .build_single_degradation import call_degradation
from .lineage import (
    DegradationSample,
    append_jsonl,
    build_multi_lineage,
    combine_parent_and_step_masks,
    combo_key,
    count_by_combo,
    existing_signatures,
    group_single_by_clean_and_degradation,
    load_effect_config,
    load_yaml,
    ordered_key,
    parse_combo_filters,
    resolve_path,
    safe_name,
    scan_stage_samples,
    stage_dirs,
    write_summary,
)


def resolve_double_output_dir(cfg: dict[str, Any], override: str | None = None) -> Path:
    """Resolve the double stage output directory."""

    if override:
        path = resolve_path(override)
    else:
        stage_cfg = cfg.get("stages", {}).get("double", {}) or {}
        path = resolve_path(stage_cfg.get("output_dir") or Path(cfg["output_root"]) / "double")
    assert path is not None
    return path


def double_signature(parent: DegradationSample, support: DegradationSample) -> str:
    """Build an order-aware signature used to avoid duplicate regeneration."""

    return f"double:{parent.sample_id}->{support.sample_id}"


def collect_double_candidates(
    single_samples: list[DegradationSample],
    rng: random.Random,
    combo_filter: set[str] | None,
    candidate_cap_per_combo: int,
) -> list[tuple[str, DegradationSample, DegradationSample]]:
    """Collect same-clean double candidates."""

    grouped = group_single_by_clean_and_degradation(single_samples)
    candidates: list[tuple[str, DegradationSample, DegradationSample]] = []
    combo_counts: dict[str, int] = {}
    for _clean_id, by_deg in grouped.items():
        degradations = sorted(by_deg)
        for deg_a, deg_b in combinations(degradations, 2):
            combo = combo_key([deg_a, deg_b])
            if combo_filter is not None and combo not in combo_filter:
                continue
            if combo_counts.get(combo, 0) >= candidate_cap_per_combo:
                continue
            samples_a = list(by_deg[deg_a])
            samples_b = list(by_deg[deg_b])
            rng.shuffle(samples_a)
            rng.shuffle(samples_b)
            for sample_a in samples_a:
                if combo_counts.get(combo, 0) >= candidate_cap_per_combo:
                    break
                for sample_b in samples_b:
                    if combo_counts.get(combo, 0) >= candidate_cap_per_combo:
                        break
                    if (
                        sample_a.mode is None
                        or sample_a.level is None
                        or sample_a.seed is None
                        or sample_b.mode is None
                        or sample_b.level is None
                        or sample_b.seed is None
                    ):
                        continue
                    if rng.random() < 0.5:
                        parent, support = sample_a, sample_b
                    else:
                        parent, support = sample_b, sample_a
                    candidates.append((combo, parent, support))
                    combo_counts[combo] = combo_counts.get(combo, 0) + 1
    rng.shuffle(candidates)
    return candidates


def generate_double_one(
    *,
    cfg: dict[str, Any],
    output_dir: Path,
    combo: str,
    parent: DegradationSample,
    support: DegradationSample,
) -> dict[str, Any]:
    """Generate one double-degradation sample."""

    if support.mode is None or support.level is None or support.seed is None:
        raise ValueError(f"Support sample lacks mode/level/seed: {support.sample_id}")

    dirs = stage_dirs(output_dir)
    parent_image = load_rgb_float(parent.image_path)
    degraded, step_mask, step_field, step_meta = call_degradation(
        cfg=cfg,
        degradation=support.primary_degradation,
        image=parent_image,
        level=int(support.level),
        mode=str(support.mode),
        seed=int(support.seed),
    )
    combined_mask, combined_field = combine_parent_and_step_masks(parent, step_mask, step_field)
    ordered_degradations = tuple(parent.ordered_degradations) + (support.primary_degradation,)
    sample_id = safe_name(
        f"{parent.clean_image_id}@double@{ordered_key(ordered_degradations)}@{uuid.uuid4().hex[:8]}",
        max_len=150,
    )
    image_path = dirs["images"] / f"{sample_id}.png"
    mask_path = dirs["masks"] / f"{sample_id}.png"
    field_path = dirs["fields"] / f"{sample_id}.png"
    meta_path = dirs["meta"] / f"{sample_id}.json"
    lineage_path = dirs["lineage"] / f"{sample_id}.json"

    save_rgb_float(image_path, degraded)
    save_gray_float(mask_path, combined_mask)
    save_gray_float(field_path, combined_field)

    lineage = build_multi_lineage(
        stage="double",
        sample_id=sample_id,
        parent=parent,
        support=support,
        output_image_path=image_path,
        output_mask_path=mask_path,
        output_field_path=field_path,
        output_meta_path=meta_path,
        output_lineage_path=lineage_path,
        step_meta=step_meta,
    )
    meta = {
        "sample_id": sample_id,
        "stage": "double",
        "degradation": "multi",
        "degradations": sorted(set(parent.degradations + support.degradations)),
        "ordered_degradations": list(ordered_degradations),
        "combo": combo,
        "lineage_signature": double_signature(parent, support),
        "clean_source_domain": parent.clean_source.get("domain"),
        "clean_source_name": parent.clean_source.get("name"),
        "clean_image_id": parent.clean_image_id,
        "clean_image_path": parent.clean.get("image_path"),
        "clean_star_mask_path": parent.clean.get("star_mask_path"),
        "clean_target_mask_path": parent.clean.get("target_mask_path"),
        "clean_background_mask_path": parent.clean.get("background_mask_path"),
        "clean_valid_mask_path": parent.clean.get("valid_mask_path"),
        "clean_stars_label_path": parent.clean.get("stars_label_path"),
        "clean_targets_label_path": parent.clean.get("targets_label_path"),
        "parent": {
            "stage": parent.stage,
            "sample_id": parent.sample_id,
            "image_path": str(parent.image_path),
            "degradations": list(parent.degradations),
            "ordered_degradations": list(parent.ordered_degradations),
        },
        "support_single": {
            "sample_id": support.sample_id,
            "image_path": str(support.image_path),
            "degradation": support.primary_degradation,
            "mode": support.mode,
            "level": support.level,
            "seed": support.seed,
            "mask_path": str(support.mask_path) if support.mask_path else None,
            "field_path": str(support.field_path) if support.field_path else None,
            "meta_path": str(support.meta_path),
            "lineage_path": str(support.lineage_path) if support.lineage_path else None,
        },
        "last_step_meta": step_meta,
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "field_path": str(field_path),
        "meta_path": str(meta_path),
        "lineage_path": str(lineage_path),
        "mask_area_ratio": float(np.mean(combined_mask > 0)),
        "field_mean": float(np.mean(combined_field)),
        "field_max": float(np.max(combined_field)),
    }
    write_json(meta_path, meta)
    write_json(lineage_path, lineage)
    append_jsonl(dirs["manifests"] / "double_manifest.jsonl", meta)
    return meta


def build_double_dataset(args: argparse.Namespace) -> dict[str, Any]:
    """Build double-degradation data from single-degradation samples."""

    cfg = load_yaml(args.config)
    effect_cfg = load_effect_config(cfg)
    source_single_root = resolve_path(args.source_single_root or cfg.get("source_single_root"), must_exist=True)
    if source_single_root is None:
        raise ValueError("Missing source_single_root in config or CLI.")
    output_dir = resolve_double_output_dir(cfg, args.output_dir)
    stage_cfg = cfg.get("stages", {}).get("double", {}) or {}
    max_images = args.max_images if args.max_images is not None else stage_cfg.get("max_images")
    per_combo_limit = args.per_combo_limit if args.per_combo_limit is not None else stage_cfg.get("per_combo_limit", 4000)
    max_images = None if max_images is None else int(max_images)
    per_combo_limit = int(per_combo_limit)
    rng = random.Random(args.seed)
    combo_filter = parse_combo_filters(args.combo)

    single_samples = scan_stage_samples(source_single_root, expected_stage="single")
    existing = scan_stage_samples(output_dir, expected_stage="double") if output_dir.exists() else []
    counts = count_by_combo(existing)
    signatures = existing_signatures(existing)
    candidate_cap = max(per_combo_limit * 8, per_combo_limit + 100)
    candidates = collect_double_candidates(single_samples, rng, combo_filter, candidate_cap)
    summary: dict[str, Any] = {
        "stage": "double",
        "config": str(resolve_path(args.config)),
        "source_single_root": str(source_single_root),
        "output_dir": str(output_dir),
        "single_samples": len(single_samples),
        "existing_samples": len(existing),
        "candidate_pairs": len(candidates),
        "per_combo_limit": per_combo_limit,
        "max_images": max_images,
        "generated": 0,
        "skipped_existing_signature": 0,
        "skipped_limit": 0,
        "counts_before": dict(counts),
        "generated_by_combo": {},
    }

    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary

    total_existing = len(existing)
    for combo, parent, support in candidates:
        if max_images is not None and total_existing + summary["generated"] >= max_images:
            break
        if counts.get(combo, 0) >= per_combo_limit:
            summary["skipped_limit"] += 1
            continue
        sig = double_signature(parent, support)
        if sig in signatures:
            summary["skipped_existing_signature"] += 1
            continue
        meta = generate_double_one(
            cfg=effect_cfg,
            output_dir=output_dir,
            combo=combo,
            parent=parent,
            support=support,
        )
        signatures.add(sig)
        counts[combo] = counts.get(combo, 0) + 1
        summary["generated"] += 1
        summary["generated_by_combo"][combo] = summary["generated_by_combo"].get(combo, 0) + 1
        print(
            f"[GENERATED] double {summary['generated']} | combo={combo} | "
            f"parent={parent.sample_id} + support={support.sample_id} | out={Path(meta['image_path']).name}"
        )
        if args.max_new is not None and summary["generated"] >= args.max_new:
            break

    summary["counts_after"] = dict(counts)
    dirs = stage_dirs(output_dir)
    write_summary(dirs["manifests"] / "double_generation_summary.json", summary)
    print("[OK] double degradation generation finished")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    """Parse direct double builder arguments."""

    parser = argparse.ArgumentParser(description="Build STAR-Agent double degradation dataset.")
    parser.add_argument("--config", default="configs/data_generation/degradation_multi.yaml")
    parser.add_argument("--source_single_root", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--per_combo_limit", type=int, default=None)
    parser.add_argument("--combo", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--max_new", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""

    build_double_dataset(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
