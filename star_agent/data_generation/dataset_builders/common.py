#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Executor/tool paired dataset builder.

本模块负责把 STAR-Agent 已经生成好的 single / double / triple degradation 数据，
转换成各个 restoration executor 可以直接训练和测试的 paired dataset。

核心原则：
- 按 domain/source 分开生成，例如 synthetic/synthetic_v002_targets 和 real/real_selected_v001。
- 训练某个 executor 时，target 不是永远等于 clean，而是“只移除当前 executor 负责的退化，保留其他退化”。
- split 按 clean_image_id 划分，避免同一张 clean 的不同退化版本同时出现在 train/test 中造成泄漏。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..common.io import ensure_dir, write_json
from ..composition.build_single_degradation import call_degradation
from ..composition.lineage import (
    DegradationSample,
    load_effect_config,
    load_yaml,
    read_gray_float,
    resolve_path,
    scan_stage_samples,
)
from ..degradations._single_common import load_rgb_float, save_gray_float, save_rgb_float

IMAGE_SUFFIX = ".png"
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class ToolPair:
    """一个 executor 训练/测试样本对。

    输入:
    - `input_path`: 当前带退化图像。
    - `target_path`: 只移除当前 subtask 负责退化后的目标图像。
    - `task_mask_path`: 当前 subtask 负责退化的 mask。
    - `task_field_path`: 当前 subtask 负责退化的连续场，可选。

    输出:
    - dataclass 记录本身，后续会被写入 manifest。
    """

    pair_id: str
    subtask: str
    sample: DegradationSample
    removed_degradations: tuple[str, ...]
    remaining_degradations: tuple[str, ...]
    input_path: Path
    target_path: Path
    task_mask_path: Path | None
    task_field_path: Path | None
    target_mode: str
    stage: str


def project_root() -> Path:
    """返回 STAR_Agent 仓库根目录。

    输入:
    - 无。

    输出:
    - `STAR_Agent/` 的绝对路径。
    """

    return Path(__file__).resolve().parents[3]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """读取 JSONL 文件。

    输入:
    - `path`: JSONL 文件路径。

    输出:
    - 记录列表；空行会被跳过。
    """

    p = Path(path)
    records: list[dict[str, Any]] = []
    if not p.exists():
        return records
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """追加写入一条 JSONL 记录。

    输入:
    - `path`: 输出 JSONL 路径。
    - `record`: 当前 pair 元信息。

    输出:
    - 无。
    """

    p = Path(path)
    ensure_dir(p.parent)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_config(path: str | Path) -> dict[str, Any]:
    """读取 tool dataset 配置。

    输入:
    - `path`: YAML 配置路径。

    输出:
    - 配置字典。
    """

    cfg = load_yaml(path)
    if not isinstance(cfg, dict):
        raise ValueError(f"Tool dataset config must be a mapping: {path}")
    return cfg


def normalize_degradation_name(name: str) -> str:
    """统一退化名称写法。

    输入:
    - `name`: 配置或 meta 中的退化名。

    输出:
    - 用下划线表示的规范名称。
    """

    return str(name).strip().replace(" ", "_")


def split_clean_ids(clean_ids: Iterable[str], split_cfg: dict[str, Any]) -> dict[str, str]:
    """按 clean_image_id 生成 train/val/test 划分。

    输入:
    - `clean_ids`: 所有样本涉及的 clean id。
    - `split_cfg`: 包含 train/val/test 比例和 seed。

    输出:
    - `{clean_image_id: split}`。

    设计目的:
    - 防止同一张 clean 图的不同退化版本同时进入 train 和 test。
    """

    ids = sorted({str(x) for x in clean_ids})
    rng = random.Random(int(split_cfg.get("seed", 131)))
    rng.shuffle(ids)
    n = len(ids)
    train_ratio = float(split_cfg.get("train", 0.8))
    val_ratio = float(split_cfg.get("val", 0.1))
    train_n = int(round(n * train_ratio))
    val_n = int(round(n * val_ratio))
    if train_n + val_n > n:
        val_n = max(0, n - train_n)
    out: dict[str, str] = {}
    for idx, clean_id in enumerate(ids):
        if idx < train_n:
            out[clean_id] = "train"
        elif idx < train_n + val_n:
            out[clean_id] = "val"
        else:
            out[clean_id] = "test"
    return out


def sample_stage_roots(degraded_root: Path, include_multi: bool) -> list[tuple[str, Path]]:
    """返回需要扫描的 degradation stage 目录。

    输入:
    - `degraded_root`: `data/degraded/<domain>/<source>/`。
    - `include_multi`: 是否包含 double/triple。

    输出:
    - `(stage, path)` 列表。
    """

    roots = [("single", degraded_root / "single")]
    if include_multi:
        roots.extend([("double", degraded_root / "double"), ("triple", degraded_root / "triple")])
    return [(stage, path) for stage, path in roots if path.exists()]


def collect_samples(degraded_root: Path, include_multi: bool) -> list[DegradationSample]:
    """扫描 single/double/triple meta，收集可用退化样本。

    输入:
    - `degraded_root`: 退化数据根目录。
    - `include_multi`: 是否读取 double/triple。

    输出:
    - DegradationSample 列表。
    """

    samples: list[DegradationSample] = []
    for stage, root in sample_stage_roots(degraded_root, include_multi):
        stage_samples = scan_stage_samples(root, expected_stage=stage)
        print(f"[INFO] scanned {stage}: {len(stage_samples)} samples from {root}")
        samples.extend(stage_samples)
    return samples


def target_step_degradations(sample: DegradationSample, removed: set[str]) -> list[dict[str, Any]]:
    """从 lineage steps 中选出当前 subtask 需要移除的退化步骤。

    输入:
    - `sample`: 当前样本。
    - `removed`: 当前 subtask 负责的退化集合。

    输出:
    - 需要移除的 step 列表。
    """

    return [step for step in sample.steps if normalize_degradation_name(step.get("degradation", "")) in removed]


def remaining_steps(sample: DegradationSample, removed: set[str]) -> list[dict[str, Any]]:
    """从 lineage steps 中保留非当前 subtask 的退化步骤。

    输入:
    - `sample`: 当前样本。
    - `removed`: 当前 subtask 负责的退化集合。

    输出:
    - 需要保留并体现在 target 中的退化步骤。
    """

    return [step for step in sample.steps if normalize_degradation_name(step.get("degradation", "")) not in removed]


def is_prefix_steps(all_steps: list[dict[str, Any]], kept_steps: list[dict[str, Any]]) -> bool:
    """判断 kept steps 是否为原始 steps 的前缀。

    输入:
    - `all_steps`: 原始退化链。
    - `kept_steps`: 移除当前退化后需要保留的链。

    输出:
    - True 表示可以直接复用前缀最后一步的输出图。
    """

    if len(kept_steps) > len(all_steps):
        return False
    for left, right in zip(all_steps[: len(kept_steps)], kept_steps):
        if int(left.get("step_index", -1)) != int(right.get("step_index", -2)):
            return False
    return True


def step_single_support_path(step: dict[str, Any]) -> Path | None:
    """读取某一步对应的 single support 图像路径。

    输入:
    - `step`: lineage 中的一个 step。

    输出:
    - support single 图像路径；没有则返回 None。
    """

    support = step.get("support_single") or {}
    image_path = support.get("image_path")
    if not image_path:
        return None
    p = resolve_path(image_path)
    if p is not None and p.exists():
        return p
    return None


def clean_image_path(sample: DegradationSample) -> Path:
    """返回当前样本对应 clean 图路径。

    输入:
    - `sample`: 当前退化样本。

    输出:
    - clean 图绝对路径。
    """

    path = resolve_path(sample.clean.get("image_path"), must_exist=True)
    if path is None:
        raise FileNotFoundError(f"Missing clean image for sample: {sample.sample_id}")
    return path


def reusable_target_path(sample: DegradationSample, kept_steps: list[dict[str, Any]]) -> tuple[Path | None, str]:
    """尽量复用已有 clean/parent/support 图作为 target。

    输入:
    - `sample`: 当前样本。
    - `kept_steps`: target 中需要保留的退化步骤。

    输出:
    - `(path, mode)`；如果无法复用则 path 为 None。

    设计目的:
    - 多退化训练集中大量 target 其实已经存在于 parent 或 support single 中。
    - 复用可显著减少重建计算和磁盘写入。
    """

    all_steps = list(sample.steps)
    if not kept_steps:
        return clean_image_path(sample), "reuse_clean"

    if is_prefix_steps(all_steps, kept_steps):
        last = kept_steps[-1]
        path_value = last.get("output_image_path")
        if path_value:
            p = resolve_path(path_value)
            if p is not None and p.exists():
                return p, "reuse_prefix_parent"

    if len(kept_steps) == 1:
        support_path = step_single_support_path(kept_steps[0])
        if support_path is not None:
            return support_path, "reuse_support_single"

    parent = sample.meta.get("parent") or {}
    parent_path = parent.get("image_path")
    parent_degs = {normalize_degradation_name(x) for x in parent.get("degradations", [])}
    kept_degs = {normalize_degradation_name(step.get("degradation", "")) for step in kept_steps}
    if parent_path and parent_degs == kept_degs:
        p = resolve_path(parent_path)
        if p is not None and p.exists():
            return p, "reuse_parent_meta"

    return None, "rebuild_replay"


def replay_target_image(
    *,
    sample: DegradationSample,
    kept_steps: list[dict[str, Any]],
    effect_cfg: dict[str, Any],
    output_path: Path,
) -> Path:
    """按 lineage 参数重放保留退化，生成 target 图。

    输入:
    - `sample`: 当前退化样本。
    - `kept_steps`: target 中要保留的退化步骤。
    - `effect_cfg`: single degradation 配置，用于重放退化。
    - `output_path`: target 输出路径。

    输出:
    - 生成后的 target 路径。
    """

    if output_path.exists():
        return output_path
    image = load_rgb_float(clean_image_path(sample))
    for step in kept_steps:
        degradation = normalize_degradation_name(step["degradation"])
        mode = str(step["mode"])
        level = int(step["level"])
        seed = int(step["seed"])
        image, _mask, _field, _meta = call_degradation(
            cfg=effect_cfg,
            degradation=degradation,
            image=image,
            level=level,
            mode=mode,
            seed=seed,
        )
    save_rgb_float(output_path, image)
    return output_path


def step_mask_path(step: dict[str, Any]) -> Path | None:
    """读取 step 对应的单步退化 mask。

    输入:
    - `step`: lineage step。

    输出:
    - mask 路径；没有则返回 None。

    注意:
    - double/triple 的 step['mask_path'] 往往是累计 mask。
    - 对 support step，优先使用 support_single['mask_path']，这样得到的是当前 step 自身的 mask。
    """

    support = step.get("support_single") or {}
    raw = support.get("mask_path") or step.get("mask_path")
    if not raw:
        return None
    p = resolve_path(raw)
    if p is not None and p.exists():
        return p
    return None


def step_field_path(step: dict[str, Any]) -> Path | None:
    """读取 step 对应的单步退化 field。

    输入:
    - `step`: lineage step。

    输出:
    - field 路径；没有则返回 None。
    """

    support = step.get("support_single") or {}
    raw = support.get("field_path") or step.get("field_path")
    if not raw:
        return None
    p = resolve_path(raw)
    if p is not None and p.exists():
        return p
    return None


def union_gray(paths: Iterable[Path | None], output_path: Path) -> Path | None:
    """把多个灰度 mask/field 合并保存。

    输入:
    - `paths`: 待合并灰度图路径。
    - `output_path`: 输出路径。

    输出:
    - 输出路径；如果没有任何有效输入，则返回 None。
    """

    valid = [p for p in paths if p is not None and p.exists()]
    if not valid:
        return None
    if output_path.exists():
        return output_path
    merged: np.ndarray | None = None
    for path in valid:
        arr = read_gray_float(path)
        merged = arr if merged is None else np.maximum(merged, arr)
    assert merged is not None
    save_gray_float(output_path, merged)
    return output_path


def make_pair_id(sample: DegradationSample, subtask: str, removed: Iterable[str]) -> str:
    """生成稳定的 pair id。

    输入:
    - `sample`: 当前样本。
    - `subtask`: executor 子任务。
    - `removed`: 当前 pair 中要移除的退化。

    输出:
    - 文件名安全的 pair id。
    """

    removed_token = "-".join(sorted(removed))
    return f"{sample.sample_id}@remove-{subtask}-{removed_token}"


def build_pair_for_sample(
    *,
    sample: DegradationSample,
    subtask: str,
    removed_degradations: set[str],
    effect_cfg: dict[str, Any],
    cache_root: Path,
) -> ToolPair | None:
    """为一个退化样本构建当前 subtask 的 input/target pair。

    输入:
    - `sample`: single/double/triple 退化样本。
    - `subtask`: executor 子任务名。
    - `removed_degradations`: 当前 subtask 负责移除的退化集合。
    - `effect_cfg`: 用于重放 target 的退化配置。
    - `cache_root`: 需要重放 target/mask 时的缓存目录。

    输出:
    - ToolPair；如果样本不包含该 subtask 负责退化，则返回 None。
    """

    sample_degs = {normalize_degradation_name(x) for x in sample.degradations}
    removed = tuple(sorted(sample_degs & removed_degradations))
    if not removed:
        return None

    kept_steps = remaining_steps(sample, removed_degradations)
    removed_steps = target_step_degradations(sample, removed_degradations)
    pair_id = make_pair_id(sample, subtask, removed)
    target_path, mode = reusable_target_path(sample, kept_steps)
    if target_path is None:
        target_path = replay_target_image(
            sample=sample,
            kept_steps=kept_steps,
            effect_cfg=effect_cfg,
            output_path=cache_root / "targets" / f"{pair_id}{IMAGE_SUFFIX}",
        )

    task_mask_path = union_gray(
        [step_mask_path(step) for step in removed_steps],
        cache_root / "task_masks" / f"{pair_id}{IMAGE_SUFFIX}",
    )
    task_field_path = union_gray(
        [step_field_path(step) for step in removed_steps],
        cache_root / "task_fields" / f"{pair_id}{IMAGE_SUFFIX}",
    )

    remaining_degs = tuple(sorted({normalize_degradation_name(step.get("degradation", "")) for step in kept_steps}))
    return ToolPair(
        pair_id=pair_id,
        subtask=subtask,
        sample=sample,
        removed_degradations=removed,
        remaining_degradations=remaining_degs,
        input_path=sample.image_path,
        target_path=target_path,
        task_mask_path=task_mask_path,
        task_field_path=task_field_path,
        target_mode=mode,
        stage=sample.stage,
    )


def maybe_filter_by_level(pair: ToolPair, min_removed_level: int) -> bool:
    """根据当前 subtask 负责退化的 level 过滤样本。

    输入:
    - `pair`: 当前 pair。
    - `min_removed_level`: 最低退化等级。

    输出:
    - True 表示保留。
    """

    if min_removed_level <= 1:
        return True
    removed = set(pair.removed_degradations)
    levels = [int(step.get("level", 1)) for step in pair.sample.steps if normalize_degradation_name(step.get("degradation", "")) in removed]
    return bool(levels) and max(levels) >= min_removed_level


def stage_weighted_subsample(
    pairs: list[ToolPair],
    max_pairs: int | None,
    stage_weights: dict[str, float] | None,
    seed: int,
) -> list[ToolPair]:
    """按 single/double/triple 比例可选抽样。

    输入:
    - `pairs`: 候选 pair。
    - `max_pairs`: 最大 pair 数；为 None 时不抽样。
    - `stage_weights`: stage 权重，例如 single/double/triple = 0.6/0.3/0.1。
    - `seed`: 随机种子。

    输出:
    - 抽样后的 pair 列表。
    """

    if max_pairs is None or len(pairs) <= max_pairs:
        return pairs
    rng = random.Random(seed)
    by_stage: dict[str, list[ToolPair]] = {stage: [] for stage in ("single", "double", "triple")}
    for pair in pairs:
        by_stage.setdefault(pair.stage, []).append(pair)
    weights = stage_weights or {"single": 0.6, "double": 0.3, "triple": 0.1}
    selected: list[ToolPair] = []
    remaining_budget = max_pairs
    for stage in ("single", "double", "triple"):
        bucket = by_stage.get(stage, [])
        rng.shuffle(bucket)
        want = int(round(max_pairs * float(weights.get(stage, 0.0))))
        take = min(want, len(bucket), remaining_budget)
        selected.extend(bucket[:take])
        remaining_budget -= take
    if remaining_budget > 0:
        already = {id(pair) for pair in selected}
        rest = [pair for pair in pairs if id(pair) not in already]
        rng.shuffle(rest)
        selected.extend(rest[:remaining_budget])
    rng.shuffle(selected)
    return selected[:max_pairs]


def link_or_copy(src: Path, dst: Path, mode: str) -> str:
    """把源文件放入工具数据集目录。

    输入:
    - `src`: 源文件。
    - `dst`: 目标路径。
    - `mode`: `hardlink` / `symlink` / `copy`。

    输出:
    - 实际采用的模式。
    """

    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        return "exists"
    if mode == "symlink":
        rel = os.path.relpath(src, dst.parent)
        os.symlink(rel, dst)
        return "symlink"
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            shutil.copy2(src, dst)
            return "copy_fallback"
    shutil.copy2(src, dst)
    return "copy"


def optional_link(src: Path | None, dst: Path, mode: str) -> str | None:
    """可选文件 link/copy。

    输入:
    - `src`: 源文件，可能为空。
    - `dst`: 输出路径。
    - `mode`: link/copy 模式。

    输出:
    - 输出路径字符串；没有源文件则返回 None。
    """

    if src is None or not src.exists():
        return None
    link_or_copy(src, dst, mode)
    return str(dst)


def clean_optional_path(sample: DegradationSample, key: str) -> Path | None:
    """读取 clean 相关 mask/label 路径。

    输入:
    - `sample`: 当前样本。
    - `key`: clean 字段名，例如 star_mask_path。

    输出:
    - Path 或 None。
    """

    raw = sample.clean.get(key)
    if not raw:
        return None
    p = resolve_path(raw)
    if p is not None and p.exists():
        return p
    return None


def output_dirs(root: Path, split: str) -> dict[str, Path]:
    """返回一个 split 下的标准输出目录。

    输入:
    - `root`: 当前 tool 数据集根目录。
    - `split`: train/val/test。

    输出:
    - 各类输出目录。
    """

    base = root / split
    dirs = {
        "input": base / "input",
        "target": base / "target",
        "degradation_mask": base / "degradation_mask",
        "degradation_field": base / "degradation_field",
        "star_mask": base / "star_mask",
        "target_mask": base / "target_mask",
        "background_mask": base / "background_mask",
        "valid_mask": base / "valid_mask",
        "meta": base / "meta",
    }
    for path in dirs.values():
        ensure_dir(path)
    return dirs


def materialize_pair(
    *,
    pair: ToolPair,
    output_root: Path,
    split: str,
    link_mode: str,
    domain: str,
    clean_source_name: str,
    tool_id: str,
) -> dict[str, Any]:
    """把 ToolPair 写成具体 input/target/mask 文件和 manifest 记录。

    输入:
    - `pair`: 当前 pair。
    - `output_root`: 当前 tool 根目录。
    - `split`: train/val/test。
    - `link_mode`: hardlink/symlink/copy。
    - `domain`: synthetic/real。
    - `clean_source_name`: clean 数据源名称。
    - `tool_id`: 工具版本名。

    输出:
    - manifest 记录。
    """

    dirs = output_dirs(output_root, split)
    name = f"{pair.pair_id}{IMAGE_SUFFIX}"
    input_out = dirs["input"] / name
    target_out = dirs["target"] / name
    link_or_copy(pair.input_path, input_out, link_mode)
    link_or_copy(pair.target_path, target_out, link_mode)

    task_mask_out = optional_link(pair.task_mask_path, dirs["degradation_mask"] / name, link_mode)
    task_field_out = optional_link(pair.task_field_path, dirs["degradation_field"] / name, link_mode)
    star_mask_out = optional_link(clean_optional_path(pair.sample, "star_mask_path"), dirs["star_mask"] / name, link_mode)
    target_mask_out = optional_link(clean_optional_path(pair.sample, "target_mask_path"), dirs["target_mask"] / name, link_mode)
    bg_mask_out = optional_link(clean_optional_path(pair.sample, "background_mask_path"), dirs["background_mask"] / name, link_mode)
    valid_mask_out = optional_link(clean_optional_path(pair.sample, "valid_mask_path"), dirs["valid_mask"] / name, link_mode)

    record = {
        "pair_id": pair.pair_id,
        "domain": domain,
        "clean_source": clean_source_name,
        "subtask": pair.subtask,
        "tool_id": tool_id,
        "split": split,
        "stage": pair.stage,
        "clean_image_id": pair.sample.clean_image_id,
        "sample_id": pair.sample.sample_id,
        "degradations": list(pair.sample.degradations),
        "ordered_degradations": list(pair.sample.ordered_degradations),
        "removed_degradations": list(pair.removed_degradations),
        "remaining_degradations": list(pair.remaining_degradations),
        "target_mode": pair.target_mode,
        "input_path": str(input_out),
        "target_path": str(target_out),
        "degradation_mask_path": task_mask_out,
        "degradation_field_path": task_field_out,
        "star_mask_path": star_mask_out,
        "target_mask_path": target_mask_out,
        "background_mask_path": bg_mask_out,
        "valid_mask_path": valid_mask_out,
        "source_input_path": str(pair.input_path),
        "source_target_path": str(pair.target_path),
        "source_meta_path": str(pair.sample.meta_path),
        "source_lineage_path": str(pair.sample.lineage_path) if pair.sample.lineage_path else None,
    }
    append_jsonl(output_root / split / "manifest.jsonl", record)
    write_json(dirs["meta"] / f"{pair.pair_id}.json", record)
    return record


def reset_split_manifests(tool_root: Path) -> None:
    """清空 train/val/test manifest，避免续跑时重复追加。

    输入:
    - `tool_root`: 当前 tool 数据集根目录。

    输出:
    - 无。
    """

    for split in SPLITS:
        manifest = tool_root / split / "manifest.jsonl"
        if manifest.exists():
            manifest.unlink()


def tool_splits(base_splits: dict[str, Any], tool_cfg: dict[str, Any]) -> dict[str, Any]:
    """合并全局 split 和 tool 专属 split。

    输入:
    - `base_splits`: 配置中的全局 split。
    - `tool_cfg`: 当前工具配置。

    输出:
    - 最终 split 配置。
    """

    merged = dict(base_splits)
    if isinstance(tool_cfg.get("splits"), dict):
        merged.update(tool_cfg["splits"])
    return merged


def selected_tool_ids(subtask_cfg: dict[str, Any], only_tools: set[str] | None) -> list[str]:
    """确定当前 subtask 需要生成哪些 tool 数据目录。

    输入:
    - `subtask_cfg`: 当前 subtask 配置。
    - `only_tools`: CLI 指定的工具过滤集合。

    输出:
    - tool id 列表。
    """

    tools = subtask_cfg.get("tools") or [subtask_cfg.get("dataset_name") or "default_v001"]
    out: list[str] = []
    for item in tools:
        tool_id = item.get("id") if isinstance(item, dict) else str(item)
        if only_tools is not None and tool_id not in only_tools:
            continue
        out.append(tool_id)
    return out


def tool_cfg_by_id(subtask_cfg: dict[str, Any], tool_id: str) -> dict[str, Any]:
    """根据 tool id 取工具配置。

    输入:
    - `subtask_cfg`: 当前 subtask 配置。
    - `tool_id`: 工具版本名。

    输出:
    - 工具配置字典。
    """

    for item in subtask_cfg.get("tools") or []:
        if isinstance(item, dict) and item.get("id") == tool_id:
            return dict(item)
    return {"id": tool_id}


def build_subtask_pairs(
    *,
    cfg: dict[str, Any],
    subtask: str,
    subtask_cfg: dict[str, Any],
    samples: list[DegradationSample],
    effect_cfg: dict[str, Any],
    cache_root: Path,
    seed: int,
) -> list[ToolPair]:
    """构建某个 subtask 的全部候选 pairs。

    输入:
    - `cfg`: 全局配置。
    - `subtask`: 子任务名。
    - `subtask_cfg`: 子任务配置。
    - `samples`: 所有退化样本。
    - `effect_cfg`: 退化重放配置。
    - `cache_root`: target/mask 缓存目录。
    - `seed`: 随机种子。

    输出:
    - ToolPair 列表。
    """

    removed = {normalize_degradation_name(x) for x in subtask_cfg.get("degradations", [])}
    if not removed:
        raise ValueError(f"Subtask has no degradations configured: {subtask}")
    min_level = int(subtask_cfg.get("min_removed_level", cfg.get("min_removed_level", 1)))
    max_pairs = subtask_cfg.get("max_pairs") or cfg.get("max_pairs_per_subtask")
    max_pairs = int(max_pairs) if max_pairs is not None else None
    stage_weights = subtask_cfg.get("stage_weights") or cfg.get("stage_weights")

    pairs: list[ToolPair] = []
    for idx, sample in enumerate(samples, start=1):
        pair = build_pair_for_sample(
            sample=sample,
            subtask=subtask,
            removed_degradations=removed,
            effect_cfg=effect_cfg,
            cache_root=cache_root / subtask,
        )
        if pair is None:
            continue
        if not maybe_filter_by_level(pair, min_level):
            continue
        pairs.append(pair)
        if idx % 5000 == 0:
            print(f"[PROGRESS] {subtask}: scanned={idx} kept={len(pairs)}")

    pairs = stage_weighted_subsample(pairs, max_pairs, stage_weights, seed)
    print(f"[INFO] subtask={subtask} candidate_pairs={len(pairs)} removed={sorted(removed)}")
    return pairs


def build_tool_dataset(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """构建当前配置下的 executor/tool 数据集。

    输入:
    - `cfg`: tool_dataset YAML 配置。
    - `args`: CLI 参数。

    输出:
    - summary 字典，同时写入 output_root/_manifests/tool_dataset_summary.json。
    """

    root = project_root()
    clean_source = dict(cfg.get("clean_source") or {})
    domain = str(clean_source.get("domain") or cfg.get("domain") or "unknown")
    clean_source_name = str(clean_source.get("name") or "unknown_source")
    degraded_root = resolve_path(args.degraded_root or cfg.get("degraded_root"), root=root, must_exist=True)
    output_root = resolve_path(args.output_root or cfg.get("output_root"), root=root)
    if degraded_root is None or output_root is None:
        raise ValueError("degraded_root/output_root is required")
    include_multi = bool(cfg.get("include_multi_degradation", True))
    link_mode = str(args.link_mode or cfg.get("link_mode", "hardlink"))
    seed = int(args.seed if args.seed is not None else cfg.get("splits", {}).get("seed", 131))
    effect_cfg = load_effect_config(cfg)
    samples = collect_samples(degraded_root, include_multi=include_multi)
    if args.max_source_samples is not None:
        rng = random.Random(seed)
        rng.shuffle(samples)
        samples = samples[: int(args.max_source_samples)]
        print(f"[INFO] max_source_samples applied: {len(samples)}")

    only_subtasks = set(args.subtask or []) or None
    only_tools = set(args.tool or []) or None

    if getattr(args, "dry_run", False):
        dry_summary = {
            "domain": domain,
            "clean_source": clean_source_name,
            "degraded_root": str(degraded_root),
            "output_root": str(output_root),
            "source_samples": len(samples),
            "stages": {},
            "subtasks": {},
        }
        for sample in samples:
            dry_summary["stages"][sample.stage] = dry_summary["stages"].get(sample.stage, 0) + 1
        for subtask, subtask_cfg in (cfg.get("subtasks") or {}).items():
            if only_subtasks is not None and subtask not in only_subtasks:
                continue
            removed = {normalize_degradation_name(x) for x in subtask_cfg.get("degradations", [])}
            dry_summary["subtasks"][subtask] = sum(
                1 for sample in samples if {normalize_degradation_name(x) for x in sample.degradations} & removed
            )
        print(json.dumps(dry_summary, ensure_ascii=False, indent=2))
        return dry_summary

    base_splits = dict(cfg.get("splits") or {"train": 0.8, "val": 0.1, "test": 0.1, "seed": seed})
    summary: dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "domain": domain,
        "clean_source": clean_source_name,
        "degraded_root": str(degraded_root),
        "output_root": str(output_root),
        "link_mode": link_mode,
        "source_samples": len(samples),
        "subtasks": {},
    }

    for subtask, subtask_cfg in (cfg.get("subtasks") or {}).items():
        if only_subtasks is not None and subtask not in only_subtasks:
            continue
        subtask_pairs = build_subtask_pairs(
            cfg=cfg,
            subtask=subtask,
            subtask_cfg=subtask_cfg,
            samples=samples,
            effect_cfg=effect_cfg,
            cache_root=output_root / "_cache",
            seed=seed,
        )
        clean_split = split_clean_ids((pair.sample.clean_image_id for pair in subtask_pairs), tool_splits(base_splits, subtask_cfg))
        tool_ids = selected_tool_ids(subtask_cfg, only_tools)
        sub_summary: dict[str, Any] = {
            "candidate_pairs": len(subtask_pairs),
            "removed_degradations": subtask_cfg.get("degradations", []),
            "tools": {},
        }
        for tool_id in tool_ids:
            tcfg = tool_cfg_by_id(subtask_cfg, tool_id)
            splits = tool_splits(tool_splits(base_splits, subtask_cfg), tcfg)
            split_map = split_clean_ids((pair.sample.clean_image_id for pair in subtask_pairs), splits)
            tool_root = output_root / subtask / tool_id
            reset_split_manifests(tool_root)
            counts = {split: 0 for split in SPLITS}
            stage_counts: dict[str, int] = {}
            for idx, pair in enumerate(subtask_pairs, start=1):
                split = split_map.get(pair.sample.clean_image_id) or clean_split.get(pair.sample.clean_image_id) or "train"
                if split not in SPLITS:
                    split = "train"
                record = materialize_pair(
                    pair=pair,
                    output_root=tool_root,
                    split=split,
                    link_mode=link_mode,
                    domain=domain,
                    clean_source_name=clean_source_name,
                    tool_id=tool_id,
                )
                counts[split] += 1
                stage_counts[record["stage"]] = stage_counts.get(record["stage"], 0) + 1
                if idx % 1000 == 0:
                    print(f"[PROGRESS] {subtask}/{tool_id}: {idx}/{len(subtask_pairs)} pairs materialized")
            tool_summary = {
                "root": str(tool_root),
                "counts": counts,
                "stage_counts": stage_counts,
                "splits": splits,
            }
            write_json(tool_root / "dataset_summary.json", tool_summary)
            sub_summary["tools"][tool_id] = tool_summary
            print(f"[OK] {subtask}/{tool_id}: {counts}")
        summary["subtasks"][subtask] = sub_summary

    write_json(output_root / "_manifests" / "tool_dataset_summary.json", summary)
    print(f"[OK] tool dataset summary saved: {output_root / '_manifests' / 'tool_dataset_summary.json'}")
    return summary


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """给 CLI parser 添加通用参数。

    输入:
    - `parser`: argparse parser。

    输出:
    - 同一个 parser，便于链式使用。
    """

    parser.add_argument("--config", default="configs/data_generation/tool_dataset.yaml", help="tool dataset YAML config")
    parser.add_argument("--subtask", nargs="*", default=None, help="only build selected subtasks")
    parser.add_argument("--tool", nargs="*", default=None, help="only build selected tool ids")
    parser.add_argument("--degraded_root", default=None, help="override degraded source root")
    parser.add_argument("--output_root", default=None, help="override output root")
    parser.add_argument("--link_mode", choices=["hardlink", "symlink", "copy"], default=None, help="how to materialize files")
    parser.add_argument("--seed", type=int, default=None, help="override split seed")
    parser.add_argument("--max_source_samples", type=int, default=None, help="debug: scan at most N source samples")
    parser.add_argument("--dry_run", action="store_true", help="scan and summarize without materializing files")
    return parser
