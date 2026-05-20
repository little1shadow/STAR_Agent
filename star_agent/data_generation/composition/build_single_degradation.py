#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""生成 STAR-Agent single degradation 数据集。

这个脚本是正式批量生成入口，不是 10 张 preview 入口。
核心目标：
- 从 clean manifest 随机抽取图像。
- 按 degradation / mode / level 分层输出。
- 每个 mode 的每个 level 默认补到 200 张。
- 多终端并行时使用 level 级 lock，避免同一目录被多个进程同时写爆。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

import yaml

from ..common.io import ensure_dir, write_json
from ..degradations._single_common import load_rgb_float, save_gray_float, save_rgb_float
from ..degradations.cosmic_ray import COSMIC_RAY_MODES, add_cosmic_ray
from ..degradations.dead_pixels import DEAD_PIXEL_MODES, add_dead_pixels
from ..degradations.dqg import DQG_MODES, add_dqg
from ..degradations.motion_blur import MOTION_BLUR_MODES, add_motion_blur
from ..degradations.noise import NOISE_MODES, add_noise
from ..degradations.smear import SMEAR_MODES, add_smear
from ..degradations.solar_stray_light import SOLAR_STRAY_LIGHT_MODES, add_solar_stray_light

ImageAddFunc = Callable[..., tuple[Any, Any, Any, dict[str, Any]]]

DEGRADATION_REGISTRY: dict[str, dict[str, Any]] = {
    "noise": {"modes": NOISE_MODES, "func": add_noise},
    "smear": {"modes": SMEAR_MODES, "func": add_smear},
    "cosmic_ray": {"modes": COSMIC_RAY_MODES, "func": add_cosmic_ray},
    "dead_pixels": {"modes": DEAD_PIXEL_MODES, "func": add_dead_pixels},
    "dqg": {"modes": DQG_MODES, "func": add_dqg},
    "solar_stray_light": {"modes": SOLAR_STRAY_LIGHT_MODES, "func": add_solar_stray_light},
    "motion_blur": {"modes": MOTION_BLUR_MODES, "func": add_motion_blur},
}

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def project_root() -> Path:
    """返回 STAR_Agent 仓库根目录。

    输入:
    - 无。

    输出:
    - `STAR_Agent/` 的绝对路径。

    用途:
    - 让脚本无论从 `AgenticIR/` 还是 `AgenticIR/STAR_Agent/` 执行，都能正确解析配置里的路径。
    """

    return Path(__file__).resolve().parents[3]


def resolve_path(path_value: str | Path, root: Path | None = None) -> Path:
    """解析配置或 manifest 中的路径。

    输入:
    - `path_value`: 相对路径或绝对路径。
    - `root`: STAR_Agent 仓库根目录；为 None 时自动推断。

    输出:
    - 解析后的绝对 Path。

    说明:
    - 兼容两类写法：`STAR_Agent/data/...` 和 `data/...`。
    """

    root = root or project_root()
    p = Path(path_value)
    if p.is_absolute():
        return p
    candidates = [
        Path.cwd() / p,
        root / p,
        root.parent / p,
    ]
    if p.parts and p.parts[0] == root.name:
        candidates.append(root / Path(*p.parts[1:]))
    for item in candidates:
        if item.exists() or item.parent.exists():
            return item.resolve()
    return (root / p).resolve()


def load_yaml(path: str | Path) -> dict[str, Any]:
    """读取 YAML 配置。

    输入:
    - `path`: 配置文件路径。

    输出:
    - 配置字典。
    """

    p = resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """读取 clean manifest。

    输入:
    - `path`: JSONL 文件路径，每行对应一张 clean 图。

    输出:
    - clean 记录列表。
    """

    records: list[dict[str, Any]] = []
    p = resolve_path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise RuntimeError(f"No clean records found in manifest: {p}")
    return records


def image_count(images_dir: Path) -> int:
    """统计当前 level 已经生成的图片数量。

    输入:
    - `images_dir`: 当前 level 的 images 目录。

    输出:
    - 图片数量，不统计 `.gitkeep`。
    """

    if not images_dir.exists():
        return 0
    return sum(1 for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


@contextmanager
def directory_lock(lock_dir: Path, stale_seconds: int = 7200, poll_seconds: float = 0.25):
    """基于 mkdir 的轻量目录锁。

    输入:
    - `lock_dir`: 锁目录路径。
    - `stale_seconds`: 超过该时间的锁视为上次异常中断残留，可以清理。
    - `poll_seconds`: 等待锁释放时的轮询间隔。

    输出:
    - 上下文管理器，无显式返回。

    用途:
    - 多终端同时生成时，保证同一个 `degradation/mode/level` 目录不会同时写入，
      从而避免明显超过 200 张。
    """

    ensure_dir(lock_dir.parent)
    while True:
        try:
            os.mkdir(lock_dir)
            (lock_dir / "owner.json").write_text(
                json.dumps({"pid": os.getpid(), "time": time.time()}, ensure_ascii=False),
                encoding="utf-8",
            )
            break
        except FileExistsError:
            try:
                age = time.time() - lock_dir.stat().st_mtime
                if age > stale_seconds:
                    shutil.rmtree(lock_dir, ignore_errors=True)
                    continue
            except FileNotFoundError:
                continue
            time.sleep(poll_seconds)
    try:
        yield
    finally:
        shutil.rmtree(lock_dir, ignore_errors=True)


def level_dirs(output_root: Path, degradation: str, mode: str, level: int) -> dict[str, Path]:
    """构造当前 degradation/mode/level 的输出目录。

    输入:
    - `output_root`: single 根目录，例如 `data/degraded/synthetic/synthetic_v002_targets/single`。
    - `degradation`: 退化类型。
    - `mode`: 退化子模式。
    - `level`: 退化等级。

    输出:
    - 包含 images/masks/fields/meta/locks 的目录字典。
    """

    base = output_root / degradation / mode / f"level_{level}"
    dirs = {
        "base": base,
        "images": base / "images",
        "masks": base / "masks",
        "fields": base / "fields",
        "meta": base / "meta",
        "locks": base / ".locks",
    }
    for key, path in dirs.items():
        if key != "locks":
            ensure_dir(path)
    ensure_dir(dirs["locks"])
    return dirs


def current_manifest_path(output_root: Path, degradation: str) -> Path:
    """返回当前退化类型的生成 manifest 路径。

    输入:
    - `output_root`: single 根目录。
    - `degradation`: 退化类型。

    输出:
    - JSONL manifest 路径。
    """

    return output_root / degradation / "_manifests" / f"{degradation}_single_manifest.jsonl"


def append_manifest(path: Path, record: dict[str, Any]) -> None:
    """追加写入 single degradation manifest。

    输入:
    - `path`: manifest 路径。
    - `record`: 当前样本元信息。

    输出:
    - 无。
    """

    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def threshold_map(cfg: dict[str, Any], degradation: str) -> dict[int, float] | None:
    """从配置中读取当前退化的 mask 阈值。

    输入:
    - `cfg`: 配置字典。
    - `degradation`: 退化类型。

    输出:
    - `{level: threshold}` 或 None。
    """

    raw = cfg.get(degradation, {}).get("mask_threshold_by_level")
    if not raw:
        return None
    return {int(k): float(v) for k, v in raw.items()}


def call_degradation(
    cfg: dict[str, Any],
    degradation: str,
    image: Any,
    level: int,
    mode: str,
    seed: int,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """调用具体退化函数。

    输入:
    - `cfg`: 配置字典。
    - `degradation`: 退化类型。
    - `image`: clean 图像数组。
    - `level`: 退化等级。
    - `mode`: 退化子模式。
    - `seed`: 当前样本随机种子。

    输出:
    - degraded image、mask、field、meta。

    说明:
    - solar stray light 额外需要 side_weights，所以单独处理。
    """

    func: ImageAddFunc = DEGRADATION_REGISTRY[degradation]["func"]
    thresholds = threshold_map(cfg, degradation)
    if degradation == "solar_stray_light":
        solar_cfg = cfg.get("solar_stray_light", {})
        return func(
            image=image,
            level=level,
            seed=seed,
            mode=mode,
            side_weights=solar_cfg.get("side_weights"),
            mask_threshold_by_level=thresholds,
        )
    return func(image=image, level=level, seed=seed, mode=mode, mask_threshold_by_level=thresholds)


def select_degradations(args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, list[str]]:
    """确定本次需要生成哪些 degradation 和 mode。

    输入:
    - `args`: 命令行参数。
    - `cfg`: 配置字典。

    输出:
    - `{degradation: [mode, ...]}`。
    """

    cfg_categories = cfg.get("categories", {})
    requested = args.degradation or list(cfg_categories.keys() or DEGRADATION_REGISTRY.keys())
    selected: dict[str, list[str]] = {}
    for degradation in requested:
        if degradation not in DEGRADATION_REGISTRY:
            raise ValueError(f"Unknown degradation: {degradation}")
        cfg_modes = cfg_categories.get(degradation, {}).get("modes")
        modes = list(cfg_modes or DEGRADATION_REGISTRY[degradation]["modes"])
        if args.mode:
            modes = [m for m in modes if m in set(args.mode)]
        if not modes:
            raise ValueError(f"No modes selected for {degradation}")
        selected[degradation] = modes
    return selected


def make_output_name(clean_id: str, degradation: str, mode: str, level: int, seq: int) -> str:
    """生成唯一输出文件名。

    输入:
    - `clean_id`: clean 图像 ID。
    - `degradation`: 退化类型。
    - `mode`: 子模式。
    - `level`: 等级。
    - `seq`: 当前目录中的序号。

    输出:
    - 不含扩展名前缀。
    """

    token = uuid.uuid4().hex[:8]
    safe_mode = mode.replace("/", "_")
    return f"{clean_id}@{degradation}@{safe_mode}@l{level}@{seq:04d}_{token}"


def generate_one(
    cfg: dict[str, Any],
    clean_records: list[dict[str, Any]],
    output_root: Path,
    degradation: str,
    mode: str,
    level: int,
    rng: random.Random,
    seed: int,
    clean_source: dict[str, Any],
) -> dict[str, Any]:
    """生成一张 single degradation 图。

    输入:
    - `cfg`: 配置字典。
    - `clean_records`: clean manifest 记录列表。
    - `output_root`: single 输出根目录。
    - `degradation`: 退化类型。
    - `mode`: 子模式。
    - `level`: 等级。
    - `rng`: 当前进程随机数生成器。
    - `seed`: 当前样本退化随机种子。
    - `clean_source`: clean 来源信息。

    输出:
    - 当前样本 meta。
    """

    dirs = level_dirs(output_root, degradation, mode, level)
    before_count = image_count(dirs["images"])
    clean_record = rng.choice(clean_records)
    clean_id = str(clean_record.get("image_id") or Path(clean_record["image_path"]).stem)
    clean_path = resolve_path(clean_record["image_path"])
    image = load_rgb_float(clean_path)
    degraded, mask, field, meta = call_degradation(cfg, degradation, image, level, mode, seed)

    name = make_output_name(clean_id, degradation, mode, level, before_count + 1)
    image_path = dirs["images"] / f"{name}.png"
    mask_path = dirs["masks"] / f"{name}.png"
    field_path = dirs["fields"] / f"{name}.png"
    meta_path = dirs["meta"] / f"{name}.json"

    save_rgb_float(image_path, degraded)
    save_gray_float(mask_path, mask)
    save_gray_float(field_path, field)

    meta.update(
        {
            "stage": "single",
            "clean_source_domain": clean_source.get("domain"),
            "clean_source_name": clean_source.get("name"),
            "clean_image_id": clean_id,
            "clean_image_path": str(clean_path),
            "source_clean_record": clean_record,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "field_path": str(field_path),
            "meta_path": str(meta_path),
        }
    )
    write_json(meta_path, meta)
    append_manifest(current_manifest_path(output_root, degradation), meta)
    return meta


def build_single_dataset(args: argparse.Namespace) -> dict[str, Any]:
    """批量补齐 single degradation 数据集。

    输入:
    - `args`: 命令行参数。

    输出:
    - summary 字典。

    调度逻辑:
    - 遍历 degradation -> mode -> level。
    - 每进入一个 level 先统计已有图片数，够 200 就跳过。
    - 每生成 10 张重新统计一次，适配多进程/多终端同时生成。
    - 每写一张图都在当前 level 的 lock 下进行，避免并发写入超量。
    """

    cfg = load_yaml(args.config)
    clean_source = cfg.get("clean_source") or cfg.get("source_clean") or {}
    manifest_path = args.manifest or clean_source.get("manifest_path")
    if not manifest_path:
        raise ValueError("Missing clean manifest. Set clean_source.manifest_path or --manifest.")
    clean_records = read_jsonl(manifest_path)

    output_cfg = cfg.get("output", {})
    output_root = resolve_path(args.output_root or output_cfg.get("root_dir"))
    ensure_dir(output_root)

    selected = select_degradations(args, cfg)
    levels = args.level or [int(x) for x in cfg.get("levels", [1, 2, 3, 4, 5])]
    target = int(args.per_level_target)
    recount_interval = max(1, int(args.recount_interval))
    rng = random.Random(args.seed)

    summary: dict[str, Any] = {
        "config": str(resolve_path(args.config)),
        "manifest": str(resolve_path(manifest_path)),
        "output_root": str(output_root),
        "per_level_target": target,
        "recount_interval": recount_interval,
        "clean_source": clean_source,
        "generated": 0,
        "skipped_slots": [],
        "processed_slots": [],
    }

    if args.dry_run:
        print("[DRY-RUN] single degradation generation plan")
        print(json.dumps({**summary, "selected": selected, "levels": levels}, ensure_ascii=False, indent=2))
        return summary

    for degradation, modes in selected.items():
        for mode in modes:
            for level in levels:
                dirs = level_dirs(output_root, degradation, mode, level)
                slot = f"{degradation}/{mode}/level_{level}"
                current = image_count(dirs["images"])
                print(f"[CHECK] {slot} | current={current} target={target}")
                if current >= target:
                    print(f"[SKIP] {slot} already has {current} images")
                    summary["skipped_slots"].append({"slot": slot, "count": current})
                    continue

                slot_generated = 0
                while current < target:
                    lock_path = dirs["locks"] / "write.lock"
                    with directory_lock(lock_path, stale_seconds=args.stale_lock_seconds):
                        current = image_count(dirs["images"])
                        if current >= target:
                            break
                        sample_seed = args.seed + summary["generated"] + os.getpid() + int(time.time() * 1000) % 100000
                        meta = generate_one(
                            cfg=cfg,
                            clean_records=clean_records,
                            output_root=output_root,
                            degradation=degradation,
                            mode=mode,
                            level=level,
                            rng=rng,
                            seed=sample_seed,
                            clean_source=clean_source,
                        )
                        summary["generated"] += 1
                        slot_generated += 1
                        current = image_count(dirs["images"])
                        print(
                            f"[GENERATED] {slot} | current={current}/{target} | "
                            f"clean={meta['clean_image_id']} | out={Path(meta['image_path']).name}"
                        )

                    if slot_generated % recount_interval == 0:
                        current = image_count(dirs["images"])
                        print(
                            f"[RECOUNT] {slot} | current={current}/{target} | "
                            f"slot_generated={slot_generated} | total_generated={summary['generated']}"
                        )
                    if args.max_new is not None and summary["generated"] >= args.max_new:
                        print(f"[STOP] reach --max_new={args.max_new}")
                        summary["processed_slots"].append({"slot": slot, "count": current, "generated": slot_generated})
                        write_json(output_root / "_manifests" / "single_generation_summary.json", summary)
                        return summary

                current = image_count(dirs["images"])
                print(f"[DONE] {slot} | current={current}/{target} | generated_now={slot_generated}")
                summary["processed_slots"].append({"slot": slot, "count": current, "generated": slot_generated})

    write_json(output_root / "_manifests" / "single_generation_summary.json", summary)
    print("[OK] single degradation generation finished")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    输入:
    - 命令行参数。

    输出:
    - argparse Namespace。
    """

    parser = argparse.ArgumentParser(description="Build STAR-Agent single degradation dataset.")
    parser.add_argument("--config", default="configs/data_generation/degradation_single.yaml")
    parser.add_argument("--manifest", default=None, help="Override clean manifest path.")
    parser.add_argument("--output_root", default=None, help="Override single degradation output root.")
    parser.add_argument("--degradation", nargs="*", default=None, help="Degradation names. Default: all in config.")
    parser.add_argument("--mode", nargs="*", default=None, help="Optional mode filter.")
    parser.add_argument("--level", nargs="*", type=int, default=None, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--per_level_target", type=int, default=200)
    parser.add_argument("--recount_interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--max_new", type=int, default=None, help="Debug limit for newly generated images.")
    parser.add_argument("--dry_run", action="store_true", help="Only print generation plan.")
    parser.add_argument("--stale_lock_seconds", type=int, default=7200)
    return parser.parse_args()


def main() -> int:
    """命令行入口。

    输入:
    - 无，读取命令行参数。

    输出:
    - 进程退出码。
    """

    build_single_dataset(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
