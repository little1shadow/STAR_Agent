#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Unified CLI entry point for double/triple degradation generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def add_repo_path() -> Path:
    """Add STAR_Agent root to sys.path and return it."""

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    return root


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Build STAR-Agent multi-degradation datasets.")
    parser.add_argument("--config", default="configs/data_generation/degradation_multi.yaml")
    parser.add_argument("--stage", choices=["double", "triple", "all"], default="double")
    parser.add_argument("--source_single_root", default=None, help="Override single degradation root.")
    parser.add_argument("--double_input_dir", default=None, help="Override double input dir for triple stage.")
    parser.add_argument("--output_dir", default=None, help="Override current stage output dir.")
    parser.add_argument("--max_images", type=int, default=None, help="Override total output cap for the selected stage.")
    parser.add_argument("--per_combo_limit", type=int, default=None, help="Override per-combo output cap.")
    parser.add_argument("--combo", nargs="*", default=None, help="Optional combo filters, e.g. noise+dqg.")
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--max_new", type=int, default=None, help="Debug cap for newly generated samples.")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Run the selected multi-degradation stage."""

    add_repo_path()
    from star_agent.data_generation.composition.build_double_degradation import build_double_dataset
    from star_agent.data_generation.composition.build_triple_degradation import build_triple_dataset

    args = parse_args()
    if args.stage in {"double", "all"}:
        build_double_dataset(args)
    if args.stage in {"triple", "all"}:
        build_triple_dataset(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
