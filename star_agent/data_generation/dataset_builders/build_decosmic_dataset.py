#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Decosmic-ray tool dataset builder wrapper."""

from __future__ import annotations

import argparse

from .common import add_common_args, build_tool_dataset, load_config


def main() -> int:
    """只构建 decosmic_ray 子任务数据。

    输入:
    - 通用 tool dataset CLI 参数。

    输出:
    - 进程退出码。
    """

    parser = argparse.ArgumentParser(description="Build decosmic-ray executor datasets.")
    add_common_args(parser)
    args = parser.parse_args()
    args.subtask = ["decosmic_ray"]
    build_tool_dataset(load_config(args.config), args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
