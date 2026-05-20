#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""通用 IO 工具。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在。

    输入:
    - `path`: 目录路径。

    输出:
    - Path 对象。
    """

    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, data: Any) -> None:
    """写 JSON 文件。

    输入:
    - `path`: 输出路径。
    - `data`: 可 JSON 序列化对象。

    输出:
    - 无返回值。
    """

    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: str | Path, record: dict) -> None:
    """追加一行 JSONL。

    输入:
    - `path`: JSONL 文件路径。
    - `record`: 当前记录。

    输出:
    - 无返回值。
    """

    p = Path(path)
    ensure_dir(p.parent)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
