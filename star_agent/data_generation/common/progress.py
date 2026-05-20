#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""简单进度输出工具。"""

from __future__ import annotations

import time


class ProgressPrinter:
    """命令行进度打印器。

    用途:
    - 在生成数据时显示当前进度、速度和预计剩余时间。
    """

    def __init__(self, total: int, stage: str = "progress", report_every: int = 50) -> None:
        """初始化进度打印器。

        输入:
        - `total`: 总任务数。
        - `stage`: 阶段名称。
        - `report_every`: 每隔多少步打印一次，避免大规模生成时刷屏。
        """

        self.total = max(1, int(total))
        self.stage = stage
        self.report_every = max(1, int(report_every))
        self.started = time.time()

    def update(self, current: int, extra: str = "") -> None:
        """打印当前进度。

        输入:
        - `current`: 已完成数量。
        - `extra`: 可选附加信息。

        输出:
        - 无返回值。
        """

        if current not in (1, self.total) and current % self.report_every != 0:
            return

        now = time.time()
        elapsed = max(now - self.started, 1e-6)
        speed = current / elapsed
        remain = max(self.total - current, 0)
        eta = remain / max(speed, 1e-6)
        pct = 100.0 * current / self.total
        msg = (
            f"[PROGRESS] {self.stage} | {current}/{self.total} "
            f"({pct:.2f}%) | speed={speed:.2f}/s | eta={eta:.1f}s"
        )
        if extra:
            msg += f" | {extra}"
        print(msg, flush=True)
