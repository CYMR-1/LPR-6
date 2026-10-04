# -*- coding: utf-8 -*-
"""仅由磁盘上的运行日志重建汇总表（不训练、不评测）。

为什么需要
----------
汇总表 ``reports/tables/exp_E*_summary.csv`` 原本由 ``reports/run_all.py``
写出，而 run_all 会**先训练再汇总**。合成域测试集切换后端后需要"只重测
不重训"，因此本脚本复用 run_all 里同一套纯函数
（``scan_runs_from_disk`` / ``aggregate`` / ``write_runs_csv`` /
``write_summary_csv``），只从 ``reports/logs/*_run.json`` 重建：

* ``exp_<Ei>_runs.csv`` 逐运行明细；
* ``exp_<Ei>_summary.csv`` 按变体聚合的均值±标准差；
* ``all_experiments_runs.csv`` / ``all_experiments_summary.csv``。

不读配置里的"启用实验"列表、不调用训练入口，因此不可能重训。

用法
----
    python reports/rebuild_summary_tables.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.config import ensure_dirs, load_config, resolve_path  # noqa: E402
from reports.run_all import (  # noqa: E402
    aggregate,
    scan_runs_from_disk,
    write_runs_csv,
    write_summary_csv,
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="由运行日志重建汇总表（不训练）")
    ap.add_argument("--config", default=None, help="配置路径")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    tables = resolve_path(cfg, "tables_dir")
    tables.mkdir(parents=True, exist_ok=True)

    disk = scan_runs_from_disk(cfg)
    if not disk:
        print("[tables] 没有找到 *_run.json，退出")
        return 1

    agg_all: List[Dict[str, Any]] = []
    for exp in sorted({r["exp"] for r in disk}):
        sub = [r for r in disk if r["exp"] == exp]
        write_runs_csv(sub, tables / f"exp_{exp}_runs.csv")
        rows = aggregate(sub)
        write_summary_csv(rows, tables / f"exp_{exp}_summary.csv")
        agg_all.extend(rows)
        print(f"[tables] {exp}: {len(sub)} 次运行 -> {len(rows)} 个变体")

    write_runs_csv(disk, tables / "all_experiments_runs.csv")
    write_summary_csv(agg_all, tables / "all_experiments_summary.csv")
    print(f"[tables] 全部重建完成（共 {len(disk)} 次运行）")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())