# -*- coding: utf-8 -*-
"""由实验汇总表重画「各测试集变体对比」柱状图（不重训）。

为什么单独写这个脚本
--------------------
``reports/run_all.py`` 把绘图与训练耦合在一起（训练完顺手画图），
而合成域测试集切换后端后需要**只重画图、不重训**。汇总数值已经由
``reports/build_report_tables.py`` 落在 ``reports/tables/exp_E*_summary.csv``，
本脚本只读这些 CSV 并调用 :func:`evaluate.visualize.plot_variant_comparison`，
产出与 run_all 相同命名的图（``<EXP>_<split>_<metric>.png``）。

用法
----
    python reports/make_split_bar_figs.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate.visualize import plot_variant_comparison, setup_chinese_font  # noqa: E402
from models.config import load_config, resolve_path  # noqa: E402

#: 与 run_all 一致的图组合：(划分, 指标)
FIG_SPECS = (("test", "char_acc"), ("test", "plate_acc"),
             ("hard_test", "char_acc"), ("synth_test", "char_acc"))


def _read_summary(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as fp:
        return list(csv.DictReader(fp))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="由汇总表重画变体对比柱状图")
    ap.add_argument("--config", default=None, help="配置路径")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    setup_chinese_font(str(cfg.viz.chinese_font))

    tables_dir = resolve_path(cfg, "tables_dir")
    fig_dir = resolve_path(cfg, "figs_dir")
    fig_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for csv_path in sorted(tables_dir.glob("exp_E*_summary.csv")):
        rows = _read_summary(csv_path)
        if not rows:
            continue
        exp = rows[0].get("exp", csv_path.stem)
        for split, metric in FIG_SPECS:
            summary: Dict[str, Dict[str, Tuple[float, float]]] = {}
            for r in rows:
                mean = float(r.get(f"{split}_{metric}_mean") or "nan")
                std = float(r.get(f"{split}_{metric}_std") or "0")
                if mean == mean:  # 非 NaN
                    summary[r["variant"]] = {metric: (mean, std)}
            if not summary:
                continue
            tag = f"{exp}_{split}_{metric}"
            out = fig_dir / f"{tag}.png"
            plot_variant_comparison(
                summary, metric=metric, out_path=out,
                title=f"{exp}：{split} 的 {metric}"
                      f"（均值±标准差，{len(summary)} 个变体）",
                ylabel=metric)
            print(f"[figs] {tag}.png（{len(summary)} 个变体）")
            total += 1
    print(f"[figs] 共重画 {total} 张 -> {fig_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())