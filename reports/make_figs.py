# -*- coding: utf-8 -*-
"""把一次运行的数值产物画成报告用图（§6.3 / §0.3）。

设计原则（为什么"图"必须由"数"驱动）
------------------------------------
规格 §0.3 明确禁止"只把结论放在截图里"。因此本脚本 **只做呈现**：

* 输入全部来自 :mod:`evaluate.main` 写出的 ``.npz`` / ``.json`` / ``.csv``；
* 任何结论都能由 ``reports/tables/`` 与 ``reports/logs/`` 里的数值文件复核；
* 图片可随时重画（本脚本幂等）。

生成内容
--------
reports/figs/<run>_curves.png            训练曲线（损失/字符准确率/整牌准确率）
reports/figs/<run>_confusion_pos<k>.png  6 个位置的混淆矩阵（测试集）
reports/figs/<run>_errors.png            错误样本可视化（优先高置信错误）

命令行
------
    python reports/make_figs.py --runs baseline_s42
    python reports/make_figs.py --runs baseline_s42 baseline_s43 --no-errors
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

# --- 包引导 ---------------------------------------------------------------
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate.visualize import (  # noqa: E402
    plot_confusion_matrix,
    plot_error_samples,
    plot_training_curves,
    setup_chinese_font,
)
from models.config import load_config, resolve_path  # noqa: E402


def read_history(log_dir: Path, run_name: str) -> Dict[str, List[float]]:
    """从 ``<run>_history.csv`` 读训练曲线。

    参数
    ----
    log_dir : Path
        日志目录。
    run_name : str
        运行短名。

    返回
    ----
    dict
        ``{列名: [值, ...]}``；文件不存在时返回空字典。

    形状
    ----
    CSV -> dict of list
    """
    path = log_dir / f"{run_name}_history.csv"
    if not path.is_file():
        return {}
    with open(path, "r", newline="", encoding="utf-8-sig") as fp:
        rows = list(csv.DictReader(fp))
    history: Dict[str, List[float]] = {}
    for key in rows[0] if rows else []:
        vals: List[float] = []
        for r in rows:
            try:
                vals.append(float(r[key]))
            except (TypeError, ValueError):
                continue
        if vals:
            history[key] = vals
    return history


def make_figs_for_run(
    cfg,
    run_name: str,
    do_curves: bool = True,
    do_confusion: bool = True,
    do_errors: bool = True,
) -> List[str]:
    """为一次运行生成全部图。

    参数
    ----
    cfg : Config
        基线配置。
    run_name : str
        运行短名。
    do_curves, do_confusion, do_errors : bool
        是否生成对应类别。

    返回
    ----
    list of str
        生成的图片路径。

    形状
    ----
    标量 -> list
    """
    setup_chinese_font(str(cfg.viz.chinese_font))
    log_dir = resolve_path(cfg, "logs_dir")
    fig_dir = resolve_path(cfg, "figs_dir")
    fig_dir.mkdir(parents=True, exist_ok=True)
    made: List[str] = []

    # ---- 训练曲线 -------------------------------------------------------
    if do_curves:
        hist = read_history(log_dir, run_name)
        if hist:
            cols = [c for c in ("train_loss", "val_loss", "val_char_acc",
                                "val_plate_acc") if c in hist]
            if cols:
                p = fig_dir / f"{run_name}_curves.png"
                plot_training_curves(hist, p, title=f"{run_name} 训练曲线",
                                     metrics=cols)
                made.append(str(p))
        else:
            print(f"  （{run_name}: 无 history CSV，跳过曲线）")

    # ---- 混淆矩阵 -------------------------------------------------------
    if do_confusion:
        arr_path = log_dir / f"{run_name}_eval_arrays.npz"
        if arr_path.is_file():
            with np.load(arr_path, allow_pickle=False) as npz:
                keys = set(npz.files)
                for k in range(1, 7):
                    key = f"test_confusion_pos{k}"
                    if key not in keys:
                        continue
                    p = fig_dir / f"{run_name}_confusion_pos{k}.png"
                    plot_confusion_matrix(
                        np.asarray(npz[key]), p, position=k - 1,
                        normalize=bool(cfg.viz.confusion_normalize),
                        title=f"{run_name} 位置 {k} 混淆矩阵（测试集，行归一化）")
                    made.append(str(p))
        else:
            print(f"  （{run_name}: 无 eval_arrays.npz，跳过混淆矩阵）")

    # ---- 错误样本 -------------------------------------------------------
    if do_errors:
        err_path = log_dir / f"{run_name}_errors.npz"
        if err_path.is_file():
            with np.load(err_path, allow_pickle=False) as npz:
                imgs = npz["error_images"]
                truth = npz["error_truth"]
                pred = npz["error_pred"]
                conf = (np.asarray(npz["error_conf"])
                        if "error_conf" in npz.files
                        else np.ones(len(imgs), dtype=np.float32))
            if len(imgs):
                p = fig_dir / f"{run_name}_errors.png"
                plot_error_samples(
                    imgs, truth, pred, conf, p,
                    n=min(int(cfg.viz.grid_samples), len(imgs)),
                    cols=int(cfg.viz.grid_cols),
                    title=f"{run_name} 测试集错误样本（真实 → 预测）")
                made.append(str(p))
            else:
                print(f"  （{run_name}: 测试集零错误，无错误样本图）")
        else:
            print(f"  （{run_name}: 无 errors.npz，跳过错误样本）")
    return made


def main(argv: Optional[Sequence[str]] = None) -> int:
    """命令行入口。

    参数
    ----
    argv : Sequence[str] or None
        命令行参数。

    返回
    ----
    int
        0 表示成功。
    """
    ap = argparse.ArgumentParser(description="由数值产物生成报告用图（§6.3）")
    ap.add_argument("--runs", nargs="+", required=True, help="运行短名列表")
    ap.add_argument("--config", default=None, help="配置路径")
    ap.add_argument("--no-curves", action="store_true")
    ap.add_argument("--no-confusion", action="store_true")
    ap.add_argument("--no-errors", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    total: List[str] = []
    for run in args.runs:
        print(f"[figs] {run}")
        made = make_figs_for_run(
            cfg, run,
            do_curves=not args.no_curves,
            do_confusion=not args.no_confusion,
            do_errors=not args.no_errors)
        total.extend(made)
        for m in made:
            print(f"    {m}")
    print(f"[figs] 共生成 {len(total)} 张图 -> {resolve_path(cfg, 'figs_dir')}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())