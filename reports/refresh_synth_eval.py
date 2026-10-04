# -*- coding: utf-8 -*-
"""一次性刷新：合成域切换为 generator_repo 后端后，重测全部历史检查点。

为什么需要
----------
报告表格（build_report_tables.py）读取 ``*_run.json`` 的 ``synth_test``
字段——那是训练结束时按**当时**的合成集测得的。合成域换成外部生成器后
这些字段全部过期。模型权重不变（合成域从不参与训练/调参，§2.5 要求 3），
因此用各运行的**最佳检查点**在新合成集上重测并原地更新即可，无需重训。
重测口径与 ``train/train.py`` 训练末评测完全一致：同一检查点、同一
evaluate_dataset、同一 eval.batch_size / loss / l2 配置。

两种产物
--------
1. 每个有 ``*_run.json`` 且有检查点的运行：原地更新
   ``<run>_run.json`` 的 ``synth_test`` 字段（并记录 ``synth_backend``）
   与 ``<run>_per_position.csv`` 的 ``synth_test`` 列；
2. 每个有 ``*_eval.json`` 的运行：完整重跑
   :func:`evaluate.main.evaluate_run`，刷新
   ``*_eval.json`` / ``*_eval_arrays.npz`` / ``*_eval_per_position.csv``
   （含逐位置混淆矩阵、错误样本数据、CPU 计时，全部按新合成集重测）。

用法
----
    python reports/refresh_synth_eval.py             # 全部刷新
    python reports/refresh_synth_eval.py --only-run-json   # 只刷 run.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate.main import (  # noqa: E402
    evaluate_run,
    load_run_config,
    load_standardizer,
    rebuild_params,
)
from evaluate.model_eval import evaluate_dataset  # noqa: E402
from models.backend import get_backend, set_backend_env  # noqa: E402
from models.charset import SEQ_LEN, is_position_legal  # noqa: E402
from models.config import ensure_dirs, load_config, resolve_path  # noqa: E402
from models.dataset import PlateDataset  # noqa: E402

SYNTH_BACKEND_TAG = "generator_repo@43bac43"


def list_runs(log_dir: Path, ckpt_dir: Path) -> List[str]:
    """所有同时有 run.json 与检查点的运行名。"""
    runs = []
    for f in sorted(log_dir.glob("*_run.json")):
        name = f.name[: -len("_run.json")]
        if (ckpt_dir / f"{name}_best.npz").is_file():
            runs.append(name)
    return runs


def refresh_run_json(cfg, run_name: str) -> Dict[str, float]:
    """重测单个运行的新合成域指标并原地更新 run.json / per_position.csv。

    返回
    ----
    dict
        新的 synth_test 指标（``metrics.as_dict()``）。
    """
    run_cfg = load_run_config(cfg, run_name)
    ckpt_dir = resolve_path(run_cfg, "models_dir")
    log_dir = resolve_path(run_cfg, "logs_dir")

    params = rebuild_params(run_cfg, run_name, ckpt_dir)
    std = load_standardizer(run_cfg)

    split_name = str(run_cfg.get("paths.splits_file", "splits.npz"))
    split_path = resolve_path(run_cfg, "processed_dir") / split_name
    with np.load(split_path, allow_pickle=False) as d:
        s_img = d["synth_images"]
        s_lab = d["synth_labels"]
    ds = PlateDataset(
        images=s_img, labels=s_lab, standardizer=std,
        flatten=True, augment_fn=None, seed=int(run_cfg.split.split_seed),
        name="synth_test",
    )

    backend = get_backend(str(run_cfg.optim.backend), verbose=False)
    set_backend_env(backend)

    head_mask_fn = None
    if int(list(run_cfg.charset.positions)[0]) < 34:
        def head_mask_fn(labels):  # noqa: ANN001
            m0 = np.array([1.0 if is_position_legal(0, int(v)) else 0.0
                           for v in labels[:, 0]], dtype=np.float32)
            return [m0] + [np.ones(len(labels), dtype=np.float32)] * (SEQ_LEN - 1)

    res = evaluate_dataset(
        params, ds, run_cfg, backend=backend,
        batch_size=int(run_cfg.eval.batch_size),
        l2_lambda=float(run_cfg.loss.l2_lambda),
        loss_type=str(run_cfg.loss.type),
        head_mask_fn=head_mask_fn,
        with_confusion=True,
    )
    metrics = res.metrics.as_dict()

    # ---- 更新 run.json ----------------------------------------------------
    run_json = log_dir / f"{run_name}_run.json"
    with open(run_json, "r", encoding="utf-8") as fp:
        payload = json.load(fp)
    payload["synth_test"] = metrics
    payload["synth_backend"] = SYNTH_BACKEND_TAG
    with open(run_json, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)

    # ---- 更新 per_position.csv 的 synth_test 列 ---------------------------
    pp_csv = log_dir / f"{run_name}_per_position.csv"
    if pp_csv.is_file():
        with open(pp_csv, "r", encoding="utf-8-sig", newline="") as fp:
            rows = list(csv.reader(fp))
        header, body = rows[0], rows[1:]
        col = header.index("synth_test")
        for i, row in enumerate(body):
            if i < SEQ_LEN:
                row[col] = round(float(res.metrics.per_position[i]), 6)
        with open(pp_csv, "w", encoding="utf-8-sig", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(header)
            w.writerows(body)

    return metrics


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="刷新全部运行的新合成域指标")
    ap.add_argument("--only-run-json", action="store_true",
                    help="只刷 run.json / per_position.csv，不重跑完整评测")
    args = ap.parse_args(argv)

    cfg = load_config()
    ensure_dirs(cfg)
    log_dir = resolve_path(cfg, "logs_dir")
    ckpt_dir = resolve_path(cfg, "models_dir")

    runs = list_runs(log_dir, ckpt_dir)
    print(f"[refresh] 共 {len(runs)} 个运行待刷新 run.json")

    t0 = time.perf_counter()
    for i, name in enumerate(runs, 1):
        m = refresh_run_json(cfg, name)
        print(f"[refresh] ({i}/{len(runs)}) {name}: "
              f"synth 字符={m['char_acc'] * 100:.2f}% "
              f"整牌={m['plate_acc'] * 100:.2f}%", flush=True)
    print(f"[refresh] run.json 刷新完成，耗时 {time.perf_counter() - t0:.1f}s")

    if args.only_run_json:
        return 0

    # ---- 完整重评有 eval 产物的运行 ---------------------------------------
    eval_runs = [f.name[: -len("_eval.json")]
                 for f in sorted(log_dir.glob("*_eval.json"))]
    print(f"[refresh] 共 {len(eval_runs)} 个运行待完整重评")
    for i, name in enumerate(eval_runs, 1):
        print(f"[refresh] ({i}/{len(eval_runs)}) 完整重评 {name} …", flush=True)
        evaluate_run(cfg, name)
    print("[refresh] 全部完成")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
