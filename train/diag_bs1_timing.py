# -*- coding: utf-8 -*-
"""bs=1 全量（9000 张）单轮耗时实测（报告 §7.5 的 U4 证据）。

只跑 2 个 epoch、不写检查点、不参与任何汇总表；产物
``reports/logs/bs1_epoch_timing.json`` 记录每轮秒数与推算口径。

用法
----
    python train/diag_bs1_timing.py [--epochs 2]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.backend import get_backend  # noqa: E402
from models.charset import resolve_positions  # noqa: E402
from models.config import git_info, load_config  # noqa: E402
from models.model import (backward, build_model, build_onehot,  # noqa: E402
                          compute_loss, forward)
from models.optim import OptimConfig, SGDMomentum  # noqa: E402
from train.train import load_data_bundle  # noqa: E402


def main(argv: Optional[Sequence[str]] = None) -> int:
    """测量 bs=1 全量训练集的单轮墙钟耗时。

    返回
    ----
    int
        0 表示成功。
    """
    ap = argparse.ArgumentParser(description="bs=1 全量单轮耗时实测")
    ap.add_argument("--epochs", type=int, default=2, help="测量轮数（默认 2）")
    args = ap.parse_args(argv)

    cfg = load_config()
    backend = get_backend("auto")
    data = load_data_bundle(cfg)
    head_dims = resolve_positions(cfg.charset.positions)

    n_train = len(data.train)
    params = build_model(int(cfg.model.input_dim), int(cfg.model.hidden_dim),
                         head_dims, arch=str(cfg.model.arch),
                         activation=str(cfg.model.activation),
                         init=str(cfg.model.init), seed=42)
    optim = SGDMomentum(params, OptimConfig(lr=0.05, momentum=0.9))

    print(f"[bs1计时] 训练集 {n_train} 张，bs=1，后端 {backend.name}，"
          f"测 {args.epochs} 轮（只前向+反向+更新，不评估、不写盘）")

    epoch_seconds = []
    for ep in range(1, int(args.epochs) + 1):
        t0 = time.perf_counter()
        n = 0
        for x, y in data.train.iter_batches(batch_size=1, shuffle=True,
                                            augment=True, seed=42 + ep,
                                            drop_last=False):
            probs, cache = forward(params, x, backend, with_cache=True)
            targets = build_onehot(y, head_dims, backend)
            compute_loss(probs, targets, backend, l2_lambda=1e-4, params=params)
            grads = backward(params, cache, targets, backend, l2_lambda=1e-4)
            optim.step(grads)
            n += 1
        dt = time.perf_counter() - t0
        epoch_seconds.append(round(dt, 2))
        print(f"  第 {ep} 轮：{dt:.1f}s（{n} 次更新，{dt / n * 1000:.1f} ms/次）")

    gi = git_info()
    out = {
        "setting": "bs=1, 全量训练集, sigmoid+shared, lr=0.05, μ=0.9, 弱增强",
        "n_train": n_train,
        "epochs_measured": int(args.epochs),
        "epoch_seconds": epoch_seconds,
        "ms_per_update": round(epoch_seconds[-1] / n_train * 1000, 2),
        "backend": backend.name,
        "commit": gi.commit, "dirty": gi.dirty,
        "note": "用于报告 §7.5 的『bs=1 在 9000 张口径每轮耗时』证据；"
                "测量期间 GPU 无其他训练任务",
    }
    out_path = Path("reports/logs/bs1_epoch_timing.json")
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=2)
    print(f"[bs1计时] 已写出 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())