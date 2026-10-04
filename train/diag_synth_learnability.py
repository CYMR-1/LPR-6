# -*- coding: utf-8 -*-
"""合成域可学性对照：直接在合成域小样本上训练，验证"合成域本身是可学的"（§8.2）。

为什么需要这个脚本
------------------
报告 §8.2 声称"让模型直接在合成域上训练：合成 300 张在 μ=0.9、lr=0.1/0.05 下
可学到 100% 字符准确率，lr=0.5 发散；真实 300 张 lr=0.1 同样可学"，但该声明
**没有对应的 JSON 产物**。本脚本把该对照变成可复跑的证据：

* 数据：``splits.npz`` 的 ``synth_images``/``synth_labels`` 前 300 张（合成域），
  以及 CCPD 缓存 ``train`` 划分的前 300 张（真实域）；
* 标准化：**只许用 train 拟合的 ``GlobalStandardizer``**（splits.npz 里的
  ``standardizer`` 键，0 维 JSON 字符串数组）——两域用同一标准化器，
  这正是报告 §8.2 第 2 条"合成域标准化后均值 +0.51σ"的实验条件；
* 模型：与基线一致（shared, hidden=256, sigmoid, xavier, seed=42）；
* 训练：numpy 后端、全批量（batch=300）、momentum=0.9、3000 轮、无增强、
  无 L2（纯可学性探针；L2 只抬高损失地板，不影响"能否记住"的判定）；
* 学习率：合成域 {0.5, 0.1, 0.05}，真实域 {0.1}。

每 100 轮记录一次损失（外加第 1 轮），最终记录训练字符/整牌准确率。
结果写入 ``reports/logs/synth_learnability.json``。

注意：旧实验无产物，本脚本**不追求逐位复现报告数字**，重点是结论方向
（合成域在足够大的 lr 下可学到 ~100%）。若结果方向与报告相反，会在 JSON
的 ``note`` 字段如实说明。

用法
----
    python train/diag_synth_learnability.py
    python train/diag_synth_learnability.py --n 300 --epochs 3000
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# --- 包引导：支持直接运行本文件 ---------------------------------------------
import sys as _sys

if __package__ in (None, ""):
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.charset import SEQ_LEN, resolve_positions  # noqa: E402
from models.config import git_info, load_config, resolve_path, set_seed  # noqa: E402
from models.dataset import GlobalStandardizer, load_cache  # noqa: E402
from models.model import (  # noqa: E402
    backward,
    build_model,
    build_onehot,
    compute_loss,
    forward,
    predict,
)
from models.backend import get_backend  # noqa: E402
from models.optim import OptimConfig, SGDMomentum  # noqa: E402


def _json_safe_loss(value: float) -> Any:
    """把损失值转成 JSON 安全形式。

    参数
    ----
    value : float
        损失值；发散实验里可能出现 ``inf`` / ``nan``。

    返回
    ----
    float or str
        有限值原样返回（float）；``inf``/``-inf``/``nan`` 转成对应字符串，
        保证产物是严格合法 JSON。

    形状
    ----
    标量 -> 标量
    """
    v = float(value)
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    return v


def _standardize(images: np.ndarray, std: GlobalStandardizer) -> np.ndarray:
    """把 uint8 图像批标准化并展平（与 ``PlateDataset.get_batch`` 口径一致）。

    参数
    ----
    images : numpy.ndarray
        形状 ``(N, H, W)``，uint8，取值 ``[0, 255]``。
    std : GlobalStandardizer
        **train 拟合**的标准化器。

    返回
    ----
    numpy.ndarray
        形状 ``(N, H*W)``，float32，``(x/255 - mean) / std``。

    形状
    ----
    ``(N, H, W) uint8`` -> ``(N, H*W) float32``
    """
    x = images.astype(np.float32) / 255.0
    x = std.transform(x)
    return x.reshape(x.shape[0], -1).astype(np.float32)


def train_one_run(
    x: np.ndarray,
    y: np.ndarray,
    lr: float,
    epochs: int,
    momentum: float,
    seed: int,
    head_dims: Sequence[int],
    log_every: int = 100,
    verbose: bool = True,
) -> Dict[str, Any]:
    """在同一批数据上以固定超参训练，并记录损失曲线。

    参数
    ----
    x : numpy.ndarray
        输入，形状 ``(N, D)``，float32（已标准化）。
    y : numpy.ndarray
        标签，形状 ``(N, 6)``，int64。
    lr : float
        学习率（恒定，无退火）。
    epochs : int
        训练轮数（全批量，一轮 = 一次参数更新）。
    momentum : float
        动量系数 μ。
    seed : int
        初始化种子（所有对照 run 共用，保证参数初始值一致）。
    head_dims : Sequence[int]
        六个头的类别数。
    log_every : int
        每多少轮记录一次损失（第 1 轮必记）。
    verbose : bool
        是否打印进度。

    返回
    ----
    dict
        ``{"lr", "momentum", "epochs", "initial_loss", "final_loss",
          "final_char_acc", "final_plate_acc", "best_finite_loss",
          "diverged", "loss_increased", "curve"}``；
        ``curve`` 为 ``[{"epoch", "loss", "char_acc"}, ...]``；
        ``diverged`` 表示训练中出现过非有限损失（inf/nan）；
        ``loss_increased`` 表示最终损失高于第 1 轮（报告 §8.2 语境下的
        "发散"——损失不降反升，如 27 -> 51）。

    形状
    ----
    ``(N, D)`` + ``(N, 6)`` -> 训练曲线 + 终局指标
    """
    backend = get_backend("numpy", verbose=False)  # 铁律：训练一律 numpy 后端
    set_seed(seed)
    params = build_model(int(x.shape[1]), 256, list(head_dims), arch="shared",
                         activation="sigmoid", init="xavier", seed=seed)
    optim = SGDMomentum(params, OptimConfig(lr=float(lr), momentum=float(momentum)))
    targets = build_onehot(y, head_dims, backend)

    curve: List[Dict[str, Any]] = []
    best_finite = float("inf")
    diverged = False
    initial_loss = float("nan")
    final_loss = float("nan")

    for ep in range(1, epochs + 1):
        probs, cache = forward(params, x, backend, with_cache=True)
        total, _ = compute_loss(probs, targets, backend, l2_lambda=0.0)
        grads = backward(params, cache, targets, backend, l2_lambda=0.0)
        optim.step(grads)

        final_loss = float(total)
        if ep == 1:
            initial_loss = final_loss
        if math.isfinite(final_loss):
            best_finite = min(best_finite, final_loss)
        else:
            diverged = True

        if ep == 1 or ep % log_every == 0 or ep == epochs:
            preds, _ = predict(probs, backend)
            char_acc = float((preds == y).mean())
            curve.append({
                "epoch": int(ep),
                "loss": _json_safe_loss(final_loss),
                "char_acc": round(char_acc, 6),
            })
            if verbose:
                print(f"    epoch {ep:5d}  loss={final_loss:.4e}  "
                      f"字符准确率={char_acc * 100:6.2f}%", flush=True)

    # 终局指标用最终权重再算一次（与曲线记录节奏解耦）
    probs, _ = forward(params, x, backend, with_cache=False)
    total, _ = compute_loss(probs, targets, backend, l2_lambda=0.0)
    preds, _ = predict(probs, backend)
    final_loss = float(total)
    if not math.isfinite(final_loss):
        diverged = True
    # 报告 §8.2 语境下的"发散"：损失不降反升（如 27 -> 51），未必到 inf
    loss_increased = bool(math.isfinite(initial_loss) and math.isfinite(final_loss)
                          and final_loss > initial_loss)
    return {
        "lr": float(lr),
        "momentum": float(momentum),
        "epochs": int(epochs),
        "initial_loss": _json_safe_loss(initial_loss),
        "final_loss": _json_safe_loss(final_loss),
        "final_char_acc": float((preds == y).mean()),
        "final_plate_acc": float((preds == y).all(axis=1).mean()),
        "best_finite_loss": (_json_safe_loss(best_finite)
                             if math.isfinite(best_finite) else "nan"),
        "diverged": bool(diverged),
        "loss_increased": loss_increased,
        "curve": curve,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """合成域可学性对照 CLI。

    参数
    ----
    argv : Sequence[str] or None
        命令行参数。

    返回
    ----
    int
        0 = 已写出产物。
    """
    ap = argparse.ArgumentParser(description="§8.2 合成域可学性对照")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--n", type=int, default=300, help="每域取样张数")
    ap.add_argument("--epochs", type=int, default=3000, help="训练轮数（全批量）")
    ap.add_argument("--momentum", type=float, default=0.9, help="动量系数 μ")
    ap.add_argument("--seed", type=int, default=42, help="初始化种子")
    ap.add_argument("--out", type=str, default=None,
                    help="输出路径；默认 reports/logs/synth_learnability.json")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    processed = resolve_path(cfg, "processed_dir")
    head_dims = resolve_positions(cfg.charset.positions)

    # ---- 数据：合成域前 n 张 + 真实域 train 划分前 n 张 --------------------
    tag = f"ccpd_{int(cfg.ccpd.input_size[0])}x{int(cfg.ccpd.input_size[1])}"
    images, labels, _ = load_cache(processed / f"{tag}.npz")
    with np.load(processed / "splits.npz", allow_pickle=False) as d:
        tr = d["train"].astype(np.int64)[: args.n]
        std = GlobalStandardizer.from_dict(json.loads(str(d["standardizer"])))
        synth_images = np.asarray(d["synth_images"])[: args.n]
        synth_labels = np.asarray(d["synth_labels"]).astype(np.int64)[: args.n]

    x_synth = _standardize(synth_images, std)                 # (n, 4096) float32
    y_synth = synth_labels                                    # (n, 6) int64
    x_real = _standardize(images[tr], std)                    # (n, 4096) float32
    y_real = labels[tr].astype(np.int64)                      # (n, 6) int64

    print("=" * 74)
    print("§8.2 合成域可学性对照（numpy 后端，全批量，μ=%.2f，%d 轮）"
          % (args.momentum, args.epochs))
    print(f"  合成域 {x_synth.shape}  真实域 {x_real.shape}  "
          f"标准化器 mean={std.mean:.4f} std={std.std:.4f}（train 拟合）")
    print("=" * 74)

    # ---- 对照矩阵：合成域三个 lr + 真实域一个 lr ----------------------------
    plan = [
        ("synth_lr0.5", x_synth, y_synth, 0.5, "合成域"),
        ("synth_lr0.1", x_synth, y_synth, 0.1, "合成域"),
        ("synth_lr0.05", x_synth, y_synth, 0.05, "合成域"),
        ("real_lr0.1", x_real, y_real, 0.1, "真实域"),
    ]
    runs: Dict[str, Any] = {}
    for name, x, y, lr, domain_cn in plan:
        print(f"\n--- {name}（{domain_cn} {len(x)} 张，lr={lr}）---", flush=True)
        runs[name] = train_one_run(
            x, y, lr=lr, epochs=int(args.epochs), momentum=float(args.momentum),
            seed=int(args.seed), head_dims=head_dims, log_every=100, verbose=True,
        )
        runs[name]["domain"] = domain_cn
        runs[name]["n_samples"] = int(len(x))
        print(f"  => 最终损失 {runs[name]['final_loss']}  "
              f"字符准确率 {runs[name]['final_char_acc'] * 100:.2f}%  "
              f"整牌 {runs[name]['final_plate_acc'] * 100:.2f}%  "
              f"发散={runs[name]['diverged']}", flush=True)

    # ---- 结论方向判定（与报告 §8.2 对照，不追求逐位复现） -------------------
    synth_ok = (isinstance(runs["synth_lr0.1"]["final_loss"], float)
                and runs["synth_lr0.1"]["final_char_acc"] >= 0.99)
    note_parts = []
    if synth_ok:
        note_parts.append("合成域在 lr=0.1、μ=0.9 下可学到 ≥99% 字符准确率，"
                          "方向与报告 §8.2 一致（合成域本身可学）。")
    else:
        note_parts.append("合成域在 lr=0.1、μ=0.9 下未达到 99% 字符准确率，"
                          "方向与报告 §8.2 不一致，需人工复核。")
    r05 = runs["synth_lr0.5"]
    if r05["diverged"] or r05["loss_increased"]:
        note_parts.append(
            f"lr=0.5 在合成域发散（损失从 {r05['initial_loss']} 升到 "
            f"{r05['final_loss']}，字符准确率停在 "
            f"{r05['final_char_acc'] * 100:.1f}%），与报告 §8.2 第 2 条方向一致。")
    else:
        note_parts.append("lr=0.5 在合成域损失下降且未发散，与报告 §8.2 第 2 条不一致。")

    gi = git_info()
    report: Dict[str, Any] = {
        "meta": {
            "n_per_domain": int(args.n),
            "epochs": int(args.epochs),
            "momentum": float(args.momentum),
            "batch": "full(全批量)",
            "l2_lambda": 0.0,
            "lr_schedule": "constant(无退火)",
            "augmentation": "none",
            "arch": "shared", "hidden_dim": 256, "activation": "sigmoid",
            "init": "xavier", "seed": int(args.seed),
            "backend": "numpy",
            "standardizer": std.as_dict(),
            "standardizer_source": "splits.npz 的 standardizer 键（train 拟合）",
            "note": "；".join(note_parts),
            "commit": gi.commit, "dirty": gi.dirty,
            "config_fingerprint": cfg.fingerprint(),
        },
        "runs": runs,
    }

    out = (Path(args.out) if args.out
           else resolve_path(cfg, "logs_dir") / "synth_learnability.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fp:
        json.dump(report, fp, ensure_ascii=False, indent=2, allow_nan=False)

    print()
    print("=" * 74)
    print("结论方向：" + note_parts[0])
    print("发散判定：" + note_parts[1])
    print(f"  报告：{out}")
    print("=" * 74)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
