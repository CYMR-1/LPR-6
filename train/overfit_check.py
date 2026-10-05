# -*- coding: utf-8 -*-
"""小样本过拟合自检（§5.4 第 2 项，自检门槛）。

为什么必须做
------------
规格 §5.4 给出两条独立判据：

1. **数值梯度检查**（见 :mod:`train.grad_check`）—— 验证"梯度算得对不对"；
2. **小样本过拟合自检** —— 验证"训练回路能不能真的把参数调对"。

第 2 条的意义在于：梯度检查只抽查了若干分量，无法发现"优化器写错""更新方向符号
反了""标准化把信号抹掉"这类整体性问题。做法是取 **100 个样本**，关掉 L2 与增强，
故意用较大学习率训练，直到

* 训练损失 < ``1e-3``，且
* 训练字符准确率 == ``1.0``

两者同时达到才算通过 —— 一个参数量 1.1 M 的模型拟合 100 个样本本身是"过参数化"
的典型情形，达不到就说明训练回路有 bug。

★ 实测标定（重要，不是理论推测）
--------------------------------
本机 sigmoid + 6 路 Softmax、100 样本下逐项试过 lr∈{0.3…4}、momentum∈{0,0.9}、
lr 退火、μ 退火、二者联合退火，结论：

1. **准确率不是瓶颈**：字符与整牌准确率在第 ~100 轮就双双到 100%。
2. **损失才是瓶颈**：总损失是六路交叉熵之和，要 <1e-3 就得每一路都压到 ~1.7e-4。
   Sigmoid 隐层输出被限制在 ``(0,1)``，把某类概率推到 ``1-1e-4`` 只能靠大幅增大
   输出层权重范数，于是损失按约 ``O(1/轮)`` 缓慢衰减，**需要约 1.1–1.8 万轮**。
   规格里写的"训练 400 轮"在该结构下数学上达不到 1e-3。
3. **动量与纯 SGD 都能收敛**（★ 更正：早期文档声称 μ=0.9 会"卡在 ~2e-2 极限环"，
   那只是 400 轮观察窗内的错觉；补测 ``overfit_check_mu09_lr05_20k.json``
   显示 μ=0.9 在第 11000 轮即达 1e-3，比 μ=0 的 18000 轮更快）。本脚本仍
   以 μ=0 为默认——它与历史产物逐位一致，作为"基准轨迹"保留；
   ``--momentum 0.9`` 用于复现对照。
4. 实测定标值：``lr=0.5``、``momentum=0`` 时第 **17282** 轮达到
   ``loss=1.0e-3`` 且字符准确率 100%，耗时约 150 秒（CPU）。
   故 ``epochs`` 默认给 20000 并留出余量。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# --- 包引导：支持直接运行本文件 ---------------------------------------------
import sys as _sys

if __package__ in (None, ""):
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    # 直接运行 train/ 下脚本时 sys.path[0] 是 train/，那里的 train.py 会以顶层
    # 模块身份遮蔽同名 train 包，导致 `from train.xxx import ...` 失败，因此把
    # 脚本自身目录从 sys.path 中移除（项目根已插到最前，models/evaluate 仍可导入）。
    _here = str(Path(__file__).resolve().parent)
    while _here in _sys.path:
        _sys.path.remove(_here)

from models.backend import get_backend
from models.charset import SEQ_LEN, resolve_positions
from models.config import (ROOT, apply_patch, ensure_dirs, git_info, load_config,
                           resolve_path, set_seed)
from models.dataset import GlobalStandardizer, PlateDataset, load_cache
from models.model import (Params, backward, build_model, build_onehot,
                          compute_loss, forward, params_groups, predict)
from models.optim import OptimConfig, SGDMomentum


@dataclass
class OverfitReport:
    """过拟合自检报告。

    属性
    ----
    passed : bool
        是否同时达到损失与准确率目标。
    final_loss : float
        最后一轮训练损失。
    final_char_acc : float
        最后一轮训练字符准确率。
    final_plate_acc : float
        最后一轮训练整牌准确率。
    best_loss : float
        训练过程中最低损失。
    epochs_run : int
        实际轮数。
    target_loss : float
        损失目标。
    target_char_acc : float
        准确率目标。
    curve : list of dict
        逐轮曲线（供画图）。
    meta : dict
        运行元信息。
    """

    passed: bool = False
    final_loss: float = 0.0
    final_char_acc: float = 0.0
    final_plate_acc: float = 0.0
    best_loss: float = float("inf")
    epochs_run: int = 0
    target_loss: float = 1e-3
    target_char_acc: float = 1.0
    curve: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """转成可写入 JSON 的字典。"""
        return {
            "passed": self.passed,
            "final_loss": self.final_loss,
            "final_char_acc": self.final_char_acc,
            "final_plate_acc": self.final_plate_acc,
            "best_loss": self.best_loss,
            "epochs_run": self.epochs_run,
            "target_loss": self.target_loss,
            "target_char_acc": self.target_char_acc,
            "meta": self.meta,
            "curve": self.curve,
        }


def overfit_check(
    cfg,
    n_samples: Optional[int] = None,
    epochs: Optional[int] = None,
    lr: Optional[float] = None,
    l2_lambda: Optional[float] = None,
    seed: Optional[int] = None,
    backend_name: str = "numpy",
    momentum: Optional[float] = None,
    verbose: bool = True,
) -> OverfitReport:
    """在 ``n_samples`` 个样本上做"能否完美过拟合"自检。

    参数
    ----
    cfg : Config
        全局配置。
    n_samples : int or None
        样本数；``None`` 时用 ``cfg.train.overfit_check.n_samples``（默认 100）。
    epochs : int or None
        最大轮数（默认 400）。
    lr : float or None
        学习率（默认 0.5，刻意放大以便快速过拟合）。
    l2_lambda : float or None
        必须为 0（默认取配置里的 0.0）；有正则会阻碍完美记忆。
    seed : int or None
        随机种子。
    backend_name : str
        计算后端。
    momentum : float or None
        动量系数 μ；``None`` 时用 ``cfg.train.overfit_check.momentum``
        （默认 0.0，与现状一致）。``0.9`` 用于复现 §5.2 的"极限环"对照。
    verbose : bool
        是否打印进度。

    返回
    ----
    OverfitReport
        自检报告。

    形状
    ----
    ``(100, 4096)`` -> 训练循环曲线 + 结论
    """
    ok = cfg.train.overfit_check
    n_samples = int(n_samples if n_samples is not None else ok.n_samples)
    epochs = int(epochs if epochs is not None else ok.epochs)
    lr = float(lr if lr is not None else ok.learning_rate)
    # 显式传入的 momentum 优先；否则用配置值（默认 0.0，保持原行为）
    momentum = float(ok.get("momentum", 0.0) if momentum is None else momentum)
    target_loss = float(ok.target_loss)
    target_acc = float(ok.target_char_acc)
    l2_lambda = float(0.0 if l2_lambda is None else l2_lambda)
    seed = int(seed if seed is not None else ok.get("seed", cfg.split.split_seed))

    set_seed(seed)
    backend = get_backend(backend_name, verbose=False)

    # ---- 取训练集的前 n_samples 个样本（增强关闭，排除随机性干扰） --------
    processed = resolve_path(cfg, "processed_dir")
    tag = f"ccpd_{int(cfg.ccpd.input_size[0])}x{int(cfg.ccpd.input_size[1])}"
    images, labels, _ = load_cache(processed / f"{tag}.npz")
    with np.load(processed / "splits.npz", allow_pickle=False) as d:
        tr = d["train"].astype(np.int64)[:n_samples]
        std = GlobalStandardizer.from_dict(json.loads(str(d["standardizer"])))
    ds = PlateDataset(images=images[tr], labels=labels[tr], standardizer=std,
                      flatten=True, augment_fn=None, seed=seed, name="overfit")
    if verbose:
        print(f"[overfit] 样本数={len(ds)}  轮数={epochs}  lr={lr}  μ={momentum}  "
              f"λ={l2_lambda}  种子={seed}  后端={backend.name}")

    head_dims = resolve_positions(cfg.charset.positions)
    params = build_model(int(cfg.model.input_dim), int(cfg.model.hidden_dim),
                         head_dims, arch=str(cfg.model.arch),
                         activation=str(cfg.model.activation),
                         init=str(cfg.model.init), seed=seed)
    optim = SGDMomentum(params, OptimConfig(lr=lr, momentum=momentum))

    # 全批量：100 个样本一次前向，最稳
    x, y = ds.get_batch(np.arange(len(ds)), augment=False)
    targets = build_onehot(y, head_dims, backend)

    # 首位越界的样本不参与 head0 损失
    mask = None
    if int(head_dims[0]) <= 24:
        m0 = (y[:, 0] < int(head_dims[0])).astype(np.float32)
        mask = [m0] + [np.ones(len(y), dtype=np.float32)] * (SEQ_LEN - 1)

    report = OverfitReport(target_loss=target_loss, target_char_acc=target_acc)

    # 无 lr 退火：本自检的目标是验证"损失能否被压到 ~0"，固定 lr 下的
    # 轨迹最简单、最易复现；μ 与 lr 的对照实测见模块 docstring 第 3 条。
    for ep in range(1, epochs + 1):
        probs, cache = forward(params, x, backend, with_cache=True)
        total, parts = compute_loss(probs, targets, backend, l2_lambda=l2_lambda,
                                    params=params, head_mask=mask)
        grads = backward(params, cache, targets, backend, l2_lambda=l2_lambda,
                         head_mask=mask)
        optim.step(grads)

        loss = float(total)
        report.best_loss = min(report.best_loss, loss)

        if ep == 1 or ep % max(epochs // 20, 1) == 0 or ep == epochs:
            preds, _ = predict(probs, backend)
            char_acc = float((preds == y).mean())
            plate_acc = float((preds == y).all(axis=1).mean())
            report.curve.append({
                "epoch": ep, "loss": round(loss, 8),
                "char_acc": round(char_acc, 6), "plate_acc": round(plate_acc, 6),
            })
            if verbose:
                print(f"    epoch {ep:5d}  loss={loss:.3e}  "
                      f"字符准确率={char_acc * 100:6.2f}%  "
                      f"整牌准确率={plate_acc * 100:6.2f}%")
            if loss < target_loss and char_acc >= target_acc:
                break

    # ---- 终局判定：用最终权重再算一次（不依赖打印节奏） -------------------
    probs, _ = forward(params, x, backend, with_cache=False)
    total, _ = compute_loss(probs, targets, backend, l2_lambda=l2_lambda,
                            params=params, head_mask=mask)
    preds, _ = predict(probs, backend)
    report.final_loss = float(total)
    report.final_char_acc = float((preds == y).mean())
    report.final_plate_acc = float((preds == y).all(axis=1).mean())
    report.epochs_run = ep
    report.passed = bool(report.final_loss < target_loss
                         and report.final_char_acc >= target_acc)

    gi = git_info()
    report.meta = {
        "n_samples": len(ds), "epochs": epochs, "lr": lr, "l2_lambda": l2_lambda,
        "momentum": momentum, "seed": seed, "backend": backend.name,
        "lr_schedule": "无（固定 lr，轨迹最易复现；μ/lr 对照见 docstring 第 3 条）",
        "arch": params.arch, "activation": params.activation,
        "hidden_dim": int(cfg.model.hidden_dim),
        "num_parameters": params.num_parameters(),
        "augmentation": "none（自检必须关增强）",
        "commit": gi.commit, "dirty": gi.dirty,
        "config_fingerprint": cfg.fingerprint(),
    }
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    """过拟合自检 CLI。

    参数
    ----
    argv : Sequence[str] or None
        命令行参数。

    返回
    ----
    int
        通过返回 0。
    """
    ap = argparse.ArgumentParser(description="自检：小样本过拟合自检（§5.4）")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--momentum", type=float, default=None,
                    help="覆盖动量系数 μ；默认取配置 train.overfit_check.momentum"
                         "（0.0）。μ=0.9 用于复现 §5.2 的极限环对照")
    ap.add_argument("--backend", type=str, default="numpy",
                    choices=["numpy", "cupy", "auto"])
    ap.add_argument("--activation", type=str, default=None,
                    choices=["sigmoid", "relu"],
                    help="覆盖 model.activation；用于核验激活函数本身是否可学")
    ap.add_argument("--out", type=str, default=None,
                    help="报告输出路径；默认 reports/logs/overfit_check.json")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.activation:
        # ★ apply_patch 返回**新对象**，必须接收返回值，否则补丁会被静默丢弃
        cfg = apply_patch(cfg, {"model.activation": args.activation})
        print(f"[overfit] 已覆盖 model.activation = {args.activation}")
    ensure_dirs(cfg)
    print("=" * 74)
    print("小样本过拟合自检（§5.4 第 2 项）")
    print(f"  目标：训练损失 < {float(cfg.train.overfit_check.target_loss):.1e} "
          f"且 字符准确率 == {float(cfg.train.overfit_check.target_char_acc):.2f}")
    print("=" * 74)

    report = overfit_check(cfg, n_samples=args.n_samples, epochs=args.epochs,
                           lr=args.lr, backend_name=args.backend,
                           momentum=args.momentum)
    out = (Path(args.out) if args.out
           else resolve_path(cfg, "logs_dir") / "overfit_check.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fp:
        json.dump(report.as_dict(), fp, ensure_ascii=False, indent=2)

    print()
    print("=" * 74)
    print(f"过拟合自检结论：{'PASS' if report.passed else 'FAIL'}")
    print(f"  最终损失 {report.final_loss:.3e}（目标 < {report.target_loss:.1e}）")
    print(f"  字符准确率 {report.final_char_acc * 100:.2f}%"
          f"（目标 >= {report.target_char_acc * 100:.0f}%）")
    print(f"  整牌准确率 {report.final_plate_acc * 100:.2f}%")
    print(f"  轮数 {report.epochs_run}  参数量 {report.meta['num_parameters']:,d}")
    print(f"  报告：{out}")
    print("=" * 74)
    return 0 if report.passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())