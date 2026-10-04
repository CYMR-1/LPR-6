# -*- coding: utf-8 -*-
"""E2 机制诊断：训练后的 ReLU 模型发生了不可逆的隐层死亡（§7.2 证据）。

E2（sigmoid vs relu，固定 lr=0.05）中 relu 崩溃到边缘分布水平。
``train/grad_check.py --activation relu`` 已证明反向传播实现正确；
本脚本进一步给出**机制层面**的证据：统计随机初始化 vs 训练后的
relu 模型在训练集上的隐层激活情况。

实测结论（E2_relu_s42）：
- 随机初始化时 0% 单元死亡、z1 均值 ≈ 0.04（健康）；
- 训练后 z1 均值 = **-110.7**，99.5% 的（样本×单元）激活为 0。
  即预激活在前几个 epoch 被爆炸式更新推向极大负值，之后 relu 梯度恒零，
  参数永远无法恢复，模型退化成只剩 b2 偏置学边缘分布。

输出
----
``reports/logs/relu_death_diag.json``：三种模型的激活统计。

用法
----
    python train/diag_relu_death.py [--run E2_relu_s42]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.backend import get_backend  # noqa: E402
from models.charset import resolve_positions  # noqa: E402
from models.config import git_info, load_config, resolve_path  # noqa: E402
from models.dataset import load_cache  # noqa: E402
from models.model import Params, build_model, forward  # noqa: E402


def activation_stats(params: Params, x: np.ndarray, be) -> dict:
    """统计一个模型在给定输入上的隐层激活情况。

    参数
    ----
    params : Params
        模型参数。
    x : np.ndarray
        形状 ``(N, input_dim)`` 的已标准化输入。
    be : BackendInfo
        计算后端。

    返回
    ----
    dict
        ``never_active_frac``：全批从未激活的单元比例；
        ``mean_h`` / ``mean_z1``：隐层输出/预激活均值；
        ``h_zero_frac``：h==0 的（样本×单元）比例。

    形状
    ----
    ``(N, D)`` -> 标量字典
    """
    probs, cache = forward(params, x, be)   # 返回 (probs, ForwardCache)
    z1 = np.asarray(cache.z)                # (N, H)，shared 结构下是单数组
    h = np.asarray(cache.h)                 # (N, H)
    ever_active = (z1.max(axis=0) > 0)      # 任一样本上为正就算活着
    return {
        "never_active_frac": float(1.0 - ever_active.mean()),
        "mean_h": float(h.mean()),
        "mean_z1": float(z1.mean()),
        "h_zero_frac": float((h == 0).mean()),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """诊断入口。

    返回
    ----
    int
        0 表示成功。
    """
    ap = argparse.ArgumentParser(description="E2 机制诊断：ReLU 隐层死亡")
    ap.add_argument("--run", type=str, default="E2_relu_s42",
                    help="训练后的 relu 运行名（读其最佳检查点）")
    ap.add_argument("--n-samples", type=int, default=512,
                    help="统计用的训练集样本数")
    args = ap.parse_args(argv)

    cfg = load_config()
    be = get_backend("numpy")
    head_dims = resolve_positions(cfg.charset.positions)

    p = resolve_path(cfg, "processed_dir")
    images, labels, _ = load_cache(p / "ccpd_128x32.npz")
    with np.load(p / "splits.npz", allow_pickle=False) as d:
        tr = d["train"].astype(np.int64)[: int(args.n_samples)]
        std = json.loads(str(d["standardizer"]))
    x = images[tr].astype(np.float64).reshape(len(tr), -1) / 255.0
    x = (x - std["mean"]) / std["std"]

    out = {"run": args.run, "n_samples": int(args.n_samples)}

    # 1) 随机初始化的 relu 模型（同 seed，对照组）
    fresh = build_model(input_dim=4096, hidden_dim=256, head_dims=head_dims,
                        arch="shared", activation="relu", init="xavier",
                        seed=42, dtype=np.float64)
    out["fresh_relu"] = activation_stats(fresh, x, be)
    print(f"[随机初始化 relu] 死亡单元 {out['fresh_relu']['never_active_frac'] * 100:.1f}%  "
          f"h均值 {out['fresh_relu']['mean_h']:.4f}  "
          f"z1均值 {out['fresh_relu']['mean_z1']:.4f}  "
          f"h==0 比例 {out['fresh_relu']['h_zero_frac'] * 100:.1f}%")

    # 2) 训练后的 E2 relu 检查点
    ckpt = resolve_path(cfg, "models_dir") / f"{args.run}_best.npz"
    if not ckpt.exists():
        print(f"未找到 {ckpt}，请先训练 {args.run}")
        return 1
    trained, extra = Params.load(ckpt)   # ★ classmethod，必须用返回值
    out["trained_relu"] = activation_stats(trained, x, be)
    print(f"[训练后 {args.run}] 死亡单元 {out['trained_relu']['never_active_frac'] * 100:.1f}%  "
          f"h均值 {out['trained_relu']['mean_h']:.4f}  "
          f"z1均值 {out['trained_relu']['mean_z1']:.4f}  "
          f"h==0 比例 {out['trained_relu']['h_zero_frac'] * 100:.1f}%")

    # 3) 对照：训练后的 sigmoid 基线
    ckpt2 = resolve_path(cfg, "models_dir") / "baseline_s42_best.npz"
    if ckpt2.exists():
        base, _ = Params.load(ckpt2)
        out["trained_sigmoid_baseline"] = activation_stats(base, x, be)
        b = out["trained_sigmoid_baseline"]
        print(f"[训练后 baseline(sigmoid)] h均值 {b['mean_h']:.4f}  "
              f"h==0 比例 {b['h_zero_frac'] * 100:.1f}%"
              f"（sigmoid 的 h 恒在 (0,1)，不存在真正的『死亡』）")

    gi = git_info()
    out["commit"] = gi.commit
    out["dirty"] = gi.dirty
    out_path = resolve_path(cfg, "logs_dir") / "relu_death_diag.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=2)
    print(f"诊断结果已写出：{out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())