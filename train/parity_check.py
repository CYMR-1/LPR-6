# -*- coding: utf-8 -*-
"""后端一致性校验：NumPy（CPU）与 CuPy（GPU）前向/反向输出的数值差异（§5.3）。

为什么需要这个脚本
------------------
报告 §5.3 声称"同一权重分别用 CuPy 与 NumPy 前向/反向，前向输出最大绝对差
5.2e-08、反向梯度最大绝对差 2.2e-08、预测索引完全一致"，但该声明**没有对应的
JSON 产物**。本脚本把这项对比变成可复跑的证据：

1. 用同一随机种子构造 ``shared`` 与 ``independent`` 两组参数
   （隐层 256、8 样本随机数据）；
2. 分别用 numpy 与 cupy 后端执行 :func:`models.model.forward` 与
   :func:`models.model.backward`；
3. 逐数组比较六个头的输出概率与全部梯度（max/mean 绝对差），并核对预测索引
   是否完全一致；
4. **对 float64 与 float32 两种精度各做一遍**（float32 是训练实际用的精度；
   报告 §5.3 的旧数字 5.2e-08/2.2e-08 与 float32 机器精度量级吻合）；
5. 结果写入 ``reports/logs/backend_parity.json``，按 ``{"float64": {...},
   "float32": {...}}`` 组织，每种精度含逐数组差异明细与汇总最大值。

若 CuPy 因故不可用（未安装 / CUDA 不可用 / NVRTC 编译失败），脚本**如实记录
回退原因并跳过对比**，这不算失败 —— 产物仍然落盘，``cupy_available=false``。

用法
----
    python train/parity_check.py
    python train/parity_check.py --batch 8 --seed 42 --out reports/logs/backend_parity.json
"""

from __future__ import annotations

import argparse
import json
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

from models.backend import BackendInfo, asnumpy, get_backend  # noqa: E402
from models.charset import NUM_CLASSES, SEQ_LEN  # noqa: E402
from models.config import git_info, load_config, resolve_path  # noqa: E402
from models.model import (  # noqa: E402
    Params,
    backward,
    build_model,
    build_onehot,
    compute_loss,
    forward,
    params_groups,
    predict,
)


def _diff_entry(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    """计算两个同形状数组的绝对差统计。

    参数
    ----
    a : numpy.ndarray
        后端 A（numpy）产出的数组，任意形状。
    b : numpy.ndarray
        后端 B（cupy）产出的数组，形状须与 ``a`` 相同。

    返回
    ----
    dict
        ``{"shape", "max_abs_diff", "mean_abs_diff", "max_abs_value"}``；
        ``max_abs_value`` 是两数组绝对值的最大值，用于判断差异的量级背景。

    形状
    ----
    ``(...)`` + ``(...)`` -> dict（标量统计）
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"形状不一致：{a.shape} vs {b.shape}")
    d = np.abs(a - b)
    return {
        "shape": [int(s) for s in a.shape],
        "max_abs_diff": float(d.max()) if d.size else 0.0,
        "mean_abs_diff": float(d.mean()) if d.size else 0.0,
        "max_abs_value": float(max(np.abs(a).max(), np.abs(b).max())) if d.size else 0.0,
    }


def _forward_backward(
    params: Params,
    x: np.ndarray,
    labels: np.ndarray,
    backend: BackendInfo,
    l2_lambda: float,
    dtype: np.dtype = np.float64,
) -> Dict[str, Any]:
    """在指定后端上执行一次前向 + 反向，收集全部可比较的量。

    参数
    ----
    params : Params
        模型参数（主机端 numpy 数组；``forward`` 内部会搬到设备）。
    x : numpy.ndarray
        输入，形状 ``(B, D)``，精度与 ``params`` 一致。
    labels : numpy.ndarray
        标签，形状 ``(B, 6)``，int64。
    backend : BackendInfo
        计算后端（numpy 或 cupy）。
    l2_lambda : float
        L2 强度；取配置值以同时覆盖梯度的 ``+λW`` 项。
    dtype : numpy.dtype
        one-hot 目标的精度，须与参数精度一致。

    返回
    ----
    dict
        ``{"probs": [六个 (B, C_i) numpy 数组], "loss": float,
          "grads": {参数名: numpy 数组}, "preds": (B, 6) numpy 数组}``；
        全部已转换回主机端 numpy，便于跨后端比较。

    形状
    ----
    ``(B, D)`` + ``(B, 6)`` -> 六个 ``(B, C_i)`` 概率 + 与参数同形状的梯度
    """
    probs, cache = forward(params, x, backend, with_cache=True)
    targets = build_onehot(labels, params.head_dims, backend, dtype=dtype)
    total, _ = compute_loss(probs, targets, backend, l2_lambda=l2_lambda,
                            params=params)
    grads = backward(params, cache, targets, backend, l2_lambda=l2_lambda)
    preds, _ = predict(probs, backend)
    return {
        "probs": [asnumpy(p).astype(np.float64) for p in probs],
        "loss": float(asnumpy(total)),
        "grads": {name: np.asarray(g, dtype=np.float64) for name, g in grads.items()},
        "preds": np.asarray(preds, dtype=np.int64),
    }


def parity_check(
    backend_np: BackendInfo,
    backend_cp: BackendInfo,
    hidden_dim: int = 256,
    batch: int = 8,
    seed: int = 42,
    l2_lambda: float = 1e-4,
    input_dim: int = 4096,
    dtype: np.dtype = np.float64,
) -> Dict[str, Any]:
    """对 ``shared`` 与 ``independent`` 两种结构执行 numpy/cupy 一致性对比。

    参数
    ----
    backend_np : BackendInfo
        NumPy（CPU）后端。
    backend_cp : BackendInfo
        CuPy（GPU）后端（必须真实可用，调用方负责确认）。
    hidden_dim : int
        隐层维度 H。
    batch : int
        随机小批量的样本数 B。
    seed : int
        初始化与随机数据的种子（两后端共用同一批参数与数据）。
    l2_lambda : float
        L2 强度。
    input_dim : int
        输入维度 D。
    dtype : numpy.dtype
        参数与数据的精度。``float64`` 用于探测后端实现的理论差异；
        ``float32`` 对应训练实际精度（报告 §5.3 旧数字的量级语境）。

    返回
    ----
    dict
        ``{"shared": {...}, "independent": {...}}``；每个结构含
        ``forward``（逐头概率 + 损失的差异明细）、``backward``（逐参数梯度
        差异明细）、``predictions_identical`` 与该结构的汇总最大值。

    形状
    ----
    ``(8, 4096)`` 随机数据 -> 逐数组差异统计
    """
    head_dims = [NUM_CLASSES] * SEQ_LEN
    results: Dict[str, Any] = {}

    for arch in ("shared", "independent"):
        # 同一种子构造参数与随机数据（两后端完全共用，精度由 dtype 决定）
        params = build_model(input_dim, hidden_dim, head_dims, arch=arch,
                             activation="sigmoid", init="xavier", seed=seed,
                             dtype=dtype)
        rng = np.random.default_rng(seed + (0 if arch == "shared" else 1))
        x = rng.normal(0.0, 1.0, size=(batch, input_dim)).astype(dtype)
        labels = rng.integers(0, NUM_CLASSES, size=(batch, SEQ_LEN)).astype(np.int64)

        out_np = _forward_backward(params, x, labels, backend_np, l2_lambda, dtype)
        out_cp = _forward_backward(params, x, labels, backend_cp, l2_lambda, dtype)

        # ---- 前向：逐头概率 + 损失标量 -----------------------------------
        fwd: Dict[str, Any] = {}
        for i in range(SEQ_LEN):
            fwd[f"prob_head_{i}"] = _diff_entry(out_np["probs"][i], out_cp["probs"][i])
        fwd["loss"] = {
            "shape": [],
            "max_abs_diff": abs(out_np["loss"] - out_cp["loss"]),
            "mean_abs_diff": abs(out_np["loss"] - out_cp["loss"]),
            "max_abs_value": max(abs(out_np["loss"]), abs(out_cp["loss"])),
        }
        fwd_max = max(v["max_abs_diff"] for v in fwd.values())

        # ---- 反向：逐参数梯度（键与 params_groups 一致） ------------------
        bwd: Dict[str, Any] = {}
        grad_names = [name for name, _ in params_groups(params)]
        for name in grad_names:
            g_np = out_np["grads"][name]
            g_cp = out_cp["grads"][name]
            bwd[name] = _diff_entry(g_np, g_cp)
        bwd_max = max(v["max_abs_diff"] for v in bwd.values())

        preds_identical = bool(np.array_equal(out_np["preds"], out_cp["preds"]))
        results[arch] = {
            "forward": fwd,
            "backward": bwd,
            "forward_max_abs_diff": float(fwd_max),
            "backward_max_abs_diff": float(bwd_max),
            "predictions_identical": preds_identical,
            "loss_numpy": out_np["loss"],
            "loss_cupy": out_cp["loss"],
        }
        print(f"  [{np.dtype(dtype).name:8s}][{arch:11s}] "
              f"前向 max|Δ|={fwd_max:.3e}  反向 max|Δ|={bwd_max:.3e}  "
              f"预测一致={preds_identical}")

    return results


def _summarize(per_arch: Dict[str, Any]) -> Dict[str, Any]:
    """把逐结构结果汇总成该精度的总结论。

    参数
    ----
    per_arch : dict
        :func:`parity_check` 的输出，``{"shared": {...}, "independent": {...}}``。

    返回
    ----
    dict
        ``{"forward_max_abs_diff", "backward_max_abs_diff",
          "predictions_identical"}``——取两结构中的最大值/全真判定。

    形状
    ----
    dict -> dict（标量汇总）
    """
    return {
        "forward_max_abs_diff": float(max(
            per_arch[a]["forward_max_abs_diff"] for a in per_arch)),
        "backward_max_abs_diff": float(max(
            per_arch[a]["backward_max_abs_diff"] for a in per_arch)),
        "predictions_identical": bool(all(
            per_arch[a]["predictions_identical"] for a in per_arch)),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """后端一致性校验 CLI。

    参数
    ----
    argv : Sequence[str] or None
        命令行参数。

    返回
    ----
    int
        0 = 已写出产物（cupy 不可用时如实记录并跳过，不算失败）。
    """
    ap = argparse.ArgumentParser(description="§5.3 NumPy/CuPy 前后端一致性校验")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--batch", type=int, default=8, help="随机小批量样本数")
    ap.add_argument("--seed", type=int, default=42, help="初始化与数据种子")
    ap.add_argument("--hidden-dim", type=int, default=256, help="隐层维度")
    ap.add_argument("--out", type=str, default=None,
                    help="输出路径；默认 reports/logs/backend_parity.json")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    l2 = float(cfg.loss.l2_lambda)
    input_dim = int(cfg.model.input_dim)

    print("=" * 74)
    print("§5.3 后端一致性校验（NumPy vs CuPy，float64 + float32 两种精度）")
    print(f"  结构：shared + independent  隐层 H={args.hidden_dim}  "
          f"批量 B={args.batch}  种子={args.seed}  λ={l2}")
    print("=" * 74)

    gi = git_info()
    backend_np = get_backend("numpy", verbose=False)
    backend_cp = get_backend("cupy", verbose=False)

    report: Dict[str, Any] = {
        "meta": {
            "seed": int(args.seed),
            "batch": int(args.batch),
            "hidden_dim": int(args.hidden_dim),
            "input_dim": int(input_dim),
            "dtypes": ["float64", "float32"],
            "l2_lambda": l2,
            "arches": ["shared", "independent"],
            "device_name": backend_cp.device_name if backend_cp.is_gpu else "CPU",
            "commit": gi.commit,
            "dirty": gi.dirty,
            "config_fingerprint": cfg.fingerprint(),
        },
        "cupy_available": bool(backend_cp.is_gpu),
        "cupy_fallback_reason": backend_cp.fallback_reason,
    }

    if not backend_cp.is_gpu:
        # 铁律：cupy 不可用不算失败，如实记录并跳过对比
        report["skipped"] = True
        report["note"] = ("CuPy 不可用，已跳过后端对比（不算失败）："
                          + (backend_cp.fallback_reason or "未知原因"))
        print(f"  [跳过] CuPy 不可用：{backend_cp.fallback_reason}")
    else:
        report["skipped"] = False
        # 两种精度各跑一遍：float64 探测理论差异，float32 对应训练实际精度
        for dt in (np.float64, np.float32):
            dt_name = np.dtype(dt).name
            per_arch = parity_check(backend_np, backend_cp,
                                    hidden_dim=int(args.hidden_dim),
                                    batch=int(args.batch), seed=int(args.seed),
                                    l2_lambda=l2, input_dim=input_dim, dtype=dt)
            report[dt_name] = {
                "per_arch": per_arch,
                "summary": _summarize(per_arch),
            }

    out = (Path(args.out) if args.out
           else resolve_path(cfg, "logs_dir") / "backend_parity.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fp:
        json.dump(report, fp, ensure_ascii=False, indent=2)

    print()
    print("=" * 74)
    if report.get("skipped"):
        print(f"结论：CuPy 不可用，对比已跳过（详见 JSON note）")
    else:
        for dt_name in ("float64", "float32"):
            s = report[dt_name]["summary"]
            print(f"结论[{dt_name}]：前向输出最大绝对差 {s['forward_max_abs_diff']:.3e}  "
                  f"反向梯度最大绝对差 {s['backward_max_abs_diff']:.3e}  "
                  f"预测索引{'完全一致' if s['predictions_identical'] else '不一致！'}")
    print(f"  报告：{out}")
    print("=" * 74)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
