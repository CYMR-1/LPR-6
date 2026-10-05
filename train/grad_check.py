# -*- coding: utf-8 -*-
"""数值梯度检查（§5.4 第 1 项，P3 验收门槛）。

为什么必须做
------------
规格 §5 的排错提示指出：

    如果某一头的回传梯度写错，表现通常不是训练崩掉，而是总损失照样下降、
    但某些位置长期学不会。

因此必须用**中心差分**独立地验证解析梯度，并逐位置统计准确率。本模块对

* 共享隐层权重 ``W1`` 与偏置 ``b1``
* 六个输出头权重 ``W2_i`` 与偏置 ``b2_i``
* （独立模型时）六个独立隐层 ``W1i_i`` / ``b1i_i``

分别抽查若干分量，计算相对误差：

.. math::

    \\text{rel\\_err} = \\frac{|g_{\\text{解析}} - g_{\\text{数值}}|}
    {\\max(|g_{\\text{解析}}| + |g_{\\text{数值}}|,\\ \\epsilon)}

要求 **相对误差 < 1e-5**。

关键细节
--------
* **损失函数必须包含 L2 项**才能检验 ``+ λW`` 是否正确，因此检查时用配置里的 λ
  （默认 1e-4），并在报告里分别给出权重与偏置的误差 —— 偏置**不应**出现 L2 贡献。
* 逐位置的交叉熵也单独检查，用于定位"某一头写错"的情况。
* 检查在 CPU（numpy）上做，避免 GPU 浮点差异干扰。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# --- 包引导：支持直接运行本文件 ---------------------------------------------
import sys as _sys

if __package__ in (None, ""):
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.backend import get_backend
from models.charset import NUM_CLASSES, SEQ_LEN, resolve_positions
from models.config import ROOT, apply_patch, git_info, load_config, resolve_path
from models.model import (
    Params,
    backward,
    build_model,
    build_onehot,
    compute_loss,
    forward,
    params_groups,
)

# =============================================================================
# 1. 相对误差
# =============================================================================


def relative_error(analytic: float, numeric: float, eps: float = 1e-12) -> float:
    """计算相对误差（分母带下限，避免除零放大噪声）。

    参数
    ----
    analytic : float
        解析梯度。
    numeric : float
        数值梯度（中心差分）。
    eps : float
        分母下限。

    返回
    ----
    float
        相对误差，非负。

    形状
    ----
    标量 -> 标量
    """
    num = abs(float(analytic) - float(numeric))
    den = max(abs(float(analytic)) + abs(float(numeric)), eps)
    return num / den


def is_within_tolerance(
    analytic: float,
    numeric: float,
    rel_tol: float,
    abs_tol: float,
) -> bool:
    """判定一个分量是否通过检查（**相对或绝对误差满足其一**）。

    为什么需要绝对误差兜底
    ----------------------
    中心差分的绝对误差下界约为 ``|L| · ε_machine / h``。本项目 ``|L| ≈ 20``、
    ``h = 1e-6``、``ε_machine ≈ 2.2e-16``，故绝对误差下限约 ``4e-9``。
    当某个梯度分量本身接近 0（例如 ``∂L/∂b1i_0 ≈ -1.2e-05``）时，
    相对误差会被这个下限放大到 ``1e-4`` 量级，**这并不代表实现错误**。
    因此采用数值优化中的通用判据：相对误差达标 **或** 绝对误差达标即算通过。

    参数
    ----
    analytic : float
        解析梯度。
    numeric : float
        数值梯度。
    rel_tol : float
        相对误差阈值。
    abs_tol : float
        绝对误差阈值。

    返回
    ----
    bool
        通过返回 ``True``。

    形状
    ----
    标量 -> 标量
    """
    if abs(float(analytic) - float(numeric)) <= abs_tol:
        return True
    return relative_error(analytic, numeric) < rel_tol


# =============================================================================
# 2. 损失函数（供数值差分调用）
# =============================================================================


def loss_at(
    params: Params,
    x: np.ndarray,
    labels: np.ndarray,
    backend,
    l2_lambda: float,
    loss_type: str,
    head_dims: Sequence[int],
    head_mask: Optional[Sequence] = None,
    dtype: np.dtype = np.float64,
) -> float:
    """在给定参数下计算标量损失（数值差分的被求导对象）。

    参数
    ----
    params : Params
        参数（会被临时修改，调用方负责复原）。
    x : numpy.ndarray
        形状 ``(B, D)`` 输入。
    labels : numpy.ndarray
        形状 ``(B, 6)`` 标签索引。
    backend : BackendInfo
        计算后端。
    l2_lambda : float
        L2 强度。
    loss_type : str
        损失类型。
    head_dims : Sequence[int]
        六个头类别数。
    head_mask : Sequence or None
        逐头掩码。
    dtype : numpy.dtype
        one-hot 精度，应与参数一致（梯度检查用 ``float64``）。

    返回
    ----
    float
        标量损失值。

    形状
    ----
    ``(B, D)`` + ``(B, 6)`` -> 标量
    """
    probs, _ = forward(params, x, backend, with_cache=False)
    targets = build_onehot(labels, head_dims, backend, dtype=dtype)
    total, _ = compute_loss(
        probs, targets, backend, l2_lambda=l2_lambda,
        params=params, loss_type=loss_type, head_mask=head_mask,
    )
    return float(total)


# =============================================================================
# 3. 检查结果容器
# =============================================================================


@dataclass
class CheckRecord:
    """单个参数分量的检查记录。

    属性
    ----
    name : str
        参数名（``W1`` / ``b1`` / ``W2_3`` / ``b2_3`` / ``W1i_2`` …）。
    index : tuple of int
        分量下标。
    analytic : float
        解析梯度。
    numeric : float
        数值梯度。
    rel_err : float
        相对误差。
    """

    name: str
    index: Tuple[int, ...]
    analytic: float
    numeric: float
    rel_err: float
    abs_err: float = 0.0

    def as_dict(self) -> Dict[str, object]:
        """转成可写入 JSON 的字典。"""
        return {
            "name": self.name,
            "index": list(self.index),
            "analytic": self.analytic,
            "numeric": self.numeric,
            "rel_err": self.rel_err,
            "abs_err": self.abs_err,
        }


@dataclass
class GradCheckReport:
    """梯度检查总报告。

    属性
    ----
    records : list of CheckRecord
        全部分量的检查记录。
    tol : float
        相对误差判定阈值。
    abs_tol : float
        绝对误差判定阈值（用于接近 0 的梯度分量）。
    per_head_ce : dict
        逐位置交叉熵值（用于确认六头都参与损失）。
    meta : dict
        运行元信息（配置、种子、commit 等）。
    """

    records: List[CheckRecord] = field(default_factory=list)
    tol: float = 1e-5
    abs_tol: float = 1e-8
    per_head_ce: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def max_rel_err(self) -> float:
        """所有分量的最大相对误差。"""
        return max((r.rel_err for r in self.records), default=0.0)

    @property
    def max_abs_err(self) -> float:
        """所有分量的最大绝对误差。"""
        return max((r.abs_err for r in self.records), default=0.0)

    @property
    def n_failed(self) -> int:
        """未通过的分量个数（相对与绝对判据都不满足）。"""
        return sum(
            1 for r in self.records
            if not is_within_tolerance(r.analytic, r.numeric, self.tol, self.abs_tol)
        )

    @property
    def passed(self) -> bool:
        """是否全部通过（相对误差达标 **或** 绝对误差达标）。"""
        return bool(self.records) and self.n_failed == 0

    def by_group(self) -> Dict[str, Dict[str, float]]:
        """按参数组汇总相对/绝对误差。

        返回
        ----
        dict
            ``{参数组: {"max_rel_err":..., "max_abs_err":..., "n":..., "mean_rel_err":...}}``。
        """
        out: Dict[str, Dict[str, float]] = {}
        for r in self.records:
            g = out.setdefault(r.name, {
                "max_rel_err": 0.0, "max_abs_err": 0.0,
                "mean_rel_err": 0.0, "n": 0,
            })
            g["max_rel_err"] = max(g["max_rel_err"], r.rel_err)
            g["max_abs_err"] = max(g["max_abs_err"], r.abs_err)
            g["mean_rel_err"] += r.rel_err
            g["n"] += 1
        for g in out.values():
            if g["n"]:
                g["mean_rel_err"] /= g["n"]
        return out

    def as_dict(self) -> Dict[str, object]:
        """转成可写入 JSON 的完整报告。"""
        return {
            "tol": self.tol,
            "abs_tol": self.abs_tol,
            "passed": self.passed,
            "max_rel_err": self.max_rel_err,
            "max_abs_err": self.max_abs_err,
            "n_failed": self.n_failed,
            "criterion": "相对误差 < tol 或 绝对误差 <= abs_tol"
                         "（后者兜底接近 0 的梯度分量）",
            "by_group": self.by_group(),
            "per_head_ce": self.per_head_ce,
            "n_checked": len(self.records),
            "meta": self.meta,
            "records": [r.as_dict() for r in self.records],
        }


# =============================================================================
# 4. 检查主流程
# =============================================================================


def check_gradients(
    params: Params,
    x: np.ndarray,
    labels: np.ndarray,
    backend,
    l2_lambda: float = 1e-4,
    loss_type: str = "cross_entropy",
    n_per_param: int = 3,
    epsilon: float = 1e-6,
    tol: float = 1e-5,
    abs_tol: float = 1e-8,
    seed: int = 42,
    head_mask: Optional[Sequence] = None,
    dtype: np.dtype = np.float64,
) -> GradCheckReport:
    """对模型做数值梯度检查（中心差分）。

    参数
    ----
    params : Params
        模型参数。
    x : numpy.ndarray
        形状 ``(B, D)`` 输入（建议 B 取 4–16，太大则差分太慢）。
    labels : numpy.ndarray
        形状 ``(B, 6)`` 标签。
    backend : BackendInfo
        计算后端（建议 ``numpy``，避免 GPU 浮点噪声）。
    l2_lambda : float
        L2 强度；**必须 > 0 才能检验 ``+λW`` 项**。
    loss_type : str
        损失类型。
    n_per_param : int
        每个参数张量抽查的分量数。
    epsilon : float
        差分步长 h。
    tol : float
        相对误差阈值。
    abs_tol : float
        绝对误差阈值（兜底接近 0 的梯度分量）。
    seed : int
        抽查下标的随机种子。
    head_mask : Sequence or None
        逐头掩码。
    dtype : numpy.dtype
        计算精度。**必须用 ``float64``**：float32 下总损失约 20 量级，
        ``h=1e-6`` 带来的损失变化（~1e-7）落在 float32 分辨率之内，
        数值梯度会退化成量化噪声，导致检查必然失败。

    返回
    ----
    GradCheckReport
        完整报告。

    形状
    ----
    参数张量 -> 逐分量 (解析, 数值) 对比
    """
    rng = np.random.default_rng(int(seed))
    head_dims = params.head_dims

    # ---- ① 解析梯度 -------------------------------------------------------
    probs, cache = forward(params, x, backend, with_cache=True)
    targets = build_onehot(labels, head_dims, backend, dtype=dtype)
    total, parts = compute_loss(
        probs, targets, backend, l2_lambda=l2_lambda,
        params=params, loss_type=loss_type, head_mask=head_mask,
    )
    grads = backward(params, cache, targets, backend,
                     l2_lambda=l2_lambda, loss_type=loss_type,
                     head_mask=head_mask)

    report = GradCheckReport(tol=float(tol), abs_tol=float(abs_tol))
    report.per_head_ce = {
        f"ce_{i}": float(parts[f"ce_{i}"]) for i in range(SEQ_LEN)
    }
    report.meta["initial_loss"] = float(total)

    # ---- ② 逐参数、逐分量做中心差分 --------------------------------------
    for name, arr in params_groups(params):
        if name not in grads:
            continue
        flat = arr.reshape(-1)
        n = flat.size
        k = min(int(n_per_param), n)
        idx = rng.choice(n, size=k, replace=False)

        for fi in idx:
            fi = int(fi)
            orig = float(flat[fi])
            multi = np.unravel_index(fi, arr.shape)

            flat[fi] = orig + epsilon
            lp = loss_at(params, x, labels, backend, l2_lambda, loss_type,
                         head_dims, head_mask, dtype)
            flat[fi] = orig - epsilon
            lm = loss_at(params, x, labels, backend, l2_lambda, loss_type,
                         head_dims, head_mask, dtype)
            flat[fi] = orig  # 复原

            numeric = (lp - lm) / (2.0 * epsilon)
            analytic = float(grads[name].reshape(-1)[fi])
            report.records.append(CheckRecord(
                name=name, index=tuple(int(v) for v in multi),
                analytic=analytic, numeric=numeric,
                rel_err=relative_error(analytic, numeric),
                abs_err=abs(analytic - numeric),
            ))

    return report


# =============================================================================
# 5. CLI
# =============================================================================


def main(argv: Optional[Sequence[str]] = None) -> int:
    """梯度检查入口。

    参数
    ----
    argv : Sequence[str] or None
        命令行参数。

    返回
    ----
    int
        通过返回 0，失败返回 1。
    """
    ap = argparse.ArgumentParser(description="P3 验收：数值梯度检查（§5.4）")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--n-samples", type=int, default=None, help="差分用的样本数")
    ap.add_argument("--n-checks", type=int, default=None, help="每个参数抽查的分量数")
    ap.add_argument("--epsilon", type=float, default=None)
    ap.add_argument("--tol", type=float, default=None)
    ap.add_argument("--abs-tol", type=float, default=None,
                    help="绝对误差阈值，兜底接近 0 的梯度分量")
    ap.add_argument("--arch", type=str, default=None, choices=["shared", "independent", "both"],
                    help="检查哪种结构")
    ap.add_argument("--activation", type=str, default=None,
                    choices=["sigmoid", "relu"],
                    help="覆盖 model.activation；输出文件名会带上激活名以免覆盖默认结果")
    ap.add_argument("--out-tag", type=str, default=None,
                    help="输出文件名附加标签（如 e9_194），避免覆盖基线结果")
    ap.add_argument("--hidden-dim", type=int, default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.activation:
        # ★ apply_patch 返回新对象，必须接收返回值
        cfg = apply_patch(cfg, {"model.activation": args.activation})
    gc = cfg.grad_check

    n_samples = int(args.n_samples if args.n_samples is not None else gc.n_samples)
    n_checks = int(args.n_checks if args.n_checks is not None else gc.n_checks_per_param)
    tol = float(args.tol if args.tol is not None else gc.relative_error_tol)
    abs_tol = float(args.abs_tol if args.abs_tol is not None
                    else gc.get_path("absolute_error_tol", 1.0e-8))
    epsilon = float(args.epsilon if args.epsilon is not None else gc.epsilon)
    hidden = int(args.hidden_dim if args.hidden_dim is not None else cfg.model.hidden_dim)
    input_dim = int(cfg.model.input_dim)
    loss_type = str(cfg.loss.type)
    l2 = float(cfg.loss.l2_lambda)
    seed = int(gc.seed)

    backend = get_backend("numpy", verbose=True)

    # 用**随机数据**做检查：梯度正确性与数据内容无关，随机数据更能暴露实现错误。
    # ★ 精度必须是 float64：float32 下 h=1e-6 引起的损失变化低于分辨率，
    #   数值梯度会退化为量化噪声，检查必然失败（这是实测踩到的坑）。
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=(n_samples, input_dim)).astype(np.float64)
    labels = rng.integers(0, NUM_CLASSES, size=(n_samples, SEQ_LEN)).astype(np.int64)

    # ★ 逐头掩码始终构造（labels < 该头类别数）：基线 34×6 时全 1、不改变数值，
    #   194 节点结构下则**正是训练时的真实语义**（首位数字样本被掩盖）。
    #   这样 masked 反向路径在每次校验中都被覆盖——此前 mask 只进损失不进
    #   反向时，该结构下校验必然 FAIL（code_audit 缺陷 #1 的复现路径）。
    head_dims_for_mask = resolve_positions(cfg.charset.positions)
    head_mask = [
        (labels[:, i] < int(c)).astype(np.float64)
        for i, c in enumerate(head_dims_for_mask)
    ]

    arches = ["shared", "independent"] if args.arch in (None, "both") else [args.arch]

    print("=" * 74)
    print("P3 数值梯度检查（§5.4 第 1 项）")
    print(f"  样本数 B={n_samples}  输入维 D={input_dim}  隐层 H={hidden}")
    print(f"  损失={loss_type}  λ={l2}  步长 h={epsilon}  精度=float64")
    print(f"  判据：相对误差 < {tol:.1e} 或 绝对误差 <= {abs_tol:.1e}")
    print(f"  每参数抽查分量数={n_checks}  种子={seed}")
    print("=" * 74)

    all_passed = True
    written: List[str] = []

    for arch in arches:
        head_dims = resolve_positions(cfg.charset.positions)
        params = build_model(
            input_dim=input_dim, hidden_dim=hidden, head_dims=head_dims,
            arch=arch, activation=str(cfg.model.activation),
            init=str(cfg.model.init), seed=seed, dtype=np.float64,
        )
        print()
        print(f"--- 结构：{arch} ---")
        print(f"    参数量 {params.num_parameters():,d}  "
              f"头维度 {head_dims}  激活 {params.activation}")
        print(f"    W1{params.W1.shape}  W2_i{params.W2[0].shape}")

        rep = check_gradients(
            params, x, labels, backend,
            l2_lambda=l2, loss_type=loss_type,
            n_per_param=n_checks, epsilon=epsilon, tol=tol, abs_tol=abs_tol,
            seed=seed, head_mask=head_mask,
        )
        gi = git_info()
        rep.meta = {
            "arch": arch, "hidden_dim": hidden, "input_dim": input_dim,
            "n_samples": n_samples, "loss_type": loss_type, "l2_lambda": l2,
            "epsilon": epsilon, "n_checks_per_param": n_checks, "seed": seed,
            "rel_tol": tol, "abs_tol": abs_tol, "dtype": "float64",
            "backend": "numpy",
            "commit": gi.commit, "dirty": gi.dirty,
            "config_fingerprint": cfg.fingerprint(),
        }

        print(f"    逐位置交叉熵: " +
              "  ".join(f"ce{i}={rep.per_head_ce[f'ce_{i}']:.4f}" for i in range(SEQ_LEN)))
        print(f"    抽查分量数 {len(rep.records)}  最大相对误差 {rep.max_rel_err:.3e}  "
              f"最大绝对误差 {rep.max_abs_err:.3e}  未通过 {rep.n_failed} 个  -> "
              f"{'PASS' if rep.passed else 'FAIL'}")
        print("    分组误差：")
        for g, st in sorted(rep.by_group().items()):
            ok = all(
                is_within_tolerance(r.analytic, r.numeric, tol, abs_tol)
                for r in rep.records if r.name == g
            )
            flag = "OK " if ok else "BAD"
            print(f"      [{flag}] {g:8s} n={int(st['n']):3d}  "
                  f"max_rel={st['max_rel_err']:.3e}  max_abs={st['max_abs_err']:.3e}")

        act_suffix = "" if str(cfg.model.activation) == "sigmoid" else f"_{cfg.model.activation}"
        tag_suffix = f"_{args.out_tag}" if args.out_tag else ""
        out = resolve_path(cfg, "logs_dir") / f"grad_check_{arch}{act_suffix}{tag_suffix}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fp:
            json.dump(rep.as_dict(), fp, ensure_ascii=False, indent=2)
        written.append(str(out))
        all_passed = all_passed and rep.passed

    print()
    print("=" * 74)
    print(f"梯度检查结论：{'全部通过' if all_passed else '存在失败项'}")
    print(f"  判据：相对误差 < {tol:.1e} 或 绝对误差 <= {abs_tol:.1e}")
    for w in written:
        print(f"  报告：{w}")
    print("=" * 74)
    return 0 if all_passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())