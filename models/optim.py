# -*- coding: utf-8 -*-
"""优化器（§4.3）：**手写 SGD + Momentum**，支持 Nesterov 与无动量对照。

为什么不用框架优化器
--------------------
规格 §8.1 要求不使用 PyTorch/TensorFlow，优化器同样手写；E5 需要对比
"Momentum vs 纯 SGD"，E6 需要对比"有无 L2"，因此优化器必须把这两项作为
显式参数。

更新式（Momentum，§4.3）
------------------------
::

    E_t = λ · W                     （权重衰减项；本工程把 L2 直接写进损失，
                                     故默认 decay=0，避免重复惩罚）
    v_t = μ · v_{t-1} + g_t
    W_t = W_{t-1} - lr · v_t

Nesterov 变体（``nesterov: true``）::

    v_t = μ · v_{t-1} + g_t
    W_t = W_{t-1} - lr · (g_t + μ · v_t)

**偏置不参与权重衰减**，与 :mod:`models.model` 中"L2 不惩罚偏置"保持一致。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# --- 包引导：支持直接运行本文件 ---------------------------------------------
import sys as _sys

if __package__ in (None, ""):
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.model import Params, params_groups


@dataclass
class OptimConfig:
    """优化器超参数（全部来自 ``configs/default.yaml`` 的 ``optim`` 段）。

    属性
    ----
    lr : float
        学习率。
    momentum : float
        动量系数 μ（0 表示纯 SGD，用于 E5 对照）。
    nesterov : bool
        是否使用 Nesterov 动量。
    decay : float
        权重衰减系数（默认 0；L2 已在损失中，避免重复）。
    weight_decay_on_bias : bool
        是否对偏置施加衰减；默认 ``False``（§4.1 只惩罚权重）。
    """

    lr: float = 0.05
    momentum: float = 0.9
    nesterov: bool = False
    decay: float = 0.0
    weight_decay_on_bias: bool = False

    @classmethod
    def from_config(cls, cfg) -> "OptimConfig":
        """从全局配置构造。

        参数
        ----
        cfg : models.config.Config
            全局配置。

        返回
        ----
        OptimConfig
        """
        o = cfg.optim
        return cls(
            lr=float(o.lr),
            momentum=float(o.get("momentum", 0.0)),
            nesterov=bool(o.get("nesterov", False)),
            decay=float(o.get("decay", 0.0)),
            weight_decay_on_bias=bool(o.get("weight_decay_on_bias", False)),
        )

    def as_dict(self) -> Dict[str, object]:
        """转成可写入日志的字典。"""
        return {
            "lr": self.lr, "momentum": self.momentum, "nesterov": self.nesterov,
            "decay": self.decay, "weight_decay_on_bias": self.weight_decay_on_bias,
        }


class SGDMomentum:
    """手写 SGD + Momentum 优化器。

    参数
    ----
    params : Params
        模型参数（原地更新）。
    config : OptimConfig
        超参数。

    说明
    ----
    动量缓存 ``v`` 与参数同名同形状；每步 :meth:`step` 接收一个梯度字典。
    参数按 :func:`models.model.params_groups` 的固定顺序遍历，保证可复现。
    """

    def __init__(self, params: Params, config: OptimConfig) -> None:
        self.params = params
        self.config = config
        self.velocity: Dict[str, np.ndarray] = {
            name: np.zeros_like(arr) for name, arr in params_groups(params)
        }
        self.step_count = 0

    # ------------------------------------------------------------------ 更新
    def step(self, grads: Dict[str, np.ndarray], lr: Optional[float] = None) -> None:
        """执行一次参数更新。

        参数
        参数
        ----
        grads : dict
            ``{参数名: 梯度}``，键须与 :func:`params_groups` 一致。
        lr : float or None
            本次使用的学习率；``None`` 时用配置值（供学习率衰减使用）。

        返回
        ----
        None

        形状
        ----
        与参数同形状的梯度 -> 原地更新参数
        """
        cfg = self.config
        lr = float(cfg.lr if lr is None else lr)
        mu = float(cfg.momentum)

        for name, arr in params_groups(self.params):
            g = grads.get(name)
            if g is None:
                continue
            g = np.asarray(g, dtype=arr.dtype)

            # 权重衰减（默认关闭；L2 已写进损失函数，避免重复惩罚）
            if cfg.decay > 0 and (cfg.weight_decay_on_bias or not name.startswith("b")):
                g = g + cfg.decay * arr

            if mu > 0:
                v = self.velocity[name]
                v *= mu
                v += g
                if cfg.nesterov:
                    arr -= lr * (g + mu * v)
                else:
                    arr -= lr * v
            else:
                arr -= lr * g          # 纯 SGD（μ = 0，E5 对照）

        self.step_count += 1

    # ------------------------------------------------------------ 状态读写
    def state_dict(self) -> Dict[str, object]:
        """导出优化器状态（用于断点续训/复现）。

        返回
        返回
        ----
        dict
            含动量缓存与步数。
        """
        return {
            "velocity": {k: v.copy() for k, v in self.velocity.items()},
            "step_count": self.step_count,
            "config": self.config.as_dict(),
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        """载入优化器状态。

        参数
        ----
        state : dict
            :meth:`state_dict` 的输出。

        返回
        ----
        None
        """
        for k, v in state["velocity"].items():          # type: ignore[union-attr]
            self.velocity[k] = np.array(v)
        self.step_count = int(state["step_count"])      # type: ignore[arg-type]


def lr_at_step(
    base_lr: float,
    step: int,
    total_steps: int,
    schedule: str = "constant",
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.0,
) -> float:
    """学习率调度（供 E8 对照与长训练使用）。

    参数
    ----
    base_lr : float
        基准学习率。
    step : int
        当前全局步数（从 0 开始）。
    total_steps : int
        总步数。
    schedule : str
        ``constant``、``step``（每 1/3 处降为 1/10）、``cosine``。
    warmup_steps : int
        线性预热步数。
    min_lr_ratio : float
        余弦退火的下限比例。

    返回
    ----
    float
        当前学习率。

    形状
    ----
    标量 -> 标量
    """
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)

    if schedule == "constant":
        return base_lr
    if schedule == "step":
        frac = step / max(total_steps, 1)
        if frac < 1.0 / 3.0:
            return base_lr
        if frac < 2.0 / 3.0:
            return base_lr * 0.1
        return base_lr * 0.01
    if schedule == "cosine":
        frac = min(max(step / max(total_steps, 1), 0.0), 1.0)
        cos = 0.5 * (1.0 + np.cos(np.pi * frac))
        return float(base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cos))
    raise ValueError(f"未知学习率调度 {schedule!r}")


if __name__ == "__main__":  # pragma: no cover
    # 自检：用手工构造的二次型损失验证动量更新公式
    from models.model import build_model
    from models.charset import NUM_CLASSES, SEQ_LEN

    print("=== 优化器自检 ===")
    p = build_model(8, 4, [NUM_CLASSES] * SEQ_LEN, seed=0)
    opt = SGDMomentum(p, OptimConfig(lr=0.1, momentum=0.9))
    before = p.W1.copy()
    grads = {name: np.ones_like(arr) for name, arr in params_groups(p)}
    opt.step(grads)
    # 第一步步长应为 lr * g = 0.1（v_1 = g）
    moved = np.abs((before - p.W1) - 0.1).max()
    print(f"  第 1 步 W1 位移与 lr*g 的最大偏差 = {moved:.3e}（应≈0）")
    opt.step(grads)
    # 第二步步长应为 lr * (0.9*1 + 1) = 0.19
    moved2 = np.abs((before - p.W1) - (0.1 + 0.19)).max()
    print(f"  两步累计位移与 lr*(g + 1.9g) 的最大偏差 = {moved2:.3e}（应≈0）")

    # 纯 SGD 对照
    p2 = build_model(8, 4, [NUM_CLASSES] * SEQ_LEN, seed=0)
    opt2 = SGDMomentum(p2, OptimConfig(lr=0.1, momentum=0.0))
    b2 = p2.W1.copy()
    opt2.step(grads)
    opt2.step(grads)
    print(f"  纯 SGD 两步累计位移 = {np.abs(b2 - p2.W1).max():.4f}（应 = 0.2）")

    # 学习率调度
    print("  lr 调度 step:", [round(lr_at_step(0.05, s, 300, "step"), 5)
                              for s in (0, 99, 100, 199, 200)])
    print("  优化器自检通过。")