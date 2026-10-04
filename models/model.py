# -*- coding: utf-8 -*-
"""模型：共享隐层 + 六个位置 Softmax 分类头，以及六个独立 MLP 对照（§3）。

结构（§3.1，主模型）
--------------------
::

    输入 x            : (batch, 4096)
    共享隐层          : W1 (4096, H), b1 (H)
    共享表示 h        : sigmoid(x @ W1 + b1)          (batch, H)
    位置 i 的 logits  : v_i = h @ W2_i + b2_i          (batch, C_i), i = 1..6
    位置 i 的概率     : y_i = softmax(v_i)             (batch, C_i)

对照模型（§3.2）
----------------
六个位置各自拥有独立的隐层 ``W1_i (4096, H)`` 与输出头，位置之间无任何参数共享，
参数量约为共享模型的 6 倍。

**前向与反向全部手写**，不使用任何自动求导（§8.1 路线 A）。反向传播公式严格对应
§5.2：``δ_i = y_i − d_i``、``∂L/∂W2_i = hᵀδ_i + λW2_i``、
``∂L/∂h = Σ_i δ_i W2_iᵀ``、``δ_h = (∂L/∂h) ⊙ h ⊙ (1−h)``、
``∂L/∂W1 = xᵀδ_h + λW1``。**六路梯度必须累加**，这是与普通单头网络最不同的地方。
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

from models.backend import BackendInfo, get_backend
from models.charset import NUM_CLASSES, SEQ_LEN, resolve_positions

# =============================================================================
# 1. 激活函数与 Softmax（前向 + 反向）
# =============================================================================


def sigmoid(z, xp=np):
    """Sigmoid 激活。

    参数
    ----
    z : 后端数组
        形状任意。
    xp : module
        计算后端（numpy 或 cupy）。

    返回
    ----
    后端数组
        与 ``z`` 同形状，取值 ``(0, 1)``。

    形状
    ----
    ``(...)`` -> ``(...)``
    """
    # 数值稳定写法：对正负分别处理，避免 exp 溢出
    out = xp.empty_like(z)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + xp.exp(-z[pos]))
    ez = xp.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out


def sigmoid_grad_from_output(h, xp=np):
    """由 **Sigmoid 的输出** 求其导数：``σ'(z) = σ(z)·(1 − σ(z))``。

    参数
    ----
    h : 后端数组
        Sigmoid 的输出（不是输入 z）。
    xp : module
        计算后端。

    返回
    ----
    后端数组
        与 ``h`` 同形状。

    形状
    ----
    ``(...)`` -> ``(...)``
    """
    return h * (1.0 - h)


def relu(z, xp=np):
    """ReLU 激活（E2 对照用）。

    参数
    ----
    z : 后端数组
        形状任意。
    xp : module
        计算后端。

    返回
    ----
    后端数组
        与 ``z`` 同形状。

    形状
    ----
    ``(...)`` -> ``(...)``
    """
    return xp.maximum(z, 0.0)


def relu_grad_from_output(h, xp=np):
    """由 **ReLU 的输出** 求其导数（``z > 0`` 时为 1，否则 0）。

    参数
    ----
    h : 后端数组
        ReLU 的输出。
    xp : module
        计算后端。

    返回
    ----
    后端数组
        与 ``h`` 同形状。

    形状
    ----
    ``(...)`` -> ``(...)``
    """
    return (h > 0).astype(h.dtype)


def softmax_stable(v, xp=np):
    """数值稳定的 Softmax（**减最大值**技巧，§5.3）。

    参数
    ----
    v : 后端数组
        形状 ``(batch, C)`` 的 logits。
    xp : module
        计算后端。

    返回
    ----
    后端数组
        形状 ``(batch, C)``，每行和为 1。

    形状
    ----
    ``(B, C)`` -> ``(B, C)``
    """
    m = xp.max(v, axis=1, keepdims=True)      # 减最大值，防 exp 溢出
    e = xp.exp(v - m)
    return e / xp.sum(e, axis=1, keepdims=True)


def cross_entropy_from_logits(v, d, xp=np, eps: float = 1e-12):
    """由 logits 直接计算交叉熵（log-sum-exp 形式，避免 ``log(0)``）。

    参数
    ----
    v : 后端数组
        形状 ``(batch, C)`` 的 logits。
    d : 后端数组
        形状 ``(batch, C)`` 的 one-hot 目标。
    xp : module
        计算后端。
    eps : float
        极小值，仅用于兜底（正常情况下 log-sum-exp 不会产生 ``log(0)``）。

    返回
    ----
    后端数组
        形状 ``(batch,)``，每个样本的交叉熵（**未对 batch 求均值**）。

    形状
    ----
    ``(B, C), (B, C)`` -> ``(B,)``
    """
    m = xp.max(v, axis=1, keepdims=True)
    shifted = v - m
    logsumexp = xp.log(xp.sum(xp.exp(shifted), axis=1, keepdims=True) + eps)
    log_prob = shifted - logsumexp
    return -xp.sum(d * log_prob, axis=1)


def mse_from_probs(probs: Sequence, targets: Sequence, xp=np):
    """由六个头的概率计算平方误差之和（E3 对照用）。

    参数
    ----
    probs : Sequence
        长度 6 的列表，每项形状 ``(batch, C_i)``。
    targets : Sequence
        长度 6 的列表，每项形状 ``(batch, C_i)`` one-hot。
    xp : module
        计算后端。

    返回
    ----
    后端数组
        形状 ``(batch,)``，六个位置平方误差之和（**未对 batch 求均值**）。

    形状
    ----
    ``[(B,C_i)] × 6`` -> ``(B,)``
    """
    total = None
    for y, d in zip(probs, targets):
        diff = y - d
        term = xp.sum(diff * diff, axis=1) * 0.5   # 0.5 * ||y - d||² 使梯度为 (y-d)
        total = term if total is None else total + term
    return total


# =============================================================================
# 2. 模型参数容器
# =============================================================================


@dataclass
class Params:
    """模型参数（主机端 numpy 数组）。

    属性
    ----
    W1 : numpy.ndarray
        共享隐层权重，形状 ``(input_dim, H)``。
    b1 : numpy.ndarray
        共享隐层偏置，形状 ``(H,)``。
    W2 : list of numpy.ndarray
        六个输出头权重，第 i 个形状 ``(H, C_i)``。
    b2 : list of numpy.ndarray
        六个输出头偏置，第 i 个形状 ``(C_i,)``。
    arch : str
        结构类型（``shared`` / ``independent``）。
    input_dim : int
        输入维度。
    hidden_dim : int
        隐层维度。
    head_dims : list of int
        六个头的输出维度。
    activation : str
        隐层激活函数名。

    说明
    ----
    ``independent`` 结构下 ``W1`` 形状仍是 ``(input_dim, H)``（第 0 个位置），
    但 ``W2`` 的长度为 6 且每个头各自配一份隐层 —— 具体存储见
    :attr:`W1_list`，由 :meth:`params_groups` 统一暴露给优化器。
    """

    W1: np.ndarray
    b1: np.ndarray
    W2: List[np.ndarray]
    b2: List[np.ndarray]
    arch: str = "shared"
    input_dim: int = 4096
    hidden_dim: int = 256
    head_dims: List[int] = field(default_factory=lambda: [NUM_CLASSES] * SEQ_LEN)
    activation: str = "sigmoid"
    # independent 结构：每个位置的独立隐层（长度 6）。shared 结构下为空列表。
    W1_list: List[np.ndarray] = field(default_factory=list)
    b1_list: List[np.ndarray] = field(default_factory=list)

    # ------------------------------------------------------------- 统计信息
    def num_parameters(self, trainable_only: bool = True) -> int:
        """参数总量。

        参数
        ----
        trainable_only : bool
            ``True``（默认）只统计**参与前向计算**的参数；
            ``False`` 时把 ``independent`` 结构下未使用的 ``W1``/``b1`` 也计入。

        返回
        ----
        int
            标量参数个数。
        """
        if self.arch == "independent":
            total = sum(int(w.size + b.size) for w, b in zip(self.W2, self.b2))
            total += sum(int(w.size + b.size)
                         for w, b in zip(self.W1_list, self.b1_list))
            if not trainable_only:
                total += int(self.W1.size + self.b1.size)
            return total

        total = int(self.W1.size + self.b1.size)
        for w, b in zip(self.W2, self.b2):
            total += int(w.size + b.size)
        return total

    def describe(self) -> Dict[str, object]:
        """结构摘要，写入日志与报告。"""
        return {
            "arch": self.arch,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "head_dims": list(self.head_dims),
            "activation": self.activation,
            "output_nodes": int(sum(self.head_dims)),
            "num_parameters": self.num_parameters(),
            "W1_shape": list(self.W1.shape),
            "W2_shapes": [list(w.shape) for w in self.W2],
            "n_independent_hidden": len(self.W1_list),
        }

    # ------------------------------------------------------------- 序列化
    def save(self, path: Path, extra: Optional[dict] = None) -> None:
        """保存为 ``.npz``（最佳验证模型，§4.3）。

        参数
        ----
        path : Path
            输出路径。
        extra : dict or None
            额外元信息（配置指纹、commit、指标等）。

        返回
        ----
        None

        形状
        ----
        参数张量 -> ``.npz``
        """
        import json

        path.parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "W1": self.W1, "b1": self.b1,
            "arch": np.asarray(self.arch),
            "input_dim": np.asarray(self.input_dim),
            "hidden_dim": np.asarray(self.hidden_dim),
            "head_dims": np.asarray(self.head_dims, dtype=np.int64),
            "activation": np.asarray(self.activation),
            "extra": np.asarray(json.dumps(extra or {}, ensure_ascii=False)),
        }
        for i, (w, b) in enumerate(zip(self.W2, self.b2)):
            blob[f"W2_{i}"] = w
            blob[f"b2_{i}"] = b
        for i, (w, b) in enumerate(zip(self.W1_list, self.b1_list)):
            blob[f"W1i_{i}"] = w
            blob[f"b1i_{i}"] = b
        np.savez_compressed(path, **blob)

    @classmethod
    def load(cls, path: Path) -> Tuple["Params", dict]:
        """从 ``.npz`` 载入。

        参数
        ----
        path : Path
            权重路径。

        返回
        ----
        tuple
            ``(Params, extra 元信息 dict)``。

        形状
        ----
        ``.npz`` -> 参数张量
        """
        import json

        with np.load(path, allow_pickle=False) as d:
            head_dims = [int(v) for v in d["head_dims"]]
            n_heads = len(head_dims)
            W2 = [d[f"W2_{i}"] for i in range(n_heads)]
            b2 = [d[f"b2_{i}"] for i in range(n_heads)]
            W1_list, b1_list = [], []
            i = 0
            while f"W1i_{i}" in d:
                W1_list.append(d[f"W1i_{i}"])
                b1_list.append(d[f"b1i_{i}"])
                i += 1
            params = cls(
                W1=d["W1"], b1=d["b1"], W2=W2, b2=b2,
                arch=str(d["arch"]), input_dim=int(d["input_dim"]),
                hidden_dim=int(d["hidden_dim"]), head_dims=head_dims,
                activation=str(d["activation"]),
                W1_list=W1_list, b1_list=b1_list,
            )
            extra = json.loads(str(d["extra"])) if "extra" in d else {}
        return params, extra


# =============================================================================
# 3. 初始化（§3.3）
# =============================================================================


def init_weights(
    shape: Tuple[int, int],
    kind: str,
    rng: np.random.Generator,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """按指定方式初始化权重矩阵。

    参数
    ----
    shape : tuple of int
        ``(fan_in, fan_out)``。
    kind : str
        ``xavier``（``std = sqrt(2/(fan_in+fan_out))``）、
        ``xavier_in``（``sqrt(1/fan_in)``）、
        ``he``（``sqrt(2/fan_in)``）、
        ``small_normal``（``std = 0.01``）。
    rng : numpy.random.Generator
        随机源。
    dtype : numpy.dtype
        输出精度。训练用 ``float32``；**数值梯度检查必须用 ``float64``**，
        否则损失的变化量会落在 float32 的分辨率之下。

    返回
    ----
    numpy.ndarray
        形状 ``shape``，指定精度。

    形状
    ----
    ``(fan_in, fan_out)`` -> ``(fan_in, fan_out)``
    """
    fan_in, fan_out = int(shape[0]), int(shape[1])
    if kind == "xavier":
        std = np.sqrt(2.0 / (fan_in + fan_out))
    elif kind == "xavier_in":
        std = np.sqrt(1.0 / fan_in)
    elif kind == "he":
        std = np.sqrt(2.0 / fan_in)
    elif kind == "small_normal":
        std = 0.01
    else:
        raise ValueError(f"未知初始化方式 {kind!r}")
    return rng.normal(0.0, std, size=(fan_in, fan_out)).astype(dtype)


def build_model(
    input_dim: int,
    hidden_dim: int,
    head_dims: Sequence[int],
    arch: str = "shared",
    activation: str = "sigmoid",
    init: str = "xavier",
    seed: int = 42,
    dtype: np.dtype = np.float32,
) -> Params:
    """构造模型参数（§3.3：必须固定随机种子）。

    参数
    ----
    input_dim : int
        输入维度（基线 4096）。
    hidden_dim : int
        隐层维度 H（基线 256）。
    head_dims : Sequence[int]
        六个头的输出维度（基线 ``[34]*6``；E9 可 ``[24,34,34,34,34,34]``）。
    arch : str
        ``shared``（主模型）或 ``independent``（E1 对照）。
    activation : str
        ``sigmoid``（基线）或 ``relu``（E2 对照）。
    init : str
        初始化方式，见 :func:`init_weights`。
    seed : int
        随机种子（覆盖所有初始化）。
    dtype : numpy.dtype
        参数精度。训练用 ``float32``；梯度检查用 ``float64``
        （float32 下损失变化量会低于分辨率，导致数值梯度全是量化噪声）。

    返回
    ----
    Params
        初始化好的参数。

    形状
    ----
    标量 -> ``W1 (input_dim, H)``、``W2_i (H, C_i)``
    """
    head_dims = [int(c) for c in head_dims]
    if len(head_dims) != SEQ_LEN:
        raise ValueError(f"head_dims 长度必须为 {SEQ_LEN}，实际 {len(head_dims)}")
    if activation not in ("sigmoid", "relu"):
        raise ValueError(f"不支持的激活函数 {activation!r}")
    if arch not in ("shared", "independent"):
        raise ValueError(f"不支持的结构 {arch!r}")

    dtype = np.dtype(dtype)
    rng = np.random.default_rng(int(seed))

    W1 = init_weights((input_dim, hidden_dim), init, rng, dtype)
    b1 = np.zeros(hidden_dim, dtype=dtype)  # 偏置初始化为 0（§3.3）
    W2 = [init_weights((hidden_dim, c), init, rng, dtype) for c in head_dims]
    b2 = [np.zeros(c, dtype=dtype) for c in head_dims]

    W1_list: List[np.ndarray] = []
    b1_list: List[np.ndarray] = []
    if arch == "independent":
        # 六个位置**各自独立**的隐层。
        # ⚠️ 这里刻意不再让位置 0 复用 W1/b1：
        #    如果 W1_list[0] is W1（同一个数组对象），那么对 W1 做数值差分时
        #    会连带改变位置 0 的隐层，使数值梯度变成"总导数" ∂L/∂W1 + ∂L/∂W1i_0，
        #    而解析梯度只含直连项，梯度检查会出现 ~1e-4 的假失败（实测踩到的坑）。
        #    independent 结构下 W1/b1 不参与前向，只保留以兼容 save/load 的字段形状。
        W1_list = [init_weights((input_dim, hidden_dim), init, rng, dtype)
                   for _ in range(SEQ_LEN)]
        b1_list = [np.zeros(hidden_dim, dtype=dtype) for _ in range(SEQ_LEN)]

    return Params(
        W1=W1, b1=b1, W2=W2, b2=b2,
        arch=arch, input_dim=int(input_dim), hidden_dim=int(hidden_dim),
        head_dims=head_dims, activation=activation,
        W1_list=W1_list, b1_list=b1_list,
    )


def params_groups(params: Params) -> List[Tuple[str, np.ndarray]]:
    """把参数摊平成 ``[(名字, 数组)]``，顺序固定。

    优化器与梯度检查都依赖这个**固定顺序**来对齐梯度。

    **结构差异（重要）**：``independent`` 结构下 ``W1``/``b1`` 不参与前向计算，
    因此不被列为可训练参数；六个位置各自的可训练隐层是 ``W1i_0..W1i_5``。
    这样可避免"同一个数组出现在两个名字下"导致的重复更新与梯度检查假失败。

    参数
    ----
    params : Params
        模型参数。

    返回
    ----
    list of (str, numpy.ndarray)
        shared：``[("W1", ...), ("b1", ...), ("W2_0", ...), ("b2_0", ...), ...]``
        independent：``[("W2_0", ...), ("b2_0", ...), ("W1i_0", ...), ("b1i_0", ...), ...]``

    形状
    ----
    -> 长度等于可训练参数张量个数的列表
    """
    groups: List[Tuple[str, np.ndarray]] = []
    if params.arch == "independent":
        for i in range(len(params.W2)):
            groups.append((f"W2_{i}", params.W2[i]))
            groups.append((f"b2_{i}", params.b2[i]))
        for i, (w, b) in enumerate(zip(params.W1_list, params.b1_list)):
            groups.append((f"W1i_{i}", w))
            groups.append((f"b1i_{i}", b))
        return groups

    groups.append(("W1", params.W1))
    groups.append(("b1", params.b1))
    for i, (w, b) in enumerate(zip(params.W2, params.b2)):
        groups.append((f"W2_{i}", w))
        groups.append((f"b2_{i}", b))
    return groups


def zero_grads_like(params: Params) -> Dict[str, np.ndarray]:
    """生成与参数同形状的零梯度字典。

    参数
    ----
    params : Params
        模型参数。

    返回
    ----
    dict
        ``{参数名: 零梯度数组}``，与 :func:`params_groups` 键一致。

    形状
    ----
    同参数形状。
    """
    return {name: np.zeros_like(arr) for name, arr in params_groups(params)}


# =============================================================================
# 4. 前向传播（§5.1）
# =============================================================================


@dataclass
class ForwardCache:
    """前向传播的中间量，供反向传播复用（§5.1）。

    属性
    ----
    x : 后端数组
        形状 ``(B, D)`` 输入。
    z : 后端数组 or list
        shared：``(B, H)`` 隐层线性输出；independent：长度 6 的列表。
    h : 后端数组 or list
        隐层激活输出，与 ``z`` 对应。
    v : list of 后端数组
        六个位置 logits，各 ``(B, C_i)``。
    y : list of 后端数组
        六个位置 softmax 概率，各 ``(B, C_i)``。
    """

    x: object
    z: object
    h: object
    v: List[object]
    y: List[object]


def forward(params: Params, x, backend: BackendInfo, with_cache: bool = True):
    """前向传播。

    参数
    ----
    params : Params
        模型参数（主机端 numpy）。
    x : 后端数组 or numpy.ndarray
        输入，形状 ``(B, D)``；若为 numpy 且后端是 GPU，会自动搬到设备。
    backend : BackendInfo
        计算后端。
    with_cache : bool
        是否返回中间量（训练需要 ``True``，纯推理可 ``False`` 省内存）。

    返回
    ----
    tuple
        ``(probs, cache)``：``probs`` 是长度 6 的列表，各 ``(B, C_i)``；
        ``with_cache=False`` 时 ``cache`` 为 ``None``。

    形状
    ----
    ``(B, D)`` -> 六个 ``(B, C_i)``
    """
    xp = backend.module
    from models.backend import to_device

    # ⚠️ 不能写 `np.asarray(x) if not backend.is_gpu else x`：
    #    GPU 后端下调用方传进来的 x 可能**已经**是 cupy 数组，
    #    此时 np.asarray 会触发 cupy 的"禁止隐式转 numpy"异常。
    #    正确做法是先判断类型，只在需要时搬运。
    xb = x if type(x).__module__.startswith("cupy") else to_device(np.asarray(x), backend)

    act = sigmoid if params.activation == "sigmoid" else relu

    if params.arch == "shared":
        W1 = to_device(params.W1, backend)
        b1 = to_device(params.b1, backend)
        z = xp.dot(xb, W1) + b1                     # (B, H)
        h = act(z, xp)                              # (B, H)
        v = []
        y = []
        for i in range(SEQ_LEN):
            W2 = to_device(params.W2[i], backend)
            b2 = to_device(params.b2[i], backend)
            vi = xp.dot(h, W2) + b2                 # (B, C_i)
            v.append(vi)
            y.append(softmax_stable(vi, xp))
        cache = ForwardCache(x=xb, z=z, h=h, v=v, y=y) if with_cache else None
        return y, cache

    # ---- independent：六个位置各有一套隐层 --------------------------------
    h_list, z_list, v, y = [], [], [], []
    for i in range(SEQ_LEN):
        W1 = to_device(params.W1_list[i], backend)
        b1 = to_device(params.b1_list[i], backend)
        W2 = to_device(params.W2[i], backend)
        b2 = to_device(params.b2[i], backend)
        zi = xp.dot(xb, W1) + b1
        hi = act(zi, xp)
        vi = xp.dot(hi, W2) + b2
        z_list.append(zi)
        h_list.append(hi)
        v.append(vi)
        y.append(softmax_stable(vi, xp))
    cache = ForwardCache(x=xb, z=z_list, h=h_list, v=v, y=y) if with_cache else None
    return y, cache


# =============================================================================
# 5. 反向传播（§5.2）—— 手写核心
# =============================================================================


def backward_shared(
    params: Params,
    cache: ForwardCache,
    targets: Sequence,
    backend: BackendInfo,
    l2_lambda: float = 0.0,
    loss_type: str = "cross_entropy",
    loss_scale: float = 1.0,
    head_mask: Optional[Sequence] = None,
) -> Dict[str, np.ndarray]:
    """共享模型的**手写反向传播**（§5.2 逐条对应）。

    步骤
    ----
    ① 输出层梯度：``δ_i = ∂L/∂v_i``

       * 交叉熵 + Softmax：``δ_i = (y_i − d_i) / B``
       * 平方误差 + Softmax：``δ_i = (y_i − d_i) ⊙ y_i ⊙ (1 − y_i) / B``
         （因为 ``∂L/∂v = (y−d) ⊙ σ'(v)``，Softmax 的雅可比在此按对角近似，
         这是 §4.2 所批评的"MSE + Softmax 梯度饱和"现象的直接体现，
         也正是 E3 要观察的对象）
       * 若给了 ``head_mask``：``δ_i ← δ_i ⊙ mask_i``（被掩盖样本的梯度置零，
         与 :func:`compute_loss` 中的 ``ce_i * mask`` 严格对应——
         ★ 历史上 mask 只进损失不进反向，导致两者不一致，梯度校验在 E9
         的 194 节点结构下必然 FAIL；修复后两侧一致）

    ② 输出头梯度：``∂L/∂W2_i = hᵀδ_i + λW2_i``，``∂L/∂b2_i = Σ_batch δ_i``
    ③ 回传共享隐层：``∂L/∂h = Σ_i δ_i W2_iᵀ``（**六路累加**）
    ④ 隐层反传：``δ_h = (∂L/∂h) ⊙ h ⊙ (1 − h)``（sigmoid）或 ``⊙ 1[h>0]``（relu）
    ⑤ 共享层梯度：``∂L/∂W1 = xᵀδ_h + λW1``，``∂L/∂b1 = Σ_batch δ_h``

    参数
    ----
    params : Params
        模型参数。
    cache : ForwardCache
        前向中间量（``arch`` 必须为 ``shared``）。
    targets : Sequence
        长度 6 的 one-hot 目标列表，各 ``(B, C_i)``（后端数组）。
    backend : BackendInfo
        计算后端。
    l2_lambda : float
        L2 强度 λ；**只作用于权重，不惩罚偏置**（§4.1）。
    loss_type : str
        ``cross_entropy`` 或 ``mse``。
    loss_scale : float
        额外缩放系数，默认 1（损失已按 batch 平均）。
    head_mask : Sequence or None
        长度 6 的 0/1 掩码列表，各 ``(B,)``；被掩盖样本不参与该头的梯度。

    返回
    ----
    dict
        ``{参数名: 梯度数组}``（主机端 numpy），键与 :func:`params_groups` 一致。

    形状
    ----
    前向缓存 -> 与参数同形状的梯度
    """
    xp = backend.module
    from models.backend import asnumpy, to_device

    B = int(cache.x.shape[0])
    grads: Dict[str, np.ndarray] = {}

    # ---- ① 输出层梯度 -----------------------------------------------------
    deltas: List[object] = []
    for i in range(SEQ_LEN):
        y = cache.y[i]
        d = targets[i]
        diff = y - d
        if loss_type == "cross_entropy":
            delta = diff / B
        elif loss_type == "mse":
            # MSE 经 Softmax 回传：乘以 Jacobian 的对角项 y(1-y)
            delta = (diff * y * (1.0 - y)) / B
        else:
            raise ValueError(f"不支持的损失类型 {loss_type!r}")
        if loss_scale != 1.0:
            delta = delta * float(loss_scale)
        if head_mask is not None:
            m = head_mask[i]
            if not type(m).__module__.startswith("cupy"):
                m = to_device(np.asarray(m), backend)
            delta = delta * m.reshape(-1, 1)
        deltas.append(delta)

    # ---- ② 输出头梯度 + ③ 回传隐层（六路累加） ---------------------------
    h = cache.h
    dL_dh = None
    for i in range(SEQ_LEN):
        W2 = to_device(params.W2[i], backend)
        # ∂L/∂W2_i = hᵀ δ_i + λ W2_i（**偏置不参与 L2**）
        gW2 = xp.dot(h.T, deltas[i])
        if l2_lambda > 0:
            gW2 = gW2 + l2_lambda * W2
        grads[f"W2_{i}"] = asnumpy(gW2)
        grads[f"b2_{i}"] = asnumpy(xp.sum(deltas[i], axis=0))

        # ∂L/∂h 的第 i 路贡献：δ_i W2_iᵀ  —— 六路必须累加
        contrib = xp.dot(deltas[i], W2.T)
        dL_dh = contrib if dL_dh is None else dL_dh + contrib

    # ---- ④ 隐层反传 -------------------------------------------------------
    assert dL_dh is not None
    if params.activation == "sigmoid":
        delta_h = dL_dh * sigmoid_grad_from_output(h, xp)
    else:
        delta_h = dL_dh * relu_grad_from_output(h, xp)

    # ---- ⑤ 共享层梯度 -----------------------------------------------------
    x = cache.x
    gW1 = xp.dot(x.T, delta_h)
    if l2_lambda > 0:
        gW1 = gW1 + l2_lambda * to_device(params.W1, backend)
    grads["W1"] = asnumpy(gW1)
    grads["b1"] = asnumpy(xp.sum(delta_h, axis=0))

    return grads


def backward_independent(
    params: Params,
    cache: ForwardCache,
    targets: Sequence,
    backend: BackendInfo,
    l2_lambda: float = 0.0,
    loss_type: str = "cross_entropy",
    loss_scale: float = 1.0,
    head_mask: Optional[Sequence] = None,
) -> Dict[str, np.ndarray]:
    """六个独立 MLP 的**手写反向传播**（E1 对照，§3.2）。

    与共享模型的唯一区别：**没有六路梯度累加**，每个位置各自从自己的隐层回传。

    参数
    ----
    params : Params
        模型参数（``arch="independent"``）。
    cache : ForwardCache
        前向中间量。
    targets : Sequence
        长度 6 的 one-hot 目标列表。
    backend : BackendInfo
        计算后端。
    l2_lambda : float
        L2 强度 λ（不惩罚偏置）。
    loss_type : str
        损失类型。
    loss_scale : float
        额外缩放系数。
    head_mask : Sequence or None
        长度 6 的 0/1 掩码列表，各 ``(B,)``；与 :func:`backward_shared`
        中的语义一致（掩盖样本的梯度置零）。

    返回
    ----
    dict
        ``{参数名: 梯度数组}``，键与 :func:`params_groups` 一致。

    形状
    ----
    前向缓存 -> 与参数同形状的梯度
    """
    xp = backend.module
    from models.backend import asnumpy, to_device

    B = int(cache.x.shape[0])
    grads: Dict[str, np.ndarray] = {}
    x = cache.x

    for i in range(SEQ_LEN):
        y = cache.y[i]
        d = targets[i]
        diff = y - d
        if loss_type == "cross_entropy":
            delta = diff / B
        else:
            delta = (diff * y * (1.0 - y)) / B
        if loss_scale != 1.0:
            delta = delta * float(loss_scale)
        if head_mask is not None:
            m = head_mask[i]
            if not type(m).__module__.startswith("cupy"):
                m = to_device(np.asarray(m), backend)
            delta = delta * m.reshape(-1, 1)

        W2 = to_device(params.W2[i], backend)
        h = cache.h[i]

        gW2 = xp.dot(h.T, delta)
        if l2_lambda > 0:
            gW2 = gW2 + l2_lambda * W2
        grads[f"W2_{i}"] = asnumpy(gW2)
        grads[f"b2_{i}"] = asnumpy(xp.sum(delta, axis=0))

        # 各自独立回传，没有跨位置累加
        dL_dh = xp.dot(delta, W2.T)
        if params.activation == "sigmoid":
            delta_h = dL_dh * sigmoid_grad_from_output(h, xp)
        else:
            delta_h = dL_dh * relu_grad_from_output(h, xp)

        W1 = to_device(params.W1_list[i], backend)
        gW1 = xp.dot(x.T, delta_h)
        if l2_lambda > 0:
            gW1 = gW1 + l2_lambda * W1
        grads[f"W1i_{i}"] = asnumpy(gW1)
        grads[f"b1i_{i}"] = asnumpy(xp.sum(delta_h, axis=0))

    # 位置 0 的参数已经在 W1i_0 / b1i_0 名下产出；independent 结构下
    # W1/b1 不参与前向，故不再重复写入同名梯度（避免优化器重复更新同一数组）。
    return grads


def backward(
    params: Params,
    cache: ForwardCache,
    targets: Sequence,
    backend: BackendInfo,
    l2_lambda: float = 0.0,
    loss_type: str = "cross_entropy",
    loss_scale: float = 1.0,
    head_mask: Optional[Sequence] = None,
) -> Dict[str, np.ndarray]:
    """按结构分派到对应的手写反向传播。

    参数
    ----
    params, cache, targets, backend, l2_lambda, loss_type, loss_scale, head_mask
        含义见 :func:`backward_shared` / :func:`backward_independent`。

    返回
    ----
    dict
        ``{参数名: 梯度数组}``。

    形状
    ----
    前向缓存 -> 与参数同形状的梯度
    """
    if params.arch == "shared":
        return backward_shared(params, cache, targets, backend,
                               l2_lambda=l2_lambda, loss_type=loss_type,
                               loss_scale=loss_scale, head_mask=head_mask)
    return backward_independent(params, cache, targets, backend,
                                l2_lambda=l2_lambda, loss_type=loss_type,
                                loss_scale=loss_scale, head_mask=head_mask)


# =============================================================================
# 6. 损失与辅助量
# =============================================================================


def build_onehot(
    labels: np.ndarray,
    head_dims: Sequence[int],
    backend: BackendInfo,
    dtype: np.dtype = np.float32,
):
    """把 ``(B, 6)`` 标签转成六个 one-hot 目标矩阵。

    参数
    ----
    labels : numpy.ndarray
        形状 ``(B, 6)``，int64 类别索引。
    head_dims : Sequence[int]
        六个头的类别数。
    backend : BackendInfo
        计算后端。
    dtype : numpy.dtype
        输出精度；梯度检查时用 ``float64`` 与参数精度保持一致。

    返回
    ----
    list
        长度 6 的列表，第 i 项形状 ``(B, C_i)``，后端数组。

    形状
    ----
    ``(B, 6)`` -> 六个 ``(B, C_i)``
    """
    xp = backend.module
    labels = np.asarray(labels, dtype=np.int64)
    B = int(labels.shape[0])
    targets = []
    for i, c in enumerate(head_dims):
        t = np.zeros((B, int(c)), dtype=np.dtype(dtype))
        # 标签索引可能 >= 该头的类别数（E9 首位 24 类时），此时该位置无有效目标：
        # 这里把越界样本的 one-hot 置零，并由调用方通过 mask 排除其损失贡献。
        valid = labels[:, i] < int(c)
        t[np.flatnonzero(valid), labels[valid, i]] = 1.0
        targets.append(xp.asarray(t) if backend.is_gpu else t)
    return targets


def compute_loss(
    probs: Sequence,
    targets: Sequence,
    backend: BackendInfo,
    l2_lambda: float = 0.0,
    params: Optional[Params] = None,
    loss_type: str = "cross_entropy",
    eps: float = 1e-12,
    head_mask: Optional[Sequence] = None,
):
    """计算总损失 ``L = Σ_i 交叉熵_i + (λ/2)‖W‖²``（§4.1）。

    参数
    ----
    probs : Sequence
        长度 6 的列表，各 ``(B, C_i)`` 后验概率。
    targets : Sequence
        长度 6 的列表，各 ``(B, C_i)`` one-hot。
    backend : BackendInfo
        计算后端。
    l2_lambda : float
        L2 强度 λ；**只对权重求和，偏置不参与**（§4.1）。
    params : Params or None
        模型参数；给了才能算 L2 项。
    loss_type : str
        ``cross_entropy`` 或 ``mse``。
    eps : float
        防 ``log(0)`` 的极小值（§5.3）。
    head_mask : Sequence or None
        长度 6 的 0/1 掩码，用于 E9 中排除首位越界样本的损失贡献。

    返回
    ----
    tuple
        ``(total, parts)``：``total`` 为标量后端数组；
        ``parts`` 为 dict，含 ``ce``、``l2`` 与逐位置 ``ce_i``。

    形状
    ----
    六个 ``(B, C_i)`` -> 标量
    """
    xp = backend.module
    B = int(probs[0].shape[0])

    per_pos = []
    for i, (y, d) in enumerate(zip(probs, targets)):
        if loss_type == "cross_entropy":
            ce_i = cross_entropy_from_logits(_log(y, xp, eps), d, xp, eps)
        elif loss_type == "mse":
            diff = y - d
            ce_i = 0.5 * xp.sum(diff * diff, axis=1)
        else:
            raise ValueError(f"不支持的损失类型 {loss_type!r}")
        if head_mask is not None:
            m = head_mask[i]
            # GPU 后端下掩码必须在同一设备上，否则 cupy 与 numpy 相乘会报错
            # （E9 评测链路实测踩坑；训练链路在 Trainer._head_mask 已转换）
            if backend.is_gpu and not type(m).__module__.startswith("cupy"):
                from models.backend import to_device
                m = to_device(np.asarray(m), backend)
            ce_i = ce_i * m
        per_pos.append(ce_i)

    data_loss = per_pos[0]
    for term in per_pos[1:]:
        data_loss = data_loss + term
    data_loss = xp.sum(data_loss) / max(B, 1)

    l2 = xp.zeros((), dtype=xp.float32)
    if l2_lambda > 0 and params is not None:
        from models.backend import to_device

        acc = None
        for name, arr in params_groups(params):
            if name.startswith("b"):     # ★ 偏置不参与 L2 惩罚
                continue
            a = to_device(arr, backend)
            s = xp.sum(a * a)
            acc = s if acc is None else acc + s
        l2 = (l2_lambda / 2.0) * acc

    total = data_loss + l2
    parts = {
        "total": total,
        "data": data_loss,
        "l2": l2,
    }
    for i, t in enumerate(per_pos):
        parts[f"ce_{i}"] = xp.sum(t) / max(B, 1)
    return total, parts


def _log(y, xp, eps: float):
    """安全的对数（``log(y + eps)``，§5.3 防 ``log(0)``）。"""
    return xp.log(y + eps)


def predict(probs: Sequence, backend: BackendInfo) -> Tuple[np.ndarray, np.ndarray]:
    """由六个头的概率得到预测类别与置信度。

    参数
    ----
    probs : Sequence
        长度 6 的列表，各 ``(B, C_i)``。
    backend : BackendInfo
        计算后端。

    返回
    ----
    tuple
        ``(pred (B,6) int64, confidence (B,) float32)``。
        置信度取六位概率的**乘积**（整牌联合置信度），更严格。

    形状
    ----
    六个 ``(B, C_i)`` -> ``(B, 6)`` + ``(B,)``
    """
    from models.backend import asnumpy

    preds = []
    joint = None
    for y in probs:
        yn = asnumpy(y)
        k = np.argmax(yn, axis=1)
        preds.append(k.astype(np.int64))
        p = yn[np.arange(yn.shape[0]), k]
        joint = p if joint is None else joint * p
    return np.stack(preds, axis=1), joint.astype(np.float32)


if __name__ == "__main__":  # pragma: no cover
    # 形状与参数量自检：验证 §3.1 的参数量估算表
    print("=== 参数量估算（§3.1）===")
    for H in (128, 256, 512):
        p = build_model(4096, H, [NUM_CLASSES] * SEQ_LEN, arch="shared", seed=0)
        shared = p.W1.size + p.b1.size
        heads = sum(w.size + b.size for w, b in zip(p.W2, p.b2))
        print(f"  H={H:4d}  共享层 {shared:>10,d}  六头 {heads:>8,d}  "
              f"合计 {p.num_parameters():>10,d}  (~{p.num_parameters() / 1e6:.2f} M)")
    p_ind = build_model(4096, 256, [NUM_CLASSES] * SEQ_LEN, arch="independent", seed=0)
    print(f"  独立模型 H=256 合计 {p_ind.num_parameters():,d} "
          f"(~{p_ind.num_parameters() / 1e6:.2f} M，约为共享的 "
          f"{p_ind.num_parameters() / build_model(4096, 256, [NUM_CLASSES] * SEQ_LEN).num_parameters():.1f} 倍)")
    print()
    print("=== 前向形状自检 ===")
    be = get_backend("numpy", verbose=False)
    p = build_model(4096, 256, [NUM_CLASSES] * SEQ_LEN)
    x = np.random.default_rng(0).normal(size=(4, 4096)).astype(np.float32)
    y, cache = forward(p, x, be)
    print(f"  x{x.shape} -> h{cache.h.shape} -> y_i{y[0].shape} × {len(y)}")
    assert cache.h.shape == (4, 256)
    assert all(t.shape == (4, NUM_CLASSES) for t in y)
    assert abs(float(y[0].sum(axis=1).mean()) - 1.0) < 1e-5
    print("  输出节点总数 =", sum(t.shape[1] for t in y), "（应为 204）")
    print("  自检通过：h 为 (batch, H)，六路 y_i 各为 (batch, 34)")