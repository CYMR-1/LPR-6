# -*- coding: utf-8 -*-
"""评价指标（§6.1）：**字符准确率**与**整牌准确率**必须同时给出。

指标定义
--------
字符准确率（character accuracy）
    六个位置分别统计，正确字符数 / 总字符数。
    报告里同时给出**逐位置准确率**（§6.1 要求），因为 MLP 的典型故障模式是
    "某一位长期学不会"，只看总体字符准确率会掩盖它。

整牌准确率（full-plate accuracy）
    六个位置**全部**正确的样本比例。若各位置独立且准确率为 ``p``，则整牌准确率
    约 ``p⁶`` —— 所以 95% 的字符准确率只能得到约 73.5% 的整牌准确率。
    报告必须同时给出两者，且说明这一关系。

其他
----
* ``loss``：六个位置交叉熵之和（+ L2）；
* ``confusion``：逐位置混淆矩阵；
* ``per_position``：逐位置正确率、样本数、最常混淆的类别对（便于定位问题）。
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

from models.charset import NUM_CLASSES, SEQ_LEN, decode_batch, index_to_char


# =============================================================================
# 1. 指标容器
# =============================================================================


@dataclass
class Metrics:
    """一次评价的完整指标。

    属性
    ----
    char_acc : float
        字符准确率（全体位置汇总）。
    plate_acc : float
        整牌准确率（六位全对）。
    per_position : list of float
        长度 6，逐位置字符准确率。
    per_position_n : list of int
        长度 6，逐位置参与统计的样本数。
    loss : float
        平均总损失（六个位置交叉熵之和 + L2）。
    data_loss : float
        平均数据损失（不含 L2）。
    n_samples : int
        样本数。
    confusion : list of numpy.ndarray or None
        长度 6，逐位置混淆矩阵 ``(C_i, C_i)``。
    top_confusions : list of list
        长度 6，逐位置最常混淆的 ``(真实字符, 预测字符, 次数)``。
    """

    char_acc: float = 0.0
    plate_acc: float = 0.0
    per_position: List[float] = field(default_factory=lambda: [0.0] * SEQ_LEN)
    per_position_n: List[int] = field(default_factory=lambda: [0] * SEQ_LEN)
    loss: float = 0.0
    data_loss: float = 0.0
    n_samples: int = 0
    confusion: Optional[List[np.ndarray]] = None
    top_confusions: Optional[List[List[Tuple[str, str, int]]]] = None

    def as_dict(self, digits: int = 6) -> Dict[str, object]:
        """转成可写入 CSV/JSON 的扁平字典。

        参数
        ----
        digits : int
            浮点保留位数。

        返回
        ----
        dict
            键名与报告表格列对应，形如 ``char_acc``、``per_position_3``。
        """
        out: Dict[str, object] = {
            "n_samples": int(self.n_samples),
            "char_acc": round(float(self.char_acc), digits),
            "plate_acc": round(float(self.plate_acc), digits),
            "loss": round(float(self.loss), digits),
            "data_loss": round(float(self.data_loss), digits),
        }
        for i, v in enumerate(self.per_position):
            out[f"per_position_{i}"] = round(float(v), digits)
        out["char_acc_mean_pos"] = round(
            float(np.mean(self.per_position)) if self.per_position else 0.0, digits
        )
        out["char_acc_min_pos"] = round(
            float(np.min(self.per_position)) if self.per_position else 0.0, digits
        )
        out["char_acc_max_pos"] = round(
            float(np.max(self.per_position)) if self.per_position else 0.0, digits
        )
        return out

    def summary_lines(self) -> List[str]:
        """生成适合打印/写报告的多行摘要。

        返回
        返回
        ----
        list of str
        """
        lines = [
            f"样本数 {self.n_samples}  损失 {self.loss:.4f}（数据项 {self.data_loss:.4f}）",
            f"字符准确率 {self.char_acc * 100:.2f}%   整牌准确率 {self.plate_acc * 100:.2f}%",
            "  逐位置准确率：" + "  ".join(
                f"位置{i + 1}={v * 100:.2f}%" for i, v in enumerate(self.per_position)
            ),
        ]
        if self.top_confusions:
            for i, lst in enumerate(self.top_confusions):
                if lst:
                    txt = "、".join(f"{a}→{b}({c})" for a, b, c in lst[:3])
                    lines.append(f"  位置{i + 1} 主要混淆：{txt}")
        return lines


# =============================================================================
# 2. 计算
# =============================================================================


def compute_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    probs: Optional[Sequence] = None,
    loss: float = 0.0,
    data_loss: float = 0.0,
    head_dims: Optional[Sequence[int]] = None,
    top_k_confusions: int = 5,
) -> Metrics:
    """计算字符/整牌/逐位置准确率与混淆矩阵。

    参数
    ----
    preds : numpy.ndarray
        形状 ``(N, 6)`` 预测类别索引。
    labels : numpy.ndarray
        形状 ``(N, 6)`` 真实类别索引。
    probs : Sequence or None
        长度 6 的预测概率列表，各 ``(N, C_i)``；给了才算混淆矩阵。
    loss : float
        平均总损失。
    data_loss : float
        平均数据损失。
    head_dims : Sequence[int] or None
        六个头类别数；``None`` 时从 ``probs`` 推断，否则用 ``[34]*6``。
    top_k_confusions : int
        逐位置回报的主要混淆对数。

    返回
    ----
    Metrics
        指标容器。

    形状
    ----
    ``(N, 6)`` + ``(N, 6)`` -> Metrics
    """
    preds = np.asarray(preds, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    if preds.shape != labels.shape:
        raise ValueError(f"预测 {preds.shape} 与标签 {labels.shape} 形状不一致")

    N = int(labels.shape[0])
    correct = preds == labels                                # (N, 6) bool
    per_pos_hits = correct.sum(axis=0)                       # (6,)
    per_position = (per_pos_hits / max(N, 1)).astype(float).tolist()
    char_acc = float(correct.mean()) if N else 0.0
    plate_acc = float(correct.all(axis=1).mean()) if N else 0.0

    m = Metrics(
        char_acc=char_acc,
        plate_acc=plate_acc,
        per_position=per_position,
        per_position_n=[N] * SEQ_LEN,
        loss=float(loss),
        data_loss=float(data_loss),
        n_samples=N,
    )

    if probs is not None:
        if head_dims is None:
            head_dims = [int(np.asarray(p).shape[1]) for p in probs]
        confusion: List[np.ndarray] = []
        tops: List[List[Tuple[str, str, int]]] = []
        for i in range(SEQ_LEN):
            c = int(head_dims[i])
            cm = np.zeros((c, c), dtype=np.int64)
            valid = labels[:, i] < c
            np.add.at(cm, (labels[valid, i], preds[valid, i]), 1)
            confusion.append(cm)
            # 主要混淆对（排除对角线）
            off = cm.copy()
            np.fill_diagonal(off, 0)
            flat = np.argsort(off.reshape(-1))[::-1][:top_k_confusions]
            lst: List[Tuple[str, str, int]] = []
            for fi in flat:
                fi = int(fi)
                n = int(off.reshape(-1)[fi])
                if n <= 0:
                    break
                t, p = divmod(fi, c)
                lst.append((index_to_char(t), index_to_char(p), n))
            tops.append(lst)
        m.confusion = confusion
        m.top_confusions = tops

    return m


def expected_plate_acc_from_char(
    char_acc: float, per_position: Optional[Sequence[float]] = None
) -> float:
    """由字符准确率估算整牌准确率（用于报告里解释两者的差距）。

    参数
    ----
    char_acc : float
        总体字符准确率；当 ``per_position`` 为 ``None`` 时按"六位同分布"估算
        ``char_acc ** 6``。
    per_position : Sequence[float] or None
        逐位置准确率；给了则按乘积估算，更贴近实际。

    返回
    ----
    float
        估算的整牌准确率。

    形状
    ----
    标量 -> 标量
    """
    if per_position is not None and len(per_position) > 0:
        out = 1.0
        for p in per_position:
            out *= float(p)
        return out
    return float(char_acc) ** SEQ_LEN


def error_samples(
    preds: np.ndarray,
    labels: np.ndarray,
    probs: Sequence,
    indices: Optional[np.ndarray] = None,
    max_n: int = 200,
) -> List[Dict[str, object]]:
    """挑出错误样本，供可视化与报告分析（§6.2 错误样本可视化）。

    参数
    ----
    preds : numpy.ndarray
        形状 ``(N, 6)`` 预测索引。
    labels : numpy.ndarray
        形状 ``(N, 6)`` 真实索引。
    probs : Sequence
        长度 6 的概率列表，各 ``(N, C_i)``。
    indices : numpy.ndarray or None
        预测在原始数据集中的下标；``None`` 时用 ``0..N-1``。
    max_n : int
        最多返回多少条。

    返回
    ----
    list of dict
        每条含 ``dataset_index``、``true_label``、``pred_label``、
        ``wrong_positions``、``per_position_conf``、``n_wrong``，
        并按错误位置数降序（错得最多的排前面）。

    形状
    ----
    ``(N, 6)`` -> list
    """
    preds = np.asarray(preds, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    if indices is None:
        indices = np.arange(labels.shape[0])

    wrong = preds != labels
    bad_rows = np.flatnonzero(wrong.any(axis=1))
    if bad_rows.size == 0:
        return []

    recs: List[Dict[str, object]] = []
    for r in bad_rows:
        r = int(r)
        wp = [int(i) for i in np.flatnonzero(wrong[r])]
        conf = []
        for i in range(SEQ_LEN):
            p = np.asarray(probs[i])[r]
            conf.append(round(float(p[preds[r, i]]), 4))
        recs.append({
            "dataset_index": int(indices[r]),
            "n_wrong": len(wp),
            "wrong_positions": wp,
            "true_label": decode_batch(labels[r][None, :])[0],
            "pred_label": decode_batch(preds[r][None, :])[0],
            "true_classes": labels[r].tolist(),
            "pred_classes": preds[r].tolist(),
            "per_position_conf": conf,
        })
    # 错误位多的排前面；同分时按逐位置信度低的排前面
    recs.sort(key=lambda d: (-int(d["n_wrong"]), float(np.mean(d["per_position_conf"]))))
    return recs[:max_n]


def confusion_pairs_table(
    confusion: Sequence[np.ndarray],
    head_dims: Sequence[int],
    top_k: int = 10,
) -> List[Dict[str, object]]:
    """把逐位置混淆矩阵转成"真实→预测"计数的长表（供写 CSV 与报告）。

    参数
    ----
    confusion : Sequence
        长度 6 的混淆矩阵列表。
    head_dims : Sequence[int]
        六个头类别数。
    top_k : int
        每个位置最多输出多少条。

    返回
    ----
    list of dict
        每行含 ``position``、``true_char``、``pred_char``、``count``。

    形状
    ----
    六个 ``(C_i, C_i)`` -> list
    """
    rows: List[Dict[str, object]] = []
    for i, cm in enumerate(confusion):
        c = int(head_dims[i])
        off = np.asarray(cm).copy()
        np.fill_diagonal(off, 0)
        flat = np.argsort(off.reshape(-1))[::-1][:top_k]
        for fi in flat:
            fi = int(fi)
            n = int(off.reshape(-1)[fi])
            if n <= 0:
                break
            t, p = divmod(fi, c)
            rows.append({
                "position": i, "position_1based": i + 1,
                "true_char": index_to_char(t), "pred_char": index_to_char(p),
                "count": n,
            })
    return rows


if __name__ == "__main__":  # pragma: no cover
    # 自检：构造已知答案，验证指标算法
    rng = np.random.default_rng(0)
    N = 1000
    labels = rng.integers(0, NUM_CLASSES, size=(N, SEQ_LEN)).astype(np.int64)
    preds = labels.copy()
    # 人为把位置 2（下标 2）在 20% 的样本上改错
    k = int(N * 0.2)
    rows = rng.choice(N, size=k, replace=False)
    preds[rows, 2] = (preds[rows, 2] + 1) % NUM_CLASSES

    probs = []
    for i in range(SEQ_LEN):
        p = np.full((N, NUM_CLASSES), 0.01, dtype=np.float32)
        p[np.arange(N), preds[:, i]] = 0.5
        p /= p.sum(axis=1, keepdims=True)
        probs.append(p)

    m = compute_metrics(preds, labels, probs=probs, loss=1.23, data_loss=1.20)
    print("=== 指标自检 ===")
    for line in m.summary_lines():
        print(" ", line)
    exp = expected_plate_acc_from_char(m.char_acc, m.per_position)
    print(f"  由逐位置准确率估算的整牌准确率 = {exp * 100:.2f}%"
          f"（实际 {m.plate_acc * 100:.2f}%，应非常接近）")
    assert abs(m.per_position[2] - 0.8) < 0.03, m.per_position
    assert abs(exp - m.plate_acc) < 0.01, (exp, m.plate_acc)
    errs = error_samples(preds, labels, probs)
    print(f"  错误样本条数 {len(errs)}，最差示例：{errs[0]['true_label']} -> "
          f"{errs[0]['pred_label']}（错 {errs[0]['n_wrong']} 位）")
    print("  指标自检通过（位置 3 准确率应≈80%，其余≈100%）。")