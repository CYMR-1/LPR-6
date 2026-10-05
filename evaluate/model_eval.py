# -*- coding: utf-8 -*-
"""模型评价的公共实现（训练中验证、最终测试、位置约束共用同一份代码）。

为什么要独立成模块
------------------
规格 §6.1 要求验证集与三个测试集用**完全相同**的评价口径。如果训练循环里写一份、
最终评估脚本里再写一份，很容易出现"数值对不上但没人发现"。因此统一放在这里：
:class:`models.dataset.PlateDataset` + :class:`models.model` + :class:`models.metrics`。

**重要**：本模块默认在 CPU（numpy）上做推理，并单独测量 **CPU 单张推理耗时**
（§附录 B 第 12 条要求：项目需报告 CPU 推理时间）。
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
from models.charset import SEQ_LEN, resolve_positions
from models.metrics import Metrics, compute_metrics, error_samples
from models.model import (
    Params,
    build_onehot,
    compute_loss,
    forward,
    predict,
)


@dataclass
class EvalResult:
    """一次数据集评价的返回值。

    属性
    ----
    metrics : Metrics
        指标容器。
    preds : numpy.ndarray
        形状 ``(N, 6)`` 预测索引。
    labels : numpy.ndarray
        形状 ``(N, 6)`` 真实索引。
    dataset_indices : numpy.ndarray
        形状 ``(N,)``。**相对传入数据集**的下标（``0..N-1``）。若传入的是
        ``PlateDataset.subset(...)`` 得到的子集，调用方需用自己保存的索引数组
        把它映射回原始缓存行（见 :func:`models.dataset.PlateDataset.subset`）。
    probs : list of numpy.ndarray
        长度 6 的概率矩阵，各 ``(N, C_i)``。
    """

    metrics: Metrics
    preds: np.ndarray
    labels: np.ndarray
    dataset_indices: np.ndarray
    probs: List[np.ndarray] = field(default_factory=list)


def evaluate_dataset(
    params: Params,
    dataset,
    cfg,
    backend: Optional[BackendInfo] = None,
    batch_size: int = 256,
    l2_lambda: float = 0.0,
    loss_type: str = "cross_entropy",
    head_mask_fn=None,
    with_confusion: bool = True,
    max_eval_samples: Optional[int] = None,
) -> EvalResult:
    """在给定数据集上做前向推理并计算全部指标。

    参数
    ----
    params : Params
        模型参数。
    dataset : models.dataset.PlateDataset
        数据集（**必须关闭增强**，否则指标不可比）。
    cfg : models.config.Config
        全局配置（用于取 ``head_dims`` / ``eval.batch_size``）。
    backend : BackendInfo or None
        计算后端；``None`` 时强制用 **CPU**（保证报告里的推理耗时可比）。
    batch_size : int
        推理批大小（不影响指标，只影响速度）。
    l2_lambda : float
        L2 强度，仅影响汇报的 loss 数值。
    loss_type : str
        损失类型。
    head_mask_fn : callable or None
        形如 ``fn(labels_batch) -> list of (B,) mask``，用于排除首位越界样本。
    with_confusion : bool
        是否计算混淆矩阵（大测试集上可关掉省内存）。
    max_eval_samples : int or None
        只评估前多少个样本（调试用）。

    返回
    ----
    EvalResult
        指标、预测、标签、原始下标与概率。

    形状
    ----
    ``(N, 32, 128)`` -> 六个 ``(N, C_i)`` 概率 + Metrics
    """
    if backend is None:
        # 评价默认走 CPU：报告要求给出 CPU 推理时间，且指标与设备无关
        backend = get_backend("numpy", verbose=False)

    head_dims = resolve_positions(cfg.charset.positions)
    n = len(dataset) if max_eval_samples is None else min(len(dataset), int(max_eval_samples))

    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_probs: List[List[np.ndarray]] = []
    loss_sum = 0.0
    data_sum = 0.0
    seen = 0

    for start in range(0, n, int(batch_size)):
        stop = min(start + int(batch_size), n)
        # 注意：augment 必须为 False，否则测出来的指标不可比
        x, y = dataset.get_batch(np.arange(start, stop), augment=False)
        probs, _ = forward(params, x, backend, with_cache=False)
        targets = build_onehot(y, head_dims, backend)
        hm = head_mask_fn(y) if head_mask_fn is not None else None
        total, parts = compute_loss(
            probs, targets, backend, l2_lambda=l2_lambda,
            params=params, loss_type=loss_type, head_mask=hm,
        )
        preds, _ = predict(probs, backend)

        from models.backend import asnumpy

        all_preds.append(preds)
        all_labels.append(np.asarray(y, dtype=np.int64))
        all_probs.append([asnumpy(p) for p in probs])
        b = stop - start
        loss_sum += float(total) * b
        data_sum += float(parts["data"]) * b
        seen += b

    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    probs = [np.concatenate([p[i] for p in all_probs], axis=0) for i in range(SEQ_LEN)]

    metrics = compute_metrics(
        preds, labels, probs=probs if with_confusion else None,
        loss=loss_sum / max(seen, 1), data_loss=data_sum / max(seen, 1),
        head_dims=head_dims,
    )
    return EvalResult(metrics=metrics, preds=preds, labels=labels,
                      dataset_indices=np.arange(n, dtype=np.int64), probs=probs)


def measure_cpu_inference_time(
    params: Params,
    dataset,
    n_samples: int = 200,
    warmup: int = 5,
    repeats: int = 3,
) -> Dict[str, float]:
    """测量 **CPU 单张推理耗时**（§附录 B 第 12 条）。

    参数
    ----
    params : Params
        模型参数。
    dataset : models.dataset.PlateDataset
        数据集。
    n_samples : int
        计时用的样本数（取前 N 个）。
    warmup : int
        预热次数（排除首次内存分配开销）。
    repeats : int
        重复轮数，取中位数。

    返回
    ----
    dict
        ``{"cpu_inference_ms_per_image": ..., "cpu_inference_images_per_sec": ...,
        "n_samples": ..., "batch_size": ...}``。

    形状
    ----
    ``(1, 32, 128)`` -> 毫秒/张
    """
    import time

    be = get_backend("numpy", verbose=False)
    n = min(int(n_samples), len(dataset))
    x, _ = dataset.get_batch(np.arange(n), augment=False)

    # 预热
    for _ in range(int(warmup)):
        forward(params, x[:1], be, with_cache=False)

    times: List[float] = []
    for _ in range(int(repeats)):
        t0 = time.perf_counter()
        for i in range(n):
            forward(params, x[i:i + 1], be, with_cache=False)   # 单张推理
        times.append(time.perf_counter() - t0)

    med = float(np.median(times))
    return {
        "cpu_inference_ms_per_image": round(med / n * 1000.0, 4),
        "cpu_inference_images_per_sec": round(n / med, 2),
        "n_samples": n,
        "repeats": int(repeats),
        "batch_size": 1,
        "note": "CPU(numpy) 单张前向推理，含标准化与六头 softmax",
    }


def measure_batched_inference_time(
    params: Params,
    dataset,
    batch_size: int = 256,
    n_samples: int = 2000,
) -> Dict[str, float]:
    """测量 **CPU 批量推理吞吐**（用于报告里对比单张延迟）。

    参数
    ----
    params : Params
        模型参数。
    dataset : models.dataset.PlateDataset
        数据集。
    batch_size : int
        批大小。
    n_samples : int
        计时样本数。

    返回
    ----
    dict
        ``{"cpu_batched_ms_per_image": ..., "cpu_batched_images_per_sec": ...}``。

    形状
    ----
    ``(B, 32, 128)`` -> 毫秒/张
    """
    import time

    be = get_backend("numpy", verbose=False)
    n = min(int(n_samples), len(dataset))
    x, _ = dataset.get_batch(np.arange(n), augment=False)

    forward(params, x[:batch_size], be, with_cache=False)   # 预热
    t0 = time.perf_counter()
    for start in range(0, n, int(batch_size)):
        forward(params, x[start:start + int(batch_size)], be, with_cache=False)
    dt = time.perf_counter() - t0
    return {
        "cpu_batched_ms_per_image": round(dt / n * 1000.0, 4),
        "cpu_batched_images_per_sec": round(n / dt, 2),
        "batch_size": int(batch_size),
        "n_samples": n,
    }


def collect_error_samples(result: EvalResult, max_n: int = 60) -> List[Dict[str, object]]:
    """从评价结果里提取错误样本清单。

    参数
    ----
    result : EvalResult
        :func:`evaluate_dataset` 的返回。
    max_n : int
        最多返回条数。

    返回
    ----
    list of dict
        见 :func:`models.metrics.error_samples`。

    形状
    ----
    EvalResult -> list
    """
    return error_samples(
        result.preds, result.labels, result.probs,
        indices=result.dataset_indices, max_n=max_n,
    )


if __name__ == "__main__":  # pragma: no cover
    # 自检：用随机权重在真实缓存上跑一遍评价链路，确认形状与指标口径
    from models.config import load_config, resolve_path
    from models.dataset import GlobalStandardizer, PlateDataset, load_cache
    from models.model import build_model

    cfg = load_config()
    tag = f"ccpd_{int(cfg.ccpd.input_size[0])}x{int(cfg.ccpd.input_size[1])}"
    cache = resolve_path(cfg, "processed_dir") / f"{tag}.npz"
    images, labels, _ = load_cache(cache)
    std = GlobalStandardizer.fit(images[:512])      # 自检用：只在子集上拟合
    ds = PlateDataset(images=images[:64], labels=labels[:64], standardizer=std,
                      flatten=True, augment_fn=None, name="selftest")

    print("=== 评价链路自检 ===")
    print(f"  数据集 {len(ds)} 张，图像 {ds.images.shape}")
    p = build_model(int(cfg.model.input_dim), int(cfg.model.hidden_dim),
                    resolve_positions(cfg.charset.positions), seed=0)
    res = evaluate_dataset(p, ds, cfg, batch_size=32)
    for line in res.metrics.summary_lines():
        print(" ", line)
    assert res.preds.shape == (64, SEQ_LEN)
    assert len(res.probs) == SEQ_LEN
    t = measure_cpu_inference_time(p, ds, n_samples=16, warmup=2, repeats=2)
    print(f"  CPU 单张推理：{t['cpu_inference_ms_per_image']:.3f} ms/张 "
          f"({t['cpu_inference_images_per_sec']:.1f} 张/秒)")
    tb = measure_batched_inference_time(p, ds, batch_size=32, n_samples=64)
    print(f"  CPU 批量推理：{tb['cpu_batched_ms_per_image']:.3f} ms/张 "
          f"(batch={tb['batch_size']})")
    errs = collect_error_samples(res, max_n=3)
    if errs:
        print(f"  错误样本示例：{errs[0]['true_label']} -> {errs[0]['pred_label']}")
    print("  评价链路自检通过（随机权重，准确率接近随机是正常的）。")