# -*- coding: utf-8 -*-
"""训练主程序（§4 训练细节 / §5.4 P4 验收）。

实现要点
--------
======================  ==================================================
规格条目                 实现位置
======================  ==================================================
六路交叉熵 + L2 损失      :func:`models.model.compute_loss`
手写反向传播（六路累加）  :func:`models.model.backward`
SGD + Momentum           :class:`models.optim.SGDMomentum`
早停（监控 val_loss）     :meth:`Trainer._should_stop`
save_best_only           :meth:`Trainer._save_best`
每轮记录指标并写 CSV/JSON  :class:`TrainHistory`
可复现（commit/配置/种子） :func:`run_training` 里的 ``meta`` 组装
CPU / GPU 后端切换        :func:`models.backend.get_backend`
======================  ==================================================

**数据泄漏防护**：训练只读 ``splits.npz`` 里 ``train`` 下标；标准化统计量来自
``splits.npz`` 中"仅在训练集上拟合"的 ``standardizer``，本脚本不会重新拟合。
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# --- 包引导：支持直接运行本文件 ---------------------------------------------
import sys as _sys

if __package__ in (None, ""):
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate.model_eval import evaluate_dataset
from models.augment import AugmentConfig, build_augment_fn
from models.backend import (BackendInfo, get_backend, memory_info, set_backend_env)
from models.charset import LETTER_MAX_INDEX, SEQ_LEN, resolve_positions
from models.config import (Config, ROOT, ensure_dirs, git_info, load_config,
                           resolve_path, set_seed)
from models.dataset import GlobalStandardizer, PlateDataset, load_cache
from models.metrics import expected_plate_acc_from_char
from models.model import (Params, backward, build_model, build_onehot,
                          compute_loss, forward, params_groups)
from models.optim import OptimConfig, SGDMomentum

# =============================================================================
# 1. 训练超参数与记录
# =============================================================================


@dataclass
class TrainingConfig:
    """训练循环的全部超参数（**全部来自配置文件**，不在此处硬编码）。

    属性
    ----
    epochs : int
        最大轮数。
    batch_size : int
        批大小；``<= 0`` 表示全批量。
    lr : float
        学习率。
    momentum : float
        动量系数。
    nesterov : bool
        是否 Nesterov。
    l2_lambda : float
        L2 强度。
    loss_type : str
        损失类型。
    early_stop_patience : int
        早停耐心值。
    early_stop_min_delta : float
        早停最小改善量。
    monitor : str
        监控指标（``val_loss`` / ``val_char_acc`` / ``val_plate_acc``）。
    save_best_only : bool
        是否只保存最佳模型。
    eval_every : int
        每多少轮验证一次。
    log_every : int
        每多少轮打印一次。
    clip_grad_norm : float or None
        梯度范数裁剪阈值。
    lr_decay : float
        每轮学习率乘 ``(1 - lr_decay)``。
    augment_level : str
        数据增强档位名。
    seed : int
        随机种子。
    backend : str
        计算后端名。
    """

    epochs: int = 80
    batch_size: int = 64
    lr: float = 0.05
    momentum: float = 0.9
    nesterov: bool = False
    l2_lambda: float = 1e-4
    loss_type: str = "cross_entropy"
    early_stop_patience: int = 8
    early_stop_min_delta: float = 1e-5
    monitor: str = "val_loss"
    save_best_only: bool = True
    eval_every: int = 1
    log_every: int = 1
    clip_grad_norm: Optional[float] = None
    lr_decay: float = 0.0
    augment_level: str = "weak"
    seed: int = 42
    backend: str = "numpy"

    @classmethod
    def from_config(cls, cfg: Config, seed: Optional[int] = None,
                    backend: Optional[str] = None) -> "TrainingConfig":
        """从全局配置构造。

        参数
        ----
        cfg : Config
            全局配置。
        seed : int or None
            覆盖种子；``None`` 时用 ``cfg.train.seed`` 或 42。
        backend : str or None
            覆盖后端；``None`` 时用 ``cfg.optim.backend``。

        返回
        ----
        TrainingConfig
        """
        t = cfg.train
        o = cfg.optim
        return cls(
            epochs=int(t.epochs),
            batch_size=int(o.batch_size),
            lr=float(o.learning_rate),
            momentum=float(o.get("momentum", 0.0)),
            nesterov=bool(o.get("nesterov", False)),
            l2_lambda=float(cfg.loss.l2_lambda),
            loss_type=str(cfg.loss.type),
            early_stop_patience=int(t.early_stop_patience),
            early_stop_min_delta=float(t.early_stop_min_delta),
            monitor=str(t.monitor),
            save_best_only=bool(t.save_best_only),
            eval_every=int(t.eval_every),
            log_every=int(t.log_every),
            clip_grad_norm=(None if t.get("clip_grad_norm", None) is None
                            else float(t.clip_grad_norm)),
            lr_decay=float(o.get("lr_decay", 0.0)),
            augment_level=str(cfg.augmentation.get("baseline_level", "weak")),
            seed=int(seed if seed is not None else cfg.train.get("seed", 42)),
            backend=str(backend if backend is not None else o.get("backend", "numpy")),
        )

    def as_dict(self) -> Dict[str, Any]:
        """转成可写入日志的字典。"""
        return asdict(self)


@dataclass
class TrainHistory:
    """逐轮训练记录。

    属性
    ----
    epochs : list of dict
        每轮一行，含 ``epoch``、``train_loss``、``train_char_acc``、
        ``val_loss``、``val_char_acc``、``val_plate_acc``、``per_position_i``、
        ``lr``、``epoch_seconds``、``is_best``。
    best_epoch : int
        最佳轮次（从 1 开始）。
    best_score : float
        最佳监控值。
    stopped_epoch : int
        实际停止轮次。
    early_stopped : bool
        是否触发早停。
    """

    epochs: List[Dict[str, Any]] = field(default_factory=list)
    best_epoch: int = 0
    best_score: float = float("inf")
    stopped_epoch: int = 0
    early_stopped: bool = False

    def append(self, row: Dict[str, Any]) -> None:
        """追加一行。"""
        self.epochs.append(row)

    def as_dict(self) -> Dict[str, Any]:
        """转成可写入 JSON 的字典。"""
        return {
            "epochs": self.epochs,
            "best_epoch": self.best_epoch,
            "best_score": self.best_score,
            "stopped_epoch": self.stopped_epoch,
            "early_stopped": self.early_stopped,
            "n_epochs": len(self.epochs),
        }

    def to_csv(self, path: Path) -> None:
        """写出逐轮指标 CSV（§9 要求可复现的数字，而非截图）。

        参数
        ----
        path : Path
            输出路径。

        返回
        ----
        None
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.epochs:
            return
        cols = list(self.epochs[0].keys())
        with open(path, "w", newline="", encoding="utf-8-sig") as fp:
            w = csv.DictWriter(fp, fieldnames=cols)
            w.writeheader()
            for row in self.epochs:
                w.writerow(row)


# =============================================================================
# 2. 数据准备
# =============================================================================


@dataclass
class DataBundle:
    """训练/验证/三个测试集与标准化器。

    属性
    ----
    train : PlateDataset
        训练集（带增强）。
    val : PlateDataset
        验证集（**无增强**）。
    test : PlateDataset
        同分布测试集（**无增强**）。
    hard : PlateDataset
        强扰动测试集（**无增强**）。
    synth : PlateDataset
        合成域测试集（**无增强**）。
    standardizer : GlobalStandardizer
        只在训练集上拟合的标准化器。
    meta : dict
        划分元信息。
    """

    train: PlateDataset
    val: PlateDataset
    test: PlateDataset
    hard: PlateDataset
    synth: PlateDataset
    standardizer: GlobalStandardizer
    meta: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, int]:
        """各集合规模摘要。"""
        return {
            "train": len(self.train), "val": len(self.val),
            "test": len(self.test), "hard_test": len(self.hard),
            "synth_test": len(self.synth),
        }


def load_data_bundle(
    cfg: Config,
    augment_level: Optional[str] = None,
    seed: Optional[int] = None,
    limit: Optional[int] = None,
) -> DataBundle:
    """载入 ``splits.npz`` 与缓存，构造五个数据集。

    参数
    ----
    cfg : Config
        全局配置。
    augment_level : str or None
        增强档位；``None`` 时用 ``cfg.augmentation.baseline_level``。
    seed : int or None
        数据集随机种子。
    limit : int or None
        只取训练集前 N 个样本（**仅用于冒烟测试**，正式实验不要用）。

    返回
    ----
    DataBundle
        数据集集合。

    形状
    ----
    ``splits.npz`` -> 五个 ``PlateDataset``
    """
    processed = resolve_path(cfg, "processed_dir")
    tag = f"ccpd_{int(cfg.ccpd.input_size[0])}x{int(cfg.ccpd.input_size[1])}"
    cache = processed / f"{tag}.npz"
    split_path = processed / "splits.npz"
    if not cache.exists():
        raise FileNotFoundError(f"缺少预处理缓存 {cache}，请先运行 train/phase1_prepare.py")
    if not split_path.exists():
        raise FileNotFoundError(f"缺少划分文件 {split_path}，请先运行 train/phase15_split.py")

    images, labels, _ = load_cache(cache)
    with np.load(split_path, allow_pickle=False) as d:
        tr = d["train"].astype(np.int64)
        va = d["val"].astype(np.int64)
        te = d["test"].astype(np.int64)
        ha = d["hard"].astype(np.int64)
        synth_images = d["synth_images"]
        synth_labels = d["synth_labels"].astype(np.int64)
        std = GlobalStandardizer.from_dict(json.loads(str(d["standardizer"])))
        meta = json.loads(str(d["config"]))

    if limit is not None and int(limit) > 0:
        tr = tr[:int(limit)]

    level = augment_level or str(cfg.augmentation.get("baseline_level", "weak"))
    acfg = AugmentConfig.from_config(cfg, level)
    augment_fn = build_augment_fn(acfg)
    ds_seed = int(seed if seed is not None else cfg.split.get("split_seed", 42))

    def _mk(imgs, labs, name, aug) -> PlateDataset:
        return PlateDataset(images=imgs, labels=labs, standardizer=std,
                            flatten=True, augment_fn=aug, seed=ds_seed, name=name)

    return DataBundle(
        train=_mk(images[tr], labels[tr], "train", augment_fn),
        val=_mk(images[va], labels[va], "val", None),
        test=_mk(images[te], labels[te], "test", None),
        hard=_mk(images[ha], labels[ha], "hard_test", None),
        synth=_mk(synth_images, synth_labels, "synth_test", None),
        standardizer=std,
        meta=meta,
    )


# =============================================================================
# 3. 训练器
# =============================================================================


class Trainer:
    """训练循环（§4.3）。

    参数
    ----
    params : Params
        模型参数。
    data : DataBundle
        数据集合。
    tcfg : TrainingConfig
        训练超参数。
    cfg : Config
        全局配置（用于评价口径与目录）。
    backend : BackendInfo
        计算后端。
    name : str
        本次运行的短名（写入日志/权重文件名）。
    """

    def __init__(self, params: Params, data: DataBundle, tcfg: TrainingConfig,
                 cfg: Config, backend: BackendInfo, name: str = "run") -> None:
        self.params = params
        self.data = data
        self.tcfg = tcfg
        self.cfg = cfg
        self.backend = backend
        self.name = name
        self.head_dims = resolve_positions(cfg.charset.positions)
        self.optim = SGDMomentum(params, OptimConfig(
            lr=tcfg.lr, momentum=tcfg.momentum, nesterov=tcfg.nesterov,
            decay=0.0,                      # L2 已在损失里，避免重复惩罚
            weight_decay_on_bias=False,
        ))
        self.history = TrainHistory()
        self.checkpoint_dir = resolve_path(cfg, "models_dir")
        self.best: Optional[Tuple[float, Params]] = None

        # E9：首位 24 类时，训练集中首位为数字的样本其 head0 无有效目标，
        # 必须用 mask 排除其损失贡献（否则会朝 one-hot 全零的方向优化）。
        self._use_head_mask = (int(self.head_dims[0]) <= LETTER_MAX_INDEX + 1)

    # ------------------------------------------------------------- 辅助
    def _head_mask(self, labels: np.ndarray) -> Optional[List[np.ndarray]]:
        """构造逐头掩码：首位越界的样本在 head0 上权重为 0。

        参数
        ----
        labels : numpy.ndarray
            形状 ``(B, 6)`` 本批标签。

        返回
        ----
        list or None
            长度 6 的掩码列表；不需要掩码时返回 ``None``。

        形状
        ----
        ``(B, 6)`` -> 六个 ``(B,)``
        """
        if not self._use_head_mask:
            return None
        m0 = (labels[:, 0] < int(self.head_dims[0])).astype(np.float32)
        masks = [m0] + [np.ones(labels.shape[0], dtype=np.float32)] * (SEQ_LEN - 1)
        return [self.backend.module.asarray(m) if self.backend.is_gpu else m
                for m in masks]

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """跑一轮训练。

        参数
        ----
        epoch : int
            轮次（从 1 开始），用于打乱种子。

        返回
        ----
        dict
            ``{"train_loss":..., "train_char_acc":..., "lr":...}``。

        形状
        ----
        ``(N, D)`` -> 标量指标
        """
        tcfg = self.tcfg
        params = self.params
        backend = self.backend
        loss_sum = 0.0
        data_sum = 0.0
        correct = 0
        total_chars = 0
        n_seen = 0

        lr = tcfg.lr * ((1.0 - tcfg.lr_decay) ** (epoch - 1))

        for x, y in self.data.train.iter_batches(
            batch_size=tcfg.batch_size, shuffle=True, augment=True,
            seed=tcfg.seed + epoch, drop_last=False,
        ):
            probs, cache = forward(params, x, backend, with_cache=True)
            targets = build_onehot(y, self.head_dims, backend)
            hm = self._head_mask(y)
            total, parts = compute_loss(
                probs, targets, backend, l2_lambda=tcfg.l2_lambda,
                params=params, loss_type=tcfg.loss_type, head_mask=hm,
            )
            grads = backward(params, cache, targets, backend,
                             l2_lambda=tcfg.l2_lambda, loss_type=tcfg.loss_type)

            if tcfg.clip_grad_norm is not None:
                grads = clip_gradients(grads, float(tcfg.clip_grad_norm))

            self.optim.step(grads, lr=lr)

            b = x.shape[0]
            loss_sum += float(total) * b
            data_sum += float(parts["data"]) * b
            preds = np.stack([np.argmax(_np(p), axis=1) for p in probs], axis=1)
            correct += int((preds == y).sum())
            total_chars += int(y.size)
            n_seen += b

        return {
            "train_loss": loss_sum / max(n_seen, 1),
            "train_data_loss": data_sum / max(n_seen, 1),
            "train_char_acc": correct / max(total_chars, 1),
            "lr": lr,
        }

    def _should_stop(self, score: float) -> bool:
        """判断是否早停（监控指标连续 ``patience`` 轮无改善）。

        参数
        ----
        score : float
            本轮监控指标值。

        返回
        ----
        bool
            应停止返回 ``True``。
        """
        better = score < self.history.best_score - self.tcfg.early_stop_min_delta
        if better:
            self.history.best_score = float(score)
            self.history.best_epoch = self.history.stopped_epoch
            return False
        gap = self.history.stopped_epoch - self.history.best_epoch
        return gap >= self.tcfg.early_stop_patience

    def _save_best(self, metrics, epoch: int, extra: Dict[str, Any]) -> Path:
        """保存最佳验证模型（``save_best_only``，§4.3）。

        参数
        ----
        metrics : models.metrics.Metrics
            本轮验证指标。
        epoch : int
            轮次。
        extra : dict
            额外元信息。

        返回
        ----
        Path
            权重路径。

        形状
        ----
        参数张量 -> ``.npz``
        """
        path = self.checkpoint_dir / f"{self.name}_best.npz"
        meta = dict(extra)
        meta.update({
            "epoch": epoch,
            "val_loss": metrics.loss,
            "val_char_acc": metrics.char_acc,
            "val_plate_acc": metrics.plate_acc,
            "per_position": list(metrics.per_position),
            "head_dims": list(self.head_dims),
            "arch": self.params.arch,
        })
        self.params.save(path, extra=meta)
        return path

    # ------------------------------------------------------------- 主循环
    def fit(self, meta: Optional[Dict[str, Any]] = None) -> TrainHistory:
        """执行完整训练循环。

        参数
        ----
        meta : dict or None
            写入权重与日志的运行元信息（commit / 配置指纹 / 种子）。

        返回
        ----
        TrainHistory
            逐轮记录。

        形状
        ----
        ``(N, D)`` x epochs -> TrainHistory
        """
        tcfg = self.tcfg
        meta = dict(meta or {})
        ckpt_path: Optional[Path] = None
        t_start = time.perf_counter()

        for epoch in range(1, tcfg.epochs + 1):
            t0 = time.perf_counter()
            self.history.stopped_epoch = epoch
            tr = self._train_epoch(epoch)

            row: Dict[str, Any] = {"epoch": epoch, **tr}

            do_eval = (epoch % max(tcfg.eval_every, 1) == 0)
            if do_eval:
                res = evaluate_dataset(
                    self.params, self.data.val, self.cfg, backend=self.backend,
                    batch_size=int(self.cfg.eval.batch_size),
                    l2_lambda=tcfg.l2_lambda, loss_type=tcfg.loss_type,
                    head_mask_fn=self._head_mask if self._use_head_mask else None,
                    with_confusion=False,
                )
                m = res.metrics
                row.update({
                    "val_loss": m.loss, "val_data_loss": m.data_loss,
                    "val_char_acc": m.char_acc, "val_plate_acc": m.plate_acc,
                })
                for i, v in enumerate(m.per_position):
                    row[f"per_position_{i}"] = v
                row["val_plate_acc_est_from_char"] = expected_plate_area(
                    m.per_position
                )

                score = float(row.get(tcfg.monitor, m.loss))
                is_best = score < self.history.best_score - tcfg.early_stop_min_delta
                row["is_best"] = int(is_best)
                row["epoch_seconds"] = round(time.perf_counter() - t0, 3)
                self.history.append(row)

                if is_best:
                    self.history.best_score = score
                    self.history.best_epoch = epoch
                    if tcfg.save_best_only:
                        ckpt_path = self._save_best(m, epoch, meta)

                if epoch % max(tcfg.log_every, 1) == 0:
                    print(f"    epoch {epoch:3d}/{tcfg.epochs}  "
                          f"train_loss={tr['train_loss']:.4f} "
                          f"train_char={tr['train_char_acc'] * 100:.2f}%  "
                          f"val_loss={m.loss:.4f} "
                          f"val_char={m.char_acc * 100:.2f}% "
                          f"val_plate={m.plate_acc * 100:.2f}%  "
                          f"lr={tr['lr']:.4f}  {row['epoch_seconds']:.1f}s"
                          + ("  *best*" if is_best else ""))

                if self._should_stop(score):
                    self.history.early_stopped = True
                    print(f"    [早停] 验证损失连续 {tcfg.early_stop_patience} 轮未改善，"
                          f"最佳轮次 {self.history.best_epoch}")
                    break
            else:
                row["epoch_seconds"] = round(time.perf_counter() - t0, 3)
                self.history.append(row)

        self.history.stopped_epoch = len(self.history.epochs)
        total = time.perf_counter() - t_start
        print(f"    训练结束：{len(self.history.epochs)} 轮，"
              f"最佳轮次 {self.history.best_epoch}，"
              f"最佳 {tcfg.monitor}={self.history.best_score:.4f}，"
              f"耗时 {total:.1f}s")

        if not tcfg.save_best_only:
            self.params.save(self.checkpoint_dir / f"{self.name}_last.npz",
                             extra=dict(meta, epoch=len(self.history.epochs)))

        # 训练结束后回滚到最佳权重，保证后续测试用的是最佳模型（§4.3）
        if ckpt_path is not None and ckpt_path.exists():
            best_params, _ = Params.load(ckpt_path)
            self._copy_params(best_params)
            print(f"    已回滚到最佳权重：{ckpt_path.name}")
        return self.history

    def _copy_params(self, src: Params) -> None:
        """把 ``src`` 的数值原地拷进 ``self.params``（保持对象身份）。

        参数
        ----
        src : Params
            源参数。

        返回
        ----
        None
        """
        for (n, dst), (n2, s) in zip(params_groups(self.params), params_groups(src)):
            if n == n2 and dst.shape == s.shape:
                dst[...] = s


def expected_plate_area(per_position: Sequence[float]) -> float:
    """由逐位置准确率估算整牌准确率（独立假设）。

    参数
    ----
    per_position : Sequence[float]
        长度 6 的逐位置准确率。

    返回
    ----
    float
        估算的整牌准确率。

    形状
    ----
    6 -> 标量
    """
    return round(expected_plate_acc_from_char(0.0, per_position), 6)


def _np(x):
    """把后端数组转成 numpy（内部小工具）。"""
    from models.backend import asnumpy

    return asnumpy(x)


def clip_gradients(grads: Dict[str, np.ndarray], max_norm: float) -> Dict[str, np.ndarray]:
    """全局范数梯度裁剪（§附录 B 第 11 条，本项目默认关闭）。

    参数
    ----
    grads : dict
        ``{参数名: 梯度}``。
    max_norm : float
        全局范数上限。

    返回
    ----
    dict
        裁剪后的梯度（新字典）。

    形状
    ----
    与参数同形状
    """
    total = 0.0
    for g in grads.values():
        total += float(np.sum(np.square(g)))
    norm = float(np.sqrt(total))
    if norm <= max_norm or norm == 0.0:
        return grads
    scale = max_norm / (norm + 1e-12)
    return {k: v * scale for k, v in grads.items()}


# =============================================================================
# 4. 单次运行入口
# =============================================================================


def run_training(
    cfg: Config,
    name: str,
    seed: Optional[int] = None,
    backend_name: Optional[str] = None,
    limit: Optional[int] = None,
    epochs: Optional[int] = None,
    quiet: bool = False,
) -> Dict[str, Any]:
    """执行一次完整训练并落盘所有产物。

    参数
    ----
    cfg : Config
        全局配置（**应已应用过变体补丁**）。
    name : str
        运行短名，用于文件名。
    seed : int or None
        随机种子。
    backend_name : str or None
        计算后端覆盖。
    limit : int or None
        训练集截断（冒烟测试）。
    epochs : int or None
        轮数覆盖。
    quiet : bool
        是否少打印。

    返回
    ----
    dict
        含 ``history``、``train``、``val``、``test``、``hard_test``、``synth_test``
        指标、``meta``、``paths`` 等。

    形状
    ----
    数据集 -> 指标字典 + 落盘文件
    """
    ensure_dirs(cfg)
    tcfg = TrainingConfig.from_config(cfg, seed=seed, backend=backend_name)
    if epochs is not None:
        tcfg.epochs = int(epochs)
    set_seed(tcfg.seed)

    backend = get_backend(tcfg.backend, verbose=not quiet)
    set_backend_env(backend)
    if not quiet:
        print(f"[train] {name}: seed={tcfg.seed} backend={backend.name} "
              f"({backend.device_name}) arch={cfg.model.arch} "
              f"act={cfg.model.activation} H={cfg.model.hidden_dim} "
              f"loss={tcfg.loss_type} λ={tcfg.l2_lambda} "
              f"bs={tcfg.batch_size} lr={tcfg.lr} μ={tcfg.momentum} "
              f"aug={tcfg.augment_level}")

    data = load_data_bundle(cfg, augment_level=tcfg.augment_level,
                            seed=tcfg.seed, limit=limit)
    if not quiet:
        print(f"[train] 数据：{data.summary()}  标准化 "
              f"mean={data.standardizer.mean:.6f} std={data.standardizer.std:.6f} "
              f"(n={data.standardizer.n_samples})")

    head_dims = resolve_positions(cfg.charset.positions)
    params = build_model(
        input_dim=int(cfg.model.input_dim), hidden_dim=int(cfg.model.hidden_dim),
        head_dims=head_dims, arch=str(cfg.model.arch),
        activation=str(cfg.model.activation), init=str(cfg.model.init),
        seed=tcfg.seed,
    )
    if not quiet:
        print(f"[train] 模型：{params.describe()}")

    gi = git_info()
    meta: Dict[str, Any] = {
        "run_name": name,
        "seed": tcfg.seed,
        "commit": gi.commit,
        "branch": gi.branch,
        "dirty": gi.dirty,
        "config_fingerprint": cfg.fingerprint(),
        "backend": backend.name,
        "device_name": backend.device_name,
        "training": tcfg.as_dict(),
        "model": params.describe(),
        "head_dims": list(head_dims),
        "split": data.summary(),
        "standardizer": data.standardizer.as_dict(),
        "preprocess_version": str(cfg.project.preprocess_version),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    trainer = Trainer(params, data, tcfg, cfg, backend, name=name)
    t0 = time.perf_counter()
    history = trainer.fit(meta=meta)
    train_seconds = time.perf_counter() - t0

    # ---- 训练集与验证集指标（用最佳权重做最终口径） ----------------------
    def _eval(ds, with_conf: bool = False):
        return evaluate_dataset(
            trainer.params, ds, cfg, backend=backend,
            batch_size=int(cfg.eval.batch_size),
            l2_lambda=tcfg.l2_lambda, loss_type=tcfg.loss_type,
            head_mask_fn=trainer._head_mask if trainer._use_head_mask else None,
            with_confusion=with_conf,
        )

    res_train = _eval(data.train)
    res_val = _eval(data.val, with_conf=True)
    res_test = _eval(data.test, with_conf=True)
    res_hard = _eval(data.hard, with_conf=True)
    res_synth = _eval(data.synth, with_conf=True)

    from models.backend import memory_info as _mem

    result: Dict[str, Any] = {
        "meta": meta,
        "train_seconds": round(train_seconds, 2),
        "epochs_run": len(history.epochs),
        "early_stopped": history.early_stopped,
        "best_epoch": history.best_epoch,
        "history": history.as_dict(),
        "train": res_train.metrics.as_dict(),
        "val": res_val.metrics.as_dict(),
        "test": res_test.metrics.as_dict(),
        "hard_test": res_hard.metrics.as_dict(),
        "synth_test": res_synth.metrics.as_dict(),
        "gpu_memory": _mem(backend),
    }

    # ---- 落盘 -------------------------------------------------------------
    log_dir = resolve_path(cfg, "logs_dir")
    history.to_csv(log_dir / f"{name}_history.csv")
    with open(log_dir / f"{name}_run.json", "w", encoding="utf-8") as fp:
        json.dump(result, fp, ensure_ascii=False, indent=2)

    # 逐位置准确率对比（三个测试集）
    with open(log_dir / f"{name}_per_position.csv", "w", newline="",
              encoding="utf-8-sig") as fp:
        w = csv.writer(fp)
        w.writerow(["position_1based"] + ["train", "val", "test",
                                          "hard_test", "synth_test"])
        for i in range(SEQ_LEN):
            w.writerow([i + 1] + [round(res_train.metrics.per_position[i], 6),
                                  round(res_val.metrics.per_position[i], 6),
                                  round(res_test.metrics.per_position[i], 6),
                                  round(res_hard.metrics.per_position[i], 6),
                                  round(res_synth.metrics.per_position[i], 6)])

    result["paths"] = {
        "history_csv": str(log_dir / f"{name}_history.csv"),
        "run_json": str(log_dir / f"{name}_run.json"),
        "per_position_csv": str(log_dir / f"{name}_per_position.csv"),
        "checkpoint": str(trainer.checkpoint_dir / f"{name}_best.npz"),
    }

    if not quiet:
        print(f"[train] 最终指标（最佳权重）：")
        for tag, res in (("验证集", res_val), ("同分布测试集", res_test),
                         ("强扰动测试集", res_hard), ("合成域测试集", res_synth)):
            m = res.metrics
            print(f"    {tag:8s} 字符={m.char_acc * 100:6.2f}%  "
                  f"整牌={m.plate_acc * 100:6.2f}%  损失={m.loss:.4f}")
    return result


# =============================================================================
# 5. CLI
# =============================================================================


def main(argv: Optional[Sequence[str]] = None) -> int:
    """训练命令行入口。

    参数
    ----
    argv : Sequence[str] or None
        命令行参数。

    返回
    ----
    int
        0 表示成功。
    """
    ap = argparse.ArgumentParser(description="ProjectX 训练（§4）")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--name", type=str, default=None, help="运行短名")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--backend", type=str, default=None, choices=["numpy", "cupy", "auto"])
    ap.add_argument("--epochs", type=int, default=None, help="覆盖轮数（冒烟测试）")
    ap.add_argument("--limit", type=int, default=None, help="训练集截断（冒烟测试）")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    name = args.name or f"baseline_s{args.seed if args.seed is not None else 'default'}"
    res = run_training(cfg, name=name, seed=args.seed, backend_name=args.backend,
                       limit=args.limit, epochs=args.epochs)
    print(f"[train] 完成：{json.dumps(res['paths'], ensure_ascii=False, indent=2)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())