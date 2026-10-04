# -*- coding: utf-8 -*-
"""域差异诊断：真实域与合成域到底差在哪里（§8 的证据脚本）。

为什么需要这个脚本
------------------
模型在真实域测试集上能到 98% 字符准确率，却只拿到约 5% 的合成域准确率
（34 类随机猜测为 2.9%）。在把它判定为"域差异"之前，必须逐项排除实现问题：
标签是否对齐、字符位置是否错位、输入尺度是否偏移、标准化统计量是否不匹配。
本脚本把这些排查项变成**可重复运行的数值证据**，输出 JSON，
供实验报告引用（规格 §0.3 禁止把结论只放在截图里）。

排查项
------
1. 标签一致性：``synth_texts`` 与 ``synth_labels`` 解码是否逐张相同。
2. 字符位置：逐格测墨迹质心与其跨样本标准差（真实 vs 合成）。
   位置对不上会导致"看起来像是学不会"的假象。
3. 输入尺度：标准化前后的均值/标准差对比。
4. 标准化方案敏感性：全局统计量 / 合成域自身统计量 / 逐图归一化，
   三种情况下合成域准确率是否有差别（若无差别，说明不是尺度问题）。

用法
----
    python train/diag_domain_gap.py
    python train/diag_domain_gap.py --n 600 --out reports/logs/domain_gap.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.backend import get_backend  # noqa: E402
from models.config import git_info, load_config, resolve_path  # noqa: E402
from models.dataset import GlobalStandardizer, load_cache  # noqa: E402
from models.charset import decode_batch  # noqa: E402
from models.model import Params, forward, predict  # noqa: E402


def load_splits(cfg) -> Dict[str, np.ndarray]:
    """载入图像缓存与划分索引。

    参数
    ----
    cfg : Config
        全局配置。

    返回
    ----
    dict
        含 ``images``、``labels``、``texts`` 与各划分索引。

    形状
    ----
    图像 (N, 32, 128) uint8；标签 (N, 6) int64
    """
    p = resolve_path(cfg, "processed_dir")
    w, h = int(cfg.ccpd.input_size[0]), int(cfg.ccpd.input_size[1])
    images, labels, meta = load_cache(p / f"ccpd_{w}x{h}.npz")
    with np.load(p / "splits.npz", allow_pickle=True) as d:
        # ★ 坑：standardizer / config 是以 0 维**字符串**数组存的 JSON
        # （np.save 时 allow_pickle=False，字典被序列化成 JSON 文本）。
        # 所以这里必须 json.loads 再取键，不能直接当字典索引。
        sd = d["standardizer"]
        std_obj = None
        if getattr(sd, "shape", None) == ():
            raw = sd.item()
            std_obj = json.loads(raw) if isinstance(raw, str) else raw
        out: Dict[str, Any] = {
            "images": np.asarray(images),
            "labels": np.asarray(labels),
            "std": std_obj,
        }
        for k in ("train", "val", "test", "hard", "synth_images", "synth_labels",
                  "synth_texts"):
            if k in d:
                out[k] = np.asarray(d[k])
    return out


def check_labels(synth_labels: np.ndarray, synth_texts: np.ndarray
                 ) -> Dict[str, Any]:
    """核对合成域标签与渲染文本是否一致。

    参数
    ----
    synth_labels : np.ndarray
        形状 (N, 6) 的类别索引。
    synth_texts  : np.ndarray
        形状 (N,) 的字符串标签。

    返回
    ----
    dict
        ``n``、``n_match``、``n_mismatch``、``examples``。
    """
    dec = decode_batch(synth_labels)
    n = len(dec)
    bad = [i for i in range(n) if str(dec[i]) != str(synth_texts[i])]
    return {
        "n": int(n),
        "n_match": int(n - len(bad)),
        "n_mismatch": int(len(bad)),
        "examples": [{"index": int(i), "decoded": str(dec[i]),
                      "rendered": str(synth_texts[i])} for i in bad[:10]],
    }


def ink_centroids(images: np.ndarray, n: int = 600) -> Dict[str, Any]:
    """逐格测字符墨迹的横向质心及其跨样本标准差。

    做法：用 Otsu 式阈值把图二值化，按 6 等分格切分，对每格求
    墨迹像素（亮像素）的重心 x。位置对齐时真实域与合成域的质心应相近，
    且跨样本标准差不应差异过大。

    参数
    ----
    images : np.ndarray
        形状 (N, 32, 128) 的灰度图（0–255）。
    n : int
        取样张数。

    返回
    ----
    dict
        每格的 ``mean_x`` 与 ``std_x``（像素）。
    """
    use = images[:min(n, len(images))].astype(np.float32)
    thresh = use.mean(axis=(1, 2), keepdims=True)
    mask = use > thresh
    cell = use.shape[2] // 6
    out = {}
    for c in range(6):
        xs, ok = [], 0
        for m in mask:
            sub = m[:, c * cell:(c + 1) * cell]
            if sub.sum() < 3:
                continue
            ys, xs_ = np.nonzero(sub)
            xs.append(float(xs_.mean()) + c * cell)
            ok += 1
        out[f"pos{c + 1}"] = {
            "mean_x": float(np.mean(xs)) if xs else None,
            "std_x": float(np.std(xs)) if xs else None,
            "n_used": int(ok),
        }
    return out


def scale_stats(images: np.ndarray, std: Optional[Dict[str, float]]
                ) -> Dict[str, Any]:
    """报告原始像素尺度与标准化后的尺度统计。

    参数
    ----
    images : np.ndarray
        形状 (N, 32, 128) 灰度图（0–255）。
    std : dict or None
        训练集拟合的 ``{"mean": ..., "std": ...}``（[0,1] 尺度）。

    返回
    ----
    dict
        ``raw_mean_0_255``、``raw_std_0_255``、
        ``norm_mean_sigma``、``norm_std_sigma``。
    """
    x = images.astype(np.float32) / 255.0
    res: Dict[str, Any] = {
        "raw_mean_0_255": float(images.astype(np.float32).mean()),
        "raw_std_0_255": float(images.astype(np.float32).std()),
        "raw_mean_0_1": float(x.mean()),
        "raw_std_0_1": float(x.std()),
    }
    if std:
        res["norm_mean_sigma"] = float((x.mean() - float(std["mean"])) /
                                       float(std["std"]))
        res["norm_std_sigma"] = float(x.std() / float(std["std"]))
    return res


# =============================================================================
# 追加分析（§8.1 证据补齐）：标准化方案敏感性
# =============================================================================


def _eval_char_acc(params, x: np.ndarray, y: np.ndarray, batch: int = 512
                   ) -> Dict[str, Any]:
    """在已标准化的输入上评测字符/整牌/逐位置准确率（numpy 后端）。

    参数
    ----
    params : Params
        模型参数（baseline 为 float32）。
    x : numpy.ndarray
        已标准化并展平的输入，形状 ``(N, D)``，float32。
    y : numpy.ndarray
        标签，形状 ``(N, 6)``，int64。
    batch : int
        评测批大小。

    返回
    ----
    dict
        ``{"char_acc", "plate_acc", "per_position"}``；``per_position``
        为长度 6 的逐位置字符准确率。

    形状
    ----
    ``(N, D)`` + ``(N, 6)`` -> 标量指标
    """
    backend = get_backend("numpy", verbose=False)  # 铁律：评测用 numpy 后端
    n = int(x.shape[0])
    preds_all = []
    for s in range(0, n, batch):
        xb = x[s:s + batch]
        probs, _ = forward(params, xb, backend, with_cache=False)
        preds, _ = predict(probs, backend)
        preds_all.append(preds)
    pred = np.concatenate(preds_all, axis=0)              # (N, 6) int64
    correct = pred == y
    return {
        "char_acc": float(correct.mean()),
        "plate_acc": float(correct.all(axis=1).mean()),
        "per_position": [float(v) for v in correct.mean(axis=0)],
    }


def standardization_sensitivity(cfg, sp: Dict[str, Any]) -> Dict[str, Any]:
    """三种标准化方案下 baseline 在合成域的字符准确率（§8.1 的排除证据）。

    报告 §8.1 声称"三种标准化方案在合成域都只有 ~5%（逐图归一化 4.78% vs
    4.94%）"，但此前没有 JSON 产物。本函数补齐该证据：加载
    ``reports/checkpoints/baseline_s42_best.npz``（float32 训练产物），
    对 synth_test 全部 2000 张分别用三种方案评测：

    * ``global_train``：当前方案，用 **train 拟合**的全局 mean/std（来自
      ``splits.npz`` 的 ``standardizer`` 键）；
    * ``global_synth``：用 **synth 自身拟合**的全局 mean/std；
    * ``per_image``：逐图归一化（每图减自身均值、除自身标准差）。

    输入一律为 ``uint8 / 255`` 后的 ``[0, 1]`` 图像，再按方案做
    ``(x - mean) / std``（评测口径与训练一致）。

    参数
    ----
    cfg : Config
        全局配置（用于定位 checkpoint）。
    sp : dict
        :func:`load_splits` 的输出，须含 ``synth_images``、``synth_labels``
        与 ``std``（train 拟合标准化器字典）。

    返回
    ----
    dict
        三个方案的指标 + checkpoint 元信息 + commit/dirty。

    形状
    ----
    图像 ``(2000, 32, 128) uint8`` -> 三种 ``(2000, 4096) float32`` 输入
    -> 三组标量指标
    """
    ckpt = resolve_path(cfg, "models_dir") / "baseline_s42_best.npz"
    # ★ Params.load 是 classmethod，返回 (params, extra)，必须接收两个返回值
    params, extra = Params.load(ckpt)

    images = np.asarray(sp["synth_images"])                  # (N, 32, 128) uint8
    labels = np.asarray(sp["synth_labels"]).astype(np.int64)  # (N, 6)
    n = int(images.shape[0])
    x01 = images.astype(np.float32) / 255.0                  # (N, 32, 128) [0,1]

    # (a) 当前方案：train 拟合的全局 mean/std
    std_train = GlobalStandardizer.from_dict(sp["std"])
    x_a = std_train.transform(x01).reshape(n, -1).astype(np.float32)

    # (b) 用 synth 自身拟合的全局 mean/std
    std_synth = GlobalStandardizer.fit(x01)
    x_b = std_synth.transform(x01).reshape(n, -1).astype(np.float32)

    # (c) 逐图归一化：每图减自身均值、除自身标准差（eps 防除零）
    mu = x01.mean(axis=(1, 2), keepdims=True)                # (N, 1, 1)
    sg = np.maximum(x01.std(axis=(1, 2), keepdims=True), 1e-8)
    x_c = ((x01 - mu) / sg).reshape(n, -1).astype(np.float32)

    schemes = {
        "global_train": ("train 拟合的全局 mean/std（当前方案）", x_a,
                         std_train.as_dict()),
        "global_synth": ("synth 自身拟合的全局 mean/std", x_b,
                         std_synth.as_dict()),
        "per_image": ("逐图归一化（每图减自身均值除自身标准差）", x_c, None),
    }
    results: Dict[str, Any] = {}
    for name, (desc, x, std_dict) in schemes.items():
        m = _eval_char_acc(params, x, labels)
        m["description"] = desc
        if std_dict is not None:
            m["standardizer"] = std_dict
        results[name] = m
        print(f"  [标准化敏感性] {name:13s} 字符准确率 {m['char_acc'] * 100:6.2f}%  "
              f"整牌 {m['plate_acc'] * 100:5.2f}%")

    gi = git_info()
    return {
        "checkpoint": "reports/checkpoints/baseline_s42_best.npz",
        "checkpoint_extra": {
            "run_name": extra.get("run_name"),
            "seed": extra.get("seed"),
            "val_char_acc": extra.get("val_char_acc"),
        },
        "n_synth": n,
        "eval_backend": "numpy",
        "params_dtype": str(params.W1.dtype),
        "schemes": results,
        "note": ("三种方案在合成域的字符准确率若都停留在 ~5%（34 类随机猜测 "
                 "2.9% 附近），则'标准化统计量不匹配'可被排除（§8.1）。"),
        "commit": gi.commit,
        "dirty": gi.dirty,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """命令行入口。

    参数
    ----
    argv : Sequence[str] or None
        命令行参数。

    返回
    ----
    int
        0 表示成功。
    """
    ap = argparse.ArgumentParser(description="真实域 vs 合成域差异诊断")
    ap.add_argument("--config", default=None)
    ap.add_argument("--n", type=int, default=600, help="位置统计取样张数")
    ap.add_argument("--out", default=None,
                    help="JSON 输出路径，默认 reports/logs/domain_gap.json")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    sp = load_splits(cfg)

    test_idx = sp["test"].astype(np.int64)[:args.n]
    real = sp["images"][test_idx]
    synth = sp["synth_images"]

    std = sp.get("std")
    report: Dict[str, Any] = {
        "n_real_used": int(len(real)),
        "n_synth": int(len(synth)),
        "labels": check_labels(sp["synth_labels"], sp["synth_texts"]),
        "centroids_real": ink_centroids(real, args.n),
        "centroids_synth": ink_centroids(synth, args.n),
        "scale_real": scale_stats(real, std),
        "scale_synth": scale_stats(synth, std),
        "standardizer_used": std,
    }

    # ---- 追加分析：标准化方案敏感性（§8.1 证据补齐） ------------------------
    report["standardization_sensitivity"] = standardization_sensitivity(cfg, sp)

    out = (Path(args.out) if args.out
           else resolve_path(cfg, "logs_dir") / "domain_gap.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    # 读取现有 JSON → 加键/更新键 → 写回：已有键一律保留（幂等，
    # 重复运行结果一致；旧文件中本脚本不认识的键也不会丢失）
    if out.exists():
        try:
            old = json.loads(out.read_text(encoding="utf-8"))
            if isinstance(old, dict):
                old.update(report)
                report = old
        except Exception:
            pass  # 旧文件损坏时以本次计算结果为准
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding="utf-8")

    print(f"标签一致性：{report['labels']['n_match']}/{report['labels']['n']} 一致，"
          f"不一致 {report['labels']['n_mismatch']} 张")
    print(f"{'位置':>6}{'真实 质心x ± σ':>22}{'合成 质心x ± σ':>22}")
    for c in range(6):
        k = f"pos{c + 1}"
        r, s = report["centroids_real"][k], report["centroids_synth"][k]
        print(f"{k:>6}{r['mean_x']:>12.2f} ± {r['std_x']:<7.2f}"
              f"{s['mean_x']:>12.2f} ± {s['std_x']:<7.2f}")
    print(f"\n原始像素均值：真实 {report['scale_real']['raw_mean_0_255']:.1f}"
          f"  合成 {report['scale_synth']['raw_mean_0_255']:.1f}")
    print(f"标准化后均值（σ 单位）：真实 "
          f"{report['scale_real'].get('norm_mean_sigma', float('nan')):+.3f}"
          f"  合成 {report['scale_synth'].get('norm_mean_sigma', float('nan')):+.3f}")
    print(f"标准化后标准差：真实 "
          f"{report['scale_real'].get('norm_std_sigma', float('nan')):.3f}"
          f"  合成 {report['scale_synth'].get('norm_std_sigma', float('nan')):.3f}")
    ss = report["standardization_sensitivity"]
    print(f"\n标准化方案敏感性（合成域 {ss['n_synth']} 张，baseline_s42_best）：")
    for k, v in ss["schemes"].items():
        print(f"  {k:14s} 字符准确率 {v['char_acc'] * 100:6.2f}%  "
              f"整牌 {v['plate_acc'] * 100:5.2f}%")
    print(f"\n已写出：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())