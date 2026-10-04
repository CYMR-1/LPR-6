# -*- coding: utf-8 -*-
"""增强强度诊断：逐算子拆解，量化各增强项对图像的破坏程度（§7.4 的证据脚本）。

为什么需要这个脚本
------------------
E7 的实测结果是 **增强越强越差**（测试整牌 91.8% → 43.4% → 2.4%）。
这个结论很容易被误读成"增强实现有 bug（比如把图弄坏了）"，因此必须给出
**逐算子**的定量证据：每次只开一个算子，测量它与原图的差异，以及各档位
叠加后的效果。

指标含义
--------
* ``mean_abs_diff``：与未增强图像的平均绝对差（0–1 尺度）。越大改得越多。
* ``contrast_std``  ：增强后图像的全局像素标准差。越小说明对比度被压掉。
* ``edge_energy``   ：相邻像素横向差分均值。越小说明边缘越糊（字越看不清）。

★ 注意：增强函数期望输入在 ``[0, 1]``。若误传 0–255，亮度/对比度/JPEG
等算子会产出垃圾结果（本项目第一次诊断就踩了这个坑）。本脚本显式做
``/255.0`` 归一化。

用法
----
    python train/diag_augment.py
    python train/diag_augment.py --samples 8 --out reports/logs/augment_diag.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.augment import AugmentConfig, apply_augmentation  # noqa: E402
from models.config import load_config, resolve_path  # noqa: E402
from models.dataset import load_cache  # noqa: E402


# 逐算子隔离：把其余算子关掉，只留要测的那一个。
# 键是展示名，值是要覆盖回默认（关闭）的字段。
_SINGLE_OPS: Dict[str, Dict[str, Any]] = {
    "几何(旋转/平移/缩放)": {
        "gaussian_blur_prob": 0, "motion_blur_prob": 0,
        "gaussian_noise_prob": 0, "salt_pepper_prob": 0, "jpeg_prob": 0,
    },
    "高斯模糊": {
        "rotation_deg": 0, "translate_px": 0, "scale_range": [1.0, 1.0],
        "motion_blur_prob": 0, "gaussian_noise_prob": 0,
        "salt_pepper_prob": 0, "jpeg_prob": 0,
    },
    "运动模糊": {
        "rotation_deg": 0, "translate_px": 0, "scale_range": [1.0, 1.0],
        "gaussian_blur_prob": 0, "gaussian_noise_prob": 0,
        "salt_pepper_prob": 0, "jpeg_prob": 0,
    },
    "高斯噪声": {
        "rotation_deg": 0, "translate_px": 0, "scale_range": [1.0, 1.0],
        "gaussian_blur_prob": 0, "motion_blur_prob": 0,
        "salt_pepper_prob": 0, "jpeg_prob": 0,
    },
    "椒盐噪声": {
        "rotation_deg": 0, "translate_px": 0, "scale_range": [1.0, 1.0],
        "gaussian_blur_prob": 0, "motion_blur_prob": 0,
        "gaussian_noise_prob": 0, "jpeg_prob": 0,
    },
    "JPEG 压缩": {
        "rotation_deg": 0, "translate_px": 0, "scale_range": [1.0, 1.0],
        "gaussian_blur_prob": 0, "motion_blur_prob": 0,
        "gaussian_noise_prob": 0, "salt_pepper_prob": 0,
    },
    "亮度/对比度": {
        "rotation_deg": 0, "translate_px": 0, "scale_range": [1.0, 1.0],
        "gaussian_blur_prob": 0, "motion_blur_prob": 0,
        "gaussian_noise_prob": 0, "salt_pepper_prob": 0, "jpeg_prob": 0,
    },
}

# 关闭一切增强用的"复位"字段（与 _SINGLE_OPS 的键合集一致）
_ALL_OFF: Dict[str, Any] = {
    "rotation_deg": 0, "translate_px": 0, "scale_range": [1.0, 1.0],
    "gaussian_blur_prob": 0, "motion_blur_prob": 0,
    "gaussian_noise_prob": 0, "salt_pepper_prob": 0, "jpeg_prob": 0,
}


def _stats(arr: np.ndarray, ref: np.ndarray) -> Dict[str, float]:
    """计算一组增强图像相对参考图的破坏程度指标。

    参数
    ----
    arr : np.ndarray
        形状 (N, 32, 128)、取值 [0,1] 的增强后图像。
    ref : np.ndarray
        形状 (N, 32, 128)、取值 [0,1] 的未增强参考图。

    返回
    ----
    dict
        ``mean_abs_diff``、``contrast_std``、``edge_energy``。
    """
    return {
        "mean_abs_diff": float(np.abs(arr - ref).mean()),
        "contrast_std": float(arr.std(axis=(1, 2)).mean()),
        "edge_energy": float(np.abs(np.diff(arr, axis=2)).mean()),
    }


def diagnose(cfg, images: np.ndarray, n: int, seed: int) -> Dict[str, Any]:
    """逐算子与逐档位测量破坏程度。

    参数
    ----
    cfg : Config
        全局配置。
    images : np.ndarray
        形状 (N, 32, 128) uint8 的真实域图像。
    n : int
        取样张数。
    seed : int
        增强随机种子。

    返回
    ----
    dict
        ``reference``、``by_operator``、``by_level``。
    """
    ref = images[:n].astype(np.float32) / 255.0   # ★ 必须是 [0,1]
    weak = dict(dict(cfg.augmentation.levels.weak))

    out: Dict[str, Any] = {
        "n_samples": int(n),
        "reference": {
            "mean_abs_diff": 0.0,
            "contrast_std": float(ref.std(axis=(1, 2)).mean()),
            "edge_energy": float(np.abs(np.diff(ref, axis=2)).mean()),
        },
        "by_operator": {},
        "by_level": {},
    }

    for name, override in _SINGLE_OPS.items():
        d = dict(weak)
        d.update(override)
        d["enabled"] = True
        acfg = AugmentConfig(**d)
        rng = np.random.default_rng(seed)
        arr = np.stack([np.asarray(apply_augmentation(a, acfg, rng),
                                   dtype=np.float32) for a in ref])
        out["by_operator"][name] = _stats(arr, ref)

    for level in ("none", "weak", "strong"):
        acfg = AugmentConfig.from_config(cfg, level)
        rng = np.random.default_rng(seed)
        arr = np.stack([np.asarray(apply_augmentation(a, acfg, rng),
                                   dtype=np.float32) for a in ref])
        st = _stats(arr, ref)
        st["enabled"] = bool(acfg.enabled)
        out["by_level"][level] = st

    return out


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
    ap = argparse.ArgumentParser(description="数据增强逐算子破坏程度诊断")
    ap.add_argument("--config", default=None)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default=None,
                    help="默认 reports/logs/augment_diag.json")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    p = resolve_path(cfg, "processed_dir")
    w, h = int(cfg.ccpd.input_size[0]), int(cfg.ccpd.input_size[1])
    images, labels, _ = load_cache(p / f"ccpd_{w}x{h}.npz")
    with np.load(p / "splits.npz", allow_pickle=True) as d:
        idx = d["test"].astype(np.int64)[:args.samples]

    rep = diagnose(cfg, images[idx], args.samples, args.seed)

    out = (Path(args.out) if args.out
           else resolve_path(cfg, "logs_dir") / "augment_diag.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                   encoding="utf-8")

    print(f"{'配置':26s}{'与原图差异':>12}{'对比度std':>12}{'边缘能量':>12}")
    r = rep["reference"]
    print(f"{'原始（未增强）':26s}{r['mean_abs_diff']:>12.4f}"
          f"{r['contrast_std']:>12.4f}{r['edge_energy']:>12.4f}")
    for name, st in rep["by_operator"].items():
        print(f"{name:26s}{st['mean_abs_diff']:>12.4f}"
              f"{st['contrast_std']:>12.4f}{st['edge_energy']:>12.4f}")
    print()
    for lv, st in rep["by_level"].items():
        print(f"[档位 {lv:6s}] enabled={st['enabled']} 差异="
              f"{st['mean_abs_diff']:.4f} 对比度={st['contrast_std']:.4f} "
              f"边缘={st['edge_energy']:.4f}")
    print(f"\n已写出：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())