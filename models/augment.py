# -*- coding: utf-8 -*-
"""数据增强（§2.6）：训练时随机施加，强度分「无 / 弱（基线）/ 强」三档。

设计约束
--------
* 增强作用在**单张 float32 图像** ``(H, W)``、取值 ``[0, 1]`` 上，
  由 :class:`models.dataset.PlateDataset` 在取批时按样本调用；
* **只在训练集启用**；验证/测试集必须保持原样，否则指标不可比；
* 强度档位由 ``configs/default.yaml`` 的 ``augmentation.levels`` 定义，
  强度档位通过 ``augmentation.baseline_level`` 切换；
* 依赖只用 numpy 与 Pillow（§8.1 依赖约束）。

支持的增强（§2.6 表）
---------------------
============================  ==========================================
类别                           实现
============================  ==========================================
几何：旋转 / 平移 / 轻微缩放    :func:`geom_rotate` / :func:`geom_translate`
                              / :func:`geom_scale`
成像：高斯模糊                  :func:`blur_gaussian`
成像：运动模糊                  :func:`blur_motion`
成像：高斯噪声                  :func:`noise_gaussian`
成像：椒盐噪声                  :func:`noise_salt_pepper`
成像：亮度 / 对比度             :func:`photometric`
编码：JPEG 压缩                 :func:`jpeg_compress`
============================  ==========================================

所有函数都遵循「形状 ``(H, W)`` -> 形状 ``(H, W)``」的约定，并保持取值在
``[0, 1]``（JPEG 压缩因 8 位量化会有微小损失，属预期）。
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from PIL import Image, ImageFilter

# --- 包引导：支持直接运行本文件 ---------------------------------------------
import sys as _sys

if __package__ in (None, ""):
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# =============================================================================
# 1. 单项增强算子
# =============================================================================


def geom_rotate(img: np.ndarray, rng: np.random.Generator, max_deg: float) -> np.ndarray:
    """随机旋转（绕图像中心，边缘用边界像素填充）。

    参数
    ----
    img : numpy.ndarray
        形状 ``(H, W)``，float32，取值 ``[0, 1]``。
    rng : numpy.random.Generator
        随机源。
    max_deg : float
        最大旋转角度（度），实际角度在 ``[-max_deg, max_deg]`` 均匀采样。

    返回
    ----
    numpy.ndarray
        形状 ``(H, W)``，float32。

    形状
    ----
    ``(H, W)`` -> ``(H, W)``
    """
    if max_deg <= 0:
        return img
    ang = float(rng.uniform(-max_deg, max_deg))
    pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8), mode="L")
    pil = pil.rotate(ang, resample=Image.BILINEAR, expand=False)
    return np.asarray(pil, dtype=np.float32) / 255.0


def geom_translate(img: np.ndarray, rng: np.random.Generator, max_px: float) -> np.ndarray:
    """随机平移（水平与垂直独立采样，越界处补 0）。

    参数
    ----
    img : numpy.ndarray
        形状 ``(H, W)``，float32，取值 ``[0, 1]``。
    rng : numpy.random.Generator
        随机源。
    max_px : float
        最大平移像素数。

    返回
    ----
    numpy.ndarray
        形状 ``(H, W)``，float32。

    形状
    ----
    ``(H, W)`` -> ``(H, W)``
    """
    if max_px <= 0:
        return img
    dx = int(round(float(rng.uniform(-max_px, max_px))))
    dy = int(round(float(rng.uniform(-max_px, max_px))))
    if dx == 0 and dy == 0:
        return img
    out = np.zeros_like(img)
    h, w = img.shape
    xs0, xs1 = max(0, -dx), min(w, w - dx)
    ys0, ys1 = max(0, -dy), min(h, h - dy)
    if xs0 >= xs1 or ys0 >= ys1:
        return img
    out[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx] = img[ys0:ys1, xs0:xs1]
    return out


def geom_scale(img: np.ndarray, rng: np.random.Generator, lo: float, hi: float) -> np.ndarray:
    """随机缩放（以图像中心为准，缩放后裁剪/补零回原尺寸）。

    参数
    ----
    img : numpy.ndarray
        形状 ``(H, W)``，float32，取值 ``[0, 1]``。
    rng : numpy.random.Generator
        随机源。
    lo, hi : float
        缩放系数下界与上界（如 ``0.9, 1.1``）。

    返回
    ----
    numpy.ndarray
        形状 ``(H, W)``，float32。

    形状
    ----
    ``(H, W)`` -> ``(H, W)``
    """
    if hi <= 0 or (lo == 1.0 and hi == 1.0):
        return img
    k = float(rng.uniform(lo, hi))
    h, w = img.shape
    nh, nw = max(1, int(round(h * k))), max(1, int(round(w * k)))
    pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8), mode="L")
    pil = pil.resize((nw, nh), resample=Image.BILINEAR)
    arr = np.asarray(pil, dtype=np.float32) / 255.0

    out = np.zeros((h, w), dtype=np.float32)
    # 居中放置
    y0, x0 = (nh - h) // 2, (nw - w) // 2
    if nh >= h and nw >= w:
        out = arr[y0:y0 + h, x0:x0 + w]
    else:
        sy0, sx0 = max(0, -y0), max(0, -x0)
        out[sy0:sy0 + min(h, nh), sx0:sx0 + min(w, nw)] = \
            arr[:min(h, nh), :min(w, nw)]
    return out


def blur_gaussian(img: np.ndarray, rng: np.random.Generator, kernel: int) -> np.ndarray:
    """高斯模糊（对应 §2.6 表中"核 3×3 / 5×5"）。

    参数
    ----
    img : numpy.ndarray
        形状 ``(H, W)``，float32。
    rng : numpy.random.Generator
        随机源（用于在候选中挑选核）。
    kernel : int
        模糊半径（像素），建议 1（≈3×3）或 2（≈5×5）。

    返回
    ----
    numpy.ndarray
        形状 ``(H, W)``，float32。

    形状
    ----
    ``(H, W)`` -> ``(H, W)``
    """
    if kernel <= 0:
        return img
    radius = float(kernel)
    pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8), mode="L")
    pil = pil.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(pil, dtype=np.float32) / 255.0


def blur_motion(
    img: np.ndarray, rng: np.random.Generator, max_len: int = 5
) -> np.ndarray:
    """运动模糊（线性核，方向与长度随机，对应"随机方向长度"）。

    参数
    ----
    img : numpy.ndarray
        形状 ``(H, W)``，float32。
    rng : numpy.random.Generator
        随机源。
    max_len : int
        最长核长度；**PIL 只支持 3×3 与 5×5**，因此实际取 3 或 5。

    返回
    ----
    numpy.ndarray
        形状 ``(H, W)``，float32。

    形状
    ----
    ``(H, W)`` -> ``(H, W)``

    说明
    ----
    PIL ``ImageFilter.Kernel`` 仅支持 3×3 / 5×5 两种尺寸（见 Pillow 文档），
    因此这里在 {3, 5} 中随机取核长，而不是任意奇数。
    """
    # PIL 的 ImageFilter.Kernel 只接受 3×3 或 5×5，必须遵守
    choices = [3] if max_len < 5 else [3, 5]
    length = int(rng.choice(choices))
    angle = float(rng.uniform(0.0, 180.0))
    ker = np.zeros((length, length), dtype=np.float64)
    c = length // 2
    rad = np.deg2rad(angle)
    for t in range(-c, c + 1):
        x = int(round(c + t * np.cos(rad)))
        y = int(round(c + t * np.sin(rad)))
        if 0 <= x < length and 0 <= y < length:
            ker[y, x] = 1.0
    if ker.sum() == 0:
        ker[c, c] = 1.0
    ker /= ker.sum()

    pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8), mode="L")
    pil = pil.filter(ImageFilter.Kernel(
        size=(length, length), kernel=ker.flatten().tolist(), scale=1.0, offset=0.0
    ))
    return np.asarray(pil, dtype=np.float32) / 255.0


def noise_gaussian(
    img: np.ndarray, rng: np.random.Generator, sigma: float
) -> np.ndarray:
    """加性高斯噪声（§2.6：``σ ∈ [0, 0.02]``）。

    参数
    ----
    img : numpy.ndarray
        形状 ``(H, W)``，float32，取值 ``[0, 1]``。
    rng : numpy.random.Generator
        随机源。
    sigma : float
        噪声标准差（在 ``[0, 1]`` 量纲下）。

    返回
    ----
    numpy.ndarray
        形状 ``(H, W)``，float32，已裁剪到 ``[0, 1]``。

    形状
    ----
    ``(H, W)`` -> ``(H, W)``
    """
    if sigma <= 0:
        return img
    s = float(rng.uniform(0.0, sigma))
    noise = rng.normal(0.0, s, size=img.shape).astype(np.float32)
    return np.clip(img + noise, 0.0, 1.0)


def noise_salt_pepper(
    img: np.ndarray, rng: np.random.Generator, ratio: float
) -> np.ndarray:
    """椒盐噪声（§2.6：比例 0.1%–1%）。

    参数
    ----
    img : numpy.ndarray
        形状 ``(H, W)``，float32，取值 ``[0, 1]``。
    rng : numpy.random.Generator
        随机源。
    ratio : float
        噪声像素比例上界；实际比例在 ``[0, ratio]`` 均匀采样。

    返回
    ----
    numpy.ndarray
        形状 ``(H, W)``，float32。

    形状
    ----
    ``(H, W)`` -> ``(H, W)``
    """
    if ratio <= 0:
        return img
    r = float(rng.uniform(0.0, ratio))
    n = int(img.size * r)
    if n <= 0:
        return img
    out = img.copy()
    flat = out.reshape(-1)
    idx = rng.choice(flat.size, size=n, replace=False)
    half = n // 2
    flat[idx[:half]] = 0.0
    flat[idx[half:]] = 1.0
    return out


def photometric(
    img: np.ndarray,
    rng: np.random.Generator,
    brightness: tuple,
    contrast: tuple,
) -> np.ndarray:
    """亮度 / 对比度扰动。

    参数
    ----
    img : numpy.ndarray
        形状 ``(H, W)``，float32，取值 ``[0, 1]``。
    rng : numpy.random.Generator
        随机源。
    brightness : tuple of float
        增益区间 ``(lo, hi)``，对应 §2.6 的 ``×[0.6, 1.4]``。
    contrast : tuple of float
        对比度区间 ``(lo, hi)``，围绕图像均值缩放。

    返回
    ----
    numpy.ndarray
        形状 ``(H, W)``，float32，已裁剪到 ``[0, 1]``。

    形状
    ----
    ``(H, W)`` -> ``(H, W)``
    """
    out = img
    blo, bhi = float(brightness[0]), float(brightness[1])
    if not (blo == 1.0 and bhi == 1.0):
        out = out * float(rng.uniform(blo, bhi))
    clo, chi = float(contrast[0]), float(contrast[1])
    if not (clo == 1.0 and chi == 1.0):
        k = float(rng.uniform(clo, chi))
        m = float(out.mean())
        out = (out - m) * k + m
    return np.clip(out, 0.0, 1.0)


def jpeg_compress(
    img: np.ndarray, rng: np.random.Generator, quality_range: tuple
) -> np.ndarray:
    """JPEG 压缩失真（编码噪声）。

    参数
    ----
    img : numpy.ndarray
        形状 ``(H, W)``，float32，取值 ``[0, 1]``。
    rng : numpy.random.Generator
        随机源。
    quality_range : tuple of int
        质量区间 ``(lo, hi)``。

    返回
    ----
    numpy.ndarray
        形状 ``(H, W)``，float32。

    形状
    ----
    ``(H, W)`` -> ``(H, W)``
    """
    lo, hi = int(quality_range[0]), int(quality_range[1])
    q = int(rng.integers(lo, hi + 1))
    buf = io.BytesIO()
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8), mode="L").save(
        buf, format="JPEG", quality=q
    )
    buf.seek(0)
    with Image.open(buf) as im:
        return np.asarray(im.convert("L"), dtype=np.float32) / 255.0


# =============================================================================
# 2. 增强档位配置
# =============================================================================


@dataclass
class AugmentConfig:
    """一档增强的完整参数（对应 ``augmentation.levels.<档位名>``）。

    属性
    ----
    enabled : bool
        总开关；``False`` 时增强函数直接返回原图。
    rotation_deg, translate_px : float
        几何：旋转角上界、平移像素上界。
    scale_range : tuple of float
        缩放系数区间。
    gaussian_blur_prob, motion_blur_prob : float
        触发概率。
    gaussian_noise_prob, gaussian_noise_sigma : float
        高斯噪声触发概率与 σ 上界。
    salt_pepper_prob, salt_pepper_ratio : float
        椒盐触发概率与比例上界。
    brightness_range, contrast_range : tuple of float
        亮度、对比度区间。
    jpeg_prob : float
        JPEG 触发概率。
    jpeg_quality_range : tuple of int
        质量区间。
    name : str
        档位名（none / weak / strong），写入日志。
    """

    enabled: bool = True
    rotation_deg: float = 0.0
    translate_px: float = 0.0
    scale_range: tuple = (1.0, 1.0)
    gaussian_blur_prob: float = 0.0
    motion_blur_prob: float = 0.0
    gaussian_noise_prob: float = 0.0
    gaussian_noise_sigma: float = 0.0
    salt_pepper_prob: float = 0.0
    salt_pepper_ratio: float = 0.0
    brightness_range: tuple = (1.0, 1.0)
    contrast_range: tuple = (1.0, 1.0)
    jpeg_prob: float = 0.0
    jpeg_quality_range: tuple = (95, 100)
    name: str = "weak"

    @classmethod
    def from_config(cls, cfg, level: str) -> "AugmentConfig":
        """从全局配置构造指定档位。

        参数
        ----
        cfg : models.config.Config
            全局配置。
        level : str
            档位名（``none`` / ``weak`` / ``strong``）。

        返回
        返回
        ----
        AugmentConfig

        异常
        ------
        KeyError
            配置中不存在该档位。
        """
        levels = cfg.augmentation.levels
        if level not in levels:
            raise KeyError(
                f"augmentation.levels 中没有档位 {level!r}；"
                f"可选：{list(levels.keys())}"
            )
        d = levels[level]
        return cls(
            enabled=bool(d.get("enabled", True)),
            rotation_deg=float(d.get("rotation_deg", 0.0)),
            translate_px=float(d.get("translate_px", 0.0)),
            scale_range=tuple(float(v) for v in d.get("scale_range", (1.0, 1.0))),
            gaussian_blur_prob=float(d.get("gaussian_blur_prob", 0.0)),
            motion_blur_prob=float(d.get("motion_blur_prob", 0.0)),
            gaussian_noise_prob=float(d.get("gaussian_noise_prob", 0.0)),
            gaussian_noise_sigma=float(d.get("gaussian_noise_sigma", 0.0)),
            salt_pepper_prob=float(d.get("salt_pepper_prob", 0.0)),
            salt_pepper_ratio=float(d.get("salt_pepper_ratio", 0.0)),
            brightness_range=tuple(float(v) for v in d.get("brightness_range", (1.0, 1.0))),
            contrast_range=tuple(float(v) for v in d.get("contrast_range", (1.0, 1.0))),
            jpeg_prob=float(d.get("jpeg_prob", 0.0)),
            jpeg_quality_range=tuple(int(v) for v in d.get("jpeg_quality_range", (95, 100))),
            name=str(level),
        )

    def as_dict(self) -> Dict[str, Any]:
        """转成可写入日志的字典。"""
        return {
            "level": self.name, "enabled": self.enabled,
            "rotation_deg": self.rotation_deg, "translate_px": self.translate_px,
            "scale_range": list(self.scale_range),
            "gaussian_blur_prob": self.gaussian_blur_prob,
            "motion_blur_prob": self.motion_blur_prob,
            "gaussian_noise_prob": self.gaussian_noise_prob,
            "gaussian_noise_sigma": self.gaussian_noise_sigma,
            "salt_pepper_prob": self.salt_pepper_prob,
            "salt_pepper_ratio": self.salt_pepper_ratio,
            "brightness_range": list(self.brightness_range),
            "contrast_range": list(self.contrast_range),
            "jpeg_prob": self.jpeg_prob,
            "jpeg_quality_range": list(self.jpeg_quality_range),
        }


def apply_augmentation(
    img: np.ndarray, acfg: AugmentConfig, rng: np.random.Generator
) -> np.ndarray:
    """对单张图像按档位施加随机增强。

    执行顺序：几何（缩放 → 旋转 → 平移）→ 成像（模糊 → 噪声 → 亮度对比度）
    → 编码（JPEG）。顺序固定以保证同一档位在不同运行中行为一致。

    参数
    ----
    img : numpy.ndarray
        形状 ``(H, W)``，float32，取值 ``[0, 1]``。
    acfg : AugmentConfig
        增强档位。
    rng : numpy.random.Generator
        随机源。

    返回
    ----
    numpy.ndarray
        形状 ``(H, W)``，float32，取值 ``[0, 1]``。

    形状
    ----
    ``(H, W)`` -> ``(H, W)``
    """
    if not acfg.enabled:
        return np.clip(img, 0.0, 1.0)

    out = img.astype(np.float32, copy=True)

    # ---- 几何 ----
    out = geom_scale(out, rng, acfg.scale_range[0], acfg.scale_range[1])
    out = geom_rotate(out, rng, acfg.rotation_deg)
    out = geom_translate(out, rng, acfg.translate_px)

    # ---- 成像 ----
    if acfg.gaussian_blur_prob > 0 and rng.random() < acfg.gaussian_blur_prob:
        out = blur_gaussian(out, rng, int(rng.integers(1, 3)))
    if acfg.motion_blur_prob > 0 and rng.random() < acfg.motion_blur_prob:
        out = blur_motion(out, rng)
    if acfg.gaussian_noise_prob > 0 and rng.random() < acfg.gaussian_noise_prob:
        out = noise_gaussian(out, rng, acfg.gaussian_noise_sigma)
    if acfg.salt_pepper_prob > 0 and rng.random() < acfg.salt_pepper_prob:
        out = noise_salt_pepper(out, rng, acfg.salt_pepper_ratio)
    out = photometric(out, rng, acfg.brightness_range, acfg.contrast_range)

    # ---- 编码 ----
    if acfg.jpeg_prob > 0 and rng.random() < acfg.jpeg_prob:
        out = jpeg_compress(out, rng, acfg.jpeg_quality_range)

    return np.clip(out, 0.0, 1.0)


def build_augment_fn(acfg: AugmentConfig) -> Callable[[np.ndarray, np.random.Generator], np.ndarray]:
    """构造可直接传给 :class:`models.dataset.PlateDataset` 的增强函数。

    参数
    ----
    acfg : AugmentConfig
        增强档位。

    返回
    ----
    callable
        签名 ``(img (H,W), rng) -> (H,W)``。

    形状
    ----
    ``(H, W)`` -> ``(H, W)``
    """
    def _fn(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return apply_augmentation(img, acfg, rng)

    return _fn


# =============================================================================
# 3. 调试入口
# =============================================================================

if __name__ == "__main__":  # pragma: no cover
    from models.config import load_config
    from models.dataset import load_cache
    from evaluate.visualize import plot_check_grid

    cfg = load_config()
    tag = f"ccpd_{int(cfg.ccpd.input_size[0])}x{int(cfg.ccpd.input_size[1])}"
    cache = Path(cfg.paths.processed_dir) / f"{tag}.npz"
    if not cache.exists():
        raise SystemExit(f"缺少缓存 {cache}，请先跑 train/prepare_data.py")

    images, labels, _ = load_cache(cache)
    pick = np.arange(min(8, len(images)))
    imgs = images[pick].astype(np.float32) / 255.0
    labs = labels[pick]

    for level in ("none", "weak", "strong"):
        acfg = AugmentConfig.from_config(cfg, level)
        rng = np.random.default_rng(1234)
        aug = np.stack([apply_augmentation(im, acfg, rng) for im in imgs])
        plot_check_grid(
            aug, labs, Path(f"reports/figs/_aug_{level}.png"),
            title=f"数据增强档位：{level}", n=8, seed=0,
        )
        print(f"档位 {level:6s} 完成：均值 {aug.mean():.4f} 标准差 {aug.std():.4f}")
    print("增强自检通过，图见 reports/figs/_aug_*.png")