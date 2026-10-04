# -*- coding: utf-8 -*-
"""合成域测试集生成（§2.4）。

定位
----
本模块生成的是**仿真中国车牌图像**，用于检验「真实训练 → 合成测试」的跨域泛化。
按规格：**只测试、不训练**，生成种子固定、参数全部记录（§2.4 要求 1）。

几何一致性（关键）
------------------
为了让我们测的是"**外观**跨域差距"而不是"**几何**处理差异"，合成图与 CCPD 真实图
走**完全相同的几何管线**：

1. 先在 ``rectify_size``（基线 168×52）上渲染完整 7 位车牌
   （首位为省份汉字，同样渲染，以便它被同样地裁掉）；
2. 再调用 :func:`models.ccpd_parse.PrepParams` 定义的同一裁剪比例
   （裁掉左侧 1/7）与同一缩放尺寸（128×32）；
3. 转灰度、缩放到 ``input_size``，输出 uint8 ``(32, 128)``。

这样合成域与真实域之间只剩「成像风格」差异：字体、底色、噪声、笔画粗细、
字符间距等，正是 §1.3 第 3 个问题要考察的对象。

两种后端
--------
* ``pil_renderer``（默认）：本仓库内置的确定性渲染器，依赖仅 Pillow + 系统字体；
* ``generator_repo``：外部仓库
  `Nenger/chinese_licence_plate_generator <https://github.com/Nenger/chinese_licence_plate_generator>`_
  的整图输出。注意它产出的是"车牌贴在背景世界图里"的整图，**必须再做检测切分**
  才能得到 32×128 输入；本项目不做检测（§0.3），若用它则需自行提供坐标切分。
  因此默认后端为内置渲染器，``generator_repo`` 仅作为可选扩展保留。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# --- 包引导：支持直接运行本文件 ---------------------------------------------
import sys as _sys

if __package__ in (None, ""):
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.ccpd_parse import PrepParams
from models.charset import (
    CCPD_PROVINCES,
    DIGITS,
    LETTERS,
    NUM_CLASSES,
    SEQ_LEN,
    check_label_legal,
    encode_label,
)

# =============================================================================
# 1. 配置对象
# =============================================================================


@dataclass
class PlateStyle:
    """一种车牌版式（底色 / 字色 / 噪声强度）。

    属性
    ----
    name : str
        版式名，写入 manifest 便于分层分析。
    bg : tuple of int
        背景 RGB。
    fg : tuple of int
        字符 RGB。
    noise : float
        高斯噪声标准差（0–255 量纲）。
    """

    name: str
    bg: Tuple[int, int, int]
    fg: Tuple[int, int, int]
    noise: float = 0.0


@dataclass
class SynthConfig:
    """合成域测试集生成配置（从 ``configs/default.yaml`` 的 ``synth`` 段构造）。

    属性
    ----
    seed : int
        固定随机种子。
    width, height : int
        最终输出尺寸（128×32）。
    rectify_size : tuple of int
        渲染尺寸 ``(168, 52)``，与真实管线一致。
    keep_right_fraction : float
        裁掉首位汉字的比例，与真实管线一致。
    fonts : list of str
        可用字体文件路径。
    styles : list of PlateStyle
        版式列表。
    jitter, rotation_deg, blur_sigma : float
        成像扰动参数。
    jpeg_quality : int or None
        JPEG 压缩质量；``None`` 表示不压缩。
    province_chars : list of str
        可选省份汉字（不参与识别，仅用于渲染）。
    """

    seed: int = 2024
    width: int = 128
    height: int = 32
    rectify_size: Tuple[int, int] = (168, 52)
    keep_right_fraction: float = 6.0 / 7.0
    fonts: List[str] = field(default_factory=list)
    styles: List[PlateStyle] = field(default_factory=list)
    jitter: float = 1.0
    rotation_deg: float = 0.0
    blur_sigma: float = 0.0
    jpeg_quality: Optional[int] = None
    province_chars: List[str] = field(default_factory=list)

    @classmethod
    def from_config(cls, cfg) -> "SynthConfig":
        """从全局配置构造。

        参数
        ----
        cfg : models.config.Config
            全局配置。

        返回
        ----
        SynthConfig
        """
        styles = [
            PlateStyle(
                name=str(s["name"]),
                bg=tuple(int(v) for v in s["bg"]),
                fg=tuple(int(v) for v in s["fg"]),
                noise=float(s.get("noise", 0.0)),
            )
            for s in cfg.synth.plate_styles
        ]
        return cls(
            seed=int(cfg.synth.seed),
            width=int(cfg.synth.width),
            height=int(cfg.synth.height),
            rectify_size=(
                int(cfg.ccpd.rectify_size[0]),
                int(cfg.ccpd.rectify_size[1]),
            ),
            keep_right_fraction=float(cfg.ccpd.keep_right_fraction),
            fonts=[str(f) for f in cfg.synth.fonts],
            styles=styles,
            jitter=float(cfg.synth.jitter),
            rotation_deg=float(cfg.synth.rotation_deg),
            blur_sigma=float(cfg.synth.blur_sigma),
            jpeg_quality=cfg.synth.jpeg_quality,
            province_chars=[str(p) for p in cfg.synth.get("province_chars", [])],
        )

    def as_dict(self) -> Dict[str, object]:
        """转成可写入日志 / manifest 的字典（记录生成参数，§2.4 要求 1）。"""
        return {
            "seed": self.seed,
            "width": self.width,
            "height": self.height,
            "rectify_size": list(self.rectify_size),
            "keep_right_fraction": self.keep_right_fraction,
            "fonts": [Path(f).name for f in self.fonts],
            "styles": [s.name for s in self.styles],
            "jitter": self.jitter,
            "rotation_deg": self.rotation_deg,
            "blur_sigma": self.blur_sigma,
            "jpeg_quality": self.jpeg_quality,
        }


# =============================================================================
# 2. 标签采样
# =============================================================================


def sample_label(rng: np.random.Generator, positions: Optional[Sequence[int]] = None) -> str:
    """随机采样一条合法的六位标签。

    参数
    ----
    rng : numpy.random.Generator
        随机源。
    positions : Sequence[int] or None
        各位置类别数；``None`` 表示全 34 类。若某位置为 24（仅字母），
        该位置只从字母表采样。

    返回
    ----
    str
        长度 6 的合法标签。

    形状
    ----
    无 -> ``str``
    """
    chars: List[str] = []
    for pos in range(SEQ_LEN):
        if positions is not None and int(positions[pos]) == len(LETTERS):
            pool = LETTERS
        else:
            # 真实车牌后 5 位字母与数字混排，这里据此调整采样比例
            pool = LETTERS if (pos == 0 or rng.random() < 0.45) else DIGITS
        chars.append(pool[int(rng.integers(0, len(pool)))])
    label = "".join(chars)
    ok, _ = check_label_legal(label, positions)
    assert ok, f"采样出非法标签：{label}"
    return label


# =============================================================================
# 3. 单张车牌渲染
# =============================================================================


def _load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    """加载字体，失败时回退到 Pillow 默认字体。

    参数
    ----
    path : str
        字体文件路径。
    size : int
        字号（像素）。

    返回
    ----
    PIL.ImageFont.FreeTypeFont
        字体对象。
    """
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def render_plate(
    label: str,
    style: PlateStyle,
    cfg: SynthConfig,
    rng: np.random.Generator,
    province: str = "京",
) -> Image.Image:
    """在 ``cfg.rectify_size`` 上渲染一张完整 7 位车牌（含省份汉字）。

    参数
    ----
    label : str
        六位标签（不含省份）。
    style : PlateStyle
        版式。
    cfg : SynthConfig
        生成配置。
    rng : numpy.random.Generator
        随机源。
    province : str
        省份汉字，仅用于渲染，随后会被裁掉。

    返回
    ----
    PIL.Image.Image
        RGB 图像，尺寸 ``cfg.rectify_size``。

    形状
    ----
    ``str`` -> ``(H, W, 3)``
    """
    w, h = int(cfg.rectify_size[0]), int(cfg.rectify_size[1])
    img = Image.new("RGB", (w, h), color=style.bg)
    draw = ImageDraw.Draw(img)

    # --- 外框（真实车牌有一圈白/黑边框） -----------------------------------
    border = 1 if style.fg[0] > 128 else 1
    draw.rectangle([0, 0, w - 1, h - 1], outline=style.fg, width=border)

    # --- 版式参数 ---------------------------------------------------------
    # 7 个字符：首位省份占 1/7，其余 6 位平分剩余宽度
    full_seq = province + label
    n = len(full_seq)
    cell_w = w / n
    # 字号略小于格宽，留出间距
    font_size = int(min(cell_w * 1.35, (h - 4) * 1.15))
    font_size = max(8, font_size)

    for i, ch in enumerate(full_seq):
        font = _load_font(
            cfg.fonts[int(rng.integers(0, len(cfg.fonts)))] if cfg.fonts else "",
            font_size,
        )
        # 量测字符尺寸以便居中
        try:
            bbox = draw.textbbox((0, 0), ch, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            ox, oy = bbox[0], bbox[1]
        except Exception:
            tw, th, ox, oy = font_size * 0.6, font_size * 0.8, 0, 0

        # 抖动：模拟字符位置与真实车牌的细微差异
        jx = float(rng.uniform(-cfg.jitter, cfg.jitter))
        jy = float(rng.uniform(-cfg.jitter, cfg.jitter))
        x = cell_w * i + (cell_w - tw) / 2.0 - ox + jx
        y = (h - th) / 2.0 - oy + jy
        draw.text((x, y), ch, font=font, fill=style.fg)

    return img


def augment_synth(
    img: Image.Image,
    cfg: SynthConfig,
    rng: np.random.Generator,
) -> Image.Image:
    """对合成车牌施加轻微成像扰动（可配置为关闭）。

    参数
    ----
    img : PIL.Image.Image
        输入图像。
    cfg : SynthConfig
        生成配置。
    rng : numpy.random.Generator
        随机源。

    返回
    ----
    PIL.Image.Image
        处理后的图像。

    形状
    ----
    ``(H, W, 3)`` -> ``(H, W, 3)``
    """
    if cfg.rotation_deg and cfg.rotation_deg > 0:
        ang = float(rng.uniform(-cfg.rotation_deg, cfg.rotation_deg))
        img = img.rotate(ang, resample=Image.BICUBIC, expand=False,
                         fillcolor=img.getpixel((0, 0)))
    if cfg.blur_sigma and cfg.blur_sigma > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=float(cfg.blur_sigma)))
    if cfg.jpeg_quality is not None:
        import io

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(cfg.jpeg_quality))
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    return img


def synth_one(
    label: str,
    style: PlateStyle,
    cfg: SynthConfig,
    rng: np.random.Generator,
    params: PrepParams,
) -> Image.Image:
    """生成一张与真实管线几何一致的 32×128 灰度合成车牌。

    参数
    ----
    label : str
        六位标签。
    style : PlateStyle
        版式。
    cfg : SynthConfig
        生成配置。
    rng : numpy.random.Generator
        随机源。
    params : PrepParams
        真实管线参数（提供裁剪比例与输出尺寸，保证几何一致）。

    返回
    ----
    PIL.Image.Image
        灰度图像，尺寸 ``(params.input_size[1], params.input_size[0])``。

    形状
    ----
    ``str`` -> ``(32, 128)`` 灰度
    """
    province = cfg.province_chars[int(rng.integers(0, len(cfg.province_chars)))] \
        if cfg.province_chars else "京"
    img = render_plate(label, style, cfg, rng, province=province)
    img = augment_synth(img, cfg, rng)

    # --- 与真实管线完全相同的几何处理 -------------------------------------
    w, h = img.size
    x0 = int(round(w * (1.0 - params.keep_right_fraction)))
    if x0 > 0:
        img = img.crop((x0, 0, w, h))
    img = img.convert("L").resize(params.input_size, resample=Image.BILINEAR)
    return img


def add_noise(arr: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """给灰度数组加高斯噪声。

    参数
    ----
    arr : numpy.ndarray
        形状 ``(H, W)``，float32，取值 ``[0, 255]``。
    sigma : float
        噪声标准差。
    rng : numpy.random.Generator
        随机源。

    返回
    ----
    numpy.ndarray
        同形状，已裁剪到 ``[0, 255]``。

    形状
    ----
    ``(H, W)`` -> ``(H, W)``
    """
    if sigma <= 0:
        return arr
    noise = rng.normal(0.0, sigma, size=arr.shape).astype(np.float32)
    return np.clip(arr + noise, 0.0, 255.0)


# =============================================================================
# 4. 批量生成
# =============================================================================


@dataclass
class SynthResult:
    """合成域测试集生成结果。

    属性
    ----
    images : numpy.ndarray
        形状 ``(N, H, W)``，uint8。
    labels : numpy.ndarray
        形状 ``(N, 6)``，int64。
    texts : list of str
        标签字符串。
    style_names : list of str
        每张图的版式名。
    """

    images: np.ndarray
    labels: np.ndarray
    texts: List[str]
    style_names: List[str]


def generate_dataset(
    n: int,
    cfg: SynthConfig,
    params: PrepParams,
    out_dir: Optional[Path] = None,
    positions: Optional[Sequence[int]] = None,
    verbose: bool = True,
) -> SynthResult:
    """批量生成合成域测试集。

    参数
    ----
    n : int
        生成张数。
    cfg : SynthConfig
        生成配置（含固定种子）。
    params : PrepParams
        真实管线参数。
    out_dir : Path or None
        若给出，则把每张图另存为 JPEG 以便人工查看（不入库）。
    positions : Sequence[int] or None
        位置约束。
    verbose : bool
        是否打印进度。

    返回
    ----
    SynthResult
        图像 ``(N, H, W)`` uint8、标签 ``(N, 6)`` int64、文本与版式名。

    形状
    ----
    ``int`` -> ``(N, 32, 128) uint8`` + ``(N, 6) int64``
    """
    rng = np.random.default_rng(int(cfg.seed))
    if not cfg.fonts:
        raise ValueError("synth.fonts 为空，无法渲染合成车牌")
    if not cfg.styles:
        raise ValueError("synth.plate_styles 为空，无法渲染合成车牌")

    images: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    texts: List[str] = []
    style_names: List[str] = []

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(int(n)):
        style = cfg.styles[int(rng.integers(0, len(cfg.styles)))]
        label = sample_label(rng, positions)
        img = synth_one(label, style, cfg, rng, params)

        arr = np.asarray(img, dtype=np.float32)
        arr = add_noise(arr, style.noise, rng)
        arr_u8 = np.clip(arr, 0, 255).astype(np.uint8)

        images.append(arr_u8)
        labels.append(encode_label(label))
        texts.append(label)
        style_names.append(style.name)

        if out_dir is not None:
            Image.fromarray(arr_u8, mode="L").save(
                out_dir / f"{i:06d}_{style.name}_{label}.jpg", quality=95
            )

        if verbose and ((i + 1) % 500 == 0 or i + 1 == n):
            print(f"  [synth] {i + 1}/{n}", flush=True)

    h, w = int(params.input_size[1]), int(params.input_size[0])
    if not images:
        return SynthResult(np.zeros((0, h, w), dtype=np.uint8),
                           np.zeros((0, SEQ_LEN), dtype=np.int64), [], [])
    return SynthResult(
        images=np.stack(images, axis=0),
        labels=np.stack(labels, axis=0),
        texts=texts,
        style_names=style_names,
    )


# =============================================================================
# 5. 调试入口
# =============================================================================

if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="合成域测试集生成（§2.4）")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", type=str, default="data/synth_test_preview")
    args = ap.parse_args()

    from models.config import ensure_dirs, load_config
    from models.dataset import BuildResult, save_cache
    from evaluate.visualize import plot_check_grid

    cfg_all = load_config()
    ensure_dirs(cfg_all)
    params = PrepParams.from_config(cfg_all)
    scfg = SynthConfig.from_config(cfg_all)
    print("合成配置：", json.dumps(scfg.as_dict(), ensure_ascii=False))

    res = generate_dataset(args.n, scfg, params, out_dir=Path(args.out))
    print("图像数组：", res.images.shape, res.images.dtype)
    print("标签数组：", res.labels.shape, res.labels.dtype)
    print("前 8 个标签：", res.texts[:8])
    print("版式分布：", {s: res.style_names.count(s) for s in set(res.style_names)})
    # 标签合法性
    for t in res.texts:
        ok, bad = check_label_legal(t, None)
        assert ok, (t, bad)
    print("标签全部合法（34 类内）")

    plot_check_grid(
        res.images, res.labels, Path("reports/figs/synth_check_grid.png"),
        title="合成域测试集人工核对（§2.4）", n=min(args.n, 20), seed=scfg.seed,
    )