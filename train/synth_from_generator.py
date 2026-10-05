# -*- coding: utf-8 -*-
"""合成域测试集后端 ``generator_repo``：Nenger/chinese_licence_plate_generator。

定位
----
本模块把外部开源仓库
`Nenger/chinese_licence_plate_generator <https://github.com/Nenger/chinese_licence_plate_generator>`_
（本仓库固定 commit ``43bac43``，克隆于 ``data/external/``，不入库）作为
**合成域测试集**的唯一来源，替代早期版本的内置 PIL 渲染器。按规格：合成域
**只测试、不训练**，生成种子固定、参数全部记录（§2.4 要求 1）。

为什么用牌面级输出而不是场景整图
--------------------------------
该仓库的主打产物是"车牌贴进街景图"的**检测**数据集（``main.py``），而本项目
任务定义（§0.3）是"输入已裁剪、已对齐的车牌区域，**不做检测**"。因此本模块
直接调用其 ``FakePlateGenerator.generate_one_plate()`` 的**牌面级**输出——
这同样是该仓库自带的用法（见上游 ``fake_plate_generator.py`` 的 ``__main__``）：

1. 以本项目的 ``rectify_size``（168×52）渲染完整 7 位牌面
   （首位省份汉字照常渲染，随后被同样地裁掉）；
2. 依次施加该仓库自带的成像扰动链：``jittering_color`` → ``add_noise``
   → ``jittering_blur`` → ``jittering_scale``（与其示例完全一致的顺序，
   ``jittering_scale`` 先缩后放回原尺寸，只损失清晰度、不改变几何）；
3. 走**与 CCPD 完全相同的几何管线**：裁掉左侧 1/7（``PrepParams.crop_x0``）
   → 缩放为 ``input_size``（128×32）→ 转灰度，输出 uint8 ``(32, 128)``。

这样合成域与真实域之间只剩「成像风格」差异：模板字体、描边、蓝底质感、
噪声与模糊链路，正是跨域泛化要考察的对象。

标签合法性（关键）
------------------
上游字符素材 ``fake_resource/letters/`` **包含 i.png / o.png**，而本项目
34 类字符集排除 I、O（§2.1）。因此逐张校验：``plate_name[2:]`` 大写化后
必须通过 :func:`models.charset.check_label_legal`，否则**拒绝重采**。
拒绝采样只改变随机流的消耗数量，不改变合法字符上的分布；给定种子即可复现。

复现性说明
----------
* 上游生成器使用 stdlib ``random`` 与 ``np.random``，本模块同时播种两者；
* 字符/模板素材按 ``os.listdir`` 顺序加载，同一台机器上顺序稳定；
  跨机器复现请以固定的 commit 号为准（记录于 ``split_summary.json``
  与 ``manifest.csv`` 的 ``source_file`` 字段）；
* 上游依赖 **OpenCV（cv2）**：cv2 只出现在本数据生成环节，模型、训练、
  评测与 CCPD 预处理管线均不依赖 OpenCV。
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from PIL import Image

# --- 包引导：支持直接运行本文件 ---------------------------------------------
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    # 直接运行 train/ 下脚本时 sys.path[0] 是 train/，那里的 train.py 会以顶层
    # 模块身份遮蔽同名 train 包，导致 `from train.xxx import ...` 失败，因此把
    # 脚本自身目录从 sys.path 中移除（项目根已插到最前，models 仍可导入）。
    _here = str(Path(__file__).resolve().parent)
    while _here in sys.path:
        sys.path.remove(_here)

from models.ccpd_parse import PrepParams
from models.charset import SEQ_LEN, check_label_legal, encode_label
from train.synth_plates import SynthConfig, SynthResult

#: 上游仓库固定版本（克隆后剥离 .git，版本以此为准）
GENERATOR_COMMIT = "43bac435127a0270143994f980e2a997561dafea"

#: 默认克隆位置（相对项目根；可用 synth.generator_dir 覆盖）
DEFAULT_GENERATOR_DIR = "data/external/chinese_licence_plate_generator"


def resolve_generator_dir(cfg=None, override: Optional[str] = None) -> Path:
    """定位生成器仓库目录。

    参数
    ----
    cfg : models.config.Config or None
        全局配置；若提供且含 ``synth.generator_dir``，优先采用。
    override : str or None
        命令行显式覆盖。

    返回
    ----
    Path
        生成器仓库根目录（含 ``fake_resource/`` 的那一级）。

    异常
    ------
    FileNotFoundError
        目录或 ``fake_resource/`` 不存在时抛出，并给出克隆命令。
    """
    rel = override
    if rel is None and cfg is not None:
        try:
            rel = str(cfg.synth.get("generator_dir", DEFAULT_GENERATOR_DIR))
        except AttributeError:
            rel = DEFAULT_GENERATOR_DIR
    if rel is None:
        rel = DEFAULT_GENERATOR_DIR
    path = Path(rel)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path
    if not (path / "fake_resource").is_dir():
        raise FileNotFoundError(
            f"找不到生成器资源目录：{path}\n"
            f"请先克隆：git clone https://github.com/Nenger/"
            f"chinese_licence_plate_generator {rel}\n"
            f"固定 commit：{GENERATOR_COMMIT}"
        )
    return path


def _patch_cv2_imread_unicode() -> None:
    """让上游的 ``cv2.imread`` 支持非 ASCII 路径（本工作区路径含中文）。

    OpenCV 在 Windows 上的 ``cv2.imread`` 无法读取含非 ASCII 字符的路径，
    会静默返回 ``None``。这里**不改动上游代码**，而是在导入前给 ``cv2.imread``
    包一层：原调用失败时回退到 ``np.fromfile`` + ``cv2.imdecode``。
    """
    import cv2

    if getattr(cv2, "_projectx_unicode_patch", False):
        return
    orig_imread = cv2.imread

    def _imread_unicode(path, flags=-1):  # noqa: ANN001
        # 非 ASCII 路径直接走 imdecode，避免 cv2 先失败一次并刷告警日志
        if any(ord(c) > 127 for c in str(path)):
            data = np.fromfile(str(path), dtype=np.uint8)
            return cv2.imdecode(data, flags) if data.size else None
        img = orig_imread(path, flags)
        if img is None:
            data = np.fromfile(str(path), dtype=np.uint8)
            img = cv2.imdecode(data, flags) if data.size else None
        return img

    cv2.imread = _imread_unicode
    cv2._projectx_unicode_patch = True


def _import_generator_modules(gen_dir: Path):
    """把上游仓库目录加入 ``sys.path`` 并导入其模块。

    参数
    ----
    gen_dir : Path
        生成器仓库根目录。

    返回
    ----
    tuple
        ``(FakePlateGenerator 类, jittering_methods 模块, img_utils 模块)``。
    """
    _patch_cv2_imread_unicode()
    gdir = str(gen_dir)
    if gdir not in sys.path:
        sys.path.insert(0, gdir)
    import fake_plate_generator as fpg  # type: ignore  # 上游仓库模块
    import img_utils as iu  # type: ignore
    import jittering_methods as jm  # type: ignore

    return fpg.FakePlateGenerator, jm, iu


def _repo_plate_to_input(img_bgr: np.ndarray, params: PrepParams) -> Image.Image:
    """上游 BGR 牌面 -> 与真实管线同口径的灰度输入图。

    步骤与 :func:`train.synth_plates.synth_one` 的几何部分逐行对应：
    裁掉左侧首位汉字区域（``crop_x0``）→ 缩放为 ``input_size`` → 转灰度。

    参数
    ----
    img_bgr : numpy.ndarray
        上游输出的 BGR 彩色牌面，尺寸 ``params.rectify_size``。
    params : PrepParams
        真实管线参数。

    返回
    ----
    PIL.Image.Image
        灰度图像，尺寸 ``(params.input_size[0], params.input_size[1])``。

    形状
    ----
    ``(52, 168, 3) BGR`` -> ``(32, 128) 灰度``
    """
    rgb = img_bgr[:, :, ::-1]  # BGR -> RGB（不依赖 cv2 的颜色转换，少一处口径分歧）
    img = Image.fromarray(rgb, mode="RGB")
    w, h = img.size
    x0 = int(round(w * (1.0 - params.keep_right_fraction)))
    if x0 > 0:
        img = img.crop((x0, 0, w, h))
    img = img.convert("L").resize(params.input_size, resample=Image.BILINEAR)
    return img


def generate_dataset_from_repo(
    n: int,
    cfg: SynthConfig,
    params: PrepParams,
    generator_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    positions: Optional[Sequence[int]] = None,
    verbose: bool = True,
) -> SynthResult:
    """用外部生成器批量生成合成域测试集（牌面级）。

    参数
    ----
    n : int
        生成张数（拒绝重采不计入）。
    cfg : SynthConfig
        合成配置（使用其 ``seed``；其余渲染参数属于内置渲染器，这里忽略）。
    params : PrepParams
        真实管线参数（``rectify_size`` 作为牌面渲染尺寸，
        ``input_size`` / ``keep_right_fraction`` 决定最终几何）。
    generator_dir : Path or None
        生成器仓库目录；``None`` 用默认位置。
    out_dir : Path or None
        若给出，每张另存 JPEG 供人工查看（不入库）。
    positions : Sequence[int] or None
        位置类别约束（首位仅字母时；上游构造天然满足）。
    verbose : bool
        是否打印进度。

    返回
    ----
    SynthResult
        图像 ``(N, H, W)`` uint8、标签 ``(N, 6)`` int64、文本与版式名
        （版式名统一为 ``nenger_blue``：上游模板只有蓝底一种）。

    形状
    ----
    ``int`` -> ``(N, 32, 128) uint8`` + ``(N, 6) int64``
    """
    gen_dir = Path(generator_dir) if generator_dir is not None else resolve_generator_dir()
    FakePlateGenerator, jm, iu = _import_generator_modules(gen_dir)

    # 上游同时使用 stdlib random 与 np.random，两者都播种才能保证复现
    random.seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    # 牌面直接渲染为 rectify_size（168×52），与真实管线几何一致
    plate_size = (int(params.rectify_size[0]), int(params.rectify_size[1]))
    gen = FakePlateGenerator(str(gen_dir / "fake_resource") + "/", plate_size)

    images: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    texts: List[str] = []
    n_rejected = 0

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    while len(images) < int(n):
        plate_bgr, plate_name = gen.generate_one_plate()
        # plate_name = 两位省份编号 + 6 位字符（小写字母/数字）
        label = str(plate_name[2:]).upper()
        if len(label) != SEQ_LEN:
            n_rejected += 1
            continue
        ok, _ = check_label_legal(label, positions)
        if not ok:
            # 含 I / O（上游素材包含这两个字母，但 34 类字符集不含）
            n_rejected += 1
            continue

        # 上游示例的扰动链（fake_plate_generator.py __main__ 原顺序）
        plate_bgr = jm.jittering_color(plate_bgr)
        plate_bgr = iu.add_noise(plate_bgr)
        plate_bgr = jm.jittering_blur(plate_bgr)
        plate_bgr = jm.jittering_scale(plate_bgr)

        img = _repo_plate_to_input(plate_bgr, params)
        arr_u8 = np.asarray(img, dtype=np.uint8)

        images.append(arr_u8)
        labels.append(encode_label(label))
        texts.append(label)

        if out_dir is not None:
            Image.fromarray(arr_u8, mode="L").save(
                out_dir / f"{len(images) - 1:06d}_nenger_blue_{label}.jpg",
                quality=95,
            )

        if verbose and (len(images) % 500 == 0):
            print(f"  [synth/repo] {len(images)}/{n}（拒绝 {n_rejected} 张）",
                  flush=True)

    if verbose:
        print(f"  [synth/repo] 完成 {len(images)} 张，拒绝 {n_rejected} 张"
              f"（I/O 或长度不合法）")

    h, w = int(params.input_size[1]), int(params.input_size[0])
    return SynthResult(
        images=np.stack(images, axis=0) if images else
        np.zeros((0, h, w), dtype=np.uint8),
        labels=np.stack(labels, axis=0) if labels else
        np.zeros((0, SEQ_LEN), dtype=np.int64),
        texts=texts,
        style_names=["nenger_blue"] * len(texts),
    )


def provenance_dict(cfg: SynthConfig) -> dict:
    """写入划分文件 / manifest 的溯源信息（§2.4 要求 1）。"""
    return {
        "backend": "generator_repo",
        "generator_repo": "https://github.com/Nenger/chinese_licence_plate_generator",
        "generator_commit": GENERATOR_COMMIT,
        "generator_level": "plate",  # 牌面级输出，非场景整图
        "seed": int(cfg.seed),
        "rectify_size": list(cfg.rectify_size),
        "keep_right_fraction": cfg.keep_right_fraction,
        "jitter_chain": ["jittering_color", "add_noise",
                         "jittering_blur", "jittering_scale"],
        "io_rejection": True,
    }


# =============================================================================
# 调试入口：预览 + 人工核对网格
# =============================================================================

if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(
        description="合成域测试集生成（generator_repo 后端，§2.4）")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", type=str, default="data/synth_test_preview")
    args = ap.parse_args()

    from models.config import ensure_dirs, load_config
    from evaluate.visualize import plot_check_grid

    cfg_all = load_config()
    ensure_dirs(cfg_all)
    params = PrepParams.from_config(cfg_all)
    scfg = SynthConfig.from_config(cfg_all)
    gen_dir = resolve_generator_dir(cfg_all)
    print("生成器目录：", gen_dir)
    print("溯源信息：", json.dumps(provenance_dict(scfg), ensure_ascii=False))

    res = generate_dataset_from_repo(
        args.n, scfg, params, generator_dir=gen_dir, out_dir=Path(args.out))
    print("图像数组：", res.images.shape, res.images.dtype)
    print("标签数组：", res.labels.shape, res.labels.dtype)
    print("前 8 个标签：", res.texts[:8])
    for t in res.texts:
        ok, bad = check_label_legal(t, None)
        assert ok, (t, bad)
    print("标签全部合法（34 类内，无 I/O）")

    plot_check_grid(
        res.images, res.labels, Path("reports/figs/synth_check_grid.png"),
        title="合成域测试集人工核对（generator_repo，§2.4）",
        n=min(args.n, 20), seed=scfg.seed,
    )
    print("核对图：reports/figs/synth_check_grid.png")
