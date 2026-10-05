# -*- coding: utf-8 -*-
"""用训练好的检查点识别**用户自己准备**的车牌图片（推理入口）。

用途
----
训练与评测链路（``train/train.py`` / ``evaluate/main.py``）只处理仓库内的
CCPD / 合成数据集。本脚本把同一套预处理与前向传播开放给**任意一张图片**：

* 从 ``reports/checkpoints/<run>_best.npz`` 载入权重（不重新训练）；
* 从该次运行的部署配置（``reports/configs/<run>.yaml``，缺失时回退
  ``reports/configs/final.yaml``）恢复输入尺寸、裁剪比例等预处理参数，
  **与训练口径严格一致**；
* 从划分文件读取**训练集拟合的标准化统计量**（禁止用用户图片现算，
  否则分布就变了）；
* 输出六位字符、逐位置置信度与整牌联合置信度。

**本项目不做车牌检测**（§0.3）：输入必须是用户自行裁好的车牌区域。
支持两种输入形态：

* ``--mode full``（默认）：图片是**完整的 7 位车牌正视图**（含首位省份
  汉字），脚本按训练口径自动裁掉左侧 1/7 的汉字区域；
* ``--mode cropped``：图片**已经是后 6 位字符区域**（已裁掉汉字），
  脚本直接缩放到模型输入尺寸。

若车牌在照片里带倾斜/透视，可用 ``--corners`` 给出四角顶点
（顺序与 CCPD 一致：**右下、左下、左上、右上**），脚本先做与训练相同的
透视矫正再裁剪。

命令行
------
    python predict.py my_plate.jpg
    python predict.py a.jpg b.jpg --run final_s43 --mode cropped
    python predict.py car.jpg --corners "433,341;120,315;128,272;445,295"
    python predict.py my_plate.jpg --save-debug reports/figs/_debug --json
    python predict.py my_plate.jpg --ckpt reports/checkpoints/final_s42_best.npz

形状
----
输入图片 ``(H0, W0)`` -> ``(1, input_dim)`` -> 六个 ``(1, C_i)`` 概率头
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

# --- 包引导：本脚本位于项目根目录，根目录即 sys.path[0]，无需调整； ----------
# --- 但显式断言一下，避免从别的目录以绝对路径调用时出错。
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluate.main import load_run_config, load_standardizer  # noqa: E402
from models.backend import asnumpy, get_backend, set_backend_env  # noqa: E402
from models.ccpd_parse import perspective_rectify  # noqa: E402
from models.charset import decode_label, index_to_char  # noqa: E402
from models.config import load_config, resolve_path  # noqa: E402
from models.model import Params, forward, predict  # noqa: E402

# 默认识别用检查点：最终交付模型 final_s42（同分布测试字符 98.28% / 整牌 91.35%，
# 3 个种子中最常用的默认；另两个种子为 final_s43 / final_s44）。
DEFAULT_RUN = "final_s42"


def parse_corners(text: str) -> np.ndarray:
    """解析 ``--corners`` 参数为四角顶点数组。

    参数
    ----
    text : str
        ``"x1,y1;x2,y2;x3,y3;x4,y4"``，顺序为**右下、左下、左上、右上**
        （与 CCPD 文件名标注一致）。

    返回
    ----
    numpy.ndarray
        形状 ``(4, 2)``，float64。

    形状
    ----
    字符串 -> ``(4, 2)``
    """
    pts: List[List[float]] = []
    for chunk in text.split(";"):
        xy = chunk.split(",")
        if len(xy) != 2:
            raise ValueError(f"角点格式应为 x,y：{chunk!r}")
        pts.append([float(xy[0]), float(xy[1])])
    arr = np.asarray(pts, dtype=np.float64)
    if arr.shape != (4, 2):
        raise ValueError(f"需要恰好 4 个角点，实际 {arr.shape[0]} 个")
    return arr


def preprocess_custom(
    image: Image.Image,
    run_cfg,
    mode: str = "full",
    corners: Optional[np.ndarray] = None,
) -> np.ndarray:
    """把用户图片处理成与训练完全一致的模型输入（未标准化）。

    步骤（与 :func:`models.ccpd_parse.preprocess_crop` 同口径）
    ----------------------------------------------------------
    1. （可选）按四角顶点透视矫正为 ``rectify_size``；
    2. ``mode="full"`` 时把画面视为完整 7 位车牌：先缩放到
       ``rectify_size``，再裁掉左侧 ``1 - keep_right_fraction`` 的汉字区域；
       ``mode="cropped"`` 时跳过此步（图片已是后 6 位区域）；
    3. 缩放为 ``input_size``、转灰度、``[0, 1]`` 归一化。

    参数
    ----
    image : PIL.Image.Image
        用户图片（任意颜色模式、任意尺寸）。
    run_cfg : Config
        该次运行的配置（提供 ``rectify_size`` / ``input_size`` /
        ``keep_right_fraction``）。
    mode : str
        ``"full"`` 或 ``"cropped"``。
    corners : numpy.ndarray or None
        四角顶点 ``(4, 2)``；给出时先做透视矫正。

    返回
    ----
    numpy.ndarray
        形状 ``(H, W)``，float32，取值 ``[0, 1]``；**未**做全局标准化。

    形状
    ----
    ``(H0, W0)`` -> ``(input_size[1], input_size[0])``
    """
    rectify_size: Tuple[int, int] = (
        int(run_cfg.ccpd.rectify_size[0]),
        int(run_cfg.ccpd.rectify_size[1]),
    )
    input_size: Tuple[int, int] = (
        int(run_cfg.ccpd.input_size[0]),
        int(run_cfg.ccpd.input_size[1]),
    )
    keep_right = float(run_cfg.ccpd.keep_right_fraction)

    img = image.convert("L")

    if corners is not None:
        # 带倾斜/透视的车牌：先做与训练相同的透视矫正
        img = perspective_rectify(img, corners, rectify_size)

    if mode == "full":
        if corners is None:
            # 正视完整车牌：先归一到 rectify_size，使"裁掉左侧 1/7"与训练口径一致
            img = img.resize(rectify_size, resample=Image.BILINEAR)
        w, h = img.size
        x0 = int(round(rectify_size[0] * (1.0 - keep_right)))
        img = img.crop((x0, 0, w, h))
    elif mode != "cropped":
        raise ValueError(f"未知 mode：{mode!r}（应为 full / cropped）")

    img = img.resize(input_size, resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def predict_image(
    params: Params,
    arr: np.ndarray,
    standardizer,
    backend,
) -> Dict[str, Any]:
    """对单张预处理后的图片做前向传播并解码。

    参数
    ----
    params : Params
        已载入权重的模型参数。
    arr : numpy.ndarray
        形状 ``(H, W)``，float32，``[0, 1]``（未标准化）。
    standardizer : GlobalStandardizer
        训练集拟合的标准化器。
    backend : BackendInfo
        计算后端。

    返回
    ----
    dict
        ``text`` 六位字符串；``per_position`` 逐位 ``(字符, 置信度)``；
        ``joint_confidence`` 整牌联合置信度（六位概率乘积）。

    形状
    ----
    ``(H, W)`` -> 六位预测
    """
    x = standardizer.transform(arr[None, ...])          # (1, H, W)
    x = x.reshape(1, -1).astype(np.float32)             # (1, D)
    probs, _ = forward(params, x, backend, with_cache=False)
    pred, joint = predict(probs, backend)               # (1, 6), (1,)
    per_position = []
    for i, y in enumerate(probs):
        yn = asnumpy(y)[0]                              # (C_i,)
        k = int(pred[0, i])
        per_position.append(
            {"position": i + 1, "char": index_to_char(k),
             "confidence": round(float(yn[k]), 4)})
    return {
        "text": decode_label(pred[0]),
        "per_position": per_position,
        "joint_confidence": round(float(joint[0]), 4),
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
        0 表示成功；1 表示存在无法处理的图片。
    """
    ap = argparse.ArgumentParser(
        description="用训练好的检查点识别自己的车牌图片（不做检测，需先裁好车牌）")
    ap.add_argument("images", nargs="+", help="图片路径（可多张）")
    ap.add_argument("--run", default=DEFAULT_RUN,
                    help=f"检查点运行短名（默认 {DEFAULT_RUN}，同分布最优）；"
                         f"对应 reports/checkpoints/<run>_best.npz")
    ap.add_argument("--ckpt", default=None, metavar="PATH",
                    help="直接指定检查点文件（*.npz），优先于 --run；"
                         "配置仍按 --run 解析")
    ap.add_argument("--mode", choices=["full", "cropped"], default="full",
                    help="full=图片含完整 7 位车牌（自动裁掉首位汉字）；"
                         "cropped=图片已是后 6 位字符区域")
    ap.add_argument("--corners", default=None,
                    help='可选：车牌四角顶点 "x1,y1;x2,y2;x3,y3;x4,y4"，'
                         "顺序为右下、左下、左上、右上；给出时先做透视矫正")
    ap.add_argument("--backend", default="numpy", choices=["numpy", "cupy"],
                    help="推理后端（默认 numpy；单张推理本就按 CPU 口径）")
    ap.add_argument("--save-debug", default=None, metavar="DIR",
                    help="把送入模型的 32×128 灰度图存到该目录（核对预处理用）")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出结果")
    args = ap.parse_args(argv)

    cfg = load_config(None)
    run_cfg = load_run_config(cfg, args.run)
    ckpt = (Path(args.ckpt) if args.ckpt
            else resolve_path(run_cfg, "models_dir") / f"{args.run}_best.npz")
    if not ckpt.is_file():
        print(f"[predict] 错误：检查点不存在 {ckpt}", file=sys.stderr)
        return 1
    params, _ = Params.load(ckpt)
    standardizer = load_standardizer(run_cfg)

    backend = get_backend(args.backend, verbose=False)
    set_backend_env(backend)

    corners = parse_corners(args.corners) if args.corners else None
    debug_dir: Optional[Path] = None
    if args.save_debug:
        debug_dir = Path(args.save_debug)
        debug_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    n_fail = 0
    for path_str in args.images:
        path = Path(path_str)
        if not path.is_file():
            print(f"[predict] 跳过（文件不存在）：{path}", file=sys.stderr)
            n_fail += 1
            continue
        with Image.open(path) as im:
            arr = preprocess_custom(im, run_cfg, mode=args.mode, corners=corners)
        res = predict_image(params, arr, standardizer, backend)
        res["image"] = str(path)
        results.append(res)

        if debug_dir is not None:
            dbg = Image.fromarray((arr * 255.0).round().astype(np.uint8), mode="L")
            dbg.save(debug_dir / f"{path.stem}_model_input.png")

        if not args.json:
            per_pos = " ".join(
                f"{p['char']}({p['confidence']:.2f})" for p in res["per_position"])
            print(f"{path.name}: {res['text']}  "
                  f"整牌置信度 {res['joint_confidence']:.3f}  |  {per_pos}")

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    return 1 if n_fail == len(args.images) else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
