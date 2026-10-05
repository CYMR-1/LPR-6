# -*- coding: utf-8 -*-
"""数据准备：CCPD 全量解析、透视矫正裁剪、过滤统计与预处理缓存。

产出
----
* ``data/processed/ccpd_<W>x<H>.npz`` —— 预处理缓存（uint8 灰度裁剪图 + 标签）
* ``data/crops/<tag>/*.png``          —— 剪裁后的图片本体，可肉眼核对（默认写出）
                                          文件名 = 缓存行号_车牌文本_来源文件名.png
* ``data/manifest.csv``               —— 每张图可追溯的来源记录（§2.5 要求 2）
* ``reports/logs/prepare_stats.json`` —— 过滤统计（各类丢弃原因计数）

自检（§9）
----------
1. 字符集断言 = 34（由 ``models.charset`` 在导入时完成）；
2. 抽样 20 张人工核对通过（由 ``evaluate/visualize.py check-grid`` 产出网格图）；
   逐张核对可直接看 ``data/crops/<tag>/`` 下的 PNG（文件名含行号与车牌文本）；
3. 过滤规则生效且丢弃统计已记录。

用法
----
::

    python train/prepare_data.py                 # 全量（默认同时保存剪裁图 PNG）
    python train/prepare_data.py --limit 2000    # 只处理前 2000 张（快速验证）
    python train/prepare_data.py --workers 8     # 指定并行进程数
    python train/prepare_data.py --no-save-crops # 只写缓存，不写剪裁图
    python train/prepare_data.py --save-crops D  # 剪裁图写到目录 D
    python train/prepare_data.py --save-crops-n  # 只保存前 N 张剪裁图
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# --- 包引导：支持直接运行本文件 ---------------------------------------------
import sys as _sys

if __package__ in (None, ""):
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    # 直接运行 train/ 下脚本时 sys.path[0] 是 train/，那里的 train.py 会以顶层
    # 模块身份遮蔽同名 train 包，导致 `from train.xxx import ...` 失败，因此把
    # 脚本自身目录从 sys.path 中移除（项目根已插到最前，models/evaluate 仍可导入）。
    _here = str(Path(__file__).resolve().parent)
    while _here in _sys.path:
        _sys.path.remove(_here)

from models.ccpd_parse import (
    CcpdRecord,
    FilterStats,
    PrepParams,
    parse_filename,
    preprocess_crop,
    validate_record,
)
from models.charset import SEQ_LEN, decode_batch
from models.config import ROOT, ensure_dirs, load_config, resolve_path


# =============================================================================
# 工作进程：处理单张图片（必须定义在模块级，供 ProcessPoolExecutor 序列化）
# =============================================================================

#: 子进程内的全局预处理参数（避免每张图都传一次 dataclass）
_WORKER_PARAMS: Optional[PrepParams] = None
_WORKER_POSITIONS: Optional[Sequence[int]] = None


def _worker_init(params: PrepParams, positions: Optional[Sequence[int]]) -> None:
    """子进程初始化：缓存预处理参数。

    参数
    ----
    params : PrepParams
        预处理参数。
    positions : Sequence[int] or None
        位置约束。

    返回
    ----
    None
    """
    global _WORKER_PARAMS, _WORKER_POSITIONS
    _WORKER_PARAMS = params
    _WORKER_POSITIONS = positions


def _process_one(path_str: str) -> Tuple[str, Optional[str], Optional[np.ndarray],
                                         Optional[dict], Optional[dict]]:
    """在子进程中处理一张图片。

    参数
    ----
    path_str : str
        图片路径字符串。

    返回
    ----
    tuple
        ``(path, drop_reason, image_array, record_dict, meta_dict)``。
        成功时 ``drop_reason is None``；失败时其余字段为 ``None``。

    形状
    ----
    单张图片 -> ``(H, W) uint8``
    """
    from PIL import Image

    params = _WORKER_PARAMS
    assert params is not None, "worker 未初始化"
    path = Path(path_str)

    # ① 文件名解析
    try:
        rec = parse_filename(path)
    except ValueError:
        n_fields = len(path.name.rsplit(".", 1)[0].split("-"))
        return path_str, ("field_count" if n_fields != 7 else "parse_error"), None, None, None
    except Exception:
        return path_str, "parse_error", None, None, None

    # ② 打开图片并跑过滤 + 预处理
    try:
        with Image.open(path) as im:
            image_size = im.size
            reason = validate_record(
                rec,
                image_size=image_size,
                min_quad_area_px=params.min_quad_area_px,
                allow_out_of_bounds_px=params.allow_out_of_bounds_px,
                positions=_WORKER_POSITIONS,
            )
            if reason is not None:
                return path_str, reason, None, None, None
            work = im.convert("RGB") if im.mode not in ("L", "RGB") else im
            arr = preprocess_crop(work, rec.corners, params)
    except Exception:
        return path_str, "read_error", None, None, None

    img_u8 = (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    rec_dict = {
        "subset": rec.subset,
        "label": rec.label,
        "label_classes": rec.label_classes.tolist(),
        "label_indices": rec.label_indices,
        "corners": rec.corners.tolist(),
        "brightness": rec.brightness,
        "blurriness": rec.blurriness,
        "quad_area": rec.quad_area,
    }
    return path_str, None, img_u8, rec_dict, {"image_size": list(image_size)}


def _crop_filename(i: int, text: str, src: str) -> str:
    """拼剪裁图文件名：``<行号>_<车牌文本>_<来源文件名主干>.png``。

    参数
    ----
    i : int
        缓存行号（排序后的下标）。
    text : str
        车牌字符文本（6 位）。
    src : str
        来源图片路径。

    返回
    ----
    str
        Windows 安全（非法字符已替换为 ``_``）的文件名，含 ``.png``。

    形状
    ----
    标量 -> 标量

    说明
    ----
    行号即缓存下标，与 ``splits.npz`` 用同一坐标系，因此"文件名 ↔ 标签 ↔
    划分"三者可逐张对照；来源文件名主干本身已含 CCPD 的全部字段（含号码去重
    键），故不再重复拼进去。
    """
    safe_src = re.sub(r"[^0-9A-Za-z._-]", "_", Path(src).stem)[:140]
    return f"{i:06d}_{text or 'unknown'}_{safe_src}.png"


def save_crops(
    images: np.ndarray,
    labels: np.ndarray,
    sources: Sequence[str],
    out_dir: Path,
    limit: int = 0,
    verbose: bool = True,
) -> int:
    """把剪裁后的图片逐张写成 PNG，供人工核对。

    参数
    ----
    images : np.ndarray
        ``(N, H, W)`` uint8 灰度裁剪图，与缓存逐行相同。
    labels : np.ndarray
        ``(N, 6)`` int64 标签。
    sources : Sequence[str]
        每行对应的来源图片路径。
    out_dir : Path
        输出目录。
    limit : int
        只写前 N 张；``0`` 表示全部。
    verbose : bool
        是否打印进度。

    返回
    ----
    int
        成功写出的张数。

    形状
    ----
    ``(N, H, W)`` uint8 -> ``N`` 个 PNG 文件
    """
    from PIL import Image  # 局部导入：与 worker 内已有写法一致，不增模块级依赖

    n = int(images.shape[0]) if limit <= 0 else min(int(limit), int(images.shape[0]))
    out_dir.mkdir(parents=True, exist_ok=True)
    texts = decode_batch(labels[:n]) if n else []
    t0 = time.time()
    saved = 0
    for i in range(n):
        name = _crop_filename(i, texts[i], str(sources[i]))
        try:
            Image.fromarray(np.asarray(images[i], dtype=np.uint8), mode="L").save(
                out_dir / name, optimize=True)
            saved += 1
        except OSError as exc:  # 单张失败不应中断整批
            print(f"  ! 剪裁图写出失败（跳过）：{name}  {exc}", flush=True)
        if verbose and (saved % 2000 == 0 or i + 1 == n):
            el = time.time() - t0
            print(f"  [prepare] 剪裁图 {i + 1}/{n}  已写出 {saved}  {el:.0f}s"
                  f" ({saved / max(el, 1e-9):.0f} 张/s)", flush=True)
    if verbose:
        print(f"  [prepare] 剪裁图完成：{out_dir}（{saved} 张 PNG，"
              f"耗时 {time.time() - t0:.0f}s）", flush=True)
    return saved


def run_build(
    files: Sequence[Path],
    params: PrepParams,
    positions: Optional[Sequence[int]] = None,
    workers: int = 1,
    verbose: bool = True,
    save_crops_dir: Optional[Path] = None,
    save_crops_n: int = 0,
) -> Tuple[np.ndarray, np.ndarray, List[dict], List[str], FilterStats, int]:
    """并行跑完整预处理管线。

    参数
    ----
    files : Sequence[Path]
        图片路径列表。
    params : PrepParams
        预处理参数。
    positions : Sequence[int] or None
        位置约束。
    workers : int
        并行进程数；``1`` 表示串行（便于调试）。
    verbose : bool
        是否打印进度。
    save_crops_dir : Path or None
        非 None 时把剪裁后的图片写成 PNG 到该目录；``None`` 表示不写。
    save_crops_n : int
        写剪裁图的上限张数；``0`` 表示全部。

    返回
    ----
    tuple
        ``(images (N,H,W) uint8, labels (N,6) int64, meta_list, source_list, stats,
        saved_crops)``。三个列表与 ``images`` 逐行对应；``saved_crops`` 为实际
        写出的剪裁图张数（未开启落盘时为 0）。

    形状
    ----
    ``list[Path]`` -> ``(N, H, W) uint8`` + ``(N, 6) int64``

    说明
    ----
    剪裁图落盘接在**排序之后**，因此文件名里的行号与缓存下标、``splits.npz``
    下标严格一一对应；落盘只是旁路写文件，缓存内容不受影响。
    """
    stats = FilterStats()
    images: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    metas: List[dict] = []
    sources: List[str] = []

    total = len(files)
    t0 = time.time()
    done = 0

    def _consume(path_str: str, reason, arr, rec_dict, meta) -> None:
        """处理单条结果：计入统计或收集。"""
        nonlocal done
        done += 1
        if reason is not None:
            stats.add_drop(reason, Path(path_str).name)
        else:
            img_u8 = arr
            images.append(img_u8)
            labels.append(np.asarray(rec_dict["label_classes"], dtype=np.int64))
            metas.append(rec_dict)
            sources.append(path_str)
            stats.add_keep()
        if verbose and (done % 2000 == 0 or done == total):
            el = time.time() - t0
            print(f"  {done}/{total}  保留 {stats.kept}  丢弃 {sum(stats.dropped.values())}"
                  f"  {el:.0f}s ({done / max(el, 1e-9):.0f} 张/s)", flush=True)

    if workers > 1:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
            initargs=(params, positions),
        ) as ex:
            futures = [ex.submit(_process_one, str(p)) for p in files]
            for fut in as_completed(futures):
                _consume(*fut.result())
    else:
        _worker_init(params, positions)
        for p in files:
            _consume(*_process_one(str(p)))

    if not images:
        h, w = params.input_size[1], params.input_size[0]
        return (np.zeros((0, h, w), dtype=np.uint8),
                np.zeros((0, SEQ_LEN), dtype=np.int64), [], [], stats, 0)

    # ★ 可复现性修复：多进程时上面按"完成顺序"收集结果，顺序随运行时机而变，
    #    导致同一数据集两次跑出的缓存样本顺序不同（重建 24×96 缓存时实测
    #    25592/28341 个位置不同）。划分文件按下标引用样本，顺序不定 ⇒
    #    重建数据后训练轨迹无法逐位复现。这里统一按 sources 字典序排序，
    #    保证任何机器、任何并行度下产出**逐位一致**的缓存。
    order = sorted(range(len(sources)), key=lambda i: sources[i])
    images = [images[i] for i in order]
    labels = [labels[i] for i in order]
    metas = [metas[i] for i in order]
    sources = [sources[i] for i in order]

    # 剪裁图落盘：必须在排序之后，文件名里的行号才等于缓存下标
    saved_crops = 0
    if save_crops_dir is not None:
        saved_crops = save_crops(
            np.stack(images, axis=0),
            np.stack(labels, axis=0),
            sources,
            Path(save_crops_dir),
            limit=int(save_crops_n),
            verbose=True,
        )

    return (np.stack(images, axis=0), np.stack(labels, axis=0),
            metas, sources, stats, saved_crops)


# =============================================================================
# 主流程
# =============================================================================


def main(argv: Optional[Sequence[str]] = None) -> int:
    """数据准备主入口。

    参数
    ----
    argv : Sequence[str] or None
        命令行参数；``None`` 时取 ``sys.argv[1:]``。

    返回
    ----
    int
        进程退出码。
    """
    ap = argparse.ArgumentParser(description="数据准备：CCPD 解析、裁剪、过滤统计与缓存")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 张（0 = 全部）")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1),
                    help="并行进程数")
    ap.add_argument("--config", type=str, default=None, help="配置文件路径")
    ap.add_argument("--out-tag", type=str, default=None, help="缓存文件名标签，默认按输入尺寸")
    ap.add_argument("--save-crops", nargs="?", const="", default=None, metavar="DIR",
                    help="保存剪裁后的图片到 DIR（不带 DIR 时用 data/crops/<tag>/）；"
                         "不加该参数时也会保存，等价于默认开启")
    ap.add_argument("--no-save-crops", action="store_true",
                    help="只写缓存，不保存剪裁图")
    ap.add_argument("--save-crops-n", type=int, default=0,
                    help="只保存前 N 张剪裁图（0 = 全部）")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    params = PrepParams.from_config(cfg)

    ccpd_root = Path(cfg.paths.ccpd_root)
    from models.dataset import discover_images

    files = discover_images(ccpd_root)
    if not files:
        print(f"[prepare] 错误：{ccpd_root} 下没有找到图片")
        return 2
    if args.limit and args.limit > 0:
        files = files[: args.limit]

    print(f"[prepare] 发现 {len(files)} 张图片，预处理参数：")
    print(f"     矫正尺寸 {params.rectify_size}  输入尺寸 {params.input_size}  "
          f"裁剪起始列 {params.crop_x0}  保留比例 {params.keep_right_fraction:.6f}")
    print(f"     面积下限 {params.min_quad_area_px}  越界容差 {params.allow_out_of_bounds_px}")
    print(f"     并行进程 {args.workers}")

    # 剪裁图落盘：默认开启（--no-save-crops 关闭；--save-crops DIR 改址）
    save_crops_dir: Optional[Path] = None
    if not args.no_save_crops:
        tag_for_crops = args.out_tag or \
            f"ccpd_{params.input_size[0]}x{params.input_size[1]}"
        if args.save_crops:
            save_crops_dir = Path(args.save_crops).expanduser().resolve()
        else:
            save_crops_dir = ROOT / "data" / "crops" / tag_for_crops
        print(f"     剪裁图目录 {save_crops_dir}"
              f"（{'全部' if args.save_crops_n <= 0 else f'前 {args.save_crops_n} 张'}）")
    else:
        print("     剪裁图目录 不保存（--no-save-crops）")

    t0 = time.time()
    images, labels, metas, sources, stats, n_crops_saved = run_build(
        files, params, workers=int(args.workers),
        save_crops_dir=save_crops_dir, save_crops_n=int(args.save_crops_n),
    )
    elapsed = time.time() - t0

    tag = args.out_tag or f"ccpd_{params.input_size[0]}x{params.input_size[1]}"
    cache_path = resolve_path(cfg, "processed_dir") / f"{tag}.npz"

    summary = stats.summary()
    meta_blob = {
        "tag": tag,
        "n_total": len(files),
        "n_kept": int(images.shape[0]),
        "input_size": list(params.input_size),
        "rectify_size": list(params.rectify_size),
        "keep_right_fraction": params.keep_right_fraction,
        "crop_x0": params.crop_x0,
        "min_quad_area_px": params.min_quad_area_px,
        "allow_out_of_bounds_px": params.allow_out_of_bounds_px,
        "preprocess_version": cfg.project.preprocess_version,
        "filter_summary": summary,
        "elapsed_sec": round(elapsed, 1),
        "images_shape": list(images.shape),
        "labels_shape": list(labels.shape),
        "saved_crops_dir": str(save_crops_dir) if save_crops_dir is not None else None,
        "saved_crops_count": int(n_crops_saved),
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        images=images,
        labels=labels,
        meta=np.asarray(json.dumps(meta_blob, ensure_ascii=False)),
        sources=np.asarray(sources, dtype=object) if sources else np.asarray([], dtype=object),
        records=np.asarray([json.dumps(m, ensure_ascii=False) for m in metas], dtype=object)
        if metas else np.asarray([], dtype=object),
    )
    print(f"[prepare] 缓存已写出：{cache_path}  ({cache_path.stat().st_size / 1024 ** 2:.1f} MB)")

    # 过滤统计落盘
    log_path = resolve_path(cfg, "logs_dir") / "prepare_stats.json"
    if tag != "ccpd_128x32":
        # ★ 非主分辨率的统计不得覆盖主口径文件（24×96 曾把
        # prepare_stats.json 覆盖成自己的口径——与 split_summary 同款坑）
        log_path = log_path.with_name(f"prepare_stats_{tag.replace('ccpd_', '')}.json")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as fp:
        json.dump(meta_blob, fp, ensure_ascii=False, indent=2)
    print(f"[prepare] 过滤统计已写出：{log_path}")

    # ---- 控制台摘要 -------------------------------------------------------
    print()
    print("=" * 72)
    print(f"数据准备完成：扫描 {summary['total_scanned']} 张，"
          f"保留 {summary['kept']}，丢弃 {summary['dropped_total']} "
          f"（丢弃率 {summary['drop_rate'] * 100:.2f}%），耗时 {elapsed:.0f}s")
    print("丢弃原因明细：")
    for reason, info in summary["by_reason"].items():
        print(f"  {info['count']:6d}  {reason:18s} {info['desc']}")
    print(f"图像数组：{images.shape} {images.dtype}   标签数组：{labels.shape} {labels.dtype}")
    if save_crops_dir is not None:
        print(f"剪裁图：{save_crops_dir}（{n_crops_saved} 张 PNG，"
              f"文件名 = 行号_车牌文本_来源文件名.png）")
    print("=" * 72)
    print()
    print("下一步（人工核对必须先做）：")
    print("  python evaluate/visualize.py check-grid     # 抽样 20 张人工核对")
    print("  核对通过后再执行： python train/split_dataset.py")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())