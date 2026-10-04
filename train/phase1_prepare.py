# -*- coding: utf-8 -*-
"""P1 阶段：CCPD 全量解析、透视矫正裁剪、过滤统计与预处理缓存。

产出
----
* ``data/processed/ccpd_<W>x<H>.npz`` —— 预处理缓存（uint8 灰度裁剪图 + 标签）
* ``data/manifest.csv``               —— 每张图可追溯的来源记录（§2.5 要求 2）
* ``reports/logs/prepare_stats.json`` —— 过滤统计（各类丢弃原因计数）

验收（§9 P1）
-------------
1. 字符集断言 = 34（由 ``models.charset`` 在导入时完成）；
2. 抽样 20 张人工核对通过（由 ``evaluate/visualize.py check-grid`` 产出网格图）；
3. 过滤规则生效且丢弃统计已记录。

用法
----
::

    python train/phase1_prepare.py                 # 全量
    python train/phase1_prepare.py --limit 2000    # 只处理前 2000 张（快速验证）
    python train/phase1_prepare.py --workers 8     # 指定并行进程数
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# --- 包引导：支持直接运行本文件 ---------------------------------------------
import sys as _sys

if __package__ in (None, ""):
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.ccpd_parse import (
    CcpdRecord,
    FilterStats,
    PrepParams,
    parse_filename,
    preprocess_crop,
    validate_record,
)
from models.charset import SEQ_LEN
from models.config import ensure_dirs, load_config, resolve_path


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


def run_build(
    files: Sequence[Path],
    params: PrepParams,
    positions: Optional[Sequence[int]] = None,
    workers: int = 1,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[dict], List[str], FilterStats]:
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

    返回
    ----
    tuple
        ``(images (N,H,W) uint8, labels (N,6) int64, meta_list, source_list, stats)``。
        三个列表与 ``images`` 逐行对应。

    形状
    ----
    ``list[Path]`` -> ``(N, H, W) uint8`` + ``(N, 6) int64``
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
                np.zeros((0, SEQ_LEN), dtype=np.int64), [], [], stats)

    # ★ 可复现性修复：多进程时上面按"完成顺序"收集结果，顺序随运行时机而变，
    #    导致同一数据集两次跑出的缓存样本顺序不同（E8 重建 24×96 缓存时实测
    #    25592/28341 个位置不同）。划分文件按下标引用样本，顺序不定 ⇒
    #    重建数据后训练轨迹无法逐位复现。这里统一按 sources 字典序排序，
    #    保证任何机器、任何并行度下产出**逐位一致**的缓存。
    order = sorted(range(len(sources)), key=lambda i: sources[i])
    images = [images[i] for i in order]
    labels = [labels[i] for i in order]
    metas = [metas[i] for i in order]
    sources = [sources[i] for i in order]

    return (np.stack(images, axis=0), np.stack(labels, axis=0),
            metas, sources, stats)


# =============================================================================
# 主流程
# =============================================================================


def main(argv: Optional[Sequence[str]] = None) -> int:
    """P1 主入口。

    参数
    ----
    argv : Sequence[str] or None
        命令行参数；``None`` 时取 ``sys.argv[1:]``。

    返回
    ----
    int
        进程退出码。
    """
    ap = argparse.ArgumentParser(description="P1：CCPD 解析、裁剪、过滤统计与缓存")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 张（0 = 全部）")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1),
                    help="并行进程数")
    ap.add_argument("--config", type=str, default=None, help="配置文件路径")
    ap.add_argument("--out-tag", type=str, default=None, help="缓存文件名标签，默认按输入尺寸")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    params = PrepParams.from_config(cfg)

    ccpd_root = Path(cfg.paths.ccpd_root)
    from models.dataset import discover_images

    files = discover_images(ccpd_root)
    if not files:
        print(f"[P1] 错误：{ccpd_root} 下没有找到图片")
        return 2
    if args.limit and args.limit > 0:
        files = files[: args.limit]

    print(f"[P1] 发现 {len(files)} 张图片，预处理参数：")
    print(f"     矫正尺寸 {params.rectify_size}  输入尺寸 {params.input_size}  "
          f"裁剪起始列 {params.crop_x0}  保留比例 {params.keep_right_fraction:.6f}")
    print(f"     面积下限 {params.min_quad_area_px}  越界容差 {params.allow_out_of_bounds_px}")
    print(f"     并行进程 {args.workers}")

    t0 = time.time()
    images, labels, metas, sources, stats = run_build(
        files, params, workers=int(args.workers)
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
    print(f"[P1] 缓存已写出：{cache_path}  ({cache_path.stat().st_size / 1024 ** 2:.1f} MB)")

    # 过滤统计落盘
    log_path = resolve_path(cfg, "logs_dir") / "prepare_stats.json"
    if tag != "ccpd_128x32":
        # ★ 非主分辨率的统计不得覆盖主口径文件（E8 的 24×96 曾把
        # prepare_stats.json 覆盖成自己的口径——与 split_summary 同款坑）
        log_path = log_path.with_name(f"prepare_stats_{tag.replace('ccpd_', '')}.json")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as fp:
        json.dump(meta_blob, fp, ensure_ascii=False, indent=2)
    print(f"[P1] 过滤统计已写出：{log_path}")

    # ---- 控制台摘要 -------------------------------------------------------
    print()
    print("=" * 72)
    print(f"P1 预处理完成：扫描 {summary['total_scanned']} 张，"
          f"保留 {summary['kept']}，丢弃 {summary['dropped_total']} "
          f"（丢弃率 {summary['drop_rate'] * 100:.2f}%），耗时 {elapsed:.0f}s")
    print("丢弃原因明细：")
    for reason, info in summary["by_reason"].items():
        print(f"  {info['count']:6d}  {reason:18s} {info['desc']}")
    print(f"图像数组：{images.shape} {images.dtype}   标签数组：{labels.shape} {labels.dtype}")
    print("=" * 72)
    print()
    print("下一步（P1 验收必须先做）：")
    print("  python evaluate/visualize.py check-grid     # 抽样 20 张人工核对")
    print("  核对通过后再执行： python train/phase15_split.py")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())