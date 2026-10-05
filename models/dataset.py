# -*- coding: utf-8 -*-
"""数据集、预处理缓存与批加载器（§2.5 / §4.3 / §8.3）。

职责
----
1. 把 CCPD 原图批量跑预处理管线，缓存成 ``.npz``（避免每次训练重解码 JPEG）；
2. 维护 ``manifest.csv``：每张图可追溯到来源文件、子集名、六字符标签、
   裁剪参数与脚本版本（§2.5 要求 2）；
3. 提供 :class:`PlateDataset` 与批迭代器，**支持 batch=1 与全批量**
   （``{1, 32, 64, 128, full}`` 全部可跑）；
4. 提供 :class:`GlobalStandardizer`：全局零均值/单位方差标准化，
   **统计量只在训练集上计算**（§2.3.2）。

内存布局约定
------------
缓存中的图像以 ``uint8`` 存储，形状 ``(N, H, W)``；标签形状 ``(N, 6)``。
标准化在取批时进行，因此同一份 ``uint8`` 缓存可服务于任意标准化配置。
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

# --- 包引导：支持直接运行本文件 ---------------------------------------------
import sys as _sys

if __package__ in (None, ""):
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.ccpd_parse import (
    CcpdRecord,
    FilterStats,
    PrepParams,
    preprocess_crop,
    parse_filename,
    validate_record,
)
from models.charset import SEQ_LEN, check_label_legal, decode_batch

# =============================================================================
# 1. manifest.csv 的列定义（§2.5 要求 2）
# =============================================================================

#: manifest.csv 的列顺序。列的选择直接对应规格中"每张测试图可追溯到来源"的要求。
MANIFEST_COLUMNS: List[str] = [
    "index",            # 在缓存中的行号
    "subset",           # 子集名（来源文件名内标注）
    "source_file",      # 来源文件相对路径
    "label",            # 六字符标签
    "label_classes",    # 六位类别索引，空格分隔
    "split",            # train / val / test / hard_test / synth_test
    "plate_number",     # 完整 7 位车牌号（含省份索引），用于号码去重
    "plate_hash",       # 前 6 位的稳定标识（号码去重键）
    "corners",          # 四角顶点，分号分隔
    "rectify_size",     # 透视矫正目标尺寸 WxH
    "input_size",       # 最终输入尺寸 WxH
    "crop_x0",          # 裁掉首位汉字后的起始列
    "keep_right_fraction",
    "brightness",       # CCPD 字段 5
    "blurriness",       # CCPD 字段 6
    "quad_area",        # 四边形面积
    "preprocess_version",  # 预处理脚本版本
    "split_seed",       # 划分随机种子
]


def _join_ints(values: Sequence[int]) -> str:
    """把整数序列拼成空格分隔字符串（写 CSV 用）。"""
    return " ".join(str(int(v)) for v in values)


def _join_corners(corners: np.ndarray) -> str:
    """把 ``(4, 2)`` 顶点拼成 ``"x,y;x,y;..."`` 字符串（写 CSV 用）。"""
    return ";".join(f"{float(x):.1f},{float(y):.1f}" for x, y in corners)


# =============================================================================
# 2. 全局标准化（统计量只在训练集上计算）
# =============================================================================


@dataclass
class GlobalStandardizer:
    """全局零均值 / 单位方差标准化器。

    属性
    ----
    mean : float
        训练集像素均值；未拟合时为 0。
    std : float
        训练集像素标准差；未拟合时为 1。
    fitted : bool
        是否已用训练集统计量拟合。
    n_samples : int
        拟合时使用的样本数。
    """

    mean: float = 0.0
    std: float = 1.0
    fitted: bool = False
    n_samples: int = 0

    @classmethod
    def fit(cls, images: np.ndarray, eps: float = 1e-8) -> "GlobalStandardizer":
        """在给定图像上统计均值与标准差（**只允许传训练集**）。

        参数
        ----
        images : numpy.ndarray
            形状 ``(N, H, W)`` 或 ``(N, H*W)``，取值 ``[0, 1]`` 或 ``[0, 255]``
            均可（本函数按 float64 统一计算）。
        eps : float
            防止除零的下限。

        返回
        ----
        GlobalStandardizer
            已拟合的标准化器。

        形状
        ----
        ``(N, H, W)`` -> 标量统计量
        """
        arr = np.asarray(images, dtype=np.float64)
        mean = float(arr.mean())
        std = float(arr.std())
        if std < eps:
            std = 1.0
        return cls(mean=mean, std=std, fitted=True, n_samples=int(arr.shape[0]))

    def transform(self, images: np.ndarray) -> np.ndarray:
        """标准化。

        参数
        ----
        images : numpy.ndarray
            形状 ``(N, H, W)`` 或 ``(N, D)``，float32。

        返回
        ----
        numpy.ndarray
            同形状，float32。

        形状
        ----
        ``(N, ...)`` -> ``(N, ...)``
        """
        arr = np.asarray(images, dtype=np.float32)
        if not self.fitted:
            raise RuntimeError("标准化器尚未用训练集拟合，禁止在未拟合状态调用 transform()")
        return (arr - np.float32(self.mean)) / np.float32(self.std)

    def as_dict(self) -> Dict[str, float]:
        """转成可写入 JSON 的字典。"""
        return {
            "mean": self.mean,
            "std": self.std,
            "fitted": self.fitted,
            "n_samples": self.n_samples,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "GlobalStandardizer":
        """从字典恢复。"""
        return cls(
            mean=float(d.get("mean", 0.0)),
            std=float(d.get("std", 1.0)),
            fitted=bool(d.get("fitted", False)),
            n_samples=int(d.get("n_samples", 0)),
        )


# =============================================================================
# 3. 预处理缓存构建
# =============================================================================


@dataclass
class BuildResult:
    """一次预处理缓存构建的结果。

    属性
    ----
    images : numpy.ndarray
        形状 ``(N, H, W)``，uint8。
    labels : numpy.ndarray
        形状 ``(N, 6)``，int64。
    records : list of CcpdRecord
        与 ``images`` 逐行对应的解析记录。
    stats : FilterStats
        过滤统计。
    """

    images: np.ndarray
    labels: np.ndarray
    records: List[CcpdRecord]
    stats: FilterStats


def discover_images(root: Path) -> List[Path]:
    """递归收集目录下所有图片（支持平铺与按子集分目录两种布局）。

    参数
    ----
    root : Path
        CCPD 图片根目录。

    返回
    ----
    list of Path
        排序后的图片路径列表。

    形状
    ----
    目录 -> ``list[Path]``
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [p for p in root.rglob("*") if p.suffix.lower() in exts]
    return sorted(files)


def build_cache(
    paths: Sequence[Path],
    params: PrepParams,
    positions: Optional[Sequence[int]] = None,
    require_image_size_check: bool = True,
    verbose: bool = True,
) -> BuildResult:
    """把一批 CCPD 图片跑完整预处理管线并汇总成数组。

    参数
    ----
    paths : Sequence[Path]
        图片路径列表。
    params : PrepParams
        预处理参数。
    positions : Sequence[int] or None
        位置约束；给了就按它做位置合法性过滤。
    require_image_size_check : bool
        是否启用顶点越界检查（需要打开图片读尺寸）。
    verbose : bool
        是否打印进度。

    返回
    ----
    BuildResult
        图像数组 ``(N, H, W)`` uint8、标签数组 ``(N, 6)`` int64、记录与过滤统计。

    形状
    ----
    ``list[Path]`` -> ``(N, H, W) uint8`` + ``(N, 6) int64``
    """
    from PIL import Image

    stats = FilterStats()
    images: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    records: List[CcpdRecord] = []

    total = len(paths)
    for i, path in enumerate(paths, 1):
        # ---- ① 文件名解析 --------------------------------------------------
        try:
            rec = parse_filename(path)
        except ValueError:
            stats.add_drop("field_count" if len(path.name.rsplit(".", 1)[0].split("-")) != 7
                           else "parse_error", path.name)
            continue
        except Exception:
            stats.add_drop("parse_error", path.name)
            continue

        # ---- ② 打开图片一次，同时拿到尺寸与像素 -----------------------------
        try:
            with Image.open(path) as im:
                image_size = im.size if require_image_size_check else None
                # ---- ③ 过滤规则 --------------------------------------------
                reason = validate_record(
                    rec,
                    image_size=image_size,
                    min_quad_area_px=params.min_quad_area_px,
                    allow_out_of_bounds_px=params.allow_out_of_bounds_px,
                    positions=positions,
                )
                if reason is not None:
                    stats.add_drop(reason, path.name)
                    continue
                # ---- ④ 预处理管线 ------------------------------------------
                work = im.convert("RGB") if im.mode not in ("L", "RGB") else im
                arr = preprocess_crop(work, rec.corners, params)
        except Exception:
            stats.add_drop("read_error", path.name)
            continue

        images.append((np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8))
        labels.append(rec.label_classes)
        records.append(rec)
        stats.add_keep()

        if verbose and (i % 2000 == 0 or i == total):
            print(f"  [build_cache] {i}/{total}  保留 {stats.kept}  丢弃 {sum(stats.dropped.values())}",
                  flush=True)

    if not images:
        h, w = params.input_size[1], params.input_size[0]
        return BuildResult(
            images=np.zeros((0, h, w), dtype=np.uint8),
            labels=np.zeros((0, SEQ_LEN), dtype=np.int64),
            records=[],
            stats=stats,
        )

    return BuildResult(
        images=np.stack(images, axis=0),
        labels=np.stack(labels, axis=0),
        records=records,
        stats=stats,
    )


def save_cache(
    out_path: Path,
    result: BuildResult,
    params: PrepParams,
    extra_meta: Optional[Dict[str, object]] = None,
) -> None:
    """把预处理结果保存为 ``.npz``。

    参数
    ----
    out_path : Path
        输出路径。
    result : BuildResult
        构建结果。
    params : PrepParams
        预处理参数（写入元信息，便于校验缓存与配置是否匹配）。
    extra_meta : dict or None
        额外元信息。

    返回
    ----
    None

    形状
    ----
    ``(N, H, W) uint8`` + ``(N, 6) int64`` -> ``.npz``
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "rectify_size": list(params.rectify_size),
        "input_size": list(params.input_size),
        "keep_right_fraction": params.keep_right_fraction,
        "crop_x0": params.crop_x0,
        "grayscale": params.grayscale,
        "filter_summary": result.stats.summary(),
    }
    if extra_meta:
        meta.update(extra_meta)
    np.savez_compressed(
        out_path,
        images=result.images,
        labels=result.labels,
        meta=np.asarray(json.dumps(meta, ensure_ascii=False)),
    )


def load_cache(path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """读取 ``.npz`` 缓存。

    参数
    ----
    path : Path
        缓存路径。

    返回
    ----
    tuple
        ``(images (N,H,W) uint8, labels (N,6) int64, meta dict)``。

    形状
    ----
    ``.npz`` -> ``(N, H, W)`` + ``(N, 6)`` + dict
    """
    with np.load(path, allow_pickle=False) as data:
        images = data["images"]
        labels = data["labels"]
        meta = json.loads(str(data["meta"])) if "meta" in data else {}
    return images, labels, meta


# =============================================================================
# 4. 数据集与批加载器
# =============================================================================


@dataclass
class PlateDataset:
    """车牌字符识别数据集。

    属性
    ----
    images : numpy.ndarray
        形状 ``(N, H, W)``，uint8，取值 ``[0, 255]``。
    labels : numpy.ndarray
        形状 ``(N, 6)``，int64。
    standardizer : GlobalStandardizer
        标准化器；未拟合时会抛错以强制"统计量只在训练集上计算"。
    flatten : bool
        是否在取批时展平为 ``(B, H*W)``（MLP 需要 ``True``）。
    augment_fn : callable or None
        数据增强函数，签名 ``(image_hw: ndarray, rng) -> ndarray``，作用在单张
        ``(H, W)`` float32 图像上。
    seed : int
        本数据集实例的随机种子（用于打乱与增强）。
    name : str
        数据集名（train / val / test / ...），仅用于日志。
    """

    images: np.ndarray
    labels: np.ndarray
    standardizer: GlobalStandardizer
    flatten: bool = True
    augment_fn: Optional[object] = None
    seed: int = 42
    name: str = "train"

    # ------------------------------------------------------------------ 基本信息
    def __len__(self) -> int:
        """样本数 ``N``。"""
        return int(self.images.shape[0])

    @property
    def input_shape(self) -> Tuple[int, int]:
        """单张输入形状 ``(H, W)``。"""
        return int(self.images.shape[1]), int(self.images.shape[2])

    @property
    def input_dim(self) -> int:
        """展平后的维度 ``H*W``。"""
        h, w = self.input_shape
        return h * w

    def subset(self, indices: np.ndarray) -> "PlateDataset":
        """按索引取子集，返回新数据集（共享底层数组，不复制像素）。

        参数
        ----
        indices : numpy.ndarray
            形状 ``(M,)`` 的整数索引。

        返回
        ----
        PlateDataset
            新数据集。

        形状
        ----
        ``(N, H, W)`` -> ``(M, H, W)``
        """
        idx = np.asarray(indices, dtype=np.int64)
        return PlateDataset(
            images=self.images[idx],
            labels=self.labels[idx],
            standardizer=self.standardizer,
            flatten=self.flatten,
            augment_fn=self.augment_fn,
            seed=self.seed,
            name=self.name,
        )

    # ------------------------------------------------------------------ 取批
    def get_batch(self, indices: np.ndarray, augment: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """按索引取一个批，完成增强、标准化与展平。

        参数
        ----
        indices : numpy.ndarray
            形状 ``(B,)`` 的整数索引。
        augment : bool
            是否对**本批**施加数据增强（只有训练集应传 ``True``）。

        返回
        ----
        tuple
            ``(x, y)``：``x`` 形状 ``(B, D)`` 或 ``(B, H, W)`` float32；
            ``y`` 形状 ``(B, 6)`` int64。

        形状
        ----
        ``(B,)`` -> ``(B, D) float32`` + ``(B, 6) int64``
        """
        idx = np.asarray(indices, dtype=np.int64)
        batch = self.images[idx].astype(np.float32) / 255.0  # (B, H, W)
        labels = self.labels[idx].astype(np.int64)

        if augment and self.augment_fn is not None:
            rng = np.random.default_rng(int(self.seed) + int(idx.sum()) % (2 ** 31))
            batch = np.stack([self.augment_fn(batch[i], rng) for i in range(batch.shape[0])])

        batch = self.standardizer.transform(batch)
        if self.flatten:
            batch = batch.reshape(batch.shape[0], -1)
        return batch.astype(np.float32), labels

    def iter_batches(
        self,
        batch_size: int,
        shuffle: bool = True,
        augment: bool = False,
        seed: Optional[int] = None,
        drop_last: bool = False,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """迭代批。

        参数
        ----
        batch_size : int
            批大小。``-1`` 或 ``<= 0`` 表示**全批量**（一个批包含全部样本）。
        shuffle : bool
            是否打乱顺序。
        augment : bool
            是否施加数据增强。
        seed : int or None
            打乱种子；``None`` 时用数据集自身 ``seed``。
        drop_last : bool
            是否丢弃最后一个不满批。

        返回
        ----
        iterator
            产出 ``(x, y)``。

        形状
        ----
        ``(N, H, W)`` -> 多个 ``(B, D)``
        """
        n = len(self)
        rng = np.random.default_rng(int(seed if seed is not None else self.seed))
        order = rng.permutation(n) if shuffle else np.arange(n)

        if batch_size is None or batch_size <= 0:  # 全批量
            yield self.get_batch(order, augment=augment)
            return

        for start in range(0, n, batch_size):
            chunk = order[start:start + batch_size]
            if len(chunk) < batch_size and drop_last:
                continue
            if len(chunk) == 0:
                continue
            yield self.get_batch(chunk, augment=augment)

    def num_batches(self, batch_size: int) -> int:
        """给定批大小下的批数（``batch_size<=0`` 记为 1，即全批量）。

        参数
        ----
        batch_size : int
            批大小。

        返回
        ----
        int
            批数。
        """
        if batch_size is None or batch_size <= 0:
            return 1
        return (len(self) + batch_size - 1) // batch_size

    def decode(self, indices: np.ndarray) -> List[str]:
        """把标签索引解码为字符串列表（调试用）。

        参数
        ----
        indices : numpy.ndarray
            形状 ``(B, 6)``。

        返回
        ----
        list of str
            长度 ``B``。

        形状
        ----
        ``(B, 6)`` -> ``list[str]``
        """
        return decode_batch(indices)


# =============================================================================
# 5. 写 manifest.csv
# =============================================================================


def write_manifest(
    out_path: Path,
    records: Sequence[CcpdRecord],
    splits: Sequence[str],
    params: PrepParams,
    preprocess_version: str,
    split_seed: int,
    plate_keys: Optional[Sequence[str]] = None,
) -> None:
    """写出 ``manifest.csv``（§2.5 要求 2）。

    参数
    ----
    out_path : Path
        输出路径。
    records : Sequence[CcpdRecord]
        与缓存逐行对应的记录。
    splits : Sequence[str]
        每行所属集合。
    params : PrepParams
        裁剪参数。
    preprocess_version : str
        预处理脚本版本。
    split_seed : int
        划分随机种子。
    plate_keys : Sequence[str] or None
        号码去重键；``None`` 时用标签本身。

    返回
    ----
    None

    形状
    ----
    逐行写 CSV，无张量。
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.writer(fp)
        writer.writerow(MANIFEST_COLUMNS)
        for i, rec in enumerate(records):
            key = plate_keys[i] if plate_keys is not None else rec.label
            writer.writerow([
                i,
                rec.subset,
                str(rec.path),
                rec.label,
                _join_ints(rec.label_classes),
                splits[i] if i < len(splits) else "",
                _join_ints(rec.label_indices),
                key,
                _join_corners(rec.corners),
                f"{params.rectify_size[0]}x{params.rectify_size[1]}",
                f"{params.input_size[0]}x{params.input_size[1]}",
                params.crop_x0,
                f"{params.keep_right_fraction:.6f}",
                rec.brightness,
                rec.blurriness,
                f"{rec.quad_area:.1f}",
                preprocess_version,
                split_seed,
            ])


def read_manifest(path: Path) -> List[Dict[str, str]]:
    """读取 ``manifest.csv`` 为字典列表。

    参数
    ----
    path : Path
        CSV 路径。

    返回
    ----
    list of dict
        每行一个字典。

    形状
    ----
    CSV -> ``list[dict]``
    """
    with open(path, "r", encoding="utf-8-sig", newline="") as fp:
        return list(csv.DictReader(fp))


# =============================================================================
# 6. 调试入口
# =============================================================================

if __name__ == "__main__":  # pragma: no cover
    import time

    root = Path("data/ccpd")
    files = discover_images(root)
    print(f"发现图片 {len(files)} 张")
    if not files:
        raise SystemExit("未找到图片，请先准备 data/ccpd/")

    params = PrepParams()
    t0 = time.time()
    # 先只跑一小批，确认管线可用
    small = files[:200]
    res = build_cache(small, params, verbose=False)
    dt = time.time() - t0
    print(f"试跑 {len(small)} 张：保留 {res.stats.kept}，丢弃 {sum(res.stats.dropped.values())}，"
          f"耗时 {dt:.1f}s（{dt / max(1, len(small)) * 1000:.1f} ms/张）")
    print("图像数组形状 :", res.images.shape, res.images.dtype)
    print("标签数组形状 :", res.labels.shape, res.labels.dtype)
    print("前 5 个标签  :", decode_batch(res.labels[:5]))
    print("首图像素范围 :", int(res.images[0].min()), "-", int(res.images[0].max()))
    print(json.dumps(res.stats.summary(), ensure_ascii=False, indent=2)[:1200])

    std = GlobalStandardizer.fit(res.images[:100].astype(np.float32) / 255.0)
    print("标准化统计   :", std.as_dict())
    ds = PlateDataset(res.images, res.labels, std, name="debug")
    x, y = ds.get_batch(np.arange(8))
    print("批形状       :", x.shape, y.shape, "| 期望 (8, 4096) (8, 6)")
    full = list(ds.iter_batches(-1))
    print("全批量批数   :", len(full), "| 该批形状", full[0][0].shape)