# -*- coding: utf-8 -*-
"""P1.5 阶段：号码去重划分 + 合成域测试集 + 强扰动测试集 + manifest.csv。

本模块落地 **§2.5 号码去重协议**，该协议优先于任何"按图像随机划分"的方案：

    CCPD 中同一辆车可能有多张不同角度/光照的照片，若按图像随机划分，
    同一车牌号码会同时出现在训练集与测试集，模型只需记住号码纹理即可拿高分，
    测试结论不成立。

三条硬要求（§2.5）
------------------
1. **先按车牌号码字符串去重再划分**：同一号码的全部图片归入同一集合；
2. 所有划分保存随机种子，并把来源文件、子集名、六字符标签、裁剪参数、
   预处理脚本版本写入 ``manifest.csv``，做到每张测试图可追溯到来源；
3. 划分协议必须先于模型训练确定；**合成域测试集与强扰动测试集在调参阶段
   不得参与任何选择决策**。

划分策略
--------
::

    ccpd_base  ──(按号码去重)──┬─ train      (基础训练)
                              ├─ val        (调参：学习率 / L2 / 批大小 / 早停)
                              └─ test       (同分布测试，确认实现正确性)
    ccpd_blur/challenge/rotate/tilt/weather/fn ── hard_test (强扰动，检验鲁棒性)
    synth (生成器)                              ── synth_test (跨域泛化)

实现要点：号码去重键取 ``label``（汉字之后的 **6 位字符**）。CCPD 同一辆车的多帧
照片字符完全相同，因此用 6 位字符串即可唯一标识"同一号码"；这也避免了依赖省份
汉字（本项目不识别汉字）。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# --- 包引导：支持直接运行本文件 ---------------------------------------------
import sys as _sys

if __package__ in (None, ""):
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.ccpd_parse import PrepParams
from models.charset import SEQ_LEN, decode_batch
from models.config import ensure_dirs, load_config, resolve_path
from models.dataset import (
    MANIFEST_COLUMNS,
    GlobalStandardizer,
    PlateDataset,
    read_manifest,
)
from models.augment import build_augment_fn
from train.synth_plates import SynthConfig, generate_dataset

# =============================================================================
# 1. 号码去重划分
# =============================================================================

#: 各集合名称
SPLIT_TRAIN = "train"
SPLIT_VAL = "val"
SPLIT_TEST = "test"
SPLIT_HARD = "hard_test"
SPLIT_SYNTH = "synth_test"


@dataclass
class SplitIndices:
    """一次划分得到的索引集合。

    属性
    ----
    train, val, test : numpy.ndarray
        来自 base 子集的索引（行号，指向 base 缓存）。
    n_unique_plates : dict
        各集合的唯一号码数。
    seed : int
        划分随机种子。
    """

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    n_unique_plates: Dict[str, int]
    seed: int

    def as_dict(self) -> Dict[str, object]:
        """转成可写入 JSON 的摘要（不含完整索引）。"""
        return {
            "n_train": int(len(self.train)),
            "n_val": int(len(self.val)),
            "n_test": int(len(self.test)),
            "n_unique_plates": self.n_unique_plates,
            "split_seed": int(self.seed),
        }


def dedup_split_by_plate(
    labels: np.ndarray,
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int = 42,
) -> SplitIndices:
    """**按车牌号码去重**划分 train / val / test（§2.5 ★核心）。

    算法
    ----
    1. 以标签字符串（6 位）作为号码键，把同键样本聚成一组；
    2. 打乱**组**的顺序（而不是样本顺序），保证同一号码整体落在一个集合；
    3. 按组依次填充 train → val → test，直到各自达到目标规模。

    参数
    ----
    labels : numpy.ndarray
        形状 ``(N, 6)`` 的类别索引。
    train_size, val_size, test_size : int
        三个集合的目标规模（以**图像数**计）。
    seed : int
        划分随机种子。

    返回
    ----
    SplitIndices
        三个索引数组与唯一号码数统计。

    形状
    ----
    ``(N, 6)`` -> 三个 ``(M,)`` 索引数组

    异常
    ------
    ValueError
        可用样本不足以满足请求规模时抛出（避免静默产出不完整划分）。
    """
    n = int(labels.shape[0])
    texts = decode_batch(labels)

    # ---- ① 按号码分组 -----------------------------------------------------
    groups: Dict[str, List[int]] = {}
    for i, t in enumerate(texts):
        groups.setdefault(t, []).append(i)

    keys = np.array(sorted(groups.keys()), dtype=object)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(keys)

    # ---- ② 按组填充三个集合 ----------------------------------------------
    train: List[int] = []
    val: List[int] = []
    test: List[int] = []
    for k in keys:
        members = groups[str(k)]
        if len(train) < train_size:
            train.extend(members)
        elif len(val) < val_size:
            val.extend(members)
        elif len(test) < test_size:
            test.extend(members)
        else:
            break

    if len(train) < train_size or len(val) < val_size or len(test) < test_size:
        raise ValueError(
            f"样本不足：可用 {n} 张 / {len(groups)} 个唯一号码，"
            f"请求 train={train_size} val={val_size} test={test_size}，"
            f"实际得到 {len(train)}/{len(val)}/{len(test)}。请调小 split.* 配置。"
        )

    train_arr = np.asarray(sorted(train), dtype=np.int64)
    val_arr = np.asarray(sorted(val), dtype=np.int64)
    test_arr = np.asarray(sorted(test), dtype=np.int64)

    # ---- ③ 切分后再次自检：号码不得跨集 --------------------------------
    def _keys_of(idx: np.ndarray) -> set:
        return {texts[int(i)] for i in idx}

    k_tr, k_va, k_te = _keys_of(train_arr), _keys_of(val_arr), _keys_of(test_arr)
    inter_tv = k_tr & k_va
    inter_tt = k_tr & k_te
    inter_vt = k_va & k_te
    if inter_tv or inter_tt or inter_vt:
        raise AssertionError(
            f"号码去重协议被破坏：train∩val={len(inter_tv)} "
            f"train∩test={len(inter_tt)} val∩test={len(inter_vt)}"
        )

    return SplitIndices(
        train=train_arr,
        val=val_arr,
        test=test_arr,
        n_unique_plates={
            "train": len(k_tr), "val": len(k_va), "test": len(k_te),
            "base_total": len(groups),
        },
        seed=int(seed),
    )


def assert_no_leakage(
    labels: np.ndarray,
    split_assignment: Sequence[str],
) -> Dict[str, object]:
    """独立复核号码是否跨集（供测试脚本再次验证，§2.5 硬要求 1）。

    参数
    ----
    labels : numpy.ndarray
        形状 ``(N, 6)``。
    split_assignment : Sequence[str]
        长度 ``N``，每行的集合名。

    返回
    ----
    dict
        含各集合唯一号码数与两两交集大小；若存在泄漏会抛 ``AssertionError``。

    形状
    ----
    ``(N, 6)`` + ``list[str]`` -> dict
    """
    texts = decode_batch(labels)
    by_split: Dict[str, set] = {}
    for t, s in zip(texts, split_assignment):
        by_split.setdefault(s, set()).add(t)

    names = sorted(by_split.keys())
    report: Dict[str, object] = {
        "unique_plates": {k: len(v) for k, v in by_split.items()},
        "intersections": {},
    }
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            inter = by_split[a] & by_split[b]
            report["intersections"][f"{a}∩{b}"] = len(inter)
            if inter:
                raise AssertionError(f"号码跨集泄漏：{a} ∩ {b} = {len(inter)} 个号码")

    # 同一集合内部不应有重复号码占据多张图之外的问题；此处只记录
    report["image_counts"] = {
        k: sum(1 for s in split_assignment if s == k) for k in names
    }
    return report


# =============================================================================
# 2. 主流程
# =============================================================================


def _load_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, List[str], List[dict]]:
    """读取预处理缓存（含来源文件名与记录）。

    参数
    ----
    path : Path
        ``.npz`` 路径。

    返回
    ----
    tuple
        ``(images (N,H,W) uint8, labels (N,6) int64, sources list[str], records list[dict])``。
    """
    with np.load(path, allow_pickle=True) as data:
        images = data["images"]
        labels = data["labels"]
        sources = [str(s) for s in data["sources"]] if "sources" in data else []
        records = ([json.loads(str(r)) for r in data["records"]]
                   if "records" in data else [])
    return images, labels, sources, records


def _subset_of_records(records: List[dict], idx: np.ndarray) -> List[dict]:
    """按索引取记录子集。"""
    return [records[int(i)] for i in idx]


def _write_manifest_rows(
    out_path: Path,
    rows: List[Dict[str, object]],
) -> None:
    """写出 manifest.csv（列顺序固定为 MANIFEST_COLUMNS）。

    参数
    ----
    out_path : Path
        输出路径。
    rows : list of dict
        行字典列表。

    返回
    ----
    None
    """
    import csv

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in MANIFEST_COLUMNS})


def main(argv: Optional[Sequence[str]] = None) -> int:
    """P1.5 主入口。

    参数
    ----
    argv : Sequence[str] or None
        命令行参数。

    返回
    ----
    int
        退出码。
    """
    ap = argparse.ArgumentParser(description="P1.5：号码去重划分 + 合成域 + 强扰动")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--force", action="store_true", help="覆盖已存在的划分产物")
    ap.add_argument("--out-name", type=str, default="splits.npz",
                    help="划分文件名；E8 的 24×96 口径用 splits_24x96.npz，"
                         "避免覆盖主划分")
    ap.add_argument("--skip-manifest", action="store_true",
                    help="不写 manifest.csv（E8 重建划分时用：主 manifest 属于"
                         " 32×128 口径，不能被覆盖）")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    params = PrepParams.from_config(cfg)
    processed = resolve_path(cfg, "processed_dir")
    tag = f"ccpd_{params.input_size[0]}x{params.input_size[1]}"
    cache_path = processed / f"{tag}.npz"
    if not cache_path.exists():
        print(f"[P1.5] 错误：找不到预处理缓存 {cache_path}")
        print("       请先运行： python train/phase1_prepare.py")
        return 2

    images, labels, sources, records = _load_npz(cache_path)
    print(f"[P1.5] 载入缓存 {cache_path}: images{images.shape} labels{labels.shape}")

    # ---- ① 按子集拆分 base 与 hard ---------------------------------------
    subsets = np.array([r.get("subset", "") for r in records], dtype=object)
    base_mask = subsets == "ccpd_base"
    hard_mask = np.isin(subsets, [
        "ccpd_blur", "ccpd_challenge", "ccpd_rotate",
        "ccpd_tilt", "ccpd_weather", "ccpd_fn",
    ])
    base_idx = np.flatnonzero(base_mask)
    hard_idx_all = np.flatnonzero(hard_mask)
    print(f"[P1.5] base 样本 {len(base_idx)}，强扰动候选 {len(hard_idx_all)}")

    if len(base_idx) == 0:
        print("[P1.5] 错误：没有 ccpd_base 样本")
        return 2

    # ---- ② 号码去重划分 ---------------------------------------------------
    si = dedup_split_by_plate(
        labels[base_idx],
        train_size=int(cfg.split.train_size),
        val_size=int(cfg.split.val_size),
        test_size=int(cfg.split.test_size),
        seed=int(cfg.split.split_seed),
    )
    train_idx = base_idx[si.train]
    val_idx = base_idx[si.val]
    test_idx = base_idx[si.test]

    # ---- ③ 强扰动测试集：按号码去重后采样（指标签去重、不跨集即可） -----
    hard_size = int(cfg.split.hard_test_size)
    rng = np.random.default_rng(int(cfg.split.split_seed) + 1)
    hard_texts = decode_batch(labels[hard_idx_all])
    # 强扰动集与 base 三集合的号码不得重叠，避免"见过同号码"
    base_keys = set(decode_batch(labels[base_idx]))
    hard_candidates = np.array(
        [i for i, t in zip(hard_idx_all, hard_texts) if t not in base_keys],
        dtype=np.int64,
    )
    if len(hard_candidates) > hard_size:
        hard_idx = np.sort(rng.choice(hard_candidates, size=hard_size, replace=False))
    else:
        hard_idx = hard_candidates
    print(f"[P1.5] 强扰动测试集 {len(hard_idx)} 张"
          f"（候选 {len(hard_candidates)}，已排除与 base 号码重叠者）")

    # ---- ④ 合成域测试集 ---------------------------------------------------
    scfg = SynthConfig.from_config(cfg)
    synth_size = int(cfg.split.synth_test_size)
    print(f"[P1.5] 生成合成域测试集 {synth_size} 张（种子 {scfg.seed}）…")
    synth = generate_dataset(
        synth_size, scfg, params, out_dir=None, verbose=True,
    )
    print(f"[P1.5] 合成域：images{synth.images.shape} labels{synth.labels.shape}")

    # ---- ⑤ 全局标准化统计量：**只用训练集**（§2.3.2） --------------------
    train_images = images[train_idx].astype(np.float32) / 255.0
    standardizer = GlobalStandardizer.fit(train_images)
    print(f"[P1.5] 训练集标准化统计量：mean={standardizer.mean:.6f} "
          f"std={standardizer.std:.6f}（n={standardizer.n_samples}，仅训练集）")

    # ---- ⑥ 保存划分与标准化器 --------------------------------------------
    split_path = processed / str(args.out_name)
    if split_path.exists() and not args.force:
        print(f"[P1.5] 错误：{split_path} 已存在；如需覆盖请加 --force")
        return 2
    np.savez_compressed(
        split_path,
        train=train_idx, val=val_idx, test=test_idx, hard=hard_idx,
        synth_images=synth.images, synth_labels=synth.labels,
        synth_texts=np.asarray(synth.texts, dtype=object),
        standardizer=np.asarray(json.dumps(standardizer.as_dict(), ensure_ascii=False)),
        config=np.asarray(json.dumps({
            "preprocess_version": cfg.project.preprocess_version,
            "split_seed": int(cfg.split.split_seed),
            "synth": scfg.as_dict(),
            "input_size": list(params.input_size),
        }, ensure_ascii=False)),
    )
    print(f"[P1.5] 划分已写出：{split_path} ({split_path.stat().st_size / 1024 ** 2:.1f} MB)")

    # ---- ⑦ 写 manifest.csv ------------------------------------------------
    rows: List[Dict[str, object]] = []
    all_idx = [(SPLIT_TRAIN, train_idx), (SPLIT_VAL, val_idx),
               (SPLIT_TEST, test_idx), (SPLIT_HARD, hard_idx)]
    counter = 0
    for split_name, idx in all_idx:
        for i in idx:
            rec = records[int(i)]
            corners = rec.get("corners", [])
            rows.append({
                "index": counter,
                "subset": rec.get("subset", ""),
                "source_file": sources[int(i)] if int(i) < len(sources) else "",
                "label": rec.get("label", ""),
                "label_classes": " ".join(str(c) for c in rec.get("label_classes", [])),
                "split": split_name,
                "plate_number": " ".join(str(v) for v in rec.get("label_indices", [])),
                "plate_hash": rec.get("label", ""),
                "corners": ";".join(f"{float(x):.1f},{float(y):.1f}" for x, y in corners),
                "rectify_size": f"{params.rectify_size[0]}x{params.rectify_size[1]}",
                "input_size": f"{params.input_size[0]}x{params.input_size[1]}",
                "crop_x0": params.crop_x0,
                "keep_right_fraction": f"{params.keep_right_fraction:.6f}",
                "brightness": rec.get("brightness", ""),
                "blurriness": rec.get("blurriness", ""),
                "quad_area": f"{float(rec.get('quad_area', 0.0)):.1f}",
                "preprocess_version": cfg.project.preprocess_version,
                "split_seed": int(cfg.split.split_seed),
            })
            counter += 1
    # 合成域：来源是生成器，没有原图路径，用生成参数溯源
    for j, t in enumerate(synth.texts):
        rows.append({
            "index": counter,
            "subset": "synth",
            "source_file": f"generated:{scfg.seed}:{j}",
            "label": t,
            "label_classes": " ".join(str(int(c)) for c in synth.labels[j]),
            "split": SPLIT_SYNTH,
            "plate_number": "",
            "plate_hash": t,
            "corners": "",
            "rectify_size": f"{params.rectify_size[0]}x{params.rectify_size[1]}",
            "input_size": f"{params.input_size[0]}x{params.input_size[1]}",
            "crop_x0": params.crop_x0,
            "keep_right_fraction": f"{params.keep_right_fraction:.6f}",
            "brightness": "",
            "blurriness": "",
            "quad_area": "",
            "preprocess_version": cfg.project.preprocess_version,
            "split_seed": int(cfg.split.split_seed),
        })
        counter += 1

    manifest_path = resolve_path(cfg, "manifest")
    if args.skip_manifest:
        print("[P1.5] --skip-manifest：跳过 manifest.csv 写出"
              "（主 manifest 属于 32×128 口径，不覆盖）")
    else:
        _write_manifest_rows(manifest_path, rows)
        print(f"[P1.5] manifest 已写出：{manifest_path}（{len(rows)} 行）")

    # ---- ⑧ 泄漏自检与摘要 -------------------------------------------------
    base_split_assignment = (
        [SPLIT_TRAIN] * len(train_idx)
        + [SPLIT_VAL] * len(val_idx)
        + [SPLIT_TEST] * len(test_idx)
    )
    base_labels_all = labels[np.concatenate([train_idx, val_idx, test_idx])]
    leak = assert_no_leakage(base_labels_all, base_split_assignment)
    # 强扰动集也必须与 base 无重叠
    hard_leak = set(synth.texts) & set()
    hard_keys = set(decode_batch(labels[hard_idx]))
    overlap = hard_keys & base_keys
    if overlap:
        raise AssertionError(f"强扰动集与 base 号码重叠 {len(overlap)} 个")
    synth_overlap = set(synth.texts) & (base_keys | hard_keys)
    if synth_overlap:
        print(f"[P1.5] 提示：合成域与真实集有 {len(synth_overlap)} 个号码巧合重复"
              f"（概率事件，不构成泄漏：字体/成像完全不同）")

    summary = {
        "counts": {
            SPLIT_TRAIN: int(len(train_idx)),
            SPLIT_VAL: int(len(val_idx)),
            SPLIT_TEST: int(len(test_idx)),
            SPLIT_HARD: int(len(hard_idx)),
            SPLIT_SYNTH: int(len(synth.texts)),
        },
        "unique_plates": leak["unique_plates"],
        "intersections": leak["intersections"],
        "hard_overlap_with_base": len(overlap),
        "synth_overlap_with_real": len(synth_overlap),
        "standardizer": standardizer.as_dict(),
        "split_seed": int(cfg.split.split_seed),
        "preprocess_version": cfg.project.preprocess_version,
        "synth_config": scfg.as_dict(),
        "note": "合成域与强扰动集不参与任何调参决策（§2.5 要求 3）",
    }
    log_path = resolve_path(cfg, "logs_dir") / "split_summary.json"
    if str(args.out_name) != "splits.npz":
        # ★ 非主划分（如 E8 的 splits_24x96.npz）不得覆盖主摘要，
        # 否则主口径的标准化统计量会被别的分辨率覆盖（实测踩坑）。
        log_path = log_path.with_name(
            "split_summary_" + Path(str(args.out_name)).stem.replace("splits_", "") + ".json")
    with open(log_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    print(f"[P1.5] 划分摘要已写出：{log_path}")

    print()
    print("=" * 72)
    print("P1.5 号码去重划分完成（§2.5 ★）")
    for k, v in summary["counts"].items():
        print(f"  {k:12s} {v:7d} 张")
    print(f"  唯一号码数: {summary['unique_plates']}")
    print(f"  两两交集  : {summary['intersections']}   ← 必须全为 0")
    print(f"  标准化统计: mean={standardizer.mean:.6f} std={standardizer.std:.6f}（仅训练集）")
    print("=" * 72)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())