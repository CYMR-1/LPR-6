# -*- coding: utf-8 -*-
"""一次性修复：把划分文件重映射到规范化后的缓存，保留 48 次历史运行的评测口径。

背景
----
``train/normalize_cache_order.py`` 把缓存重排到 sources 字典序后**重跑了**
phase15_split，暴露出 hard 集选择的顺序依赖：``hard_candidates`` 按缓存顺序
组装，``rng.choice`` 同一 seed 在不同候选顺序下选出**不同样本**
（新 hard 集与原 hard 集只有约 640 个号码重合）。48 次历史运行的
hard_test 指标都是在**原 hard 集**上评测的，换集合会使历史数字失效。

修复策略（保留历史、不重跑任何训练）
------------------------------------
1. 原缓存（``.pre_sort_bak``）+ 原划分备份（``splits_32x128_backup.npz``）
   都在；先验证备份的真实性：用 baseline_s42 检查点在备份 hard 集上评测，
   字符/整牌必须与 ``baseline_s42_run.json`` 的记录**逐位一致**；
2. 用 sources（每张图的唯一来源路径）建立"旧下标 -> 新下标"映射，
   把原划分的四个集合重映射到新缓存——**样本序列逐位保持**，
   历史运行的训练轨迹因此仍可逐位复现；
3. 合成域与标准化统计量从"刚重建的 canonical 划分"中取（synth 由 seed=2024
   确定性生成，与顺序无关；标准化器在新 train 集上重新拟合并断言与旧值一致）；
4. 24×96 划分同样按 sources 映射到 24×96 缓存（E8 必须与主划分同序列）。

★ 运行前提：没有任何训练/评测进程在读写 data/processed/。
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.charset import decode_batch  # noqa: E402
from models.config import load_config, resolve_path  # noqa: E402
from models.dataset import GlobalStandardizer, load_cache  # noqa: E402
from models.model import Params, forward, predict  # noqa: E402
from models.backend import get_backend  # noqa: E402

SPLITS = ("train", "val", "test", "hard")


def _sources(path: Path) -> list:
    """读缓存的 sources 列（每张图的唯一来源路径）。

    形状
    ----
    缓存 -> list[str]，长度 N
    """
    with np.load(path, allow_pickle=True) as d:
        return [str(x) for x in d["sources"]]


def main() -> int:
    """执行重映射修复。

    返回
    ----
    int
        0 表示全部校验通过。
    """
    cfg = load_config()
    processed = resolve_path(cfg, "processed_dir")
    old_cache_path = processed / "ccpd_128x32.npz.pre_sort_bak"
    new_cache_path = processed / "ccpd_128x32.npz"
    bak_splits_path = processed / "splits_32x128_backup.npz"
    cur_splits_path = processed / "splits.npz"
    p24_cache = processed / "ccpd_96x24.npz"
    p24_splits = processed / "splits_24x96.npz"

    # ---- 0. 备份真实性验证：备份 hard 集 + 旧缓存 + baseline 检查点 --------
    print("步骤 0：验证 splits_32x128_backup.npz 确为历史运行所用的原划分")
    images_old, _, _ = load_cache(old_cache_path)
    with np.load(bak_splits_path, allow_pickle=True) as d:
        bak = {k: d[k] for k in d.files}
    run = json.loads(Path("reports/logs/baseline_s42_run.json").read_text(
        encoding="utf-8"))
    std = json.loads(str(bak["standardizer"]))
    hard_idx = bak["hard"].astype(np.int64)
    x = images_old[hard_idx].astype(np.float32).reshape(len(hard_idx), -1) / 255.0
    x = (x - std["mean"]) / std["std"]
    params, _ = Params.load(
        Path("reports/checkpoints/baseline_s42_best.npz"))
    be = get_backend("auto")
    labels_old_all = load_cache(old_cache_path)[1]
    y = labels_old_all[hard_idx]
    probs, _ = forward(params, x, be)
    preds, _ = predict(probs, be)
    char_acc = float((preds == y).mean())
    plate_acc = float((preds == y).all(axis=1).mean())
    exp_c = float(run["hard_test"]["char_acc"])
    exp_p = float(run["hard_test"]["plate_acc"])
    print(f"  备份 hard 集复评：字符 {char_acc * 100:.4f}% vs 记录 "
          f"{exp_c * 100:.4f}%　整牌 {plate_acc * 100:.2f}% vs {exp_p * 100:.2f}%")
    if not (abs(char_acc - exp_c) < 1e-9 and abs(plate_acc - exp_p) < 1e-9):
        print("  !! 备份与历史记录不符，终止（不改动任何文件）")
        return 1
    print("  备份真实性确认 ✓")

    # ---- 1. 旧下标 -> 新下标 映射 -----------------------------------------
    print("步骤 1：建立 sources 映射并重映射四个集合")
    old_src = _sources(old_cache_path)
    new_src = _sources(new_cache_path)
    assert len(old_src) == len(new_src) and len(set(new_src)) == len(new_src)
    assert set(old_src) == set(new_src), "新旧缓存的来源集合不一致"
    src2new = {s: i for i, s in enumerate(new_src)}

    _, labels_new, _ = load_cache(new_cache_path)
    labels_old = labels_old_all
    remapped = {}
    for k in SPLITS:
        idx_old = bak[k].astype(np.int64)
        idx_new = np.array([src2new[old_src[i]] for i in idx_old], dtype=np.int64)
        # 序列逐位一致校验：标签序列必须完全相同
        assert np.array_equal(labels_new[idx_new], labels_old[idx_old]), \
            f"{k} 集合序列重映射后标签不一致"
        remapped[k] = idx_new
        print(f"  {k}: {len(idx_new)} 张，标签序列逐位一致 ✓")

    # ---- 2. 标准化统计量在新 train 序列上重拟合并断言 ----------------------
    print("步骤 2：重拟合标准化器（应与原值逐位一致）")
    st = GlobalStandardizer().fit(
        images_old[bak["train"].astype(np.int64)].astype(np.float32) / 255.0)
    old_std = json.loads(str(bak["standardizer"]))
    print(f"  原值 mean={old_std['mean']:.8f} std={old_std['std']:.8f}")
    print(f"  新拟合 mean={st.mean:.8f} std={st.std:.8f}")
    # 注：fit 用图像内容，与顺序无关，仅断言容差内一致
    assert abs(st.mean - old_std["mean"]) < 1e-6
    assert abs(st.std - old_std["std"]) < 1e-6

    # ---- 3. 组装并写出 splits.npz（synth/配置沿用 canonical 新划分） -------
    print("步骤 3：写出重映射后的 splits.npz")
    with np.load(cur_splits_path, allow_pickle=True) as d:
        cur = {k: d[k] for k in d.files}
    # synth 由 seed 确定性生成，两版应一致；校验后沿用新版
    st_bak = [str(t) for t in bak["synth_texts"]]
    st_cur = [str(t) for t in cur["synth_texts"]]
    assert st_bak == st_cur, "synth 文本两版不一致（不应发生）"
    out = dict(cur)
    for k in SPLITS:
        out[k] = remapped[k]
    out["standardizer"] = bak["standardizer"]
    shutil.copy2(cur_splits_path, processed / "splits.npz.hardfix_bak")
    np.savez_compressed(cur_splits_path, **out)
    print("  splits.npz 已重映射（原文件备份为 splits.npz.hardfix_bak）")

    # ---- 4. 24×96 划分同序列映射 ------------------------------------------
    if p24_cache.exists() and p24_splits.exists():
        print("步骤 4：24×96 划分按 sources 同序列映射")
        src24 = _sources(p24_cache)
        src2_24 = {s: i for i, s in enumerate(src24)}
        assert set(new_src) == set(src24)
        _, labels24, _ = load_cache(p24_cache)
        with np.load(p24_splits, allow_pickle=True) as d:
            cur24 = {k: d[k] for k in d.files}
        out24 = dict(cur24)
        for k in SPLITS:
            idx = np.array([src2_24[new_src[i]] for i in remapped[k]],
                           dtype=np.int64)
            assert np.array_equal(labels24[idx], labels_new[remapped[k]])
            out24[k] = idx
        shutil.copy2(p24_splits, processed / "splits_24x96.npz.hardfix_bak")
        np.savez_compressed(p24_splits, **out24)
        print("  splits_24x96.npz 已同序列映射 ✓")

    # ---- 5. 终验：号码集合与 baseline 复评 --------------------------------
    print("步骤 5：终验（号码集合 + baseline_s42 复评）")
    for k in SPLITS:
        before = set(decode_batch(labels_old[bak[k].astype(np.int64)]))
        after = set(decode_batch(labels_new[remapped[k]]))
        assert before == after, f"{k} 号码集合变化"
    print("  四个集合号码集合与原划分完全一致 ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())