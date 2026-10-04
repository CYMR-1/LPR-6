# -*- coding: utf-8 -*-
"""一次性修复：把被 24×96 口径覆盖的 split_summary.json 恢复为 32×128 主口径。

起因：phase15_split.py 曾把摘要固定写到 ``split_summary.json``，
E8 生成 splits_24x96.npz 时把主摘要覆盖成了 24×96 口径
（standardizer 0.4212/0.2156 vs 主口径 0.4212/0.2212）。
phase15_split.py 已修复（非主划分自动改写 split_summary_<tag>.json），
本脚本从**冻结的 splits.npz 与缓存**只读重算主口径摘要并写回。

只写 reports/logs/split_summary.json，不触碰任何划分/缓存文件。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.charset import decode_batch  # noqa: E402
from models.config import load_config, resolve_path  # noqa: E402
from models.dataset import load_cache  # noqa: E402


def main() -> int:
    """重建主口径 split_summary.json。

    返回
    ----
    int
        0 表示成功。
    """
    cfg = load_config()
    processed = resolve_path(cfg, "processed_dir")

    _, labels, _ = load_cache(processed / "ccpd_128x32.npz")
    # synth_texts 是 object 数组（字符串列表），必须 allow_pickle=True
    with np.load(processed / "splits.npz", allow_pickle=True) as d:
        idx = {k: d[k].astype(np.int64) for k in ("train", "val", "test", "hard")}
        synth_texts = [str(t) for t in d["synth_texts"]]
        std = json.loads(str(d["standardizer"]))

    texts = np.asarray(decode_batch(labels), dtype=object)
    sets = {k: set(texts[v].tolist()) for k, v in idx.items()}
    keys = ("train", "val", "test", "hard")
    inter = {f"{a}∩{b}": len(sets[a] & sets[b])
             for i, a in enumerate(keys) for b in keys[i + 1:]}
    synth_overlap = set(synth_texts) & (sets["train"] | sets["val"]
                                        | sets["test"] | sets["hard"])

    summary = {
        "counts": {k: int(len(v)) for k, v in idx.items()},
        "unique_plates": {k: len(sets[k]) for k in keys},
        "intersections": inter,
        "hard_overlap_with_base": len(
            sets["hard"] & (sets["train"] | sets["val"] | sets["test"])),
        "synth_overlap_with_real": len(synth_overlap),
        "standardizer": std,
        "split_seed": int(cfg.split.split_seed),
        "preprocess_version": cfg.project.preprocess_version,
        "input_size": [int(cfg.ccpd.input_size[0]), int(cfg.ccpd.input_size[1])],
        "note": "合成域与强扰动集不参与任何调参决策（§2.5 要求 3）；"
                "本文件由 train/restore_split_summary.py 从冻结的 splits.npz "
                "只读重建（曾被 24×96 口径覆盖，phase15_split.py 已修复）",
        "restored_from_frozen_split": True,
    }
    out = resolve_path(cfg, "logs_dir") / "split_summary.json"
    with open(out, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    print(f"已重建 {out}")
    print(f"  standardizer: mean={std['mean']:.6f} std={std['std']:.6f}")
    print(f"  synth_overlap_with_real={len(synth_overlap)}（32×128 口径应为 0）")
    print(f"  intersections: {inter}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())