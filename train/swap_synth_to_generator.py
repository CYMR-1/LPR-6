# -*- coding: utf-8 -*-
"""一次性迁移：把划分文件中的合成域测试集切换为 ``generator_repo`` 后端。

为什么要"手术式"替换而不是重跑 phase15_split --force
--------------------------------------------------
仓库现存的 ``splits.npz`` 保留的是**历史**强扰动（hard）测试集——它是在
"候选先排序再抽样"这一规范化修复**之前**选出的（见 phase15_split.py 第③步
的注释）。重跑 phase15 会得到同规模但内容不同的 canonical hard 集，导致
历史 48 次运行的 hard_test 指标无法对齐。因此本脚本只替换划分文件中的
**合成域三个数组**（``synth_images / synth_labels / synth_texts``）与
``config`` 里的 synth 溯源字段，其余内容（train/val/test/hard 索引、
标准化统计量）**逐字节保持历史值**。

一致性校验（脚本内置，失败即中止）
----------------------------------
1. 替换后 train/val/test/hard 索引与 standardizer 必须与备份逐字节一致；
2. 新合成集标签全部合法（34 类，无 I/O）；
3. E8 的 24×96 划分文件做同样的替换，且与 32×128 使用**同一批号码**
   （同种子重跑生成器，标签序列相同，仅最终缩放分辨率不同）。

用法
----
    python train/swap_synth_to_generator.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.ccpd_parse import PrepParams
from models.charset import check_label_legal, decode_batch
from models.config import ensure_dirs, load_config, resolve_path
from train.phase15_split import _write_manifest_rows
from train.synth_from_generator import (
    generate_dataset_from_repo,
    provenance_dict,
    resolve_generator_dir,
)
from train.synth_plates import SynthConfig

SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST = "train", "val", "test"
SPLIT_HARD, SPLIT_SYNTH = "hard_test", "synth_test"


def _swap_one(split_path: Path, backup_path: Path, params: PrepParams,
              scfg: SynthConfig, gen_dir: Path, n: int) -> Dict[str, int]:
    """替换单个划分文件中的合成域数组，返回计数统计。"""
    if not backup_path.is_file():
        raise FileNotFoundError(f"缺少划分备份：{backup_path}（无法保留历史 hard 集）")

    print(f"[swap] 生成合成域 {n} 张，尺寸 {params.input_size}（种子 {scfg.seed}）…")
    synth = generate_dataset_from_repo(
        n, scfg, params, generator_dir=gen_dir, out_dir=None, verbose=True)

    for t in synth.texts:
        ok, bad = check_label_legal(t, None)
        if not ok:
            raise AssertionError(f"合成标签非法：{t} 位置 {bad}")

    with np.load(backup_path, allow_pickle=False) as old:
        # synth_texts 是 object 数组，allow_pickle=False 下不能枚举全部键，
        # 只读取需要的非 object 键
        old_arrays = {k: old[k] for k in ("train", "val", "test", "hard")}
        old_std = str(old["standardizer"])
        old_cfg = json.loads(str(old["config"]))

    # 只有 synth 三个数组与 config.synth 溯源字段更新，其余原样保留
    new_cfg = dict(old_cfg)
    new_cfg["synth"] = provenance_dict(scfg)

    np.savez_compressed(
        split_path,
        train=old_arrays["train"], val=old_arrays["val"],
        test=old_arrays["test"], hard=old_arrays["hard"],
        synth_images=synth.images, synth_labels=synth.labels,
        synth_texts=np.asarray(synth.texts, dtype=object),
        standardizer=np.asarray(old_std),
        config=np.asarray(json.dumps(new_cfg, ensure_ascii=False)),
    )

    # ---- 内置校验：除 synth 外逐字节一致 ----------------------------------
    with np.load(split_path, allow_pickle=False) as chk:
        for k in ("train", "val", "test", "hard"):
            if not np.array_equal(chk[k], old_arrays[k]):
                raise AssertionError(f"{split_path.name}: {k} 索引被改动！")
        if str(chk["standardizer"]) != old_std:
            raise AssertionError(f"{split_path.name}: standardizer 被改动！")
    print(f"[swap] {split_path.name} 替换完成并校验通过"
          f"（synth {synth.images.shape}）")
    return {"synth": len(synth.texts), "synth_texts": synth.texts}


def _rebuild_manifest(cfg, cache_path: Path, backup_path: Path,
                      synth_texts: List[str], scfg: SynthConfig,
                      params: PrepParams) -> None:
    """按历史索引 + 新合成集重建 manifest.csv（与 phase15 的行格式一致）。"""
    with np.load(cache_path, allow_pickle=True) as data:
        sources = [str(s) for s in data["sources"]] if "sources" in data else []
        records = ([json.loads(str(r)) for r in data["records"]]
                   if "records" in data else [])
    with np.load(backup_path, allow_pickle=False) as old:
        idx_map = {SPLIT_TRAIN: old["train"], SPLIT_VAL: old["val"],
                   SPLIT_TEST: old["test"], SPLIT_HARD: old["hard"]}

    rows: List[Dict[str, object]] = []
    counter = 0
    for split_name, idx in idx_map.items():
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
    for j, t in enumerate(synth_texts):
        rows.append({
            "index": counter,
            "subset": "synth",
            "source_file": f"generator_repo:43bac43:{scfg.seed}:{j}",
            "label": t,
            "label_classes": "",
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

    _write_manifest_rows(resolve_path(cfg, "manifest"), rows)
    print(f"[swap] manifest.csv 重建完成（{len(rows)} 行）")


def _restore_split_summary(cfg, scfg: SynthConfig) -> None:
    """把 split_summary.json 恢复为 git 中的历史版本，仅更新 synth_config 字段。"""
    log_path = resolve_path(cfg, "logs_dir") / "split_summary.json"
    rel = "reports/logs/split_summary.json"
    raw = subprocess.run(
        ["git", "show", f"HEAD:{rel}"], capture_output=True, text=True,
        encoding="utf-8", cwd=str(Path(__file__).resolve().parent.parent),
    )
    if raw.returncode != 0:
        raise RuntimeError(f"git show HEAD:{rel} 失败：{raw.stderr}")
    summary = json.loads(raw.stdout)
    summary["synth_config"] = provenance_dict(scfg)
    summary["note"] = (summary.get("note", "")
                       + "；合成域已切换为 generator_repo 后端"
                         "（Nenger/chinese_licence_plate_generator @43bac43），"
                         "hard 集保持历史口径不变")
    with open(log_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    print(f"[swap] split_summary.json 已恢复历史版本并更新 synth_config")


def main() -> int:
    cfg = load_config()
    ensure_dirs(cfg)
    scfg = SynthConfig.from_config(cfg)
    gen_dir = resolve_generator_dir(cfg)
    processed = resolve_path(cfg, "processed_dir")
    params = PrepParams.from_config(cfg)

    # 1) 主划分（32×128）
    main_split = processed / "splits.npz"
    res = _swap_one(main_split, processed / "splits.npz.pre_genrepo_bak",
                    params, scfg, gen_dir, n=int(cfg.split.synth_test_size))

    # 2) E8 划分（24×96）：同种子重跑 => 同一批标签，仅分辨率不同
    p24 = PrepParams(rectify_size=params.rectify_size, input_size=(96, 24),
                     keep_right_fraction=params.keep_right_fraction)
    split24 = processed / "splits_24x96.npz"
    res24 = _swap_one(split24, processed / "splits_24x96.npz.pre_genrepo_bak",
                      p24, scfg, gen_dir, n=int(cfg.split.synth_test_size))
    if res24["synth_texts"] != res["synth_texts"]:
        raise AssertionError("24×96 与 32×128 的合成标签序列不一致！")
    print("[swap] 两种分辨率的合成标签序列一致（同种子）")

    # 3) manifest 与 split_summary
    tag = f"ccpd_{params.input_size[0]}x{params.input_size[1]}"
    _rebuild_manifest(cfg, processed / f"{tag}.npz",
                      processed / "splits.npz.pre_genrepo_bak",
                      res["synth_texts"], scfg, params)
    _restore_split_summary(cfg, scfg)

    print("[swap] 全部完成")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
