# -*- coding: utf-8 -*-
"""一次性修复：把既有数据产物规范化到"按 sources 字典序"的标准顺序。

背景
----
phase1_prepare.py 之前按多进程完成顺序组装缓存，样本顺序随运行时机不同
（E8 重建 24×96 缓存时实测 25592/28341 个位置与 32×128 缓存不同）。
由于 data/ 不入库、划分文件按下标引用样本，任何人在本机重建数据后都会
得到不同的训练轨迹（统计可复现、逐位不复现）。

phase1_prepare.py 已修复为统一按 sources 排序；本脚本把**既有**产物迁移到
该标准顺序，使"未来重建"与"当前已提交结果"逐位一致：

1. 重排 ``ccpd_128x32.npz`` / ``ccpd_96x24.npz``（若已是字典序则跳过）；
2. 重跑 ``train/phase15_split.py``（32×128 用 --force 覆盖；24×96 用
   ``--out-name splits_24x96.npz --skip-manifest``）。划分逻辑对标签是
   确定性的，重跑产物即标准产物；
3. 校验：新旧划分的**号码集合**逐集合一致（划分的不变量），并用
   baseline_s42 检查点重评测试集，指标必须与修复前逐位一致。

★ 运行前提：没有任何训练/评测进程在读写 data/processed/（避免半新半旧）。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.ccpd_parse import PrepParams  # noqa: E402
from models.charset import decode_batch  # noqa: E402
from models.config import load_config, resolve_path  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def _run_module(module: str, extra: list) -> int:
    """以 ``-m`` 形式运行项目模块（sys.path[0]=项目根，避免 train/train.py
    遮蔽 ``train`` 包）。"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return subprocess.run([sys.executable, "-u", "-m", module] + list(extra),
                          cwd=str(ROOT), env=env).returncode


def normalize_cache(path: Path) -> bool:
    """把单个缓存重排到 sources 字典序。

    参数
    ----
    path : Path
        缓存文件路径。

    返回
    ----
    bool
        True 表示文件内容最终处于标准顺序（无论是否发生了重排）。
    """
    d = np.load(path, allow_pickle=True)
    sources = [str(x) for x in d["sources"]]
    order = sorted(range(len(sources)), key=lambda i: sources[i])
    if order == list(range(len(sources))):
        print(f"  {path.name}: 已是标准顺序，跳过")
        return True
    bak = path.with_suffix(".npz.pre_sort_bak")
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"  {path.name}: 备份 -> {bak.name}")
    out = {k: d[k] for k in d.files}
    perm = np.array(order, dtype=np.int64)
    for k in ("images", "labels", "sources", "records"):
        a = out.get(k)
        if a is not None and getattr(a, "ndim", 0) > 0 and len(a) == len(perm):
            out[k] = a[perm]
    np.savez_compressed(path, **out)
    print(f"  {path.name}: 已重排到标准顺序")
    return True


def split_plate_sets(split_path: Path, labels: np.ndarray) -> dict:
    """读出划分文件每个集合的号码集合（用于修复前后对比）。

    参数
    ----
    split_path : Path
        划分文件。
    labels : np.ndarray
        缓存标签 ``(N, 6)``。

    返回
    ----
    dict
        ``{split: set[str]}``。
    """
    with np.load(split_path, allow_pickle=False) as d:
        return {k: set(decode_batch(labels[d[k].astype(np.int64)]))
                for k in ("train", "val", "test", "hard")}


def main() -> int:
    """执行规范化迁移。

    返回
    ----
    int
        0 表示成功。
    """
    cfg = load_config()
    processed = resolve_path(cfg, "processed_dir")
    p32 = processed / "ccpd_128x32.npz"
    p24 = processed / "ccpd_96x24.npz"
    split_main = processed / "splits.npz"
    split_24 = processed / "splits_24x96.npz"

    # ---- 0. 修复前的号码集合快照（不变量） -------------------------------
    from models.dataset import load_cache
    _, labels_old, _ = load_cache(p32)
    before = split_plate_sets(split_main, labels_old)
    print("修复前号码集合快照：train/val/test/hard = "
          + "/".join(str(len(before[k])) for k in ("train", "val", "test", "hard")))

    # ---- 1. 重排两个缓存 -------------------------------------------------
    print("步骤 1：规范化缓存顺序")
    normalize_cache(p32)
    if p24.exists():
        normalize_cache(p24)

    # ---- 2. 重建划分（确定性逻辑 + 标准顺序缓存 = 标准产物） --------------
    print("步骤 2：重建 32×128 划分（--force）")
    rc = _run_module("train.phase15_split", ["--force"])
    if rc != 0:
        return rc
    if p24.exists():
        print("步骤 2b：重建 24×96 划分")
        rc = _run_module("train.phase15_split", [
            "--config", str(ROOT / "configs" / "_e8_24x96.yaml"), "--force",
            "--out-name", "splits_24x96.npz", "--skip-manifest"])
        if rc != 0:
            return rc

    # ---- 3. 校验号码集合不变量 -------------------------------------------
    _, labels_new, _ = load_cache(p32)
    after = split_plate_sets(split_main, labels_new)
    ok = all(before[k] == after[k] for k in before)
    print("步骤 3：修复前后各集合号码集合一致 =", ok)
    if not ok:
        for k in before:
            print(f"  {k}: |前|={len(before[k])} |后|={len(after[k])} "
                  f"仅前={len(before[k] - after[k])} 仅后={len(after[k] - before[k])}")
        return 1

    # ---- 4. 用既有检查点复评，指标必须逐位一致 ----------------------------
    print("步骤 4：用 baseline_s42 检查点复评测试集（应与修复前逐位一致）")
    rc = _run_module("evaluate.main", ["--run", "baseline_s42"])
    if rc != 0:
        return rc
    ev = json.loads((resolve_path(cfg, "logs_dir")
                     / "baseline_s42_eval.json").read_text(encoding="utf-8"))
    tc = ev["datasets"]["test"]["char_acc"] * 100
    tp = ev["datasets"]["test"]["plate_acc"] * 100
    print(f"  复评 test：字符 {tc:.4f}% 整牌 {tp:.2f}%"
          f"（修复前 86.7333% / 42.60%，必须一致）")
    ok2 = abs(tc - 86.7333) < 1e-6 and abs(tp - 42.60) < 1e-9
    print("  逐位一致 =", ok2)
    return 0 if ok2 else 1


if __name__ == "__main__":
    raise SystemExit(main())