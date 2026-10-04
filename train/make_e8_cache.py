# -*- coding: utf-8 -*-
"""E8 需要的 24×96 输入尺寸预处理缓存与划分文件生成脚本。

为什么需要单独一步
------------------
E8 比较输入尺寸 32×128 与 24×96。预处理缓存按输入尺寸命名
（``ccpd_<W>x<H>.npz``），所以 24×96 必须重新从 CCPD 原图裁一遍。
与之配套，合成域测试集与划分文件（含只在训练集上拟合的标准化
统计量）也必须在 24×96 下重建，否则特征维度不一致。

本脚本做三件事：
1. 生成 ``configs/_e8_24x96.yaml``（只改 ``ccpd.input_size``）；
2. 用该配置跑 ``train/phase1_prepare.py`` 生成 ``ccpd_96x24.npz``；
3. 用 ``--out-name splits_24x96.npz --skip-manifest`` 跑
   ``train/phase15_split.py``，生成 24×96 口径的划分文件，
   **不触碰**主 ``splits.npz`` 与 ``manifest.csv``（无共享文件竞争）。

★ 划分索引的一致性由 ``split.split_seed`` 保证：phase15_split 的号码去重
排序是确定性的，因此 24×96 与 32×128 的 train/val/test/hard 索引完全相同
（本脚本会显式核对，不一致就报错而不是静默继续）。

训练侧的配合：``configs/default.yaml`` 的 E8 size_24x96 变体已通过
``paths.splits_file: splits_24x96.npz`` 指向该划分文件。

用法
----
    python train/make_e8_cache.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.config import load_config, resolve_path  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TARGET = (96, 24)          # (W, H)
SPLIT_24 = "splits_24x96.npz"


def _run(cmd: list) -> int:
    """在项目根目录下运行子进程。

    参数
    ----
    cmd : list of str
        命令与参数（子脚本请用 ``-m train.xxx`` 模块形式，见 main 注释）。

    返回
    ----
    int
        子进程退出码。
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return subprocess.run(cmd, cwd=str(ROOT), env=env).returncode


def main() -> int:
    """生成 24×96 缓存与划分文件。

    返回
    ----
    int
        0 表示成功。
    """
    cfg = load_config()
    processed = resolve_path(cfg, "processed_dir")
    base_split = processed / "splits.npz"
    if not base_split.exists():
        print("主划分 splits.npz 不存在，请先跑 P1.5（32×128 口径）")
        return 2

    # 1) 写临时配置
    raw = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    raw["ccpd"]["input_size"] = [TARGET[0], TARGET[1]]
    tmp_cfg = ROOT / "configs" / "_e8_24x96.yaml"
    tmp_cfg.write_text(
        yaml.safe_dump(raw, allow_unicode=True, sort_keys=False),
        encoding="utf-8")
    print(f"已写临时配置：{tmp_cfg}  input_size={list(TARGET)}")

    # 2) 预处理 24×96（已存在则跳过）
    # ★ 必须用 ``-m train.phase1_prepare`` 而不是 ``python train/phase1_prepare.py``：
    #    后者把 train/ 放到 sys.path[0]，而 train/ 里恰好有 train.py，
    #    ``import train`` 会解析成那个模块而不是包（"train is not a package"）。
    cache = processed / f"ccpd_{TARGET[0]}x{TARGET[1]}.npz"
    if cache.exists():
        print(f"[1/2] 缓存已存在，跳过预处理：{cache.name}")
    else:
        print("[1/2] 预处理 24×96 ……")
        rc = _run([sys.executable, "-u", "-m", "train.phase1_prepare",
                   "--config", str(tmp_cfg)])
        if rc != 0:
            print(f"预处理失败（exit {rc}）")
            return rc

    # 3) 生成 24×96 口径划分（独立文件名，不动主 splits.npz / manifest.csv）
    print("[2/2] 生成 24×96 口径划分 ……")
    rc = _run([sys.executable, "-u", "-m", "train.phase15_split",
               "--config", str(tmp_cfg), "--force",
               "--out-name", SPLIT_24, "--skip-manifest"])
    if rc != 0:
        print(f"划分失败（exit {rc}）")
        return rc

    # 4) 核对划分索引与主划分完全一致（防泄漏前提：E8 必须与主口径同号同集）
    with np.load(base_split, allow_pickle=False) as d:
        ref = {k: d[k].copy() for k in ("train", "val", "test", "hard")}
    with np.load(processed / SPLIT_24, allow_pickle=False) as d:
        same = all(np.array_equal(ref[k], d[k]) for k in ref)
        syn_shape = d["synth_images"].shape
        std24 = str(d["standardizer"])

    print("\n=== 一致性核对 ===")
    print(f"  划分索引与 32×128 完全一致：{same}")
    print(f"  合成域图像形状：{syn_shape}（应为 (2000, {TARGET[1]}, {TARGET[0]})）")
    print(f"  24×96 标准化统计量：{std24}")
    if not same:
        print("⚠️ 划分索引发生变化，E8 与其它实验不可比！")
        return 1
    if syn_shape != (2000, TARGET[1], TARGET[0]):
        print("⚠️ 合成域形状不符合预期！")
        return 1
    print(f"\n完成：{processed / SPLIT_24}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())