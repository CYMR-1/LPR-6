# -*- coding: utf-8 -*-
"""单次运行的独立评测入口（§6 / §附录 B 第 12 条）。

用途
----
``train/train.py`` 在训练结束时已经评测过一次。本脚本用于**不重新训练**、
只从一个已保存的检查点复现评测，生成：

* 三个测试集（同分布 / 强扰动 / 合成域）与验证集的完整指标；
* 每位的完整混淆矩阵（供 §6.3 可视化）；
* 错误样本可视化所需的数据（图片 + 真实/预测标签 + 置信度）；
* **CPU 单张推理耗时**（规格明确要求报告，且必须与"用了 GPU 训练"分开陈述）；
* 逐位准确率表。

为什么需要它
------------
规格 §0.3 禁止"只在截图里给结论"，因此混淆矩阵与错误样本必须落成可复现的
数值文件（``.npz`` / ``.csv`` / ``.json``），图片只是这些文件的呈现方式。

命令行
------
    python evaluate/main.py --run final_s42
    python evaluate/main.py --run final_s42 --backend numpy

命名说明
--------
本文件**故意不叫** ``evaluate/evaluate.py``：那样直接运行时 Python 会把脚本
自身注册成顶层模块 ``evaluate``，把 ``evaluate/`` 这个包整个遮住，
``from evaluate.model_eval import ...`` 就会失败或无限递归。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

# --- 包引导：支持直接运行本文件 ---------------------------------------------
if __package__ in (None, ""):
    _root = str(Path(__file__).resolve().parent.parent)
    if _root not in sys.path:
        sys.path.insert(0, _root)
    # 直接运行时 sys.path[0] 是脚本所在目录（evaluate/），需要让项目根优先，
    # 否则 `models` 等顶层包找不到。
    _here = str(Path(__file__).resolve().parent)
    while _here in sys.path:
        sys.path.remove(_here)

from evaluate.model_eval import (  # noqa: E402
    collect_error_samples,
    evaluate_dataset,
    measure_cpu_inference_time,
)
from models.backend import asnumpy, get_backend, set_backend_env  # noqa: E402
from models.charset import SEQ_LEN, decode_batch, resolve_positions  # noqa: E402
from models.config import ensure_dirs, git_info, load_config, resolve_path  # noqa: E402
from models.dataset import GlobalStandardizer, PlateDataset, load_cache  # noqa: E402
from models.model import Params, build_model  # noqa: E402

SPLIT_ORDER = ("train", "val", "test", "hard_test", "synth_test")


def load_run_config(cfg, run_name: str):
    """读取某次运行的部署配置。

    解析顺序（找到即用）
    --------------------
    1. ``reports/configs/<run_name>.yaml``：该次运行自己的配置；
    2. ``reports/configs/final.yaml``：交付模型的通用部署口径
       （``augmentation.baseline_level: none``，与最终检查点一致）；
    3. 都没有时抛出 :class:`FileNotFoundError`。

    ★ 为什么不再静默回退到 ``configs/default.yaml``：默认配置的数据增强档位是
    ``weak``，而最终模型的检查点是在关闭增强（``none``）的口径下训练的。
    拿默认配置去评测会得到口径错误的数字（同一模型两种口径的整牌准确率相差
    数十个百分点），因此必须显式报错，让调用方指定正确的配置。

    参数
    ----
    cfg : Config
        基线配置（用于定位目录）。
    run_name : str
        运行短名。

    返回
    ----
    Config
        该次运行实际使用的配置。
    """
    import yaml

    from models.config import Config

    var_dir = resolve_path(cfg, "logs_dir").parent / "configs"
    candidates = [var_dir / f"{run_name}.yaml", var_dir / "final.yaml"]
    for path in candidates:
        if path.is_file():
            with open(path, "r", encoding="utf-8") as fp:
                raw = yaml.safe_load(fp)
            if not isinstance(raw, dict):
                raise ValueError(f"配置不是映射：{path}")
            print(f"[eval] 使用配置 {path}")
            return Config(raw)
    raise FileNotFoundError(
        f"找不到运行 {run_name!r} 的配置；已尝试 "
        f"{', '.join(str(p) for p in candidates)}。"
        f"请确认检查点与其配置同名，或用 --config 指定配置路径")


def rebuild_params(cfg, run_name: str, ckpt_dir: Path) -> Params:
    """按配置重建模型结构并载入检查点权重。

    参数
    ----
    cfg : Config
        该次运行的配置（决定结构：arch / hidden_dim / activation / head_dims）。
    run_name : str
        运行短名。
    ckpt_dir : Path
        检查点目录。

    返回
    ----
    Params
        已载入权重的参数对象。

    形状
    ----
    W1 ``(input_dim, hidden_dim)``；W2_i ``(hidden_dim, C_i)``。
    """
    import json as _json

    ckpt = ckpt_dir / f"{run_name}_best.npz"
    if not ckpt.is_file():
        raise FileNotFoundError(f"检查点不存在：{ckpt}")

    # ★ 注意：Params.load 是 **classmethod**，返回 (Params, extra)，不是就地载入。
    # 早期版本写成 `params = build_model(...); params.load(ckpt)`，返回值被丢弃，
    # 于是评测用的是**随机初始权重**，表现为"真实集字符准确率 3%"，
    # 与训练脚本汇报的 86.7% 严重不符 —— 这是必须靠交叉核对才能发现的坑。
    params, extra = Params.load(ckpt)

    # 用配置交叉校验结构，防止"配置改了但检查点是旧的"而无人察觉
    want_head_dims = resolve_positions(cfg.charset.positions)
    if list(params.head_dims) != [int(v) for v in want_head_dims]:
        raise ValueError(
            f"{run_name}: 检查点 head_dims={list(params.head_dims)} 与配置 "
            f"{list(want_head_dims)} 不一致")
    want_arch = str(cfg.model.arch)
    if str(params.arch) != want_arch:
        raise ValueError(
            f"{run_name}: 检查点 arch={params.arch!r} 与配置 {want_arch!r} 不一致")
    want_h = int(cfg.model.hidden_dim)
    if int(params.hidden_dim) != want_h:
        raise ValueError(
            f"{run_name}: 检查点 hidden_dim={params.hidden_dim} 与配置 "
            f"{want_h} 不一致")
    want_in = int(cfg.model.input_dim)
    if int(params.input_dim) != want_in:
        raise ValueError(
            f"{run_name}: 检查点 input_dim={params.input_dim} 与配置 "
            f"{want_in} 不一致")
    if extra:
        print(f"[eval] {run_name}: 检查点附带信息 {list(extra)[:6]}")
    return params


def load_standardizer(cfg) -> GlobalStandardizer:
    """从划分文件读取训练集拟合出的标准化统计量。

    参数
    ----
    cfg : Config
        全局配置。

    返回
    ----
    GlobalStandardizer
        含 mean / std / n_samples 的标准化器。
    """
    split_name = str(cfg.get("paths.splits_file", "splits.npz"))
    split_path = resolve_path(cfg, "processed_dir") / split_name
    with np.load(split_path, allow_pickle=False) as d:
        return GlobalStandardizer.from_dict(json.loads(str(d["standardizer"])))


def build_datasets(cfg, standardizer: GlobalStandardizer) -> Dict[str, PlateDataset]:
    """构建训练/验证/三个测试集的 ``PlateDataset``（评测一律不增强）。

    参数
    ----
    cfg : Config
        全局配置。
    standardizer : GlobalStandardizer
        标准化器。

    返回
    ----
    dict
        键为 ``train`` / ``val`` / ``test`` / ``hard_test`` / ``synth_test``。

    形状
    ----
    每个数据集 ``images (N, 32, 128)``，``labels (N, 6)``。
    """
    processed = resolve_path(cfg, "processed_dir")
    # ★ 缓存与划分文件都要按配置取（input_size 与 paths.splits_file 必须配对），
    # 不能写死 128x32 / splits.npz。
    tag = f"ccpd_{int(cfg.ccpd.input_size[0])}x{int(cfg.ccpd.input_size[1])}"
    split_name = str(cfg.get("paths.splits_file", "splits.npz"))
    images, labels, _ = load_cache(processed / f"{tag}.npz")
    # ★ 响亮防线（与 train.load_data_bundle 相同）：越界标签必须报错，
    # 不能在 one-hot 阶段被静默置零。
    if int(((labels < 0) | (labels >= 34)).sum()):
        raise ValueError(f"缓存 {tag}.npz 中存在越界标签（合法范围 0..33），缓存已损坏")
    with np.load(processed / split_name, allow_pickle=False) as d:
        out: Dict[str, PlateDataset] = {}
        for key in ("train", "val", "test", "hard"):
            idx = d[key].astype(np.int64)
            name = "hard_test" if key == "hard" else key
            out[name] = PlateDataset(
                images=images[idx], labels=labels[idx], standardizer=standardizer,
                flatten=True, augment_fn=None, seed=int(cfg.split.split_seed),
                name=name,
            )
        s_img = d["synth_images"]
        s_lab = d["synth_labels"]
    out["synth_test"] = PlateDataset(
        images=s_img, labels=s_lab, standardizer=standardizer,
        flatten=True, augment_fn=None, seed=int(cfg.split.split_seed),
        name="synth_test",
    )
    return out


def evaluate_run(
    cfg,
    run_name: str,
    backend_name: Optional[str] = None,
    max_eval_samples: Optional[int] = None,
    save_arrays: bool = True,
) -> Dict[str, Any]:
    """评测一次运行并落盘全部产物。

    参数
    ----
    cfg : Config
        基线配置。
    run_name : str
        运行短名。
    backend_name : str or None
        推理后端覆盖；``None`` 用配置里的。
    max_eval_samples : int or None
        每个测试集最多评测多少张（用于快速冒烟）。
    save_arrays : bool
        是否保存混淆矩阵与错误样本数组（``.npz``）。

    返回
    ----
    dict
        指标汇总 + 产物路径。

    形状
    ----
    数据集 -> 指标字典。
    """
    run_cfg = load_run_config(cfg, run_name)
    ensure_dirs(run_cfg)
    log_dir = resolve_path(run_cfg, "logs_dir")
    ckpt_dir = resolve_path(run_cfg, "models_dir")
    fig_dir = resolve_path(run_cfg, "figs_dir")
    fig_dir.mkdir(parents=True, exist_ok=True)

    be_name = backend_name or str(run_cfg.optim.backend)
    backend = get_backend(be_name, verbose=True)
    set_backend_env(backend)

    params = rebuild_params(run_cfg, run_name, ckpt_dir)
    std = load_standardizer(run_cfg)
    datasets = build_datasets(run_cfg, std)

    # 位置合法性掩码：仅当首位类别数 < 34（即首位被约束为字母）时才需要，
    # 由数据集标签动态生成——首位为数字的样本其 head0 无有效目标。
    head_mask_fn = None
    use_mask = int(list(run_cfg.charset.positions)[0]) < 34
    if use_mask:
        def head_mask_fn(labels):  # noqa: ANN001
            from models.charset import is_position_legal

            m0 = np.array([1.0 if is_position_legal(0, int(v)) else 0.0
                           for v in labels[:, 0]], dtype=np.float32)
            return [m0] + [np.ones(len(labels), dtype=np.float32)] * (SEQ_LEN - 1)

    results: Dict[str, Any] = {}
    arrays: Dict[str, np.ndarray] = {}
    for tag, ds in datasets.items():
        t0 = time.perf_counter()
        res = evaluate_dataset(
            params, ds, run_cfg, backend=backend,
            batch_size=int(run_cfg.eval.batch_size),
            l2_lambda=float(run_cfg.loss.l2_lambda),
            loss_type=str(run_cfg.loss.type),
            head_mask_fn=head_mask_fn,
            with_confusion=True,
            max_eval_samples=max_eval_samples,
        )
        secs = time.perf_counter() - t0

        # 构造"真实/预测 车牌文本"便于人工核对
        truth = decode_batch(asnumpy(res.labels))
        preds = decode_batch(asnumpy(res.preds))

        m = res.metrics
        results[tag] = {
            **m.as_dict(),
            "eval_seconds": round(secs, 2),
            "n_evaluated": int(len(res.labels)),
            "backend": backend.name,
            "device_name": backend.device_name,
            "top_confusions": m.top_confusions or [],
            "sample_truth": list(truth[:20]),
            "sample_pred": list(preds[:20]),
        }

        if save_arrays:
            arrays[f"{tag}_preds"] = asnumpy(res.preds)
            arrays[f"{tag}_labels"] = asnumpy(res.labels)
            arrays[f"{tag}_indices"] = asnumpy(res.dataset_indices)
            for i, c in enumerate(m.confusion):
                if c is not None:
                    arrays[f"{tag}_confusion_pos{i + 1}"] = np.asarray(c)

    # ---- CPU 单张推理耗时（§附录 B 第 12 条） ----------------------------
    cpu_backend = get_backend("numpy", verbose=False)
    cpu_time = measure_cpu_inference_time(
        params, datasets["test"], n_samples=int(run_cfg.eval.cpu_timing_samples),
        warmup=int(run_cfg.eval.cpu_time_warmup),
        repeats=int(run_cfg.eval.cpu_time_repeat),
    )

    # ---- 错误样本可视化数据 ---------------------------------------------
    err_arrays: Dict[str, np.ndarray] = {}
    res_test = evaluate_dataset(
        params, datasets["test"], run_cfg, backend=cpu_backend,
        batch_size=int(run_cfg.eval.batch_size),
        l2_lambda=float(run_cfg.loss.l2_lambda),
        loss_type=str(run_cfg.loss.type),
        head_mask_fn=head_mask_fn, with_confusion=False,
    )
    errs = collect_error_samples(res_test, max_n=int(run_cfg.eval.error_table_top_n))
    idx = (np.array([e["dataset_index"] for e in errs], dtype=np.int64)
           if errs else np.zeros(0, dtype=np.int64))
    if len(idx):
        err_arrays["error_images"] = np.asarray(datasets["test"].images[idx])
        err_arrays["error_truth"] = np.asarray(res_test.labels[idx])
        err_arrays["error_pred"] = np.asarray(res_test.preds[idx])
        # 整牌置信度取"六位里最低的那一位"的置信度 —— 只要有一位没把握，
        # 整牌就不可信，因此取最小值比取乘积更直观（乘积会被位数放大）。
        err_arrays["error_conf"] = np.asarray(
            [float(np.min(e["per_position_conf"])) for e in errs],
            dtype=np.float32)
        # 高置信错误（§6.2：优先展示"自信地答错"的样本）单独统计
        thr = float(run_cfg.eval.high_confidence_threshold)
        payload_high_conf = int(sum(
            1 for e in errs if float(np.min(e["per_position_conf"])) >= thr))
    else:
        payload_high_conf = 0

    payload: Dict[str, Any] = {
        "run_name": run_name,
        "config_name": run_cfg.get("project", {}).get("name", "default"),
        "commit": git_info().commit,
        "git_dirty": git_info().dirty,
        "backend": backend.name,
        "device_name": backend.device_name,
        "model": {
            "arch": str(run_cfg.model.arch),
            "activation": str(run_cfg.model.activation),
            "hidden_dim": int(run_cfg.model.hidden_dim),
            "head_dims": resolve_positions(run_cfg.charset.positions),
            "num_parameters": params.num_parameters(),
        },
        "cpu_inference": cpu_time,
        "test_errors": {
            "n_errors": int(len(idx)),
            "n_high_confidence_errors": payload_high_conf
            if len(idx) else 0,
            "high_confidence_threshold": float(
                run_cfg.eval.high_confidence_threshold),
        },
        "datasets": results,
    }

    out_json = log_dir / f"{run_name}_eval.json"
    with open(out_json, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    if save_arrays:
        np.savez_compressed(log_dir / f"{run_name}_eval_arrays.npz", **arrays)
        if err_arrays:
            np.savez_compressed(log_dir / f"{run_name}_errors.npz", **err_arrays)

    # 逐位准确率 CSV
    with open(log_dir / f"{run_name}_eval_per_position.csv", "w", newline="",
              encoding="utf-8-sig") as fp:
        w = csv.writer(fp)
        w.writerow(["position_1based"] + list(SPLIT_ORDER))
        for i in range(SEQ_LEN):
            w.writerow([i + 1] + [
                round(float(results[t][f"per_position_{i}"]), 6)
                if t in results and f"per_position_{i}" in results[t] else ""
                for t in SPLIT_ORDER])

    payload["paths"] = {
        "eval_json": str(out_json),
        "eval_arrays": str(log_dir / f"{run_name}_eval_arrays.npz"),
        "error_arrays": str(log_dir / f"{run_name}_errors.npz"),
        "per_position_csv": str(log_dir / f"{run_name}_eval_per_position.csv"),
    }

    print(f"[eval] {run_name} 完成（{backend.name} / {backend.device_name}）")
    for tag in SPLIT_ORDER:
        if tag not in results:
            continue
        r = results[tag]
        print(f"    {tag:11s} 字符={r['char_acc'] * 100:6.2f}%  "
              f"整牌={r['plate_acc'] * 100:6.2f}%  损失={r['loss']:.4f}  "
              f"({r['eval_seconds']:.1f}s)")
    print(f"    CPU 单张推理 {cpu_time['cpu_inference_ms_per_image']:.3f} ms/张 "
          f"（{cpu_time['n_samples']} 张，median）")
    print(f"    产物：{out_json}")
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    """评测命令行入口。

    参数
    ----
    argv : Sequence[str] or None
        命令行参数。

    返回
    ----
    int
        0 表示成功。
    """
    ap = argparse.ArgumentParser(description="ProjectX 单次运行评测（§6）")
    ap.add_argument("--run", required=True, help="运行短名，如 final_s42")
    ap.add_argument("--config", default=None, help="基线配置路径")
    ap.add_argument("--backend", default=None, help="推理后端 numpy/cupy")
    ap.add_argument("--max-eval-samples", type=int, default=None,
                    help="每个测试集最多评测张数（冒烟用）")
    ap.add_argument("--no-arrays", action="store_true", help="不保存 npz 数组")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    evaluate_run(cfg, args.run, backend_name=args.backend,
                 max_eval_samples=args.max_eval_samples,
                 save_arrays=not args.no_arrays)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())