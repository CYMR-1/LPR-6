# -*- coding: utf-8 -*-
"""E1–E9 对照实验一键执行器（P5 对照开关 + P6 一键跑，§7 / §8.4）。

职责
----
1. **P5 对照开关**：把 §7 的六类对照（独立模型 / MSE / 批量策略 / 动量 / L2 /
   激活函数）全部表达为 ``configs/default.yaml`` 里 ``experiments`` 段的
   变体补丁，本脚本只负责"应用补丁 -> 训练 -> 汇总"，不把超参数写死在代码里。
2. **P6 一键跑**：对每个启用实验的每个变体、每个随机种子各训练一次，
   产出可复现的 CSV/JSON 汇总表与对比图。

产物
----
reports/configs/<run>.yaml          该次运行实际生效的完整配置（可复现的关键）
reports/logs/<run>_run.json         单次运行指标（由 train/train.py 写）
reports/logs/<run>_history.csv      训练曲线
reports/tables/exp_<Ei>_runs.csv    逐次运行明细（每行 = 一次运行）
reports/tables/exp_<Ei>_summary.csv 按变体聚合的均值±标准差
reports/tables/all_experiments_runs.csv    全部实验合并明细
reports/figs/exp_<Ei>_<metric>.png  变体对比图

为什么要把"实际生效的配置"另存一份
----------------------------------
变体补丁是点分路径覆盖，只有把它落盘才能回答"这个数是用什么配置跑出来的"。
配合 run.json 里的 config 指纹与 git commit，任何一行汇总都能回溯到具体代码与参数。

命令行
------
    # 快速冒烟：每个实验只跑 1 个种子、3 轮、1000 样本
    python reports/run_all.py --smoke

    # 正式执行（读 configs/default.yaml 的 experiments.enabled）
    python reports/run_all.py

    # 只跑某几个实验
    python reports/run_all.py --only E1 E3
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

# --- 包引导 ---------------------------------------------------------------
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.config import (  # noqa: E402
    Config,
    _to_plain,
    apply_patch,
    ensure_dirs,
    git_info,
    load_config,
    resolve_path,
)

# 汇总表里每个指标一组列：均值 / 标准差 / 各次运行的原始值
SUMMARY_METRICS = (
    ("test", "char_acc"),
    ("test", "plate_acc"),
    ("hard_test", "char_acc"),
    ("hard_test", "plate_acc"),
    ("synth_test", "char_acc"),
    ("synth_test", "plate_acc"),
    ("val", "char_acc"),
    ("train", "char_acc"),
)

# 曲线图默认画的指标
PLOT_METRICS = ("char_acc", "plate_acc")


# =============================================================================
# 1. 配置与命名
# =============================================================================


def variant_config(base: Config, patches: Dict[str, Any]) -> Config:
    """在基线配置上应用变体补丁，返回独立副本。

    参数
    ----
    base : Config
        基线配置。
    patches : dict
        点分路径 -> 值，例如 ``{"model.arch": "independent"}``。

    返回
    ----
    Config
        应用补丁后的新配置（不修改 ``base``）。

    形状
    ----
    标量配置树。
    """
    import copy

    cfg = Config(copy.deepcopy(_to_plain(base)))
    # 注意：apply_patch 返回**新配置**（内部先深拷贝），不会就地修改 cfg。
    # 早期版本漏了接收返回值，导致所有变体补丁被静默丢弃 —— 表现为
    # "E3 的 mse 变体与 cross_entropy 结果完全相同"，极具迷惑性。
    # 这里用断言兜底：补丁必须真的体现在返回配置上。
    patched = apply_patch(cfg, patches or {})
    for dotted in (patches or {}):
        sentinel = object()
        if patched.get_path(dotted, sentinel) is sentinel:
            raise RuntimeError(f"变体补丁未生效：{dotted}")
    return patched


def run_name_for(exp: str, variant: str, seed: int, prefix: str = "") -> str:
    """生成运行短名，保证可读且唯一。

    参数
    ----
    exp : str
        实验号，如 ``"E1"``。
    variant : str
        变体名，如 ``"shared"``。
    seed : int
        随机种子。
    prefix : str
        可选前缀（如 ``smoke_``）。

    返回
    ----
    str
        ``<prefix><exp>_<variant>_s<seed>``，例如 ``E1_shared_s42``。

    形状
    ----
    标量 -> str
    """
    return f"{prefix}{exp}_{variant}_s{int(seed)}"


def save_variant_config(cfg: Config, run_name: str, base: Config) -> Path:
    """把该次运行实际生效的配置写到 ``reports/configs/<run>.yaml``。

    参数
    ----
    cfg : Config
        生效配置。
    run_name : str
        运行短名。
    base : Config
        基线配置（用于取目录）。

    返回
    ----
    Path
        写出的文件路径。

    形状
    ----
    配置树 -> YAML 文件
    """
    out_dir = resolve_path(base, "logs_dir").parent / "configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_name}.yaml"
    with open(path, "w", encoding="utf-8") as fp:
        yaml.safe_dump(_to_plain(cfg), fp, allow_unicode=True, sort_keys=False)
    return path


# =============================================================================
# 2. 单次运行
# =============================================================================


def execute_one(
    base: Config,
    exp: str,
    variant: str,
    patches: Dict[str, Any],
    seed: int,
    prefix: str = "",
    limit: Optional[int] = None,
    epochs: Optional[int] = None,
    quiet: bool = False,
    extra_patches: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """跑一次完整的"变体 + 种子"训练。

    参数
    ----
    base : Config
        基线配置。
    exp, variant : str
        实验号与变体名。
    patches : dict
        变体补丁。
    seed : int
        随机种子。
    prefix : str
        运行名前缀。
    limit : int or None
        训练集截断（冒烟用）。
    epochs : int or None
        轮数覆盖。
    quiet : bool
        是否少打印。
    extra_patches : dict or None
        额外补丁（如冒烟时强制 batch_size）。

    返回
    ----
    dict
        训练结果 + ``run_name`` / ``exp`` / ``variant`` / ``seed`` / 配置路径。

    形状
    ----
    配置 -> 指标字典
    """
    from train.train import run_training

    merged = dict(patches or {})
    if extra_patches:
        merged.update(extra_patches)
    cfg = variant_config(base, merged)
    run_name = run_name_for(exp, variant, seed, prefix=prefix)
    cfg_path = save_variant_config(cfg, run_name, base)

    t0 = time.perf_counter()
    result = run_training(cfg, run_name, seed=seed, limit=limit,
                          epochs=epochs, quiet=quiet)
    wall = time.perf_counter() - t0

    result["run_name"] = run_name
    result["exp"] = exp
    result["variant"] = variant
    result["patches"] = merged
    result["variant_config"] = str(cfg_path)
    result["wall_seconds"] = round(wall, 2)
    return result


# =============================================================================
# 3. 汇总
# =============================================================================


def _get_metric(result: Dict[str, Any], split: str, metric: str) -> Optional[float]:
    """从单次运行结果里取一个指标；缺失返回 None。"""
    block = result.get(split)
    if not isinstance(block, dict):
        return None
    v = block.get(metric)
    return float(v) if isinstance(v, (int, float)) else None


_RUN_NAME_RE = re.compile(
    r"^(?:(e4n|smoke|verify|e2lr)_)?(E\d+|baseline)_(.+)_s(\d+)$")


def scan_runs_from_disk(cfg: Config) -> List[Dict[str, Any]]:
    """从 ``logs_dir`` 扫描全部 ``*_run.json``，重建结果列表供 CSV 汇总。

    ★ 为什么从磁盘重建：按"本次进程跑到的运行"写 CSV 时，``skip_existing``
    跳过的运行会从汇总表里消失（exp_E4 的 batch_1 行曾因此丢失）；磁盘扫描
    保证 CSV 始终是**当前全部产物的完整镜像**，且 seed 从运行名解析、
    不再依赖进程内传参。探针/冒烟/验证运行（``smoke_``/``verify_``/``e2lr_``
    前缀）不计入。

    参数
    ----
    cfg : Config
        全局配置（用于定位 ``logs_dir``）。

    返回
    ----
    list of dict
        与 ``run_training`` 返回值同构的列表，额外含
        ``run_name`` / ``exp`` / ``variant`` / ``seed``。

    形状
    ----
    ``*_run.json`` 文件集 -> 结果列表
    """
    logs = resolve_path(cfg, "logs_dir")
    out: List[Dict[str, Any]] = []
    for p in sorted(logs.glob("*_run.json")):
        name = p.name[: -len("_run.json")]
        m = _RUN_NAME_RE.match(name)
        if not m:
            continue                      # 非实验命名（如 e2lr 探针以外的临时运行）
        prefix, exp, variant, seed = m.groups()
        if prefix in ("smoke", "verify", "e2lr"):
            continue                      # 探针与验证运行不进汇总表
        with open(p, encoding="utf-8") as fp:
            r = json.load(fp)
        # run.json 的逐位准确率是扁平键 per_position_0..5，还原成列表
        test = r.get("test") or {}
        if "per_position" not in test:
            pp = [test[f"per_position_{i}"] for i in range(6)
                  if f"per_position_{i}" in test]
            if pp:
                test["per_position"] = pp
                r["test"] = test
        r["run_name"] = name
        r["exp"] = exp
        # 前缀并入变体名，避免 e4n_（2000 张口径）与主口径同名变体被聚合到一起
        r["variant"] = f"{prefix}_{variant}" if prefix else variant
        r["seed"] = seed
        out.append(r)
    return out


def write_runs_csv(results: List[Dict[str, Any]], path: Path) -> None:
    """写"逐次运行明细"CSV（每行一次运行）。

    参数
    ----
    results : list of dict
        多次运行的返回值。
    path : Path
        输出 CSV 路径。

    返回
    ----
    None

    形状
    ----
    list -> 表格文件
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = (["exp", "variant", "seed", "run_name", "commit", "git_dirty",
             "backend", "device_name", "epochs_run", "best_epoch",
             "early_stopped", "train_seconds", "wall_seconds",
             "num_parameters", "patches_json"]
            + [f"{s}_{m}" for s, m in SUMMARY_METRICS]
            + [f"per_pos_test_{i + 1}" for i in range(6)])

    with open(path, "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.writer(fp)
        w.writerow(cols)
        for r in results:
            meta = r.get("meta", {}) or {}
            model = meta.get("model", {}) or {}
            row = [
                r.get("exp", ""), r.get("variant", ""), r.get("seed", ""),
                r.get("run_name", ""), meta.get("commit", ""),
                meta.get("dirty", ""), meta.get("backend", ""),
                meta.get("device_name", ""), r.get("epochs_run", ""),
                r.get("best_epoch", ""), r.get("early_stopped", ""),
                r.get("train_seconds", ""), r.get("wall_seconds", ""),
                model.get("num_parameters", ""),
                json.dumps(r.get("patches", {}), ensure_ascii=False),
            ]
            for split, metric in SUMMARY_METRICS:
                v = _get_metric(r, split, metric)
                row.append("" if v is None else round(v, 6))
            pp = (r.get("test") or {}).get("per_position") or []
            row.extend([round(float(x), 6) for x in pp[:6]])
            row.extend([""] * (6 - len(pp[:6])))
            w.writerow(row)


def aggregate(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按"实验 + 变体"聚合多次运行，给出均值与样本标准差。

    参数
    ----
    results : list of dict
        多次运行结果。

    返回
    ----
    list of dict
        每个变体一行，含 ``n_seeds`` 与各指标的 ``mean`` / ``std`` / ``values``。

    形状
    ----
    list -> list
    """
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in results:
        groups.setdefault((r.get("exp", ""), r.get("variant", "")), []).append(r)

    rows: List[Dict[str, Any]] = []
    for (exp, variant), rs in sorted(groups.items()):
        row: Dict[str, Any] = {
            "exp": exp, "variant": variant, "n_seeds": len(rs),
            "seeds": sorted(int(x.get("seed", 0)) for x in rs),
            "patches_json": json.dumps(rs[0].get("patches", {}), ensure_ascii=False),
            "epochs_run_mean": round(
                statistics.fmean([float(x.get("epochs_run", 0)) for x in rs]), 2),
            "planned_epochs": rs[0].get("planned_epochs", ""),
            # 有几次运行因为触到时间预算而没跑满目标轮数（口径必须说明）
            "n_budget_exhausted": sum(
                1 for x in rs if bool(x.get("budget_exhausted", False))),
            "n_early_stopped": sum(
                1 for x in rs if bool(x.get("early_stopped", False))),
            "train_seconds_mean": round(
                statistics.fmean([float(x.get("train_seconds", 0)) for x in rs]), 2),
        }
        model = (rs[0].get("meta", {}) or {}).get("model", {}) or {}
        row["num_parameters"] = model.get("num_parameters", "")

        for split, metric in SUMMARY_METRICS:
            vals = [v for v in (_get_metric(x, split, metric) for x in rs)
                    if v is not None]
            if not vals:
                row[f"{split}_{metric}_mean"] = ""
                row[f"{split}_{metric}_std"] = ""
                row[f"{split}_{metric}_values"] = ""
                continue
            row[f"{split}_{metric}_mean"] = round(statistics.fmean(vals), 6)
            row[f"{split}_{metric}_std"] = (
                round(statistics.stdev(vals), 6) if len(vals) > 1 else 0.0)
            row[f"{split}_{metric}_values"] = ";".join(
                f"{v:.6f}" for v in vals)
        rows.append(row)
    return rows


def write_summary_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    """写"按变体聚合"的汇总 CSV。

    参数
    ----
    rows : list of dict
        :func:`aggregate` 的输出。
    path : Path
        输出路径。

    返回
    ----
    None

    形状
    ----
    list -> 表格文件
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols: List[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.DictWriter(fp, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# =============================================================================
# 4. 主流程
# =============================================================================


def enabled_experiments(cfg: Config, only: Optional[Sequence[str]]) -> List[str]:
    """确定要执行的实验列表。

    参数
    ----
    cfg : Config
        配置。
    only : Sequence[str] or None
        显式指定要跑的实验；``None`` 表示读 ``experiments.enabled``。

    返回
    ----
    list of str
        实验号列表。

    形状
    ----
    标量 -> list
    """
    defined = list(cfg.experiments.all_defined)
    if only:
        bad = [e for e in only if e not in defined]
        if bad:
            raise ValueError(f"未定义的实验：{bad}；已定义 {defined}")
        return list(only)
    return list(cfg.experiments.enabled)


def run_finished(base: Config, run_name: str) -> bool:
    """判断某次运行是否已有完整的产物（可跳过重跑）。

    参数
    ----
    base : Config
        基线配置（用于定位日志目录）。
    run_name : str
        运行短名。

    返回
    ----
    bool
        同时存在 ``<run>_run.json`` 与 ``<run>_history.csv`` 时为 True。

    形状
    ----
    标量 -> bool
    """
    log_dir = resolve_path(base, "logs_dir")
    return ((log_dir / f"{run_name}_run.json").is_file()
            and (log_dir / f"{run_name}_history.csv").is_file())


def load_finished(base: Config, run_name: str, exp: str,
                  variant: str) -> Dict[str, Any]:
    """从已完成运行的 JSON 里恢复结果，供汇总使用。

    参数
    ----
    base : Config
        基线配置。
    run_name, exp, variant : str
        运行标识。

    返回
    ----
    dict
        与 :func:`execute_one` 结构一致的结果字典（``resumed`` 标记为 True）。

    形状
    ----
    JSON -> dict
    """
    log_dir = resolve_path(base, "logs_dir")
    with open(log_dir / f"{run_name}_run.json", "r", encoding="utf-8") as fp:
        result = json.load(fp)
    result["run_name"] = run_name
    result["exp"] = exp
    result["variant"] = variant
    result.setdefault("patches", {})
    result["resumed"] = True
    return result


def run_experiments(
    cfg: Config,
    exps: Sequence[str],
    seeds_override: Optional[Sequence[int]] = None,
    limit: Optional[int] = None,
    epochs: Optional[int] = None,
    prefix: str = "",
    quiet: bool = False,
    extra_patches: Optional[Dict[str, Any]] = None,
    skip_existing: bool = False,
    skip_variants: Optional[Sequence[str]] = None,
) -> tuple:
    """按实验矩阵逐个训练，返回全部运行结果。

    参数
    ----
    cfg : Config
        基线配置。
    exps : Sequence[str]
        要执行的实验号。
    seeds_override : Sequence[int] or None
        覆盖每个实验的种子列表（冒烟用）。
    limit : int or None
        训练集截断。
    epochs : int or None
        轮数覆盖。
    prefix : str
        运行名前缀。
    quiet : bool
        是否少打印。
    extra_patches : dict or None
        额外补丁。
    skip_existing : bool
        已有完整产物的运行直接复用 JSON 结果，不重跑（长实验的断点续跑）。
    skip_variants : Sequence[str] or None
        显式跳过的变体名（例如 ``("batch_1",)``）。用于某变体已得出
        **结论性结果**、继续跑其余种子无信息增益时，把时间留给其它变体。

    返回
    ----
    tuple
        ``(results, failures)``：成功运行的完整结果列表，以及失败明细列表。

    形状
    ----
    实验矩阵 -> (list, list)
    """
    results: List[Dict[str, Any]] = []
    total = sum(len(list(cfg.experiments[e].variants))
                * len(list(seeds_override if seeds_override is not None
                           else cfg.experiments[e].seeds))
                for e in exps)
    done = 0
    failures: List[Dict[str, str]] = []

    _skip = {str(v) for v in (skip_variants or ())}
    for exp in exps:
        node = cfg.experiments[exp]
        seeds = list(seeds_override if seeds_override is not None else node.seeds)
        for variant, patches in list(node.variants.items()):
            for seed in seeds:
                done += 1
                name = run_name_for(exp, variant, seed, prefix=prefix)
                if str(variant) in _skip:
                    print(f"\n[{done}/{total}] {exp} / {variant} / seed={seed} "
                          f"-> {name}  （按 --skip-variant 显式跳过）")
                    continue
                if skip_existing and run_finished(cfg, name):
                    print(f"\n[{done}/{total}] {exp} / {variant} / seed={seed} "
                          f"-> {name}  （已有产物，跳过重跑）")
                    results.append(load_finished(cfg, name, exp, variant))
                    continue
                print(f"\n[{done}/{total}] {exp} / {variant} / seed={seed} "
                      f"-> {name}")
                try:
                    res = execute_one(
                        cfg, exp, variant, dict(patches), int(seed),
                        prefix=prefix, limit=limit, epochs=epochs,
                        quiet=quiet, extra_patches=extra_patches)
                    results.append(res)
                except Exception as exc:                     # noqa: BLE001
                    print(f"    !! 失败：{type(exc).__name__}: {exc}")
                    traceback.print_exc()
                    failures.append({"exp": exp, "variant": variant,
                                     "seed": str(seed), "run_name": name,
                                     "error": f"{type(exc).__name__}: {exc}"})
    if failures:
        print(f"\n[run_all] 共 {len(failures)} 次运行失败，明细见汇总 JSON。")
    return results, failures


def make_plots(cfg: Config, rows: List[Dict[str, Any]], exp: str) -> List[str]:
    """为某个实验生成变体对比图（每个指标一张）。

    参数
    ----
    cfg : Config
        基线配置。
    rows : list of dict
        该实验的聚合行。
    exp : str
        实验号。

    返回
    ----
    list of str
        生成的图片路径。

    形状
    ----
    list -> list
    """
    try:
        from evaluate.visualize import plot_variant_comparison, setup_chinese_font
    except Exception as exc:                                    # noqa: BLE001
        print(f"    （跳过绘图：{exc}）")
        return []

    try:
        setup_chinese_font(cfg.viz.chinese_font)
    except Exception:
        pass

    fig_dir = resolve_path(cfg, "figs_dir")
    fig_dir.mkdir(parents=True, exist_ok=True)
    out: List[str] = []

    # 统一用同分布测试集主指标 + 另两个测试集，便于观察"分布偏移"下的差异
    for split, metric in (("test", "char_acc"), ("test", "plate_acc"),
                          ("hard_test", "char_acc"),
                          ("synth_test", "char_acc")):
        # plot_variant_comparison 期望 {变体名: {指标名: (均值, 标准差)}}
        summary: Dict[str, Dict[str, tuple]] = {}
        for r in rows:
            if r.get("exp") != exp:
                continue
            mean = r.get(f"{split}_{metric}_mean")
            std = r.get(f"{split}_{metric}_std")
            if isinstance(mean, (int, float)):
                summary[r["variant"]] = {
                    metric: (float(mean),
                             float(std) if isinstance(std, (int, float)) else 0.0),
                }
        if not summary:
            continue
        tag = f"{exp}_{split}_{metric}"
        path = fig_dir / f"{tag}.png"
        try:
            plot_variant_comparison(
                summary, metric=metric, out_path=path,
                title=f"{exp}：{split} 的 {metric}（均值±标准差，"
                      f"{len(summary)} 个变体）",
                ylabel=metric)
            out.append(str(path))
        except Exception as exc:                                # noqa: BLE001
            print(f"    （{tag} 绘图失败：{type(exc).__name__}: {exc}）")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    """命令行入口。

    参数
    ----
    argv : Sequence[str] or None
        命令行参数。

    返回
    ----
    int
        0 表示全部成功；1 表示有运行失败。
    """
    ap = argparse.ArgumentParser(description="ProjectX E1–E9 一键实验（§7）")
    ap.add_argument("--config", default=None, help="配置路径")
    ap.add_argument("--only", nargs="*", default=None,
                    help="只跑指定实验，如 --only E1 E3")
    ap.add_argument("--limit", type=int, default=None, help="训练集截断")
    ap.add_argument("--epochs", type=int, default=None, help="轮数覆盖")
    ap.add_argument("--seeds", nargs="*", type=int, default=None,
                    help="种子覆盖（默认读配置）")
    ap.add_argument("--prefix", default="", help="运行名前缀")
    ap.add_argument("--no-plot", action="store_true", help="不生成对比图")
    ap.add_argument("--skip-existing", action="store_true",
                    help="已有完整产物的运行直接复用，不重跑（断点续跑）")
    ap.add_argument("--skip-variant", nargs="*", default=None,
                    help="显式跳过的变体名（如 batch_1）；用于某变体已有结论性结果时")
    ap.add_argument("--run-timeout", type=float, default=None,
                    help="单次运行的时间预算（秒）；超出则提前结束并如实记录轮次")
    ap.add_argument("--evals", nargs="*", default=None,
                    help="对指定运行额外执行 evaluate/main.py（生成混淆矩阵等）")
    ap.add_argument("--baseline", action="store_true",
                    help="先按 base_seeds 跑基线配置（无补丁），供报告做主结果")
    ap.add_argument("--smoke", action="store_true",
                    help="快速冒烟：1 个种子、3 轮、1500 样本、bs 统一为 64")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    limit = args.limit
    epochs = args.epochs
    seeds = args.seeds
    prefix = args.prefix
    extra: Dict[str, Any] = {}
    if args.smoke:
        limit = limit if limit is not None else 1500
        epochs = epochs if epochs is not None else 3
        seeds = seeds if seeds is not None else [42]
        prefix = prefix or "smoke_"
        # 冒烟时把极端的批大小拉回 64，避免 batch=1 耗掉全部时间
        extra = {"optim.batch_size": 64, "train.epochs": 3}

    # 单次运行时间预算（秒）。E4 的 batch_size=1 每轮上百秒，必须设上限；
    # 超出预算的运行会提前结束并在 CSV 里标注"轮次不足"，不当作跑满的口径。
    if args.run_timeout is not None:
        extra = dict(extra)
        extra["train.time_budget_seconds"] = float(args.run_timeout)
        cfg = apply_patch(cfg, {"train.time_budget_seconds":
                                float(args.run_timeout)})

    exps = enabled_experiments(cfg, args.only)
    print(f"[run_all] 实验：{exps}  种子：{seeds or '（按配置）'}  "
          f"limit={limit} epochs={epochs} 前缀='{prefix}'")
    print(f"[run_all] commit={git_info().commit} dirty={git_info().dirty}")

    # ---- 可选：先跑基线（无补丁） ---------------------------------------
    if args.baseline:
        base_seeds = list(seeds if seeds is not None
                          else cfg.base_seeds)
        print(f"\n[run_all] 基线运行，种子 {base_seeds}")
        for sd in base_seeds:
            name = f"{prefix}baseline_s{int(sd)}"
            if args.skip_existing and run_finished(cfg, name):
                print(f"  baseline / seed={sd} -> {name}（已有产物，跳过）")
                continue
            try:
                res = execute_one(cfg, "baseline", "default", {}, int(sd),
                                  prefix=prefix, limit=limit, epochs=epochs,
                                  quiet=False, extra_patches=extra or None)
                print(f"    基线 seed={sd} 完成："
                      f"test_char={_get_metric(res, 'test', 'char_acc')}")
            except Exception as exc:                            # noqa: BLE001
                print(f"    !! 基线 seed={sd} 失败：{type(exc).__name__}: {exc}")
                traceback.print_exc()

    results, failures = run_experiments(
        cfg, exps, seeds_override=seeds, limit=limit, epochs=epochs,
        prefix=prefix, extra_patches=extra or None,
        skip_existing=args.skip_existing,
        skip_variants=args.skip_variant)

    # 可选：对指定运行做独立评测（混淆矩阵、CPU 推理耗时、错误样本）
    if args.evals:
        from evaluate.main import evaluate_run as _eval_run

        for run in args.evals:
            print(f"\n[eval] 独立评测 {run}")
            try:
                _eval_run(cfg, run, backend_name=None)
            except Exception as exc:                            # noqa: BLE001
                print(f"    !! 评测失败：{type(exc).__name__}: {exc}")
                traceback.print_exc()

    tables = resolve_path(cfg, "tables_dir")
    tables.mkdir(parents=True, exist_ok=True)

    if not results:
        print("[run_all] 本次没有新执行/成功的运行；仍会从磁盘重建汇总表。")

    # 逐实验汇总：★ 永远从磁盘全量重建，避免 skip_existing 跳过的运行
    # 从汇总表中消失（历史缺陷：exp_E4 缺 batch_1、all_experiments 只剩
    # 本次跑到的运行）。
    agg_all: List[Dict[str, Any]] = []
    disk = scan_runs_from_disk(cfg)
    for exp in sorted({r["exp"] for r in disk}):
        sub = [r for r in disk if r["exp"] == exp]
        write_runs_csv(sub, tables / f"exp_{exp}_runs.csv")
        rows = aggregate(sub)
        write_summary_csv(rows, tables / f"exp_{exp}_summary.csv")
        agg_all.extend(rows)

    write_runs_csv(disk, tables / "all_experiments_runs.csv")
    write_summary_csv(agg_all, tables / "all_experiments_summary.csv")

    # 本次实际执行的运行仍单独作图（避免把磁盘上全部历史运行都重画一遍）
    if not args.no_plot:
        for exp in exps:
            sub = [r for r in results if r.get("exp") == exp]
            if sub:
                make_plots(cfg, aggregate(sub), exp)

    payload = {
        "commit": git_info().commit,
        "dirty": git_info().dirty,
        "experiments": list(exps),
        "seeds": seeds,
        "limit": limit,
        "epochs": epochs,
        "prefix": prefix,
        "smoke": bool(args.smoke),
        "n_runs_ok": len(results),
        "n_runs_failed": len(failures),
        "failures": failures,
        "tables": {
            "all_runs": str(tables / "all_experiments_runs.csv"),
            "all_summary": str(tables / "all_experiments_summary.csv"),
        },
    }
    with open(resolve_path(cfg, "logs_dir") / "run_all_summary.json", "w",
              encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)

    print(f"\n[run_all] 完成：成功 {len(results)} 次，失败 {len(failures)} 次。")
    print(f"[run_all] 汇总表：{tables / 'all_experiments_summary.csv'}")
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())