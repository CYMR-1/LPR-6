# -*- coding: utf-8 -*-
"""把 reports/tables 与 reports/logs 里的数值汇总成报告用的 Markdown 表格。

为什么单独写这个脚本（§0.3）
--------------------------
规格 §0.3 禁止"把结论只放在截图里"。报告里的每个数字都应该能由
``reports/tables/*.csv`` 与 ``reports/logs/*_run.json`` 复核。手工把数字抄进
报告容易抄错、也容易在重跑后失效；因此这里**由数值文件生成 Markdown 表格**，
再插入到 ``reports/实验报告.md`` 的对应占位符处。

用法
----
    python reports/build_report_tables.py                # 打印全部表格
    python reports/build_report_tables.py --write        # 写 reports/tables/report_tables.md
    python reports/build_report_tables.py --inject       # 注入实验报告.md 的占位符
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# --- 包引导 ---------------------------------------------------------------
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.config import load_config, resolve_path  # noqa: E402

REPORT_MARKERS = {
    "baseline": "<!-- TABLE:BASELINE -->",
    "e1": "<!-- TABLE:E1 -->",
    "e3": "<!-- TABLE:E3 -->",
    "e4": "<!-- TABLE:E4 -->",
    "e7": "<!-- TABLE:E7 -->",
    "midtrain": "<!-- TABLE:MIDTRAIN -->",
    "ranking": "<!-- TABLE:RANKING -->",
    "position": "<!-- TABLE:POSITION -->",
    "cpu": "<!-- TABLE:CPU -->",
}


def load_runs(cfg) -> List[Dict[str, Any]]:
    """读取全部 ``*_run.json``，跳过没有 ``test`` 字段的残缺文件。

    参数
    ----
    cfg : Config
        全局配置。

    返回
    ----
    list of dict
        每个元素是一次运行的结果，额外带 ``run_name``。

    形状
    ----
    JSON 目录 -> list
    """
    out = []
    for f in sorted(glob.glob(str(resolve_path(cfg, "logs_dir") / "*_run.json"))):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                d = json.load(fp)
        except Exception:                                          # noqa: BLE001
            continue
        if "test" not in d or "meta" not in d:
            continue
        d["run_name"] = Path(f).name.replace("_run.json", "")
        out.append(d)
    return out


def _acc(d: Dict[str, Any], split: str, metric: str) -> Optional[float]:
    """安全取某划分某指标。"""
    node = (d or {}).get(split) or {}
    v = node.get(metric, None)
    return None if v in (None, "") else float(v)


def runs_of(runs: Sequence[Dict[str, Any]], prefix: str,
            exp: Optional[str] = None) -> List[Dict[str, Any]]:
    """按运行名前缀筛选（如 ``E3_``），并兼容 ``<scope>_E3_`` 这类带作用域前缀的命名。

    参数
    ----
    runs : list of dict
        全部运行。
    prefix : str
        实验号前缀（如 ``E3``）。
    exp : str or None
        未使用，保留以便调用处可读。

    返回
    ----
    list of dict
        匹配的运行。
    """
    key = f"_{prefix.strip('_')}_"
    return [r for r in runs
            if str(r.get("run_name", "")).startswith(f"{prefix.strip('_')}_")
            or key in str(r.get("run_name", ""))]


def variant_of(run_name: str, exp: str) -> str:
    """从运行名里取出变体名：``E3_mse_s42`` -> ``mse``。

    ★ 坑：变体名本身可能含 ``_s``（例如 ``E7_aug_strong_s42``），
    所以不能简单地按 ``_s`` 切分；正确做法是剥掉实验前缀与**结尾的**
    ``_s<种子>``。运行名可能还带作用域前缀（如 ``e4n_E4_batch_1_s42``），
    这里一并处理。

    参数
    ----
    run_name : str
        运行短名。
    exp : str
        实验号（如 ``E3``）。

    返回
    ----
    str
        变体名。
    """
    body = run_name
    if "_" in body and not body.startswith(exp):
        # 形如 e4n_E4_batch_1_s42 -> 取 "E4_batch_1_s42"
        idx = body.find(f"{exp}_")
        if idx >= 0:
            body = body[idx:]
    for pre in ("smoke_", f"{exp}_"):
        if body.startswith(pre):
            body = body[len(pre):]
            break
    m = re.search(r"_s\d+$", body)
    if m:
        body = body[:m.start()]
    return body or run_name


def group_by_variant(runs: Sequence[Dict[str, Any]], exp: str
                     ) -> Dict[str, List[Dict[str, Any]]]:
    """把同一实验的运行按变体分组。"""
    g: Dict[str, List[Dict[str, Any]]] = {}
    for r in runs:
        g.setdefault(variant_of(r["run_name"], exp), []).append(r)
    return g


def fmt_mean_std(vals: Sequence[float], pct: bool = True) -> str:
    """把一组值格式化成 ``mean ± std``。"""
    vals = [v for v in vals if v is not None]
    if not vals:
        return "—"
    m = statistics.fmean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{m * 100:.2f} ± {s * 100:.2f}" if pct else f"{m:.4f} ± {s:.4f}"


def table_experiment(runs: Sequence[Dict[str, Any]], exp: str,
                     title: str, anchor: Optional[str] = None) -> str:
    """生成一个实验的对比表（各变体 × 各划分的字符/整牌准确率）。

    参数
    ----
    runs : list of dict
        该实验的全部运行。
    exp : str
        实验号。
    title : str
        表标题。

    返回
    ----
    str
        Markdown 表格。
    """
    anchor = anchor or exp
    groups = group_by_variant(runs, exp)
    if not groups:
        return f"*（{exp} 暂无结果）*\n"

    lines = [f"<!-- TABLE:{anchor}:BEGIN -->", f"**{title}**", ""]
    lines.append("| 变体 | 种子数 | 平均轮数 | 测试集字符 | 测试集整牌 | "
                 "强扰动字符 | 合成域字符 | 参数量 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    order = sorted(groups)
    for var in order:
        rs = groups[var]
        lines.append(
            "| {v} | {n} | {ep} | {tc} | {tp} | {hc} | {sc} | {prm} |".format(
                v=var,
                n=len(rs),
                ep=(f"{statistics.fmean([float(r.get('epochs_run', 0)) for r in rs]):.1f}"
                    + ("（含预算截断）"
                       if any(r.get("budget_exhausted") for r in rs) else "")),
                tc=fmt_mean_std([_acc(r, "test", "char_acc") for r in rs]),
                tp=fmt_mean_std([_acc(r, "test", "plate_acc") for r in rs]),
                hc=fmt_mean_std([_acc(r, "hard_test", "char_acc") for r in rs]),
                sc=fmt_mean_std([_acc(r, "synth_test", "char_acc") for r in rs]),
                prm=((rs[0].get("meta", {}) or {}).get("model", {}) or {}
                     ).get("num_parameters", "—"),
            ))
    lines.append(f"<!-- TABLE:{anchor}:END -->")
    lines.append("")
    return "\n".join(lines) + "\n"


def table_position(runs: Sequence[Dict[str, Any]]) -> str:
    """生成基线运行（多随机种子）的逐位准确率表。

    参数
    ----
    runs : list of dict
        基线运行列表。

    返回
    ----
    str
        Markdown 表格。
    """
    if not runs:
        return "*（暂无基线结果）*\n"
    lines = [f"<!-- TABLE:POSITION:BEGIN -->","| 位置 | 真实测试集字符准确率 | 强扰动测试集 | 合成域测试集 |",
             "|---|---|---|---|"]
    for i in range(6):
        row = [f"第 {i + 1} 位"]
        for split in ("test", "hard_test", "synth_test"):
            vals = []
            for r in runs:
                v = _acc(r, split, f"per_position_{i}")
                if v is not None:
                    vals.append(v)
            row.append(fmt_mean_std(vals))
        lines.append("| " + " | ".join(row) + " |")
    # 汇总行
    row = ["**整牌（6 位全对）**"]
    for split in ("test", "hard_test", "synth_test"):
        row.append(fmt_mean_std([_acc(r, split, "plate_acc") for r in runs]))
    lines.append("| " + " | ".join(row) + " |")
    row = ["**字符（逐位平均）**"]
    for split in ("test", "hard_test", "synth_test"):
        row.append(fmt_mean_std([_acc(r, split, "char_acc") for r in runs]))
    lines.append("| " + " | ".join(row) + " |")
    lines.append("<!-- TABLE:POSITION:END -->")
    return "\n".join(lines) + "\n"


def load_evals(cfg) -> List[Dict[str, Any]]:
    """读取全部 ``*_eval.json``（由 ``evaluate/main.py`` 写出）。

    ★ 注意：CPU 推理计时在 ``*_eval.json`` 里，**不在** ``*_run.json`` 里
    —— 后者由训练脚本写出，此时还没做 CPU 计时。图表/报告要取 CPU 时间
    必须走这个函数。

    参数
    ----
    cfg : Config
        全局配置。

    返回
    ----
    list of dict
        每次独立评测的结果，含 ``run_name``。
    """
    out = []
    for f in sorted(glob.glob(str(resolve_path(cfg, "logs_dir") / "*_eval.json"))):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                d = json.load(fp)
        except Exception:                                          # noqa: BLE001
            continue
        d.setdefault("run_name", Path(f).name.replace("_eval.json", ""))
        out.append(d)
    return out


def table_cpu(runs: Sequence[Dict[str, Any]]) -> str:
    """生成 CPU 单张推理耗时表。

    参数
    ----
    runs : list of dict
        ``*_eval.json`` 的内容列表。

    返回
    ----
    str
        Markdown 表格。
    """
    rows = []
    for r in runs:
        ci = r.get("cpu_inference") or {}
        ms = ci.get("cpu_inference_ms_per_image")
        if ms is None:
            continue
        rows.append((str(r.get("run_name", "?")), float(ms),
                     float(ci.get("cpu_inference_images_per_sec") or 0.0),
                     int(ci.get("n_samples") or 0),
                     str(r.get("backend", "")), str(r.get("device_name", ""))))
    if not rows:
        return ("*（暂无 CPU 计时；请先对某次运行执行 "
                "`python evaluate/main.py --run <run>`）*\n")
    rows.sort(key=lambda x: x[1])
    lines = [f"<!-- TABLE:CPU:BEGIN -->","| 运行 | CPU 单张推理 (ms) | 吞吐 (张/秒) | 计时张数 | 训练后端 | 训练设备 |",
             "|---|---|---|---|---|---|"]
    for name, ms, ips, ns, be, dev in rows:
        lines.append(f"| {name} | {ms:.3f} | {ips:.1f} | {ns} | {be} | {dev} |")
    lines.append("<!-- TABLE:CPU:END -->")
    return "\n".join(lines) + "\n"


def table_ranking(runs: Sequence[Dict[str, Any]]) -> str:
    """按"测试集整牌准确率"给全部运行排名，便于一眼看出最优配置。

    参数
    ----
    runs : list of dict
        运行列表。

    返回
    ----
    str
        Markdown 表格。
    """
    rows = []
    for r in runs:
        tp = _acc(r, "test", "plate_acc")
        tc = _acc(r, "test", "char_acc")
        if tp is None:
            continue
        rows.append((r["run_name"], tc, tp,
                     _acc(r, "hard_test", "char_acc"),
                     _acc(r, "synth_test", "char_acc"),
                     r.get("epochs_run")))
    rows.sort(key=lambda x: -(x[2] or 0.0))
    lines = [f"<!-- TABLE:RANKING:BEGIN -->","| 排名 | 运行 | 测试字符 | 测试整牌 | 强扰动字符 | 合成域字符 | 轮数 |",
             "|---|---|---|---|---|---|---|"]
    for i, (name, tc, tp, hc, sc, ep) in enumerate(rows, 1):
        lines.append(
            f"| {i} | {name} | {tc * 100:.2f}% | {tp * 100:.2f}% | "
            f"{(hc or 0) * 100:.2f}% | {(sc or 0) * 100:.2f}% | {ep} |")
    lines.append("<!-- TABLE:RANKING:END -->")
    return "\n".join(lines) + "\n"


def table_midtrain(runs: Sequence[Dict[str, Any]]) -> str:
    """生成"中期收敛"对照表。

    ★ 为什么同时给"验证准确率"而不是只给 val_loss：交叉熵与平方误差的
    **绝对数值不可比**（交叉熵是 6 路之和、MSE 是 6 路平方和，量纲不同）。
    要比"学得快不快"，必须看**同一把尺子**上的指标，即验证字符准确率。

    参数
    ----
    runs : list of dict
        运行列表。

    返回
    ----
    str
        Markdown 表格。
    """
    def at(d: Dict[str, Any], epoch: int, key: str) -> Optional[float]:
        """取第 epoch 轮的某指标（找不到则取不超过它的最大轮）。"""
        hist = ((d.get("history") or {}).get("epochs")) or []
        cand = [h for h in hist if int(h.get("epoch", 0)) <= epoch
                and h.get(key) is not None]
        return float(cand[-1][key]) if cand else None

    def first_epoch_reaching(d: Dict[str, Any], key: str,
                             thr: float) -> Optional[int]:
        """首次达到某阈值的轮次；没达到返回 None。"""
        hist = ((d.get("history") or {}).get("epochs")) or []
        for h in hist:
            v = h.get(key)
            if v is not None and float(v) >= thr:
                return int(h.get("epoch", 0))
        return None

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in runs:
        nm = str(r.get("run_name", ""))
        if "_E3_" in nm or nm.startswith("E3_"):
            groups.setdefault(variant_of(nm, "E3"), []).append(r)
    if not groups:
        return "*（暂无 E3 结果）*\n"

    lines = [f"<!-- TABLE:MIDTRAIN:BEGIN -->","**同一把尺子：验证字符准确率（%）**", "",
             "| 变体 | 第 1 轮 | 第 5 轮 | 第 10 轮 | 第 20 轮 | 最终（最佳权重） |",
             "|---|---|---|---|---|---|"]
    for var in sorted(groups):
        rs = groups[var]
        cells = [f"{statistics.fmean(vs) * 100:.2f}" if (vs :=
                 [v for v in (at(r, ep, "val_char_acc") for r in rs)
                  if v is not None]) else "—"
                 for ep in (1, 5, 10, 20)]
        fin = [v for v in (_acc(r, "val", "char_acc") for r in rs)
               if v is not None]
        cells.append(f"{statistics.fmean(fin) * 100:.2f}" if fin else "—")
        lines.append(f"| {var} | " + " | ".join(cells) + " |")

    # 首个达到 50% / 80% 验证字符准确率的轮次
    lines += ["", "**首次达到目标验证字符准确率的轮次**（越小越快；"
              "`—` = 该轮数内未达到）", "",
              "| 变体 | 达到 50% | 达到 80% |", "|---|---|---|"]
    for var in sorted(groups):
        rs = groups[var]
        c50, c80 = [], []
        for r in rs:
            e50 = first_epoch_reaching(r, "val_char_acc", 0.50)
            e80 = first_epoch_reaching(r, "val_char_acc", 0.80)
            if e50:
                c50.append(e50)
            if e80:
                c80.append(e80)
        f50 = f"{statistics.fmean(c50):.0f} 轮" if c50 else "—"
        f80 = f"{statistics.fmean(c80):.0f} 轮" if c80 else "—"
        lines.append(f"| {var} | {f50} | {f80} |")

    # val_loss 仅作参考，并明确标注不可跨损失类型比较
    lines += ["", "**参考：val_loss 绝对值**（⚠️ 交叉熵是 6 路之和、MSE 是"
              " 6 路平方和，**两者不可直接比较**，此处仅供同变体内观察下降趋势）",
              "",
              "| 变体 | 第 1 轮 | 第 20 轮 | 最终 |", "|---|---|---|---|"]
    for var in sorted(groups):
        rs = groups[var]
        cells = []
        for ep in (1, 20):
            vs = [v for v in (at(r, ep, "val_loss") for r in rs) if v is not None]
            cells.append(f"{statistics.fmean(vs):.4f}" if vs else "—")
        fin = [v for v in (_acc(r, "val", "loss") for r in rs) if v is not None]
        cells.append(f"{statistics.fmean(fin):.4f}" if fin else "—")
        lines.append(f"| {var} | " + " | ".join(cells) + " |")
    lines.append("<!-- TABLE:MIDTRAIN:END -->")
    return "\n".join(lines) + "\n"


def build_all(cfg) -> Dict[str, str]:
    """生成全部表格。

    参数
    ----
    cfg : Config
        全局配置。

    返回
    ----
    dict
        键为占位符名，值为 Markdown 文本。
    """
    runs = load_runs(cfg)
    # 基线与 E1 的 shared 变体在配置上完全等价（arch=shared + 交叉熵 + 弱增强），
    # 因此把 baseline_s* 与 E1_shared_s* 合并统计，得到 3 个种子的基线均值。
    base = [r for r in runs if r["run_name"].startswith("baseline_")
            or r["run_name"].startswith("E1_shared_")]
    return {
        "baseline": (table_experiment(base, "E1", "基线（共享六头 + 弱增强）",
                                      anchor="BASELINE")
                     if base else "*（暂无基线结果）*\n"),
        "e1": table_experiment(runs_of(runs, "E1_"), "E1",
                               "E1 共享六头 vs 六个独立 MLP", anchor="E1"),
        "e3": table_experiment(runs_of(runs, "E3_"), "E3",
                               "E3 交叉熵 vs 平方误差", anchor="E3"),
        "e4": table_experiment(runs_of(runs, "E4_"), "E4",
                               "E4 批量大小 / SGD / 小批量", anchor="E4"),
        "e7": table_experiment(runs_of(runs, "E7_"), "E7",
                               "E7 数据增强强度", anchor="E7"),
        "midtrain": table_midtrain(runs),
        "ranking": table_ranking(runs),
        "position": table_position(base),
        "cpu": table_cpu(load_evals(cfg)),
    }


def inject(cfg, tables: Dict[str, str], report_path: Path) -> int:
    """把表格写入报告里的**锚点区间**，锚点本身保留，便于反复重注入。

    设计说明
    --------
    最初的做法是把 ``<!-- TABLE:E1 -->`` 整个替换成表格文本，问题是
    **替换后占位符就消失了**，第二次运行 ``--inject`` 就无法再更新。
    现在改成锚点对的形式：

    ```
    <!-- TABLE:E1:BEGIN -->
    ...（表格内容，可反复覆盖）...
    <!-- TABLE:E1:END -->
    ```

    注入时只覆盖 BEGIN 与 END 之间的内容，锚点保留。这样重跑实验后
    可以直接再执行一次 ``--inject`` 刷新全部表格。

    参数
    ----
    cfg : Config
        全局配置。
    tables : dict
        :func:`build_all` 的输出。
    report_path : Path
        报告路径。

    返回
    ----
    int
        被更新的锚点区间个数。
    """
    if not report_path.is_file():
        print(f"报告不存在：{report_path}")
        return 0
    text = report_path.read_text(encoding="utf-8")
    n = 0
    # ★ 锚点用大写（TABLE:E1:BEGIN），而 build_all 的键是小写（"e1"），
    # 这里统一 upper()，否则一个都匹配不上（返回"已注入 0 个"）。
    for key in tables:
        anchor = str(key).upper()
        begin = f"<!-- TABLE:{anchor}:BEGIN -->"
        end = f"<!-- TABLE:{anchor}:END -->"
        # ★ 表格生成函数自己也会带上锚点；这里先剥掉，再由 inject 统一加一对，
        # 否则每注入一次锚点就会翻倍（且重叠的锚点对会让正则匹配错位）。
        body = str(tables.get(key, "")).replace(begin, "").replace(end, "")
        block = f"{begin}\n{body}{end}"
        if begin in text and end in text:
            # 非贪婪匹配"最近的一个 END"，避免跨表吞并
            pattern = re.compile(
                re.escape(begin) + r"(?:(?!" + re.escape(begin) + r").)*?"
                + re.escape(end), re.DOTALL)
            text, cnt = pattern.subn(lambda _m: block, text)
            n += cnt
        elif f"<!-- TABLE:{anchor} -->" in text:
            # 兼容旧的单锚点写法
            text = text.replace(f"<!-- TABLE:{anchor} -->", block)
            n += 1
    report_path.write_text(text, encoding="utf-8")
    return n


def main(argv: Optional[Sequence[str]] = None) -> int:
    """命令行入口。

    参数
    ----
    argv : Sequence[str] or None
        命令行参数。

    返回
    ----
    int
        0 表示成功。
    """
    ap = argparse.ArgumentParser(description="由数值文件生成报告用 Markdown 表格")
    ap.add_argument("--config", default=None)
    ap.add_argument("--write", action="store_true",
                    help="写到 reports/tables/report_tables.md")
    ap.add_argument("--inject", action="store_true",
                    help="注入 reports/实验报告.md 的占位符")
    ap.add_argument("--report", default="reports/实验报告.md")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    tables = build_all(cfg)

    if args.write or not (args.write or args.inject):
        body = ["# 报告用表格（由 reports/build_report_tables.py 自动生成）",
                "",
                "> 本文件由数值产物生成，**不要手工编辑**；重跑实验后重新生成即可。",
                ""]
        for key in ("baseline", "e1", "e3", "e4", "e7", "midtrain", "position",
                    "cpu", "ranking"):
            body.append(f"## {key}")
            body.append("")
            body.append(tables.get(key, ""))
            body.append("")
        if args.write:
            out = resolve_path(cfg, "tables_dir") / "report_tables.md"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("\n".join(body), encoding="utf-8")
            print(f"已写出：{out}")
        else:
            print("\n".join(body))

    if args.inject:
        n = inject(cfg, tables, Path(args.report))
        print(f"已注入 {n} 个占位符 -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())