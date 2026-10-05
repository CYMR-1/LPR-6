# -*- coding: utf-8 -*-
"""可视化：训练曲线、样本网格、混淆矩阵、错误样本（§2.3.3 / §6 / §8.3）。

包含两类产出
------------
1. **人工核对网格**（:func:`plot_check_grid`）：数据准备的自检门槛（§2.3.3 必做），
   随机抽 20 张输出「裁剪图 + 标签字符串」，用于确认①四角顶点顺序解析无误、
   ②后 6 位字符与图像逐位对应。**核对通过前不得进入训练。**
2. **评测用图**：训练曲线、逐位置混淆矩阵、错误样本网格。

所有绘图函数都会在文件名与标题中使用中文，因此统一通过
:func:`setup_chinese_font` 配置 matplotlib 的中文字体。

用法
----
::

    python evaluate/visualize.py check-grid                  # 数据准备的自检网格
    python evaluate/visualize.py selftest                    # 绘图链路自检
    python evaluate/visualize.py error-grid --run final_s42  # 错误样本网格（需先评测）
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

# --- 包引导：支持直接运行本文件 ---------------------------------------------
import sys as _sys

if __package__ in (None, ""):
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")  # 服务器/命令行环境无显示设备
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from models.charset import INDEX_TO_CHAR, NUM_CLASSES, SEQ_LEN, decode_batch  # noqa: E402

# =============================================================================
# 0. 中文字体
# =============================================================================

_FONT_READY = False


def setup_chinese_font(font_path: str = "C:/Windows/Fonts/simhei.ttf") -> bool:
    """配置 matplotlib 使用支持中文的字体（幂等）。

    参数
    ----
    font_path : str
        中文字体文件路径。

    返回
    ----
    bool
        配置成功返回 ``True``；字体缺失时返回 ``False``（不抛异常，
        以免因环境缺少字体导致整条实验流水线中断）。

    形状
    ----
    标量 -> 标量
    """
    global _FONT_READY
    if _FONT_READY:
        return True
    try:
        if not Path(font_path).exists():
            print(f"[viz] 警告：中文字体不存在 {font_path}，图中的中文可能显示为方块")
            return False
        font_manager.fontManager.addfont(font_path)
        name = font_manager.FontProperties(fname=font_path).get_name()
        plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示
        _FONT_READY = True
        return True
    except Exception as exc:  # pragma: no cover
        print(f"[viz] 警告：配置中文字体失败：{exc}")
        return False


# =============================================================================
# 1. 人工核对网格（数据准备自检）
# =============================================================================


def plot_check_grid(
    images: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    title: str = "CCPD 裁剪结果人工核对（§2.3.3）",
    n: int = 20,
    cols: int = 4,
    seed: int = 42,
    source_names: Optional[Sequence[str]] = None,
) -> Path:
    """输出「裁剪图 + 标签字符串」网格图，用于人工核对。

    核对要点（§2.3.3）
    ------------------
    ① 四角顶点顺序解析无误 —— 图像不应上下翻转或左右镜像；
    ② 后 6 位字符与图像逐位对应 —— 图中可见的每个字符应与标签逐位一致。

    参数
    ----
    images : numpy.ndarray
        形状 ``(N, H, W)``，取值 ``[0, 255]``（uint8）或 ``[0, 1]``（float）。
    labels : numpy.ndarray
        形状 ``(N, 6)``，int64 类别索引。
    out_path : Path
        输出图片路径。
    title : str
        图标题。
    n : int
        抽样张数。
    cols : int
        列数。
    seed : int
        抽样随机种子（保证核对样本可复现）。
    source_names : Sequence[str] or None
        来源文件名，用于副标题溯源。

    返回
    ----
    Path
        实际写出路径。

    形状
    ----
    ``(N, H, W)`` + ``(N, 6)`` -> PNG 文件
    """
    setup_chinese_font()
    n_total = int(images.shape[0])
    n = min(int(n), n_total)
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(n_total, size=n, replace=False)

    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 1.5 + 1.0))
    axes = np.atleast_2d(axes)

    texts = decode_batch(labels[idx])
    for k, i in enumerate(idx):
        r, c = divmod(k, cols)
        ax = axes[r][c]
        img = np.asarray(images[i], dtype=np.float32)
        if img.max() > 1.5:  # 说明是 0-255 量纲
            img = img / 255.0
        ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_title(f"{texts[k]}", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    # 清空多余子图
    for k in range(n, rows * cols):
        r, c = divmod(k, cols)
        axes[r][c].axis("off")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    # 同时落一份文本清单，便于逐条核对与追溯
    txt_path = out_path.with_suffix(".txt")
    with open(txt_path, "w", encoding="utf-8") as fp:
        fp.write(f"# {title}\n# 抽样 {n} 张，种子 {seed}\n")
        for k, i in enumerate(idx):
            src = source_names[int(i)] if source_names is not None else ""
            fp.write(f"{k + 1:3d}. index={int(i):6d}  label={texts[k]}  source={src}\n")
    print(f"[viz] 人工核对网格已写出：{out_path}")
    print(f"[viz] 核对清单已写出：{txt_path}")
    return out_path


# =============================================================================
# 2. 训练曲线
# =============================================================================


def plot_training_curves(
    history: Dict[str, List[float]],
    out_path: Path,
    title: str = "训练曲线",
    metrics: Optional[Sequence[str]] = None,
) -> Path:
    """绘制训练/验证 loss 与准确率曲线。

    参数
    ----
    history : dict
        ``{指标名: 逐 epoch 数值列表}``。常见键：
        ``train_loss``、``val_loss``、``train_char_acc``、``val_char_acc``、
        ``val_seq_acc``。
    out_path : Path
        输出路径。
    title : str
        图标题。
    metrics : Sequence[str] or None
        要画的指标；``None`` 时自动选取存在的指标。

    返回
    ----
    Path
        写出路径。

    形状
    ----
    ``{str: list[float]}`` -> PNG 文件
    """
    setup_chinese_font()
    if metrics is None:
        want = ["train_loss", "val_loss", "train_char_acc", "val_char_acc", "val_seq_acc"]
        metrics = [m for m in want if m in history and len(history[m]) > 0]

    if not metrics:
        raise ValueError("history 中没有可绘制的指标")

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.6))
    axes = np.atleast_1d(axes)
    pretty = {
        "train_loss": "训练损失",
        "val_loss": "验证损失",
        "train_char_acc": "训练字符准确率",
        "val_char_acc": "验证字符准确率",
        "val_plate_acc": "验证整牌准确率",
        "val_seq_acc": "验证整牌准确率",
        "train_plate_acc": "训练整牌准确率",
        "lr": "学习率",
    }
    for ax, m in zip(axes, metrics):
        ax.plot(range(1, len(history[m]) + 1), history[m], linewidth=1.6)
        ax.set_xlabel("epoch")
        ax.set_ylabel(pretty.get(m, m))
        ax.set_title(pretty.get(m, m))
        ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# =============================================================================
# 3. 混淆矩阵
# =============================================================================


def plot_confusion_matrix(
    matrix: np.ndarray,
    out_path: Path,
    position: int,
    normalize: bool = True,
    title: Optional[str] = None,
) -> Path:
    """绘制某个位置的 34×34 混淆矩阵。

    参数
    ----
    matrix : numpy.ndarray
        形状 ``(34, 34)``，行 = 真实类别，列 = 预测类别。
    out_path : Path
        输出路径。
    position : int
        位置编号（0–5），用于标题。
    normalize : bool
        是否按真实类别归一化（行归一化）。
    title : str or None
        自定义标题。

    返回
    ----
    Path
        写出路径。

    形状
    ----
    ``(34, 34)`` -> PNG 文件
    """
    setup_chinese_font()
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"混淆矩阵必须是方阵，实际形状 {mat.shape}")
    c = mat.shape[0]

    if normalize:
        denom = mat.sum(axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        mat = mat / denom

    labels = [INDEX_TO_CHAR[i] for i in range(c)]
    fig, ax = plt.subplots(figsize=(9.5, 8.2))
    im = ax.imshow(mat, cmap="viridis", vmin=0.0, vmax=mat.max() if mat.max() > 0 else 1.0)
    ax.set_xticks(range(c))
    ax.set_yticks(range(c))
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title(title or f"位置 {position + 1} 混淆矩阵" + ("（行归一化）" if normalize else ""),
                 fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# =============================================================================
# 4. 错误样本网格
# =============================================================================


def plot_error_samples(
    images: np.ndarray,
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    confidences: np.ndarray,
    out_path: Path,
    n: int = 20,
    cols: int = 4,
    title: str = "错误样本（优先高置信错误）",
) -> Path:
    """绘制错误样本网格，标题标注「真实 → 预测」与置信度。

    参数
    ----
    images : numpy.ndarray
        形状 ``(M, H, W)``，错误样本图像。
    true_labels : numpy.ndarray
        形状 ``(M, 6)``，真实类别索引。
    pred_labels : numpy.ndarray
        形状 ``(M, 6)``，预测类别索引。
    confidences : numpy.ndarray
        形状 ``(M,)``，整牌置信度（六位概率乘积或最小值）。
    out_path : Path
        输出路径。
    n : int
        展示张数。
    cols : int
        列数。
    title : str
        图标题。

    返回
    ----
    Path
        写出路径。

    形状
    ----
    ``(M, H, W)`` + ``(M, 6)`` + ``(M, 6)`` + ``(M,)`` -> PNG 文件
    """
    setup_chinese_font()
    m = int(images.shape[0])
    if m == 0:
        print("[viz] 没有错误样本，跳过绘制")
        return out_path
    n = min(int(n), m)
    # 按置信度降序：优先展示"模型很自信却错了"的样本
    order = np.argsort(-np.asarray(confidences, dtype=np.float64))[:n]

    true_text = decode_batch(true_labels[order])
    pred_text = decode_batch(pred_labels[order])

    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.6, rows * 1.7 + 1.0))
    axes = np.atleast_2d(axes)
    for k in range(n):
        r, c = divmod(k, cols)
        ax = axes[r][c]
        img = np.asarray(images[order[k]], dtype=np.float32)
        if img.max() > 1.5:
            img = img / 255.0
        ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_title(f"真:{true_text[k]}\n预:{pred_text[k]}  p={confidences[order[k]]:.2f}",
                     fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    for k in range(n, rows * cols):
        r, c = divmod(k, cols)
        axes[r][c].axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="可视化工具")
    ap.add_argument("command", choices=["check-grid", "selftest", "error-grid"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=None,
                    help="输出文件；缺省时 check-grid 用 reports/figs/ccpd_check_grid.png，"
                         "error-grid 用 <figs_dir>/<run>_errors.png")
    ap.add_argument("--run", type=str, default="final_s42",
                    help="error-grid：运行短名，读 <logs_dir>/<run>_errors.npz")
    ap.add_argument("--cols", type=int, default=0,
                    help="error-grid：网格列数，0 表示取配置 viz.grid_cols")
    ap.add_argument("--title", type=str, default=None,
                    help="error-grid：自定义图标题")
    args = ap.parse_args()

    if args.command == "selftest":
        # 用随机噪声与随机标签自检绘图链路
        rng = np.random.default_rng(0)
        imgs = rng.random((10, 32, 128)).astype(np.float32)
        labs = rng.integers(0, NUM_CLASSES, size=(10, SEQ_LEN))
        p = plot_check_grid(imgs, labs, Path("reports/figs/_selftest_grid.png"), n=8, seed=1)
        print("绘图自检通过：", p)

    elif args.command == "error-grid":
        from models.config import load_config, resolve_path

        cfg = load_config()
        run = str(args.run)
        npz = resolve_path(cfg, "logs_dir") / f"{run}_errors.npz"
        if not npz.is_file():
            raise SystemExit(
                f"[viz] 缺少 {npz}\n"
                f"      请先用评测生成错误样本：python evaluate/main.py --run {run}")
        with np.load(npz, allow_pickle=False) as data:
            need = ("error_images", "error_truth", "error_pred", "error_conf")
            missing = [k for k in need if k not in data.files]
            if missing:
                raise SystemExit(
                    f"[viz] {npz} 缺少字段 {missing}；实际字段 = {list(data.files)}")
            images = data["error_images"]
            n_err = int(images.shape[0])
            if n_err == 0:
                print(f"[viz] {run} 在该测试集上没有错误样本，无需绘图")
                raise SystemExit(0)
            out_path = (Path(args.out) if args.out
                        else resolve_path(cfg, "figs_dir") / f"{run}_errors.png")
            title = args.title or f"{run} 错误样本（优先高置信错误，共 {n_err} 张）"
            plot_error_samples(
                images, data["error_truth"], data["error_pred"], data["error_conf"],
                out_path, n=args.n, cols=int(args.cols) or int(cfg.viz.grid_cols),
                title=title,
            )
        print(f"[viz] {run}：错误样本共 {n_err} 张，图中展示 {min(int(args.n), n_err)} 张")

    else:  # check-grid
        from models.config import ensure_dirs, load_config
        from models.ccpd_parse import PrepParams
        from models.dataset import build_cache, discover_images

        cfg = load_config()
        ensure_dirs(cfg)
        files = discover_images(Path(cfg.paths.ccpd_root))
        if not files:
            raise SystemExit("data/ccpd/ 下没有图片，请先准备数据")
        params = PrepParams.from_config(cfg)
        # 抽取时用固定种子挑选，保证核对样本可复现
        rng = np.random.default_rng(int(args.seed))
        pick = rng.choice(len(files), size=min(args.n * 3, len(files)), replace=False)
        chosen = [files[int(i)] for i in sorted(pick)]
        res = build_cache(chosen, params, verbose=False)
        print(f"核对用样本：请求 {len(chosen)}，成功预处理 {res.stats.kept}")
        out_path = Path(args.out) if args.out else Path("reports/figs/ccpd_check_grid.png")
        plot_check_grid(
            res.images, res.labels, out_path,
            n=args.n, cols=int(cfg.viz.grid_cols), seed=int(args.seed),
            source_names=[r.path.name for r in res.records],
        )
        print("请打开图片人工核对：①顶点顺序是否上下/左右翻转 ②字符是否与标签逐位一致")