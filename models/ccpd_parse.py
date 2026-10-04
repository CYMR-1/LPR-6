# -*- coding: utf-8 -*-
"""CCPD 文件名解析、透视矫正、裁剪与标准化（§2.3）。

本模块实现训练侧预处理管线，**不做任何车牌检测**：只使用 CCPD 文件名中已标注的
四角顶点做透视矫正与裁剪（§0.3 禁止事项）。

管线总览（§2.3.2）
------------------
::

    原图 → 读四角顶点 → 透视矫正为正视车牌（168×52）
         → 按比例裁掉首位汉字区域（取右侧 6/7 宽度）
         → 缩放为 128×32 → 转灰度 → [0,1] 归一化
         → 全局零均值/单位方差标准化（统计量只在训练集上计算）

依赖约束
--------
* 透视矫正使用 **PIL 的 ``Image.PERSPECTIVE``**，8 个系数由 numpy 解线性方程组求得；
* **不引入 OpenCV**（§附录 B 第 9 条）。

关键约定
--------
* **四角顶点顺序为「右下、左下、左上、右上」**（与 CCPD 仓库 README 一致）。
  这是最容易搞错的一处，顺序错了会让矫正结果上下或左右翻转，
  必须通过 §2.3.3 的抽样人工核对确认。
* 顶点坐标分隔符在实际数据中可能是 ``&`` 或 ``,``，两种都要支持。
* 车牌号字段为 ``_`` 分隔的 7 个索引：索引 0 → 省份表，索引 1 → 字母表，
  索引 2–6 → 字母/数字表。本项目取**后 6 个索引**（跳过省份汉字）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

# --- 包引导：支持 `python models/ccpd_parse.py` 直接运行 --------------------
import sys as _sys
from pathlib import Path as _Path

if __package__ in (None, ""):
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from models.charset import (
    CCPD_ADS,
    CCPD_ALPHABETS,
    CCPD_PROVINCES,
    CHAR_TO_INDEX,
    NUM_CLASSES,
    SEQ_LEN,
    check_label_legal,
    encode_label,
)

# =============================================================================
# 1. 文件名解析
# =============================================================================

#: 顶点坐标成对解析：支持 "x&y" 与 "x,y" 两种分隔符
_SEP_CHARS = ("&", ",")


@dataclass
class CcpdRecord:
    """一条 CCPD 文件名标注解析结果。

    属性
    ----
    path : Path
        图片路径。
    area_ratio : int
        字段 0：车牌区域占整图面积比的整数表示。
    tilt_h : int
        字段 1：水平倾斜度。
    tilt_v : int
        字段 1：垂直倾斜度。
    bbox : tuple
        字段 2：车牌外接框 ``(xmin, ymin, xmax, ymax)``。
    corners : numpy.ndarray
        字段 3：四角顶点，形状 ``(4, 2)``，float64，
        **顺序为右下、左下、左上、右上**。
    label_indices : list of int
        字段 4：原始 7 个索引（含省份）。
    label : str
        后 6 位字符标签串（已跳过省份汉字）。
    label_classes : numpy.ndarray
        标签的 34 类索引，形状 ``(6,)``，int64。
    brightness : int
        字段 5。
    blurriness : int
        字段 6。
    subset : str
        子集名（如 ``"ccpd_base"``）；从文件名尾部提取，缺失时为空串。
    n_label_indices : int
        标签索引个数（7 位标准牌为 7，新能源 8 位牌为 8）。
    """

    path: Path
    area_ratio: int
    tilt_h: int
    tilt_v: int
    bbox: Tuple[int, int, int, int]
    corners: np.ndarray
    label_indices: List[int]
    label: str
    label_classes: np.ndarray
    brightness: int
    blurriness: int
    subset: str = ""
    n_label_indices: int = 7

    # -------------------------------------------------------------- 便捷属性
    @property
    def quad_area(self) -> float:
        """四角顶点构成的四边形面积（鞋带公式，单位：原图像素²）。

        返回
        ----
        float
            面积；``corners`` 形状 ``(4, 2)`` -> 标量。
        """
        x = self.corners[:, 0]
        y = self.corners[:, 1]
        return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)

    @property
    def bbox_area(self) -> float:
        """外接框面积。"""
        xmin, ymin, xmax, ymax = self.bbox
        return float(max(0, xmax - xmin) * max(0, ymax - ymin))


def parse_pair(text: str) -> Tuple[float, float]:
    """解析 ``"x&y"`` 或 ``"x,y"`` 形式的坐标对。

    参数
    ----
    text : str
        形如 ``"154&383"`` 或 ``"154,383"``。

    返回
    ----
    tuple of float
        ``(x, y)``。

    异常
    ----
    ValueError
        无法解析出两个数值。
    """
    for sep in _SEP_CHARS:
        if sep in text:
            parts = text.split(sep)
            if len(parts) == 2:
                return float(parts[0]), float(parts[1])
    raise ValueError(f"无法解析坐标对：{text!r}")


def parse_corners(text: str) -> np.ndarray:
    """解析四角顶点字段。

    参数
    ----
    text : str
        ``"x1&y1_x2&y2_x3&y3_x4&y4"``，顺序为**右下、左下、左上、右上**。
        分隔符 ``&`` 与 ``,`` 均可。

    返回
    ----
    numpy.ndarray
        形状 ``(4, 2)``，float64。

    异常
    ----
    ValueError
        顶点个数不为 4。
    """
    pts = [parse_pair(p) for p in text.split("_") if p]
    if len(pts) != 4:
        raise ValueError(f"四角顶点个数必须为 4，实际为 {len(pts)}：{text!r}")
    return np.asarray(pts, dtype=np.float64)


def parse_bbox(text: str) -> Tuple[int, int, int, int]:
    """解析外接框字段 ``"xmin&ymin_xmax&ymax"``。

    参数
    ----
    text : str
        外接框字符串。

    返回
    ----
    tuple of int
        ``(xmin, ymin, xmax, ymax)``。
    """
    p1, p2 = text.split("_")
    xmin, ymin = parse_pair(p1)
    xmax, ymax = parse_pair(p2)
    return int(xmin), int(ymin), int(xmax), int(ymax)


def parse_label_indices(text: str) -> List[int]:
    """解析车牌号字段的索引序列。

    参数
    ----
    text : str
        ``"_"`` 分隔的索引，例如 ``"0_0_22_27_27_33_16"``（7 个）。

    返回
    ----
    list of int
        索引列表，长度 7（标准牌）或 8（新能源牌）。
    """
    return [int(v) for v in text.split("_") if v != ""]


def indices_to_label(indices: Sequence[int]) -> str:
    """把 7 个 CCPD 索引还原为「后 6 位」字符标签串。

    参数
    ----
    indices : Sequence[int]
        至少 7 个索引：``[省份, 字母, ads×5]``。

    返回
    ----
    str
        长度 6 的标签串。

    异常
    ----
    ValueError
        索引个数不足 7，或索引越界（含哨兵值 ``O``）。
    """
    if len(indices) < 7:
        raise ValueError(f"索引个数必须 ≥ 7，实际为 {len(indices)}：{indices}")
    if indices[0] >= len(CCPD_PROVINCES):
        raise ValueError(f"省份索引越界：{indices[0]}")

    # 位置 1（汉字后第一位）使用 CCPD_ALPHABETS 表，索引 24 为哨兵 'O'
    if indices[1] >= len(CCPD_ALPHABETS) - 1:
        raise ValueError(f"字母索引非法（哨兵或越界）：{indices[1]}")
    chars: List[str] = [CCPD_ALPHABETS[indices[1]]]

    # 位置 2–6 使用 CCPD_ADS 表，索引 34 为哨兵 'O'
    for pos, idx in enumerate(indices[2:7]):
        if idx >= len(CCPD_ADS) - 1:
            raise ValueError(f"第 {pos + 2} 位 ads 索引非法（哨兵或越界）：{idx}")
        chars.append(CCPD_ADS[idx])

    # 后 6 位必须全部落在项目 34 类字符集内
    label = "".join(chars)
    for i, ch in enumerate(label):
        if ch not in CHAR_TO_INDEX:
            raise ValueError(f"第 {i} 位字符 {ch!r} 不在 34 类字符集内")
    return label


def extract_subset(name: str) -> str:
    """从文件名中提取子集名。

    参数
    ----
    name : str
        文件名，例如 ``"...-134-129_ccpd_base_005128.jpg"``。

    返回
    ----
    str
        子集名，如 ``"ccpd_base"``；找不到时返回 ``""``。

    说明
    ----
    实际遇到的书写形式不止一种，都需要兼容：

    * ``"...-118-10_ccpd_base_012856.jpg"`` —— ``ccpd`` 与 ``base`` 被拆成两个 token；
    * ``"..._ccpd_base_005128.jpg"``        —— 可能连写成 ``ccpd_base``；
    * 也可能带其它后缀，如 ``ccpd_fn``、``ccpd_green``、``ccpd_np``。

    做法：以 ``"_"`` 切分并去掉扩展名，找到 ``ccpd`` 相关的 token，
    拼成 ``ccpd_<name>`` 形式返回。
    """
    stem = name.rsplit(".", 1)[0]
    tokens = stem.split("_")
    for i, token in enumerate(tokens):
        # 形式一：连写 "ccpd_base"
        if token.startswith("ccpd_") and len(token) > len("ccpd_"):
            return token
        # 形式二：被拆开 "ccpd", "base"
        if token == "ccpd" and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            # 子集名只含小写字母；若下一个 token 是文件名数字串则说明没有子集名
            if nxt and all(c.islower() for c in nxt):
                return f"ccpd_{nxt}"
    return ""


def parse_filename(path: str | Path) -> CcpdRecord:
    """解析一个 CCPD 文件名，得到全部标注。

    参数
    ----
    path : str or Path
        图片路径；标注取自文件名本身。

    返回
    ----
    CcpdRecord
        解析结果。

    异常
    异常统一由调用方捕获并归类统计：
    ``ValueError``（字段数/格式错误）、``IndexError``。

    形状
    ----
    标量输入 -> dataclass（其中 ``corners`` 为 ``(4, 2)``，``label_classes`` 为 ``(6,)``）
    """
    path = Path(path)
    stem = path.name.rsplit(".", 1)[0]
    parts = stem.split("-")
    if len(parts) != 7:
        raise ValueError(f"文件名必须由 7 个 '-' 分隔字段组成，实际 {len(parts)}：{path.name}")

    area = int(parts[0])
    tilt = parts[1].split("_")
    tilt_h, tilt_v = (int(tilt[0]), int(tilt[1])) if len(tilt) == 2 else (0, 0)
    bbox = parse_bbox(parts[2])
    corners = parse_corners(parts[3])
    indices = parse_label_indices(parts[4])
    # 字段 5 为亮度；字段 6 为模糊度。
    # 注意：部分镜像/衍生版本会在模糊度字段后附加子集名
    # （例如 "...-118-10_ccpd_base_012856.jpg"），因此先按 "_" 取首段再转 int。
    brightness = int(parts[5].split("_")[0])
    blurriness = int(parts[6].split("_")[0])

    label = indices_to_label(indices)
    classes = encode_label(label)

    return CcpdRecord(
        path=path,
        area_ratio=area,
        tilt_h=tilt_h,
        tilt_v=tilt_v,
        bbox=bbox,
        corners=corners,
        label_indices=indices,
        label=label,
        label_classes=classes,
        brightness=brightness,
        blurriness=blurriness,
        subset=extract_subset(path.name),
        n_label_indices=len(indices),
    )


# =============================================================================
# 2. 过滤规则（§2.3.2）
# =============================================================================

#: 丢弃原因代码 -> 中文说明。所有被丢弃的样本都必须按此归类计数并写入日志。
DROP_REASONS: Dict[str, str] = {
    "field_count": "文件名字段数不为 7",
    "parse_error": "文件名解析异常（数值/坐标格式错误）",
    "not_7_chars": "不是 7 位标准牌（含 8 位新能源牌）",
    "illegal_char": "含 34 类之外的字符（如 I/O 哨兵、非法索引）",
    "position_illegal": "位置约束不满足（首位非字母等）",
    "coord_negative": "顶点坐标含负值",
    "coord_oob": "顶点坐标越界（超出图像范围）",
    "quad_too_small": "四边形面积过小",
    "bbox_invalid": "外接框非法（宽或高为 0）",
    "read_error": "图像无法读取",
    "green_plate": "新能源 8 位牌（子集 ccpd_green）",
}


@dataclass
class FilterStats:
    """过滤统计器：记录各类丢弃原因的数量（§2.3.2 必须记录）。

    属性
    ----
    kept : int
        保留的样本数。
    dropped : dict
        原因代码 -> 计数。
    dropped_examples : dict
        原因代码 -> 若干示例文件名（便于排查）。
    """

    kept: int = 0
    dropped: Dict[str, int] = field(default_factory=dict)
    dropped_examples: Dict[str, List[str]] = field(default_factory=dict)

    def add_drop(self, reason: str, name: str, max_examples: int = 5) -> None:
        """记录一次丢弃。

        参数
        ----
        reason : str
            原因代码（:data:`DROP_REASONS` 的键）。
        name : str
            触发的文件名。
        max_examples : int
            每类最多保留的示例数。

        返回
        ----
        None
        """
        self.dropped[reason] = self.dropped.get(reason, 0) + 1
        ex = self.dropped_examples.setdefault(reason, [])
        if len(ex) < max_examples:
            ex.append(name)

    def add_keep(self) -> None:
        """记录一次保留。"""
        self.kept += 1

    @property
    def total(self) -> int:
        """扫描的样本总数。"""
        return self.kept + sum(self.dropped.values())

    def summary(self) -> Dict[str, object]:
        """汇总为可写入 JSON 日志的结构。

        返回
        ----
        dict
            含 ``total`` / ``kept`` / ``dropped_total`` / ``drop_rate`` /
            ``by_reason``（带中文说明）。
        """
        dropped_total = sum(self.dropped.values())
        return {
            "total_scanned": self.total,
            "kept": self.kept,
            "dropped_total": dropped_total,
            "drop_rate": round(dropped_total / self.total, 6) if self.total else 0.0,
            "by_reason": {
                reason: {
                    "count": self.dropped.get(reason, 0),
                    "desc": DROP_REASONS.get(reason, reason),
                    "examples": self.dropped_examples.get(reason, []),
                }
                for reason in DROP_REASONS
                if self.dropped.get(reason, 0) > 0
            },
        }


def validate_record(
    rec: CcpdRecord,
    image_size: Optional[Tuple[int, int]] = None,
    min_quad_area_px: float = 200.0,
    allow_out_of_bounds_px: float = 2.0,
    positions: Optional[Sequence[int]] = None,
) -> Optional[str]:
    """按 §2.3.2 的过滤规则校验一条记录。

    参数
    ----
    rec : CcpdRecord
        待校验记录。
    image_size : tuple or None
        ``(宽, 高)``；给出时启用越界检查。
    min_quad_area_px : float
        四边形面积下限。
    allow_out_of_bounds_px : float
        允许的轻微越界容差（像素）。
    positions : Sequence[int] or None
        位置约束；给了就做位置合法性检查（E9 用）。

    返回
    ----
    str or None
        合法返回 ``None``；否则返回原因代码。
    """
    # ① 只保留 7 位标准牌（新能源 8 位牌在解析阶段就已带 n_label_indices=8）
    if rec.n_label_indices != 7:
        return "green_plate" if rec.n_label_indices == 8 else "not_7_chars"

    # ② 标签必须是 6 位且字符全部落在 34 类内
    if len(rec.label) != SEQ_LEN:
        return "not_7_chars"
    legal, bad = check_label_legal(rec.label, positions)
    if not legal and positions is not None:
        return "position_illegal"

    # ③ 顶点坐标：负值
    if np.any(rec.corners < 0):
        return "coord_negative"

    # ④ 顶点坐标：越界
    if image_size is not None:
        w, h = image_size
        if (
            np.any(rec.corners[:, 0] > w - 1 + allow_out_of_bounds_px)
            or np.any(rec.corners[:, 1] > h - 1 + allow_out_of_bounds_px)
        ):
            return "coord_oob"

    # ⑤ 四边形面积过小
    if rec.quad_area < min_quad_area_px:
        return "quad_too_small"

    # ⑥ 外接框非法
    xmin, ymin, xmax, ymax = rec.bbox
    if xmax <= xmin or ymax <= ymin:
        return "bbox_invalid"

    return None


# =============================================================================
# 3. 透视矫正（PIL Image.PERSPECTIVE + numpy 解系数）
# =============================================================================


def find_perspective_coeffs(
    dst_pts: np.ndarray, src_pts: np.ndarray
) -> List[float]:
    """求解 PIL ``Image.PERSPECTIVE`` 所需的 8 个系数。

    PIL 的变换约定是**从目标坐标反查源坐标**（逆映射）::

        x_src = (a·x_dst + b·y_dst + c) / (g·x_dst + h·y_dst + 1)
        y_src = (d·x_dst + e·y_dst + f) / (g·x_dst + h·y_dst + 1)

    因此 :func:`perspective_rectify` 中传入的 ``dst_pts`` 是矫正后的矩形四角，
    ``src_pts`` 是原图中的车牌四角。

    参数
    ----
    dst_pts : numpy.ndarray
        目标点，形状 ``(4, 2)``。
    src_pts : numpy.ndarray
        源点，形状 ``(4, 2)``，与 ``dst_pts`` 逐点对应。

    返回
    ----
    list of float
        ``[a, b, c, d, e, f, g, h]``。

    形状
    ----
    ``(4, 2), (4, 2)`` -> ``list[8]``

    说明
    ----
    对每个对应点对可建立两个线性方程，把 8 个未知数排成向量后
    用 ``numpy.linalg.lstsq`` 解超定方程组（8 个方程、8 个未知数）。
    """
    dst = np.asarray(dst_pts, dtype=np.float64)
    src = np.asarray(src_pts, dtype=np.float64)
    if dst.shape != (4, 2) or src.shape != (4, 2):
        raise ValueError(f"点集形状必须为 (4, 2)，实际为 {dst.shape} / {src.shape}")

    a_mat = np.zeros((8, 8), dtype=np.float64)
    b_vec = np.zeros(8, dtype=np.float64)
    for i in range(4):
        xd, yd = dst[i]
        xs, ys = src[i]
        a_mat[2 * i] = [xd, yd, 1, 0, 0, 0, -xs * xd, -xs * yd]
        b_vec[2 * i] = xs
        a_mat[2 * i + 1] = [0, 0, 0, xd, yd, 1, -ys * xd, -ys * yd]
        b_vec[2 * i + 1] = ys

    coeffs, *_ = np.linalg.lstsq(a_mat, b_vec, rcond=None)
    return [float(c) for c in coeffs]


def perspective_rectify(
    image: Image.Image,
    corners: np.ndarray,
    out_size: Tuple[int, int],
) -> Image.Image:
    """按四角顶点把车牌透视矫正为正视矩形。

    参数
    ----
    image : PIL.Image.Image
        原图（任意颜色模式）。
    corners : numpy.ndarray
        四角顶点，形状 ``(4, 2)``，顺序为**右下、左下、左上、右上**。
    out_size : tuple of int
        输出尺寸 ``(宽 W, 高 H)``。

    返回
    ----
    PIL.Image.Image
        矫正后的图像，尺寸 ``(W, H)``，模式与输入一致。

    形状
    ----
    输入图像 ``(H0, W0)`` -> 输出 ``(H, W)``；``corners`` 为 ``(4, 2)``

    说明
    ----
    目标四角顺序与源点严格对应，即
    ``右下 → (W-1, H-1)``、``左下 → (0, H-1)``、``左上 → (0, 0)``、``右上 → (W-1, 0)``。
    这一点决定了矫正结果的方向正确性，顺序写错会导致图像翻转。
    """
    w, h = int(out_size[0]), int(out_size[1])
    dst_pts = np.array(
        [
            [w - 1, h - 1],  # 右下
            [0, h - 1],      # 左下
            [0, 0],          # 左上
            [w - 1, 0],      # 右上
        ],
        dtype=np.float64,
    )
    coeffs = find_perspective_coeffs(dst_pts, np.asarray(corners, dtype=np.float64))
    return image.transform(
        (w, h), Image.PERSPECTIVE, coeffs, resample=Image.BICUBIC
    )


# =============================================================================
# 4. 裁剪 / 缩放 / 灰度 / 归一化 的完整管线
# =============================================================================


@dataclass
class PrepParams:
    """预处理参数集合（从 ``configs/default.yaml`` 构造）。

    属性
    ----
    rectify_size : tuple of int
        透视矫正目标 ``(宽, 高)``，基线 ``(168, 52)``。
    input_size : tuple of int
        最终输入尺寸 ``(宽, 高)``，基线 ``(128, 32)``。
    keep_right_fraction : float
        保留右侧宽度比例，``6/7`` 表示裁掉首位汉字区域。
    grayscale : bool
        是否转灰度（本项目恒定为 ``True``）。
    min_quad_area_px : float
        四边形面积下限。
    allow_out_of_bounds_px : float
        越界容差。
    """

    rectify_size: Tuple[int, int] = (168, 52)
    input_size: Tuple[int, int] = (128, 32)
    keep_right_fraction: float = 6.0 / 7.0
    grayscale: bool = True
    min_quad_area_px: float = 200.0
    allow_out_of_bounds_px: float = 2.0

    @classmethod
    def from_config(cls, cfg) -> "PrepParams":
        """从配置对象构造。

        参数
        ----
        cfg : models.config.Config
            全局配置。

        返回
        ----
        PrepParams
        """
        return cls(
            rectify_size=(
                int(cfg.ccpd.rectify_size[0]),
                int(cfg.ccpd.rectify_size[1]),
            ),
            input_size=(
                int(cfg.ccpd.input_size[0]),
                int(cfg.ccpd.input_size[1]),
            ),
            keep_right_fraction=float(cfg.ccpd.keep_right_fraction),
            grayscale=bool(cfg.ccpd.grayscale),
            min_quad_area_px=float(cfg.ccpd.min_quad_area_px),
            allow_out_of_bounds_px=float(cfg.ccpd.allow_out_of_bounds_px),
        )

    # ------------------------------------------------------------------ 派生量
    @property
    def crop_x0(self) -> int:
        """裁掉首位汉字后的水平起始像素（在矫正图上）。

        返回
        ----
        int
            起始列；取 ``round(W · (1 − keep_right_fraction))``。
        """
        w = self.rectify_size[0]
        return int(round(w * (1.0 - self.keep_right_fraction)))

    @property
    def input_dim(self) -> int:
        """展平后的输入维度（基线 32 × 128 = 4096）。"""
        return int(self.input_size[0] * self.input_size[1])


def preprocess_crop(
    image: Image.Image,
    corners: np.ndarray,
    params: PrepParams,
) -> np.ndarray:
    """把一张 CCPD 原图处理为归一化后的灰度数组（未做全局标准化）。

    步骤（严格按 §2.3.2）
    --------------------
    1. 透视矫正为 ``params.rectify_size``；
    2. 裁掉左侧首位汉字区域（取右侧 ``keep_right_fraction`` 宽度）；
    3. 缩放为 ``params.input_size``；
    4. 转灰度；
    5. ``[0, 1]`` 归一化。

    参数
    ----
    image : PIL.Image.Image
        原图。
    corners : numpy.ndarray
        四角顶点 ``(4, 2)``，顺序为右下、左下、左上、右上。
    params : PrepParams
        预处理参数。

    返回
    ----
    numpy.ndarray
        形状 ``(H, W)``，float32，取值范围 ``[0, 1]``。
        本函数**不**做全局零均值/单位方差标准化，那一步只允许用训练集统计量，
        由 :class:`~models.dataset.GlobalStandardizer` 在数据集层完成。

    形状
    ----
    原图 ``(H0, W0)`` -> ``(params.input_size[1], params.input_size[0])``
    """
    rect = perspective_rectify(image, corners, params.rectify_size)
    if params.grayscale:
        rect = rect.convert("L")

    # 裁掉首位汉字：取右侧 6/7
    w, h = rect.size
    x0 = params.crop_x0
    if x0 > 0:
        rect = rect.crop((x0, 0, w, h))

    # 缩放为最终输入尺寸
    rect = rect.resize(params.input_size, resample=Image.BILINEAR)

    arr = np.asarray(rect, dtype=np.float32)
    arr = arr / 255.0
    return arr


def process_record(
    rec: CcpdRecord,
    params: PrepParams,
) -> np.ndarray:
    """读取图片并按管线处理为输入数组。

    参数
    ----
    rec : CcpdRecord
        解析后的记录。
    params : PrepParams
        预处理参数。

    返回
    ----
    numpy.ndarray
        形状 ``(H, W)``，float32，取值 ``[0, 1]``。

    形状
    ----
    磁盘图像 -> ``(params.input_size[1], params.input_size[0])``
    """
    with Image.open(rec.path) as im:
        im = im.convert("RGB") if im.mode not in ("L", "RGB") else im
        return preprocess_crop(im, rec.corners, params)


# =============================================================================
# 5. 调试入口
# =============================================================================

if __name__ == "__main__":  # pragma: no cover
    import sys

    if len(sys.argv) > 1:
        rec = parse_filename(sys.argv[1])
        print("path        :", rec.path.name)
        print("subset      :", rec.subset)
        print("corners     :\n", rec.corners)
        print("bbox        :", rec.bbox)
        print("indices     :", rec.label_indices, f"(n={rec.n_label_indices})")
        print("label       :", rec.label)
        print("classes     :", rec.label_classes.tolist())
        print("quad_area   :", round(rec.quad_area, 1))
    else:
        demo = (
            "025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-"
            "0_0_22_27_27_33_16-37-15.jpg"
        )
        rec = parse_filename(demo)
        print("示例解析：")
        print("  label     :", rec.label)
        print("  classes   :", rec.label_classes.tolist())
        print("  corners   :", rec.corners.tolist())
        print("  quad_area :", round(rec.quad_area, 1))
        # 官方示例 0_0_22_27_27_33_16：
        #   位置 1 用 CCPD_ALPHABETS[0] = 'A'
        #   位置 2-6 用 CCPD_ADS[22,27,27,33,16] = Y,3,3,9,S
        #   故后 6 位 = 'A' + 'Y339S' = AY339S
        assert rec.label == "AY339S", rec.label
        print("  自检通过：后 6 位 = AY339S")
        # 幂等交叉验证：直接按映射表算一遍，必须与解析结果一致
        expect = CCPD_ALPHABETS[0] + "".join(CCPD_ADS[i] for i in (22, 27, 27, 33, 16))
        assert rec.label == expect == "AY339S", (rec.label, expect)
        # 位置 1 必须走 alphabets 表而不是 ads 表：两表仅在前 24 项相同
        assert CCPD_ADS[:24] == CCPD_ALPHABETS[:24]
        assert len(CCPD_ALPHABETS) == 25 and len(CCPD_ADS) == 35
        print("  自检通过：位置1 走 alphabets 表；两表前 24 项一致、长度分别为 25 / 35")
        # 数字位交叉验证：索引 24 在 ads 中是数字 '0'（而在 alphabets 中是哨兵 'O'）
        rec3 = parse_filename(
            "025-95_113-154,383_386,473-386,473_177,454_154,383_363,402-"
            "0_16_15_29_24_33_27-37-15.jpg"
        )
        expect3 = CCPD_ALPHABETS[16] + "".join(CCPD_ADS[i] for i in (15, 29, 24, 33, 27))
        assert rec3.label == expect3 == "SR5093", (rec3.label, expect3)
        print(f"  自检通过：0_16_15_29_24_33_27 -> {rec3.label}（16->S, 15->R, 29->5, 24->0, 33->9, 27->3）")
        # 逗号分隔版本
        rec2 = parse_filename(
            "025-95_113-154,383_386,473-386,473_177,454_154,383_363,402-"
            "0_0_22_27_27_33_16-37-15.jpg"
        )
        assert np.allclose(rec2.corners, rec.corners)
        print("  自检通过：& 与 , 两种分隔符解析一致")