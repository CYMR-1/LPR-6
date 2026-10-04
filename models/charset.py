# -*- coding: utf-8 -*-
"""字符集定义与 CCPD 索引映射表。

本模块是整个工程编码口径的唯一来源，对应规格 §2.1 与 §2.3.1。

设计要点
--------
1. 本项目最终类别空间为 **34 类**：数字 ``0-9`` + 大写字母 ``A-Z`` 去掉 ``I``、``O``。
2. **类别索引顺序遵循 CCPD 官方 ``ads`` 表的顺序**，即先 24 个字母、后 10 个数字：

   ==========  ================================================
   索引区间     含义
   ==========  ================================================
   0 – 23      字母 ``A-Z`` 去掉 ``I``、``O``（24 个）
   24 – 33     数字 ``0-9``
   ==========  ================================================

   这样做的直接好处是：CCPD 文件名中的 ``ads`` 索引 **本身就是本项目的类别索引**，
   无需二次映射表，从源头消除映射错位的可能。

3. 官方映射表来源（**已逐项核对，非凭记忆编写**）：

   * CCPD 仓库 README 的 *Dataset Annotations* 一节（本地存档
     ``data/CCPD_README.md``）；
   * CCPD 仓库源码 ``rpnet/demo.py``（第 29–35 行）中的 ``provinces / alphabets / ads``
     三个数组，2024 年核对结果与 README 完全一致。

4. 位置约束（E9）：第 1 位（汉字后第一位）仅字母，共 24 类；第 2–6 位为全 34 类。

注意：CCPD 三张表的**最后一个元素都是字母 ``O``**，作者用它作为"无字符"的哨兵值
（中国车牌字符集里没有 ``O``）。任何读到索引等于"表长度 - 1"的记录都必须视为
非法样本丢弃，不得当作 ``0`` 或字母处理。
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

# =============================================================================
# 1. 本项目 34 类字符集（§2.1）
# =============================================================================

#: 数字字符按数值升序
DIGITS: str = "0123456789"

#: 大写字母，剔除易混的 I 与 O
LETTERS: str = "ABCDEFGHJKLMNPQRSTUVWXYZ"  # 26 - 2 = 24 个

#: 完整字符集，顺序与 CCPD 官方 ``ads`` 表一致（先字母后数字）
CHARSET: str = LETTERS + DIGITS  # 长度 34

#: 类别数
NUM_CLASSES: int = 34

#: 每个车牌需要识别的字符位置数（汉字之后的六位）
SEQ_LEN: int = 6

#: 字符 -> 类别索引
CHAR_TO_INDEX: Dict[str, int] = {ch: i for i, ch in enumerate(CHARSET)}

#: 类别索引 -> 字符
INDEX_TO_CHAR: Dict[int, str] = {i: ch for i, ch in enumerate(CHARSET)}

#: 字母类别的索引上界（含）。索引 0..23 为字母，24..33 为数字。
LETTER_MAX_INDEX: int = len(LETTERS) - 1  # 23

#: 数字类别的索引下界（含）
DIGIT_MIN_INDEX: int = len(LETTERS)  # 24

# 规格 §2.1 的自检断言
assert len(CHARSET) == NUM_CLASSES == 34, "字符集长度必须严格为 34"
assert len(set(CHARSET)) == 34, "字符集不得有重复字符"
assert "I" not in CHARSET and "O" not in CHARSET, "字符集必须剔除 I 与 O"
assert LETTER_MAX_INDEX == 23 and DIGIT_MIN_INDEX == 24

# =============================================================================
# 2. CCPD 官方映射表（原文照抄，已与 README 及 rpnet/demo.py 核对）
# =============================================================================

#: 省份表，34 项 + 哨兵 'O' = 35 项。本项目不识别汉字，仅用于长度校验。
CCPD_PROVINCES: List[str] = [
    "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁",
    "新", "警", "学", "O",
]

#: CCPD 字母表：24 个字母 + 哨兵 'O' = 25 项。索引 1..24 有效，索引 24 为哨兵。
CCPD_ALPHABETS: List[str] = [
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N",
    "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "O",
]

#: CCPD 字母/数字表：24 个字母 + 10 个数字 + 哨兵 'O' = 35 项。
#: 索引 0..33 恰与 :data:`CHARSET` 一一对应，索引 34 为哨兵。
CCPD_ADS: List[str] = [
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N",
    "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "O",
]

# --- 一致性断言：官方表的前 34 项必须与本项目 CHARSET 完全相同 -------------
assert CCPD_ADS[:NUM_CLASSES] == list(CHARSET), (
    "CCPD ads 表前 34 项必须与本项目 CHARSET 完全一致（顺序亦须相同）"
)
assert CCPD_ALPHABETS[: len(LETTERS)] == list(LETTERS)
# 哨兵必须都是字母 'O'，且不在 CHARSET 中
assert CCPD_ADS[-1] == CCPD_ALPHABETS[-1] == CCPD_PROVINCES[-1] == "O"
assert "O" not in CHARSET


# =============================================================================
# 3. 编码 / 解码
# =============================================================================


def char_to_index(ch: str) -> int:
    """单个字符 -> 类别索引。

    参数
    ----
    ch : str
        单个字符，必须属于 :data:`CHARSET`。

    返回
    ----
    int
        类别索引，范围 ``[0, 33]``。

    异常
    ----
    KeyError
        字符不在 34 类字符集内（例如 ``I``、``O`` 或汉字）。

    形状
    ----
    标量输入 -> 标量输出。
    """
    return CHAR_TO_INDEX[ch]


def index_to_char(idx: int) -> str:
    """类别索引 -> 单个字符。

    参数
    ----
    idx : int
        类别索引，范围 ``[0, 33]``。

    返回
    ----
    str
        对应字符。

    形状
    ----
    标量输入 -> 标量输出。
    """
    return INDEX_TO_CHAR[int(idx)]


def encode_label(text: str) -> np.ndarray:
    """六字符标签串 -> 类别索引向量。

    参数
    ----
    text : str
        长度必须为 6，每个字符属于 :data:`CHARSET`。

    返回
    ----
    numpy.ndarray
        形状 ``(6,)``，dtype ``int64``。

    异常
    ----
    ValueError
        长度不为 6，或含非法字符。
    """
    if len(text) != SEQ_LEN:
        raise ValueError(f"标签长度必须为 {SEQ_LEN}，实际为 {len(text)}：{text!r}")
    out = np.empty(SEQ_LEN, dtype=np.int64)
    for i, ch in enumerate(text):
        if ch not in CHAR_TO_INDEX:
            raise ValueError(f"标签第 {i} 位字符 {ch!r} 不在 34 类字符集内：{text!r}")
        out[i] = CHAR_TO_INDEX[ch]
    return out


def decode_label(indices: Sequence[int]) -> str:
    """类别索引向量 -> 六字符标签串。

    参数
    ----
    indices : Sequence[int]
        长度 6 的类别索引序列。

    返回
    ----
    str
        长度 6 的标签字符串。

    形状
    ----
    ``(6,)`` -> ``str``
    """
    if len(indices) != SEQ_LEN:
        raise ValueError(f"索引长度必须为 {SEQ_LEN}，实际为 {len(indices)}")
    return "".join(INDEX_TO_CHAR[int(i)] for i in indices)


def decode_batch(indices: np.ndarray) -> List[str]:
    """批量索引矩阵 -> 标签字符串列表。

    参数
    ----
    indices : numpy.ndarray
        形状 ``(N, 6)``，dtype 为整型。

    返回
    ----
    list of str
        长度 ``N`` 的标签列表。

    形状
    ----
    ``(N, 6)`` -> ``list[str]``，长度 ``N``
    """
    arr = np.asarray(indices)
    if arr.ndim != 2 or arr.shape[1] != SEQ_LEN:
        raise ValueError(f"输入形状必须为 (N, {SEQ_LEN})，实际为 {arr.shape}")
    return ["".join(INDEX_TO_CHAR[int(c)] for c in row) for row in arr]


# =============================================================================
# 4. 位置合法性与类别维度
# =============================================================================


def is_position_legal(ch: str, pos: int, positions: Sequence[int] | None = None) -> bool:
    """判断位置 ``pos`` 上的字符 ``ch`` 是否合法。

    参数
    ----
    ch : str
        待检查字符。
    pos : int
        位置编号，``0`` 表示汉字后的第一位（§2.1 的"第 1 位"）。
    positions : Sequence[int] or None
        各位置的类别数。若为 ``None`` 则使用全 34 类。
        当 ``positions[pos] == len(LETTERS)``（即 24）时，该位置仅允许字母。

    返回
    ----
    bool
        合法返回 ``True``。

    形状
    ----
    标量 -> 标量
    """
    if ch not in CHAR_TO_INDEX:
        return False
    if positions is None:
        return True
    idx = CHAR_TO_INDEX[ch]
    return idx < int(positions[pos])


def check_label_legal(
    text: str, positions: Sequence[int] | None = None
) -> Tuple[bool, List[int]]:
    """检查整条六位标签是否满足各位置约束。

    参数
    ----
    text : str
        长度 6 的标签串。
    positions : Sequence[int] or None
        各位置类别数，语义同 :func:`is_position_legal`。

    返回
    ----
    tuple
        ``(是否全部合法, 非法位置下标列表)``。

    形状
    ----
    ``str`` -> ``(bool, list[int])``
    """
    if len(text) != SEQ_LEN:
        return False, list(range(SEQ_LEN))
    bad = [
        i for i, ch in enumerate(text) if not is_position_legal(ch, i, positions)
    ]
    return (len(bad) == 0), bad


def resolve_positions(
    positions_cfg: Sequence[int] | None,
    position_letter_max_index: int = LETTER_MAX_INDEX,
) -> List[int]:
    """把配置中的 ``charset.positions`` 归一化为六个头的类别数列表。

    参数
    ----
    positions_cfg : Sequence[int] or None
        配置文件中的位置列表。``None`` 或空 -> 全 34 类。
        **特殊值 0 表示"该位置仅字母"**，会被展开为 ``LETTER_MAX_INDEX + 1``（= 24）；
        这是为了让配置文件可写 ``[0, 34, 34, 34, 34, 34]`` 表示"首位仅字母"。
    position_letter_max_index : int
        字母类别索引上界，默认 23。

    返回
    ----
    list of int
        长度 6 的各头输出维度。

    形状
    ----
    ``(6,)`` -> ``(6,)``
    """
    if not positions_cfg:
        return [NUM_CLASSES] * SEQ_LEN
    dims: List[int] = []
    for v in positions_cfg:
        v = int(v)
        dims.append(position_letter_max_index + 1 if v == 0 else v)
    if len(dims) != SEQ_LEN:
        raise ValueError(f"positions 长度必须为 {SEQ_LEN}，实际为 {len(dims)}")
    for d in dims:
        if d not in (len(LETTERS), NUM_CLASSES):
            raise ValueError(f"各位置类别数只允许 24 或 34，实际出现 {d}")
    return dims


def total_output_nodes(positions: Sequence[int] | None = None) -> int:
    """六个头的输出节点总数（基线 34 × 6 = 204）。

    参数
    ----
    positions : Sequence[int] or None
        各位置类别数；``None`` 表示全 34 类。

    返回
    ----
    int
        输出节点总数。

    形状
    ----
    ``(6,)`` -> 标量
    """
    dims = resolve_positions(positions)
    return int(sum(dims))


if __name__ == "__main__":  # pragma: no cover - 手工自检入口
    print(f"CHARSET      = {CHARSET}")
    print(f"NUM_CLASSES  = {NUM_CLASSES}")
    print(f"SEQ_LEN      = {SEQ_LEN}")
    print(f"输出节点总数 = {total_output_nodes()}  (基线应为 204)")
    print(f"E9 首个位置24类时 = {total_output_nodes([0, 34, 34, 34, 34, 34])} (应为 194)")
    demo = "A1B2C3"
    enc = encode_label(demo)
    print(f"编码 {demo} -> {enc.tolist()}")
    print(f"解码回       -> {decode_label(enc)}")
    assert decode_label(enc) == demo
    print("索引顺序自检：",
          ", ".join(f"{i}:{INDEX_TO_CHAR[i]}" for i in (0, 23, 24, 33)))
    print("全部自检通过。")