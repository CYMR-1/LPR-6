# -*- coding: utf-8 -*-
"""可执行的包引导模块。

用途
----
本项目的脚本既支持 ``python -m train.train`` 运行，也支持
``python train/train.py`` 直接运行（README 中采用后者，更直观）。
直接运行时 ``sys.path[0]`` 是脚本所在目录，``import models`` 会失败。

因此每个脚本的**第一行**（文档字符串之后）都写::

    from _bootstrap import setup
    setup()

它会把仓库根目录插入 ``sys.path``，使两种运行方式等价。

说明：这里刻意使用 ``_bootstrap`` 而不是 ``sitecustomize``，
因为 ``sitecustomize`` 是 Python 启动时的全局钩子名，容易与其它环境冲突。
"""

import sys
from pathlib import Path

#: 仓库根目录（本文件位于仓库根）
ROOT: Path = Path(__file__).resolve().parent


def setup() -> Path:
    """把仓库根目录插入 ``sys.path``（幂等）。

    返回
    ----
    pathlib.Path
        仓库根目录。
    """
    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    # 同时保证脚本自身所在目录可被导入（train.xxx / evaluate.xxx）
    for sub in ("train", "evaluate", "models", "reports"):
        p = ROOT / sub
        if p.is_dir() and str(p) not in sys.path:
            sys.path.append(str(p))
    return ROOT