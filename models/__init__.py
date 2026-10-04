# -*- coding: utf-8 -*-
"""ProjectX 的核心包：配置、字符集、数据、模型、损失与优化器。

子模块
------
charset
    34 类字符集与 CCPD 索引映射表。
config
    配置加载、变体补丁、Git 溯源。
ccpd_parse
    CCPD 文件名解析与透视矫正裁剪。
dataset
    数据集、全局标准化与批加载器。
model
    共享模型与独立模型（手写前向 / 反向）。
losses
    交叉熵 / MSE / L2 及其梯度。
optim
    SGD、动量与批量策略。
"""

from pathlib import Path

#: 仓库根目录，供各子模块直接运行时定位包
ROOT: Path = Path(__file__).resolve().parent.parent

__all__ = ["ROOT"]