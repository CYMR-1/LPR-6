# -*- coding: utf-8 -*-
"""配置加载与应用模块。

对应规格 §0.2（要求 2：所有超参数集中在 ``configs/default.yaml``）与 §8.4
（实验与代码绑定：日志必须记录 ``commit`` / ``config`` / ``seed`` / ``dirty``）。

设计要点
--------
* :class:`Config` 把嵌套字典包装成"点分路径"访问的对象，
  例如 ``cfg.model.hidden_dim``、``cfg["optim.batch_size"]``、``cfg.get("loss.type")``。
* :func:`apply_patch` 用于对照实验：把 ``{"model.arch": "independent"}`` 这类
  点分路径补丁**深拷贝**到配置上，保证各变体互不污染。
* :func:`git_info` 调用 ``git rev-parse --short HEAD`` 与 ``git status --porcelain``，
  产出实验日志表头所需的四个溯源字段。

任何脚本都只通过本模块取超参数，不得在代码里硬编码超参数。
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from collections.abc import Mapping as _Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import yaml

#: 仓库根目录（本文件位于 <root>/models/config.py）
ROOT: Path = Path(__file__).resolve().parent.parent

#: 默认配置文件路径
DEFAULT_CONFIG_PATH: Path = ROOT / "configs" / "default.yaml"


# =============================================================================
# 1. 点分路径访问的配置对象
# =============================================================================


class Config(dict):
    """支持点分路径访问的配置对象（dict 的子类）。

    访问方式::

        cfg = load_config()
        cfg.model.hidden_dim          # 256
        cfg["optim.batch_size"]       # 64
        cfg.get("loss.l2_lambda", 0)  # 1e-4
        cfg.get_path("model.hidden_dim")   # 等价写法

    说明
    ----
    嵌套的 ``dict`` 在访问时会被自动包装成 :class:`Config`（惰性包装，不复制数据），
    因此 ``cfg.a.b.c`` 可以一直链下去。

    形状约定
    --------
    本类仅承载标量 / 列表 / 字典形式的超参数，不含张量。
    """

    # ---------------------------------------------------------------- 属性访问
    def __getitem__(self, key: str) -> Any:
        """取键；键名含 ``.`` 时按点分路径解析。

        这样 ``cfg["optim.batch_size"]`` 与 ``cfg["optim"]["batch_size"]`` 等价。
        """
        if isinstance(key, str) and "." in key:
            sentinel = object()
            value = self.get_path(key, sentinel)
            if value is sentinel:
                raise KeyError(key)
            return value
        return self._wrap(dict.__getitem__(self, key))

    def __getattr__(self, name: str) -> Any:
        """按属性名取键；不存在时抛 ``AttributeError``（保持 Python 语义）。"""
        try:
            value = dict.__getitem__(self, name)
        except KeyError as exc:
            raise AttributeError(f"配置中不存在键 {name!r}") from exc
        return self._wrap(value)

    def __setattr__(self, name: str, value: Any) -> None:
        """按属性名写键。"""
        self[name] = value

    # ---------------------------------------------------------------- 包装
    @staticmethod
    def _wrap(value: Any) -> Any:
        """把嵌套 dict / Config 统一包装为 Config，list 内的元素递归包装。

        注意：只能用 ``Mapping`` 判定，因为 :class:`Config` 本身是 dict 子类，
        直接 ``isinstance(x, dict)`` 会漏判嵌套的 Config。
        """
        if isinstance(value, Config):
            return value
        if isinstance(value, _Mapping):
            return Config(value)
        if isinstance(value, list):
            return [Config._wrap(v) for v in value]
        return value

    # ---------------------------------------------------------------- 点分路径
    def get_path(self, dotted: str, default: Any = None) -> Any:
        """按点分路径取值，缺失时返回 ``default``。

        参数
        ----
        dotted : str
            形如 ``"optim.batch_size"`` 的路径。
        default : Any
            缺失时的返回值。

        返回
        ----
        Any
            取到的值（嵌套 dict 会被包装为 Config）。
        """
        node: Any = self
        for part in dotted.split("."):
            if isinstance(node, _Mapping):
                if part not in node:
                    return default
                node = node[part]
            else:
                return default
        return self._wrap(node)

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        """兼容两种写法：``cfg.get("a.b.c")`` 与 ``cfg.get("a")``。"""
        if isinstance(key, str) and "." in key:
            return self.get_path(key, default)
        if key in self:
            return self._wrap(dict.__getitem__(self, key))
        return default

    def __contains__(self, key: object) -> bool:
        """支持 ``"optim.batch_size" in cfg`` 的点分路径判定。"""
        if isinstance(key, str) and "." in key:
            sentinel = object()
            return self.get_path(key, sentinel) is not sentinel
        return dict.__contains__(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        """写键；键名含 ``.`` 时按点分路径写入。

        实现注意：``__getitem__`` 返回的是**包装后的副本**，直接
        ``cfg["a"]["b"] = v`` 会把值写到临时对象上而丢失。因此
        :meth:`set_path` 内部一律用 ``dict.__getitem__`` 取原始子字典。
        """
        if isinstance(key, str) and "." in key:
            self.set_path(key, value)
            return
        dict.__setitem__(self, key, _to_plain(value))

    def set_path(self, dotted: str, value: Any) -> None:
        """按点分路径写值（中间层不存在时自动创建 dict）。

        参数
        ----
        dotted : str
            形如 ``"model.hidden_dim"`` 的路径。
        value : Any
            新值。

        返回
        ----
        None
        """
        parts = dotted.split(".")
        node: Any = self
        for part in parts[:-1]:
            child = dict.get(node, part) if isinstance(node, dict) else None
            if not isinstance(child, dict):
                child = {}
                dict.__setitem__(node, part, child)
            node = child
        dict.__setitem__(node, parts[-1], _to_plain(value))

    def to_dict(self) -> Dict[str, Any]:
        """递归转回纯 ``dict``（供 JSON 序列化 / 日志记录）。"""
        return _to_plain(self)

    def flat(self, prefix: str = "") -> Dict[str, Any]:
        """展平为 ``{"a.b.c": value}`` 形式，便于写入 CSV 表头。

        参数
        ----
        prefix : str
            递归时的前缀，外部调用保持默认空串。

        返回
        ----
        dict
            展平后的键值对。
        """
        out: Dict[str, Any] = {}
        for key, value in self.items():
            full = f"{prefix}{key}"
            if isinstance(value, _Mapping):
                out.update(Config(value).flat(f"{full}."))
            else:
                out[full] = value
        return out

    def diff(self, other: Mapping[str, Any]) -> Dict[str, Any]:
        """返回本配置与 ``other`` 的展平差异，用于记录"本实验改了什么"。

        参数
        ----
        other : Mapping
            参照配置（通常是基线）。

        返回
        ----
        dict
            ``{路径: 本配置的值}``，仅含与参照不同的项。
        """
        mine, theirs = self.flat(), Config(dict(other)).flat()
        return {
            k: v
            for k, v in mine.items()
            if k not in theirs or json.dumps(theirs[k], sort_keys=True, default=str)
            != json.dumps(v, sort_keys=True, default=str)
        }

    def fingerprint(self) -> str:
        """配置内容的稳定指纹（sha1 前 12 位），用于识别"同一组超参数"。"""
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str, ensure_ascii=False)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def _to_plain(value: Any) -> Any:
    """把 Config / dict / list 递归转换为纯 Python 对象。"""
    if isinstance(value, _Mapping):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_plain(v) for v in value]
    return value


# =============================================================================
# 2. 加载与补丁
# =============================================================================


def load_config(path: str | os.PathLike | None = None, **overrides: Any) -> Config:
    """读取 YAML 配置。

    参数
    ----
    path : str or Path or None
        配置文件路径；``None`` 表示使用 ``configs/default.yaml``。
    **overrides : Any
        额外的点分路径覆盖，例如 ``load_config(hidden_dim=128)`` 不推荐，
        应使用 :func:`apply_patch`。此处仅支持 ``key_with_dot=value`` 形式的直接覆盖，
        键名中的 ``____`` 会被还原为 ``.``（因为 Python 形参不能含点）。

    返回
    ----
    Config
        可点分访问的配置对象。

    形状
    ----
    标量配置树，无张量。
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"配置文件不存在：{cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as fp:
        raw = yaml.safe_load(fp)
    if not isinstance(raw, dict):
        raise ValueError(f"配置文件顶层必须是映射：{cfg_path}")
    cfg = Config(raw)
    for key, value in overrides.items():
        cfg.set_path(key.replace("____", "."), value)
    return cfg


def apply_patch(cfg: Config, patch: Mapping[str, Any] | None) -> Config:
    """把点分路径补丁应用到配置的**深拷贝**上，返回新配置。

    参数
    ----
    cfg : Config
        基线配置。
    patch : Mapping or None
        ``{"model.arch": "independent", "optim.batch_size": 32}``。
        ``None`` 或空映射表示不改动。

    返回
    ----
    Config
        打了补丁的新配置（原 ``cfg`` 不被修改）。

    形状
    ----
    标量配置树，无张量。
    """
    new_cfg = Config(copy.deepcopy(cfg.to_dict()))
    if patch:
        for dotted, value in patch.items():
            new_cfg.set_path(dotted, value)
    return new_cfg


def resolve_path(cfg: Config, dotted: str) -> Path:
    """把配置中的相对路径解析为仓库内的绝对路径。

    参数
    ----
    cfg : Config
        配置对象。
    dotted : str
        ``paths`` 下的键名，例如 ``"processed_dir"``。

    返回
    ----
    pathlib.Path
        绝对路径（父目录会被创建）。

    形状
    ----
    标量 -> 标量
    """
    rel = cfg.get(f"paths.{dotted}")
    if rel is None:
        raise KeyError(f"paths.{dotted} 未在配置中定义")
    p = Path(rel)
    if not p.is_absolute():
        p = ROOT / p
    return p


def ensure_dirs(cfg: Config) -> None:
    """创建配置中声明为输出目录的路径（幂等）。

    参数
    ----
    cfg : Config
        配置对象。

    返回
    ----
    None
    """
    for key in (
        "ccpd_root", "synth_root", "processed_dir",
        "models_dir", "logs_dir", "figs_dir", "tables_dir",
    ):
        resolve_path(cfg, key).mkdir(parents=True, exist_ok=True)
    resolve_path(cfg, "manifest").parent.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 3. Git 溯源（§8.4）
# =============================================================================


@dataclass
class GitInfo:
    """实验溯源所需的 Git 状态。

    属性
    ----
    commit : str
        短 7 位 commit hash；仓库无提交时为 ``"unknown"``。
    dirty : bool
        工作区是否有未提交改动。**``dirty=True`` 的结果不可作为最终结论。**
    branch : str
        当前分支名。
    describe : str
        ``git describe --tags --always`` 结果，失败时与 commit 相同。
    """

    commit: str = "unknown"
    dirty: bool = True
    branch: str = ""
    describe: str = ""

    def as_dict(self) -> Dict[str, Any]:
        """转成可写入日志表头的字典。"""
        return {
            "commit": self.commit,
            "dirty": bool(self.dirty),
            "branch": self.branch,
            "git_describe": self.describe,
        }


def _run_git(args: Iterable[str], cwd: Path | None = None) -> str:
    """执行 git 命令并返回 stdout（失败返回空串，绝不抛异常）。

    参数
    ----
    args : Iterable[str]
        git 子命令与参数，例如 ``["rev-parse", "--short", "HEAD"]``。
    cwd : Path or None
        执行目录，默认仓库根。

    返回
    ----
    str
        stdout 去空白后的内容。
    """
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd or ROOT),
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        return proc.stdout.strip()
    except Exception:
        return ""


def git_info(cwd: Path | None = None) -> GitInfo:
    """采集当前仓库的 ``commit`` / ``dirty`` / ``branch``。

    参数
    ----
    cwd : Path or None
        仓库路径，默认仓库根。

    返回
    ----
    GitInfo
        采集结果；任何 git 调用失败都不会抛异常，而是回退为
        ``commit="unknown"``、``dirty=True``（保守判定，避免把脏工作区当干净）。

    形状
    ----
    标量 -> dataclass
    """
    commit = _run_git(["rev-parse", "--short", "HEAD"], cwd) or "unknown"
    porcelain = _run_git(["status", "--porcelain"], cwd)
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd)
    describe = _run_git(["describe", "--tags", "--always"], cwd) or commit
    return GitInfo(
        commit=commit,
        dirty=bool(porcelain.strip()),
        branch=branch or "unknown",
        describe=describe,
    )


# =============================================================================
# 4. 随机种子与环境快照
# =============================================================================


def set_seed(seed: int) -> None:
    """固定所有随机源（§3.3：必须覆盖初始化、批顺序与数据采样）。

    参数
    ----
    seed : int
        随机种子。

    返回
    ----
    None

    说明
    ----
    同时固定 ``random``、``numpy`` 与 ``PYTHONHASHSEED`` 相关的行为。
    项目不使用任何自动求导框架，故无需设置框架级种子。
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    os.environ["PYTHONHASHSEED"] = str(seed)


def rng_for(seed: int, stream: str) -> np.random.Generator:
    """由 ``seed`` 与用途名派生一个独立的随机数发生器。

    参数
    ----
    seed : int
        基础种子。
    stream : str
        用途名（如 ``"augment"``、``"init"``、``"shuffle"``），
        用于把不同用途的随机流解耦，避免互相扰动。

    返回
    ----
    numpy.random.Generator
        ``default_rng`` 实例。

    形状
    ----
    标量 -> Generator
    """
    h = hashlib.sha1(f"{int(seed)}::{stream}".encode("utf-8")).digest()
    derived = int.from_bytes(h[:8], "little", signed=False)
    return np.random.default_rng(derived)


def env_snapshot() -> Dict[str, Any]:
    """采集运行环境快照，写入实验日志表头（可复现性证据）。

    返回
    ----
    dict
        含 python 版本、numpy 版本、平台、GPU 后端可用性等字段。

    形状
    ----
    无输入 -> dict
    """
    snap: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "executable": sys.executable,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    try:  # CuPy 为可选依赖
        import cupy  # type: ignore

        snap["cupy"] = getattr(cupy, "__version__", "unknown")
        try:
            snap["cuda_device"] = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
            snap["cuda_free_mem_mb"] = round(
                cupy.cuda.runtime.memGetInfo()[0] / 1024 ** 2, 1
            )
        except Exception:
            snap["cuda_device"] = "unavailable"
    except Exception:
        snap["cupy"] = None
    return snap


@dataclass
class RunMeta:
    """一次实验运行的完整溯源元信息（§8.4 的四个必备字段 + 环境快照）。"""

    commit: str = "unknown"
    dirty: bool = True
    config_name: str = ""
    config_fingerprint: str = ""
    seed: int = 42
    env: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def build(cls, cfg: Config, seed: int, **extra: Any) -> "RunMeta":
        """从配置与种子构造元信息。

        参数
        ----
        cfg : Config
            本次实验生效的配置（已含变体补丁）。
        seed : int
            随机种子。
        **extra : Any
            额外字段，例如 ``experiment="E1"``、``variant="shared"``。

        返回
        ----
        RunMeta
        """
        gi = git_info()
        return cls(
            commit=gi.commit,
            dirty=gi.dirty,
            config_name=str(cfg.get("project.name", "ProjectX")),
            config_fingerprint=cfg.fingerprint(),
            seed=int(seed),
            env=env_snapshot(),
            extra=dict(extra),
        )

    def as_dict(self) -> Dict[str, Any]:
        """转成可写入 CSV / JSON 的扁平字典。"""
        out: Dict[str, Any] = {
            "commit": self.commit,
            "dirty": self.dirty,
            "config": self.config_name,
            "config_fingerprint": self.config_fingerprint,
            "seed": self.seed,
        }
        out.update({k: v for k, v in self.extra.items()})
        out.update({f"env_{k}": v for k, v in self.env.items()})
        return out


if __name__ == "__main__":  # pragma: no cover - 手工自检入口
    cfg = load_config()
    print("hidden_dim      =", cfg.model.hidden_dim)
    print("batch_size      =", cfg["optim.batch_size"])
    print("l2_lambda       =", cfg.get("loss.l2_lambda"))
    print("positions       =", cfg.charset.positions)
    print("fingerprint     =", cfg.fingerprint())
    patched = apply_patch(cfg, {"model.arch": "independent", "optim.batch_size": 32})
    print("patch diff      =", patched.diff(cfg))
    print("baseline intact =", cfg.model.arch, cfg["optim.batch_size"])
    print("git info        =", git_info().as_dict())
    print("config ok.")