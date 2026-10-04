# -*- coding: utf-8 -*-
"""计算后端：NumPy（CPU）与 CuPy（GPU）的统一入口（§8.1 / §附录 B 第 12 条）。

定位
----
规格要求"训练使用 GPU，后端为 CuPy（numpy 的 GPU 兼容替代）；无 GPU 时回退
numpy 纯 CPU"。本模块把这一要求收敛成一处：

* :func:`get_backend` 按名字或自动探测返回一个"numpy 兼容模块"；
* **反向传播仍然是手写的**（见 :mod:`models.model`），CuPy 只承担矩阵运算，
  不使用任何自动求导，因此不违反课程约束；
* 无 GPU、未装 CuPy、CUDA 不可用等任何异常都会**回退到 numpy 并打印告警**，
  保证工程在无 GPU 环境下仍可完整复现。

约定
----
所有 ``models`` 包内的数值代码都通过 :func:`get_backend` 取 ``xp``，
然后写 ``xp.dot / xp.exp / ...``，不在模块顶层 ``import numpy as np`` 后直接用
它做张量运算（否则会绕过 GPU 后端）。
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

#: 允许的后端名
BACKEND_CHOICES: Tuple[str, ...] = ("numpy", "cupy")


@dataclass
class BackendInfo:
    """后端信息，写入实验日志（证明"是否真的用了 GPU"）。

    属性
    ----
    name : str
        实际生效的后端名（``numpy`` 或 ``cupy``）。
    requested : str
        配置中请求的后端名。
    is_gpu : bool
        是否运行在 GPU 上。
    device_name : str
        GPU 型号；CPU 后端时为 ``"CPU"``。
    fallback_reason : str
        发生回退时的原因；未回退为空串。
    module : Any
        numpy 或 cupy 模块本身。
    """

    name: str
    requested: str
    is_gpu: bool
    device_name: str = "CPU"
    fallback_reason: str = ""
    module: Any = None

    def as_dict(self) -> dict:
        """转成可写入日志的字典（不含模块对象）。"""
        return {
            "backend": self.name,
            "backend_requested": self.requested,
            "is_gpu": self.is_gpu,
            "device_name": self.device_name,
            "backend_fallback_reason": self.fallback_reason,
        }


def _prepare_cuda_dll_paths() -> None:
    """把 pip 安装的 CUDA 运行库目录（``nvidia-*-cu12``）加入 DLL 搜索路径。

    动机
    ----
    本工程只通过 pip 装了 ``nvidia-cublas-cu12 / nvidia-cuda-runtime-cu12 /
    nvidia-cuda-nvrtc-cu12``，**没有安装完整的 CUDA Toolkit**。这些 wheel 把 DLL
    放在 ``site-packages/nvidia/<组件>/bin``，而 ``cuda_nvrtc/bin`` 里的
    ``nvrtc64_*.dll`` 还依赖同目录的 ``nvrtc-builtins64_*.dll``。默认情况下这些
    目录不在搜索路径上，于是会报::

        RuntimeError: CuPy failed to load nvrtc64_120_0.dll
        ... failed to open nvrtc-builtins64_120.dll

    这里在 **导入 cupy 之前** 把它们补进 ``PATH`` 并调用 ``os.add_dll_directory``，
    使工程无需手动设置 ``CUDA_PATH`` 即可使用 GPU。

    若机器上装有正式 CUDA Toolkit（``CUDA_PATH`` 已设置），则不做任何改动。

    返回
    ----
    None

    形状
    ----
    无
    """
    if not sys.platform.startswith("win"):
        return
    if os.environ.get("CUDA_PATH"):
        return                            # 已有正式 Toolkit，尊重用户环境

    # 注意：pip 装的 nvidia 是 **命名空间包**，没有 __file__，
    # 必须用 find_spec 的 submodule_search_locations 定位。
    try:
        import importlib.util

        spec = importlib.util.find_spec("nvidia")
        locs = list(getattr(spec, "submodule_search_locations", None) or [])
    except Exception:
        return
    if not locs:
        return
    nvidia_root = locs[0]

    bins = []
    for comp in ("cuda_nvrtc", "cublas", "cuda_runtime"):
        d = os.path.join(nvidia_root, comp, "bin")
        if os.path.isdir(d):
            bins.append(d)
    if not bins:
        return

    os.environ["PATH"] = os.pathsep.join(bins + [os.environ.get("PATH", "")])
    # CuPy 在 Windows 上还会用 CUDA_PATH 定位 <root>/bin；未设置时会打印
    # "CUDA path could not be detected" 告警。这里指向 nvidia 命名空间根，
    # 其 bin 子目录正好由各组件 wheel 提供。
    for comp in ("cuda_nvrtc", "cublas", "cuda_runtime"):
        d = os.path.join(nvidia_root, comp, "bin")
        if os.path.isdir(d):
            try:
                os.add_dll_directory(d)
            except Exception:
                pass


def _try_cupy() -> Tuple[Optional[Any], str, str]:
    """尝试导入并初始化 CuPy。

    返回
    ----
    tuple
        ``(cupy 模块或 None, 设备名, 失败原因)``。

    说明
    ----
    初始化分三步，任一步失败都返回回退原因：

    1. 导入 ``cupy``；
    2. 若工程路径含非 ASCII 字符（例如本机的 ``…/深度学习导论/projectX``），
       施加 :func:`_apply_windows_nvrtc_pathfix` —— 这是本机实测必须的一步，
       详见该函数 docstring；
    3. 真正按 **逐元素核 JIT** 跑一次运算（``xp.exp``），确认 NVRTC 可用；
       只做 ``cupy.zeros(1)`` 不够 —— 它走的是 cudaMalloc 而非 NVRTC，会在
       "显存能分配但核编译失败"时给出假阳性。
    """
    _prepare_cuda_dll_paths()
    try:
        import warnings

        with warnings.catch_warnings():
            # 只忽略 CuPy 自己的 "CUDA path could not be detected" 提示：
            # 本工程用 pip 版 CUDA 运行库，没有完整 Toolkit，该提示无实际影响
            # （DLL 搜索路径已在上一步补好）。
            warnings.filterwarnings("ignore", message=".*CUDA path could not.*")
            import cupy  # type: ignore
    except Exception as exc:
        return None, "CPU", f"未安装 CuPy（{type(exc).__name__}）"

    try:
        _apply_windows_nvrtc_pathfix(cupy)
    except Exception as exc:
        return None, "CPU", f"CuPy 路径兼容处理失败（{type(exc).__name__}: {exc}）"

    try:
        # 真正触碰一次设备，确认 CUDA 可用且驱动匹配
        cupy.zeros(1)
        name = cupy.cuda.runtime.getDeviceProperties(0)["name"]
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
    except Exception as exc:
        return None, "CPU", f"CuPy 已安装但 CUDA 设备不可用（{type(exc).__name__}: {exc}）"

    # 第二步：确认 JIT 逐元素核真的能编译（NVRTC）。
    # 这一步是本机踩坑后加的：只测 cupy.zeros(1) 会误判为"GPU 可用"，
    # 而实际训练里第一个 xp.exp 就会抛 CompileException。
    try:
        probe = cupy.asarray([-1.0, 0.0, 1.0], dtype=cupy.float32)
        _ = cupy.asnumpy(cupy.exp(probe) + cupy.maximum(probe, 0.0))
    except Exception as exc:
        return None, "CPU", (
            f"CuPy 设备可分配显存但 NVRTC 逐元素核编译失败"
            f"（{type(exc).__name__}: {str(exc).splitlines()[0][:120]}）"
        )

    return cupy, str(name), ""


def _apply_windows_nvrtc_pathfix(cupy: Any) -> str:
    """修正 Windows + 非 ASCII 工程路径下 NVRTC 找不到 CuPy 头文件的问题。

    背景（本机实测结论，非推测）
    ---------------------------
    CuPy 的 JIT 会把 ``-I<cupy包路径>/_core/include`` 等参数传给 NVRTC。Windows
    版 NVRTC **无法处理含非 ASCII 字符的 ``-I`` 目录 —— 它既不报错也不使用，
    而是静默丢弃该搜索路径**，于是编译 ``#include <cupy/complex.cuh>`` 时报
    "catastrophic error: cannot open source file"。

    本工程的路径含中文（``…/深度学习导论/projectX``），正好命中该缺陷。
    实测对照（同一台机器、同一份 CuPy 14.2.0）：

    * ``-I`` 指向中文路径          -> ``cannot open source file "cupy/complex.cuh"``
    * 同样头文件复制到纯 ASCII 路径 -> 编译成功

    解决方式
    --------
    把 CuPy 自带的 ``_core/include``（以及 ``nvidia-cuda-runtime`` 的 ``include``）
    复制到临时目录下的纯 ASCII 路径，并在 NVRTC 边界把含非 ASCII 字符的
    ``-I`` 重写成对应的 ASCII 副本。重写点是 ``_NVRTCProgram.compile`` ——
    它是最靠近 NVRTC 的必经过滤口；``assemble_cupy_compiler_options`` 看着更"正统"，
    但实测逐元素核根本不走它（调用计数为 0），所以补丁无效。

    参数
    ----
    cupy : Any
        已导入的 cupy 模块。

    返回
    ----
    str
        实际使用的 ASCII 头文件根目录；无需修正时返回空串。

    形状
    ----
    标量 -> str
    """
    if not sys.platform.startswith("win"):
        return ""
    cupy_root = os.path.dirname(os.path.abspath(cupy.__file__))
    if cupy_root.isascii():
        return ""                      # 路径本来就安全，不动它

    # 目标：临时目录下的纯 ASCII 头文件根
    ascii_root = os.path.join(tempfile.gettempdir(), "projectx_cupy_inc")
    real_inc = os.path.join(cupy_root, "_core", "include")
    stamp = os.path.join(ascii_root, ".ok")
    if not os.path.isfile(stamp):
        os.makedirs(ascii_root, exist_ok=True)
        shutil.copytree(real_inc, ascii_root, dirs_exist_ok=True)
        rt_inc = os.path.join(os.path.dirname(cupy_root), "nvidia",
                              "cuda_runtime", "include")
        if os.path.isdir(rt_inc):
            shutil.copytree(rt_inc,
                            os.path.join(ascii_root, "cuda_runtime", "include"),
                            dirs_exist_ok=True)
        with open(stamp, "w", encoding="utf-8") as fp:
            fp.write("projectx nvrtc ascii include root\n")

    marker = "_core/include"

    def _rewrite(path: str) -> str:
        """把单个 -I 目标路径映射成 ASCII 副本路径。"""
        p = path.replace("\\", "/")
        if "cuda_runtime" in p:
            return os.path.join(ascii_root, "cuda_runtime", "include").replace(
                os.sep, "/")
        idx = p.find(marker)
        rel = p[idx + len(marker):].strip("/") if idx >= 0 else ""
        if not rel:
            return ascii_root.replace(os.sep, "/")
        return os.path.join(ascii_root, rel).replace(os.sep, "/")

    from cupy.cuda import compiler as _compiler  # type: ignore

    if getattr(_compiler, "_projectx_pathfix", False):
        return ascii_root                # 已打过补丁，避免重复包装

    orig_compile = _compiler._NVRTCProgram.compile

    def patched_compile(self, options=(), log_stream=None):  # noqa: ANN001
        new_opts = tuple(
            ("-I" + _rewrite(o[2:]))
            if isinstance(o, str) and o.startswith("-I") and not o[2:].isascii()
            else o
            for o in options
        )
        return orig_compile(self, new_opts, log_stream)

    _compiler._NVRTCProgram.compile = patched_compile
    _compiler._projectx_pathfix = True
    return ascii_root


def get_backend(
    name: str = "numpy",
    verbose: bool = True,
) -> BackendInfo:
    """获取计算后端。

    参数
    ----
    name : str
        请求的后端名，``"numpy"`` 或 ``"cupy"``；``"auto"`` 表示优先 CuPy。
    verbose : bool
        回退时是否打印告警。

    返回
    ----
    BackendInfo
        实际生效的后端信息；``module`` 字段是可直接当 numpy 用的模块。

    形状
    ----
    标量 -> dataclass
    """
    name = str(name).lower()
    if name not in BACKEND_CHOICES and name != "auto":
        raise ValueError(f"未知后端 {name!r}，可选 {BACKEND_CHOICES} 或 'auto'")

    if name == "numpy":
        return BackendInfo(name="numpy", requested=name, is_gpu=False,
                           device_name="CPU", module=np)

    cupy_mod, dev, reason = _try_cupy()
    if cupy_mod is None:
        if verbose:
            print(f"[backend] 回退到 numpy 纯 CPU 运行：{reason}")
        return BackendInfo(name="numpy", requested=name, is_gpu=False,
                           device_name="CPU", fallback_reason=reason, module=np)

    if verbose:
        print(f"[backend] 使用 CuPy GPU 后端：{dev}")
    return BackendInfo(name="cupy", requested=name, is_gpu=True,
                       device_name=dev, module=cupy_mod)


def asnumpy(x: Any) -> np.ndarray:
    """把后端数组转换回 numpy 数组（CPU 上为无拷贝视图）。

    参数
    ----
    x : Any
        可能是 numpy 或 cupy 的数组。

    返回
    ----
    numpy.ndarray
        主机端数组。

    形状
    ----
    与输入同形状。
    """
    if x is None:
        return None  # type: ignore[return-value]
    mod = type(x).__module__
    if mod.startswith("cupy"):
        import cupy  # type: ignore

        return cupy.asnumpy(x)
    return np.asarray(x)


def to_device(x: np.ndarray, backend: BackendInfo) -> Any:
    """把主机端数组搬到后端设备上。

    参数
    ----
    x : numpy.ndarray
        主机数组。
    backend : BackendInfo
        目标后端。

    返回
    ----
    Any
        后端数组（CPU 后端时即原数组）。

    形状
    ----
    同形状。
    """
    if backend.is_gpu:
        import cupy  # type: ignore

        return cupy.asarray(x)
    return np.asarray(x)


def device_synchronize(backend: BackendInfo) -> None:
    """等待设备上的计算完成（GPU 计时必须调用，否则时间不准）。

    参数
    ----
    backend : BackendInfo
        后端信息。

    返回
    ----
    None
    """
    if backend.is_gpu:
        try:
            import cupy  # type: ignore

            cupy.cuda.Stream.null.synchronize()
        except Exception:
            pass


def memory_info(backend: BackendInfo) -> dict:
    """查询设备显存/内存信息。

    参数
    ----
    backend : BackendInfo
        后端信息。

    返回
    ----
    dict
        ``{"used_mb": ..., "total_mb": ...}``；CPU 后端返回空字典。
    """
    if backend.is_gpu:
        try:
            import cupy  # type: ignore

            free_b, total_b = cupy.cuda.runtime.memGetInfo()
            return {
                "used_mb": round((total_b - free_b) / 1024 ** 2, 1),
                "total_mb": round(total_b / 1024 ** 2, 1),
            }
        except Exception:
            return {}
    return {}


def set_backend_env(backend: BackendInfo) -> None:
    """把后端名写入环境变量，便于子进程/日志统一。

    参数
    ----
    backend : BackendInfo
        后端信息。

    返回
    ----
    None
    """
    os.environ["PROJECTX_BACKEND"] = backend.name


if __name__ == "__main__":  # pragma: no cover
    import json

    for req in ("numpy", "cupy", "auto"):
        info = get_backend(req)
        print(f"请求 {req:6s} -> 生效 {info.name:6s} is_gpu={info.is_gpu} "
              f"device={info.device_name} 回退原因={info.fallback_reason or '无'}")
        x = to_device(np.arange(6, dtype=np.float32).reshape(2, 3), info)
        y = info.module.dot(x, x.T)
        print("   dot 结果:", asnumpy(y).tolist(), " 显存:", memory_info(info))
    print("后端自检通过。")