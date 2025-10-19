# ManimGL 通用数学与工具函数集：提供一系列基础数学运算、函数分析工具和数据处理函数，
# 支持标量与数组操作，为动画逻辑、图形生成和数值计算提供核心支持。


from __future__ import annotations

from functools import lru_cache  # 缓存装饰器，优化重复计算
import hashlib  # 哈希算法库，用于字符串唯一标识
import inspect  # 函数反射工具，获取参数信息
import math

import numpy as np  # 数值计算库，支持数组操作

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Callable, TypeVar, Iterable
    from manimlib.typing import FloatArray  # 浮点数组类型注解

    # 泛型类型：支持 float 或 FloatArray 的可缩放类型
    Scalable = TypeVar("Scalable", float, FloatArray)


# ------------------------------ 基础数学函数 ------------------------------
def sigmoid(x: float | FloatArray):
    """
    Sigmoid 激活函数（S 形曲线）：将输入值映射到 (0, 1) 区间，常用于平滑过渡。
    
    数学公式：f(x) = 1 / (1 + e^(-x))
    
    参数：x - 输入值（标量或 numpy 数组）
    返回：映射后的结果（0 附近变化陡峭，两端趋于平缓）
    """
    return 1.0 / (1 + np.exp(-x))


@lru_cache(maxsize=10)
def choose(n: int, k: int) -> int:
    """
    组合数计算（n 选 k）：计算从 n 个元素中无序选择 k 个的方案数，带缓存优化。
    
    缓存机制：最多缓存 10 个结果，适合频繁调用的场景（如排列组合动画）。
    
    参数：
        n - 总元素数量（非负整数）
        k - 选择的元素数量（0 ≤ k ≤ n）
    返回：组合数结果（整数）
    """
    return math.comb(n, k)  # 调用 Python 内置组合数函数


def gen_choose(n: int, r: int) -> int:
    """
    广义组合数计算：适用于 n 为任意整数，计算连乘形式的组合数。
    
    计算逻辑：n × (n-1) × ... × (n-r+1) / r!
    
    参数：
        n - 起始整数
        r - 连乘项数（正整数）
    返回：广义组合数结果（整数）
    """
    return int(np.prod(range(n, n - r, -1)) / math.factorial(r))


# ------------------------------ 函数参数检查 ------------------------------
def get_num_args(function: Callable) -> int:
    """
    获取函数的参数数量（不包含可变参数 *args 和 **kwargs）。
    
    参数：function - 待检查的函数对象
    返回：参数数量（整数）
    """
    return function.__code__.co_argcount  # 从函数代码对象中提取参数计数


def get_parameters(function: Callable) -> Iterable[str]:
    """
    获取函数的所有参数名称（包含位置参数和关键字参数）。
    
    参数：function - 待检查的函数对象
    返回：参数名称的可迭代序列
    """
    return inspect.signature(function).parameters.keys()  # 通过函数签名获取参数名


# ------------------------------ 数值范围裁剪 ------------------------------
def clip(a: float, min_a: float, max_a: float) -> float:
    """
    将标量值裁剪到指定范围 [min_a, max_a]。
    
    参数：
        a - 输入标量
        min_a - 最小值下界
        max_a - 最大值上界
    返回：裁剪后的值（若 a 超出范围则返回边界值，否则返回原值）
    """
    if a < min_a:
        return min_a
    elif a > max_a:
        return max_a
    return a


def arr_clip(arr: np.ndarray, min_a: float, max_a: float) -> np.ndarray:
    """
    将数组元素裁剪到指定范围 [min_a, max_a]（原地修改数组）。
    
    参数：
        arr - 输入 numpy 数组
        min_a - 最小值下界
        max_a - 最大值上界
    返回：裁剪后的数组（与输入同形状）
    """
    arr[arr < min_a] = min_a  # 小于下界的元素设为 min_a
    arr[arr > max_a] = max_a  # 大于上界的元素设为 max_a
    return arr


# ------------------------------ 安全除法运算 ------------------------------
def fdiv(a: Scalable, b: Scalable, zero_over_zero_value: Scalable | None = None) -> Scalable:
    """
    安全除法：基于 numpy.true_divide，支持自定义 0/0 的结果（避免 NaN）。
    
    参数：
        a - 被除数（标量或数组）
        b - 除数（标量或数组）
        zero_over_zero_value - 0/0 时的返回值（默认 None 保留 NaN）
    返回：除法结果 a/b
    """
    if zero_over_zero_value is not None:
        # 初始化输出数组为指定默认值
        out = np.full_like(a, zero_over_zero_value)
        # 仅在 a 或 b 非零时执行除法（避免 0/0）
        where = np.logical_or(a != 0, b != 0)
    else:
        out = None
        where = True  # 所有位置都执行除法

    return np.true_divide(a, b, out=out, where=where)


# ------------------------------ 二分查找算法 ------------------------------
def binary_search(
    function: Callable[[float], float],
    target: float,
    lower_bound: float,
    upper_bound: float,
    tolerance: float = 1e-4
) -> float | None:
    """
    二分查找：在区间 [lower_bound, upper_bound] 内寻找使函数值等于 target 的输入值。
    
    适用场景：单调函数的根求解或目标值查找（如动画中物体碰撞时间点计算）。
    
    参数：
        function - 待查找的单变量函数（假设在区间内单调）
        target - 目标函数值
        lower_bound - 查找下界
        upper_bound - 查找上界
        tolerance - 精度阈值（区间长度小于此值时停止）
    返回：
        找到的输入值（满足精度要求）或 None（目标值不在区间内）
    """
    lh = lower_bound  # 左边界
    rh = upper_bound  # 右边界
    mh = (lh + rh) / 2  # 中点

    while abs(rh - lh) > tolerance:
        # 计算三点的函数值
        lx, mx, rx = [function(h) for h in (lh, mh, rh)]
        # 检查边界点是否正好命中目标
        if lx == target:
            return lh
        if rx == target:
            return rh

        # 判断目标是否在当前区间内
        if (lx <= target <= rx) or (rx <= target <= lx):
            # 调整边界
            if mx > target:
                rh = mh  # 中点值过大，收缩右边界
            else:
                lh = mh  # 中点值过小，收缩左边界
        else:
            # 目标不在区间内
            return None
        mh = (lh + rh) / 2  # 更新中点

    return mh  # 返回满足精度的近似解


# ------------------------------ 字符串哈希 ------------------------------
def hash_string(string: str, n_bytes=16) -> str:
    """
    计算字符串的 SHA-256 哈希值，返回前 n_bytes 个字符作为唯一标识。
    
    应用场景：生成缓存键、临时文件名或资源唯一 ID。
    
    参数：
        string - 输入字符串
        n_bytes - 哈希结果的长度（默认 16）
    返回：哈希字符串（小写十六进制）
    """
    hasher = hashlib.sha256(string.encode())  # 使用 SHA-256 算法
    return hasher.hexdigest()[:n_bytes]  # 截取前 n_bytes 个字符