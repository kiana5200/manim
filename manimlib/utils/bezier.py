# ManimGL 贝塞尔曲线工具库：提供贝塞尔曲线生成、分段截取等核心功能，
# 支持任意阶贝塞尔曲线的参数化表示，以及从曲线中提取指定区间的子曲线，
# 是图形绘制（如 SVG 解析、自定义路径）和动画路径生成的基础。


from __future__ import annotations

import numpy as np
from scipy import linalg
from fontTools.cu2qu.cu2qu import curve_to_quadratic  #  cubic→quadratic 曲线转换（字体工具）

from manimlib.logger import log  # 日志工具
from manimlib.utils.simple_functions import choose  # 组合数计算
from manimlib.utils.space_ops import cross2d  # 2D 叉积
from manimlib.utils.space_ops import cross  # 3D 叉积
from manimlib.utils.space_ops import find_intersection  # 求交
from manimlib.utils.space_ops import midpoint  # 中点计算
from manimlib.utils.space_ops import get_norm  # 向量模长
from manimlib.utils.space_ops import z_to_vector  # z值转向量

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable, Sequence, TypeVar, Tuple
    from manimlib.typing import VectN, FloatArray, VectNArray, Vect3Array

    Scalable = TypeVar("Scalable", float, FloatArray)  # 支持标量和数组的泛型类型


# 曲线闭合阈值：判断曲线是否闭合的距离阈值
CLOSED_THRESHOLD = 0.001


# ------------------------------ 贝塞尔曲线生成 ------------------------------
def bezier(
    points: Sequence[float | FloatArray] | VectNArray
) -> Callable[[float], float | FloatArray]:
    """
    生成贝塞尔曲线函数：根据控制点序列创建参数化曲线函数，输入参数 t∈[0,1]，返回对应点坐标。
    
    贝塞尔曲线公式：B(t) = Σ (C(n,k) * t^k * (1-t)^(n-k) * Pk)，其中 n 为控制点数量-1，
    C(n,k) 为组合数，Pk 为第 k 个控制点。
    
    参数：points - 控制点序列（长度 ≥1，支持标量或 N 维向量）
    返回：函数，接收 t∈[0,1]，返回对应点的坐标（与控制点同维度）。
    异常：控制点为空时抛出异常。
    """
    if len(points) == 0:
        raise Exception("bezier cannot be called on an empty list")

    n = len(points) - 1  # 曲线阶数 = 控制点数量 - 1

    def result(t: float) -> float | FloatArray:
        """参数 t 对应的贝塞尔曲线上的点"""
        return sum(
            # 组合数 * t^k * (1-t)^(n-k) * 控制点
            ((1 - t)**(n - k)) * (t**k) * choose(n, k) * point
            for k, point in enumerate(points)
        )

    return result


# ------------------------------ 贝塞尔曲线分段截取 ------------------------------
def partial_bezier_points(
    points: Sequence[Scalable],
    a: float,
    b: float
) -> list[Scalable]:
    """
    截取贝塞尔曲线的 [a, b] 区间：根据原曲线的控制点，计算子区间 [a, b] 对应的新控制点，
    生成与原曲线同阶的子曲线，适用于任意阶贝塞尔曲线。
    
    算法逻辑：基于 de Casteljau 算法，通过递归细分曲线，提取指定区间的控制点。
    
    参数：
        points : 原曲线的控制点序列；
        a, b : 截取区间（0 ≤ a < b ≤ 1）。
    返回：list[Scalable] - 子曲线的控制点序列（长度与原控制点相同）。
    """
    if a == 1:
        # 特殊情况：a=1 时，子曲线退化为终点
        return [points[-1]] * len(points)

    # 步骤1：计算原曲线从 a 到 1 的子曲线控制点
    a_to_1 = [
        bezier(points[i:])(a)  # 对原曲线的子序列应用 a 参数
        for i in range(len(points))
    ]
    # 步骤2：计算 [a, b] 在 [a, 1] 中的相对比例
    end_prop = (b - a) / (1. - a)
    # 步骤3：从 a_to_1 中截取 [0, end_prop] 区间，得到 [a, b] 对应的控制点
    return [
        bezier(a_to_1[:i + 1])(end_prop)
        for i in range(len(points))
    ]


def partial_quadratic_bezier_points(
    points: Sequence[VectN] | VectNArray,
    a: float,
    b: float
) -> list[VectN]:
    """
    二次贝塞尔曲线的分段截取（优化版本）：针对二次贝塞尔曲线（3个控制点）的高效实现，
    比通用的 partial_bezier_points 更快，适用于高频调用场景（如 SVG 路径解析）。
    
    参数：
        points : 二次贝塞尔曲线的控制点（3个点）；
        a, b : 截取区间（0 ≤ a < b ≤ 1）。
    返回：list[VectN] - 子曲线的3个控制点。
    """
    if a == 1:
        # 特殊情况：a=1 时，子曲线退化为终点
        return 3 * [points[-1]]

    # 二次贝塞尔曲线公式（直接展开，避免调用通用 bezier 函数的开销）
    def curve(t):
        return (
            points[0] * (1 - t) * (1 - t) +  # 起点项
            2 * points[1] * t * (1 - t) +    # 控制点项
            points[2] * t * t                # 终点项
        )

    # 计算子曲线的起点 h0 和终点 h2
    h0 = curve(a) if a > 0 else points[0]  # a=0 时直接取原起点
    h2 = curve(b) if b < 1 else points[2]  # b=1 时直接取原终点

    # 计算中间控制点 h1（基于 de Casteljau 算法的简化）
    h1_prime = (1 - a) * points[1] + a * points[2]  # 辅助点
    end_prop = (b - a) / (1. - a)  # 相对比例
    h1 = (1 - end_prop) * h0 + end_prop * h1_prime   # 子曲线控制点

    return [h0, h1, h2]

# Linear interpolation variants


# ManimGL 插值计算工具集：提供多种插值方式（线性、外插、整数插值等），
# 支持数值、向量、数组的平滑过渡，是动画参数变化、图形变形、路径生成的核心计算模块。


from __future__ import annotations

import numpy as np

from manimlib.logger import log  # 日志工具
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable
    from manimlib.typing import Scalable, VectN, FloatArray  # 类型注解


# ------------------------------ 基础线性插值 ------------------------------
def interpolate(start: Scalable, end: Scalable, alpha: float | VectN) -> Scalable:
    """
    线性插值：计算 start 到 end 之间按 alpha 比例的插值结果，支持标量、向量和数组。
    
    公式：result = (1 - alpha) * start + alpha * end
    
    参数：
        start : 起始值（标量、向量或数组，与 end 同类型/形状）；
        end : 目标值（与 start 兼容）；
        alpha : 插值比例（0→返回 start，1→返回 end，可超出范围实现外插）。
    返回：插值结果（与 start 同类型/形状）。
    异常处理：类型不兼容时输出调试信息并退出。
    """
    try:
        return (1 - alpha) * start + alpha * end
    except TypeError:
        # 输出调试信息帮助定位类型错误
        log.debug(f"`start` type: `{type(start)}`, dtype: `{start.dtype}`")
        log.debug(f"`end` type: `{type(end)}`, dtype: `{end.dtype}`")
        log.debug(f"`alpha` value: `{alpha}`")
        import sys
        sys.exit(2)  # 非零退出码标识错误


# ------------------------------ 外插值（多维度批量计算） ------------------------------
def outer_interpolate(
    start: Scalable,
    end: Scalable,
    alpha: Scalable,
) -> np.ndarray:
    """
    外插值：对 alpha 数组中的每个元素与 start/end 进行插值，适用于批量计算多组插值。
    
    应用场景：当需要为多个 alpha 值（如动画的每帧）计算插值结果时，避免循环提高效率。
    
    参数：
        start : 起始值（标量或数组）；
        end : 目标值（与 start 同形状）；
        alpha : 插值比例数组（1D 或高维）。
    返回：np.ndarray - 插值结果数组，形状为 (*alpha.shape, *start.shape)。
    """
    # 计算外积插值：(1-alpha)与start的外积 + alpha与end的外积
    result = np.outer(1 - alpha, start) + np.outer(alpha, end)
    # 调整形状以匹配 (alpha维度 + start维度)
    return result.reshape((*np.shape(alpha), *np.shape(start)))


# ------------------------------ 数组插值（原地修改） ------------------------------
def set_array_by_interpolation(
    arr: np.ndarray,
    arr1: np.ndarray,
    arr2: np.ndarray,
    alpha: float,
    interp_func: Callable[[np.ndarray, np.ndarray, float], np.ndarray] = interpolate
) -> np.ndarray:
    """
    数组插值并原地修改：将 arr1 和 arr2 的插值结果写入目标数组 arr（避免创建新对象），
    支持自定义插值函数（默认线性插值）。
    
    优势：减少内存占用，适用于大型数组或高频调用场景。
    
    参数：
        arr : 目标数组（将被修改）；
        arr1 : 起始数组；
        arr2 : 目标数组；
        alpha : 插值比例；
        interp_func : 自定义插值函数（默认使用 linear_interpolate）。
    返回：修改后的 arr 数组（与输入 arr 同引用）。
    """
    arr[:] = interp_func(arr1, arr2, alpha)  # 原地赋值，覆盖原有数据
    return arr


# ------------------------------ 整数插值（带余数） ------------------------------
def integer_interpolate(
    start: int,
    end: int,
    alpha: float
) -> tuple[int, float]:
    """
    整数插值：在整数 start 到 end 之间插值，返回当前整数和到下一个整数的比例，
    适用于需要离散整数但保持平滑过渡的场景（如帧索引、计数器动画）。
    
    示例：start=0, end=10, alpha=0.46 → (4, 0.6)，表示处于 4 到 5 之间的 60% 处。
    
    参数：
        start : 起始整数；
        end : 目标整数；
        alpha : 插值比例（0→start，1→end）。
    返回：tuple - (当前整数, 到下一个整数的比例)。
    """
    if alpha >= 1:
        return (end - 1, 1.0)  # alpha≥1 时取终点前一个整数，比例为1
    if alpha <= 0:
        return (start, 0)       # alpha≤0 时取起点，比例为0
    # 计算插值后的整数
    value = int(interpolate(start, end, alpha))
    # 计算到下一个整数的比例（余数）
    residue = ((end - start) * alpha) % 1
    return (value, residue)


# ------------------------------ 中点计算 ------------------------------
def mid(start: Scalable, end: Scalable) -> Scalable:
    """
    计算 start 和 end 的中点（alpha=0.5 时的插值结果）。
    
    公式：mid = (start + end) / 2.0
    
    参数：start, end - 两个值（标量、向量或数组，需兼容）
    返回：中点值（与输入同类型）。
    """
    return (start + end) / 2.0


# ------------------------------ 反向插值 ------------------------------
def inverse_interpolate(start: Scalable, end: Scalable, value: Scalable) -> np.ndarray:
    """
    反向插值：计算 value 在 start 到 end 区间内的相对比例（alpha 值）。
    
    公式：alpha = (value - start) / (end - start)
    
    参数：
        start : 起始值；
        end : 目标值；
        value : 位于 [start, end] 区间内的值。
    返回：np.ndarray - 相对比例 alpha（0→start，1→end，可外插）。
    """
    return np.true_divide(value - start, end - start)  # 安全除法，支持数组


# ------------------------------ 匹配插值（映射区间） ------------------------------
def match_interpolate(
    new_start: Scalable,
    new_end: Scalable,
    old_start: Scalable,
    old_end: Scalable,
    old_value: Scalable
) -> Scalable:
    """
    匹配插值：将 old_value 在 [old_start, old_end] 中的比例映射到 [new_start, new_end] 区间，
    实现不同区间之间的比例转换。
    
    示例：old区间 [0,10] 中的 5 映射到 new区间 [100,200] 中为 150。
    
    参数：
        new_start, new_end : 新区间的起止；
        old_start, old_end : 原区间的起止；
        old_value : 原区间中的值。
    返回：新区间中对应比例的值。
    """
    return interpolate(
        new_start, new_end,
        inverse_interpolate(old_start, old_end, old_value)  # 先计算原比例
    )


# ------------------------------ 圆弧的二次贝塞尔控制点生成 ------------------------------
def quadratic_bezier_points_for_arc(angle: float, n_components: int = 8) -> np.ndarray:
    """
    生成圆弧的二次贝塞尔曲线控制点：将圆弧分段，用多段二次贝塞尔曲线逼近圆弧，
    适用于需要用贝塞尔曲线绘制圆弧的场景（如SVG路径、平滑曲线动画）。
    
    参数：
        angle : 圆弧角度（弧度）；
        n_components : 分段数量（越多越逼近圆弧，默认8）。
    返回：np.ndarray - 控制点数组（形状为 (2n_components+1, 3)，包含起点、控制点、终点）。
    """
    n_points = 2 * n_components + 1  # 总控制点数量（每段2个控制点）
    # 生成从0到angle的均匀角度序列
    angles = np.linspace(0, angle, n_points)
    # 计算单位圆上的点（x=cosθ, y=sinθ, z=0）
    points = np.array([np.cos(angles), np.sin(angles), np.zeros(n_points)]).T
    # 调整控制点位置（使贝塞尔曲线更逼近圆弧）
    theta = angle / n_components  # 每段的角度
    points[1::2] /= np.cos(theta / 2)  # 奇数索引为控制点，按三角函数调整
    return points


# ManimGL 贝塞尔曲线平滑处理工具：提供二次/三次贝塞尔曲线的平滑控制点计算、
# 路径逼近等功能，用于将离散点序列转换为平滑曲线，适用于图形绘制和路径动画。


from __future__ import annotations

import numpy as np
from scipy import linalg
from fontTools.cu2qu.cu2qu import curve_to_quadratic  # cubic→quadratic 转换

# 导入辅助函数
from manimlib.utils.space_ops import cross  # 3D 叉积
from manimlib.utils.space_ops import get_norm  # 向量模长
from manimlib.utils.space_ops import z_to_vector  # z轴向量转换
from manimlib.utils.space_ops import is_closed  # 判断曲线是否闭合
from manimlib.utils.simple_functions import midpoint  # 中点计算
from manimlib.utils.bezier import get_quadratic_approximation_of_cubic  # 三次转二次逼近
from manimlib.utils.bezier import diag_to_matrix  # 对角矩阵转换

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from manimlib.typing import FloatArray, Vect3Array, VectN, VectNArray  # 类型注解


# ------------------------------ 二次贝塞尔平滑控制点计算 ------------------------------
def approx_smooth_quadratic_bezier_handles(
    points: FloatArray
) -> FloatArray:
    """
    计算二次贝塞尔曲线的平滑控制点：为离散点序列生成中间控制点，使相邻曲线段平滑连接（一阶导数连续）。
    
    算法逻辑：
    1. 对每个点 P1，计算两个候选控制点：
       - smooth_to_right：使曲线 (P0, h, P1) 经过 P2；
       - smooth_to_left：使曲线 (P1, h, P2) 经过 P0；
    2. 取两个候选点的中点作为最终控制点，平衡平滑性和路径精度。
    
    参数：points - 离散点序列（N×D 数组，D为维度）
    返回：FloatArray - 控制点数组（与输入点数量相同，每个点对应一个控制点）。
    """
    if len(points) == 1:
        return points[0]  # 单点时控制点为自身
    elif len(points) == 2:
        return midpoint(*points)  # 两点时控制点为中点

    # 计算向右/向左平滑的候选控制点
    smooth_to_right, smooth_to_left = [
        0.25 * ps[0:-2] + ps[1:-1] - 0.25 * ps[2:]  # 公式推导：确保曲线连贯性
        for ps in (points, points[::-1])  # 正向和反向计算
    ]

    # 处理闭合曲线的首尾连接（确保首尾控制点平滑）
    if np.isclose(points[0], points[-1]).all():
        # 闭合曲线：单独计算首尾控制点
        last_str = 0.25 * points[-2] + points[-1] - 0.25 * points[1]
        last_stl = 0.25 * points[1] + points[0] - 0.25 * points[-2]
    else:
        # 开放曲线：复用边缘的候选控制点
        last_str = smooth_to_left[0]
        last_stl = smooth_to_right[0]

    # 合并候选控制点，取中点作为最终结果
    handles = 0.5 * np.vstack([smooth_to_right, [last_str]])
    handles += 0.5 * np.vstack([last_stl, smooth_to_left[::-1]])
    return handles


# ------------------------------ 平滑二次贝塞尔路径生成 ------------------------------
def smooth_quadratic_path(anchors: Vect3Array) -> Vect3Array:
    """
    生成平滑二次贝塞尔路径：将锚点序列转换为连续的二次贝塞尔曲线段，确保整体平滑（适用于3D点）。
    
    流程：
    1. 处理3D锚点：若不在同一平面，旋转到2D平面计算；
    2. 先生成三次贝塞尔曲线的平滑控制点；
    3. 将三次曲线分段逼近为二次贝塞尔曲线；
    4. 转回原3D坐标系（若有旋转）。
    
    参数：anchors - 3D锚点序列（N×3 数组）
    返回：Vect3Array - 二次贝塞尔路径的控制点数组（包含锚点和中间控制点）。
    """
    if len(anchors) < 2:
        return anchors  # 少于2个点直接返回
    elif len(anchors) == 2:
        # 两个点：生成简单二次曲线（起点-中点-终点）
        return np.array([anchors[0], anchors.mean(0), anchors[1]])

    # 检查锚点是否在同一平面（z坐标均为0）
    is_flat = (anchors[:, 2] == 0).all()
    if not is_flat:
        # 非平面曲线：计算旋转矩阵，将3D点投影到2D平面
        normal = cross(anchors[2] - anchors[1], anchors[1] - anchors[0])  # 平面法向量
        rot = z_to_vector(normal)  # 生成旋转矩阵（将法向量转为z轴）
        anchors = np.dot(anchors, rot)  # 旋转到2D平面（z坐标对齐）
        shift = anchors[0, 2]  # 记录z轴偏移
        anchors[:, 2] -= shift  # 归零z坐标

    # 生成三次贝塞尔曲线的平滑控制点
    h1s, h2s = get_smooth_cubic_bezier_handle_points(anchors)
    quads = [anchors[0, :2]]  # 存储二次曲线控制点（先添加起点）

    # 遍历每个三次曲线段，转换为二次曲线
    for cub_bs in zip(anchors[:-1], h1s, h2s, anchors[1:]):
        # 尝试用fontTools的高精度转换（cubic→quadratic）
        new_quads = curve_to_quadratic(
            [b[:2] for b in cub_bs],  # 取2D坐标
            max_err=0.1 * get_norm(cub_bs[3] - cub_bs[0])  # 误差阈值（与线段长度相关）
        )
        # 转换失败时使用自定义逼近方法
        if new_quads is None or len(new_quads) % 2 == 0:
            new_quads = get_quadratic_approximation_of_cubic(*cub_bs)[:, :2]
        quads.extend(new_quads[1:])  # 跳过重复的起点

    # 构建3D路径（恢复z坐标）
    new_path = np.zeros((len(quads), 3))
    new_path[:, :2] = quads
    if not is_flat:
        new_path[:, 2] += shift  # 恢复z轴偏移
        new_path = np.dot(new_path, rot.T)  # 旋转回原坐标系

    return new_path


# ------------------------------ 三次贝塞尔平滑控制点计算 ------------------------------
def get_smooth_cubic_bezier_handle_points(
    points: Sequence[VectN] | VectNArray
) -> tuple[FloatArray, FloatArray]:
    """
    计算三次贝塞尔曲线的平滑控制点：为离散点序列生成两组控制点（h1, h2），
    使相邻三次贝塞尔曲线段（P0, h1, h2, P1）平滑连接（一阶导数连续）。
    
    算法：通过求解带状线性方程组，确保各段曲线在连接点处导数连续，适用于开放和闭合曲线。
    
    参数：points - 离散点序列（N×D 数组）
    返回：tuple[FloatArray, FloatArray] - (h1s, h2s)，每组控制点数量为 N-1。
    """
    points = np.array(points)
    num_handles = len(points) - 1  # 控制点对数量 = 点数量 - 1
    dim = points.shape[1]  # 维度（2D或3D）
    if num_handles < 1:
        return np.zeros((0, dim)), np.zeros((0, dim))  # 不足1段时返回空

    # 构建带状矩阵（用于求解线性方程组 Ax = b）
    # 参考：https://www.particleincell.com/2012/bezier-splines/
    l, u = 2, 1  # 下三角/上三角带宽
    diag = np.zeros((l + u + 1, 2 * num_handles))  # 对角矩阵表示
    # 填充矩阵元素（根据平滑条件推导的系数）
    diag[0, 1::2] = -1
    diag[0, 2::2] = 1
    diag[1, 0::2] = 2
    diag[1, 1::2] = 1
    diag[2, 1:-2:2] = -2
    diag[3, 0:-3:2] = 1
    # 最后一行特殊处理
    diag[2, -2] = -1
    diag[1, -1] = 2

    # 构建方程组的右侧向量 b（每个维度独立求解）
    b = np.zeros((2 * num_handles, dim))
    b[1::2] = 2 * points[1:]  # 奇数索引：2×中间点
    b[0] = points[0]           # 第一个元素：起点
    b[-1] = points[-1]         # 最后一个元素：终点

    # 求解带状矩阵方程组的函数
    def solve_func(b):
        return linalg.solve_banded((l, u), diag, b)

    # 处理闭合曲线（首尾点相同）
    use_closed_solve_function = is_closed(points)
    if use_closed_solve_function:
        # 闭合曲线需要额外约束：首尾导数连续
        matrix = diag_to_matrix((l, u), diag)  # 转换为完整矩阵
        # 最后一行：二阶导数约束
        matrix[-1, [0, 1, -2, -1]] = [2, -1, 1, -2]
        # 第一行：一阶导数约束
        matrix[0, :] = np.zeros(matrix.shape[1])
        matrix[0, [0, -1]] = [1, 1]
        # 调整右侧向量 b
        b[0] = 2 * points[0]
        b[-1] = np.zeros(dim)

        # 闭合曲线的求解函数
        def closed_curve_solve_func(b):
            return linalg.solve(matrix, b)

    # 求解各维度的控制点
    handle_pairs = np.zeros((2 * num_handles, dim))
    for i in range(dim):
        if use_closed_solve_function:
            handle_pairs[:, i] = closed_curve_solve_func(b[:, i])
        else:
            handle_pairs[:, i] = solve_func(b[:, i])

    # 分离两组控制点（h1s 对应每个段的第一个控制点，h2s 对应第二个）
    return handle_pairs[0::2], handle_pairs[1::2]

# ManimGL 贝塞尔曲线辅助工具：提供对角矩阵转换、曲线闭合判断、三次转二次曲线逼近等功能，
# 辅助处理复杂曲线的平滑化和格式转换，是贝塞尔曲线生成与优化的重要支撑。


from __future__ import annotations

import numpy as np
from scipy import linalg

# 导入基础工具函数
from manimlib.utils.bezier import bezier  # 贝塞尔曲线生成
from manimlib.utils.space_ops import cross2d  # 2D 叉积计算
from manimlib.utils.space_ops import find_intersection  # 直线交点计算

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from manimlib.typing import FloatArray, VectN, Sequence  # 类型注解


# ------------------------------ 对角矩阵转换 ------------------------------
def diag_to_matrix(
    l_and_u: tuple[int, int], 
    diag: np.ndarray
) -> np.ndarray:
    """
    将带状对角矩阵表示转换为完整矩阵形式，与 scipy.linalg.solve_banded 兼容。
    
    带状矩阵存储格式：每行代表矩阵的一条对角线（从下到上），适用于高效求解线性方程组，
    该函数将其还原为标准的二维矩阵，便于后续处理（如闭合曲线的额外约束）。
    
    参数：
        l_and_u : 元组 (l, u)，表示下三角和上三角的带宽；
        diag : 带状矩阵的对角线数组（形状为 (l+u+1, n)）。
    返回：np.ndarray - 完整的二维矩阵（形状为 (n, n)）。
    """
    l, u = l_and_u
    dim = diag.shape[1]  # 矩阵维度
    matrix = np.zeros((dim, dim))  # 初始化完整矩阵

    # 填充每条对角线到完整矩阵
    for i in range(l + u + 1):
        # 计算对角线在完整矩阵中的起始位置
        row_start = max(0, i - u)
        col_start = max(0, u - i)
        # 填充对角线元素
        np.fill_diagonal(
            matrix[row_start:, col_start:],  # 目标对角线位置
            diag[i, col_start:]  # 对应对角线的元素
        )
    return matrix


# ------------------------------ 曲线闭合判断 ------------------------------
def is_closed(points: FloatArray) -> bool:
    """
    判断点序列是否形成闭合曲线（起点与终点是否接近）。
    
    参数：points - 点序列数组（N×D 形状）
    返回：bool - 若起点与终点的所有维度均接近（默认精度），则返回 True。
    """
    return np.allclose(points[0], points[-1])  # 检查起点和终点是否接近


# ------------------------------ 三次贝塞尔转二次贝塞尔逼近 ------------------------------
def get_quadratic_approximation_of_cubic(
    a0: FloatArray,
    h0: FloatArray,
    h1: FloatArray,
    a1: FloatArray
) -> FloatArray:
    """
    将三次贝塞尔曲线用两段二次贝塞尔曲线逼近，平衡精度与计算效率。
    
    算法流程：
    1. 计算三次曲线的起点、终点和切线；
    2. 检测拐点（若存在，以此为分段点；否则用中点）；
    3. 计算分段点处的切线，求切线交点作为二次曲线的控制点；
    4. 生成两段二次曲线的控制点序列。
    
    参数：
        a0, a1 : 三次贝塞尔曲线的起点和终点；
        h0, h1 : 三次贝塞尔曲线的两个控制点。
    返回：np.ndarray - 二次贝塞尔曲线的控制点数组（形状为 (5*N, D)，每5个点代表两段二次曲线）。
    """
    # 确保输入为二维数组（支持批量处理多个曲线）
    a0 = np.array(a0, ndmin=2)
    h0 = np.array(h0, ndmin=2)
    h1 = np.array(h1, ndmin=2)
    a1 = np.array(a1, ndmin=2)

    # 计算起点和终点的切线向量
    T0 = h0 - a0  # 起点切线（h0 - a0）
    T1 = a1 - h1  # 终点切线（a1 - h1）

    # 检测三次曲线的拐点（基于二阶导数为零的条件）
    # 参考：http://www.caffeineowl.com/graphics/2d/vectorial/cubic-inflexion.html
    has_infl = np.ones(len(a0), dtype=bool)  # 标记是否有拐点

    # 三次曲线的系数（用于计算拐点）
    p = h0 - a0
    q = h1 - 2 * h0 + a0
    r = a1 - 3 * h1 + 3 * h0 - a0

    # 计算二阶导数为零的方程系数（ax² + bx + c = 0）
    a = cross2d(q, r)
    b = cross2d(p, r)
    c = cross2d(p, q)

    # 判别式（判断是否有实根）
    disc = b * b - 4 * a * c
    has_infl &= (disc > 0)  # 仅保留有实根的曲线
    sqrt_disc = np.sqrt(np.abs(disc))  # 平方根（取绝对值避免负数）

    # 计算拐点的参数 t（暂时忽略警告）
    settings = np.seterr(all='ignore')
    ti_bounds = []
    for sgn in [-1, +1]:
        ti = (-b + sgn * sqrt_disc) / (2 * a)
        # 处理 a=0 的特殊情况（降为一次方程）
        ti[a == 0] = (-c / b)[a == 0]
        # 处理 a=0 且 b=0 的退化情况
        ti[(a == 0) & (b == 0)] = 0
        ti_bounds.append(ti)
    ti_min, ti_max = ti_bounds
    np.seterr(** settings)  # 恢复警告设置

    # 判断拐点是否在 [0, 1] 区间内
    ti_min_in_range = has_infl & (0 < ti_min) & (ti_min < 1)
    ti_max_in_range = has_infl & (0 < ti_max) & (ti_max < 1)

    # 选择分段点 t（优先用拐点，否则用中点）
    t_mid = 0.5 * np.ones(len(a0))  # 默认中点 t=0.5
    t_mid[ti_min_in_range] = ti_min[ti_min_in_range]  # 替换为有效拐点
    t_mid[ti_max_in_range] = ti_max[ti_max_in_range]

    # 扩展 t_mid 维度以匹配点的维度（支持批量计算）
    m, n = a0.shape
    t_mid = t_mid.repeat(n).reshape((m, n))

    # 计算分段点的坐标和切线
    mid = bezier([a0, h0, h1, a1])(t_mid)  # 分段点坐标
    Tm = bezier([h0 - a0, h1 - h0, a1 - h1])(t_mid)  # 分段点切线

    # 计算切线交点作为二次曲线的控制点
    i0 = find_intersection(a0, T0, mid, Tm)  # 起点切线与分段点切线的交点
    i1 = find_intersection(a1, T1, mid, Tm)  # 终点切线与分段点切线的交点

    # 构建二次贝塞尔曲线的控制点序列（每5个点代表两段曲线：a0→i0→mid→i1→a1）
    m, n = np.shape(a0)
    result = np.zeros((5 * m, n))
    result[0::5] = a0   # 第一段起点
    result[1::5] = i0   # 第一段控制点
    result[2::5] = mid  # 两段连接点（第一段终点，第二段起点）
    result[3::5] = i1   # 第二段控制点
    result[4::5] = a1   # 第二段终点
    return result


# ------------------------------ 平滑二次贝塞尔路径生成（待完善） ------------------------------
def get_smooth_quadratic_bezier_path_through(
    points: Sequence[VectN]
) -> np.ndarray:
    """
    生成通过指定点的平滑二次贝塞尔路径（未完善）。
    
    思路：先计算三次贝塞尔曲线的平滑控制点，再转换为二次贝塞尔曲线逼近。
    
    参数：points - 点序列
    返回：np.ndarray - 二次贝塞尔曲线的控制点数组。
    """
    # TODO: 完善路径生成逻辑，处理端点和平滑性
    h0, h1 = get_smooth_cubic_bezier_handle_points(points)  # 获取三次曲线控制点
    a0 = points[:-1]  # 各段起点
    a1 = points[1:]   # 各段终点
    return get_quadratic_approximation_of_cubic(a0, h0, h1, a1)  # 转换为二次曲线