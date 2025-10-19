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


def approx_smooth_quadratic_bezier_handles(
    points: FloatArray
) -> FloatArray:
    """
    Figuring out which bezier curves most smoothly connect a sequence of points.

    Given three successive points, P0, P1 and P2, you can compute that by defining
    h = (1/4) P0 + P1 - (1/4)P2, the bezier curve defined by (P0, h, P1) will pass
    through the point P2.

    So for a given set of four successive points, P0, P1, P2, P3, if we want to add
    a handle point h between P1 and P2 so that the quadratic bezier (P1, h, P2) is
    part of a smooth curve passing through all four points, we calculate one solution
    for h that would produce a parbola passing through P3, call it smooth_to_right, and
    another that would produce a parabola passing through P0, call it smooth_to_left,
    and use the midpoint between the two.
    """
    if len(points) == 1:
        return points[0]
    elif len(points) == 2:
        return midpoint(*points)
    smooth_to_right, smooth_to_left = [
        0.25 * ps[0:-2] + ps[1:-1] - 0.25 * ps[2:]
        for ps in (points, points[::-1])
    ]
    if np.isclose(points[0], points[-1]).all():
        last_str = 0.25 * points[-2] + points[-1] - 0.25 * points[1]
        last_stl = 0.25 * points[1] + points[0] - 0.25 * points[-2]
    else:
        last_str = smooth_to_left[0]
        last_stl = smooth_to_right[0]
    handles = 0.5 * np.vstack([smooth_to_right, [last_str]])
    handles += 0.5 * np.vstack([last_stl, smooth_to_left[::-1]])
    return handles


def smooth_quadratic_path(anchors: Vect3Array) -> Vect3Array:
    """
    Returns a path defining a smooth quadratic bezier spline
    through anchors.
    """
    if len(anchors) < 2:
        return anchors
    elif len(anchors) == 2:
        return np.array([anchors[0], anchors.mean(0), anchors[1]])

    is_flat = (anchors[:, 2] == 0).all()
    if not is_flat:
        normal = cross(anchors[2] - anchors[1], anchors[1] - anchors[0])
        rot = z_to_vector(normal)
        anchors = np.dot(anchors, rot)
        shift = anchors[0, 2]
        anchors[:, 2] -= shift
    h1s, h2s = get_smooth_cubic_bezier_handle_points(anchors)
    quads = [anchors[0, :2]]
    for cub_bs in zip(anchors[:-1], h1s, h2s, anchors[1:]):
        # Try to use fontTools curve_to_quadratic
        new_quads = curve_to_quadratic(
            [b[:2] for b in cub_bs],
            max_err=0.1 * get_norm(cub_bs[3] - cub_bs[0])
        )
        # Otherwise fall back on home baked solution
        if new_quads is None or len(new_quads) % 2 == 0:
            new_quads = get_quadratic_approximation_of_cubic(*cub_bs)[:, :2]
        quads.extend(new_quads[1:])
    new_path = np.zeros((len(quads), 3))
    new_path[:, :2] = quads
    if not is_flat:
        new_path[:, 2] += shift
        new_path = np.dot(new_path, rot.T)
    return new_path


def get_smooth_cubic_bezier_handle_points(
    points: Sequence[VectN] | VectNArray
) -> tuple[FloatArray, FloatArray]:
    points = np.array(points)
    num_handles = len(points) - 1
    dim = points.shape[1]
    if num_handles < 1:
        return np.zeros((0, dim)), np.zeros((0, dim))
    # Must solve 2*num_handles equations to get the handles.
    # l and u are the number of lower an upper diagonal rows
    # in the matrix to solve.
    l, u = 2, 1
    # diag is a representation of the matrix in diagonal form
    # See https://www.particleincell.com/2012/bezier-splines/
    # for how to arrive at these equations
    diag = np.zeros((l + u + 1, 2 * num_handles))
    diag[0, 1::2] = -1
    diag[0, 2::2] = 1
    diag[1, 0::2] = 2
    diag[1, 1::2] = 1
    diag[2, 1:-2:2] = -2
    diag[3, 0:-3:2] = 1
    # last
    diag[2, -2] = -1
    diag[1, -1] = 2
    # This is the b as in Ax = b, where we are solving for x,
    # and A is represented using diag.  However, think of entries
    # to x and b as being points in space, not numbers
    b = np.zeros((2 * num_handles, dim))
    b[1::2] = 2 * points[1:]
    b[0] = points[0]
    b[-1] = points[-1]

    def solve_func(b):
        return linalg.solve_banded((l, u), diag, b)

    use_closed_solve_function = is_closed(points)
    if use_closed_solve_function:
        # Get equations to relate first and last points
        matrix = diag_to_matrix((l, u), diag)
        # last row handles second derivative
        matrix[-1, [0, 1, -2, -1]] = [2, -1, 1, -2]
        # first row handles first derivative
        matrix[0, :] = np.zeros(matrix.shape[1])
        matrix[0, [0, -1]] = [1, 1]
        b[0] = 2 * points[0]
        b[-1] = np.zeros(dim)

        def closed_curve_solve_func(b):
            return linalg.solve(matrix, b)

    handle_pairs = np.zeros((2 * num_handles, dim))
    for i in range(dim):
        if use_closed_solve_function:
            handle_pairs[:, i] = closed_curve_solve_func(b[:, i])
        else:
            handle_pairs[:, i] = solve_func(b[:, i])
    return handle_pairs[0::2], handle_pairs[1::2]


def diag_to_matrix(
    l_and_u: tuple[int, int], 
    diag: np.ndarray
) -> np.ndarray:
    """
    Converts array whose rows represent diagonal
    entries of a matrix into the matrix itself.
    See scipy.linalg.solve_banded
    """
    l, u = l_and_u
    dim = diag.shape[1]
    matrix = np.zeros((dim, dim))
    for i in range(l + u + 1):
        np.fill_diagonal(
            matrix[max(0, i - u):, max(0, u - i):],
            diag[i, max(0, u - i):]
        )
    return matrix


def is_closed(points: FloatArray) -> bool:
    return np.allclose(points[0], points[-1])


# Given 4 control points for a cubic bezier curve (or arrays of such)
# return control points for 2 quadratics (or 2n quadratics) approximating them.
def get_quadratic_approximation_of_cubic(
    a0: FloatArray,
    h0: FloatArray,
    h1: FloatArray,
    a1: FloatArray
) -> FloatArray:
    a0 = np.array(a0, ndmin=2)
    h0 = np.array(h0, ndmin=2)
    h1 = np.array(h1, ndmin=2)
    a1 = np.array(a1, ndmin=2)
    # Tangent vectors at the start and end.
    T0 = h0 - a0
    T1 = a1 - h1

    # Search for inflection points.  If none are found, use the
    # midpoint as a cut point.
    # Based on http://www.caffeineowl.com/graphics/2d/vectorial/cubic-inflexion.html
    has_infl = np.ones(len(a0), dtype=bool)

    p = h0 - a0
    q = h1 - 2 * h0 + a0
    r = a1 - 3 * h1 + 3 * h0 - a0

    a = cross2d(q, r)
    b = cross2d(p, r)
    c = cross2d(p, q)

    disc = b * b - 4 * a * c
    has_infl &= (disc > 0)
    sqrt_disc = np.sqrt(np.abs(disc))
    settings = np.seterr(all='ignore')
    ti_bounds = []
    for sgn in [-1, +1]:
        ti = (-b + sgn * sqrt_disc) / (2 * a)
        ti[a == 0] = (-c / b)[a == 0]
        ti[(a == 0) & (b == 0)] = 0
        ti_bounds.append(ti)
    ti_min, ti_max = ti_bounds
    np.seterr(**settings)
    ti_min_in_range = has_infl & (0 < ti_min) & (ti_min < 1)
    ti_max_in_range = has_infl & (0 < ti_max) & (ti_max < 1)

    # Choose a value of t which starts at 0.5,
    # but is updated to one of the inflection points
    # if they lie between 0 and 1

    t_mid = 0.5 * np.ones(len(a0))
    t_mid[ti_min_in_range] = ti_min[ti_min_in_range]
    t_mid[ti_max_in_range] = ti_max[ti_max_in_range]

    m, n = a0.shape
    t_mid = t_mid.repeat(n).reshape((m, n))

    # Compute bezier point and tangent at the chosen value of t
    mid = bezier([a0, h0, h1, a1])(t_mid)
    Tm = bezier([h0 - a0, h1 - h0, a1 - h1])(t_mid)

    # Intersection between tangent lines at end points
    # and tangent in the middle
    i0 = find_intersection(a0, T0, mid, Tm)
    i1 = find_intersection(a1, T1, mid, Tm)

    m, n = np.shape(a0)
    result = np.zeros((5 * m, n))
    result[0::5] = a0
    result[1::5] = i0
    result[2::5] = mid
    result[3::5] = i1
    result[4::5] = a1
    return result


def get_smooth_quadratic_bezier_path_through(
    points: Sequence[VectN]
) -> np.ndarray:
    # TODO
    h0, h1 = get_smooth_cubic_bezier_handle_points(points)
    a0 = points[:-1]
    a1 = points[1:]
    return get_quadratic_approximation_of_cubic(a0, h0, h1, a1)
