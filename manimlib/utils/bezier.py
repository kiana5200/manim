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


def interpolate(start: Scalable, end: Scalable, alpha: float | VectN) -> Scalable:
    try:
        return (1 - alpha) * start + alpha * end
    except TypeError:
        log.debug(f"`start` parameter with type `{type(start)}` and dtype `{start.dtype}`")
        log.debug(f"`end` parameter with type `{type(end)}` and dtype `{end.dtype}`")
        log.debug(f"`alpha` parameter with value `{alpha}`")
        import sys
        sys.exit(2)


def outer_interpolate(
    start: Scalable,
    end: Scalable,
    alpha: Scalable,
) -> np.ndarray:
    result = np.outer(1 - alpha, start) + np.outer(alpha, end)
    return result.reshape((*np.shape(alpha), *np.shape(start)))


def set_array_by_interpolation(
    arr: np.ndarray,
    arr1: np.ndarray,
    arr2: np.ndarray,
    alpha: float,
    interp_func: Callable[[np.ndarray, np.ndarray, float], np.ndarray] = interpolate
) -> np.ndarray:
    arr[:] = interp_func(arr1, arr2, alpha)
    return arr


def integer_interpolate(
    start: int,
    end: int,
    alpha: float
) -> tuple[int, float]:
    """
    alpha is a float between 0 and 1.  This returns
    an integer between start and end (inclusive) representing
    appropriate interpolation between them, along with a
    "residue" representing a new proportion between the
    returned integer and the next one of the
    list.

    For example, if start=0, end=10, alpha=0.46, This
    would return (4, 0.6).
    """
    if alpha >= 1:
        return (end - 1, 1.0)
    if alpha <= 0:
        return (start, 0)
    value = int(interpolate(start, end, alpha))
    residue = ((end - start) * alpha) % 1
    return (value, residue)


def mid(start: Scalable, end: Scalable) -> Scalable:
    return (start + end) / 2.0


def inverse_interpolate(start: Scalable, end: Scalable, value: Scalable) -> np.ndarray:
    return np.true_divide(value - start, end - start)


def match_interpolate(
    new_start: Scalable,
    new_end: Scalable,
    old_start: Scalable,
    old_end: Scalable,
    old_value: Scalable
) -> Scalable:
    return interpolate(
        new_start, new_end,
        inverse_interpolate(old_start, old_end, old_value)
    )


def quadratic_bezier_points_for_arc(angle: float, n_components: int = 8):
    n_points = 2 * n_components + 1
    angles = np.linspace(0, angle, n_points)
    points = np.array([np.cos(angles), np.sin(angles), np.zeros(n_points)]).T
    # Adjust handles
    theta = angle / n_components
    points[1::2] /= np.cos(theta / 2)
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
