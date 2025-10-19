# ManimGL 路径生成工具：提供点集从起始位置到目标位置的移动路径计算，
# 支持直线运动和圆弧运动（顺时针/逆时针），是动画中物体位置过渡的核心逻辑。


from __future__ import annotations

import math

import numpy as np

# 导入常量和工具函数：3D 方向向量、插值函数、向量运算
from manimlib.constants import OUT  # 3D 向外方向向量（z轴正方向）
from manimlib.utils.bezier import interpolate  # 线性插值函数
from manimlib.utils.space_ops import get_norm  # 向量求模（长度）
from manimlib.utils.space_ops import rotation_matrix_transpose  # 旋转矩阵的转置（用于坐标变换）

from typing import TYPE_CHECKING

# 仅静态类型检查阶段导入类型注解
if TYPE_CHECKING:
    from typing import Callable
    from manimlib.typing import Vect3, Vect3Array  # 3D 向量、3D 向量数组


# 直线路径阈值：当圆弧角度小于此值时，近似为直线运动（避免微小圆弧的计算开销）
STRAIGHT_PATH_THRESHOLD = 0.01


# ------------------------------ 直线路径 ------------------------------
def straight_path(
    start_points: np.ndarray,
    end_points: np.ndarray,
    alpha: float
) -> np.ndarray:
    """
    直线路径：计算点集从起始位置到目标位置的线性插值位置，是最基础的运动路径。
    
    本质上是对每个点执行 `start + alpha * (end - start)` 的线性插值，
    适用于物体沿直线移动的动画（如 `MoveAlongPath`、`Transform` 中的位置过渡）。
    
    参数：
        start_points : 起始点集（N×3 数组，每个行向量为一个点的 3D 坐标）；
        end_points : 目标点集（与 start_points 同形状）；
        alpha : 插值参数（0→起始位置，1→目标位置）。
    返回：np.ndarray - alpha 时刻的点集位置（与输入同形状）。
    """
    return interpolate(start_points, end_points, alpha)  # 直接调用线性插值函数


# ------------------------------ 圆弧路径生成器 ------------------------------
def path_along_arc(
    arc_angle: float, 
    axis: Vect3 = OUT
) -> Callable[[Vect3Array, Vect3Array, float], Vect3Array]:
    """
    生成圆弧路径函数：返回一个函数，该函数计算点集沿圆弧从起始位置到目标位置的过渡位置，
    圆弧的角度和旋转轴可自定义，适用于物体沿曲线运动的动画（如弧形移动、旋转过渡）。
    
    核心逻辑：
    1. 若圆弧角度接近 0（小于阈值），返回直线路径（优化计算）；
    2. 计算每个点对的圆弧圆心（基于起始点、目标点和旋转轴）；
    3. 对每个点，绕圆心按指定角度和轴旋转，得到 alpha 时刻的位置。
    
    参数：
        arc_angle : 圆弧角度（弧度，正值为逆时针，负值为顺时针）；
        axis : 旋转轴（3D 向量，默认 OUT 即 z 轴，控制圆弧所在平面）。
    返回：函数，接收 start_points、end_points、alpha，返回对应时刻的点集位置。
    """
    # 微小角度时，直接使用直线路径（减少计算量，视觉效果近似）
    if abs(arc_angle) < STRAIGHT_PATH_THRESHOLD:
        return straight_path
    # 确保旋转轴不为零向量（否则使用默认 OUT 轴）
    if get_norm(axis) == 0:
        axis = OUT
    unit_axis = axis / get_norm(axis)  # 归一化旋转轴（确保是单位向量）

    def path(start_points, end_points, alpha):
        # 计算起始点到目标点的向量（位移向量）
        vects = end_points - start_points
        # 圆弧的圆心初步定位在起点和终点的中点
        centers = start_points + 0.5 * vects

        # 非半圆时，调整圆心位置（确保圆弧半径正确）
        if arc_angle != np.pi:
            # 计算垂直于位移向量和旋转轴的方向（圆弧的径向方向）
            # 公式推导：圆心偏移量 = (旋转轴 × 位移向量/2) / tan(角度/2)
            centers += np.cross(unit_axis, vects / 2.0) / math.tan(arc_angle / 2)

        # 计算旋转矩阵的转置（用于将点绕轴旋转指定角度）
        # alpha * arc_angle 表示当前插值对应的旋转角度
        rot_matrix_T = rotation_matrix_transpose(alpha * arc_angle, unit_axis)
        # 每个点绕圆心旋转：(起点 - 圆心) 旋转后 + 圆心
        return centers + np.dot(start_points - centers, rot_matrix_T)

    return path


# ------------------------------ 特定方向圆弧路径 ------------------------------
def clockwise_path() -> Callable[[Vect3Array, Vect3Array, float], Vect3Array]:
    """
    生成顺时针半圆路径函数：圆弧角度为 -π（180度），默认绕 OUT 轴旋转，
    适用于物体沿顺时针方向移动半个圆弧的场景。
    
    返回：圆弧路径函数（同 path_along_arc 的返回类型）。
    """
    return path_along_arc(-np.pi)  # 负角度表示顺时针


def counterclockwise_path() -> Callable[[Vect3Array, Vect3Array, float], Vect3Array]:
    """
    生成逆时针半圆路径函数：圆弧角度为 π（180度），默认绕 OUT 轴旋转，
    适用于物体沿逆时针方向移动半个圆弧的场景。
    
    返回：圆弧路径函数（同 path_along_arc 的返回类型）。
    """
    return path_along_arc(np.pi)  # 正角度表示逆时针