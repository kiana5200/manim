# ManimGL 空间运算工具库：提供向量运算、距离计算、归一化、多边形长度等基础空间几何功能，
# 支持 2D/3D 向量操作，是图形变换、碰撞检测、路径计算的核心工具。


from __future__ import annotations

from functools import reduce
import math
import operator as op
import platform

from mapbox_earcut import triangulate_float32 as earcut  # 多边形三角化工具
import numpy as np
from scipy.spatial.transform import Rotation  # 旋转处理
from tqdm.auto import tqdm as ProgressDisplay  # 进度条

# 导入常量和工具函数
from manimlib.constants import DOWN, OUT, RIGHT, UP  # 方向向量
from manimlib.constants import PI, TAU  # 圆周率常量
from manimlib.utils.iterables import adjacent_pairs  # 相邻元素对
from manimlib.utils.simple_functions import clip  # 数值裁剪

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Callable, Sequence, List, Tuple
    from manimlib.typing import Vect2, Vect3, Vect4, VectN, Matrix3x3, Vect3Array, Vect2Array  # 类型注解


# ------------------------------ 向量叉积 ------------------------------
def cross(
    v1: Vect3 | List[float],
    v2: Vect3 | List[float],
    out: np.ndarray | None = None
) -> Vect3 | Vect3Array:
    """
    计算 3D 向量的叉积（向量积），支持单个向量和向量数组（批量计算）。
    
    叉积公式：
    v1 × v2 = [
        v1.y*v2.z - v1.z*v2.y,
        v1.z*v2.x - v1.x*v2.z,
        v1.x*v2.y - v1.y*v2.x
    ]
    
    参数：
        v1 : 第一个 3D 向量（或向量数组）；
        v2 : 第二个 3D 向量（或向量数组）；
        out : 输出数组（可选，用于原地计算）。
    返回：Vect3 或 Vect3Array - 叉积结果，与输入同形状。
    """
    # 判断是否为批量处理（2D 数组，形状为 (N, 3)）
    is2d = isinstance(v1, np.ndarray) and len(v1.shape) == 2
    if is2d:
        # 批量处理：分离 x, y, z 分量（按列提取）
        x1, y1, z1 = v1[:, 0], v1[:, 1], v1[:, 2]
        x2, y2, z2 = v2[:, 0], v2[:, 1], v2[:, 2]
    else:
        # 单个向量：直接解包分量
        x1, y1, z1 = v1
        x2, y2, z2 = v2

    # 初始化输出数组（若未提供则自动创建）
    if out is None:
        out = np.empty(np.shape(v1))
    # 计算叉积并赋值（按行填充结果）
    out.T[:] = [
        y1 * z2 - z1 * y2,  # x 分量
        z1 * x2 - x1 * z2,  # y 分量
        x1 * y2 - y1 * x2   # z 分量
    ]
    return out


# ------------------------------ 向量模长与距离 ------------------------------
def get_norm(vect: VectN | List[float]) -> float:
    """
    计算向量的模长（L2 范数）。
    
    公式：||v|| = sqrt(v1² + v2² + ... + vn²)
    
    参数：vect - N 维向量（或列表）
    返回：float - 向量的模长。
    """
    return sum((x**2 for x in vect))** 0.5


def get_dist(vect1: VectN, vect2: VectN) -> float:
    """
    计算两个向量之间的欧氏距离（即两向量差的模长）。
    
    公式：distance = ||v2 - v1||
    
    参数：
        vect1 : 第一个 N 维向量；
        vect2 : 第二个 N 维向量。
    返回：float - 两向量间的距离。
    """
    return get_norm(vect2 - vect1)


# ------------------------------ 向量归一化 ------------------------------
def normalize(
    vect: VectN | List[float],
    fall_back: VectN | List[float] | None = None
) -> VectN:
    """
    将向量归一化（单位向量），即模长为 1 的向量。
    
    处理逻辑：
    - 若向量模长 > 0，返回 vect / ||vect||；
    - 若模长为 0 且提供 fall_back，返回 fall_back（默认归一化）；
    - 否则返回零向量。
    
    参数：
        vect : 待归一化的 N 维向量；
        fall_back : 模长为 0 时的 fallback 向量（可选）。
    返回：VectN - 归一化后的向量。
    """
    norm = get_norm(vect)
    if norm > 0:
        return np.array(vect) / norm
    elif fall_back is not None:
        return np.array(fall_back)
    else:
        return np.zeros(len(vect))


# ------------------------------ 多边形周长计算 ------------------------------
def poly_line_length(points: VectNArray) -> float:
    """
    计算折线（多边形边）的总长度，即相邻点之间距离的总和。
    
    参数：points - 点序列数组（形状为 (N, D)，D 为维度）
    返回：float - 折线的总长度。
    """
    # 计算相邻点之间的差值
    diffs = points[1:] - points[:-1]
    # 计算每个差值向量的模长并求和
    return np.sqrt((diffs**2).sum(1)).sum()

# Operations related to rotation


# ManimGL 三维旋转与向量运算工具库：提供四元数操作、旋转矩阵、角度计算等功能，
# 支持三维空间中的向量旋转、坐标系转换和角度分析，是3D动画和图形变换的核心数学支撑。


from __future__ import annotations

import math
import numpy as np
from scipy.spatial.transform import Rotation  # 旋转处理库

# 导入基础工具函数
from manimlib.constants import OUT, RIGHT, UP, DOWN  # 方向向量常量
from manimlib.utils.simple_functions import clip  # 数值裁剪
from manimlib.utils.space_ops import cross  # 向量叉积
from manimlib.utils.space_ops import get_norm  # 向量模长
from manimlib.utils.space_ops import normalize  # 向量归一化

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Tuple
    from manimlib.typing import Vect2, Vect3, Vect4, VectN, Matrix3x3  # 类型注解


# ------------------------------ 四元数运算 ------------------------------
def quaternion_mult(*quats: Vect4) -> Vect4:
    """
    四元数乘法：计算多个四元数的乘积（遵循scipy旋转 convention，实部为最后一个元素）。
    
    四元数乘法用于组合旋转：q = q1 * q2 表示先应用 q2 旋转，再应用 q1 旋转。
    
    参数：*quats - 可变数量的四元数（每个为 [x, y, z, w]，w 为实部）
    返回：Vect4 - 乘积四元数。
    """
    if len(quats) == 0:
        return np.array([0, 0, 0, 1])  # 单位四元数（无旋转）
    result = np.array(quats[0])
    for next_quat in quats[1:]:
        x1, y1, z1, w1 = result
        x2, y2, z2, w2 = next_quat
        # 四元数乘法公式
        result[:] = [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,  # x 分量
            w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2,  # y 分量
            w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2,  # z 分量
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,  # 实部 w
        ]
    return result


def quaternion_from_angle_axis(
    angle: float,
    axis: Vect3,
) -> Vect4:
    """
    从角度和轴创建四元数：根据旋转角度和旋转轴生成对应的四元数。
    
    参数：
        angle : 旋转角度（弧度）；
        axis : 旋转轴向量（3D）。
    返回：Vect4 - 四元数 [x, y, z, w]。
    """
    # 利用scipy的Rotation：旋转向量 = 角度 × 单位轴向量
    return Rotation.from_rotvec(angle * normalize(axis)).as_quat()


def angle_axis_from_quaternion(quat: Vect4) -> Tuple[float, Vect3]:
    """
    从四元数提取角度和轴：将四元数转换为对应的旋转角度和旋转轴。
    
    参数：quat - 四元数 [x, y, z, w]
    返回：Tuple[float, Vect3] - (旋转角度, 旋转轴向量)。
    """
    # 旋转向量 = 角度 × 单位轴向量
    rot_vec = Rotation.from_quat(quat).as_rotvec()
    norm = get_norm(rot_vec)
    return norm, rot_vec / norm if norm != 0 else (0.0, np.array([0, 0, 1]))


def quaternion_conjugate(quaternion: Vect4) -> Vect4:
    """
    四元数共轭：计算四元数的共轭（虚部取反），用于逆转旋转。
    
    参数：quaternion - 四元数 [x, y, z, w]
    返回：Vect4 - 共轭四元数 [-x, -y, -z, w]。
    """
    result = np.array(quaternion)
    result[:3] *= -1  # 虚部取反
    return result


# ------------------------------ 向量旋转 ------------------------------
def rotate_vector(
    vector: Vect3,
    angle: float,
    axis: Vect3 = OUT
) -> Vect3:
    """
    旋转3D向量：绕指定轴旋转向量一定角度。
    
    参数：
        vector : 待旋转的3D向量；
        angle : 旋转角度（弧度）；
        axis : 旋转轴（默认OUT，即z轴正方向）。
    返回：Vect3 - 旋转后的向量。
    """
    # 创建旋转对象：旋转向量 = 角度 × 单位轴
    rot = Rotation.from_rotvec(angle * normalize(axis))
    # 应用旋转（矩阵转置是为了与Manim的坐标变换兼容）
    return np.dot(vector, rot.as_matrix().T)


def rotate_vector_2d(vector: Vect2, angle: float) -> Vect2:
    """
    旋转2D向量：在xy平面内绕原点旋转向量（利用复数运算简化）。
    
    参数：
        vector : 待旋转的2D向量；
        angle : 旋转角度（弧度，逆时针为正）。
    返回：Vect2 - 旋转后的向量。
    """
    # 复数乘法实现旋转：z' = z * e^(iθ)
    z = complex(*vector) * np.exp(complex(0, angle))
    return np.array([z.real, z.imag])


# ------------------------------ 旋转矩阵 ------------------------------
def rotation_matrix_transpose_from_quaternion(quat: Vect4) -> Matrix3x3:
    """
    从四元数获取旋转矩阵的转置（用于Manim的坐标变换）。
    
    参数：quat - 四元数 [x, y, z, w]
    返回：Matrix3x3 - 旋转矩阵的转置。
    """
    return Rotation.from_quat(quat).as_matrix()


def rotation_matrix_from_quaternion(quat: Vect4) -> Matrix3x3:
    """
    从四元数获取旋转矩阵。
    
    参数：quat - 四元数 [x, y, z, w]
    返回：Matrix3x3 - 旋转矩阵。
    """
    return np.transpose(rotation_matrix_transpose_from_quaternion(quat))


def rotation_matrix(angle: float, axis: Vect3) -> Matrix3x3:
    """
    生成3D旋转矩阵：绕指定轴旋转一定角度的旋转矩阵。
    
    参数：
        angle : 旋转角度（弧度）；
        axis : 旋转轴（3D向量）。
    返回：Matrix3x3 - 旋转矩阵。
    """
    return Rotation.from_rotvec(angle * normalize(axis)).as_matrix()


def rotation_matrix_transpose(angle: float, axis: Vect3) -> Matrix3x3:
    """
    生成旋转矩阵的转置（用于Manim的坐标变换）。
    
    参数：
        angle : 旋转角度（弧度）；
        axis : 旋转轴（3D向量）。
    返回：Matrix3x3 - 旋转矩阵的转置。
    """
    return rotation_matrix(angle, axis).T


def rotation_about_z(angle: float) -> Matrix3x3:
    """
    绕z轴旋转的矩阵（2D旋转的3D扩展）。
    
    参数：angle - 旋转角度（弧度）
    返回：Matrix3x3 - z轴旋转矩阵。
    """
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return np.array([
        [cos_a, -sin_a, 0],  # x' = x cosθ - y sinθ
        [sin_a, cos_a, 0],   # y' = x sinθ + y cosθ
        [0, 0, 1]            # z不变
    ])


def rotation_between_vectors(v1: Vect3, v2: Vect3) -> Matrix3x3:
    """
    生成两个向量之间的旋转矩阵：将v1旋转到与v2同向的矩阵。
    
    参数：
        v1 : 起始向量；
        v2 : 目标向量。
    返回：Matrix3x3 - 旋转矩阵。
    """
    atol = 1e-8
    # 若两向量已接近，返回单位矩阵
    if get_norm(v1 - v2) < atol:
        return np.identity(3)
    # 旋转轴为两向量的叉积（垂直于两向量所在平面）
    axis = cross(v1, v2)
    # 若叉积为零（两向量共线），则选择与v1垂直的轴
    if get_norm(axis) < atol:
        axis = cross(v1, RIGHT)  # 尝试RIGHT方向
    if get_norm(axis) < atol:
        axis = cross(v1, UP)     # 再尝试UP方向
    # 旋转角度为两向量的夹角
    return rotation_matrix(
        angle=angle_between_vectors(v1, v2),
        axis=axis,
    )


def z_to_vector(vector: Vect3) -> Matrix3x3:
    """
    生成将z轴（OUT方向）旋转到目标向量方向的矩阵。
    
    参数：vector - 目标方向向量（3D）
    返回：Matrix3x3 - 旋转矩阵。
    """
    return rotation_between_vectors(OUT, vector)


# ------------------------------ 角度计算 ------------------------------
def angle_of_vector(vector: Vect2 | Vect3) -> float:
    """
    计算向量在xy平面上的极角（与x轴正方向的夹角）。
    
    参数：vector - 2D或3D向量（3D时取xy分量）
    返回：float - 极角（弧度，范围[-π, π]）。
    """
    return math.atan2(vector[1], vector[0])  # atan2(y, x)


def angle_between_vectors(v1: VectN, v2: VectN) -> float:
    """
    计算两个向量之间的夹角（范围[0, π]）。
    
    参数：
        v1 : N维向量；
        v2 : N维向量。
    返回：float - 夹角（弧度）。
    """
    n1 = get_norm(v1)
    n2 = get_norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0  # 零向量夹角为0
    # 点积公式：v1·v2 = |v1||v2|cosθ → cosθ = 点积 / (|v1||v2|)
    cos_angle = np.dot(v1, v2) / np.float64(n1 * n2)
    # 裁剪到[-1, 1]避免数值误差导致的acos错误
    return math.acos(clip(cos_angle, -1, 1))


# ------------------------------ 向量投影与归一化 ------------------------------
def project_along_vector(point: Vect3, vector: Vect3) -> Vect3:
    """
    将点沿指定向量方向投影到垂直于该向量的平面上。
    
    参数：
        point : 3D点；
        vector : 投影方向向量（平面法向量）。
    返回：Vect3 - 投影后的点。
    """
    # 投影矩阵：I - vv^T（v为单位向量）
    matrix = np.identity(3) - np.outer(vector, vector)
    return np.dot(point, matrix.T)


def normalize_along_axis(
    array: np.ndarray,
    axis: int,
) -> np.ndarray:
    """
    沿指定轴归一化数组（使每个子数组的模长为1）。
    
    参数：
        array : 多维数组；
        axis : 要归一化的轴。
    返回：np.ndarray - 归一化后的数组。
    """
    norms = np.sqrt((array * array).sum(axis))
    norms[norms == 0] = 1  # 避免除以零
    return array / norms[:, np.newaxis]


def get_unit_normal(
    v1: Vect3,
    v2: Vect3,
    tol: float = 1e-6
) -> Vect3:
    """
    计算两个向量所在平面的单位法向量（垂直于v1和v2）。
    
    参数：
        v1, v2 : 3D向量；
        tol : 数值容差（判断向量是否共线）。
    返回：Vect3 - 单位法向量。
    """
    v1 = normalize(v1)
    v2 = normalize(v2)
    cp = cross(v1, v2)  # 叉积即法向量
    cp_norm = get_norm(cp)
    if cp_norm < tol:
        # 两向量共线，在与z轴形成的平面中找法向量
        new_cp = cross(cross(v1, OUT), v1)
        new_cp_norm = get_norm(new_cp)
        if new_cp_norm < tol:
            return DOWN  # 最终 fallback
        return new_cp / new_cp_norm
    return cp / cp_norm


###


# ManimGL 几何工具库：提供对角线矩阵、方向向量、坐标转换、交点计算等几何操作，
# 支持2D/3D空间中的点、线、路径关系分析，是图形绘制和碰撞检测的基础工具。


from __future__ import annotations

import numpy as np

# 导入基础工具函数和常量
from manimlib.constants import TAU, PI  # 圆周率常量（TAU=2π）
from manimlib.utils.iterables import adjacent_pairs  # 相邻元素对生成
from manimlib.utils.space_ops import cross2d  # 2D叉积
from manimlib.utils.space_ops import cross  # 3D叉积
from manimlib.utils.space_ops import rotate_vector  # 向量旋转
from manimlib.utils.space_ops import angle_of_vector  # 向量极角

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Callable, Sequence, Tuple
    from manimlib.typing import Vect2, Vect3, VectN, Vect2Array, Vect3Array  # 类型注解


# ------------------------------ 特殊矩阵生成 ------------------------------
def thick_diagonal(dim: int, thickness: int = 2) -> np.ndarray:
    """
    生成带"粗对角线"的矩阵：对角线及附近区域为1，其余为0，用于掩码或权重矩阵。
    
    参数：
        dim : 矩阵维度（dim×dim）；
        thickness : 对角线厚度（默认2，即主对角线±1范围内为1）。
    返回：np.ndarray - 二进制矩阵（0或1）。
    """
    # 生成行索引和列索引矩阵
    row_indices = np.arange(dim).repeat(dim).reshape((dim, dim))
    col_indices = np.transpose(row_indices)
    # 行索引与列索引的差的绝对值小于厚度的位置为1
    return (np.abs(row_indices - col_indices) < thickness).astype('uint8')


# ------------------------------ 方向向量生成 ------------------------------
def compass_directions(n: int = 4, start_vect: Vect3 = RIGHT) -> Vect3Array:
    """
    生成均匀分布的方向向量（如指南针方向），沿圆周等角度分布。
    
    参数：
        n : 方向数量（默认4，即上下左右）；
        start_vect : 起始方向（默认RIGHT，即x轴正方向）。
    返回：Vect3Array - 形状为(n, 3)的方向向量数组。
    示例：n=4 → [RIGHT, UP, LEFT, DOWN]
    """
    angle = TAU / n  # 每个方向的角度间隔
    return np.array([
        rotate_vector(start_vect, k * angle)  # 旋转起始向量得到每个方向
        for k in range(n)
    ])


# ------------------------------ 复数与3D坐标转换 ------------------------------
def complex_to_R3(complex_num: complex) -> Vect3:
    """
    将复数转换为3D坐标（z分量为0）。
    
    参数：complex_num - 复数（实部为x，虚部为y）
    返回：Vect3 - (real, imag, 0)
    """
    return np.array((complex_num.real, complex_num.imag, 0))


def R3_to_complex(point: Vect3) -> complex:
    """
    将3D坐标转换为复数（取x和y分量）。
    
    参数：point - 3D点 (x, y, z)
    返回：complex - x + y*i
    """
    return complex(*point[:2])  # 取前两个分量作为实部和虚部


def complex_func_to_R3_func(complex_func: Callable[[complex], complex]) -> Callable[[Vect3], Vect3]:
    """
    将复数函数转换为3D坐标函数（仅作用于x-y平面）。
    
    参数：complex_func - 输入输出均为复数的函数
    返回：Callable[[Vect3], Vect3] - 接收3D点，返回转换后的3D点。
    示例：f(z) = z^2 → 转换后函数将(x,y,0)映射为(x²-y², 2xy, 0)
    """
    def result(p: Vect3):
        # 将3D点转为复数→应用函数→转回3D点
        return complex_to_R3(complex_func(R3_to_complex(p)))
    return result


# ------------------------------ 质心与中点计算 ------------------------------
def center_of_mass(points: Sequence[Vect3]) -> Vect3:
    """
    计算点集的质心（重心），即所有点的平均坐标。
    
    参数：points - 3D点序列
    返回：Vect3 - 质心坐标。
    """
    return np.array(points).sum(0) / len(points)  # 按列求和再除以点数


def midpoint(point1: VectN, point2: VectN) -> VectN:
    """
    计算两点的中点（质心的特例，仅两个点）。
    
    参数：
        point1 : N维点；
        point2 : N维点。
    返回：VectN - 中点坐标。
    """
    return center_of_mass([point1, point2])  # 复用质心计算


# ------------------------------ 直线交点计算 ------------------------------
def line_intersection(
    line1: Tuple[Vect3, Vect3],
    line2: Tuple[Vect3, Vect3]
) -> Vect3:
    """
    计算两条2D直线的交点（假设在同一平面z=0）。
    
    参数：
        line1 : 第一条直线的两个端点；
        line2 : 第二条直线的两个端点。
    返回：Vect3 - 交点坐标（z=0）。
    异常：直线平行时抛出异常。
    """
    # 提取x和y方向的差分
    x_diff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    y_diff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    # 2D行列式计算
    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    # 计算分母（判断是否平行）
    div = det(x_diff, y_diff)
    if div == 0:
        raise Exception("Lines do not intersect")  # 平行或重合

    # 计算分子（Cramer法则）
    d = (det(*line1), det(*line2))
    x = det(d, x_diff) / div
    y = det(d, y_diff) / div
    return np.array([x, y, 0])


def find_intersection(
    p0: Vect3 | Vect3Array,
    v0: Vect3 | Vect3Array,
    p1: Vect3 | Vect3Array,
    v1: Vect3 | Vect3Array,
    threshold: float = 1e-5,
) -> Vect3 | Vect3Array:
    """
    计算两条射线（或直线）的交点（3D中为最近点）。
    
    射线定义：
    - 射线1：p0 + t*v0（t≥0）
    - 射线2：p1 + s*v1（s≥0）
    
    参数：
        p0, p1 : 射线起点；
        v0, v1 : 射线方向向量；
        threshold : 数值容差（避免除以零）。
    返回：
        2D：精确交点；
        3D：两条射线的最近点（若不共面）。
    """
    d = len(p0.shape)  # 维度（1为单个向量，2为数组）
    # 判断是否为3D（z分量不全为0）
    if d == 1:
        is_3d = any(arr[2] for arr in (p0, v0, p1, v1))
    else:
        is_3d = any(z for arr in (p0, v0, p1, v1) for z in arr.T[2])

    if not is_3d:
        # 2D情况：用叉积计算交点
        numer = np.array(cross2d(v1, p1 - p0))
        denom = np.array(cross2d(v1, v0))
    else:
        # 3D情况：计算最近点（基于向量投影）
        cp1 = cross(v1, p1 - p0)
        cp2 = cross(v1, v0)
        numer = np.array((cp1 * cp1).sum(d - 1))  # 分子为点积和
        denom = np.array((cp1 * cp2).sum(d - 1))   # 分母为点积和

    # 处理接近零的分母（避免无穷大）
    denom[abs(denom) < threshold] = np.inf
    ratio = numer / denom  # 计算参数t
    return p0 + (ratio * v0.T).T  # 计算交点


# ------------------------------ 路径相交检测 ------------------------------
def line_intersects_path(
    start: Vect2 | Vect3,
    end: Vect2 | Vect3,
    path: Vect2Array | Vect3Array,
) -> bool:
    """
    判断直线段（start, end）是否与多边形路径相交。
    
    参数：
        start, end : 直线段的端点；
        path : 多边形路径的顶点数组（按顺序连接）。
    返回：bool - 若相交则返回True，否则False。
    """
    n = len(path) - 1  # 路径的线段数量
    # 初始化线段端点数组（批量处理）
    p1 = np.empty((n, 2))
    q1 = np.empty((n, 2))
    p1[:] = start[:2]  # 直线段的起点（复制n次）
    q1[:] = end[:2]    # 直线段的终点（复制n次）
    p2 = path[:-1, :2]  # 路径各线段的起点
    q2 = path[1:, :2]   # 路径各线段的终点

    # 计算方向向量
    v1 = q1 - p1
    v2 = q2 - p2

    # 检测相交（基于叉积符号判断）
    # mis1：直线段与路径线段的起点/终点在v1两侧
    mis1 = cross2d(v1, p2 - p1) * cross2d(v1, q2 - p1) < 0
    # mis2：路径线段与直线段的起点/终点在v2两侧
    mis2 = cross2d(v2, p1 - p2) * cross2d(v2, q1 - p2) < 0
    # 两者均为True时表示相交
    return bool((mis1 * mis2).any())


# ------------------------------ 直线上最近点 ------------------------------
def get_closest_point_on_line(a: VectN, b: VectN, p: VectN) -> VectN:
    """
    计算点p在直线ab上的最近点（若超出线段ab，则返回最近的端点）。
    
    参数：
        a, b : 直线的两个端点；
        p : 待投影的点。
    返回：VectN - 直线上的最近点。
    """
    # 计算参数t：t=0→b，t=1→a，t∈[0,1]为线段ab上的点
    t = np.dot(p - b, a - b) / np.dot(a - b, a - b)
    # 裁剪t到[0,1]范围（超出线段则取端点）
    t = np.clip(t, 0, 1)
    # 计算最近点
    return t * a + (1 - t) * b


# ------------------------------ 绕数计算 ------------------------------
def get_winding_number(points: Sequence[Vect2 | Vect3]) -> float:
    """
    计算多边形的绕数（winding number）：描述多边形环绕原点的次数，
    用于判断点是否在多边形内部（绕数非零则在内部）。
    
    参数：points - 多边形顶点序列（闭合或非闭合）
    返回：float - 绕数（整数，正数为逆时针，负数为顺时针）。
    """
    total_angle = 0.0
    # 累加相邻顶点的角度差
    for p1, p2 in adjacent_pairs(points):
        d_angle = angle_of_vector(p2) - angle_of_vector(p1)
        # 归一化角度差到[-π, π]
        d_angle = ((d_angle + PI) % TAU) - PI
        total_angle += d_angle
    # 绕数 = 总角度 / TAU（2π）
    return total_angle / TAU


##

def cross2d(a: Vect2 | Vect2Array, b: Vect2 | Vect2Array) -> Vect2 | Vect2Array:
    if len(a.shape) == 2:
        return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    else:
        return a[0] * b[1] - b[0] * a[1]


def tri_area(
    a: Vect2,
    b: Vect2,
    c: Vect2
) -> float:
    return 0.5 * abs(
        a[0] * (b[1] - c[1]) +
        b[0] * (c[1] - a[1]) +
        c[0] * (a[1] - b[1])
    )


def is_inside_triangle(
    p: Vect2,
    a: Vect2,
    b: Vect2,
    c: Vect2
) -> bool:
    """
    Test if point p is inside triangle abc
    """
    crosses = np.array([
        cross2d(p - a, b - p),
        cross2d(p - b, c - p),
        cross2d(p - c, a - p),
    ])
    return bool(np.all(crosses > 0) or np.all(crosses < 0))


def norm_squared(v: VectN | List[float]) -> float:
    return sum(x * x for x in v)


# TODO, fails for polygons drawn over themselves
def earclip_triangulation(verts: Vect3Array | Vect2Array, ring_ends: list[int]) -> list[int]:
    """
    Returns a list of indices giving a triangulation
    of a polygon, potentially with holes

    - verts is a numpy array of points

    - ring_ends is a list of indices indicating where
    the ends of new paths are
    """

    rings = [
        list(range(e0, e1))
        for e0, e1 in zip([0, *ring_ends], ring_ends)
    ]
    epsilon = 1e-6

    def is_in(point, ring_id):
        return abs(abs(get_winding_number([i - point for i in verts[rings[ring_id]]])) - 1) < epsilon

    def ring_area(ring_id):
        ring = rings[ring_id]
        s = 0
        for i, j in zip(ring[1:], ring):
            s += cross2d(verts[i], verts[j])
        return abs(s) / 2

    # Points at the same position may cause problems
    for i in rings:
        if len(i) < 2:
            continue
        verts[i[0]] += (verts[i[1]] - verts[i[0]]) * epsilon
        verts[i[-1]] += (verts[i[-2]] - verts[i[-1]]) * epsilon

    # First, we should know which rings are directly contained in it for each ring

    right = [max(verts[rings[i], 0]) for i in range(len(rings))]
    left = [min(verts[rings[i], 0]) for i in range(len(rings))]
    top = [max(verts[rings[i], 1]) for i in range(len(rings))]
    bottom = [min(verts[rings[i], 1]) for i in range(len(rings))]
    area = [ring_area(i) for i in range(len(rings))]

    # The larger ring must be outside
    rings_sorted = list(range(len(rings)))
    rings_sorted.sort(key=lambda x: area[x], reverse=True)

    def is_in_fast(ring_a, ring_b):
        # Whether a is in b
        return reduce(op.and_, (
            left[ring_b] <= left[ring_a] <= right[ring_a] <= right[ring_b],
            bottom[ring_b] <= bottom[ring_a] <= top[ring_a] <= top[ring_b],
            is_in(verts[rings[ring_a][0]], ring_b)
        ))

    chilren = [[] for i in rings]
    ringenum = ProgressDisplay(
        enumerate(rings_sorted),
        total=len(rings),
        leave=False,
        ascii=True if platform.system() == 'Windows' else None,
        dynamic_ncols=True,
        desc="SVG Triangulation",
        delay=3,
    )
    for idx, i in ringenum:
        for j in rings_sorted[:idx][::-1]:
            if is_in_fast(i, j):
                chilren[j].append(i)
                break

    res = []

    # Then, we can use earcut for each part
    used = [False] * len(rings)
    for i in rings_sorted:
        if used[i]:
            continue
        v = rings[i]
        ring_ends = [len(v)]
        for j in chilren[i]:
            used[j] = True
            v += rings[j]
            ring_ends.append(len(v))
        res += [v[i] for i in earcut(verts[v, :2], ring_ends)]

    return res
