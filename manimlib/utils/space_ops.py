# 从未来版本导入类型注解支持
from __future__ import annotations

# 导入所需模块和函数
from functools import reduce
import math
import operator as op
import platform

# 导入mapbox_earcut的三角形化函数
from mapbox_earcut import triangulate_float32 as earcut
import numpy as np
# 导入scipy的旋转变换模块
from scipy.spatial.transform import Rotation
# 导入进度条显示工具
from tqdm.auto import tqdm as ProgressDisplay

# 从manimlib导入常量和工具函数
from manimlib.constants import DOWN, OUT, RIGHT, UP
from manimlib.constants import PI, TAU
from manimlib.utils.iterables import adjacent_pairs
from manimlib.utils.simple_functions import clip

# 导入类型检查相关模块
from typing import TYPE_CHECKING

# 条件导入类型定义（仅在类型检查时生效）
if TYPE_CHECKING:
    from typing import Callable, Sequence, List, Tuple
    from manimlib.typing import Vect2, Vect3, Vect4, VectN, Matrix3x3, Vect3Array, Vect2Array


def cross(
    v1: Vect3 | List[float],
    v2: Vect3 | List[float],
    out: np.ndarray | None = None
) -> Vect3 | Vect3Array:
    # 判断输入是否为二维数组（向量数组）
    is2d = isinstance(v1, np.ndarray) and len(v1.shape) == 2
    if is2d:
        # 提取数组中每个向量的x、y、z分量
        x1, y1, z1 = v1[:, 0], v1[:, 1], v1[:, 2]
        x2, y2, z2 = v2[:, 0], v2[:, 1], v2[:, 2]
    else:
        # 提取单个向量的x、y、z分量
        x1, y1, z1 = v1
        x2, y2, z2 = v2
    # 若未指定输出数组则创建新数组
    if out is None:
        out = np.empty(np.shape(v1))
    # 计算叉积并存储到输出数组（叉积公式：v1 × v2 = (y1z2-z1y2, z1x2-x1z2, x1y2-y1x2)）
    out.T[:] = [
        y1 * z2 - z1 * y2,
        z1 * x2 - x1 * z2,
        x1 * y2 - y1 * x2,
    ]
    return out


def get_norm(vect: VectN | List[float]) -> float:
    # 计算向量的模长（L2范数）：各分量平方和的平方根
    return sum((x**2 for x in vect))**0.5


def get_dist(vect1: VectN, vect2: VectN):
    # 计算两个向量之间的距离：向量差的模长
    return get_norm(vect2 - vect1)


def normalize(
    vect: VectN | List[float],
    fall_back: VectN | List[float] | None = None
) -> VectN:
    # 计算向量的模长
    norm = get_norm(vect)
    # 若模长大于0，则返回单位向量
    if norm > 0:
        return np.array(vect) / norm
    # 若模长为0且指定了 fallback 向量，则返回fallback向量
    elif fall_back is not None:
        return np.array(fall_back)
    # 否则返回零向量
    else:
        return np.zeros(len(vect))


def poly_line_length(points):
    """
    返回相邻点之间距离的总和（折线长度）
    """
    # 计算相邻点之间的差值
    diffs = points[1:] - points[:-1]
    # 计算每个差值的模长并求和
    return np.sqrt((diffs**2).sum(1)).sum()


# 与旋转相关的操作

def quaternion_mult(*quats: Vect4) -> Vect4:
    """
    输入被视为四元数，其中实部是最后一个元素，
    以遵循scipy的Rotation约定。
    """
    # 若没有输入四元数，返回单位四元数（表示无旋转）
    if len(quats) == 0:
        return np.array([0, 0, 0, 1])
    # 初始化结果为第一个四元数
    result = np.array(quats[0])
    # 依次与后续四元数相乘
    for next_quat in quats[1:]:
        x1, y1, z1, w1 = result
        x2, y2, z2, w2 = next_quat
        # 四元数乘法公式
        result[:] = [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2,
            w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    return result


def quaternion_from_angle_axis(
    angle: float,
    axis: Vect3,
) -> Vect4:
    # 从旋转角度和轴创建四元数：使用scipy的Rotation，先创建旋转向量再转为四元数
    return Rotation.from_rotvec(angle * normalize(axis)).as_quat()


def angle_axis_from_quaternion(quat: Vect4) -> Tuple[float, Vect3]:
    # 从四元数提取旋转角度和轴：先转为旋转向量，再计算模长（角度）和单位向量（轴）
    rot_vec = Rotation.from_quat(quat).as_rotvec()
    norm = get_norm(rot_vec)
    return norm, rot_vec / norm


def quaternion_conjugate(quaternion: Vect4) -> Vect4:
    # 计算四元数的共轭：虚部取反，实部不变
    result = np.array(quaternion)
    result[:3] *= -1
    return result


def rotate_vector(
    vector: Vect3,
    angle: float,
    axis: Vect3 = OUT
) -> Vect3:
    # 创建旋转对象：绕指定轴旋转指定角度
    rot = Rotation.from_rotvec(angle * normalize(axis))
    # 应用旋转：向量与旋转矩阵的转置相乘（等价于旋转向量）
    return np.dot(vector, rot.as_matrix().T)


def rotate_vector_2d(vector: Vect2, angle: float) -> Vect2:
    # 使用复数实现2D向量旋转：复数乘法等价于旋转
    z = complex(*vector) * np.exp(complex(0, angle))
    return np.array([z.real, z.imag])


def rotation_matrix_transpose_from_quaternion(quat: Vect4) -> Matrix3x3:
    # 从四元数获取旋转矩阵的转置（即旋转矩阵本身，scipy的as_matrix返回旋转矩阵）
    return Rotation.from_quat(quat).as_matrix()


def rotation_matrix_from_quaternion(quat: Vect4) -> Matrix3x3:
    # 从四元数获取旋转矩阵（通过转置其转置矩阵）
    return np.transpose(rotation_matrix_transpose_from_quaternion(quat))


def rotation_matrix(angle: float, axis: Vect3) -> Matrix3x3:
    """
    R^3中绕指定旋转轴的旋转矩阵
    """
    # 创建旋转矩阵：从旋转向量（角度×单位轴）转换而来
    return Rotation.from_rotvec(angle * normalize(axis)).as_matrix()


def rotation_matrix_transpose(angle: float, axis: Vect3) -> Matrix3x3:
    # 旋转矩阵的转置（等于其逆矩阵，因为旋转矩阵是正交矩阵）
    return rotation_matrix(angle, axis).T


def rotation_about_z(angle: float) -> Matrix3x3:
    # 计算绕z轴旋转的矩阵
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])


def rotation_between_vectors(v1: Vect3, v2: Vect3) -> Matrix3x3:
    # 容忍误差
    atol = 1e-8
    # 若两向量几乎相同，返回单位矩阵
    if get_norm(v1 - v2) < atol:
        return np.identity(3)
    # 计算两向量的叉积作为旋转轴
    axis = cross(v1, v2)
    # 若叉积模长接近0（向量共线），则尝试其他轴
    if get_norm(axis) < atol:
        axis = cross(v1, RIGHT)
    if get_norm(axis) < atol:
        axis = cross(v1, UP)
    # 创建从v1旋转到v2的旋转矩阵
    return rotation_matrix(
        angle=angle_between_vectors(v1, v2),
        axis=axis,
    )


def z_to_vector(vector: Vect3) -> Matrix3x3:
    # 生成将z轴（OUT）旋转到指定向量方向的矩阵
    return rotation_between_vectors(OUT, vector)


def angle_of_vector(vector: Vect2 | Vect3) -> float:
    """
    返回向量在xy平面上投影的极坐标角度theta
    """
    # 计算向量的辐角（与x轴正方向的夹角）
    return math.atan2(vector[1], vector[0])


def angle_between_vectors(v1: VectN, v2: VectN) -> float:
    """
    返回两个3D向量之间的角度，该角度始终在0到pi之间
    """
    # 计算两向量的模长
    n1 = get_norm(v1)
    n2 = get_norm(v2)
    # 若任一向量为零向量，返回0
    if n1 == 0 or n2 == 0:
        return 0
    # 计算夹角的余弦值（点积除以模长乘积）
    cos_angle = np.dot(v1, v2) / np.float64(n1 * n2)
    # 确保余弦值在[-1,1]范围内，再计算反余弦
    return math.acos(clip(cos_angle, -1, 1))


def project_along_vector(point: Vect3, vector: Vect3) -> Vect3:
    # 计算沿向量方向的投影矩阵（单位矩阵减去外积）
    matrix = np.identity(3) - np.outer(vector, vector)
    # 应用投影矩阵
    return np.dot(point, matrix.T)


def normalize_along_axis(
    array: np.ndarray,
    axis: int,
) -> np.ndarray:
    # 沿指定轴计算数组的模长
    norms = np.sqrt((array * array).sum(axis))
    # 避免除零错误（模长为0时设为1）
    norms[norms == 0] = 1
    # 沿轴归一化
    return array / norms[:, np.newaxis]


def get_unit_normal(
    v1: Vect3,
    v2: Vect3,
    tol: float = 1e-6
) -> Vect3:
    # 归一化输入向量
    v1 = normalize(v1)
    v2 = normalize(v2)
    # 计算叉积（法向量）
    cp = cross(v1, v2)
    cp_norm = get_norm(cp)
    # 若叉积模长过小（向量接近共线）
    if cp_norm < tol:
        # 计算在与z轴共享的平面内与v1垂直的向量
        new_cp = cross(cross(v1, OUT), v1)
        new_cp_norm = get_norm(new_cp)
        if new_cp_norm < tol:
            return DOWN
        return new_cp / new_cp_norm
    # 返回单位法向量
    return cp / cp_norm


###


def thick_diagonal(dim: int, thickness: int = 2) -> np.ndarray:
    # 生成行索引矩阵：0到dim-1重复dim次后重塑为dim×dim矩阵
    row_indices = np.arange(dim).repeat(dim).reshape((dim, dim))
    # 生成列索引矩阵：行索引矩阵的转置
    col_indices = np.transpose(row_indices)
    # 生成厚对角线矩阵：行列索引差的绝对值小于厚度时为1，否则为0（uint8类型）
    return (np.abs(row_indices - col_indices) < thickness).astype('uint8')


def compass_directions(n: int = 4, start_vect: Vect3 = RIGHT) -> Vect3:
    # 计算每个方向的角度间隔（圆周角除以方向数）
    angle = TAU / n
    # 生成n个方向向量：从起始向量开始，每次旋转angle角度
    return np.array([
        rotate_vector(start_vect, k * angle)
        for k in range(n)
    ])


def complex_to_R3(complex_num: complex) -> Vect3:
    # 复数转三维向量：实部为x，虚部为y，z为0
    return np.array((complex_num.real, complex_num.imag, 0))


def R3_to_complex(point: Vect3) -> complex:
    # 三维向量转复数：x为实部，y为虚部（忽略z）
    return complex(*point[:2])


def complex_func_to_R3_func(complex_func: Callable[[complex], complex]) -> Callable[[Vect3], Vect3]:
    # 定义转换后的三维向量函数
    def result(p: Vect3):
        # 将三维向量转为复数，应用复数函数，再转回三维向量
        return complex_to_R3(complex_func(R3_to_complex(p)))
    return result


def center_of_mass(points: Sequence[Vect3]) -> Vect3:
    # 计算质心：所有点坐标求和后除以点的数量
    return np.array(points).sum(0) / len(points)


def midpoint(point1: VectN, point2: VectN) -> VectN:
    # 计算中点：两点的质心
    return center_of_mass([point1, point2])


def line_intersection(
    line1: Tuple[Vect3, Vect3],
    line2: Tuple[Vect3, Vect3]
) -> Vect3:
    """
    返回两条直线的交点，每条直线由一对端点定义
    """
    # 计算两条直线在x方向上的差值（x1-x2）
    x_diff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    # 计算两条直线在y方向上的差值（y1-y2）
    y_diff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    # 定义2x2矩阵的行列式计算函数
    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    # 计算分母（两条直线方向向量的行列式）
    div = det(x_diff, y_diff)
    # 若分母为0，直线平行或重合，抛出异常
    if div == 0:
        raise Exception("Lines do not intersect")
    # 计算分子部分的行列式
    d = (det(*line1), det(*line2))
    # 计算交点x坐标
    x = det(d, x_diff) / div
    # 计算交点y坐标
    y = det(d, y_diff) / div
    # 返回三维交点（z=0）
    return np.array([x, y, 0])


def find_intersection(
    p0: Vect3 | Vect3Array,
    v0: Vect3 | Vect3Array,
    p1: Vect3 | Vect3Array,
    v1: Vect3 | Vect3Array,
    threshold: float = 1e-5,
) -> Vect3:
    """
    返回通过p0沿v0方向的直线与通过p1沿v1方向的直线的交点。
    （或从这类点/方向的数组中返回交点数组）。
    对于3D值，返回射线p0 + v0 * t上最接近射线p1 + v1 * t的点
    """
    # 获取输入数组的维度（1表示单个向量，2表示向量数组）
    d = len(p0.shape)
    # 判断是否为3D场景：检查z分量是否有非零值
    if d == 1:
        is_3d = any(arr[2] for arr in (p0, v0, p1, v1))
    else:
        is_3d = any(z for arr in (p0, v0, p1, v1) for z in arr.T[2])
    
    # 2D场景下的计算（使用二维叉积）
    if not is_3d:
        numer = np.array(cross2d(v1, p1 - p0))
        denom = np.array(cross2d(v1, v0))
    # 3D场景下的计算（使用三维叉积和点积）
    else:
        cp1 = cross(v1, p1 - p0)
        cp2 = cross(v1, v0)
        numer = np.array((cp1 * cp1).sum(d - 1))
        denom = np.array((cp1 * cp2).sum(d - 1))
    
    # 处理接近零的分母（避免除零错误）
    denom[abs(denom) < threshold] = np.inf
    # 计算比例系数
    ratio = numer / denom
    # 计算并返回交点
    return p0 + (ratio * v0.T).T


def line_intersects_path(
    start: Vect2 | Vect3,
    end: Vect2 | Vect3,
    path: Vect2Array | Vect3Array,
) -> bool:
    """
    测试直线（start, end）是否与由顶点定义的多边形路径相交
    """
    # 路径段数量 = 顶点数 - 1
    n = len(path) - 1
    # 创建存储直线端点的数组（与路径段数量匹配）
    p1 = np.empty((n, 2))
    q1 = np.empty((n, 2))
    p1[:] = start[:2]  # 填充直线起点（x,y分量）
    q1[:] = end[:2]    # 填充直线终点（x,y分量）
    
    # 提取路径的连续顶点对（每段路径的起点和终点）
    p2 = path[:-1, :2]
    q2 = path[1:, :2]

    # 计算两条直线的方向向量
    v1 = q1 - p1
    v2 = q2 - p2

    # 判断路径段的两个端点是否在测试直线的两侧
    mis1 = cross2d(v1, p2 - p1) * cross2d(v1, q2 - p1) < 0
    # 判断测试直线的两个端点是否在路径段的两侧
    mis2 = cross2d(v2, p1 - p2) * cross2d(v2, q1 - p2) < 0
    # 若存在同时满足的线段，则相交
    return bool((mis1 * mis2).any())


def get_closest_point_on_line(a: VectN, b: VectN, p: VectN) -> VectN:
    """
    返回点x，使得x在直线ab上且xp垂直于ab。
    若x超出线段ab，则返回最近的端点（a或b）
    """
    # 计算参数t：t=0时x=b，t=1时x=a，t∈[0,1]时x在线段ab上
    t = np.dot(p - b, a - b) / np.dot(a - b, a - b)
    # 限制t在[0,1]范围内
    if t < 0:
        t = 0
    if t > 1:
        t = 1
    # 计算直线上的最近点
    return ((t * a) + ((1 - t) * b))


def get_winding_number(points: Sequence[Vect2 | Vect3]) -> float:
    # 初始化总角度
    total_angle = 0
    # 遍历相邻点对
    for p1, p2 in adjacent_pairs(points):
        # 计算角度差
        d_angle = angle_of_vector(p2) - angle_of_vector(p1)
        # 标准化角度差到[-π, π]
        d_angle = ((d_angle + PI) % TAU) - PI
        # 累加角度差
        total_angle += d_angle
    # 缠绕数 = 总角度 / 圆周角
    return total_angle / TAU


##

def cross2d(a: Vect2 | Vect2Array, b: Vect2 | Vect2Array) -> Vect2 | Vect2Array:
    # 计算二维叉积（标量，相当于z分量）
    if len(a.shape) == 2:
        # 数组情况：逐个计算叉积
        return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    else:
        # 单个向量情况
        return a[0] * b[1] - b[0] * a[1]


def tri_area(
    a: Vect2,
    b: Vect2,
    c: Vect2
) -> float:
    # 计算三角形面积（基于 shoelace 公式的简化）
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
    测试点p是否在三角形abc内部
    """
    # 计算点p与三角形各边的叉积
    crosses = np.array([
        cross2d(p - a, b - p),
        cross2d(p - b, c - p),
        cross2d(p - c, a - p),
    ])
    # 若所有叉积同号（均正或均负），则点在内部
    return bool(np.all(crosses > 0) or np.all(crosses < 0))


def norm_squared(v: VectN | List[float]) -> float:
    # 计算向量模长的平方（避免开方运算，提高效率）
    return sum(x * x for x in v)


# TODO: 对于自相交多边形可能失败
def earclip_triangulation(verts: Vect3Array | Vect2Array, ring_ends: list[int]) -> list[int]:
    """
    返回多边形（可能包含孔洞）的三角形化索引列表

    - verts 是点的numpy数组

    - ring_ends 是指示新路径结束位置的索引列表
    """

    # 将顶点分组为环（每个环是一个路径）
    rings = [
        list(range(e0, e1))
        for e0, e1 in zip([0, *ring_ends], ring_ends)
    ]
    epsilon = 1e-6  # 微小量，用于处理重合点

    # 判断点是否在指定环内部（基于缠绕数）
    def is_in(point, ring_id):
        return abs(abs(get_winding_number([i - point for i in verts[rings[ring_id]]])) - 1) < epsilon

    # 计算环的面积
    def ring_area(ring_id):
        ring = rings[ring_id]
        s = 0
        for i, j in zip(ring[1:], ring):
            s += cross2d(verts[i], verts[j])
        return abs(s) / 2

    # 处理可能导致问题的重合点（轻微偏移端点）
    for i in rings:
        if len(i) < 2:
            continue
        verts[i[0]] += (verts[i[1]] - verts[i[0]]) * epsilon
        verts[i[-1]] += (verts[i[-2]] - verts[i[-1]]) * epsilon

    # 计算每个环的边界框和面积（用于快速判断包含关系）
    right = [max(verts[rings[i], 0]) for i in range(len(rings))]
    left = [min(verts[rings[i], 0]) for i in range(len(rings))]
    top = [max(verts[rings[i], 1]) for i in range(len(rings))]
    bottom = [min(verts[rings[i], 1]) for i in range(len(rings))]
    area = [ring_area(i) for i in range(len(rings))]

    # 按面积降序排序环（大面积环更可能是外部环）
    rings_sorted = list(range(len(rings)))
    rings_sorted.sort(key=lambda x: area[x], reverse=True)

    # 快速判断环a是否包含在环b内部
    def is_in_fast(ring_a, ring_b):
        return reduce(op.and_, (
            left[ring_b] <= left[ring_a] <= right[ring_a] <= right[ring_b],
            bottom[ring_b] <= bottom[ring_a] <= top[ring_a] <= top[ring_b],
            is_in(verts[rings[ring_a][0]], ring_b)
        ))

    # 构建环的包含关系（子环列表）
    chilren = [[] for i in rings]
    # 创建进度条显示
    ringenum = ProgressDisplay(
        enumerate(rings_sorted),
        total=len(rings),
        leave=False,
        ascii=True if platform.system() == 'Windows' else None,
        dynamic_ncols=True,
        desc="SVG Triangulation",
        delay=3,
    )
    # 为每个环找到其父环（直接包含它的环）
    for idx, i in ringenum:
        for j in rings_sorted[:idx][::-1]:
            if is_in_fast(i, j):
                chilren[j].append(i)
                break

    res = []

    # 对每个外部环及其子环（孔洞）进行三角形化
    used = [False] * len(rings)
    for i in rings_sorted:
        if used[i]:
            continue
        v = rings[i]
        ring_ends = [len(v)]
        # 合并子环（孔洞）到当前环
        for j in chilren[i]:
            used[j] = True
            v += rings[j]
            ring_ends.append(len(v))
        # 使用earcut进行三角形化，并收集结果
        res += [v[i] for i in earcut(verts[v, :2], ring_ends)]

    return res