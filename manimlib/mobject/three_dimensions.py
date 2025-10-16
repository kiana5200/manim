# 从__future__导入annotations，用于支持Python 3.7及以下版本的类型提示语法
from __future__ import annotations

# 导入math模块，用于数学计算
import math

# 导入numpy库，用于数值计算和数组操作
import numpy as np

# 从manimlib.constants导入常用颜色常量
from manimlib.constants import BLUE, BLUE_D, BLUE_E, GREY_A, BLACK
# 从manimlib.constants导入方向和位置常量
from manimlib.constants import IN, ORIGIN, OUT, RIGHT
# 从manimlib.constants导入数学常量（圆周率相关）
from manimlib.constants import PI, TAU
# 从manimlib.mobject导入Mobject基类，所有可显示对象的父类
from manimlib.mobject.mobject import Mobject
# 从manimlib.mobject.types.surface导入表面相关的类
from manimlib.mobject.types.surface import SGroup
from manimlib.mobject.types.surface import Surface
# 从manimlib.mobject.types.vectorized_mobject导入向量图形相关的组合类
from manimlib.mobject.types.vectorized_mobject import VGroup
from manimlib.mobject.types.vectorized_mobject import VMobject
# 从manimlib.mobject.geometry导入几何图形类
from manimlib.mobject.geometry import Polygon
from manimlib.mobject.geometry import Square
# 从manimlib.utils.bezier导入贝塞尔曲线插值函数
from manimlib.utils.bezier import interpolate
# 从manimlib.utils.iterables导入用于处理可迭代对象的函数
from manimlib.utils.iterables import adjacent_pairs
# 从manimlib.utils.space_ops导入空间操作相关的函数
from manimlib.utils.space_ops import compass_directions
from manimlib.utils.space_ops import get_norm
from manimlib.utils.space_ops import z_to_vector

# 导入typing模块中的TYPE_CHECKING常量，用于条件类型检查
from typing import TYPE_CHECKING
# 如果是类型检查阶段（非运行时），导入所需的类型提示
if TYPE_CHECKING:
    from typing import Tuple, TypeVar
    from manimlib.typing import ManimColor, Vect3, Sequence

    # 定义类型变量T，限定为Mobject的子类
    T = TypeVar("T", bound=Mobject)


class SurfaceMesh(VGroup):
    """表面网格类，继承自VGroup，用于为Manim中的Surface对象生成可视化网格线"""
    
    def __init__(
        self,
        uv_surface: Surface,        # 待添加网格的Surface对象（核心依赖）
        resolution: Tuple[int, int] = (21, 11),  # 网格分辨率（u方向线数, v方向线数），默认(21,11)
        stroke_width: float = 1,    # 网格线的线条宽度，默认1
        stroke_color: ManimColor = GREY_A,  # 网格线的颜色，默认浅灰色
        normal_nudge: float = 1e-2, # 网格线相对于原始表面的偏移量（沿法向量方向），默认0.01
        depth_test: bool = True,    # 是否启用深度测试（影响3D场景中的遮挡显示），默认True
        joint_type: str = 'no_joint',  # 网格线连接点的样式，默认无特殊样式
        **kwargs                    # 其他关键字参数，传递给父类VGroup
    ):
        # 存储核心属性：待处理的Surface对象
        self.uv_surface = uv_surface
        # 存储网格分辨率：决定u、v两个方向各生成多少条网格线
        self.resolution = resolution
        # 存储网格线偏移量：避免网格线与原始表面完全重叠
        self.normal_nudge = normal_nudge

        # 调用父类VGroup的构造函数，传递样式相关参数
        super().__init__(
            stroke_color=stroke_color,
            stroke_width=stroke_width,
            depth_test=depth_test,
            joint_type=joint_type,
            **kwargs
        )

    def init_points(self) -> None:
        """初始化网格线的顶点数据，生成u、v两个方向的网格线并添加到组合中"""
        # 简化变量名：获取待处理的Surface对象
        uv_surface = self.uv_surface

        # 获取原始Surface的分辨率（u方向点数, v方向点数）
        full_nu, full_nv = uv_surface.resolution
        # 获取网格线的分辨率（u方向线数, v方向线数）
        part_nu, part_nv = self.resolution
        # 注：此处的“indices”为浮点型，后续会通过插值计算，实现网格线在两点间的平滑过渡
        # 生成u方向网格线对应的索引（从0到原始u方向最大索引，均匀生成part_nu个点）
        u_indices = np.linspace(0, full_nu - 1, part_nu)
        # 生成v方向网格线对应的索引（从0到原始v方向最大索引，均匀生成part_nv个点）
        v_indices = np.linspace(0, full_nv - 1, part_nv)

        # 获取原始Surface的所有顶点坐标
        points = uv_surface.get_points()
        # 获取原始Surface每个顶点的单位法向量（用于计算网格线偏移方向）
        normals = uv_surface.get_unit_normals()
        # 简化变量名：获取网格线偏移量
        nudge = self.normal_nudge
        # 计算偏移后的顶点坐标：原始顶点 + 偏移量×法向量（沿法向量方向轻微偏移）
        nudged_points = points + nudge * normals

        # -------------------------- 生成u方向的网格线 --------------------------
        # 遍历u方向的每个索引，生成一条沿v方向延伸的网格线
        for ui in u_indices:
            # 创建一个VMobject对象，用于存储单条u方向网格线
            path = VMobject()
            # 计算当前u索引对应的下限顶点索引（整数，向下取整）
            low_ui = full_nv * int(math.floor(ui))
            # 计算当前u索引对应的上限顶点索引（整数，向上取整）
            high_ui = full_nv * int(math.ceil(ui))
            # 为网格线设置平滑顶点：通过插值计算当前u索引对应的v方向所有顶点
            path.set_points_smoothly(interpolate(
                # 下限u对应的所有v方向偏移后顶点
                nudged_points[low_ui:low_ui + full_nv],
                # 上限u对应的所有v方向偏移后顶点
                nudged_points[high_ui:high_ui + full_nv],
                # 插值系数（ui的小数部分，0~1之间，决定两点间的插值比例）
                ui % 1
            ))
            # 将当前u方向网格线添加到SurfaceMesh组合中
            self.add(path)

        # -------------------------- 生成v方向的网格线 --------------------------
        # 遍历v方向的每个索引，生成一条沿u方向延伸的网格线
        for vi in v_indices:
            # 创建一个VMobject对象，用于存储单条v方向网格线
            path = VMobject()
            # 为网格线设置平滑顶点：通过插值计算当前v索引对应的u方向所有顶点
            path.set_points_smoothly(interpolate(
                # 下限v对应的所有u方向偏移后顶点（步长为full_nv，按列取点）
                nudged_points[int(math.floor(vi))::full_nv],
                # 上限v对应的所有u方向偏移后顶点（步长为full_nv，按列取点）
                nudged_points[int(math.ceil(vi))::full_nv],
                # 插值系数（vi的小数部分，0~1之间，决定两点间的插值比例）
                vi % 1
            ))
            # 将当前v方向网格线添加到SurfaceMesh组合中
            self.add(path)


# 3D shapes

class Sphere(Surface):
    """球体类，继承自Surface，用于在Manim中创建3D球体表面"""
    
    def __init__(
        self,
        u_range: Tuple[float, float] = (0, TAU),  # u参数范围，默认0到2π（完整圆周）
        v_range: Tuple[float, float] = (0, PI),   # v参数范围，默认0到π（半个圆周）
        resolution: Tuple[int, int] = (101, 51),  # 表面分辨率（u方向点数, v方向点数），默认(101,51)
        radius: float = 1.0,                      # 球体半径，默认1.0
        true_normals: bool = True,                # 是否使用真实法向量，默认True（避免极点处的问题）
        clockwise=False,                          # 是否顺时针方向生成，默认False
        **kwargs,                                 # 其他关键字参数，传递给父类Surface
    ):
        # 存储球体半径
        self.radius = radius
        # 存储旋转方向标志
        self.clockwise = clockwise
        
        # 调用父类Surface的构造函数，初始化表面基本属性
        super().__init__(
            u_range=u_range,
            v_range=v_range,
            resolution=resolution,** kwargs
        )
        
        # 为避免极点处的法向量问题，使用自定义的法向量点
        if true_normals:
            # 法向量点 = 表面点 × (半径+法向量偏移)/半径（按比例调整法向量长度）
            self.data['d_normal_point'] = self.data['point'] * ((radius + self.normal_nudge) / radius)

    def uv_func(self, u: float, v: float) -> np.ndarray:
        """
        球坐标系到笛卡尔坐标系的转换函数
        
        参数:
            u: 方位角（绕z轴旋转角度）
            v: 极角（与z轴夹角）
        
        返回:
            三维坐标点(x, y, z)
        """
        # 根据旋转方向设置符号（顺时针为-1，逆时针为+1）
        sign = -1 if self.clockwise else +1
        
        # 球坐标到笛卡尔坐标的转换公式（乘以半径缩放）
        return self.radius * np.array([
            math.cos(sign * u) * math.sin(v),  # x坐标
            math.sin(sign * u) * math.sin(v),  # y坐标
            -math.cos(v)                       # z坐标（负号调整方向）
        ])


class Torus(Surface):
    """圆环面类，继承自Surface，用于在Manim中创建3D圆环表面（甜甜圈形状）"""
    
    def __init__(
        self,
        u_range: Tuple[float, float] = (0, TAU),  # u参数范围，默认0到2π（绕中心轴旋转）
        v_range: Tuple[float, float] = (0, TAU),  # v参数范围，默认0到2π（绕圆环自身旋转）
        r1: float = 3.0,                          # 圆环中心到管中心的距离（大半径），默认3.0
        r2: float = 1.0,                          # 管的半径（小半径），默认1.0
        **kwargs,                                 # 其他关键字参数，传递给父类Surface
    ):
        # 存储大半径（中心到管中心距离）
        self.r1 = r1
        # 存储小半径（管自身半径）
        self.r2 = r2
        
        # 调用父类Surface的构造函数，初始化表面基本属性
        super().__init__(
            u_range=u_range,
            v_range=v_range,** kwargs,
        )

    def uv_func(self, u: float, v: float) -> np.ndarray:
        """
        圆环面的参数方程，计算三维坐标点
        
        参数:
            u: 绕中心轴旋转的角度
            v: 绕圆环管自身旋转的角度
        
        返回:
            三维坐标点(x, y, z)
        """
        # 计算绕中心轴旋转的单位向量（x-y平面内）
        P = np.array([math.cos(u), math.sin(u), 0])
        
        # 圆环面参数方程：
        # (大半径 - 小半径×cos(v)) × 旋转向量 - 小半径×sin(v) × 外方向向量
        return (self.r1 - self.r2 * math.cos(v)) * P - self.r2 * math.sin(v) * OUT


class Cylinder(Surface):
    def __init__(
        self,
        u_range: Tuple[float, float] = (0, TAU),
        v_range: Tuple[float, float] = (-1, 1),
        resolution: Tuple[int, int] = (101, 11),
        height: float = 2,
        radius: float = 1,
        axis: Vect3 = OUT,
        **kwargs,
    ):
        self.height = height
        self.radius = radius
        self.axis = axis
        super().__init__(
            u_range=u_range,
            v_range=v_range,
            resolution=resolution,
            **kwargs
        )

    def init_points(self):
        super().init_points()
        self.scale(self.radius)
        self.set_depth(self.height, stretch=True)
        self.apply_matrix(z_to_vector(self.axis))

    def uv_func(self, u: float, v: float) -> np.ndarray:
        return np.array([np.cos(u), np.sin(u), v])


class Cone(Cylinder):
    def __init__(
        self,
        u_range: Tuple[float, float] = (0, TAU),
        v_range: Tuple[float, float] = (0, 1),
        *args,
        **kwargs,
    ):
        super().__init__(u_range=u_range, v_range=v_range, *args, **kwargs)

    def uv_func(self, u: float, v: float) -> np.ndarray:
        return np.array([(1 - v) * np.cos(u), (1 - v) * np.sin(u), v])


class Line3D(Cylinder):
    def __init__(
        self,
        start: Vect3,
        end: Vect3,
        width: float = 0.05,
        resolution: Tuple[int, int] = (21, 25),
        **kwargs
    ):
        axis = end - start
        super().__init__(
            height=get_norm(axis),
            radius=width / 2,
            axis=axis,
            resolution=resolution,
            **kwargs
        )
        self.shift((start + end) / 2)


class Disk3D(Surface):
    def __init__(
        self,
        radius: float = 1,
        u_range: Tuple[float, float] = (0, 1),
        v_range: Tuple[float, float] = (0, TAU),
        resolution: Tuple[int, int] = (2, 100),
        **kwargs
    ):
        super().__init__(
            u_range=u_range,
            v_range=v_range,
            resolution=resolution,
            **kwargs,
        )
        self.scale(radius)

    def uv_func(self, u: float, v: float) -> np.ndarray:
        return np.array([
            u * math.cos(v),
            u * math.sin(v),
            0
        ])


class Square3D(Surface):
    def __init__(
        self,
        side_length: float = 2.0,
        u_range: Tuple[float, float] = (-1, 1),
        v_range: Tuple[float, float] = (-1, 1),
        resolution: Tuple[int, int] = (2, 2),
        **kwargs,
    ):
        super().__init__(
            u_range=u_range, 
            v_range=v_range, 
            resolution=resolution, 
            **kwargs
        )
        self.scale(side_length / 2)

    def uv_func(self, u: float, v: float) -> np.ndarray:
        return np.array([u, v, 0])


def square_to_cube_faces(square: T) -> list[T]:
    radius = square.get_height() / 2
    square.move_to(radius * OUT)
    result = [square.copy()]
    result.extend([
        square.copy().rotate(PI / 2, axis=vect, about_point=ORIGIN)
        for vect in compass_directions(4)
    ])
    result.append(square.copy().rotate(PI, RIGHT, about_point=ORIGIN))
    return result


class Cube(SGroup):
    def __init__(
        self,
        color: ManimColor = BLUE,
        opacity: float = 1,
        shading: Tuple[float, float, float] = (0.1, 0.5, 0.1),
        square_resolution: Tuple[int, int] = (2, 2),
        side_length: float = 2,
        **kwargs,
    ):
        face = Square3D(
            resolution=square_resolution,
            side_length=side_length,
            color=color,
            opacity=opacity,
            shading=shading,
        )
        super().__init__(*square_to_cube_faces(face), **kwargs)


class Prism(Cube):
    def __init__(
        self,
        width: float = 3.0,
        height: float = 2.0,
        depth: float = 1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        for dim, value in enumerate([width, height, depth]):
            self.rescale_to_fit(value, dim, stretch=True)


class VGroup3D(VGroup):
    def __init__(
        self,
        *vmobjects: VMobject,
        depth_test: bool = True,
        shading: Tuple[float, float, float] = (0.2, 0.2, 0.2),
        joint_type: str = "no_joint",
        **kwargs
    ):
        super().__init__(*vmobjects, **kwargs)
        self.set_shading(*shading)
        self.set_joint_type(joint_type)
        if depth_test:
            self.apply_depth_test()


class VCube(VGroup3D):
    def __init__(
        self,
        side_length: float = 2.0,
        fill_color: ManimColor = BLUE_D,
        fill_opacity: float = 1,
        stroke_width: float = 0,
        **kwargs
    ):
        style = dict(
            fill_color=fill_color,
            fill_opacity=fill_opacity,
            stroke_width=stroke_width,
            **kwargs
        )
        face = Square(side_length=side_length, **style)
        super().__init__(*square_to_cube_faces(face), **style)


class VPrism(VCube):
    def __init__(
        self,
        width: float = 3.0,
        height: float = 2.0,
        depth: float = 1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        for dim, value in enumerate([width, height, depth]):
            self.rescale_to_fit(value, dim, stretch=True)


class Dodecahedron(VGroup3D):
    def __init__(
        self,
        fill_color: ManimColor = BLUE_E,
        fill_opacity: float = 1,
        stroke_color: ManimColor = BLUE_E,
        stroke_width: float = 1,
        shading: Tuple[float, float, float] = (0.2, 0.2, 0.2),
        **kwargs,
    ):
        style = dict(
            fill_color=fill_color,
            fill_opacity=fill_opacity,
            stroke_color=stroke_color,
            stroke_width=stroke_width,
            shading=shading,
            **kwargs
        )

        # Start by creating two of the pentagons, meeting
        # back to back on the positive x-axis
        phi = (1 + math.sqrt(5)) / 2
        x, y, z = np.identity(3)
        pentagon1 = Polygon(
            np.array([phi, 1 / phi, 0]),
            np.array([1, 1, 1]),
            np.array([1 / phi, 0, phi]),
            np.array([1, -1, 1]),
            np.array([phi, -1 / phi, 0]),
            **style
        )
        pentagon2 = pentagon1.copy().stretch(-1, 2, about_point=ORIGIN)
        pentagon2.reverse_points()
        x_pair = VGroup(pentagon1, pentagon2)
        z_pair = x_pair.copy().apply_matrix(np.array([z, -x, -y]).T)
        y_pair = x_pair.copy().apply_matrix(np.array([y, z, x]).T)

        pentagons = [*x_pair, *y_pair, *z_pair]
        for pentagon in list(pentagons):
            pc = pentagon.copy()
            pc.apply_function(lambda p: -p)
            pc.reverse_points()
            pentagons.append(pc)

        super().__init__(*pentagons, **style)


class Prismify(VGroup3D):
    def __init__(self, vmobject, depth=1.0, direction=IN, **kwargs):
        # At the moment, this assume stright edges
        vect = depth * direction
        pieces = [vmobject.copy()]
        points = vmobject.get_anchors()
        for p1, p2 in adjacent_pairs(points):
            wall = VMobject()
            wall.match_style(vmobject)
            wall.set_points_as_corners([p1, p2, p2 + vect, p1 + vect])
            pieces.append(wall)
        top = vmobject.copy()
        top.shift(vect)
        top.reverse_points()
        pieces.append(top)
        super().__init__(*pieces, **kwargs)
