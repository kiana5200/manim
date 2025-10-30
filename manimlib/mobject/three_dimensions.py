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
    """圆柱体类，继承自Surface，用于在Manim中创建3D圆柱体表面"""
    
    def __init__(
        self,
        u_range: Tuple[float, float] = (0, TAU),  # u参数范围，默认0到2π（绕轴线旋转）
        v_range: Tuple[float, float] = (-1, 1),   # v参数范围，默认-1到1（沿轴线方向）
        resolution: Tuple[int, int] = (101, 11),  # 表面分辨率（u方向点数, v方向点数）
        height: float = 2,                        # 圆柱体高度，默认2
        radius: float = 1,                        # 圆柱体半径，默认1
        axis: Vect3 = OUT,                        # 圆柱体轴线方向，默认向外(OUT)
        **kwargs,                                 # 其他关键字参数，传递给父类Surface
    ):
        # 存储圆柱体高度
        self.height = height
        # 存储圆柱体半径
        self.radius = radius
        # 存储圆柱体轴线方向
        self.axis = axis
        
        # 调用父类Surface的构造函数，初始化表面基本属性
        super().__init__(
            u_range=u_range,
            v_range=v_range,
            resolution=resolution,** kwargs
        )

    def init_points(self):
        """初始化顶点数据，对基本圆柱进行缩放、调整深度和方向"""
        # 调用父类的init_points方法，生成基础顶点
        super().init_points()
        # 按半径缩放圆柱体（基础圆柱半径为1）
        self.scale(self.radius)
        # 调整圆柱体深度以匹配指定高度
        self.set_depth(self.height, stretch=True)
        # 应用矩阵变换，使圆柱体沿指定轴线方向排列
        self.apply_matrix(z_to_vector(self.axis))

    def uv_func(self, u: float, v: float) -> np.ndarray:
        """
        圆柱体的参数方程
        
        参数:
            u: 绕轴线旋转的角度
            v: 沿轴线方向的位置参数
        
        返回:
            三维坐标点(x, y, z)
        """
        # 圆柱坐标到笛卡尔坐标的转换（基础圆柱半径为1）
        return np.array([np.cos(u), np.sin(u), v])


class Cone(Cylinder):
    """圆锥体类，继承自Cylinder，用于在Manim中创建3D圆锥体表面"""
    
    def __init__(
        self,
        u_range: Tuple[float, float] = (0, TAU),  # u参数范围，默认0到2π（绕轴线旋转）
        v_range: Tuple[float, float] = (0, 1),    # v参数范围，默认0到1（从顶点到底部）
        *args,                                    # 可变参数，传递给父类Cylinder
        **kwargs,                                 # 关键字参数，传递给父类Cylinder
    ):
        # 调用父类Cylinder的构造函数
        super().__init__(u_range=u_range, v_range=v_range, *args, **kwargs)

    def uv_func(self, u: float, v: float) -> np.ndarray:
        """
        圆锥体的参数方程
        
        参数:
            u: 绕轴线旋转的角度
            v: 从顶点到底部的位置参数（0为顶点，1为底部）
        
        返回:
            三维坐标点(x, y, z)
        """
        # 圆锥参数方程：半径随v线性变化（v=0时半径为1，v=1时半径为0）
        return np.array([(1 - v) * np.cos(u), (1 - v) * np.sin(u), v])


class Line3D(Cylinder):
    """3D线段类，继承自Cylinder，用于在Manim中创建有宽度的3D线段"""
    
    def __init__(
        self,
        start: Vect3,                             # 线段起点坐标
        end: Vect3,                               # 线段终点坐标
        width: float = 0.05,                      # 线段宽度，默认0.05
        resolution: Tuple[int, int] = (21, 25),   # 线段表面分辨率
        **kwargs                                  # 其他关键字参数，传递给父类Cylinder
    ):
        # 计算线段方向向量（从起点到终点）
        axis = end - start
        # 调用父类Cylinder的构造函数，将圆柱体变形为线段
        super().__init__(
            height=get_norm(axis),                # 线段长度（方向向量的模）
            radius=width / 2,                     # 线段半径（宽度的一半）
            axis=axis,                            # 线段方向
            resolution=resolution,** kwargs
        )
        # 将线段移动到起点和终点的中间位置（使线段居中于两点之间）
        self.shift((start + end) / 2)


class Disk3D(Surface):
    """3D圆盘类，继承自Surface，用于在Manim中创建3D圆盘表面（圆形平面）"""
    
    def __init__(
        self,
        radius: float = 1,                        # 圆盘半径，默认1
        u_range: Tuple[float, float] = (0, 1),    # u参数范围，默认0到1（从中心到边缘）
        v_range: Tuple[float, float] = (0, TAU),  # v参数范围，默认0到2π（角度方向）
        resolution: Tuple[int, int] = (2, 100),   # 圆盘表面分辨率
        **kwargs,                                 # 其他关键字参数，传递给父类Surface
    ):
        # 调用父类Surface的构造函数，初始化表面基本属性
        super().__init__(
            u_range=u_range,
            v_range=v_range,
            resolution=resolution,** kwargs,
        )
        # 按指定半径缩放圆盘（基础圆盘半径为1）
        self.scale(radius)

    def uv_func(self, u: float, v: float) -> np.ndarray:
        """
        圆盘的参数方程
        
        参数:
            u: 从中心到边缘的距离参数（0为中心，1为边缘）
            v: 角度参数
        
        返回:
            三维坐标点(x, y, z)（位于z=0平面）
        """
        return np.array([
            u * math.cos(v),  # x坐标：距离参数×cos(角度)
            u * math.sin(v),  # y坐标：距离参数×sin(角度)
            0                 # z坐标：0（位于xy平面）
        ])

class Square3D(Surface):
    """3D正方形平面类，继承自Surface，用于创建3D空间中的正方形表面"""
    
    def __init__(
        self,
        side_length: float = 2.0,         # 正方形边长，默认2.0
        u_range: Tuple[float, float] = (-1, 1),  # u参数范围，默认-1到1
        v_range: Tuple[float, float] = (-1, 1),  # v参数范围，默认-1到1
        resolution: Tuple[int, int] = (2, 2),    # 表面分辨率，默认(2,2)（低分辨率足够）
        **kwargs,                         # 其他关键字参数，传递给父类Surface
    ):
        # 调用父类Surface的构造函数，初始化基础属性
        super().__init__(
            u_range=u_range, 
            v_range=v_range, 
            resolution=resolution, 
            **kwargs
        )
        # 按边长缩放正方形（基础正方形边长为2，所以除以2）
        self.scale(side_length / 2)

    def uv_func(self, u: float, v: float) -> np.ndarray:
        """
        正方形平面的参数方程
        
        参数:
            u: u方向参数
            v: v方向参数
        
        返回:
            三维坐标点(x, y, z)（位于z=0平面）
        """
        return np.array([u, v, 0])  # 正方形位于xy平面


def square_to_cube_faces(square: T) -> list[T]:
    """
    将单个正方形转换为立方体的6个面
    
    参数:
        square: 基础正方形对象
    
    返回:
        组成立方体的6个面的列表
    """
    # 计算从中心到边缘的距离（半径）
    radius = square.get_height() / 2
    # 将正方形移动到z轴正方向（朝外）
    square.move_to(radius * OUT)
    # 初始化结果列表，包含第一个面（前面）
    result = [square.copy()]
    # 添加四个侧面：围绕四个方向向量旋转90度
    result.extend([
        square.copy().rotate(PI / 2, axis=vect, about_point=ORIGIN)
        for vect in compass_directions(4)  # 四个方向：右、上、左、下
    ])
    # 添加最后一个面（后面）：绕RIGHT轴旋转180度
    result.append(square.copy().rotate(PI, RIGHT, about_point=ORIGIN))
    return result


class Cube(SGroup):
    """立方体类，继承自SGroup，由6个Square3D面组成"""
    
    def __init__(
        self,
        color: ManimColor = BLUE,              # 立方体颜色，默认蓝色
        opacity: float = 1,                    # 不透明度，默认完全不透明
        shading: Tuple[float, float, float] = (0.1, 0.5, 0.1),  # 阴影参数
        square_resolution: Tuple[int, int] = (2, 2),  # 每个面的分辨率
        side_length: float = 2,                # 边长，默认2
        **kwargs,                             # 其他关键字参数
    ):
        # 创建一个正方形面作为基础
        face = Square3D(
            resolution=square_resolution,
            side_length=side_length,
            color=color,
            opacity=opacity,
            shading=shading,
        )
        # 调用父类SGroup的构造函数，传入6个面
        super().__init__(*square_to_cube_faces(face), **kwargs)


class Prism(Cube):
    """棱柱类，继承自Cube，可创建长方体（各边可不等长）"""
    
    def __init__(
        self,
        width: float = 3.0,    # 宽度（x方向），默认3.0
        height: float = 2.0,   # 高度（y方向），默认2.0
        depth: float = 1.0,    # 深度（z方向），默认1.0
        **kwargs               # 其他关键字参数
    ):
        # 调用父类Cube的构造函数
        super().__init__(** kwargs)
        # 分别在三个维度上调整大小以匹配指定的宽、高、深
        for dim, value in enumerate([width, height, depth]):
            self.rescale_to_fit(value, dim, stretch=True)


class VGroup3D(VGroup):
    """3D向量图形组合类，继承自VGroup，为3D对象提供额外属性"""
    
    def __init__(
        self,
        *vmobjects: VMobject,    # 要包含的向量图形对象
        depth_test: bool = True,  # 是否启用深度测试，默认True
        shading: Tuple[float, float, float] = (0.2, 0.2, 0.2),  # 阴影参数
        joint_type: str = "no_joint",  # 连接点类型，默认无特殊样式
        **kwargs                  # 其他关键字参数
    ):
        # 调用父类VGroup的构造函数
        super().__init__(*vmobjects, **kwargs)
        # 设置阴影效果
        self.set_shading(*shading)
        # 设置连接点类型
        self.set_joint_type(joint_type)
        # 如果需要，应用深度测试
        if depth_test:
            self.apply_depth_test()


class VCube(VGroup3D):
    """向量立方体类，继承自VGroup3D，使用2D正方形构建3D立方体"""
    
    def __init__(
        self,
        side_length: float = 2.0,        # 边长，默认2.0
        fill_color: ManimColor = BLUE_D, # 填充颜色，默认深蓝色
        fill_opacity: float = 1,         # 填充不透明度，默认1
        stroke_width: float = 0,         # 边框宽度，默认0（无边界）
        **kwargs                         # 其他关键字参数
    ):
        # 整合样式参数
        style = dict(
            fill_color=fill_color,
            fill_opacity=fill_opacity,
            stroke_width=stroke_width,** kwargs
        )
        # 创建一个2D正方形作为基础面
        face = Square(side_length=side_length, **style)
        # 调用父类VGroup3D的构造函数，传入6个面
        super().__init__(*square_to_cube_faces(face), ** style)


class VPrism(VCube):
    """向量棱柱类，继承自VCube，可创建长方体"""
    
    def __init__(
        self,
        width: float = 3.0,    # 宽度（x方向），默认3.0
        height: float = 2.0,   # 高度（y方向），默认2.0
        depth: float = 1.0,    # 深度（z方向），默认1.0
        **kwargs               # 其他关键字参数
    ):
        # 调用父类VCube的构造函数
        super().__init__(** kwargs)
        # 分别在三个维度上调整大小以匹配指定的宽、高、深
        for dim, value in enumerate([width, height, depth]):
            self.rescale_to_fit(value, dim, stretch=True)


class Dodecahedron(VGroup3D):
    """十二面体类，继承自VGroup3D，创建正十二面体（12个五边形面）"""
    
    def __init__(
        self,
        fill_color: ManimColor = BLUE_E,  # 填充颜色，默认浅蓝色
        fill_opacity: float = 1,          # 填充不透明度，默认1
        stroke_color: ManimColor = BLUE_E, # 边框颜色，默认浅蓝色
        stroke_width: float = 1,          # 边框宽度，默认1
        shading: Tuple[float, float, float] = (0.2, 0.2, 0.2),  # 阴影参数
        **kwargs,                         # 其他关键字参数
    ):
        # 整合样式参数
        style = dict(
            fill_color=fill_color,
            fill_opacity=fill_opacity,
            stroke_color=stroke_color,
            stroke_width=stroke_width,
            shading=shading,** kwargs
        )

        # 黄金比例（正十二面体的关键比例）
        phi = (1 + math.sqrt(5)) / 2
        # 创建x、y、z轴单位向量
        x, y, z = np.identity(3)
        
        # 创建第一个五边形面
        pentagon1 = Polygon(
            np.array([phi, 1 / phi, 0]),
            np.array([1, 1, 1]),
            np.array([1 / phi, 0, phi]),
            np.array([1, -1, 1]),
            np.array([phi, -1 / phi, 0]),
            **style
        )
        # 创建背面五边形（沿z轴翻转）
        pentagon2 = pentagon1.copy().stretch(-1, 2, about_point=ORIGIN)
        pentagon2.reverse_points()  # 反转点顺序以确保法线方向正确
        
        # 创建x方向的五边形对
        x_pair = VGroup(pentagon1, pentagon2)
        # 创建z方向的五边形对（通过矩阵变换旋转x方向的对）
        z_pair = x_pair.copy().apply_matrix(np.array([z, -x, -y]).T)
        # 创建y方向的五边形对（通过矩阵变换旋转x方向的对）
        y_pair = x_pair.copy().apply_matrix(np.array([y, z, x]).T)

        # 收集所有五边形面
        pentagons = [*x_pair, *y_pair, *z_pair]
        # 添加剩余的六个面（通过中心对称创建）
        for pentagon in list(pentagons):
            pc = pentagon.copy()
            pc.apply_function(lambda p: -p)  # 中心对称
            pc.reverse_points()  # 反转点顺序
            pentagons.append(pc)

        # 调用父类VGroup3D的构造函数，传入所有12个五边形面
        super().__init__(*pentagons, **style)


class Prismify(VGroup3D):
    """3D棱柱化类，继承自VGroup3D，用于将2D向量图形（VMobject）转换为3D棱柱"""
    
    def __init__(self, vmobject, depth=1.0, direction=IN, **kwargs):
        # 注：当前版本假设输入的2D图形（vmobject）仅包含直边（非曲线边）
        # 计算棱柱的延伸向量：深度 × 延伸方向（控制棱柱的厚度和朝向）
        vect = depth * direction
        # 初始化存储棱柱所有组成部分的列表，先加入2D图形的"底面"（原始图形副本）
        pieces = [vmobject.copy()]
        # 获取2D图形的所有锚点（顶点坐标），用于构建侧面
        points = vmobject.get_anchors()
        
        # 遍历相邻的顶点对，为每对顶点构建棱柱的一个侧面
        for p1, p2 in adjacent_pairs(points):
            # 创建一个空的向量图形对象，作为当前侧面
            wall = VMobject()
            # 让侧面的样式（颜色、边框等）与原始2D图形保持一致
            wall.match_style(vmobject)
            # 设置侧面的四个顶点，形成矩形侧面：
            # 顺序为“底面顶点1 → 底面顶点2 → 顶面顶点2 → 顶面顶点1”
            wall.set_points_as_corners([p1, p2, p2 + vect, p1 + vect])
            # 将当前侧面添加到棱柱组成部分列表中
            pieces.append(wall)
        
        # 创建棱柱的"顶面"：复制原始2D图形
        top = vmobject.copy()
        # 将顶面沿延伸向量移动，与底面形成厚度（顶面位置 = 底面位置 + 延伸向量）
        top.shift(vect)
        # 反转顶面的顶点顺序：确保顶面的法向量方向与底面相反（避免3D渲染时的面朝向错误）
        top.reverse_points()
        # 将顶面添加到棱柱组成部分列表中
        pieces.append(top)
        
        # 调用父类VGroup3D的构造函数，将底面、所有侧面、顶面整合为3D棱柱组合
        super().__init__(*pieces, **kwargs)
