# 从__future__导入annotations，用于支持字符串类型注解（Python 3.7+特性）
from __future__ import annotations

# 从abc模块导入ABC（抽象基类）和abstractmethod（抽象方法装饰器）
from abc import ABC, abstractmethod
# 导入numbers模块，用于处理数值类型相关操作
import numbers

# 导入numpy库并简写为np，用于数值计算和数组操作
import numpy as np
# 导入itertools库并简写为it，用于处理迭代器和组合操作
import itertools as it

# 从manimlib.constants导入各种常量：
# 颜色常量（黑色、蓝色系列、绿色、灰色、红色、默认物体颜色）
from manimlib.constants import BLACK, BLUE, BLUE_D, BLUE_E, GREEN, GREY_A, RED, DEFAULT_MOBJECT_COLOR
# 角度相关常量（度、π）
from manimlib.constants import DEG, PI
# 方向向量常量（左下、左上、下、右下、左、原点、外、右、上）
from manimlib.constants import DL, UL, DOWN, DR, LEFT, ORIGIN, OUT, RIGHT, UP
# 帧尺寸相关常量（帧的X半径、Y半径）
from manimlib.constants import FRAME_X_RADIUS, FRAME_Y_RADIUS
# 缓冲距离常量（中小编号缓冲、小编号缓冲）
from manimlib.constants import MED_SMALL_BUFF, SMALL_BUFF

# 从manimlib.mobject.functions导入ParametricCurve（参数曲线类）
from manimlib.mobject.functions import ParametricCurve
# 从manimlib.mobject.geometry导入几何相关类：
# Arrow（箭头）、DashedLine（虚线）、Line（直线）、Rectangle（矩形）
from manimlib.mobject.geometry import Arrow, DashedLine, Line, Rectangle
# 从manimlib.mobject.number_line导入NumberLine（数轴类）
from manimlib.mobject.number_line import NumberLine
# 从manimlib.mobject.svg.tex_mobject导入Tex（TeX文本对象类）
from manimlib.mobject.svg.tex_mobject import Tex
# 从manimlib.mobject.types.dot_cloud导入DotCloud（点云类）
from manimlib.mobject.types.dot_cloud import DotCloud
# 从manimlib.mobject.types.surface导入ParametricSurface（参数曲面类）
from manimlib.mobject.types.surface import ParametricSurface
# 从manimlib.mobject.types.vectorized_mobject导入：
# VGroup（向量对象组）、VMobject（向量化物体基类）
from manimlib.mobject.types.vectorized_mobject import VGroup, VMobject

# 从manimlib.utils.bezier导入inverse_interpolate（贝塞尔曲线反插值函数）
from manimlib.utils.bezier import inverse_interpolate
# 从manimlib.utils.dict_ops导入merge_dicts_recursively（字典递归合并函数）
from manimlib.utils.dict_ops import merge_dicts_recursively
# 从manimlib.utils.simple_functions导入binary_search（二分查找函数）
from manimlib.utils.simple_functions import binary_search
# 从manimlib.utils.space_ops导入空间操作相关函数：
# angle_of_vector（向量角度计算）、get_norm（向量范数计算）
# rotate_vector（向量旋转）、normalize（向量归一化）
from manimlib.utils.space_ops import angle_of_vector, get_norm, rotate_vector, normalize

# 导入typing模块中的类型检查相关工具
from typing import TYPE_CHECKING

# 条件导入，仅在类型检查时生效（不影响运行时）
if TYPE_CHECKING:
    # 从typing导入各种类型提示工具
    from typing import Callable, Iterable, Sequence, Type, TypeVar, Optional
    # 导入manimlib中的Mobject（物体基类）
    from manimlib.mobject.mobject import Mobject
    # 导入manimlib中的自定义类型
    from manimlib.typing import ManimColor, Vect3, Vect3Array, VectN, RangeSpecifier, Self
    # 定义类型变量T，约束为Mobject的子类
    T = TypeVar("T", bound=Mobject)


# 定义一个极小值，用于避免除以零或处理浮点数精度问题
EPSILON = 1e-8
# 默认X轴范围：(最小值, 最大值, 刻度间隔)
DEFAULT_X_RANGE = (-8.0, 8.0, 1.0)
# 默认Y轴范围：(最小值, 最大值, 刻度间隔)
DEFAULT_Y_RANGE = (-4.0, 4.0, 1.0)


def full_range_specifier(range_args):
    """
    确保范围参数是完整的三元组形式
    
    如果输入的范围参数只有两个值（最小值和最大值），
    则自动补充第三个值为1.0作为刻度间隔；
    如果已经是三元组，则直接返回
    """
    if len(range_args) == 2:
        return (*range_args, 1)
    return range_args


class CoordinateSystem(ABC):
    """
    坐标系的抽象基类，为Axes（坐标轴）和NumberPlane（坐标系）提供基础功能
    """
    # 维度默认为2D（二维）
    dimension: int = 2

    def __init__(
        self,
        x_range: RangeSpecifier = DEFAULT_X_RANGE,
        y_range: RangeSpecifier = DEFAULT_Y_RANGE,
        num_sampled_graph_points_per_tick: int = 5,
    ):
        """
        初始化坐标系
        
        参数:
            x_range: X轴范围，形式为(最小值, 最大值, 刻度间隔)
            y_range: Y轴范围，形式为(最小值, 最大值, 刻度间隔)
            num_sampled_graph_points_per_tick: 每个刻度间隔内用于绘制图形的采样点数
        """
        # 处理X轴范围，确保是三元组形式
        self.x_range = full_range_specifier(x_range)
        # 处理Y轴范围，确保是三元组形式
        self.y_range = full_range_specifier(y_range)
        # 每个刻度的采样点数
        self.num_sampled_graph_points_per_tick = num_sampled_graph_points_per_tick

    @abstractmethod
    def coords_to_point(self, *coords: float | VectN) -> Vect3 | Vect3Array:
        """
        抽象方法：将坐标值转换为场景中的点（像素坐标）
        
        需要在子类中实现具体转换逻辑
        """
        raise Exception("Not implemented")

    @abstractmethod
    def point_to_coords(self, point: Vect3 | Vect3Array) -> tuple[float | VectN, ...]:
        """
        抽象方法：将场景中的点（像素坐标）转换为坐标系中的坐标值
        
        需要在子类中实现具体转换逻辑
        """
        raise Exception("Not implemented")

    def c2p(self, *coords: float) -> Vect3 | Vect3Array:
        """coords_to_point方法的缩写，用于快速调用"""
        return self.coords_to_point(*coords)

    def p2c(self, point: Vect3) -> tuple[float | VectN, ...]:
        """point_to_coords方法的缩写，用于快速调用"""
        return self.point_to_coords(point)

    def get_origin(self) -> Vect3:
        """获取坐标系原点在场景中的位置"""
        return self.c2p(*[0] * self.dimension)

    @abstractmethod
    def get_axes(self) -> VGroup:
        """
        抽象方法：获取所有坐标轴组成的组
        
        需要在子类中实现具体逻辑
        """
        raise Exception("Not implemented")

    @abstractmethod
    def get_all_ranges(self) -> list[np.ndarray]:
        """
        抽象方法：获取所有轴的范围信息
        
        需要在子类中实现具体逻辑
        """
        raise Exception("Not implemented")

    def get_axis(self, index: int) -> NumberLine:
        """
        根据索引获取特定的轴
        
        参数:
            index: 轴的索引（0通常为X轴，1为Y轴，2为Z轴）
        返回:
            指定的坐标轴（NumberLine对象）
        """
        return self.get_axes()[index]

    def get_x_axis(self) -> NumberLine:
        """获取X轴（索引为0的轴）"""
        return self.get_axis(0)

    def get_y_axis(self) -> NumberLine:
        """获取Y轴（索引为1的轴）"""
        return self.get_axis(1)

    def get_z_axis(self) -> NumberLine:
        """获取Z轴（索引为2的轴）"""
        return self.get_axis(2)

    def get_x_axis_label(
        self,
        label_tex: str,
        edge: Vect3 = RIGHT,
        direction: Vect3 = DL,** kwargs
    ) -> Tex:
        """
        创建X轴的标签
        
        参数:
            label_tex: 标签的TeX文本
            edge: 标签靠近X轴的边缘方向
            direction: 标签相对于轴边缘的方向
            **kwargs: 传递给get_axis_label的其他参数
        返回:
            标签对象（Tex）
        """
        return self.get_axis_label(
            label_tex, self.get_x_axis(),
            edge, direction, **kwargs
        )

    def get_y_axis_label(
        self,
        label_tex: str,
        edge: Vect3 = UP,
        direction: Vect3 = DR,** kwargs
    ) -> Tex:
        """
        创建Y轴的标签
        
        参数:
            label_tex: 标签的TeX文本
            edge: 标签靠近Y轴的边缘方向
            direction: 标签相对于轴边缘的方向
            **kwargs: 传递给get_axis_label的其他参数
        返回:
            标签对象（Tex）
        """
        return self.get_axis_label(
            label_tex, self.get_y_axis(),
            edge, direction, **kwargs
        )

    def get_axis_label(
        self,
        label_tex: str,
        axis: Vect3,
        edge: Vect3,
        direction: Vect3,
        buff: float = MED_SMALL_BUFF,
        ensure_on_screen: bool = False
    ) -> Tex:
        """
        为指定轴创建标签
        
        参数:
            label_tex: 标签的TeX文本
            axis: 要添加标签的轴
            edge: 标签靠近轴的边缘方向
            direction: 标签相对于轴边缘的方向
            buff: 标签与轴之间的缓冲距离
            ensure_on_screen: 是否确保标签在屏幕内
        返回:
            标签对象（Tex）
        """
        # 创建TeX文本标签
        label = Tex(label_tex)
        # 将标签放置在轴的指定边缘和方向
        label.next_to(
            axis.get_edge_center(edge), direction,
            buff=buff
        )
        # 如果需要，将标签移到屏幕内
        if ensure_on_screen:
            label.shift_onto_screen(buff=MED_SMALL_BUFF)
        return label

    def get_axis_labels(
        self,
        x_label_tex: str = "x",
        y_label_tex: str = "y"
    ) -> VGroup:
        """
        创建并返回X轴和Y轴的标签组
        
        参数:
            x_label_tex: X轴标签的TeX文本，默认为"x"
            y_label_tex: Y轴标签的TeX文本，默认为"y"
        返回:
            包含X轴和Y轴标签的组（VGroup）
        """
        self.axis_labels = VGroup(
            self.get_x_axis_label(x_label_tex),
            self.get_y_axis_label(y_label_tex),
        )
        return self.axis_labels

def get_line_from_axis_to_point(
        self, 
        index: int,
        point: Vect3,
        line_func: Type[T] = DashedLine,
        color: ManimColor = GREY_A,
        stroke_width: float = 2
    ) -> T:
    """
    创建从指定轴到给定点的连接线
    
    参数:
        index: 轴的索引（0为X轴，1为Y轴等）
        point: 目标点的坐标
        line_func: 用于创建线的类（默认为虚线DashedLine）
        color: 线的颜色（默认为灰色GREY_A）
        stroke_width: 线的宽度（默认为2）
    返回:
        创建的线对象
    """
    # 获取指定索引的轴
    axis = self.get_axis(index)
    # 创建从点在轴上的投影到该点的线
    line = line_func(axis.get_projection(point), point)
    # 设置线的颜色和宽度
    line.set_stroke(color, stroke_width)
    return line

def get_v_line(self, point: Vect3, **kwargs):
    """
    创建从X轴到给定点的垂直线（vertical line）
    
    参数:
        point: 目标点的坐标
        **kwargs: 传递给get_line_from_axis_to_point的其他参数
    返回:
        创建的垂直线对象
    """
    # 调用通用方法，指定索引0（X轴）
    return self.get_line_from_axis_to_point(0, point,** kwargs)

def get_h_line(self, point: Vect3, **kwargs):
    """
    创建从Y轴到给定点的水平线（horizontal line）
    
    参数:
        point: 目标点的坐标
        **kwargs: 传递给get_line_from_axis_to_point的其他参数
    返回:
        创建的水平线对象
    """
    # 调用通用方法，指定索引1（Y轴）
    return self.get_line_from_axis_to_point(1, point,** kwargs)

# 用于图形绘制的工具方法
def get_graph(
    self,
    function: Callable[[float], float],
    x_range: Sequence[float] | None = None,
    bind: bool = False,
    **kwargs
) -> ParametricCurve:
    """
    根据给定的函数创建参数曲线（图形）
    
    参数:
        function: 要绘制的函数，形式为y = f(x)
        x_range: X轴范围，形式为(最小值, 最大值, 采样间隔)，默认为坐标系的x_range
        bind: 是否将图形与函数绑定（用于动态更新）
        **kwargs: 传递给ParametricCurve的其他参数
    返回:
        创建的参数曲线对象
    """
    # 如果未指定x_range，使用坐标系的默认x_range
    x_range = x_range or self.x_range
    # 初始化t_range为全1数组，用于参数化曲线的采样范围
    t_range = np.ones(3)
    # 将x_range的值赋给t_range（确保长度匹配）
    t_range[:len(x_range)] = x_range
    # 调整采样间隔：每个刻度间隔内采样num_sampled_graph_points_per_tick次
    t_range[2] /= self.num_sampled_graph_points_per_tick

    def parametric_function(t: float) -> Vect3:
        """参数化函数，将t映射到坐标系中的点"""
        return self.c2p(t, function(t))

    # 创建参数曲线对象
    graph = ParametricCurve(
        parametric_function,
        t_range=tuple(t_range),** kwargs
    )
    # 存储原始函数引用，便于后续使用
    graph.underlying_function = function
    # 存储使用的x范围
    graph.x_range = x_range

    # 如果需要绑定，将图形与函数绑定以支持动态更新
    if bind:
        self.bind_graph_to_func(graph, function)

    return graph

def get_parametric_curve(
    self,
    function: Callable[[float], Vect3],
    **kwargs
) -> ParametricCurve:
    """
    创建参数曲线（适用于多变量参数函数）
    
    参数:
        function: 参数函数，输入参数t，输出坐标值
        **kwargs: 传递给ParametricCurve的其他参数
    返回:
        创建的参数曲线对象
    """
    # 获取坐标系维度
    dim = self.dimension
    # 创建参数曲线，将函数输出转换为坐标系中的点
    graph = ParametricCurve(
        lambda t: self.coords_to_point(*function(t)[:dim]),** kwargs
    )
    # 存储原始函数引用
    graph.underlying_function = function
    return graph

def input_to_graph_point(
    self,
    x: float,
    graph: ParametricCurve
) -> Vect3 | None:
    """
    根据输入的x值，获取函数图形上对应的点
    
    参数:
        x: 输入的x坐标值
        graph: 函数图形对象
    返回:
        图形上对应点的坐标，若未找到则返回None
    """
    # 如果图形有底层函数，直接计算对应点
    if hasattr(graph, "underlying_function"):
        return self.coords_to_point(x, graph.underlying_function(x))
    else:
        # 否则通过二分查找找到对应的点
        alpha = binary_search(
            # 匿名函数：将比例值a转换为对应的x坐标
            function=lambda a: self.point_to_coords(
                graph.quick_point_from_proportion(a)
            )[0],
            target=x,  # 目标x值
            lower_bound=self.x_range[0],  # 搜索下界
            upper_bound=self.x_range[1],  # 搜索上界
        )
        # 如果找到对应的比例值，返回该点
        if alpha is not None:
            return graph.quick_point_from_proportion(alpha)
        else:
            return None

def i2gp(self, x: float, graph: ParametricCurve) -> Vect3 | None:
    """
    input_to_graph_point方法的缩写，用于快速调用
    """
    return self.input_to_graph_point(x, graph)

def bind_graph_to_func(
    self,
    graph: VMobject,
    func: Callable[[VectN], VectN],
    jagged: bool = False,
    get_discontinuities: Optional[Callable[[], Vect3]] = None
) -> VMobject:
    """
    将图形与函数绑定，使图形能随函数动态更新（适用于随时间变化的函数）
    
    参数:
        graph: 要绑定的图形对象
        func: 函数对象
        jagged: 是否保持图形的锯齿状（不进行平滑处理）
        get_discontinuities: 获取函数不连续点的方法
    返回:
        绑定后的图形对象
    """
    # 获取图形上所有点对应的x值（从X轴坐标转换）
    x_values = np.array([self.x_axis.p2n(p) for p in graph.get_points()])

    def get_graph_points():
        """获取更新后的图形点坐标"""
        xs = x_values
        # 如果有不连续点，在不连续点附近添加采样点以正确显示
        if get_discontinuities:
            ds = get_discontinuities()
            ep = 1e-6  # 极小值，用于不连续点两侧的采样
            # 在每个不连续点两侧添加采样点
            added_xs = it.chain(*((d - ep, d + ep) for d in ds))
            # 合并并排序所有采样点，保持原长度
            xs[:] = sorted([*x_values, *added_xs])[:len(x_values)]
        # 计算所有x对应的y值，并转换为坐标系中的点
        return self.c2p(xs, func(xs))

    # 为图形添加更新器：更新点坐标
    graph.add_updater(
        lambda g: g.set_points_as_corners(get_graph_points())
    )
    # 如果不需要锯齿状，对图形进行平滑处理
    if not jagged:
        graph.add_updater(lambda g: g.make_smooth(approx=True))
    return graph

def get_graph_label(
    self,
    graph: ParametricCurve,
    label: str | Mobject = "f(x)",
    x: float | None = None,
    direction: Vect3 = RIGHT,
    buff: float = MED_SMALL_BUFF,
    color: ManimColor | None = None
) -> Tex | Mobject:
    """
    为函数图形添加标签
    
    参数:
        graph: 函数图形对象
        label: 标签内容（字符串或Mobject对象），默认为"f(x)"
        x: 标签所在的x坐标位置，默认为自动选择
        direction: 标签相对于图形的方向
        buff: 标签与图形之间的缓冲距离
        color: 标签颜色，默认为与图形颜色一致
    返回:
        创建的标签对象
    """
    # 如果标签是字符串，转换为Tex对象
    if isinstance(label, str):
        label = Tex(label)
    # 如果未指定颜色，与图形颜色保持一致
    if color is None:
        label.match_color(graph)
    # 如果未指定x位置，自动选择合适的位置
    if x is None:
        # 从右侧开始搜索，找到一个y值在屏幕范围内的点
        max_y = FRAME_Y_RADIUS - label.get_height()  # Y方向最大可用范围
        max_x = FRAME_X_RADIUS - label.get_width()   # X方向最大可用范围
        # 从右向左遍历x_range中的点
        for x0 in np.arange(*self.x_range)[::-1]:
            pt = self.i2gp(x0, graph)  # 获取图形上x0对应的点
            # 检查点是否在屏幕范围内
            if abs(pt[0]) < max_x and abs(pt[1]) < max_y:
                x = x0
                break
        # 如果未找到合适的点，使用x_range的最大值
        if x is None:
            x = self.x_range[1]

    # 获取标签所在的点
    point = self.input_to_graph_point(x, graph)
    # 获取该点处切线的角度
    angle = self.angle_of_tangent(x, graph)
    # 计算法线方向（垂直于切线）
    normal = rotate_vector(RIGHT, angle + 90 * DEG)
    # 确保法线方向向上（如果向下则反转）
    if normal[1] < 0:
        normal *= -1
    # 将标签放置在点的法线方向上
    label.next_to(point, normal, buff=buff)
    # 确保标签在屏幕内
    label.shift_onto_screen()
    return label

def get_v_line_to_graph(self, x: float, graph: ParametricCurve, **kwargs):
    """
    创建从X轴到图形上指定x点的垂直线
    
    参数:
        x: 指定的x坐标
        graph: 函数图形对象
        **kwargs: 传递给get_v_line的其他参数
    返回:
        创建的垂直线对象
    """
    return self.get_v_line(self.i2gp(x, graph),** kwargs)

def get_h_line_to_graph(self, x: float, graph: ParametricCurve, **kwargs):
    """
    创建从Y轴到图形上指定x点的水平线
    
    参数:
        x: 指定的x坐标
        graph: 函数图形对象
        **kwargs: 传递给get_h_line的其他参数
    返回:
        创建的水平线对象
    """
    return self.get_h_line(self.i2gp(x, graph),** kwargs)

def get_scatterplot(self,
                    x_values: Vect3Array,
                    y_values: Vect3Array,
                    **dot_config):
    """
    创建散点图
    
    参数:
        x_values: x坐标数组
        y_values: y坐标数组
        **dot_config: 点的配置参数（颜色、大小等）
    返回:
        创建的散点图对象（DotCloud）
    """
    # 将x和y值转换为坐标系中的点，并创建点云
    return DotCloud(self.c2p(x_values, y_values),** dot_config)

# 微积分相关功能
def angle_of_tangent(
    self,
    x: float,
    graph: ParametricCurve,
    dx: float = EPSILON
) -> float:
    """
    计算函数图像在指定x处切线与水平方向的夹角
    
    参数:
        x: 计算切线角度的x坐标
        graph: 函数图像对象
        dx: 微小增量，用于近似计算导数（默认为极小值EPSILON）
    返回:
        切线角度（弧度）
    """
    # 获取x处的点坐标
    p0 = self.input_to_graph_point(x, graph)
    # 获取x+dx处的点坐标（用于近似切线方向）
    p1 = self.input_to_graph_point(x + dx, graph)
    # 计算两点连线向量的角度，即切线角度
    return angle_of_vector(p1 - p0)

def slope_of_tangent(
    self,
    x: float,
    graph: ParametricCurve,** kwargs
) -> float:
    """
    计算函数图像在指定x处切线的斜率
    
    参数:
        x: 计算切线斜率的x坐标
        graph: 函数图像对象
        **kwargs: 传递给angle_of_tangent的参数（如dx）
    返回:
        切线斜率
    """
    # 斜率 = tan(切线角度)
    return np.tan(self.angle_of_tangent(x, graph, **kwargs))

def get_tangent_line(
    self,
    x: float,
    graph: ParametricCurve,
    length: float = 5,
    line_func: Type[T] = Line
) -> T:
    """
    获取函数图像在指定x处的切线
    
    参数:
        x: 切线位置的x坐标
        graph: 函数图像对象
        length: 切线长度（默认为5）
        line_func: 线的类型（默认为直线Line）
    返回:
        切线对象
    """
    # 创建一条水平基准线
    line = line_func(LEFT, RIGHT)
    # 设置切线长度
    line.set_width(length)
    # 按切线角度旋转线
    line.rotate(self.angle_of_tangent(x, graph))
    # 将切线移动到函数图像上的x点位置
    line.move_to(self.input_to_graph_point(x, graph))
    return line

def get_riemann_rectangles(
    self,
    graph: ParametricCurve,
    x_range: Sequence[float] = None,
    dx: float | None = None,
    input_sample_type: str = "left",
    stroke_width: float = 1,
    stroke_color: ManimColor = BLACK,
    fill_opacity: float = 1,
    colors: Iterable[ManimColor] = (BLUE, GREEN),
    negative_color: ManimColor = RED,
    stroke_background: bool = True,
    show_signed_area: bool = True
) -> VGroup:
    """
    创建黎曼矩形（用于近似定积分）
    
    参数:
        graph: 函数图像对象
        x_range: 积分区间(x_min, x_max)，默认使用坐标系x_range
        dx: 矩形宽度，默认使用坐标系x_range的间隔
        input_sample_type: 采样点类型：left/right/center（左/右/中点）
        stroke_width: 矩形边框宽度
        stroke_color: 矩形边框颜色
        fill_opacity: 填充透明度
        colors: 填充渐变色
        negative_color: 负值区域填充色
        stroke_background: 是否将边框置于填充之下
        show_signed_area: 是否通过颜色区分正负面积
    返回:
        包含所有黎曼矩形的VGroup对象
    """
    # 处理x_range默认值
    if x_range is None:
        x_range = self.x_range[:2]
    # 处理dx默认值
    if dx is None:
        dx = self.x_range[2]
    # 确保x_range是三元组(x_min, x_max, dx)
    if len(x_range) < 3:
        x_range = [*x_range, dx]

    rects = []  # 存储所有矩形
    # 扩展上限以确保最后一个矩形完整
    x_range[1] = x_range[1] + dx
    # 生成所有矩形左边界x值
    xs = np.arange(*x_range)
    # 遍历每个矩形区间[x0, x1)
    for x0, x1 in zip(xs, xs[1:]):
        # 根据采样类型选择高度计算点
        if input_sample_type == "left":
            sample = x0  # 左端点
        elif input_sample_type == "right":
            sample = x1  # 右端点
        elif input_sample_type == "center":
            sample = 0.5 * x0 + 0.5 * x1  # 中点
        else:
            raise Exception("Invalid input sample type")
        
        # 计算矩形高度向量（从x轴到函数图像）
        height_vect = self.i2gp(sample, graph) - self.c2p(sample, 0)
        # 创建矩形
        rect = Rectangle(
            # 宽度：x1到x0的像素距离
            width=self.x_axis.n2p(x1)[0] - self.x_axis.n2p(x0)[0],
            # 高度：高度向量的模长
            height=get_norm(height_vect),
        )
        # 标记矩形是否在x轴上方
        rect.positive = height_vect[1] > 0
        # 定位矩形：上方区域从左下角对齐，下方区域从左上角对齐
        rect.move_to(self.c2p(x0, 0), DL if rect.positive else UL)
        rects.append(rect)
    
    # 将所有矩形组合成组
    result = VGroup(*rects)
    # 设置渐变色填充
    result.set_submobject_colors_by_gradient(*colors)
    # 设置矩形样式
    result.set_style(
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        fill_opacity=fill_opacity,
        stroke_behind=stroke_background  # 边框在填充之后
    )
    # 为负值区域矩形设置特殊颜色
    for rect in result:
        if not rect.positive:
            rect.set_fill(negative_color)
    
    return result

def get_area_under_graph(self, graph, x_range, fill_color=BLUE, fill_opacity=0.5):
    """
    获取函数图像下指定区间的面积区域（多边形）
    
    参数:
        graph: 函数图像对象（需有x_range属性）
        x_range: 积分区间(x_min, x_max)
        fill_color: 填充颜色
        fill_opacity: 填充透明度
    返回:
        表示面积的多边形对象
    """
    # 检查图形是否有x_range属性
    if not hasattr(graph, "x_range"):
        raise Exception("Argument `graph` must have attribute `x_range`")

    # 计算区间在图形参数化中的比例范围
    alpha_bounds = [
        inverse_interpolate(*graph.x_range, x)
        for x in x_range
    ]
    # 复制图形并截取指定区间部分
    sub_graph = graph.copy()
    sub_graph.pointwise_become_partial(graph, *alpha_bounds)
    # 添加从图形右端点到x轴的垂线
    sub_graph.add_line_to(self.c2p(x_range[1], 0))
    # 添加从x轴右端点到左端点的水平线
    sub_graph.add_line_to(self.c2p(x_range[0], 0))
    # 添加从x轴左端点回到图形起点的线（闭合多边形）
    sub_graph.add_line_to(sub_graph.get_start())

    # 设置样式：无边框，有填充
    sub_graph.set_stroke(width=0)
    sub_graph.set_fill(fill_color, fill_opacity)

    return sub_graph


class Axes(VGroup, CoordinateSystem):
    """坐标轴类，继承自VGroup（向量对象组）和CoordinateSystem（坐标系）"""
    # 默认轴配置
    default_axis_config: dict = dict()
    # 默认X轴配置
    default_x_axis_config: dict = dict()
    # 默认Y轴配置（数字标签方向向左）
    default_y_axis_config: dict = dict(line_to_number_direction=LEFT)

    def __init__(
        self,
        x_range: RangeSpecifier = DEFAULT_X_RANGE,
        y_range: RangeSpecifier = DEFAULT_Y_RANGE,
        axis_config: dict = dict(),
        x_axis_config: dict = dict(),
        y_axis_config: dict = dict(),
        height: float | None = None,
        width: float | None = None,
        unit_size: float = 1.0,
        **kwargs
    ):
        """
        初始化坐标轴
        
        参数:
            x_range: X轴范围(x_min, x_max, step)
            y_range: Y轴范围(y_min, y_max, step)
            axis_config: 通用轴配置
            x_axis_config: X轴特定配置
            y_axis_config: Y轴特定配置
            height: Y轴长度
            width: X轴长度
            unit_size: 单位长度的像素大小
            **kwargs: 传递给VGroup的参数
        """
        # 初始化父类CoordinateSystem
        CoordinateSystem.__init__(self, x_range, y_range,** kwargs)
        # 移除不需要的参数
        kwargs.pop("num_sampled_graph_points_per_tick", None)
        # 初始化父类VGroup
        VGroup.__init__(self, **kwargs)

        # 合并轴配置（包含单位大小）
        axis_config = dict(** axis_config, unit_size=unit_size)
        # 创建X轴
        self.x_axis = self.create_axis(
            self.x_range,
            # 合并配置：默认配置 → X轴默认 → 通用配置 → X轴特定
            axis_config=merge_dicts_recursively(
                self.default_axis_config,
                self.default_x_axis_config,
                axis_config,
                x_axis_config
            ),
            length=width,  # X轴长度
        )
        # 创建Y轴
        self.y_axis = self.create_axis(
            self.y_range,
            # 合并配置：默认配置 → Y轴默认 → 通用配置 → Y轴特定
            axis_config=merge_dicts_recursively(
                self.default_axis_config,
                self.default_y_axis_config,
                axis_config,
                y_axis_config
            ),
            length=height,  # Y轴长度
        )
        # Y轴旋转90度（垂直方向），旋转中心为原点
        self.y_axis.rotate(90 * DEG, about_point=ORIGIN)
        # 将坐标轴组合成组（便于统一管理）
        self.axes = VGroup(self.x_axis, self.y_axis)
        # 将坐标轴添加到当前对象
        self.add(*self.axes)
        # 居中显示
        self.center()

    def create_axis(
        self,
        range_terms: RangeSpecifier,
        axis_config: dict,
        length: float | None
    ) -> NumberLine:
        """
        创建单个坐标轴（X轴或Y轴）
        
        参数:
            range_terms: 轴范围
            axis_config: 轴配置
            length: 轴长度
        返回:
            坐标轴对象（NumberLine）
        """
        # 创建数轴对象
        axis = NumberLine(range_terms, width=length, **axis_config)
        # 将轴的原点（0点）移动到场景原点
        axis.shift(-axis.n2p(0))
        return axis

    def coords_to_point(self, *coords: float | VectN) -> Vect3 | Vect3Array:
        origin = self.x_axis.number_to_point(0)
        return origin + sum(
            axis.number_to_point(coord) - origin
            for axis, coord in zip(self.get_axes(), coords)
        )

    def point_to_coords(self, point: Vect3 | Vect3Array) -> tuple[float | VectN, ...]:
        return tuple([
            axis.point_to_number(point)
            for axis in self.get_axes()
        ])

    def get_axes(self) -> VGroup:
        return self.axes

    def get_all_ranges(self) -> list[Sequence[float]]:
        return [self.x_range, self.y_range]

    def add_coordinate_labels(
        self,
        x_values: Iterable[float] | None = None,
        y_values: Iterable[float] | None = None,
        excluding: Iterable[float] = [0],
        **kwargs
    ) -> VGroup:
        axes = self.get_axes()
        self.coordinate_labels = VGroup()
        for axis, values in zip(axes, [x_values, y_values]):
            labels = axis.add_numbers(values, excluding=excluding, **kwargs)
            self.coordinate_labels.add(labels)
        return self.coordinate_labels


class ThreeDAxes(Axes):
    dimension: int = 3
    default_z_axis_config: dict = dict()

    def __init__(
        self,
        x_range: RangeSpecifier = (-6.0, 6.0, 1.0),
        y_range: RangeSpecifier = (-5.0, 5.0, 1.0),
        z_range: RangeSpecifier = (-4.0, 4.0, 1.0),
        z_axis_config: dict = dict(),
        z_normal: Vect3 = DOWN,
        depth: float | None = None,
        **kwargs
    ):
        Axes.__init__(self, x_range, y_range, **kwargs)

        self.z_range = full_range_specifier(z_range)
        self.z_axis = self.create_axis(
            self.z_range,
            axis_config=merge_dicts_recursively(
                self.default_axis_config,
                self.default_z_axis_config,
                kwargs.get("axis_config", {}),
                z_axis_config
            ),
            length=depth,
        )
        self.z_axis.rotate(-PI / 2, UP, about_point=ORIGIN)
        self.z_axis.rotate(
            angle_of_vector(z_normal), OUT,
            about_point=ORIGIN
        )
        self.z_axis.shift(self.x_axis.n2p(0))
        self.axes.add(self.z_axis)
        self.add(self.z_axis)

    def get_all_ranges(self) -> list[Sequence[float]]:
        return [self.x_range, self.y_range, self.z_range]

    def add_axis_labels(self, x_tex="x", y_tex="y", z_tex="z", font_size=24, buff=0.2):
        x_label, y_label, z_label = labels = VGroup(*(
            Tex(tex, font_size=font_size)
            for tex in [x_tex, y_tex, z_tex]
        ))
        z_label.rotate(PI / 2, RIGHT)
        for label, axis in zip(labels, self):
            label.next_to(axis, normalize(np.round(axis.get_vector()), 2), buff=buff)
            axis.add(label)
        self.axis_labels = labels

    def get_graph(
        self,
        func,
        color=BLUE_E,
        opacity=0.9,
        u_range=None,
        v_range=None,
        **kwargs
    ) -> ParametricSurface:
        xu = self.x_axis.get_unit_size()
        yu = self.y_axis.get_unit_size()
        zu = self.z_axis.get_unit_size()
        x0, y0, z0 = self.get_origin()
        u_range = u_range or self.x_range[:2]
        v_range = v_range or self.y_range[:2]
        return ParametricSurface(
            lambda u, v: [xu * u + x0, yu * v + y0, zu * func(u, v) + z0],
            u_range=u_range,
            v_range=v_range,
            color=color,
            opacity=opacity,
            **kwargs
        )

    def get_parametric_surface(
        self,
        func,
        color=BLUE_E,
        opacity=0.9,
        **kwargs
    ) -> ParametricSurface:
        surface = ParametricSurface(func, color=color, opacity=opacity, **kwargs)
        axes = [self.x_axis, self.y_axis, self.z_axis]
        for dim, axis in zip(range(3), axes):
            surface.stretch(axis.get_unit_size(), dim, about_point=ORIGIN)
        surface.shift(self.get_origin())
        return surface


class NumberPlane(Axes):
    default_axis_config: dict = dict(
        stroke_color=DEFAULT_MOBJECT_COLOR,
        stroke_width=2,
        include_ticks=False,
        include_tip=False,
        line_to_number_buff=SMALL_BUFF,
        line_to_number_direction=DL,
    )
    default_y_axis_config: dict = dict(
        line_to_number_direction=DL,
    )

    def __init__(
        self,
        x_range: RangeSpecifier = (-8.0, 8.0, 1.0),
        y_range: RangeSpecifier = (-4.0, 4.0, 1.0),
        background_line_style: dict = dict(
            stroke_color=BLUE_D,
            stroke_width=2,
            stroke_opacity=1,
        ),
        # Defaults to a faded version of line_config
        faded_line_style: dict = dict(),
        faded_line_ratio: int = 4,
        make_smooth_after_applying_functions: bool = True,
        **kwargs
    ):
        super().__init__(x_range, y_range, **kwargs)
        self.background_line_style = dict(background_line_style)
        self.faded_line_style = dict(faded_line_style)
        self.faded_line_ratio = faded_line_ratio
        self.make_smooth_after_applying_functions = make_smooth_after_applying_functions
        self.init_background_lines()

    def init_background_lines(self) -> None:
        if not self.faded_line_style:
            style = dict(self.background_line_style)
            # For anything numerical, like stroke_width
            # and stroke_opacity, chop it in half
            for key in style:
                if isinstance(style[key], numbers.Number):
                    style[key] *= 0.5
            self.faded_line_style = style

        self.background_lines, self.faded_lines = self.get_lines()
        self.background_lines.set_style(**self.background_line_style)
        self.faded_lines.set_style(**self.faded_line_style)
        self.add_to_back(
            self.faded_lines,
            self.background_lines,
        )

    def get_lines(self) -> tuple[VGroup, VGroup]:
        x_axis = self.get_x_axis()
        y_axis = self.get_y_axis()

        x_lines1, x_lines2 = self.get_lines_parallel_to_axis(x_axis, y_axis)
        y_lines1, y_lines2 = self.get_lines_parallel_to_axis(y_axis, x_axis)
        lines1 = VGroup(*x_lines1, *y_lines1)
        lines2 = VGroup(*x_lines2, *y_lines2)
        return lines1, lines2

    def get_lines_parallel_to_axis(
        self,
        axis1: NumberLine,
        axis2: NumberLine
    ) -> tuple[VGroup, VGroup]:
        freq = axis2.x_step
        ratio = self.faded_line_ratio
        line = Line(axis1.get_start(), axis1.get_end())
        dense_freq = (1 + ratio)
        step = (1 / dense_freq) * freq

        lines1 = VGroup()
        lines2 = VGroup()
        inputs = np.arange(axis2.x_min, axis2.x_max + step, step)
        for i, x in enumerate(inputs):
            if abs(x) < 1e-8:
                continue
            new_line = line.copy()
            new_line.shift(axis2.n2p(x) - axis2.n2p(0))
            if i % (1 + ratio) == 0:
                lines1.add(new_line)
            else:
                lines2.add(new_line)
        return lines1, lines2

    def get_x_unit_size(self) -> float:
        return self.get_x_axis().get_unit_size()

    def get_y_unit_size(self) -> list:
        return self.get_x_axis().get_unit_size()

    def get_axes(self) -> VGroup:
        return self.axes

    def get_vector(self, coords: Iterable[float], **kwargs) -> Arrow:
        kwargs["buff"] = 0
        return Arrow(self.c2p(0, 0), self.c2p(*coords), **kwargs)

    def prepare_for_nonlinear_transform(self, num_inserted_curves: int = 50) -> Self:
        for mob in self.family_members_with_points():
            num_curves = mob.get_num_curves()
            if num_inserted_curves > num_curves:
                mob.insert_n_curves(num_inserted_curves - num_curves)
            mob.make_smooth_after_applying_functions = True
        return self


class ComplexPlane(NumberPlane):
    def number_to_point(self, number: complex | float) -> Vect3:
        number = complex(number)
        return self.coords_to_point(number.real, number.imag)

    def n2p(self, number: complex | float) -> Vect3:
        return self.number_to_point(number)

    def point_to_number(self, point: Vect3) -> complex:
        x, y = self.point_to_coords(point)
        return complex(x, y)

    def p2n(self, point: Vect3) -> complex:
        return self.point_to_number(point)

    def get_default_coordinate_values(
        self,
        skip_first: bool = True
    ) -> list[complex]:
        x_numbers = self.get_x_axis().get_tick_range()[1:]
        y_numbers = self.get_y_axis().get_tick_range()[1:]
        y_numbers = [complex(0, y) for y in y_numbers if y != 0]
        return [*x_numbers, *y_numbers]

    def add_coordinate_labels(
        self,
        numbers: list[complex] | None = None,
        skip_first: bool = True,
        font_size: int = 36,
        **kwargs
    ) -> Self:
        if numbers is None:
            numbers = self.get_default_coordinate_values(skip_first)

        self.coordinate_labels = VGroup()
        for number in numbers:
            z = complex(number)
            if abs(z.imag) > abs(z.real):
                axis = self.get_y_axis()
                value = z.imag
                kwargs["unit_tex"] = "i"
            else:
                axis = self.get_x_axis()
                value = z.real
            number_mob = axis.get_number_mobject(value, font_size=font_size, **kwargs)
            self.coordinate_labels.add(number_mob)
        self.add(self.coordinate_labels)
        return self
