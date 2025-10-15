from __future__ import annotations

# 导入抽象基类（ABC）和抽象方法（abstractmethod），用于定义接口和强制子类实现特定方法
from abc import ABC, abstractmethod

# 导入numbers模块，用于检查数值类型
import numbers

# 导入NumPy库，用于高效的数值计算和数组操作
import numpy as np

# 导入itertools模块，提供创建和操作迭代器的工具函数
import itertools as it

from manimlib.constants import BLACK, BLUE, BLUE_D, BLUE_E, GREEN, GREY_A, RED, DEFAULT_MOBJECT_COLOR
from manimlib.constants import DEG, PI
from manimlib.constants import DL, UL, DOWN, DR, LEFT, ORIGIN, OUT, RIGHT, UP
from manimlib.constants import FRAME_X_RADIUS, FRAME_Y_RADIUS
from manimlib.constants import MED_SMALL_BUFF, SMALL_BUFF
from manimlib.mobject.functions import ParametricCurve
from manimlib.mobject.geometry import Arrow
from manimlib.mobject.geometry import DashedLine
from manimlib.mobject.geometry import Line
from manimlib.mobject.geometry import Rectangle
from manimlib.mobject.number_line import NumberLine
from manimlib.mobject.svg.tex_mobject import Tex
from manimlib.mobject.types.dot_cloud import DotCloud
from manimlib.mobject.types.surface import ParametricSurface
from manimlib.mobject.types.vectorized_mobject import VGroup
from manimlib.mobject.types.vectorized_mobject import VMobject
from manimlib.utils.bezier import inverse_interpolate
from manimlib.utils.dict_ops import merge_dicts_recursively
from manimlib.utils.simple_functions import binary_search
from manimlib.utils.space_ops import angle_of_vector
from manimlib.utils.space_ops import get_norm
from manimlib.utils.space_ops import rotate_vector
from manimlib.utils.space_ops import normalize

from typing import TYPE_CHECKING

# 仅在类型检查模式下导入类型注解（避免运行时依赖）
if TYPE_CHECKING:
    from typing import Callable, Iterable, Sequence, Type, TypeVar, Optional
    from manimlib.mobject.mobject import Mobject
    from manimlib.typing import ManimColor, Vect3, Vect3Array, VectN, RangeSpecifier, Self

    T = TypeVar("T", bound=Mobject)


# 数学/数值计算相关常量
EPSILON = 1e-8  # 极小值，用于比较浮点数时避免精度问题
DEFAULT_X_RANGE = (-8.0, 8.0, 1.0)  # 默认x轴范围 (min, max, step)
DEFAULT_Y_RANGE = (-4.0, 4.0, 1.0)  # 默认y轴范围 (min, max, step)


def full_range_specifier(range_args):
    """
    确保范围参数是三元组 (min, max, step)。
    如果只提供了 (min, max)，则自动添加 step=1。

    参数:
        range_args: 范围参数，可以是二元组 (min, max) 或三元组 (min, max, step)。
    """
    if len(range_args) == 2:
        return (*range_args, 1)
    return range_args


class CoordinateSystem(ABC):
    """
    Abstract class for Axes and NumberPlane
    """
    """
    坐标系的抽象基类（ABC），为所有具体坐标系（如Axes, NumberPlane）提供统一的接口。
    它定义了坐标与点之间相互转换的核心方法，以及配置坐标轴范围的基本属性。
    """
    # 类属性：坐标系的维度，默认为2（二维平面）
    dimension: int = 2

    # 初始化坐标系的基本参数
    def __init__(
        self,
        x_range: RangeSpecifier = DEFAULT_X_RANGE,
        y_range: RangeSpecifier = DEFAULT_Y_RANGE,
        num_sampled_graph_points_per_tick: int = 5,
    ):
        # 使用辅助函数确保x_range和y_range都是三元组 (min, max, step)
        self.x_range = full_range_specifier(x_range)
        self.y_range = full_range_specifier(y_range)
        # 存储每个刻度间隔的采样点数
        self.num_sampled_graph_points_per_tick = num_sampled_graph_points_per_tick

    @abstractmethod
    # 抽象方法：将一组数学坐标（如(x, y)）转换为Manim场景中的三维空间点。
    def coords_to_point(self, *coords: float | VectN) -> Vect3 | Vect3Array:
        raise Exception("Not implemented")

    @abstractmethod
    # 抽象方法：将Manim场景中的三维空间点转换回数学坐标。
    def point_to_coords(self, point: Vect3 | Vect3Array) -> tuple[float | VectN, ...]:
        raise Exception("Not implemented")

    #  `coords_to_point` 方法的缩写，用于快速调用。
    def c2p(self, *coords: float) -> Vect3 | Vect3Array:
        """Abbreviation for coords_to_point"""
        return self.coords_to_point(*coords)

    # `point_to_coords` 方法的缩写，用于快速调用。
    def p2c(self, point: Vect3) -> tuple[float | VectN, ...]:
        """Abbreviation for point_to_coords"""
        return self.point_to_coords(point)

    # 获取坐标系原点在Manim场景中的三维点坐标。
    def get_origin(self) -> Vect3:
        return self.c2p(*[0] * self.dimension)

    @abstractmethod
    # 抽象方法：获取包含所有坐标轴的VGroup。
    def get_axes(self) -> VGroup:
        raise Exception("Not implemented")

    @abstractmethod
    # 抽象方法：获取所有坐标轴的数值范围数组。
    def get_all_ranges(self) -> list[np.ndarray]:
        raise Exception("Not implemented")

    # 根据索引获取特定的坐标轴。
    def get_axis(self, index: int) -> NumberLine:
        return self.get_axes()[index]

    # 获取x轴。
    def get_x_axis(self) -> NumberLine:
        return self.get_axis(0)

    # 获取y轴。
    def get_y_axis(self) -> NumberLine:
        return self.get_axis(1)

    # 获取z轴。
    def get_z_axis(self) -> NumberLine:
        return self.get_axis(2)

    # 为x轴创建并定位一个标签。
    def get_x_axis_label(
        self,
        label_tex: str,
        edge: Vect3 = RIGHT,
        direction: Vect3 = DL,
        **kwargs
    ) -> Tex:
        return self.get_axis_label(
            label_tex, self.get_x_axis(),
            edge, direction, **kwargs
        )

    # 为y轴创建并定位一个标签。
    def get_y_axis_label(
        self,
        label_tex: str,
        edge: Vect3 = UP,
        direction: Vect3 = DR,
        **kwargs
    ) -> Tex:
        return self.get_axis_label(
            label_tex, self.get_y_axis(),
            edge, direction, **kwargs
        )

    # 为指定坐标轴创建标签并定位。
    def get_axis_label(
        self,
        label_tex: str,
        axis: Vect3,
        edge: Vect3,
        direction: Vect3,
        buff: float = MED_SMALL_BUFF,
        ensure_on_screen: bool = False
    ) -> Tex:
        # 创建LaTeX标签对象
        label = Tex(label_tex)
        # 将标签定位在坐标轴指定端点的指定方向
        label.next_to(
            axis.get_edge_center(edge), direction,
            buff=buff
        )
        # 如果需要，将标签移至屏幕内
        if ensure_on_screen:
            label.shift_onto_screen(buff=MED_SMALL_BUFF)
        return label

    # 同时创建x轴和y轴的标签，并将它们组合成一个VGroup。
    def get_axis_labels(
        self,
        x_label_tex: str = "x",
        y_label_tex: str = "y"
    ) -> VGroup:
        # 创建并组合x轴和y轴标签
        self.axis_labels = VGroup(
            self.get_x_axis_label(x_label_tex),
            self.get_y_axis_label(y_label_tex),
        )
        return self.axis_labels

    # 从指定坐标轴向给定点绘制一条辅助线（默认是虚线）。
    # 这条线从点在坐标轴上的投影开始，延伸到点本身，用于直观展示点与坐标轴的对应关系。
    def get_line_from_axis_to_point(
        self, 
        index: int,
        point: Vect3,
        line_func: Type[T] = DashedLine,
        color: ManimColor = GREY_A,
        stroke_width: float = 2
    ) -> T:
        # 获取指定索引的坐标轴
        axis = self.get_axis(index)
        # 创建从点在坐标轴上的投影到点本身的线
        line = line_func(axis.get_projection(point), point)
        # 设置线的样式（颜色和宽度）
        line.set_stroke(color, stroke_width)
        return line

    # 从x轴向给定点绘制一条垂直线（辅助线）。
    # 是`get_line_from_axis_to_point`方法的便捷包装，固定使用x轴（索引0）。
    def get_v_line(self, point: Vect3, **kwargs):
        return self.get_line_from_axis_to_point(0, point, **kwargs)

    # 从y轴向给定点绘制一条水平线（辅助线）。
    # 是`get_line_from_axis_to_point`方法的便捷包装，固定使用y轴（索引1）。
    def get_h_line(self, point: Vect3, **kwargs):
        return self.get_line_from_axis_to_point(1, point, **kwargs)

    # Useful for graphing
    # 根据给定的函数创建并返回一个参数曲线（函数图像）。
    # 该曲线会自动适配当前坐标系的范围和转换规则。
    def get_graph(
        self,
        function: Callable[[float], float],
        x_range: Sequence[float] | None = None,
        bind: bool = False,
        **kwargs
    ) -> ParametricCurve:
        # 确定函数图像的x范围，若未指定则使用坐标系默认的x_range
        x_range = x_range or self.x_range
        # 准备参数t的范围（用于参数曲线采样）
        t_range = np.ones(3)
        # 将x_range的值复制到t_range（最多前3个元素）
        t_range[:len(x_range)] = x_range
        # For axes, the third coordinate of x_range indicates
        # tick frequency.  But for functions, it indicates a
        # sample frequency
        # 调整采样步长：原x_range的步长表示刻度间隔，这里转换为采样间隔
        # 通过除以每个刻度的采样点数，使采样更密集，曲线更平滑
        t_range[2] /= self.num_sampled_graph_points_per_tick

        # 定义参数曲线函数：将t（即x值）转换为坐标系中的三维点
        def parametric_function(t: float) -> Vect3:
            # 先计算函数值y = function(t)，再通过c2p转换为场景中的点坐标
            return self.c2p(t, function(t))

        # 创建参数曲线对象
        graph = ParametricCurve(
            parametric_function,
            t_range=tuple(t_range),  # 采样范围和步长
            **kwargs
        )
        # 为曲线添加额外属性，存储原始函数和x范围
        graph.underlying_function = function
        graph.x_range = x_range

        # 如果需要绑定，则将曲线与函数关联（通常用于动态更新）
        if bind:
            self.bind_graph_to_func(graph, function)

        return graph

    # 创建一条参数曲线，该曲线会自动适配当前坐标系的坐标转换规则。
    # 适用于绘制参数方程定义的曲线（如圆、椭圆、螺旋线等）。
    def get_parametric_curve(
        self,
        function: Callable[[float], Vect3],
        **kwargs
    ) -> ParametricCurve:
        # 获取坐标系的维度（通常为2，三维坐标系为3）
        dim = self.dimension
        # 创建参数曲线：将函数返回的坐标通过坐标系转换为场景中的点
        # 截取与坐标系维度匹配的坐标分量（例如2D时忽略z分量）
        graph = ParametricCurve(
            lambda t: self.coords_to_point(*function(t)[:dim]),
            **kwargs
        )
        # 存储原始参数函数，便于后续引用或动态更新
        graph.underlying_function = function
        return graph

    # 根据给定的x值，找到函数图像上对应的点在场景中的坐标。
    # 支持两种方式：直接计算（若曲线绑定了原始函数）或通过二分查找（通用方法）。
    def input_to_graph_point(
        self,
        x: float,
        graph: ParametricCurve
    ) -> Vect3 | None:
        # 情况1：如果曲线存储了原始函数，直接计算对应点
        if hasattr(graph, "underlying_function"):
            return self.coords_to_point(x, graph.underlying_function(x))
        # 情况2：通用方法，通过二分查找在曲线上定位x对应的点
        else:
            alpha = binary_search(
                function=lambda a: self.point_to_coords(
                    graph.quick_point_from_proportion(a)
                )[0],
                target=x,
                lower_bound=self.x_range[0],  # 搜索范围下限
                upper_bound=self.x_range[1],  # 搜索范围上限
            )
            # 若找到有效比例，则返回对应点；否则返回None
            if alpha is not None:
                return graph.quick_point_from_proportion(alpha)
            else:
                return None

    # `input_to_graph_point` 方法的缩写，用于快速调用。
    def i2gp(self, x: float, graph: ParametricCurve) -> Vect3 | None:
        """
        Alias for input_to_graph_point
        """
        return self.input_to_graph_point(x, graph)

    # 将函数图像与函数绑定，使图像能随函数动态更新（适用于会随时间或条件变化的函数）。
    # 当函数发生变化时，图像会自动重新计算并更新形状。
    def bind_graph_to_func(
        self,
        graph: VMobject,
        func: Callable[[VectN], VectN],
        jagged: bool = False,
        get_discontinuities: Optional[Callable[[], Vect3]] = None
    ) -> VMobject:
        """
        Use for graphing functions which might change over time, or change with
        conditions
        """
        # 初始化x值数组：从图像当前点反推对应的x坐标（数值坐标，非场景坐标）
        x_values = np.array([self.x_axis.p2n(p) for p in graph.get_points()])

        # 定义用于更新图像点的函数
        def get_graph_points():
            # 基础x值使用初始化的数组
            xs = x_values
            # 若提供了不连续点获取函数，则在不连续点附近添加额外采样点
            if get_discontinuities:
                ds = get_discontinuities()
                ep = 1e-6  # 微小偏移量，用于在不连续点两侧采样
                # 在每个不连续点d的两侧添加d-epsilon和d+epsilon两个点
                added_xs = it.chain(*((d - ep, d + ep) for d in ds))
                # 将新增点与原有x值合并，排序后截取与原长度相同的点（保持采样密度）
                xs[:] = sorted([*x_values, *added_xs])[:len(x_values)]
            # 计算所有x对应的y值，并转换为场景中的点坐标
            return self.c2p(xs, func(xs))

        # 为图像添加更新器：每帧重新计算并设置图像的顶点
        graph.add_updater(
            lambda g: g.set_points_as_corners(get_graph_points())
        )
        # 若不需要锯齿状边缘，则添加平滑处理的更新器
        if not jagged:
            graph.add_updater(lambda g: g.make_smooth(approx=True))
        return graph

    # 为函数图像添加标签，并自动定位在合适的位置。
    def get_graph_label(
        self,
        graph: ParametricCurve,
        label: str | Mobject = "f(x)",
        x: float | None = None,
        direction: Vect3 = RIGHT,
        buff: float = MED_SMALL_BUFF,
        color: ManimColor | None = None
    ) -> Tex | Mobject:
        # 1. 处理标签对象：若为字符串则转换为Tex对象
        if isinstance(label, str):
            label = Tex(label)
        # 2. 设置标签颜色：默认与图像颜色一致
        if color is None:
            label.match_color(graph)
        # 3. 自动确定标签位置的x坐标（如果未指定）
        if x is None:
            # Searching from the right, find a point
            # whose y value is in bounds
            # 定义屏幕范围内的最大y和x值（留出标签空间）
            max_y = FRAME_Y_RADIUS - label.get_height()
            max_x = FRAME_X_RADIUS - label.get_width()
            # 从右向左搜索合适的x位置（优先右侧）
            # 寻找图像上y值在屏幕范围内的点
            for x0 in np.arange(*self.x_range)[::-1]:
                pt = self.i2gp(x0, graph)  # 获取x0对应的图像点
                # 检查点是否在屏幕范围内
                if abs(pt[0]) < max_x and abs(pt[1]) < max_y:
                    x = x0
                    break
            # 若未找到合适位置，默认放在x轴范围的右端点
            if x is None:
                x = self.x_range[1]

        # 4. 获取标签附着点在图像上的坐标
        point = self.input_to_graph_point(x, graph)
        # 5. 计算图像在该点的切线角度，确定标签的法线方向（垂直于切线）
        angle = self.angle_of_tangent(x, graph)
        normal = rotate_vector(RIGHT, angle + 90 * DEG)
        # 6. 确保法线方向向上（避免标签出现在图像下方）
        if normal[1] < 0:
            normal *= -1
        # 7. 定位标签：沿法线方向放置在附着点旁边
        label.next_to(point, normal, buff=buff)
        # 8. 确保标签在屏幕内
        label.shift_onto_screen()
        return label

    # 从x轴绘制一条垂直线到函数图像上指定x值对应的点。
    def get_v_line_to_graph(self, x: float, graph: ParametricCurve, **kwargs):
        # 先获取x对应的图像点，再绘制垂直线
        return self.get_v_line(self.i2gp(x, graph), **kwargs)

    # 从y轴绘制一条水平线到函数图像上指定x值对应的点。
    def get_h_line_to_graph(self, x: float, graph: ParametricCurve, **kwargs):
        # 先获取x对应的图像点，再绘制水平线
        return self.get_h_line(self.i2gp(x, graph), **kwargs)

    # 根据给定的x和y值数组创建散点图。
    def get_scatterplot(self,
                        x_values: Vect3Array,
                        y_values: Vect3Array,
                        **dot_config):
        # 将(x, y)坐标转换为场景中的点，再创建点云
        return DotCloud(self.c2p(x_values, y_values), **dot_config)

    # For calculus
    # # 微积分相关工具方法
    # 计算函数图像在指定x处的切线与水平方向的夹角（弧度）。
    def angle_of_tangent(
        self,
        x: float,
        graph: ParametricCurve,
        dx: float = EPSILON
    ) -> float:
        # 获取x和x+dx处的图像点
        p0 = self.input_to_graph_point(x, graph)
        p1 = self.input_to_graph_point(x + dx, graph)
        # 计算两点连线（近似切线）的角度
        return angle_of_vector(p1 - p0)

    # 计算函数图像在指定x处的切线斜率。
    def slope_of_tangent(
        self,
        x: float,
        graph: ParametricCurve,
        **kwargs
    ) -> float:
        # 斜率 = tan(切线角度)
        return np.tan(self.angle_of_tangent(x, graph, **kwargs))

    # 计算并返回函数图像在指定x处的切线。
    def get_tangent_line(
        self,
        x: float,
        graph: ParametricCurve,
        length: float = 5,
        line_func: Type[T] = Line
    ) -> T:
        # 创建一条水平线段作为基础
        line = line_func(LEFT, RIGHT)
        # 设置线段长度
        line.set_width(length)
        # 按照切线角度旋转线段
        line.rotate(self.angle_of_tangent(x, graph))
        # 将线段移动到图像上的目标点
        line.move_to(self.input_to_graph_point(x, graph))
        return line

    # 创建用于黎曼求和的矩形组，可视化函数与x轴之间的面积近似。
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
        # 处理x范围和步长
        if x_range is None:
            x_range = self.x_range[:2]
        if dx is None:
            dx = self.x_range[2]
        if len(x_range) < 3:
            x_range = [*x_range, dx]  # 确保x_range是三元组 (min, max, dx)

        rects = []
        # 扩展x范围上限，确保最后一个矩形完整
        x_range[1] = x_range[1] + dx
        # 生成所有矩形的左端点
        xs = np.arange(*x_range)
        # 逐个创建矩形
        for x0, x1 in zip(xs, xs[1:]):
            # 根据采样方式确定矩形高度的采样点
            if input_sample_type == "left":
                sample = x0  # 左端点采样
            elif input_sample_type == "right":
                sample = x1  # 右端点采样
            elif input_sample_type == "center":
                sample = 0.5 * x0 + 0.5 * x1  # 中点采样
            else:
                raise Exception("Invalid input sample type")
            
            # 计算矩形高度（函数值对应的场景距离）
            # 从x轴上的点到图像上的点的向量
            height_vect = self.i2gp(sample, graph) - self.c2p(sample, 0)
            # 创建矩形
            rect = Rectangle(
                # 宽度：x1和x0在场景中的距离
                width=self.x_axis.n2p(x1)[0] - self.x_axis.n2p(x0)[0],
                # 高度：函数值向量的长度
                height=get_norm(height_vect),
            )
            # 标记矩形是否在x轴上方（正值区域）
            rect.positive = height_vect[1] > 0
            # 定位矩形：根据正负区域决定锚点
            rect.move_to(self.c2p(x0, 0), DL if rect.positive else UL)
            rects.append(rect)
        # 将所有矩形组合成VGroup
        result = VGroup(*rects)
        # 设置矩形的渐变色
        result.set_submobject_colors_by_gradient(*colors)
        # 设置矩形的基础样式
        result.set_style(
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            fill_opacity=fill_opacity,
            stroke_behind=stroke_background  # 边框在填充下方
        )
        # 对负区域的矩形设置特殊颜色
        for rect in result:
            if not rect.positive:
                rect.set_fill(negative_color)
        return result

    # 计算并创建函数图像与x轴在指定x范围内围成的填充区域。
    def get_area_under_graph(self, graph, x_range, fill_color=BLUE, fill_opacity=0.5):
        # 检查graph是否有x_range属性（用于后续计算比例）
        if not hasattr(graph, "x_range"):
            raise Exception("Argument `graph` must have attribute `x_range`")

        # 1. 将x轴范围转换为图像上的比例范围（0到1之间）
        # inverse_interpolate用于计算x在graph.x_range中的相对位置
        alpha_bounds = [
            inverse_interpolate(*graph.x_range, x)
            for x in x_range
        ]
        # 2. 提取指定x范围内的子图像
        sub_graph = graph.copy()
        # 只保留图像在alpha_bounds比例范围内的部分
        sub_graph.pointwise_become_partial(graph, *alpha_bounds)
        # 3. 连接子图像与x轴，形成闭合区域
        sub_graph.add_line_to(self.c2p(x_range[1], 0))  # 从子图像终点连到(x_max, 0)
        sub_graph.add_line_to(self.c2p(x_range[0], 0))  # 从(x_max, 0)连到(x_min, 0)
        sub_graph.add_line_to(sub_graph.get_start())    # 从(x_min, 0)连回子图像起点，闭合区域

        # 4. 设置填充样式（隐藏边框，添加填充色和透明度）
        sub_graph.set_stroke(width=0)  # 边框宽度设为0，不显示边框
        sub_graph.set_fill(fill_color, fill_opacity)  # 设置填充色和透明度

        return sub_graph


class Axes(VGroup, CoordinateSystem):
    """
    具体的坐标轴类，继承自复合图形类VGroup和坐标系抽象类CoordinateSystem。
    用于在Manim场景中创建带刻度、标签的标准二维坐标轴，支持坐标转换、函数绘制等功能。
    """
    # 坐标轴的默认配置字典，用于统一设置轴的样式（如颜色、线宽等）
    default_axis_config: dict = dict()
    # x轴的专属默认配置，会覆盖default_axis_config中的同名配置
    default_x_axis_config: dict = dict()
    # y轴的专属默认配置，默认设置数值标签在轴的左侧（line_to_number_direction=LEFT）
    default_y_axis_config: dict = dict(line_to_number_direction=LEFT)

    # 初始化Axes对象，创建带有刻度和标签的二维坐标轴。
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
        # 1. 初始化父类CoordinateSystem（处理坐标范围等核心属性）
        CoordinateSystem.__init__(self, x_range, y_range, **kwargs)
        # 移除可能从父类传递过来的不相关参数
        kwargs.pop("num_sampled_graph_points_per_tick", None)
        # 2. 初始化父类VGroup（作为复合图形容器）
        VGroup.__init__(self, **kwargs)

        # 3. 准备坐标轴配置（合并默认配置与用户配置）
        # 基础配置中加入单位大小
        axis_config = dict(**axis_config, unit_size=unit_size)
        # 创建x轴：合并多层配置（默认通用配置 → 默认x轴配置 → 通用用户配置 → x轴用户配置）
        self.x_axis = self.create_axis(
            self.x_range,
            axis_config=merge_dicts_recursively(
                self.default_axis_config,
                self.default_x_axis_config,
                axis_config,
                x_axis_config
            ),
            length=width,  # 指定x轴长度
        )
        # 创建y轴：类似x轴，但使用y轴专属配置
        self.y_axis = self.create_axis(
            self.y_range,
            axis_config=merge_dicts_recursively(
                self.default_axis_config,
                self.default_y_axis_config,
                axis_config,
                y_axis_config
            ),
            length=height,  # 指定y轴长度
        )
        # 4. 旋转y轴使其垂直于x轴（绕原点旋转90度）
        self.y_axis.rotate(90 * DEG, about_point=ORIGIN)
        # Add as a separate group in case various other
        # mobjects are added to self, as for example in
        # NumberPlane below
        # 5. 将坐标轴组合成单独的VGroup（便于统一管理）
        # 这样即使后续添加其他图形，坐标轴仍能作为整体被操作
        self.axes = VGroup(self.x_axis, self.y_axis)
        # 6. 将坐标轴添加到当前Axes对象中，并居中显示
        self.add(*self.axes)
        self.center()

    def create_axis(
        self,
        range_terms: RangeSpecifier,
        axis_config: dict,
        length: float | None
    ) -> NumberLine:
        # 创建数轴对象，参数包括范围、长度和其他轴配置
        axis = NumberLine(range_terms, width=length,** axis_config)
        # 将数轴平移，使数轴上的0点与原点对齐（消除初始偏移）
        axis.shift(-axis.n2p(0))  # n2p(0)表示0对应的点坐标，取反后平移实现对齐
        return axis

    def coords_to_point(self, *coords: float | VectN) -> Vect3 | Vect3Array:
        # 获取x轴上0点对应的坐标作为原点基准
        origin = self.x_axis.number_to_point(0)
        # 计算坐标对应的空间点：
        # 1. 遍历每个轴和对应的坐标
        # 2. 将每个轴上的坐标转换为点坐标
        # 3. 减去原点后累加，最后加上原点得到最终点
        return origin + sum(
            axis.number_to_point(coord) - origin
            for axis, coord in zip(self.get_axes(), coords)
        )

    def point_to_coords(self, point: Vect3 | Vect3Array) -> tuple[float | VectN, ...]:
        # 将空间点转换为各轴对应的坐标：
        # 对每个轴调用point_to_number方法，得到点在该轴上的坐标值
        return tuple([
            axis.point_to_number(point)
            for axis in self.get_axes()
        ])

    def get_axes(self) -> VGroup:
        # 返回包含所有轴的组（VGroup），作为获取轴的统一接口
        return self.axes

    def get_all_ranges(self) -> list[Sequence[float]]:
        # 返回包含x轴和y轴范围的列表，用于快速获取所有轴的取值范围
        return [self.x_range, self.y_range]

    def add_coordinate_labels(
        self,
        x_values: Iterable[float] | None = None,
        y_values: Iterable[float] | None = None,
        excluding: Iterable[float] = [0],
        **kwargs
    ) -> VGroup:
        # 获取所有轴对象（如x轴、y轴）
        axes = self.get_axes()
        # 创建用于存储坐标标签的组
        self.coordinate_labels = VGroup()
        # 遍历每个轴和对应的坐标值列表（x_values对应x轴，y_values对应y轴）
        for axis, values in zip(axes, [x_values, y_values]):
            # 为当前轴添加数字标签：
            # - values指定要显示的坐标值，None则使用轴默认值
            # - excluding指定不显示标签的坐标（默认排除0点，避免与原点标记冲突）
            # - **kwargs传递额外的标签样式配置（如字体、颜色等）
            labels = axis.add_numbers(values, excluding=excluding,** kwargs)
            # 将生成的标签添加到坐标标签组中
            self.coordinate_labels.add(labels)
        return self.coordinate_labels


class ThreeDAxes(Axes):
    # 三维坐标系的维度标识为3
    dimension: int = 3
    # 三维坐标系中z轴的默认配置字典（可包含长度、颜色、刻度等参数）
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
        # 调用父类Axes的初始化方法，传入x轴和y轴范围及其他参数
        Axes.__init__(self, x_range, y_range, **kwargs)

        # 处理z轴范围，确保其格式符合RangeSpecifier规范（起始值、结束值、步长）
        self.z_range = full_range_specifier(z_range)
        # 创建z轴：
        # 1. 合并轴配置（默认配置→z轴默认配置→全局轴配置→z轴专用配置，后者覆盖前者）
        # 2. 使用create_axis方法生成数轴，长度由depth参数指定
        self.z_axis = self.create_axis(
            self.z_range,
            axis_config=merge_dicts_recursively(
                self.default_axis_config,          # 基础默认配置
                self.default_z_axis_config,        # z轴默认配置
                kwargs.get("axis_config", {}),     # 全局轴配置（如果有）
                z_axis_config                      # z轴专用配置（优先级最高）
            ),
            length=depth,
        )
        # 旋转z轴：
        # 1. 绕UP方向旋转-90度（PI/2弧度），以调整初始朝向
        # 2. 绕OUT方向旋转，角度由z_normal向量决定，使z轴与指定法向量对齐
        self.z_axis.rotate(-PI / 2, UP, about_point=ORIGIN)
        self.z_axis.rotate(
            angle_of_vector(z_normal), OUT,
            about_point=ORIGIN
        )
        # 将z轴平移到x轴的0点位置，确保三轴在原点交汇
        self.z_axis.shift(self.x_axis.n2p(0))
    
        # 将z轴添加到轴组和当前三维坐标系中
        self.axes.add(self.z_axis)
        self.add(self.z_axis)

    def get_all_ranges(self) -> list[Sequence[float]]:
        # 重写父类方法，返回包含x、y、z三轴范围的列表（适配三维坐标系）
        return [self.x_range, self.y_range, self.z_range]

    def add_axis_labels(self, x_tex="x", y_tex="y", z_tex="z", font_size=24, buff=0.2):
        # 生成x、y、z轴的标签文本（默认显示"x"、"y"、"z"），并组合成VGroup
        # 用Tex创建公式文本对象，统一设置字体大小
        x_label, y_label, z_label = labels = VGroup(*(
            Tex(tex, font_size=font_size)
            for tex in [x_tex, y_tex, z_tex]
        ))
        # 旋转z轴标签：绕RIGHT方向旋转90度（PI/2弧度），使其与z轴朝向匹配
        z_label.rotate(PI / 2, RIGHT)
        # 遍历每个标签与对应的轴，完成标签定位与绑定
        for label, axis in zip(labels, self):
            # 将标签放在轴的对应方向旁，设置间距为buff
            label.next_to(axis, normalize(np.round(axis.get_vector()), 2), buff=buff)
            # 将标签添加到对应轴上，确保标签随轴移动
            axis.add(label)
        # 存储轴标签组，便于后续管理
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
        # 获取三个轴的单位尺寸（每个单位对应的实际长度）
        xu = self.x_axis.get_unit_size()
        yu = self.y_axis.get_unit_size()
        zu = self.z_axis.get_unit_size()
        # 获取坐标系原点的三维坐标（x0, y0, z0）
        x0, y0, z0 = self.get_origin()
        # 确定参数u、v的范围：默认使用x轴、y轴的起始-结束范围（忽略步长）
        u_range = u_range or self.x_range[:2]
        v_range = v_range or self.y_range[:2]
        # 创建并返回参数化曲面（即函数func对应的三维图形）
        return ParametricSurface(
            # 曲面参数方程：将(u, v)映射为三维点，结合轴单位尺寸和原点偏移
            lambda u, v: [xu * u + x0, yu * v + y0, zu * func(u, v) + z0],
            u_range=u_range,    # u参数范围（对应x轴方向）
            v_range=v_range,    # v参数范围（对应y轴方向）
            color=color,        # 曲面颜色（默认蓝绿色BLUE_E）
            opacity=opacity,    # 曲面透明度（默认0.9）
            **kwargs            # 传递额外参数（如分辨率、边框等）
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
