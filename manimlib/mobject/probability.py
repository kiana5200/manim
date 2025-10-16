# 从__future__导入annotations，用于支持Python 3.7及以下版本的类型提示语法
from __future__ import annotations

# 导入numpy库，用于数值计算和数组操作
import numpy as np

# 从manimlib.constants导入常用颜色常量
from manimlib.constants import BLUE, BLUE_E, GREEN_E, GREY_B, GREY_D, MAROON_B, YELLOW
# 从manimlib.constants导入方向向量常量
from manimlib.constants import DOWN, LEFT, RIGHT, UP
# 从manimlib.constants导入间距常量
from manimlib.constants import MED_LARGE_BUFF, MED_SMALL_BUFF, SMALL_BUFF
# 从manimlib.mobject.geometry导入几何图形类
from manimlib.mobject.geometry import Line
from manimlib.mobject.geometry import Rectangle
# 从manimlib.mobject导入Mobject基类，所有可显示对象的父类
from manimlib.mobject.mobject import Mobject
# 从manimlib.mobject.svg.brace导入Brace类，用于创建花括号标记
from manimlib.mobject.svg.brace import Brace
# 从manimlib.mobject.svg.tex_mobject导入Tex和TexText类，用于处理LaTeX文本
from manimlib.mobject.svg.tex_mobject import Tex
from manimlib.mobject.svg.tex_mobject import TexText
# 从manimlib.mobject.types.vectorized_mobject导入VGroup类，用于组合多个可显示对象
from manimlib.mobject.types.vectorized_mobject import VGroup
# 从manimlib.utils.color导入color_gradient函数，用于创建颜色渐变
from manimlib.utils.color import color_gradient
# 从manimlib.utils.iterables导入listify函数，用于将输入转换为列表
from manimlib.utils.iterables import listify

# 导入typing模块中的TYPE_CHECKING常量，用于条件类型检查
from typing import TYPE_CHECKING

# 如果是类型检查阶段（非运行时），导入所需的类型提示
if TYPE_CHECKING:
    from typing import Iterable
    from manimlib.typing import ManimColor

# 定义一个极小值epsilon，用于避免浮点数计算中的精度问题
EPSILON = 0.0001


class SampleSpace(Rectangle):
    """样本空间类，继承自Rectangle，用于在Manim中创建概率样本空间的可视化表示"""
    
    def __init__(
        self,
        width: float = 3,          # 样本空间矩形的宽度，默认值为3
        height: float = 3,         # 样本空间矩形的高度，默认值为3
        fill_color: ManimColor = GREY_D,  # 填充颜色，默认为深灰色
        fill_opacity: float = 1,   # 填充不透明度，默认完全不透明
        stroke_width: float = 0.5, # 边框宽度，默认0.5
        stroke_color: ManimColor = GREY_B,  # 边框颜色，默认为中灰色
        default_label_scale_val: float = 1, # 默认标签缩放值
        **kwargs,                  # 其他关键字参数，传递给父类
    ):
        # 调用父类Rectangle的构造函数，初始化基本矩形属性
        super().__init__(
            width, height,
            fill_color=fill_color,
            fill_opacity=fill_opacity,
            stroke_width=stroke_width,
            stroke_color=stroke_color,** kwargs
        )
        # 存储默认标签缩放值
        self.default_label_scale_val = default_label_scale_val

    def add_title(
        self,
        title: str = "Sample space",  # 标题文本，默认"Sample space"
        buff: float = MED_SMALL_BUFF  # 与样本空间的距离，默认中等小间距
    ) -> None:
        # TODO, 这个方法是否应该存在于SampleSpaceScene中
        # 创建标题文本对象
        title_mob = TexText(title)
        # 如果标题宽度超过样本空间宽度，则调整标题宽度
        if title_mob.get_width() > self.get_width():
            title_mob.set_width(self.get_width())
        # 将标题放置在样本空间上方
        title_mob.next_to(self, UP, buff=buff)
        # 存储标题对象并添加到场景中
        self.title = title_mob
        self.add(title_mob)

    def add_label(self, label: str) -> None:
        """为样本空间添加标签"""
        self.label = label

    def complete_p_list(self, p_list: list[float]) -> list[float]:
        """
        补全概率列表，确保概率和为1
        
        参数:
            p_list: 概率列表
        
        返回:
            补全后的概率列表，总和为1
        """
        # 将输入转换为列表（确保是列表类型）
        new_p_list = listify(p_list)
        # 计算剩余概率（1减去现有概率总和）
        remainder = 1.0 - sum(new_p_list)
        # 如果剩余概率大于极小值（考虑浮点数误差），则添加到列表中
        if abs(remainder) > EPSILON:
            new_p_list.append(remainder)
        return new_p_list

    def get_division_along_dimension(
        self,
        p_list: list[float],        # 概率列表，决定每个部分的比例
        dim: int,                   # 划分维度，0为水平，1为垂直
        colors: Iterable[ManimColor],  # 颜色迭代器，用于每个部分的填充色
        vect: np.ndarray            # 方向向量，指示划分方向
    ) -> VGroup:
        """沿指定维度按概率比例划分样本空间"""
        # 补全概率列表，确保总和为1
        p_list = self.complete_p_list(p_list)
        # 根据概率列表长度创建颜色渐变
        colors = color_gradient(colors, len(p_list))

        # 获取起始点（沿反方向向量的边缘中心）
        last_point = self.get_edge_center(-vect)
        # 创建用于存储所有部分的VGroup
        parts = VGroup()
        
        # 遍历每个概率和对应的颜色
        for factor, color in zip(p_list, colors):
            # 创建新的样本空间部分
            part = SampleSpace()
            # 设置部分的填充颜色和不透明度
            part.set_fill(color, 1)
            # 替换为当前样本空间的大小和位置
            part.replace(self, stretch=True)
            # 沿指定维度按概率因子拉伸
            part.stretch(factor, dim)
            # 移动到正确位置
            part.move_to(last_point, -vect)
            # 更新下一个部分的起始点
            last_point = part.get_edge_center(vect)
            # 将部分添加到VGroup
            parts.add(part)
        
        return parts

    def get_horizontal_division(
        self,
        p_list: list[float],        # 概率列表
        colors: Iterable[ManimColor] = [GREEN_E, BLUE_E],  # 默认颜色列表
        vect: np.ndarray = DOWN     # 默认方向向量为向下
    ) -> VGroup:
        """获取水平方向的划分（按高度划分）"""
        return self.get_division_along_dimension(p_list, 1, colors, vect)

    def get_vertical_division(
        self,
        p_list: list[float],        # 概率列表
        colors: Iterable[ManimColor] = [MAROON_B, YELLOW],  # 默认颜色列表
        vect: np.ndarray = RIGHT    # 默认方向向量为向右
    ) -> VGroup:
        """获取垂直方向的划分（按宽度划分）"""
        return self.get_division_along_dimension(p_list, 0, colors, vect)

    def divide_horizontally(self, *args, **kwargs) -> None:
        """执行水平划分并将结果添加到当前对象"""
        self.horizontal_parts = self.get_horizontal_division(*args, **kwargs)
        self.add(self.horizontal_parts)

    def divide_vertically(self, *args, **kwargs) -> None:
        """执行垂直划分并将结果添加到当前对象"""
        self.vertical_parts = self.get_vertical_division(*args, **kwargs)
        self.add(self.vertical_parts)

    def get_subdivision_braces_and_labels(
        self,
        parts: VGroup,              # 要添加花括号和标签的部分组
        labels: str,                # 标签文本
        direction: np.ndarray,      # 花括号和标签的方向
        buff: float = SMALL_BUFF,   # 间距，默认小间距
    ) -> VGroup:
        """为划分的部分添加花括号和标签"""
        # 创建存储标签和花括号的VGroup
        label_mobs = VGroup()
        braces = VGroup()
        
        # 为每个部分添加花括号和标签
        for label, part in zip(labels, parts):
            # 创建花括号
            brace = Brace(
                part, direction,
                buff=buff
            )
            # 处理标签，如果是Mobject则直接使用，否则创建Tex对象
            if isinstance(label, Mobject):
                label_mob = label
            else:
                label_mob = Tex(label)
                label_mob.scale(self.default_label_scale_val)
            # 将标签放置在花括号的指定方向
            label_mob.next_to(brace, direction, buff)

            # 添加花括号和标签到对应的组
            braces.add(brace)
            label_mobs.add(label_mob)
        
        # 将花括号和标签存储在parts对象中
        parts.braces = braces
        parts.labels = label_mobs
        parts.label_kwargs = {
            "labels": label_mobs.copy(),
            "direction": direction,
            "buff": buff,
        }
        
        return VGroup(parts.braces, parts.labels)

    def get_side_braces_and_labels(
        self,
        labels: str,                # 标签文本
        direction: np.ndarray = LEFT,  # 方向，默认为左
        **kwargs
    ) -> VGroup:
        """为水平划分的部分添加侧边花括号和标签"""
        # 确保对象已进行水平划分
        assert hasattr(self, "horizontal_parts")
        parts = self.horizontal_parts
        return self.get_subdivision_braces_and_labels(parts, labels, direction,** kwargs)

    def get_top_braces_and_labels(
        self,
        labels: str,                # 标签文本
        **kwargs
    ) -> VGroup:
        """为垂直划分的部分添加顶部花括号和标签"""
        # 确保对象已进行垂直划分
        assert hasattr(self, "vertical_parts")
        parts = self.vertical_parts
        return self.get_subdivision_braces_and_labels(parts, labels, UP,** kwargs)

    def get_bottom_braces_and_labels(
        self,
        labels: str,                # 标签文本
        **kwargs
    ) -> VGroup:
        """为垂直划分的部分添加底部花括号和标签"""
        # 确保对象已进行垂直划分
        assert hasattr(self, "vertical_parts")
        parts = self.vertical_parts
        return self.get_subdivision_braces_and_labels(parts, labels, DOWN,** kwargs)

    def add_braces_and_labels(self) -> None:
        """将已创建的花括号和标签添加到场景中"""
        # 检查水平和垂直划分部分
        for attr in "horizontal_parts", "vertical_parts":
            if not hasattr(self, attr):
                continue
            parts = getattr(self, attr)
            # 添加花括号和标签
            for subattr in "braces", "labels":
                if hasattr(parts, subattr):
                    self.add(getattr(parts, subattr))

    def __getitem__(self, index: int | slice) -> VGroup:
        """
        重载索引运算符，允许通过索引访问划分的部分
        
        参数:
            index: 索引或切片
        
        返回:
            对应的部分或部分组
        """
        if hasattr(self, "horizontal_parts"):
            return self.horizontal_parts[index]
        elif hasattr(self, "vertical_parts"):
            return self.vertical_parts[index]
        # 如果没有划分，则返回分割后的部分
        return self.split()[index]


class BarChart(VGroup):
    def __init__(
        self,
        values: Iterable[float],
        height: float = 4,
        width: float = 6,
        n_ticks: int = 4,
        include_x_ticks: bool = False,
        tick_width: float = 0.2,
        tick_height: float = 0.15,
        label_y_axis: bool = True,
        y_axis_label_height: float = 0.25,
        max_value: float = 1,
        bar_colors: list[ManimColor] = [BLUE, YELLOW],
        bar_fill_opacity: float = 0.8,
        bar_stroke_width: float = 3,
        bar_names: list[str] = [],
        bar_label_scale_val: float = 0.75,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.height = height
        self.width = width
        self.n_ticks = n_ticks
        self.include_x_ticks = include_x_ticks
        self.tick_width = tick_width
        self.tick_height = tick_height
        self.label_y_axis = label_y_axis
        self.y_axis_label_height = y_axis_label_height
        self.max_value = max_value
        self.bar_colors = bar_colors
        self.bar_fill_opacity = bar_fill_opacity
        self.bar_stroke_width = bar_stroke_width
        self.bar_names = bar_names
        self.bar_label_scale_val = bar_label_scale_val

        if self.max_value is None:
            self.max_value = max(values)

        self.n_ticks_x = len(values)
        self.add_axes()
        self.add_bars(values)
        self.center()

    def add_axes(self) -> None:
        x_axis = Line(self.tick_width * LEFT / 2, self.width * RIGHT)
        y_axis = Line(MED_LARGE_BUFF * DOWN, self.height * UP)
        y_ticks = VGroup()
        heights = np.linspace(0, self.height, self.n_ticks + 1)
        values = np.linspace(0, self.max_value, self.n_ticks + 1)
        for y, value in zip(heights, values):
            y_tick = Line(LEFT, RIGHT)
            y_tick.set_width(self.tick_width)
            y_tick.move_to(y * UP)
            y_ticks.add(y_tick)
        y_axis.add(y_ticks)

        if self.include_x_ticks == True:
            x_ticks = VGroup()
            widths = np.linspace(0, self.width, self.n_ticks_x + 1)
            label_values = np.linspace(0, len(self.bar_names), self.n_ticks_x + 1)
            for x, value in zip(widths, label_values):
                x_tick = Line(UP, DOWN)
                x_tick.set_height(self.tick_height)
                x_tick.move_to(x * RIGHT)
                x_ticks.add(x_tick)
            x_axis.add(x_ticks)

        self.add(x_axis, y_axis)
        self.x_axis, self.y_axis = x_axis, y_axis

        if self.label_y_axis:
            labels = VGroup()
            for y_tick, value in zip(y_ticks, values):
                label = Tex(str(np.round(value, 2)))
                label.set_height(self.y_axis_label_height)
                label.next_to(y_tick, LEFT, SMALL_BUFF)
                labels.add(label)
            self.y_axis_labels = labels
            self.add(labels)

    def add_bars(self, values: Iterable[float]) -> None:
        buff = float(self.width) / (2 * len(values))
        bars = VGroup()
        for i, value in enumerate(values):
            bar = Rectangle(
                height=(value / self.max_value) * self.height,
                width=buff,
                stroke_width=self.bar_stroke_width,
                fill_opacity=self.bar_fill_opacity,
            )
            bar.move_to((2 * i + 0.5) * buff * RIGHT, DOWN + LEFT * 5)
            bars.add(bar)
        bars.set_color_by_gradient(*self.bar_colors)

        bar_labels = VGroup()
        for bar, name in zip(bars, self.bar_names):
            label = Tex(str(name))
            label.scale(self.bar_label_scale_val)
            label.next_to(bar, DOWN, SMALL_BUFF)
            bar_labels.add(label)

        self.add(bars, bar_labels)
        self.bars = bars
        self.bar_labels = bar_labels

    def change_bar_values(self, values: Iterable[float]) -> None:
        for bar, value in zip(self.bars, values):
            bar_bottom = bar.get_bottom()
            bar.stretch_to_fit_height(
                (value / self.max_value) * self.height
            )
            bar.move_to(bar_bottom, DOWN)
