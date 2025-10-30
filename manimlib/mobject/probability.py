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
    """柱状图类，继承自VGroup，用于在Manim中创建可视化柱状图"""
    
    def __init__(
        self,
        values: Iterable[float],    # 柱状图的核心数据值，每个值对应一根柱子
        height: float = 4,          # 柱状图整体高度（含坐标轴），默认4
        width: float = 6,           # 柱状图整体宽度（含坐标轴），默认6
        n_ticks: int = 4,           # Y轴刻度数量，默认4
        include_x_ticks: bool = False,  # 是否显示X轴刻度，默认不显示
        tick_width: float = 0.2,    # Y轴刻度线宽度，默认0.2
        tick_height: float = 0.15,  # X轴刻度线高度，默认0.15
        label_y_axis: bool = True,  # 是否为Y轴添加数值标签，默认是
        y_axis_label_height: float = 0.25,  # Y轴标签高度，默认0.25
        max_value: float = 1,       # Y轴最大值（柱子高度的参考基准），默认1
        bar_colors: list[ManimColor] = [BLUE, YELLOW],  # 柱子颜色渐变列表
        bar_fill_opacity: float = 0.8,  # 柱子填充不透明度，默认0.8
        bar_stroke_width: float = 3,    # 柱子边框宽度，默认3
        bar_names: list[str] = [],  # 每根柱子的名称（X轴下方标签），默认空列表
        bar_label_scale_val: float = 0.75,  # 柱子名称标签缩放比例，默认0.75
        **kwargs                    # 其他关键字参数，传递给父类VGroup
    ):
        # 调用父类VGroup的构造函数，初始化组合对象属性
        super().__init__(**kwargs)
        # 存储柱状图的基础配置属性
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

        # 若未指定Y轴最大值，则自动设为数据值中的最大值
        if self.max_value is None:
            self.max_value = max(values)

        # X轴刻度数量等于数据值数量（每根柱子对应一个刻度位置）
        self.n_ticks_x = len(values)
        # 绘制并添加坐标轴
        self.add_axes()
        # 绘制并添加柱状图的柱子
        self.add_bars(values)
        # 将整个柱状图居中显示
        self.center()

    def add_axes(self) -> None:
        """创建并添加柱状图的X轴和Y轴（含刻度线和标签）"""
        # 创建X轴：从左侧刻度线一半宽度处延伸到指定宽度的右侧
        x_axis = Line(self.tick_width * LEFT / 2, self.width * RIGHT)
        # 创建Y轴：从下方中等大间距处延伸到指定高度的上方
        y_axis = Line(MED_LARGE_BUFF * DOWN, self.height * UP)
        # 用于存储所有Y轴刻度线的组合对象
        y_ticks = VGroup()
        # 生成Y轴刻度线的高度位置（从0到总高度，分n_ticks+1个点）
        heights = np.linspace(0, self.height, self.n_ticks + 1)
        # 生成Y轴刻度对应的数值（从0到max_value，分n_ticks+1个点）
        values = np.linspace(0, self.max_value, self.n_ticks + 1)
        
        # 遍历每个刻度的高度和对应数值，创建Y轴刻度线
        for y, value in zip(heights, values):
            # 创建单条Y轴刻度线（水平方向）
            y_tick = Line(LEFT, RIGHT)
            # 设置刻度线宽度
            y_tick.set_width(self.tick_width)
            # 将刻度线移动到对应高度位置
            y_tick.move_to(y * UP)
            # 将刻度线添加到Y轴刻度组合中
            y_ticks.add(y_tick)
        # 将所有Y轴刻度线添加到Y轴对象中
        y_axis.add(y_ticks)

        # 若需要显示X轴刻度线
        if self.include_x_ticks == True:
            # 用于存储所有X轴刻度线的组合对象
            x_ticks = VGroup()
            # 生成X轴刻度线的宽度位置（从0到总宽度，分n_ticks_x+1个点）
            widths = np.linspace(0, self.width, self.n_ticks_x + 1)
            # 生成X轴刻度对应的标签值（从0到柱子数量，分n_ticks_x+1个点）
            label_values = np.linspace(0, len(self.bar_names), self.n_ticks_x + 1)
            
            # 遍历每个刻度的宽度和对应标签值，创建X轴刻度线
            for x, value in zip(widths, label_values):
                # 创建单条X轴刻度线（垂直方向）
                x_tick = Line(UP, DOWN)
                # 设置刻度线高度
                x_tick.set_height(self.tick_height)
                # 将刻度线移动到对应宽度位置
                x_tick.move_to(x * RIGHT)
                # 将刻度线添加到X轴刻度组合中
                x_ticks.add(x_tick)
            # 将所有X轴刻度线添加到X轴对象中
            x_axis.add(x_ticks)

        # 将X轴和Y轴添加到柱状图组合中
        self.add(x_axis, y_axis)
        # 存储X轴和Y轴对象，方便后续调用
        self.x_axis, self.y_axis = x_axis, y_axis

        # 若需要为Y轴添加数值标签
        if self.label_y_axis:
            # 用于存储所有Y轴标签的组合对象
            labels = VGroup()
            # 遍历每个Y轴刻度线和对应数值，创建标签
            for y_tick, value in zip(y_ticks, values):
                # 创建标签文本（保留2位小数）
                label = Tex(str(np.round(value, 2)))
                # 设置标签高度
                label.set_height(self.y_axis_label_height)
                # 将标签放在刻度线左侧，保留小间距
                label.next_to(y_tick, LEFT, SMALL_BUFF)
                # 将标签添加到Y轴标签组合中
                labels.add(label)
            # 存储Y轴标签组合，方便后续调用
            self.y_axis_labels = labels
            # 将Y轴标签添加到柱状图组合中
            self.add(labels)

    def add_bars(self, values: Iterable[float]) -> None:
        """根据输入数据值，创建并添加所有柱状图的柱子（含柱子名称标签）"""
        # 计算每根柱子的宽度：总宽度除以2倍柱子数量（预留间距）
        buff = float(self.width) / (2 * len(values))
        # 用于存储所有柱子的组合对象
        bars = VGroup()
        
        # 遍历每个数据值，创建对应的柱子
        for i, value in enumerate(values):
            # 创建矩形柱子
            bar = Rectangle(
                # 柱子高度：数据值与max_value的比例乘以总高度（按比例缩放）
                height=(value / self.max_value) * self.height,
                # 柱子宽度：使用预计算的buff值
                width=buff,
                # 柱子边框宽度
                stroke_width=self.bar_stroke_width,
                # 柱子填充不透明度
                fill_opacity=self.bar_fill_opacity,
            )
            # 将柱子移动到指定位置：X方向按索引排列，Y方向靠下对齐
            bar.move_to((2 * i + 0.5) * buff * RIGHT, DOWN + LEFT * 5)
            # 将柱子添加到柱子组合中
            bars.add(bar)
        # 为柱子组合设置颜色渐变（从第一个颜色过渡到最后一个颜色）
        bars.set_color_by_gradient(*self.bar_colors)

        # 用于存储所有柱子名称标签的组合对象
        bar_labels = VGroup()
        # 遍历每个柱子和对应的名称，创建标签
        for bar, name in zip(bars, self.bar_names):
            # 创建柱子名称标签文本
            label = Tex(str(name))
            # 按指定比例缩放标签
            label.scale(self.bar_label_scale_val)
            # 将标签放在柱子正下方，保留小间距
            label.next_to(bar, DOWN, SMALL_BUFF)
            # 将标签添加到柱子名称标签组合中
            bar_labels.add(label)

        # 将柱子和柱子名称标签添加到柱状图组合中
        self.add(bars, bar_labels)
        # 存储柱子和柱子名称标签组合，方便后续调用（如更新柱子高度）
        self.bars = bars
        self.bar_labels = bar_labels

    def change_bar_values(self, values: Iterable[float]) -> None:
        """更新柱状图中所有柱子的高度（根据新数据值）"""
        # 遍历每个柱子和对应的新数据值
        for bar, value in zip(self.bars, values):
            # 记录柱子当前的底部位置（确保更新高度后底部位置不变）
            bar_bottom = bar.get_bottom()
            # 按新数据值缩放柱子高度：新高度 = (新值/max_value) * 总高度
            bar.stretch_to_fit_height(
                (value / self.max_value) * self.height
            )
            # 将更新高度后的柱子移动回原底部位置（保持底部对齐）
            bar.move_to(bar_bottom, DOWN)
