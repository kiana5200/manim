from __future__ import annotations

import numpy as np

from manimlib.constants import BLUE, BLUE_E, GREEN_E, GREY_B, GREY_D, MAROON_B, YELLOW
from manimlib.constants import DOWN, LEFT, RIGHT, UP
from manimlib.constants import MED_LARGE_BUFF, MED_SMALL_BUFF, SMALL_BUFF
from manimlib.mobject.geometry import Line
from manimlib.mobject.geometry import Rectangle
from manimlib.mobject.mobject import Mobject
from manimlib.mobject.svg.brace import Brace
from manimlib.mobject.svg.tex_mobject import Tex
from manimlib.mobject.svg.tex_mobject import TexText
from manimlib.mobject.types.vectorized_mobject import VGroup
from manimlib.utils.color import color_gradient
from manimlib.utils.iterables import listify

from typing import TYPE_CHECKING

# 仅在类型检查时导入相关模块，避免运行时循环导入或不必要的依赖
if TYPE_CHECKING:
    from typing import Iterable
    from manimlib.typing import ManimColor


# 浮点精度阈值：用于判断数值是否接近（如概率和是否接近1）
EPSILON = 0.0001


class SampleSpace(Rectangle):
    """
    样本空间图形类：继承自Rectangle，用于可视化概率中的样本空间，
    支持添加标题、标签，补全概率列表，以及沿指定维度（水平/垂直）按概率划分样本空间。
    """
    def __init__(
        self,
        width: float = 3,  # 样本空间矩形宽度，默认3
        height: float = 3,  # 样本空间矩形高度，默认3
        fill_color: ManimColor = GREY_D,  # 填充色，默认深灰色
        fill_opacity: float = 1,  # 填充透明度，默认1（完全不透明）
        stroke_width: float = 0.5,  # 描边宽度，默认0.5
        stroke_color: ManimColor = GREY_B,  # 描边色，默认中灰色
        default_label_scale_val: float = 1,  # 标签默认缩放值，默认1
        **kwargs,
    ):
        # 调用父类Rectangle的初始化方法，传入矩形样式参数
        super().__init__(
            width, height,
            fill_color=fill_color,
            fill_opacity=fill_opacity,
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            **kwargs
        )
        # 存储标签默认缩放值，用于后续标签相关操作
        self.default_label_scale_val = default_label_scale_val

    def add_title(
        self,
        title: str = "Sample space",  # 标题文本，默认"Sample space"（样本空间）
        buff: float = MED_SMALL_BUFF  # 标题与样本空间的间距，默认中小间距
    ) -> None:
        # TODO, should this really exist in SampleSpaceScene（待优化：该方法是否应归属SampleSpaceScene）
        # 创建标题文本对象（使用TexText支持Latex语法）
        title_mob = TexText(title)
        # 若标题宽度超过样本空间宽度，缩放标题以适配
        if title_mob.get_width() > self.get_width():
            title_mob.set_width(self.get_width())
        # 将标题移动到样本空间正上方，保持指定间距
        title_mob.next_to(self, UP, buff=buff)
        # 存储标题对象到实例属性，便于后续操作
        self.title = title_mob
        # 将标题添加到样本空间图形中（作为子对象）
        self.add(title_mob)

    def add_label(self, label: str) -> None:
        """
        为样本空间添加标签：将标签文本存储到实例属性，便于后续标识样本空间。
        
        参数
        -----
        label : str
            样本空间的标签文本（如"A"、"事件1"等）
        """
        self.label = label

    def complete_p_list(self, p_list: list[float]) -> list[float]:
        """
        补全概率列表：确保概率列表中所有概率之和为1，若存在剩余概率（与1的差值超过EPSILON），
        则在列表末尾添加剩余概率，返回补全后的列表。
        
        参数
        -----
        p_list : list[float]
            初始概率列表（各元素为0-1的浮点数）
        
        返回
        -----
        list[float]
            补全后的概率列表，总和为1
        """
        # 将输入转换为列表（确保可修改，处理非列表的可迭代对象）
        new_p_list = listify(p_list)
        # 计算当前概率列表的总和与1的差值（剩余概率）
        remainder = 1.0 - sum(new_p_list)
        # 若剩余概率的绝对值超过精度阈值，将其添加到列表末尾
        if abs(remainder) > EPSILON:
            new_p_list.append(remainder)
        return new_p_list

    def get_division_along_dimension(
        self,
        p_list: list[float],
        dim: int,
        colors: Iterable[ManimColor],
        vect: np.ndarray
    ) -> VGroup:
        """
        沿指定维度划分样本空间：按概率列表p_list的比例，将样本空间沿dim维度（0为水平，1为垂直）分割，
        为每个分割部分分配渐变颜色，返回包含所有分割部分的VGroup。
        
        参数
        -----
        p_list : list[float]
            概率列表，用于确定各分割部分的比例（会先调用complete_p_list补全）
        dim : int
            划分维度，0表示沿水平方向（x轴），1表示沿垂直方向（y轴）
        colors : Iterable[ManimColor]
            颜色集合，用于生成分割部分的渐变颜色（颜色数量自动匹配分割数量）
        vect : np.ndarray
            方向向量，用于确定划分的起始和终止方向（如RIGHT表示从左到右，UP表示从下到上）
        
        返回
        -----
        VGroup
            包含所有分割部分的向量组，每个部分为SampleSpace实例，按概率比例划分
        """
        # 补全概率列表，确保总和为1
        p_list = self.complete_p_list(p_list)
        # 生成渐变颜色列表：根据颜色集合和分割数量（概率列表长度）生成均匀渐变
        colors = color_gradient(colors, len(p_list))

        # 确定划分的起始点：样本空间沿"负vect方向"的边缘中心（如vect=RIGHT时，起始点为左边缘中心）
        last_point = self.get_edge_center(-vect)
        # 存储所有分割部分的VGroup
        parts = VGroup()

        # 遍历概率和对应的颜色，生成每个分割部分
        for factor, color in zip(p_list, colors):
            # 创建分割部分的SampleSpace实例
            part = SampleSpace()
            # 设置分割部分的填充色和透明度
            part.set_fill(color, 1)
            # 按当前样本空间的形状拉伸分割部分（确保初始形状匹配）
            part.replace(self, stretch=True)
            # 沿指定维度按概率比例拉伸分割部分（factor为该部分占总样本空间的比例）
            part.stretch(factor, dim)
            # 将分割部分移动到起始点，对齐"负vect方向"的边缘（确保与前一部分衔接）
            part.move_to(last_point, -vect)
            # 更新下一个分割部分的起始点：当前分割部分沿"vect方向"的边缘中心
            last_point = part.get_edge_center(vect)
            # 将分割部分添加到VGroup中
            parts.add(part)
        
        return parts

    def get_horizontal_division(
            self,
            p_list: list[float],
            colors: Iterable[ManimColor] = [GREEN_E, BLUE_E],  # 水平分割的默认渐变颜色（浅绿、浅蓝）
            vect: np.ndarray = DOWN  # 水平分割的方向向量（默认向下，即从顶部到底部划分）
    ) -> VGroup:
        """
        水平分割样本空间：调用通用的维度划分方法，指定沿垂直维度（dim=1）划分，
        返回包含所有水平分割部分的VGroup。
        
        参数
        -----
        p_list : list[float]
            概率列表，确定各水平分割部分的高度比例（会自动补全使总和为1）
        colors : Iterable[ManimColor], optional
            水平分割部分的渐变颜色集合，默认[GREEN_E, BLUE_E]
        vect : np.ndarray, optional
            划分方向向量，默认DOWN（从顶部开始向下划分）
        
        返回
        -----
        VGroup
            包含所有水平分割部分的向量组，每个部分为SampleSpace实例
        """
        # 调用get_division_along_dimension，指定dim=1（垂直维度，对应水平分割）
        return self.get_division_along_dimension(p_list, 1, colors, vect)

    def get_vertical_division(
            self,
            p_list: list[float],
            colors: Iterable[ManimColor] = [MAROON_B, YELLOW],  # 垂直分割的默认渐变颜色（深红、黄色）
            vect: np.ndarray = RIGHT  # 垂直分割的方向向量（默认向右，即从左侧到右侧划分）
    ) -> VGroup:
        """
        垂直分割样本空间：调用通用的维度划分方法，指定沿水平维度（dim=0）划分，
        返回包含所有垂直分割部分的VGroup。
        
        参数
        -----
        p_list : list[float]
            概率列表，确定各垂直分割部分的宽度比例（会自动补全使总和为1）
        colors : Iterable[ManimColor], optional
            垂直分割部分的渐变颜色集合，默认[MAROON_B, YELLOW]
        vect : np.ndarray, optional
            划分方向向量，默认RIGHT（从左侧开始向右划分）
        
        返回
        -----
        VGroup
            包含所有垂直分割部分的向量组，每个部分为SampleSpace实例
        """
        # 调用get_division_along_dimension，指定dim=0（水平维度，对应垂直分割）
        return self.get_division_along_dimension(p_list, 0, colors, vect)

    def divide_horizontally(self, *args, **kwargs) -> None:
        """
        执行水平分割并添加到样本空间：调用get_horizontal_division获取水平分割部分，
        将其存储为实例属性，并添加到当前样本空间图形中（作为子对象）。
        
        参数
        -----
        *args, **kwargs
            传递给get_horizontal_division的参数（如p_list、colors等）
        """
        # 获取水平分割部分并存储到实例属性horizontal_parts
        self.horizontal_parts = self.get_horizontal_division(*args, **kwargs)
        # 将水平分割部分添加到当前样本空间中
        self.add(self.horizontal_parts)

    def divide_vertically(self, *args, **kwargs) -> None:
        """
        执行垂直分割并添加到样本空间：调用get_vertical_division获取垂直分割部分，
        将其存储为实例属性，并添加到当前样本空间图形中（作为子对象）。
        
        参数
        -----
        *args, **kwargs
            传递给get_vertical_division的参数（如p_list、colors等）
        """
        # 获取垂直分割部分并存储到实例属性vertical_parts
        self.vertical_parts = self.get_vertical_division(*args, **kwargs)
        # 将垂直分割部分添加到当前样本空间中
        self.add(self.vertical_parts)

    def get_subdivision_braces_and_labels(
            self,
            parts: VGroup,
            labels: str,
            direction: np.ndarray,
            buff: float = SMALL_BUFF,  # 花括号/标签与分割部分的间距，默认小间距
    ) -> VGroup:
        """
        为分割部分添加花括号和标签：为VGroup中的每个分割部分（如水平/垂直分割块）
        创建对应方向的花括号（Brace）和标签文本，返回包含所有花括号和标签的VGroup，
        并将相关信息存储到分割部分的VGroup属性中。
        
        参数
        -----
        parts : VGroup
            包含分割部分的VGroup（如horizontal_parts、vertical_parts）
        labels : str
            标签文本，需与分割部分数量匹配（可传入单个字符串或字符串列表）
        direction : np.ndarray
            花括号和标签的朝向（如UP表示在分割部分上方，RIGHT表示在右侧）
        buff : float, optional
            花括号与分割部分、标签与花括号的间距，默认SMALL_BUFF
        
        返回
        -----
        VGroup
            包含所有花括号（braces）和标签（labels）的向量组
        """
        # 存储所有标签文本对象的VGroup
        label_mobs = VGroup()
        # 存储所有花括号对象的VGroup
        braces = VGroup()

        # 遍历每个标签和对应的分割部分，创建花括号和标签
        for label, part in zip(labels, parts):
            # 为当前分割部分创建花括号，朝向指定方向，保持指定间距
            brace = Brace(
                part, direction,
                buff=buff
            )
            # 判断标签是否为Mobject实例（如已创建的Tex对象），是则直接使用，否则创建Tex对象
            if isinstance(label, Mobject):
                label_mob = label
            else:
                label_mob = Tex(label)
                # 按样本空间的默认标签缩放值缩放标签
                label_mob.scale(self.default_label_scale_val)
            # 将标签移动到花括号的指定方向外侧，保持指定间距
            label_mob.next_to(brace, direction, buff)

            # 将花括号和标签分别添加到对应VGroup
            braces.add(brace)
            label_mobs.add(label_mob)

        # 将花括号和标签存储到分割部分VGroup的属性中，便于后续访问和修改
        parts.braces = braces
        parts.labels = label_mobs
        # 存储标签相关配置参数到分割部分VGroup，便于复用
        parts.label_kwargs = {
            "labels": label_mobs.copy(),
            "direction": direction,
            "buff": buff,
        }

        # 返回包含花括号和标签的组合VGroup
        return VGroup(parts.braces, parts.labels)

    def get_side_braces_and_labels(
            self,
            labels: str,
            direction: np.ndarray = LEFT,  # 花括号和标签的朝向，默认左侧（LEFT）
            **kwargs
    ) -> VGroup:
        """
        为水平分割部分添加侧边花括号和标签：仅在样本空间已执行水平分割（存在horizontal_parts）时生效，
        调用通用标签方法为水平分割块添加指定方向的花括号和标签，默认朝向左侧。
        
        参数
        -----
        labels : str
            标签文本，需与水平分割块数量匹配（单个字符串或字符串列表）
        direction : np.ndarray, optional
            花括号和标签的朝向，默认LEFT（左侧），可传入RIGHT（右侧）等方向向量
        **kwargs
            传递给get_subdivision_braces_and_labels的额外参数（如buff）
        
        返回
        -----
        VGroup
            包含侧边花括号和标签的向量组
        
        断言
        -----
        断言样本空间存在horizontal_parts属性（即已执行水平分割），否则报错
        """
        # 断言样本空间已执行水平分割（存在horizontal_parts），未分割则触发断言错误
        assert hasattr(self, "horizontal_parts")
        # 获取水平分割部分的VGroup
        parts = self.horizontal_parts
        # 调用通用方法为水平分割部分添加花括号和标签，指定朝向和其他参数
        return self.get_subdivision_braces_and_labels(parts, labels, direction, **kwargs)

    def get_top_braces_and_labels(
        self,
        labels: str,
        **kwargs
    ) -> VGroup:
        """
        为垂直分割部分添加顶部花括号和标签：仅在样本空间已执行垂直分割（存在vertical_parts）时生效，
        调用通用标签方法为垂直分割块添加顶部（UP方向）的花括号和标签。
        
        参数
        -----
        labels : str
            标签文本，需与垂直分割块数量匹配（单个字符串或字符串列表）
        **kwargs
            传递给get_subdivision_braces_and_labels的额外参数（如buff）
        
        返回
        -----
        VGroup
            包含顶部花括号和标签的向量组
        
        断言
        -----
        断言样本空间存在vertical_parts属性（即已执行垂直分割），否则报错
        """
        # 断言样本空间已执行垂直分割（存在vertical_parts），未分割则触发断言错误
        assert hasattr(self, "vertical_parts")
        # 获取垂直分割部分的VGroup
        parts = self.vertical_parts
        # 调用通用方法为垂直分割部分添加顶部（UP方向）的花括号和标签
        return self.get_subdivision_braces_and_labels(parts, labels, UP, **kwargs)

    def get_bottom_braces_and_labels(
        self,
        labels: str,
        **kwargs
    ) -> VGroup:
        """
        为垂直分割部分添加底部花括号和标签：仅在样本空间已执行垂直分割（存在vertical_parts）时生效，
        调用通用标签方法为垂直分割块添加底部（DOWN方向）的花括号和标签。
        
        参数
        -----
        labels : str
            标签文本，需与垂直分割块数量匹配（单个字符串或字符串列表）
        **kwargs
            传递给get_subdivision_braces_and_labels的额外参数（如buff）
        
        返回
        -----
        VGroup
            包含底部花括号和标签的向量组
        
        断言
        -----
        断言样本空间存在vertical_parts属性（即已执行垂直分割），否则报错
        """
        # 断言样本空间已执行垂直分割（存在vertical_parts），未分割则触发断言错误
        assert hasattr(self, "vertical_parts")
        # 获取垂直分割部分的VGroup
        parts = self.vertical_parts
        # 调用通用方法为垂直分割部分添加底部（DOWN方向）的花括号和标签
        return self.get_subdivision_braces_and_labels(parts, labels, DOWN, **kwargs)

    def add_braces_and_labels(self) -> None:
        """
        将分割部分的花括号和标签添加到样本空间：遍历水平分割（horizontal_parts）和垂直分割（vertical_parts），
        若分割部分已存在花括号（braces）或标签（labels），则将其添加到当前样本空间图形中（作为子对象）。
        """
        # 遍历水平分割和垂直分割的属性名
        for attr in "horizontal_parts", "vertical_parts":
            # 若样本空间不存在该分割属性（未执行对应分割），跳过
            if not hasattr(self, attr):
                continue
            # 获取分割部分的VGroup（horizontal_parts或vertical_parts）
            parts = getattr(self, attr)
            # 遍历分割部分可能存在的花括号和标签属性
            for subattr in "braces", "labels":
                # 若分割部分存在该子属性（已创建花括号或标签），将其添加到样本空间
                if hasattr(parts, subattr):
                    self.add(getattr(parts, subattr))

    def __getitem__(self, index: int | slice) -> VGroup:
        """
        索引访问魔法方法：支持通过索引或切片获取样本空间的分割部分，优先级为：
        1. 若存在水平分割（horizontal_parts），返回水平分割块的对应索引部分
        2. 若存在垂直分割（vertical_parts），返回垂直分割块的对应索引部分
        3. 若未分割，将样本空间本身分割后返回对应索引部分
        
        参数
        -----
        index : int | slice
            索引（如0、1）或切片（如1:3），用于指定获取的分割部分
        
        返回
        -----
        VGroup
            对应索引的分割部分（单个或多个SampleSpace实例组成的VGroup）
        """
        # 若存在水平分割，优先通过索引获取水平分割块
        if hasattr(self, "horizontal_parts"):
            return self.horizontal_parts[index]
        # 若存在垂直分割，通过索引获取垂直分割块
        elif hasattr(self, "vertical_parts"):
            return self.vertical_parts[index]
        # 若未分割，将样本空间按默认方式分割后返回对应索引部分
        return self.split()[index]


class BarChart(VGroup):
    """
    柱状图组件：继承自VGroup，用于可视化一组数值数据，包含坐标轴、刻度、柱状图本体及可选标签，
    支持自定义尺寸、颜色、刻度数量等样式，数据映射自动适配设定的最大数值和图表高度。
    """
    def __init__(
        self,
        values: Iterable[float],  # 用于绘制柱状图的数值集合（如[1,3,2]）
        height: float = 4,  # 柱状图整体高度（含Y轴高度），默认4
        width: float = 6,   # 柱状图整体宽度（含X轴宽度），默认6
        n_ticks: int = 4,   # Y轴刻度数量，默认4（含0刻度，共n_ticks+1个刻度）
        include_x_ticks: bool = False,  # 是否显示X轴刻度，默认不显示
        tick_width: float = 0.2,  # Y轴刻度线宽度，默认0.2
        tick_height: float = 0.15,  # X轴刻度线高度，默认0.15
        label_y_axis: bool = True,  # 是否为Y轴刻度添加数值标签，默认显示
        y_axis_label_height: float = 0.25,  # Y轴标签文本高度，默认0.25
        max_value: float = 1,  # 数值映射的最大值（Y轴顶端对应的值），默认1
        bar_colors: list[ManimColor] = [BLUE, YELLOW],  # 柱状图颜色渐变集合，默认[蓝色,黄色]
        bar_fill_opacity: float = 0.8,  # 柱状图填充透明度，默认0.8
        bar_stroke_width: float = 3,  # 柱状图描边宽度，默认3
        bar_names: list[str] = [],  # 每个柱子的名称标签（与values长度匹配），默认空列表
        bar_label_scale_val: float = 0.75,  # 柱子名称标签的缩放比例，默认0.75
        **kwargs
    ):
        super().__init__(**kwargs)
        # 存储柱状图核心配置参数
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

        # 若未指定max_value（或为None），则将max_value设为输入数值的最大值
        if self.max_value is None:
            self.max_value = max(values)

        # X轴刻度数量（含0刻度）与数值数量一致，用于后续X轴刻度绘制
        self.n_ticks_x = len(values)
        # 添加坐标轴（X轴、Y轴及刻度）
        self.add_axes()
        # 添加柱状图本体（根据values绘制对应高度的柱子）
        self.add_bars(values)
        # 将整个柱状图组件居中显示
        self.center()

    def add_axes(self) -> None:
        """
        绘制并添加坐标轴：创建X轴、Y轴，生成Y轴刻度及可选的X轴刻度，
        为Y轴刻度添加数值标签（若开启label_y_axis），并将坐标轴添加到组件中。
        """
        # 创建X轴：从左侧偏移（tick_width/2）处开始，到设定宽度的右侧结束
        x_axis = Line(self.tick_width * LEFT / 2, self.width * RIGHT)
        # 创建Y轴：从下方中等大间距处开始，到设定高度的上方结束
        y_axis = Line(MED_LARGE_BUFF * DOWN, self.height * UP)
        # 存储Y轴刻度的VGroup
        y_ticks = VGroup()
        # 生成Y轴刻度的高度位置（从0到self.height，均匀分布n_ticks+1个点）
        heights = np.linspace(0, self.height, self.n_ticks + 1)
        # 生成Y轴刻度对应的数值（从0到self.max_value，与高度位置一一对应）
        values = np.linspace(0, self.max_value, self.n_ticks + 1)

        # 遍历每个高度和对应数值，创建Y轴刻度线
        for y, value in zip(heights, values):
            y_tick = Line(LEFT, RIGHT)  # 创建水平刻度线
            y_tick.set_width(self.tick_width)  # 设置刻度线宽度
            y_tick.move_to(y * UP)  # 将刻度线移动到对应高度
            y_ticks.add(y_tick)  # 将刻度线添加到Y轴刻度组
        y_axis.add(y_ticks)  # 将Y轴刻度组添加到Y轴（作为子对象）

        # 若开启显示X轴刻度，生成并添加X轴刻度线
        if self.include_x_ticks == True:
            x_ticks = VGroup()  # 存储X轴刻度的VGroup
            # 生成X轴刻度的宽度位置（从0到self.width，均匀分布n_ticks_x+1个点）
            widths = np.linspace(0, self.width, self.n_ticks_x + 1)
            # 生成X轴刻度对应的标签数值（从0到柱子数量，与宽度位置一一对应）
            label_values = np.linspace(0, len(self.bar_names), self.n_ticks_x + 1)

            # 遍历每个宽度和对应数值，创建X轴刻度线
            for x, value in zip(widths, label_values):
                x_tick = Line(UP, DOWN)  # 创建垂直刻度线
                x_tick.set_height(self.tick_height)  # 设置刻度线高度
                x_tick.move_to(x * RIGHT)  # 将刻度线移动到对应宽度
                x_ticks.add(x_tick)  # 将刻度线添加到X轴刻度组
            x_axis.add(x_ticks)  # 将X轴刻度组添加到X轴（作为子对象）

        # 将X轴和Y轴添加到柱状图组件中
        self.add(x_axis, y_axis)
        # 存储X轴和Y轴到实例属性，便于后续访问
        self.x_axis, self.y_axis = x_axis, y_axis

        # 若开启Y轴标签，为每个Y轴刻度添加数值标签
        if self.label_y_axis:
            labels = VGroup()  # 存储Y轴标签的VGroup
            # 遍历每个Y轴刻度和对应数值，创建标签文本
            for y_tick, value in zip(y_ticks, values):
                label = Tex(str(np.round(value, 2)))  # 创建标签文本，数值保留2位小数
                label.set_height(self.y_axis_label_height)  # 设置标签文本高度
                label.next_to(y_tick, LEFT, SMALL_BUFF)  # 将标签放在刻度线左侧，保持小间距
                labels.add(label)  # 将标签添加到标签组
            self.y_axis_labels = labels  # 存储Y轴标签组到实例属性
            self.add(labels)  # 将Y轴标签组添加到柱状图组件中

    def add_bars(self, values: Iterable[float]) -> None:
        """
        绘制并添加柱状图本体及柱子标签：根据输入数值计算每个柱子的高度，
        生成带渐变颜色的柱子，为每个柱子添加名称标签（若bar_names非空），
        最终将柱子和标签添加到柱状图组件中。
        
        参数
        -----
        values : Iterable[float]
            用于确定每个柱子高度的数值集合，与柱子数量一一对应
        """
        # 计算柱子之间的间距及柱子宽度：总宽度均分，单个柱子宽度为buff
        buff = float(self.width) / (2 * len(values))
        # 存储所有柱子的VGroup
        bars = VGroup()

        # 遍历每个数值，创建对应的柱子
        for i, value in enumerate(values):
            # 计算柱子高度：数值与max_value的比例乘以图表总高度（实现数值到高度的映射）
            bar_height = (value / self.max_value) * self.height
            # 创建柱子矩形：设置高度、宽度、描边宽度和填充透明度
            bar = Rectangle(
                height=bar_height,
                width=buff,
                stroke_width=self.bar_stroke_width,
                fill_opacity=self.bar_fill_opacity,
            )
            # 计算柱子位置并移动：按索引均匀分布在X轴上，初始Y轴位置偏下（后续整体居中调整）
            bar_x_pos = (2 * i + 0.5) * buff * RIGHT
            bar.move_to(bar_x_pos, DOWN + LEFT * 5)
            bars.add(bar)  # 将柱子添加到柱子组

        # 为柱子组设置渐变颜色：使用配置的bar_colors生成颜色渐变
        bars.set_color_by_gradient(*self.bar_colors)

        # 存储所有柱子名称标签的VGroup
        bar_labels = VGroup()
        # 遍历每个柱子和对应的名称，创建标签（仅当bar_names非空时生效）
        for bar, name in zip(bars, self.bar_names):
            label = Tex(str(name))  # 创建标签文本（显示柱子名称）
            label.scale(self.bar_label_scale_val)  # 按配置缩放标签
            label.next_to(bar, DOWN, SMALL_BUFF)  # 将标签放在柱子正下方，保持小间距
            bar_labels.add(label)  # 将标签添加到标签组

        # 将柱子组和标签组添加到柱状图组件中
        self.add(bars, bar_labels)
        # 存储柱子组和标签组到实例属性，便于后续修改（如更新数值）
        self.bars = bars
        self.bar_labels = bar_labels

    def change_bar_values(self, values: Iterable[float]) -> None:
        """
        更新柱状图的数值及对应柱子高度：遍历每个柱子和新数值，
        按新数值比例调整柱子高度，并保持柱子底部位置不变（避免整体偏移）。
        
        参数
        -----
        values : Iterable[float]
            新的数值集合，需与现有柱子数量一致，用于更新柱子高度
        """
        # 遍历每个柱子和对应的新数值
        for bar, value in zip(self.bars, values):
            # 记录当前柱子的底部位置（确保高度调整后底部位置不变）
            bar_bottom = bar.get_bottom()
            # 计算新的柱子高度：新数值与max_value的比例乘以图表总高度
            new_bar_height = (value / self.max_value) * self.height
            # 拉伸柱子以适配新高度（仅调整高度，宽度保持不变）
            bar.stretch_to_fit_height(new_bar_height)
            # 将调整后的柱子移动回原底部位置，确保底部对齐
            bar.move_to(bar_bottom, DOWN)