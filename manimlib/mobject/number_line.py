from __future__ import annotations

import numpy as np

from manimlib.constants import DOWN, LEFT, RIGHT, UP
from manimlib.constants import DEFAULT_LIGHT_COLOR
from manimlib.constants import MED_SMALL_BUFF
from manimlib.mobject.geometry import Line
from manimlib.mobject.numbers import DecimalNumber
from manimlib.mobject.types.vectorized_mobject import VGroup
from manimlib.utils.bezier import interpolate
from manimlib.utils.bezier import outer_interpolate
from manimlib.utils.dict_ops import merge_dicts_recursively
from manimlib.utils.simple_functions import fdiv

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Iterable, Optional
    from manimlib.typing import ManimColor, Vect3, Vect3Array, VectN, RangeSpecifier


class NumberLine(Line):
    def __init__(
        self,
        x_range: RangeSpecifier = (-8, 8, 1),
        color: ManimColor = DEFAULT_LIGHT_COLOR,
        stroke_width: float = 2.0,
        # 数值轴上1个单位对应的实际空间距离
        unit_size: float = 1.0,
        width: Optional[float] = None,
        include_ticks: bool = True,
        tick_size: float = 0.1,
        longer_tick_multiple: float = 1.5,
        tick_offset: float = 0.0,
        # 重命名（大刻度间距）
        big_tick_spacing: Optional[float] = None,
        big_tick_numbers: list[float] = [],
        include_numbers: bool = False,
        line_to_number_direction: Vect3 = DOWN,
        line_to_number_buff: float = MED_SMALL_BUFF,
        include_tip: bool = False,
        tip_config: dict = dict(
            width=0.25,
            length=0.25,
        ),
        decimal_number_config: dict = dict(
            num_decimal_places=0,
            font_size=36,
        ),
        numbers_to_exclude: list | None = None,** kwargs,
    ):
        # 存储数值轴范围（起始、结束、步长）
        self.x_range = x_range
        # 普通刻度的大小
        self.tick_size = tick_size
        # 大刻度相对于普通刻度的长度倍数
        self.longer_tick_multiple = longer_tick_multiple
        # 刻度的偏移量
        self.tick_offset = tick_offset
        
        # 确定大刻度对应的数值：优先用间距计算，无间距则用指定列表
        if big_tick_spacing is not None:
            self.big_tick_numbers = np.arange(
                x_range[0],
                x_range[1] + big_tick_spacing,
                big_tick_spacing,
            )
        else:
            self.big_tick_numbers = list(big_tick_numbers)
        
        # 数字标签相对于轴线的方向（默认向下）
        self.line_to_number_direction = line_to_number_direction
        # 数字标签与轴线的间距
        self.line_to_number_buff = line_to_number_buff
        # 是否显示轴端点箭头
        self.include_tip = include_tip
        # 箭头的配置参数
        self.tip_config = dict(tip_config)
        # 数字标签的配置参数（小数位数、字体大小等）
        self.decimal_number_config = dict(decimal_number_config)
        # 需要排除不显示的数字
        self.numbers_to_exclude = numbers_to_exclude

        # 解析数值轴的起始、结束值和步长
        self.x_min, self.x_max = x_range[:2]
        self.x_step = 1 if len(x_range) == 2 else x_range[2]

        # 调用父类Line的初始化：以x_min和x_max为端点创建线段
        super().__init__(
            self.x_min * RIGHT, self.x_max * RIGHT,
            color=color,
            stroke_width=stroke_width,
            **kwargs
        )

        # 调整数值轴宽度：有指定宽度则直接设置，否则按单位尺寸缩放
        if width:
            self.set_width(width)
        else:
            self.scale(unit_size)
        # 将数值轴居中
        self.center()

        # 若需要显示箭头，添加箭头并同步样式
        if include_tip:
            self.add_tip()
            self.tip.set_stroke(
                self.stroke_color,
                self.stroke_width,
            )
        # 若需要显示刻度，添加刻度
        if include_ticks:
            self.add_ticks()
        # 若需要显示数字标签，添加标签（排除指定数字）
        if include_numbers:
            self.add_numbers(excluding=self.numbers_to_exclude)

    def get_tick_range(self) -> np.ndarray:
        # 确定刻度范围的最大值：有箭头时用x_max，无箭头时扩展一个步长（避免终点漏刻度）
        if self.include_tip:
            x_max = self.x_max
        else:
            x_max = self.x_max + self.x_step
        # 生成从x_min到x_max（不含）、步长为x_step的刻度值序列
        result = np.arange(self.x_min, x_max, self.x_step)
        # 过滤掉超过x_max的刻度（确保不超出轴的实际终点）
        return result[result <= self.x_max]

    def add_ticks(self) -> None:
        # 初始化刻度组
        ticks = VGroup()
        # 遍历所有刻度位置，生成对应刻度线
        for x in self.get_tick_range():
            size = self.tick_size
            # 若当前刻度是大刻度，将大小乘以倍数（放大）
            if np.isclose(self.big_tick_numbers, x).any():
                size *= self.longer_tick_multiple
            ticks.add(self.get_tick(x, size))
        # 将刻度组添加到数值轴，并保存为实例属性
        self.add(ticks)
        self.ticks = ticks

    def get_tick(self, x: float, size: float | None = None) -> Line:
        # 若未指定刻度大小，使用默认值
        if size is None:
            size = self.tick_size
        # 创建垂直方向的刻度线（从下到上）
        result = Line(size * DOWN, size * UP)
        # 让刻度线旋转到与数值轴一致的角度（适配非水平轴）
        result.rotate(self.get_angle())
        # 将刻度线移动到x对应的轴上位置
        result.move_to(self.number_to_point(x))
        # 让刻度线样式与数值轴匹配（颜色、线宽等）
        result.match_style(self)
        return result

    def get_tick_marks(self) -> VGroup:
        # 返回已创建的刻度组
        return self.ticks

    def number_to_point(self, number: float | VectN) -> Vect3 | Vect3Array:
        # 获取数值轴的起点和终点坐标
        start = self.get_points()[0]
        end = self.get_points()[-1]
        # 计算数值在轴范围中的比例（0到1之间）
        alpha = (number - self.x_min) / (self.x_max - self.x_min)
        # 按比例在轴的起点和终点间插值，得到数值对应的坐标点
        return outer_interpolate(start, end, alpha)

    def point_to_number(self, point: Vect3 | Vect3Array) -> float | VectN:
        # 获取数值轴的起点和终点坐标
        start = self.get_points()[0]
        end = self.get_points()[-1]
        # 计算轴的方向向量
        vect = end - start
        # 计算点在轴方向上的比例（投影比例）
        proportion = fdiv(
            np.dot(point - start, vect),  # 点到起点的向量与轴方向向量的点积
            np.dot(end - start, vect),    # 轴方向向量自身的点积（长度平方）
        )
        # 按比例在轴的数值范围间插值，得到点对应的数值
        return interpolate(self.x_min, self.x_max, proportion)

    def n2p(self, number: float | VectN) -> Vect3 | Vect3Array:
        """number_to_point的缩写，将数值转换为坐标点"""
        return self.number_to_point(number)

    def p2n(self, point: Vect3 | Vect3Array) -> float | VectN:
        """point_to_number的缩写，将坐标点转换为数值"""
        return self.point_to_number(point)

    def get_unit_size(self) -> float:
        # 计算数值轴上1个单位对应的实际空间长度（总长度除以数值范围跨度）
        return self.get_length() / (self.x_max - self.x_min)

    def get_number_mobject(
        self,
        x: float,
        direction: Vect3 | None = None,
        buff: float | None = None,
        unit: float = 1.0,
        unit_tex: str = "",
        **number_config
    ) -> DecimalNumber:
        # 合并默认数字配置和传入配置（传入配置优先级更高）
        number_config = merge_dicts_recursively(
            self.decimal_number_config, number_config,
        )
        # 若未指定方向，使用默认的数字标签方向
        if direction is None:
            direction = self.line_to_number_direction
        # 若未指定间距，使用默认的数字与轴线间距
        if buff is None:
            buff = self.line_to_number_buff
        # 若指定了单位文本，将其加入数字配置
        if unit_tex:
            number_config["unit"] = unit_tex

        # 创建DecimalNumber对象：数值按单位缩放（x除以单位）
        num_mob = DecimalNumber(x / unit, **number_config)
        # 将数字标签定位到x对应的轴上点的指定方向，保持间距
        num_mob.next_to(
            self.number_to_point(x),
            direction=direction,
            buff=buff
        )
        # 处理负数标签：若方向无水平分量（如上下），调整负号位置使其对齐
        if x < 0 and direction[0] == 0:
            # 不含负号对齐（向左偏移负号宽度的一半）
            num_mob.shift(num_mob[0].get_width() * LEFT / 2)
        # 处理单位为1的特殊情况：移除数字"1"，只保留符号和单位（如"-i"而非"-1i"）
        if abs(x) == unit and unit_tex:
            # 记录当前数字标签的中心位置
            center = num_mob.get_center()
            if x > 0:
                # 正数：移除数字部分（索引0），保留单位
                num_mob.remove(num_mob[0])
            else:
                # 负数：移除数字部分（索引1），调整负号（索引0）与单位的间距
                num_mob.remove(num_mob[1])
                num_mob[0].next_to(num_mob[1], LEFT, buff=num_mob[0].get_width() / 4)
            # 将调整后的标签移回原中心位置
            num_mob.move_to(center)
        return num_mob

    def add_numbers(
        self,
        x_values: Iterable[float] | None = None,
        excluding: Iterable[float] | None = None,
        font_size: int = 24,
        **kwargs
    ) -> VGroup:
        # 若未指定数值列表，默认使用刻度范围的数值
        if x_values is None:
            x_values = self.get_tick_range()

        kwargs["font_size"] = font_size

        # 若未指定排除的数值，使用实例中保存的排除列表
        if excluding is None:
            excluding = self.numbers_to_exclude

        numbers = VGroup()
        # 遍历每个数值，生成并添加数字标签
        for x in x_values:
            # 跳过需要排除的数值
            if excluding is not None and x in excluding:
                continue
            # 生成该数值对应的标签并添加到组中
            numbers.add(self.get_number_mobject(x, **kwargs))
        # 将数字标签组添加到数值轴，并保存为实例属性
        self.add(numbers)
        self.numbers = numbers
        return numbers


class UnitInterval(NumberLine):
    def __init__(
        self,
        x_range: RangeSpecifier = (0, 1, 0.1),
        unit_size: float = 10,
        big_tick_numbers: list[float] = [0, 1],
        decimal_number_config: dict = dict(
            num_decimal_places=1,
        ),** kwargs
    ):
        # 调用父类NumberLine的初始化方法，实现单位区间（0到1）的数值轴
        super().__init__(
            x_range=x_range,          # 数值范围默认0到1，步长0.1
            unit_size=unit_size,      # 1个单位对应的空间长度默认10
            big_tick_numbers=big_tick_numbers,  # 大刻度默认在0和1处
            decimal_number_config=decimal_number_config,  # 数字默认保留1位小数
            **kwargs  # 传递其他额外参数
        )
