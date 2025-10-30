# 从__future__导入annotations，用于支持 postponed evaluation of annotations（延迟类型注解评估）
from __future__ import annotations

# 导入numpy库，并简写为np，用于数值计算
import numpy as np

# 从manimlib.constants模块导入常用方向常量：下、左、右、上
from manimlib.constants import DOWN, LEFT, RIGHT, UP
# 从manimlib.constants模块导入默认的光源颜色常量
from manimlib.constants import DEFAULT_LIGHT_COLOR
# 从manimlib.constants模块导入中等偏小的缓冲距离常量
from manimlib.constants import MED_SMALL_BUFF
# 从manimlib.mobject.geometry模块导入Line类，用于创建线对象
from manimlib.mobject.geometry import Line
# 从manimlib.mobject.numbers模块导入DecimalNumber类，用于创建十进制数字对象
from manimlib.mobject.numbers import DecimalNumber
# 从manimlib.mobject.types.vectorized_mobject模块导入VGroup类，用于创建矢量对象组
from manimlib.mobject.types.vectorized_mobject import VGroup
# 从manimlib.utils.bezier模块导入interpolate函数，用于贝塞尔插值计算
from manimlib.utils.bezier import interpolate
# 从manimlib.utils.bezier模块导入outer_interpolate函数，用于外部贝塞尔插值计算
from manimlib.utils.bezier import outer_interpolate
# 从manimlib.utils.dict_ops模块导入merge_dicts_recursively函数，用于递归合并字典
from manimlib.utils.dict_ops import merge_dicts_recursively
# 从manimlib.utils.simple_functions模块导入fdiv函数，用于浮点数除法
from manimlib.utils.simple_functions import fdiv

# 从typing模块导入TYPE_CHECKING常量，用于条件性导入类型提示
from typing import TYPE_CHECKING

# 如果处于类型检查阶段（非运行时）
if TYPE_CHECKING:
    # 从typing模块导入需要的类型提示：可迭代对象、可选类型
    from typing import Iterable, Optional
    # 从manimlib.typing模块导入特定的类型提示：Manim颜色类型、三维向量、三维向量数组、N维向量、范围说明符
    from manimlib.typing import ManimColor, Vect3, Vect3Array, VectN, RangeSpecifier


class NumberLine(Line):
    """
    数值线类，继承自Line类，用于创建带有刻度和数字标记的数轴
    """
    def __init__(
        self,
        x_range: RangeSpecifier = (-8, 8, 1),  # 数轴范围，格式为(最小值, 最大值, 步长)
        color: ManimColor = DEFAULT_LIGHT_COLOR,  # 数轴颜色
        stroke_width: float = 2.0,  # 线条宽度
        # 数轴上一个单位在空间中的实际距离
        unit_size: float = 1.0,
        width: Optional[float] = None,  # 数轴总宽度，若指定则忽略unit_size
        include_ticks: bool = True,  # 是否包含刻度
        tick_size: float = 0.1,  # 普通刻度的长度
        longer_tick_multiple: float = 1.5,  # 长刻度相对于普通刻度的倍数
        tick_offset: float = 0.0,  # 刻度偏移量
        # 大刻度的间隔
        big_tick_spacing: Optional[float] = None,
        big_tick_numbers: list[float] = [],  # 大刻度对应的数值列表
        include_numbers: bool = False,  # 是否包含数字标记
        line_to_number_direction: Vect3 = DOWN,  # 数字相对于数轴的方向
        line_to_number_buff: float = MED_SMALL_BUFF,  # 数字与数轴的距离
        include_tip: bool = False,  # 是否包含数轴末端的箭头
        tip_config: dict = dict(  # 箭头的配置参数
            width=0.25,
            length=0.25,
        ),
        decimal_number_config: dict = dict(  # 数字的配置参数
            num_decimal_places=0,  # 小数位数
            font_size=36,  # 字体大小
        ),
        numbers_to_exclude: list | None = None,  # 要排除的数字标记
        **kwargs,
    ):
        # 初始化实例变量
        self.x_range = x_range
        self.tick_size = tick_size
        self.longer_tick_multiple = longer_tick_multiple
        self.tick_offset = tick_offset
        
        # 确定大刻度的位置
        if big_tick_spacing is not None:
            self.big_tick_numbers = np.arange(
                x_range[0],
                x_range[1] + big_tick_spacing,  # +spacing确保包含终点
                big_tick_spacing,
            )
        else:
            self.big_tick_numbers = list(big_tick_numbers)
            
        # 设置数字相关的属性
        self.line_to_number_direction = line_to_number_direction
        self.line_to_number_buff = line_to_number_buff
        self.include_tip = include_tip
        self.tip_config = dict(tip_config)
        self.decimal_number_config = dict(decimal_number_config)
        self.numbers_to_exclude = numbers_to_exclude

        # 解析x_range获取最小值、最大值和步长
        self.x_min, self.x_max = x_range[:2]
        self.x_step = 1 if len(x_range) == 2 else x_range[2]

        # 调用父类Line的初始化方法
        super().__init__(
            self.x_min * RIGHT, self.x_max * RIGHT,  # 起点和终点
            color=color,
            stroke_width=stroke_width,** kwargs
        )

        # 设置数轴宽度或缩放比例
        if width:
            self.set_width(width)
        else:
            self.scale(unit_size)
        self.center()  # 将数轴居中

        # 如果需要，添加箭头
        if include_tip:
            self.add_tip()
            self.tip.set_stroke(
                self.stroke_color,
                self.stroke_width,
            )
            
        # 如果需要，添加刻度
        if include_ticks:
            self.add_ticks()
            
        # 如果需要，添加数字标记
        if include_numbers:
            self.add_numbers(excluding=self.numbers_to_exclude)

    def get_tick_range(self) -> np.ndarray:
        """获取所有刻度的位置范围"""
        if self.include_tip:
            x_max = self.x_max
        else:
            x_max = self.x_max + self.x_step  # 扩展范围以确保包含最后一个刻度
            
        # 生成从x_min到x_max的等差数列
        result = np.arange(self.x_min, x_max, self.x_step)
        # 过滤掉超过最大值的刻度
        return result[result <= self.x_max]

    def add_ticks(self) -> None:
        """添加所有刻度到数轴上"""
        ticks = VGroup()  # 创建一个矢量对象组来存储所有刻度
        
        # 为每个刻度位置创建刻度线
        for x in self.get_tick_range():
            size = self.tick_size
            # 如果是大刻度位置，使用更长的刻度
            if np.isclose(self.big_tick_numbers, x).any():
                size *= self.longer_tick_multiple
            ticks.add(self.get_tick(x, size))
            
        self.add(ticks)  # 将刻度添加到数轴
        self.ticks = ticks  # 保存刻度引用

    def get_tick(self, x: float, size: float | None = None) -> Line:
        """创建单个刻度线"""
        if size is None:
            size = self.tick_size
            
        # 创建垂直方向的刻度线
        result = Line(size * DOWN, size * UP)
        # 旋转刻度线以匹配数轴的角度
        result.rotate(self.get_angle())
        # 将刻度线移动到对应位置
        result.move_to(self.number_to_point(x))
        # 匹配数轴的样式
        result.match_style(self)
        return result

    def get_tick_marks(self) -> VGroup:
        """返回所有刻度的集合"""
        return self.ticks

    def number_to_point(self, number: float | VectN) -> Vect3 | Vect3Array:
        """将数值转换为空间中的点坐标"""
        start = self.get_points()[0]  # 数轴起点
        end = self.get_points()[-1]   # 数轴终点
        # 计算数值在数轴上的比例位置
        alpha = (number - self.x_min) / (self.x_max - self.x_min)
        # 计算对应位置的空间坐标
        return outer_interpolate(start, end, alpha)

    def point_to_number(self, point: Vect3 | Vect3Array) -> float | VectN:
        """将空间中的点坐标转换为对应的数值"""
        start = self.get_points()[0]  # 数轴起点
        end = self.get_points()[-1]   # 数轴终点
        vect = end - start  # 数轴方向向量
        
        # 计算点在数轴上的比例位置
        proportion = fdiv(
            np.dot(point - start, vect),  # 点到起点的向量与数轴方向的点积
            np.dot(end - start, vect),    # 数轴长度的平方
        )
        
        # 根据比例计算对应的数值
        return interpolate(self.x_min, self.x_max, proportion)

    def n2p(self, number: float | VectN) -> Vect3 | Vect3Array:
        """number_to_point方法的缩写"""
        return self.number_to_point(number)

    def p2n(self, point: Vect3 | Vect3Array) -> float | VectN:
        """point_to_number方法的缩写"""
        return self.point_to_number(point)

    def get_unit_size(self) -> float:
        """获取数轴上一个单位对应的实际空间长度"""
        return self.get_length() / (self.x_max - self.x_min)

    def get_number_mobject(
        self,
        x: float,  # 要显示的数值
        direction: Vect3 | None = None,  # 数字相对于数轴的方向
        buff: float | None = None,  # 数字与数轴的距离
        unit: float = 1.0,  # 单位换算因子
        unit_tex: str = "",  # 单位的LaTeX表示
        **number_config  # 数字的额外配置
    ) -> DecimalNumber:
        """创建并返回表示指定数值的DecimalNumber对象"""
        # 合并默认配置和传入的配置
        number_config = merge_dicts_recursively(
            self.decimal_number_config, number_config,
        )
        
        # 使用默认方向和距离如果未指定
        if direction is None:
            direction = self.line_to_number_direction
        if buff is None:
            buff = self.line_to_number_buff
            
        # 设置单位（如果有）
        if unit_tex:
            number_config["unit"] = unit_tex

        # 创建DecimalNumber对象，处理单位换算
        num_mob = DecimalNumber(x / unit,** number_config)
        # 将数字移动到数轴上对应位置的指定方向
        num_mob.next_to(
            self.number_to_point(x),
            direction=direction,
            buff=buff
        )
        
        # 对齐负数的位置（不考虑负号的宽度）
        if x < 0 and direction[0] == 0:
            num_mob.shift(num_mob[0].get_width() * LEFT / 2)
            
        # 处理单位为1的特殊情况
        if abs(x) == unit and unit_tex:
            center = num_mob.get_center()
            if x > 0:
                num_mob.remove(num_mob[0])  # 移除数字"1"
            else:
                num_mob.remove(num_mob[1])  # 移除数字"1"
                # 调整负号位置
                num_mob[0].next_to(num_mob[1], LEFT, buff=num_mob[0].get_width() / 4)
            num_mob.move_to(center)  # 保持原中心位置
            
        return num_mob

    def add_numbers(
        self,
        x_values: Iterable[float] | None = None,  # 要添加数字的位置列表，默认为None
        excluding: Iterable[float] | None = None,  # 要排除的数字位置，默认为None
        font_size: int = 24,  # 数字的字体大小，默认为24
        **kwargs  # 传递给数字对象的其他配置参数
    ) -> VGroup:
        """
        向数轴添加数字标记，返回包含所有数字的VGroup对象
        """
        # 如果未指定x_values，则使用刻度范围作为默认值
        if x_values is None:
            x_values = self.get_tick_range()

        # 将字体大小添加到配置参数中
        kwargs["font_size"] = font_size

        # 如果未指定排除的数字，则使用类中预定义的numbers_to_exclude
        if excluding is None:
            excluding = self.numbers_to_exclude

        # 创建一个VGroup来存储所有数字对象
        numbers = VGroup()
        # 遍历每个要添加数字的位置
        for x in x_values:
            # 跳过需要排除的数字
            if excluding is not None and x in excluding:
                continue
            # 为当前位置创建数字对象并添加到组中
            numbers.add(self.get_number_mobject(x,** kwargs))
        # 将数字组添加到数轴
        self.add(numbers)
        # 保存数字组的引用
        self.numbers = numbers
        # 返回数字组
        return numbers


class UnitInterval(NumberLine):
    """
    单位区间类，继承自NumberLine，专门用于表示[0,1]区间的数轴
    """
    def __init__(
        self,
        x_range: RangeSpecifier = (0, 1, 0.1),  # 区间范围，默认0到1，步长0.1
        unit_size: float = 10,  # 单位大小，默认10
        big_tick_numbers: list[float] = [0, 1],  # 大刻度位置，默认0和1处
        decimal_number_config: dict = dict(
            num_decimal_places=1,  # 小数位数，默认1位
        ),
        **kwargs  # 传递给父类的其他参数
    ):
        # 调用父类的初始化方法，设置单位区间的特定参数
        super().__init__(
            x_range=x_range,
            unit_size=unit_size,
            big_tick_numbers=big_tick_numbers,
            decimal_number_config=decimal_number_config,** kwargs
        )
