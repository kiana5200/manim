from __future__ import annotations

from colour import Color

from manimlib.config import manim_config
from manimlib.constants import BLACK, RED, YELLOW, DEFAULT_MOBJECT_COLOR
from manimlib.constants import DL, DOWN, DR, LEFT, RIGHT, UL, UR
from manimlib.constants import SMALL_BUFF
from manimlib.mobject.geometry import Line
from manimlib.mobject.geometry import Rectangle
from manimlib.mobject.types.vectorized_mobject import VGroup
from manimlib.mobject.types.vectorized_mobject import VMobject

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Sequence
    from manimlib.mobject.mobject import Mobject
    from manimlib.typing import ManimColor, Self


class SurroundingRectangle(Rectangle):
    def __init__(
        self,
        mobject: Mobject,
        buff: float = SMALL_BUFF,
        color: ManimColor = YELLOW,** kwargs
    ):
        # 调用父类Rectangle的初始化方法，设置颜色及其他参数
        super().__init__(color=color, **kwargs)
        # 存储包围时的间距
        self.buff = buff
        # 包围目标mobject
        self.surround(mobject)
        # 如果目标mobject固定在帧中，则当前矩形也固定
        if mobject.is_fixed_in_frame():
            self.fix_in_frame()

    def surround(self, mobject, buff=None) -> Self:
        # 记录被包围的mobject
        self.mobject = mobject
        # 确定间距，优先使用传入的buff，否则用实例自身的buff
        self.buff = buff if buff is not None else self.buff
        # 调用父类的surround方法实现包围
        super().surround(mobject, self.buff)
        return self

    def set_buff(self, buff) -> Self:
        # 设置新的间距值
        self.buff = buff
        # 根据新间距重新包围mobject
        self.surround(self.mobject)
        return self


class BackgroundRectangle(SurroundingRectangle):
    def __init__(
        self,
        mobject: Mobject,
        color: ManimColor = None,
        stroke_width: float = 0,
        stroke_opacity: float = 0,
        fill_opacity: float = 0.75,
        buff: float = 0,** kwargs
    ):
        # 若未指定颜色，默认使用摄像机背景色
        if color is None:
            color = manim_config.camera.background_color
        # 调用父类SurroundingRectangle的初始化方法
        super().__init__(
            mobject,
            color=color,
            stroke_width=stroke_width,
            stroke_opacity=stroke_opacity,
            fill_opacity=fill_opacity,
            buff=buff,
            **kwargs
        )
        # 保存初始填充不透明度，用于后续部分显示时计算
        self.original_fill_opacity = fill_opacity

    def pointwise_become_partial(self, mobject: Mobject, a: float, b: float) -> Self:
        # 根据比例b设置填充不透明度（基于初始值）
        self.set_fill(opacity=b * self.original_fill_opacity)
        return self

    def set_style(
        self,
        stroke_color: ManimColor | None = None,
        stroke_width: float | None = None,
        fill_color: ManimColor | None = None,
        fill_opacity: float | None = None,
        family: bool = True,** kwargs
    ) -> Self:
        # 除填充不透明度外，其他样式不可更改（固定为黑色、无描边）
        VMobject.set_style(
            self,
            stroke_color=BLACK,  # 固定描边颜色为黑色
            stroke_width=0,      # 固定描边宽度为0（无描边）
            fill_color=BLACK,    # 固定填充颜色为黑色
            fill_opacity=fill_opacity  # 仅填充不透明度可通过参数设置
        )
        return self

    def get_fill_color(self) -> Color:
        # 返回当前对象的填充颜色（基于自身color属性）
        return Color(self.color)


class Cross(VGroup):
    def __init__(
        self,
        mobject: Mobject,
        stroke_color: ManimColor = RED,
        stroke_width: float | Sequence[float] = [0, 6, 0],** kwargs
    ):
        # 初始化父类VGroup，包含两条交叉线（左上到右下、右上到左下）
        super().__init__(
            Line(UL, DR),  # 从左上到右下的线
            Line(UR, DL),  # 从右上到左下的线
        )
        # 插入曲线段使线条更平滑
        self.insert_n_curves(20)
        # 替换为与目标mobject相同大小和位置（拉伸适配）
        self.replace(mobject, stretch=True)
        # 设置交叉线的颜色和线宽
        self.set_stroke(stroke_color, width=stroke_width)


class Underline(Line):
    def __init__(
        self,
        mobject: Mobject,
        buff: float = SMALL_BUFF,
        stroke_color=DEFAULT_MOBJECT_COLOR,
        stroke_width: float | Sequence[float] = [0, 3, 3, 0],
        stretch_factor=1.2,
        **kwargs
    ):
        # 初始化父类Line，方向从左到右
        super().__init__(LEFT, RIGHT, **kwargs)
        # 若线宽是序列（渐变线宽），插入对应数量的曲线段
        if not isinstance(stroke_width, (float, int)):
            self.insert_n_curves(len(stroke_width) - 2)
        # 设置下划线的颜色和线宽
        self.set_stroke(stroke_color, stroke_width)
        # 设置下划线宽度（目标对象宽度乘以拉伸因子）
        self.set_width(mobject.get_width() * stretch_factor)
        # 将下划线定位到目标对象下方，保持指定间距
        self.next_to(mobject, DOWN, buff=buff)