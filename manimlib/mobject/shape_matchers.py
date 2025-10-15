from __future__ import annotations

from colour import Color

from manimlib.config import manim_config  # 导入配置模块
from manimlib.constants import BLACK, RED, YELLOW, DEFAULT_MOBJECT_COLOR  # 导入颜色常量
from manimlib.constants import DL, DOWN, DR, LEFT, RIGHT, UL, UR  # 导入方向常量
from manimlib.constants import SMALL_BUFF  # 导入间距常量
from manimlib.mobject.geometry import Line  # 导入基础图形Line
from manimlib.mobject.geometry import Rectangle  # 导入基础图形Rectangle
from manimlib.mobject.types.vectorized_mobject import VGroup  # 导入向量组类
from manimlib.mobject.types.vectorized_mobject import VMobject  # 导入向量对象基类

from typing import TYPE_CHECKING  # 用于类型检查

if TYPE_CHECKING:
    from typing import Sequence
    from manimlib.mobject.mobject import Mobject
    from manimlib.typing import ManimColor, Self


class SurroundingRectangle(Rectangle):
    """围绕目标对象的矩形边框"""
    def __init__(
        self,
        mobject: Mobject,  # 要围绕的目标对象
        buff: float = SMALL_BUFF,  # 与目标对象的间距
        color: ManimColor = YELLOW,  # 边框颜色，默认黄色
        **kwargs  # 传递给Rectangle的其他参数
    ):
        # 调用父类Rectangle的构造方法
        super().__init__(color=color,** kwargs)
        self.buff = buff  # 存储间距值
        self.surround(mobject)  # 围绕目标对象调整大小和位置
        # 如果目标对象固定在屏幕上，当前边框也随之固定
        if mobject.is_fixed_in_frame():
            self.fix_in_frame()

    def surround(self, mobject, buff=None) -> Self:
        """调整矩形大小和位置以围绕目标对象"""
        self.mobject = mobject  # 存储目标对象引用
        # 使用指定的间距，默认使用初始化时的buff
        self.buff = buff if buff is not None else self.buff
        # 调用父类的surround方法实现围绕逻辑
        super().surround(mobject, self.buff)
        return self

    def set_buff(self, buff) -> Self:
        """更新间距并重新调整围绕位置"""
        self.buff = buff  # 更新间距值
        self.surround(self.mobject)  # 重新围绕目标对象
        return self


class BackgroundRectangle(SurroundingRectangle):
    """作为目标对象背景的矩形（填充区域）"""
    def __init__(
        self,
        mobject: Mobject,  # 要作为背景的目标对象
        color: ManimColor = None,  # 背景颜色，默认使用场景背景色
        stroke_width: float = 0,  # 边框宽度，默认为0（无边界）
        stroke_opacity: float = 0,  # 边框透明度，默认为0
        fill_opacity: float = 0.75,  # 填充透明度，默认0.75
        buff: float = 0,  # 与目标对象的间距，默认为0（紧密包围）
        **kwargs  # 传递给父类的其他参数
    ):
        # 如果未指定颜色，使用配置中的相机背景色
        if color is None:
            color = manim_config.camera.background_color
        # 调用父类构造方法，设置背景相关样式
        super().__init__(
            mobject,
            color=color,
            stroke_width=stroke_width,
            stroke_opacity=stroke_opacity,
            fill_opacity=fill_opacity,
            buff=buff,** kwargs
        )
        # 保存初始填充透明度，用于后续动画
        self.original_fill_opacity = fill_opacity

    def pointwise_become_partial(self, mobject: Mobject, a: float, b: float) -> Self:
        """部分显示背景（用于动画过渡）"""
        # 根据比例b设置填充透明度（0~original_fill_opacity）
        self.set_fill(opacity=b * self.original_fill_opacity)
        return self

    def set_style(
        self,
        stroke_color: ManimColor | None = None,
        stroke_width: float | None = None,
        fill_color: ManimColor | None = None,
        fill_opacity: float | None = None,
        family: bool = True,
        **kwargs
    ) -> Self:
        """重写样式设置方法，固定部分样式（仅允许修改填充透明度）"""
        # 调用VMobject的set_style方法，强制设置部分样式为固定值
        VMobject.set_style(
            self,
            stroke_color=BLACK,  # 固定边框颜色为黑色
            stroke_width=0,      # 固定边框宽度为0
            fill_color=BLACK,    # 固定填充颜色为黑色（实际由color属性控制）
            fill_opacity=fill_opacity  # 仅允许修改填充透明度
        )
        return self

    def get_fill_color(self) -> Color:
        """获取填充颜色（返回自身color属性）"""
        return Color(self.color)


class Cross(VGroup):
    """十字交叉图形，用于覆盖在目标对象上"""
    def __init__(
        self,
        mobject: Mobject,  # 要覆盖的目标对象
        stroke_color: ManimColor = RED,  # 线条颜色，默认红色
        stroke_width: float | Sequence[float] = [0, 6, 0],  # 线条宽度，支持渐变
        **kwargs  # 传递给VGroup的其他参数
    ):
        # 初始化十字：由两条对角线组成
        super().__init__(
            Line(UL, DR),  # 左上到右下的线
            Line(UR, DL),  # 右上到左下的线
        )
        self.insert_n_curves(20)  # 插入曲线段，使线条更平滑
        # 调整十字大小以匹配目标对象（拉伸模式）
        self.replace(mobject, stretch=True)
        # 设置线条样式（颜色和宽度）
        self.set_stroke(stroke_color, width=stroke_width)


class Underline(Line):
    """下划线图形，用于目标对象下方"""
    def __init__(
        self,
        mobject: Mobject,  # 要添加下划线的目标对象
        buff: float = SMALL_BUFF,  # 与目标对象的垂直间距
        stroke_color=DEFAULT_MOBJECT_COLOR,  # 线条颜色，默认使用对象颜色
        stroke_width: float | Sequence[float] = [0, 3, 3, 0],  # 线条宽度，支持渐变
        stretch_factor=1.2,  # 拉伸因子，使下划线略宽于目标对象
        **kwargs  # 传递给Line的其他参数
    ):
        # 初始化一条从左到右的线作为下划线基础
        super().__init__(LEFT, RIGHT,** kwargs)
        # 如果线条宽度是序列（渐变宽度），插入对应数量的曲线段
        if not isinstance(stroke_width, (float, int)):
            self.insert_n_curves(len(stroke_width) - 2)
        # 设置线条样式
        self.set_stroke(stroke_color, stroke_width)
        # 调整下划线宽度（目标对象宽度 × 拉伸因子）
        self.set_width(mobject.get_width() * stretch_factor)
        # 将下划线放置在目标对象下方，保持指定间距
        self.next_to(mobject, DOWN, buff=buff)