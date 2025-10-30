# 导入Python 3.7+的特性：允许在类定义中直接引用类名作为类型注解（无需提前声明）
from __future__ import annotations

# 从Manim常量库导入常用颜色（黑色、浅灰色）和帧高度常量
from manimlib.constants import BLACK, GREY_E
from manimlib.constants import FRAME_HEIGHT

# 从Manim几何图形模块导入基础矩形类
from manimlib.mobject.geometry import Rectangle

# 导入类型检查相关模块，避免运行时依赖
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from manimlib.typing import ManimColor  # 仅在类型检查时导入颜色类型注解


class ScreenRectangle(Rectangle):
    """
    基础屏幕比例矩形类,继承自Manim的Rectangle,支持自定义宽高比和高度。
    核心功能是根据指定的宽高比自动计算宽度,生成符合屏幕比例的矩形(如16:9)。
    """
    def __init__(
        self,
        aspect_ratio: float = 16.0 / 9.0,  # 默认宽高比为16:9（常见屏幕比例）
        height: float = 4,                 # 默认高度为4（Manim坐标系单位）
        **kwargs                           # 传递给父类Rectangle的额外参数（如颜色、边框等）
    ):
        # 调用父类Rectangle的初始化方法
        # 宽度由“宽高比 × 高度”自动计算，确保符合指定的屏幕比例
        super().__init__(
            width=aspect_ratio * height,  # 自动计算宽度
            height=height,                # 传入指定高度
            **kwargs                      # 传递额外配置（如fill_color、stroke_width等）
        )


class FullScreenRectangle(ScreenRectangle):
    """
    全屏矩形类,继承自ScreenRectangle,默认占据Manim的整个帧高度,常用于作为背景。
    预设了浅灰色填充、无边框的样式，可直接作为基础全屏背景使用。
    """
    def __init__(
        self,
        height: float = FRAME_HEIGHT,          # 高度默认等于Manim的帧高度（全屏高度）
        fill_color: ManimColor = GREY_E,       # 填充色默认浅灰色（GREY_E）
        fill_opacity: float = 1,               # 填充透明度默认1（完全不透明）
        stroke_width: float = 0,               # 边框宽度默认0（无边框）
        **kwargs,                              # 传递给父类ScreenRectangle的额外参数
    ):
        # 调用父类ScreenRectangle的初始化方法
        # 继承16:9宽高比，高度设为全屏高度，同时应用预设样式
        super().__init__(
            height=height,
            fill_color=fill_color,
            fill_opacity=fill_opacity,
            stroke_width=stroke_width,
            **kwargs
        )


class FullScreenFadeRectangle(FullScreenRectangle):
    """
    全屏半透明遮罩矩形类,继承自FullScreenRectangle,常用于作为画面遮罩（如聚焦特定内容）。
    预设了黑色半透明填充、无边框的样式，可直接作为遮罩层使用。
    """
    def __init__(
        self,
        stroke_width: float = 0.0,             # 边框宽度默认0（无边框）
        fill_color: ManimColor = BLACK,        # 填充色默认黑色（BLACK）
        fill_opacity: float = 0.7,             # 填充透明度默认0.7（半透明，保留部分背景可见）
        **kwargs,                              # 传递给父类FullScreenRectangle的额外参数
    ):
        # 调用父类FullScreenRectangle的初始化方法
        # 继承全屏尺寸，应用黑色半透明的遮罩样式
        super().__init__(
            stroke_width=stroke_width,
            fill_color=fill_color,
            fill_opacity=fill_opacity,
            **kwargs
        )