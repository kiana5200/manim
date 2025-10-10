#从__future__导入annotations，支持在类型提示中使用尚未定义的类
from __future__ import annotations

# 从manimlib.constants导入常用颜色常量和帧高度常量
from manimlib.constants import BLACK, GREY_E
from manimlib.constants import FRAME_HEIGHT
# 从manimlib.mobject.geometry导入Rectangle基类
from manimlib.mobject.geometry import Rectangle

# 导入类型检查相关模块
from typing import TYPE_CHECKING
# 如果是类型检查阶段
if TYPE_CHECKING:
    # 导入ManimColor类型用于类型提示
    from manimlib.typing import ManimColor


class ScreenRectangle(Rectangle):
    """屏幕比例的矩形类，继承自Rectangle"""
    
    def __init__(
        self,
        aspect_ratio: float = 16.0 / 9.0,  # 宽高比，默认16:9
        height: float = 4,  # 矩形高度，默认4个单位
        **kwargs  # 其他传递给父类的关键字参数
    ):
        # 调用父类的初始化方法
        super().__init__(
            width=aspect_ratio * height,  # 宽度由宽高比和高度计算得出
            height=height,  # 设置高度
            **kwargs  # 传递其他参数
        )


class FullScreenRectangle(ScreenRectangle):
    def __init__(
        self,
        height: float = FRAME_HEIGHT,
        fill_color: ManimColor = GREY_E,
        fill_opacity: float = 1,
        stroke_width: float = 0,
        **kwargs,
    ):
        super().__init__(
            height=height,
            fill_color=fill_color,
            fill_opacity=fill_opacity,
            stroke_width=stroke_width,
            **kwargs
        )


class FullScreenFadeRectangle(FullScreenRectangle):
    def __init__(
        self,
        stroke_width: float = 0.0,
        fill_color: ManimColor = BLACK,
        fill_opacity: float = 0.7,
        **kwargs,
    ):
        super().__init__(
            stroke_width=stroke_width,
            fill_color=fill_color,
            fill_opacity=fill_opacity,
        )
