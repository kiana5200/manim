# 从未来版本导入注解功能，允许在类型注解中使用尚未定义的类
from __future__ import annotations

# 导入numpy库，用于数值计算和数组操作
import numpy as np

# 从manimlib导入常量（各种颜色）
from manimlib.constants import BLUE_B, BLUE_D, BLUE_E, GREY_BROWN, DEFAULT_MOBJECT_COLOR
# 导入Mobject基类，所有可显示对象的基类
from manimlib.mobject.mobject import Mobject
# 导入VGroup类，用于组合多个向量图形对象
from manimlib.mobject.types.vectorized_mobject import VGroup
# 导入VMobject类，可矢量化的图形对象基类
from manimlib.mobject.types.vectorized_mobject import VMobject
# 导入平滑速率函数，用于动画过渡
from manimlib.utils.rate_functions import smooth

# 导入类型检查相关模块
from typing import TYPE_CHECKING

# 条件导入，仅在类型检查时生效
if TYPE_CHECKING:
    # 导入类型注解所需的类型
    from typing import Callable, List, Iterable
    from manimlib.typing import ManimColor, Vect3, Self


class AnimatedBoundary(VGroup):
    """
    为向量图形对象创建动画边界效果的类
    用于生成沿着图形边界移动的动态描边效果
    """
    def __init__(
        self,
        vmobject: VMobject,
        colors: List[ManimColor] = [BLUE_D, BLUE_B, BLUE_E, GREY_BROWN],
        max_stroke_width: float = 3.0,
        cycle_rate: float = 0.5,
        back_and_forth: bool = True,
        draw_rate_func: Callable[[float], float] = smooth,
        fade_rate_func: Callable[[float], float] = smooth,
        **kwargs
    ):
        # 调用父类VGroup的初始化方法，传入关键字参数
        super().__init__(** kwargs)
        # 存储要添加动画边界的向量图形对象
        self.vmobject: VMobject = vmobject
        # 定义动画边界使用的颜色序列
        self.colors = colors
        # 定义动画边界的最大描边宽度
        self.max_stroke_width = max_stroke_width
        # 定义动画循环的速率（每秒循环次数）
        self.cycle_rate = cycle_rate
        # 定义动画是否来回移动（True为来回，False为单向循环）
        self.back_and_forth = back_and_forth
        # 定义绘制动画的速率函数（控制动画进度的变化方式）
        self.draw_rate_func = draw_rate_func
        # 定义淡出动画的速率函数
        self.fade_rate_func = fade_rate_func

        # 创建两个边界副本，用于实现动画效果
        self.boundary_copies: list[VMobject] = [
            # 复制原始图形，设置初始样式（无描边宽度，无填充透明度）
            vmobject.copy().set_style(
                stroke_width=0,
                fill_opacity=0
            )
            for x in range(2)  # 创建两个副本
        ]
        # 将边界副本添加到当前组中
        self.add(*self.boundary_copies)
        # 记录总动画时间
        self.total_time: float = 0
        # 添加更新器，用于每一帧更新边界副本的状态
        self.add_updater(
            # 定义更新函数，接收当前实例和时间增量dt
            lambda m, dt: self.update_boundary_copies(dt)
        )

    def update_boundary_copies(self, dt: float) -> Self:
        # Not actual time, but something which passes at
        # an altered rate to make the implementation below
        # cleaner
        time = self.total_time * self.cycle_rate
        growing, fading = self.boundary_copies
        colors = self.colors
        msw = self.max_stroke_width
        vmobject = self.vmobject

        index = int(time % len(colors))
        alpha = time % 1
        draw_alpha = self.draw_rate_func(alpha)
        fade_alpha = self.fade_rate_func(alpha)

        if self.back_and_forth and int(time) % 2 == 1:
            bounds = (1 - draw_alpha, 1)
        else:
            bounds = (0, draw_alpha)
        self.full_family_become_partial(growing, vmobject, *bounds)
        growing.set_stroke(colors[index], width=msw)

        if time >= 1:
            self.full_family_become_partial(fading, vmobject, 0, 1)
            fading.set_stroke(
                color=colors[index - 1],
                width=(1 - fade_alpha) * msw
            )

        self.total_time += dt
        return self

    def full_family_become_partial(
        self,
        mob1: VMobject,
        mob2: VMobject,
        a: float,
        b: float
    ) -> Self:
        family1 = mob1.family_members_with_points()
        family2 = mob2.family_members_with_points()
        for sm1, sm2 in zip(family1, family2):
            sm1.pointwise_become_partial(sm2, a, b)
        return self


class TracedPath(VMobject):
    def __init__(
        self,
        traced_point_func: Callable[[], Vect3],
        time_traced: float = np.inf,
        time_per_anchor: float = 1.0 / 15,
        stroke_width: float | Iterable[float] = 2.0,
        stroke_color: ManimColor = DEFAULT_MOBJECT_COLOR,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.traced_point_func = traced_point_func
        self.time_traced = time_traced
        self.time_per_anchor = time_per_anchor
        self.time: float = 0
        self.traced_points: list[np.ndarray] = []
        self.add_updater(lambda m, dt: m.update_path(dt))
        self.always.set_stroke(stroke_color, stroke_width)

    def update_path(self, dt: float) -> Self:
        if dt == 0:
            return self
        point = self.traced_point_func().copy()
        self.traced_points.append(point)

        if self.time_traced < np.inf:
            n_relevant_points = int(self.time_traced / dt + 0.5)
            n_tps = len(self.traced_points)
            if n_tps < n_relevant_points:
                points = self.traced_points + [point] * (n_relevant_points - n_tps)
            else:
                points = self.traced_points[n_tps - n_relevant_points:]
            # Every now and then refresh the list
            if n_tps > 10 * n_relevant_points:
                self.traced_points = self.traced_points[-n_relevant_points:]
        else:
            points = self.traced_points

        if points:
            self.set_points_smoothly(points)

        self.time += dt
        return self


class TracingTail(TracedPath):
    def __init__(
        self,
        mobject_or_func: Mobject | Callable[[], np.ndarray],
        time_traced: float = 1.0,
        stroke_width: float | Iterable[float] = (0, 3),
        stroke_opacity: float | Iterable[float] = (0, 1),
        stroke_color: ManimColor = DEFAULT_MOBJECT_COLOR,
        **kwargs
    ):
        if isinstance(mobject_or_func, Mobject):
            func = mobject_or_func.get_center
        else:
            func = mobject_or_func
        super().__init__(
            func,
            time_traced=time_traced,
            stroke_width=stroke_width,
            stroke_opacity=stroke_opacity,
            stroke_color=stroke_color,
            **kwargs
        )
        self.add_updater(lambda m: m.set_stroke(width=stroke_width, opacity=stroke_opacity))
