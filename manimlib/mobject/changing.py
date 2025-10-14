# 导入Python 3.7+的特性：允许在类定义中直接引用类名作为类型注解（无需提前声明）
from __future__ import annotations

# 导入NumPy库，用于数值计算和数组操作
import numpy as np

# 从Manim常量库导入常用颜色（不同深浅的蓝色、灰棕色、默认图形颜色）
from manimlib.constants import BLUE_B, BLUE_D, BLUE_E, GREY_BROWN, DEFAULT_MOBJECT_COLOR

# 导入Manim的基础图形对象类Mobject
from manimlib.mobject.mobject import Mobject

# 导入Manim的矢量图形对象类（VGroup用于组合多个VMobject，VMobject是可填充的矢量图形）
from manimlib.mobject.types.vectorized_mobject import VGroup
from manimlib.mobject.types.vectorized_mobject import VMobject

# 导入Manim的缓动函数，用于平滑动画效果
from manimlib.utils.rate_functions import smooth

# 导入类型检查相关模块，避免运行时依赖
from typing import TYPE_CHECKING

# 仅在类型检查模式下导入类型注解（避免运行时循环依赖或冗余导入）
if TYPE_CHECKING:
    from typing import Callable, List, Iterable
    from manimlib.typing import ManimColor, Vect3, Self


class AnimatedBoundary(VGroup):
    """
    一个用于创建“流动边界”动画效果的复合图形类。
    它通过创建原图形的两个边界副本，并在时间上连续更新它们的绘制进度和样式，
    实现类似脉冲或流光沿着图形边界流动的视觉效果。

    核心思想：
    1. 创建两个不可见的图形副本（仅保留边界）。
    2. 随时间推移，一个副本逐渐“绘制”边界，另一个副本逐渐“擦除”边界。
    3. 同时调整边界的颜色和宽度，以增强流动感。
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
        """
        初始化AnimatedBoundary对象。

        参数:
            vmobject (VMobject): 要添加动画边界的目标矢量图形对象。
            colors (List[ManimColor]): 边界动画使用的颜色序列，用于颜色渐变。
            max_stroke_width (float): 边界动画的最大线宽。
            cycle_rate (float): 动画循环的速率（周期的倒数）。值越大，动画越快。
            back_and_forth (bool): 是否在动画结束时反向播放。如果为False，则动画到达终点后直接跳回起点。
            draw_rate_func (Callable[[float], float]): 控制边界绘制进度的缓动函数（输入0-1，输出0-1）。
            fade_rate_func (Callable[[float], float]): 控制边界透明度和线宽变化的缓动函数。
            **kwargs: 传递给父类VGroup的额外参数。
        """
        # 1. 调用父类VGroup的初始化方法
        super().__init__(**kwargs)

        # 2. 保存动画配置参数
        self.vmobject: VMobject = vmobject
        self.colors = colors
        self.max_stroke_width = max_stroke_width
        self.cycle_rate = cycle_rate
        self.back_and_forth = back_and_forth
        self.draw_rate_func = draw_rate_func
        self.fade_rate_func = fade_rate_func

        # 3. 创建两个边界副本
        # 这两个副本将用于实现“绘制”和“擦除”的动画效果
        self.boundary_copies: list[VMobject] = [
            vmobject.copy().set_style(
                stroke_width=0,  # 初始线宽为0（不可见）
                fill_opacity=0   # 填充透明度为0（不可见）
            )
            for _ in range(2)  # 创建两个完全相同的副本
        ]

        # 4. 将边界副本添加到VGroup中
        # 这样它们就会作为一个整体被管理和显示
        self.add(*self.boundary_copies)

        # 5. 初始化时间跟踪变量
        # 用于记录自动画开始以来经过的总时间
        self.total_time: float = 0

        # 6. 添加动画更新器
        # 这是实现连续动画的关键。每帧都会调用update_boundary_copies方法。
        self.add_updater(
            lambda m, dt: self.update_boundary_copies(dt)
        )

    # 动画更新函数，每帧被调用一次，用于更新两个边界副本的状态，实现流动边界效果。
    def update_boundary_copies(self, dt: float) -> Self:
        # Not actual time, but something which passes at
        # an altered rate to make the implementation below
        # cleaner
        # 1. 计算动画进度时间
        # 不是真实时间，而是根据循环速率调整后的时间，使后续逻辑更简洁
        time = self.total_time * self.cycle_rate

        # 2. 提取内部状态和配置
        growing, fading = self.boundary_copies  # 两个边界副本：一个“生长”，一个“消退”
        colors = self.colors                    # 颜色列表
        msw = self.max_stroke_width            # 最大线宽
        vmobject = self.vmobject               # 原始图形

        # 3. 计算当前颜色索引和动画相位
        index = int(time % len(colors))  # 确定当前使用的颜色在列表中的索引
        alpha = time % 1                 # 计算当前周期内的相位（0到1之间）

        # 4. 应用缓动函数计算绘制和消退的实际进度
        draw_alpha = self.draw_rate_func(alpha)  # 绘制进度（考虑缓动效果）
        fade_alpha = self.fade_rate_func(alpha)  # 消退进度（考虑缓动效果）

        # 5. 确定“生长”边界的绘制范围
        if self.back_and_forth and int(time) % 2 == 1:
            # 如果是往返模式且当前周期为奇数，则反向绘制（从终点向起点）
            bounds = (1 - draw_alpha, 1)
        else:
            # 否则，正向绘制（从起点向终点）
            bounds = (0, draw_alpha)

        # 6. 更新“生长”边界的外观
        # 将“生长”副本设置为原始图形的指定部分（由bounds定义）
        self.full_family_become_partial(growing, vmobject, *bounds)
        # 设置“生长”边界的颜色和最大线宽
        growing.set_stroke(colors[index], width=msw)

        # 7. 更新“消退”边界的外观（当动画至少进行了一个完整周期后）
        if time >= 1:
            # 将“消退”副本设置为完整的原始图形
            self.full_family_become_partial(fading, vmobject, 0, 1)
            # 设置“消退”边界的颜色（上一个颜色）和逐渐减小的线宽
            fading.set_stroke(
                color=colors[index - 1],
                width=(1 - fade_alpha) * msw
            )

        # 8. 更新总时间，为下一帧做准备
        self.total_time += dt

        return self


    # 将一个VMobject及其所有子对象（整个家族）的形状，设置为另一个VMobject家族的指定部分。
    # 这是一个“批量”操作，确保所有相关的子对象都被同步更新，常用于实现复杂图形的部分显示动画。
    def full_family_become_partial(
        self,
        mob1: VMobject,
        mob2: VMobject,
        a: float,
        b: float
    ) -> Self:
        # 1. 获取两个VMobject家族中所有包含顶点数据的成员
        # family_members_with_points()会递归地收集所有子对象，但只保留那些有顶点数据的
        family1 = mob1.family_members_with_points()
        family2 = mob2.family_members_with_points()
        # 2. 遍历两个家族的对应子对象，并进行部分形状的同步
        # zip会将两个家族中位置对应的子对象配对
        for sm1, sm2 in zip(family1, family2):
            # 对每个子对象对，调用pointwise_become_partial方法
            # 该方法会将sm1的顶点设置为sm2顶点路径上从比例a到比例b的那一部分
            sm1.pointwise_become_partial(sm2, a, b)
            
        return self


class TracedPath(VMobject):
    """
    用于动态追踪一个点的运动轨迹并绘制路径的矢量图形类。
    它通过周期性地记录一个函数返回的点的位置，将这些点连接成一条连续的曲线，从而实现轨迹的实时绘制。
    常用于展示物体的运动路径、函数的动态变化等场景。
    """
    def __init__(
        self,
        traced_point_func: Callable[[], Vect3],
        time_traced: float = np.inf,
        time_per_anchor: float = 1.0 / 15,
        stroke_width: float | Iterable[float] = 2.0,
        stroke_color: ManimColor = DEFAULT_MOBJECT_COLOR,
        **kwargs
    ):
        """
        初始化TracedPath对象。

        参数:
            traced_point_func (Callable[[], Vect3]): 一个无参数的函数，每次调用时返回一个需要被追踪的点的坐标（Vect3）。
            time_traced (float): 轨迹的最长记录时间。超过这个时间的旧点将被丢弃，实现“轨迹逐渐消失”的效果。默认为无穷大（保留所有历史轨迹）。
            time_per_anchor (float): 每记录一个轨迹点（锚点）的时间间隔。值越小，轨迹越平滑，但性能开销越大。默认为1/15秒（约15fps）。
            stroke_width (float | Iterable[float]): 轨迹线的宽度。可以是一个固定值，也可以是一个可迭代对象以实现渐变宽度。
            stroke_color (ManimColor): 轨迹线的颜色。默认为Manim的默认图形颜色。
            **kwargs: 传递给父类VMobject的额外参数。
        """
        # 1. 调用父类VMobject的初始化方法
        super().__init__(**kwargs)

        # 2. 保存核心追踪配置
        self.traced_point_func = traced_point_func  # 用于获取被追踪点坐标的函数
        self.time_traced = time_traced              # 轨迹的最长记录时间
        self.time_per_anchor = time_per_anchor      # 记录点的时间间隔

        # 3. 初始化状态变量
        self.time: float = 0                        # 自追踪开始以来的累计时间
        self.traced_points: list[np.ndarray] = []   # 用于存储所有记录的轨迹点坐标

        # 4. 添加动画更新器
        # 这是实现动态追踪的关键。每帧都会调用update_path方法来检查是否需要记录新点。
        self.add_updater(lambda m, dt: m.update_path(dt))

        # 5. 设置轨迹线的样式
        # 使用always.set_stroke确保样式（颜色和宽度）在任何时候都保持一致
        self.always.set_stroke(stroke_color, stroke_width)

    def update_path(self, dt: float) -> Self:
        """
        轨迹更新函数，每帧被调用一次，用于记录新的轨迹点并更新显示的路径。

        参数:
            dt (float): 从上一帧到当前帧的时间间隔。

        返回:
            Self: 返回自身，便于链式调用。
        """
        # 1. 如果时间间隔为0，则不做任何更新，直接返回
        if dt == 0:
            return self
        # 2. 获取并记录当前被追踪点的坐标
        # 调用追踪函数获取点坐标，并使用.copy()确保我们拥有独立的数据副本
        point = self.traced_point_func().copy()
        self.traced_points.append(point)

        # 3. 根据最大追踪时间，筛选出需要显示的相关轨迹点
        if self.time_traced < np.inf:
            # 计算在`time_traced`时间段内大约能记录多少个点
            n_relevant_points = int(self.time_traced / dt + 0.5)
            n_tps = len(self.traced_points)
            # 如果当前记录的点少于需要显示的点数，则用最新的点填充不足的部分
            # 这可以避免在动画开始时轨迹太短
            if n_tps < n_relevant_points:
                points = self.traced_points + [point] * (n_relevant_points - n_tps)
            # 否则，只保留最近的`n_relevant_points`个点
            else:
                points = self.traced_points[n_tps - n_relevant_points:]
            # Every now and then refresh the list
            # 性能优化：偶尔清理一下`traced_points`列表，防止它变得过大
            if n_tps > 10 * n_relevant_points:
                self.traced_points = self.traced_points[-n_relevant_points:]
        # 4. 如果不限制追踪时间，则使用所有记录的点
        else:
            points = self.traced_points

        # 5. 使用筛选后的点更新路径
        # 如果有需要显示的点，则使用`set_points_smoothly`来平滑地更新路径
        if points:
            self.set_points_smoothly(points)

        # 6. 更新总时间，为下一帧做准备
        self.time += dt
        return self


class TracingTail(TracedPath):
    """
    用于为移动的Mobject创建一个带有渐变尾迹效果的轨迹。
    继承自TracedPath，它不仅记录物体的运动路径，还能使轨迹从起点到终点在宽度、透明度上产生平滑的渐变，模拟出“尾巴”逐渐消失的视觉效果。
    """
    def __init__(
        self,
        mobject_or_func: Mobject | Callable[[], np.ndarray],
        time_traced: float = 1.0,
        stroke_width: float | Iterable[float] = (0, 3),
        stroke_opacity: float | Iterable[float] = (0, 1),
        stroke_color: ManimColor = DEFAULT_MOBJECT_COLOR,
        **kwargs
    ):
        """
        初始化TracingTail对象。

        参数:
            mobject_or_func (Mobject | Callable[[], np.ndarray]): 要么是一个Mobject（将追踪其中心），要么是一个返回坐标点的函数。
            time_traced (float): 尾迹的持续时间（秒）。默认为1秒。
            stroke_width (float | Iterable[float]): 尾迹的宽度。如果是一个可迭代对象（如(0, 3)），则尾迹从起点到终点宽度渐变。
            stroke_opacity (float | Iterable[float]): 尾迹的透明度。如果是一个可迭代对象（如(0, 1)），则尾迹从起点到终点透明度渐变。
            stroke_color (ManimColor): 尾迹的颜色。默认为Manim的默认图形颜色。
            **kwargs: 传递给父类TracedPath的额外参数。
        """
        # 1. 确定追踪函数
        if isinstance(mobject_or_func, Mobject):
            # 如果传入的是一个Mobject，默认追踪其中心点
            func = mobject_or_func.get_center
        else:
            # 否则，直接使用传入的函数
            func = mobject_or_func

        # 2. 调用父类TracedPath的初始化方法
        # 传递追踪函数和所有样式参数
        super().__init__(
            func,
            time_traced=time_traced,
            stroke_width=stroke_width,
            stroke_opacity=stroke_opacity,
            stroke_color=stroke_color,
            **kwargs
        )

        # 3. 添加样式更新器
        # 这个更新器确保尾迹的样式（宽度和透明度）在每一帧都被重新应用。
        # 这对于实现渐变效果至关重要，因为它会根据路径上点的位置动态计算样式。
        self.add_updater(lambda m: m.set_stroke(width=stroke_width, opacity=stroke_opacity))
