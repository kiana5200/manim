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
    """
    更新边界副本的状态，实现动画效果
    dt: 时间增量，即从上一帧到当前帧的时间间隔
    """
    # 不是实际时间，而是经过调整速率的时间，使下面的实现更简洁
    # 用于控制动画的进度，结合循环速率调整时间流逝速度
    time = self.total_time * self.cycle_rate
    # 分别获取用于"生长"和"淡出"的两个边界副本
    growing, fading = self.boundary_copies
    # 获取颜色序列
    colors = self.colors
    # 获取最大描边宽度
    msw = self.max_stroke_width
    # 获取原始向量图形对象
    vmobject = self.vmobject

    # 根据时间计算当前应使用的颜色索引（循环使用颜色序列）
    index = int(time % len(colors))
    # 计算当前周期内的进度（0到1之间）
    alpha = time % 1
    # 根据绘制速率函数计算绘制进度
    draw_alpha = self.draw_rate_func(alpha)
    # 根据淡出速率函数计算淡出进度
    fade_alpha = self.fade_rate_func(alpha)

    # 如果是来回模式且处于奇数周期，反转绘制方向
    if self.back_and_forth and int(time) % 2 == 1:
        # 边界范围为从(1-draw_alpha)到1（反向绘制）
        bounds = (1 - draw_alpha, 1)
    else:
        # 边界范围为从0到draw_alpha（正向绘制）
        bounds = (0, draw_alpha)
    # 更新"生长"中的边界副本，使其显示原始图形的指定部分
    self.full_family_become_partial(growing, vmobject, *bounds)
    # 设置"生长"边界的颜色和宽度
    growing.set_stroke(colors[index], width=msw)

    # 当时间超过1（即至少完成一个周期），开始处理"淡出"效果
    if time >= 1:
        # 让"淡出"的边界副本显示原始图形的完整部分
        self.full_family_become_partial(fading, vmobject, 0, 1)
        # 设置"淡出"边界的颜色（上一个颜色）和宽度（随时间减小）
        fading.set_stroke(
            color=colors[index - 1],
            width=(1 - fade_alpha) * msw
        )

    # 累加总时间
    self.total_time += dt
    # 返回自身实例，支持链式调用
    return self

def full_family_become_partial(
    self,
    mob1: VMobject,
    mob2: VMobject,
    a: float,
    b: float
) -> Self:
    """
    使一个图形对象(mob1)的所有子对象成为另一个图形对象(mob2)子对象的部分副本
    a和b定义了部分副本的范围（0到1之间的比例）
    """
    # 获取mob1中所有包含点数据的子对象
    family1 = mob1.family_members_with_points()
    # 获取mob2中所有包含点数据的子对象
    family2 = mob2.family_members_with_points()
    # 遍历两个图形对象的子对象并一一对应
    for sm1, sm2 in zip(family1, family2):
        # 使sm1成为sm2从a到b比例范围内的部分副本
        sm1.pointwise_become_partial(sm2, a, b)
    # 返回自身实例，支持链式调用
    return self


class TracedPath(VMobject):
    """
    用于追踪某个点的运动轨迹并生成平滑路径的类
    核心功能：根据传入的点函数，实时记录点的位置，形成动态更新的轨迹图形
    """
    def __init__(
        self,
        traced_point_func: Callable[[], Vect3],  # 无参函数，返回要追踪的点的3D坐标（Vect3）
        time_traced: float = np.inf,            # 轨迹保留时间（默认无限久，即不自动删除历史轨迹）
        time_per_anchor: float = 1.0 / 15,      # 每个锚点的时间间隔（控制轨迹平滑度，默认15帧/秒）
        stroke_width: float | Iterable[float] = 2.0,  # 轨迹描边宽度（可传入单个值或可迭代的渐变值）
        stroke_color: ManimColor = DEFAULT_MOBJECT_COLOR,  # 轨迹描边颜色（默认使用全局默认颜色）
        **kwargs                                # 传递给父类VMobject的关键字参数（如位置、旋转等）
    ):
        # 调用父类VMobject的初始化方法，处理通用图形属性
        super().__init__(** kwargs)
        # 存储追踪点的函数（后续每帧会调用该函数获取最新点坐标）
        self.traced_point_func = traced_point_func
        # 存储轨迹保留时间（超过该时间的历史点会被删除）
        self.time_traced = time_traced
        # 存储每个锚点的时间间隔（用于控制轨迹采样频率）
        self.time_per_anchor = time_per_anchor
        # 记录轨迹累计时间（用于计算采样进度）
        self.time: float = 0
        # 存储所有追踪到的点坐标（历史轨迹数据）
        self.traced_points: list[np.ndarray] = []
        # 添加更新器：每帧调用update_path方法，传入时间增量dt，更新轨迹
        self.add_updater(lambda m, dt: m.update_path(dt))
        # 永久设置轨迹的描边样式（颜色和宽度，后续可通过其他方法修改）
        self.always.set_stroke(stroke_color, stroke_width)

    def update_path(self, dt: float) -> Self:
        """
        轨迹更新方法：每帧调用，根据时间增量dt更新追踪点并刷新轨迹
        dt: 上一帧到当前帧的时间间隔（单位：秒）
        """
        # 若时间增量为0（无时间流逝），直接返回，不更新轨迹
        if dt == 0:
            return self
        # 调用追踪点函数，获取当前帧的点坐标，并创建副本（避免原数据被修改）
        point = self.traced_point_func().copy()
        # 将当前点坐标添加到历史轨迹列表中
        self.traced_points.append(point)

        # 处理轨迹保留时间：若设置了有限保留时间，筛选出最近的相关点
        if self.time_traced < np.inf:
            # 计算需要保留的点的数量（保留时间 / 时间增量，四舍五入）
            n_relevant_points = int(self.time_traced / dt + 0.5)
            # 获取当前已追踪的总点数
            n_tps = len(self.traced_points)
            
            # 若当前点数不足需保留的数量，用当前点填充（避免轨迹过短）
            if n_tps < n_relevant_points:
                points = self.traced_points + [point] * (n_relevant_points - n_tps)
            # 若当前点数超过需保留的数量，只保留最近的n_relevant_points个点
            else:
                points = self.traced_points[n_tps - n_relevant_points:]
            
            # 优化内存：当总点数超过需保留数量的10倍时，直接截取最近的点（避免列表过大）
            if n_tps > 10 * n_relevant_points:
                self.traced_points = self.traced_points[-n_relevant_points:]
        # 若保留时间为无限久，直接使用所有历史点
        else:
            points = self.traced_points

        # 若存在有效点，用这些点生成平滑路径（通过插值使轨迹无棱角）
        if points:
            self.set_points_smoothly(points)

        # 累加累计时间，记录轨迹总时长
        self.time += dt
        # 返回自身实例，支持方法链式调用
        return self


class TracingTail(TracedPath):
    """
    TracedPath的子类，专用于生成"尾随轨迹"效果
    特点：支持基于Mobject对象或自定义函数追踪，轨迹可实现宽度/透明度渐变，模拟"尾巴"消失效果
    """
    def __init__(
        self,
        mobject_or_func: Mobject | Callable[[], np.ndarray],  # 追踪目标：Mobject对象或返回点坐标的函数
        time_traced: float = 1.0,            # 轨迹保留时间（默认1秒，即尾巴长度对应1秒内的运动）
        stroke_width: float | Iterable[float] = (0, 3),  # 描边宽度（默认从0渐变到3，模拟尾巴从细到粗）
        stroke_opacity: float | Iterable[float] = (0, 1),  # 描边透明度（默认从0渐变到1，模拟尾巴从透明到不透明）
        stroke_color: ManimColor = DEFAULT_MOBJECT_COLOR,  # 轨迹颜色（默认使用全局默认颜色）
        **kwargs                                # 传递给父类TracedPath的关键字参数
    ):
        # 判断追踪目标类型：若为Mobject对象，自动追踪其中心点；若为函数，直接使用该函数
        if isinstance(mobject_or_func, Mobject):
            func = mobject_or_func.get_center  # Mobject的get_center()方法返回中心点坐标
        else:
            func = mobject_or_func  # 直接使用自定义点函数
        
        # 调用父类TracedPath的初始化方法，传递必要参数
        super().__init__(
            func,                          # 追踪点函数（中心点函数或自定义函数）
            time_traced=time_traced,       # 轨迹保留时间
            stroke_width=stroke_width,     # 描边宽度（支持渐变）
            stroke_opacity=stroke_opacity, # 描边透明度（支持渐变）
            stroke_color=stroke_color,     # 描边颜色
            **kwargs                       # 其他父类参数
        )
        
        # 添加更新器：每帧重新设置描边样式（确保宽度和透明度的渐变效果实时生效）
        self.add_updater(lambda m: m.set_stroke(width=stroke_width, opacity=stroke_opacity))
