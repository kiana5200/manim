from __future__ import annotations

from isosurfaces import plot_isoline
import numpy as np

from manimlib.constants import FRAME_X_RADIUS, FRAME_Y_RADIUS
from manimlib.constants import YELLOW
from manimlib.mobject.types.vectorized_mobject import VMobject

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable, Sequence, Tuple
    from manimlib.typing import ManimColor, Vect3


class ParametricCurve(VMobject):
    def __init__(
        self,
        t_func: Callable[[float], Sequence[float] | Vect3],
        t_range: Tuple[float, float, float] = (0, 1, 0.1),
        epsilon: float = 1e-8,
        # 待办：自动检测不连续点
        discontinuities: Sequence[float] = [],
        use_smoothing: bool = True,
        **kwargs
    ):
        # 存储参数方程函数：输入t返回坐标点
        self.t_func = t_func
        # 存储参数t的范围：(最小值, 最大值, 步长)
        self.t_range = t_range
        # 用于处理不连续点的微小偏移量
        self.epsilon = epsilon
        # 手动指定的不连续点列表
        self.discontinuities = discontinuities
        # 是否启用平滑处理
        self.use_smoothing = use_smoothing
        # 调用父类初始化方法
        super().__init__(** kwargs)

    def get_point_from_function(self, t: float) -> Vect3:
        # 将参数t对应的函数值转换为三维点坐标
        return np.array(self.t_func(t))

    def init_points(self):
        # 解析t的范围参数
        t_min, t_max, step = self.t_range

        # 处理不连续点：在跳跃点附近添加边界，避免跨不连续点绘制
        jumps = np.array(self.discontinuities)
        # 筛选出在[t_min, t_max]范围内的不连续点
        jumps = jumps[(jumps > t_min) & (jumps < t_max)]
        # 为每个不连续点创建两个边界点（t±epsilon），避免跨越绘制
        boundary_times = [t_min, t_max, *(jumps - self.epsilon), *(jumps + self.epsilon)]
        # 排序边界点
        boundary_times.sort()

        # 遍历每对边界点，在区间内绘制曲线段
        for t1, t2 in zip(boundary_times[0::2], boundary_times[1::2]):
            # 生成当前区间内的t值序列（包含端点）
            t_range = [*np.arange(t1, t2, step), t2]
            # 计算每个t对应的点坐标
            points = np.array([self.t_func(t) for t in t_range])
            # 开始新的路径（处理不连续点）
            self.start_new_path(points[0])
            # 添加后续点作为角点
            self.add_points_as_corners(points[1:])

        # 如果启用平滑处理，对曲线进行平滑处理
        if self.use_smoothing:
            self.make_smooth(approx=True)

        # 确保至少有一个点（避免空曲线）
        if not self.has_points():
            self.set_points(np.array([self.t_func(t_min)]))

        return self

    def get_t_func(self):
        # 返回参数方程函数
        return self.t_func

    def get_function(self):
        # 尝试返回底层函数（兼容可能的属性名）
        if hasattr(self, "underlying_function"):
            return self.underlying_function
        if hasattr(self, "function"):
            return self.function

    def get_x_range(self):
        # 如果存在x_range属性，返回它
        if hasattr(self, "x_range"):
            return self.x_range


class FunctionGraph(ParametricCurve):
    def __init__(
        self,
        function: Callable[[float], float],
        x_range: Tuple[float, float, float] = (-8, 8, 0.25),
        color: ManimColor = YELLOW,** kwargs
    ):
        # 存储函数本身
        self.function = function
        # 存储x轴范围：(最小值, 最大值, 步长)
        self.x_range = x_range

        # 定义参数化函数：将x作为参数t，返回对应的(x, f(x), 0)三维坐标
        # （z坐标为0，因为是2D函数图像）
        def parametric_function(t):
            return [t, function(t), 0]

        # 调用父类ParametricCurve的初始化方法
        # 使用x_range作为参数t的范围，传入参数化函数和其他参数
        super().__init__(parametric_function, self.x_range, **kwargs)

class ImplicitFunction(VMobject):
    def __init__(
        self,
        func: Callable[[float, float], float],
        x_range: Tuple[float, float] = (-FRAME_X_RADIUS, FRAME_X_RADIUS),
        y_range: Tuple[float, float] = (-FRAME_Y_RADIUS, FRAME_Y_RADIUS),
        min_depth: int = 5,
        max_quads: int = 1500,
        use_smoothing: bool = False,
        joint_type: str = 'no_joint',** kwargs
    ):
        # 调用父类VMobject初始化，设置线段连接方式
        super().__init__(joint_type=joint_type, **kwargs)

        # 定义平面范围的最小/最大点（2D坐标，x和y分别对应x_range、y_range）
        p_min, p_max = (
            np.array([x_range[0], y_range[0]]),
            np.array([x_range[1], y_range[1]]),
        )
        # 调用plot_isoline函数绘制隐函数曲线：找到满足func(x,y)=0的点集
        # 返回值是2D点列表组成的列表（每条子列表对应一段连续曲线）
        curves = plot_isoline(
            fn=lambda u: func(u[0], u[1]),  # 适配plot_isoline的输入格式（接收2D向量u）
            pmin=p_min,                     # 范围最小值
            pmax=p_max,                     # 范围最大值
            min_depth=min_depth,            # 细分网格的最小深度（控制精度）
            max_quads=max_quads,            # 最大四边形数量（控制计算量）
        )
        # 为每条2D曲线添加z坐标（设为0，转为3D点），并过滤空曲线
        curves = [
            np.pad(curve, [(0, 0), (0, 1)])  # 维度扩展：(n,2)→(n,3)，z列补0
            for curve in curves
            if curve != []                   # 跳过空的点列表
        ]
        # 把每条曲线的点添加到VMobject中，形成完整图形
        for curve in curves:
            self.start_new_path(curve[0])    # 从曲线第一个点开始新路径
            self.add_points_as_corners(curve[1:])  # 添加后续点作为角点
        # 若启用平滑，对曲线进行平滑处理
        if use_smoothing:
            self.make_smooth()