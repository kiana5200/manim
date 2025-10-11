# 从__future__导入annotations，支持在类型提示中使用尚未定义的类
from __future__ import annotations

# 从isosurfaces导入plot_isoline函数用于绘制等值线
from isosurfaces import plot_isoline
# 导入numpy用于数值计算
import numpy as np

# 从manimlib.constants导入帧的X/Y半径常量
from manimlib.constants import FRAME_X_RADIUS, FRAME_Y_RADIUS
# 从manimlib.constants导入颜色常量YELLOW
from manimlib.constants import YELLOW
# 从manimlib.mobject.types.vectorized_mobject导入VMobject基类
from manimlib.mobject.types.vectorized_mobject import VMobject

# 导入类型检查相关模块
from typing import TYPE_CHECKING

# 如果是类型检查阶段
if TYPE_CHECKING:
    # 导入所需的类型提示
    from typing import Callable, Sequence, Tuple
    from manimlib.typing import ManimColor, Vect3


class ParametricCurve(VMobject):
    """参数曲线类，继承自VMobject"""
    
    def __init__(
        self,
        t_func: Callable[[float], Sequence[float] | Vect3],  # 参数函数，输入参数t返回三维坐标
        t_range: Tuple[float, float, float] = (0, 1, 0.1),  # 参数t的范围（起始、结束、步长）
        epsilon: float = 1e-8,  # 用于处理不连续点的小量
        discontinuities: Sequence[float] = [],  # 不连续点的参数t值列表
        use_smoothing: bool = True,  # 是否使用平滑处理
        **kwargs  # 其他传递给父类的关键字参数
    ):
        # 存储参数曲线的相关属性
        self.t_func = t_func
        self.t_range = t_range
        self.epsilon = epsilon
        self.discontinuities = discontinuities
        self.use_smoothing = use_smoothing
        # 调用父类的初始化方法
        super().__init__(**kwargs)

    def get_point_from_function(self, t: float) -> Vect3:
        """根据参数t获取曲线上的点"""
        return np.array(self.t_func(t))

    def init_points(self):
        """初始化曲线的点"""
        t_min, t_max, step = self.t_range

        # 处理不连续点
        jumps = np.array(self.discontinuities)
        jumps = jumps[(jumps > t_min) & (jumps < t_max)]
        boundary_times = [t_min, t_max, *(jumps - self.epsilon), *(jumps + self.epsilon)]
        boundary_times.sort()
        for t1, t2 in zip(boundary_times[0::2], boundary_times[1::2]):
            t_range = [*np.arange(t1, t2, step), t2]
            points = np.array([self.t_func(t) for t in t_range])
            self.start_new_path(points[0])
            self.add_points_as_corners(points[1:])
        # 如果需要平滑处理
        if self.use_smoothing:
            self.make_smooth(approx=True)
        # 如果没有点，设置默认点
        if not self.has_points():
            self.set_points(np.array([self.t_func(t_min)]))
        return self

    def get_t_func(self):
        """获取参数函数"""
        return self.t_func

    def get_function(self):
        """获取底层函数"""
        if hasattr(self, "underlying_function"):
            return self.underlying_function
        if hasattr(self, "function"):
            return self.function

    def get_x_range(self):
        """获取x范围"""
        if hasattr(self, "x_range"):
            return self.x_range


class FunctionGraph(ParametricCurve):
    """函数图像类，继承自ParametricCurve"""
    
    def __init__(
        self,
        function: Callable[[float], float],  # 函数，输入x返回y
        x_range: Tuple[float, float, float] = (-8, 8, 0.25),  # x的范围（起始、结束、步长）
        color: ManimColor = YELLOW,  # 曲线颜色，默认为黄色
        **kwargs  # 其他传递给父类的关键字参数
    ):
        self.function = function
        self.x_range = x_range

        # 定义参数函数（将x作为参数t）
        def parametric_function(t):
            return [t, function(t), 0]

        # 调用父类的初始化方法
        super().__init__(parametric_function, self.x_range, **kwargs)


class ImplicitFunction(VMobject):
    """隐函数类，继承自VMobject"""
    
    def __init__(
        self,
        func: Callable[[float, float], float],  # 隐函数，输入x和y返回函数值
        x_range: Tuple[float, float] = (-FRAME_X_RADIUS, FRAME_X_RADIUS),  # x的范围
        y_range: Tuple[float, float] = (-FRAME_Y_RADIUS, FRAME_Y_RADIUS),  # y的范围
        min_depth: int = 5,  # 最小递归深度
        max_quads: int = 1500,  # 最大四边形数量
        use_smoothing: bool = False,  # 是否使用平滑处理
        joint_type: str = 'no_joint',  # 连接类型
        **kwargs  # 其他传递给父类的关键字参数
    ):
        # 调用父类的初始化方法
        super().__init__(joint_type=joint_type, **kwargs)

        # 定义绘图范围的最小和最大点
        p_min, p_max = (
            np.array([x_range[0], y_range[0]]),
            np.array([x_range[1], y_range[1]]),
        )
        # 绘制等值线（函数值为0的曲线）
        curves = plot_isoline(
            fn=lambda u: func(u[0], u[1]),
            pmin=p_min,
            pmax=p_max,
            min_depth=min_depth,
            max_quads=max_quads,
        )  # 返回等值线的点列表
        # 为每个点添加z坐标（设为0）
        curves = [
            np.pad(curve, [(0, 0), (0, 1)])
            for curve in curves
            if curve != []
        ]
        # 将每个等值线添加到对象中
        for curve in curves:
            self.start_new_path(curve[0])
            self.add_points_as_corners(curve[1:])
        # 如果需要平滑处理
        if use_smoothing:
            self.make_smooth()
