# 从__future__导入annotations，支持在类型注解中使用尚未定义的类
from __future__ import annotations

# 从manimlib.animation.animation模块导入Animation类，用于动画基础类
from manimlib.animation.animation import Animation
# 从manimlib.utils.rate_functions模块导入linear函数，用于线性速率控制
from manimlib.utils.rate_functions import linear

# 从typing模块导入TYPE_CHECKING，用于条件导入类型提示
from typing import TYPE_CHECKING

# 如果是类型检查阶段（非运行时），则导入所需的类型提示
if TYPE_CHECKING:
    # 从typing模块导入Callable（可调用对象类型）和Sequence（序列类型）
    from typing import Callable, Sequence

    # 导入numpy并别名np，用于numpy相关类型注解
    import numpy as np

    # 从manimlib.mobject.mobject导入Mobject类，用于物体类型注解
    from manimlib.mobject.mobject import Mobject
    # 从manimlib.mobject.types.vectorized_mobject导入VMobject类，用于矢量物体类型注解
    from manimlib.mobject.types.vectorized_mobject import VMobject



"""
实现了基于同伦函数的通用动画变换
核心原理：
1.通过一个自定义的同伦函数控制物体的连续变形
2.同伦函数接收空间坐标 (x,y,z) 和时间参数 t (0 到 1)，返回变换后的坐标
3.动画过程中，随着时间进度 alpha 从 0 到 1 变化，不断应用对应时刻的变换函数
4.实现物体从初始状态到目标状态的平滑过渡
"""
# 定义Homotopy类，继承自Animation，用于实现同伦变换动画
# 同伦变换是一种连续变形，通过函数将物体从初始状态平滑过渡到目标状态
class Homotopy(Animation):
    # 应用函数的配置字典，可用于传递额外的配置参数
    apply_function_config: dict = dict()

    def __init__(
        self,
        homotopy: Callable[[float, float, float, float], Sequence[float]],
        # 同伦函数，接收(x, y, z, t)四个参数，返回变换后的坐标(x', y', z')
        # 其中t是时间参数（0到1），表示动画进度
        mobject: Mobject,  # 要应用同伦变换的物体
        run_time: float = 3.0,  # 动画运行时间，默认3秒
        **kwargs  # 其他关键字参数，传递给父类Animation
    ):
        """
        同伦函数是一个从(x, y, z, t)到(x', y', z')的映射
        其中t∈[0,1]表示动画进度，(x,y,z)是物体上点的原始坐标
        """
        self.homotopy = homotopy  # 保存同伦函数到实例变量
        # 调用父类Animation的初始化方法，传递参数
        super().__init__(mobject, run_time=run_time,** kwargs)

    # 生成特定时间t的变换函数
    def function_at_time_t(self, t: float) -> Callable[[np.ndarray], Sequence[float]]:
        # 定义一个函数，接收点坐标p（包含x,y,z），返回变换后的坐标
        def result(p):
            # 调用同伦函数，将点p的坐标(x,y,z)和时间t传入
            return self.homotopy(*p, t)
        return result  # 返回配置好的变换函数

    # 插值方法，更新子物体的状态
    def interpolate_submobject(
        self,
        submob: Mobject,  # 要更新的子物体
        start: Mobject,   # 起始状态的子物体
        alpha: float      # 动画进度（0到1）
    ) -> None:
        # 使子物体与起始状态的点结构匹配
        submob.match_points(start)
        # 应用当前时间的变换函数到子物体
        submob.apply_function(
            self.function_at_time_t(alpha),  # 获取对应进度的变换函数
            **self.apply_function_config    # 传递配置参数
        )



"""
Homotopy子类
作用：
1.继承了Homotopy的所有同伦变换功能，保持了通过函数实现连续变形的特性
2.重写了apply_function_config类属性，默认添加make_smooth=True参数
3.确保在对矢量物体 (VMobject) 应用变换时，能自动进行平滑处理，避免出现锯齿或变形 artifacts
"""
# 定义SmoothedVectorizedHomotopy类，继承自Homotopy，用于实现平滑的矢量同伦变换动画
class SmoothedVectorizedHomotopy(Homotopy):
    # 应用函数的配置字典，设置make_smooth=True以确保变换过程平滑
    # 这对于矢量物体(VMobject)尤为重要，可保持边缘和曲线的平滑性
    apply_function_config: dict = dict(make_smooth=True)



"""
核心功能是将复平面上的变换扩展到三维空间
特点：
1.接收一个复变函数（处理复数输入输出），而非直接处理三维坐标
2.内部将复变函数转换为适用于 3D 空间的同伦函数：
 把 x,y 坐标视为复数的实部和虚部
 变换仅作用于 x,y 平面（z 坐标保持不变）
 将变换后的复数实部和虚部分别作为新的 x,y 坐标
3.继承了Homotopy类的所有动画特性，实现复平面变换的平滑动画效果
"""
# 定义ComplexHomotopy类，继承自Homotopy，用于处理复平面上的同伦变换动画
class ComplexHomotopy(Homotopy):
    def __init__(
        self,
        complex_homotopy: Callable[[complex, float], complex],
        # 复变同伦函数，接收一个复数z和时间t，返回变换后的复数w
        mobject: Mobject,  # 要应用复变同伦变换的物体
        **kwargs  # 其他关键字参数，传递给父类Homotopy
    ):
        """
        给定一个从(z, t)到w的函数，其中z和w是复数，t是时间，
        这个类用于实现随时间变化的动画效果
        """
        # 定义三维空间中的同伦函数，将复变函数转换为适用于3D坐标的变换
        def homotopy(x, y, z, t):
            # 将x,y坐标转换为复数z = x + yi
            # 调用复变同伦函数得到变换后的复数c
            c = complex_homotopy(complex(x, y), t)
            # 返回变换后的三维坐标：实部作为x，虚部作为y，保持z坐标不变
            return (c.real, c.imag, z)

        # 调用父类Homotopy的初始化方法，传入转换后的同伦函数和其他参数
        super().__init__(homotopy, mobject,** kwargs)



"""
实现了基于向量场的动画效果
工作原理：
1.通过一个向量场函数function定义空间中每个点的运动方向和速度
2.动画过程中，物体上的每个点会根据向量场和时间流逝不断更新位置
3.采用增量更新方式：每帧只计算与上一帧的位置变化，避免累积误差
"""
# 定义PhaseFlow类，继承自Animation，用于实现基于向量场的相位流动画
class PhaseFlow(Animation):
    def __init__(
        self,
        function: Callable[[np.ndarray], np.ndarray],
        # 向量场函数，接收点坐标p（np.ndarray），返回该点的向量（运动方向和速度）
        mobject: Mobject,  # 要应用相位流动画的物体
        virtual_time: float | None = None,  # 虚拟时间，控制动画的总"时间量"，默认为None（使用运行时间）
        suspend_mobject_updating: bool = False,  # 是否暂停物体的更新，默认不暂停
        rate_func: Callable[[float], float] = linear,  # 速率函数，控制动画速度变化，默认线性
        run_time: float = 3.0,  # 动画实际运行时间，默认3秒
        **kwargs  # 其他关键字参数，传递给父类
    ):
        self.function = function  # 保存向量场函数到实例变量
        # 设置虚拟时间：若未指定则使用实际运行时间
        self.virtual_time = virtual_time or run_time
        # 调用父类Animation的初始化方法，传递参数
        super().__init__(
            mobject,
            rate_func=rate_func,
            run_time=run_time,
            suspend_mobject_updating=suspend_mobject_updating,** kwargs
        )

    # 插值方法，更新物体状态以实现相位流效果
    def interpolate_mobject(self, alpha: float) -> None:
        # 检查是否存在上一帧的alpha值（即是否不是第一帧）
        if hasattr(self, "last_alpha"):
            # 计算当前帧与上一帧之间的虚拟时间差
            # 虚拟时间 × (当前进度 - 上一帧进度)
            dt = self.virtual_time * (alpha - self.last_alpha)
            # 对物体应用位移：每个点沿向量场方向移动，位移量 = 时间差 × 向量场强度
            self.mobject.apply_function(
                lambda p: p + dt * self.function(p)
            )
        # 记录当前帧的alpha值，供下一帧计算时间差
        self.last_alpha = alpha



"""
核心功能是使物体沿着预先定义的路径移动
工作原理：
1.接收一个path参数（矢量物体）作为移动轨迹
2.动画过程中，根据当前进度alpha（0 到 1）计算路径上的对应位置
3.通过rate_func可以控制移动速率（如匀速、加速、减速等）
4.每帧将物体移动到路径上的对应点，形成沿路径运动的动画效果
"""
# 定义MoveAlongPath类，继承自Animation，用于实现物体沿指定路径移动的动画
class MoveAlongPath(Animation):
    def __init__(
        self,
        mobject: Mobject,  # 要沿路径移动的物体
        path: VMobject,    # 移动所沿的路径（矢量物体，如曲线、线段等）
        suspend_mobject_updating: bool = False,  # 是否暂停物体的更新，默认不暂停
        **kwargs  # 其他关键字参数，传递给父类Animation
    ):
        self.path = path  # 保存路径到实例变量
        # 调用父类Animation的初始化方法，传递参数
        super().__init__(mobject, suspend_mobject_updating=suspend_mobject_updating, **kwargs)

    # 插值方法，更新物体位置以实现沿路径移动的效果
    def interpolate_mobject(self, alpha: float) -> None:
        # 根据动画进度获取路径上的对应点
        # rate_func(alpha)将进度转换为符合速率函数的比例（如非线性速率）
        # quick_point_from_proportion获取路径上该比例位置的点坐标
        point = self.path.quick_point_from_proportion(self.rate_func(alpha))
        # 将物体移动到路径上的该点
        self.mobject.move_to(point)
