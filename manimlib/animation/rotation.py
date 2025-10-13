# 启用Python 3.7+的注解向前兼容支持，允许在类型注解中使用尚未定义的类
from __future__ import annotations

# 从Manim的动画模块导入基础动画类
from manimlib.animation.animation import Animation

# 从常量模块导入常用的空间坐标和方向常量
# ORIGIN：原点坐标 (0, 0, 0)
# OUT：垂直于屏幕向外的方向向量
from manimlib.constants import ORIGIN, OUT

# 导入数学常量：PI(π)和TAU(τ=2π)
from manimlib.constants import PI, TAU

# 从速率函数工具模块导入常用的速率函数
# linear：线性速率函数（匀速）
# smooth：平滑速率函数（缓进缓出）
from manimlib.utils.rate_functions import linear, smooth

# 导入类型检查相关模块
from typing import TYPE_CHECKING

# 条件导入，仅在类型检查时执行（运行时不执行）
# 用于解决循环导入问题，同时提供完整的类型提示
if TYPE_CHECKING:
    import numpy as np
    from typing import Callable
    from manimlib.mobject.mobject import Mobject


# 定义Rotating类，继承自Animation类，用于实现物体的旋转动画
class Rotating(Animation):
    # 初始化方法，设置旋转动画的各种参数
    def __init__(
        self,
        mobject: Mobject,  # 要进行旋转动画的Mobject对象
        angle: float = TAU,  # 旋转的总角度，默认值为TAU（2π，即360度）
        axis: np.ndarray = OUT,  # 旋转轴，默认值为OUT（指向屏幕外的轴）
        about_point: np.ndarray | None = None,  # 围绕旋转的点，默认为None
        about_edge: np.ndarray | None = None,  # 围绕旋转的边，默认为None
        run_time: float = 5.0,  # 动画运行时间，默认5秒
        rate_func: Callable[[float], float] = linear,  # 速率函数，控制动画速度变化，默认线性
        suspend_mobject_updating: bool = False,  # 是否暂停物体的更新，默认不暂停
        **kwargs  # 其他关键字参数，用于传递给父类
    ):
        self.angle = angle  # 保存旋转角度到实例变量
        self.axis = axis  # 保存旋转轴到实例变量
        self.about_point = about_point  # 保存旋转点到实例变量
        self.about_edge = about_edge  # 保存旋转边到实例变量
        # 调用父类的初始化方法，传递必要的参数
        super().__init__(
            mobject,
            run_time=run_time,
            rate_func=rate_func,
            suspend_mobject_updating=suspend_mobject_updating,** kwargs
        )

    # 插值方法，用于在动画的每个帧更新物体状态
    def interpolate_mobject(self, alpha: float) -> None:
        # 配对当前物体和初始状态物体的所有带有点数据的子物体
        pairs = zip(
            self.mobject.family_members_with_points(),  # 当前物体的所有带点的子物体
            self.starting_mobject.family_members_with_points(),  # 初始状态物体的所有带点的子物体
        )
        # 遍历每对子物体
        for sm1, sm2 in pairs:
            # 遍历所有点数据键（如顶点、控制点等）
            for key in sm1.pointlike_data_keys:
                # 将初始状态的点数据复制到当前物体，重置位置
                sm1.data[key][:] = sm2.data[key]
        # 对物体进行旋转
        self.mobject.rotate(
            # 计算当前帧的旋转角度：速率函数值 × 总角度
            self.rate_func(self.time_spanned_alpha(alpha)) * self.angle,
            axis=self.axis,  # 使用指定的旋转轴
            about_point=self.about_point,  # 使用指定的旋转点
            about_edge=self.about_edge,  # 使用指定的旋转边
        )


# 定义Rotate类，继承自Rotating类，用于实现更简洁的旋转动画
class Rotate(Rotating):
    # 初始化方法，设置旋转动画的参数（基于父类Rotating进行了默认值调整）
    def __init__(
        self,
        mobject: Mobject,  # 要旋转的Mobject对象
        angle: float = PI,  # 旋转总角度，默认值为PI（π，即180度，区别于父类的TAU）
        axis: np.ndarray = OUT,  # 旋转轴，默认值为OUT（与父类一致）
        run_time: float = 1,  # 动画运行时间，默认1秒（比父类的5秒更短）
        rate_func: Callable[[float], float] = smooth,  # 速率函数，默认平滑曲线（父类为线性）
        about_edge: np.ndarray = ORIGIN,  # 围绕旋转的边，默认值为原点（父类默认为None）
        **kwargs  # 其他关键字参数，用于传递给父类
    ):
        # 调用父类Rotating的初始化方法，传递参数
        super().__init__(
            mobject, angle, axis,  # 传递前三个位置参数（mobject, angle, axis）
            run_time=run_time,  # 传递运行时间参数
            rate_func=rate_func,  # 传递速率函数参数
            about_edge=about_edge,  # 传递旋转边参数
            **kwargs  # 传递其他关键字参数
        )
