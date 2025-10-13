# 从__future__模块导入annotations，支持类型注释的延迟评估
# 允许在类型提示中使用尚未定义的类或函数
from __future__ import annotations

# 从manimlib.animation.animation模块导入Animation基类
# Animation是Manim中所有动画类的父类，提供动画的基本框架
from manimlib.animation.animation import Animation

# 从typing模块导入TYPE_CHECKING常量
# 用于在类型检查阶段执行特定代码，运行时不执行
from typing import TYPE_CHECKING

# 条件判断：仅在类型检查时执行以下代码块
# 运行时TYPE_CHECKING的值为False，因此不会执行
if TYPE_CHECKING:
    # 从typing模块导入Callable类型，用于标注可调用对象（如函数）
    from typing import Callable

    # 从manimlib.mobject.mobject模块导入Mobject类
    # Mobject是Manim中所有可移动对象的基类
    from manimlib.mobject.mobject import Mobject


class UpdateFromFunc(Animation):
    """
    从Animation基类继承的更新动画类
    该类使用指定的更新函数来更新mobject（可移动对象）的状态
    主要用于当一个对象的状态依赖于另一个同时被动画的对象时
    
    update_function的形式为func(mobject)，接收一个mobject参数
    """
    def __init__(
        self,
        mobject: Mobject,  # 要被更新的可移动对象
        update_function: Callable[[Mobject], Mobject | None],  # 更新函数：接收Mobject并返回Mobject或None
        suspend_mobject_updating: bool = False,  # 是否暂停对象自身的更新器
        **kwargs  # 传递给父类Animation的其他参数
    ):
        # 保存更新函数为实例属性
        self.update_function = update_function
        # 调用父类Animation的初始化方法
        super().__init__(
            mobject,
            suspend_mobject_updating=suspend_mobject_updating,** kwargs
        )

    def interpolate_mobject(self, alpha: float) -> None:
        """
        重写父类的插值方法，定义动画的具体更新逻辑
        alpha表示动画进度，范围从0到1
        
        在这个实现中，忽略alpha值，直接调用更新函数来更新对象
        """
        # 调用更新函数，传入当前的mobject进行更新
        self.update_function(self.mobject)


class UpdateFromAlphaFunc(Animation):
    """
    继承自Animation基类，用于通过依赖动画进度(alpha)的函数更新mobject
    
    与UpdateFromFunc的区别在于，更新函数会接收动画进度作为第二个参数，
    便于根据动画的不同阶段（从0到1）来精确控制对象的状态变化
    """
    def __init__(
        self,
        mobject: Mobject,  # 要被更新的可移动对象
        # 更新函数：接收Mobject实例和alpha值(0-1)，返回Mobject或None
        update_function: Callable[[Mobject, float], Mobject | None],
        suspend_mobject_updating: bool = False,  # 是否暂停对象自身的更新器
        **kwargs  # 传递给父类Animation的其他参数（如run_time等）
    ):
        # 保存更新函数为实例属性
        self.update_function = update_function
        # 调用父类构造方法初始化动画
        super().__init__(
            mobject,
            suspend_mobject_updating=suspend_mobject_updating,** kwargs
        )

    def interpolate_mobject(self, alpha: float) -> None:
        """
        重写父类的插值方法，实现动画帧更新逻辑
        
        参数alpha：动画进度值，范围从0（开始状态）到1（结束状态）
        """
        # 调用更新函数，将当前mobject和进度alpha传入
        # 由外部定义的函数决定如何根据进度更新对象状态
        self.update_function(self.mobject, alpha)


class MaintainPositionRelativeTo(Animation):
    """
    继承自Animation基类，用于维持一个对象相对于另一个被跟踪对象的位置关系
    确保当被跟踪对象移动时，当前对象能保持初始的相对位置不变
    """
    def __init__(
        self,
        mobject: Mobject,  # 需要维持相对位置的对象
        tracked_mobject: Mobject,  # 被跟踪的参考对象
        **kwargs  # 传递给父类Animation的其他参数
    ):
        # 保存被跟踪对象为实例属性
        self.tracked_mobject = tracked_mobject
        # 计算并保存两个对象初始的中心位置差值（当前对象 - 被跟踪对象）
        # 这个差值定义了它们之间的相对位置关系
        self.diff = mobject.get_center() - tracked_mobject.get_center()
        # 调用父类构造方法初始化动画
        super().__init__(mobject,** kwargs)

    def interpolate_mobject(self, alpha: float) -> None:
        """
        重写父类的插值方法，实现位置维持逻辑
        
        参数alpha：动画进度值（此处未直接使用，因为是持续跟踪）
        """
        # 获取被跟踪对象当前的中心位置
        target = self.tracked_mobject.get_center()
        # 获取当前对象当前的中心位置
        location = self.mobject.get_center()
        # 计算需要移动的距离：(被跟踪对象新位置 + 初始差值) - 当前对象位置
        # 确保当前对象始终保持与被跟踪对象的初始相对位置
        self.mobject.shift(target - location + self.diff)
