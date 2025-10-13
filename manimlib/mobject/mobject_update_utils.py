# 从未来版本导入注解功能，支持在函数参数和返回值中使用类名作为类型提示
from __future__ import annotations

# 导入inspect模块，用于检查对象的类型和属性
import inspect

# 从manimlib常量中导入角度单位和方向向量
from manimlib.constants import DEG
from manimlib.constants import RIGHT
# 导入Mobject基类，所有可渲染对象的基类
from manimlib.mobject.mobject import Mobject
# 导入剪辑函数，用于将值限制在指定范围内
from manimlib.utils.simple_functions import clip

# 导入类型检查相关模块
from typing import TYPE_CHECKING

# 条件导入类型提示，仅在类型检查时生效，避免运行时依赖
if TYPE_CHECKING:
    from typing import Callable

    import numpy as np

    from manimlib.animation.animation import Animation


def assert_is_mobject_method(method):
    """断言给定的方法是Mobject实例的方法"""
    # 检查是否为方法
    assert inspect.ismethod(method)
    # 获取方法所属的实例对象
    mobject = method.__self__
    # 检查实例是否为Mobject类型
    assert isinstance(mobject, Mobject)


def always(method, *args, **kwargs):
    """
    为Mobject添加一个持续执行指定方法的更新器
    每次场景更新时都会调用该方法
    """
    # 验证输入是Mobject的方法
    assert_is_mobject_method(method)
    # 获取方法所属的Mobject实例
    mobject = method.__self__
    # 获取方法本身（剥离实例绑定）
    func = method.__func__
    # 为Mobject添加更新器，每次更新时调用该方法
    mobject.add_updater(lambda m: func(m, *args, **kwargs))
    # 返回Mobject实例，支持链式调用
    return mobject


def f_always(method, *arg_generators, **kwargs):
    """
    功能更强大的always版本，参数由生成器函数提供
    每次更新时会先调用生成器函数获取参数，再调用方法
    """
    # 验证输入是Mobject的方法
    assert_is_mobject_method(method)
    # 获取方法所属的Mobject实例
    mobject = method.__self__
    # 获取方法本身（剥离实例绑定）
    func = method.__func__

    # 定义更新器函数
    def updater(mob):
        # 调用所有参数生成器，获取最新参数
        args = [
            arg_generator()
            for arg_generator in arg_generators
        ]
        # 使用生成的参数调用方法
        func(mob, *args, **kwargs)

    # 添加更新器到Mobject
    mobject.add_updater(updater)
    # 返回Mobject实例，支持链式调用
    return mobject


def always_redraw(func: Callable[..., Mobject], *args, **kwargs) -> Mobject:
    """
    创建一个会持续重绘的Mobject
    每次更新时都会调用func重新生成对象，并替换当前对象
    """
    # 初始调用函数创建Mobject
    mob = func(*args, **kwargs)
    # 添加更新器：每次更新时用新生成的对象替换当前对象
    mob.add_updater(lambda m: mob.become(func(*args, **kwargs)))
    # 返回创建的Mobject
    return mob


def always_shift(
    mobject: Mobject,
    direction: np.ndarray = RIGHT,
    rate: float = 0.1
) -> Mobject:
    """
    使Mobject持续向指定方向移动
    direction: 移动方向向量，默认为RIGHT（向右）
    rate: 移动速率，默认为0.1单位/秒
    """
    # 添加更新器：根据时间增量(dt)计算移动距离并应用
    mobject.add_updater(
        lambda m, dt: m.shift(dt * rate * direction)
    )
    # 返回Mobject实例，支持链式调用
    return mobject


def always_rotate(
    mobject: Mobject,
    rate: float = 20 * DEG,
    **kwargs
) -> Mobject:
    """
    使Mobject持续旋转
    rate: 旋转速率，默认为20度/秒
    **kwargs: 传递给rotate方法的其他参数（如旋转轴等）
    """
    # 添加更新器：根据时间增量(dt)计算旋转角度并应用
    mobject.add_updater(
        lambda m, dt: m.rotate(dt * rate, **kwargs)
    )
    # 返回Mobject实例，支持链式调用
    return mobject


def turn_animation_into_updater(
    animation: Animation,
    cycle: bool = False,** kwargs
) -> Mobject:
    """
    将动画转换为Mobject的更新器，使动画可以作为持续更新的一部分
    如果cycle为True，动画会循环播放；否则播放一次后停止
    """
    # 获取动画作用的Mobject
    mobject = animation.mobject
    # 更新动画的速率信息
    animation.update_rate_info(**kwargs)
    # 不暂停Mobject的更新
    animation.suspend_mobject_updating = False
    # 开始动画（执行动画的初始化操作）
    animation.begin()
    # 初始化动画总时间
    animation.total_time = 0

    # 定义更新器函数
    def update(m, dt):
        # 获取动画的运行时间
        run_time = animation.get_run_time()
        # 计算时间比率（已运行时间/总运行时间）
        time_ratio = animation.total_time / run_time
        
        if cycle:
            # 如果循环播放，使用取模运算使alpha在0-1之间循环
            alpha = time_ratio % 1
        else:
            # 如果不循环，将alpha限制在0-1之间
            alpha = clip(time_ratio, 0, 1)
            # 当动画完成时（alpha >= 1）
            if alpha >= 1:
                # 执行动画的结束操作
                animation.finish()
                # 移除更新器，停止动画
                m.remove_updater(update)
                return
        
        # 根据alpha值插值计算动画状态
        animation.interpolate(alpha)
        # 更新动画中的所有Mobject
        animation.update_mobjects(dt)
        # 累加动画总时间
        animation.total_time += dt

    # 为Mobject添加更新器
    mobject.add_updater(update)
    # 返回Mobject实例
    return mobject


def cycle_animation(animation: Animation, **kwargs) -> Mobject:
    """
    创建一个循环播放的动画更新器
    是turn_animation_into_updater的便捷包装，固定cycle=True
    """
    return turn_animation_into_updater(
        animation, cycle=True,** kwargs
    )