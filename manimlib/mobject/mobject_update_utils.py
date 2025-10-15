from __future__ import annotations

import inspect

from manimlib.constants import DEG
from manimlib.constants import RIGHT
from manimlib.mobject.mobject import Mobject
from manimlib.utils.simple_functions import clip

from typing import TYPE_CHECKING

# 类型检查专用导入，仅在类型检查时生效，不影响运行时
if TYPE_CHECKING:
    from typing import Callable

    import numpy as np

    from manimlib.animation.animation import Animation


def assert_is_mobject_method(method):
    """
    验证传入的对象是否为Mobject实例的方法。
    
    参数:
        method: 需要验证的对象
        
    断言:
        - 该对象必须是一个方法（而非函数或其他类型）
        - 方法所属的实例必须是Mobject类（或其子类）的实例
    """
    # 断言传入的是方法（绑定到实例的函数）
    assert inspect.ismethod(method)
    # 获取方法所属的实例对象
    mobject = method.__self__
    # 断言该实例是Mobject类型
    assert isinstance(mobject, Mobject)


def always(method, *args, **kwargs):
    """
    为Mobject实例的方法添加一个持续更新器，使其在每一帧都被调用。
    
    参数:
        method: Mobject实例的方法（将被持续调用）
        *args: 传递给方法的位置参数
        **kwargs: 传递给方法的关键字参数
        
    返回:
        Mobject: 方法所属的Mobject实例（便于链式调用）
    """
    # 验证传入的是Mobject实例的方法
    assert_is_mobject_method(method)
    # 获取方法所属的Mobject实例
    mobject = method.__self__
    # 获取方法对应的函数对象（剥离实例绑定）
    func = method.__func__
    # 为Mobject添加更新器：在每一帧调用该方法，并传递参数
    # lambda中的m是Mobject实例自身，确保方法调用时的self正确
    mobject.add_updater(lambda m: func(m, *args, **kwargs))
    # 返回Mobject实例，支持链式操作
    return mobject


def f_always(method, *arg_generators, **kwargs):
    """
    功能化版本的always函数，接收生成参数的函数而非直接接收参数。
    """
    # 验证传入的是Mobject实例的方法
    assert_is_mobject_method(method)
    # 获取方法所属的Mobject实例
    mobject = method.__self__
    # 获取方法对应的函数对象
    func = method.__func__

    # 定义更新器：调用参数生成器获取参数后执行方法
    def updater(mob):
        args = [
            arg_generator()  # 调用每个参数生成器获取实际参数
            for arg_generator in arg_generators
        ]
        func(mob, *args, **kwargs)  # 执行方法

    # 为Mobject添加更新器
    mobject.add_updater(updater)
    return mobject


def always_redraw(func: Callable[..., Mobject], *args, **kwargs) -> Mobject:
    # 首次调用函数生成Mobject
    mob = func(*args, **kwargs)
    # 添加更新器：每帧用新生成的Mobject替换当前对象
    mob.add_updater(lambda m: mob.become(func(*args, **kwargs)))
    return mob


def always_shift(
    mobject: Mobject,
    direction: np.ndarray = RIGHT,
    rate: float = 0.1
) -> Mobject:
    # 为Mobject添加更新器：每帧按指定方向和速率移动
    mobject.add_updater(
        lambda m, dt: m.shift(dt * rate * direction)  # dt为帧间隔时间
    )
    return mobject


def always_rotate(
    mobject: Mobject,
    rate: float = 20 * DEG,
    **kwargs
) -> Mobject:
    mobject.add_updater(
        lambda m, dt: m.rotate(dt * rate, **kwargs)
    )
    return mobject


def turn_animation_into_updater(
    animation: Animation,
    cycle: bool = False,** kwargs
) -> Mobject:
    """
    将动画转换为更新器，使动画的插值和更新函数通过mobject的更新器生效。
    如果cycle为True，动画会循环播放；否则，动画完成后会移除更新器。
    """
    # 获取动画作用的mobject
    mobject = animation.mobject
    # 更新动画的速率信息
    animation.update_rate_info(**kwargs)
    # 不暂停mobject的更新
    animation.suspend_mobject_updating = False
    # 开始动画（执行初始化操作）
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
            # 循环模式：取时间比率的余数，使动画循环
            alpha = time_ratio % 1
        else:
            # 非循环模式：限制alpha在0到1之间
            alpha = clip(time_ratio, 0, 1)
            # 如果动画完成（alpha >= 1），执行收尾并移除更新器
            if alpha >= 1:
                animation.finish()
                m.remove_updater(update)
                return
        # 根据alpha值插值计算动画状态
        animation.interpolate(alpha)
        # 更新动画中的mobjects
        animation.update_mobjects(dt)
        # 累加总时间
        animation.total_time += dt

    # 为mobject添加更新器
    mobject.add_updater(update)
    return mobject


def cycle_animation(animation: Animation, **kwargs) -> Mobject:
    # 调用turn_animation_into_updater并开启循环模式
    return turn_animation_into_updater(
        animation, cycle=True,** kwargs
    )
