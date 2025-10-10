# 从__future__导入annotations，支持在类型注解中使用尚未定义的类
from __future__ import annotations

# 从manimlib的animation模块导入Animation基类和prepare_animation函数
from manimlib.animation.animation import Animation
from manimlib.animation.animation import prepare_animation

# 从mobject模块导入动画构建器和组合对象类
from manimlib.mobject.mobject import _AnimationBuilder  # 动画构建器，用于简化动画创建
from manimlib.mobject.mobject import Group  # 普通对象组合类

# 从vectorized_mobject模块导入矢量对象相关类
from manimlib.mobject.types.vectorized_mobject import VGroup  # 矢量对象组合类
from manimlib.mobject.types.vectorized_mobject import VMobject  # 矢量图形对象基类

# 从工具模块导入插值和贝塞尔曲线相关函数
from manimlib.utils.bezier import integer_interpolate  # 整数插值函数
from manimlib.utils.bezier import interpolate  # 通用插值函数
from manimlib.utils.iterables import remove_list_redundancies  # 移除列表中的冗余元素
from manimlib.utils.simple_functions import clip  # 截断函数，将值限制在指定范围内

# 导入类型检查相关模块
from typing import TYPE_CHECKING, Union, Iterable
# 如果是类型检查阶段，则导入相关类型（避免运行时循环导入问题）
if TYPE_CHECKING:
    # 定义AnimationType类型别名，表示可以是Animation实例或_AnimationBuilder
    AnimationType = Union[Animation, _AnimationBuilder]


DEFAULT_LAGGED_START_LAG_RATIO = 0.05


class AnimationGroup(Animation):
    def __init__(
        self,
        *args: AnimationType | Iterable[AnimationType],
        run_time: float = -1,  # If negative, default to sum of inputed animation runtimes
        lag_ratio: float = 0.0,
        group: Optional[Mobject] = None,
        group_type: Optional[type] = None,
        **kwargs
    ):
        animations = args[0] if isinstance(args[0], Iterable) else args
        self.animations = [prepare_animation(anim) for anim in animations]
        self.build_animations_with_timings(lag_ratio)
        self.max_end_time = max((awt[2] for awt in self.anims_with_timings), default=0)
        self.run_time = self.max_end_time if run_time < 0 else run_time
        self.lag_ratio = lag_ratio
        mobs = remove_list_redundancies([a.mobject for a in self.animations])
        if group is not None:
            self.group = group
        elif group_type is not None:
            self.group = group_type(*mobs)
        elif all(isinstance(anim.mobject, VMobject) for anim in animations):
            self.group = VGroup(*mobs)
        else:
            self.group = Group(*mobs)

        super().__init__(
            self.group,
            run_time=self.run_time,
            lag_ratio=lag_ratio,
            **kwargs
        )

    def get_all_mobjects(self) -> Mobject:
        return self.group

    def begin(self) -> None:
        self.group.set_animating_status(True)
        for anim in self.animations:
            anim.begin()
        # self.init_run_time()

    def finish(self) -> None:
        self.group.set_animating_status(False)
        for anim in self.animations:
            anim.finish()

    def clean_up_from_scene(self, scene: Scene) -> None:
        for anim in self.animations:
            anim.clean_up_from_scene(scene)

    def update_mobjects(self, dt: float) -> None:
        for anim in self.animations:
            anim.update_mobjects(dt)

    def calculate_max_end_time(self) -> None:
        self.max_end_time = max(
            (awt[2] for awt in self.anims_with_timings),
            default=0,
        )
        if self.run_time < 0:
            self.run_time = self.max_end_time

    def build_animations_with_timings(self, lag_ratio: float) -> None:
        """
        Creates a list of triplets of the form
        (anim, start_time, end_time)
        """
        self.anims_with_timings = []
        curr_time = 0
        for anim in self.animations:
            start_time = curr_time
            end_time = start_time + anim.get_run_time()
            self.anims_with_timings.append(
                (anim, start_time, end_time)
            )
            # Start time of next animation is based on the lag_ratio
            curr_time = interpolate(
                start_time, end_time, lag_ratio
            )

    def interpolate(self, alpha: float) -> None:
        # Note, if the run_time of AnimationGroup has been
        # set to something other than its default, these
        # times might not correspond to actual times,
        # e.g. of the surrounding scene.  Instead they'd
        # be a rescaled version.  But that's okay!
        time = alpha * self.max_end_time
        for anim, start_time, end_time in self.anims_with_timings:
            anim_time = end_time - start_time
            if anim_time == 0:
                sub_alpha = 0
            else:
                sub_alpha = clip((time - start_time) / anim_time, 0, 1)
            anim.interpolate(sub_alpha)


class Succession(AnimationGroup):
    def __init__(
        self,
        *animations: Animation,
        lag_ratio: float = 1.0,
        **kwargs
    ):
        super().__init__(*animations, lag_ratio=lag_ratio, **kwargs)

    def begin(self) -> None:
        assert len(self.animations) > 0
        self.active_animation = self.animations[0]
        self.active_animation.begin()

    def finish(self) -> None:
        self.active_animation.finish()

    def update_mobjects(self, dt: float) -> None:
        self.active_animation.update_mobjects(dt)

    def interpolate(self, alpha: float) -> None:
        index, subalpha = integer_interpolate(
            0, len(self.animations), alpha
        )
        animation = self.animations[index]
        if animation is not self.active_animation:
            self.active_animation.finish()
            animation.begin()
            self.active_animation = animation
        animation.interpolate(subalpha)


class LaggedStart(AnimationGroup):
    def __init__(
        self,
        *animations,
        lag_ratio: float = DEFAULT_LAGGED_START_LAG_RATIO,
        **kwargs
    ):
        super().__init__(*animations, lag_ratio=lag_ratio, **kwargs)


class LaggedStartMap(LaggedStart):
    def __init__(
        self,
        anim_func: Callable[[Mobject], Animation],
        group: Mobject,
        run_time: float = 2.0,
        lag_ratio: float = DEFAULT_LAGGED_START_LAG_RATIO,
        **kwargs
    ):
        anim_kwargs = dict(kwargs)
        anim_kwargs.pop("lag_ratio", None)
        super().__init__(
            *(anim_func(submob, **anim_kwargs) for submob in group),
            run_time=run_time,
            lag_ratio=lag_ratio,
            group=group
        )
