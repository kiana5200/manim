# 从 __future__ 导入 annotations，支持在类型注解中使用尚未定义的类型（延迟解析）
from __future__ import annotations

# 导入 deepcopy 用于深拷贝对象（完全复制对象及其所有嵌套结构）
from copy import deepcopy

# 从 manimlib 的 mobject 模块导入动画构建器类和基本可渲染对象类
from manimlib.mobject.mobject import _AnimationBuilder  # 用于构建动画的工具类
from manimlib.mobject.mobject import Mobject  # 所有可渲染对象的基类

# 导入工具函数：移除列表冗余元素、平滑速率函数、数值截断函数
from manimlib.utils.iterables import remove_list_redundancies  # 移除列表中重复元素并保持顺序
from manimlib.utils.rate_functions import smooth  # 平滑过渡的速率函数（控制动画节奏）
from manimlib.utils.simple_functions import clip  # 将数值限制在特定范围内的截断函数

# 导入类型检查相关工具
from typing import TYPE_CHECKING

# 类型检查条件块：仅在静态类型检查时执行，运行时不执行
if TYPE_CHECKING:
    # 导入Callable用于注解可调用对象（如函数、方法）
    from typing import Callable
    # 导入Scene类用于类型注解（场景类，动画的容器）
    from manimlib.scene.scene import Scene


# 动画默认参数定义
DEFAULT_ANIMATION_RUN_TIME = 1.0  # 默认动画运行时间为1秒
DEFAULT_ANIMATION_LAG_RATIO = 0  # 默认动画延迟比例为0（无延迟）


class Animation(object):
    """动画基类，所有具体动画类的父类，定义了动画的基本属性和行为"""
    
    def __init__(
        self,
        mobject: Mobject,
        run_time: float = DEFAULT_ANIMATION_RUN_TIME,
        # 动画运行的时间区间（元组形式）
        time_span: tuple[float, float] | None = None,
        # 延迟比例：
        # - 0表示所有子对象同时开始动画
        # - 1表示按顺序依次应用到每个子对象
        # - 0到1之间表示每个子对象按滞后时间依次开始
        lag_ratio: float = DEFAULT_ANIMATION_LAG_RATIO,
        rate_func: Callable[[float], float] = smooth,
        name: str = "",
        # 该动画是否在屏幕上添加或移除mobject
        remover: bool = False,
        # 动画完成时更新函数的最终alpha值
        final_alpha_value: float = 1.0,
        # 如果设为True，mobject自身的内部更新器会被调用，
        # 但起始或目标mobject不会被暂停。
        # 若要完全暂停更新，请在动画前调用mobject.suspend_updating()
        suspend_mobject_updating: bool = False,
    ):
        # 验证输入的mobject类型是否合法
        self._validate_input_type(mobject)
        # 动画作用的mobject（Manim中的可动画对象）
        self.mobject = mobject
        # 动画运行时间（秒），默认使用全局默认值
        self.run_time = run_time
        # 动画运行的时间区间，None表示使用默认时间线
        self.time_span = time_span
        # 速率函数，控制动画进度的变化速率（如平滑过渡、先快后慢等）
        self.rate_func = rate_func
        # 动画名称，默认使用"类名+对象标识"的形式
        self.name = name or self.__class__.__name__ + str(self.mobject)
        # 标记该动画是否用于移除mobject
        self.remover = remover
        # 动画结束时的最终alpha值（用于透明度等渐变属性）
        self.final_alpha_value = final_alpha_value
        # 子对象动画的延迟比例
        self.lag_ratio = lag_ratio
        # 是否暂停mobject的自动更新
        self.suspend_mobject_updating = suspend_mobject_updating

    def _validate_input_type(self, mobject: Mobject) -> None:
        """验证输入对象是否为Mobject类型"""
    if not isinstance(mobject, Mobject):
        # 如果不是Mobject类型则抛出类型错误
        raise TypeError("Animation only works for Mobjects.")

    def __str__(self) -> str:
        """返回动画的名称字符串表示"""
        return self.name

    def begin(self) -> None:
        """动画开始时调用的初始化方法"""
    # 如果指定了时间区间，调整运行时间以适应区间
        if self.time_span is not None:
            start, end = self.time_span
            self.run_time = max(end, self.run_time)
    # 标记mobject为正在动画状态
        self.mobject.set_animating_status(True)
    # 创建mobject的初始状态副本
        self.starting_mobject = self.create_starting_mobject()
    # 根据设置暂停mobject的自动更新
        if self.suspend_mobject_updating:
            self.mobject_was_updating = not self.mobject.updating_suspended
            self.mobject.suspend_updating()
    # 获取所有相关mobject家族并压缩组合
        self.families = list(self.get_all_families_zipped())
    # 初始化为alpha=0的状态
        self.interpolate(0)

    def finish(self) -> None:
        """动画结束时调用的收尾方法"""
        # 将动画插值到最终状态
        self.interpolate(self.final_alpha_value)
        # 标记mobject为非动画状态
        self.mobject.set_animating_status(False)
        # 如果需要，恢复mobject的自动更新
        if self.suspend_mobject_updating and self.mobject_was_updating:
            self.mobject.resume_updating()

    def clean_up_from_scene(self, scene: Scene) -> None:
        """从场景中清理动画相关资源"""
        # 如果是移除型动画，从场景中移除mobject
        if self.is_remover():
            scene.remove(self.mobject)

    def create_starting_mobject(self) -> Mobject:
        """创建mobject的初始状态副本，记录动画开始前的状态"""
        # Keep track of where the mobject starts
        return self.mobject.copy()

    def get_all_mobjects(self) -> tuple[Mobject, Mobject]:
        """
        Ordering must match the ording of arguments to interpolate_submobject
        """
        """
        获取所有与动画相关的mobject
        顺序需与interpolate_submobject方法的参数顺序一致
        """
        return self.mobject, self.starting_mobject

    def get_all_families_zipped(self) -> zip[tuple[Mobject]]:
        """获取所有相关mobject的家族树并按层级压缩"""
        return zip(*[
            mob.get_family()  # 获取每个mobject的家族成员（包括自身和子对象）
            for mob in self.get_all_mobjects()
        ])

    def update_mobjects(self, dt: float) -> None:
        """
        Updates things like starting_mobject, and (for
        Transforms) target_mobject.
        """
        """更新动画相关的mobject（如起始状态副本）"""
        # 对需要更新的mobject调用update方法
        for mob in self.get_all_mobjects_to_update():
            mob.update(dt)

    def get_all_mobjects_to_update(self) -> list[Mobject]:
        """获取需要在动画过程中更新的mobject列表"""
        # 排除主mobject（通常由场景负责更新）
        items = list(filter(
            lambda m: m is not self.mobject,
            self.get_all_mobjects()
        ))
        # 移除列表中的重复项
        items = remove_list_redundancies(items)
        return items

    def copy(self):
        """创建动画对象的深拷贝"""
        return deepcopy(self)

    def update_rate_info(
        self,
        run_time: float | None = None,
        rate_func: Callable[[float], float] | None = None,
        lag_ratio: float | None = None,
    ):
        """更新动画的速率相关参数（运行时间、速率函数、延迟比例）"""
        self.run_time = run_time or self.run_time
        self.rate_func = rate_func or self.rate_func
        self.lag_ratio = lag_ratio or self.lag_ratio
        return self # 支持链式调用
    
    # 插值相关方法（动画的核心逻辑）
    # Methods for interpolation, the mean of an Animation
    def interpolate(self, alpha: float) -> None:
        """根据alpha值（0到1）插值更新mobject状态"""
        self.interpolate_mobject(alpha)

    def update(self, alpha: float) -> None:
        """
        兼容旧版本的更新方法
        本应被移除，但为了兼容旧场景保留
        """
        """
        This method shouldn't exist, but it's here to
        keep many old scenes from breaking
        """
        self.interpolate(alpha)

    def time_spanned_alpha(self, alpha: float) -> float:
        """根据时间区间调整alpha值"""
        if self.time_span is not None:
            start, end = self.time_span
            # 将alpha值映射到指定的时间区间内
            return clip(alpha * self.run_time - start, 0, end - start) / (end - start)
        return alpha

    def interpolate_mobject(self, alpha: float) -> None:
        """对mobject及其子对象进行插值更新"""
        for i, mobs in enumerate(self.families):
            # 计算每个子对象的插值比例（考虑延迟）
            sub_alpha = self.get_sub_alpha(self.time_spanned_alpha(alpha), i, len(self.families))
            # 对子对象进行插值
            self.interpolate_submobject(*mobs, sub_alpha)

    def interpolate_submobject(
        self,
        submobject: Mobject,
        starting_submobject: Mobject,
        alpha: float
    ):
        """
        子对象插值的具体实现
        通常由子类重写以实现特定动画效果
        """
        # Typically ipmlemented by subclass
        pass

    def get_sub_alpha(
        self,
        alpha: float,
        index: int,
        num_submobjects: int
    ) -> float:
        # TODO, make this more understanable, and/or combine
        # its functionality with AnimationGroup's method
        # build_animations_with_timings
        """计算子对象的插值比例（考虑延迟比例）"""
        lag_ratio = self.lag_ratio
        # 计算总动画长度（考虑所有子对象的延迟）
        full_length = (num_submobjects - 1) * lag_ratio + 1
        value = alpha * full_length
        # 计算当前子对象的起始位置
        lower = index * lag_ratio
        # 计算原始子对象alpha值并限制在0-1范围内
        raw_sub_alpha = clip((value - lower), 0, 1)
        # 应用速率函数调整动画节奏
        return self.rate_func(raw_sub_alpha)

    # Getters and setters
    def set_run_time(self, run_time: float):
        """设置动画运行时间"""
        self.run_time = run_time
        return self  # 支持链式调用

    def get_run_time(self) -> float:
        """获取动画运行时间（考虑时间区间）"""
        if self.time_span:
            return max(self.run_time, self.time_span[1])
        return self.run_time

    def set_rate_func(self, rate_func: Callable[[float], float]):
        """设置动画速率函数"""
        self.rate_func = rate_func
        return self# 支持链式调用

    def get_rate_func(self) -> Callable[[float], float]:
        """获取当前动画速率函数"""
        return self.rate_func

    def set_name(self, name: str):
        """设置动画名称"""
        self.name = name
        return self# 支持链式调用

    def is_remover(self) -> bool:
        """判断当前动画是否为移除型动画"""
        return self.remover


def prepare_animation(anim: Animation | _AnimationBuilder):
    """
    将动画构建器转换为实际动画对象
    用于统一处理Animation实例和_AnimationBuilder实例
    """
    if isinstance(anim, _AnimationBuilder):
        return anim.build() # 构建动画对象

    if isinstance(anim, Animation):
        return anim # 直接返回动画实例

    # 类型不匹配时抛出错误
    raise TypeError(f"Object {anim} cannot be converted to an animation")
