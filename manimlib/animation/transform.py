# 从__future__模块导入annotations，支持类型注释的延迟评估
# 允许在类型提示中使用尚未定义的类或函数，提升代码灵活性
from __future__ import annotations

# 导入inspect模块，用于获取对象的运行时信息（如函数参数、类结构等）
import inspect

# 导入numpy库并简写为np，用于科学计算和数组操作
import numpy as np

# 从manimlib.animation.animation导入Animation基类
# Animation是所有动画类的父类，提供动画的基本框架和生命周期方法
from manimlib.animation.animation import Animation

# 从manimlib.constants导入DEG常量，表示角度单位（度）
from manimlib.constants import DEG
# 从manimlib.constants导入OUT常量，表示3D空间中的向外方向（垂直屏幕向外）
from manimlib.constants import OUT

# 从manimlib.mobject.mobject导入Group类
# Group是多个Mobject的容器，用于统一管理一组对象
from manimlib.mobject.mobject import Group
# 从manimlib.mobject.mobject导入Mobject基类
# Mobject是所有可移动对象的基类，提供位置、旋转、缩放等基础功能
from manimlib.mobject.mobject import Mobject

# 从manimlib.utils.paths导入path_along_arc函数
# 用于生成沿圆弧的路径，常用于定义对象的运动轨迹
from manimlib.utils.paths import path_along_arc
# 从manimlib.utils.paths导入straight_path函数
# 用于生成直线路径，定义对象从起点到终点的直线运动轨迹
from manimlib.utils.paths import straight_path

# 从typing模块导入TYPE_CHECKING常量
# 该常量在类型检查时为True，运行时为False，用于条件导入类型提示
from typing import TYPE_CHECKING

# 条件判断：仅在类型检查阶段执行以下代码
if TYPE_CHECKING:
    # 从typing模块导入Callable类型，用于标注可调用对象（如函数、方法）
    from typing import Callable
    # 导入numpy.typing模块并简写为npt，用于numpy数组的类型标注
    import numpy.typing as npt
    # 从manimlib.scene.scene导入Scene类
    # Scene是场景基类，用于组织和播放动画
    from manimlib.scene.scene import Scene
    # 从manimlib.typing导入ManimColor类型，用于颜色相关的类型标注
    from manimlib.typing import ManimColor


# 定义Transform类，继承自Animation类，用于实现动画中物体的变换效果
class Transform(Animation):
    # 布尔值属性，指示是否在场景中用目标物体替换原物体
    replace_mobject_with_target_in_scene: bool = False

    # 初始化方法，设置变换动画的各种参数
    def __init__(
        self,
        mobject: Mobject,          # 要进行变换的物体
        target_mobject: Mobject | None = None,  # 变换的目标物体，可为None
        path_arc: float = 0.0,     # 路径弧度，控制物体移动的弧线程度
        path_arc_axis: np.ndarray = OUT,  # 路径弧线的旋转轴，默认为OUT方向
        path_func: Callable | None = None,  # 自定义路径函数，控制物体移动路径
        **kwargs                   # 其他传递给父类的参数
    ):
        self.target_mobject = target_mobject  # 保存目标物体
        self.path_arc = path_arc              # 保存路径弧度
        self.path_arc_axis = path_arc_axis    # 保存路径弧线轴
        self.path_func = path_func            # 保存路径函数
        super().__init__(mobject,** kwargs)   # 调用父类的初始化方法
        self.init_path_func()                 # 初始化路径函数

    # 初始化路径函数的方法
    def init_path_func(self) -> None:
        if self.path_func is not None:        # 如果已设置自定义路径函数，直接返回
            return
        elif self.path_arc == 0:              # 如果路径弧度为0，使用直线路径
            self.path_func = straight_path
        else:                                 # 否则，使用弧线路径
            self.path_func = path_along_arc(
                self.path_arc,
                self.path_arc_axis,
            )

    # 动画开始时执行的方法
    def begin(self) -> None:
        self.target_mobject = self.create_target()  # 创建目标物体（子类可重写）
        self.check_target_mobject_validity()        # 检查目标物体是否有效

        # 检查原物体与目标物体是否对齐
        if self.mobject.is_aligned_with(self.target_mobject):
            self.target_copy = self.target_mobject  # 对齐则直接使用目标物体
        else:
            # 不对齐则创建目标物体的副本，用于数据对齐（不修改原始目标物体）
            self.target_copy = self.target_mobject.copy()
        # 对齐原物体与目标副本的数据结构，确保能正确插值
        self.mobject.align_data_and_family(self.target_copy)
        super().begin()  # 调用父类的begin方法
        # 如果原物体没有更新器，则锁定其与起始状态和目标副本的数据匹配
        if not self.mobject.has_updaters():
            self.mobject.lock_matching_data(
                self.starting_mobject,
                self.target_copy,
            )

    # 动画结束时执行的方法
    def finish(self) -> None:
        super().finish()  # 调用父类的finish方法
        self.mobject.unlock_data()  # 解锁物体的数据锁定

    # 创建目标物体的方法，子类可重写以实现自定义目标
    def create_target(self) -> Mobject:
        # 这里直接返回已设置的目标物体，在子类中可能有更复杂的实现
        return self.target_mobject

    # 检查目标物体是否有效的方法
    def check_target_mobject_validity(self) -> None:
        if self.target_mobject is None:  # 如果目标物体为None，抛出异常
            raise Exception(
                f"{self.__class__.__name__}.create_target not properly implemented"
            )

    # 从场景中清理动画资源的方法
    def clean_up_from_scene(self, scene: Scene) -> None:
        super().clean_up_from_scene(scene)  # 调用父类的清理方法
        # 如果需要替换原物体，则从场景中移除原物体并添加目标物体
        if self.replace_mobject_with_target_in_scene:
            scene.remove(self.mobject)
            scene.add(self.target_mobject)

    # 更新配置参数的方法
    def update_config(self, **kwargs) -> None:
        Animation.update_config(self,** kwargs)  # 调用父类的更新配置方法
        # 如果更新了路径弧度，则重新设置路径函数
        if "path_arc" in kwargs:
            self.path_func = path_along_arc(
                kwargs["path_arc"],
                kwargs.get("path_arc_axis", OUT)
            )

    # 获取所有相关物体的方法
    def get_all_mobjects(self) -> list[Mobject]:
        return [
            self.mobject,            # 原物体
            self.starting_mobject,   # 起始状态的物体
            self.target_mobject,     # 目标物体
            self.target_copy,        # 目标物体的副本
        ]

    # 获取所有相关物体家族并打包的方法，用于插值计算
    def get_all_families_zipped(self) -> zip[tuple[Mobject]]:
        return zip(*[
            mob.get_family()  # 获取每个物体的家族（包括子物体）
            for mob in [
                self.mobject,          # 原物体
                self.starting_mobject, # 起始状态的物体
                self.target_copy,      # 目标物体的副本
            ]
        ])

    # 插值计算子物体的方法，控制每个子物体如何从起始状态过渡到目标状态
    def interpolate_submobject(
        self,
        submob: Mobject,        # 要插值的子物体
        start: Mobject,         # 起始状态的子物体
        target_copy: Mobject,   # 目标状态的子物体副本
        alpha: float            # 插值系数，0表示起始状态，1表示目标状态
    ):
        # 调用子物体的插值方法，应用路径函数
        submob.interpolate(start, target_copy, alpha, self.path_func)
        return self  # 返回自身，支持链式调用


# 定义ReplacementTransform类，继承自Transform
class ReplacementTransform(Transform):
    # 重写父类属性，设置为True表示动画结束后用目标物体替换原物体
    replace_mobject_with_target_in_scene: bool = True


# 定义TransformFromCopy类，继承自Transform
class TransformFromCopy(Transform):
    # 动画结束后用目标物体替换原物体
    replace_mobject_with_target_in_scene: bool = True

    # 初始化方法，接收原物体和目标物体
    def __init__(self, mobject: Mobject, target_mobject: Mobject, **kwargs):
        # 调用父类初始化方法，但使用原物体的副本作为动画起始物体
        # 这样原物体会保持不动，由其副本执行变换动画到目标物体
        super().__init__(mobject.copy(), target_mobject,** kwargs)


# 定义MoveToTarget类，继承自Transform
class MoveToTarget(Transform):
    # 初始化方法，只需要传入要移动的物体
    def __init__(self, mobject: Mobject, **kwargs):
        # 检查输入的有效性
        self.check_validity_of_input(mobject)
        # 调用父类初始化方法，目标物体是该物体自身的target属性
        super().__init__(mobject, mobject.target,** kwargs)

    # 检查输入物体是否有效
    def check_validity_of_input(self, mobject: Mobject) -> None:
        # 如果物体没有target属性，抛出异常
        if not hasattr(mobject, "target"):
            raise Exception(
                "MoveToTarget called on mobject without attribute 'target'"
            )


class _MethodAnimation(MoveToTarget):
    def __init__(self, mobject: Mobject, methods: list[Callable], **kwargs):
        self.methods = methods
        super().__init__(mobject, **kwargs)


class ApplyMethod(Transform):
    def __init__(self, method: Callable, *args, **kwargs):
        """
        method is a method of Mobject, *args are arguments for
        that method.  Key word arguments should be passed in
        as the last arg, as a dict, since **kwargs is for
        configuration of the transform itself

        Relies on the fact that mobject methods return the mobject
        """
        self.check_validity_of_input(method)
        self.method = method
        self.method_args = args
        super().__init__(method.__self__, **kwargs)

    def check_validity_of_input(self, method: Callable) -> None:
        if not inspect.ismethod(method):
            raise Exception(
                "Whoops, looks like you accidentally invoked "
                "the method you want to animate"
            )
        assert isinstance(method.__self__, Mobject)

    def create_target(self) -> Mobject:
        method = self.method
        # Make sure it's a list so that args.pop() works
        args = list(self.method_args)

        if len(args) > 0 and isinstance(args[-1], dict):
            method_kwargs = args.pop()
        else:
            method_kwargs = {}
        target = method.__self__.copy()
        method.__func__(target, *args, **method_kwargs)
        return target


class ApplyPointwiseFunction(ApplyMethod):
    def __init__(
        self,
        function: Callable[[np.ndarray], np.ndarray],
        mobject: Mobject,
        run_time: float = 3.0,
        **kwargs
    ):
        super().__init__(mobject.apply_function, function, run_time=run_time, **kwargs)


class ApplyPointwiseFunctionToCenter(Transform):
    def __init__(
        self,
        function: Callable[[np.ndarray], np.ndarray],
        mobject: Mobject,
        **kwargs
    ):
        self.function = function
        super().__init__(mobject, **kwargs)

    def create_target(self) -> Mobject:
        return self.mobject.copy().move_to(self.function(self.mobject.get_center()))


class FadeToColor(ApplyMethod):
    def __init__(
        self,
        mobject: Mobject,
        color: ManimColor,
        **kwargs
    ):
        super().__init__(mobject.set_color, color, **kwargs)


class ScaleInPlace(ApplyMethod):
    def __init__(
        self,
        mobject: Mobject,
        scale_factor: npt.ArrayLike,
        **kwargs
    ):
        super().__init__(mobject.scale, scale_factor, **kwargs)


class ShrinkToCenter(ScaleInPlace):
    def __init__(self, mobject: Mobject, **kwargs):
        super().__init__(mobject, 0, **kwargs)


class Restore(Transform):
    def __init__(self, mobject: Mobject, **kwargs):
        if not hasattr(mobject, "saved_state") or mobject.saved_state is None:
            raise Exception("Trying to restore without having saved")
        super().__init__(mobject, mobject.saved_state, **kwargs)


class ApplyFunction(Transform):
    def __init__(
        self,
        function: Callable[[Mobject], Mobject],
        mobject: Mobject,
        **kwargs
    ):
        self.function = function
        super().__init__(mobject, **kwargs)

    def create_target(self) -> Mobject:
        target = self.function(self.mobject.copy())
        if not isinstance(target, Mobject):
            raise Exception("Functions passed to ApplyFunction must return object of type Mobject")
        return target


class ApplyMatrix(ApplyPointwiseFunction):
    def __init__(
        self,
        matrix: npt.ArrayLike,
        mobject: Mobject,
        **kwargs
    ):
        matrix = self.initialize_matrix(matrix)

        def func(p):
            return np.dot(p, matrix.T)

        super().__init__(func, mobject, **kwargs)

    def initialize_matrix(self, matrix: npt.ArrayLike) -> np.ndarray:
        matrix = np.array(matrix)
        if matrix.shape == (2, 2):
            new_matrix = np.identity(3)
            new_matrix[:2, :2] = matrix
            matrix = new_matrix
        elif matrix.shape != (3, 3):
            raise Exception("Matrix has bad dimensions")
        return matrix


class ApplyComplexFunction(ApplyMethod):
    def __init__(
        self,
        function: Callable[[complex], complex],
        mobject: Mobject,
        **kwargs
    ):
        self.function = function
        method = mobject.apply_complex_function
        super().__init__(method, function, **kwargs)

    def init_path_func(self) -> None:
        func1 = self.function(complex(1))
        self.path_arc = np.log(func1).imag
        super().init_path_func()

###


class CyclicReplace(Transform):
    def __init__(self, *mobjects: Mobject, path_arc=90 * DEG, **kwargs):
        super().__init__(Group(*mobjects), path_arc=path_arc, **kwargs)

    def create_target(self) -> Mobject:
        group = self.mobject
        target = group.copy()
        cycled_targets = [target[-1], *target[:-1]]
        for m1, m2 in zip(cycled_targets, group):
            m1.move_to(m2)
        return target


class Swap(CyclicReplace):
    """Alternate name for CyclicReplace"""
    pass
