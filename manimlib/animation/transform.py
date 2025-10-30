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


# 定义_MethodAnimation类，继承自MoveToTarget
# 这是一个内部使用的动画类（类名前的下划线通常表示内部使用）
class _MethodAnimation(MoveToTarget):
    # 初始化方法，接收一个物体和一个方法列表
    def __init__(self, mobject: Mobject, methods: list[Callable], **kwargs):
        # 存储方法列表，这些方法将用于修改目标状态
        self.methods = methods
        # 调用父类MoveToTarget的初始化方法，传入物体和其他参数
        super().__init__(mobject, **kwargs)


# 定义ApplyMethod类，继承自Transform，用于将方法应用于物体的动画
class ApplyMethod(Transform):
    def __init__(self, method: Callable, *args, **kwargs):
        """
        method是Mobject的一个方法，*args是该方法的参数。
        关键字参数应作为最后一个参数以字典形式传入，
        因为**kwargs用于配置变换本身
        依赖于mobject方法返回mobject这一特性
        """
        # 检查输入的方法是否有效
        self.check_validity_of_input(method)
        # 存储要应用的方法
        self.method = method
        # 存储方法的参数
        self.method_args = args
        # 调用父类Transform的初始化方法，传入方法所属的物体和其他参数
        # method.__self__指的是调用该方法的Mobject实例
        super().__init__(method.__self__, **kwargs)

    # 检查输入的方法是否有效的辅助方法
    def check_validity_of_input(self, method: Callable) -> None:
        # 检查是否为方法（而非函数或已调用的方法结果）
        if not inspect.ismethod(method):
            raise Exception(
                "Whoops, looks like you accidentally invoked "
                "the method you want to animate"
            )
        # 确保方法的所有者是Mobject实例
        assert isinstance(method.__self__, Mobject)

    # 创建目标物体的方法，重写父类方法
    def create_target(self) -> Mobject:
        # 获取要应用的方法
        method = self.method
        # 将参数转为列表，以便使用pop()方法
        args = list(self.method_args)

        # 如果最后一个参数是字典，则视为方法的关键字参数
        if len(args) > 0 and isinstance(args[-1], dict):
            method_kwargs = args.pop()
        else:
            method_kwargs = {}
        # 创建方法所属物体的副本作为目标物体的基础
        target = method.__self__.copy()
        # 对目标物体应用该方法（使用函数形式调用，传入目标作为第一个参数）
        # 这样原物体不变，而目标物体成为应用方法后的状态
        method.__func__(target, *args, **method_kwargs)
        # 返回处理后的目标物体
        return target


# 定义ApplyPointwiseFunction类，继承自ApplyMethod
# 用于对物体的每个点应用指定函数的动画
class ApplyPointwiseFunction(ApplyMethod):
    # 初始化方法
    def __init__(
        self,
        function: Callable[[np.ndarray], np.ndarray],  # 点变换函数，接收并返回numpy数组
        mobject: Mobject,  # 要应用函数的物体
        run_time: float = 3.0,  # 动画运行时间，默认3秒
        **kwargs  # 其他传递给父类的参数
    ):
        # 调用父类ApplyMethod的初始化方法
        # 将mobject的apply_function方法作为要执行的方法
        # 传入的参数为自定义的function，同时指定运行时间和其他参数
        super().__init__(mobject.apply_function, function, run_time=run_time,** kwargs)


# 定义ApplyPointwiseFunctionToCenter类，继承自Transform
# 用于对物体中心点应用指定函数的变换动画
class ApplyPointwiseFunctionToCenter(Transform):
    # 初始化方法
    def __init__(
        self,
        function: Callable[[np.ndarray], np.ndarray],  # 作用于中心点的函数，接收并返回numpy数组（坐标）
        mobject: Mobject,  # 要变换的物体
        **kwargs  # 其他传递给父类的参数
    ):
        self.function = function  # 存储作用于中心点的函数
        super().__init__(mobject, **kwargs)  # 调用父类Transform的初始化方法

    # 重写创建目标物体的方法
    def create_target(self) -> Mobject:
        # 1. 创建原物体的副本
        # 2. 将副本移动到新位置：原物体中心点经过function变换后的坐标
        # 最终返回这个经过位置变换的副本作为目标状态
        return self.mobject.copy().move_to(self.function(self.mobject.get_center()))


# 定义FadeToColor类，继承自ApplyMethod
# 用于实现物体渐变为指定颜色的动画
class FadeToColor(ApplyMethod):
    def __init__(
        self,
        mobject: Mobject,       # 要改变颜色的物体
        color: ManimColor,      # 目标颜色
        **kwargs                # 其他动画参数（如运行时间等）
    ):
        # 调用父类ApplyMethod的初始化方法
        # 使用mobject的set_color方法作为要执行的方法，传入目标颜色
        super().__init__(mobject.set_color, color, **kwargs)


# 定义ScaleInPlace类，继承自ApplyMethod
# 用于实现物体原地缩放的动画
class ScaleInPlace(ApplyMethod):
    def __init__(
        self,
        mobject: Mobject,           # 要缩放的物体
        scale_factor: npt.ArrayLike, # 缩放因子（可以是单个数值或数组）
        **kwargs                    # 其他动画参数
    ):
        # 调用父类初始化方法，使用mobject的scale方法实现缩放
        super().__init__(mobject.scale, scale_factor, **kwargs)


# 定义ShrinkToCenter类，继承自ScaleInPlace
# 用于实现物体缩小到中心消失的动画
class ShrinkToCenter(ScaleInPlace):
    def __init__(self, mobject: Mobject, **kwargs):
        # 调用父类ScaleInPlace的初始化方法，缩放因子设为0
        # 当缩放因子为0时，物体将缩小到中心点
        super().__init__(mobject, 0, **kwargs)


# 定义Restore类，继承自Transform
# 用于实现物体从当前状态恢复到之前保存状态的动画
class Restore(Transform):
    def __init__(self, mobject: Mobject, **kwargs):
        # 检查物体是否有已保存的状态
        if not hasattr(mobject, "saved_state") or mobject.saved_state is None:
            raise Exception("Trying to restore without having saved")
        # 调用父类Transform的初始化方法，目标状态为保存的状态
        super().__init__(mobject, mobject.saved_state, **kwargs)


# 定义ApplyFunction类，继承自Transform
# 用于将自定义函数应用于物体的动画
class ApplyFunction(Transform):
    def __init__(
        self,
        function: Callable[[Mobject], Mobject],  # 作用于物体的函数，输入输出都是Mobject
        mobject: Mobject,                        # 要处理的物体
        **kwargs                                  # 其他动画参数
    ):
        self.function = function  # 存储自定义函数
        super().__init__(mobject, **kwargs)  # 调用父类初始化方法

    # 重写创建目标物体的方法
    def create_target(self) -> Mobject:
        # 对物体的副本应用自定义函数，生成目标状态
        target = self.function(self.mobject.copy())
        # 检查函数返回值是否为Mobject类型
        if not isinstance(target, Mobject):
            raise Exception("Functions passed to ApplyFunction must return object of type Mobject")
        return target


# 定义ApplyMatrix类，继承自ApplyPointwiseFunction
# 用于对物体应用矩阵变换的动画（如旋转、缩放、剪切等线性变换）
class ApplyMatrix(ApplyPointwiseFunction):
    def __init__(
        self,
        matrix: npt.ArrayLike,  # 变换矩阵，可以是2x2或3x3矩阵
        mobject: Mobject,       # 要应用矩阵变换的物体
        **kwargs                # 其他动画参数
    ):
        # 初始化矩阵，确保格式正确
        matrix = self.initialize_matrix(matrix)

        # 定义对点应用矩阵变换的函数
        def func(p):
            # 对每个点p应用矩阵变换（通过点积实现）
            return np.dot(p, matrix.T)

        # 调用父类构造方法，传入变换函数和物体
        super().__init__(func, mobject, **kwargs)

    # 初始化矩阵，确保其为3x3矩阵（适合3D空间变换）
    def initialize_matrix(self, matrix: npt.ArrayLike) -> np.ndarray:
        matrix = np.array(matrix)
        # 如果是2x2矩阵，扩展为3x3矩阵（保持z坐标不变）
        if matrix.shape == (2, 2):
            new_matrix = np.identity(3)  # 创建3x3单位矩阵
            new_matrix[:2, :2] = matrix  # 将2x2矩阵放入左上角
            matrix = new_matrix
        # 检查矩阵是否为3x3，否则抛出异常
        elif matrix.shape != (3, 3):
            raise Exception("Matrix has bad dimensions")
        return matrix


# 定义ApplyComplexFunction类，继承自ApplyMethod
# 用于对物体应用复变函数变换的动画（适合2D平面变换）
class ApplyComplexFunction(ApplyMethod):
    def __init__(
        self,
        function: Callable[[complex], complex],  # 复变函数，输入输出均为复数
        mobject: Mobject,                        # 要变换的物体
        **kwargs                                  # 其他动画参数
    ):
        self.function = function  # 存储复变函数
        # 使用物体的apply_complex_function方法
        method = mobject.apply_complex_function
        super().__init__(method, function, **kwargs)

    # 初始化路径函数，计算变换的旋转角度
    def init_path_func(self) -> None:
        # 计算函数在z=1处的取值，用于确定旋转角度
        func1 = self.function(complex(1))
        # 计算复数的辐角（虚部的对数），作为路径弧度
        self.path_arc = np.log(func1).imag
        # 调用父类的路径初始化方法
        super().init_path_func()


# 定义CyclicReplace类，继承自Transform
# 用于实现多个物体循环替换位置的动画
class CyclicReplace(Transform):
    def __init__(self, *mobjects: Mobject, path_arc=90 * DEG, **kwargs):
        # 将所有物体组合成一个组，作为变换的对象
        super().__init__(Group(*mobjects), path_arc=path_arc, **kwargs)

    # 创建目标状态
    def create_target(self) -> Mobject:
        group = self.mobject  # 获取包含所有物体的组
        target = group.copy()  # 创建组的副本作为目标基础
        # 循环移位：最后一个物体移到第一个位置，其余依次后移
        cycled_targets = [target[-1], *target[:-1]]
        # 每个目标物体移动到下一个物体的原始位置
        for m1, m2 in zip(cycled_targets, group):
            m1.move_to(m2)
        return target


# 定义Swap类，继承自CyclicReplace
# 作为CyclicReplace的别名，特别适合两个物体交换位置的场景
class Swap(CyclicReplace):
    """Alternate name for CyclicReplace"""
    pass