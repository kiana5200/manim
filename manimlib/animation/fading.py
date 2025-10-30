# 启用Python 3.7+的注解向前兼容支持，允许在类型注解中使用尚未定义的类
from __future__ import annotations

# 导入numpy库，用于数值计算和数组操作
import numpy as np

# 从Manim的动画模块导入基础动画类Animation
from manimlib.animation.animation import Animation
# 从变换动画模块导入Transform类，用于对象间的变换动画
from manimlib.animation.transform import Transform
# 从常量模块导入原点坐标常量ORIGIN（通常为(0, 0, 0)）
from manimlib.constants import ORIGIN
# 导入矢量图形对象类VMobject，用于处理可矢量动画的图形
from manimlib.mobject.types.vectorized_mobject import VMobject
# 导入组合对象类Group，用于管理多个Mobject的集合
from manimlib.mobject.mobject import Group
# 从贝塞尔曲线工具模块导入插值函数interpolate
from manimlib.utils.bezier import interpolate
# 从速率函数模块导入there_and_back函数，一种往返式的速率曲线
from manimlib.utils.rate_functions import there_and_back

# 导入类型检查相关模块
from typing import TYPE_CHECKING

# 条件导入，仅在类型检查时执行（运行时不生效）
# 用于解决循环导入问题，同时提供完整的类型提示支持
if TYPE_CHECKING:
    from typing import Callable  # 导入Callable类型，用于注解可调用对象
    from manimlib.mobject.mobject import Mobject  # 导入基础可移动对象类Mobject
    from manimlib.scene.scene import Scene  # 导入场景类Scene
    from manimlib.typing import Vect3  # 导入三维向量类型Vect3


# 定义Fade类，继承自Transform类，用于实现淡入淡出效果的动画
class Fade(Transform):
    # 初始化方法，设置淡入淡出动画的参数
    def __init__(
        self,
        mobject: Mobject,  # 要应用淡入淡出效果的Mobject对象
        shift: np.ndarray = ORIGIN,  # 动画过程中的位移向量，默认值为原点（不位移）
        scale: float = 1,  # 动画过程中的缩放因子，默认值为1（不缩放）
        **kwargs  # 其他关键字参数，用于传递给父类Transform
    ):
        # 保存位移向量到实例变量，供后续动画计算使用
        self.shift_vect = shift
        # 保存缩放因子到实例变量，供后续动画计算使用
        self.scale_factor = scale
        # 调用父类Transform的初始化方法，传递必要参数
        super().__init__(mobject,** kwargs)


# 定义FadeIn类，继承自Fade类，用于实现淡入动画效果
class FadeIn(Fade):
    # 创建目标状态的方法，定义动画结束时物体的状态
    def create_target(self) -> Mobject:
        # 返回原物体的副本作为目标状态（即淡入结束时显示原物体）
        return self.mobject.copy()

    # 创建起始状态的方法，定义动画开始时物体的状态
    def create_starting_mobject(self) -> Mobject:
        # 调用父类Fade的方法获取基础起始状态
        start = super().create_starting_mobject()
        # 设置起始状态的透明度为0（完全透明）
        start.set_opacity(0)
        # 根据缩放因子反向缩放（如果scale_factor>1，则起始状态更小）
        start.scale(1.0 / self.scale_factor)
        # 根据位移向量反向移动（为后续正向移动做准备）
        start.shift(-self.shift_vect)
        # 返回配置好的起始状态物体
        return start


# 定义FadeOut类，继承自Fade类，用于实现淡出动画效果
class FadeOut(Fade):
    # 初始化方法，设置淡出动画的参数
    def __init__(
        self,
        mobject: Mobject,  # 要应用淡出效果的Mobject对象
        shift: Vect3 = ORIGIN,  # 淡出过程中的位移向量，默认值为原点（不位移）
        remover: bool = True,  # 动画结束后是否移除物体，默认True（移除）
        final_alpha_value: float = 0.0,  # 最终透明度值，默认0.0（完全透明）
        **kwargs  # 其他关键字参数，传递给父类
    ):
        # 调用父类Fade的初始化方法，传递参数
        super().__init__(
            mobject, shift,  # 传递物体和位移参数
            remover=remover,  # 传递是否移除物体的参数
            final_alpha_value=final_alpha_value,  # 传递最终透明度参数
            **kwargs  # 传递其他关键字参数
        )

    # 创建目标状态的方法，定义动画结束时物体的状态
    def create_target(self) -> Mobject:
        # 复制原物体作为基础目标状态
        result = self.mobject.copy()
        # 设置目标状态的透明度为0（完全透明）
        result.set_opacity(0)
        # 按照位移向量移动目标状态物体
        result.shift(self.shift_vect)
        # 按照缩放因子缩放目标状态物体
        result.scale(self.scale_factor)
        # 返回配置好的目标状态物体
        return result


# 定义FadeInFromPoint类，继承自FadeIn，实现从指定点淡入的效果
class FadeInFromPoint(FadeIn):
    # 初始化方法，设置从指定点淡入的参数
    def __init__(self, mobject: Mobject, point: Vect3, **kwargs):
        # 调用父类FadeIn的初始化方法，传递参数
        super().__init__(
            mobject,  # 要淡入的物体
            # 计算位移向量：物体最终位置到起始点的向量（从point移动到物体原始位置）
            shift=mobject.get_center() - point,
            scale=np.inf,  # 缩放因子设为无穷大（起点状态会极度缩小）
            **kwargs,  # 其他关键字参数
        )


# 定义FadeOutToPoint类，继承自FadeOut，实现淡出到指定点的效果
class FadeOutToPoint(FadeOut):
    # 初始化方法，设置淡出到指定点的参数
    def __init__(self, mobject: Mobject, point: Vect3, **kwargs):
        # 调用父类FadeOut的初始化方法，传递参数
        super().__init__(
            mobject,  # 要淡出的物体
            # 计算位移向量：从物体原始位置到目标点的向量
            shift=point - mobject.get_center(),
            scale=0,  # 缩放因子设为0（终点状态会缩成一个点）
            **kwargs,  # 其他关键字参数
        )


# 定义FadeTransform类，继承自Transform，用于实现一个物体淡入同时另一个物体淡出的变换动画
class FadeTransform(Transform):
    # 初始化方法，设置淡入淡出变换的参数
    def __init__(
        self,
        mobject: Mobject,  # 源物体（将被淡出的物体）
        target_mobject: Mobject,  # 目标物体（将被淡入的物体）
        stretch: bool = True,  # 是否在变换时拉伸物体以匹配尺寸
        dim_to_match: int = 1,  # 用于匹配尺寸的维度
        **kwargs  # 其他关键字参数，传递给父类
    ):
        # 保存动画完成后要添加到场景的目标物体
        self.to_add_on_completion = target_mobject
        self.stretch = stretch  # 保存是否拉伸的参数
        self.dim_to_match = dim_to_match  # 保存用于匹配的维度参数

        # 保存源物体的当前状态，以便后续恢复
        mobject.save_state()
        # 调用父类Transform的初始化方法，传入包含源物体和目标物体副本的组
        super().__init__(Group(mobject, target_mobject.copy()),** kwargs)

    # 动画开始时执行的方法
    def begin(self) -> None:
        # 创建当前物体的副本作为结束状态的参考
        self.ending_mobject = self.mobject.copy()
        # 调用父类Animation的begin方法进行初始化
        Animation.begin(self)
        
        # 起始状态和结束状态都包含源物体和目标物体
        # 起始时，目标物体应透明并替换源物体；结束时则相反
        start, end = self.starting_mobject, self.ending_mobject
        
        # 配置起始和结束状态的物体
        # (start[1], start[0])：处理起始状态中目标物体到源物体的映射
        # (end[0], end[1])：处理结束状态中源物体到目标物体的映射
        for m0, m1 in ((start[1], start[0]), (end[0], end[1])):
            self.ghost_to(m0, m1)

    # 辅助方法：将源物体"幽灵化"为目标物体（位置大小匹配但透明）
    def ghost_to(self, source: Mobject, target: Mobject) -> None:
        # 将源物体替换为目标物体的位置和大小，保持拉伸和维度匹配参数
        source.replace(target, stretch=self.stretch, dim_to_match=self.dim_to_match)
        # 复制目标物体的所有统一属性（如颜色等）
        source.set_uniform(**target.get_uniforms())
        # 将源物体设置为完全透明
        source.set_opacity(0)

    # 获取所有参与动画的物体列表
    def get_all_mobjects(self) -> list[Mobject]:
        return [
            self.mobject,  # 当前物体组
            self.starting_mobject,  # 起始状态物体组
            self.ending_mobject,  # 结束状态物体组
        ]

    # 获取所有家族物体的压缩列表（用于动画插值）
    def get_all_families_zipped(self) -> zip[tuple[Mobject]]:
        # 调用父类Animation的方法获取压缩的家族物体列表
        return Animation.get_all_families_zipped(self)

    # 动画完成后从场景中清理的方法
    def clean_up_from_scene(self, scene: Scene) -> None:
        # 调用父类的清理方法
        Animation.clean_up_from_scene(self, scene)
        # 从场景中移除动画使用的物体组
        scene.remove(self.mobject)
        # 恢复源物体的原始状态
        self.mobject[0].restore()
        # 如果不需要移除物体，将目标物体添加到场景中
        if not self.remover:
            scene.add(self.to_add_on_completion)


# FadeTransformPieces类，继承自FadeTransform，用于对物体的多个部分分别进行淡入淡出变换
class FadeTransformPieces(FadeTransform):
    # 动画开始时执行的方法
    def begin(self) -> None:
        # 将源物体（索引0）与目标物体（索引1）的家族成员对齐
        # 确保变换时各对应部分位置匹配
        self.mobject[0].align_family(self.mobject[1])
        # 调用父类的begin方法完成初始化
        super().begin()

    # 重写ghost_to方法，对家族中的每个子物体分别处理
    def ghost_to(self, source: Mobject, target: Mobject) -> None:
        # 遍历源和目标物体的所有家族成员（子物体）
        for sm0, sm1 in zip(source.get_family(), target.get_family()):
            # 对每个子物体调用父类的ghost_to方法，实现逐部分变换
            super().ghost_to(sm0, sm1)


# VFadeIn类，继承自Animation，专门用于VMobject的淡入动画
class VFadeIn(Animation):
    """
    VFadeIn和VFadeOut仅适用于VMobjects（矢量物体），
    分别控制描边和填充的透明度
    """
    # 初始化方法
    def __init__(self, vmobject: VMobject, suspend_mobject_updating: bool = False, **kwargs):
        # 调用父类构造方法，传入矢量物体和其他参数
        super().__init__(
            vmobject,
            suspend_mobject_updating=suspend_mobject_updating,** kwargs
        )

    # 插值方法，控制子物体的淡入效果
    def interpolate_submobject(
        self,
        submob: VMobject,  # 要处理的子矢量物体
        start: VMobject,   # 起始状态
        alpha: float       # 动画进度（0到1）
    ) -> None:
        # 插值计算描边透明度：从0到起始状态的描边透明度
        submob.set_stroke(
            opacity=interpolate(0, start.get_stroke_opacity(), alpha)
        )
        # 插值计算填充透明度：从0到起始状态的填充透明度
        submob.set_fill(
            opacity=interpolate(0, start.get_fill_opacity(), alpha)
        )


# VFadeOut类，继承自VFadeIn，实现矢量物体的淡出动画
class VFadeOut(VFadeIn):
    # 初始化方法，增加了淡出相关参数
    def __init__(
        self,
        vmobject: VMobject,
        remover: bool = True,  # 动画结束后是否移除物体
        final_alpha_value: float = 0.0,  # 最终透明度值
        **kwargs
    ):
        # 调用父类构造方法
        super().__init__(
            vmobject,
            remover=remover,
            final_alpha_value=final_alpha_value,** kwargs
        )

    # 重写插值方法，实现淡出效果
    def interpolate_submobject(
        self,
        submob: VMobject,
        start: VMobject,
        alpha: float
    ) -> None:
        # 调用父类的插值方法，但将alpha反转（1-alpha）
        # 实现从原始透明度到0的过渡
        super().interpolate_submobject(submob, start, 1 - alpha)


# VFadeInThenOut类，继承自VFadeIn，实现先淡入再淡出的效果
class VFadeInThenOut(VFadeIn):
    # 初始化方法
    def __init__(
        self,
        vmobject: VMobject,
        # 速率函数默认使用there_and_back（去而复返），实现先增后减
        rate_func: Callable[[float], float] = there_and_back,
        remover: bool = True,  # 动画结束后是否移除物体
        final_alpha_value: float = 0.5,  # 最终保留的透明度
        **kwargs
    ):
        # 调用父类构造方法
        super().__init__(
            vmobject,
            rate_func=rate_func,
            remover=remover,
            final_alpha_value=final_alpha_value,** kwargs
        )
