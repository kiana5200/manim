# 从 __future__ 模块导入 annotations 特性。
# 这是一个非常有用的特性，它允许在函数注解（类型提示）中直接使用尚未完全定义的类名或函数名，
# 而无需将其放在字符串中。这使得代码更具可读性。
# 例如，在类定义内部，你可以写 `def get_parent(self) -> Node:`，而不是 `def get_parent(self) -> 'Node':`。
from __future__ import annotations

# 导入 Python 标准库中的 math 模块，用于执行各种数学运算，如 sqrt, sin, cos, pi 等。
import math

# 导入 Python 标准库中的 copy 模块，用于创建对象的副本。
# 例如，copy.copy() 用于浅拷贝，copy.deepcopy() 用于深拷贝。
import copy

# 导入 NumPy 库，并将其别名为 np。NumPy 是 Python 进行科学计算的核心库，
# 提供了高性能的多维数组对象（ndarray）以及大量用于数组操作的函数。
import numpy as np

from manimlib.constants import DEFAULT_MOBJECT_TO_MOBJECT_BUFF, SMALL_BUFF
from manimlib.constants import DOWN, LEFT, ORIGIN, RIGHT, DL, DR, UL, UP
from manimlib.constants import PI
from manimlib.animation.composition import AnimationGroup
from manimlib.animation.fading import FadeIn
from manimlib.animation.growing import GrowFromCenter
from manimlib.mobject.svg.tex_mobject import Tex
from manimlib.mobject.svg.tex_mobject import TexText
from manimlib.mobject.svg.text_mobject import Text
from manimlib.mobject.types.vectorized_mobject import VGroup
from manimlib.mobject.types.vectorized_mobject import VMobject
from manimlib.utils.iterables import listify
from manimlib.utils.space_ops import get_norm

# 从 typing 模块导入 TYPE_CHECKING 常量。
# 这个常量在程序运行时的值永远是 False，但在静态类型检查工具（如 mypy, pyright）分析代码时，
# 它会被视为 True。这使得我们可以编写仅在类型检查阶段执行的代码。
from typing import TYPE_CHECKING

# 如果正在进行类型检查，则执行以下代码块。
# 这是一种常见的模式，用于导入仅在类型注解中使用的模块，从而避免在程序运行时产生不必要的
# 导入开销或解决潜在的循环导入问题。
if TYPE_CHECKING:
    # 从 typing 模块导入 Iterable 类型，用于注解一个可迭代对象（如列表、元组、生成器等）。
    from typing import Iterable

    # 从 manimlib.animation.animation 模块导入 Animation 类。
    # 这是 Manim 动画库中所有动画的基类。
    from manimlib.animation.animation import Animation
    
    # 从 manimlib.mobject.mobject 模块导入 Mobject 类。
    # 这是 Manim 动画库中所有可显示对象（如形状、文本、图像）的基类。
    from manimlib.mobject.mobject import Mobject
    
    # 从 manimlib.typing 模块导入 Vect3 类型。
    # 这是一个自定义类型注解，通常代表一个三维向量（x, y, z），在 Manim 中用于表示位置、
    # 方向或缩放比例。它通常是一个 NumPy 数组或包含三个浮点数的元组。
    from manimlib.typing import Vect3


# 定义一个名为 Brace 的类，它继承自 Tex 类。
# 这意味着 Brace 是一个特殊的 Tex 对象，它的主要功能是在另一个 Mobject 下方（或旁边）绘制一个花括号。
class Brace(Tex):
    def __init__(
        self,
        mobject: Mobject,
        direction: Vect3 = DOWN,
        buff: float = 0.2,
        tex_string: str = R"\underbrace{\qquad}",
        **kwargs
    ):
        # 调用父类 Tex 的构造函数，用指定的 LaTeX 字符串（默认是带花括号的）和其他关键字参数来初始化对象。
        # 此时，self 是一个包含标准尺寸花括号的 Tex 对象。
        super().__init__(tex_string, **kwargs)

        # 1. 计算旋转角度
        # 综合效果是：计算出一个角度，使得当我们按这个角度旋转时，'direction' 向量会指向正下方 (DOWN)。
        angle = -math.atan2(*direction[:2]) + PI

        # 2. 临时旋转目标对象
        # about_point=ORIGIN 表示旋转的支点是坐标系原点。
        mobject.rotate(-angle, about_point=ORIGIN)
        # 3. 获取目标宽度
        # 在旋转后的坐标系中，获取 mobject 的左下角 (DL) 和右下角 (DR) 的坐标。
        left = mobject.get_corner(DL)
        right = mobject.get_corner(DR)
        # 目标宽度就是这两个点在 x 轴上的差值。
        target_width = right[0] - left[0]

        # 4. 调整花括号自身的尺寸
        # 找到花括号尖端在所有点中的索引。花括号的尖端 y 坐标最小，所以用 np.argmin 找到 y 坐标最小的点。
        # 这一步是为了后续更精确的定位或宽度调整做准备。
        self.tip_point_index = np.argmin(self.get_all_points()[:, 1])
        # 调用之前定义的 set_initial_width 方法，将花括号的宽度拉伸或收缩到与 mobject 匹配的 target_width。
        self.set_initial_width(target_width)
        # 5. 定位花括号
        # - self.get_corner(UL) 是花括号自身（在其局部坐标系中）的左上角。
        # - left - self.get_corner(UL) 计算出一个偏移向量，这个向量能将花括号的左上角移动到 mobject 的左下角。
        # - + buff * DOWN 在垂直方向上再增加一个缓冲距离，将花括号向下移动一点，使其与 mobject 分开。
        # - self.shift(...) 将计算出的总偏移量应用到花括号上，使其移动到 mobject 正下方的正确位置。
        self.shift(left - self.get_corner(UL) + buff * DOWN)
        # 6. 恢复原始方向
        # 循环遍历 mobject 和 self（花括号）。
        # 将它们按最初计算的 'angle' 旋转回来。
        # 因为之前 mobject 被反向旋转了，现在旋转回来就恢复了它原来的方向。
        # 花括号因为已经被定位在 mobject 的“下方”，所以当它和 mobject 一起旋转回来时，
        # 就会出现在 mobject 相对于其原始方向的正确一侧（下方、左侧、右侧或上方）。
        for mob in mobject, self:
            mob.rotate(angle, about_point=ORIGIN)

    # 设置对象的初始宽度。
    def set_initial_width(self, width: float):
        # 1. 计算当前宽度与目标宽度的差值
        width_diff = width - self.get_width()
        # 2. 如果目标宽度大于当前宽度（需要增加宽度）
        if width_diff > 0:
            # 遍历需要调整的两个部分：右侧部分和左侧部分
            # - self[0], self[1], RIGHT: 代表右侧的箭头尖(tip)、右侧的矩形(rect)以及方向向量(RIGHT)
            # - self[5], self[4], LEFT: 代表左侧的箭头尖(tip)、左侧的矩形(rect)以及方向向量(LEFT)
            for tip, rect, vect in [(self[0], self[1], RIGHT), (self[5], self[4], LEFT)]:
                # a. 调整矩形的宽度
                # 将宽度差的一半加到矩形当前的宽度上
                # 'about_edge=vect' 表示缩放操作的锚点是矩形的特定边缘（右侧矩形以右边缘为锚，左侧矩形以左边缘为锚）
                # 'stretch=True' 表示通过拉伸/压缩矩形本身来改变宽度，而不是进行整体缩放（scaling）
                rect.set_width(
                    width_diff / 2 + rect.get_width(),
                    about_edge=vect, stretch=True
                )
                # b. 移动箭头尖的位置
                # 将箭头尖沿着相反的方向移动宽度差的一半，以保持它与矩形新边缘的正确对齐
                tip.shift(-width_diff / 2 * vect)

        # 3. 如果目标宽度小于或等于当前宽度（需要减少宽度）                
        else:
            self.set_width(width, stretch=True)
        # 4. 返回 self 以支持链式调用
        return self

    def put_at_tip(
        self,
        mob: Mobject,
        use_next_to: bool = True,
        **kwargs
    ):
        if use_next_to:
            mob.next_to(
                self.get_tip(),
                np.round(self.get_direction()),
                **kwargs
            )
        else:
            mob.move_to(self.get_tip())
            buff = kwargs.get("buff", DEFAULT_MOBJECT_TO_MOBJECT_BUFF)
            shift_distance = mob.get_width() / 2.0 + buff
            mob.shift(self.get_direction() * shift_distance)
        return self

    def get_text(self, text: str, **kwargs) -> Text:
        buff = kwargs.pop("buff", SMALL_BUFF)
        text_mob = Text(text, **kwargs)
        self.put_at_tip(text_mob, buff=buff)
        return text_mob

    def get_tex(self, *tex: str, **kwargs) -> Tex:
        buff = kwargs.pop("buff", SMALL_BUFF)
        tex_mob = Tex(*tex, **kwargs)
        self.put_at_tip(tex_mob, buff=buff)
        return tex_mob

    def get_tip(self) -> np.ndarray:
        # Very specific to the LaTeX representation
        # of a brace, but it's the only way I can think
        # of to get the tip regardless of orientation.
        return self.get_all_points()[self.tip_point_index]

    def get_direction(self) -> np.ndarray:
        vect = self.get_tip() - self.get_center()
        return vect / get_norm(vect)


class BraceLabel(VMobject):
    label_constructor: type = Tex

    def __init__(
        self,
        obj: VMobject | list[VMobject],
        text: str | Iterable[str],
        brace_direction: np.ndarray = DOWN,
        label_scale: float = 1.0,
        label_buff: float = DEFAULT_MOBJECT_TO_MOBJECT_BUFF,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.brace_direction = brace_direction
        self.label_scale = label_scale
        self.label_buff = label_buff

        if isinstance(obj, list):
            obj = VGroup(*obj)
        self.brace = Brace(obj, brace_direction, **kwargs)

        self.label = self.label_constructor(*listify(text), **kwargs)
        self.label.scale(self.label_scale)

        self.brace.put_at_tip(self.label, buff=self.label_buff)
        self.set_submobjects([self.brace, self.label])

    def creation_anim(
        self,
        label_anim: Animation = FadeIn,
        brace_anim: Animation = GrowFromCenter
    ) -> AnimationGroup:
        return AnimationGroup(brace_anim(self.brace), label_anim(self.label))

    def shift_brace(self, obj: VMobject | list[VMobject], **kwargs):
        if isinstance(obj, list):
            obj = VMobject(*obj)
        self.brace = Brace(obj, self.brace_direction, **kwargs)
        self.brace.put_at_tip(self.label)
        self.submobjects[0] = self.brace
        return self

    def change_label(self, *text: str, **kwargs):
        self.label = self.label_constructor(*text, **kwargs)
        if self.label_scale != 1:
            self.label.scale(self.label_scale)

        self.brace.put_at_tip(self.label)
        self.submobjects[1] = self.label
        return self

    def change_brace_label(self, obj: VMobject | list[VMobject], *text: str):
        self.shift_brace(obj)
        self.change_label(*text)
        return self

    def copy(self):
        copy_mobject = copy.copy(self)
        copy_mobject.brace = self.brace.copy()
        copy_mobject.label = self.label.copy()
        copy_mobject.set_submobjects([copy_mobject.brace, copy_mobject.label])

        return copy_mobject


class BraceText(BraceLabel):
    label_constructor: type = TexText


class LineBrace(Brace):
    def __init__(self, line: Line, direction=UP, **kwargs):
        angle = line.get_angle()
        line.rotate(-angle)
        super().__init__(line, direction, **kwargs)
        line.rotate(angle)
        self.rotate(angle, about_point=line.get_center())
