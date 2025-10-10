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

    # 将另一个 Mobject (mob) 放置在当前对象（self）的尖端（tip）位置
    def put_at_tip(
        self,
        mob: Mobject,
        use_next_to: bool = True,
        **kwargs
    ):
        # 如果开关为 True，则执行此代码块
        if use_next_to:
            # 调用 mob 的 next_to 方法来进行放置
            mob.next_to(
                # 参数 1: 参照点。
                # self.get_tip() 返回当前对象（如箭头、花括号）尖端的坐标点 (np.ndarray)。
                self.get_tip(),
                # 参数 2: 方向。
                # self.get_direction() 返回一个表示当前对象指向的方向向量 (Vect3)。
                # np.round() 将方向向量的每个分量四舍五入到最近的整数。这是一个安全措施，
                # 确保方向向量是像 RIGHT (1, 0, 0)、UP (0, 1, 0) 这样的标准单位向量，
                # 避免因浮点精度问题导致 `next_to` 方法行为异常。
                np.round(self.get_direction()),
                # 参数 3: 其他关键字参数。
                # 例如，可以传入 buff=0.5 来设置 mob 和尖端之间的距离。
                **kwargs
            )
        else:
            # 如果 use_next_to 为 False，则执行此代码块。
            # 这种方式手动实现了类似 next_to 的功能，但提供了更底层的控制。

            # 1. 先将 mob 的中心移动到尖端的精确位置。
            # 此时，mob 和尖端会完全重叠。
            mob.move_to(self.get_tip())
            # 2. 计算需要移动的总距离。
            # - 从 kwargs 中获取 'buff' 参数，如果没有提供，则使用 Manim 的默认缓冲距离。
            buff = kwargs.get("buff", DEFAULT_MOBJECT_TO_MOBJECT_BUFF)
            # - 总距离 = mob 自身宽度的一半 + 缓冲距离。
            #   这样可以确保 mob 的边缘与尖端之间有 'buff' 的空隙。
            shift_distance = mob.get_width() / 2.0 + buff
            # 3. 执行移动。
            # - 将 mob 沿着 self 的方向向量移动计算出的总距离。
            # - 这会将 mob 从尖端位置“推开”，使其刚好贴在尖端的前方。
            mob.shift(self.get_direction() * shift_distance)

        # 4. 返回 self 以支持方法链式调用。    
        return self

    # 创建一个 Text 对象，并将其自动放置在当前对象（self）的尖端。
    # 这是一个非常便捷的“一站式”方法，用于给箭头、花括号等对象添加标签。
    def get_text(self, text: str, **kwargs) -> Text:
        # 1. 从 kwargs 中提取并移除 'buff' 参数。
        # - kwargs.pop("buff", SMALL_BUFF) 的作用是：
        #   a. 尝试从 kwargs 字典中获取 'buff' 的值。
        #   b. 如果 'buff' 存在，则返回它的值，并将 'buff' 键值对从 kwargs 中删除。
        #   c. 如果 'buff' 不存在，则返回默认值 SMALL_BUFF，而 kwargs 保持不变。
        # - 这样做是为了将 'buff' 参数专门用于后续的 put_at_tip 调用，
        #   而不会被意外地传递给 Text 的构造函数（因为 Text 不接受 'buff' 参数）。
        buff = kwargs.pop("buff", SMALL_BUFF)
        # 2. 创建 Text 对象。
        # - 使用传入的 text 字符串和剩余的 **kwargs 创建一个新的 Text mobject。
        text_mob = Text(text, **kwargs)
        # 3. 将 Text 对象放置在尖端。
        # - 调用之前定义的 put_at_tip 方法，将新创建的 text_mob 放置在 self 的尖端。
        # - 显式地将刚刚提取的 buff 值传递给 put_at_tip。
        self.put_at_tip(text_mob, buff=buff)
        # 4. 返回创建的 Text 对象。
        # - 将定位好的 text_mob 返回给调用者，这样调用者就可以对它进行后续操作，
        #   比如添加动画（如 FadeIn）或进一步调整。
        return text_mob

    # 创建一个 Tex 对象，并将其自动放置在当前对象（self）的尖端。
    # 这是 `get_text` 方法的 LaTeX 版本，用于在尖端添加格式化的数学公式或文本。
    def get_tex(self, *tex: str, **kwargs) -> Tex:
        buff = kwargs.pop("buff", SMALL_BUFF)
        tex_mob = Tex(*tex, **kwargs)
        self.put_at_tip(tex_mob, buff=buff)
        return tex_mob

    # 获取当前对象（如花括号、箭头）尖端的坐标。
    # 该方法通过预存的尖端点索引，从对象的所有顶点中精准提取尖端位置。
    # np.ndarray: 尖端的三维坐标（格式为 [x, y, z]），数据类型为 NumPy 数组。
    def get_tip(self) -> np.ndarray:
        # Very specific to the LaTeX representation
        # of a brace, but it's the only way I can think
        # of to get the tip regardless of orientation.
        return self.get_all_points()[self.tip_point_index]

    # 获取当前对象（如花括号、箭头）的“指向方向”，返回一个标准化的单位向量。
    # np.ndarray: 代表对象指向的三维单位向量（格式为 [x, y, z]），数据类型为 NumPy 数组。
    def get_direction(self) -> np.ndarray:
        vect = self.get_tip() - self.get_center()
        return vect / get_norm(vect)

# 定义一个名为 BraceLabel 的类，它继承自 VMobject。
# VMobject (Vectorized Mobject) 是 Manim 中所有可渲染对象的基础，
class BraceLabel(VMobject):
    # 类属性，指定用于创建标签的构造函数。默认为 Tex，
    # 这意味着默认情况下标签是 LaTeX 公式。
    # 用户可以通过继承或直接修改这个属性来使用 Text 或其他 Mobject 作为标签。
    label_constructor: type = Tex

    # 初始化一个 BraceLabel 对象。这个对象会自动创建一个花括号（Brace）和一个标签（Tex/Text），
    # 并将它们组合成一个单一的、可操作的对象。
    def __init__(
        self,
        obj: VMobject | list[VMobject],
        text: str | Iterable[str],
        brace_direction: np.ndarray = DOWN,
        label_scale: float = 1.0,
        label_buff: float = DEFAULT_MOBJECT_TO_MOBJECT_BUFF,
        **kwargs
    ) -> None:
        # 调用父类 VMobject 的构造函数进行初始化。
        super().__init__(**kwargs)
        # 将传入的参数存储为实例属性，以便在后续的 setup 方法中使用。
        self.brace_direction = brace_direction
        self.label_scale = label_scale
        self.label_buff = label_buff

        # 检查传入的 'obj' 是否是一个列表或其他可迭代对象。
        if isinstance(obj, list):
            # VGroup (Vectorized Group) 是 Manim 中用于管理一组 Mobject 的容器，可以像操作单个对象一样操作整个组。
            # 这样做的好处是，Brace 只需要附着到这个 VGroup 上，它会自动包围所有子对象的边界框。
            obj = VGroup(*obj)
        # 1. 创建花括号 (Brace)
        # 使用之前处理过的 'obj' 和指定的方向 'brace_direction' 来创建一个 Brace 对象。
        # **kwargs 会传递给 Brace 的构造函数，用于设置花括号的样式（如颜色）。
        self.brace = Brace(obj, brace_direction, **kwargs)

        # 2. 创建标签 (Label)
        self.label = self.label_constructor(*listify(text), **kwargs)
        # 3. 缩放标签
        self.label.scale(self.label_scale)
        
        # 4. 将标签定位在花括号尖端
        # 调用花括号对象的 put_at_tip 方法，将标签自动放置在其尖端。
        # 使用 label_buff 参数来控制标签与尖端之间的距离。
        self.brace.put_at_tip(self.label, buff=self.label_buff)
        # 5. 将花括号和标签组合成一个整体
        # 这是最关键的一步。调用 set_submobjects 方法，将 self.brace 和 self.label 设为当前 BraceLabel 对象的子对象。
        self.set_submobjects([self.brace, self.label])

    # 创建一个动画组，用于同时或按顺序展示花括号和标签的出现过程。
    def creation_anim(
        self,
        label_anim: Animation = FadeIn,
        brace_anim: Animation = GrowFromCenter
    ) -> AnimationGroup:
        return AnimationGroup(brace_anim(self.brace), label_anim(self.label))

    # 将花括号移动到一个新的对象或一组对象上，同时保持标签的位置和指向不变。
    def shift_brace(self, obj: VMobject | list[VMobject], **kwargs):
        if isinstance(obj, list):
            obj = VMobject(*obj)
        # 根据新的目标对象 'obj' 和原来存储的方向 'self.brace_direction'，创建一个全新的 Brace 对象。
        self.brace = Brace(obj, self.brace_direction, **kwargs)
         # 调用新花括号的 put_at_tip 方法，将现有的标签 self.label 重新放置在新花括号的尖端。
        self.brace.put_at_tip(self.label)
        # 将 self.submobjects 列表中的第一个元素（即旧的花括号）替换为新创建的花括号。
        self.submobjects[0] = self.brace
        # 4. 返回 self 以支持链式调用
        return self

    # 动态地改变标签的文本内容，同时保持其相对于花括号尖端的位置不变。
    def change_label(self, *text: str, **kwargs):
        # 使用存储的 label_constructor (默认为 Tex) 和新的 text 内容，创建一个全新的 label 对象。
        # **kwargs 允许在改变文本的同时，也改变标签的样式。
        self.label = self.label_constructor(*text, **kwargs)
        # 检查在初始化时是否设置了非默认的缩放比例。
        if self.label_scale != 1:
            self.label.scale(self.label_scale)

        # 重新放置在花括号的尖端，保持了视觉上的连贯性。
        self.brace.put_at_tip(self.label)
        # 将 self.submobjects 列表中的第二个元素（即旧的标签）替换为新创建的标签。
        self.submobjects[1] = self.label
        return self

    # 同时改变花括号的附着对象和标签的文本内容。这是一个便捷的组合方法。
    def change_brace_label(self, obj: VMobject | list[VMobject], *text: str):
        # 调用已有的 shift_brace 方法，将花括号移动到新的对象 obj 上。
        self.shift_brace(obj)
        # 调用已有的 change_label 方法，用新的 text 内容替换旧的标签。
        self.change_label(*text)
        return self

    # 创建一个 `BraceLabel` 对象的深拷贝。
    def copy(self):
        copy_mobject = copy.copy(self)
        copy_mobject.brace = self.brace.copy()
        copy_mobject.label = self.label.copy()
        # 更新副本的子对象列表，将新拷贝的 brace 和 label 设置为新创建的 copy_mobject 的子对象。
        copy_mobject.set_submobjects([copy_mobject.brace, copy_mobject.label])

        return copy_mobject

# 定义一个名为 BraceText 的类，它继承自 BraceLabel；唯一目的是改变默认的标签构造函数。
class BraceText(BraceLabel):
    label_constructor: type = TexText

# 定义一个名为 LineBrace 的类，它继承自 Brace。
# 这个类是 Brace 的一个特化版本，专门用于附着在 Line 对象上。
class LineBrace(Brace):
    def __init__(self, line: Line, direction=UP, **kwargs):
        # 1. 计算线条的角度
        angle = line.get_angle()
        # 2. 临时旋转线条使其水平
        line.rotate(-angle)
        # 3. 调用父类 Brace 的构造函数
        super().__init__(line, direction, **kwargs)
        # 4. 恢复线条的原始角度
        line.rotate(angle)
        # 5. 旋转花括号以匹配线条角度
        self.rotate(angle, about_point=line.get_center())
