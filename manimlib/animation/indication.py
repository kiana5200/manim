# 从__future__导入annotations，支持在类型注解中使用尚未定义的类名
from __future__ import annotations

# 导入numpy库并简写为np，用于数值计算和数组操作
import numpy as np

# 从manimlib.animation模块导入动画相关类
from manimlib.animation.animation import Animation  # 动画基类
from manimlib.animation.composition import AnimationGroup  # 动画组，同时播放多个动画
from manimlib.animation.composition import Succession  # 连续动画，按顺序播放多个动画
from manimlib.animation.creation import ShowCreation  # 用于创建对象的动画（逐步绘制）
from manimlib.animation.creation import ShowPartial  # 用于部分显示对象的动画
from manimlib.animation.fading import FadeOut  # 淡出动画
from manimlib.animation.fading import FadeIn  # 淡入动画
from manimlib.animation.movement import Homotopy  # 同伦变换动画（连续变形）
from manimlib.animation.transform import Transform  # 变换动画（从一个对象变为另一个）

# 从manimlib.constants导入常量
from manimlib.constants import FRAME_X_RADIUS, FRAME_Y_RADIUS  # 帧的X和Y方向半径
from manimlib.constants import ORIGIN, RIGHT, UP  # 原点坐标、右方向向量、上方向向量
from manimlib.constants import SMALL_BUFF  # 小间距常量
from manimlib.constants import DEG  # 角度单位：度
from manimlib.constants import TAU  # 数学常量：2π（约6.283）
from manimlib.constants import GREY, YELLOW  # 颜色常量：灰色、黄色

# 从manimlib.mobject.geometry导入几何图形类
from manimlib.mobject.geometry import Circle  # 圆形类
from manimlib.mobject.geometry import Dot  # 点类
from manimlib.mobject.geometry import Line  # 线段类

# 从manimlib.mobject.shape_matchers导入形状匹配类
from manimlib.mobject.shape_matchers import SurroundingRectangle  # 环绕矩形（包围目标对象）
from manimlib.mobject.shape_matchers import Underline  # 下划线类

# 从manimlib.mobject.types导入矢量图形对象类
from manimlib.mobject.types.vectorized_mobject import VMobject  # 矢量图形对象基类
from manimlib.mobject.types.vectorized_mobject import VGroup  # 矢量图形对象组

# 从manimlib.utils导入工具函数
from manimlib.utils.bezier import interpolate  # 贝塞尔插值函数
from manimlib.utils.rate_functions import smooth  # 平滑速率函数（缓入缓出）
from manimlib.utils.rate_functions import squish_rate_func  # 压缩速率函数（调整时间范围）
from manimlib.utils.rate_functions import there_and_back  # 往返速率函数（去而复返）
from manimlib.utils.rate_functions import wiggle  # 摆动速率函数

# 从typing模块导入TYPE_CHECKING，用于条件类型检查
from typing import TYPE_CHECKING

# 仅在类型检查时执行以下导入（不影响运行时）
if TYPE_CHECKING:
    from typing import Callable  # 导入Callable类型，用于注解可调用对象
    from manimlib.typing import ManimColor  # 导入Manim颜色类型
    from manimlib.mobject.mobject import Mobject  # 导入所有可移动对象的基类


# 定义FocusOn类，继承自Transform（变换基类）
class FocusOn(Transform):
    # 构造方法，初始化FocusOn实例
    def __init__(
        self,
        focus_point: np.ndarray | Mobject,  # 聚焦点，可以是numpy数组（坐标）或Mobject对象
        opacity: float = 0.2,  # 透明度，默认值0.2
        color: ManimColor = GREY,  # 颜色，默认灰色
        run_time: float = 2,  # 动画运行时间，默认2秒
        remover: bool = True,  # 是否在动画结束后移除，默认True
        **kwargs  # 其他关键字参数，传递给父类
    ):
        self.focus_point = focus_point  # 保存聚焦点到实例变量
        self.opacity = opacity  # 保存透明度到实例变量
        self.color = color  # 保存颜色到实例变量
        # 调用父类构造方法，初始化一个空的VMobject作为基础
        # 实际的动画内容由create_target和create_starting_mobject处理
        super().__init__(VMobject(), run_time=run_time, remover=remover,** kwargs)

    # 创建动画的目标状态Mobject
    def create_target(self) -> Dot:
        little_dot = Dot(radius=0)  # 创建一个半径为0的点（不可见的点）
        little_dot.set_fill(self.color, opacity=self.opacity)  # 设置填充颜色和透明度
        # 添加更新器，使这个点始终跟随聚焦点移动
        little_dot.add_updater(lambda d: d.move_to(self.focus_point))
        return little_dot  # 返回目标状态的点

    # 创建动画的起始状态Mobject
    def create_starting_mobject(self) -> Dot:
        return Dot(
            radius=FRAME_X_RADIUS + FRAME_Y_RADIUS,  # 半径设为帧宽半径+帧高半径（覆盖整个画面）
            stroke_width=0,  # 描边宽度为0（无描边）
            fill_color=self.color,  # 填充颜色
            fill_opacity=0,  # 初始填充透明度为0（完全透明）
        )



# 定义Indicate类，继承自Transform（变换基类）
class Indicate(Transform):
    # 构造方法，初始化Indicate实例
    def __init__(
        self,
        mobject: Mobject,  # 要强调的Mobject对象
        scale_factor: float = 1.2,  # 缩放因子，默认1.2倍
        color: ManimColor = YELLOW,  # 强调颜色，默认黄色
        rate_func: Callable[[float], float] = there_and_back,  # 速率函数，默认there_and_back（去而复返）
        **kwargs  # 其他关键字参数，传递给父类
    ):
        self.scale_factor = scale_factor  # 保存缩放因子到实例变量
        self.color = color  # 保存颜色到实例变量
        # 调用父类构造方法，传入要变换的mobject和速率函数等参数
        super().__init__(mobject, rate_func=rate_func,** kwargs)

    # 创建动画的目标状态Mobject
    def create_target(self) -> Mobject:
        target = self.mobject.copy()  # 复制原始Mobject作为目标基础
        target.scale(self.scale_factor)  # 按缩放因子缩放目标
        target.set_color(self.color)  # 设置目标的颜色
        return target  # 返回目标状态的Mobject


# 定义Flash类，继承自AnimationGroup（动画组），用于创建闪烁动画效果
class Flash(AnimationGroup):
    # 构造方法，初始化Flash动画实例
    def __init__(
        self,
        point: np.ndarray | Mobject,  # 闪烁中心点，可以是坐标数组或Mobject对象
        color: ManimColor = YELLOW,  # 闪烁线条颜色，默认黄色
        line_length: float = 0.2,  # 每条闪烁线的长度，默认0.2
        num_lines: int = 12,  # 闪烁线的数量，默认12条
        flash_radius: float = 0.3,  # 闪烁效果的初始半径，默认0.3
        line_stroke_width: float = 3.0,  # 线条粗细，默认3.0
        run_time: float = 1.0,  # 动画运行时间，默认1秒
        **kwargs  # 其他关键字参数，传递给父类
    ):
        # 将参数保存为实例变量，供后续方法使用
        self.point = point
        self.color = color
        self.line_length = line_length
        self.num_lines = num_lines
        self.flash_radius = flash_radius
        self.line_stroke_width = line_stroke_width

        # 创建闪烁效果所需的所有线条
        self.lines = self.create_lines()
        # 为每条线创建对应的动画
        animations = self.create_line_anims()
        # 调用父类构造方法，将所有线条动画组合成一个动画组
        super().__init__(
            *animations,  # 解包动画列表，作为动画组的子动画
            group=self.lines,  # 指定动画组操作的Mobject组
            run_time=run_time,  # 动画总时长
            **kwargs,  # 传递其他关键字参数
        )

    # 创建组成闪烁效果的所有线条，返回一个VGroup（矢量对象组）
    def create_lines(self) -> VGroup:
        lines = VGroup()  # 初始化一个空的矢量对象组，用于存放所有线条
        # 按角度均匀分布创建线条（TAU是2π，即360度）
        for angle in np.arange(0, TAU, TAU / self.num_lines):
            # 创建一条从原点到右侧的线段，长度为line_length
            line = Line(ORIGIN, self.line_length * RIGHT)
            # 将线段向右偏移，使线段起点位于闪烁半径位置
            # 偏移距离 = 闪烁半径 - 线段长度（确保线段末端在半径边界）
            line.shift((self.flash_radius - self.line_length) * RIGHT)
            # 绕原点旋转线段到当前角度，实现放射状分布
            line.rotate(angle, about_point=ORIGIN)
            # 将线段添加到线条组中
            lines.add(line)
        # 设置所有线条的样式：颜色和粗细
        lines.set_stroke(
            color=self.color,
            width=self.line_stroke_width
        )
        # 添加更新器，使整个线条组始终跟随目标点移动
        lines.add_updater(lambda l: l.move_to(self.point))
        return lines  # 返回创建好的线条组

    # 为每条线创建显示后消失的动画，返回动画列表
    def create_line_anims(self) -> list[Animation]:
        # 对线条组中的每条线，创建"先显示再消失"的动画
        return [
            ShowCreationThenDestruction(line)
            for line in self.lines
        ]


# 定义CircleIndicate类，继承自Transform
# 用于实现“圆形指示器”动画，通过圆形缩放高亮目标物体
class CircleIndicate(Transform):
    def __init__(
        self,
        mobject: Mobject,               # 要被高亮的目标物体
        scale_factor: float = 1.2,      # 圆形的缩放倍数（相对于目标物体）
        rate_func: Callable[[float], float] = there_and_back,  # 动画速率函数（默认“去而复返”）
        stroke_color: ManimColor = YELLOW,  # 圆形边框颜色（默认黄色）
        stroke_width: float = 3.0,      # 圆形边框宽度（默认3.0）
        remover: bool = True,           # 动画结束后是否移除圆形（默认移除）
        **kwargs                        # 其他传递给父类的参数
    ):
        # 创建高亮用的圆形：设置边框颜色和宽度，无填充
        circle = Circle(stroke_color=stroke_color, stroke_width=stroke_width)
        circle.surround(mobject)  # 让圆形包围目标物体
        
        # 创建动画起始状态的圆形：复制最终圆形，将边框宽度设为0（初始不可见）
        pre_circle = circle.copy().set_stroke(width=0)
        pre_circle.scale(1 / scale_factor)  # 起始圆形缩小（为后续缩放做准备）
        
        # 调用父类Transform的初始化：从pre_circle变换到circle
        super().__init__(
            pre_circle, circle,
            rate_func=rate_func,
            remover=remover,** kwargs
        )


# 定义ShowPassingFlash类，继承自ShowPartial
# 用于实现“滚动闪现”动画，让物体按区域逐步显示（类似扫描效果）
class ShowPassingFlash(ShowPartial):
    def __init__(
        self,
        mobject: Mobject,               # 要添加闪现效果的物体
        time_width: float = 0.1,        # 闪现区域的宽度（占物体总长的比例，默认0.1）
        remover: bool = True,           # 动画结束后是否移除物体（默认移除）
        **kwargs                        # 其他传递给父类的参数
    ):
        self.time_width = time_width    # 存储闪现区域宽度
        # 调用父类ShowPartial的初始化：基于区域显示实现闪现
        super().__init__(
            mobject,
            remover=remover,** kwargs
        )

    # 计算不同动画进度（alpha）下，闪现区域的上下边界
    def get_bounds(self, alpha: float) -> tuple[float, float]:
        tw = self.time_width  # 闪现区域宽度
        # 上边界：随alpha从0逐步移动到1+tw（超出物体范围，确保完全闪过）
        upper = interpolate(0, 1 + tw, alpha)
        lower = upper - tw  # 下边界：上边界减去闪现宽度，形成移动的“窗口”
        
        # 边界裁剪：确保上下边界不超出0-1范围（避免显示异常）
        upper = min(upper, 1)
        lower = max(lower, 0)
        return (lower, upper)  # 返回（下边界，上边界）

    # 动画结束时的收尾处理
    def finish(self) -> None:
        super().finish()  # 调用父类收尾方法
        # 让所有子物体完全显示（避免动画结束后物体处于部分显示状态）
        for submob, start in self.get_all_families_zipped():
            submob.pointwise_become_partial(start, 0, 1)


# 定义VShowPassingFlash类，继承自Animation
# 用于实现“垂直滚动闪光”动画，通过边框宽度变化模拟闪光扫描效果（针对VMobject）
class VShowPassingFlash(Animation):
    def __init__(
        self,
        vmobject: VMobject,             # 要添加闪光效果的矢量物体（VMobject）
        time_width: float = 0.3,        # 闪光区域的时间宽度（控制闪光范围，默认0.3）
        taper_width: float = 0.05,      # 闪光边缘的渐变宽度（避免生硬边界，默认0.05）
        remover: bool = True,           # 动画结束后是否移除物体（默认移除）
        **kwargs                        # 其他传递给父类的参数
    ):
        self.time_width = time_width    # 存储闪光区域宽度
        self.taper_width = taper_width  # 存储边缘渐变宽度
        # 调用父类Animation的初始化：指定动画物体为vmobject
        super().__init__(vmobject, remover=remover, **kwargs)
        self.mobject = vmobject  # 显式存储物体（确保后续方法可调用）

    # 计算闪光边缘的渐变系数（让闪光两端逐渐变细）
    def taper_kernel(self, x):
        if x < self.taper_width:  # 左侧边缘：随x线性增加（从0到1）
            return x
        elif x > 1 - self.taper_width:  # 右侧边缘：随x线性减少（从1到0）
            return 1.0 - x
        return 1.0  # 中间区域：保持最大系数1.0

    # 动画开始前的初始化操作
    def begin(self) -> None:
        # 为每个子物体计算“带渐变的边框宽度”（存储哈希与宽度数组的映射）
        self.submob_to_widths = dict()
        for sm in self.mobject.get_family():  # 遍历物体的所有子部分
            widths = sm.get_stroke_widths()  # 获取子物体原始的边框宽度数组
            # 计算每个位置的渐变后宽度：原始宽度 × 渐变系数
            self.submob_to_widths[hash(sm)] = np.array([
                width * self.taper_kernel(x)
                for width, x in zip(widths, np.linspace(0, 1, len(widths)))
            ])
        super().begin()  # 调用父类begin方法（初始化起始状态）

    # 子物体的插值逻辑（核心：控制闪光的移动）
    def interpolate_submobject(
        self,
        submobject: VMobject,           # 当前要插值的子物体
        starting_sumobject: None,       # 起始状态（此处未使用，设为None）
        alpha: float                    # 动画进度（0→1）
    ) -> None:
        # 获取当前子物体的“带渐变边框宽度”数组
        widths = self.submob_to_widths[hash(submobject)]

        # 计算高斯分布参数（模拟闪光的“亮斑”效果）
        tw = self.time_width
        sigma = tw / 6  # 标准差：确保3倍标准差覆盖闪光宽度（99.7%的能量集中在闪光区）
        # 高斯分布的均值（随alpha移动：从-tw/2到1+tw/2，确保闪光完全扫过物体）
        mu = interpolate(-tw / 2, 1 + tw / 2, alpha)
        xs = np.linspace(0, 1, len(widths))  # 生成0→1的均匀坐标（对应子物体的每个点）
        zs = (xs - mu) / sigma  # 计算每个点到均值的“标准化距离”
        gaussian = np.exp(-0.5 * zs * zs)  # 高斯分布值（模拟闪光亮度）
        # 裁剪高斯值：距离均值超过3倍标准差的位置设为0（消除微弱尾迹）
        gaussian[abs(xs - mu) > 3 * sigma] = 0

        # 避免空数组报错：当计算出的宽度数组非空时，设置子物体边框宽度
        if len(widths * gaussian) != 0:
            submobject.set_stroke(width=widths * gaussian)

    # 动画结束时的收尾处理
    def finish(self) -> None:
        super().finish()  # 调用父类收尾方法
        # 让所有子物体恢复原始样式（避免残留闪光效果）
        for submob, start in self.get_all_families_zipped():
            submob.match_style(start)


# 定义 FlashAround 类，继承自 VShowPassingFlash（一个显示闪烁动画的基础类）
class FlashAround(VShowPassingFlash):
    # 构造方法，初始化动画的各种参数
    def __init__(
        self,
        mobject: Mobject,  # 要围绕其创建闪烁效果的图形对象
        time_width: float = 1.0,  # 动画持续的时间宽度
        taper_width: float = 0.0,  # 闪烁效果的渐变宽度
        stroke_width: float = 4.0,  # 边框线条宽度
        color: ManimColor = YELLOW,  # 闪烁效果的颜色，默认为黄色
        buff: float = SMALL_BUFF,  # 与目标对象的缓冲距离，使用预定义的小缓冲值
        n_inserted_curves: int = 100,  # 插入的曲线段数量，用于使动画更平滑
        **kwargs  # 其他传递给父类的关键字参数
    ):
        # 获取围绕目标对象的路径（默认是边框矩形）
        path = self.get_path(mobject, buff)
        # 如果目标对象是固定在帧中的，则路径也固定在帧中
        if mobject.is_fixed_in_frame():
            path.fix_in_frame()
        # 向路径中插入指定数量的曲线段，使动画更平滑
        path.insert_n_curves(n_inserted_curves)
        # 移除路径中的空曲线点，确保路径有效
        path.set_points(path.get_points_without_null_curves())
        # 设置路径的外观：颜色和线条宽度
        path.set_stroke(color, stroke_width)
        # 调用父类的构造方法，传递路径和其他参数
        super().__init__(path, time_width=time_width, taper_width=taper_width,** kwargs)

    # 定义获取路径的方法，返回围绕目标对象的矩形
    def get_path(self, mobject: Mobject, buff: float) -> SurroundingRectangle:
        # 创建并返回一个围绕目标对象的矩形，使用指定的缓冲距离
        return SurroundingRectangle(mobject, buff=buff)


# 定义 FlashUnder 类，继承自 FlashAround
class FlashUnder(FlashAround):
    # 重写 get_path 方法，返回下划线而不是矩形
    def get_path(self, mobject: Mobject, buff: float) -> Underline:
        # 创建并返回一个位于目标对象下方的下划线，缓冲距离和拉伸因子为1.0
        return Underline(mobject, buff=buff, stretch_factor=1.0)


# 定义 ShowCreationThenDestruction 类，继承自 ShowPassingFlash
class ShowCreationThenDestruction(ShowPassingFlash):
    # 构造方法，初始化动画参数
    def __init__(self, vmobject: VMobject, time_width: float = 2.0, **kwargs):
        # 调用父类的构造方法，传递向量图形对象和时间宽度等参数
        # 这个类本质上是对 ShowPassingFlash 的简单封装，使用默认的2.0秒动画时长
        super().__init__(vmobject, time_width=time_width,** kwargs)


# 定义 ShowCreationThenFadeOut 类，继承自 Succession（序列动画类）
class ShowCreationThenFadeOut(Succession):
    # 构造方法，初始化动画序列
    def __init__(self, mobject: Mobject, remover: bool = True, **kwargs):
        # 调用父类的构造方法，创建一个动画序列
        super().__init__(
            ShowCreation(mobject),  # 第一个动画：创建对象（逐渐显示）
            FadeOut(mobject),       # 第二个动画：淡出对象（逐渐消失）
            remover=remover,        # 是否在动画结束后移除对象，默认为True
            **kwargs                # 其他传递给父类的关键字参数
        )


# 定义一个围绕目标对象矩形的动画组基类，继承自 AnimationGroup
class AnimationOnSurroundingRectangle(AnimationGroup):
    # 类属性：矩形要使用的动画类型，默认为基础 Animation 类
    RectAnimationType: type = Animation

    # 构造方法，初始化围绕目标对象的矩形及相关动画
    def __init__(
        self,
        mobject: Mobject,  # 要围绕其创建矩形的图形对象
        stroke_width: float = 2.0,  # 矩形边框宽度，默认2.0
        stroke_color: ManimColor = YELLOW,  # 矩形边框颜色，默认黄色
        buff: float = SMALL_BUFF,  # 矩形与目标对象的缓冲距离，使用预定义小缓冲值
        **kwargs  # 传递给父类或动画的其他关键字参数
    ):
        # 创建围绕目标对象的矩形
        rect = SurroundingRectangle(
            mobject,  # 目标对象
            stroke_width=stroke_width,  # 边框宽度
            stroke_color=stroke_color,  # 边框颜色
            buff=buff,  # 缓冲距离
        )
        # 为矩形添加更新器：使矩形始终跟随目标对象移动
        # 当目标对象位置变化时，矩形会自动移动到相同位置
        rect.add_updater(lambda r: r.move_to(mobject))
        # 调用父类构造方法，将矩形的指定类型动画添加到动画组
        super().__init__(self.RectAnimationType(rect, **kwargs))


# 定义闪烁效果围绕目标对象的动画类，继承自 AnimationOnSurroundingRectangle
class ShowPassingFlashAround(AnimationOnSurroundingRectangle):
    # 重写矩形动画类型为 ShowPassingFlash（闪烁效果动画）
    RectAnimationType = ShowPassingFlash


# 定义先创建后销毁矩形的动画类，继承自 AnimationOnSurroundingRectangle
class ShowCreationThenDestructionAround(AnimationOnSurroundingRectangle):
    # 重写矩形动画类型为 ShowCreationThenDestruction（先创建后销毁动画）
    RectAnimationType = ShowCreationThenDestruction


# 定义先创建后淡出矩形的动画类，继承自 AnimationOnSurroundingRectangle
class ShowCreationThenFadeAround(AnimationOnSurroundingRectangle):
    # 重写矩形动画类型为 ShowCreationThenFadeOut（先创建后淡出动画）
    RectAnimationType = ShowCreationThenFadeOut


class ApplyWave(Homotopy):
    def __init__(
        self,
        mobject: Mobject,
        direction: np.ndarray = UP,
        amplitude: float = 0.2,
        run_time: float = 1.0,
        **kwargs
    ):

        left_x = mobject.get_left()[0]
        right_x = mobject.get_right()[0]
        vect = amplitude * direction

        def homotopy(x, y, z, t):
            alpha = (x - left_x) / (right_x - left_x)
            power = np.exp(2.0 * (alpha - 0.5))
            nudge = there_and_back(t**power)
            return np.array([x, y, z]) + nudge * vect

        super().__init__(homotopy, mobject, **kwargs)


class WiggleOutThenIn(Animation):
    def __init__(
        self,
        mobject: Mobject,
        scale_value: float = 1.1,
        rotation_angle: float = 0.01 * TAU,
        n_wiggles: int = 6,
        scale_about_point: np.ndarray | None = None,
        rotate_about_point: np.ndarray | None = None,
        run_time: float = 2,
        **kwargs
    ):
        self.scale_value = scale_value
        self.rotation_angle = rotation_angle
        self.n_wiggles = n_wiggles
        self.scale_about_point = scale_about_point
        self.rotate_about_point = rotate_about_point
        super().__init__(mobject, run_time=run_time, **kwargs)

    def get_scale_about_point(self) -> np.ndarray:
        return self.scale_about_point or self.mobject.get_center()

    def get_rotate_about_point(self) -> np.ndarray:
        return self.rotate_about_point or self.mobject.get_center()

    def interpolate_submobject(
        self,
        submobject: Mobject,
        starting_sumobject: Mobject,
        alpha: float
    ) -> None:
        submobject.match_points(starting_sumobject)
        submobject.scale(
            interpolate(1, self.scale_value, there_and_back(alpha)),
            about_point=self.get_scale_about_point()
        )
        submobject.rotate(
            wiggle(alpha, self.n_wiggles) * self.rotation_angle,
            about_point=self.get_rotate_about_point()
        )


class TurnInsideOut(Transform):
    def __init__(self, mobject: Mobject, path_arc: float = 90 * DEG, **kwargs):
        super().__init__(mobject, path_arc=path_arc, **kwargs)

    def create_target(self) -> Mobject:
        result = self.mobject.copy().reverse_points()
        if isinstance(result, VMobject):
            result.refresh_triangulation()
        return result


class FlashyFadeIn(AnimationGroup):
    def __init__(self,
        vmobject: VMobject,
        stroke_width: float = 2.0,
        fade_lag: float = 0.0,
        time_width: float = 1.0,
        **kwargs
    ):
        outline = vmobject.copy()
        outline.set_fill(opacity=0)
        outline.set_stroke(width=stroke_width, opacity=1)

        rate_func = kwargs.get("rate_func", smooth)
        super().__init__(
            FadeIn(vmobject, rate_func=squish_rate_func(rate_func, fade_lag, 1)),
            VShowPassingFlash(outline, time_width=time_width),
            **kwargs
        )
