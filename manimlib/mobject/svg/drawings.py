# 从 __future__ 模块导入 annotations 特性。
# 这允许在类型注解中直接使用尚未完全定义的类名或函数名，
# 而无需将其放在字符串中，从而使代码更具可读性。
# 例如，可以写 `def get_item(self) -> MyClass:` 而不是 `def get_item(self) -> 'MyClass':`。
from __future__ import annotations

# 导入 NumPy 库，并将其别名为 np。NumPy 是 Python 进行科学计算的核心库，
# 提供了高性能的多维数组对象（ndarray）和大量的数组操作函数。
import numpy as np

# 导入 itertools 模块，并将其别名为 it。这个模块提供了一系列用于创建
# 高效迭代器的工具，常用于循环、排列组合、笛卡尔积等场景。
import itertools as it

# 导入 random 模块。这个模块提供了各种生成伪随机数的函数，
# 用于实现随机选择、洗牌、生成随机数等功能。
import random

from manimlib.animation.composition import AnimationGroup
from manimlib.animation.rotation import Rotating
from manimlib.constants import BLACK
from manimlib.constants import BLUE_A
from manimlib.constants import BLUE_B
from manimlib.constants import BLUE_C
from manimlib.constants import BLUE_D
from manimlib.constants import DOWN
from manimlib.constants import DOWN
from manimlib.constants import FRAME_WIDTH
from manimlib.constants import GREEN
from manimlib.constants import GREEN_SCREEN
from manimlib.constants import GREEN_E
from manimlib.constants import GREY
from manimlib.constants import GREY_A
from manimlib.constants import GREY_B
from manimlib.constants import GREY_E
from manimlib.constants import LEFT
from manimlib.constants import LEFT
from manimlib.constants import MED_LARGE_BUFF
from manimlib.constants import MED_SMALL_BUFF
from manimlib.constants import ORIGIN
from manimlib.constants import OUT
from manimlib.constants import PI
from manimlib.constants import RED
from manimlib.constants import RED_E
from manimlib.constants import RIGHT
from manimlib.constants import SMALL_BUFF
from manimlib.constants import SMALL_BUFF
from manimlib.constants import UP
from manimlib.constants import UL
from manimlib.constants import UR
from manimlib.constants import DL
from manimlib.constants import DR
from manimlib.constants import WHITE
from manimlib.constants import YELLOW
from manimlib.constants import TAU
from manimlib.mobject.boolean_ops import Difference
from manimlib.mobject.boolean_ops import Union
from manimlib.mobject.geometry import Arc
from manimlib.mobject.geometry import Circle
from manimlib.mobject.geometry import Dot
from manimlib.mobject.geometry import Line
from manimlib.mobject.geometry import Polygon
from manimlib.mobject.geometry import Rectangle
from manimlib.mobject.geometry import Square
from manimlib.mobject.geometry import AnnularSector
from manimlib.mobject.numbers import Integer
from manimlib.mobject.shape_matchers import SurroundingRectangle
from manimlib.mobject.svg.svg_mobject import SVGMobject
from manimlib.mobject.svg.special_tex import TexTextFromPresetString
from manimlib.mobject.three_dimensions import Prismify
from manimlib.mobject.three_dimensions import VCube
from manimlib.mobject.types.vectorized_mobject import VGroup
from manimlib.mobject.types.vectorized_mobject import VMobject
from manimlib.mobject.svg.text_mobject import Text
from manimlib.utils.bezier import interpolate
from manimlib.utils.iterables import adjacent_pairs
from manimlib.utils.rate_functions import linear
from manimlib.utils.space_ops import angle_of_vector
from manimlib.utils.space_ops import compass_directions
from manimlib.utils.space_ops import get_norm
from manimlib.utils.space_ops import midpoint
from manimlib.utils.space_ops import rotate_vector

# 导入 TYPE_CHECKING，用于条件导入
from typing import TYPE_CHECKING

# 如果在进行类型检查，则导入类型提示
if TYPE_CHECKING:
    from typing import Tuple, Sequence, Callable
    from manimlib.typing import ManimColor, Vect3

# 定义一个 Checkmark 类，继承自预设字符串的 TexText
class Checkmark(TexTextFromPresetString):
    # LaTeX 代码，用于显示一个绿色的勾选符号
    tex: str = R"\ding{51}"
    # 默认颜色为绿色
    default_color: ManimColor = GREEN

# 定义一个 Exmark 类，继承自预设字符串的 TexText
class Exmark(TexTextFromPresetString):
    # LaTeX 代码，用于显示一个红色的叉号
    tex: str = R"\ding{55}"
    # 默认颜色为红色
    default_color: ManimColor = RED


# 定义一个 Lightbulb 类，它继承自 SVGMobject，用于显示 SVG 图像。
class Lightbulb(SVGMobject):
    # 指定要加载的 SVG 文件的名称（不含 .svg 后缀）。
    file_name = "lightbulb"

    def __init__(
        self,
        height: float = 1.0,
        color: ManimColor = YELLOW,
        stroke_width: float = 3.0,
        fill_opacity: float = 0.0,
        **kwargs
    ):
        # 调用父类 SVGMobject 的构造函数，传递所有参数来初始化 SVG 对象。
        super().__init__(
            height=height,
            color=color,
            stroke_width=stroke_width,
            fill_opacity=fill_opacity,
            **kwargs
        )
        # 增加 SVG 路径的曲线分段数，使其在进行变形动画时更加平滑。
        self.insert_n_curves(25)


# 定义一个 Speedometer 类，继承自 VMobject，使其成为一个可独立操作的组合对象。
class Speedometer(VMobject):
    def __init__(
        self,
        arc_angle: float = 4 * PI / 3,
        num_ticks: int = 8,
        tick_length: float = 0.2,
        needle_width: float = 0.1,
        needle_height: float = 0.8,
        needle_color: ManimColor = YELLOW,
        **kwargs,
    ):
        # 调用父类构造函数，完成基础初始化。
        super().__init__(**kwargs)

        # 存储所有参数为实例属性，以便后续使用和修改。
        self.arc_angle = arc_angle
        self.num_ticks = num_ticks
        self.tick_length = tick_length
        self.needle_width = needle_width
        self.needle_height = needle_height
        self.needle_color = needle_color

        # 1. 计算弧形刻度盘的几何参数
        # 速度表通常是一个从左下方到右下方的弧形。
        # PI/2 是向上的垂直方向，以此为中心，向左右各延伸 arc_angle/2，得到弧形的起止点。
        start_angle = PI / 2 + arc_angle / 2  # 左侧起始点（例如，210度）
        end_angle = PI / 2 - arc_angle / 2    # 右侧结束点（例如，-30度）
        
        # 2. 创建并添加弧形刻度盘
        # 创建一个从 start_angle 开始，逆时针旋转 arc_angle 角度的弧形。
        self.arc = Arc(
            start_angle=start_angle,
            angle=-self.arc_angle  # 负号表示逆时针绘制
        )
        self.add(self.arc)

        # 3. 创建并添加刻度线和数字标签
        # 使用 np.linspace 在起始和结束角度之间均匀生成 num_ticks 个角度值。
        tick_angle_range = np.linspace(start_angle, end_angle, num_ticks)
        for index, angle in enumerate(tick_angle_range):
            # 计算该角度对应的单位向量，用于定位刻度和标签。
            vect = rotate_vector(RIGHT, angle)
            # 创建刻度线：从圆上的一点 (1 - tick_length) * vect 指向圆周 vect。
            tick = Line((1 - tick_length) * vect, vect)
            # 创建数字标签：值为 10 * index (如 0, 10, 20...)。
            label = Integer(10 * index)
            # 调整标签大小，使其与刻度线长度协调。
            label.set_height(tick_length)
            # 将标签移动到刻度线外侧一点的位置。
            label.shift((1 + tick_length) * vect)
            # 将刻度线和标签添加到速度表组件中。
            self.add(tick, label)

        # 4. 创建并添加指针
        # 创建一个等腰三角形作为指针。
        needle = Polygon(
            LEFT, UP, RIGHT,
            stroke_width=0,      # 无描边
            fill_opacity=1,      # 完全填充
            fill_color=self.needle_color
        )
        # 按指定尺寸拉伸三角形，使其成为细长的指针形状。
        needle.stretch_to_fit_width(needle_width)
        needle.stretch_to_fit_height(needle_height)
        # 将指针旋转到起始角度位置（即速度为0的位置）。
        # 三角形默认尖端朝上(UP)，需要旋转到 start_angle 方向。
        # 减去 PI/2 是为了将其从朝上的方向旋转到与 start_angle 对齐。
        needle.rotate(start_angle - np.pi / 2, about_point=ORIGIN)
        self.add(needle)
        # 将指针保存为实例属性，以便后续通过动画旋转它。
        self.needle = needle

        # 5. 记录中心点偏移量
        # 计算整个速度表的几何中心，并存储起来。
        self.center_offset = self.get_center()


    # 计算并返回速度表的“逻辑”中心。
    # 这个方法重写了父类的 `get_center`，目的是将速度表的中心定义为其弧形的圆心，
    # 而不是整个图形（包括标签）的几何中心。
    def get_center(self):
        result = VMobject.get_center(self) # 1. 调用父类方法获取几何中心
        # 2. 如果存在偏移量，则进行修正
        if hasattr(self, "center_offset"):
            # 通过减去这个偏移量，将几何中心“修正”回弧形的圆心位置。
            result -= self.center_offset
        return result

    # 获取指针尖端的坐标。
    def get_needle_tip(self):
        # 返回尖端对应的锚点
        # 根据定义，UP 方向的点是指针的尖端。在 Polygon(LEFT, UP, RIGHT) 中，
        # UP 是第二个元素，所以索引为 1。
        return self.needle.get_anchors()[1]

    # 计算并返回指针当前指向的角度。
    def get_needle_angle(self):
        # 计算该向量的角度
        # Manim 的 angle_of_vector 函数可以计算一个向量与正 x 轴之间的夹角，返回值为弧度。
        return angle_of_vector(
            self.get_needle_tip() - self.get_center()
        )

    # 旋转指针，用于以速度表的圆心为支点来旋转指针。
    def rotate_needle(self, angle):
        # 执行旋转；调用指针对象的 rotate 方法。
        # - angle: 旋转的角度。
        # - about_point=self.arc.get_arc_center(): 这是关键。它指定了旋转的支点是弧形的圆心，
        #   而不是指针自身的中心或坐标系原点。这确保了指针能像真实速度表那样绕中心转动。
        self.needle.rotate(angle, about_point=self.arc.get_arc_center())
        return self

    # 将速度表的指针移动到与给定速度值相对应的位置。
    # 通过计算速度值在整个量程中的比例，将其转换为指针应指向的角度，
    # 然后调用 `rotate_needle` 方法来完成移动。
    def move_needle_to_velocity(self, velocity):
        max_velocity = 10 * (self.num_ticks - 1) # 计算速度表的最大量程
        proportion = float(velocity) / max_velocity # 计算目标速度占最大量程的比例
        # 计算指针应指向的目标角度
        # - start_angle: 速度为 0 时指针的初始角度（最左侧）。
        # - self.arc_angle * proportion: 指针需要从起点向终点移动的角度距离。
        # - target_angle: 从起点减去移动的角度距离，得到指针最终应指向的角度。
        start_angle = np.pi / 2 + self.arc_angle / 2
        target_angle = start_angle - self.arc_angle * proportion
        # 计算需要旋转的角度差并旋转指针
        # - self.get_needle_angle(): 获取指针当前的角度。
        # - target_angle - self.get_needle_angle(): 计算出从当前角度到目标角度需要旋转的差值。
        # - self.rotate_needle(...): 调用 rotate_needle 方法，将指针旋转这个角度差，使其到达目标位置。
        self.rotate_needle(target_angle - self.get_needle_angle())
        return self


# 定义一个 Laptop 类，继承自 VGroup，使其成为一个可以整体操作的组合对象。
class Laptop(VGroup):
    def __init__(
        self,
        width: float = 3,
        body_dimensions: Tuple[float, float, float] = (4.0, 3.0, 0.05),
        screen_thickness: float = 0.01,
        keyboard_width_to_body_width: float = 0.9,
        keyboard_height_to_body_height: float = 0.5,
        screen_width_to_screen_plate_width: float = 0.9,
        key_color_kwargs: dict = dict(
            stroke_width=0,
            fill_color=BLACK,
            fill_opacity=1,
        ),
        fill_opacity: float = 1.0,
        stroke_width: float = 0.0,
        body_color: ManimColor = GREY_B,
        shaded_body_color: ManimColor = GREY,
        open_angle: float = np.pi / 4,
        **kwargs
    ):
        # 调用父类 VGroup 的构造函数进行初始化。
        super().__init__(**kwargs)

        # 1. 创建笔记本电脑的底座 (body)
        # 使用一个立方体 (VCube) 作为基础，通过拉伸得到所需的长宽高。
        body = VCube(side_length=1)
        for dim, scale_factor in enumerate(body_dimensions):
            body.stretch(scale_factor, dim=dim)
        body.set_width(width)
        # 设置底座的颜色，并为顶部表面（放置键盘的地方）设置不同的颜色。
        body.set_fill(shaded_body_color, opacity=1)
        body.sort(lambda p: p[2])
        body[-1].set_fill(body_color)

        # 2. 创建键盘 (keyboard)
        # 通过两层循环创建 4 行按键，每行按键数量略有不同，形成类似真实键盘的交错布局。
        keyboard = VGroup(*[
            VGroup(*[
                Square(**key_color_kwargs)
                for x in range(12 - y % 2)
            ]).arrange(RIGHT, buff=SMALL_BUFF)
            for y in range(4)
        ]).arrange(DOWN, buff=MED_SMALL_BUFF)
        # 将创建好的按键组拉伸到合适的尺寸，并定位到底座上。
        keyboard.stretch_to_fit_width(keyboard_width_to_body_width * body.get_width())
        keyboard.stretch_to_fit_height(keyboard_height_to_body_height * body.get_height())
        keyboard.next_to(body, OUT, buff=0.1 * SMALL_BUFF)
        keyboard.shift(MED_SMALL_BUFF * UP)
        body.add(keyboard)

        # 3. 创建屏幕部分 (screen_plate 和 screen)
        # 复制底座并将其拉伸成一个薄片，作为屏幕的边框（screen_plate）。
        screen_plate = body.copy()
        screen_plate.stretch(screen_thickness / body_dimensions[2], dim=2)
        # 创建一个黑色矩形作为屏幕显示区域 (screen)，并将其定位到边框上。
        screen = Rectangle(stroke_width=0, fill_color=BLACK, fill_opacity=1)
        screen.replace(screen_plate, stretch=True)
        screen.scale(screen_width_to_screen_plate_width)
        screen.next_to(screen_plate, OUT, buff=0.1 * SMALL_BUFF)
        screen_plate.add(screen)

        # 4. 将屏幕与底座组合并旋转
        # 将屏幕边框放置在底座上方，并以其底部边缘为轴，按指定角度 (open_angle) 旋转，模拟打开的状态。
        screen_plate.next_to(body, UP, buff=0)
        screen_plate.rotate(open_angle, RIGHT, about_point=screen_plate.get_bottom())
        self.screen_plate = screen_plate
        self.screen = screen

        # 5. 创建连接轴 (axis)
        # 创建一条黑色的细线，模拟连接屏幕和底座的铰链轴，并将其添加到组合中。
        axis = Line(
            body.get_corner(UP + LEFT + OUT),
            body.get_corner(UP + RIGHT + OUT),
            color=BLACK,
            stroke_width=2
        )
        self.axis = axis

        # 6. 组合所有部分
        # 将底座、屏幕组件和铰链轴添加到 Laptop 对象中，完成最终的组合。
        self.add(body, screen_plate, axis)


# 定义一个 VideoIcon 类，继承自 SVGMobject，用于显示一个视频图标。
class VideoIcon(SVGMobject):
    # 指定要加载的 SVG 文件的名称（不含 .svg 后缀）。
    file_name: str = "video_icon"

    def __init__(
        self,
        width: float = 1.2,
        color=BLUE_A,
        **kwargs
    ):
        # 调用父类 SVGMobject 的构造函数，设置颜色等属性。
        super().__init__(color=color, **kwargs)
        # 将加载的 SVG 图标设置为指定的宽度。
        self.set_width(width)


# 定义一个 VideoSeries 类，继承自 VGroup，用于创建一排视频图标。
class VideoSeries(VGroup):
    def __init__(
        self,
        num_videos: int = 11,
        gradient_colors: Sequence[ManimColor] = [BLUE_B, BLUE_D],
        width: float = FRAME_WIDTH - MED_LARGE_BUFF,
        **kwargs
    ):
        # 1. 调用父类 VGroup 的构造函数，并在其中一次性创建并添加多个 VideoIcon。
        # 使用生成器表达式 `(VideoIcon() for x in range(num_videos))` 来创建 num_videos 个图标实例。
        super().__init__(
            *(VideoIcon() for x in range(num_videos)),
            **kwargs
        )
        
        # 2. 将这一排图标从左到右排列。
        self.arrange(RIGHT)
        
        # 3. 将整个图标组的总宽度设置为指定值。
        # 这会自动缩放所有图标，使它们的总宽度符合要求。
        self.set_width(width)
        
        # 4. 为整个图标组应用渐变色。
        # 从左到右，图标颜色会从 gradient_colors 的第一个颜色平滑过渡到最后一个颜色。
        self.set_color_by_gradient(*gradient_colors)


# 定义一个 Clock 类，继承自 VGroup，使其成为一个可以整体操作的组合对象。
class Clock(VGroup):
    def __init__(
        self,
        stroke_color: ManimColor = WHITE,
        stroke_width: float = 3.0,
        hour_hand_height: float = 0.3,
        minute_hand_height: float = 0.6,
        tick_length: float = 0.1,
        **kwargs,
    ):
        # 1. 准备通用样式
        # 将颜色和线宽等通用样式参数打包到一个字典中，方便后续使用。
        style = dict(stroke_color=stroke_color, stroke_width=stroke_width)
        
        # 2. 创建时钟外圈
        # 创建一个圆形作为时钟的表盘。
        circle = Circle(**style)
        
        # 3. 创建时钟刻度
        # compass_directions(12, UP) 会在圆周上生成 12 个均匀分布的点（方向向量）。
        ticks = []
        for x, point in enumerate(compass_directions(12, UP)):
            length = tick_length
            # 让每 3 个刻度（即 3, 6, 9, 12 点方向）的长度加倍，作为小时刻度。
            if x % 3 == 0:
                length *= 2
            # 从圆周上的点向圆心方向画一条线，作为刻度。
            ticks.append(Line(point, (1 - length) * point, **style))
        
        # 4. 创建时针和分针
        # 创建两条线作为指针，初始时都指向 12 点方向。
        self.hour_hand = Line(ORIGIN, hour_hand_height * UP, **style)
        self.minute_hand = Line(ORIGIN, minute_hand_height * UP, **style)

        # 5. 组合所有部分
        # 调用父类 VGroup 的构造函数，将外圈、指针和所有刻度组合成一个整体。
        super().__init__(
            circle, self.hour_hand, self.minute_hand,
            *ticks
        )


# 定义一个 ClockPassesTime 类，继承自 Animation_group，用于制作时钟指针转动的动画。
class ClockPassesTime(AnimationGroup):
    def __init__(
        self,
        clock: Clock,
        run_time: float = 5.0,
        hours_passed: float = 12.0,
        rate_func: Callable[[float], float] = linear,
        **kwargs
    ):
        # 1. 准备旋转的通用参数
        # 创建一个字典，存储时针和分针共同的旋转参数：
        # - axis=OUT: 绕着垂直于屏幕向外的轴旋转（即顺时针/逆时针旋转）。
        # - about_point=clock.get_center(): 以时钟的中心为旋转支点。
        rot_kwargs = dict(
            axis=OUT,
            about_point=clock.get_center()
        )
        
        # 2. 计算时针旋转的总角度
        # 将小时数转换为弧度。一圈是 2*PI，12 小时转一圈，所以每小时是 2*PI/12。
        # 负号表示顺时针旋转。
        hour_radians = -hours_passed * 2 * PI / 12
        
        # 3. 创建并组合动画
        # 调用父类 AnimationGroup 的构造函数，将两个旋转动画组合在一起：
        # - 时针动画：旋转 hour_radians 角度。
        # - 分针动画：旋转 12 * hour_radians 角度（分针速度是时针的 12 倍）。
        # - group=clock: 将动画的“主体”设置为 clock 对象，这有助于在动画编辑器中正确显示。
        # - run_time=run_time: 设置整个动画的持续时间。
        super().__init__(
            Rotating(
                clock.hour_hand,
                angle=hour_radians,
                **rot_kwargs
            ),
            Rotating(
                clock.minute_hand,
                angle=12 * hour_radians,
                **rot_kwargs
            ),
            group=clock,
            run_time=run_time,
            **kwargs
        )


# 定义一个 Bubble 类，继承自 VGroup，用于创建一个带内容的对话框气泡。
class Bubble(VGroup):
    # 指定要加载的 SVG 文件的名称（不含 .svg 后缀）。
    file_name: str = "Bubbles_speech.svg"
    # 用于微调气泡中心点的调整因子。
    bubble_center_adjustment_factor = 0.125

    def __init__(
        self,
        content: str | VMobject | None = None,
        buff: float = 1.0,
        filler_shape: Tuple[float, float] = (3.0, 2.0),
        pin_point: Vect3 | None = None,
        direction: Vect3 = LEFT,
        add_content: bool = True,
        fill_color: ManimColor = BLACK,
        fill_opacity: float = 0.8,
        stroke_color: ManimColor = WHITE,
        stroke_width: float = 3.0,
        **kwargs
    ):
        # 调用父类 VGroup 的构造函数进行初始化。
        super().__init__(**kwargs)
        self.direction = direction

        # 1. 处理内容 (content)
        # 如果没有提供内容，则创建一个不可见的矩形作为占位符。
        if content is None:
            content = Rectangle(*filler_shape)
            content.set_fill(opacity=0)
            content.set_stroke(width=0)
        # 如果提供的内容是字符串，则将其转换为 Text 对象。
        elif isinstance(content, str):
            content = Text(content)
        self.content = content

        # 2. 创建气泡主体 (body)
        # 调用 get_body 方法，根据内容、方向和边距创建气泡的形状。
        self.body = self.get_body(content, direction, buff)
        # 设置气泡主体的样式（填充和描边）。
        self.body.set_fill(fill_color, fill_opacity)
        self.body.set_stroke(stroke_color, stroke_width)
        # 将气泡主体添加到组合中。
        self.add(self.body)

        # 3. 添加内容到气泡中
        # 如果 add_content 为 True，则将内容对象也添加到组合中。
        if add_content:
            self.add(self.content)

        # 4. 将气泡固定到指定点
        # 如果提供了 pin_point，则调用 pin_to 方法将气泡的尖端固定到该点。
        if pin_point is not None:
            self.pin_to(pin_point)

def get_body(self, content: VMobject, direction: Vect3, buff: float) -> VMobject:
    """
    创建并返回气泡的主体形状（SVG），使其能够包裹住给定的内容。

    参数:
        content (VMobject): 气泡要包裹的内容。
        direction (Vect3): 气泡尖端的指向（如 LEFT, RIGHT）。
        buff (float): 内容与气泡边缘之间的缓冲距离。

    返回:
        VMobject: 一个已经调整好大小和位置的气泡主体。
    """
    # 1. 加载 SVG 并根据方向翻转
    # 加载预设的气泡 SVG 文件。
    body = SVGMobject(self.file_name)
    # 如果方向向量的 x 分量大于 0（即指向右侧），则水平翻转气泡，使其尖端朝右。
    if direction[0] > 0:
        body.flip()

    # 2. 计算并设置气泡的目标尺寸
    # 获取内容的宽和高。
    width = content.get_width()
    height = content.get_height()
    # 根据内容尺寸和缓冲距离计算气泡的目标宽度和高度。
    target_width = width + min(buff, height)
    target_height = 1.35 * (height + buff)  # 1.35 是一个经验系数，用于美观地适配高度
    # 将气泡的形状拉伸/压缩到计算出的目标尺寸。
    body.set_shape(target_width, target_height)

    # 3. 定位气泡主体
    # 将气泡移动到与内容相同的位置。
    body.move_to(content)
    # 将气泡向下微调一点，以补偿尖端导致的视觉中心偏移。
    body.shift(self.bubble_center_adjustment_factor * body.get_height() * DOWN)

    return body


def get_tip(self):
    """
    获取气泡尖端的坐标。

    返回:
        np.ndarray: 气泡尖端的三维坐标。
    """
    # 气泡的尖端位于其右下角或左下角，具体取决于方向。
    # self.get_corner(DOWN + self.direction) 是一个巧妙的组合：
    # - 如果 direction 是 LEFT，则为左下角。
    # - 如果 direction 是 RIGHT，则为右下角。
    return self.get_corner(DOWN + self.direction)


def get_bubble_center(self):
    """
    获取气泡主体的视觉中心点（相对于尖端进行了微调）。

    返回:
        np.ndarray: 气泡视觉中心的三维坐标。
    """
    # 获取整个组合的几何中心，然后向上微调一个因子，
    # 这个因子与气泡的高度成正比，以得到一个更符合视觉预期的中心。
    factor = self.bubble_center_adjustment_factor
    return self.get_center() + factor * self.get_height() * UP

def move_tip_to(self, point):
    """
    将气泡的尖端移动到指定坐标点。

    参数:
        point (Vect3): 气泡尖端要移动到的目标坐标。

    返回:
        self: 返回自身，支持链式调用。
    """
    # 计算“当前尖端位置到目标点”的偏移向量，让气泡整体移动这个向量，实现尖端精准定位
    self.shift(point - self.get_tip())
    return self


def flip(self, axis=UP, only_body=True, **kwargs):
    """
    沿指定轴翻转气泡（默认垂直轴），可控制是否仅翻转主体。

    参数:
        axis (Vect3): 翻转所围绕的轴，默认 UP（垂直轴，实现水平翻转）。
        only_body (bool): 是否仅翻转气泡主体，True 时内容也会同步翻转。
        **kwargs: 传递给父类 flip 方法的额外参数。

    返回:
        self: 返回自身，支持链式调用。
    """
    # 调用父类 VGroup 的 flip 方法，对气泡整体（含主体、内容）执行翻转
    super().flip(axis=axis, **kwargs)
    # 若 only_body 为 True，额外对内容单独翻转（确保内容方向与主体匹配）
    if only_body:
        self.content.flip(axis=axis)
    # 若沿垂直轴（UP）翻转，说明气泡尖端方向左右切换，更新 direction 向量
    if abs(axis[1]) > 0:
        self.direction = -np.array(self.direction)
    return self


def pin_to(self, mobject, auto_flip=False):
    """
    将气泡尖端“固定”到目标对象（mobject）的边缘，实现气泡与对象的关联。

    参数:
        mobject (VMobject): 气泡要关联的目标对象（如文字、图形）。
        auto_flip (bool): 是否根据目标对象位置自动翻转气泡方向（左右切换）。

    返回:
        self: 返回自身，支持链式调用。
    """
    # 获取目标对象的几何中心
    mob_center = mobject.get_center()
    # 判断是否需要翻转：目标对象中心的 x 方向（左右）与气泡当前 direction 的 x 方向是否相反
    want_to_flip = np.sign(mob_center[0]) != np.sign(self.direction[0])
    # 若需要翻转且开启 auto_flip，执行翻转调整方向
    if want_to_flip and auto_flip:
        self.flip()
    # 获取目标对象边界框上“朝向气泡方向”的边缘点（作为气泡尖端的附着点）
    boundary_point = mobject.get_bounding_box_point(UP - self.direction)
    # 计算“目标对象中心到边缘点”的向量，确定气泡尖端的最终位置
    vector_from_center = 1.0 * (boundary_point - mob_center)
    # 将气泡尖端移动到“目标中心 + 边缘偏移向量”的位置，完成固定
    self.move_tip_to(mob_center + vector_from_center)
    return self

def position_mobject_inside(self, mobject, buff=MED_LARGE_BUFF):
    """
    将目标对象（mobject）调整尺寸并定位到气泡内部，确保不超出气泡边界。

    参数:
        mobject (VMobject): 要放入气泡的对象（如文本、图形）。
        buff (float): 对象与气泡内壁之间的缓冲距离，默认使用中等大缓冲。

    返回:
        mobject: 已调整好尺寸和位置的目标对象。
    """
    # 限制对象最大宽度：不超过气泡主体宽度减去两侧缓冲
    mobject.set_max_width(self.body.get_width() - 2 * buff)
    # 限制对象最大高度：不超过气泡主体高度的1/1.5（适配气泡形状）减去上下缓冲
    mobject.set_max_height(self.body.get_height() / 1.5 - 2 * buff)
    # 将对象移动到气泡视觉中心：用气泡中心坐标减去对象自身中心坐标，实现精准对齐
    mobject.shift(self.get_bubble_center() - mobject.get_center())
    return mobject


def add_content(self, mobject):
    """
    将目标对象作为新内容添加到气泡中，并自动调整其在气泡内的位置。

    参数:
        mobject (VMobject): 要添加的新内容对象。

    返回:
        self.content: 已添加并定位好的新内容对象。
    """
    # 先调用方法将新对象调整尺寸并放入气泡内部
    self.position_mobject_inside(mobject)
    # 更新气泡的 content 属性为新对象
    self.content = mobject
    return self.content


def write(self, text):
    """
    便捷方法：将字符串转换为 Text 对象，作为内容添加到气泡中。

    参数:
        text (str): 要显示在气泡内的文本内容。

    返回:
        self: 返回自身，支持链式调用。
    """
    # 先创建 Text 对象，再通过 add_content 方法添加到气泡
    self.add_content(Text(text))
    return self


def resize_to_content(self, buff=1.0):  # TODO
    """
    （待完善）根据当前内容调整气泡主体的尺寸，使其适配内容大小。

    参数:
        buff (float): 内容与气泡内壁之间的缓冲距离，默认1.0。
    """
    # 调用 get_body 重新生成适配当前内容的气泡主体，再通过 match_points 同步气泡形状
    self.body.match_points(self.get_body(self.content, self.direction, buff))


def clear(self):
    """
    清除气泡中的当前内容（从气泡组合中移除 content 对象）。

    返回:
        self: 返回自身，支持链式调用。
    """
    # 从气泡（VGroup）中移除内容对象
    self.remove(self.content)
    return self


# 定义一个 SpeechBubble 类，继承自 Bubble，用于创建一个带尖角（stem）的对话框气泡。
class SpeechBubble(Bubble):
    def __init__(
        self,
        content: str | VMobject | None = None,
        buff: float = MED_SMALL_BUFF,
        filler_shape: Tuple[float, float] = (2.0, 1.0),
        stem_height_to_bubble_height: float = 0.5,
        stem_top_x_props: Tuple[float, float] = (0.2, 0.3),
        **kwargs
    ):
        # 存储与气泡尖角（stem）相关的参数。
        self.stem_height_to_bubble_height = stem_height_to_bubble_height
        self.stem_top_x_props = stem_top_x_props
        # 调用父类 Bubble 的构造函数进行初始化。
        super().__init__(content, buff, filler_shape, **kwargs)

    def get_body(self, content: VMobject, direction: Vect3, buff: float) -> VMobject:
        """
        重写父类方法，创建一个由矩形和三角形组成的气泡主体。

        参数:
            content (VMobject): 气泡要包裹的内容。
            direction (Vect3): 气泡尖角的指向（如 LEFT, RIGHT）。
            buff (float): 内容与气泡边缘之间的缓冲距离。

        返回:
            VMobject: 一个由矩形和三角形组成的组合气泡主体。
        """
        # 1. 创建圆角矩形作为气泡主体
        # 创建一个包围内容的矩形，并设置圆角。
        rect = SurroundingRectangle(content, buff=buff)
        rect.round_corners()

        # 2. 创建气泡尖角（stem）
        # 获取矩形左下角和右下角的点。
        lp = rect.get_corner(DL)
        rp = rect.get_corner(DR)
        # 根据矩形高度计算尖角的高度。
        stem_height = self.stem_height_to_bubble_height * rect.get_height()
        # 获取用于定位尖角顶部两个顶点的比例值。
        low_prop, high_prop = self.stem_top_x_props
        # 创建一个三角形作为尖角，其顶部两个顶点在矩形底部边上，底部顶点向下延伸。
        triangle = Polygon(
            interpolate(lp, rp, low_prop),
            interpolate(lp, rp, high_prop),
            lp + stem_height * DOWN,
        )

        # 3. 组合并调整气泡
        # 将矩形和三角形合并成一个单一的形状。
        result = Union(rect, triangle)
        # 增加曲线分段数，使形状在变形时更平滑。
        result.insert_n_curves(20)
        # 如果方向向右，则水平翻转整个气泡，使尖角朝右。
        if direction[0] > 0:
            result.flip()

        return result


# 定义一个 ThoughtBubble 类，继承自 Bubble，用于创建一个类似漫画中“思考泡泡”的对话框。
class ThoughtBubble(Bubble):
    def __init__(
        self,
        content: str | VMobject | None = None,
        buff: float = SMALL_BUFF,
        filler_shape: Tuple[float, float] = (2.0, 1.0),
        bulge_radius: float = 0.35,
        bulge_overlap: float = 0.25,
        noise_factor: float = 0.1,
        circle_radii: list[float] = [0.1, 0.15, 0.2],
        **kwargs
    ):
        # 存储用于构建“思考泡泡”的特殊参数。
        self.bulge_radius = bulge_radius
        self.bulge_overlap = bulge_overlap
        self.noise_factor = noise_factor
        self.circle_radii = circle_radii
        # 调用父类 Bubble 的构造函数进行初始化。
        super().__init__(content, buff, filler_shape, **kwargs)

    def get_body(self, content: VMobject, direction: Vect3, buff: float) -> VMobject:
        """
        重写父类方法，创建一个不规则的、带小气泡链的思考泡泡主体。

        参数:
            content (VMobject): 气泡要包裹的内容。
            direction (Vect3): 气泡链的指向（如 LEFT, RIGHT）。
            buff (float): 内容与气泡边缘之间的缓冲距离。

        返回:
            VMobject: 一个由不规则云状图形和小气泡链组成的组合对象。
        """
        # 1. 创建基础矩形和噪声点
        # 创建一个包围内容的矩形。
        rect = SurroundingRectangle(content, buff)
        perimeter = rect.get_arc_length()
        radius = self.bulge_radius
        step = (1 - self.bulge_overlap) * (2 * radius)
        nf = self.noise_factor
        # 获取矩形的四个角点。
        corners = [rect.get_corner(v) for v in [DL, UL, UR, DR]]
        points = []
        # 遍历每一条边，在边上生成一系列带有随机噪声的点。
        for c1, c2 in adjacent_pairs(corners):
            n_alphas = int(get_norm(c1 - c2) / step) + 1
            for alpha in np.linspace(0, 1, n_alphas):
                points.append(interpolate(
                    c1, c2, alpha + nf * (step / n_alphas) * (random.random() - 0.5)
                ))

        # 2. 创建云状主体
        # 将基础矩形和一系列随机大小的圆（bulges）合并，形成不规则的云状外形。
        cloud = Union(rect, *(
            Circle(radius=radius * (1 + nf * random.random())).move_to(point)
            for point in points
        ))
        cloud.set_stroke(WHITE, 2)

        # 3. 创建小气泡链
        # 创建一系列大小递增的小圆，模拟思考泡泡的气泡链。
        circles = VGroup(Circle(radius=radius) for radius in self.circle_radii)
        circ_buff = 0.25 * self.circle_radii[0]
        circles.arrange(UR, buff=circ_buff)
        circles[1].shift(circ_buff * DR)
        # 将气泡链定位到云状主体的下方。
        circles.next_to(cloud, DOWN, 4 * circ_buff, aligned_edge=LEFT)
        circles.set_stroke(WHITE, 2)

        # 4. 组合并调整方向
        # 将气泡链和云状主体组合成一个整体。
        result = VGroup(*circles, cloud)

        # 如果方向向右，则水平翻转整个思考泡泡。
        if direction[0] > 0:
            result.flip()

        return result


# 定义 OldSpeechBubble 类，继承自 Bubble，是基础对话框气泡的“旧版”实现。
# 直接复用父类 Bubble 的逻辑，仅指定了特定的 SVG 文件作为气泡主体。
class OldSpeechBubble(Bubble):
    # 指定旧版对话框气泡对应的 SVG 文件名（不含 .svg 后缀）。
    file_name: str = "Bubbles_speech.svg"


# 定义 DoubleSpeechBubble 类，继承自 Bubble，用于创建双框样式的对话框气泡。
# 同样复用父类逻辑，仅通过指定 SVG 文件实现独特的双框外观。
class DoubleSpeechBubble(Bubble):
    # 指定双框对话框气泡对应的 SVG 文件名（不含 .svg 后缀）。
    file_name: str = "Bubbles_double_speech.svg"


# 定义 OldThoughtBubble 类，继承自 Bubble，是“思考泡泡”的旧版实现。
class OldThoughtBubble(Bubble):
    # 指定旧版思考泡泡对应的 SVG 文件名（不含 .svg 后缀）。
    file_name: str = "Bubbles_thought.svg"

    def get_body(self, content: VMobject, direction: Vect3, buff: float) -> VMobject:
        """
        重写父类 get_body 方法，对加载的 SVG 气泡主体进行额外排序处理。

        参数:
            content (VMobject): 气泡要包裹的内容。
            direction (Vect3): 气泡指向（如 LEFT, RIGHT）。
            buff (float): 内容与气泡边缘的缓冲距离。

        返回:
            VMobject: 处理后的气泡主体（按 y 轴坐标排序）。
        """
        # 先调用父类 Bubble 的 get_body 方法，加载基础 SVG 气泡主体。
        body = super().get_body(content, direction, buff)
        # 对气泡主体的子部分按 y 轴坐标（p[1]，即垂直方向）排序，确保图层顺序正确。
        body.sort(lambda p: p[1])
        return body

    def make_green_screen(self):
        """
        专门用于将气泡主体的特定部分设置为绿幕颜色（方便后期抠图）。
        """
        # 将气泡主体的最后一个子部分（通常是背景层）填充为绿幕色，并设置完全不透明。
        self.body[-1].set_fill(GREEN_SCREEN, opacity=1)
        return self


# 定义 VectorizedEarth 类，继承自 SVGMobject，用于显示矢量地球图形
class VectorizedEarth(SVGMobject):
    # 指定要加载的地球相关 SVG 文件名称（不含 .svg 后缀）
    file_name: str = "earth"

    def __init__(
        self,
        height: float = 2.0,  # 地球图形的默认高度
        **kwargs
    ):
        # 调用父类 SVGMobject 构造函数，传入高度及其他参数初始化
        super().__init__(height=height, **kwargs)
        # 增加 SVG 路径的曲线分段数至 20，让地球图形变形时更平滑
        self.insert_n_curves(20)
        
        # 创建一个圆形作为地球的底层背景
        circle = Circle(
            stroke_width=3,       # 圆形描边宽度
            stroke_color=GREEN,   # 圆形描边颜色（绿色）
            fill_opacity=1,       # 圆形填充完全不透明
            fill_color=BLUE_C     # 圆形填充颜色（蓝色，模拟海洋）
        )
        # 让圆形尺寸、位置与加载的地球 SVG 图形匹配
        circle.replace(self)
        # 将圆形添加到地球图形的最底层（作为背景）
        self.add_to_back(circle)


# 定义 Piano 类，继承自 VGroup，用于创建钢琴键盘组合对象
class Piano(VGroup):
    def __init__(
        self,
        n_white_keys = 52,          # 白键总数量，默认52个
        black_pattern = [0, 2, 3, 5, 6],  # 黑键在八度内的位置索引（对应白键间的间隔）
        white_keys_per_octave = 7,  # 每个八度的白键数量，默认7个
        white_key_dims = (0.15, 1.0),  # 白键尺寸（宽、高）
        black_key_dims = (0.1, 0.66),  # 黑键尺寸（宽、高）
        key_buff = 0.02,            # 琴键间的间隔距离
        white_key_color = WHITE,    # 白键颜色，默认白色
        black_key_color = GREY_E,   # 黑键颜色，默认深灰色
        total_width = 13,           # 钢琴键盘总宽度，默认13
        **kwargs
    ):
        # 存储钢琴键盘的各项参数为实例属性
        self.n_white_keys = n_white_keys
        self.black_pattern = black_pattern
        self.white_keys_per_octave = white_keys_per_octave
        self.white_key_dims = white_key_dims
        self.black_key_dims = black_key_dims
        self.key_buff = key_buff
        self.white_key_color = white_key_color
        self.black_key_color = black_key_color
        self.total_width = total_width

        # 调用父类 VGroup 构造函数完成基础初始化
        super().__init__(**kwargs)
        # 添加白键到钢琴组合中
        self.add_white_keys()
        # 添加黑键到钢琴组合中
        self.add_black_keys()
        # 对所有琴键进行排序（确保显示层级正确）
        self.sort_keys()
        # 反转除最后一个外所有键的点顺序（调整图形绘制方向）
        self[:-1].reverse_points()
        # 将整个钢琴键盘缩放到指定总宽度
        self.set_width(self.total_width)

def add_white_keys(self):
    """
    创建并添加所有的白色琴键到钢琴中。
    """
    # 1. 创建单个白键模板
    # 创建一个矩形作为白键的基础形状。
    key = Rectangle(*self.white_key_dims)
    # 设置白键的填充颜色和不透明度。
    key.set_fill(self.white_key_color, 1)
    # 设置白键无边框。
    key.set_stroke(width=0)

    # 2. 创建并排列所有白键
    # 使用 get_grid 方法快速创建一个 1 行 n_white_keys 列的白键网格。
    # buff=self.key_buff 设置了每个白键之间的水平间距。
    self.white_keys = key.get_grid(1, self.n_white_keys, buff=self.key_buff)

    # 3. 将白键添加到钢琴组合中
    # 将创建好的所有白键添加到 Piano 对象（VGroup）中。
    self.add(*self.white_keys)


def add_black_keys(self):
    """
    创建并添加所有的黑色琴键到钢琴中，并在白键上为黑键“挖出”位置。
    """
    # 1. 创建单个黑键模板
    # 创建一个矩形作为黑键的基础形状。
    key = Rectangle(*self.black_key_dims)
    # 设置黑键的填充颜色和不透明度。
    key.set_fill(self.black_key_color, 1)
    # 设置黑键无边框。
    key.set_stroke(width=0)

    # 2. 初始化黑键组合
    # 创建一个空的 VGroup 来存放所有的黑键。
    self.black_keys = VGroup()

    # 3. 循环创建并定位黑键
    # 遍历所有白键之间的间隙。
    for i in range(len(self.white_keys) - 1):
        # 根据黑键位置模式（black_pattern）判断当前位置是否需要放置黑键。
        if i % self.white_keys_per_octave not in self.black_pattern:
            continue

        # 获取相邻的两个白键。
        wk1 = self.white_keys[i]
        wk2 = self.white_keys[i + 1]

        # 创建一个黑键实例并将其定位在两个白键顶部的中间。
        bk = key.copy()
        bk.move_to(midpoint(wk1.get_top(), wk2.get_top()), UP)

        # 4. 在白键上为黑键“挖洞”
        # 创建一个比黑键稍大的矩形，用于从白键上减去。
        big_bk = bk.copy()
        big_bk.stretch((bk.get_width() + self.key_buff) / bk.get_width(), 0)
        big_bk.stretch((bk.get_height() + self.key_buff) / bk.get_height(), 1)
        big_bk.move_to(bk, UP)

        # 使用 Difference 操作，从相邻的两个白键中减去 big_bk 的形状，
        # 这样白键在黑键下方的部分就被“挖掉”了，避免了视觉上的重叠。
        for wk in wk1, wk2:
            wk.become(Difference(wk, big_bk).match_style(wk))

        # 5. 将黑键添加到组合中
        self.black_keys.add(bk)

    # 6. 将所有黑键添加到钢琴组合中
    self.add(*self.black_keys)

def sort_keys(self):
    """
    对钢琴的所有琴键进行排序，按琴键的 x 轴坐标（水平方向）从左到右排列
    """
    # 根据琴键的 x 轴坐标（p[0]）排序，确保琴键从左到右顺序正确
    self.sort(lambda p: p[0])


# 定义 Piano3D 类，继承自 VGroup，用于创建 3D 效果的钢琴
class Piano3D(VGroup):
    def __init__(
        self,
        shading: Tuple[float, float, float] = (1.0, 0.2, 0.2),  # 3D 模型的阴影参数
        stroke_width: float = 0.25,                            # 琴键描边宽度
        stroke_color: ManimColor = BLACK,                      # 琴键描边颜色
        key_depth: float = 0.1,                                # 琴键的 3D 深度（前后厚度）
        black_key_shift: float = 0.05,                         # 黑键在 3D 空间中向外（OUT）偏移的距离
        piano_2d_config: dict = dict(                          # 传递给 2D 钢琴的配置参数
            white_key_color=GREY_A,
            key_buff=0.001
        ),
        **kwargs
    ):
        # 1. 创建 2D 钢琴作为基础
        # 根据配置创建一个 2D 钢琴实例，作为 3D 钢琴的形状基础
        piano_2d = Piano(**piano_2d_config)
        
        # 2. 将 2D 琴键转换为 3D 棱柱（Prism）
        # 遍历 2D 钢琴的每个琴键，用 Prismify 转换为带深度的 3D 形状，作为 3D 钢琴的子对象
        super().__init__(*(
            Prismify(key, key_depth)
            for key in piano_2d
        ))
        
        # 3. 设置 3D 钢琴的基础样式
        self.set_stroke(stroke_color, stroke_width)  # 设置所有琴键的描边
        self.set_shading(*shading)                   # 设置 3D 阴影效果
        self.apply_depth_test()                      # 开启深度测试，让 3D 效果更真实（近大远小、遮挡）

        # 4. 调整黑键的 3D 位置（向外偏移，模拟真实钢琴黑键略突出的效果）
        for i, key in enumerate(self):
            # 判断当前 3D 琴键对应的 2D 琴键是否为黑键
            if piano_2d[i] in piano_2d.black_keys:
                key.shift(black_key_shift * OUT)  # 黑键向外偏移
                key.set_color(BLACK)              # 确保黑键颜色为黑色


# 定义 DieFace 类，继承自 VGroup，用于创建骰子的单个面（含点数）
class DieFace(VGroup):
    def __init__(
        self,
        value: int,                  # 骰子面的点数（1-6）
        side_length: float = 1.0,    # 骰子面的边长
        corner_radius: float = 0.15, # 骰子面的圆角半径
        stroke_color: ManimColor = WHITE, # 骰子面的边框颜色
        stroke_width: float = 2.0,   # 骰子面的边框宽度
        fill_color: ManimColor = GREY_E,  # 骰子面的填充颜色
        dot_radius: float = 0.08,    # 点数（圆点）的半径
        dot_color: ManimColor = WHITE,    # 点数（圆点）的颜色
        dot_coalesce_factor: float = 0.5  # 点数间的间距调整因子
    ):
        # 1. 创建单个点数（圆点）模板
        dot = Dot(radius=dot_radius, fill_color=dot_color)
        
        # 2. 创建骰子面的正方形底座
        square = Square(
            side_length=side_length,
            stroke_color=stroke_color,
            stroke_width=stroke_width,
            fill_color=fill_color,
            fill_opacity=1.0,  # 完全填充
        )
        square.round_corners(corner_radius)  # 给正方形加圆角

        # 3. 校验点数合法性（仅支持1-6）
        if not (1 <= value <= 6):
            raise Exception("DieFace only accepts integer inputs between 1 and 6")

        # 4. 定义不同点数对应的圆点位置（基于正方形的角/中心向量）
        # 索引对应点数-1（1点→索引0，6点→索引5），每个元组存储该点数的圆点位置向量
        edge_group = [
            (ORIGIN,),                          # 1点：中心
            (UL, DR),                           # 2点：左上角、右下角
            (UL, ORIGIN, DR),                   # 3点：左上角、中心、右下角
            (UL, UR, DL, DR),                   # 4点：左上、右上、左下、右下
            (UL, UR, ORIGIN, DL, DR),           # 5点：左上、右上、中心、左下、右下
            (UL, UR, LEFT, RIGHT, DL, DR),      # 6点：左上、右上、左中、右中、左下、右下
        ][value - 1]

        # 5. 创建并排列当前点数的所有圆点
        # 根据位置向量，在正方形对应位置生成圆点，组成点数组
        arrangement = VGroup(*(
            dot.copy().move_to(square.get_bounding_box_point(vect))
            for vect in edge_group
        ))
        arrangement.space_out_submobjects(dot_coalesce_factor)  # 调整点数间间距

        # 6. 组合骰子面（正方形底座+点数），并存储关键属性
        super().__init__(square, arrangement)
        self.dots = arrangement  # 存储点数组，方便后续操作
        self.value = value       # 存储当前面的点数
        self.index = value       # 存储索引（与点数一致，便于索引调用）


class Dartboard(VGroup):
    radius = 3
    n_sectors = 20

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        n_sectors = self.n_sectors
        angle = TAU / n_sectors

        segments = VGroup(*[
            VGroup(*[
                AnnularSector(
                    inner_radius=in_r,
                    outer_radius=out_r,
                    start_angle=n * angle,
                    angle=angle,
                    fill_color=color,
                )
                for n, color in zip(
                    range(n_sectors),
                    it.cycle(colors)
                )
            ])
            for colors, in_r, out_r in [
                ([GREY_B, GREY_E], 0, 1),
                ([GREEN_E, RED_E], 0.5, 0.55),
                ([GREEN_E, RED_E], 0.95, 1),
            ]
        ])
        segments.rotate(-angle / 2)
        bullseyes = VGroup(*[
            Circle(radius=r)
            for r in [0.07, 0.035]
        ])
        bullseyes.set_fill(opacity=1)
        bullseyes.set_stroke(width=0)
        bullseyes[0].set_color(GREEN_E)
        bullseyes[1].set_color(RED_E)

        self.bullseye = bullseyes[1]
        self.add(*segments, *bullseyes)
        self.scale(self.radius)
