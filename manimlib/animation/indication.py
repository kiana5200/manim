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


class FocusOn(Transform):
    def __init__(
        self,
        focus_point: np.ndarray | Mobject,
        opacity: float = 0.2,
        color: ManimColor = GREY,
        run_time: float = 2,
        remover: bool = True,
        **kwargs
    ):
        self.focus_point = focus_point
        self.opacity = opacity
        self.color = color
        # Initialize with blank mobject, while create_target
        # and create_starting_mobject handle the meat
        super().__init__(VMobject(), run_time=run_time, remover=remover, **kwargs)

    def create_target(self) -> Dot:
        little_dot = Dot(radius=0)
        little_dot.set_fill(self.color, opacity=self.opacity)
        little_dot.add_updater(lambda d: d.move_to(self.focus_point))
        return little_dot

    def create_starting_mobject(self) -> Dot:
        return Dot(
            radius=FRAME_X_RADIUS + FRAME_Y_RADIUS,
            stroke_width=0,
            fill_color=self.color,
            fill_opacity=0,
        )


class Indicate(Transform):
    def __init__(
        self,
        mobject: Mobject,
        scale_factor: float = 1.2,
        color: ManimColor = YELLOW,
        rate_func: Callable[[float], float] = there_and_back,
        **kwargs
    ):
        self.scale_factor = scale_factor
        self.color = color
        super().__init__(mobject, rate_func=rate_func, **kwargs)

    def create_target(self) -> Mobject:
        target = self.mobject.copy()
        target.scale(self.scale_factor)
        target.set_color(self.color)
        return target


class Flash(AnimationGroup):
    def __init__(
        self,
        point: np.ndarray | Mobject,
        color: ManimColor = YELLOW,
        line_length: float = 0.2,
        num_lines: int = 12,
        flash_radius: float = 0.3,
        line_stroke_width: float = 3.0,
        run_time: float = 1.0,
        **kwargs
    ):
        self.point = point
        self.color = color
        self.line_length = line_length
        self.num_lines = num_lines
        self.flash_radius = flash_radius
        self.line_stroke_width = line_stroke_width

        self.lines = self.create_lines()
        animations = self.create_line_anims()
        super().__init__(
            *animations,
            group=self.lines,
            run_time=run_time,
            **kwargs,
        )

    def create_lines(self) -> VGroup:
        lines = VGroup()
        for angle in np.arange(0, TAU, TAU / self.num_lines):
            line = Line(ORIGIN, self.line_length * RIGHT)
            line.shift((self.flash_radius - self.line_length) * RIGHT)
            line.rotate(angle, about_point=ORIGIN)
            lines.add(line)
        lines.set_stroke(
            color=self.color,
            width=self.line_stroke_width
        )
        lines.add_updater(lambda l: l.move_to(self.point))
        return lines

    def create_line_anims(self) -> list[Animation]:
        return [
            ShowCreationThenDestruction(line)
            for line in self.lines
        ]


class CircleIndicate(Transform):
    def __init__(
        self,
        mobject: Mobject,
        scale_factor: float = 1.2,
        rate_func: Callable[[float], float] = there_and_back,
        stroke_color: ManimColor = YELLOW,
        stroke_width: float = 3.0,
        remover: bool = True,
        **kwargs
    ):
        circle = Circle(stroke_color=stroke_color, stroke_width=stroke_width)
        circle.surround(mobject)
        pre_circle = circle.copy().set_stroke(width=0)
        pre_circle.scale(1 / scale_factor)
        super().__init__(
            pre_circle, circle,
            rate_func=rate_func,
            remover=remover,
            **kwargs
        )


class ShowPassingFlash(ShowPartial):
    def __init__(
        self,
        mobject: Mobject,
        time_width: float = 0.1,
        remover: bool = True,
        **kwargs
    ):
        self.time_width = time_width
        super().__init__(
            mobject,
            remover=remover,
            **kwargs
        )

    def get_bounds(self, alpha: float) -> tuple[float, float]:
        tw = self.time_width
        upper = interpolate(0, 1 + tw, alpha)
        lower = upper - tw
        upper = min(upper, 1)
        lower = max(lower, 0)
        return (lower, upper)

    def finish(self) -> None:
        super().finish()
        for submob, start in self.get_all_families_zipped():
            submob.pointwise_become_partial(start, 0, 1)


class VShowPassingFlash(Animation):
    def __init__(
        self,
        vmobject: VMobject,
        time_width: float = 0.3,
        taper_width: float = 0.05,
        remover: bool = True,
        **kwargs
    ):
        self.time_width = time_width
        self.taper_width = taper_width
        super().__init__(vmobject, remover=remover, **kwargs)
        self.mobject = vmobject

    def taper_kernel(self, x):
        if x < self.taper_width:
            return x
        elif x > 1 - self.taper_width:
            return 1.0 - x
        return 1.0

    def begin(self) -> None:
        # Compute an array of stroke widths for each submobject
        # which tapers out at either end
        self.submob_to_widths = dict()
        for sm in self.mobject.get_family():
            widths = sm.get_stroke_widths()
            self.submob_to_widths[hash(sm)] = np.array([
                width * self.taper_kernel(x)
                for width, x in zip(widths, np.linspace(0, 1, len(widths)))
            ])
        super().begin()

    def interpolate_submobject(
        self,
        submobject: VMobject,
        starting_sumobject: None,
        alpha: float
    ) -> None:
        widths = self.submob_to_widths[hash(submobject)]

        # Create a gaussian such that 3 sigmas out on either side
        # will equals time_width
        tw = self.time_width
        sigma = tw / 6
        mu = interpolate(-tw / 2, 1 + tw / 2, alpha)
        xs = np.linspace(0, 1, len(widths))
        zs = (xs - mu) / sigma
        gaussian = np.exp(-0.5 * zs * zs)
        gaussian[abs(xs - mu) > 3 * sigma] = 0

        if len(widths * gaussian) !=0:
            submobject.set_stroke(width=widths * gaussian)


    def finish(self) -> None:
        super().finish()
        for submob, start in self.get_all_families_zipped():
            submob.match_style(start)


class FlashAround(VShowPassingFlash):
    def __init__(
        self,
        mobject: Mobject,
        time_width: float = 1.0,
        taper_width: float = 0.0,
        stroke_width: float = 4.0,
        color: ManimColor = YELLOW,
        buff: float = SMALL_BUFF,
        n_inserted_curves: int = 100,
        **kwargs
    ):
        path = self.get_path(mobject, buff)
        if mobject.is_fixed_in_frame():
            path.fix_in_frame()
        path.insert_n_curves(n_inserted_curves)
        path.set_points(path.get_points_without_null_curves())
        path.set_stroke(color, stroke_width)
        super().__init__(path, time_width=time_width, taper_width=taper_width, **kwargs)

    def get_path(self, mobject: Mobject, buff: float) -> SurroundingRectangle:
        return SurroundingRectangle(mobject, buff=buff)


class FlashUnder(FlashAround):
    def get_path(self, mobject: Mobject, buff: float) -> Underline:
        return Underline(mobject, buff=buff, stretch_factor=1.0)


class ShowCreationThenDestruction(ShowPassingFlash):
    def __init__(self, vmobject: VMobject, time_width: float = 2.0, **kwargs):
        super().__init__(vmobject, time_width=time_width, **kwargs)


class ShowCreationThenFadeOut(Succession):
    def __init__(self, mobject: Mobject, remover: bool = True, **kwargs):
        super().__init__(
            ShowCreation(mobject),
            FadeOut(mobject),
            remover=remover,
            **kwargs
        )


class AnimationOnSurroundingRectangle(AnimationGroup):
    RectAnimationType: type = Animation

    def __init__(
        self,
        mobject: Mobject,
        stroke_width: float = 2.0,
        stroke_color: ManimColor = YELLOW,
        buff: float = SMALL_BUFF,
        **kwargs
    ):
        rect = SurroundingRectangle(
            mobject,
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            buff=buff,
        )
        rect.add_updater(lambda r: r.move_to(mobject))
        super().__init__(self.RectAnimationType(rect, **kwargs))


class ShowPassingFlashAround(AnimationOnSurroundingRectangle):
    RectAnimationType = ShowPassingFlash


class ShowCreationThenDestructionAround(AnimationOnSurroundingRectangle):
    RectAnimationType = ShowCreationThenDestruction


class ShowCreationThenFadeAround(AnimationOnSurroundingRectangle):
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
