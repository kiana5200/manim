# 导入__future__模块的annotations，支持字符串形式的类型注解（如在类定义内部引用自身）
# 该特性在Python 3.7+可用，主要用于解决类型注解中的循环引用问题
from __future__ import annotations

# 导入NumPy库，用于高效的数值计算、数组操作和数学函数
import numpy as np

# 从pyglet.window模块导入key对象（别名PygletWindowKeys），用于处理键盘按键事件
# 常用于获取按键的键码（如PygletWindowKeys.A、PygletWindowKeys.ENTER），实现交互功能
from pyglet.window import key as PygletWindowKeys

from manimlib.constants import FRAME_HEIGHT, FRAME_WIDTH
from manimlib.constants import DOWN, LEFT, ORIGIN, RIGHT, UP
from manimlib.constants import MED_LARGE_BUFF, MED_SMALL_BUFF, SMALL_BUFF
from manimlib.constants import BLACK, BLUE, GREEN, GREY_A, GREY_C, RED, WHITE, DEFAULT_MOBJECT_COLOR
from manimlib.mobject.mobject import Group
from manimlib.mobject.mobject import Mobject
from manimlib.mobject.geometry import Circle
from manimlib.mobject.geometry import Dot
from manimlib.mobject.geometry import Line
from manimlib.mobject.geometry import Rectangle
from manimlib.mobject.geometry import RoundedRectangle
from manimlib.mobject.geometry import Square
from manimlib.mobject.svg.text_mobject import Text
from manimlib.mobject.types.vectorized_mobject import VGroup
from manimlib.mobject.value_tracker import ValueTracker
from manimlib.utils.color import rgb_to_hex
from manimlib.utils.space_ops import get_closest_point_on_line
from manimlib.utils.space_ops import get_norm

# 仅在类型检查模式下导入所需类型（避免运行时依赖）
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable
    from manimlib.typing import ManimColor


# Interactive Mobjects
# 交互式图形类（可拖拽）
class MotionMobject(Mobject):
    """
        You could hold and drag this object to any position
        可通过鼠标拖拽的交互式图形容器类。
        包装一个普通Mobject，使其支持鼠标按住并拖动到任意位置，常用于实现场景中的交互功能。
    """
    def __init__(self, mobject: Mobject, **kwargs):
        """
        初始化可拖拽图形容器。

        参数:
            mobject (Mobject): 需要被包装成可拖拽的目标图形对象。
            **kwargs: 传递给父类Mobject的额外参数（如位置、颜色等）。

        断言:
            若传入的`mobject`不是Mobject类型，会触发断言错误。
        """
        # 1. 调用父类Mobject的初始化方法
        super().__init__(**kwargs)
        # 2. 验证传入对象是否为Mobject类型
        assert isinstance(mobject, Mobject)
        # 3. 存储被包装的目标图形，并为其添加鼠标拖拽监听器
        self.mobject = mobject
        # 绑定拖拽事件处理函数：当鼠标拖拽时触发mob_on_mouse_drag
        self.mobject.add_mouse_drag_listner(self.mob_on_mouse_drag)
        # To avoid locking it as static mobject
        # 4. 为目标图形添加空更新器，避免其被标记为“静态图形”（静态图形可能不响应交互）
        self.mobject.add_updater(lambda mob: None)
        # 5. 将目标图形添加到当前容器中，使其成为容器的子图形
        self.add(mobject)

    def mob_on_mouse_drag(self, mob: Mobject, event_data: dict[str, np.ndarray]) -> bool:
        """
        鼠标拖拽事件的处理函数。
        当鼠标拖拽被包装的图形时，会调用此方法更新图形位置。

        参数:
            mob (Mobject): 被拖拽的目标图形对象（即self.mobject）。
            event_data (dict[str, np.ndarray]): 拖拽事件数据，包含当前鼠标位置（key为"point"）。

        返回:
            bool: 返回False，表示不阻止后续事件传播（允许其他监听器继续处理）。
        """
        # 将目标图形移动到鼠标当前位置（event_data["point"]存储鼠标坐标）
        mob.move_to(event_data["point"])
        return False


class Button(Mobject):
    """
        Pass any mobject and register an on_click method

        The on_click method takes mobject as argument like updater
        交互式按钮类，可包装任意Mobject并绑定点击事件。
        通过为目标图形添加鼠标按压监听器，实现点击按钮时触发指定的回调函数。

        点击事件的回调函数需接收一个Mobject参数（通常是按钮本身），类似于更新器(updater)的参数形式。
    """

    # 初始化按钮对象。
    def __init__(self, mobject: Mobject, on_click: Callable[[Mobject]], **kwargs):
        # 调用父类Mobject的初始化方法
        super().__init__(**kwargs)
        # 验证传入对象是否为Mobject类型
        assert isinstance(mobject, Mobject)
        # 存储点击事件的回调函数和按钮图形
        self.on_click = on_click
        self.mobject = mobject
        # 为按钮图形添加鼠标按压监听器，绑定到内部处理方法
        self.mobject.add_mouse_press_listner(self.mob_on_mouse_press)
        # 将按钮图形添加到当前按钮容器中
        self.add(self.mobject)

    # 鼠标按压事件的内部处理方法。
    # 当鼠标点击按钮时，会调用此方法，进而触发用户注册的点击回调函数。
    def mob_on_mouse_press(self, mob: Mobject, event_data) -> bool:
        # 触发用户注册的点击回调函数，将按钮图形作为参数传入
        self.on_click(mob)
        return False


# Controls

class ControlMobject(ValueTracker):
    """
    带有可视化图形的控制值追踪器，继承自ValueTracker。
    用于在场景中创建可交互的控制器（如滑块、旋钮等），既可以追踪数值变化，
    又能通过附加的Mobject提供可视化界面。
    """
    # 初始化控制值追踪器。
    def __init__(self, value: float, *mobjects: Mobject, **kwargs):
        # 1. 调用父类ValueTracker的初始化方法，设置初始值
        super().__init__(value=value, **kwargs)
        # 2. 添加所有传入的可视化图形对象
        self.add(*mobjects)

        # To avoid lock_static_mobject_data while waiting in scene
        # 3. 添加空更新器，避免在场景等待时被标记为静态图形（可能导致交互失效）
        self.add_updater(lambda mob: None)
        # 4. 将控制器固定在帧中，使其不会随相机移动而改变位置
        self.fix_in_frame()

    # 重写父类方法，设置新值时会先验证并执行动画过渡。
    def set_value(self, value: float):
        # 验证值的有效性（子类可重写assert_value实现自定义验证）
        self.assert_value(value)
        # 执行值变化的动画（子类可重写set_value_anim实现自定义动画）
        self.set_value_anim(value)
        # 调用父类方法实际更新值
        return ValueTracker.set_value(self, value)

    # 验证值的有效性，子类可重写此方法实现自定义验证逻辑（如范围检查）。
    def assert_value(self, value):
        # To be implemented in subclasses
        # 预留接口，由子类实现
        pass

    # 值变化时的动画过渡，子类可重写此方法实现自定义动画效果（如滑动、旋转）。
    def set_value_anim(self, value):
        # To be implemented in subclasses
        # 预留接口，由子类实现
        pass


class EnableDisableButton(ControlMobject):
    """
    启用/禁用状态切换按钮，继承自ControlMobject。
    以矩形图形为基础，通过点击切换状态（启用/禁用），并以不同颜色显示当前状态。
    """
    # 初始化启用/禁用切换按钮。
    def __init__(
        self,
        value: bool = True,
        value_type: np.dtype = np.dtype(bool),
        rect_kwargs: dict = {
            "width": 0.5,
            "height": 0.5,
            "fill_opacity": 1.0
        },
        enable_color: ManimColor = GREEN,
        disable_color: ManimColor = RED,
        **kwargs
    ):
        # 存储按钮状态和样式配置
        self.value = value
        self.value_type = value_type
        self.rect_kwargs = rect_kwargs
        self.enable_color = enable_color
        self.disable_color = disable_color

        # 创建按钮的矩形图形
        self.box = Rectangle(**self.rect_kwargs)
        # 调用父类构造方法，传入初始值和矩形图形
        super().__init__(value, self.box, **kwargs)
        # 添加鼠标按压监听器，绑定到点击事件处理方法
        self.add_mouse_press_listner(self.on_mouse_press)

    # 验证值是否为布尔类型，确保按钮状态只能是True或False。
    def assert_value(self, value: bool) -> None:
        assert isinstance(value, bool)

    # 根据状态值更新按钮颜色的动画方法。
    def set_value_anim(self, value: bool) -> None:
        if value:
            # 启用状态：设置为启用颜色
            self.box.set_fill(self.enable_color)
        else:
            # 禁用状态：设置为禁用颜色
            self.box.set_fill(self.disable_color)

    def toggle_value(self) -> None:
        """切换按钮状态（启用→禁用或禁用→启用）。"""
        # 调用父类方法设置新值（当前值的反值）
        super().set_value(not self.get_value())

    # 鼠标点击事件处理方法，触发状态切换。
    def on_mouse_press(self, mob: Mobject, event_data) -> bool:
        mob.toggle_value()  # 切换按钮状态
        return False


class Checkbox(ControlMobject):
    """
    复选框控件，继承自ControlMobject。
    用于表示二选一状态（选中/未选中），通过点击切换状态，并分别显示勾选标记或叉号。
    """
    def __init__(
        self,
        value: bool = True,
        value_type: np.dtype = np.dtype(bool),
        rect_kwargs: dict = {
            "width": 0.5,
            "height": 0.5,
            "fill_opacity": 0.0  # 默认透明填充，仅显示边框
        },
        checkmark_kwargs: dict = {
            "stroke_color": GREEN,  # 勾选标记颜色为绿色
            "stroke_width": 6,      # 勾选标记线宽
        },
        cross_kwargs: dict = {
            "stroke_color": RED,  # 叉号颜色为红色
            "stroke_width": 6,    # 叉号线宽
        },
        box_content_buff: float = SMALL_BUFF,  # 标记与方框的间距
        **kwargs
    ):
        """
        初始化复选框控件。

        参数:
            value (bool): 初始状态，True为选中（显示勾选），False为未选中（显示叉号），默认为True。
            value_type (np.dtype): 值类型，固定为布尔型。
            rect_kwargs (dict): 复选框方框的样式配置。
            checkmark_kwargs (dict): 勾选标记的样式配置。
            cross_kwargs (dict): 叉号的样式配置。
            box_content_buff (float): 标记与方框边缘的缓冲距离。
            **kwargs: 传递给父类ControlMobject的额外参数。
        """
        # 存储样式配置参数
        self.value_type = value_type
        self.rect_kwargs = rect_kwargs
        self.checkmark_kwargs = checkmark_kwargs
        self.cross_kwargs = cross_kwargs
        self.box_content_buff = box_content_buff

        # 创建复选框的方框
        self.box = Rectangle(**self.rect_kwargs)
        # 根据初始状态创建对应的标记（勾选或叉号）
        self.box_content = self.get_checkmark() if value else self.get_cross()
        # 调用父类构造方法，传入初始值和图形元素
        super().__init__(value, self.box, self.box_content, **kwargs)
        # 添加鼠标按压监听器，绑定到点击事件处理方法
        self.add_mouse_press_listner(self.on_mouse_press)

    def assert_value(self, value: bool) -> None:
        """验证值是否为布尔类型，确保状态只能是True或False。"""
        assert isinstance(value, bool)

    def toggle_value(self) -> None:
        """切换复选框状态（选中→未选中或未选中→选中）。"""
        super().set_value(not self.get_value())

    # 根据状态值更新显示的标记（平滑切换勾选/叉号）。
    def set_value_anim(self, value: bool) -> None:
        if value:
            # 切换为勾选标记
            self.box_content.become(self.get_checkmark())
        else:
            # 切换为叉号标记
            self.box_content.become(self.get_cross())

    def on_mouse_press(self, mob: Mobject, event_data) -> None:
        """
        鼠标点击事件处理方法，触发状态切换。

        参数:
            mob (Mobject): 被点击的复选框对象
            event_data: 鼠标点击事件的相关数据
        """
        mob.toggle_value()
        return False

    # Helper methods
    # 创建并返回勾选标记（对勾），自动适配方框大小。
    def get_checkmark(self) -> VGroup:
        # 创建对勾的两条线段
        checkmark = VGroup(
            # 第一条线段（左下到中间）
            Line(UP / 2 + 2 * LEFT, DOWN + LEFT, **self.checkmark_kwargs),
            # 第二条线段（中间到右上）
            Line(DOWN + LEFT, UP + RIGHT, **self.checkmark_kwargs)
        )

        # 自动调整对勾大小以适应方框
        checkmark.stretch_to_fit_width(self.box.get_width())  # 宽度适配方框
        checkmark.stretch_to_fit_height(self.box.get_height())  # 高度适配方框
        checkmark.scale(0.5)  # 整体缩小一点，避免超出方框
        checkmark.move_to(self.box)  # 移动到方框中心
        return checkmark

    # 创建并返回叉号标记，自动适配方框大小。
    def get_cross(self) -> VGroup:
        # 创建叉号的两条对角线
        cross = VGroup(
            Line(UP + LEFT, DOWN + RIGHT, **self.cross_kwargs),  # 左上到右下
            Line(UP + RIGHT, DOWN + LEFT, **self.cross_kwargs)  # 右上到左下
        )

        # 自动调整叉号大小以适应方框
        cross.stretch_to_fit_width(self.box.get_width())  # 宽度适配方框
        cross.stretch_to_fit_height(self.box.get_height())  # 高度适配方框
        cross.scale(0.5)  # 整体缩小一点，避免超出方框
        cross.move_to(self.box)  # 移动到方框中心
        return cross


class LinearNumberSlider(ControlMobject):
    """
    线性数字滑块控件，继承自ControlMobject。
    用于在指定范围内选择数值，通过拖拽滑块按钮调整值，支持步长限制。
    """
    def __init__(
        self,
        value: float = 0,
        value_type: type = np.float64,
        min_value: float = -10.0,
        max_value: float = 10.0,
        step: float = 1.0,
        rounded_rect_kwargs: dict = {
            "height": 0.075,       # 滑道高度
            "width": 2,            # 滑道宽度
            "corner_radius": 0.0375 # 滑道圆角半径
        },
        circle_kwargs: dict = {
            "radius": 0.1,         # 滑块半径
            "stroke_color": GREY_A, # 滑块边框颜色
            "fill_color": GREY_A,   # 滑块填充颜色
            "fill_opacity": 1.0     # 滑块填充透明度
        },
        **kwargs
    ):
        """
        初始化线性滑块控件。

        参数:
            value (float): 初始数值，默认为0。
            value_type (type): 值类型，默认为np.float64。
            min_value (float): 最小值限制，默认为-10.0。
            max_value (float): 最大值限制，默认为10.0。
            step (float): 调整步长，数值变化的最小单位，默认为1.0。
            rounded_rect_kwargs (dict): 滑道（圆角矩形）的样式配置。
            circle_kwargs (dict): 滑块（圆形）的样式配置。
            **kwargs: 传递给父类ControlMobject的额外参数。
        """
        # 存储滑块配置参数
        self.value_type = value_type
        self.min_value = min_value
        self.max_value = max_value
        self.step = step
        self.rounded_rect_kwargs = rounded_rect_kwargs
        self.circle_kwargs = circle_kwargs

        # 创建滑道（圆角矩形）
        self.bar = RoundedRectangle(**self.rounded_rect_kwargs)
        # 创建滑块按钮（圆形）
        self.slider = Circle(**self.circle_kwargs)
        # 创建滑块的拖拽参考轴（不可见）
        self.slider_axis = Line(
            start=self.bar.get_bounding_box_point(LEFT),  # 滑道左端
            end=self.bar.get_bounding_box_point(RIGHT)  # 滑道右端
        )
        self.slider_axis.set_opacity(0.0)  # 隐藏参考轴
        self.slider.move_to(self.slider_axis)  # 初始化滑块位置（根据初始值计算）

        # 为滑块添加拖拽监听器，绑定到拖拽处理方法
        self.slider.add_mouse_drag_listner(self.slider_on_mouse_drag)
        # 调用父类构造方法，传入初始值和图形元素
        super().__init__(value, self.bar, self.slider, self.slider_axis, **kwargs)

    # 验证值是否在有效范围内。
    def assert_value(self, value: float) -> None:
        assert self.min_value <= value <= self.max_value

    # 根据数值更新滑块位置（无动画过渡，直接定位）。
    def set_value_anim(self, value: float) -> None:
        # 计算数值在[min, max]范围内的比例（0到1）
        prop = (value - self.min_value) / (self.max_value - self.min_value)
        # 根据比例获取滑块轴上的对应点，并移动滑块
        self.slider.move_to(self.slider_axis.point_from_proportion(prop))

    # 滑块拖拽事件处理方法，通过鼠标位置计算并更新数值。
    def slider_on_mouse_drag(self, mob, event_data: dict[str, np.ndarray]) -> bool:
        # 从鼠标位置计算对应数值并更新
        self.set_value(self.get_value_from_point(event_data["point"]))
        return False

    # Helper Methods
    # 辅助方法
    def get_value_from_point(self, point: np.ndarray) -> float:
        """
        将场景中的点坐标转换为滑块对应的数值（考虑步长限制）。

        参数:
            point (np.ndarray): 场景中的三维点坐标

        返回:
            float: 转换后的数值（已按步长对齐）
        """
        # 获取滑块轴的起点和终点
        start, end = self.slider_axis.get_start_and_end()
        # 计算点在滑块轴上的投影点
        point_on_line = get_closest_point_on_line(start, end, point)
        # 计算投影点在滑块轴上的比例（0到1）
        prop = get_norm(point_on_line - start) / get_norm(end - start)
        # 根据比例计算原始数值
        value = self.min_value + prop * (self.max_value - self.min_value)
        # 按步长对齐数值（四舍五入到最近的步长点）
        no_of_steps = int((value - self.min_value) / self.step)
        value_nearest_to_step = self.min_value + no_of_steps * self.step
        return value_nearest_to_step


class ColorSliders(Group):
    """
    RGB+透明度颜色调节滑块组，继承自Group。
    通过4个线性滑块分别控制红（R）、绿（G）、蓝（B）三原色和透明度（A），
    并实时显示当前选择的颜色，支持自定义样式和初始值。
    """
    def __init__(
        self,
        sliders_kwargs: dict = {},
        rect_kwargs: dict = {
            "width": 2.0,          # 颜色显示框宽度
            "height": 0.5,         # 颜色显示框高度
            "stroke_opacity": 1.0  # 颜色显示框边框透明度
        },
        background_grid_kwargs: dict = {
            "colors": [GREY_A, GREY_C],  # 背景网格配色
            "single_square_len": 0.1     # 背景网格单个方块边长
        },
        sliders_buff: float = MED_LARGE_BUFF,  # 滑块之间的间距
        default_rgb_value: int = 255,          # RGB通道默认值（0-255）
        default_a_value: int = 1,              # 透明度默认值（0-1）
        **kwargs
    ):
        """初始化颜色调节滑块组。"""
        # 存储配置参数
        self.sliders_kwargs = sliders_kwargs
        self.rect_kwargs = rect_kwargs
        self.background_grid_kwargs = background_grid_kwargs
        self.sliders_buff = sliders_buff
        self.default_rgb_value = default_rgb_value
        self.default_a_value = default_a_value

        # 1. 配置并创建4个滑块（R/G/B/A）
        # RGB滑块配置：范围0-255，步长1（整数调节）
        rgb_kwargs = {"value": self.default_rgb_value, "min_value": 0, "max_value": 255, "step": 1}
        # 透明度滑块配置：范围0-1，步长0.04（25级调节）
        a_kwargs = {"value": self.default_a_value, "min_value": 0, "max_value": 1, "step": 0.04}

        # 创建滑块并合并通用配置
        self.r_slider = LinearNumberSlider(**self.sliders_kwargs, **rgb_kwargs)
        self.g_slider = LinearNumberSlider(**self.sliders_kwargs, **rgb_kwargs)
        self.b_slider = LinearNumberSlider(**self.sliders_kwargs, **rgb_kwargs)
        self.a_slider = LinearNumberSlider(**self.sliders_kwargs, **a_kwargs)
        # 将滑块组合并垂直排列
        self.sliders = Group(
            self.r_slider,
            self.g_slider,
            self.b_slider,
            self.a_slider
        )
        self.sliders.arrange(DOWN, buff=self.sliders_buff)

        # 2. 为滑块设置标志性颜色（便于区分功能）
        self.r_slider.slider.set_color(RED)  # R滑块→红色
        self.g_slider.slider.set_color(GREEN)  # G滑块→绿色
        self.b_slider.slider.set_color(BLUE)  # B滑块→蓝色
        self.a_slider.slider.set_color_by_gradient(BLACK, WHITE)  # A滑块→黑白渐变（代表透明度）

        # 3. 创建颜色显示框（实时显示当前选择的颜色）
        self.selected_color_box = Rectangle(**self.rect_kwargs)
        # 添加更新器：滑块值变化时，同步更新显示框颜色和透明度
        self.selected_color_box.add_updater(
            lambda mob: mob.set_fill(
                self.get_picked_color(),  # 获取当前RGB颜色
                self.get_picked_opacity()  # 获取当前透明度
            )
        )
        # 4. 创建颜色显示框的背景（网格背景，便于观察透明度）
        self.background = self.get_background()

        # 5. 初始化父类Group，组合背景+颜色框+滑块
        super().__init__(
            Group(self.background, self.selected_color_box).fix_in_frame(),
            self.sliders,
            **kwargs
        )
        # 6. 整体垂直排列（颜色显示区在上，滑块组在下）
        self.arrange(DOWN)


    # 为颜色显示框创建网格背景（用于直观观察透明度），返回由小方块组成的网格组。
    def get_background(self) -> VGroup:
        # 从配置中提取网格参数
        single_square_len = self.background_grid_kwargs["single_square_len"]  # 单个方块边长
        colors = self.background_grid_kwargs["colors"]  # 网格交替颜色（如[GREY_A, GREY_C]）
        width = self.rect_kwargs["width"]  # 背景网格总宽度（与颜色显示框一致）
        height = self.rect_kwargs["height"]  # 背景网格总高度（与颜色显示框一致）

        # 计算网格的行数和列数（确保能铺满整个背景区域）
        rows = int(height / single_square_len)  # 行数 = 总高度 ÷ 单个方块边长
        cols = int(width / single_square_len)  # 列数 = 总宽度 ÷ 单个方块边长
        # 确保列数为奇数：避免边缘方块显示不完整，让网格对称
        cols = (cols + 1) if (cols % 2 == 0) else cols

        # 1. 创建单个小方块模板
        single_square = Square(single_square_len)
        # 2. 基于模板生成完整网格（n_rows行、n_cols列，无间距）
        grid = single_square.get_grid(n_rows=rows, n_cols=cols, buff=0.0)
        # 3. 拉伸网格以精确匹配颜色显示框的尺寸
        grid.stretch_to_fit_width(width)
        grid.stretch_to_fit_height(height)
        # 4. 将网格移动到颜色显示框的位置（与显示框对齐）
        grid.move_to(self.selected_color_box)

        # 5. 为网格方块设置交替颜色（类似棋盘格，便于观察透明度）
        for idx, square in enumerate(grid):
            # 确保每个子元素都是Square类型
            assert isinstance(square, Square)
            # 隐藏方块边框（只保留填充色）
            square.set_stroke(width=0.0, opacity=0.0)
            # 按索引交替设置颜色（idx % len(colors)实现循环取色）
            square.set_fill(colors[idx % len(colors)], 1.0)

        return grid

    # 同时设置RGB和透明度的值，同步更新所有滑块位置。
    def set_value(self, r: float, g: float, b: float, a: float):
        self.r_slider.set_value(r)
        self.g_slider.set_value(g)
        self.b_slider.set_value(b)
        self.a_slider.set_value(a)

    # 获取当前选择的RGBA值（归一化到0-1范围）。
    def get_value(self) -> np.ndarary:
        # 将RGB值从0-255范围转换为0-1范围
        r = self.r_slider.get_value() / 255
        g = self.g_slider.get_value() / 255
        b = self.b_slider.get_value() / 255
        # 透明度直接使用0-1范围的值
        alpha = self.a_slider.get_value()
        return np.array((r, g, b, alpha))

    # 获取当前选择的颜色的十六进制字符串表示（如"#FF0080"）。
    def get_picked_color(self) -> str:
        rgba = self.get_value()
        # 取前三个RGB通道值，转换为十六进制格式
        return rgb_to_hex(rgba[:3])

    # 获取当前选择的透明度值。
    def get_picked_opacity(self) -> float:
        rgba = self.get_value()
        return rgba[3]


class Textbox(ControlMobject):
    """
    可交互文本输入框控件，继承自ControlMobject。
    支持点击激活/取消激活，通过键盘输入修改文本内容，并有激活/未激活状态的视觉区分。
    """
    def __init__(
        self,
        value: str = "",
        value_type: np.dtype = np.dtype(object),
        box_kwargs: dict = {
            "width": 2.0,          # 输入框宽度
            "height": 1.0,         # 输入框高度
            "fill_color": DEFAULT_MOBJECT_COLOR,  # 输入框填充色
            "fill_opacity": 1.0,   # 输入框填充透明度
        },
        text_kwargs: dict = {
            "color": BLUE          # 文本颜色
        },
        text_buff: float = MED_SMALL_BUFF,  # 文本与输入框边缘的间距
        isInitiallyActive: bool = False,  # 初始是否处于激活状态
        active_color: ManimColor = BLUE,  # 激活状态的颜色（如边框色）
        deactive_color: ManimColor = RED,  # 未激活状态的颜色（如边框色）
        **kwargs
    ):
        """
        初始化文本输入框控件。

        参数:
            value (str): 初始文本内容，默认为空字符串。
            value_type (np.dtype): 值类型，固定为object（用于文本）。
            box_kwargs (dict): 输入框（矩形）的样式配置。
            text_kwargs (dict): 文本的样式配置（如字体、大小、颜色）。
            text_buff (float): 文本与输入框内壁的缓冲距离，避免文本贴边。
            isInitiallyActive: 初始状态是否激活，激活后才能接收键盘输入。
            active_color: 激活状态下的视觉颜色（通常用于边框）。
            deactive_color: 未激活状态下的视觉颜色（通常用于边框）。
            **kwargs: 传递给父类ControlMobject的额外参数。
        """
        # 存储配置参数与状态
        self.value_type = value_type
        self.box_kwargs = box_kwargs
        self.text_kwargs = text_kwargs
        self.text_buff = text_buff
        self.isInitiallyActive = isInitiallyActive
        self.active_color = active_color
        self.deactive_color = deactive_color

        # 记录当前激活状态（激活时可接收键盘输入）
        self.isActive = self.isInitiallyActive
        # 1. 创建输入框容器（矩形）
        self.box = Rectangle(**self.box_kwargs)
        # 为输入框添加鼠标按压监听器，用于切换激活/未激活状态
        self.box.add_mouse_press_listner(self.box_on_mouse_press)
        # 2. 创建文本对象（显示输入内容）
        self.text = Text(value, **self.text_kwargs)
        # 3. 调用父类构造方法，传入初始值与图形元素
        super().__init__(value, self.box, self.text, **kwargs)
        # 4. 初始化文本显示（确保文本在输入框内居中且不超出）
        self.update_text(value)
        # 5. 初始化状态视觉效果（激活/未激活的颜色区分）
        self.active_anim(self.isActive)
        # 6. 添加键盘按压监听器，用于处理文本输入
        self.add_key_press_listner(self.on_key_press)

    # 文本值变化时的动画更新方法，通过更新文本显示实现值的同步。
    def set_value_anim(self, value: str) -> None:
        self.update_text(value)

    # 更新文本内容并调整布局，确保文本在输入框内正确显示。
    def update_text(self, value: str) -> None:
        text = self.text
        # 移除当前文本对象（准备替换）
        self.remove(text)
        text.__init__(value, **self.text_kwargs)
        height = text.get_height()
        text.set_width(self.box.get_width() - 2 * self.text_buff)
        if text.get_height() > height:
            text.set_height(height)
        # 添加更新器：确保文本始终居中于输入框
        text.add_updater(lambda mob: mob.move_to(self.box))
        text.fix_in_frame()
        # 将新文本添加到控件中
        self.add(text)

    # 根据激活状态更新输入框边框颜色，提供直观的状态反馈。
    def active_anim(self, isActive: bool) -> None:
        if isActive:
            # 激活状态：使用激活颜色
            self.box.set_stroke(self.active_color)
        else:
            # 未激活状态：使用未激活颜色
            self.box.set_stroke(self.deactive_color)

    # 输入框鼠标按压事件处理：切换激活/未激活状态，并同步更新视觉效果。
    def box_on_mouse_press(self, mob, event_data) -> bool:
        # 切换激活状态（激活→未激活，未激活→激活）
        self.isActive = not self.isActive
        # 根据新状态更新输入框视觉样式（边框颜色）
        self.active_anim(self.isActive)
        return False

    # 键盘按压事件处理：仅在激活状态下，根据按键类型生成对应文本，更新输入框内容。
    def on_key_press(self, mob: Mobject, event_data: dict[str, int]) -> bool | None:
        # 从键盘事件数据中提取按键的键码（如字母A、退格键等的编码）
        symbol = event_data["symbol"]
        # 从键盘事件数据中提取修饰键状态（如是否按下Shift、CapsLock等）
        modifiers = event_data["modifiers"]
        # 将键码转换为对应的字符（如键码对应字母A，则转换为'a'或'A'）
        char = chr(symbol)

        # 仅当文本框处于激活状态时，才处理键盘输入
        if mob.isActive:
            # 获取文本框当前的文本内容
            old_value = mob.get_value()
            # 初始化新文本为当前文本（后续根据按键修改）
            new_value = old_value

            # 处理字母/数字键（判断字符是否为字母或数字）
            if char.isalnum():
                # 若按下Shift键或开启CapsLock，添加大写字符；否则添加小写字符
                if (modifiers & PygletWindowKeys.MOD_SHIFT) or (modifiers & PygletWindowKeys.MOD_CAPSLOCK):
                    new_value = old_value + char.upper()
                else:
                    new_value = old_value + char.lower()
            # 处理空格键：在当前文本后添加空格字符
            elif symbol in [PygletWindowKeys.SPACE]:
                new_value = old_value + char
            # 处理Tab键：在当前文本后添加制表符（'\t'）
            elif symbol == PygletWindowKeys.TAB:
                new_value = old_value + '\t'
            # 处理退格键：删除当前文本的最后一个字符，若文本为空则保持空字符串
            elif symbol == PygletWindowKeys.BACKSPACE:
                new_value = old_value[:-1] or ''
            # 将更新后的文本设置为文本框的新值（触发文本显示更新）
            mob.set_value(new_value)
            return False  # 返回False，标识事件已处理且不阻止后续事件传播


class ControlPanel(Group):
    def __init__(
        self,
        *controls: ControlMobject,
        panel_kwargs: dict = {
            "width": FRAME_WIDTH / 4,
            "height": MED_SMALL_BUFF + FRAME_HEIGHT,
            "fill_color": GREY_C,
            "fill_opacity": 1.0,
            "stroke_width": 0.0
        },
        opener_kwargs: dict = {
            "width": FRAME_WIDTH / 8,
            "height": 0.5,
            "fill_color": GREY_C,
            "fill_opacity": 1.0
        },
        opener_text_kwargs: dict = {
            "text": "Control Panel",
            "font_size": 20
        },
        **kwargs
    ):
        # 存储控制面板、开启器及文本的样式配置参数
        self.panel_kwargs = panel_kwargs
        self.opener_kwargs = opener_kwargs
        self.opener_text_kwargs = opener_text_kwargs

        # 创建控制面板主体（灰色矩形）
        self.panel = Rectangle(**self.panel_kwargs)
        # 将控制面板定位到帧的左上角（UP+LEFT方向），无额外间距
        self.panel.to_corner(UP + LEFT, buff=0)
        # 向上偏移控制面板自身高度的距离（初始状态可能为隐藏或错位，后续会调整）
        self.panel.shift(self.panel.get_height() * UP)
        # 为控制面板添加鼠标滚动监听器，绑定滚动处理方法
        self.panel.add_mouse_scroll_listner(self.panel_on_mouse_scroll)

        # 创建控制面板开启器的矩形背景
        self.panel_opener_rect = Rectangle(**self.opener_kwargs)
        # 创建开启器上的文本（显示"Control Panel"）
        self.panel_info_text = Text(**self.opener_text_kwargs)
        # 将文本移动到开启器矩形的中心，实现文本与矩形对齐
        self.panel_info_text.move_to(self.panel_opener_rect)

        # 将开启器矩形与文本组合成开启器整体
        self.panel_opener = Group(self.panel_opener_rect, self.panel_info_text)
        # 将开启器定位到控制面板的正下方，与控制面板下边缘对齐
        self.panel_opener.next_to(self.panel, DOWN, aligned_edge=DOWN)
        # 为开启器添加鼠标拖拽监听器，绑定拖拽处理方法（用于拖动控制面板）
        self.panel_opener.add_mouse_drag_listner(self.panel_opener_on_mouse_drag)

        # 将传入的所有控制组件（如滑块、按钮等）组合成一个组
        self.controls = Group(*controls)
        # 控制组件在组内垂直向下排列，不居中对齐，以原点为对齐边（保持左侧对齐等）
        self.controls.arrange(DOWN, center=False, aligned_edge=ORIGIN)
        # 将控制组件组移动到控制面板的中心，实现组件在面板内居中
        self.controls.move_to(self.panel)

        # 调用父类Group的初始化方法，将控制面板、开启器、控制组件添加到当前组
        super().__init__(
            self.panel, self.panel_opener,
            self.controls,
            **kwargs
        )

        # 调用方法，将控制面板和控制组件移动到与开启器对齐的位置（初始化位置校准）
        self.move_panel_and_controls_to_panel_opener()
        # 将整个控制面板组固定在帧中，避免随相机移动而改变位置
        self.fix_in_frame()

    def move_panel_and_controls_to_panel_opener(self) -> None:
        # 将控制面板主体（panel）移动到开启器矩形（panel_opener_rect）的正上方
        # 方向为UP（向上），间距buff=0（无间隙，面板底部与开启器顶部紧贴）
        self.panel.next_to(
            self.panel_opener_rect,
            direction=UP,
            buff=0
        )

        # 记录控制组件组（controls）当前的X轴坐标（保持水平位置不变）
        controls_old_x = self.controls.get_x()
        # 将控制组件组移动到开启器矩形的正上方，间距为MED_SMALL_BUFF（预留小间隙）
        self.controls.next_to(
            self.panel_opener_rect,
            direction=UP,
            buff=MED_SMALL_BUFF
        )

        # 恢复控制组件组的X轴坐标（确保水平位置不偏移，仅调整垂直位置）
        self.controls.set_x(controls_old_x)

    def add_controls(self, *new_controls: ControlMobject) -> None:
        # 向控制组件组（controls）中添加新的控制组件（如滑块、按钮等）
        self.controls.add(*new_controls)
        # 重新调整面板和控制组件的位置，确保新增组件后布局仍与开启器对齐
        self.move_panel_and_controls_to_panel_opener()

    def remove_controls(self, *controls_to_remove: ControlMobject) -> None:
        # 从控制组件组（controls）中移除指定的控制组件
        self.controls.remove(*controls_to_remove)
        # 重新调整面板和控制组件的位置，确保移除组件后布局仍与开启器对齐
        self.move_panel_and_controls_to_panel_opener()

    def open_panel(self):
        # 记录开启器当前的X轴坐标（保持水平位置不变）
        panel_opener_x = self.panel_opener.get_x()
        # 将开启器移动到左下角（DOWN+LEFT方向），无额外间距
        self.panel_opener.to_corner(DOWN + LEFT, buff=0.0)
        # 恢复开启器的X轴坐标（确保水平位置不变，仅调整垂直位置）
        self.panel_opener.set_x(panel_opener_x)
        # 重新调整面板和控制组件位置，使其与移动后的开启器对齐
        self.move_panel_and_controls_to_panel_opener()
        return self

    def close_panel(self):
        # 记录开启器当前的X轴坐标（保持水平位置不变）
        panel_opener_x = self.panel_opener.get_x()
        # 将开启器移动到左上角（UP+LEFT方向），无额外间距
        self.panel_opener.to_corner(UP + LEFT, buff=0.0)
        # 恢复开启器的X轴坐标（确保水平位置不变，仅调整垂直位置）
        self.panel_opener.set_x(panel_opener_x)
        # 重新调整面板和控制组件位置，使其与移动后的开启器对齐
        self.move_panel_and_controls_to_panel_opener()
        return self

    def panel_opener_on_mouse_drag(self, mob, event_data: dict[str, np.ndarray]) -> bool:
        # 从事件数据中获取鼠标当前位置
        point = event_data["point"]
        # 使开启器的Y坐标与鼠标位置的Y坐标匹配（仅在垂直方向拖动）
        self.panel_opener.match_y(Dot(point))
        # 拖动后重新调整面板和控制组件位置，保持与开启器对齐
        self.move_panel_and_controls_to_panel_opener()
        return False

    def panel_on_mouse_scroll(self, mob, event_data: dict[str, np.ndarray]) -> bool:
        # 从事件数据中获取滚动偏移量（通常是一个二维数组，y方向为[1]或[-1]）
        offset = event_data["offset"]
        # 根据滚动方向计算移动距离（放大滚动效果，使操作更明显）
        factor = 10 * offset[1]
        # 沿Y轴移动控制组件组（上滚上移，下滚下移，实现控件滚动浏览）
        self.controls.set_y(self.controls.get_y() + factor)
        return False
