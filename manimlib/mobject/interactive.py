from __future__ import annotations

import numpy as np
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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable
    from manimlib.typing import ManimColor


# Interactive Mobjects（交互式图形对象）

class MotionMobject(Mobject):
    """
        可拖动的交互式图形对象：支持鼠标按住并拖动到任意位置。
        核心是为传入的基础图形添加鼠标拖动监听，实现实时位置跟随。
    """
    def __init__(self, mobject: Mobject, **kwargs):
        super().__init__(**kwargs)
        # 断言传入的必须是Mobject实例，确保基础图形合法
        assert isinstance(mobject, Mobject)
        self.mobject = mobject  # 存储基础图形对象
        # 为基础图形添加鼠标拖动监听器，拖动时触发mob_on_mouse_drag方法
        self.mobject.add_mouse_drag_listner(self.mob_on_mouse_drag)
        # 添加空更新器：避免基础图形被标记为“静态对象”，确保拖动功能生效
        self.mobject.add_updater(lambda mob: None)
        self.add(mobject)  # 将基础图形添加到当前MotionMobject中

    def mob_on_mouse_drag(self, mob: Mobject, event_data: dict[str, np.ndarray]) -> bool:
        # 拖动时的核心逻辑：将基础图形移动到鼠标当前位置（event_data["point"]为鼠标坐标）
        mob.move_to(event_data["point"])
        return False  # 返回False，表示不阻止后续事件传递


class Button(Mobject):
    """
        交互式按钮对象：传入任意基础图形，注册点击回调方法，点击时触发自定义逻辑。
        点击回调方法需接收“基础图形”作为参数，类似更新器（updater）的参数形式。
    """

    def __init__(self, mobject: Mobject, on_click: Callable[[Mobject]], **kwargs):
        super().__init__(**kwargs)
        # 断言传入的必须是Mobject实例，确保按钮的视觉载体合法
        assert isinstance(mobject, Mobject)
        self.on_click = on_click  # 存储点击回调方法
        self.mobject = mobject    # 存储按钮的视觉载体（基础图形）
        # 为视觉载体添加鼠标按压监听器，按压时触发mob_on_mouse_press方法
        self.mobject.add_mouse_press_listner(self.mob_on_mouse_press)
        self.add(mobject)  # 将视觉载体添加到当前Button中

    def mob_on_mouse_press(self, mob: Mobject, event_data) -> bool:
        # 点击时的核心逻辑：调用注册的回调方法，传入按钮的视觉载体
        self.on_click(mob)
        return False  # 返回False，表示不阻止后续事件传递


# Controls（控制类图形对象）

class ControlMobject(ValueTracker):
    """
    带视觉载体的数值控制对象：继承自ValueTracker（数值跟踪器），可关联多个图形对象（Mobject），
    实现“数值变化+视觉反馈”联动，且默认固定在画面中（不随场景相机移动）。
    """
    def __init__(self, value: float, *mobjects: Mobject, **kwargs):
        # 调用父类ValueTracker的初始化方法，传入初始数值
        super().__init__(value=value, **kwargs)
        # 将传入的所有图形对象添加为自身的子对象，作为控制组件的视觉载体
        self.add(*mobjects)

        # 添加空更新器：避免控制对象在场景等待时被标记为“静态数据锁定”，确保交互功能生效
        self.add_updater(lambda mob: None)
        # 固定控制对象在画面坐标系中：不随相机的平移、旋转而移动，始终显示在固定位置
        self.fix_in_frame()

    def set_value(self, value: float):
        """
        重写数值设置方法：先验证数值合法性，再执行数值变化的动画，最后更新跟踪的数值。
        """
        self.assert_value(value)          # 验证数值（子类需实现具体规则，如范围限制）
        self.set_value_anim(value)        # 执行数值变化的视觉动画（子类需实现具体效果）
        return ValueTracker.set_value(self, value)  # 调用父类方法，更新跟踪的核心数值

    def assert_value(self, value):
        """
        数值验证抽象方法：子类需重写此方法，实现对输入数值的合法性检查（如范围、类型）。
        示例：限制数值在0-100之间，非法数值抛出异常或自动修正。
        """
        # To be implemented in subclasses（需在子类中实现）
        pass

    def set_value_anim(self, value):
        """
        数值变化动画抽象方法：子类需重写此方法，实现数值变化时的视觉反馈动画。
        示例：滑动条随数值移动、数值显示文本更新等。
        """
        # To be implemented in subclasses（需在子类中实现）
        pass


class EnableDisableButton(ControlMobject):
    """
    启用/禁用切换按钮：继承自ControlMobject，是带视觉反馈的布尔值控制器，
    通过点击切换状态（启用/禁用），并以不同颜色区分两种状态。
    """
    def __init__(
        self,
        value: bool = True,  # 初始状态：True为启用，False为禁用，默认启用
        value_type: np.dtype = np.dtype(bool),  # 数值类型，固定为布尔型
        rect_kwargs: dict = {  # 按钮视觉载体（矩形）的配置参数
            "width": 0.5,
            "height": 0.5,
            "fill_opacity": 1.0
        },
        enable_color: ManimColor = GREEN,  # 启用状态的填充色，默认绿色
        disable_color: ManimColor = RED,   # 禁用状态的填充色，默认红色
        **kwargs
    ):
        # 存储按钮核心配置参数
        self.value = value
        self.value_type = value_type
        self.rect_kwargs = rect_kwargs
        self.enable_color = enable_color
        self.disable_color = disable_color

        # 创建按钮的视觉载体：矩形（可通过rect_kwargs自定义尺寸、透明度）
        self.box = Rectangle(**self.rect_kwargs)
        # 调用父类ControlMobject的初始化方法：传入初始值和视觉载体
        super().__init__(value, self.box, **kwargs)
        # 为按钮添加鼠标按压监听器：点击时触发状态切换
        self.add_mouse_press_listner(self.on_mouse_press)

    def assert_value(self, value: bool) -> None:
        """
        数值验证：确保传入的状态值是布尔类型（True/False），非法类型直接报错。
        """
        assert isinstance(value, bool), "EnableDisableButton only accepts boolean values (True/False)"

    def set_value_anim(self, value: bool) -> None:
        """
        状态切换的视觉动画：根据目标状态更新矩形的填充色，无额外动画时直接切换颜色。
        """
        if value:  # 启用状态：设置为启用色
            self.box.set_fill(self.enable_color)
        else:      # 禁用状态：设置为禁用色
            self.box.set_fill(self.disable_color)

    def toggle_value(self) -> None:
        """
        状态切换方法：获取当前状态并取反，调用父类方法更新数值（自动触发验证和视觉反馈）。
        """
        super().set_value(not self.get_value())

    def on_mouse_press(self, mob: Mobject, event_data) -> bool:
        """
        鼠标按压回调：点击按钮时触发状态切换，实现“点击即切换”的交互逻辑。
        """
        mob.toggle_value()
        return False  # 返回False，不阻止后续事件传递


class Checkbox(ControlMobject):
    """
    复选框控制器：继承自ControlMobject，通过布尔值控制选中（显示对勾）/未选中（显示叉号）状态，
    支持点击切换，包含矩形框作为载体，对勾和叉号作为状态标识。
    """
    def __init__(
        self,
        value: bool = True,  # 初始状态：True为选中（显示对勾），False为未选中（显示叉号），默认选中
        value_type: np.dtype = np.dtype(bool),  # 数值类型，固定为布尔型
        rect_kwargs: dict = {  # 复选框矩形框的配置参数
            "width": 0.5,      # 矩形框宽度，默认0.5
            "height": 0.5,     # 矩形框高度，默认0.5
            "fill_opacity": 0.0  # 矩形框填充透明度，默认0（无填充，仅显示边框）
        },
        checkmark_kwargs: dict = {  # 对勾（选中标识）的配置参数
            "stroke_color": GREEN,  # 对勾描边颜色，默认绿色
            "stroke_width": 6,      # 对勾描边宽度，默认6
        },
        cross_kwargs: dict = {  # 叉号（未选中标识）的配置参数
            "stroke_color": RED,    # 叉号描边颜色，默认红色
            "stroke_width": 6,      # 叉号描边宽度，默认6
        },
        box_content_buff: float = SMALL_BUFF,  # 矩形框与内部标识（对勾/叉号）的间距，默认小间距
        **kwargs
    ):
        # 存储复选框的核心配置参数
        self.value_type = value_type
        self.rect_kwargs = rect_kwargs
        self.checkmark_kwargs = checkmark_kwargs
        self.cross_kwargs = cross_kwargs
        self.box_content_buff = box_content_buff

        # 创建复选框的矩形框载体，按rect_kwargs配置样式
        self.box = Rectangle(**self.rect_kwargs)
        # 根据初始状态创建对应的内部标识：选中则创建对勾，未选中则创建叉号
        self.box_content = self.get_checkmark() if value else self.get_cross()
        # 调用父类ControlMobject的初始化方法：传入初始值、矩形框、内部标识（作为视觉载体）
        super().__init__(value, self.box, self.box_content, **kwargs)
        # 为复选框添加鼠标按压监听器：点击时触发状态切换
        self.add_mouse_press_listner(self.on_mouse_press)

    def assert_value(self, value: bool) -> None:
        """
        数值验证：确保传入的状态值是布尔类型（True/False），非法类型直接抛出断言错误。
        """
        assert isinstance(value, bool)

    def toggle_value(self) -> None:
        """
        状态切换方法：获取当前状态并取反，调用父类set_value方法更新数值（自动触发验证和视觉反馈）。
        """
        super().set_value(not self.get_value())

    def set_value_anim(self, value: bool) -> None:
        """
        状态切换的视觉动画：根据目标状态，将内部标识替换为对勾或叉号（通过become方法实现图形替换）。
        """
        if value:  # 目标状态为选中：将内部标识替换为对勾
            self.box_content.become(self.get_checkmark())
        else:      # 目标状态为未选中：将内部标识替换为叉号
            self.box_content.become(self.get_cross())

    def on_mouse_press(self, mob: Mobject, event_data) -> None:
        """
        鼠标按压回调：点击复选框时，调用toggle_value方法切换状态，实现“点击即切换选中/未选中”的交互逻辑。
        """
        mob.toggle_value()
        return False  # 返回False，不阻止后续事件传递
    
    # Helper methods

    def get_checkmark(self) -> VGroup:
        """
        创建并返回复选框的“对勾”标识：由两条线段组成，适配矩形框尺寸并居中显示。
        返回值为包含两条线段的VGroup（向量组），便于整体操作。
        """
        # 创建对勾的两条线段：第一条从左上到中下，第二条从中下到右上，应用对勾样式配置
        checkmark = VGroup(
            Line(UP / 2 + 2 * LEFT, DOWN + LEFT, **self.checkmark_kwargs),
            Line(DOWN + LEFT, UP + RIGHT, **self.checkmark_kwargs)
        )

        # 调整对勾尺寸以适配矩形框：宽度拉伸至与矩形框相同
        checkmark.stretch_to_fit_width(self.box.get_width())
        # 高度拉伸至与矩形框相同
        checkmark.stretch_to_fit_height(self.box.get_height())
        # 整体缩小至50%（避免对勾过大超出矩形框）
        checkmark.scale(0.5)
        # 将对勾移动到矩形框的中心位置，确保居中显示
        checkmark.move_to(self.box)
        return checkmark

    def get_cross(self) -> VGroup:
        """
        创建并返回复选框的“叉号”标识：由两条交叉线段组成，适配矩形矩形矩形框尺寸并居中显示。
        返回值为包含两条线段的VGroup（向量组），便于整体操作。
        """
        # 创建叉号的两条交叉线段：第一条从左上到右下，第二条从右上到左下，应用叉号样式配置
        cross = VGroup(
            Line(UP + LEFT, DOWN + RIGHT, **self.cross_kwargs),
            Line(UP + RIGHT, DOWN + LEFT, **self.cross_kwargs)
        )

        # 调整叉号尺寸以适配矩形框：宽度拉伸至与矩形框相同
        cross.stretch_to_fit_width(self.box.get_width())
        # 高度拉伸至与矩形框相同
        cross.stretch_to_fit_height(self.box.get_height())
        # 整体缩小至50%（避免叉号过大超出矩形框）
        cross.scale(0.5)
        # 将叉号移动到矩形框的中心位置，确保居中显示
        cross.move_to(self.box)
        return cross


class LinearNumberSlider(ControlMobject):
    """
    线性数值滑动条控制器：继承自ControlMobject，支持在指定数值范围内（min-max）通过拖动滑块调整数值，
    包含圆角矩形（滑动条背景）、圆形（滑块）和隐藏轴线（用于定位滑块），数值变化时滑块同步移动。
    """
    def __init__(
        self,
        value: float = 0,  # 初始数值，默认0
        value_type: type = np.float64,  # 数值类型，默认浮点型
        min_value: float = -10.0,  # 数值范围最小值，默认-10.0
        max_value: float = 10.0,   # 数值范围最大值，默认10.0
        step: float = 1.0,         # 数值调整步长（此处初始化存储，具体生效需结合其他逻辑），默认1.0
        rounded_rect_kwargs: dict = {  # 滑动条背景（圆角矩形）的配置参数
            "height": 0.075,       # 背景高度，默认0.075
            "width": 2,            # 背景宽度，默认2
            "corner_radius": 0.0375  # 背景圆角半径，默认0.0375
        },
        circle_kwargs: dict = {  # 滑块（圆形）的配置参数
            "radius": 0.1,        # 滑块半径，默认0.1
            "stroke_color": GREY_A,  # 滑块描边颜色，默认浅灰色
            "fill_color": GREY_A,    # 滑块填充颜色，默认浅灰色
            "fill_opacity": 1.0      # 滑块填充透明度，默认1.0
        },
        **kwargs
    ):
        # 存储滑动条的核心配置参数
        self.value_type = value_type
        self.min_value = min_value
        self.max_value = max_value
        self.step = step
        self.rounded_rect_kwargs = rounded_rect_kwargs
        self.circle_kwargs = circle_kwargs

        # 创建滑动条背景：圆角矩形，按配置参数设置样式
        self.bar = RoundedRectangle(**self.rounded_rect_kwargs)
        # 创建滑块：圆形，按配置参数设置样式
        self.slider = Circle(**self.circle_kwargs)
        # 创建滑块定位轴线：从背景左侧边界点到右侧边界点，用于确定滑块移动范围
        self.slider_axis = Line(
            start=self.bar.get_bounding_box_point(LEFT),  # 轴线起点：背景左边界点
            end=self.bar.get_bounding_box_point(RIGHT)    # 轴线终点：背景右边界点
        )
        self.slider_axis.set_opacity(0.0)  # 隐藏轴线（仅用于定位，不显示）
        self.slider.move_to(self.slider_axis)  # 将滑块初始移动到轴线上

        # 为滑块添加鼠标拖动监听器：拖动时触发滑块位置更新逻辑
        self.slider.add_mouse_drag_listner(self.slider_on_mouse_drag)

        # 调用父类ControlMobject的初始化方法：传入初始值、背景、滑块、轴线（作为视觉/功能组件）
        super().__init__(value, self.bar, self.slider, self.slider_axis, **kwargs)

    def assert_value(self, value: float) -> None:
        """
        数值验证：确保传入的数值在[min_value, max_value]范围内，超出范围则抛出断言错误。
        """
        assert self.min_value <= value <= self.max_value

    def set_value_anim(self, value: float) -> None:
        """
        数值变化的视觉动画：根据目标数值计算滑块在轴线上的比例位置，将滑块移动到对应位置。
        """
        # 计算目标数值在范围内的比例（0对应min_value，1对应max_value）
        prop = (value - self.min_value) / (self.max_value - self.min_value)
        # 按比例获取轴线上的对应点，将滑块移动到该点
        self.slider.move_to(self.slider_axis.point_from_proportion(prop))

    def slider_on_mouse_drag(self, mob, event_data: dict[str, np.ndarray]) -> bool:
        """
        滑块拖动回调：根据鼠标当前位置（event_data["point"]）计算对应的数值，调用set_value更新数值（同步滑块位置）。
        注：get_value_from_point方法需额外实现，用于将鼠标位置转换为范围内的数值。
        """
        self.set_value(self.get_value_from_point(event_data["point"]))
        return False  # 返回False，不阻止后续事件传递

    # Helper Methods（辅助方法）

    def get_value_from_point(self, point: np.ndarray) -> float:
        """
        根据鼠标点击/拖动的位置计算对应的滑动条数值，确保数值符合步长要求。
        核心逻辑：将点坐标映射到滑动条的数值范围，并取最接近的步长整数倍值。
        
        参数
        -----
        point : np.ndarray
            鼠标在场景中的坐标点（通常来自事件数据event_data["point"]）
        
        返回
        -----
        float
            映射后的数值，确保在[min_value, max_value]范围内且为step的整数倍
        """
        # 获取滑块轴线的起点和终点坐标（即滑动条的左右边界点）
        start, end = self.slider_axis.get_start_and_end()
        # 计算鼠标点在轴线上的最近点（将鼠标位置投影到滑动条的直线上）
        point_on_line = get_closest_point_on_line(start, end, point)
        # 计算该最近点在轴线上的比例（0.0对应起点，1.0对应终点）
        prop = get_norm(point_on_line - start) / get_norm(end - start)
        # 根据比例计算对应的原始数值（线性映射到[min_value, max_value]范围）
        value = self.min_value + prop * (self.max_value - self.min_value)
        # 计算该数值距离最小值有多少个步长（向下取整）
        no_of_steps = int((value - self.min_value) / self.step)
        # 将数值调整为最接近的步长整数倍（确保符合step间隔要求）
        value_nearest_to_step = self.min_value + no_of_steps * self.step
        return value_nearest_to_step


class ColorSliders(Group):
    """
    颜色调节滑块组：继承自Group，整合4个LinearNumberSlider（R/G/B/A通道），
    支持独立调节RGB颜色值和透明度，实时显示当前选中的颜色，并包含背景网格和颜色预览框。
    """
    def __init__(
        self,
        sliders_kwargs: dict = {},  # 传递给每个LinearNumberSlider的额外配置参数，默认空字典
        rect_kwargs: dict = {       # 颜色预览框（矩形）的配置参数
            "width": 2.0,           # 预览框宽度，默认2.0
            "height": 0.5,          # 预览框高度，默认0.5
            "stroke_opacity": 1.0   # 预览框描边透明度，默认1.0（完全显示边框）
        },
        background_grid_kwargs: dict = {  # 背景网格的配置参数
            "colors": [GREY_A, GREY_C],   # 网格交替颜色，默认浅灰和中灰
            "single_square_len": 0.1      # 网格单个方块的边长，默认0.1
        },
        sliders_buff: float = MED_LARGE_BUFF,  # 滑块之间的垂直间距，默认中等大间距
        default_rgb_value: int = 255,          # R/G/B通道的默认初始值，默认255（最亮）
        default_a_value: int = 1,              # A（透明度）通道的默认初始值，默认1（完全不透明）
        **kwargs
    ):
        # 存储颜色滑块组的核心配置参数
        self.sliders_kwargs = sliders_kwargs
        self.rect_kwargs = rect_kwargs
        self.background_grid_kwargs = background_grid_kwargs
        self.sliders_buff = sliders_buff
        self.default_rgb_value = default_rgb_value
        self.default_a_value = default_a_value

        # 配置R/G/B通道滑块的通用参数：值范围0-255，步长1（符合RGB颜色值标准）
        rgb_kwargs = {"value": self.default_rgb_value, "min_value": 0, "max_value": 255, "step": 1}
        # 配置A通道滑块的参数：值范围0-1，步长0.04（透明度从完全透明到不透明）
        a_kwargs = {"value": self.default_a_value, "min_value": 0, "max_value": 1, "step": 0.04}

        # 创建4个线性数值滑块，分别对应R、G、B、A通道，合并通用参数和自定义参数
        self.r_slider = LinearNumberSlider(**self.sliders_kwargs, **rgb_kwargs)  # R通道滑块
        self.g_slider = LinearNumberSlider(**self.sliders_kwargs, **rgb_kwargs)  # G通道滑块
        self.b_slider = LinearNumberSlider(**self.sliders_kwargs, **rgb_kwargs)  # B通道滑块
        self.a_slider = LinearNumberSlider(**self.sliders_kwargs, **a_kwargs)    # A通道滑块
        # 将4个滑块整合为一个Group，便于统一排版
        self.sliders = Group(
            self.r_slider,
            self.g_slider,
            self.b_slider,
            self.a_slider
        )
        # 按垂直方向（DOWN）排列滑块，滑块间间距为sliders_buff
        self.sliders.arrange(DOWN, buff=self.sliders_buff)

        # 为每个滑块设置对应颜色，便于区分通道：R滑块红色、G滑块绿色、B滑块蓝色
        self.r_slider.slider.set_color(RED)
        self.g_slider.slider.set_color(GREEN)
        self.b_slider.slider.set_color(BLUE)
        # A滑块设置黑白渐变颜色，体现透明度调节的特性
        self.a_slider.slider.set_color_by_gradient(BLACK, WHITE)

        # 创建颜色预览框：用于实时显示当前R/G/B/A组合对应的颜色
        self.selected_color_box = Rectangle(**self.rect_kwargs)
        # 为预览框添加更新器：每次滑块数值变化时，同步更新预览框的填充色和透明度
        self.selected_color_box.add_updater(
            lambda mob: mob.set_fill(
                self.get_picked_color(),  # 获取当前选中的RGB颜色（需get_picked_color方法支持）
                self.get_picked_opacity() # 获取当前选中的透明度（需get_picked_opacity方法支持）
            )
        )
        # 创建背景网格（需get_background方法支持），用于衬托预览框，便于观察颜色
        self.background = self.get_background()

        # 调用父类Group的初始化方法：整合背景+预览框（固定在画面中）和滑块组，作为整体组件
        super().__init__(
            Group(self.background, self.selected_color_box).fix_in_frame(),  # 背景和预览框固定显示
            self.sliders,  # 滑块组
            **kwargs
        )

        # 按垂直方向（DOWN）排列整个颜色滑块组的内部元素（背景预览区和滑块区）
        self.arrange(DOWN)

    def get_background(self) -> VGroup:
        """
        创建并返回颜色预览框的背景网格：由交替颜色的小方块组成，便于
        用于增强颜色预览的对比度，使颜色变化更易观察。
        
        返回
        -----
        VGroup
            包含网格方块的向量组，尺寸与颜色预览框匹配
        """
        # 从配置参数中提取网格单个方块的边长和交替颜色
        single_square_len = self.background_grid_kwargs["single_square_len"]
        colors = self.background_grid_kwargs["colors"]
        # 从预览框配置中提取宽度和高度，确保网格尺寸匹配
        width = self.rect_kwargs["width"]
        height = self.rect_kwargs["height"]
        # 计算网格的行数和列数（根据预览框尺寸和方块边长）
        rows = int(height / single_square_len)
        cols = int(width / single_square_len)
        # 确保列数为奇数（使网格图案对称），若为偶数则加1
        cols = (cols + 1) if (cols % 2 == 0) else cols

        # 创建单个方块模板，基于此生成网格
        single_square = Square(single_square_len)
        # 生成指定行列数的网格，方块间无间距距
        grid = single_square.get_grid(n_rows=rows, n_cols=cols, buff=0.0)
        # 拉伸网格以精确匹配预览框的宽度和高度
        grid.stretch_to_fit_width(width)
        grid.stretch_to_fit_height(height)
        # 将网格移动到与颜色预览框相同的位置（居中心对齐）
        grid.move_to(self.selected_color_box)

        # 为网格中的每个方块设置交替颜色（无描边，完全填充）
        for idx, square in enumerate(grid):
            assert isinstance(square, Square)  # 确保每个元素都是Square实例
            square.set_stroke(width=0.0, opacity=0.0)  # 去除描边
            # 按索引标取模循环使用配置的颜色，实现交替效果
            square.set_fill(colors[idx % len(colors)], 1.0)

        return grid

    def set_value(self, r: float, g: float, b: float, a: float):
        """
        同时时设置R、G、B、A四个通道的数值，同步滑块各滑块对应的滑块同步更新。
        
        参数
        -----
        r : float
            红色通道值（0-255）
        g : float
            绿色通道值（0-255）
        b : float
            蓝色通道值（0-255）
        a : float
            透明度值（0-1）
        """
        self.r_slider.set_value(r)  # 更新红色通道滑块
        self.g_slider.set_value(g)  # 更新绿色通道滑块
        self.b_slider.set_value(b)  # 更新蓝色通道滑块
        self.a_slider.set_value(a)  # 更新透明度滑块

    def get_value(self) -> np.ndarray:
        """
        获取当前R、G、B、A通道的归一化数值（R/G/B转为0-1范围）。
        
        返回
        -----
        np.ndarray
            包含(r, g, b, alpha)的数组，前三者范围0-1，透明度范围0-1
        """
        # R/G/B值从0-255范围归一化到0-1
        r = self.r_slider.get_value() / 255
        g = self.g_slider.get_value() / 255
        b = self.b_slider.get_value() / 255
        alpha = self.a_slider.get_value()  # 透明度已在0-1范围
        return np.array((r, g, b, alpha))

    def get_picked_color(self) -> str:
        """
        获取当前选中颜色的十六进制字符串（如"#RRGGBB"）。
        
        返回
        -----
        str
            十六进制颜色码
        """
        rgba = self.get_value()  # 获取归一化的RGBA值
        return rgb_to_hex(rgba[:3])  # 仅取RGB部分转换为十六进制

    def get_picked_opacity(self) -> float:
        """
        获取当前选中的透明度值。
        """
        rgba = self.get_value()  # 获取归一化的RGBA值
        return rgba[3]  # 返回alpha通道值


class Textbox(ControlMobject):
    """
    文本输入框控制器：继承自ControlMobject，具备文本显示、点击激活/取消、键盘输入编辑功能，
    通过矩形框作为载体，支持配置激活/未激活状态颜色，文本尺寸自动适配框体。
    """
    def __init__(
        self,
        value: str = "",  # 初始文本内容，默认空字符串
        value_type: np.dtype = np.dtype(object),  # 数值类型，默认对象类型（适配字符串）
        box_kwargs: dict = {  # 文本框载体（矩形）的配置参数
            "width": 2.0,          # 矩形宽度，默认2.0
            "height": 1.0,         # 矩形高度，默认1.0
            "fill_color": DEFAULT_MOBJECT_COLOR,  # 矩形填充色，默认图形对象颜色
            "fill_opacity": 1.0,   # 矩形填充透明度，默认1.0
        },
        text_kwargs: dict = {  # 文本的配置参数
            "color": BLUE  # 文本颜色，默认蓝色
        },
        text_buff: float = MED_SMALL_BUFF,  # 文本与矩形框的内边距，默认中小间距
        isInitiallyActive: bool = False,    # 初始是否处于激活状态（可输入），默认未激活
        active_color: ManimColor = BLUE,     # 激活状态下矩形框的描边色，默认蓝色
        deactive_color: ManimColor = RED,    # 未激活状态下矩形框的描边色，默认红色
        **kwargs
    ):
        # 存储文本框的核心配置参数
        self.value_type = value_type
        self.box_kwargs = box_kwargs
        self.text_kwargs = text_kwargs
        self.text_buff = text_buff
        self.isInitiallyActive = isInitiallyActive
        self.active_color = active_color
        self.deactive_color = deactive_color

        # 记录当前文本框的激活状态（初始值与isInitiallyActive一致）
        self.isActive = self.isInitiallyActive
        # 创建文本框的矩形载体，按配置参数设置样式
        self.box = Rectangle(**self.box_kwargs)
        # 为矩形框添加鼠标按压监听器：点击切换激活/未激活状态
        self.box.add_mouse_press_listner(self.box_on_mouse_press)
        # 创建初始文本对象，按配置参数设置样式
        self.text = Text(value, **self.text_kwargs)
        # 调用父类ControlMobject的初始化方法：传入初始文本、矩形框、文本（作为视觉/功能组件）
        super().__init__(value, self.box, self.text, **kwargs)
        # 更新文本显示（确保初始文本适配框体尺寸）
        self.update_text(value)
        # 根据初始激活状态设置矩形框描边色（视觉反馈）
        self.active_anim(self.isActive)
        # 为文本框添加键盘按压监听器：激活状态下响应键盘输入（需on_key_press方法支持）
        self.add_key_press_listner(self.on_key_press)

    def set_value_anim(self, value: str) -> None:
        """
        文本值更新的视觉动画：调用update_text方法更新文本显示，实现文本变化的视觉反馈。
        """
        self.update_text(value)

    def update_text(self, value: str) -> None:
        """
        更新文本内容并适配文本框尺寸：移除旧文本，创建新文本，确保文本宽度/高度不超出框体，
        并添加更新器使文本始终居中显示。
        """
        text = self.text  # 记录旧文本对象
        self.remove(text)  # 从文本框中移除旧文本
        # 创建新文本对象，内容为目标值，沿用文本配置参数
        text.__init__(value, **self.text_kwargs)
        height = text.get_height()  # 记录新文本的初始高度
        # 限制文本宽度：不超过文本框宽度减去2倍内边距（避免文本贴边）
        text.set_width(self.box.get_width() - 2 * self.text_buff)
        # 若宽度限制导致文本高度超出初始高度，进一步限制文本高度（避免文本拉伸变形）
        if text.get_height() > height:
            text.set_height(height)
        # 为文本添加更新器：确保文本始终跟随文本框居中显示
        text.add_updater(lambda mob: mob.move_to(self.box))
        text.fix_in_frame()  # 固定文本在画面坐标系中（不随相机移动）
        self.add(text)  # 将新文本添加到文本框中

    def active_anim(self, isActive: bool) -> None:
        """
        激活/未激活状态的视觉动画：根据状态切换矩形框的描边色，提供清晰的状态反馈。
        """
        if isActive:  # 激活状态：设置描边色为激活色
            self.box.set_stroke(self.active_color)
        else:  # 未激活状态：设置描边色为未激活色
            self.box.set_stroke(self.deactive_color)

    def box_on_mouse_press(self, mob, event_data) -> bool:
        """
        文本框矩形框的鼠标按压回调：点击时切换文本框的激活/未激活状态，
        并调用active_anim更新描边色，提供视觉状态反馈。
        """
        # 切换激活状态：当前为激活则变为未激活，反之则激活
        self.isActive = not self.isActive
        # 根据新的激活状态更新矩形框描边色（激活色/未激活色）
        self.active_anim(self.isActive)
        return False

    def on_key_press(self, mob: Mobject, event_data: dict[str, int]) -> bool | None:
        """
        键盘按压回调：仅在文本框激活状态下响应键盘输入，支持字母数字、空格、制表符输入
        和退格键删除，根据Shift/CapsLock修饰键判断字母大小写，最终更新文本框内容。
        """
        # 提取键盘事件中的键位编码和修饰键编码
        symbol = event_data["symbol"]
        modifiers = event_data["modifiers"]
        # 将键位编码转换为对应的字符（如编码对应字母则转为字符）
        char = chr(symbol)
        # 仅在文本框激活状态下处理键盘输入
        if mob.isActive:
            old_value = mob.get_value()  # 获取当前文本内容
            new_value = old_value        # 初始化新文本为当前内容（默认不变）
            
            # 处理字母数字输入：根据Shift或CapsLock判断是否大写
            if char.isalnum():
                # 若按下Shift键或开启CapsLock，新字符为大写
                if (modifiers & PygletWindowKeys.MOD_SHIFT) or (modifiers & PygletWindowKeys.MOD_CAPSLOCK):
                    new_value = old_value + char.upper()
                # 否则为小写
                else:
                    new_value = old_value + char.lower()
            # 处理空格键：添加空格字符
            elif symbol in [PygletWindowKeys.SPACE]:
                new_value = old_value + char
            # 处理制表符：添加制表符（\t）
            elif symbol == PygletWindowKeys.TAB:
                new_value = old_value + '\t'
            # 处理退格键：删除最后一个字符，若为空则保持空字符串
            elif symbol == PygletWindowKeys.BACKSPACE:
                new_value = old_value[:-1] or ''
            # 更新文本框内容（自动触发set_value_anim更新显示）
            mob.set_value(new_value)
            return False  # 返回False，不阻止后续事件传递


class ControlPanel(Group):
    """
    控制面板组件：继承自Group，整合多个ControlMobject（控制器），包含可拖动展开/收起的面板主体、
    面板开启器（带文本标识），支持鼠标滚动调整面板内控制器位置，整体固定在画面中。
    """
    def __init__(
        self,
        *controls: ControlMobject,  # 传入的多个控制器对象（如滑块、按钮等）
        panel_kwargs: dict = {      # 面板主体（矩形）的配置参数
            "width": FRAME_WIDTH / 4,  # 面板宽度，默认帧宽度的1/4
            "height": MED_SMALL_BUFF + FRAME_HEIGHT,  # 面板高度，默认帧高度加中小间距
            "fill_color": GREY_C,      # 面板填充色，默认中灰色
            "fill_opacity": 1.0,       # 面板填充透明度，默认1.0
            "stroke_width": 0.0        # 面板描边宽度，默认0（无描边）
        },
        opener_kwargs: dict = {     # 面板开启器（矩形）的配置参数
            "width": FRAME_WIDTH / 8,  # 开启器宽度，默认帧宽度的1/8
            "height": 0.5,             # 开启器高度，默认0.5
            "fill_color": GREY_C,      # 开启器填充色，默认中灰色
            "fill_opacity": 1.0        # 开启器填充透明度，默认1.0
        },
        opener_text_kwargs: dict = {  # 开启器文本的配置参数
            "text": "Control Panel",  # 文本内容，默认"Control Panel"
            "font_size": 20           # 字体大小，默认20
        },
        **kwargs
    ):
        # 存储控制面板的核心配置参数
        self.panel_kwargs = panel_kwargs
        self.opener_kwargs = opener_kwargs
        self.opener_text_kwargs = opener_text_kwargs

        # 创建面板主体：矩形，按配置参数设置样式
        self.panel = Rectangle(**self.panel_kwargs)
        # 将面板移动到画面左上角落（UP+LEFT），无间距
        self.panel.to_corner(UP + LEFT, buff=0)
        # 向上偏移面板高度的距离（初始处于画面外，通过开启器拖动显示）
        self.panel.shift(self.panel.get_height() * UP)
        # 为面板添加鼠标滚动监听器：滚动时调整面板内控制器位置（需panel_on_mouse_scroll方法支持）
        self.panel.add_mouse_scroll_listner(self.panel_on_mouse_scroll)

        # 创建面板开启器的矩形载体，按配置参数设置样式
        self.panel_opener_rect = Rectangle(**self.opener_kwargs)
        # 创建开启器的文本对象，按配置参数设置样式
        self.panel_info_text = Text(**self.opener_text_kwargs)
        # 将文本移动到开启器矩形中心，实现文本与开启器对齐
        self.panel_info_text.move_to(self.panel_opener_rect)

        # 将开启器矩形和文本整合为一个Group，作为完整的开启器组件
        self.panel_opener = Group(self.panel_opener_rect, self.panel_info_text)
        # 将开启器移动到面板正下方，与面板下边缘对齐
        self.panel_opener.next_to(self.panel, DOWN, aligned_edge=DOWN)
        # 为开启器添加鼠标拖动监听器：拖动时调整面板和开启器位置（需panel_opener_on_mouse_drag方法支持）
        self.panel_opener.add_mouse_drag_listner(self.panel_opener_on_mouse_drag)

        # 将传入的所有控制器整合为一个Group，便于统一排版
        self.controls = Group(*controls)
        # 按垂直方向（DOWN）排列控制器，不居中对齐，按原点对齐（保持左侧整齐）
        self.controls.arrange(DOWN, center=False, aligned_edge=ORIGIN)
        # 将控制器Group移动到面板主体中心，实现控制器在面板内居中
        self.controls.move_to(self.panel)

        # 调用父类Group的初始化方法：整合面板、开启器、控制器，作为整体控制面板
        super().__init__(
            self.panel, self.panel_opener,
            self.controls,
            **kwargs
        )

        # 调整面板和控制器位置，确保与开启器对齐（初始化位置校准）
        self.move_panel_and_controls_to_panel_opener()
        # 固定整个控制面板在画面坐标系中（不随相机的平移、旋转而移动）
        self.fix_in_frame()

    def move_panel_and_controls_to_panel_opener(self) -> None:
        """
        校准面板和控制器位置：将面板移动到开启器正上方（无间距），
        调整控制器水平位置不变，仅垂直方向跟随面板对齐。
        """
        # 将面板移动到开启器正上方，与开启器上边缘对齐，无间距
        self.panel.next_to(
            self.panel_opener_rect,
            direction=UP,
            buff=0
        )

        # 记录控制器当前的水平X坐标（保持水平位置不变）
        controls_old_x = self.controls.get_x()
        # 将控制器移动到开启器正上方，与开启器上边缘保持中小间距
        self.controls.next_to(
            self.panel_opener_rect,
            direction=UP,
            buff=MED_SMALL_BUFF
        )

        # 恢复控制器的水平X坐标，确保仅垂直方向移动，水平位置不变
        self.controls.set_x(controls_old_x)

    def add_controls(self, *new_controls: ControlMobject) -> None:
        """
        向控制面板中添加新的控制器：将新控制器加入控制器Group，
        并调用位置校准方法确保面板与控制器位置匹配。
        
        参数
        -----
        *new_controls : ControlMobject
            一个或多个待添加的ControlMobject实例（如滑块、按钮等）
        """
        self.controls.add(*new_controls)  # 将新控制器添加到控制器Group
        self.move_panel_and_controls_to_panel_opener()  # 校准面板和控制器位置

    def remove_controls(self, *controls_to_remove: ControlMobject) -> None:
        """
        从控制面板中移除指定控制器：从控制器Group中删除目标控制器，
        并调用位置校准方法更新面板与控制器位置。
        
        参数
        -----
        *controls_to_remove : ControlMobject
            一个或多个待移除的ControlMobject实例
        """
        self.controls.remove(*controls_to_remove)  # 从控制器Group中移除目标控制器
        self.move_panel_and_controls_to_panel_opener()  # 校准面板和控制器位置

    def open_panel(self):
        """
        展开控制面板：将面板开启器移动到画面左下角落，带动面板主体显示在画面内，
        保持开启器水平位置不变，最终返回自身以便链式调用。
        
        返回
        -----
        ControlPanel
            自身实例（支持链式调用）
        """
        panel_opener_x = self.panel_opener.get_x()  # 记录开启器当前水平X坐标（保持不变）
        self.panel_opener.to_corner(DOWN + LEFT, buff=0.0)  # 将开启器移动到左下角落
        self.panel_opener.set_x(panel_opener_x)  # 恢复开启器水平X坐标，仅垂直移动
        self.move_panel_and_controls_to_panel_opener()  # 校准面板和控制器位置
        return self

    def close_panel(self):
        """
        收起控制面板：将面板开启器移动到画面左上角落，带动面板主体隐藏到画面外，
        保持开启器水平位置不变，最终返回自身以便链式调用。
        
        返回
        -----
        ControlPanel
            自身实例（支持链式调用）
        """
        panel_opener_x = self.panel_opener.get_x()  # 记录开启器当前水平X坐标（保持不变）
        self.panel_opener.to_corner(UP + LEFT, buff=0.0)  # 将开启器移动到左上角落
        self.panel_opener.set_x(panel_opener_x)  # 恢复开启器水平X坐标，仅垂直移动
        self.move_panel_and_controls_to_panel_opener()  # 校准面板和控制器位置
        return self

    def panel_opener_on_mouse_drag(self, mob, event_data: dict[str, np.ndarray]) -> bool:
        """
        面板开启器的鼠标拖动回调：根据鼠标位置（event_data["point"]）
        同步调整开启器的Y坐标，带动面板和控制器跟随移动，实现面板的自由展开/收起。
        
        参数
        -----
        mob : Mobject
            触发事件的图形对象（此处为面板开启器Group）
        event_data : dict[str, np.ndarray]
            鼠标事件数据，包含"point"（鼠标当前坐标）
        
        返回
        -----
        bool
            返回False，不阻止后续事件传递
        """
        point = event_data["point"]  # 获取鼠标当前坐标
        self.panel_opener.match_y(Dot(point))  # 让开启器的Y坐标与鼠标Y坐标一致
        self.move_panel_and_controls_to_panel_opener()  # 校准面板和控制器位置
        return False

    def panel_on_mouse_scroll(self, mob, event_data: dict[str, np.ndarray]) -> bool:
        """
        面板的鼠标滚动回调：根据滚动偏移量（event_data["offset"]）
        调整面板内控制器的Y坐标，实现控制器的上下滚动查看（适配多控制器场景）。
        
        参数
        -----
        mob : Mobject
            触发事件的图形对象（此处为面板主体）
        event_data : dict[str, np.ndarray]
            鼠标滚动事件数据，包含"offset"（滚动偏移量，垂直方向为offset[1]）
        
        返回
        -----
        bool
            返回False，不阻止后续事件传递
        """
        offset = event_data["offset"]  # 获取滚动偏移量
        factor = 10 * offset[1]  # 计算滚动幅度（放大偏移量，使滚动更明显）
        # 调整控制器的Y坐标：向上滚动（offset[1]为正）时控制器上移，向下滚动则下移
        self.controls.set_y(self.controls.get_y() + factor)
        return False