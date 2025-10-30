# 从__future__模块导入annotations特性，支持更灵活的类型注解（如字符串形式的类型引用）
from __future__ import annotations

# 导入numpy库并简写为np，用于数值计算和数组操作
import numpy as np

# 导入moderngl_window库，这是一个现代OpenGL窗口管理库
import moderngl_window as mglw
# 从moderngl_window的pyglet上下文模块导入Window类（重命名为PygletWindow），用于基于pyglet的窗口管理
from moderngl_window.context.pyglet.window import Window as PygletWindow
# 从moderngl_window的计时器模块导入Clock类，用于时间跟踪
from moderngl_window.timers.clock import Timer
# 从functools导入wraps装饰器，用于包装函数时保留原函数元信息
from functools import wraps
# 导入screeninfo库，用于获取显示器信息（如分辨率、位置等）
import screeninfo

# 从manimlib的常量模块导入纵横比和帧形状常量
from manimlib.constants import ASPECT_RATIO
from manimlib.constants import FRAME_SHAPE

# 从typing模块导入TYPE_CHECKING常量，用于条件类型导入（仅在类型检查时执行）
from typing import TYPE_CHECKING

# 仅在类型检查阶段执行以下代码（运行时不执行）
if TYPE_CHECKING:
    # 从typing模块导入所需的类型注解工具
    from typing import Callable, TypeVar, Optional
    # 从manimlib的scene模块导入Scene类，用于场景相关的类型注解
    from manimlib.scene.scene import Scene

    # 定义一个泛型类型变量T，用于泛型函数或类的类型注解
    T = TypeVar("T")


class Window(PygletWindow):
    # 窗口默认属性：不全屏、可调整大小、使用OpenGL 3.3、开启垂直同步、显示鼠标光标
    fullscreen: bool = False
    resizable: bool = True
    gl_version: tuple[int, int] = (3, 3)
    vsync: bool = True
    cursor: bool = True

    def __init__(
        self,
        scene: Optional[Scene] = None,
        position_string: str = "UR",
        monitor_index: int = 1,
        full_screen: bool = False,
        size: Optional[tuple[int, int]] = None,
        position: Optional[tuple[int, int]] = None,
        samples: int = 0
    ):
        self.scene = scene  # 关联的场景对象
        self.monitor = self.get_monitor(monitor_index)  # 获取指定索引的显示器信息
        # 确定窗口默认大小（全屏则使用显示器尺寸，否则为显示器一半宽度并保持纵横比）
        self.default_size = size or self.get_default_size(full_screen)
        # 确定窗口默认位置（通过位置字符串或直接指定）
        self.default_position = position or self.position_from_string(position_string)
        self.pressed_keys = set()  # 记录当前按下的键

        super().__init__(samples=samples)  # 调用父类构造函数，设置抗锯齿采样数
        self.to_default_position()  # 将窗口移动到默认位置

        if self.scene:
            self.init_for_scene(scene)  # 若有场景，初始化窗口以适配场景

    def init_for_scene(self, scene: Scene):
        """
        重置窗口状态并更新关联的场景。

        当调用`scene.reload()`创建新场景实例后，需要复用现有窗口时，此方法很有必要。
        """
        self.pressed_keys.clear()  # 清空按键记录
        self._has_undrawn_event = True  # 标记存在未绘制的事件

        self.scene = scene  # 更新关联的场景
        self.title = str(scene)  # 设置窗口标题为场景名称

        self.init_mgl_context()  # 初始化ModernGL上下文

        self.timer = Timer()  # 创建计时器
        # 配置窗口上下文（关联OpenGL上下文、窗口和计时器）
        self.config = mglw.WindowConfig(ctx=self.ctx, wnd=self, timer=self.timer)
        mglw.activate_context(window=self, ctx=self.ctx)  # 激活上下文
        self.timer.start()  # 启动计时器

        # 触发 resize 事件以同步视口
        self.on_resize(*self.size)

    def get_monitor(self, index):
        """获取指定索引的显示器信息，失败时返回默认显示器参数"""
        try:
            monitors = screeninfo.get_monitors()  # 获取所有显示器信息
            # 返回索引对应的显示器（索引超出范围则返回最后一个）
            return monitors[min(index, len(monitors) - 1)]
        except screeninfo.ScreenInfoError:
            # 异常时返回默认显示器参数（1920x1080）
            return screeninfo.Monitor(width=1920, height=1080)

    def get_default_size(self, full_screen=False):
        """根据是否全屏，计算窗口默认大小（保持Manim的纵横比）"""
        # 全屏时宽度为显示器宽度，否则为一半
        width = self.monitor.width // (1 if full_screen else 2)
        # 高度根据宽度和预设纵横比计算
        height = int(width // ASPECT_RATIO)
        return (width, height)

    def position_from_string(self, position_string):
        """
        根据位置字符串（如"UR"、"DL"等）计算窗口在显示器上的位置。
        字符含义：L(左)/R(右)、U(上)/D(下)、O(中)
        """
        # 字符到系数的映射：左/上为0，中为1，右/下为2
        char_to_n = {"L": 0, "U": 0, "O": 1, "R": 2, "D": 2}
        size = self.default_size  # 窗口大小
        # 计算显示器与窗口的宽高差
        width_diff = self.monitor.width - size[0]
        height_diff = self.monitor.height - size[1]
        # 根据位置字符串计算x和y方向的偏移（基于宽高差的比例）
        x_step = char_to_n[position_string[1]] * width_diff // 2
        y_step = char_to_n[position_string[0]] * height_diff // 2
        # 计算最终位置（注意y坐标可能需要调整符号，因系统坐标原点可能在左上角）
        return (self.monitor.x + x_step, -self.monitor.y + y_step)

    def focus(self):
        """
        将焦点置于当前窗口（通过隐藏再显示的方式）。

        注意：pyglet的`activate()`方法效果不佳，因此使用此变通方法。
        可能会导致窗口轻微闪烁，位置也可能略有偏移。
        """
        self._window.set_visible(False)
        self._window.set_visible(True)

    def to_default_position(self):
        """将窗口移动到默认位置，并通过微调大小确保显示正确"""
        self.position = self.default_position
        #  hack：有时在独立窗口模式下，需要调整大小才能正确显示
        w, h = self.default_size
        self.size = (w - 1, h - 1)
        self.size = (w, h)

    # 将事件处理委托给场景

    def pixel_coords_to_space_coords(
        self,
        px: int,
        py: int,
        relative: bool = False
    ) -> np.ndarray:
        """
        将像素坐标转换为场景空间坐标。

        参数：
            px, py: 像素坐标
            relative: 是否为相对坐标（True时返回偏移量）
        """
        if self.scene is None or not hasattr(self.scene, "frame"):
            return np.zeros(3)  # 无场景或帧时返回原点

        pixel_shape = np.array(self.size)  # 窗口像素尺寸
        fixed_frame_shape = np.array(FRAME_SHAPE)  # 固定帧尺寸
        frame = self.scene.frame  # 场景的帧对象

        coords = np.zeros(3)
        # 像素坐标转换为固定帧坐标
        coords[:2] = (fixed_frame_shape / pixel_shape) * np.array([px, py])
        if not relative:
            # 非相对坐标时，转换为以帧中心为原点的坐标
            coords[:2] -= 0.5 * fixed_frame_shape
        # 从固定帧坐标转换为场景空间坐标
        return frame.from_fixed_frame_point(coords, relative)

    def has_undrawn_event(self) -> bool:
        """返回是否存在未绘制的事件"""
        return self._has_undrawn_event

    def swap_buffers(self):
        """交换缓冲区（绘制完成），并标记所有事件已绘制"""
        super().swap_buffers()
        self._has_undrawn_event = False

    @staticmethod
    def note_undrawn_event(func: Callable[..., T]) -> Callable[..., T]:
        """装饰器：标记事件为未绘制状态"""
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            func(self, *args, **kwargs)  # 执行原事件处理函数
            self._has_undrawn_event = True  # 标记有未绘制事件
        return wrapper

    @note_undrawn_event
    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int) -> None:
        """鼠标移动事件：转发给场景处理"""
        super().on_mouse_motion(x, y, dx, dy)
        if not self.scene:
            return
        # 转换像素坐标为场景坐标
        point = self.pixel_coords_to_space_coords(x, y)
        d_point = self.pixel_coords_to_space_coords(dx, dy, relative=True)
        self.scene.on_mouse_motion(point, d_point)

    @note_undrawn_event
    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int) -> None:
        """鼠标拖拽事件：转发给场景处理"""
        super().on_mouse_drag(x, y, dx, dy, buttons, modifiers)
        if not self.scene:
            return
        point = self.pixel_coords_to_space_coords(x, y)
        d_point = self.pixel_coords_to_space_coords(dx, dy, relative=True)
        self.scene.on_mouse_drag(point, d_point, buttons, modifiers)

    @note_undrawn_event
    def on_mouse_press(self, x: int, y: int, button: int, mods: int) -> None:
        """鼠标按下事件：转发给场景处理"""
        super().on_mouse_press(x, y, button, mods)
        if not self.scene:
            return
        point = self.pixel_coords_to_space_coords(x, y)
        self.scene.on_mouse_press(point, button, mods)

    @note_undrawn_event
    def on_mouse_release(self, x: int, y: int, button: int, mods: int) -> None:
        """鼠标释放事件：转发给场景处理"""
        super().on_mouse_release(x, y, button, mods)
        if not self.scene:
            return
        point = self.pixel_coords_to_space_coords(x, y)
        self.scene.on_mouse_release(point, button, mods)

    @note_undrawn_event
    def on_mouse_scroll(self, x: int, y: int, x_offset: float, y_offset: float) -> None:
        """鼠标滚轮事件：转发给场景处理"""
        super().on_mouse_scroll(x, y, x_offset, y_offset)
        if not self.scene:
            return
        point = self.pixel_coords_to_space_coords(x, y)
        offset = self.pixel_coords_to_space_coords(x_offset, y_offset, relative=True)
        self.scene.on_mouse_scroll(point, offset, x_offset, y_offset)

    @note_undrawn_event
    def on_key_press(self, symbol: int, modifiers: int) -> None:
        """键盘按下事件：记录按键并转发给场景处理"""
        self.pressed_keys.add(symbol)  # 记录按下的键（暂不处理修饰键）
        super().on_key_press(symbol, modifiers)
        if not self.scene:
            return
        self.scene.on_key_press(symbol, modifiers)

    @note_undrawn_event
    def on_key_release(self, symbol: int, modifiers: int) -> None:
        """键盘释放事件：移除按键记录并转发给场景处理"""
        self.pressed_keys.difference_update({symbol})  # 移除释放的键
        super().on_key_release(symbol, modifiers)
        if not self.scene:
            return
        self.scene.on_key_release(symbol, modifiers)

    @note_undrawn_event
    def on_resize(self, width: int, height: int) -> None:
        """窗口大小改变事件：转发给场景处理"""
        super().on_resize(width, height)
        if not self.scene:
            return
        self.scene.on_resize(width, height)

    @note_undrawn_event
    def on_show(self) -> None:
        """窗口显示事件：转发给场景处理"""
        super().on_show()
        if not self.scene:
            return
        self.scene.on_show()

    @note_undrawn_event
    def on_hide(self) -> None:
        """窗口隐藏事件：转发给场景处理"""
        super().on_hide()
        if not self.scene:
            return
        self.scene.on_hide()

    @note_undrawn_event
    def on_close(self) -> None:
        """窗口关闭事件：转发给场景处理"""
        super().on_close()
        if not self.scene:
            return
        self.scene.on_close()

    def is_key_pressed(self, symbol: int) -> bool:
        """检查指定按键是否处于按下状态"""
        return (symbol in self.pressed_keys)