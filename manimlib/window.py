from __future__ import annotations

import numpy as np

import moderngl_window as mglw
from moderngl_window.context.pyglet.window import Window as PygletWindow
from moderngl_window.timers.clock import Timer
from functools import wraps
import screeninfo

from manimlib.constants import ASPECT_RATIO
from manimlib.constants import FRAME_SHAPE

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable, TypeVar, Optional
    from manimlib.scene.scene import Scene

    T = TypeVar("T")


class Window(PygletWindow):
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
        self.scene = scene
        self.monitor = self.get_monitor(monitor_index)
        self.default_size = size or self.get_default_size(full_screen)
        self.default_position = position or self.position_from_string(position_string)
        self.pressed_keys = set()

        super().__init__(samples=samples)
        self.to_default_position()

        if self.scene:
            self.init_for_scene(scene)

    def init_for_scene(self, scene: Scene):
        """
        Resets the state and updates the scene associated to this window.

        This is necessary when we want to reuse an *existing* window after a
        `scene.reload()` was requested, which will create new scene instances.
        """
        self.pressed_keys.clear()
        self._has_undrawn_event = True

        self.scene = scene
        self.title = str(scene)

        self.init_mgl_context()

        self.timer = Timer()
        self.config = mglw.WindowConfig(ctx=self.ctx, wnd=self, timer=self.timer)
        mglw.activate_context(window=self, ctx=self.ctx)
        self.timer.start()

        # This line seems to resync the viewport
        self.on_resize(*self.size)

    def get_monitor(self, index):
        try:
            monitors = screeninfo.get_monitors()
            return monitors[min(index, len(monitors) - 1)]
        except screeninfo.ScreenInfoError:
            # Default fallback
            return screeninfo.Monitor(width=1920, height=1080)

    def get_default_size(self, full_screen=False):
        width = self.monitor.width // (1 if full_screen else 2)
        height = int(width // ASPECT_RATIO)
        return (width, height)

    def position_from_string(self, position_string):
        # Alternatively, it might be specified with a string like
        # UR, OO, DL, etc. specifying what corner it should go to
        char_to_n = {"L": 0, "U": 0, "O": 1, "R": 2, "D": 2}
        size = self.default_size
        width_diff = self.monitor.width - size[0]
        height_diff = self.monitor.height - size[1]
        x_step = char_to_n[position_string[1]] * width_diff // 2
        y_step = char_to_n[position_string[0]] * height_diff // 2
        return (self.monitor.x + x_step, -self.monitor.y + y_step)

    def focus(self):
        """
        Puts focus on this window by hiding and showing it again.

        Note that the pyglet `activate()` method didn't work as expected here,
        so that's why we have to use this workaround. This will produce a small
        flicker on the window but at least reliably focuses it. It may also
        offset the window position slightly.
        """
        self._window.set_visible(False)
        self._window.set_visible(True)

    def to_default_position(self):
        self.position = self.default_position
        # Hack. Sometimes, namely when configured to open in a separate window,
        # the window needs to be resized to display correctly.
        w, h = self.default_size
        self.size = (w - 1, h - 1)
        self.size = (w, h)

    def pixel_coords_to_space_coords(
        self,
        px: int,
        py: int,
        relative: bool = False
    ) -> np.ndarray:
        """
        将窗口像素坐标转换为场景虚拟空间坐标（核心：统一交互坐标系统，避免像素与虚拟单位混淆）。
        
        逻辑：
        1. 若未绑定场景或场景无 frame，返回原点（避免报错）；
        2. 计算像素与虚拟空间的缩放比例（固定帧尺寸 / 窗口像素尺寸）；
        3. 非相对坐标：减去帧尺寸的一半（将窗口左上角像素坐标转为场景中心为原点的坐标）；
        4. 通过场景 frame 转换为最终空间坐标（支持 3D 视角变换）。
        
        参数：
            px : 窗口像素 X 坐标（窗口左上角为原点，向右为正）；
            py : 窗口像素 Y 坐标（窗口左上角为原点，向下为正）；
            relative : 是否为相对位移（True 时不做原点偏移，直接返回缩放后的坐标）。
        返回：np.ndarray（3D 空间坐标，[x, y, 0]，适配 2D/3D 场景）。
        """
        # 未绑定场景或无 frame，返回原点
        if self.scene is None or not hasattr(self.scene, "frame"):
            return np.zeros(3)

        # 获取窗口像素尺寸和场景固定帧尺寸（虚拟单位）
        pixel_shape = np.array(self.size)
        fixed_frame_shape = np.array(FRAME_SHAPE)
        frame = self.scene.frame  # 场景的相机帧（控制视角）

        # 初始化 3D 坐标（Z 轴默认为 0）
        coords = np.zeros(3)
        # 像素坐标 → 虚拟空间坐标（应用缩放比例）
        coords[:2] = (fixed_frame_shape / pixel_shape) * np.array([px, py])
        # 非相对坐标：将窗口左上角原点转为场景中心原点
        if not relative:
            coords[:2] -= 0.5 * fixed_frame_shape
        # 转换为 frame 对应的空间坐标（支持 3D 视角变换）
        return frame.from_fixed_frame_point(coords, relative)

    def has_undrawn_event(self) -> bool:
        """
        检查是否存在未绘制的事件（如鼠标移动、按键按下），用于控制窗口重绘时机（避免无效渲染）。
        
        返回：布尔值，True 表示存在未绘制事件，需要触发重绘。
        """
        return self._has_undrawn_event

    def swap_buffers(self):
        """
        交换窗口缓冲区（OpenGL 渲染核心步骤）：将后台渲染好的帧显示到前台，同时标记无未绘制事件。
        
        逻辑：调用父类缓冲区交换方法，之后重置 `_has_undrawn_event` 为 False（表示当前帧已绘制）。
        """
        super().swap_buffers()
        self._has_undrawn_event = False

    @staticmethod
    def note_undrawn_event(func: Callable[..., T]) -> Callable[..., T]:
        """
        事件装饰器：标记触发事件的方法为“需要重绘”（设置 `_has_undrawn_event=True`）。
        
        作用：所有被装饰的事件方法（如鼠标移动、按键）触发后，自动标记窗口需要重绘，确保交互反馈实时显示。
        
        参数：func - 待装饰的事件处理方法（如 on_mouse_motion）
        返回：装饰后的方法（执行原逻辑后标记未绘制事件）。
        """
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            func(self, *args, **kwargs)  # 执行原事件处理逻辑
            self._has_undrawn_event = True  # 标记需要重绘
        return wrapper

    # ------------------------------ 鼠标事件处理（装饰器标记需重绘） ------------------------------
    @note_undrawn_event
    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int) -> None:
        """
        鼠标移动事件回调：将像素坐标转为场景空间坐标，转发给场景的 `on_mouse_motion` 方法。
        
        参数：
            x/y : 鼠标当前像素坐标；
            dx/dy : 鼠标相对于上一帧的像素位移。
        """
        super().on_mouse_motion(x, y, dx, dy)  # 调用父类事件处理
        if not self.scene:
            return
        # 像素坐标 → 场景空间坐标
        point = self.pixel_coords_to_space_coords(x, y)
        d_point = self.pixel_coords_to_space_coords(dx, dy, relative=True)
        # 转发事件给场景
        self.scene.on_mouse_motion(point, d_point)

    @note_undrawn_event
    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int) -> None:
        """
        鼠标拖动事件回调：转发给场景的 `on_mouse_drag` 方法（支持拖动平移、旋转等交互）。
        
        参数：
            x/y : 鼠标当前像素坐标；
            dx/dy : 鼠标位移像素；
            buttons : 按下的鼠标按键（如左键=1、右键=2）；
            modifiers : 按下的修饰键（如 Shift=1、Ctrl=2）。
        """
        super().on_mouse_drag(x, y, dx, dy, buttons, modifiers)
        if not self.scene:
            return
        point = self.pixel_coords_to_space_coords(x, y)
        d_point = self.pixel_coords_to_space_coords(dx, dy, relative=True)
        self.scene.on_mouse_drag(point, d_point, buttons, modifiers)

    @note_undrawn_event
    def on_mouse_press(self, x: int, y: int, button: int, mods: int) -> None:
        """
        鼠标按下事件回调：转发给场景的 `on_mouse_press` 方法（如记录拖动起点）。
        
        参数：
            x/y : 鼠标按下位置像素坐标；
            button : 按下的鼠标按键；
            mods : 按下的修饰键。
        """
        super().on_mouse_press(x, y, button, mods)
        if not self.scene:
            return
        point = self.pixel_coords_to_space_coords(x, y)
        self.scene.on_mouse_press(point, button, mods)

    @note_undrawn_event
    def on_mouse_release(self, x: int, y: int, button: int, mods: int) -> None:
        """
        鼠标释放事件回调：转发给场景的 `on_mouse_release` 方法（如结束拖动）。
        """
        super().on_mouse_release(x, y, button, mods)
        if not self.scene:
            return
        point = self.pixel_coords_to_space_coords(x, y)
        self.scene.on_mouse_release(point, button, mods)

    @note_undrawn_event
    def on_mouse_scroll(self, x: int, y: int, x_offset: float, y_offset: float) -> None:
        """
        鼠标滚轮事件回调：转发给场景的 `on_mouse_scroll` 方法（如缩放画面）。
        
        参数：
            x/y : 鼠标当前像素坐标；
            x_offset/y_offset : 滚轮水平/垂直像素偏移（正为上滚/右滚，负为下滚/左滚）。
        """
        super().on_mouse_scroll(x, y, x_offset, y_offset)
        if not self.scene:
            return
        point = self.pixel_coords_to_space_coords(x, y)
        offset = self.pixel_coords_to_space_coords(x_offset, y_offset, relative=True)
        self.scene.on_mouse_scroll(point, offset, x_offset, y_offset)

    # ------------------------------ 键盘事件处理 ------------------------------
    @note_undrawn_event
    def on_key_press(self, symbol: int, modifiers: int) -> None:
        """
        键盘按下事件回调：记录按下的按键，转发给场景的 `on_key_press` 方法（如快捷键响应）。
        
        参数：
            symbol : 按键的 ASCII 码（如 'a'=97、ESC=27）；
            modifiers : 按下的修饰键。
        """
        self.pressed_keys.add(symbol)  # 记录当前按下的按键
        super().on_key_press(symbol, modifiers)
        if not self.scene:
            return
        self.scene.on_key_press(symbol, modifiers)

    @note_undrawn_event
    def on_key_release(self, symbol: int, modifiers: int) -> None:
        """
        键盘释放事件回调：移除记录的按键，转发给场景的 `on_key_release` 方法。
        """
        self.pressed_keys.difference_update({symbol})  # 从按下集合中移除该按键
        super().on_key_release(symbol, modifiers)
        if not self.scene:
            return
        self.scene.on_key_release(symbol, modifiers)

    # ------------------------------ 窗口状态事件处理 ------------------------------
    @note_undrawn_event
    def on_resize(self, width: int, height: int) -> None:
        """
        窗口调整大小事件回调：转发给场景的 `on_resize` 方法（如适配新分辨率）。
        """
        super().on_resize(width, height)
        if not self.scene:
            return
        self.scene.on_resize(width, height)

    @note_undrawn_event
    def on_show(self) -> None:
        """
        窗口显示事件回调：转发给场景的 `on_show` 方法（如加载临时资源）。
        """
        super().on_show()
        if not self.scene:
            return
        self.scene.on_show()

    @note_undrawn_event
    def on_hide(self) -> None:
        """
        窗口隐藏事件回调：转发给场景的 `on_hide` 方法（如暂停动画）。
        """
        super().on_hide()
        if not self.scene:
            return
        self.scene.on_hide()

    @note_undrawn_event
    def on_close(self) -> None:
        """
        窗口关闭事件回调：转发给场景的 `on_close` 方法（如清理资源）。
        """
        super().on_close()
        if not self.scene:
            return
        self.scene.on_close()

    def is_key_pressed(self, symbol: int) -> bool:
        """
        检查指定按键是否处于按下状态（供场景判断快捷键组合，如 Ctrl+Z）。
        
        参数：symbol - 按键的 ASCII 码
        返回：布尔值，True 表示按键当前处于按下状态。
        """
        return (symbol in self.pressed_keys)