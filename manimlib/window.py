from __future__ import annotations

import numpy as np

# 导入现代 OpenGL 窗口相关依赖：用于创建硬件加速的渲染窗口
import moderngl_window as mglw
from moderngl_window.context.pyglet.window import Window as PygletWindow  # Pyglet 后端窗口基类
from moderngl_window.timers.clock import Timer  # 计时器：用于动画时间管理
from functools import wraps  # 函数装饰器工具（暂未使用）
import screeninfo  # 屏幕信息库：获取显示器分辨率、位置等

# 导入 Manim 核心常量：用于窗口尺寸与比例计算
from manimlib.constants import ASPECT_RATIO  # 场景宽高比（与渲染分辨率一致）
from manimlib.constants import FRAME_SHAPE  # 场景帧尺寸（虚拟坐标系，如 (16, 9)）

from typing import TYPE_CHECKING

# 仅静态类型检查阶段导入类型注解（避免运行时依赖）
if TYPE_CHECKING:
    from typing import Callable, TypeVar, Optional
    from manimlib.scene.scene import Scene  # 场景类：窗口关联的渲染场景

    T = TypeVar("T")  # 通用类型变量（暂未使用）


class Window(PygletWindow):
    """
    ManimGL 自定义窗口类：继承自 moderngl_window 的 Pyglet 后端窗口，
    适配 Manim 场景渲染需求，支持窗口位置/尺寸自动计算、多显示器适配、场景关联等功能，
    是动画交互式预览的核心载体。
    """
    # 窗口默认配置（类属性，全局生效）
    fullscreen: bool = False  # 默认非全屏
    resizable: bool = True    # 窗口可调整大小
    gl_version: tuple[int, int] = (3, 3)  # 使用 OpenGL 3.3 版本（兼容主流硬件）
    vsync: bool = True        # 开启垂直同步（避免画面撕裂）
    cursor: bool = True       # 显示鼠标光标

    def __init__(
        self,
        scene: Optional[Scene] = None,  # 关联的场景实例（初始可空，后续可通过 init_for_scene 绑定）
        position_string: str = "UR",    # 窗口初始位置字符串（如 "UR"=右上、"DL"=左下、"OO"=居中）
        monitor_index: int = 1,         # 目标显示器索引（多显示器时选择，0 为默认显示器）
        full_screen: bool = False,      # 是否全屏显示（覆盖默认配置）
        size: Optional[tuple[int, int]] = None,  # 窗口自定义尺寸（像素，如 (1920, 1080)）
        position: Optional[tuple[int, int]] = None,  # 窗口自定义位置（像素，屏幕坐标系）
        samples: int = 0                # 抗锯齿采样数（0=无抗锯齿，值越高画面越平滑但性能消耗越大）
    ):
        self.scene = scene  # 绑定场景（后续渲染画面从场景获取）
        self.monitor = self.get_monitor(monitor_index)  # 获取目标显示器信息
        # 窗口默认尺寸：优先使用自定义尺寸，否则根据是否全屏计算（全屏=显示器尺寸，非全屏=显示器一半宽度）
        self.default_size = size or self.get_default_size(full_screen)
        # 窗口默认位置：优先使用自定义位置，否则根据 position_string 计算（如 "UR" 对应右上角落）
        self.default_position = position or self.position_from_string(position_string)
        self.pressed_keys = set()  # 记录当前按下的键盘按键（用于交互事件处理）

        # 调用父类构造函数：初始化 OpenGL 上下文、窗口实例
        super().__init__(samples=samples)
        # 移动窗口到默认位置
        self.to_default_position()

        # 若初始化时绑定了场景，为场景初始化窗口相关配置
        if self.scene:
            self.init_for_scene(scene)

    def init_for_scene(self, scene: Scene):
        """
        为指定场景初始化窗口状态：重置按键记录、绑定场景、初始化 OpenGL 上下文，
        支持窗口复用（如场景重载后无需重新创建窗口，仅更新关联场景即可）。
        
        参数：scene - 待绑定的场景实例
        """
        self.pressed_keys.clear()  # 重置按键记录（避免上一个场景的按键状态影响当前）
        self._has_undrawn_event = True  # 标记存在未绘制事件（触发首次渲染）

        self.scene = scene  # 绑定当前场景
        self.title = str(scene)  # 设置窗口标题为场景名称（如 "SquareScene"）

        self.init_mgl_context()  # 初始化 ModernGL 渲染上下文（父类方法）

        # 初始化计时器：用于动画时间同步（控制帧速率、动画进度）
        self.timer = Timer()
        # 创建窗口配置对象：关联 OpenGL 上下文、窗口、计时器
        self.config = mglw.WindowConfig(ctx=self.ctx, wnd=self, timer=self.timer)
        # 激活当前窗口的 OpenGL 上下文（确保渲染命令指向当前窗口）
        mglw.activate_context(window=self, ctx=self.ctx)
        self.timer.start()  # 启动计时器

        # 触发窗口 resize 事件：同步视口尺寸（确保渲染画面适配窗口大小）
        self.on_resize(*self.size)

    def get_monitor(self, index):
        """
        获取指定索引的显示器信息（支持多显示器适配），获取失败时返回默认显示器配置。
        
        参数：index - 显示器索引（0 开始）
        返回：screeninfo.Monitor 对象（包含显示器宽度、高度、位置等信息）
        """
        try:
            monitors = screeninfo.get_monitors()  # 获取所有可用显示器
            # 取索引对应的显示器（避免索引越界，取最小值）
            return monitors[min(index, len(monitors) - 1)]
        except screeninfo.ScreenInfoError:
            # 获取失败时返回默认配置（1920x1080 显示器）
            return screeninfo.Monitor(width=1920, height=1080)

    def get_default_size(self, full_screen=False):
        """
        计算窗口默认尺寸：全屏模式使用显示器完整尺寸，非全屏模式使用显示器一半宽度（保持场景宽高比）。
        
        参数：full_screen - 是否全屏
        返回：tuple[int, int] - 窗口尺寸（像素）
        """
        # 宽度：全屏=显示器宽度，非全屏=显示器宽度//2
        width = self.monitor.width // (1 if full_screen else 2)
        # 高度：根据场景宽高比计算（确保画面不拉伸）
        height = int(width // ASPECT_RATIO)
        return (width, height)

    def position_from_string(self, position_string):
        """
        根据位置字符串计算窗口初始位置（简化窗口定位，无需手动输入像素坐标）。
        
        位置字符串规则：两位字符，第一位控制垂直方向（U=上、O=中、D=下），第二位控制水平方向（L=左、O=中、R=右），
        示例："UR"=右上、"OO"=居中、"DL"=左下、"UO"=上中。
        
        参数：position_string - 位置字符串（如 "UR"、"OO"）
        返回：tuple[int, int] - 窗口左上角像素坐标
        """
        # 字符映射：将方向字符转换为比例系数（0=靠对应边缘，1=居中，2=靠对侧边缘）
        char_to_n = {"L": 0, "U": 0, "O": 1, "R": 2, "D": 2}
        size = self.default_size  # 窗口默认尺寸
        # 计算窗口与显示器的尺寸差（用于居中/靠边定位）
        width_diff = self.monitor.width - size[0]
        height_diff = self.monitor.height - size[1]
        # 计算水平偏移：L=0（靠左）、O=width_diff//2（居中）、R=width_diff（靠右）
        x_step = char_to_n[position_string[1]] * width_diff // 2
        # 计算垂直偏移：U=0（靠上）、O=height_diff//2（居中）、D=height_diff（靠下）
        # 注：窗口坐标系 Y 轴向下，显示器坐标系 Y 轴向上，故取负号转换
        y_step = char_to_n[position_string[0]] * height_diff // 2
        # 窗口最终位置 = 显示器起始位置 + 偏移量
        return (self.monitor.x + x_step, -self.monitor.y + y_step)

    def focus(self):
        """
        将窗口置于前台（获取焦点）：通过隐藏再显示的 workaround 实现（Pyglet 原生 activate 方法效果不稳定）。
        注：可能会产生轻微闪烁，且窗口位置可能小幅偏移，但能可靠获取焦点。
        """
        self._window.set_visible(False)  # 隐藏窗口
        self._window.set_visible(True)   # 显示窗口

    def to_default_position(self):
        """
        将窗口移动到默认位置，并通过小幅调整尺寸触发窗口重绘（解决部分环境下窗口显示异常的问题）。
        """
        self.position = self.default_position  # 移动到默认位置
        # 尺寸调整 hack：解决窗口初始显示不全的问题（先缩小1像素，再恢复原尺寸）
        w, h = self.default_size
        self.size = (w - 1, h - 1)
        self.size = (w, h)
    # Delegate event handling to scene
    def pixel_coords_to_space_coords(
        self,
        px: int,
        py: int,
        relative: bool = False
    ) -> np.ndarray:
        if self.scene is None or not hasattr(self.scene, "frame"):
            return np.zeros(3)

        pixel_shape = np.array(self.size)
        fixed_frame_shape = np.array(FRAME_SHAPE)
        frame = self.scene.frame

        coords = np.zeros(3)
        coords[:2] = (fixed_frame_shape / pixel_shape) * np.array([px, py])
        if not relative:
            coords[:2] -= 0.5 * fixed_frame_shape
        return frame.from_fixed_frame_point(coords, relative)

    def has_undrawn_event(self) -> bool:
        return self._has_undrawn_event

    def swap_buffers(self):
        super().swap_buffers()
        self._has_undrawn_event = False

    @staticmethod
    def note_undrawn_event(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            func(self, *args, **kwargs)
            self._has_undrawn_event = True
        return wrapper

    @note_undrawn_event
    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int) -> None:
        super().on_mouse_motion(x, y, dx, dy)
        if not self.scene:
            return
        point = self.pixel_coords_to_space_coords(x, y)
        d_point = self.pixel_coords_to_space_coords(dx, dy, relative=True)
        self.scene.on_mouse_motion(point, d_point)

    @note_undrawn_event
    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int) -> None:
        super().on_mouse_drag(x, y, dx, dy, buttons, modifiers)
        if not self.scene:
            return
        point = self.pixel_coords_to_space_coords(x, y)
        d_point = self.pixel_coords_to_space_coords(dx, dy, relative=True)
        self.scene.on_mouse_drag(point, d_point, buttons, modifiers)

    @note_undrawn_event
    def on_mouse_press(self, x: int, y: int, button: int, mods: int) -> None:
        super().on_mouse_press(x, y, button, mods)
        if not self.scene:
            return
        point = self.pixel_coords_to_space_coords(x, y)
        self.scene.on_mouse_press(point, button, mods)

    @note_undrawn_event
    def on_mouse_release(self, x: int, y: int, button: int, mods: int) -> None:
        super().on_mouse_release(x, y, button, mods)
        if not self.scene:
            return
        point = self.pixel_coords_to_space_coords(x, y)
        self.scene.on_mouse_release(point, button, mods)

    @note_undrawn_event
    def on_mouse_scroll(self, x: int, y: int, x_offset: float, y_offset: float) -> None:
        super().on_mouse_scroll(x, y, x_offset, y_offset)
        if not self.scene:
            return
        point = self.pixel_coords_to_space_coords(x, y)
        offset = self.pixel_coords_to_space_coords(x_offset, y_offset, relative=True)
        self.scene.on_mouse_scroll(point, offset, x_offset, y_offset)

    @note_undrawn_event
    def on_key_press(self, symbol: int, modifiers: int) -> None:
        self.pressed_keys.add(symbol)  # Modifiers?
        super().on_key_press(symbol, modifiers)
        if not self.scene:
            return
        self.scene.on_key_press(symbol, modifiers)

    @note_undrawn_event
    def on_key_release(self, symbol: int, modifiers: int) -> None:
        self.pressed_keys.difference_update({symbol})  # Modifiers?
        super().on_key_release(symbol, modifiers)
        if not self.scene:
            return
        self.scene.on_key_release(symbol, modifiers)

    @note_undrawn_event
    def on_resize(self, width: int, height: int) -> None:
        super().on_resize(width, height)
        if not self.scene:
            return
        self.scene.on_resize(width, height)

    @note_undrawn_event
    def on_show(self) -> None:
        super().on_show()
        if not self.scene:
            return
        self.scene.on_show()

    @note_undrawn_event
    def on_hide(self) -> None:
        super().on_hide()
        if not self.scene:
            return
        self.scene.on_hide()

    @note_undrawn_event
    def on_close(self) -> None:
        super().on_close()
        if not self.scene:
            return
        self.scene.on_close()

    def is_key_pressed(self, symbol: int) -> bool:
        return (symbol in self.pressed_keys)
