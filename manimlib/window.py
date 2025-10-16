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
    # ------------------------------ 窗口事件处理与坐标转换方法 ------------------------------
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