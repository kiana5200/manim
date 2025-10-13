# 导入Python未来版本的注解特性（支持字符串形式类名、泛型等灵活类型提示）
from __future__ import annotations

# 导入标准库模块
from collections import OrderedDict  # 有序字典，用于保持键值对插入顺序（如动画队列、状态管理）
import platform  # 获取操作系统信息（适配不同平台的交互逻辑，如键盘、文件打开）
import random  # 生成随机数（可能用于随机化动画参数、对象位置等）
import time  # 时间相关操作（如动画延迟、性能计时）
from functools import wraps  # 函数装饰器工具（用于包装方法，添加额外逻辑如状态检查、日志）
from contextlib import contextmanager  # 上下文管理器工具（用于创建临时资源环境，如临时配置切换）
from contextlib import ExitStack  # 多上下文管理器整合工具（同时管理多个临时资源，确保正确释放）

# 导入第三方库模块
import numpy as np  # 数值计算库（核心用于向量运算、矩阵变换、像素数据处理等）
from tqdm.auto import tqdm as ProgressDisplay  # 进度条工具（可视化动画渲染、帧生成进度）
from pyglet.window import key as PygletWindowKeys  # Pyglet键盘常量（映射键盘按键编码，如MOD_SHIFT、LEFT）

# 导入Manim库核心模块
from manimlib.animation.animation import prepare_animation  # 动画预处理函数（标准化动画参数、处理嵌套动画）
from manimlib.camera.camera import Camera  # 相机类（负责场景渲染、帧捕获、视角控制）
from manimlib.camera.camera_frame import CameraFrame  # 相机帧类（定义相机视野范围、旋转、缩放等属性）
from manimlib.config import manim_config  # Manim全局配置对象（存储渲染参数、路径、交互设置等）
from manimlib.event_handler import EVENT_DISPATCHER  # 事件分发器（管理键盘、鼠标等事件的注册与触发）
from manimlib.event_handler.event_type import EventType  # 事件类型枚举（定义事件分类，如KEY_PRESS、MOUSE_DRAG）
from manimlib.logger import log  # Manim日志工具（打印信息、警告、错误，支持分级输出）
from manimlib.mobject.mobject import _AnimationBuilder  # 动画构建器（简化Mobject对象的动画创建，如obj.animate.move_to()）
from manimlib.mobject.mobject import Group  # 基础组类（用于批量管理非向量图形对象，如3D模型、文本）
from manimlib.mobject.mobject import Mobject  # 所有可见对象的基类（定义图形对象的基础属性与方法，如添加、移除、移动）
from manimlib.mobject.mobject import Point  # 点对象（用于标记位置、计算距离、作为动画锚点等）
from manimlib.mobject.types.vectorized_mobject import VGroup  # 向量组类（批量管理向量图形对象，如矩形、圆，支持向量运算）
from manimlib.mobject.types.vectorized_mobject import VMobject  # 向量图形基类（支持路径绘制、填充、描边等，如Rectangle、Circle）
from manimlib.scene.scene_embed import InteractiveSceneEmbed  # 交互式场景嵌入类（提供IPython终端交互功能）
from manimlib.scene.scene_embed import CheckpointManager  # 检查点管理器（用于保存/恢复场景状态，支持交互式开发）
from manimlib.scene.scene_file_writer import SceneFileWriter  # 场景文件写入器（负责渲染帧保存、视频编码、音频合并）
from manimlib.utils.dict_ops import merge_dicts_recursively  # 字典递归合并工具（用于合并配置字典，处理嵌套结构）
from manimlib.utils.family_ops import extract_mobject_family_members  # 对象家族提取工具（获取对象及其所有子对象，如Group的子Mobject）
from manimlib.utils.family_ops import recursive_mobject_remove  # 对象递归移除工具（从场景中移除对象及其所有子对象，避免残留）
from manimlib.utils.iterables import batch_by_property  # 按属性分批工具（根据对象属性将可迭代对象分组，如按层分组渲染）
from manimlib.utils.sounds import play_sound  # 音频播放工具（播放内置或自定义音效，如动画触发音）
from manimlib.utils.color import color_to_rgba  # 颜色转换工具（将颜色值转换为RGBA格式，用于渲染像素数据）
from manimlib.window import Window  # 窗口类（Manim的可视化窗口，负责显示渲染结果、接收用户输入）

# 导入类型提示相关模块（仅在类型检查时生效，不影响运行时）
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # 导入泛型、可调用对象等基础类型
    from typing import Callable, Iterable, TypeVar, Optional
    from manimlib.typing import Vect3  # Manim自定义的三维向量类型（如(x, y, z)坐标）

    T = TypeVar('T')  # 泛型类型变量（用于定义通用函数/类，如处理任意类型的列表）

    # 导入PIL图像类（用于类型提示，如帧图像保存、像素处理）
    from PIL.Image import Image

    # 导入动画基类（用于类型提示，如动画列表、动画参数）
    from manimlib.animation.animation import Animation


class Scene(object):
    """
    Manim场景基类，所有自定义动画场景均需继承此类。
    核心职责：管理场景状态（如Mobject对象、时间、相机）、动画播放、交互事件、渲染输出，
    是连接图形对象、动画逻辑与输出文件的核心枢纽。
    """
    # 类级静态配置（所有Scene子类共享，可在子类中重写）
    random_seed: int = 0  # 随机种子（确保场景中随机操作可复现，如随机位置生成）
    pan_sensitivity: float = 0.5  # 相机平移灵敏度（鼠标拖拽/键盘控制时的移动速度）
    scroll_sensitivity: float = 20  # 相机缩放灵敏度（鼠标滚轮控制时的缩放速度）
    drag_to_pan: bool = True  # 是否启用“拖拽平移”（鼠标拖拽场景时移动相机）
    max_num_saved_states: int = 50  # 最大状态保存数量（用于undo/redo，避免内存溢出）
    default_camera_config: dict = dict()  # 相机默认配置（子类可扩展，如分辨率、视角）
    default_file_writer_config: dict = dict()  # 文件写入器默认配置（子类可扩展，如输出格式、编码）
    samples: int = 0  # 相机抗锯齿采样数（samples>0启用抗锯齿，值越高画质越好但渲染越慢）
    default_frame_orientation = (0, 0)  # 相机帧默认欧拉角（角度制，控制初始视角方向）

    def __init__(
        self,
        window: Optional[Window] = None,
        camera_config: dict = dict(),
        file_writer_config: dict = dict(),
        skip_animations: bool = False,
        always_update_mobjects: bool = False,
        start_at_animation_number: int | None = None,
        end_at_animation_number: int | None = None,
        show_animation_progress: bool = False,
        leave_progress_bars: bool = False,
        preview_while_skipping: bool = True,
        presenter_mode: bool = False,
        default_wait_time: float = 1.0,
    ):
        """
        初始化Scene实例，整合配置、相机、窗口、文件写入器，建立场景基础环境。
        
        参数说明：
            window: 关联的可视化窗口（None则不创建窗口，仅用于后台渲染）
            camera_config: 相机个性化配置（覆盖默认配置，如fps、分辨率）
            file_writer_config: 文件写入器个性化配置（覆盖默认配置，如输出目录、视频编码）
            skip_animations: 是否跳过动画（True则直接显示动画最终状态，用于快速预览）
            always_update_mobjects: 是否强制Mobject持续更新（True则每帧重新计算对象状态，如动态参数）
            start_at_animation_number: 从指定动画编号开始播放（用于分段渲染，如从第5个动画开始）
            end_at_animation_number: 播放到指定动画编号停止（用于分段渲染，如到第10个动画停止）
            show_animation_progress: 是否显示单个动画的进度条（True则每个动画单独显示进度）
            leave_progress_bars: 动画完成后是否保留进度条（True则进度条不自动清除，便于查看历史）
            preview_while_skipping: 跳过动画时是否显示最终预览（True则快速显示每段动画的结果）
            presenter_mode: 是否启用演示者模式（True则增强交互，如暂停时保持画面更新）
            default_wait_time: 默认等待时间（wait()方法未指定时长时的默认值，单位：秒）
        """
        # 1. 初始化场景行为配置
        self.skip_animations = skip_animations  # 动画跳过开关
        self.always_update_mobjects = always_update_mobjects  # Mobject强制更新开关
        self.start_at_animation_number = start_at_animation_number  # 起始动画编号
        self.end_at_animation_number = end_at_animation_number  # 结束动画编号
        self.show_animation_progress = show_animation_progress  # 单个动画进度条开关
        self.leave_progress_bars = leave_progress_bars  # 进度条保留开关
        self.preview_while_skipping = preview_while_skipping  # 跳过动画预览开关
        self.presenter_mode = presenter_mode  # 演示者模式开关
        self.default_wait_time = default_wait_time  # 默认等待时长

        # 2. 合并相机配置（优先级：实例传入 > 子类默认 > 全局默认）
        self.camera_config = merge_dicts_recursively(
            manim_config.camera,         # 全局相机配置（manimlib.config中定义）
            self.default_camera_config,  # 子类自定义的相机默认配置
            camera_config,               # 实例化时传入的个性化相机配置
        )
        # 3. 合并文件写入器配置（优先级同上）
        self.file_writer_config = merge_dicts_recursively(
            manim_config.file_writer,          # 全局文件写入器配置
            self.default_file_writer_config,   # 子类自定义的写入器默认配置
            file_writer_config,                # 实例化时传入的个性化写入器配置
        )

        # 4. 初始化窗口与相机同步
        self.window = window  # 关联可视化窗口
        if self.window:
            self.window.init_for_scene(self)  # 窗口为场景初始化（如绑定事件处理器）
            self.camera_config["fps"] = 30    # 强制窗口模式下相机帧率为30（匹配大多数显示器）

        # 5. 初始化核心组件（相机、帧、文件写入器）
        # 创建相机（负责渲染场景，关联窗口与采样配置）
        self.camera: Camera = Camera(
            window=self.window,
            samples=self.samples,
            **self.camera_config  # 应用合并后的相机配置
        )
        self.frame: CameraFrame = self.camera.frame  # 相机帧（控制视野范围、旋转、缩放）
        self.frame.reorient(*self.default_frame_orientation)  # 应用默认视角方向
        self.frame.make_orientation_default()  # 将当前视角设为“默认方向”（便于后续重置）
        
        # 创建文件写入器（负责将渲染帧保存为图像/视频）
        self.file_writer = SceneFileWriter(self, **self.file_writer_config)

        # 6. 初始化场景状态管理
        self.mobjects: list[Mobject] = [self.camera.frame]  # 场景中所有Mobject列表（默认包含相机帧）
        self.render_groups: list[Mobject] = []  # 渲染分组列表（用于按组控制渲染顺序/层级）
        self.id_to_mobject_map: dict[int, Mobject] = dict()  # MobjectID到对象的映射（快速查找对象）
        self.num_plays: int = 0  # 动画播放次数计数器（用于分段渲染编号）
        self.time: float = 0  # 场景时间轴（记录从场景开始到当前的总时长，单位：秒）
        self.skip_time: float = 0  # 跳过的时间累计（用于计算实际播放时长）
        self.original_skipping_status: bool = self.skip_animations  # 初始动画跳过状态（用于临时切换后恢复）
        self.undo_stack = []  # 撤销栈（保存场景历史状态，用于undo操作）
        self.redo_stack = []  # 重做栈（保存撤销的状态，用于redo操作）

        # 7. 处理分段渲染的初始状态
        if self.start_at_animation_number is not None:
            self.skip_animations = True  # 从指定动画开始时，默认跳过之前的动画
        # 若文件写入器已显示全局进度条，关闭单个动画进度条（避免重复）
        if self.file_writer.has_progress_display():
            self.show_animation_progress = False

        # 8. 初始化交互相关对象
        self.mouse_point = Point()  # 鼠标当前位置的点对象（用于交互计算，如鼠标拾取）
        self.mouse_drag_point = Point()  # 鼠标拖拽起始位置的点对象（用于拖拽计算）
        self.hold_on_wait = self.presenter_mode  # 等待时是否保持交互（演示者模式下启用）
        self.quit_interaction = False  # 退出交互的标记（用于终止场景循环）

        # 9. 初始化随机种子（确保场景可复现）
        if self.random_seed is not None:
            random.seed(self.random_seed)  # 初始化Python原生随机数
            np.random.seed(self.random_seed)  # 初始化NumPy随机数（用于向量/坐标随机）

def __str__(self) -> str:
    """
    重写字符串表示方法，返回场景类的名称（便于日志打印、调试识别）
    返回：场景类名（如"MyCustomScene"，而非默认的对象内存地址）
    """
    return self.__class__.__name__

def get_window(self) -> Window | None:
    """
    获取当前场景关联的可视化窗口对象
    返回：Window对象（若已初始化）或None（若仅后台渲染无窗口）
    用途：外部模块（如事件处理器）获取窗口实例以操作界面
    """
    return self.window

def run(self) -> None:
    """
    场景的核心运行入口，启动完整的“初始化→构建→交互→收尾”流程
    流程：
        1. 初始化动画时间戳与文件写入器
        2. 执行场景初始化（setup）、动画构建（construct）、交互循环（interact）
        3. 捕获特殊异常（结束场景、键盘中断），确保流程优雅收尾
        4. 执行场景清理（tear_down），完成输出文件生成与资源释放
    """
    # 初始化动画时间戳：虚拟时间（场景内时间）与真实时间（系统时间）对齐
    self.virtual_animation_start_time: float = 0  # 场景内动画起始时间（用于时间轴计算）
    self.real_animation_start_time: float = time.time()  # 系统真实起始时间（用于同步进度）
    # 启动文件写入器（准备视频编码管道或图像保存路径）
    self.file_writer.begin()

    # 执行场景初始化（子类可重写setup添加自定义初始化逻辑）
    self.setup()
    try:
        # 执行动画构建（核心逻辑，子类必须重写construct定义动画内容）
        self.construct()
        # 进入交互循环（若有窗口，允许用户通过键盘/鼠标操作场景）
        self.interact()
    except EndScene:
        # 捕获EndScene异常（由embed退出触发），直接跳过以优雅结束流程
        pass
    except KeyboardInterrupt:
        # 捕获键盘中断（如Ctrl+C），清除终端中的中断符号，标记写入器为“中断结束”
        print("", end="\r")  # 覆盖终端中的^C符号，美化输出
        self.file_writer.ended_with_interrupt = True  # 告知写入器保留临时文件
    # 执行场景清理（无论是否异常，确保资源释放与输出完成）
    self.tear_down()

def setup(self) -> None:
    """
    场景初始化钩子方法，子类可重写以添加通用初始化逻辑
    用途：适用于被频繁继承的基础场景类（如“数学公式场景”默认加载LaTeX配置）
    说明：默认无实现，子类按需扩展（如预加载常用Mobject、配置相机初始视角）
    """
    pass

def construct(self) -> None:
    # 场景动画构建的核心方法，子类必须重写以定义具体动画逻辑
    # 示例：子类中可在此方法内创建Mobject、添加动画（如self.play(Create(circle))）
    pass

def tear_down(self) -> None:
    """
    场景收尾清理方法，确保资源释放与输出文件完整
    流程：
        1. 停止动画跳过模式（恢复正常渲染状态）
        2. 完成文件写入器收尾（合并音频、生成最终视频/图像）
        3. 销毁窗口（释放Pyglet GUI资源，避免内存泄漏）
    """
    self.stop_skipping()  # 停止跳过动画（确保收尾时渲染最终状态）
    self.file_writer.finish()  # 完成输出文件（合并音频、保存图像、打印提示）
    # 若存在窗口，销毁窗口并置空（释放GPU/CPU资源）
    if self.window:
        self.window.destroy()
        self.window = None

def interact(self) -> None:
    """
    场景交互循环（仅在有窗口时生效），持续更新画面并响应用户输入
    逻辑：
        1. 打印交互提示（告知用户可用快捷键，如d/f/z控制相机）
        2. 关闭动画跳过模式（确保交互时实时渲染画面）
        3. 循环更新场景帧，直到窗口被关闭（通过is_window_closing判断）
    """
    # 无窗口时直接退出（仅后台渲染场景无需交互）
    if self.window is None:
        return
    # 打印交互快捷键提示（引导用户操作）
    log.info(
        "\nTips: Using the keys `d`, `f`, or `z` " +
        "you can interact with the scene. " +
        "Press `command + q` or `esc` to quit"
    )
    self.skip_animations = False  # 交互时关闭跳过模式，确保画面实时更新
    # 循环更新场景帧：每次更新间隔为1/帧率（如30fps则每~33ms更新一次）
    while not self.is_window_closing():
        self.update_frame(1 / self.camera.fps)

def embed(
    self,
    close_scene_on_exit: bool = True,
    show_animation_progress: bool = False,
) -> None:
    """
    嵌入IPython交互式终端，支持实时调试与代码交互（仅在有窗口时生效）
    核心用途：开发时实时修改场景（如调整Mobject位置、添加动画），无需重启场景
    
    参数说明：
        close_scene_on_exit: 退出交互时是否结束场景（True则触发EndScene异常，终止后续流程）
        show_animation_progress: 交互时是否显示动画进度条（False则隐藏，简化交互界面）
    """
    # 无窗口时不执行嵌入（交互需可视化窗口反馈）
    if not self.window:
        return
    # 配置交互时的进度条显示（默认隐藏，避免干扰）
    self.show_animation_progress = show_animation_progress
    self.stop_skipping()  # 停止跳过模式，确保交互时画面实时更新
    self.update_frame(force_draw=True)  # 强制更新一帧，确保当前画面正确显示

    # 创建交互式场景嵌入实例并启动IPython终端
    InteractiveSceneEmbed(self).launch()

    # 退出交互时的处理：若需要结束场景，抛出EndScene异常（被run方法捕获）
    if close_scene_on_exit:
        raise EndScene()

    # Only these methods should touch the camera

    # 以下方法均属于 **Scene 类**（或其核心子类），是 Manim 动画框架中“场景渲染与生命周期管理”的核心逻辑，
# 负责图像捕获、窗口显示、帧更新、时间管理、渲染分组等关键功能，支撑动画从生成到输出的全流程。

def get_image(self) -> Image:
    """
    捕获当前场景的图像（从相机缓冲区获取），返回可用于显示或保存的 Image 对象。
    核心作用是“冻结”当前帧状态，常用于预览单帧、保存截图等场景。
    
    逻辑步骤：
    1. 若存在窗口（交互式渲染），先禁用相机的“窗口帧缓冲区（FBO）”——避免窗口实时渲染干扰图像捕获；
    2. 让相机捕获所有渲染组（render_groups）的内容，将场景画面写入相机内部缓冲区；
    3. 从相机缓冲区读取图像数据，生成 Image 对象；
    4. 若存在窗口，恢复启用窗口 FBO——确保后续窗口渲染正常；
    5. 返回捕获的 Image 对象。
    
    返回值：Image 对象，包含当前场景的像素数据（可调用 .show() 预览，或 .save() 保存）。
    """
    if self.window is not None:
        self.camera.use_window_fbo(False)  # 禁用窗口FBO，切换到“离线捕获”模式
        self.camera.capture(*self.render_groups)  # 捕获所有渲染组内容到相机缓冲区
    image = self.camera.get_image()  # 从缓冲区提取图像
    if self.window is not None:
        self.camera.use_window_fbo(True)  # 恢复窗口FBO，回归实时渲染
    return image


def show(self) -> None:
    """
    快速预览当前场景的单帧画面（调用 get_image() 并显示），适用于调试时快速查看场景状态。
    
    逻辑步骤：
    1. 强制更新场景帧（update_frame(force_draw=True)）——确保所有动画、Mobject 状态已同步；
    2. 调用 get_image() 捕获当前帧图像；
    3. 调用 Image 对象的 .show() 方法（依赖系统图像预览工具）显示截图。
    
    注意：该方法仅显示“当前时间点”的静态帧，非动态动画播放。
    """
    self.update_frame(force_draw=True)  # 强制绘制最新状态（忽略“跳过动画”标记）
    self.get_image().show()  # 显示捕获的单帧图像


def update_frame(self, dt: float = 0, force_draw: bool = False) -> None:
    """
    场景“帧更新”的核心驱动方法，负责**时间推进、Mobject 状态更新、画面渲染**，是动画播放的“心脏”。
    每帧动画都会调用一次，控制场景从“上一帧”到“当前帧”的过渡。
    
    参数说明：
        dt : 浮点数，当前帧与上一帧的时间间隔（单位：秒），默认 0（静态帧更新）；
        force_draw : 布尔值，是否强制绘制画面（即使设置了 skip_animations，也会渲染），默认 False。
    
    核心逻辑步骤：
    1. **时间推进**：调用 increment_time(dt)，更新场景总时间（驱动动画进度）；
    2. **Mobject 状态更新**：调用 update_mobjects(dt)，让所有 Mobject 按时间差 dt 更新状态（如位置、颜色变化）；
    3. **跳过渲染判断**：若开启 skip_animations（跳过动画）且未强制绘制，直接返回（不渲染画面，提升效率）；
    4. **窗口关闭检测**：若窗口正在关闭，抛出 EndScene 异常，终止场景生命周期；
    5. **冗余渲染判断**：若存在窗口、时间差为 0、无未处理事件且未强制绘制，仅处理窗口事件（不重复渲染，节省资源）；
    6. **画面渲染**：让相机捕获所有渲染组内容，生成当前帧画面；
    7. **动画时序控制**：若存在窗口且未跳过动画，计算“虚拟动画时间”与“真实时间”的差值，通过 sleep 同步时序（避免动画播放过快）。
    
    异常：当检测到窗口关闭时，抛出 EndScene 异常，用于外层逻辑终止场景。
    """
    self.increment_time(dt)  # 1. 推进场景时间
    self.update_mobjects(dt)  # 2. 更新所有Mobject状态

    # 3. 跳过渲染的快捷判断
    if self.skip_animations and not force_draw:
        return

    # 4. 窗口关闭检测
    if self.is_window_closing():
        raise EndScene()

    # 5. 避免冗余渲染：仅处理事件，不重绘
    if self.window and dt == 0 and not self.window.has_undrawn_event() and not force_draw:
        self.window._window.dispatch_events()  # 处理鼠标、键盘等窗口事件
        return

    # 6. 渲染当前帧画面
    self.camera.capture(*self.render_groups)

    # 7. 同步动画时序（确保动画按预期速度播放）
    if self.window and not self.skip_animations:
        vt = self.time - self.virtual_animation_start_time  # 虚拟动画已播放时间
        rt = time.time() - self.real_animation_start_time    # 真实已流逝时间
        time.sleep(max(vt - rt, 0))  # 若虚拟时间超前，sleep补全差值


def emit_frame(self) -> None:
    """
    将当前帧写入输出文件（由 file_writer 负责），是“动画导出为视频/图片序列”的关键步骤。
    仅在“不跳过动画”时执行（避免导出空帧）。
    
    逻辑：若未开启 skip_animations，调用文件写入器（file_writer）的 write_frame 方法，
    将相机缓冲区中的当前帧数据写入目标文件（如 .mp4、.png 序列）。
    """
    if not self.skip_animations:
        self.file_writer.write_frame(self.camera)  # 把当前相机捕获的帧写入文件


# ------------------------------ 与 Mobject 更新相关的辅助方法 ------------------------------
def update_mobjects(self, dt: float) -> None:
    """
    遍历场景中所有 Mobject，调用其 update 方法更新状态（如位置、旋转、颜色等）。
    dt 为时间差，确保动画按“时间驱动”更新（而非帧驱动），避免不同帧率下动画速度不一致。
    
    参数：dt - 帧时间差（秒），用于计算 Mobject 状态随时间的变化量。
    """
    for mobject in self.mobjects:
        mobject.update(dt)  # 调用每个Mobject的update方法，传递时间差


def should_update_mobjects(self) -> bool:
    """
    判断当前场景是否需要更新 Mobject 状态，用于优化性能（避免无必要的更新）。
    
    返回值：布尔值，满足以下任一条件则返回 True（需要更新）：
        1. 场景开启 always_update_mobjects（强制始终更新）；
        2. 至少有一个 Mobject 拥有“更新器（updater）”（即需要动态更新状态）。
    """
    return self.always_update_mobjects or any(
        mob.has_updaters() for mob in self.mobjects
    )


# ------------------------------ 与时间管理相关的辅助方法 ------------------------------
def get_time(self) -> float:
    """获取当前场景的总时间（从场景启动开始累计），用于动画进度计算（如 alpha 值推导）。"""
    return self.time


def increment_time(self, dt: float) -> None:
    """
    推进场景总时间，是“时间驱动动画”的核心——所有依赖时间的动画（如 LinearAnimation）
    均通过该方法的时间累计来计算当前帧的状态。
    
    参数：dt - 待累加的时间差（秒）。
    """
    self.time += dt


# ------------------------------ 与 Mobject 组织及渲染优化相关的方法 ------------------------------
def get_top_level_mobjects(self) -> list[Mobject]:
    """
    获取场景中的“顶级 Mobject”——即不被其他 Mobject 包含的独立对象（排除子对象），
    用于避免重复渲染（子对象会随父对象一起渲染，无需单独处理）。
    
    逻辑步骤：
    1. 获取场景中所有 Mobject；
    2. 对每个 Mobject，获取其“家族树”（get_family()，包含自身及所有子对象）；
    3. 判断一个 Mobject 是否为“顶级”：仅在一个家族树中出现（即不是任何其他 Mobject 的子对象）；
    4. 返回所有顶级 Mobject 的列表。
    
    返回值：list[Mobject]，场景中所有独立的顶级 Mobject。
    """
    mobjects = self.get_mobjects()  # 获取场景所有Mobject
    families = [m.get_family() for m in mobjects]  # 每个Mobject的家族树

    def is_top_level(mobject):
        # 统计该Mobject在所有家族树中出现的次数：仅出现1次 → 顶级（仅自身家族）
        num_families = sum([
            (mobject in family)
            for family in families
        ])
        return num_families == 1

    return list(filter(is_top_level, mobjects))  # 过滤出顶级Mobject


def get_mobject_family_members(self) -> list[Mobject]:
    """
    获取场景中所有 Mobject 的“完整家族成员”（包含所有顶级对象及其所有子对象），
    用于需要遍历“所有可见元素”的场景（如碰撞检测、全局状态重置）。
    
    依赖：调用 extract_mobject_family_members 工具函数，扁平化所有 Mobject 的家族树。
    返回值：list[Mobject]，所有 Mobject 及其子对象的扁平列表（无重复）。
    """
    return extract_mobject_family_members(self.mobjects)


def assemble_render_groups(self):
    """
    优化渲染性能的核心方法：将场景中的 Mobject 按“渲染属性”分组，
    相同属性的 Mobject 批量渲染（减少 GPU 绘制调用次数，提升效率）。
    
    核心逻辑：
    1. **按渲染属性分组**：调用 batch_by_property 工具函数，将 Mobject 按以下属性聚合：
        - 类型（str(type(m))）：确保相同类型的 Mobject 用同一 shader（着色器）；
        - 着色器包装器 ID（m.get_shader_wrapper(self.camera.ctx).get_id()）：确保相同 shader 配置的对象同组；
        - z_index（层级）：确保相同层级的对象同组（避免层级混乱）；
    2. **清空旧渲染组**：将之前的 render_groups 清空，避免残留；
    3. **创建新渲染组**：对每个分组，用该组第一个 Mobject 的“组类”（get_group_class()）创建 Group，
       确保分组后的对象仍保持原有的渲染行为（如 VMobjectGroup 用于矢量对象）。
    
    作用：减少 GPU 的“绘制调用（draw call）”次数——GPU 批量处理同属性对象比逐个处理高效得多，
    尤其在场景中存在大量小 Mobject 时（如粒子效果），性能提升显著。
    """
    # 1. 按“类型+shader ID+z_index”分组，确保同属性对象聚合
    batches = batch_by_property(
        self.mobjects,
        lambda m: str(type(m)) + str(m.get_shader_wrapper(self.camera.ctx).get_id()) + str(m.z_index)
    )

    # 2. 清空旧渲染组
    for group in self.render_groups:
        group.clear()
    # 3. 为每个分组创建对应类型的渲染组
    self.render_groups = [
        batch[0].get_group_class()(*batch)  # 用分组第一个对象的组类创建Group
        for batch, key in batches
    ]
    # 以下方法均属于 **Scene 类**，核心围绕“场景中 Mobject 的组织与管理”展开，
# 包括添加、删除、替换、层级调整等操作，同时通过装饰器确保操作后自动优化渲染分组，
# 是构建动画场景（如添加图形、切换元素、调整显示层级）的基础接口。


@staticmethod
def affects_mobject_list(func: Callable[..., T]) -> Callable[..., T]:
    """
    静态装饰器：标记“修改 Mobject 列表”的方法，确保方法执行后自动重新构建渲染分组（assemble_render_groups），
    维持渲染性能优化（避免因 Mobject 列表变化导致分组失效）。
    
    工作逻辑：
    1. 用 @wraps(func) 保留原方法的元信息（如函数名、文档字符串）；
    2. 定义包装函数 wrapper：先执行原方法，再调用场景的 assemble_render_groups() 重建渲染组；
    3. 返回包装后的函数，确保修改 Mobject 列表后渲染分组同步更新。
    
    参数：func - 待装饰的方法（如 add、remove、replace 等修改 mobjects 列表的方法）
    返回：包装后的方法
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        func(self, *args, **kwargs)  # 执行原方法（修改Mobject列表）
        self.assemble_render_groups()  # 重建渲染分组，优化后续渲染
        return self  # 返回场景对象，支持链式调用（如 scene.add(circle).add(square)）
    return wrapper


@affects_mobject_list
def add(self, *new_mobjects: Mobject):
    """
    向场景中添加一个或多个 Mobject（图形对象），添加顺序决定默认渲染层级（后添加的默认在上方，可通过 z_index 调整）。
    
    核心逻辑：
    1. 先移除待添加的 Mobject（避免重复添加，防止同一对象在场景中多次出现）；
    2. 将新 Mobject 追加到场景的 mobjects 列表；
    3. 按“z_index（层级）+ 添加顺序”重新排序 mobjects：
        - 优先按 z_index 升序排列（z_index 越大，渲染层级越高，显示在越上层）；
        - z_index 相同时，按原始添加顺序排列（后添加的在同层级上方）；
    4. 更新 id_to_mobject_map（Mobject ID 到对象的映射），包含新 Mobject 及其所有子对象，便于快速查找；
    5. 装饰器自动触发 assemble_render_groups() 重建渲染分组。
    
    参数：*new_mobjects - 可变参数，一个或多个待添加的 Mobject（如 Circle()、Text()）
    返回：当前 Scene 对象，支持链式调用（如 scene.add(circle).set_background_color(WHITE)）
    
    示例：
        >>> circle = Circle()
        >>> square = Square()
        >>> scene.add(circle, square)  # 先添加圆形（下层），再添加正方形（上层）
    """
    self.remove(*new_mobjects)  # 避免重复添加
    self.mobjects += new_mobjects  # 追加新Mobject到列表

    # 按 z_index 和添加顺序排序：确保层级正确
    id_to_scene_order = {id(m): idx for idx, m in enumerate(self.mobjects)}  # 记录原始添加顺序
    self.mobjects.sort(key=lambda m: (m.z_index, id_to_scene_order[id(m)]))  # 排序规则：z_index优先，再按添加顺序

    # 更新ID映射：包含新Mobject的所有家族成员（自身+子对象）
    self.id_to_mobject_map.update({
        id(sm): sm
        for m in new_mobjects
        for sm in m.get_family()  # get_family() 获取对象及其所有子对象
    })
    return self


def add_mobjects_among(self, values: Iterable):
    """
    从可迭代对象（如字典 values、列表）中筛选出所有 Mobject 并添加到场景，
    适用于快速原型开发（如一次性添加当前作用域中定义的所有图形对象）。
    
    逻辑：
    1. 用 filter 过滤 values 中的元素，仅保留 isinstance(m, Mobject) 为 True 的对象；
    2. 调用 add() 方法将筛选出的 Mobject 批量添加到场景。
    
    参数：values - 可迭代对象（如 locals().values()，包含当前作用域所有变量）
    返回：当前 Scene 对象，支持链式调用
    
    示例：
        >>> circle = Circle()
        >>> square = Square()
        >>> text = Text("Hello")
        >>> scene.add_mobjects_among(locals().values())  # 自动添加 circle、square、text
    """
    self.add(*filter(
        lambda m: isinstance(m, Mobject),  # 筛选Mobject类型的元素
        values
    ))
    return self


@affects_mobject_list
def replace(self, mobject: Mobject, *replacements: Mobject):
    """
    用一个或多个新 Mobject 替换场景中的指定 Mobject，保持原 Mobject 的位置（在 mobjects 列表中的索引）。
    
    逻辑：
    1. 检查待替换的 mobject 是否在场景的 mobjects 列表中；
    2. 若存在，找到其在列表中的索引，用 replacements 替换该位置的 mobject；
    3. 装饰器自动触发 assemble_render_groups() 重建渲染分组。
    
    参数：
        mobject - 待替换的 Mobject（必须已在场景中）；
        *replacements - 用于替换的一个或多个新 Mobject。
    返回：当前 Scene 对象，支持链式调用
    
    示例：
        >>> old_circle = Circle(color=RED)
        >>> scene.add(old_circle)
        >>> new_square = Square(color=BLUE)
        >>> scene.replace(old_circle, new_square)  # 用蓝色正方形替换红色圆形
    """
    if mobject in self.mobjects:
        index = self.mobjects.index(mobject)  # 找到待替换对象的索引
        # 替换列表中的元素：保留索引前的元素，插入新对象，保留索引后的元素
        self.mobjects = [
            *self.mobjects[:index],
            *replacements,
            *self.mobjects[index + 1:]
        ]
    return self


@affects_mobject_list
def remove(self, *mobjects_to_remove: Mobject):
    """
    从场景中移除一个或多个 Mobject，若移除的是“组（Group）中的子对象”，则保留组内其他子对象（避免误删整个组）。
    
    核心逻辑（解决“组内子对象移除”的特殊场景）：
    1. 先提取所有待移除 Mobject 的“家族成员”（包含子对象），存入集合 to_remove；
    2. 调用 recursive_mobject_remove 工具函数：从 mobjects 列表中移除 to_remove 中的对象，
       同时保留“被移除对象所在组的其他子对象”（如移除 Group(m1,m2) 中的 m1，会保留 m2）；
    3. 更新 mobjects 列表为移除后的结果；
    4. 装饰器自动触发 assemble_render_groups() 重建渲染分组。
    
    参数：*mobjects_to_remove - 待移除的一个或多个 Mobject
    返回：当前 Scene 对象，支持链式调用
    
    示例：
        >>> group = Group(Circle(), Square(), Text("Hi"))
        >>> scene.add(group)
        >>> scene.remove(group[0])  # 仅移除组中的圆形，保留正方形和文本
    """
    # 提取待移除对象的所有家族成员（避免遗漏子对象）
    to_remove = set(extract_mobject_family_members(mobjects_to_remove))
    # 递归移除：保留组内其他子对象
    new_mobjects, _ = recursive_mobject_remove(self.mobjects, to_remove)
    self.mobjects = new_mobjects  # 更新Mobject列表


@affects_mobject_list
def remove_all_except(self, *mobjects_to_keep: Mobject):
    """
    移除场景中除指定 Mobject 外的所有对象（先清空，再添加需保留的对象），
    适用于场景切换（如从“步骤1”切换到“步骤2”，仅保留必要元素）。
    
    逻辑：
    1. 调用 clear() 清空场景所有 Mobject；
    2. 调用 add() 添加需保留的 mobjects_to_keep；
    3. 装饰器自动触发 assemble_render_groups() 重建渲染分组。
    
    参数：*mobjects_to_keep - 需保留的一个或多个 Mobject
    返回：当前 Scene 对象，支持链式调用
    
    示例：
        >>> circle = Circle()
        >>> square = Square()
        >>> text = Text("Keep Me")
        >>> scene.add(circle, square, text)
        >>> scene.remove_all_except(text)  # 仅保留文本，移除圆形和正方形
    """
    self.clear()  # 清空所有Mobject
    self.add(*mobjects_to_keep)  # 添加需保留的对象


def bring_to_front(self, *mobjects: Mobject):
    """
    将指定 Mobject 移到渲染层级的最上层（显示在所有其他对象上方）。
    
    逻辑：利用 add() 方法的“后添加对象默认在上层”特性——先移除 Mobject，再重新添加，
    使其成为 mobjects 列表中排序靠后的元素（按 z_index+添加顺序排序后层级最高）。
    
    参数：*mobjects - 需移到顶层的一个或多个 Mobject
    返回：当前 Scene 对象，支持链式调用
    
    示例：
        >>> circle = Circle()
        >>> square = Square()
        >>> scene.add(circle, square)  # 正方形默认在顶层
        >>> scene.bring_to_front(circle)  # 圆形移到顶层，覆盖正方形
    """
    self.add(*mobjects)  # 移除后重新添加，触发排序到上层
    return self


@affects_mobject_list
def bring_to_back(self, *mobjects: Mobject):
    """
    将指定 Mobject 移到渲染层级的最下层（显示在所有其他对象下方）。
    
    逻辑：
    1. 先从场景中移除指定 Mobject；
    2. 将其插入到 mobjects 列表的最开头（排序后层级最低）；
    3. 装饰器自动触发 assemble_render_groups() 重建渲染分组。
    
    参数：*mobjects - 需移到底层的一个或多个 Mobject
    返回：当前 Scene 对象，支持链式调用
    
    示例：
        >>> circle = Circle()
        >>> square = Square()
        >>> scene.add(circle, square)  # 圆形在底层，正方形在顶层
        >>> scene.bring_to_back(square)  # 正方形移到底层，圆形在顶层
    """
    self.remove(*mobjects)  # 先移除
    self.mobjects = list(mobjects) + self.mobjects  # 插入到列表开头
    return self


@affects_mobject_list
def clear(self):
    """
    清空场景中所有 Mobject（重置 mobjects 列表为空），
    适用于完全重置场景（如切换到全新的动画片段）。
    
    逻辑：将 mobjects 列表设为空列表，装饰器自动触发 assemble_render_groups() 重建渲染分组。
    
    返回：当前 Scene 对象，支持链式调用
    
    示例：
        >>> scene.add(Circle(), Square(), Text("Hi"))
        >>> scene.clear()  # 场景中无任何对象
    """
    self.mobjects = []
    return self
# 以下方法继续补充 Scene 类的核心功能，涵盖 Mobject 检索、坐标点交互、分组管理、动画进度控制等，
# 是场景与对象交互、动画流程管理的关键接口。


def get_mobjects(self) -> list[Mobject]:
    """
    获取场景中所有 Mobject 的列表（返回副本，避免外部直接修改内部状态）。
    
    返回值：list[Mobject]，场景中当前所有 Mobject 的浅拷贝列表（修改列表本身不影响场景，
    但修改列表中的 Mobject 会影响场景中的对象）。
    """
    return list(self.mobjects)  # 返回副本，防止外部直接操作内部列表


def get_mobject_copies(self) -> list[Mobject]:
    """
    获取场景中所有 Mobject 的“深拷贝”列表（独立于原对象，修改拷贝不影响场景中的原始对象），
    适用于需要临时操作对象副本的场景（如预览修改效果、备份状态）。
    
    返回值：list[Mobject]，每个元素都是场景中对应 Mobject 的 copy() 结果。
    """
    return [m.copy() for m in self.mobjects]  # 对每个Mobject调用copy()生成独立副本


def point_to_mobject(
    self,
    point: np.ndarray,
    search_set: Iterable[Mobject] | None = None,
    buff: float = 0
) -> Mobject | None:
    """
    查找“包含指定坐标点”的顶层 Mobject（从上层到下层检查，返回第一个命中的对象），
    常用于交互场景（如鼠标点击检测：确定点击位置对应的图形对象）。
    
    参数说明：
        point : 三维坐标点（np.ndarray），待检测的位置；
        search_set : 可选，待搜索的 Mobject 集合，默认搜索场景中所有 Mobject；
        buff : 缓冲距离，默认0，扩大检测范围（buff>0 时，点在对象边缘外 buff 范围内也算命中）。
    
    逻辑：
    1. 若未指定 search_set，默认使用场景的 mobjects 列表；
    2. 按“逆序”遍历集合（从最后添加的对象开始，即顶层对象优先）；
    3. 对每个 Mobject，调用 is_point_touching(point, buff) 判断点是否在对象上（含缓冲）；
    4. 返回第一个命中的 Mobject，未命中则返回 None。
    
    返回值：Mobject 或 None，包含该点的顶层对象（或 None）。
    """
    if search_set is None:
        search_set = self.mobjects  # 默认搜索所有场景对象
    # 逆序遍历（顶层对象优先），返回第一个包含该点的Mobject
    for mobject in reversed(search_set):
        if mobject.is_point_touching(point, buff=buff):
            return mobject
    return None


def get_group(self, *mobjects):
    """
    根据输入的 Mobject 类型，自动创建对应的“组对象”（VGroup 或 Group）：
    - 若所有对象都是 VMobject（矢量图形），返回 VGroup（优化矢量渲染）；
    - 否则返回通用 Group（适用于混合类型对象）。
    
    作用：简化分组操作，自动选择最优的组类型，提升渲染效率。
    
    参数：*mobjects - 待分组的一个或多个 Mobject
    返回值：VGroup 或 Group，包含所有输入 Mobject 的组对象。
    """
    # 检查所有对象是否都是VMobject（矢量图形基类）
    if all(isinstance(m, VMobject) for m in mobjects):
        return VGroup(*mobjects)  # 矢量对象用VGroup优化渲染
    else:
        return Group(*mobjects)   # 混合类型用通用Group


def id_to_mobject(self, id_value):
    """
    通过对象的 ID（内存地址标识）查找对应的 Mobject，依赖场景维护的 id_to_mobject_map。
    
    参数：id_value - Mobject 的 ID（可通过 id(mobject) 获取）
    返回值：Mobject，与该 ID 关联的对象（若不存在会抛出 KeyError）。
    """
    return self.id_to_mobject_map[id_value]  # 从映射表中直接查找


def ids_to_group(self, *id_values):
    """
    通过多个对象 ID 创建对应的组（自动过滤无效 ID），适用于通过 ID 批量恢复对象并分组。
    
    逻辑：
    1. 用 map(self.id_to_mobject, id_values) 将每个 ID 转换为对应的 Mobject；
    2. 用 filter 过滤掉 None（无效 ID 对应的结果）；
    3. 调用 get_group() 创建组对象。
    
    参数：*id_values - 一个或多个 Mobject 的 ID
    返回值：VGroup 或 Group，包含所有有效 ID 对应 Mobject 的组。
    """
    return self.get_group(*filter(
        lambda x: x is not None,  # 过滤无效对象
        map(self.id_to_mobject, id_values)  # ID转对象
    ))


def i2g(self, *id_values):
    """ids_to_group 的简写，快速通过 ID 创建组对象（提升开发效率）。"""
    return self.ids_to_group(*id_values)


def i2m(self, id_value):
    """id_to_mobject 的简写，快速通过 ID 查找对象（提升开发效率）。"""
    return self.id_to_mobject(id_value)


# ------------------------------ 与动画跳过（Skipping）相关的方法 ------------------------------
def update_skipping_status(self) -> None:
    """
    更新动画跳过状态：根据 start_at_animation_number 和 end_at_animation_number 控制动画播放范围，
    用于从指定动画编号开始播放，或在指定编号结束播放（常用于调试特定片段）。
    
    逻辑：
    1. 若设置了 start_at_animation_number（起始动画编号）：
        - 当当前播放次数（num_plays）达到该编号时，记录当前时间为 skip_time，
          并在原本处于跳过状态时调用 stop_skipping() 开始播放；
    2. 若设置了 end_at_animation_number（结束动画编号）：
        - 当当前播放次数 >= 该编号时，抛出 EndScene 异常终止场景。
    """
    if self.start_at_animation_number is not None:
        if self.num_plays == self.start_at_animation_number:
            self.skip_time = self.time  # 记录跳过的时间点
            if not self.original_skipping_status:
                self.stop_skipping()  # 开始播放动画
    if self.end_at_animation_number is not None:
        if self.num_plays >= self.end_at_animation_number:
            raise EndScene()  # 终止场景


def stop_skipping(self) -> None:
    """
    停止“跳过动画”模式：重置动画时间基准，开始正常播放动画。
    
    逻辑：
    1. 将虚拟动画开始时间（virtual_animation_start_time）设为当前场景时间；
    2. 将真实动画开始时间（real_animation_start_time）设为当前系统时间；
    3. 关闭 skip_animations 标记（开始渲染动画帧）。
    """
    self.virtual_animation_start_time = self.time  # 虚拟时间基准
    self.real_animation_start_time = time.time()   # 真实时间基准
    self.skip_animations = False  # 关闭跳过模式


# ------------------------------ 与动画进度管理相关的方法 ------------------------------
def get_time_progression(
    self,
    run_time: float,
    n_iterations: int | None = None,
    desc: str = "",
    override_skip_animations: bool = False
) -> list[float] | np.ndarray | ProgressDisplay:
    """
    生成动画播放的“时间进度序列”（每帧对应的时间点），或返回进度条显示对象，
    用于控制动画在每帧的状态更新。
    
    参数说明：
        run_time : 动画总时长（秒）；
        n_iterations : 可选，总迭代次数（用于进度条显示）；
        desc : 进度条描述文本；
        override_skip_animations : 布尔值，是否强制生成完整时间序列（忽略 skip_animations），默认 False。
    
    逻辑：
    1. 若处于跳过动画模式且未强制覆盖，返回 [run_time]（仅处理最后一帧状态）；
    2. 否则，生成从 0 到 run_time 的时间序列，间隔为 1/帧率（确保每帧都被覆盖）；
    3. 若开启动画进度显示（show_animation_progress），返回 ProgressDisplay 对象（用于终端进度条）；
    4. 否则，返回时间序列数组（供动画帧迭代使用）。
    
    返回值：
        - 若跳过动画：[run_time]（列表）；
        - 若显示进度：ProgressDisplay 对象；
        - 否则：np.ndarray（时间点数组）。
    """
    # 跳过动画模式：仅返回最后一帧的时间点
    if self.skip_animations and not override_skip_animations:
        return [run_time]

    # 生成完整时间序列：从1/帧率开始，到run_time，间隔1/帧率（确保覆盖每帧）
    times = np.arange(0, run_time, 1 / self.camera.fps) + 1 / self.camera.fps

    # 设置文件写入器的进度描述
    self.file_writer.set_progress_display_description(sub_desc=desc)

    # 显示进度条：返回ProgressDisplay对象
    if self.show_animation_progress:
        return ProgressDisplay(
            times,
            total=n_iterations,
            leave=self.leave_progress_bars,  # 是否保留进度条
            ascii=True if platform.system() == 'Windows' else None,  # Windows用ASCII进度条
            desc=desc,  # 描述文本
            bar_format="{l_bar} {n_fmt:3}/{total_fmt:3} {rate_fmt}{postfix}",  # 进度条格式
        )
    else:
        return times  # 返回时间序列数组

# 以下方法是 **Scene 类** 中“动画播放核心流程”的实现，涵盖动画时长计算、进度管理、播放前后准备与收尾，
# 以及最核心的 `play()` 方法（动画播放入口），是 Manim 动画从定义到执行的关键链路。


def get_run_time(self, animations: Iterable[Animation]) -> float:
    """
    计算一组动画的“总播放时长”——取所有动画中最长的单动画时长（确保所有动画都能播放完成）。
    
    逻辑：遍历所有动画，调用每个动画的 `get_run_time()` 方法获取其时长，返回最大值。
    （若多个动画并行播放，总时长由最长的动画决定，避免短动画提前结束）
    
    参数：animations - 可迭代对象，包含一个或多个 Animation 对象（如 Create、MoveAlongPath）
    返回值：float，所有动画的最大时长（秒）。
    """
    return np.max([animation.get_run_time() for animation in animations])


def get_animation_time_progression(
    self,
    animations: Iterable[Animation]
) -> list[float] | np.ndarray | ProgressDisplay:
    """
    为一组动画生成“时间进度序列”——即动画播放过程中每帧对应的时间点，或返回进度条对象。
    
    逻辑步骤：
    1. 将动画转换为列表，计算总播放时长（调用 `get_run_time()`）；
    2. 生成进度描述文本（包含当前播放次数和第一个动画名称，多动画时加“etc.”）；
    3. 调用 `get_time_progression()` 生成时间序列或进度条（复用通用时间进度逻辑）。
    
    参数：animations - 可迭代对象，包含一个或多个 Animation 对象
    返回值：与 `get_time_progression()` 一致（时间序列列表/数组 或 ProgressDisplay 进度条）。
    """
    animations = list(animations)
    run_time = self.get_run_time(animations)  # 计算总时长
    # 生成进度描述（如“1 Create(Circle)”或“2 MoveAlongPath, etc.”）
    description = f"{self.num_plays} {animations[0]}"
    if len(animations) > 1:
        description += ", etc."
    # 生成时间进度（复用通用方法）
    time_progression = self.get_time_progression(run_time, desc=description)
    return time_progression


def get_wait_time_progression(
    self,
    duration: float,
    stop_condition: Callable[[], bool] | None = None
) -> list[float] | np.ndarray | ProgressDisplay:
    """
    为“等待场景”（如 `self.wait(duration)`）生成时间进度序列，支持自定义停止条件。
    
    逻辑：
    1. 基础参数设置：进度描述为“N Waiting”（N 为当前播放次数）；
    2. 若有自定义停止条件（如鼠标点击停止）：
        - 设置 `n_iterations=-1`（不显示百分比进度，因时长不确定）；
        - 强制 `override_skip_animations=True`（即使跳过动画也需等待停止条件）；
    3. 调用 `get_time_progression()` 生成时间进度。
    
    参数：
        duration : 等待时长（秒）；
        stop_condition : 可选，无参数布尔函数，返回 True 时提前停止等待（如检测鼠标点击）。
    返回值：时间序列列表/数组 或 ProgressDisplay 进度条。
    """
    kw = {"desc": f"{self.num_plays} Waiting"}  # 进度描述
    if stop_condition is not None:
        kw["n_iterations"] = -1  # 不显示百分比（时长不确定）
        kw["override_skip_animations"] = True  # 强制不跳过等待
    return self.get_time_progression(duration, **kw)


def pre_play(self):
    """
    动画播放前的“准备工作”——处理演示模式、更新跳过状态、初始化文件写入器、同步时间基准。
    
    核心逻辑：
    1. 演示模式（presenter_mode）处理：若为首次播放（num_plays=0），进入 hold_loop（等待用户操作）；
    2. 更新动画跳过状态（调用 `update_skipping_status()`），确定是否从指定帧开始播放；
    3. 若不跳过动画，通知文件写入器开始记录动画（`file_writer.begin_animation()`）；
    4. 若存在窗口，重置虚拟动画时间和真实时间基准（确保时序同步）。
    """
    # 演示模式：首次播放前等待用户操作
    if self.presenter_mode and self.num_plays == 0:
        self.hold_loop()

    # 更新跳过状态（如从指定动画编号开始）
    self.update_skipping_status()

    # 开始记录动画帧（不跳过动画时）
    if not self.skip_animations:
        self.file_writer.begin_animation()

    # 同步窗口渲染的时间基准
    if self.window:
        self.virtual_animation_start_time = self.time
        self.real_animation_start_time = time.time()


def post_play(self):
    """
    动画播放后的“收尾工作”——结束文件写入、预览跳过帧、更新播放次数。
    
    核心逻辑：
    1. 若不跳过动画，通知文件写入器结束当前动画记录（`file_writer.end_animation()`）；
    2. 若开启“跳过动画时预览”（preview_while_skipping）且有窗口，强制绘制最后一帧（让用户看到结果）；
    3. 播放次数计数器加 1（`num_plays`，用于进度跟踪和跳过状态判断）。
    """
    # 结束当前动画的帧记录
    if not self.skip_animations:
        self.file_writer.end_animation()

    # 跳过动画时预览最终帧
    if self.preview_while_skipping and self.skip_animations and self.window is not None:
        self.update_frame(dt=0, force_draw=True)

    # 更新播放次数
    self.num_plays += 1


def begin_animations(self, animations: Iterable[Animation]) -> None:
    """
    动画播放前的“初始化”——调用每个动画的 `begin()` 方法，并将未在场景中的动画对象添加到场景。
    
    核心逻辑：
    1. 获取当前场景中所有 Mobject 的家族成员（避免重复添加）；
    2. 遍历每个动画：
        a. 调用动画的 `begin()` 方法（初始化动画起始状态，如记录对象初始位置）；
        b. 若动画的目标 Mobject 不在场景中，将其添加到场景，并更新“已存在对象集合”；
    （确保动画对象能被正确渲染，且场景 Mobject 列表同步）
    """
    # 获取当前场景中所有 Mobject（含子对象）
    all_mobjects = set(self.get_mobject_family_members())
    for animation in animations:
        # 初始化动画（记录起始状态）
        animation.begin()
        # 若动画对象不在场景中，添加到场景
        if animation.mobject not in all_mobjects:
            self.add(animation.mobject)
            # 更新已存在对象集合（包含新对象的所有子对象）
            all_mobjects = all_mobjects.union(animation.mobject.get_family())


def progress_through_animations(self, animations: Iterable[Animation]) -> None:
    """
    动画播放的“核心循环”——逐帧更新动画状态、渲染画面、写入帧数据。
    
    逻辑步骤：
    1. 初始化上一帧时间（last_t=0），用于计算帧时间差（dt）；
    2. 遍历时间进度序列（每帧对应的时间点 t）：
        a. 计算当前帧与上一帧的时间差 dt = t - last_t；
        b. 更新上一帧时间为当前 t；
        c. 遍历每个动画：
            - 调用 `animation.update_mobjects(dt)`：更新动画对象的时间相关状态；
            - 计算动画进度 alpha = t / 动画时长（0~1，控制动画插值）；
            - 调用 `animation.interpolate(alpha)`：根据进度更新对象状态（如位置、颜色）；
        d. 调用 `update_frame(dt)`：渲染当前帧画面；
        e. 调用 `emit_frame()`：将当前帧写入输出文件（不跳过动画时）。
    """
    last_t = 0
    # 逐帧遍历时间进度序列
    for t in self.get_animation_time_progression(animations):
        dt = t - last_t  # 帧时间差
        last_t = t

        # 逐动画更新状态和进度
        for animation in animations:
            animation.update_mobjects(dt)  # 更新时间相关状态
            alpha = t / animation.run_time  # 计算动画进度（0~1）
            animation.interpolate(alpha)    # 根据进度插值更新对象

        # 渲染当前帧并写入文件
        self.update_frame(dt)
        self.emit_frame()


def finish_animations(self, animations: Iterable[Animation]) -> None:
    """
    动画播放后的“收尾”——确保所有动画达到最终状态，清理临时资源。
    
    逻辑步骤：
    1. 遍历每个动画：
        a. 调用 `animation.finish()`：强制动画达到最终状态（避免因帧丢失导致未完成）；
        b. 调用 `animation.clean_up_from_scene(self)`：清理动画在场景中产生的临时对象（如辅助线）；
    2. 若跳过动画：一次性更新所有 Mobject 到最终状态（时间差设为总动画时长）；
    3. 若不跳过动画：更新 Mobject 状态（时间差设为 0，仅处理必要更新）。
    """
    for animation in animations:
        animation.finish()  # 强制动画完成
        animation.clean_up_from_scene(self)  # 清理临时资源

    # 跳过动画时，一次性更新到最终状态
    if self.skip_animations:
        self.update_mobjects(self.get_run_time(animations))
    else:
        self.update_mobjects(0)  # 正常播放后，处理必要更新


@affects_mobject_list
def play(
    self,
    *proto_animations: Animation | _AnimationBuilder,
    run_time: float | None = None,
    rate_func: Callable[[float], float] | None = None,
    lag_ratio: float | None = None,
) -> None:
    """
    **Scene 类的核心方法：动画播放入口**，支持传入多个动画（并行播放），并可配置时长、速度曲线、延迟比例。
    
    参数说明：
        *proto_animations : 可变参数，动画对象（如 Create(Circle)）或动画构建器（如 circle.animate.move_to(ORIGIN)）；
        run_time : 可选，动画总时长（秒），优先级高于单个动画的 run_time；
        rate_func : 可选，速度曲线函数（如 linear、ease_in_out），控制动画速度变化；
        lag_ratio : 可选，多个动画的延迟比例（0 为完全同步，1 为依次播放）。
    
    核心流程（完整动画生命周期）：
    1. 空动画检查：若未传入动画，打印警告并返回；
    2. 动画预处理：调用 `prepare_animation()` 将动画构建器转换为实际 Animation 对象；
    3. 动画参数更新：为所有动画统一设置 run_time、rate_func、lag_ratio（覆盖单个动画配置）；
    4. 播放前准备：调用 `pre_play()`（演示模式等待、跳过状态更新、文件写入初始化）；
    5. 动画初始化：调用 `begin_animations()`（初始化动画起始状态、添加对象到场景）；
    6. 逐帧播放：调用 `progress_through_animations()`（更新状态、渲染画面、写入帧）；
    7. 播放后收尾：调用 `finish_animations()`（强制完成、清理资源）；
    8. 最终收尾：调用 `post_play()`（结束文件写入、更新播放次数）。
    
    装饰器 `@affects_mobject_list` 确保动画过程中 Mobject 列表变化后，自动重建渲染分组。
    
    示例：
        >>> circle = Circle()
        >>> square = Square()
        >>> # 并行播放“创建圆形”和“移动正方形”，总时长2秒，使用缓入缓出曲线
        >>> self.play(
        ...     Create(circle),
        ...     square.animate.move_to(RIGHT),
        ...     run_time=2,
        ...     rate_func=rate_functions.ease_in_out
        ... )
    """
    # 空动画检查
    if len(proto_animations) == 0:
        log.warning("Called Scene.play with no animations")
        return

    # 预处理动画：将构建器转换为Animation对象
    animations = list(map(prepare_animation, proto_animations))

    # 统一更新动画参数（时长、速度曲线、延迟比例）
    for anim in animations:
        anim.update_rate_info(run_time, rate_func, lag_ratio)

    # 完整动画生命周期
    self.pre_play()                  # 播放前准备
    self.begin_animations(animations)# 动画初始化
    self.progress_through_animations(animations)  # 逐帧播放
    self.finish_animations(animations)# 播放后收尾
    self.post_play()                 # 最终收尾

    def wait(
        self,
        duration: Optional[float] = None,
        stop_condition: Callable[[], bool] = None,
        note: str = None,
        ignore_presenter_mode: bool = False
    ):
        if duration is None:
            duration = self.default_wait_time
        self.pre_play()
        self.update_mobjects(dt=0)  # Any problems with this?
        if self.presenter_mode and not self.skip_animations and not ignore_presenter_mode:
            if note:
                log.info(note)
            self.hold_loop()
        else:
            time_progression = self.get_wait_time_progression(duration, stop_condition)
            last_t = 0
            for t in time_progression:
                dt = t - last_t
                last_t = t
                self.update_frame(dt)
                self.emit_frame()
                if stop_condition is not None and stop_condition():
                    break
        self.post_play()

    def hold_loop(self):
        while self.hold_on_wait:
            self.update_frame(dt=1 / self.camera.fps)
        self.hold_on_wait = True

    def wait_until(
        self,
        stop_condition: Callable[[], bool],
        max_time: float = 60
    ):
        self.wait(max_time, stop_condition=stop_condition)

    def force_skipping(self):
        self.original_skipping_status = self.skip_animations
        self.skip_animations = True
        return self

    def revert_to_original_skipping_status(self):
        if hasattr(self, "original_skipping_status"):
            self.skip_animations = self.original_skipping_status
        return self

    def add_sound(
        self,
        sound_file: str,
        time_offset: float = 0,
        gain: float | None = None,
        gain_to_background: float | None = None
    ):
        if self.skip_animations:
            return
        time = self.get_time() + time_offset
        self.file_writer.add_sound(sound_file, time, gain, gain_to_background)

    # Helpers for interactive development

    def get_state(self) -> SceneState:
        return SceneState(self)

    @affects_mobject_list
    def restore_state(self, scene_state: SceneState):
        scene_state.restore_scene(self)

    def save_state(self) -> None:
        state = self.get_state()
        if self.undo_stack and state.mobjects_match(self.undo_stack[-1]):
            return
        self.redo_stack = []
        self.undo_stack.append(state)
        if len(self.undo_stack) > self.max_num_saved_states:
            self.undo_stack.pop(0)

    def undo(self):
        if self.undo_stack:
            self.redo_stack.append(self.get_state())
            self.restore_state(self.undo_stack.pop())

    def redo(self):
        if self.redo_stack:
            self.undo_stack.append(self.get_state())
            self.restore_state(self.redo_stack.pop())

    @contextmanager
    def temp_skip(self):
        prev_status = self.skip_animations
        self.skip_animations = True
        try:
            yield
        finally:
            if not prev_status:
                self.stop_skipping()

    @contextmanager
    def temp_progress_bar(self):
        prev_progress = self.show_animation_progress
        self.show_animation_progress = True
        try:
            yield
        finally:
            self.show_animation_progress = prev_progress

    @contextmanager
    def temp_record(self):
        self.camera.use_window_fbo(False)
        self.file_writer.begin_insert()
        try:
            yield
        finally:
            self.file_writer.end_insert()
            self.camera.use_window_fbo(True)

    def temp_config_change(self, skip=False, record=False, progress_bar=False):
        stack = ExitStack()
        if skip:
            stack.enter_context(self.temp_skip())
        if record:
            stack.enter_context(self.temp_record())
        if progress_bar:
            stack.enter_context(self.temp_progress_bar())
        return stack

    def is_window_closing(self):
        return self.window and (self.window.is_closing or self.quit_interaction)

    # Event handling
    def set_floor_plane(self, plane: str = "xy"):
        if plane == "xy":
            self.frame.set_euler_axes("zxz")
        elif plane == "xz":
            self.frame.set_euler_axes("zxy")
        else:
            raise Exception("Only `xz` and `xy` are valid floor planes")

    def on_mouse_motion(
        self,
        point: Vect3,
        d_point: Vect3
    ) -> None:
        assert self.window is not None
        self.mouse_point.move_to(point)

        event_data = {"point": point, "d_point": d_point}
        propagate_event = EVENT_DISPATCHER.dispatch(EventType.MouseMotionEvent, **event_data)
        if propagate_event is not None and propagate_event is False:
            return

        frame = self.camera.frame
        # Handle perspective changes
        if self.window.is_key_pressed(ord(manim_config.key_bindings.pan_3d)):
            ff_d_point = frame.to_fixed_frame_point(d_point, relative=True)
            ff_d_point *= self.pan_sensitivity
            frame.increment_theta(-ff_d_point[0])
            frame.increment_phi(ff_d_point[1])
        # Handle frame movements
        elif self.window.is_key_pressed(ord(manim_config.key_bindings.pan)):
            frame.shift(-d_point)

    def on_mouse_drag(
        self,
        point: Vect3,
        d_point: Vect3,
        buttons: int,
        modifiers: int
    ) -> None:
        self.mouse_drag_point.move_to(point)
        if self.drag_to_pan:
            self.frame.shift(-d_point)

        event_data = {"point": point, "d_point": d_point, "buttons": buttons, "modifiers": modifiers}
        propagate_event = EVENT_DISPATCHER.dispatch(EventType.MouseDragEvent, **event_data)
        if propagate_event is not None and propagate_event is False:
            return

    def on_mouse_press(
        self,
        point: Vect3,
        button: int,
        mods: int
    ) -> None:
        self.mouse_drag_point.move_to(point)
        event_data = {"point": point, "button": button, "mods": mods}
        propagate_event = EVENT_DISPATCHER.dispatch(EventType.MousePressEvent, **event_data)
        if propagate_event is not None and propagate_event is False:
            return

    def on_mouse_release(
        self,
        point: Vect3,
        button: int,
        mods: int
    ) -> None:
        event_data = {"point": point, "button": button, "mods": mods}
        propagate_event = EVENT_DISPATCHER.dispatch(EventType.MouseReleaseEvent, **event_data)
        if propagate_event is not None and propagate_event is False:
            return

    def on_mouse_scroll(
        self,
        point: Vect3,
        offset: Vect3,
        x_pixel_offset: float,
        y_pixel_offset: float
    ) -> None:
        event_data = {"point": point, "offset": offset}
        propagate_event = EVENT_DISPATCHER.dispatch(EventType.MouseScrollEvent, **event_data)
        if propagate_event is not None and propagate_event is False:
            return

        rel_offset = y_pixel_offset / self.camera.get_pixel_height()
        self.frame.scale(
            1 - self.scroll_sensitivity * rel_offset,
            about_point=point
        )

    def on_key_release(
        self,
        symbol: int,
        modifiers: int
    ) -> None:
        event_data = {"symbol": symbol, "modifiers": modifiers}
        propagate_event = EVENT_DISPATCHER.dispatch(EventType.KeyReleaseEvent, **event_data)
        if propagate_event is not None and propagate_event is False:
            return

    def on_key_press(
        self,
        symbol: int,
        modifiers: int
    ) -> None:
        try:
            char = chr(symbol)
        except OverflowError:
            log.warning("The value of the pressed key is too large.")
            return

        event_data = {"symbol": symbol, "modifiers": modifiers}
        propagate_event = EVENT_DISPATCHER.dispatch(EventType.KeyPressEvent, **event_data)
        if propagate_event is not None and propagate_event is False:
            return

        if char == manim_config.key_bindings.reset:
            self.play(self.camera.frame.animate.to_default_state())
        elif char == "z" and (modifiers & (PygletWindowKeys.MOD_COMMAND | PygletWindowKeys.MOD_CTRL)):
            self.undo()
        elif char == "z" and (modifiers & (PygletWindowKeys.MOD_COMMAND | PygletWindowKeys.MOD_CTRL | PygletWindowKeys.MOD_SHIFT)):
            self.redo()
        # command + q
        elif char == manim_config.key_bindings.quit and (modifiers & (PygletWindowKeys.MOD_COMMAND | PygletWindowKeys.MOD_CTRL)):
            self.quit_interaction = True
        # Space or right arrow
        elif char == " " or symbol == PygletWindowKeys.RIGHT:
            self.hold_on_wait = False

    def on_resize(self, width: int, height: int) -> None:
        pass

    def on_show(self) -> None:
        pass

    def on_hide(self) -> None:
        pass

    def on_close(self) -> None:
        pass

    def focus(self) -> None:
        """
        Puts focus on the ManimGL window.
        """
        if not self.window:
            return
        self.window.focus()

    def set_background_color(self, background_color, background_opacity=1) -> None:
        self.camera.background_rgba = list(color_to_rgba(
            background_color, background_opacity
        ))


class SceneState():
    def __init__(self, scene: Scene, ignore: list[Mobject] | None = None):
        self.time = scene.time
        self.num_plays = scene.num_plays
        self.mobjects_to_copies = OrderedDict.fromkeys(scene.mobjects)
        if ignore:
            for mob in ignore:
                self.mobjects_to_copies.pop(mob, None)

        last_m2c = scene.undo_stack[-1].mobjects_to_copies if scene.undo_stack else dict()
        for mob in self.mobjects_to_copies:
            # If it hasn't changed since the last state, just point to the
            # same copy as before
            if mob in last_m2c and last_m2c[mob].looks_identical(mob):
                self.mobjects_to_copies[mob] = last_m2c[mob]
            else:
                self.mobjects_to_copies[mob] = mob.copy()

    def __eq__(self, state: SceneState):
        return all((
            self.time == state.time,
            self.num_plays == state.num_plays,
            self.mobjects_to_copies == state.mobjects_to_copies
        ))

    def mobjects_match(self, state: SceneState):
        return self.mobjects_to_copies == state.mobjects_to_copies

    def n_changes(self, state: SceneState):
        m2c = state.mobjects_to_copies
        return sum(
            1 - int(mob in m2c and mob.looks_identical(m2c[mob]))
            for mob in self.mobjects_to_copies
        )

    def restore_scene(self, scene: Scene):
        scene.time = self.time
        scene.num_plays = self.num_plays
        scene.mobjects = [
            mob.become(mob_copy)
            for mob, mob_copy in self.mobjects_to_copies.items()
        ]


class EndScene(Exception):
    pass


class ThreeDScene(Scene):
    samples = 4
    default_frame_orientation = (-30, 70)
    always_depth_test = True

    def add(self, *mobjects: Mobject, set_depth_test: bool = True, perp_stroke: bool = True):
        for mob in mobjects:
            if set_depth_test and not mob.is_fixed_in_frame() and self.always_depth_test:
                mob.apply_depth_test()
            if isinstance(mob, VMobject) and mob.has_stroke() and perp_stroke:
                mob.set_flat_stroke(False)
        super().add(*mobjects)
