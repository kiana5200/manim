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

    def get_image(self) -> Image:
        if self.window is not None:
            self.camera.use_window_fbo(False)
            self.camera.capture(*self.render_groups)
        image = self.camera.get_image()
        if self.window is not None:
            self.camera.use_window_fbo(True)
        return image

    def show(self) -> None:
        self.update_frame(force_draw=True)
        self.get_image().show()

    def update_frame(self, dt: float = 0, force_draw: bool = False) -> None:
        self.increment_time(dt)
        self.update_mobjects(dt)
        if self.skip_animations and not force_draw:
            return

        if self.is_window_closing():
            raise EndScene()

        if self.window and dt == 0 and not self.window.has_undrawn_event() and not force_draw:
            # In this case, there's no need for new rendering, but we
            # shoudl still listen for new events
            self.window._window.dispatch_events()
            return

        self.camera.capture(*self.render_groups)

        if self.window and not self.skip_animations:
            vt = self.time - self.virtual_animation_start_time
            rt = time.time() - self.real_animation_start_time
            time.sleep(max(vt - rt, 0))

    def emit_frame(self) -> None:
        if not self.skip_animations:
            self.file_writer.write_frame(self.camera)

    # Related to updating

    def update_mobjects(self, dt: float) -> None:
        for mobject in self.mobjects:
            mobject.update(dt)

    def should_update_mobjects(self) -> bool:
        return self.always_update_mobjects or any(
            mob.has_updaters() for mob in self.mobjects
        )

    # Related to time

    def get_time(self) -> float:
        return self.time

    def increment_time(self, dt: float) -> None:
        self.time += dt

    # Related to internal mobject organization

    def get_top_level_mobjects(self) -> list[Mobject]:
        # Return only those which are not in the family
        # of another mobject from the scene
        mobjects = self.get_mobjects()
        families = [m.get_family() for m in mobjects]

        def is_top_level(mobject):
            num_families = sum([
                (mobject in family)
                for family in families
            ])
            return num_families == 1
        return list(filter(is_top_level, mobjects))

    def get_mobject_family_members(self) -> list[Mobject]:
        return extract_mobject_family_members(self.mobjects)

    def assemble_render_groups(self):
        """
        Rendering can be more efficient when mobjects of the
        same type are grouped together, so this function creates
        Groups of all clusters of adjacent Mobjects in the scene
        """
        batches = batch_by_property(
            self.mobjects,
            lambda m: str(type(m)) + str(m.get_shader_wrapper(self.camera.ctx).get_id()) + str(m.z_index)
        )

        for group in self.render_groups:
            group.clear()
        self.render_groups = [
            batch[0].get_group_class()(*batch)
            for batch, key in batches
        ]

    @staticmethod
    def affects_mobject_list(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            func(self, *args, **kwargs)
            self.assemble_render_groups()
            return self
        return wrapper

    @affects_mobject_list
    def add(self, *new_mobjects: Mobject):
        """
        Mobjects will be displayed, from background to
        foreground in the order with which they are added.
        """
        self.remove(*new_mobjects)
        self.mobjects += new_mobjects

        # Reorder based on z_index
        id_to_scene_order = {id(m): idx for idx, m in enumerate(self.mobjects)}
        self.mobjects.sort(key=lambda m: (m.z_index, id_to_scene_order[id(m)]))

        self.id_to_mobject_map.update({
            id(sm): sm
            for m in new_mobjects
            for sm in m.get_family()
        })
        return self

    def add_mobjects_among(self, values: Iterable):
        """
        This is meant mostly for quick prototyping,
        e.g. to add all mobjects defined up to a point,
        call self.add_mobjects_among(locals().values())
        """
        self.add(*filter(
            lambda m: isinstance(m, Mobject),
            values
        ))
        return self

    @affects_mobject_list
    def replace(self, mobject: Mobject, *replacements: Mobject):
        if mobject in self.mobjects:
            index = self.mobjects.index(mobject)
            self.mobjects = [
                *self.mobjects[:index],
                *replacements,
                *self.mobjects[index + 1:]
            ]
        return self

    @affects_mobject_list
    def remove(self, *mobjects_to_remove: Mobject):
        """
        Removes anything in mobjects from scenes mobject list, but in the event that one
        of the items to be removed is a member of the family of an item in mobject_list,
        the other family members are added back into the list.

        For example, if the scene includes Group(m1, m2, m3), and we call scene.remove(m1),
        the desired behavior is for the scene to then include m2 and m3 (ungrouped).
        """
        to_remove = set(extract_mobject_family_members(mobjects_to_remove))
        new_mobjects, _ = recursive_mobject_remove(self.mobjects, to_remove)
        self.mobjects = new_mobjects

    @affects_mobject_list
    def remove_all_except(self, *mobjects_to_keep : Mobject):
        self.clear()
        self.add(*mobjects_to_keep)

    def bring_to_front(self, *mobjects: Mobject):
        self.add(*mobjects)
        return self

    @affects_mobject_list
    def bring_to_back(self, *mobjects: Mobject):
        self.remove(*mobjects)
        self.mobjects = list(mobjects) + self.mobjects
        return self

    @affects_mobject_list
    def clear(self):
        self.mobjects = []
        return self

    def get_mobjects(self) -> list[Mobject]:
        return list(self.mobjects)

    def get_mobject_copies(self) -> list[Mobject]:
        return [m.copy() for m in self.mobjects]

    def point_to_mobject(
        self,
        point: np.ndarray,
        search_set: Iterable[Mobject] | None = None,
        buff: float = 0
    ) -> Mobject | None:
        """
        E.g. if clicking on the scene, this returns the top layer mobject
        under a given point
        """
        if search_set is None:
            search_set = self.mobjects
        for mobject in reversed(search_set):
            if mobject.is_point_touching(point, buff=buff):
                return mobject
        return None

    def get_group(self, *mobjects):
        if all(isinstance(m, VMobject) for m in mobjects):
            return VGroup(*mobjects)
        else:
            return Group(*mobjects)

    def id_to_mobject(self, id_value):
        return self.id_to_mobject_map[id_value]

    def ids_to_group(self, *id_values):
        return self.get_group(*filter(
            lambda x: x is not None,
            map(self.id_to_mobject, id_values)
        ))

    def i2g(self, *id_values):
        return self.ids_to_group(*id_values)

    def i2m(self, id_value):
        return self.id_to_mobject(id_value)

    # Related to skipping

    def update_skipping_status(self) -> None:
        if self.start_at_animation_number is not None:
            if self.num_plays == self.start_at_animation_number:
                self.skip_time = self.time
                if not self.original_skipping_status:
                    self.stop_skipping()
        if self.end_at_animation_number is not None:
            if self.num_plays >= self.end_at_animation_number:
                raise EndScene()

    def stop_skipping(self) -> None:
        self.virtual_animation_start_time = self.time
        self.real_animation_start_time = time.time()
        self.skip_animations = False

    # Methods associated with running animations

    def get_time_progression(
        self,
        run_time: float,
        n_iterations: int | None = None,
        desc: str = "",
        override_skip_animations: bool = False
    ) -> list[float] | np.ndarray | ProgressDisplay:
        if self.skip_animations and not override_skip_animations:
            return [run_time]

        times = np.arange(0, run_time, 1 / self.camera.fps) + 1 / self.camera.fps

        self.file_writer.set_progress_display_description(sub_desc=desc)

        if self.show_animation_progress:
            return ProgressDisplay(
                times,
                total=n_iterations,
                leave=self.leave_progress_bars,
                ascii=True if platform.system() == 'Windows' else None,
                desc=desc,
                bar_format="{l_bar} {n_fmt:3}/{total_fmt:3} {rate_fmt}{postfix}",
            )
        else:
            return times

    def get_run_time(self, animations: Iterable[Animation]) -> float:
        return np.max([animation.get_run_time() for animation in animations])

    def get_animation_time_progression(
        self,
        animations: Iterable[Animation]
    ) -> list[float] | np.ndarray | ProgressDisplay:
        animations = list(animations)
        run_time = self.get_run_time(animations)
        description = f"{self.num_plays} {animations[0]}"
        if len(animations) > 1:
            description += ", etc."
        time_progression = self.get_time_progression(run_time, desc=description)
        return time_progression

    def get_wait_time_progression(
        self,
        duration: float,
        stop_condition: Callable[[], bool] | None = None
    ) -> list[float] | np.ndarray | ProgressDisplay:
        kw = {"desc": f"{self.num_plays} Waiting"}
        if stop_condition is not None:
            kw["n_iterations"] = -1  # So it doesn't show % progress
            kw["override_skip_animations"] = True
        return self.get_time_progression(duration, **kw)

    def pre_play(self):
        if self.presenter_mode and self.num_plays == 0:
            self.hold_loop()

        self.update_skipping_status()

        if not self.skip_animations:
            self.file_writer.begin_animation()

        if self.window:
            self.virtual_animation_start_time = self.time
            self.real_animation_start_time = time.time()

    def post_play(self):
        if not self.skip_animations:
            self.file_writer.end_animation()

        if self.preview_while_skipping and self.skip_animations and self.window is not None:
            # Show some quick frames along the way
            self.update_frame(dt=0, force_draw=True)

        self.num_plays += 1

    def begin_animations(self, animations: Iterable[Animation]) -> None:
        all_mobjects = set(self.get_mobject_family_members())
        for animation in animations:
            animation.begin()
            # Anything animated that's not already in the
            # scene gets added to the scene.  Note, for
            # animated mobjects that are in the family of
            # those on screen, this can result in a restructuring
            # of the scene.mobjects list, which is usually desired.
            if animation.mobject not in all_mobjects:
                self.add(animation.mobject)
                all_mobjects = all_mobjects.union(animation.mobject.get_family())

    def progress_through_animations(self, animations: Iterable[Animation]) -> None:
        last_t = 0
        for t in self.get_animation_time_progression(animations):
            dt = t - last_t
            last_t = t
            for animation in animations:
                animation.update_mobjects(dt)
                alpha = t / animation.run_time
                animation.interpolate(alpha)
            self.update_frame(dt)
            self.emit_frame()

    def finish_animations(self, animations: Iterable[Animation]) -> None:
        for animation in animations:
            animation.finish()
            animation.clean_up_from_scene(self)
        if self.skip_animations:
            self.update_mobjects(self.get_run_time(animations))
        else:
            self.update_mobjects(0)

    @affects_mobject_list
    def play(
        self,
        *proto_animations: Animation | _AnimationBuilder,
        run_time: float | None = None,
        rate_func: Callable[[float], float] | None = None,
        lag_ratio: float | None = None,
    ) -> None:
        if len(proto_animations) == 0:
            log.warning("Called Scene.play with no animations")
            return
        animations = list(map(prepare_animation, proto_animations))
        for anim in animations:
            anim.update_rate_info(run_time, rate_func, lag_ratio)
        self.pre_play()
        self.begin_animations(animations)
        self.progress_through_animations(animations)
        self.finish_animations(animations)
        self.post_play()

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
