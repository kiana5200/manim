# 导入Python未来版本的注解特性（用于支持更灵活的类型提示，如字符串形式的类名）
from __future__ import annotations

# 导入所需的标准库模块
import inspect  # 用于检查对象的源代码、类结构、函数参数等（常用于反射和调试）
import pyperclip  # 用于访问系统剪贴板，实现复制和粘贴功能（如之前代码中的复制选中对象、坐标等）
import traceback  # 用于捕获和格式化异常信息，便于调试时查看错误堆栈

# 导入IPython相关模块（用于交互式环境集成）
from IPython.terminal import pt_inputhooks  # IPython终端的输入钩子，用于处理外部事件（如GUI事件循环）
from IPython.terminal.embed import InteractiveShellEmbed  # 用于在代码中嵌入IPython交互式shell，支持实时调试和交互

# 导入Manim库相关模块（Manim是用于数学动画制作的库）
from manimlib.animation.fading import VFadeInThenOut  # 导入淡入淡出动画类（用于对象的淡入后淡出效果）
from manimlib.config import manim_config  # 导入Manim的全局配置对象（存储渲染参数、路径设置等）
from manimlib.constants import RED  # 导入Manim预定义的颜色常量（此处导入红色，用于设置对象颜色）
from manimlib.mobject.mobject import Mobject  # 导入Manim的基础图形对象类（所有可见图形对象的父类）
from manimlib.mobject.frame import FullScreenRectangle  # 导入全屏矩形类（用于创建覆盖整个屏幕的矩形，如背景、遮罩等）
from manimlib.module_loader import ModuleLoader  # 导入模块加载器类（用于动态加载Manim场景、自定义模块等）

# 导入类型提示相关模块（仅在类型检查时生效，不影响运行时）
from typing import TYPE_CHECKING
# 条件导入：仅当执行类型检查时（如使用mypy工具），才导入Scene类（避免循环导入问题）
if TYPE_CHECKING:
    from manimlib.scene.scene import Scene  # 导入场景基类，用于类型注解（如函数参数、返回值的类型提示）

class InteractiveSceneEmbed:
    """
    交互式场景嵌入类，用于在Manim场景中嵌入IPython交互式终端
    核心功能：提供实时代码交互环境，支持直接调用场景方法（如play、add），并集成GUI交互和错误反馈
    """
    def __init__(self, scene: Scene):
        """
        初始化交互式场景嵌入实例
        
        参数：
            scene: 要嵌入交互终端的Manim场景对象（所有交互操作围绕该场景展开）
        """
        # 保存目标场景对象（后续所有终端命令都作用于该场景）
        self.scene = scene
        # 初始化检查点管理器（用于管理场景状态的检查点，支持保存/恢复场景状态）
        self.checkpoint_manager = CheckpointManager()

        # 创建并配置用于场景交互的IPython嵌入式终端
        self.shell = self.get_ipython_shell_for_embedded_scene()
        # 启用GUI交互（确保终端运行时，场景窗口仍能响应鼠标/键盘事件）
        self.enable_gui()
        # 确保IPython单元格执行后，场景帧能实时更新（避免执行代码后画面无变化）
        self.ensure_frame_update_post_cell()
        # 配置错误反馈：发生异常时，场景窗口显示红色边框闪烁（提升错误感知）
        self.ensure_flash_on_error()
        # 若配置开启自动重载，启用场景自动重载功能（修改代码后自动更新场景）
        if manim_config.embed.autoreload:
            self.auto_reload()

    def launch(self):
        """启动IPython嵌入式终端，进入交互式环境"""
        self.shell()  # 执行IPython终端，此时用户可在终端中输入代码操作场景

    def get_ipython_shell_for_embedded_scene(self) -> InteractiveShellEmbed:
        """
        创建并配置适用于场景交互的IPython嵌入式终端
        核心：让终端能访问用户场景定义的局部变量，实现"在场景上下文内交互"
        
        返回：配置完成的IPython嵌入式终端对象
        """
        # 获取调用栈中"用户调用scene.embed()"的帧（找到用户场景定义的上下文）
        # f_back三次跳转：1.当前方法 → 2.__init__ → 3.用户调用embed()的代码帧
        caller_frame = inspect.currentframe().f_back.f_back.f_back

        # 加载用户场景所在的模块（从调用帧的全局变量中获取文件路径）
        module = ModuleLoader.get_module(caller_frame.f_globals["__file__"])
        # 将用户场景的局部变量（如自定义mobject、方法）注入模块命名空间
        # 目的：用户在终端中可直接使用这些变量（无需额外导入）
        module.__dict__.update(caller_frame.f_locals)
        # 将自定义快捷方法（如play、undo）注入模块命名空间，简化终端操作
        module.__dict__.update(self.get_shortcuts())
        # 从配置中获取异常显示模式（如"Context"模式显示详细错误上下文）
        exception_mode = manim_config.embed.exception_mode

        # 创建IPython嵌入式终端并配置
        return InteractiveShellEmbed(
            user_module=module,  # 终端的默认模块为用户场景所在模块（确保变量可访问）
            display_banner=False,  # 不显示IPython默认启动横幅（简化交互界面）
            xmode=exception_mode  # 设置异常显示模式（按配置展示错误信息）
        )

    def get_shortcuts(self):
        """
        定义终端中的自定义快捷方法，将场景常用方法（如play、add）直接暴露给用户
        目的：简化终端命令（如输入play(...)而非scene.play(...)）
        
        返回：包含快捷方法的字典（键为方法名，值为场景对应方法）
        """
        scene = self.scene  # 简化场景对象引用
        return dict(
            play=scene.play,          # 场景动画播放方法
            wait=scene.wait,          # 场景等待方法（暂停指定时间）
            add=scene.add,            # 向场景添加mobject方法
            remove=scene.remove,      # 从场景移除mobject方法
            clear=scene.clear,        # 清空场景所有mobject方法
            focus=scene.focus,        # 场景相机聚焦到指定对象方法
            save_state=scene.save_state,  # 保存场景当前状态方法
            undo=scene.undo,          # 撤销上一步操作（恢复到上一状态）
            redo=scene.redo,          # 重做上一步撤销的操作
            i2g=scene.i2g,            # （推测）将屏幕坐标转换为场景坐标的工具方法
            i2m=scene.i2m,            # （推测）将屏幕坐标映射到mobject的工具方法
            checkpoint_paste=self.checkpoint_paste,  # 从检查点粘贴场景状态的方法
            clear_checkpoints=self.checkpoint_manager.clear_checkpoints,  # 清空所有检查点的方法
            reload=self.reload_scene  # 场景重载方法（下方定义，用于重新加载场景代码）
        )

    def enable_gui(self):
        """
        启用GUI交互支持：确保IPython终端运行时，场景窗口仍能响应鼠标/键盘事件
        原理：通过IPython输入钩子，在等待用户输入时持续更新场景帧
        """
        def inputhook(context):
            """
            IPython输入钩子函数：在终端等待用户输入时执行，处理场景GUI事件
            context: IPython输入上下文对象（用于判断是否有用户输入）
            """
            # 当终端无输入且场景窗口未关闭时，持续更新场景帧（dt=0表示无时间流逝）
            while not context.input_is_ready():
                if not self.scene.is_window_closing():
                    self.scene.update_frame(dt=0)
            # 若场景窗口被关闭，触发IPython终端退出（避免终端悬浮）
            if self.scene.is_window_closing():
                self.shell.ask_exit()

        # 注册自定义输入钩子（命名为"manim"），并启用该GUI钩子
        pt_inputhooks.register("manim", inputhook)
        self.shell.enable_gui("manim")

    def ensure_frame_update_post_cell(self):
        """
        确保IPython单元格执行完成后，场景帧强制更新
        解决问题：用户执行修改场景的代码（如add(mob)）后，画面未实时刷新的问题
        """
        def post_cell_func(*args, **kwargs):
            """IPython单元格执行后的回调函数：强制更新场景帧"""
            # 若场景窗口未关闭，强制更新帧（force_draw=True确保重新渲染画面）
            if not self.scene.is_window_closing():
                self.scene.update_frame(dt=0, force_draw=True)

        # 为IPython注册"post_run_cell"事件：单元格执行后触发上述回调
        self.shell.events.register("post_run_cell", post_cell_func)

    def ensure_flash_on_error(self):
        """
        配置错误视觉反馈：当终端执行代码抛出异常时，场景窗口显示红色边框闪烁
        目的：让用户直观感知到代码执行出错（避免只看终端才发现错误）
        """
        def custom_exc(shell, etype, evalue, tb, tb_offset=None):
            """
            IPython自定义异常处理函数：捕获异常后显示错误信息并触发视觉反馈
            参数：
                shell: IPython终端对象
                etype: 异常类型（如ValueError）
                evalue: 异常实例
                tb: 异常追踪栈
                tb_offset: 追踪栈偏移（用于定位错误位置）
            """
            # 先调用IPython默认的异常显示方法（确保错误信息在终端正常打印）
            shell.showtraceback((etype, evalue, tb), tb_offset=tb_offset)
            
            # 创建全屏红色边框矩形（用于闪烁提示）
            # 无填充（opacity=0），仅显示30像素宽的红色描边
            error_rect = FullScreenRectangle().set_stroke(RED, 30).set_fill(opacity=0)
            error_rect.fix_in_frame()  # 固定在场景帧中（不受相机移动影响）
            
            # 播放"淡入后淡出"动画：红色边框闪烁0.5秒，提示用户发生错误
            self.scene.play(VFadeInThenOut(error_rect, run_time=0.5))

        # 为IPython注册自定义异常处理器：捕获所有Exception类型异常并触发反馈
        self.shell.set_custom_exc((Exception,), custom_exc)

    def reload_scene(self, embed_line: int | None = None) -> None:
    """
    重新加载场景，模拟`manimgl`命令的启动逻辑（使用初始启动参数）
    核心价值：场景开发时无需退出IPython内核并重新运行`manimgl`，GUI窗口保持打开，实现快速迭代
    
    参数说明：
        embed_line: 可选整数，指定场景重新加载后嵌入交互终端的行号
                    对应`extract_scene.insert_embed_line_to_module()`方法的`linemarker`参数
    工作原理：
        重新加载前，场景会被清空且状态完全重置（回到初始干净状态），
        该逻辑由`__main__.py`中的`run_scenes`函数处理，它会捕获本方法触发的`exit_raise`异常
    注意：
        无法为该功能定义自定义异常类，因为IPython内核会吞掉所有异常；
        即使通过`set_custom_exc`注册的自定义异常处理器能捕获异常，也无法通过这种方式退出IPython shell
    """
    # 更新全局运行配置：标记当前为"重新加载"状态
    run_config = manim_config.run
    run_config.is_reload = True
    # 若指定了嵌入行号，将其写入运行配置（用于重新加载后定位嵌入终端的位置）
    if embed_line:
        run_config.embed_line = embed_line

    # 打印重新加载提示，告知用户当前操作
    print("Reloading...")
    # 执行IPython行魔法命令`exit_raise`：触发退出并抛出异常，供`run_scenes`捕获以启动重新加载流程
    self.shell.run_line_magic("exit_raise", "")

def auto_reload(self):
    """
    启用"单元格执行前自动重载模块"功能
    作用：每次执行IPython单元格前，自动重新加载终端关联的用户模块，确保代码修改实时生效
    """
    def pre_cell_func(*args, **kwargs):
        """IPython单元格执行前的回调函数：重新加载用户模块并更新命名空间"""
        # 重新加载用户模块（指定`is_during_reload=True`，标识当前为重载过程）
        new_mod = ModuleLoader.get_module(self.shell.user_module.__file__, is_during_reload=True)
        # 将重新加载后的模块变量更新到IPython用户命名空间（确保单元格执行时使用最新代码）
        self.shell.user_ns.update(vars(new_mod))

    # 为IPython注册"pre_run_cell"事件：单元格执行前触发上述回调函数
    self.shell.events.register("pre_run_cell", pre_cell_func)

def checkpoint_paste(
    self,
    skip: bool = False,
    record: bool = False,
    progress_bar: bool = True
):
    """
    从剪贴板粘贴代码并结合检查点运行（支持场景状态的保存与恢复）
    作用：交互式开发中运行（或重新运行）一段场景代码，基于检查点实现状态回溯
    
    参数说明：
        skip: 是否跳过动画（True则直接显示结果，不播放动画）
        record: 是否记录动画（True则生成动画文件，False仅实时显示）
        progress_bar: 是否显示进度条（True则在动画播放时显示进度）
    实现逻辑：
        在临时配置（skip/record/progress_bar）下执行粘贴的代码，
        检查点管理由`CheckpointManager`处理（根据代码注释决定是否恢复到历史状态）
    """
    # 使用临时配置执行检查点粘贴（临时修改场景配置，执行后自动恢复原配置）
    with self.scene.temp_config_change(skip, record, progress_bar):
        # 调用检查点管理器的粘贴方法，传入IPython终端和场景对象
        self.checkpoint_manager.checkpoint_paste(self.shell, self.scene)


class CheckpointManager:
    """
    检查点管理器：用于管理场景状态的检查点，支持基于代码注释的状态保存与恢复
    核心功能：根据剪贴板代码的开头注释，关联对应的场景状态，实现"同一代码块重复运行时回溯到初始状态"
    """
    def __init__(self):
        """初始化检查点管理器：创建存储检查点状态的字典"""
        # 检查点状态字典：key=代码开头的注释字符串，value=场景状态列表（包含Mobject的状态快照）
        self.checkpoint_states: dict[str, list[tuple[Mobject, Mobject]]] = dict()

    def checkpoint_paste(self, shell, scene):
        """
        核心方法：从剪贴板获取代码，基于开头注释匹配检查点，恢复场景状态后运行代码
        
        参数说明：
            shell: IPython终端对象（用于执行剪贴板中的代码）
            scene: 当前操作的场景对象（用于保存/恢复状态）
        逻辑：
            1. 从剪贴板读取代码字符串
            2. 提取代码开头的注释作为检查点key
            3. 根据key处理检查点（恢复历史状态或保存新状态）
            4. 在IPython终端中运行剪贴板代码
        """
        # 从剪贴板读取代码字符串（用户复制的场景代码）
        code_string = pyperclip.paste()
        # 提取代码开头的注释作为检查点key（由get_leading_comment静态方法处理）
        checkpoint_key = self.get_leading_comment(code_string)
        # 根据检查点key处理场景状态（恢复或保存）
        self.handle_checkpoint_key(scene, checkpoint_key)
        # 在IPython终端中执行剪贴板中的代码
        shell.run_cell(code_string)

    @staticmethod
    def get_leading_comment(code_string: str) -> str:
        """
        静态方法：提取代码字符串开头的注释（第一行非空且以#开头的内容）
        
        参数：code_string - 从剪贴板读取的代码字符串
        返回：提取到的注释字符串（若第一行不是注释则返回空字符串）
        """
        # 分割代码字符串的第一行（以\n为分隔符，取第一个部分），并去除左侧空白
        leading_line = code_string.partition("\n")[0].lstrip()
        # 若第一行以#开头（是注释），返回该注释；否则返回空字符串
        if leading_line.startswith("#"):
            return leading_line
        return ""

    def handle_checkpoint_key(self, scene, key: str):
        """
        根据检查点key处理场景状态：
        - 若key已存在：恢复场景到该key对应的状态，并删除该key之后的所有检查点（确保状态回溯）
        - 若key不存在：保存当前场景状态到该key（首次运行该注释开头的代码时）
        
        参数说明：
            scene: 当前场景对象（用于获取/恢复状态）
            key: 检查点key（代码开头的注释字符串）
        """
        # 若key为空（代码开头无注释），不处理状态（直接运行代码）
        if not key:
            return
        # 若key已存在于检查点字典中（该注释开头的代码之前运行过）
        elif key in self.checkpoint_states:
            # 恢复场景到该key对应的状态
            scene.restore_state(self.checkpoint_states[key])

            # 删除该key之后添加的所有检查点（确保状态回溯后，后续检查点失效）
            # 获取所有检查点key的列表
            all_keys = list(self.checkpoint_states.keys())
            # 找到当前key在列表中的索引
            index = all_keys.index(key)
            # 遍历并删除索引之后的所有key（清除后续检查点）
            for later_key in all_keys[index + 1:]:
                self.checkpoint_states.pop(later_key)
        # 若key不存在（首次运行该注释开头的代码）
        else:
            # 获取当前场景的状态，并保存到检查点字典中（key为注释）
            self.checkpoint_states[key] = scene.get_state()

    def clear_checkpoints(self):
        """清空所有检查点状态：重置检查点字典，删除所有保存的场景状态"""
        self.checkpoint_states = dict()