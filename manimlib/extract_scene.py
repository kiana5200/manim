# 从__future__模块导入annotations特性，支持类型注解中的前向引用（可引用未定义的类型）
from __future__ import annotations

# 导入copy模块，用于对象的复制操作
import copy
# 导入inspect模块，用于获取对象的自省信息（如模块、函数等）
import inspect
# 导入sys模块，用于访问Python解释器的相关变量和功能
import sys

# 从manimlib.module_loader导入ModuleLoader类，用于模块加载
from manimlib.module_loader import ModuleLoader

# 从manimlib.config导入manim_config配置对象，包含manim的配置信息
from manimlib.config import manim_config
# 从manimlib.logger导入log对象，用于日志输出
from manimlib.logger import log
# 从manimlib.scene.interactive_scene导入InteractiveScene类，交互式场景基类
from manimlib.scene.interactive_scene import InteractiveScene
# 从manimlib.scene.scene导入Scene类，场景基类
from manimlib.scene.scene import Scene

# 从typing模块导入TYPE_CHECKING常量，用于类型检查阶段的条件执行
from typing import TYPE_CHECKING

# 如果处于类型检查阶段（非运行时执行）
if TYPE_CHECKING:
    # 定义Module类型为importlib.util.types中的ModuleType（模块类型）
    Module = importlib.util.types.ModuleType
    # 从typing模块导入Optional类型，用于表示可选值（可为None）
    from typing import Optional
    # 从addict模块导入Dict类型，一种支持属性访问的字典类型
    from addict import Dict


class BlankScene(InteractiveScene):
    """空白交互式场景类，继承自InteractiveScene"""
    def construct(self):
        """场景构建方法，执行通用导入并嵌入入交互式环境"""
        # 执行配置中定义的通用导入语句，方便在交互式场景中直接使用常用模块/类
        exec(manim_config.universal_import_line)
        # 嵌入交互式环境，允许用户在场景中实时操作
        self.embed()


def is_child_scene(obj, module):
    """
    判断一个对象是否为指定模块中的Scene子类（排除Scene本身）
    
    参数:
        obj: 待检查的对象
        module: 所属模块
        
    返回:
        bool: 是否为有效的场景子类
    """
    # 不是类则排除
    if not inspect.isclass(obj):
        return False
    # 不是Scene的子类则排除
    if not issubclass(obj, Scene):
        return False
    # 是Scene本身则排除
    if obj == Scene:
        return False
    # 不在指定模块中则排除（检查模块名前缀）
    if not obj.__module__.startswith(module.__name__):
        return False
    return True


def prompt_user_for_choice(scene_classes):
    """
    提示用户从场景类列表中选择要渲染的场景（支持名称或编号，可多选）
    
    参数:
        scene_classes: 场景类列表
        
    返回:
        list: 用户选择的场景类列表
    """
    # 建立场景名称到类的映射
    name_to_class = {}
    # 计算编号所需的最大位数（用于格式化输出）
    max_digits = len(str(len(scene_classes)))
    # 打印所有可选场景（带编号）
    for idx, scene_class in enumerate(scene_classes, start=1):
        name = scene_class.__name__
        print(f"{str(idx).zfill(max_digits)}: {name}")
        name_to_class[name] = scene_class
    try:
        # 获取用户输入（支持逗号分隔的多个选择，忽略空格）
        user_input = input("\nSelect which scene to render (by name or number): ")
        # 解析用户输入，返回对应的场景类列表
        return [
            # 若输入是数字则按索引取场景，否则按名称取
            name_to_class[split_str] if not split_str.isnumeric() else scene_classes[int(split_str) - 1]
            for split_str in user_input.replace(" ", "").split(",")
        ]
    except IndexError:
        # 处理无效编号错误
        log.error("Invalid scene number")
        sys.exit(2)
    except KeyError:
        # 处理无效名称错误
        log.error("Invalid scene name")
        sys.exit(2)
    except EOFError:
        # 处理输入结束错误（如Ctrl+D）
        sys.exit(1)


def compute_total_frames(scene_class, scene_config):
    """
    计算场景渲染的总帧数（通过预运行场景，跳过动画来快速估算）
    
    作用:
        - 用于渲染时显示总进度条
        - 提前暴露长时间运行场景的运行时错误
        
    参数:
        scene_class: 场景类
        scene_config: 场景配置
        
    返回:
        int: 总帧数
    """
    # 深拷贝配置，避免修改原始配置
    pre_config = copy.deepcopy(scene_config)
    # 预运行配置：不写入视频、不保存最后一帧、静默模式
    pre_config["file_writer_config"]["write_to_movie"] = False
    pre_config["file_writer_config"]["save_last_frame"] = False
    pre_config["file_writer_config"]["quiet"] = True
    # 跳过动画，加速预运行
    pre_config["skip_animations"] = True
    # 创建预运行场景实例并执行
    pre_scene = scene_class(**pre_config)
    pre_scene.run()
    # 计算实际动画时长（总时间 - 跳过的时间）
    total_time = pre_scene.time - pre_scene.skip_time
    # 转换为帧数（时长 × 帧率）
    return int(total_time * manim_config.camera.fps)


def scene_from_class(scene_class, scene_config: Dict, run_config: Dict):
    """
    从场景类创建场景实例（根据配置决定是否预计算总帧数）
    
    参数:
        scene_class: 场景类
        scene_config: 场景配置
        run_config: 运行配置
        
    返回:
        场景实例
    """
    fw_config = manim_config.file_writer
    # 若需要写入视频且启用预运行，则计算总帧数并加入配置
    if fw_config.write_to_movie and run_config.prerun:
        scene_config.file_writer_config.total_frames = compute_total_frames(scene_class, scene_config)
    # 创建并返回场景实例
    return scene_class(** scene_config)


def note_missing_scenes(arg_names, module_names):
    """
    检查并提示缺失的场景（在指定模块中未找到的场景名称）
    
    参数:
        arg_names: 待检查的场景名称列表（通常来自命令行参数）
        module_names: 模块中存在的场景名称列表
    """
    for name in arg_names:
        if name not in module_names:
            log.error(f"No scene named {name} found")


def get_scenes_to_render(all_scene_classes: list, scene_config: Dict, run_config: Dict):
    """
    确定需要渲染的场景列表
    
    参数:
        all_scene_classes: 所有可用的场景类列表
        scene_config: 场景配置字典
        run_config: 运行配置字典
        
    返回:
        list: 待渲染的场景实例列表
    """
    # 如果配置为渲染所有场景，或仅存在一个场景，则运行所有场景
    if run_config["write_all"] or len(all_scene_classes) == 1:
        classes_to_run = all_scene_classes
    else:
        # 建立场景名称到类的映射
        name_to_class = {sc.__name__: sc for sc in all_scene_classes}
        # 根据运行配置中的场景名称筛选待运行的场景类
        classes_to_run = [name_to_class.get(name) for name in run_config.scene_names]
        # 过滤掉不存在的场景（值为None的元素）
        classes_to_run = list(filter(lambda x: x, classes_to_run))
        # 提示缺失的场景名称
        note_missing_scenes(run_config.scene_names, name_to_class.keys())

    # 如果没有筛选出有效场景，提示用户选择
    if len(classes_to_run) == 0:
        classes_to_run = prompt_user_for_choice(all_scene_classes)

    # 为每个待运行的场景类创建场景实例并返回
    return [
        scene_from_class(scene_class, scene_config, run_config)
        for scene_class in classes_to_run
    ]


def get_scene_classes(module: Optional[Module]):
    """
    从模块中获取所有场景类
    
    参数:
        module: 待提取场景类的模块（可为None）
        
    返回:
        list: 场景类列表
    """
    if module is None:
        # 若模块为None，返回空白场景类
        return [BlankScene]
    # 若模块定义了SCENES_IN_ORDER（场景顺序列表），则使用该列表
    if hasattr(module, "SCENES_IN_ORDER"):
        return module.SCENES_IN_ORDER
    else:
        # 否则通过自省获取模块中所有Scene的有效子类
        return [
            member[1]  # member是(名称, 对象)元组，取对象部分
            for member in inspect.getmembers(
                module,
                # 筛选条件：是指定模块中的Scene子类（通过is_child_scene判断）
                lambda x: is_child_scene(x, module)
            )
        ]


def get_indent(code_lines: list[str], line_number: int) -> str:
    """
    获取Python代码中指定行的缩进（以空格字符串表示）
    
    参数:
        code_lines: 代码行列表
        line_number: 行号（从0开始）
        
    返回:
        str: 缩进的空格字符串
    """
    # 查找指定行之前最近的非空行
    try:
        line = next(filter(
            lambda line: line.strip(),  # 过滤空行
            code_lines[line_number - 1::-1]  # 从指定行的上一行向前遍历
        ))
    except StopIteration:
        # 若未找到非空行，返回空字符串
        return ""

    # 计算缩进空格数：行长度 - 去除左侧空格后的长度
    n_spaces = len(line) - len(line.lstrip())
    # 若该行以冒号结尾（如if/for/def/class等），缩进增加4个空格
    if line.endswith(":"):
        n_spaces += 4
    # 返回对应数量的空格字符串
    return n_spaces * " "


def insert_embed_line_to_module(module: Module, run_config: Dict) -> None:
    """
    向模块的场景类中插入self.embed()语句（用于交互式调试）
    
    说明:
        这是一种便捷但较 hack 的方式，当用户使用-e参数时，会在场景的construct方法末尾
        插入self.embed()，使场景运行到此处时进入交互式环境。若指定了参数，会插入到包含
        该字符串的最后一行之后。
    """
    # 获取模块源代码并按行分割
    lines = inspect.getsource(module).splitlines()
    # 获取要插入的行号
    line_number = run_config.embed_line

    # 获取目标行的缩进，并插入self.embed()语句
    indent = get_indent(lines, line_number)
    lines.insert(line_number, indent + "self.embed()")
    # 拼接修改后的代码
    new_code = "\n".join(lines)

    # 若未指定场景名称，自动查找插入行上方最近的类名作为场景名
    if not run_config.scene_names:
        # 筛选插入行之前所有以"class"开头的行
        classes = list(filter(
            lambda line: line.startswith("class"),
            lines[:line_number]
        ))
        if classes:
            # 使用正则提取类名（匹配class后的类名，直到左括号）
            from re import search
            scene_name = search(r"(\w+)\(", classes[-1])
            # 更新运行配置的场景名称
            run_config.update(scene_names=[scene_name.group(1)])
        else:
            log.error(f"No 'class' found above {line_number}!")

    # 编译并执行修改后的代码，以重新定义模块中的场景类（包含嵌入语句）
    code_object = compile(new_code, module.__name__, 'exec')
    exec(code_object, module.__dict__)


def get_module(run_config: Dict) -> Module:
    """
    获取指定的模块（支持重新加载），并在需要时插入嵌入语句
    
    参数:
        run_config: 运行配置字典
        
    返回:
        Module: 处理后的模块对象
    """
    # 通过模块加载器获取模块（根据文件名和是否重新加载的配置）
    module = ModuleLoader.get_module(run_config.file_name, run_config.is_reload)
    # 若配置了嵌入行号，向模块插入self.embed()语句
    if run_config.embed_line:
        insert_embed_line_to_module(module, run_config)
    return module


def main(scene_config: Dict, run_config: Dict):
    """
    主函数：获取模块、提取场景类、确定待渲染场景并返回
    
    参数:
        scene_config: 场景配置字典
        run_config: 运行配置字典
        
    返回:
        list: 待渲染的场景实例列表
    """
    # 获取模块
    module = get_module(run_config)
    # 从模块中提取所有场景类
    all_scene_classes = get_scene_classes(module)
    # 确定需要渲染的场景
    scenes = get_scenes_to_render(all_scene_classes, scene_config, run_config)
    # 若没有找到场景，打印提示信息
    if len(scenes) == 0:
        print("No scenes found to run")
    return scenes
