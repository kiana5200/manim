# ManimGL 场景提取与加载的核心模块，负责**从用户脚本中定位场景类、处理嵌入调试、生成可执行场景实例**，
# 是连接用户代码与框架渲染逻辑的关键桥梁，确保用户定义的 Scene 子类能被正确识别并执行。


# ------------------------------ 1. 空白场景类（默认 fallback） ------------------------------
class BlankScene(InteractiveScene):
    """
    空白交互式场景：当用户未指定脚本或脚本中无有效场景时，作为默认场景加载，
    自动执行通用导入并嵌入 IPython 交互环境（便于快速调试）。
    
    核心逻辑：`construct` 方法中执行全局导入（`manim_config.universal_import_line`，默认 `from manimlib import *`），
    然后调用 `self.embed()` 进入交互式会话，支持用户实时创建 Mobject 和动画。
    """
    def construct(self):
        exec(manim_config.universal_import_line)  # 执行全局导入，简化交互时的代码输入
        self.embed()  # 嵌入 IPython 交互环境


# ------------------------------ 2. 场景类判断与筛选工具 ------------------------------
def is_child_scene(obj, module):
    """
    判断一个对象是否为“有效场景类”（用户定义的 Scene 子类，且属于指定模块），
    用于从模块中筛选出可执行的场景。
    
    判断条件（需同时满足）：
    1. 对象是类（`inspect.isclass(obj)`）；
    2. 是 Scene 类的子类（`issubclass(obj, Scene)`）；
    3. 不是 Scene 类本身（排除基类）；
    4. 类的模块属于指定模块（避免导入其他库中的 Scene 子类）。
    
    参数：
        obj - 待判断的对象（如类、函数、变量）；
        module - 目标模块（用户脚本对应的模块）。
    返回：布尔值，True 表示是有效场景类。
    """
    if not inspect.isclass(obj):
        return False
    if not issubclass(obj, Scene):
        return False
    if obj == Scene:
        return False
    if not obj.__module__.startswith(module.__name__):
        return False
    return True


def prompt_user_for_choice(scene_classes):
    """
    当未指定目标场景或场景不存在时，在终端提示用户选择场景（支持按名称或编号选择，多场景用逗号分隔）。
    
    逻辑步骤：
    1. 构建“场景名→场景类”的映射，同时打印所有可选场景（带编号）；
    2. 读取用户输入，按逗号分割，分别解析为场景名或编号；
    3. 转换为对应的场景类列表，处理无效输入（如不存在的编号/名称）并退出。
    
    参数：scene_classes - 所有有效场景类的列表
    返回：list[Scene] - 用户选择的场景类列表
    """
    name_to_class = {}
    max_digits = len(str(len(scene_classes)))  # 编号的最大位数（用于对齐显示）
    # 打印可选场景（编号+名称）
    for idx, scene_class in enumerate(scene_classes, start=1):
        name = scene_class.__name__
        print(f"{str(idx).zfill(max_digits)}: {name}")  # 编号补零对齐（如 01: SquareScene）
        name_to_class[name] = scene_class

    try:
        user_input = input("\nSelect which scene to render (by name or number): ")
        # 分割输入（支持逗号分隔多场景，如 "1,SquareScene"）
        split_inputs = user_input.replace(" ", "").split(",")
        # 解析每个输入项为场景类
        selected_scenes = [
            name_to_class[split_str] if not split_str.isnumeric() 
            else scene_classes[int(split_str) - 1]  # 编号转索引（用户输入1对应索引0）
            for split_str in split_inputs
        ]
        return selected_scenes
    except IndexError:
        log.error("Invalid scene number")  # 编号超出范围
        sys.exit(2)
    except KeyError:
        log.error("Invalid scene name")    # 名称不存在
        sys.exit(2)
    except EOFError:
        sys.exit(1)  # 输入中断（如 Ctrl+D）


# ------------------------------ 3. 预计算总帧数（进度条支持） ------------------------------
def compute_total_frames(scene_class, scene_config):
    """
    预运行场景（跳过动画）计算总帧数：用于导出时显示完整进度条，同时提前暴露运行时错误（避免渲染到一半崩溃）。
    
    逻辑：
    1. 复制场景配置，修改为“不导出文件、不保存帧、静默模式、跳过动画”；
    2. 创建场景实例并运行（`pre_scene.run()`），仅执行时间累计，不渲染实际帧；
    3. 计算场景实际运行时间（总时间 - 跳过起始时间），乘以帧率得到总帧数。
    
    参数：
        scene_class - 待计算的场景类；
        scene_config - 场景配置字典。
    返回：int - 场景总帧数（用于进度条显示）。
    """
    pre_config = copy.deepcopy(scene_config)
    # 修改配置为预运行模式（不生成文件，仅计算时间）
    pre_config["file_writer_config"]["write_to_movie"] = False
    pre_config["file_writer_config"]["save_last_frame"] = False
    pre_config["file_writer_config"]["quiet"] = True
    pre_config["skip_animations"] = True

    # 预运行场景
    pre_scene = scene_class(**pre_config)
    pre_scene.run()

    # 计算总帧数（运行时间 × 帧率）
    total_time = pre_scene.time - pre_scene.skip_time  # 实际动画时间（排除跳过的初始时间）
    return int(total_time * manim_config.camera.fps)


# ------------------------------ 4. 场景实例化（含预计算逻辑） ------------------------------
def scene_from_class(scene_class, scene_config: Dict, run_config: Dict):
    """
    从场景类创建场景实例：若开启预计算（`run_config.prerun=True`），先计算总帧数并传入配置，
    确保导出时能显示完整进度条。
    
    参数：
        scene_class - 场景类；
        scene_config - 场景配置字典；
        run_config - 运行配置字典（含 `prerun` 开关）。
    返回：Scene - 配置完成的场景实例。
    """
    fw_config = manim_config.file_writer
    # 若导出视频且开启预计算，添加总帧数到配置
    if fw_config.write_to_movie and run_config.prerun:
        scene_config.file_writer_config.total_frames = compute_total_frames(scene_class, scene_config)
    # 创建并返回场景实例
    return scene_class(**scene_config)


# ------------------------------ 5. 目标场景筛选（确定要渲染的场景） ------------------------------
def note_missing_scenes(arg_names, module_names):
    """
    打印“场景不存在”的错误日志：当用户指定的场景名在脚本中不存在时，提示无效名称。
    
    参数：
        arg_names - 用户指定的场景名列表（如命令行输入的 "SquareScene"）；
        module_names - 脚本中实际存在的场景名列表。
    """
    for name in arg_names:
        if name not in module_names:
            log.error(f"No scene named {name} found")


def get_scenes_to_render(all_scene_classes: list, scene_config: Dict, run_config: Dict):
    """
    从所有有效场景中筛选出“要渲染的场景”，处理三种情况：
    1. 导出所有场景（`run_config.write_all=True`）或仅一个场景：直接使用所有场景；
    2. 用户指定场景名：筛选出匹配的场景，提示无效名称；
    3. 无匹配场景：提示用户选择。
    
    最终返回“配置完成的场景实例列表”（调用 `scene_from_class` 实例化）。
    
    参数：
        all_scene_classes - 脚本中所有有效场景类；
        scene_config - 场景配置字典；
        run_config - 运行配置字典（含 `write_all`、`scene_names` 等）。
    返回：list[Scene] - 要渲染的场景实例列表。
    """
    # 情况1：导出所有场景或仅一个场景
    if run_config["write_all"] or len(all_scene_classes) == 1:
        classes_to_run = all_scene_classes
    # 情况2：用户指定场景名，筛选匹配的场景
    else:
        name_to_class = {sc.__name__: sc for sc in all_scene_classes}  # 场景名→场景类映射
        classes_to_run = [name_to_class.get(name) for name in run_config.scene_names]  # 按名筛选
        classes_to_run = list(filter(lambda x: x, classes_to_run))  # 过滤 None（无效名称）
        note_missing_scenes(run_config.scene_names, name_to_class.keys())  # 提示无效名称

    # 情况3：无匹配场景，提示用户选择
    if len(classes_to_run) == 0:
        classes_to_run = prompt_user_for_choice(all_scene_classes)

    # 实例化所有要渲染的场景
    return [
        scene_from_class(scene_class, scene_config, run_config)
        for scene_class in classes_to_run
    ]


# ------------------------------ 6. 从模块中提取场景类 ------------------------------
def get_scene_classes(module: Optional[Module]):
    """
    从用户脚本模块中提取所有有效场景类，优先使用模块的 `SCENES_IN_ORDER` 变量（用户指定的场景顺序），
    否则自动筛选所有符合条件的 Scene 子类。
    
    参数：module - 用户脚本对应的模块（None 表示无脚本）
    返回：list[Scene] - 提取的有效场景类列表（无模块时返回 [BlankScene]）。
    """
    if module is None:
        return [BlankScene]  # 无模块时返回空白场景
    # 优先使用用户指定的场景顺序（模块中定义的 SCENES_IN_ORDER 列表）
    if hasattr(module, "SCENES_IN_ORDER"):
        return module.SCENES_IN_ORDER
    # 否则自动筛选模块中所有有效场景类
    else:
        return [
            member[1]  # member 是 (名称, 对象) 元组，取对象部分
            for member in inspect.getmembers(
                module,
                lambda x: is_child_scene(x, module)  # 筛选条件：有效场景类
            )
        ]


# ------------------------------ 7. 嵌入调试：代码注入工具 ------------------------------
def get_indent(code_lines: list[str], line_number: int) -> str:
    """
    计算指定行代码的缩进（空格字符串）：用于嵌入 `self.embed()` 时保持代码语法正确。
    
    逻辑：
    1. 从指定行向上查找最近的非空行；
    2. 若该行以冒号结尾（如 if/for/def 语句），缩进增加4个空格（符合 Python 语法）；
    3. 返回最终的缩进空格字符串。
    
    参数：
        code_lines - 脚本代码按行分割的列表；
        line_number - 目标行号（1-based）。
    返回：str - 缩进空格字符串（如 "    " 表示4个空格）。
    """
    try:
        # 向上查找最近的非空行（从 line_number-1 开始，0-based）
        line = next(filter(lambda line: line.strip(), code_lines[line_number - 1::-1]))
    except StopIteration:
        return ""  # 无有效行，返回空缩进

    # 计算该行的缩进空格数
    n_spaces = len(line) - len(line.lstrip())
    # 若该行以冒号结尾（代码块开始），缩进+4
    if line.endswith(":"):
        n_spaces += 4
    return n_spaces * " "


def insert_embed_line_to_module(module: Module, run_config: Dict) -> None:
    """
    嵌入调试：在用户脚本的指定行注入 `self.embed()` 代码，进入交互式会话（`-e <行号>` 参数触发），
    便于在场景运行过程中调试（如查看 Mobject 状态）。
    
    核心逻辑（hacky 但高效）：
    1. 读取模块源代码，按行分割；
    2. 计算目标行的缩进，插入 `self.embed()`；
    3. 重新编译并执行修改后的代码，覆盖模块中的场景类定义；
    4. 若未指定场景名，自动识别目标行上方最近的场景类作为目标。
    
    参数：
        module - 用户脚本模块；
        run_config - 运行配置字典（含 `embed_line`、`scene_names` 等）。
    """
    # 读取模块源代码并分割成行
    lines = inspect.getsource(module).splitlines()
    line_number = run_config.embed_line  # 目标行号（1-based）

    # 计算缩进并插入 self.embed()
    indent = get_indent(lines, line_number)
    lines.insert(line_number, indent + "self.embed()")  # 插入调试代码
    new_code = "\n".join(lines)  # 重组修改后的代码

    # 自动识别目标场景（未指定场景名时）
    if not run_config.scene_names:
        # 查找目标行上方所有类定义
        classes = list(filter(lambda line: line.startswith("class"), lines[:line_number]))
        if classes:
            import re
            # 提取最后一个类的名称（如 "class SquareScene(Scene):" → "SquareScene"）
            scene_name_match = re.search(r"(\w+)\(", classes[-1])
            if scene_name_match:
                run_config.update(scene_names=[scene_name_match.group(1)])
        else:
            log.error(f"No 'class' found above {line_number}!")  # 无类定义，提示错误

    # 重新编译并执行修改后的代码，覆盖模块中的场景类
    code_object = compile(new_code, module.__name__, 'exec')
    exec(code_object, module.__dict__)


# ------------------------------ 8. 模块加载与场景提取主流程 ------------------------------
def get_module(run_config: Dict) -> Module:
    """
    加载用户脚本模块：若开启嵌入调试（`run_config.embed_line` 非空），先注入调试代码，
    再返回加载后的模块。
    
    参数：run_config - 运行配置字典（含 `file_name`、`is_reload`、`embed_line` 等）
    返回：Module - 加载并处理后的用户脚本模块。
    """
    # 加载模块（支持自动重载，`is_reload` 控制是否重新加载）
    module = ModuleLoader.get_module(run_config.file_name, run_config.is_reload)
    # 若开启嵌入调试，注入 self.embed() 代码
    if run_config.embed_line:
        insert_embed_line_to_module(module, run_config)
    return module


def main(scene_config: Dict, run_config: Dict):
    """
    场景提取主函数：加载模块 → 提取场景类 → 筛选目标场景 → 实例化场景，
    是 `run_scenes` 函数中获取可执行场景的核心调用。
    
    参数：
        scene_config - 场景配置字典；
        run_config - 运行配置字典。
    返回：list[Scene] - 可执行的场景实例列表（无场景时打印提示）。
    """
    # 1. 加载用户模块（含嵌入调试处理）
    module = get_module(run_config)
    # 2. 提取模块中所有有效场景类
    all_scene_classes = get_scene_classes(module)
    # 3. 筛选并实例化要渲染的场景
    scenes = get_scenes_to_render(all_scene_classes, scene_config, run_config)

    # 无场景时打印提示
    if len(scenes) == 0:
        print("No scenes found to run")
    return scenes