# 从__future__模块导入annotations，支持延迟类型注解解析（Python 3.7+特性）
from __future__ import annotations

# 导入argparse模块，用于解析命令行参数
import argparse
# 导入colour模块，用于颜色处理和转换
import colour
# 导入importlib模块，用于动态导入Python模块
import importlib
# 导入inspect模块，用于获取对象的信息（如模块、函数的结构等）
import inspect
# 导入os模块，用于与操作系统交互（如文件路径、环境变量等）
import os
# 导入sys模块，用于访问Python解释器的相关变量和功能
import sys
# 导入yaml模块，用于解析和生成YAML格式的配置文件
import yaml
# 从pathlib模块导入Path类，用于面向对象的文件路径操作
from pathlib import Path
# 从ast模块导入literal_eval函数，用于安全地解析字符串为Python字面量
from ast import literal_eval
# 从addict库导入Dict类，提供支持属性访问的字典扩展功能
from addict import Dict

# 从manimlib.logger模块导入log对象，用于日志记录
from manimlib.logger import log
# 从manimlib.utils.dict_ops模块导入merge_dicts_recursively函数，用于递归合并字典
from manimlib.utils.dict_ops import merge_dicts_recursively

# 从typing模块导入TYPE_CHECKING常量，用于条件性导入类型注解（仅在类型检查时生效）
from typing import TYPE_CHECKING
# 如果处于类型检查阶段（非运行时执行），导入特定的类型注解
if TYPE_CHECKING:
    # 从argparse模块导入Namespace类型，用于命令行参数解析结果的类型注解
    from argparse import Namespace
    # 从typing模块导入Optional类型，用于表示可选值（可为None的类型）
    from typing import Optional


def initialize_manim_config() -> Dict:
    """
    Return default configuration for various classes in manim, such as
    Scene, Window, Camera, and SceneFileWriter, as well as configuration
    determining how the scene is run (e.g. written to file or previewed in window).

    The result is initially on the contents of default_config.yml in the manimlib directory,
    which can be further updated by a custom configuration file custom_config.yml.
    It is further updated based on command line argument.
    """
    args = parse_cli()
    global_defaults_file = os.path.join(get_manim_dir(), "manimlib", "default_config.yml")
    config = Dict(merge_dicts_recursively(
        load_yaml(global_defaults_file),
        load_yaml("custom_config.yml"),  # From current working directory
        load_yaml(args.config_file) if args.config_file else dict(),
    ))

    log.setLevel(args.log_level or config["log_level"])

    update_directory_config(config)
    update_window_config(config, args)
    update_camera_config(config, args)
    update_file_writer_config(config, args)
    update_scene_config(config, args)
    update_run_config(config, args)
    update_embed_config(config, args)

    return config


def parse_cli():
    try:
        parser = argparse.ArgumentParser()
        module_location = parser.add_mutually_exclusive_group()
        module_location.add_argument(
            "file",
            nargs="?",
            help="Path to file holding the python code for the scene",
        )
        parser.add_argument(
            "scene_names",
            nargs="*",
            help="Name of the Scene class you want to see",
        )
        parser.add_argument(
            "-w", "--write_file",
            action="store_true",
            help="Render the scene as a movie file",
        )
        parser.add_argument(
            "-s", "--skip_animations",
            action="store_true",
            help="Save the last frame",
        )
        parser.add_argument(
            "-l", "--low_quality",
            action="store_true",
            help="Render at 480p",
        )
        parser.add_argument(
            "-m", "--medium_quality",
            action="store_true",
            help="Render at 720p",
        )
        parser.add_argument(
            "--hd",
            action="store_true",
            help="Render at a 1080p",
        )
        parser.add_argument(
            "--uhd",
            action="store_true",
            help="Render at a 4k",
        )
        parser.add_argument(
            "-f", "--full_screen",
            action="store_true",
            help="Show window in full screen",
        )
        parser.add_argument(
            "-p", "--presenter_mode",
            action="store_true",
            help="Scene will stay paused during wait calls until " + \
                 "space bar or right arrow is hit, like a slide show"
        )
        parser.add_argument(
            "-i", "--gif",
            action="store_true",
            help="Save the video as gif",
        )
        parser.add_argument(
            "-t", "--transparent",
            action="store_true",
            help="Render to a movie file with an alpha channel",
        )
        parser.add_argument(
            "--vcodec",
            help="Video codec to use with ffmpeg",
        )
        parser.add_argument(
            "--pix_fmt",
            help="Pixel format to use for the output of ffmpeg, defaults to `yuv420p`",
        )
        parser.add_argument(
            "-q", "--quiet",
            action="store_true",
            help="",
        )
        parser.add_argument(
            "-a", "--write_all",
            action="store_true",
            help="Write all the scenes from a file",
        )
        parser.add_argument(
            "-o", "--open",
            action="store_true",
            help="Automatically open the saved file once its done",
        )
        parser.add_argument(
            "--finder",
            action="store_true",
            help="Show the output file in finder",
        )
        parser.add_argument(
            "--subdivide",
            action="store_true",
            help="Divide the output animation into individual movie files " +
                 "for each animation",
        )
        parser.add_argument(
            "--file_name",
            help="Name for the movie or image file",
        )
        parser.add_argument(
            "-n", "--start_at_animation_number",
            help="Start rendering not from the first animation, but " + \
                 "from another, specified by its index.  If you pass " + \
                 "in two comma separated values, e.g. \"3,6\", it will end " + \
                 "the rendering at the second value",
        )
        parser.add_argument(
            "-e", "--embed",
            metavar="LINE_NUMBER",
            help="Adds a breakpoint at the inputted file dropping into an " + \
                 "interactive iPython session at that point of the code."
        )
        parser.add_argument(
            "-r", "--resolution",
            help="Resolution, passed as \"WxH\", e.g. \"1920x1080\"",
        )
        parser.add_argument(
            "--fps",
            help="Frame rate, as an integer",
            type=int,
        )
        parser.add_argument(
            "-c", "--color",
            help="Background color",
        )
        parser.add_argument(
            "--leave_progress_bars",
            action="store_true",
            help="Leave progress bars displayed in terminal",
        )
        parser.add_argument(
            "--show_animation_progress",
            action="store_true",
            help="Show progress bar for each animation",
        )
        parser.add_argument(
            "--prerun",
            action="store_true",
            help="Calculate total framecount, to display in a progress bar, by doing " + \
                 "an initial run of the scene which skips animations."
        )
        parser.add_argument(
            "--video_dir",
            help="Directory to write video",
        )
        parser.add_argument(
            "--config_file",
            help="Path to the custom configuration file",
        )
        parser.add_argument(
            "-v", "--version",
            action="store_true",
            help="Display the version of manimgl"
        )
        parser.add_argument(
            "--log-level",
            help="Level of messages to Display, can be DEBUG / INFO / WARNING / ERROR / CRITICAL"
        )
        parser.add_argument(
            "--clear-cache",
            action="store_true",
            help="Erase the cache used for Tex and Text Mobjects"
        )
        parser.add_argument(
            "--autoreload",
            action="store_true",
            help="Automatically reload Python modules to pick up code changes " +
                 "across different files",
        )
        args = parser.parse_args()
        args.write_file = any([args.write_file, args.open, args.finder])
        return args
    except argparse.ArgumentError as err:
        log.error(str(err))
        sys.exit(2)


def update_directory_config(config: Dict):
    dir_config = config.directories
    base = dir_config.base
    for key, subdir in dir_config.subdirs.items():
        dir_config[key] = os.path.join(base, subdir)


def update_window_config(config: Dict, args: Namespace):
    window_config = config.window
    for key in "position", "size":
        if window_config.get(key):
            window_config[key] = literal_eval(window_config[key])
    if args.full_screen:
        window_config.full_screen = True


def update_camera_config(config: Dict, args: Namespace):
    camera_config = config.camera
    arg_resolution = get_resolution_from_args(args, config.resolution_options)
    camera_config.resolution = arg_resolution or literal_eval(camera_config.resolution)
    if args.fps:
        camera_config.fps = args.fps
    if args.color:
        try:
            camera_config.background_color = colour.Color(args.color)
        except Exception as err:
            log.error("Please use a valid color")
            log.error(err)
            sys.exit(2)
    if args.transparent:
        camera_config.background_opacity = 0.0


def update_file_writer_config(config: Dict, args: Namespace):
    file_writer_config = config.file_writer
    file_writer_config.update(
        write_to_movie=(not args.skip_animations and args.write_file),
        subdivide_output=args.subdivide,
        save_last_frame=(args.skip_animations and args.write_file),
        png_mode=("RGBA" if args.transparent else "RGB"),
        movie_file_extension=(get_file_ext(args)),
        output_directory=get_output_directory(args, config),
        file_name=args.file_name,
        open_file_upon_completion=args.open,
        show_file_location_upon_completion=args.finder,
        quiet=args.quiet,
    )

    if args.vcodec:
        file_writer_config.video_codec = args.vcodec
    elif args.transparent:
        file_writer_config.video_codec = 'prores_ks'
        file_writer_config.pixel_format = ''
    elif args.gif:
        file_writer_config.video_codec = ''

    if args.pix_fmt:
        file_writer_config.pixel_format = args.pix_fmt

def update_scene_config(config: Dict, args: Namespace):
    """
    初始化“场景配置”：设置场景的核心行为（如是否跳过动画、演示模式、动画范围），
    直接影响场景的渲染逻辑（如 `Scene.play()` 的执行方式）。
    
    核心逻辑：
    1. 获取动画范围：调用 `get_animations_numbers` 解析 `--start_at_animation_number` 参数，
       得到动画起始/结束编号（如 "3,6" → (3,6)，仅 "3" → (3, None)）；
    2. 基础行为配置：将命令行参数（如跳过动画、演示模式）映射到场景配置，覆盖默认值；
    3. 进度条配置：若指定 `--leave_progress_bars` 或 `--show_animation_progress`，
       开启对应的进度条显示逻辑（提升用户对渲染进度的感知）。
    
    参数：
        config - 全局配置字典（需包含 `scene` 字段）；
        args - 命令行参数对象（含 `skip_animations`、`presenter_mode` 等）。
    """
    scene_config = config.scene  # 获取场景配置子字典
    # 1. 解析动画起始/结束编号（控制渲染的动画范围）
    start_anim_num, end_anim_num = get_animations_numbers(args)
    
    # 2. 更新场景核心配置
    scene_config.update({
        "camera_config": dict(),  # 预留相机配置（后续由相机模块补充）
        "file_writer_config": dict(),  # 预留文件写入器配置（后续由写入器模块补充）
        "skip_animations": args.skip_animations,  # 是否跳过动画（仅渲染最后一帧）
        "start_at_animation_number": start_anim_num,  # 动画起始编号
        "end_at_animation_number": end_anim_num,  # 动画结束编号
        "presenter_mode": args.presenter_mode,  # 是否开启演示模式（等待空格键继续）
    })
    
    # 3. 进度条显示配置（命令行参数优先）
    if args.leave_progress_bars:
        scene_config.leave_progress_bars = True  # 保留终端进度条
    if args.show_animation_progress:
        scene_config.show_animation_progress = True  # 显示每个动画的进度条


def update_run_config(config: Dict, args: Namespace):
    """
    初始化“运行配置”：设置 ManimGL 整体的运行行为（如脚本路径、嵌入调试行号、是否显示窗口），
    是框架启动和场景加载的核心控制参数。
    
    核心逻辑：
    1. 脚本与场景配置：记录用户输入的脚本文件路径（`args.file`）和目标场景名（`args.scene_names`）；
    2. 嵌入调试配置：解析 `--embed` 参数，得到调试断点行号（如 `--embed 50` → 50）；
    3. 窗口显示逻辑：默认“不导出文件时显示窗口”（`show_in_window = not args.write_file`），
       即导出文件（-w/-o）时隐藏窗口，仅后台渲染；
    4. 静默模式配置：`--quiet` 或 `--write_all` 时开启静默模式，减少日志输出。
    
    参数：
        config - 全局配置字典（会新增 `run` 字段）；
        args - 命令行参数对象（含 `file`、`embed`、`write_file` 等）。
    """
    config.run = Dict({
        "file_name": args.file,  # 用户场景脚本路径（如 "example.py"）
        "embed_line": int(args.embed) if args.embed is not None else None,  # 嵌入调试的行号
        "is_reload": False,  # 是否是重新加载场景（默认 False，由重载逻辑后续修改）
        "prerun": args.prerun,  # 是否预运行计算总帧数（用于准确进度条）
        "scene_names": args.scene_names,  # 目标场景名列表（如 ["SquareScene"]）
        "quiet": args.quiet or args.write_all,  # 静默模式（减少日志）
        "show_in_window": not args.write_file  # 是否显示预览窗口（导出文件时不显示）
    })


def update_embed_config(config: Dict, args: Namespace):
    """
    初始化“嵌入调试配置”：仅处理 `--autoreload` 参数，开启模块自动重载（开发时无需重启框架），
    提升调试效率（如修改场景代码后，框架自动加载新代码，无需重新执行命令）。
    
    参数：
        config - 全局配置字典（需包含 `embed` 字段）；
        args - 命令行参数对象（含 `autoreload` 布尔值）。
    """
    if args.autoreload:
        config.embed.autoreload = True  # 开启自动重载模块


# ------------------------------ 2. 配置辅助工具函数 ------------------------------
def load_yaml(file_path: str):
    """
    加载 YAML 配置文件：安全读取 YAML 文件内容，若文件不存在则返回空字典（避免配置加载崩溃），
    是加载默认配置（default_config.yml）和自定义配置（custom_config.yml）的基础工具。
    
    参数：file_path - YAML 文件路径（如 "./custom_config.yml"）
    返回：dict - YAML 文件内容（解析为字典），文件不存在则返回空字典。
    """
    try:
        with open(file_path, "r") as file:
            # yaml.safe_load：安全解析 YAML（避免执行恶意代码），空文件返回 None，需转为空字典
            return yaml.safe_load(file) or {}
    except FileNotFoundError:
        # 文件不存在（如用户未创建 custom_config.yml），返回空字典（不影响后续配置合并）
        return {}


def get_manim_dir():
    """
    获取 ManimGL 库的根目录：通过导入 `manimlib` 模块，定位其安装路径，
    用于查找内置默认配置文件（default_config.yml）。
    
    返回：str - ManimGL 根目录的绝对路径（如 "/usr/local/lib/python3.10/site-packages/manimgl"）。
    """
    # 1. 导入 manimlib 模块（确保已安装）
    manimlib_module = importlib.import_module("manimlib")
    # 2. 获取模块文件的绝对路径（如 ".../manimlib/__init__.py"）
    manimlib_module_path = inspect.getabsfile(manimlib_module)
    # 3. 取目录部分，再向上一层（得到 manimlib 根目录，而非 __init__.py 所在的子目录）
    manimlib_dir = os.path.dirname(manimlib_module_path)
    return os.path.abspath(os.path.join(manimlib_dir, ".."))


def get_resolution_from_args(args: Optional[Namespace], resolution_options: dict) -> Optional[tuple[int, int]]:
    """
    从命令行参数中解析渲染分辨率：优先处理 `--resolution` 自定义分辨率，再处理质量 flags（-l/-m/--hd/--uhd），
    若均未指定则返回 None（后续使用配置文件默认值）。
    
    参数：
        args - 命令行参数对象（含 `resolution`、`low_quality` 等）；
        resolution_options - 配置文件中的分辨率选项（如 {"low": "(854,480)", "high": "(1920,1080)"}）。
    返回：Optional[tuple[int, int]] - 分辨率（如 (1920,1080)），未指定则返回 None。
    """
    # 1. 优先处理自定义分辨率（--resolution "WxH"，如 "--resolution 1280x960"）
    if args.resolution:
        return tuple(map(int, args.resolution.split("x")))
    # 2. 处理质量 flags（按优先级：低质量 → 中质量 → 高清 → 超高清）
    if args.low_quality:
        return literal_eval(resolution_options["low"])  # 低质量（如 480p → (854,480)）
    if args.medium_quality:
        return literal_eval(resolution_options["med"])  # 中质量（如 720p → (1280,720)）
    if args.hd:
        return literal_eval(resolution_options["high"])  # 高清（如 1080p → (1920,1080)）
    if args.uhd:
        return literal_eval(resolution_options["4k"])  # 超高清（如 4K → (3840,2160)）
    # 3. 未指定任何分辨率参数，返回 None
    return None


def get_file_ext(args: Namespace) -> str:
    """
    根据命令行参数确定导出文件的后缀：透明导出用 .mov，GIF 导出用 .gif，默认用 .mp4，
    确保文件格式与导出类型匹配（如透明视频需用支持 alpha 通道的 .mov 格式）。
    
    参数：args - 命令行参数对象（含 `transparent`、`gif` 布尔值）
    返回：str - 文件后缀（如 ".mp4"、".gif"、".mov"）。
    """
    if args.transparent:
        return ".mov"  # 透明视频默认用 .mov（支持 alpha 通道）
    elif args.gif:
        return ".gif"  # GIF 导出用 .gif
    else:
        return ".mp4"  # 默认导出 MP4 视频


def get_animations_numbers(args: Namespace) -> tuple[int | None, int | None]:
    """
    解析 `--start_at_animation_number` 参数，得到动画的起始和结束编号：
    - 未指定 → (None, None)（渲染所有动画）；
    - 单个数字（如 "3"）→ (3, None)（从第3个动画开始，渲染到最后）；
    - 两个数字（如 "3,6"）→ (3,6)（渲染第3到第6个动画）。
    
    参数：args - 命令行参数对象（含 `start_at_animation_number` 字符串）
    返回：tuple[int | None, int | None] - 起始编号和结束编号。
    """
    anim_num_str = args.start_at_animation_number
    if anim_num_str is None:
        return (None, None)
    # 拆分逗号分隔的字符串（如 "3,6" → ["3","6"]）
    anim_num_list = anim_num_str.split(",")
    if len(anim_num_list) == 2:
        # 两个数字：起始和结束编号
        return (int(anim_num_list[0]), int(anim_num_list[1]))
    else:
        # 单个数字：仅起始编号，结束编号为 None
        return (int(anim_num_list[0]), None)


def get_output_directory(args: Namespace, config: Dict) -> str:
    """
    确定动画导出的目标目录：优先使用命令行 `--video_dir`，其次使用配置文件默认路径，
    若开启“路径镜像”（`mirror_module_path`），则自动匹配场景脚本的目录结构（便于组织多脚本导出文件）。
    
    参数：
        args - 命令行参数对象（含 `video_dir`、`file` 字段）；
        config - 全局配置字典（含 `directories` 字段，如 `base`、`output`、`mirror_module_path`）。
    返回：str - 导出目录的绝对路径（如 "./media/videos/example/"）。
    """
    dir_config = config.directories
    # 1. 优先使用命令行指定的导出目录（--video_dir）
    output_dir = args.video_dir or dir_config.output
    
    # 2. 若开启路径镜像（mirror_module_path=True），且指定了脚本文件，按脚本路径组织导出目录
    if dir_config.mirror_module_path and args.file:
        # 获取脚本文件的绝对路径（如 "/home/user/projects/example.py"）
        script_abs_path = Path(args.file).absolute()
        # 移除路径前缀（如配置中的 "removed_mirror_prefix"，避免冗余目录）
        if str(script_abs_path).startswith(dir_config.removed_mirror_prefix):
            relative_path = script_abs_path.relative_to(dir_config.removed_mirror_prefix)
            relative_path = Path(str(relative_path).lstrip("_"))  # 去除前缀下划线（可选配置）
        else:
            # 未匹配前缀，取脚本文件名（不含后缀）作为相对路径
            relative_path = script_abs_path.stem
        # 拼接导出目录（如 "./media/videos/" + "example" → "./media/videos/example/"）
        output_dir = Path(output_dir, relative_path).with_suffix("")
    
    return str(output_dir)


# ------------------------------ 3. 生成全局配置 ------------------------------
# 调用 initialize_manim_config() 整合所有配置，生成全局生效的 manim_config，
# 后续所有模块（Scene、Camera、Window 等）均从该全局配置读取参数
manim_config: Dict = initialize_manim_config()