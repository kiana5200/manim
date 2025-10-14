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


# ManimGL 核心配置初始化与命令行参数解析模块，负责**整合默认配置、自定义配置和命令行参数**，
# 生成最终的全局运行配置（`manim_config`），是控制动画渲染、窗口显示、文件导出等行为的“总开关”。
# 注：代码中存在重复定义，以下按“功能完整性”合并解析，去除重复逻辑。


def initialize_manim_config() -> Dict:
    """
    初始化 ManimGL 全局配置：按优先级整合配置（命令行参数 > 自定义配置 > 默认配置），
    为场景、窗口、相机等核心组件提供统一参数，确保各模块行为一致。
    
    核心逻辑：
    1. **配置优先级规则**：用户命令行输入的参数拥有最高优先级，其次是本地自定义配置文件，
       最后是 ManimGL 内置的默认配置（避免用户重复配置基础参数）。
    2. **配置加载流程**：先加载默认配置，再用自定义配置覆盖，最后用命令行参数修正，
       确保最终配置符合用户预期。
    3. **模块配置细分**：将合并后的配置分配到对应模块（如窗口、相机、文件写入器），
       避免配置混乱，便于后续组件调用。
    
    返回值：Dict（全局配置字典，包含各模块的详细参数，如窗口大小、渲染分辨率、导出路径等）。
    """
    # 1. 先解析命令行参数（后续用于覆盖配置文件参数）
    args = parse_cli()
    
    # 2. 确定全局默认配置文件路径（ManimGL 安装目录下的 default_config.yml）
    # get_manim_dir()：获取 ManimGL 库的根目录（如 Python 环境的 site-packages/manimgl）
    global_defaults_file = os.path.join(get_manim_dir(), "manimlib", "default_config.yml")
    
    # 3. 按优先级合并配置（后加载的配置覆盖先加载的）
    config = Dict(merge_dicts_recursively(
        load_yaml(global_defaults_file),          # 1. 最低优先级：内置默认配置（如默认分辨率1080p、帧率30）
        load_yaml("custom_config.yml"),           # 2. 中优先级：本地自定义配置（用户当前工作目录下，可选）
        load_yaml(args.config_file) if args.config_file else dict(),  # 3. 高优先级：命令行指定的配置文件（--config_file 参数）
    ))

    # 4. 配置日志级别（命令行参数 --log-level 优先于配置文件中的 log_level）
    # 日志级别：DEBUG（最详细）→ INFO → WARNING → ERROR → CRITICAL（仅致命错误）
    log.setLevel(args.log_level or config["log_level"])

    # 5. 细分配置到各功能模块（将合并后的配置分配给对应组件，避免参数混乱）
    update_directory_config(config)       # 目录配置（如视频导出路径 ./media/videos/、缓存路径）
    update_window_config(config, args)    # 窗口配置（如大小、是否全屏、标题）
    update_camera_config(config, args)    # 相机配置（如渲染分辨率、视角、3D深度测试）
    update_file_writer_config(config, args)  # 文件写入器配置（如导出格式、FFmpeg 编码、GIF 优化）
    update_scene_config(config, args)     # 场景配置（如背景色、演示模式开关、默认等待时间）
    update_run_config(config, args)       # 运行配置（如是否显示窗口、是否自动打开导出文件）
    update_embed_config(config, args)     # 嵌入调试配置（如 IPython 断点行号、自动重载开关）

    return config


def parse_cli():
    """
    解析 ManimGL 命令行参数：使用 `argparse` 定义并解析用户输入的命令，
    将命令行指令转换为结构化的参数对象，是用户控制 ManimGL 运行的主要入口。
    
    核心作用：
    - 覆盖配置文件参数（如命令行 `-l` 强制启用低质量渲染，忽略配置文件中的分辨率设置）；
    - 提供快捷操作（如 `-w` 一键导出视频、`-s` 快速保存最后一帧）；
    - 支持调试功能（如 `-e` 嵌入 IPython 断点、`--autoreload` 自动重载代码）。
    
    返回值：argparse.Namespace（解析后的参数对象，属性对应各命令行参数值）。
    """
    try:
        # 1. 创建参数解析器（程序名称默认是脚本名，描述信息省略，可通过 add_help 添加）
        parser = argparse.ArgumentParser()

        # 2. 核心必选参数（场景脚本路径与场景名）
        # 互斥组（预留扩展，当前未实际限制互斥，仅用于逻辑分组）
        module_location = parser.add_mutually_exclusive_group()
        module_location.add_argument(
            "file",
            nargs="?",  # 可选参数：若不指定，可能仅执行版本查询、缓存清理等操作
            help="Path to file holding the python code for the scene",  # 示例：example.py
        )
        parser.add_argument(
            "scene_names",
            nargs="*",  # 0个或多个场景名：若不指定，默认运行脚本中第一个 Scene 子类；指定多个则依次运行
            help="Name of the Scene class you want to see",  # 示例：SquareScene CircleScene
        )

        # 3. 渲染与导出控制参数
        parser.add_argument(
            "-w", "--write_file",
            action="store_true",
            help="Render the scene as a movie file (MP4 by default)",  # 导出视频文件
        )
        parser.add_argument(
            "-s", "--skip_animations",
            action="store_true",
            help="Skip animations and save only the last frame (as PNG)",  # 仅保存最后一帧（快速预览最终效果）
        )
        parser.add_argument(
            "-i", "--gif",
            action="store_true",
            help="Save the video as a GIF file (auto-enables --write_file)",  # 导出为 GIF（需配合 -w）
        )
        parser.add_argument(
            "-t", "--transparent",
            action="store_true",
            help="Render video with an alpha channel (transparent background)",  # 导出带透明通道的视频
        )

        # 4. 质量控制参数（分辨率、帧率）
        parser.add_argument(
            "-l", "--low_quality",
            action="store_true",
            help="Render at 480p (854x480, overrides other resolution settings)",  # 低质量（快速渲染，调试用）
        )
        parser.add_argument(
            "-m", "--medium_quality",
            action="store_true",
            help="Render at 720p (1280x720, overrides other resolution settings)",  # 中质量
        )
        parser.add_argument(
            "--hd",
            action="store_true",
            help="Render at 1080p (1920x1080, overrides other resolution settings)",  # 高清（默认）
        )
        parser.add_argument(
            "--uhd",
            action="store_true",
            help="Render at 4K (3840x2160, overrides other resolution settings)",  # 超高清（耗时长，最终输出用）
        )
        parser.add_argument(
            "-r", "--resolution",
            help="Custom resolution (format: \"WxH\", e.g. \"1280x960\", overrides quality flags)",  # 自定义分辨率
        )
        parser.add_argument(
            "--fps",
            help="Frame rate (integer, e.g. 60 for smooth animation)",
            type=int,  # 帧率：默认30，60需更高性能，15可减少文件大小
        )

        # 5. 窗口控制参数
        parser.add_argument(
            "-f", "--full_screen",
            action="store_true",
            help="Show the preview window in full screen",  # 全屏显示预览窗口
        )

        # 6. 演示与交互参数
        parser.add_argument(
            "-p", "--presenter_mode",
            action="store_true",
            help="Pause at each wait() call until Space/Right Arrow is pressed (slide show mode)",  # 演示模式（逐页播放）
        )
        parser.add_argument(
            "-e", "--embed",
            metavar="LINE_NUMBER",
            help="Add an IPython breakpoint at the specified line in the scene file (debug mode)",  # 嵌入调试（指定行号）
        )
        parser.add_argument(
            "--autoreload",
            action="store_true",
            help="Auto-reload Python modules when code changes (no need to restart Manim)",  # 自动重载代码（开发效率）
        )

        # 7. 输出文件控制参数
        parser.add_argument(
            "--file_name",
            help="Custom name for the output file (default: scene class name)",  # 自定义输出文件名
        )
        parser.add_argument(
            "--video_dir",
            help="Custom directory to save videos (default: ./media/videos/)",  # 自定义视频导出目录
        )
        parser.add_argument(
            "-o", "--open",
            action="store_true",
            help="Automatically open the output file after rendering (auto-enables --write_file)",  # 导出后自动打开文件
        )
        parser.add_argument(
            "--finder",
            action="store_true",
            help="Show the output file in the file manager (auto-enables --write_file)",  # 在文件管理器中显示文件
        )
        parser.add_argument(
            "-a", "--write_all",
            action="store_true",
            help="Render and save all Scene classes in the input file",  # 导出脚本中所有场景
        )
        parser.add_argument(
            "--subdivide",
            action="store_true",
            help="Split the output into separate files for each animation step",  # 按动画步骤拆分输出文件
        )
        parser.add_argument(
            "--vcodec",
            help="Video codec for FFmpeg (e.g. libx264 for MP4, libvpx for WebM)",  # FFmpeg 视频编码（高级用户）
        )
        parser.add_argument(
            "--pix_fmt",
            help="Pixel format for FFmpeg (default: yuv420p for MP4, rgba for transparent videos)",  # 像素格式
        )

        # 8. 动画进度与日志参数
        parser.add_argument(
            "-q", "--quiet",
            action="store_true",
            help="Reduce log output (only show errors and warnings)",  # 静默模式（减少日志）
        )
        parser.add_argument(
            "--log-level",
            help="Log verbosity: DEBUG / INFO / WARNING / ERROR / CRITICAL",  # 日志级别（调试用）
        )
        parser.add_argument(
            "--show_animation_progress",
            action="store_true",
            help="Show a progress bar for each individual animation",  # 显示每个动画的进度条
        )
        parser.add_argument(
            "--leave_progress_bars",
            action="store_true",
            help="Keep progress bars in the terminal after rendering (default: auto-hide)",  # 保留进度条
        )
        parser.add_argument(
            "--prerun",
            action="store_true",
            help="Pre-run to calculate total frames (for accurate progress bars, adds small overhead)",  # 预计算总帧数
        )

        # 9. 动画范围控制参数
        parser.add_argument(
            "-n", "--start_at_animation_number",
            help="Start rendering from the Nth animation (e.g. 3 for 3rd); use \"N,M\" to end at M",  # 指定动画范围
        )

        # 10. 辅助工具参数
        parser.add_argument(
            "-v", "--version",
            action="store_true",
            help="Display the current version of ManimGL",  # 显示版本号
        )
        parser.add_argument(
            "--clear-cache",
            action="store_true",
            help="Delete cached files for Tex/Text Mobjects (fixes rendering issues for updated formulas)",  # 清理缓存
        )
        parser.add_argument(
            "--config_file",
            help="Path to a custom YAML configuration file (overrides default_config.yml)",  # 自定义配置文件路径
        )

        # 11. 关键逻辑修正：只要指定 -o/--finder，自动开启 --write_file（避免用户漏加导出开关）
        args = parser.parse_args()
        args.write_file = any([args.write_file, args.open, args.finder])

        return args

    # 12. 捕获参数解析错误（如无效参数、格式错误，如 --resolution 输入 "1920x1080" 而非 1920x1080）
    except argparse.ArgumentError as err:
        log.error(str(err))  # 打印错误信息（如 "argument --resolution: invalid WxH format: '1920'"）
        sys.exit(2)  # 退出程序（错误码2：表示命令行参数错误，符合 Unix 程序惯例）

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
    scene_config = config.scene
    start, end = get_animations_numbers(args)
    scene_config.update(
        # Note, Scene.__init__ makes use of both manimlib.camera and
        # manimlib.file_writer below, so the arguments here are just for
        # any future specifications beyond what the global configuration holds
        camera_config=dict(),
        file_writer_config=dict(),
        skip_animations=args.skip_animations,
        start_at_animation_number=start,
        end_at_animation_number=end,
        presenter_mode=args.presenter_mode,
    )
    if args.leave_progress_bars:
        scene_config.leave_progress_bars = True
    if args.show_animation_progress:
        scene_config.show_animation_progress = True


def update_run_config(config: Dict, args: Namespace):
    config.run = Dict(
        file_name=args.file,
        embed_line=(int(args.embed) if args.embed is not None else None),
        is_reload=False,
        prerun=args.prerun,
        scene_names=args.scene_names,
        quiet=args.quiet or args.write_all,
        write_all=args.write_all,
        show_in_window=not args.write_file
    )


def update_embed_config(config: Dict, args: Namespace):
    if args.autoreload:
        config.embed.autoreload = True


# Helpers for the functions above


def load_yaml(file_path: str):
    try:
        with open(file_path, "r") as file:
            return yaml.safe_load(file) or {}
    except FileNotFoundError:
        return {}


def get_manim_dir():
    manimlib_module = importlib.import_module("manimlib")
    manimlib_dir = os.path.dirname(inspect.getabsfile(manimlib_module))
    return os.path.abspath(os.path.join(manimlib_dir, ".."))


def get_resolution_from_args(args: Optional[Namespace], resolution_options: dict) -> Optional[tuple[int, int]]:
    if args.resolution:
        return tuple(map(int, args.resolution.split("x")))
    if args.low_quality:
        return literal_eval(resolution_options["low"])
    if args.medium_quality:
        return literal_eval(resolution_options["med"])
    if args.hd:
        return literal_eval(resolution_options["high"])
    if args.uhd:
        return literal_eval(resolution_options["4k"])
    return None


def get_file_ext(args: Namespace) -> str:
    if args.transparent:
        file_ext = ".mov"
    elif args.gif:
        file_ext = ".gif"
    else:
        file_ext = ".mp4"
    return file_ext


def get_animations_numbers(args: Namespace) -> tuple[int | None, int | None]:
    stan = args.start_at_animation_number
    if stan is None:
        return (None, None)
    elif "," in stan:
        return tuple(map(int, stan.split(",")))
    else:
        return int(stan), None


def get_output_directory(args: Namespace, config: Dict) -> str:
    dir_config = config.directories
    out_dir = args.video_dir or dir_config.output
    if dir_config.mirror_module_path and args.file:
        file_path = Path(args.file).absolute()
        if str(file_path).startswith(dir_config.removed_mirror_prefix):
            rel_path = file_path.relative_to(dir_config.removed_mirror_prefix)
            rel_path = Path(str(rel_path).lstrip("_"))
        else:
            rel_path = file_path.stem
        out_dir = Path(out_dir, rel_path).with_suffix("")
    return out_dir


# Create global configuration
manim_config: Dict = initialize_manim_config()
