# 从 __future__ 模块导入 annotations，支持在类型注解中使用尚未定义的类型（延迟解析）
from __future__ import annotations

# 导入标准库模块
import argparse  # 用于解析命令行参数，处理用户从终端输入的指令
import colour  # 用于颜色处理相关操作，可能涉及颜色空间转换、颜色值解析等
import importlib  # 提供动态导入模块的功能，可在运行时根据名称导入模块
import inspect  # 用于检查活对象（如模块、类、函数）的源代码信息，如参数、文档等
import os  # 提供与操作系统交互的功能，如文件路径操作、环境变量访问等
import sys  # 提供对Python解释器相关变量和函数的访问，如命令行参数、标准输入输出等
import yaml  # 用于解析和生成YAML格式的文件，常用于读取配置文件
from pathlib import Path  # 提供面向对象的文件路径操作，比os.path更直观易用
from ast import literal_eval  # 用于安全地将字符串解析为Python字面量（如字典、列表等），避免eval的安全风险
from addict import Dict  # 导入addict库的Dict类，这是一个增强版字典，支持通过属性方式访问键值对

# 从manimlib内部模块导入
from manimlib.logger import log  # 导入日志工具，用于输出程序运行中的日志信息（如调试、警告、错误等）
from manimlib.utils.dict_ops import merge_dicts_recursively  # 导入递归合并字典的工具函数，用于合并多层级配置

# 类型检查相关导入，仅在静态类型检查时执行（运行时不执行）
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from argparse import Namespace  # 用于注解命令行参数解析后的结果类型
    from typing import Optional  # 用于注解可选类型（即变量可能为None或指定类型）

def initialize_manim_config() -> Dict:
    """
    返回 Manim 中各类的默认配置，包括
    Scene（场景）、Window（窗口）、Camera（相机）、SceneFileWriter（场景文件写入器），
    以及决定场景运行方式的配置（如写入文件或窗口预览）。

    配置初始值来自 manimlib 目录下的 default_config.yml，
    可通过自定义配置文件 custom_config.yml 进一步更新，
    最终会根据命令行参数再次更新。
    """
    # 解析命令行参数（如 --resolution、--fps 等）
    args = parse_cli()
    # 构建全局默认配置文件（default_config.yml）的路径
    # get_manim_dir() 获取 Manim 安装目录，拼接得到默认配置文件路径
    global_defaults_file = os.path.join(get_manim_dir(), "manimlib", "default_config.yml")
    # 合并配置：依次合并默认配置、自定义配置、命令行指定的配置文件
    # merge_dicts_recursively 用于递归合并字典（深层键值不会被覆盖）
    config = Dict(merge_dicts_recursively(
        load_yaml(global_defaults_file),  # 加载全局默认配置
        load_yaml("custom_config.yml"),  # 加载当前工作目录的自定义配置（若存在）
        load_yaml(args.config_file) if args.config_file else dict(),  # 加载命令行指定的配置文件（若有）
    ))

    # 根据命令行参数或配置文件设置日志级别
    log.setLevel(args.log_level or config["log_level"])

    # 以下函数用于根据命令行参数更新特定模块的配置
    # 更新目录相关配置（如输出文件路径、临时文件路径等）
    update_directory_config(config, args)
    # 更新窗口相关配置（如窗口大小、标题等）
    update_window_config(config, args)
    # 更新相机相关配置（如分辨率、视场角等）
    update_camera_config(config, args)
    # 更新文件写入器配置（如帧率、编码格式等）
    update_file_writer_config(config, args)
    # 更新场景相关配置（如场景名称、是否循环播放等）
    update_scene_config(config, args)
    # 更新运行模式配置（如是否预览、是否保存等）
    update_run_config(config, args)
    # 更新嵌入相关配置（如是否嵌入到 Jupyter 等）
    update_embed_config(config, args)

    # 返回最终合并后的配置字典
    return config


def parse_cli():
    try:
        # 创建命令行参数解析器（argparse.ArgumentParser）
        parser = argparse.ArgumentParser()
        
        # 添加互斥参数组（module_location），用于指定模块位置（当前仅包含file参数）
        module_location = parser.add_mutually_exclusive_group()
        # 添加"file"参数：可选，指定包含场景代码的Python文件路径
        module_location.add_argument(
            "file",
            nargs="?",  # 0或1个参数（可选）
            help="Path to file holding the python code for the scene",  # 参数说明
        )
        
        # 添加"scene_names"参数：可变数量，指定要运行的场景类名称
        parser.add_argument(
            "scene_names",
            nargs="*",  # 0或多个参数
            help="Name of the Scene class you want to see",  # 参数说明
        )
        
        # 添加"-w/--write_file"参数：开关型，指定是否将场景渲染为视频文件
        parser.add_argument(
            "-w", "--write_file",
            action="store_true",  # 出现该参数则值为True
            help="Render the scene as a movie file",
        )
        
        # 添加"-s/--skip_animations"参数：开关型，仅保存最后一帧（跳过动画）
        parser.add_argument(
            "-s", "--skip_animations",
            action="store_true",
            help="Save the last frame",
        )
        
        # 添加"-l/--low_quality"参数：开关型，指定低质量渲染（480p）
        parser.add_argument(
            "-l", "--low_quality",
            action="store_true",
            help="Render at 480p",
        )
        
        # 添加"-m/--medium_quality"参数：开关型，指定中等质量渲染（720p）
        parser.add_argument(
            "-m", "--medium_quality",
            action="store_true",
            help="Render at 720p",
        )
        
        # 添加"--hd"参数：开关型，指定高清渲染（1080p）
        parser.add_argument(
            "--hd",
            action="store_true",
            help="Render at a 1080p",
        )
        
        # 添加"--uhd"参数：开关型，指定超高清渲染（4k）
        parser.add_argument(
            "--uhd",
            action="store_true",
            help="Render at a 4k",
        )
        
        # 添加"-f/--full_screen"参数：开关型，指定窗口全屏显示
        parser.add_argument(
            "-f", "--full_screen",
            action="store_true",
            help="Show window in full screen",
        )
        
        # 添加"-p/--presenter_mode"参数：开关型，演示模式（等待空格键/右箭头继续）
        parser.add_argument(
            "-p", "--presenter_mode",
            action="store_true",
            help="Scene will stay paused during wait calls until " + \
                 "space bar or right arrow is hit, like a slide show"
        )
        
        # 添加"-i/--gif"参数：开关型，指定输出为GIF格式
        parser.add_argument(
            "-i", "--gif",
            action="store_true",
            help="Save the video as gif",
        )
        
        # 添加"-t/--transparent"参数：开关型，渲染带透明通道的视频
        parser.add_argument(
            "-t", "--transparent",
            action="store_true",
            help="Render to a movie file with an alpha channel",
        )
        
        # 添加"--vcodec"参数：指定ffmpeg使用的视频编码器
        parser.add_argument(
            "--vcodec",
            help="Video codec to use with ffmpeg",
        )
        
        # 添加"--pix_fmt"参数：指定ffmpeg输出的像素格式（默认yuv420p）
        parser.add_argument(
            "--pix_fmt",
            help="Pixel format to use for the output of ffmpeg, defaults to `yuv420p`",
        )
        
        # 添加"-q/--quiet"参数：开关型，静默模式（减少输出信息）
        parser.add_argument(
            "-q", "--quiet",
            action="store_true",
            help="",  # 原注释未填写具体说明，通常为减少日志输出
        )
        
        # 添加"-a/--write_all"参数：开关型，渲染文件中所有场景
        parser.add_argument(
            "-a", "--write_all",
            action="store_true",
            help="Write all the scenes from a file",
        )
        
        # 添加"-o/--open"参数：开关型，渲染完成后自动打开文件
        parser.add_argument(
            "-o", "--open",
            action="store_true",
            help="Automatically open the saved file once its done",
        )
        
        # 添加"--finder"参数：开关型，在文件管理器中显示输出文件
        parser.add_argument(
            "--finder",
            action="store_true",
            help="Show the output file in finder",
        )
        
        # 添加"--subdivide"参数：开关型，将动画分割为单个动画文件
        parser.add_argument(
            "--subdivide",
            action="store_true",
            help="Divide the output animation into individual movie files " +
                 "for each animation",
        )
        
        # 添加"--file_name"参数：指定输出视频/图像的文件名
        parser.add_argument(
            "--file_name",
            help="Name for the movie or image file",
        )
        
        # 添加"-n/--start_at_animation_number"参数：指定动画起始索引（支持范围，如"3,6"）
        parser.add_argument(
            "-n", "--start_at_animation_number",
            help="Start rendering not from the first animation, but " + \
                 "from another, specified by its index.  If you pass " + \
                 "in two comma separated values, e.g. \"3,6\", it will end " + \
                 "the rendering at the second value",
        )
        
        # 添加"-e/--embed"参数：指定行号，在该处添加断点并进入交互式iPython会话
        parser.add_argument(
            "-e", "--embed",
            metavar="LINE_NUMBER",
            help="Adds a breakpoint at the inputted file dropping into an " + \
                 "interactive iPython session at that point of the code."
        )
        
        # 添加"-r/--resolution"参数：指定分辨率，格式为"WxH"（如"1920x1080"）
        parser.add_argument(
            "-r", "--resolution",
            help="Resolution, passed as \"WxH\", e.g. \"1920x1080\"",
        )
        
        # 添加"--fps"参数：指定帧率（整数）
        parser.add_argument(
            "--fps",
            help="Frame rate, as an integer",
            type=int,  # 强制转换为整数
        )
        
        # 添加"-c/--color"参数：指定背景颜色
        parser.add_argument(
            "-c", "--color",
            help="Background color",
        )
        
        # 添加"--leave_progress_bars"参数：开关型，保留终端中的进度条
        parser.add_argument(
            "--leave_progress_bars",
            action="store_true",
            help="Leave progress bars displayed in terminal",
        )
        
        # 添加"--show_animation_progress"参数：开关型，显示每个动画的进度条
        parser.add_argument(
            "--show_animation_progress",
            action="store_true",
            help="Show progress bar for each animation",
        )
        
        # 添加"--prerun"参数：开关型，预运行场景计算总帧数（用于进度条）
        parser.add_argument(
            "--prerun",
            action="store_true",
            help="Calculate total framecount, to display in a progress bar, by doing " + \
                 "an initial run of the scene which skips animations."
        )
        
        # 添加"--video_dir"参数：指定视频输出目录
        parser.add_argument(
            "--video_dir",
            help="Directory to write video",
        )
        
        # 添加"--config_file"参数：指定自定义配置文件路径
        parser.add_argument(
            "--config_file",
            help="Path to the custom configuration file",
        )
        
        # 添加"-v/--version"参数：开关型，显示manimgl版本
        parser.add_argument(
            "-v", "--version",
            action="store_true",
            help="Display the version of manimgl"
        )
        
        # 添加"--log-level"参数：指定日志级别（DEBUG/INFO/WARNING/ERROR/CRITICAL）
        parser.add_argument(
            "--log-level",
            help="Level of messages to Display, can be DEBUG / INFO / WARNING / ERROR / CRITICAL"
        )
        
        # 添加"--clear-cache"参数：开关型，清除Tex和Text Mobjects的缓存
        parser.add_argument(
            "--clear-cache",
            action="store_true",
            help="Erase the cache used for Tex and Text Mobjects"
        )
        
        # 添加"--autoreload"参数：开关型，自动重新加载Python模块（检测代码变化）
        parser.add_argument(
            "--autoreload",
            action="store_true",
            help="Automatically reload Python modules to pick up code changes " +
                 "across different files",
        )
        
        # 解析命令行参数，生成args对象（属性对应各个参数）
        args = parser.parse_args()
        
        # 补充逻辑：若指定--open或--finder，自动开启--write_file（需先渲染才能操作文件）
        args.write_file = any([args.write_file, args.open, args.finder])
        
        # 返回解析后的参数对象
        return args
        
    # 捕获参数解析错误（如无效参数组合）
    except argparse.ArgumentError as err:
        # 记录错误日志
        log.error(str(err))
        # 以状态码2退出程序（表示参数错误）
        sys.exit(2)


# 更新目录配置：将基础目录与子目录拼接，生成完整路径
def update_directory_config(config: Dict):
    # 从总配置中获取目录相关子配置
    dir_config = config.directories
    # 提取基础目录路径（如 "media"）
    base = dir_config.base
    # 遍历子目录配置（如 subdirs: {"videos": "videos", "images": "images"}）
    for key, subdir in dir_config.subdirs.items():
        # 拼接基础目录与子目录，生成完整路径（如 "media/videos"），存入目录配置
        dir_config[key] = os.path.join(base, subdir)


# 更新窗口配置：处理窗口位置/尺寸的格式转换，响应全屏参数
def update_window_config(config: Dict, args: Namespace):
    # 从总配置中获取窗口相关子配置
    window_config = config.window
    # 处理窗口位置（position）和尺寸（size）：将字符串格式（如 "(100,200)"）转为元组
    for key in "position", "size":
        # 若配置中存在该键且值非空
        if window_config.get(key):
            # 使用literal_eval安全解析字符串为Python对象（如元组）
            window_config[key] = literal_eval(window_config[key])
    # 若命令行指定了全屏（--full_screen），强制开启窗口全屏模式
    if args.full_screen:
        window_config.full_screen = True


# 更新相机配置：处理分辨率、帧率、背景色、透明度等相机核心参数
def update_camera_config(config: Dict, args: Namespace):
    # 从总配置中获取相机相关子配置
    camera_config = config.camera
    # 从命令行参数或配置的分辨率选项中，提取最终分辨率（如 (1920,1080)）
    arg_resolution = get_resolution_from_args(args, config.resolution_options)
    # 确定相机最终分辨率：命令行指定的优先，否则解析配置中的默认值
    camera_config.resolution = arg_resolution or literal_eval(camera_config.resolution)
    # 若命令行指定了帧率（--fps），覆盖相机配置的帧率
    if args.fps:
        camera_config.fps = args.fps
    # 若命令行指定了背景色（--color），尝试解析为颜色对象
    if args.color:
        try:
            # 使用colour库将输入的颜色字符串（如 "red"、"#FF0000"）转为Color对象
            camera_config.background_color = colour.Color(args.color)
        except Exception as err:
            # 若颜色无效，记录错误并退出程序（状态码2）
            log.error("Please use a valid color")
            log.error(err)
            sys.exit(2)
    # 若命令行指定了透明背景（--transparent），将背景透明度设为0.0（完全透明）
    if args.transparent:
        camera_config.background_opacity = 0.0


# 更新文件写入器配置：处理视频输出模式、格式、编码等参数
def update_file_writer_config(config: Dict, args: Namespace):
    # 从总配置中获取文件写入器相关子配置
    file_writer_config = config.file_writer
    # 批量更新配置：根据命令行参数设置核心输出参数
    file_writer_config.update(
        # 是否写入视频：不跳过动画（--skip_animations未指定）且需写入文件（--write_file指定）
        write_to_movie=(not args.skip_animations and args.write_file),
        # 是否分割输出：按每个动画生成单独文件（--subdivide指定）
        subdivide_output=args.subdivide,
        # 是否保存最后一帧：跳过动画（--skip_animations指定）且需写入文件（--write_file指定）
        save_last_frame=(args.skip_animations and args.write_file),
        # PNG模式：透明背景用RGBA，否则用RGB（由--transparent决定）
        png_mode=("RGBA" if args.transparent else "RGB"),
        # 视频文件后缀：根据命令行参数（如--gif）获取对应格式（如 ".gif"）
        movie_file_extension=(get_file_ext(args)),
        # 输出目录：从命令行或配置中获取最终输出路径
        output_directory=get_output_directory(args, config),
        # 输出文件名：命令行指定的--file_name，无则用默认
        file_name=args.file_name,
        # 渲染完成后自动打开文件：--open指定
        open_file_upon_completion=args.open,
        # 渲染完成后在文件管理器显示：--finder指定
        show_file_location_upon_completion=args.finder,
        # 静默模式：--quiet指定（减少输出日志）
        quiet=args.quiet,
    )

    # 处理视频编码器（vcodec）：命令行指定优先
    if args.vcodec:
        file_writer_config.video_codec = args.vcodec
    # 若为透明背景（--transparent），默认使用prores_ks编码器，清空像素格式
    elif args.transparent:
        file_writer_config.video_codec = 'prores_ks'
        file_writer_config.pixel_format = ''
    # 若输出为GIF（--gif），清空编码器（GIF无需视频编码器）
    elif args.gif:
        file_writer_config.video_codec = ''

    # 若命令行指定了像素格式（--pix_fmt），覆盖配置
    if args.pix_fmt:
        file_writer_config.pixel_format = args.pix_fmt


# 更新场景配置：处理动画起始/结束索引、演示模式等场景运行参数
def update_scene_config(config: Dict, args: Namespace):
    # 从总配置中获取场景相关子配置
    scene_config = config.scene
    # 从命令行参数（-n/--start_at_animation_number）中解析动画起始和结束索引
    start, end = get_animations_numbers(args)
    # 批量更新场景配置
    scene_config.update(
        # 相机子配置：预留字段，供后续扩展（当前使用全局相机配置）
        camera_config=dict(),
        # 文件写入器子配置：预留字段，供后续扩展（当前使用全局文件写入器配置）
        file_writer_config=dict(),
        # 是否跳过动画：--skip_animations指定（仅显示最后一帧）
        skip_animations=args.skip_animations,
        # 动画起始索引：从-n参数解析，无则为None（从第一帧开始）
        start_at_animation_number=start,
        # 动画结束索引：从-n参数解析（如 "-n 3,6" 中的6），无则为None（到最后一帧结束）
        end_at_animation_number=end,
        # 演示模式：--presenter_mode指定（按空格/右箭头继续）
        presenter_mode=args.presenter_mode,
    )
    # 若指定--leave_progress_bars，保留终端中的进度条
    if args.leave_progress_bars:
        scene_config.leave_progress_bars = True
    # 若指定--show_animation_progress，显示每个动画的进度条
    if args.show_animation_progress:
        scene_config.show_animation_progress = True


# 更新运行配置：整合文件路径、嵌入调试、预运行等全局运行参数
def update_run_config(config: Dict, args: Namespace):
    # 构建并赋值运行配置字典
    config.run = Dict(
        # 场景文件路径：命令行指定的Python文件（args.file）
        file_name=args.file,
        # 嵌入调试的行号：--embed指定的行号（转为整数），无则为None
        embed_line=(int(args.embed) if args.embed is not None else None),
        # 是否重新加载：默认False（预留字段，供自动重载功能使用）
        is_reload=False,
        # 是否预运行：--prerun指定（预计算总帧数，用于进度条）
        prerun=args.prerun,
        # 要运行的场景名称：命令行指定的scene_names，无则运行文件中所有场景
        scene_names=args.scene_names,
        # 静默模式：--quiet指定 或 --write_all指定（批量渲染时减少输出）
        quiet=args.quiet or args.write_all,
        # 是否运行所有场景：--write_all指定（渲染文件中所有Scene类）
        write_all=args.write_all,
        # 是否在窗口预览：未指定--write_file时为True（仅预览不渲染文件）
        show_in_window=not args.write_file
    )


def update_embed_config(config: Dict, args: Namespace):
    # 若命令行指定了--autoreload（自动重载模块），开启嵌入配置中的autoreload
    if args.autoreload:
        config.embed.autoreload = True


# Helpers for the functions above


# 加载YAML配置文件，返回解析后的字典（文件不存在则返回空字典）
def load_yaml(file_path: str):
    try:
        # 尝试打开并读取YAML文件
        with open(file_path, "r") as file:
            # 使用yaml.safe_load安全解析YAML内容，避免执行恶意代码；若文件为空返回空字典
            return yaml.safe_load(file) or {}
    # 若文件不存在，返回空字典（不报错，使用默认配置）
    except FileNotFoundError:
        return {}


# 获取Manim库的根目录路径
def get_manim_dir():
    # 动态导入manimlib模块
    manimlib_module = importlib.import_module("manimlib")
    # 获取manimlib模块的绝对路径（如.../manimlib/__init__.py）
    manimlib_dir = os.path.dirname(inspect.getabsfile(manimlib_module))
    # 返回manimlib的父目录（即Manim库的根目录）
    return os.path.abspath(os.path.join(manimlib_dir, ".."))


# 从命令行参数中解析分辨率（优先使用显式参数，再用质量简写参数）
def get_resolution_from_args(args: Optional[Namespace], resolution_options: dict) -> Optional[tuple[int, int]]:
    # 若显式指定了--resolution（如"1920x1080"），按"x"分割并转为整数元组
    if args.resolution:
        return tuple(map(int, args.resolution.split("x")))
    # 若指定了低质量（-l/--low_quality），返回配置中"low"对应的分辨率
    if args.low_quality:
        return literal_eval(resolution_options["low"])
    # 若指定了中等质量（-m/--medium_quality），返回配置中"med"对应的分辨率
    if args.medium_quality:
        return literal_eval(resolution_options["med"])
    # 若指定了高清（--hd），返回配置中"high"对应的分辨率（通常1080p）
    if args.hd:
        return literal_eval(resolution_options["high"])
    # 若指定了超高清（--uhd），返回配置中"4k"对应的分辨率（通常4096x2160）
    if args.uhd:
        return literal_eval(resolution_options["4k"])
    # 未指定任何分辨率相关参数，返回None（使用默认配置）
    return None


# 根据命令行参数确定输出文件的后缀（.mp4/.mov/.gif）
def get_file_ext(args: Namespace) -> str:
    # 若指定透明背景（--transparent），默认用.mov格式（支持alpha通道）
    if args.transparent:
        file_ext = ".mov"
    # 若指定输出GIF（-i/--gif），用.gif格式
    elif args.gif:
        file_ext = ".gif"
    # 默认输出.mp4格式
    else:
        file_ext = ".mp4"
    return file_ext


# 从命令行参数中解析动画的起始和结束索引（-n/--start_at_animation_number）
def get_animations_numbers(args: Namespace) -> tuple[int | None, int | None]:
    # 获取命令行参数中的动画索引字符串（如"3"或"3,6"）
    stan = args.start_at_animation_number
    # 若未指定，返回(None, None)（从第一帧到最后一帧）
    if stan is None:
        return (None, None)
    # 若包含逗号（如"3,6"），分割为起始和结束索引（整数元组）
    elif "," in stan:
        return tuple(map(int, stan.split(",")))
    # 若仅单个数字（如"3"），返回(起始索引, None)（从该索引到最后一帧）
    else:
        return int(stan), None


# 确定最终的输出目录路径（支持路径镜像功能）
def get_output_directory(args: Namespace, config: Dict) -> str:
    # 获取目录配置
    dir_config = config.directories
    # 输出目录优先级：命令行--video_dir > 配置中的默认输出目录
    out_dir = args.video_dir or dir_config.output

    # 若开启路径镜像（mirror_module_path）且指定了场景文件
    if dir_config.mirror_module_path and args.file:
        # 获取场景文件的绝对路径
        file_path = Path(args.file).absolute()
        # 若文件路径以"removed_mirror_prefix"为前缀（通常是项目根目录）
        if str(file_path).startswith(dir_config.removed_mirror_prefix):
            # 计算文件相对于前缀的相对路径（用于镜像目录结构）
            rel_path = file_path.relative_to(dir_config.removed_mirror_prefix)
            # 去除路径中的前导下划线（美化目录名）
            rel_path = Path(str(rel_path).lstrip("_"))
        # 若不满足前缀条件，用文件名作为相对路径
        else:
            rel_path = file_path.stem
        # 拼接输出目录与相对路径，作为最终输出目录（镜像原文件的目录结构）
        out_dir = Path(out_dir, rel_path).with_suffix("")

    # 返回输出目录的字符串路径
    return out_dir


# 创建全局配置对象：调用initialize_manim_config()整合所有配置，供全库使用
# Create global configuration
manim_config: Dict = initialize_manim_config()
