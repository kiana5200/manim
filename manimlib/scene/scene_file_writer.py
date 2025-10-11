# 导入Python未来版本的注解特性（支持字符串形式的类名等灵活类型提示）
from __future__ import annotations

# 导入所需标准库模块
import os  # 用于文件路径、目录操作（如创建文件夹、判断路径是否存在）
import platform  # 用于获取操作系统信息（如Windows/macOS/Linux，适配不同平台的文件打开方式）
import shutil  # 用于高级文件操作（如复制、删除目录）
import subprocess as sp  # 用于调用外部命令（核心：调用ffmpeg进行视频编码、音频处理）
import sys  # 用于访问系统参数（如命令行参数、标准输出）

# 导入第三方库模块
import numpy as np  # 用于数值计算（如处理图像像素数据）
from pydub import AudioSegment  # 用于音频文件处理（如音频格式转换、拼接）
from tqdm.auto import tqdm as ProgressDisplay  # 用于显示进度条（可视化渲染/编码进度）
from pathlib import Path  # 用于更便捷的路径对象操作（跨平台路径处理）

# 导入Manim库内部模块
from manimlib.logger import log  # Manim的日志工具（用于打印信息、警告、错误）
from manimlib.mobject.mobject import Mobject  # Manim基础图形对象类（类型提示用）
from manimlib.utils.file_ops import guarantee_existence  # 用于确保目录存在（不存在则创建）
from manimlib.utils.sounds import get_full_sound_file_path  # 用于获取音频文件的完整路径

# 导入类型提示相关模块（仅在类型检查时生效，不影响运行时）
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from PIL.Image import Image  # PIL图像类（类型提示用，处理帧图像）
    from manimlib.camera.camera import Camera  # Manim相机类（类型提示用，负责渲染帧）
    from manimlib.scene.scene import Scene  # Manim场景类（类型提示用，关联待渲染场景）


class SceneFileWriter(object):
    """
    场景文件写入器类：负责将Manim场景的渲染帧（图像）和音频合并为最终视频文件，
    同时支持单独保存帧图像、管理输出目录、显示进度等功能。
    核心依赖：ffmpeg（用于视频编码）、pydub（用于音频处理）。
    """
    def __init__(
        self,
        scene: Scene,
        write_to_movie: bool = False,
        subdivide_output: bool = False,
        png_mode: str = "RGBA",
        save_last_frame: bool = False,
        movie_file_extension: str = ".mp4",
        # 输出路径相关参数
        output_directory: str = ".",
        file_name: str | None = None,
        open_file_upon_completion: bool = False,
        show_file_location_upon_completion: bool = False,
        quiet: bool = False,
        total_frames: int = 0,
        progress_description_len: int = 40,
        # 视频编码相关参数
        ffmpeg_bin: str = "ffmpeg",
        video_codec: str = "libx264",
        pixel_format: str = "yuv420p",
        saturation: float = 1.0,
        gamma: float = 1.0,
    ):
        """
        初始化场景文件写入器，配置渲染输出的核心参数。
        
        参数说明：
            scene: 待渲染的Manim场景对象（关联相机、帧数据等）
            write_to_movie: 是否生成视频文件（True则编码为视频，False仅保存帧图像）
            subdivide_output: 是否按场景类名细分输出目录（True则在输出目录下创建场景类名子目录）
            png_mode: 保存帧图像的模式（如"RGBA"含透明通道，"RGB"无透明）
            save_last_frame: 是否单独保存场景的最后一帧（True则额外生成last_frame.png）
            movie_file_extension: 输出视频的文件后缀（如".mp4"、".mov"）
            
            output_directory: 输出目录的根路径（默认当前目录）
            file_name: 输出文件的基础名称（不含后缀，默认使用场景类名）
            open_file_upon_completion: 视频生成后是否自动打开文件（适配不同操作系统）
            show_file_location_upon_completion: 视频生成后是否打印文件路径
            quiet: 是否静默模式（True则不显示进度条和部分日志）
            total_frames: 总渲染帧数（用于进度条初始化，0则自动适配）
            progress_description_len: 进度条描述文本的最大长度（避免文字溢出）
            
            ffmpeg_bin: ffmpeg可执行文件的路径（默认"ffmpeg"，需确保系统可调用）
            video_codec: 视频编码器（如"libx264"对应H.264编码，兼容性好）
            pixel_format: 视频像素格式（如"yuv420p"适配大多数播放器，支持YUV颜色空间）
            saturation: 视频色彩饱和度（1.0为默认，>1.0更鲜艳，<1.0更灰暗）
            gamma: 视频 gamma 值（调整亮度，1.0为默认，>1.0更亮，<1.0更暗）
        """
        # 绑定核心关联对象
        self.scene: Scene = scene  # 待渲染的场景
        # 输出模式配置
        self.write_to_movie = write_to_movie  # 是否生成视频
        self.subdivide_output = subdivide_output  # 是否细分输出目录
        self.png_mode = png_mode  # 帧图像保存模式
        self.save_last_frame = save_last_frame  # 是否保存最后一帧
        self.movie_file_extension = movie_file_extension  # 视频后缀
        # 输出路径与行为配置
        self.output_directory = output_directory  # 输出根目录
        self.file_name = file_name  # 输出文件名（默认用场景类名）
        self.open_file_upon_completion = open_file_upon_completion  # 完成后自动打开
        self.show_file_location_upon_completion = show_file_location_upon_completion  # 显示文件路径
        self.quiet = quiet  # 静默模式
        # 进度显示配置
        self.total_frames = total_frames  # 总帧数（进度条用）
        self.progress_description_len = progress_description_len  # 进度条描述长度
        # 视频编码配置
        self.ffmpeg_bin = ffmpeg_bin  # ffmpeg路径
        self.video_codec = video_codec  # 视频编码器
        self.pixel_format = pixel_format  # 像素格式
        self.saturation = saturation  # 色彩饱和度
        self.gamma = gamma  # 亮度gamma值

        # 运行时状态变量（初始化默认值）
        self.writing_process: sp.Popen | None = None  # ffmpeg编码子进程（None表示未启动）
        self.progress_display: ProgressDisplay | None = None  # 进度条对象（None表示未创建）
        self.ended_with_interrupt: bool = False  # 是否因中断（如Ctrl+C）结束

        # 初始化输出目录（确保目录存在，按配置细分）
        self.init_output_directories()
        # 初始化音频处理模块（准备音频数据，如拼接、格式转换）
        self.init_audio()

    # Output directories and files
def init_output_directories(self) -> None:
    """
    初始化所有输出目录与文件路径
    逻辑：根据配置的输出模式（保存单帧、生成视频、细分输出），分别初始化对应的路径
    依赖：调用`init_image_file_path`/`init_movie_file_path`/`init_partial_movie_directory`等方法
    """
    # 1. 若需要保存最后一帧，初始化单帧图像的保存路径
    if self.save_last_frame:
        self.image_file_path = self.init_image_file_path()
    # 2. 若需要生成视频，初始化最终视频文件的保存路径
    if self.write_to_movie:
        self.movie_file_path = self.init_movie_file_path()
    # 3. 若需要细分输出（按片段拆分视频），初始化分段视频的存储目录
    if self.subdivide_output:
        self.partial_movie_directory = self.init_partial_movie_directory()

def init_image_file_path(self) -> Path:
    """
    初始化最后一帧图像的保存路径
    规则：以“输出文件根名”为基础，添加.png后缀（如output/scene_001.png）
    返回：Path对象（包含完整路径的图像文件）
    """
    # 调用get_output_file_rootname()获取基础路径，再通过with_suffix添加.png后缀
    return self.get_output_file_rootname().with_suffix(".png")

def init_movie_file_path(self) -> Path:
    """
    初始化最终视频文件的保存路径
    规则：以“输出文件根名”为基础，添加配置的视频后缀（如.mp4/.mov，默认.mp4）
    返回：Path对象（包含完整路径的视频文件）
    """
    # 调用get_output_file_rootname()获取基础路径，添加配置的视频后缀
    return self.get_output_file_rootname().with_suffix(self.movie_file_extension)

def init_partial_movie_directory(self):
    """
    初始化分段视频的存储目录
    逻辑：以“输出文件根名”为目录名，确保该目录存在（不存在则创建）
    返回：Path对象（包含完整路径的分段视频目录）
    """
    # 调用get_output_file_rootname()获取目录名，通过guarantee_existence确保目录存在
    return guarantee_existence(self.get_output_file_rootname())

def get_output_file_rootname(self) -> Path:
    """
    获取输出文件的“根路径”（不含后缀的完整路径）
    构成：输出目录（确保存在） + 输出文件名（由get_output_file_name()生成）
    返回：Path对象（包含目录和文件名，无后缀）
    """
    return Path(
        guarantee_existence(self.output_directory),  # 确保输出目录存在，不存在则创建
        self.get_output_file_name()  # 获取输出文件名（不含后缀）
    )

def get_output_file_name(self) -> str:
    """
    生成输出文件的“基础名称”（不含目录和后缀）
    规则：
        1. 若用户指定了file_name，直接使用该名称；
        2. 若未指定，默认使用场景类名，并根据“起始/结束动画编号”追加后缀
    返回：字符串形式的基础文件名
    """
    # 1. 优先使用用户指定的文件名
    if self.file_name:
        return self.file_name
    # 2. 未指定时，以场景类名作为基础
    name = str(self.scene)  # 场景对象的字符串表示（通常是场景类名，如"MyScene"）
    # 获取场景的“起始动画编号”和“结束动画编号”（用于分段渲染）
    start_anim_num = self.scene.start_at_animation_number
    end_anim_num = self.scene.end_at_animation_number
    # 若指定了起始动画编号，追加到文件名后（如"MyScene_5"）
    if start_anim_num is not None:
        name += f"_{start_anim_num}"
    # 若指定了结束动画编号，追加到文件名后（如"MyScene_5_10"）
    if end_anim_num is not None:
        name += f"_{end_anim_num}"
    return name

# 目录/文件路径获取器（Directory getters）
def get_image_file_path(self) -> str:
    """获取最后一帧图像的完整路径（字符串形式）"""
    return str(self.image_file_path)  # 转换Path对象为字符串，便于外部调用

def get_next_partial_movie_path(self) -> str:
    """
    获取“下一个分段视频”的完整路径（字符串形式）
    规则：分段目录 + 5位场景播放次数（补零） + 视频后缀（如output/scene/00001.mp4）
    用途：细分输出时，每段动画生成一个独立视频文件，按播放次数编号
    """
    # 分段视频文件名：场景播放次数（5位补零，如第1次播放为"00001"）
    partial_file_name = Path(self.partial_movie_directory, f"{self.scene.num_plays:05}")
    # 添加视频后缀，转换为字符串返回
    return str(partial_file_name.with_suffix(self.movie_file_extension))

def get_movie_file_path(self) -> str:
    """获取最终视频文件的完整路径（字符串形式）"""
    return str(self.movie_file_path)  # 转换Path对象为字符串，便于外部调用

# 音频相关方法（Sound）
def init_audio(self) -> None:
    """
    初始化音频相关状态
    逻辑：标记当前是否包含音频，默认初始化为“不包含”（后续添加音频时更新该状态）
    """
    self.includes_sound: bool = False  # 布尔值，标识场景是否包含音频（True=包含，False=不包含）

def create_audio_segment(self) -> None:
    """
    创建一个空的音频片段（静音片段）
    用途：作为音频拼接的基础容器，后续将场景中的音频片段追加到该对象中
    依赖：pydub.AudioSegment，生成的静音片段默认使用pydub的默认参数（如采样率）
    """
    # 创建静音音频片段（AudioSegment.silent()默认生成1秒静音，后续可通过拼接扩展）
    self.audio_segment = AudioSegment.silent()

def add_audio_segment(
        self,
        new_segment: AudioSegment,
        time: float | None = None,
        gain_to_background: float | None = None
    ) -> None:
    """
    向场景音频中添加一段音频片段，并处理时间对齐和音频叠加
    
    参数说明：
        new_segment: 待添加的音频片段（pydub.AudioSegment对象）
        time: 音频片段开始播放的时间点（秒），None则默认追加到现有音频末尾
        gain_to_background: 叠加时背景音频的增益调整（dB），用于控制背景音量
    逻辑：
        1. 首次添加音频时初始化音频容器（静音片段）
        2. 根据指定时间点调整音频容器长度（确保有足够空间容纳新片段）
        3. 将新片段叠加到指定时间位置（支持覆盖式叠加）
    """
    # 若首次添加音频，标记包含音频并创建基础静音片段
    if not self.includes_sound:
        self.includes_sound = True
        self.create_audio_segment()
    
    # 获取当前音频容器和其总时长（秒）
    current_segment = self.audio_segment
    current_end_time = current_segment.duration_seconds
    
    # 确定新片段的起始时间：未指定则默认接在现有音频末尾
    if time is None:
        time = current_end_time
    # 校验时间合法性：不允许在负时间点添加音频
    if time < 0:
        raise Exception("Adding sound at timestamp < 0")
    
    # 计算新片段结束时间和所需扩展的长度
    new_segment_end = time + new_segment.duration_seconds
    duration_diff = new_segment_end - current_end_time
    
    # 若新片段超出当前音频长度，扩展音频容器（添加静音）
    if duration_diff > 0:
        # 计算需要添加的静音时长（毫秒，向上取整确保足够）
        silence_duration_ms = int(np.ceil(duration_diff * 1000))
        # 追加静音片段（无交叉淡入淡出）
        current_segment = current_segment.append(
            AudioSegment.silent(silence_duration_ms),
            crossfade=0,
        )
    
    # 将新音频片段叠加到指定时间位置（以毫秒为单位计算偏移）
    self.audio_segment = current_segment.overlay(
        new_segment,
        position=int(1000 * time),  # 转换秒为毫秒
        gain_during_overlay=gain_to_background  # 叠加时调整背景音量
    )

def add_sound(
        self,
        sound_file: str,
        time: float | None = None,
        gain: float | None = None,
        gain_to_background: float | None = None
    ) -> None:
    """
    从音频文件加载音频并添加到场景中（调用add_audio_segment完成添加）
    
    参数说明：
        sound_file: 音频文件路径（支持相对路径或内置音频名称）
        time: 音频开始播放的时间点（秒），None则默认追加到末尾
        gain: 音频本身的增益调整（dB），用于增强或减弱该音频
        gain_to_background: 叠加时背景音频的增益调整（dB）
    流程：
        1. 解析音频文件的完整路径（支持内置音频库）
        2. 加载音频文件为AudioSegment对象
        3. （可选）调整音频增益
        4. 调用add_audio_segment将音频添加到场景
    """
    # 获取音频文件的完整路径（处理内置音频和用户自定义音频）
    file_path = get_full_sound_file_path(sound_file)
    # 从文件加载音频片段（pydub自动识别格式）
    new_segment = AudioSegment.from_file(file_path)
    
    # 若指定了增益调整，应用到新音频片段
    if gain:
        new_segment = new_segment.apply_gain(gain)
    
    # 调用音频片段添加方法，完成最终添加
    self.add_audio_segment(new_segment, time, gain_to_background)

# 视频写入控制方法（Writers）
def begin(self) -> None:
    """
    开始视频写入流程（针对不分段输出模式）
    逻辑：若需要生成视频且不使用分段输出，打开FFmpeg管道准备写入帧数据
    """
    # 当不细分输出且需要生成视频时，初始化视频编码管道
    if not self.subdivide_output and self.write_to_movie:
        self.open_movie_pipe(self.get_movie_file_path())

def begin_animation(self) -> None:
    """
    开始单个动画片段的写入（针对分段输出模式）
    逻辑：若使用分段输出且需要生成视频，为当前动画片段打开新的FFmpeg管道
    """
    # 当细分输出且需要生成视频时，为下一个分段视频打开编码管道
    if self.subdivide_output and self.write_to_movie:
        self.open_movie_pipe(self.get_next_partial_movie_path())

def end_animation(self) -> None:
    """
    结束单个动画片段的写入（针对分段输出模式）
    逻辑：若使用分段输出且需要生成视频，关闭当前动画片段的FFmpeg管道
    """
    # 当细分输出且需要生成视频时，关闭当前分段视频的编码管道
    if self.subdivide_output and self.write_to_movie:
        self.close_movie_pipe()

    def finish(self) -> None:
    """
    完成场景文件写入的最终流程，处理视频收尾、音频合并、帧保存与文件打开
    核心逻辑：根据输出模式（不分段视频/单帧保存）执行对应收尾操作，确保输出文件可用
    """
    # 1. 处理不分段视频的收尾（若开启视频生成且不分段）
    if not self.subdivide_output and self.write_to_movie:
        self.close_movie_pipe()  # 关闭FFmpeg编码管道，完成临时视频文件
        if self.includes_sound:  # 若场景包含音频，将音频合并到视频中
            self.add_sound_to_video()
        # 打印视频文件就绪的提示信息（含文件路径）
        self.print_file_ready_message(self.get_movie_file_path())
    
    # 2. 保存最后一帧图像（若开启该功能）
    if self.save_last_frame:
        self.scene.update_frame(force_draw=True)  # 强制更新场景帧，确保获取最新画面
        final_image = self.scene.get_image()  # 获取场景的当前图像（最后一帧）
        self.save_final_image(final_image)  # 保存图像到指定路径
    
    # 3. 自动打开输出文件（若满足配置条件）
    if self.should_open_file():
        self.open_file()

def open_movie_pipe(self, file_path: str) -> None:
    """
    打开FFmpeg编码管道，准备将渲染帧写入视频文件
    核心：构造FFmpeg命令，创建子进程处理帧数据，生成临时视频文件（后续会合并音频或重命名）
    
    参数：file_path - 最终视频文件的目标路径
    """
    # 拆分文件路径为“文件名（无后缀）”和“后缀”，用于生成临时文件路径
    stem, ext = os.path.splitext(file_path)
    self.final_file_path = file_path  # 记录最终视频文件路径
    self.temp_file_path = f"{stem}_temp{ext}"  # 临时视频文件路径（先存无音频版本）

    # 获取相机配置参数（用于FFmpeg命令）
    fps = self.scene.camera.fps  # 视频帧率（帧/秒）
    width, height = self.scene.camera.get_pixel_shape()  # 视频分辨率（宽×高）

    # 构造FFmpeg视频滤镜参数（vf）：1. 垂直翻转（解决渲染帧上下颠倒问题）；2. 调整饱和度和gamma
    vf_arg = "vflip"  # 垂直翻转（因Manim渲染帧默认上下颠倒，需修正）
    vf_arg += f",eq=saturation={self.saturation}:gamma={self.gamma}"  # 色彩饱和度+亮度调整

    # 构造FFmpeg命令列表（通过子进程执行）
    command = [
        self.ffmpeg_bin,  # FFmpeg可执行文件路径
        "-y",  # 强制覆盖已存在的输出文件（避免询问）
        "-f", "rawvideo",  # 输入格式：原始视频帧（无压缩的像素数据）
        "-s", f"{width}x{height}",  # 输入帧分辨率（宽×高）
        "-pix_fmt", "rgba",  # 输入像素格式（RGBA，含透明通道）
        "-r", str(fps),  # 输入帧率（与相机帧率一致）
        "-i", "-",  # 输入源：标准输入（通过管道传递帧数据）
        "-vf", vf_arg,  # 应用视频滤镜（翻转+色彩调整）
        "-an",  # 禁用音频输入（当前仅写入视频帧，音频后续合并）
        "-loglevel", "error",  # FFmpeg日志级别：仅显示错误（减少冗余输出）
    ]
    # 若指定了视频编码器，添加到命令（如libx264）
    if self.video_codec:
        command += ["-vcodec", self.video_codec]
    # 若指定了输出像素格式，添加到命令（如yuv420p）
    if self.pixel_format:
        command += ["-pix_fmt", self.pixel_format]
    # 命令最后指定输出路径（临时视频文件）
    command += [self.temp_file_path]

    # 创建FFmpeg子进程，开启标准输入管道（用于传递帧数据）
    self.writing_process = sp.Popen(command, stdin=sp.PIPE)

    # 若不启用静默模式，初始化进度条（显示渲染进度）
    if not self.quiet:
        self.progress_display = ProgressDisplay(
            range(self.total_frames),  # 进度条总范围（总帧数）
            leave=False,  # 进度条完成后不保留（避免占用终端空间）
            ascii=True if platform.system() == "Windows" else None,  # Windows用ASCII字符渲染进度条
            dynamic_ncols=True,  # 自动适配终端宽度
        )
        self.set_progress_display_description()  # 设置进度条描述文本（如“Rendering”）

def use_fast_encoding(self):
    """
    启用快速视频编码模式
    原理：使用RGB格式编码器（libx264rgb）和像素格式（rgb32），跳过YUV格式转换，提升编码速度
    注意：生成的视频文件体积可能更大，兼容性略低于默认的YUV420p格式
    """
    self.video_codec = "libx264rgb"  # 快速RGB编码器（无需转换色彩空间）
    self.pixel_format = "rgb32"  # 对应的RGB像素格式（32位，含透明通道）

def get_insert_file_path(self, index: int) -> Path:
    """
    生成“插入片段”的视频文件路径（用于场景中插入额外动画片段的场景）
    
    参数：index - 插入片段的序号（避免文件名冲突）
    返回：插入片段的完整路径（Path对象）
    """
    # 获取最终视频文件的路径对象
    movie_path = Path(self.get_movie_file_path())
    scene_name = movie_path.stem  # 提取场景名称（无后缀的文件名）
    # 构造插入片段的存储目录（在最终视频目录下创建“inserts”子目录）
    insert_dir = Path(movie_path.parent, "inserts")
    guarantee_existence(insert_dir)  # 确保目录存在（不存在则创建）
    # 生成插入片段的文件名（场景名_序号.后缀），返回完整路径
    return Path(insert_dir, f"{scene_name}_{index}").with_suffix(self.movie_file_extension)

def begin_insert(self):
    """
    开始处理“插入片段”的视频写入
    逻辑：临时开启视频生成模式，创建不冲突的插入片段路径，打开FFmpeg编码管道
    用途：在主场景渲染过程中，插入额外的小动画片段（如临时标注、补充说明）
    """
    # 临时开启视频写入模式（即使主场景未开启，插入片段也需生成视频）
    self.write_to_movie = True
    self.init_output_directories()  # 重新初始化输出目录（确保插入目录存在）
    
    # 查找未被占用的插入片段序号（从0开始，避免覆盖已有文件）
    index = 0
    while (insert_path := self.get_insert_file_path(index)).exists():
        index += 1
    self.inserted_file_path = insert_path  # 记录当前插入片段的路径
    
    # 打开FFmpeg编码管道，开始写入插入片段的帧数据
    self.open_movie_pipe(self.inserted_file_path)

def end_insert(self):
    """
    结束“插入片段”的视频写入
    逻辑：关闭FFmpeg编码管道，恢复原视频写入模式，打印插入片段就绪信息
    """
    self.close_movie_pipe()  # 关闭插入片段的FFmpeg编码管道
    self.write_to_movie = False  # 恢复原视频写入模式（避免影响主场景）
    # 打印插入片段文件就绪的提示信息
    self.print_file_ready_message(self.inserted_file_path)

def has_progress_display(self):
    """
    检查是否已初始化进度条对象
    返回：布尔值（True表示进度条已创建，False表示未创建或已关闭）
    """
    return self.progress_display is not None

def set_progress_display_description(self, file: str = "", sub_desc: str = "") -> None:
    """
    设置进度条的描述文本，确保文本长度不超过配置的最大长度（避免终端显示溢出）
    
    参数说明：
        file: 文件名（默认自动获取视频输出文件名）
        sub_desc: 补充描述（如“Rendering”“Encoding”，用于区分不同阶段）
    逻辑：
        1. 若未初始化进度条，直接返回（不执行操作）
        2. 自动补全文件名（未指定时从视频路径提取）
        3. 拼接完整描述并截断超长文本（末尾加“...”）
        4. 不足最大长度时用空格填充（保持进度条对齐）
    """
    # 若进度条未初始化，跳过操作
    if self.progress_display is None:
        return

    # 获取配置的进度条描述最大长度
    max_desc_len = self.progress_description_len
    # 若未指定文件名，从最终视频路径中提取文件名（含后缀）
    if not file:
        file = os.path.split(self.get_movie_file_path())[1]
    
    # 拼接完整描述文本（文件名 + 补充描述）
    full_desc = f"{file} {sub_desc}".strip()  # 去除首尾多余空格
    
    # 处理超长文本：截断并添加省略号
    if len(full_desc) > max_desc_len:
        full_desc = full_desc[:max_desc_len - 3] + "..."  # 保留max_desc_len-3个字符，加“...”
    # 处理短文本：用空格填充至最大长度（确保进度条位置对齐）
    else:
        full_desc += " " * (max_desc_len - len(full_desc))
    
    # 更新进度条描述
    self.progress_display.set_description(full_desc)

def write_frame(self, camera: Camera) -> None:
    """
    将相机渲染的当前帧写入FFmpeg编码管道（用于生成视频）
    
    参数：camera - 场景关联的相机对象（提供当前帧的原始像素数据）
    逻辑：
        1. 仅当开启视频生成时执行（write_to_movie=True）
        2. 获取相机帧的原始字节数据（RGBA格式）
        3. 将字节数据写入FFmpeg子进程的标准输入
        4. 更新进度条（若已初始化）
    """
    # 仅在需要生成视频时执行帧写入
    if self.write_to_movie:
        # 从相机获取当前帧的原始FBO（帧缓冲对象）字节数据（RGBA格式）
        raw_frame_bytes = camera.get_raw_fbo_data()
        # 将原始字节数据写入FFmpeg子进程的标准输入（管道传递）
        self.writing_process.stdin.write(raw_frame_bytes)
        # 若进度条已初始化，更新进度（前进1帧）
        if self.progress_display is not None:
            self.progress_display.update()

def close_movie_pipe(self) -> None:
    """
    关闭FFmpeg编码管道，完成当前视频片段的写入，并处理临时文件
    核心逻辑：
        1. 关闭标准输入管道，等待FFmpeg子进程结束
        2. 终止FFmpeg子进程，释放资源
        3. 关闭进度条（若已初始化）
        4. 根据是否中断，决定临时文件的处理方式（重命名为最终文件或保留临时文件）
    """
    # 关闭FFmpeg子进程的标准输入（告知FFmpeg无更多帧数据）
    self.writing_process.stdin.close()
    # 等待FFmpeg子进程完成编码（阻塞直到子进程退出）
    self.writing_process.wait()
    # 终止FFmpeg子进程（确保资源释放）
    self.writing_process.terminate()
    
    # 若进度条已初始化，关闭进度条（清理终端输出）
    if self.progress_display is not None:
        self.progress_display.close()

    # 根据是否因中断结束，处理临时文件
    if not self.ended_with_interrupt:
        # 正常结束：将临时视频文件重命名为最终视频文件（覆盖原文件，因FFmpeg已加-y参数）
        shutil.move(self.temp_file_path, self.final_file_path)
    else:
        # 中断结束：将临时文件路径作为最终文件路径（便于用户后续查看未完成文件）
        self.movie_file_path = self.temp_file_path

def add_sound_to_video(self) -> None:
    """
    将场景中的音频片段合并到视频文件中（视频生成后执行）
    流程：
        1. 生成临时WAV音频文件（从场景音频片段导出）
        2. 调用FFmpeg将音频与视频合并（视频流复制，仅添加音频流）
        3. 替换原视频文件为带音频的新文件，删除临时音频文件
    目的：解决FFmpeg管道写入时无法同步添加音频的问题（先写视频，后合并音频）
    """
    # 获取最终视频文件的路径
    final_video_path = self.get_movie_file_path()
    # 拆分视频路径为“文件名（无后缀）”和“后缀”，用于生成临时音频文件路径
    video_stem, video_ext = os.path.splitext(final_video_path)
    temp_audio_path = f"{video_stem}.wav"  # 临时WAV音频文件路径

    # 补充一个0长度静音片段（确保音频文件时长与视频完全匹配，避免FFmpeg警告）
    self.add_audio_segment(AudioSegment.silent(0))
    # 将场景中的音频片段导出为WAV文件（比特率312k，保证音频质量）
    self.audio_segment.export(
        temp_audio_path,
        bitrate='312k'  # 音频比特率（越高质量越好，312k为较高质量配置）
    )

    # 生成合并音频后的临时视频文件路径
    temp_merged_video_path = f"{video_stem}_temp{video_ext}"
    # 构造FFmpeg合并音频与视频的命令
    ffmpeg_command = [
        self.ffmpeg_bin,  # FFmpeg可执行文件路径
        "-i", final_video_path,  # 输入文件1：原视频文件（无音频）
        "-i", temp_audio_path,   # 输入文件2：临时音频文件
        "-y",  # 强制覆盖已存在的输出文件
        "-c:v", "copy",  # 视频流编码方式：直接复制（不重新编码，节省时间）
        "-c:a", "aac",   # 音频流编码方式：AAC（兼容性好，适合视频封装）
        "-b:a", "320k",  # 音频比特率：320k（高质量音频配置）
        "-map", "0:v:0", # 选择输入1的第1个视频流（原视频的视频流）
        "-map", "1:a:0", # 选择输入2的第1个音频流（临时音频的音频流）
        "-loglevel", "error",  # 日志级别：仅显示错误（减少冗余输出）
        temp_merged_video_path  # 输出文件：合并后的临时视频文件
    ]

    # 执行FFmpeg命令（阻塞直到合并完成）
    sp.call(ffmpeg_command)
    # 将合并后的临时视频文件重命名为最终视频文件（替换原无音频视频）
    shutil.move(temp_merged_video_path, final_video_path)
    # 删除临时音频文件（清理中间文件，避免占用空间）
    os.remove(temp_audio_path)

    def save_final_image(self, image: Image) -> None:
        file_path = self.get_image_file_path()
        image.save(file_path)
        self.print_file_ready_message(file_path)

    def print_file_ready_message(self, file_path: str) -> None:
        if not self.quiet:
            log.info(f"File ready at {file_path}")

    def should_open_file(self) -> bool:
        return any([
            self.show_file_location_upon_completion,
            self.open_file_upon_completion,
        ])

    def open_file(self) -> None:
        if self.quiet:
            curr_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")

        current_os = platform.system()
        file_paths = []

        if self.save_last_frame:
            file_paths.append(self.get_image_file_path())
        if self.write_to_movie:
            file_paths.append(self.get_movie_file_path())

        for file_path in file_paths:
            if current_os == "Windows":
                os.startfile(file_path)
            else:
                commands = []
                if current_os == "Linux":
                    commands.append("xdg-open")
                elif current_os.startswith("CYGWIN"):
                    commands.append("cygstart")
                else:  # Assume macOS
                    commands.append("open")

                if self.show_file_location_upon_completion:
                    commands.append("-R")

                commands.append(file_path)

                FNULL = open(os.devnull, 'w')
                sp.call(commands, stdout=FNULL, stderr=sp.STDOUT)
                FNULL.close()

        if self.quiet:
            sys.stdout.close()
            sys.stdout = curr_stdout
