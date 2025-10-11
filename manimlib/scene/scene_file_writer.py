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
        if not self.includes_sound:
            self.includes_sound = True
            self.create_audio_segment()
        segment = self.audio_segment
        curr_end = segment.duration_seconds
        if time is None:
            time = curr_end
        if time < 0:
            raise Exception("Adding sound at timestamp < 0")

        new_end = time + new_segment.duration_seconds
        diff = new_end - curr_end
        if diff > 0:
            segment = segment.append(
                AudioSegment.silent(int(np.ceil(diff * 1000))),
                crossfade=0,
            )
        self.audio_segment = segment.overlay(
            new_segment,
            position=int(1000 * time),
            gain_during_overlay=gain_to_background,
        )

    def add_sound(
        self,
        sound_file: str,
        time: float | None = None,
        gain: float | None = None,
        gain_to_background: float | None = None
    ) -> None:
        file_path = get_full_sound_file_path(sound_file)
        new_segment = AudioSegment.from_file(file_path)
        if gain:
            new_segment = new_segment.apply_gain(gain)
        self.add_audio_segment(new_segment, time, gain_to_background)

    # Writers
    def begin(self) -> None:
        if not self.subdivide_output and self.write_to_movie:
            self.open_movie_pipe(self.get_movie_file_path())

    def begin_animation(self) -> None:
        if self.subdivide_output and self.write_to_movie:
            self.open_movie_pipe(self.get_next_partial_movie_path())

    def end_animation(self) -> None:
        if self.subdivide_output and self.write_to_movie:
            self.close_movie_pipe()

    def finish(self) -> None:
        if not self.subdivide_output and self.write_to_movie:
            self.close_movie_pipe()
            if self.includes_sound:
                self.add_sound_to_video()
            self.print_file_ready_message(self.get_movie_file_path())
        if self.save_last_frame:
            self.scene.update_frame(force_draw=True)
            self.save_final_image(self.scene.get_image())
        if self.should_open_file():
            self.open_file()

    def open_movie_pipe(self, file_path: str) -> None:
        stem, ext = os.path.splitext(file_path)
        self.final_file_path = file_path
        self.temp_file_path = stem + "_temp" + ext

        fps = self.scene.camera.fps
        width, height = self.scene.camera.get_pixel_shape()

        vf_arg = 'vflip'
        vf_arg += f',eq=saturation={self.saturation}:gamma={self.gamma}'

        command = [
            self.ffmpeg_bin,
            '-y',  # overwrite output file if it exists
            '-f', 'rawvideo',
            '-s', f'{width}x{height}',  # size of one frame
            '-pix_fmt', 'rgba',
            '-r', str(fps),  # frames per second
            '-i', '-',  # The input comes from a pipe
            '-vf', vf_arg,
            '-an',  # Tells ffmpeg not to expect any audio
            '-loglevel', 'error',
        ]
        if self.video_codec:
            command += ['-vcodec', self.video_codec]
        if self.pixel_format:
            command += ['-pix_fmt', self.pixel_format]
        command += [self.temp_file_path]
        self.writing_process = sp.Popen(command, stdin=sp.PIPE)

        if not self.quiet:
            self.progress_display = ProgressDisplay(
                range(self.total_frames),
                leave=False,
                ascii=True if platform.system() == 'Windows' else None,
                dynamic_ncols=True,
            )
            self.set_progress_display_description()

    def use_fast_encoding(self):
        self.video_codec = "libx264rgb"
        self.pixel_format = "rgb32"

    def get_insert_file_path(self, index: int) -> Path:
        movie_path = Path(self.get_movie_file_path())
        scene_name = movie_path.stem
        insert_dir = Path(movie_path.parent, "inserts")
        guarantee_existence(insert_dir)
        return Path(insert_dir, f"{scene_name}_{index}").with_suffix(self.movie_file_extension)

    def begin_insert(self):
        # Begin writing process
        self.write_to_movie = True
        self.init_output_directories()
        index = 0
        while (insert_path := self.get_insert_file_path(index)).exists():
            index += 1
        self.inserted_file_path = insert_path
        self.open_movie_pipe(self.inserted_file_path)

    def end_insert(self):
        self.close_movie_pipe()
        self.write_to_movie = False
        self.print_file_ready_message(self.inserted_file_path)

    def has_progress_display(self):
        return self.progress_display is not None

    def set_progress_display_description(self, file: str = "", sub_desc: str = "") -> None:
        if self.progress_display is None:
            return

        desc_len = self.progress_description_len
        if not file:
            file = os.path.split(self.get_movie_file_path())[1]
        full_desc = f"{file} {sub_desc}"
        if len(full_desc) > desc_len:
            full_desc = full_desc[:desc_len - 3] + "..."
        else:
            full_desc += " " * (desc_len - len(full_desc))
        self.progress_display.set_description(full_desc)

    def write_frame(self, camera: Camera) -> None:
        if self.write_to_movie:
            raw_bytes = camera.get_raw_fbo_data()
            self.writing_process.stdin.write(raw_bytes)
            if self.progress_display is not None:
                self.progress_display.update()

    def close_movie_pipe(self) -> None:
        self.writing_process.stdin.close()
        self.writing_process.wait()
        self.writing_process.terminate()
        if self.progress_display is not None:
            self.progress_display.close()

        if not self.ended_with_interrupt:
            shutil.move(self.temp_file_path, self.final_file_path)
        else:
            self.movie_file_path = self.temp_file_path

    def add_sound_to_video(self) -> None:
        movie_file_path = self.get_movie_file_path()
        stem, ext = os.path.splitext(movie_file_path)
        sound_file_path = stem + ".wav"
        # Makes sure sound file length will match video file
        self.add_audio_segment(AudioSegment.silent(0))
        self.audio_segment.export(
            sound_file_path,
            bitrate='312k',
        )
        temp_file_path = stem + "_temp" + ext
        commands = [
            self.ffmpeg_bin,
            "-i", movie_file_path,
            "-i", sound_file_path,
            '-y',  # overwrite output file if it exists
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "320k",
            # select video stream from first file
            "-map", "0:v:0",
            # select audio stream from second file
            "-map", "1:a:0",
            '-loglevel', 'error',
            # "-shortest",
            temp_file_path,
        ]
        sp.call(commands)
        shutil.move(temp_file_path, movie_file_path)
        os.remove(sound_file_path)

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
