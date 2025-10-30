# ManimGL 音频处理工具：提供音频文件路径获取和跨平台播放功能，
# 支持在动画中同步播放音效（如提示音、背景音乐），适配 Windows、macOS 和 Linux 系统。


from __future__ import annotations

import subprocess  # 用于调用系统命令播放音频
import threading  # （未直接使用，预留多线程播放支持）
import platform  # 用于检测操作系统类型

# 导入目录和文件操作工具
from manimlib.utils.directories import get_sound_dir  # 音频文件目录
from manimlib.utils.file_ops import find_file  # 文件查找工具


def get_full_sound_file_path(sound_file_name: str) -> str:
    """
    获取音频文件的完整路径：在配置的音频目录中查找指定文件，
    自动尝试常见音频扩展名（WAV、MP3），确保能正确定位本地音频资源。
    
    参数：sound_file_name - 音频文件名（可不带扩展名）
    返回：str - 音频文件的完整路径
    异常：IOError - 未找到音频文件时抛出（由 find_file 触发）。
    """
    return find_file(
        sound_file_name,
        directories=[get_sound_dir()],  # 仅在音频目录中查找
        extensions=[".wav", ".mp3", ""]  # 尝试的音频扩展名（含空，即原文件名）
    )


def play_sound(sound_file):
    """
    播放音频文件：根据操作系统类型调用相应的系统音频播放器，
    支持 Windows、macOS 和 Linux，播放时隐藏输出信息（避免干扰终端）。
    
    参数：sound_file - 音频文件名或路径（会通过 get_full_sound_file_path 解析）
    说明：
        - Windows 使用 PowerShell 的 SoundPlayer 播放；
        - macOS 使用系统自带的 afplay；
        - Linux 使用 aplay（ALSA 音频播放器）。
    """
    # 获取音频文件的完整路径
    full_path = get_full_sound_file_path(sound_file)
    # 检测操作系统类型
    system = platform.system()

    if system == "Windows":
        # Windows 系统：通过 PowerShell 调用 .NET 的 SoundPlayer 播放（同步播放）
        subprocess.Popen(
            ["powershell", "-c", f"(New-Object Media.SoundPlayer '{full_path}').PlaySync()"],
            shell=True,
            stdout=subprocess.DEVNULL,  # 隐藏标准输出
            stderr=subprocess.DEVNULL   # 隐藏错误输出
        )
    elif system == "Darwin":
        # macOS 系统：使用 afplay 命令播放
        subprocess.Popen(
            ["afplay", full_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    else:
        # Linux 系统：使用 aplay 命令播放（依赖 ALSA 音频系统）
        subprocess.Popen(
            ["aplay", full_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )