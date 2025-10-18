# ManimGL 目录路径管理工具：提供框架各核心目录（缓存、输出、资源、临时文件等）的统一访问接口，
# 优先使用配置文件中的路径，未配置时自动生成系统默认路径，确保文件读写操作的路径一致性。


from __future__ import annotations

import os
import tempfile
import appdirs  # 跨平台获取系统标准目录（如缓存、下载路径）


from manimlib.config import manim_config  # 全局配置对象
from manimlib.config import get_manim_dir  # Manim 库根目录路径
from manimlib.utils.file_ops import guarantee_existence  # 确保目录存在（不存在则创建）


def get_directories() -> dict[str, str]:
    """
    获取配置中的所有目录设置：直接返回全局配置中 `directories` 字段的字典，
    包含基础目录、子目录（输出、图片、缓存等）的配置路径。
    
    返回：dict[str, str] - 配置中的目录字典（如 {"base": "./media", "output": "videos", ...}）。
    """
    return manim_config.directories


def get_cache_dir() -> str:
    """
    获取缓存目录路径：优先使用配置中的 `cache` 路径，未配置时使用系统默认缓存目录（如 ~/.cache/manim）。
    
    返回：str - 缓存目录的绝对路径（确保存在，用于存储 Tex 渲染缓存、临时计算结果等）。
    """
    # 配置中指定了缓存目录则使用，否则调用 appdirs 获取系统默认缓存目录
    return get_directories()["cache"] or appdirs.user_cache_dir("manim")


def get_temp_dir() -> str:
    """
    获取临时文件目录：优先使用配置中的 `temporary_storage` 路径，未配置时使用系统临时目录（如 /tmp）。
    
    返回：str - 临时目录路径（用于存储渲染过程中的临时帧、中间文件等）。
    """
    return get_directories()["temporary_storage"] or tempfile.gettempdir()


def get_downloads_dir() -> str:
    """
    获取下载目录路径：优先使用配置中的 `downloads` 路径，未配置时使用系统默认下载缓存目录（如 ~/.cache/manim_downloads）。
    
    返回：str - 下载目录路径（用于存储从网络自动下载的资源，如图片、字体）。
    """
    return get_directories()["downloads"] or appdirs.user_cache_dir("manim_downloads")


def get_output_dir() -> str:
    """
    获取动画输出目录：使用配置中的 `output` 路径，自动确保目录存在（不存在则创建），
    是视频、图片等最终渲染结果的保存位置。
    
    返回：str - 输出目录的绝对路径（已确保存在）。
    """
    # guarantee_existence 确保目录存在，不存在则递归创建
    return guarantee_existence(get_directories()["output"])


def get_raster_image_dir() -> str:
    """
    获取光栅图片资源目录：使用配置中的 `raster_images` 路径（如 "raster_images"），
    框架会在此目录查找 PNG、JPG 等像素图片资源。
    
    返回：str - 光栅图片目录路径。
    """
    return get_directories()["raster_images"]


def get_vector_image_dir() -> str:
    """
    获取矢量图片资源目录：使用配置中的 `vector_images` 路径（如 "vector_images"），
    框架会在此目录查找 SVG 等矢量图片资源。
    
    返回：str - 矢量图片目录路径。
    """
    return get_directories()["vector_images"]


def get_sound_dir() -> str:
    """
    获取音效资源目录：使用配置中的 `sounds` 路径（如 "sounds"），
    框架会在此目录查找音频文件（如 MP3、WAV）。
    
    返回：str - 音效目录路径。
    """
    return get_directories()["sounds"]


def get_shader_dir() -> str:
    """
    获取着色器文件目录：固定为 Manim 库安装目录下的 `manimlib/shaders`，
    存储框架内置的 GLSL 着色器代码（顶点着色器、片段着色器等）。
    
    返回：str - 着色器目录的绝对路径。
    """
    # 拼接 Manim 根目录与 "manimlib/shaders" 路径
    return os.path.join(get_manim_dir(), "manimlib", "shaders")