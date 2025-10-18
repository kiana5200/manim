# ManimGL 文件操作工具：提供文件路径确保存在、文件查找与下载功能，
# 支持本地文件检索和网络资源自动下载，确保框架能可靠获取所需资源（如图片、音频）。


from __future__ import annotations

import os
from pathlib import Path
import hashlib  # 哈希工具（未直接使用，依赖其他模块哈希函数）

import numpy as np
import validators  # 用于验证 URL 有效性
import urllib.request  # 用于下载网络文件

# 导入 Manim 目录工具和哈希函数
import manimlib.utils.directories
from manimlib.utils.simple_functions import hash_string  # 字符串哈希（用于生成下载文件唯一名称）

from typing import TYPE_CHECKING

# 仅静态类型检查阶段导入类型注解
if TYPE_CHECKING:
    from typing import Iterable


def guarantee_existence(path: str | Path) -> Path:
    """
    确保指定路径的目录存在（不存在则递归创建），并返回该路径的绝对路径。
    
    核心作用：避免因目录不存在导致的文件写入错误（如渲染输出目录、缓存目录）。
    
    参数：path - 目录路径（字符串或 Path 对象）
    返回：Path - 绝对路径对象（确保目录已存在）
    """
    path = Path(path)
    # 递归创建目录（parents=True 表示创建所有父目录，exist_ok=True 表示目录存在时不报错）
    path.mkdir(parents=True, exist_ok=True)
    return path.absolute()  # 返回绝对路径


def find_file(
    file_name: str,
    directories: Iterable[str] | None = None,
    extensions: Iterable[str] | None = None
) -> Path:
    """
    查找文件：支持网络 URL 自动下载、本地路径直接返回、指定目录+扩展名组合检索，
    确保能找到目标文件（常用于加载图片、音频等资源）。
    
    查找优先级：
    1. 若 file_name 是有效 URL，下载到本地下载目录并返回路径；
    2. 若 file_name 是本地已存在的路径，直接返回；
    3. 在指定目录中按指定扩展名组合检索，返回第一个找到的文件路径；
    4. 未找到时抛出 IOError。
    
    参数：
        file_name : 文件名、本地路径或网络 URL；
        directories : 检索的目录列表（默认包含当前目录）；
        extensions : 尝试的文件扩展名列表（默认包含空字符串，即原文件名）。
    返回：Path - 找到的文件路径
    异常：IOError - 未找到文件时抛出。
    """
    # 1. 检查是否为网络 URL，若是则下载到本地
    if validators.url(file_name):
        suffix = Path(file_name).suffix  # 获取 URL 中的文件扩展名（如 .png）
        file_hash = hash_string(file_name)  # 对 URL 哈希生成唯一文件名（避免重复下载）
        folder = manimlib.utils.directories.get_downloads_dir()  # 获取下载目录

        # 构建本地保存路径：下载目录/哈希值.扩展名
        path = Path(folder, file_hash).with_suffix(suffix)
        # 下载文件到本地路径
        urllib.request.urlretrieve(file_name, path)
        return path

    # 2. 检查是否为本地已存在的文件路径
    if os.path.exists(file_name):
        return Path(file_name)

    # 3. 在指定目录中按扩展名组合检索
    directories = directories or [""]  # 默认检索当前目录
    extensions = extensions or [""]     # 默认不添加额外扩展名
    # 生成所有可能的路径组合（目录 + 文件名 + 扩展名）
    possible_paths = (
        Path(directory, file_name + extension)
        for directory in directories
        for extension in extensions
    )
    # 遍历所有可能路径，返回第一个存在的文件
    for path in possible_paths:
        if path.exists():
            return path

    # 4. 所有途径均未找到文件
    raise IOError(f"{file_name} not Found")