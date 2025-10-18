# ManimGL 图像资源处理工具：提供光栅图像（如 PNG、JPG）和矢量图像（如 SVG）的路径获取功能，
# 以及图像反转处理，确保框架能正确定位图像资源并支持基础图像操作。


from __future__ import annotations

import numpy as np
from PIL import Image  # 用于图像加载和处理

# 导入目录工具和文件查找函数
from manimlib.utils.directories import get_raster_image_dir  # 光栅图像目录
from manimlib.utils.directories import get_vector_image_dir  # 矢量图像目录
from manimlib.utils.file_ops import find_file  # 文件查找工具

from typing import TYPE_CHECKING

# 仅静态类型检查阶段导入类型注解
if TYPE_CHECKING:
    from typing import Iterable


# ------------------------------ 图像路径获取（核心功能） ------------------------------
def get_full_raster_image_path(image_file_name: str) -> str:
    """
    获取光栅图像（如 PNG、JPG、GIF）的完整路径：在配置的光栅图像目录中查找指定文件，
    自动尝试常见图像扩展名，确保能正确定位本地图像资源。
    
    参数：image_file_name - 图像文件名（可不带扩展名）
    返回：str - 图像文件的完整路径
    异常：IOError - 未找到图像时抛出（由 find_file 触发）。
    """
    return find_file(
        image_file_name,
        directories=[get_raster_image_dir()],  # 仅在光栅图像目录中查找
        extensions=[".jpg", ".jpeg", ".png", ".gif", ""]  # 尝试的扩展名（含空，即原文件名）
    )


def get_full_vector_image_path(image_file_name: str) -> str:
    """
    获取矢量图像（如 SVG、XDV）的完整路径：在配置的矢量图像目录中查找指定文件，
    自动尝试矢量图像扩展名，适用于加载可缩放的矢量图形。
    
    参数：image_file_name - 图像文件名（可不带扩展名）
    返回：str - 图像文件的完整路径
    异常：IOError - 未找到图像时抛出。
    """
    return find_file(
        image_file_name,
        directories=[get_vector_image_dir()],  # 仅在矢量图像目录中查找
        extensions=[".svg", ".xdv", ""],  # 尝试的矢量图像扩展名
    )


# ------------------------------ 图像处理（辅助功能） ------------------------------
def invert_image(image: Iterable) -> Image.Image:
    """
    反转图像颜色（RGB 通道取反）：将图像的每个像素颜色转换为其互补色（如白色→黑色、红色→青色），
    适用于需要对比效果或适配不同背景的场景。
    
    参数：image - 输入图像（PIL.Image 对象或可转换为数组的图像数据）
    返回：PIL.Image - 颜色反转后的图像。
    """
    # 将图像转换为 numpy 数组（像素值 0-255）
    arr = np.array(image)
    # 计算反转后的像素值：255 - 原像素值（确保数据类型一致）
    arr = (255 * np.ones(arr.shape)).astype(arr.dtype) - arr
    # 转换回 PIL.Image 对象并返回
    return Image.fromarray(arr)