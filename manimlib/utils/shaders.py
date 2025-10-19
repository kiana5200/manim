# ManimGL 着色器工具库：提供着色器程序创建、纹理加载、统一变量管理、着色器代码处理等核心功能，
# 是 OpenGL 渲染 pipeline 与 Manim 逻辑交互的桥梁，负责将图形数据转换为 GPU 可执行的渲染指令。


from __future__ import annotations

import os
import re
from functools import lru_cache  # 缓存装饰器：避免重复加载相同资源
import moderngl  # ModernGL 库：OpenGL 封装接口
from PIL import Image  # 图像处理：加载纹理图片
import numpy as np

# 导入目录和文件操作工具
from manimlib.utils.directories import get_shader_dir  # 着色器文件目录
from manimlib.utils.file_ops import find_file  # 文件查找工具

from typing import TYPE_CHECKING

# 仅静态类型检查阶段导入类型注解
if TYPE_CHECKING:
    from typing import Sequence, Optional


# ------------------------------ 全局状态管理 ------------------------------
# 存储着色器程序的统一变量镜像（CPU 端缓存，避免重复向 GPU 发送相同值）
# 键：程序 ID（id(program)），值：{变量名: 变量值}
PROGRAM_UNIFORM_MIRRORS: dict[int, dict[str, float | tuple]] = dict()


# ------------------------------ 纹理加载 ------------------------------
@lru_cache()
def image_path_to_texture(path: str, ctx: moderngl.Context) -> moderngl.Texture:
    """
    将图片文件转换为 ModernGL 纹理对象（带缓存，避免重复加载）。
    
    流程：
    1. 用 PIL 打开图片并转换为 RGBA 格式（确保含透明度通道）；
    2. 创建 ModernGL 纹理，尺寸为图片尺寸，通道数为 4（RGBA）；
    3. 将图片像素数据上传到纹理。
    
    参数：
        path : 图片文件路径；
        ctx : ModernGL OpenGL 上下文。
    返回：moderngl.Texture - 纹理对象（可绑定到着色器使用）。
    缓存：相同路径和上下文的调用会返回缓存的纹理对象。
    """
    im = Image.open(path).convert("RGBA")  # 打开图片并转为 RGBA
    return ctx.texture(
        size=im.size,                  # 纹理尺寸（图片宽高）
        components=len(im.getbands()),  # 通道数（RGBA 为 4）
        data=im.tobytes(),             # 像素数据（字节流）
    )


# ------------------------------ 着色器程序创建 ------------------------------
@lru_cache()
def get_shader_program(
        ctx: moderngl.context.Context,
        vertex_shader: str,
        fragment_shader: Optional[str] = None,
        geometry_shader: Optional[str] = None,
) -> moderngl.Program:
    """
    创建 ModernGL 着色器程序（带缓存，相同代码仅编译一次）。
    
    着色器程序是 OpenGL 渲染的核心，由顶点着色器、片段着色器（可选）、几何着色器（可选）组成，
    负责将顶点数据转换为屏幕像素。
    
    参数：
        ctx : OpenGL 上下文；
        vertex_shader : 顶点着色器代码（字符串）；
        fragment_shader : 片段着色器代码（可选）；
        geometry_shader : 几何着色器代码（可选）。
    返回：moderngl.Program - 编译链接后的着色器程序。
    缓存：相同代码组合的调用会返回缓存的程序对象。
    """
    return ctx.program(
        vertex_shader=vertex_shader,
        fragment_shader=fragment_shader,
        geometry_shader=geometry_shader,
    )


# ------------------------------ 统一变量设置 ------------------------------
def set_program_uniform(
    program: moderngl.Program,
    name: str,
    value: float | tuple | np.ndarray
) -> bool:
    """
    设置着色器程序的统一变量（uniform），并通过 CPU 端缓存避免重复上传（优化性能）。
    
    统一变量是着色器中全局可见的变量（如变换矩阵、颜色），由 CPU 传递给 GPU，
    重复设置相同值会浪费带宽，因此通过 `PROGRAM_UNIFORM_MIRRORS` 缓存当前值。
    
    参数：
        program : 着色器程序；
        name : 统一变量名称（需与着色器中定义一致）；
        value : 变量值（支持标量、元组、numpy 数组）。
    返回：bool - 若变量值有变化并成功设置，返回 True；否则返回 False。
    """
    pid = id(program)  # 用程序的内存地址作为唯一标识
    # 初始化该程序的统一变量镜像（若不存在）
    if pid not in PROGRAM_UNIFORM_MIRRORS:
        PROGRAM_UNIFORM_MIRRORS[pid] = dict()
    uniform_mirror = PROGRAM_UNIFORM_MIRRORS[pid]

    # 标准化值：numpy 数组转为元组（便于缓存比较）
    if type(value) is np.ndarray and value.ndim > 0:
        value = tuple(value.flatten())
    # 若值未变化，直接返回 False（无需操作）
    if uniform_mirror.get(name, None) == value:
        return False

    try:
        # 设置着色器程序的统一变量值
        program[name].value = value
    except KeyError:
        # 变量名不存在于着色器中，返回 False
        return False
    # 更新缓存中的值
    uniform_mirror[name] = value
    return True


# ------------------------------ 着色器代码加载与处理 ------------------------------
@lru_cache()
def get_shader_code_from_file(filename: str) -> str | None:
    """
    从文件加载着色器代码，并处理代码插入（#INSERT 指令），支持模块化复用（带缓存）。
    
    核心功能：
    1. 查找着色器文件（默认在着色器目录）；
    2. 递归处理 #INSERT 指令（插入其他文件的代码，实现函数复用）；
    3. 返回完整的着色器代码字符串。
    
    参数：filename - 着色器文件名或路径
    返回：str | None - 处理后的着色器代码，文件不存在时返回 None。
    缓存：相同文件名的调用返回缓存的代码。
    """
    if not filename:
        return None

    try:
        # 查找着色器文件（优先在着色器目录，其次在系统根目录）
        filepath = find_file(
            filename,
            directories=[get_shader_dir(), "/"],
            extensions=[],  # 不自动添加扩展名（文件名需完整）
        )
    except IOError:
        return None  # 文件未找到

    # 读取文件内容
    with open(filepath, "r") as f:
        result = f.read()

    # 处理 #INSERT 指令：插入其他文件的代码（支持模块化）
    # 正则匹配所有 "#INSERT xxx.glsl" 行
    insertions = re.findall(r"^#INSERT .*\.glsl$", result, flags=re.MULTILINE)
    for line in insertions:
        # 提取被插入的文件名（去掉 "#INSERT " 前缀）
        inserted_filename = line.replace("#INSERT ", "")
        # 递归加载插入文件的代码（从 "inserts" 子目录）
        inserted_code = get_shader_code_from_file(
            os.path.join("inserts", inserted_filename)
        )
        # 替换 #INSERT 行为实际代码
        result = result.replace(line, inserted_code)
    return result


def get_colormap_code(rgb_list: Sequence[float]) -> str:
    """
    生成 GLSL 代码字符串：将 RGB 颜色列表转换为 GLSL 中的 vec3 数组，
    用于在着色器中定义颜色映射（如渐变色）。
    
    参数：rgb_list - RGB 颜色列表（每个元素为 [r, g, b]，0-1 范围）
    返回：str - GLSL 数组定义字符串（如 "vec3[2](vec3(1,0,0), vec3(0,1,0))"）。
    """
    # 生成每个颜色的 vec3 字符串
    data = ",".join(
        "vec3({}, {}, {})".format(*rgb)
        for rgb in rgb_list
    )
    # 组合为 GLSL 数组
    return f"vec3[{len(rgb_list)}]({data})"