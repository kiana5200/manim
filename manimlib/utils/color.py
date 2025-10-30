# ManimGL 颜色处理工具库：提供颜色格式转换、插值、渐变生成等核心功能，
# 统一颜色在不同格式（RGB、RGBA、HEX、HSL、Colour 对象）之间的转换逻辑，
# 支持颜色混合、随机生成、 colormap 映射等高级操作，是 Mobject 颜色设置的基础依赖。


from __future__ import annotations

# 导入颜色处理核心库：colour 用于颜色空间转换，numpy 用于数值计算
from colour import Color
from colour import hex2rgb
from colour import rgb2hex
import numpy as np
import random
from matplotlib import pyplot  # 用于获取 matplotlib 的 colormap（如 viridis）

# 导入 Manim 常量和工具函数：3B1B 风格配色、插值函数等
from manimlib.constants import COLORMAP_3B1B  # 3Blue1Brown 经典配色表
from manimlib.constants import WHITE  # 默认白色
from manimlib.utils.bezier import interpolate  # 贝塞尔插值（用于颜色渐变）
from manimlib.utils.iterables import resize_with_interpolation  # 列表插值调整长度

from typing import TYPE_CHECKING

# 仅静态类型检查阶段导入类型注解
if TYPE_CHECKING:
    from typing import Iterable, Sequence, Callable
    from manimlib.typing import ManimColor, Vect3, Vect4, Vect3Array, Vect4Array, NDArray


# ------------------------------ 颜色格式转换（核心功能） ------------------------------
def color_to_rgb(color: ManimColor) -> Vect3:
    """
    将任意颜色格式转换为 RGB 数组（float 类型，范围 0-1）。
    
    参数：
        color : 支持字符串（HEX 如 "#FF0000"、颜色名如 "red"）、colour.Color 对象。
    返回：Vect3（3元素数组，[R, G, B]，每个元素 0-1 之间）。
    异常：传入无效颜色类型时抛出异常。
    """
    if isinstance(color, str):
        return hex_to_rgb(color)  # 字符串→HEX→RGB
    elif isinstance(color, Color):
        return np.array(color.get_rgb())  # Colour 对象直接取 RGB
    else:
        raise Exception("Invalid color type")  # 不支持的类型


def color_to_rgba(color: ManimColor, alpha: float = 1.0) -> Vect4:
    """
    将颜色转换为 RGBA 数组（float 类型，范围 0-1），包含透明度通道。
    
    参数：
        color : 任意支持的颜色格式；
        alpha : 透明度（0-1，默认 1.0 完全不透明）。
    返回：Vect4（[R, G, B, A]，每个元素 0-1 之间）。
    """
    return np.array([*color_to_rgb(color), alpha])  # 拼接 RGB 和 alpha


def rgb_to_color(rgb: Vect3 | Sequence[float]) -> Color:
    """
    将 RGB 数组（0-1）转换为 colour.Color 对象。
    
    参数：rgb - 3元素数组或序列（R, G, B，0-1 范围）。
    返回：colour.Color 对象，转换失败时返回白色（WHITE）。
    """
    try:
        return Color(rgb=tuple(rgb))  # RGB→Colour 对象
    except ValueError:
        return Color(WHITE)  # 异常处理：返回白色


def rgba_to_color(rgba: Vect4) -> Color:
    """
    从 RGBA 数组中提取 RGB 部分，转换为 colour.Color 对象（忽略 alpha 通道）。
    
    参数：rgba - 4元素数组（R, G, B, A）。
    返回：colour.Color 对象。
    """
    return rgb_to_color(rgba[:3])  # 取前3个元素（RGB）


def rgb_to_hex(rgb: Vect3 | Sequence[float]) -> str:
    """
    将 RGB 数组（0-1）转换为 HEX 颜色字符串（如 "#FF0000"）。
    
    参数：rgb - 3元素数组或序列（R, G, B，0-1 范围）。
    返回：大写 HEX 字符串（带 # 前缀，6位字符）。
    """
    return rgb2hex(rgb, force_long=True).upper()  # 强制6位格式，转为大写


def hex_to_rgb(hex_code: str) -> Vect3:
    """
    将 HEX 颜色字符串转换为 RGB 数组（0-1 范围）。
    
    参数：hex_code - HEX 字符串（如 "#FF0000"、"FF0000"）。
    返回：Vect3（[R, G, B]，0-1 范围）。
    """
    return np.array(hex2rgb(hex_code))  # HEX→RGB 数组


# ------------------------------ 颜色操作（反转、量化、混合） ------------------------------
def invert_color(color: ManimColor) -> Color:
    """
    反转颜色（RGB 通道取补，1 - 原通道值），如白色→黑色，红色→青色。
    
    参数：color - 任意支持的颜色格式。
    返回：反转后的 colour.Color 对象。
    """
    return rgb_to_color(1.0 - color_to_rgb(color))  # 1 - RGB 实现反转


def color_to_int_rgb(color: ManimColor) -> np.ndarray[int, np.dtype[np.uint8]]:
    """
    将颜色转换为整数 RGB 数组（0-255 范围，用于图像存储）。
    
    参数：color - 任意支持的颜色格式。
    返回：3元素 uint8 数组（[R, G, B]，0-255 范围）。
    """
    return (255 * color_to_rgb(color)).astype('uint8')  # 0-1→0-255 并转为整数


def color_to_int_rgba(color: ManimColor, opacity: float = 1.0) -> np.ndarray[int, np.dtype[np.uint8]]:
    """
    将颜色转换为整数 RGBA 数组（0-255 范围，含透明度）。
    
    参数：
        color : 任意支持的颜色格式；
        opacity : 透明度（0-1，默认 1.0）。
    返回：4元素 uint8 数组（[R, G, B, A]，0-255 范围）。
    """
    alpha = int(255 * opacity)  # 透明度转为 0-255 整数
    return np.array([*color_to_int_rgb(color), alpha], dtype=np.uint8)


def color_to_hex(color: ManimColor) -> str:
    """
    将任意颜色格式直接转换为 HEX 字符串（通过 colour.Color 中转）。
    
    参数：color - 任意支持的颜色格式。
    返回：大写 HEX 字符串（带 # 前缀）。
    """
    return Color(color).get_hex_l().upper()  # get_hex_l 返回长格式（6位）


def hex_to_int(rgb_hex: str) -> int:
    """
    将 HEX 字符串转换为整数（如 "#FF0000" → 16711680），用于快速颜色比较。
    
    参数：rgb_hex - HEX 颜色字符串。
    返回：对应的整数表示。
    """
    return int(rgb_hex[1:], 16)  # 去掉 # 后按 16 进制解析


def int_to_hex(rgb_int: int) -> str:
    """
    将整数转换为 HEX 颜色字符串（如 16711680 → "#FF0000"）。
    
    参数：rgb_int - 颜色的整数表示。
    返回：大写 HEX 字符串（带 # 前缀，6位）。
    """
    return f"#{rgb_int:06x}".upper()  # 格式化为 6 位 16 进制，补零


# ------------------------------ 颜色渐变与插值（高级功能） ------------------------------
def color_gradient(
    reference_colors: Iterable[ManimColor],
    length_of_output: int
) -> list[Color]:
    """
    生成从参考色列表插值得到的颜色渐变（按 RGB 空间的平方插值，避免亮度偏差）。
    
    参数：
        reference_colors : 参考颜色列表（如 [RED, BLUE, GREEN]）；
        length_of_output : 输出颜色的数量（渐变的总步数）。
    返回：长度为 length_of_output 的颜色列表，平滑过渡参考色。
    """
    if length_of_output == 0:
        return []
    # 转换参考色为 RGB 数组
    rgbs = list(map(color_to_rgb, reference_colors))
    # 生成插值因子（0 到 len(rgbs)-1 之间均匀分布）
    alphas = np.linspace(0, (len(rgbs) - 1), length_of_output)
    floors = alphas.astype('int')  # 整数部分（当前参考色索引）
    alphas_mod1 = alphas % 1  # 小数部分（插值比例）
    # 处理边界情况：最后一个元素强制取前一个参考色，插值比例为 1
    alphas_mod1[-1] = 1
    floors[-1] = len(rgbs) - 2
    # 对每个位置进行平方插值（sqrt 确保亮度线性变化）
    return [
        rgb_to_color(np.sqrt(interpolate(rgbs[i]**2, rgbs[i + 1]**2, alpha)))
        for i, alpha in zip(floors, alphas_mod1)
    ]


def interpolate_color(
    color1: ManimColor,
    color2: ManimColor,
    alpha: float
) -> Color:
    """
    在两个颜色之间按比例插值（RGB 空间平方插值，更自然的亮度过渡）。
    
    参数：
        color1/color2 : 起始和结束颜色；
        alpha : 插值比例（0→color1，1→color2）。
    返回：插值后的颜色对象。
    """
    # 转换为 RGB 后平方插值，再开方（避免亮度非线性变化）
    rgb = np.sqrt(interpolate(color_to_rgb(color1)**2, color_to_rgb(color2)** 2, alpha))
    return rgb_to_color(rgb)


def interpolate_color_by_hsl(
    color1: ManimColor,
    color2: ManimColor,
    alpha: float
) -> Color:
    """
    在 HSL 颜色空间插值（更自然的色相过渡，如红→绿会经过黄）。
    
    参数：同 interpolate_color
    返回：插值后的颜色对象。
    """
    # 转换为 HSL 数组后插值
    hsl1 = np.array(Color(color1).get_hsl())
    hsl2 = np.array(Color(color2).get_hsl())
    return Color(hsl=interpolate(hsl1, hsl2, alpha))


def average_color(*colors: ManimColor) -> Color:
    """
    计算多个颜色的平均值（RGB 平方平均，避免亮度偏差）。
    
    参数：*colors - 任意数量的颜色。
    返回：平均后的颜色对象。
    """
    rgbs = np.array(list(map(color_to_rgb, colors)))  # 转换为 RGB 数组
    # 平方平均后开方（亮度平均更合理）
    return rgb_to_color(np.sqrt((rgbs**2).mean(0)))


# ------------------------------ 随机颜色生成 ------------------------------
def random_color() -> Color:
    """生成随机颜色（RGB 通道随机取值 0-1）。"""
    return Color(rgb=tuple(np.random.random(3)))  # 随机 RGB 分量


def random_bright_color(
    hue_range: tuple[float, float] = (0.0, 1.0),
    saturation_range: tuple[float, float] = (0.5, 0.8),
    luminance_range: tuple[float, float] = (0.5, 1.0),
) -> Color:
    """
    生成随机亮色（控制 HSL 范围，确保饱和度和亮度较高）。
    
    参数：
        hue_range : 色相范围（0-1）；
        saturation_range : 饱和度范围（0-1，默认 0.5-0.8 确保鲜艳）；
        luminance_range : 亮度范围（0-1，默认 0.5-1.0 确保明亮）。
    返回：随机亮色对象。
    """
    return Color(hsl=(
        interpolate(*hue_range, random.random()),  # 随机色相
        interpolate(*saturation_range, random.random()),  # 随机饱和度
        interpolate(*luminance_range, random.random()),  # 随机亮度
    ))


# ------------------------------ 颜色映射（Colormap） ------------------------------
def get_colormap_from_colors(colors: Iterable[ManimColor]) -> Callable[[Sequence[float]], Vect4Array]:
    """
    从颜色列表创建 colormap 函数：输入 0-1 之间的数值，返回对应的 RGBA 颜色数组。
    
    参数：colors - 颜色列表（渐变的关键色）。
    返回：函数，接收数值序列（0-1），返回对应的 RGBA 数组序列。
    """
    rgbas = np.array([color_to_rgba(color) for color in colors])  # 转换为 RGBA 数组

    def func(values):
        alphas = np.clip(values, 0, 1)  # 限制输入在 0-1 范围
        scaled_alphas = alphas * (len(rgbas) - 1)  # 映射到颜色列表索引范围
        indices = scaled_alphas.astype(int)  # 当前颜色索引
        next_indices = np.clip(indices + 1, 0, len(rgbas) - 1)  # 下一个颜色索引（避免越界）
        inter_alphas = scaled_alphas % 1  # 插值比例（0-1）
        # 扩展为 RGBA 四个通道的插值比例
        inter_alphas = inter_alphas.repeat(4).reshape((len(indices), 4))
        # 插值计算最终颜色
        result = interpolate(rgbas[indices], rgbas[next_indices], inter_alphas)
        return result

    return func


def get_color_map(map_name: str) -> Callable[[Sequence[float]], Vect4Array]:
    """
    获取预定义的 colormap 函数（支持 3B1B 风格和 matplotlib 内置 colormap）。
    
    参数：map_name - 颜色映射名称（如 "3b1b_colormap"、"viridis"）。
    返回：colormap 函数（输入 0-1 数值，返回 RGBA 数组）。
    """
    if map_name == "3b1b_colormap":
        return get_colormap_from_colors(COLORMAP_3B1B)  # 3B1B 经典配色
    return pyplot.get_cmap(map_name)  # 其他名称使用 matplotlib 的 colormap


def get_colormap_list(
    map_name: str = "viridis",
    n_colors: int = 9
) -> Vect3Array:
    """
    获取 colormap 的颜色列表（固定长度，用于需要离散颜色的场景）。
    
    参数：
        map_name : 颜色映射名称；
        n_colors : 输出颜色数量。
    返回：长度为 n_colors 的 RGB 数组列表。
    """
    from matplotlib.cm import cmaps_listed  # 导入 matplotlib 颜色映射列表

    if map_name == "3b1b_colormap":
        rgbs = np.array([color_to_rgb(color) for color in COLORMAP_3B1B])
    else:
        rgbs = cmaps_listed[map_name].colors  # 获取 matplotlib 颜色映射的 RGB 数组
    # 插值调整长度到 n_colors
    return resize_with_interpolation(np.array(rgbs), n_colors)