# 从__future__模块导入annotations特性，用于支持函数注解中类型的前向引用（即可以引用尚未定义的类型）
from __future__ import annotations
# 导入numpy库并简写为np，用于数值计算和数组操作
import numpy as np

# 从typing模块导入TYPE_CHECKING常量，用于类型检查相关的条件导入
from typing import TYPE_CHECKING
# 如果处于类型检查阶段（非运行时），则执行以下代码块
if TYPE_CHECKING:
    # 从typing模块导入List类型，用于类型注解
    from typing import List
    # 从manimlib.typing模块导入ManimColor和Vect3类型，用于类型注解
    from manimlib.typing import ManimColor, Vect3

# 注释：参见manimlib/default_config.yml文件（提示配置文件位置）
# 从manimlib.config模块导入manim_config对象，该对象包含manim库的默认配置信息
from manimlib.config import manim_config


# 定义默认分辨率，从manim配置的相机分辨率中获取，类型为整数元组
DEFAULT_RESOLUTION: tuple[int, int] = manim_config.camera.resolution
# 从默认分辨率中提取像素宽度（元组第一个元素）
DEFAULT_PIXEL_WIDTH: int = DEFAULT_RESOLUTION[0]
# 从默认分辨率中提取像素高度（元组第二个元素）
DEFAULT_PIXEL_HEIGHT: int = DEFAULT_RESOLUTION[1]

# 与默认相机帧相关的尺寸
# 计算宽高比，即像素宽度除以像素高度
ASPECT_RATIO: float = DEFAULT_PIXEL_WIDTH / DEFAULT_PIXEL_HEIGHT
# 从配置中获取帧高度
FRAME_HEIGHT: float = manim_config.sizes.frame_height
# 根据帧高度和宽高比计算帧宽度
FRAME_WIDTH: float = FRAME_HEIGHT * ASPECT_RATIO
# 定义帧的形状（宽和高组成的元组）
FRAME_SHAPE: tuple[float, float] = (FRAME_WIDTH, FRAME_HEIGHT)
# 帧的Y方向半径（高度的一半）
FRAME_Y_RADIUS: float = FRAME_HEIGHT / 2
# 帧的X方向半径（宽度的一半）
FRAME_X_RADIUS: float = FRAME_WIDTH / 2


# 用于放置移动对象（mobjects）的辅助间距值
# 小间距，从配置中获取
SMALL_BUFF: float = manim_config.sizes.small_buff
# 中-小间距，从配置中获取
MED_SMALL_BUFF: float = manim_config.sizes.med_small_buff
# 中-大间距，从配置中获取
MED_LARGE_BUFF: float = manim_config.sizes.med_large_buff
# 大间距，从配置中获取
LARGE_BUFF: float = manim_config.sizes.large_buff

# 对象到边缘的默认间距，从配置中获取
DEFAULT_MOBJECT_TO_EDGE_BUFF: float = manim_config.sizes.default_mobject_to_edge_buff
# 对象之间的默认间距，从配置中获取
DEFAULT_MOBJECT_TO_MOBJECT_BUFF: float = manim_config.sizes.default_mobject_to_mobject_buff


# 标准向量定义（3D向量，基于numpy数组）
# 原点向量
ORIGIN: Vect3 = np.array([0., 0., 0.])
# 向上方向向量（Y轴正方向）
UP: Vect3 = np.array([0., 1., 0.])
# 向下方向向量（Y轴负方向）
DOWN: Vect3 = np.array([0., -1., 0.])
# 向右方向向量（X轴正方向）
RIGHT: Vect3 = np.array([1., 0., 0.])
# 向左方向向量（X轴负方向）
LEFT: Vect3 = np.array([-1., 0., 0.])
# 向内方向向量（Z轴负方向）
IN: Vect3 = np.array([0., 0., -1.])
# 向外方向向量（Z轴正方向）
OUT: Vect3 = np.array([0., 0., 1.])
# X轴向量（同RIGHT）
X_AXIS: Vect3 = np.array([1., 0., 0.])
# Y轴向量（同UP）
Y_AXIS: Vect3 = np.array([0., 1., 0.])
# Z轴向量（同OUT）
Z_AXIS: Vect3 = np.array([0., 0., 1.])

# 空点数组（包含单个原点的二维数组）
NULL_POINTS = np.array([[0., 0., 0.]])

# 对角线方向的简写向量
# 上左方向（UP + LEFT）
UL: Vect3 = UP + LEFT
# 上右方向（UP + RIGHT）
UR: Vect3 = UP + RIGHT
# 下左方向（DOWN + LEFT）
DL: Vect3 = DOWN + LEFT
# 下右方向（DOWN + RIGHT）
DR: Vect3 = DOWN + RIGHT

# 帧边界的位置向量
# 顶部边界（Y轴半径×向上向量）
TOP: Vect3 = FRAME_Y_RADIUS * UP
# 底部边界（Y轴半径×向下向量）
BOTTOM: Vect3 = FRAME_Y_RADIUS * DOWN
# 左侧边界（X轴半径×向左向量）
LEFT_SIDE: Vect3 = FRAME_X_RADIUS * LEFT
# 右侧边界（X轴半径×向右向量）
RIGHT_SIDE: Vect3 = FRAME_X_RADIUS * RIGHT

# 角度相关常量
# π（圆周率），从numpy获取
PI: float = np.pi
# τ（2π，圆的周长与半径之比）
TAU: float = 2 * PI
# 1度对应的弧度值（τ/360）
DEG: float = TAU / 360
# DEG的全写别名（许多旧动画使用该名称）
DEGREES = DEG
# 弧度单位常量（用于可读性，与30*DEG等表达式对应）
RADIANS: float = 1

# 与文本相关的常量
# 正常字体样式
NORMAL: str = "NORMAL"
# 斜体字体样式
ITALIC: str = "ITALIC"
# 倾斜字体样式
OBLIQUE: str = "OBLIQUE"
# 粗体字体样式
BOLD: str = "BOLD"

# 默认描边宽度，从配置中获取
DEFAULT_STROKE_WIDTH: float = manim_config.vmobject.default_stroke_width

# 颜色常量（从配置中获取对应颜色值）
BLUE_E: ManimColor = manim_config.colors.blue_e
BLUE_D: ManimColor = manim_config.colors.blue_d
BLUE_C: ManimColor = manim_config.colors.blue_c
BLUE_B: ManimColor = manim_config.colors.blue_b
BLUE_A: ManimColor = manim_config.colors.blue_a
TEAL_E: ManimColor = manim_config.colors.teal_e
TEAL_D: ManimColor = manim_config.colors.teal_d
TEAL_C: ManimColor = manim_config.colors.teal_c
TEAL_B: ManimColor = manim_config.colors.teal_b
TEAL_A: ManimColor = manim_config.colors.teal_a
GREEN_E: ManimColor = manim_config.colors.green_e
GREEN_D: ManimColor = manim_config.colors.green_d
GREEN_C: ManimColor = manim_config.colors.green_c
GREEN_B: ManimColor = manim_config.colors.green_b
GREEN_A: ManimColor = manim_config.colors.green_a
YELLOW_E: ManimColor = manim_config.colors.yellow_e
YELLOW_D: ManimColor = manim_config.colors.yellow_d
YELLOW_C: ManimColor = manim_config.colors.yellow_c
YELLOW_B: ManimColor = manim_config.colors.yellow_b
YELLOW_A: ManimColor = manim_config.colors.yellow_a
GOLD_E: ManimColor = manim_config.colors.gold_e
GOLD_D: ManimColor = manim_config.colors.gold_d
GOLD_C: ManimColor = manim_config.colors.gold_c
GOLD_B: ManimColor = manim_config.colors.gold_b
GOLD_A: ManimColor = manim_config.colors.gold_a
RED_E: ManimColor = manim_config.colors.red_e
RED_D: ManimColor = manim_config.colors.red_d
RED_C: ManimColor = manim_config.colors.red_c
RED_B: ManimColor = manim_config.colors.red_b
RED_A: ManimColor = manim_config.colors.red_a
MAROON_E: ManimColor = manim_config.colors.maroon_e
MAROON_D: ManimColor = manim_config.colors.maroon_d
MAROON_C: ManimColor = manim_config.colors.maroon_c
MAROON_B: ManimColor = manim_config.colors.maroon_b
MAROON_A: ManimColor = manim_config.colors.maroon_a
PURPLE_E: ManimColor = manim_config.colors.purple_e
PURPLE_D: ManimColor = manim_config.colors.purple_d
PURPLE_C: ManimColor = manim_config.colors.purple_c
PURPLE_B: ManimColor = manim_config.colors.purple_b
PURPLE_A: ManimColor = manim_config.colors.purple_a
GREY_E: ManimColor = manim_config.colors.grey_e
GREY_D: ManimColor = manim_config.colors.grey_d
GREY_C: ManimColor = manim_config.colors.grey_c
GREY_B: ManimColor = manim_config.colors.grey_b
GREY_A: ManimColor = manim_config.colors.grey_a
WHITE: ManimColor = manim_config.colors.white
BLACK: ManimColor = manim_config.colors.black
GREY_BROWN: ManimColor = manim_config.colors.grey_brown
DARK_BROWN: ManimColor = manim_config.colors.dark_brown
LIGHT_BROWN: ManimColor = manim_config.colors.light_brown
PINK: ManimColor = manim_config.colors.pink
LIGHT_PINK: ManimColor = manim_config.colors.light_pink
GREEN_SCREEN: ManimColor = manim_config.colors.green_screen
ORANGE: ManimColor = manim_config.colors.orange
PURE_RED: ManimColor = manim_config.colors.pure_red
PURE_GREEN: ManimColor = manim_config.colors.pure_green
PURE_BLUE: ManimColor = manim_config.colors.pure_blue

# 所有Manim颜色的列表（从配置的颜色值中提取）
MANIM_COLORS: List[ManimColor] = list(manim_config.colors.values())

# 各种颜色"中间色调"的简写（使用C后缀的颜色）
BLUE: ManimColor = BLUE_C
TEAL: ManimColor = TEAL_C
GREEN: ManimColor = GREEN_C
YELLOW: ManimColor = YELLOW_C
GOLD: ManimColor = GOLD_C
RED: ManimColor = RED_C
MAROON: ManimColor = MAROON_C
PURPLE: ManimColor = PURPLE_C
GREY: ManimColor = GREY_C

# 3Blue1Brown风格的颜色映射列表
COLORMAP_3B1B: List[ManimColor] = [BLUE_E, GREEN, YELLOW, RED]

# 默认对象颜色应像背景色一样可配置
# DEFAULT_MOBJECT_COLOR主要用于文本、公式、线条等对象，默认为白色
# DEFAULT_LIGHT_COLOR主要用于坐标轴、箭头、圆环等浅色对象，默认为GREY_B
DEFAULT_MOBJECT_COLOR: ManimColor = manim_config.mobject.default_mobject_color or WHITE
DEFAULT_LIGHT_COLOR: ManimColor = manim_config.mobject.default_light_color or GREY_B

# 向量对象（VMobject）的默认描边颜色，从配置中获取，默认GREY_A
DEFAULT_VMOBJECT_STROKE_COLOR : ManimColor = manim_config.vmobject.default_stroke_color or GREY_A
# 向量对象（VMobject）的默认填充颜色，从配置中获取，默认GREY_C
DEFAULT_VMOBJECT_FILL_COLOR : ManimColor = manim_config.vmobject.default_fill_color or GREY_C
