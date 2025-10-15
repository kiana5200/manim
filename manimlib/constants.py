# ManimGL 核心常量定义文件，负责**统一导出全局通用常量**（如分辨率、位置向量、颜色、角度单位），
# 所有常量值均从全局配置 `manim_config` 读取或推导，确保与用户配置（如自定义分辨率、颜色）保持一致，
# 是 Mobject 定位、渲染样式、动画参数设置的基础依赖。


# ------------------------------ 1. 基础渲染分辨率与帧尺寸常量 ------------------------------
# 从全局配置读取默认相机分辨率（如 1920x1080），是所有图形渲染的像素基准
DEFAULT_RESOLUTION: tuple[int, int] = manim_config.camera.resolution
DEFAULT_PIXEL_WIDTH: int = DEFAULT_RESOLUTION[0]  # 渲染宽度（像素）
DEFAULT_PIXEL_HEIGHT: int = DEFAULT_RESOLUTION[1]  # 渲染高度（像素）

# 帧尺寸（虚拟坐标系，非像素）：决定 Mobject 定位的空间范围
ASPECT_RATIO: float = DEFAULT_PIXEL_WIDTH / DEFAULT_PIXEL_HEIGHT  # 宽高比（与像素分辨率一致）
FRAME_HEIGHT: float = manim_config.sizes.frame_height  # 帧高度（虚拟单位，默认4.0）
FRAME_WIDTH: float = FRAME_HEIGHT * ASPECT_RATIO  # 帧宽度（由高度和宽高比推导）
FRAME_SHAPE: tuple[float, float] = (FRAME_WIDTH, FRAME_HEIGHT)  # 帧宽高（虚拟单位）
FRAME_Y_RADIUS: float = FRAME_HEIGHT / 2  # 帧垂直半径（从原点到上下边缘的距离）
FRAME_X_RADIUS: float = FRAME_WIDTH / 2    # 帧水平半径（从原点到左右边缘的距离）


# ------------------------------ 2. 缓冲距离常量 ------------------------------
# 控制 Mobject 之间、Mobject 与帧边缘的间距，避免元素重叠，提升视觉美观度
SMALL_BUFF: float = manim_config.sizes.small_buff          # 小缓冲（默认0.1）
MED_SMALL_BUFF: float = manim_config.sizes.med_small_buff  # 中小缓冲（默认0.25）
MED_LARGE_BUFF: float = manim_config.sizes.med_large_buff  # 中大缓冲（默认0.5）
LARGE_BUFF: float = manim_config.sizes.large_buff          # 大缓冲（默认1.0）

DEFAULT_MOBJECT_TO_EDGE_BUFF: float = manim_config.sizes.default_mobject_to_edge_buff  # 元素到帧边缘的默认缓冲
DEFAULT_MOBJECT_TO_MOBJECT_BUFF: float = manim_config.sizes.default_mobject_to_mobject_buff  # 元素之间的默认缓冲


# ------------------------------ 3. 标准方向向量常量 ------------------------------
# 3D 坐标系中的基础方向向量，用于 Mobject 定位、移动、旋转的方向参数（如 `square.move_to(RIGHT)`）
ORIGIN: Vect3 = np.array([0., 0., 0.])  # 原点（默认位置基准）
UP: Vect3 = np.array([0., 1., 0.])       # 向上（Y轴正方向）
DOWN: Vect3 = np.array([0., -1., 0.])    # 向下（Y轴负方向）
RIGHT: Vect3 = np.array([1., 0., 0.])    # 向右（X轴正方向）
LEFT: Vect3 = np.array([-1., 0., 0.])    # 向左（X轴负方向）
IN: Vect3 = np.array([0., 0., -1.])      # 向内（Z轴负方向，远离相机）
OUT: Vect3 = np.array([0., 0., 1.])      # 向外（Z轴正方向，朝向相机）
X_AXIS: Vect3 = np.array([1., 0., 0.])   # X轴方向（同 RIGHT）
Y_AXIS: Vect3 = np.array([0., 1., 0.])   # Y轴方向（同 UP）
Z_AXIS: Vect3 = np.array([0., 0., 1.])   # Z轴方向（同 OUT）

NULL_POINTS = np.array([[0., 0., 0.]])  # 空点集（默认初始化用）


# ------------------------------ 4. 对角方向与帧边缘位置常量 ------------------------------
# 常用对角方向向量（简化斜向定位，如 `square.move_to(UR)` 表示右上）
UL: Vect3 = UP + LEFT    # 左上（向上+向左）
UR: Vect3 = UP + RIGHT   # 右上（向上+向右）
DL: Vect3 = DOWN + LEFT  # 左下（向下+向左）
DR: Vect3 = DOWN + RIGHT # 右下（向下+向右）

# 帧边缘位置（虚拟单位，直接定位到帧的上下左右边缘）
TOP: Vect3 = FRAME_Y_RADIUS * UP    # 帧顶部（Y轴最大位置）
BOTTOM: Vect3 = FRAME_Y_RADIUS * DOWN  # 帧底部（Y轴最小位置）
LEFT_SIDE: Vect3 = FRAME_X_RADIUS * LEFT  # 帧左侧（X轴最小位置）
RIGHT_SIDE: Vect3 = FRAME_X_RADIUS * RIGHT  # 帧右侧（X轴最大位置）


# ------------------------------ 5. 角度单位常量 ------------------------------
# 数学角度相关常量，统一角度计算单位（避免弧度/角度混淆）
PI: float = np.pi        # 圆周率（≈3.14159）
TAU: float = 2 * PI      # 2π（一周的弧度，≈6.28319）
DEG: float = TAU / 360   # 1度对应的弧度（≈0.01745），用于角度转弧度（如 30*DEG 表示30度）
DEGREES = DEG            # 兼容旧代码的别名（DEGREES 同 DEG）
RADIANS: float = 1       # 弧度单位标识（用于代码可读性，如 1*RADIANS 表示1弧度）


# ------------------------------ 6. 文本样式常量 ------------------------------
# 文本/字体样式标识（用于 Text、Tex 等文本 Mobject 的样式设置）
NORMAL: str = "NORMAL"   # 正常样式
ITALIC: str = "ITALIC"   # 斜体
OBLIQUE: str = "OBLIQUE" # 倾斜体（类似斜体，但字体结构不同）
BOLD: str = "BOLD"       # 粗体


# ------------------------------ 7. 渲染样式常量 ------------------------------
# 矢量对象（VMobject）的默认描边宽度（如矩形、圆形的边框粗细）
DEFAULT_STROKE_WIDTH: float = manim_config.vmobject.default_stroke_width


# ------------------------------ 8. 颜色常量 ------------------------------
# 从全局配置读取预定义颜色（支持用户自定义配置，如修改 blue_c 为特定色值）
# 颜色命名规则：基础色+等级（A-E，E最深，A最浅），如 BLUE_E（深蓝）、BLUE_A（浅蓝）
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
GREEN_SCREEN: ManimColor = manim_config.colors.green_screen  # 绿幕色（用于后期抠图）
ORANGE: ManimColor = manim_config.colors.orange
PURE_RED: ManimColor = manim_config.colors.pure_red        # 纯红（RGB(255,0,0)）
PURE_GREEN: ManimColor = manim_config.colors.pure_green    # 纯绿（RGB(0,255,0)）
PURE_BLUE: ManimColor = manim_config.colors.pure_blue      # 纯蓝（RGB(0,0,255)）

# 所有预定义颜色的列表（用于随机颜色选择等场景）
MANIM_COLORS: List[ManimColor] = list(manim_config.colors.values())

# 颜色缩写：取每个色系的“中间等级”（C级）作为默认缩写，简化使用（如 BLUE 同 BLUE_C）
BLUE: ManimColor = BLUE_C
TEAL: ManimColor = TEAL_C
GREEN: ManimColor = GREEN_C
YELLOW: ManimColor = YELLOW_C
GOLD: ManimColor = GOLD_C
RED: ManimColor = RED_C
MAROON: ManimColor = MAROON_C
PURPLE: ManimColor = PURPLE_C
GREY: ManimColor = GREY_C

# 3Blue1Brown 风格配色表（经典配色，用于还原 3B1B 视频风格）
COLORMAP_3B1B: List[ManimColor] = [BLUE_E, GREEN, YELLOW, RED]


# ------------------------------ 9. Mobject 默认颜色常量 ------------------------------
# 通用 Mobject 的默认颜色（文本、线条等），优先级：用户设置 > 此常量 > 硬编码
DEFAULT_MOBJECT_COLOR: ManimColor = manim_config.mobject.default_mobject_color or WHITE  # 默认颜色（默认白色）
DEFAULT_LIGHT_COLOR: ManimColor = manim_config.mobject.default_light_color or GREY_B    # 浅色元素默认色（默认浅灰B）

# 矢量对象（VMobject）的默认描边和填充颜色
DEFAULT_VMOBJECT_STROKE_COLOR: ManimColor = manim_config.vmobject.default_stroke_color or GREY_A  # 默认描边色（深灰A）
DEFAULT_VMOBJECT_FILL_COLOR: ManimColor = manim_config.vmobject.default_fill_color or GREY_C    # 默认填充色（中灰C）