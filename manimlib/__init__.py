# ManimGL 库的核心 __init__.py 文件，作用是**统一导出库内关键模块、类和常量**，
# 简化外部调用（用户无需深入库目录结构，直接从 manimlib 导入即可），同时定义库版本。

# ------------------------------ 1. 定义库版本 ------------------------------
import pkg_resources
# 通过 pkg_resources 获取 manimgl 包的版本号（从安装的包元数据中读取）
__version__ = pkg_resources.get_distribution("manimgl").version


# ------------------------------ 2. 类型检查优化（TYPE_CHECKING） ------------------------------
from typing import TYPE_CHECKING
# TYPE_CHECKING 为 True 时（仅在静态类型检查阶段，如 mypy），导入类型提示模块，
# 避免运行时导入不必要的类型文件，减少性能开销
if TYPE_CHECKING:
    from manimlib.typing import *  # 导入所有自定义类型提示（如 Vect3, Mobject, Animation 等的类型别名）


# ------------------------------ 3. 导出核心常量 ------------------------------
from manimlib.constants import *
# 导出常用常量，如颜色（RED, BLUE, WHITE）、方向向量（LEFT, RIGHT, UP, DOWN）、
# 数学常量（PI, TAU=2π）、缓冲距离（SMALL_BUFF, MED_BUFF）等，用户可直接使用


# ------------------------------ 4. 导出窗口相关类 ------------------------------
from manimlib.window import *
# 导出窗口管理类（如 PygletWindow），负责动画的交互式渲染、窗口事件（鼠标/键盘）处理等，
# 通常无需用户直接实例化，由 Scene 类内部管理


# ------------------------------ 5. 导出所有动画类 ------------------------------
# 按动画功能分类导出，覆盖创建、变换、移动、旋转、淡入淡出等所有动画效果
from manimlib.animation.animation import *          # 基础动画类（Animation）及核心接口
from manimlib.animation.composition import *       # 组合动画（如 Succession, Parallel, LaggedStart）
from manimlib.animation.creation import *          # 创建类动画（如 Create, Write, ShowCreation）
from manimlib.animation.fading import *            # 淡入淡出类动画（如 FadeIn, FadeOut, FadeTransform）
from manimlib.animation.growing import *           # 生长类动画（如 GrowFromCenter, GrowFromEdge）
from manimlib.animation.indication import *        # 强调类动画（如 Flash, Pulse, Indicate）
from manimlib.animation.movement import *          # 移动类动画（如 MoveToTarget, MoveAlongPath, Shift）
from manimlib.animation.numbers import *           # 数字变化类动画（如 ChangeDecimalToValue, CountInFrom）
from manimlib.animation.rotation import *          # 旋转类动画（如 Rotate, RotateAround）
from manimlib.animation.specialized import *       # 特殊动画（如 ApplyWave, VFadeIn, ShowPassingFlash）
from manimlib.animation.transform import *         # 变换类动画（如 Transform, ReplacementTransform, Scale）
from manimlib.animation.transform_matching_parts import *  # 匹配部分变换（如 TransformMatchingShapes）
from manimlib.animation.update import *            # 更新类动画（如 UpdateFromFunc, Wait）


# ------------------------------ 6. 导出相机相关类 ------------------------------
from manimlib.camera.camera import *
# 导出相机类（如 Camera, PerspectiveCamera），负责场景的渲染、视角控制（缩放、平移、旋转）、
# 帧捕获等，3D场景会使用其子类（如 ThreeDCamera）


# ------------------------------ 7. 导出所有可渲染对象（Mobject） ------------------------------
# 按功能分类导出，覆盖几何图形、文本、坐标系、3D对象等所有可在场景中显示的元素
from manimlib.mobject.boolean_ops import *         # 布尔运算对象（如 Union, Intersection, Difference）
from manimlib.mobject.changing import *            # 动态变化对象（如 Changing, UpdatedFromFunc）
from manimlib.mobject.coordinate_systems import *  # 坐标系（如 Axes, NumberPlane, ComplexPlane）
from manimlib.mobject.frame import *               # 帧对象（如 ScreenRectangle, PictureInPictureFrame）
from manimlib.mobject.functions import *           # 函数图像（如 FunctionGraph, ParametricFunction）
from manimlib.mobject.geometry import *            # 基础几何图形（如 Circle, Square, Rectangle, Line, Arrow）
from manimlib.mobject.interactive import *         # 交互式对象（如 InteractiveSquare, DraggablePoint）
from manimlib.mobject.matrix import *              # 矩阵对象（如 Matrix, IntegerMatrix, MobjectMatrix）
from manimlib.mobject.mobject import *             # 基础渲染对象类（Mobject）及核心方法
from manimlib.mobject.mobject_update_utils import *# Mobject 更新工具（如 add_updater, update_function）
from manimlib.mobject.number_line import *         # 数轴对象（如 NumberLine, UnitNumberLine）
from manimlib.mobject.numbers import *             # 数字对象（如 DecimalNumber, Integer, SingleStringMathTex）
from manimlib.mobject.probability import *         # 概率相关图形（如 BarChart, PieChart, Histogram）
from manimlib.mobject.shape_matchers import *      # 形状匹配对象（如 SurroundingRectangle, BackgroundRectangle）
from manimlib.mobject.svg.brace import *           # 括号对象（如 Brace, BraceLabel，用于标注长度/高度）
from manimlib.mobject.svg.drawings import *         # SVG 绘图对象（如 Logo, Code, SVGPathMobject）
from manimlib.mobject.svg.string_mobject import *   # 字符串SVG对象（如 StringMobject，支持字符级动画）
from manimlib.mobject.svg.svg_mobject import *     # 基础SVG对象（如 SVGMobject，加载外部SVG文件）
from manimlib.mobject.svg.special_tex import *     # 特殊TeX对象（如 Tex, MathTex，支持LaTeX公式）
from manimlib.mobject.svg.tex_mobject import *     # TeX相关对象（如 TexSymbol, TexMobject）
from manimlib.mobject.svg.text_mobject import *     # 文本对象（如 Text, MarkupText，支持富文本）
from manimlib.mobject.three_dimensions import *     # 3D对象（如 Cube, Sphere, Cylinder, Cone, Surface）
from manimlib.mobject.types.dot_cloud import *      # 点云对象（如 DotCloud，由多个Dot组成）
from manimlib.mobject.types.image_mobject import *  # 图像对象（如 ImageMobject，加载外部图片）
from manimlib.mobject.types.point_cloud_mobject import *  # 点云渲染对象（如 PointCloudMobject）
from manimlib.mobject.types.surface import *       # 表面对象（如 ParametricSurface, SurfaceFromFunction）
from manimlib.mobject.types.vectorized_mobject import *  # 矢量对象（如 VMobject，支持矢量渲染的基础类）
from manimlib.mobject.value_tracker import *       # 值跟踪器（如 ValueTracker, IntegerTracker，驱动动画参数）
from manimlib.mobject.vector_field import *        # 向量场对象（如 VectorField, StreamLines，用于数学可视化）


# ------------------------------ 8. 导出场景相关类 ------------------------------
from manimlib.scene.interactive_scene import *     # 交互式场景（如 InteractiveScene，支持实时交互）
from manimlib.scene.scene import *                 # 基础场景类（Scene）及3D场景（ThreeDScene），
                                                   # 所有用户自定义动画场景需继承此类


# ------------------------------ 9. 导出工具函数 ------------------------------
# 按功能分类导出，覆盖贝塞尔曲线、颜色处理、文件操作、数学计算等辅助功能
from manimlib.utils.bezier import *                # 贝塞尔曲线工具（如 bezier, quadratic_bezier_points）
from manimlib.utils.cache import *                 # 缓存工具（如 Cache，优化重复计算）
from manimlib.utils.color import *                 # 颜色处理工具（如 color_to_rgba, interpolate_color）
from manimlib.utils.dict_ops import *              # 字典操作工具（如 merge_dicts, update_dict_recursively）
from manimlib.utils.debug import *                 # 调试工具（如 debug_print, show_frame）
from manimlib.utils.directories import *           # 目录操作工具（如 create_dir, get_manim_dir）
from manimlib.utils.file_ops import *              # 文件操作工具（如 file_to_base64, get_file_extension）
from manimlib.utils.images import *                # 图像处理工具（如 get_image_dimensions, crop_image）
from manimlib.utils.iterables import *             # 可迭代对象工具（如 adjacent_pairs, flatten_list）
from manimlib.utils.paths import *                 # 路径工具（如 straight_path, arc_path, bezier_path）
from manimlib.utils.rate_functions import *        # 速度曲线函数（如 linear, ease_in, ease_out, ease_in_out）
from manimlib.utils.simple_functions import *      # 简单数学工具（如 sign, clamp, fdiv, normalize）
from manimlib.utils.shaders import *               # 着色器工具（如 ShaderProgram，自定义GPU渲染效果）
from manimlib.utils.sounds import *                # 音效工具（如 play_sound, add_sound_to_video）
from manimlib.utils.space_ops import *             # 空间计算工具（如 get_norm, dot, cross, angle_between_vectors）
from manimlib.utils.tex import *                   # TeX工具（如 tex_to_svg, get_tex_template，处理LaTeX编译）