# 导入未来版本的注解特性，支持更灵活的类型提示写法（如在类定义前引用类名）
from __future__ import annotations

# 导入Python标准库的XML解析模块，用于处理SVG文件中的XML结构
from xml.etree import ElementTree as ET

# 导入科学计算库numpy，用于处理SVG图形中的数值计算（如坐标、矩阵变换等）
import numpy as np

# 导入第三方SVG处理库svgelements，用于解析和操作SVG的各种元素（如路径、形状、颜色等）
import svgelements as se

# 导入Python标准库的io模块，用于处理内存中的字节流（如将SVG字符串转为可读取的流对象）
import io

# 导入pathlib模块，用于便捷地处理文件路径（如SVG文件的读取、保存路径管理）
from pathlib import Path

from manimlib.constants import RIGHT
from manimlib.constants import TAU
from manimlib.logger import log
from manimlib.mobject.geometry import Circle
from manimlib.mobject.geometry import Line
from manimlib.mobject.geometry import Polygon
from manimlib.mobject.geometry import Polyline
from manimlib.mobject.geometry import Rectangle
from manimlib.mobject.geometry import RoundedRectangle
from manimlib.mobject.types.vectorized_mobject import VMobject
from manimlib.utils.bezier import quadratic_bezier_points_for_arc
from manimlib.utils.images import get_full_vector_image_path
from manimlib.utils.iterables import hash_obj
from manimlib.utils.space_ops import rotation_about_z

# 导入类型检查相关模块
from typing import TYPE_CHECKING

# 仅在类型检查模式下导入类型注解，避免运行时依赖
if TYPE_CHECKING:
    from manimlib.typing import ManimColor, Vect3Array

# 全局缓存：SVG哈希值到图形对象列表的映射，用于复用已解析的SVG图形
SVG_HASH_TO_MOB_MAP: dict[int, list[VMobject]] = {}

# 全局缓存：SVG路径字符串到其点坐标数组的映射，用于复用路径数据
PATH_TO_POINTS: dict[str, Vect3Array] = {}


def _convert_point_to_3d(x: float, y: float) -> np.ndarray:
    """
    将2D坐标点(x, y)转换为Manim使用的3D坐标数组(x, y, 0.0)。

    参数:
        x (float): x坐标
        y (float): y坐标

    返回:
        np.ndarray: 包含3D坐标的NumPy数组
    """
    return np.array([x, y, 0.0])


class SVGMobject(VMobject):
    """
    用于处理SVG文件的图形对象基类。
    它能够解析SVG文件并将其转换为Manim可以渲染的矢量图形对象。
    """
    file_name: str = ""  # SVG文件名（不包含路径），子类应覆盖此属性
    height: float | None = 2.0  # 渲染后的图形高度，None表示不按高度缩放
    width: float | None = None  # 渲染后的图形宽度，None表示不按宽度缩放

    def __init__(
        self,
        file_name: str = "",
        svg_string: str = "",
        should_center: bool = True,
        height: float | None = None,
        width: float | None = None,
        # Style that overrides the original svg
        color: ManimColor = None,
        fill_color: ManimColor = None,
        fill_opacity: float | None = None,
        stroke_width: float | None = 0.0,
        stroke_color: ManimColor = None,
        stroke_opacity: float | None = None,
        # Style that fills only when not specified
        # If None, regarded as default values from svg standard
        svg_default: dict = dict(
            color=None,
            opacity=None,
            fill_color=None,
            fill_opacity=None,
            stroke_width=None,
            stroke_color=None,
            stroke_opacity=None,
        ),
        path_string_config: dict = dict(),
        **kwargs
    ):
        """
        初始化SVGMobject实例。

        参数:
            file_name (str): SVG文件路径。
            svg_string (str): 直接提供的SVG字符串。
            should_center (bool): 是否将图形居中。
            height (float | None): 图形的高度。
            width (float | None): 图形的宽度。
            color (ManimColor): 同时设置填充和描边颜色。
            fill_color (ManimColor): 填充颜色。
            fill_opacity (float | None): 填充透明度。
            stroke_width (float | None): 描边宽度。
            stroke_color (ManimColor): 描边颜色。
            stroke_opacity (float | None): 描边透明度。
            svg_default (dict): SVG元素的默认样式。
            path_string_config (dict): 路径字符串配置。
            **kwargs: 传递给父类VMobject的其他参数。
        """
        # 1. 确定SVG源：优先使用直接提供的字符串，其次是文件路径
        if svg_string != "":
            self.svg_string = svg_string
        elif file_name != "":
            self.svg_string = self.file_name_to_svg_string(file_name)
        elif self.file_name != "":
            self.file_name_to_svg_string(self.file_name)
        else:
            raise Exception("Must specify either a file_name or svg_string SVGMobject")

        # 2. 保存配置参数
        self.svg_default = dict(svg_default)
        self.path_string_config = dict(path_string_config)

        # 3. 初始化父类并处理SVG
        super().__init__(**kwargs)
        self.init_svg_mobject()  # 解析SVG并创建子对象
        self.ensure_positive_orientation()  # 确保路径方向正确

        # Rather than passing style into super().__init__
        # do it after svg has been taken in
        # 4. 设置样式（覆盖SVG原有的样式）
        self.set_style(
            fill_color=color or fill_color,
            fill_opacity=fill_opacity,
            stroke_color=color or stroke_color,
            stroke_width=stroke_width,
            stroke_opacity=stroke_opacity,
        )

        # Initialize position
        # 5. 设置位置和大小
        height = height or self.height
        width = width or self.width

        if should_center:
            self.center()
        if height is not None:
            self.set_height(height)
        if width is not None:
            self.set_width(width)

    def init_svg_mobject(self) -> None:
        """
        初始化SVG图形对象的核心方法。
        它负责解析SVG字符串并创建相应的子对象(submobjects)。
        为了提高性能,它会检查缓存,如果SVG已经被解析过,则直接复用缓存的结果。
        """
        # 1. 计算一个唯一的哈希值，用于缓存
        # hash_obj 是一个自定义函数，它会对传入的 `self.hash_seed` 进行哈希计算
        hash_val = hash_obj(self.hash_seed)
        # 2. 检查缓存
        # SVG_HASH_TO_MOB_MAP 是一个全局字典，用于存储已经解析过的SVG图形
        if hash_val in SVG_HASH_TO_MOB_MAP:
            # 如果缓存中存在，直接从缓存中获取子对象的副本，避免重复解析
            submobs = [sm.copy() for sm in SVG_HASH_TO_MOB_MAP[hash_val]]
        else:
            # 如果缓存中不存在，则调用 `mobjects_from_svg_string` 方法解析SVG字符串
            # 这个方法会将SVG中的路径、形状等转换为Manim的矢量图形对象（VMobject）
            submobs = self.mobjects_from_svg_string(self.svg_string)
            # 将解析结果存入缓存，以便将来复用
            SVG_HASH_TO_MOB_MAP[hash_val] = [sm.copy() for sm in submobs]

        # 3. 添加子对象到当前图形中
        self.add(*submobs)
        # 4. 翻转坐标系
        # SVG的Y轴向下，而Manim的Y轴向上。
        # 这里沿RIGHT方向（即X轴）翻转，实际上是在2D平面内翻转了Y轴，使SVG图形正确显示。
        self.flip(RIGHT)  # Flip y

    @property
    def hash_seed(self) -> tuple:
        # Returns data which can uniquely represent the result of `init_points`.
        # The hashed value of it is stored as a key in `SVG_HASH_TO_MOB_MAP`.
        """
        一个属性,返回一个可以唯一代表当前SVG图形状态的元组。
        这个元组是计算缓存键（哈希值）的基础。

        返回:
            tuple: 一个包含以下元素的元组，任何一个元素的改变都会导致哈希值改变：
                - 类名 (__class__.__name__)
                - SVG默认样式配置 (self.svg_default)
                - 路径字符串配置 (self.path_string_config)
                - SVG原始字符串 (self.svg_string)
        """
        return (
            self.__class__.__name__,
            self.svg_default,
            self.path_string_config,
            self.svg_string
        )

    def mobjects_from_svg_string(self, svg_string: str) -> list[VMobject]:
        """
        将SVG字符串解析为Manim的图形对象列表。
        这是SVGMobject将SVG数据转换为可渲染图形的核心步骤。

        参数:
            svg_string (str): 包含SVG数据的字符串。

        返回:
            list[VMobject]: 从SVG中解析出的图形对象列表。
        """
        # 1. 解析SVG字符串为ElementTree对象
        element_tree = ET.ElementTree(ET.fromstring(svg_string))
        # 2. 修改XML树结构（用于注入全局样式或修改结构）
        new_tree = self.modify_xml_tree(element_tree)

        # New svg based on tree contents
        # 3. 将修改后的ElementTree对象重新写入字节流
        data_stream = io.BytesIO()
        new_tree.write(data_stream)
        data_stream.seek(0)
        # 4. 使用svgelements库解析字节流中的SVG数据
        svg = se.SVG.parse(data_stream)
        data_stream.close()

        # 5. 将解析后的svgelements对象转换为Manim的图形对象
        return self.mobjects_from_svg(svg)

    def file_name_to_svg_string(self, file_name: str) -> str:
        """
        读取SVG文件内容并返回其字符串形式。

        参数:
            file_name (str): SVG文件的名称或路径。

        返回:
            str: SVG文件的内容字符串。
        """
        # get_full_vector_image_path 是一个Manim内部函数，用于查找图片文件的完整路径
        return Path(get_full_vector_image_path(file_name)).read_text()

    def modify_xml_tree(self, element_tree: ET.ElementTree) -> ET.ElementTree:
        """
        修改SVG的XML树结构,主要用于注入配置的全局样式。
        它会创建一个新的SVG结构,将原始内容包裹在带有配置样式的组(g)中。

        参数:
            element_tree (ET.ElementTree): 原始的SVG ElementTree对象。

        返回:
            ET.ElementTree: 修改后的SVG ElementTree对象。
        """
        # 1. 生成包含配置样式的字典（如fill, stroke等）
        config_style_attrs = self.generate_config_style_dict()
        # 2. 定义需要关注的样式属性
        style_keys = (
            "fill",
            "fill-opacity",
            "stroke",
            "stroke-opacity",
            "stroke-width",
            "style"
        )
        # 3. 获取原始SVG的根元素和其样式属性
        root = element_tree.getroot()
        style_attrs = {
            k: v
            for k, v in root.attrib.items()
            if k in style_keys
        }

        # Ignore other attributes in case that svgelements cannot parse them
        # 4. 创建新的XML树结构
        SVG_XMLNS = "{http://www.w3.org/2000/svg}"
        new_root = ET.Element("svg")
        # 5. 创建包裹层以应用样式
        # 第一层组(g)应用从配置生成的样式
        config_style_node = ET.SubElement(new_root, f"{SVG_XMLNS}g", config_style_attrs)
        # 第二层组(g)应用原始SVG根元素的样式
        root_style_node = ET.SubElement(config_style_node, f"{SVG_XMLNS}g", style_attrs)
        # 6. 将原始SVG的所有子元素移动到新的结构中
        root_style_node.extend(root)
        # 7. 返回包含新结构的ElementTree对象
        return ET.ElementTree(new_root)

    def generate_config_style_dict(self) -> dict[str, str]:
        """
        根据 `self.svg_default` 中的配置，生成一个用于注入到SVG XML中的样式属性字典。
        这个方法将Manim的样式参数名（如fill_color）映射为SVG标准的属性名（如fill）。

        返回:
            dict[str, str]: 一个包含SVG样式属性的字典，键是SVG属性名，值是对应的样式值字符串。
        """
        # 1. 定义Manim样式键到SVG属性键的映射关系
        keys_converting_dict = {
            "fill": ("color", "fill_color"),
            "fill-opacity": ("opacity", "fill_opacity"),
            "stroke": ("color", "stroke_color"),
            "stroke-opacity": ("opacity", "stroke_opacity"),
            "stroke-width": ("stroke_width",)
        }
        svg_default_dict = self.svg_default
        result = {}
        # 2. 遍历映射关系，构建SVG样式字典
        for svg_key, style_keys in keys_converting_dict.items():
            for style_key in style_keys:
                # 如果 `self.svg_default` 中对应的值不为None，则添加到结果字典中
                if svg_default_dict[style_key] is None:
                    continue
                result[svg_key] = str(svg_default_dict[style_key])
        return result

    def mobjects_from_svg(self, svg: se.SVG) -> list[VMobject]:
        """
        将一个 `svgelements.SVG` 对象转换为Manim的图形对象列表。
        它会遍历SVG中的所有元素，并根据元素类型（路径、矩形、圆形等）创建对应的Manim对象。

        参数:
            svg (se.SVG): 一个由 `svgelements` 库解析后的SVG对象。

        返回:
            list[VMobject]: 一个包含从SVG元素转换而来的Manim图形对象的列表。
        """
        result = []
        # 1. 遍历SVG中的所有元素
        for shape in svg.elements():
            # 2. 根据元素类型，创建对应的Manim对象
            if isinstance(shape, (se.Group, se.Use)):
                 # 跳过组和引用元素，它们的子元素会被单独处理
                continue
            elif isinstance(shape, se.Path):
                mob = self.path_to_mobject(shape)
            elif isinstance(shape, se.SimpleLine):
                mob = self.line_to_mobject(shape)
            elif isinstance(shape, se.Rect):
                mob = self.rect_to_mobject(shape)
            elif isinstance(shape, (se.Circle, se.Ellipse)):
                mob = self.ellipse_to_mobject(shape)
            elif isinstance(shape, se.Polygon):
                mob = self.polygon_to_mobject(shape)
            elif isinstance(shape, se.Polyline):
                mob = self.polyline_to_mobject(shape)
            # elif isinstance(shape, se.Text):
            #     mob = self.text_to_mobject(shape) # 文本处理通常单独实现
            elif type(shape) == se.SVGElement:
                # 跳过未知的基础SVG元素
                continue
            else:
                # 打印警告并跳过不支持的元素类型
                log.warning("Unsupported element type: %s", type(shape))
                continue
            # 3. 处理创建的Manim对象
            # 如果对象没有任何点数据（例如，一个零长度的线），则跳过
            if not mob.has_points():
                continue
            # 如果是图形对象，应用其自身的样式（颜色、透明度等）
            if isinstance(shape, se.GraphicObject):
                self.apply_style_to_mobject(mob, shape)
            # 如果元素有变换属性，应用这些变换到Manim对象上
            if isinstance(shape, se.Transformable) and shape.apply:
                self.handle_transform(mob, shape.transform)
            # 4. 将处理好的对象添加到结果列表中 
            result.append(mob)
        return result

    @staticmethod
    def handle_transform(mob: VMobject, matrix: se.Matrix) -> VMobject:
        """
        将SVG的2D变换矩阵应用到Manim的图形对象上。

        参数:
            mob (VMobject): 要应用变换的Manim图形对象。
            matrix (se.Matrix): 来自 `svgelements` 的变换矩阵对象。

        返回:
            VMobject: 应用了变换后的图形对象。
        """
        # 1. 提取旋转、缩放、倾斜等线性变换部分
        mat = np.array([
            [matrix.a, matrix.c],
            [matrix.b, matrix.d]
        ])
        # 2. 提取平移变换部分，并转换为Manim的3D坐标
        vec = np.array([matrix.e, matrix.f, 0.0])
        # 3. 应用线性变换矩阵
        mob.apply_matrix(mat)
        # 4. 应用平移变换
        mob.shift(vec)
        return mob

    @staticmethod
    def apply_style_to_mobject(
        mob: VMobject,
        shape: se.GraphicObject
    ) -> VMobject:
        """
        将SVG元素的样式（颜色、透明度、描边宽度等）应用到Manim的图形对象上。

        参数:
            mob (VMobject): 要应用样式的Manim图形对象。
            shape (se.GraphicObject): 包含样式信息的 `svgelements` 图形对象。

        返回:
            VMobject: 应用了样式后的图形对象。
        """
        mob.set_style(
            stroke_width=shape.stroke_width,
            stroke_color=shape.stroke.hexrgb,
            stroke_opacity=shape.stroke.opacity,
            fill_color=shape.fill.hexrgb,
            fill_opacity=shape.fill.opacity
        )
        return mob

    def path_to_mobject(self, path: se.Path) -> VMobjectFromSVGPath:
        """
        将SVG的路径元素（Path）转换为一个专门处理SVG路径的Manim图形对象。

        参数:
            path (se.Path): 来自 `svgelements` 的路径对象。

        返回:
            VMobjectFromSVGPath: 一个能够渲染SVG路径的Manim图形对象。
        """
        return VMobjectFromSVGPath(path, **self.path_string_config)

    def line_to_mobject(self, line: se.SimpleLine) -> Line:
        """
        将SVG的直线元素转换为Manim的Line对象。

        参数:
            line (se.SimpleLine): 来自svgelements的直线对象。

        返回:
            Line: 一个Manim的直线对象。
        """
        return Line(
            start=_convert_point_to_3d(line.x1, line.y1),  # 转换起点为3D坐标
            end=_convert_point_to_3d(line.x2, line.y2)     # 转换终点为3D坐标
        )

    def rect_to_mobject(self, rect: se.Rect) -> Rectangle:
        """
        将SVG的矩形元素转换为Manim的Rectangle或RoundedRectangle对象。

        参数:
            rect (se.Rect): 来自svgelements的矩形对象。

        返回:
            Rectangle | RoundedRectangle: 一个Manim的矩形对象。
        """
        # 如果矩形没有圆角，则创建一个普通的Rectangle
        if rect.rx == 0 or rect.ry == 0:
            mob = Rectangle(
                width=rect.width,
                height=rect.height,
            )
        else:
            # 如果有圆角，则创建一个RoundedRectangle
            # 先创建一个基于x方向圆角半径的矩形
            mob = RoundedRectangle(
                width=rect.width,
                height=rect.height * rect.rx / rect.ry,  # 临时高度
                corner_radius=rect.rx
            )
            # 然后将其垂直拉伸到SVG矩形的实际高度
            mob.stretch_to_fit_height(rect.height)
        # 将矩形移动到SVG中指定的位置（矩形的中心点）
        mob.shift(_convert_point_to_3d(
            rect.x + rect.width / 2,
            rect.y + rect.height / 2
        ))
        return mob

    def ellipse_to_mobject(self, ellipse: se.Circle | se.Ellipse) -> Circle:
        """
        将SVG的圆形或椭圆形元素转换为Manim的Circle对象（通过拉伸实现椭圆效果）。

        参数:
            ellipse (se.Circle | se.Ellipse): 来自svgelements的圆形或椭圆形对象。

        返回:
            Circle: 一个Manim的圆形/椭圆形对象。
        """
        # 以x轴半径为基础创建一个圆形
        mob = Circle(radius=ellipse.rx)
        # 将圆形垂直拉伸，使其y轴半径等于SVG椭圆的ry
        mob.stretch_to_fit_height(2 * ellipse.ry)
        # 将椭圆移动到SVG中指定的中心位置
        mob.shift(_convert_point_to_3d(
            ellipse.cx, ellipse.cy
        ))
        return mob

    def polygon_to_mobject(self, polygon: se.Polygon) -> Polygon:
        """
        将SVG的多边形元素转换为Manim的Polygon对象。

        参数:
            polygon (se.Polygon): 来自svgelements的多边形对象。

        返回:
            Polygon: 一个Manim的多边形对象。
        """
        # 1. 遍历多边形的所有顶点，并将其转换为Manim的3D坐标
        points = [
            _convert_point_to_3d(*point)
            for point in polygon
        ]
        # 2. 使用转换后的顶点列表创建一个Manim的Polygon对象
        return Polygon(*points)

    def polyline_to_mobject(self, polyline: se.Polyline) -> Polyline:
        """
        将SVG的折线元素转换为Manim的Polyline对象。

        参数:
            polyline (se.Polyline): 来自svgelements的折线对象。

        返回:
            Polyline: 一个Manim的折线对象。
        """
        # 1. 遍历折线的所有顶点，并将其转换为Manim的3D坐标
        points = [
            _convert_point_to_3d(*point)
            for point in polyline
        ]
        # 2. 使用转换后的顶点列表创建一个Manim的Polyline对象
        return Polyline(*points)

    def text_to_mobject(self, text: se.Text):
        """
        将SVG的文本元素转换为Manim对象。此方法目前未实现。
    
        SVG中的文本处理比较复杂，通常不会直接转换为简单的图形对象，
        而是需要使用Manim的Text或Tex等专门处理文本的类。
        """
        pass


class VMobjectFromSVGPath(VMobject):
    """
    一个专门用于处理SVG路径（Path）元素的Manim图形对象。
    它直接操作SVG路径数据，能够精确地复现SVG中的复杂路径。
    """
    def __init__(
        self,
        path_obj: se.Path,
        **kwargs
    ):
        # caches (transform.inverse(), rot, shift)
        # 初始化一个缓存，用于存储路径的逆变换、旋转和位移信息，以提高性能
        self.transform_cache: tuple[se.Matrix, np.ndarray, np.ndarray] | None = None

        # 保存原始的svgelements Path对象，以备后续使用
        self.path_obj = path_obj
        # 调用父类VMobject的初始化方法
        super().__init__(**kwargs)

    def init_points(self) -> None:
        # After a given svg_path has been converted into points, the result
        # will be saved so that future calls for the same pathdon't need to
        # retrace the same computation.
        path_string = self.path_obj.d()
        if path_string not in PATH_TO_POINTS:
            self.handle_commands()
            # Save for future use
            PATH_TO_POINTS[path_string] = self.get_points().copy()
        else:
            points = PATH_TO_POINTS[path_string]
            self.set_points(points)

    def handle_commands(self) -> None:
        segment_class_to_func_map = {
            se.Move: (self.start_new_path, ("end",)),
            se.Close: (self.close_path, ()),
            se.Line: (lambda p: self.add_line_to(p, allow_null_line=False), ("end",)),
            se.QuadraticBezier: (lambda c, e: self.add_quadratic_bezier_curve_to(c, e, allow_null_curve=False), ("control", "end")),
            se.CubicBezier: (self.add_cubic_bezier_curve_to, ("control1", "control2", "end"))
        }
        for segment in self.path_obj:
            segment_class = segment.__class__
            if segment_class is se.Arc:
                self.handle_arc(segment)
            else:
                func, attr_names = segment_class_to_func_map[segment_class]
                points = [
                    _convert_point_to_3d(*segment.__getattribute__(attr_name))
                    for attr_name in attr_names
                ]
                func(*points)

        # Get rid of the side effect of trailing "Z M" commands.
        if self.has_new_path_started():
            self.resize_points(self.get_num_points() - 2)

    def handle_arc(self, arc: se.Arc) -> None:
        if self.transform_cache is not None:
            transform, rot, shift = self.transform_cache
        else:
            # The transform obtained in this way considers the combined effect
            # of all parent group transforms in the SVG.
            # Therefore, the arc can be transformed inversely using this transform
            # to correctly compute the arc path before transforming it back.
            transform = se.Matrix(self.path_obj.values.get('transform', ''))
            rot = np.array([
                [transform.a, transform.c],
                [transform.b, transform.d]
            ])
            shift = np.array([transform.e, transform.f, 0])
            transform.inverse()
            self.transform_cache = (transform, rot, shift)

        # Apply inverse transformation to the arc so that its path can be correctly computed
        arc *= transform

        # The value of n_components is chosen based on the implementation of VMobject.arc_to
        n_components = int(np.ceil(8 * abs(arc.sweep) / TAU))

        # Obtain the required angular segments on the unit circle
        arc_points = quadratic_bezier_points_for_arc(arc.sweep, n_components)
        arc_points @= np.array(rotation_about_z(arc.get_start_t())).T

        # Transform to an ellipse, considering rotation and translating the ellipse center
        arc_points[:, 0] *= arc.rx
        arc_points[:, 1] *= arc.ry
        arc_points @= np.array(rotation_about_z(arc.get_rotation().as_radians)).T
        arc_points += [*arc.center, 0]

        # Transform back
        arc_points[:, :2] @= rot.T
        arc_points += shift

        self.append_points(arc_points[1:])
