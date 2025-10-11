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
        element_tree = ET.ElementTree(ET.fromstring(svg_string))
        new_tree = self.modify_xml_tree(element_tree)

        # New svg based on tree contents
        data_stream = io.BytesIO()
        new_tree.write(data_stream)
        data_stream.seek(0)
        svg = se.SVG.parse(data_stream)
        data_stream.close()

        return self.mobjects_from_svg(svg)

    def file_name_to_svg_string(self, file_name: str) -> str:
        return Path(get_full_vector_image_path(file_name)).read_text()

    def modify_xml_tree(self, element_tree: ET.ElementTree) -> ET.ElementTree:
        config_style_attrs = self.generate_config_style_dict()
        style_keys = (
            "fill",
            "fill-opacity",
            "stroke",
            "stroke-opacity",
            "stroke-width",
            "style"
        )
        root = element_tree.getroot()
        style_attrs = {
            k: v
            for k, v in root.attrib.items()
            if k in style_keys
        }

        # Ignore other attributes in case that svgelements cannot parse them
        SVG_XMLNS = "{http://www.w3.org/2000/svg}"
        new_root = ET.Element("svg")
        config_style_node = ET.SubElement(new_root, f"{SVG_XMLNS}g", config_style_attrs)
        root_style_node = ET.SubElement(config_style_node, f"{SVG_XMLNS}g", style_attrs)
        root_style_node.extend(root)
        return ET.ElementTree(new_root)

    def generate_config_style_dict(self) -> dict[str, str]:
        keys_converting_dict = {
            "fill": ("color", "fill_color"),
            "fill-opacity": ("opacity", "fill_opacity"),
            "stroke": ("color", "stroke_color"),
            "stroke-opacity": ("opacity", "stroke_opacity"),
            "stroke-width": ("stroke_width",)
        }
        svg_default_dict = self.svg_default
        result = {}
        for svg_key, style_keys in keys_converting_dict.items():
            for style_key in style_keys:
                if svg_default_dict[style_key] is None:
                    continue
                result[svg_key] = str(svg_default_dict[style_key])
        return result

    def mobjects_from_svg(self, svg: se.SVG) -> list[VMobject]:
        result = []
        for shape in svg.elements():
            if isinstance(shape, (se.Group, se.Use)):
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
            #     mob = self.text_to_mobject(shape)
            elif type(shape) == se.SVGElement:
                continue
            else:
                log.warning("Unsupported element type: %s", type(shape))
                continue
            if not mob.has_points():
                continue
            if isinstance(shape, se.GraphicObject):
                self.apply_style_to_mobject(mob, shape)
            if isinstance(shape, se.Transformable) and shape.apply:
                self.handle_transform(mob, shape.transform)
            result.append(mob)
        return result

    @staticmethod
    def handle_transform(mob: VMobject, matrix: se.Matrix) -> VMobject:
        mat = np.array([
            [matrix.a, matrix.c],
            [matrix.b, matrix.d]
        ])
        vec = np.array([matrix.e, matrix.f, 0.0])
        mob.apply_matrix(mat)
        mob.shift(vec)
        return mob

    @staticmethod
    def apply_style_to_mobject(
        mob: VMobject,
        shape: se.GraphicObject
    ) -> VMobject:
        mob.set_style(
            stroke_width=shape.stroke_width,
            stroke_color=shape.stroke.hexrgb,
            stroke_opacity=shape.stroke.opacity,
            fill_color=shape.fill.hexrgb,
            fill_opacity=shape.fill.opacity
        )
        return mob

    def path_to_mobject(self, path: se.Path) -> VMobjectFromSVGPath:
        return VMobjectFromSVGPath(path, **self.path_string_config)

    def line_to_mobject(self, line: se.SimpleLine) -> Line:
        return Line(
            start=_convert_point_to_3d(line.x1, line.y1),
            end=_convert_point_to_3d(line.x2, line.y2)
        )

    def rect_to_mobject(self, rect: se.Rect) -> Rectangle:
        if rect.rx == 0 or rect.ry == 0:
            mob = Rectangle(
                width=rect.width,
                height=rect.height,
            )
        else:
            mob = RoundedRectangle(
                width=rect.width,
                height=rect.height * rect.rx / rect.ry,
                corner_radius=rect.rx
            )
            mob.stretch_to_fit_height(rect.height)
        mob.shift(_convert_point_to_3d(
            rect.x + rect.width / 2,
            rect.y + rect.height / 2
        ))
        return mob

    def ellipse_to_mobject(self, ellipse: se.Circle | se.Ellipse) -> Circle:
        mob = Circle(radius=ellipse.rx)
        mob.stretch_to_fit_height(2 * ellipse.ry)
        mob.shift(_convert_point_to_3d(
            ellipse.cx, ellipse.cy
        ))
        return mob

    def polygon_to_mobject(self, polygon: se.Polygon) -> Polygon:
        points = [
            _convert_point_to_3d(*point)
            for point in polygon
        ]
        return Polygon(*points)

    def polyline_to_mobject(self, polyline: se.Polyline) -> Polyline:
        points = [
            _convert_point_to_3d(*point)
            for point in polyline
        ]
        return Polyline(*points)

    def text_to_mobject(self, text: se.Text):
        pass


class VMobjectFromSVGPath(VMobject):
    def __init__(
        self,
        path_obj: se.Path,
        **kwargs
    ):
        # caches (transform.inverse(), rot, shift)
        self.transform_cache: tuple[se.Matrix, np.ndarray, np.ndarray] | None = None

        self.path_obj = path_obj
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
