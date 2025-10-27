from __future__ import annotations

from functools import wraps

import numpy as np

from manimlib.constants import GREY_A, GREY_C, GREY_E
from manimlib.constants import DEFAULT_VMOBJECT_FILL_COLOR, DEFAULT_VMOBJECT_STROKE_COLOR
from manimlib.constants import BLACK
from manimlib.constants import DEFAULT_STROKE_WIDTH
from manimlib.constants import DEG
from manimlib.constants import ORIGIN, OUT
from manimlib.constants import PI
from manimlib.constants import TAU
from manimlib.mobject.mobject import Mobject
from manimlib.mobject.mobject import Group
from manimlib.mobject.mobject import Point
from manimlib.utils.bezier import bezier
from manimlib.utils.bezier import get_quadratic_approximation_of_cubic
from manimlib.utils.bezier import approx_smooth_quadratic_bezier_handles
from manimlib.utils.bezier import smooth_quadratic_path
from manimlib.utils.bezier import interpolate
from manimlib.utils.bezier import integer_interpolate
from manimlib.utils.bezier import inverse_interpolate
from manimlib.utils.bezier import find_intersection
from manimlib.utils.bezier import outer_interpolate
from manimlib.utils.bezier import partial_quadratic_bezier_points
from manimlib.utils.bezier import quadratic_bezier_points_for_arc
from manimlib.utils.color import color_gradient
from manimlib.utils.color import rgb_to_hex
from manimlib.utils.iterables import make_even
from manimlib.utils.iterables import resize_array
from manimlib.utils.iterables import resize_with_interpolation
from manimlib.utils.iterables import resize_preserving_order
from manimlib.utils.space_ops import angle_between_vectors
from manimlib.utils.space_ops import cross2d
from manimlib.utils.space_ops import earclip_triangulation
from manimlib.utils.space_ops import get_norm
from manimlib.utils.space_ops import get_unit_normal
from manimlib.utils.space_ops import line_intersects_path
from manimlib.utils.space_ops import midpoint
from manimlib.utils.space_ops import rotation_between_vectors
from manimlib.utils.space_ops import rotation_matrix_transpose
from manimlib.utils.space_ops import poly_line_length
from manimlib.utils.space_ops import z_to_vector
from manimlib.shader_wrapper import VShaderWrapper

from typing import TYPE_CHECKING
from typing import Generic, TypeVar, Iterable
# 定义SubVmobjectType类型变量，约束为VMobject的子类
SubVmobjectType = TypeVar('SubVmobjectType', bound='VMobject')

if TYPE_CHECKING:
    from typing import Callable, Tuple, Any, Optional
    from manimlib.typing import ManimColor, Vect3, Vect4, Vect3Array, Self
    from moderngl.context import Context


class VMobject(Mobject):
    """
    向量图形对象基类，继承自Mobject，用于表示可渲染的向量图形
    
    包含点数据、颜色、描边、填充等属性，以及处理贝塞尔曲线、路径操作的方法
    """
    # 数据类型定义，包含图形渲染所需的各种属性
    data_dtype: np.dtype = np.dtype([
        ('point', np.float32, (3,)),               # 点坐标
        ('stroke_rgba', np.float32, (4,)),         # 描边颜色(RGBA)
        ('stroke_width', np.float32, (1,)),        # 描边宽度
        ('joint_angle', np.float32, (1,)),         # 连接角度
        ('fill_rgba', np.float32, (4,)),           # 填充颜色(RGBA)
        ('base_normal', np.float32, (3,)),         # 基础法向量
        ('fill_border_width', np.float32, (1,)),   # 填充边框宽度
    ])
    # 函数处理后锚点的缩放因子
    pre_function_handle_to_anchor_scale_factor: float = 0.01
    # 应用函数后是否自动平滑处理
    make_smooth_after_applying_functions: bool = False
    # 点相等的容差范围
    tolerance_for_point_equality: float = 1e-8
    # 连接类型映射（用于着色器）
    joint_type_map: dict = {
        "no_joint": 0,
        "auto": 1,
        "bevel": 2,
        "miter": 3,
    }

    def __init__(
        self,
        color: ManimColor = None,  # 如果设置，将覆盖描边和填充颜色
        fill_color: ManimColor = None,
        fill_opacity: float | Iterable[float] | None = 0.0,
        stroke_color: ManimColor = None,
        stroke_opacity: float | Iterable[float] | None = 1.0,
        stroke_width: float | Iterable[float] | None = DEFAULT_STROKE_WIDTH,
        stroke_behind: bool = False,
        background_image_file: str | None = None,
        long_lines: bool = False,
        # 连接类型，可选"no_joint", "bevel", "miter"
        joint_type: str = "auto",
        flat_stroke: bool = False,
        scale_stroke_with_zoom: bool = False,
        use_simple_quadratic_approx: bool = False,
        # 以像素宽度为单位
        anti_alias_width: float = 1.5,
        fill_border_width: float = 0.0,
        **kwargs
    ):
        # 初始化填充和描边颜色，若未指定则使用默认值
        self.fill_color = fill_color or color or DEFAULT_VMOBJECT_FILL_COLOR
        self.fill_opacity = fill_opacity
        self.stroke_color = stroke_color or color or DEFAULT_VMOBJECT_STROKE_COLOR
        self.stroke_opacity = stroke_opacity
        self.stroke_width = stroke_width
        self.stroke_behind = stroke_behind  # 描边是否在填充后面
        self.background_image_file = background_image_file  # 背景图片文件
        self.long_lines = long_lines  # 是否使用长线条
        self.joint_type = joint_type  # 连接类型
        self.flat_stroke = flat_stroke  # 是否使用扁平描边
        self.scale_stroke_with_zoom = scale_stroke_with_zoom  # 描边是否随缩放变化
        self.use_simple_quadratic_approx = use_simple_quadratic_approx  # 是否使用简单二次近似
        self.anti_alias_width = anti_alias_width  # 抗锯齿宽度
        self.fill_border_width = fill_border_width  # 填充边框宽度

        self.needs_new_joint_angles = True  # 是否需要更新连接角度
        self.needs_new_unit_normal = True  # 是否需要更新单位法向量
        self.subpath_end_indices = None  # 子路径结束索引
        self.outer_vert_indices = np.zeros(0, dtype=int)  # 外部顶点索引

        super().__init__(** kwargs)

    def get_group_class(self):
        """返回对应的组类，默认为VGroup"""
        return VGroup

    def init_uniforms(self):
        """初始化着色器 uniforms"""
        super().init_uniforms()
        self.uniforms.update(
            anti_alias_width=self.anti_alias_width,
            joint_type=self.joint_type_map[self.joint_type],
            flat_stroke=float(self.flat_stroke),
            scale_stroke_with_zoom=float(self.scale_stroke_with_zoom)
        )

    def add(self, *vmobjects: VMobject) -> Self:
        """添加子对象，确保所有子对象都是VMobject类型"""
        if not all((isinstance(m, VMobject) for m in vmobjects)):
            raise Exception("All submobjects must be of type VMobject")
        return super().add(*vmobjects)

    # 颜色相关方法
    def init_colors(self):
        """初始化颜色属性"""
        self.set_stroke(
            color=self.stroke_color,
            width=self.stroke_width,
            opacity=self.stroke_opacity,
            behind=self.stroke_behind,
        )
        self.set_fill(
            color=self.fill_color,
            opacity=self.fill_opacity,
            border_width=self.fill_border_width,
        )
        self.set_shading(*self.shading)
        self.set_flat_stroke(self.flat_stroke)
        self.color = self.get_color()
        return self

    def set_fill(
        self,
        color: ManimColor | Iterable[ManimColor] = None,
        opacity: float | Iterable[float] | None = None,
        border_width: float | None = None,
        recurse: bool = True
    ) -> Self:
        """
        设置填充属性
        
        Args:
            color: 填充颜色
            opacity: 填充透明度
            border_width: 填充边框宽度
            recurse: 是否递归应用到子对象
        """
        self.set_rgba_array_by_color(color, opacity, 'fill_rgba', recurse)
        if border_width is not None:
            self.border_width = border_width
            for mob in self.get_family(recurse):
                data = mob.data if mob.has_points() > 0 else mob._data_defaults
                data["fill_border_width"] = border_width
        return self

    def set_stroke(
        self,
        color: ManimColor | Iterable[ManimColor] = None,
        width: float | Iterable[float] | None = None,
        opacity: float | Iterable[float] | None = None,
        behind: bool | None = None,
        flat: bool | None = None,
        recurse: bool = True
    ) -> Self:
        """
        设置描边属性
        
        Args:
            color: 描边颜色
            width: 描边宽度
            opacity: 描边透明度
            behind: 描边是否在填充后面
            flat: 是否使用扁平描边
            recurse: 是否递归应用到子对象
        """
        self.set_rgba_array_by_color(color, opacity, 'stroke_rgba', recurse)

        if width is not None:
            for mob in self.get_family(recurse):
                data = mob.data if mob.get_num_points() > 0 else mob._data_defaults
                if isinstance(width, (float, int, np.floating)):
                    data['stroke_width'][:, 0] = width
                else:
                    data['stroke_width'][:, 0] = resize_with_interpolation(
                        np.array(width), len(data)
                    ).flatten()

        if behind is not None:
            for mob in self.get_family(recurse):
                if mob.stroke_behind != behind:
                    mob.stroke_behind = behind
                    mob.refresh_shader_wrapper_id()

        if flat is not None:
            self.set_flat_stroke(flat)

        return self

    def set_backstroke(
        self,
        color: ManimColor | Iterable[ManimColor] = BLACK,
        width: float | Iterable[float] = 3,
    ) -> Self:
        """设置背景描边（在所有内容后面）"""
        self.set_stroke(color, width, behind=True)
        return self

    @Mobject.affects_family_data
    def set_style(
        self,
        fill_color: ManimColor | Iterable[ManimColor] | None = None,
        fill_opacity: float | Iterable[float] | None = None,
        fill_rgba: Vect4 | None = None,
        fill_border_width: float | None = None,
        stroke_color: ManimColor | Iterable[ManimColor] | None = None,
        stroke_opacity: float | Iterable[float] | None = None,
        stroke_rgba: Vect4 | None = None,
        stroke_width: float | Iterable[float] | None = None,
        stroke_behind: bool | None = None,
        flat_stroke: Optional[bool] = None,
        shading: Tuple[float, float, float] | None = None,
        recurse: bool = True
    ) -> Self:
        """
        批量设置样式属性
        
        可以同时设置填充和描边的多种属性，并支持递归应用到子对象
        """
        for mob in self.get_family(recurse):
            if fill_rgba is not None:
                mob.data['fill_rgba'][:] = resize_with_interpolation(fill_rgba, len(mob.data['fill_rgba']))
            else:
                mob.set_fill(
                    color=fill_color,
                    opacity=fill_opacity,
                    border_width=fill_border_width,
                    recurse=False
                )

            if stroke_rgba is not None:
                mob.data['stroke_rgba'][:] = resize_with_interpolation(stroke_rgba, len(mob.data['stroke_rgba']))
                mob.set_stroke(
                    width=stroke_width,
                    behind=stroke_behind,
                    flat=flat_stroke,
                    recurse=False,
                )
            else:
                mob.set_stroke(
                    color=stroke_color,
                    width=stroke_width,
                    opacity=stroke_opacity,
                    flat=flat_stroke,
                    behind=stroke_behind,
                    recurse=False,
                )

            if shading is not None:
                mob.set_shading(*shading, recurse=False)
        return self

    def get_style(self) -> dict[str, Any]:
        """获取当前样式属性字典"""
        data = self.data if self.get_num_points() > 0 else self._data_defaults
        return {
            "fill_rgba": data['fill_rgba'].copy(),
            "fill_border_width": data['fill_border_width'].copy(),
            "stroke_rgba": data['stroke_rgba'].copy(),
            "stroke_width": data['stroke_width'].copy(),
            "stroke_behind": self.stroke_behind,
            "flat_stroke": self.get_flat_stroke(),
            "shading": self.get_shading(),
        }

    def match_style(self, vmobject: VMobject, recurse: bool = True) -> Self:
        """匹配另一个VMobject的样式"""
        self.set_style(**vmobject.get_style(), recurse=False)
        if recurse:
            # 尽可能匹配子对象列表并相应地匹配样式
            submobs1, submobs2 = self.submobjects, vmobject.submobjects
            if len(submobs1) == 0:
                return self
            elif len(submobs2) == 0:
                submobs2 = [vmobject]
            for sm1, sm2 in zip(*make_even(submobs1, submobs2)):
                sm1.match_style(sm2)
        return self

    def set_color(
        self,
        color: ManimColor | Iterable[ManimColor] | None,
        opacity: float | Iterable[float] | None = None,
        recurse: bool = True
    ) -> Self:
        """同时设置填充和描边颜色"""
        self.set_fill(color, opacity=opacity, recurse=recurse)
        self.set_stroke(color, opacity=opacity, recurse=recurse)
        return self

    def set_opacity(
        self,
        opacity: float | Iterable[float] | None,
        recurse: bool = True
    ) -> Self:
        """同时设置填充和描边透明度"""
        self.set_fill(opacity=opacity, recurse=recurse)
        self.set_stroke(opacity=opacity, recurse=recurse)
        return self

    def set_anti_alias_width(self, anti_alias_width: float, recurse: bool = True) -> Self:
        """设置抗锯齿宽度"""
        self.set_uniform(recurse, anti_alias_width=anti_alias_width)
        return self

    def fade(self, darkness: float = 0.5, recurse: bool = True) -> Self:
        """使对象变暗（降低透明度）"""
        mobs = self.get_family() if recurse else [self]
        for mob in mobs:
            factor = 1.0 - darkness
            mob.set_fill(
                opacity=factor * mob.get_fill_opacity(),
                recurse=False,
            )
            mob.set_stroke(
                opacity=factor * mob.get_stroke_opacity(),
                recurse=False,
            )
        return self

    def get_fill_colors(self) -> list[str]:
        """获取所有点的填充颜色（十六进制字符串）"""
        return [
            rgb_to_hex(rgba[:3])
            for rgba in self.data['fill_rgba']
        ]

    def get_fill_opacities(self) -> np.ndarray:
        """获取所有点的填充透明度"""
        return self.data['fill_rgba'][:, 3]

    def get_stroke_colors(self) -> list[str]:
        """获取所有点的描边颜色（十六进制字符串）"""
        return [
            rgb_to_hex(rgba[:3])
            for rgba in self.data['stroke_rgba']
        ]

    def get_stroke_opacities(self) -> np.ndarray:
        """获取所有点的描边透明度"""
        return self.data['stroke_rgba'][:, 3]

    def get_stroke_widths(self) -> np.ndarray:
        """获取所有点的描边宽度"""
        return self.data['stroke_width'][:, 0]

    # TODO: 这些方法返回列表中的第一个元素而不是完整信息，有点奇怪
    def get_fill_color(self) -> str:
        """
        获取填充颜色
        
        如果有多种颜色（渐变），返回第一个颜色
        """
        data = self.data if self.has_points() else self._data_defaults
        return rgb_to_hex(data["fill_rgba"][0, :3])

    def get_fill_opacity(self) -> float:
        """
        获取填充透明度
        
        如果有多种透明度，返回第一个
        """
        data = self.data if self.has_points() else self._data_defaults
        return data["fill_rgba"][0, 3]

    def get_stroke_color(self) -> str:
        """获取描边颜色（第一个点的）"""
        data = self.data if self.has_points() else self._data_defaults
        return rgb_to_hex(data["stroke_rgba"][0, :3])

    def get_stroke_width(self) -> float:
        """获取描边宽度（第一个点的）"""
        data = self.data if self.has_points() else self._data_defaults
        return data["stroke_width"][0, 0]

    def get_stroke_opacity(self) -> float:
        """获取描边透明度（第一个点的）"""
        data = self.data if self.has_points() else self._data_defaults
        return data["stroke_rgba"][0, 3]

    def get_color(self) -> str:
        """获取主要颜色（有填充则返回填充色，否则返回描边色）"""
        if self.has_fill():
            return self.get_fill_color()
        return self.get_stroke_color()

    def get_anti_alias_width(self):
        """获取抗锯齿宽度"""
        return self.uniforms["anti_alias_width"]

    def has_stroke(self) -> bool:
        """判断是否有可见的描边"""
        data = self.data if len(self.data) > 0 else self._data_defaults
        return any(data['stroke_width']) and any(data['stroke_rgba'][:, 3])

    def has_fill(self) -> bool:
        """判断是否有可见的填充"""
        data = self.data if len(self.data) > 0 else self._data_defaults
        return any(data['fill_rgba'][:, 3])

    def get_opacity(self) -> float:
        """获取主要透明度（有填充则返回填充透明度，否则返回描边透明度）"""
        if self.has_fill():
            return self.get_fill_opacity()
        return self.get_stroke_opacity()

    def set_flat_stroke(self, flat_stroke: bool = True, recurse: bool = True) -> Self:
        """设置是否使用扁平描边"""
        self.set_uniform(recurse, flat_stroke=float(flat_stroke))
        return self

    def get_flat_stroke(self) -> bool:
        """判断是否使用扁平描边"""
        return self.uniforms["flat_stroke"] == 1.0

    def set_scale_stroke_with_zoom(self, scale_stroke_with_zoom: bool = True, recurse: bool = True) -> Self:
        """设置描边是否随缩放变化"""
        self.set_uniform(recurse, scale_stroke_with_zoom=float(scale_stroke_with_zoom))
        pass

    def get_scale_stroke_with_zoom(self) -> bool:
        """判断描边是否随缩放变化"""
        return self.uniforms["flat_stroke"] == 1.0

    def set_joint_type(self, joint_type: str, recurse: bool = True) -> Self:
        """设置连接类型"""
        for mob in self.get_family(recurse):
            mob.uniforms["joint_type"] = self.joint_type_map[joint_type]
        return self

    def get_joint_type(self) -> float:
        """获取连接类型"""
        return self.uniforms["joint_type"]

    def apply_depth_test(
        self,
        anti_alias_width: float = 0,
        recurse: bool = True
    ) -> Self:
        """应用深度测试"""
        super().apply_depth_test(recurse)
        self.set_anti_alias_width(anti_alias_width)
        return self

    def deactivate_depth_test(
        self,
        anti_alias_width: float = 1.0,
        recurse: bool = True
    ) -> Self:
        """禁用深度测试"""
        super().deactivate_depth_test(recurse)
        self.set_anti_alias_width(anti_alias_width)
        return self

    def use_winding_fill(self, value: bool = True, recurse: bool = True) -> Self:
        """使用环绕填充（仅为兼容旧场景保留）"""
        # 仅保留此方法，因为一些旧场景会调用它
        return self

    # 点和路径相关方法
    def set_anchors_and_handles(
        self,
        anchors: Vect3Array,
        handles: Vect3Array,
    ) -> Self:
        """
        设置锚点和控制点
        
        Args:
            anchors: 锚点数组
            handles: 控制点数组（长度应比锚点少1）
        """
        if len(anchors) == 0:
            self.clear_points()
            return self
        assert len(anchors) == len(handles) + 1
        points = resize_array(self.get_points(), 2 * len(anchors) - 1)
        points[0::2] = anchors  # 偶数索引放锚点
        points[1::2] = handles  # 奇数索引放控制点
        self.set_points(points)
        return self

    def start_new_path(self, point: Vect3) -> Self:
        """
        开始新路径
        
        通过将控制点放在前一个锚点上来标记路径结束
        """
        if self.has_points():
            self.append_points([self.get_last_point(), point])
        else:
            self.set_points([point])
        return self

    def add_cubic_bezier_curve(
        self,
        anchor1: Vect3,
        handle1: Vect3,
        handle2: Vect3,
        anchor2: Vect3
    ) -> Self:
        """添加完整的三次贝塞尔曲线"""
        self.start_new_path(anchor1)
        self.add_cubic_bezier_curve_to(handle1, handle2, anchor2)
        return self

    def add_cubic_bezier_curve_to(
        self,
        handle1: Vect3,
        handle2: Vect3,
        anchor: Vect3,
    ) -> Self:
        """
        向前一个锚点添加三次贝塞尔曲线
        
        将三次贝塞尔曲线近似为二次贝塞尔曲线后添加到路径
        """
        self.throw_error_if_no_points()
        last = self.get_last_point()
        # 注意：这里假设所有点都在xy平面上
        v1 = handle1 - last
        v2 = anchor - handle2
        angle = angle_between_vectors(v1, v2)
        # 如果角度较小，使用简单的二次近似
        if self.use_simple_quadratic_approx and angle < 45 * DEG:
            quad_approx = [last, find_intersection(last, v1, anchor, -v2), anchor]
        else:
            quad_approx = get_quadratic_approximation_of_cubic(
                last, handle1, handle2, anchor
            )
        if self.consider_points_equal(quad_approx[1], last):
            # 防止子路径被意外标记为闭合
            quad_approx[1] = midpoint(*quad_approx[1:3])
        self.append_points(quad_approx[1:])
        return self

    def add_quadratic_bezier_curve_to(self, handle: Vect3, anchor: Vect3, allow_null_curve=True) -> Self:
        """添加二次贝塞尔曲线"""
        self.throw_error_if_no_points()
        last_point = self.get_last_point()
        if not allow_null_curve and self.consider_points_equal(last_point, anchor):
            return self
        if self.consider_points_equal(handle, last_point):
            # 防止子路径被意外标记为闭合
            handle = midpoint(handle, anchor)
        self.append_points([handle, anchor])
        return self

    def add_line_to(self, point: Vect3, allow_null_line: bool = True) -> Self:
        """添加直线段"""
        self.throw_error_if_no_points()
        last_point = self.get_last_point()
        if not allow_null_line and self.consider_points_equal(last_point, point):
            return self
        # 根据是否为长线条选择不同的细分点数
        alphas = np.linspace(0, 1, 5 if self.long_lines else 3)
        self.append_points(outer_interpolate(last_point, point, alphas[1:]))
        return self

    def add_smooth_curve_to(self, point: Vect3) -> Self:
        """添加平滑曲线（使用反射的控制点确保平滑过渡）"""
        if self.has_new_path_started():
            self.add_line_to(point)
        else:
            self.throw_error_if_no_points()
            new_handle = self.get_reflection_of_last_handle()
            self.add_quadratic_bezier_curve_to(new_handle, point)
        return self

    def add_smooth_cubic_curve_to(self, handle: Vect3, point: Vect3) -> Self:
        """添加平滑三次曲线"""
        self.throw_error_if_no_points()
        if self.get_num_points() == 1:
            new_handle = handle
        else:
            new_handle = self.get_reflection_of_last_handle()
        self.add_cubic_bezier_curve_to(new_handle, handle, point)
        return self

    def add_arc_to(self, point: Vect3, angle: float, n_components: int | None = None, threshold: float = 1e-3) -> Self:
        """
        添加圆弧
        
        Args:
            point: 圆弧终点
            angle: 圆弧角度
            n_components: 组成圆弧的贝塞尔曲线数量
            threshold: 角度阈值，小于此值则用直线代替
        """
        self.throw_error_if_no_points()
        if abs(angle) < threshold:
            self.add_line_to(point)
            return self

        # 为n_components分配默认值
        if n_components is None:
            n_components = int(np.ceil(8 * abs(angle) / TAU))

        arc_points = quadratic_bezier_points_for_arc(angle, n_components)
        target_vect = point - self.get_end()
        curr_vect = arc_points[-1] - arc_points[0]

        # 旋转和缩放圆弧以匹配目标向量
        arc_points = arc_points @ rotation_between_vectors(curr_vect, target_vect).T
        arc_points *= get_norm(target_vect) / get_norm(curr_vect)
        arc_points += (self.get_end() - arc_points[0])
        self.append_points(arc_points[1:])
        return self

    def has_new_path_started(self) -> bool:
        """判断是否已开始新路径"""
        points = self.get_points()
        if len(points) == 0:
            return False
        elif len(points) == 1:
            return True
        # 路径开始的标志是控制点与前一个锚点重合
        return self.consider_points_equal(points[-3], points[-2])

    def get_last_point(self) -> Vect3:
        """获取最后一个点"""
        return self.get_points()[-1]

    def get_reflection_of_last_handle(self) -> Vect3:
        """获取最后一个控制点的反射点（用于平滑曲线）"""
        points = self.get_points()
        return 2 * points[-1] - points[-2]

    def close_path(self, smooth: bool = False) -> Self:
        """
        闭合路径
        
        Args:
            smooth: 是否平滑闭合（使用曲线而非直线）
        """
        if self.is_closed():
            return self
        ends = self.get_subpath_end_indices()
        # 获取最后一个路径的起点
        last_path_start = self.get_points()[0 if len(ends) == 1 else ends[-2] + 2]
        if smooth:
            self.add_smooth_curve_to(last_path_start)
        else:
            self.add_line_to(last_path_start)
        return self

    def is_closed(self) -> bool:
        """判断路径是否闭合"""
        points = self.get_points()
        ends = self.get_subpath_end_indices()
        last_path_start = points[0 if len(ends) == 1 else ends[-2] + 2]
        return self.consider_points_equal(last_path_start, points[-1])

    def subdivide_curves_by_condition(
        self,
        tuple_to_subdivisions: Callable,
        recurse: bool = True
    ) -> Self:
        """
        根据条件细分曲线
        
        Args:
            tuple_to_subdivisions: 接收贝塞尔曲线参数并返回细分数量的函数
            recurse: 是否递归应用到子对象
        """
        for vmob in self.get_family(recurse):
            if not vmob.has_points():
                continue
            new_points = [vmob.get_points()[0]]
            for tup in vmob.get_bezier_tuples():
                n_divisions = tuple_to_subdivisions(*tup)
                if n_divisions > 0:
                    alphas = np.linspace(0, 1, n_divisions + 2)
                    new_points.extend([
                        partial_quadratic_bezier_points(tup, a1, a2)[1:]
                        for a1, a2 in zip(alphas, alphas[1:])
                    ])
                else:
                    new_points.append(tup[1:])
            vmob.set_points(np.vstack(new_points))
        return self

    def subdivide_sharp_curves(
        self,
        angle_threshold: float = 30 * DEG,
        recurse: bool = True
    ) -> Self:
        """
        细分尖锐曲线
        
        根据角度阈值判断曲线是否尖锐，对尖锐曲线进行细分
        """
        def tuple_to_subdivisions(b0, b1, b2):
            angle = angle_between_vectors(b1 - b0, b2 - b1)
            return int(angle / angle_threshold)

        self.subdivide_curves_by_condition(tuple_to_subdivisions, recurse)
        return self

    def subdivide_intersections(self, recurse: bool = True, n_subdivisions: int = 1) -> Self:
        """细分与自身相交的曲线"""
        path = self.get_anchors()
        def tuple_to_subdivisions(b0, b1, b2):
            if line_intersects_path(b0, b1, path):
                return n_subdivisions
            return 0

        self.subdivide_curves_by_condition(tuple_to_subdivisions, recurse)
        return self

    def add_points_as_corners(self, points: Iterable[Vect3]) -> Self:
        """将点集作为角点添加（用直线连接）"""
        for point in points:
            self.add_line_to(point)
        return self

    def set_points_as_corners(self, points: Iterable[Vect3]) -> Self:
        """将点集设置为角点（用直线连接）"""
        anchors = np.array(points)
        handles = 0.5 * (anchors[:-1] + anchors[1:])  # 控制点设在中点（直线）
        self.set_anchors_and_handles(anchors, handles)
        return self

    def set_points_smoothly(
        self,
        points: Iterable[Vect3],
        approx: bool = True
    ) -> Self:
        """平滑地设置点集（创建平滑曲线）"""
        self.set_points_as_corners(points)
        self.make_smooth(approx=approx)
        return self

    def is_smooth(self, angle_tol=1 * DEG) -> bool:
        """判断曲线是否平滑（连接角度是否小于阈值）"""
        angles = np.abs(self.get_joint_angles()[0::2])
        return (angles < angle_tol).all()

    def change_anchor_mode(self, mode: str) -> Self:
        """
        改变锚点模式
        
        Args:
            mode: 模式，可选"jagged"（锯齿状）、"approx_smooth"（近似平滑）、"true_smooth"（真正平滑）
        """
        assert mode in ("jagged", "approx_smooth", "true_smooth")
        if self.get_num_points() == 0:
            return self
        subpaths = self.get_subpaths()
        self.clear_points()
        for subpath in subpaths:
            anchors = subpath[::2]
            new_subpath = np.array(subpath)
            if mode == "jagged":
                # 锯齿模式：控制点设在中点
                new_subpath[1::2] = 0.5 * (anchors[:-1] + anchors[1:])
            elif mode == "approx_smooth":
                # 近似平滑模式：使用近似平滑的控制点
                new_subpath[1::2] = approx_smooth_quadratic_bezier_handles(anchors)
            elif mode == "true_smooth":
                # 真正平滑模式：创建真正平滑的路径
                new_subpath = smooth_quadratic_path(anchors)
            # 调整任何与前一个锚点重合的控制点
            a0 = new_subpath[0:-1:2]
            h = new_subpath[1::2]
            a1 = new_subpath[2::2]
            false_ends = np.equal(a0, h).all(1)
            h[false_ends] = 0.5 * (a0[false_ends] + a1[false_ends])
            self.add_subpath(new_subpath)
        return self

    def make_smooth(self, approx=True, recurse=True) -> Self:
        """
        使路径平滑
        
        编辑路径以平滑通过所有当前锚点
        
        Args:
            approx: 是否使用近似平滑（不增加点数）
            recurse: 是否递归应用到子对象
        """
        mode = "approx_smooth" if approx else "true_smooth"
        for submob in self.get_family(recurse):
            if submob.is_smooth():
                continue
            submob.change_anchor_mode(mode)
        return self

    def make_approximately_smooth(self, recurse=True) -> Self:
        """使路径近似平滑"""
        self.make_smooth(approx=True, recurse=recurse)
        return self

    def make_jagged(self, recurse=True) -> Self:
        """使路径呈锯齿状"""
        for submob in self.get_family(recurse):
            submob.change_anchor_mode("jagged")
        return self

    def add_subpath(self, points: Vect3Array) -> Self:
        """添加子路径"""
        assert len(points) % 2 == 1 or len(points) == 0
        if not self.has_points():
            self.set_points(points)
            return self
        # 如果新子路径的起点与当前最后一点不重合，则开始新路径
        if not self.consider_points_equal(points[0], self.get_points()[-1]):
            self.start_new_path(points[0])
        self.append_points(points[1:])
        return self

    def append_vectorized_mobject(self, vmobject: VMobject) -> Self:
        """追加另一个向量图形对象的路径和数据"""
        self.add_subpath(vmobject.get_points())
        n = vmobject.get_num_points()
        self.data[-n:] = vmobject.data
        return self

    def consider_points_equal(self, p0: Vect3, p1: Vect3) -> bool:
        """判断两个点是否相等（在容差范围内）"""
        return all(abs(p1 - p0) < self.tolerance_for_point_equality)

    # 曲线信息相关方法
    def get_bezier_tuples_from_points(self, points: Vect3Array) -> Iterable[Vect3Array]:
        """从点数组中获取贝塞尔曲线元组（每组3个点：起点、控制点、终点）"""
        n_curves = (len(points) - 1) // 2
        return (points[2 * i:2 * i + 3] for i in range(n_curves))

    def get_bezier_tuples(self) -> Iterable[Vect3Array]:
        """获取当前路径的贝塞尔曲线元组"""
        return self.get_bezier_tuples_from_points(self.get_points())

    def get_subpath_end_indices_from_points(self, points: Vect3Array) -> np.ndarray:
        """从点数组中获取子路径结束索引"""
        atol = 1e-4  # TODO: 这个值太随意了
        a0, h, a1 = points[0:-1:2], points[1::2], points[2::2]
        # 一个锚点被视为路径的终点，如果它后面的控制点与它重合
        # 为了区分连续的空曲线情况，我们还检查下一个锚点确实不同
        is_end = (a0 == h).all(1) & (abs(h - a1) > atol).any(1)
        end_indices = (2 * n for n, end in enumerate(is_end) if end)
        return np.array([*end_indices, len(points) - 1])

    def get_subpath_end_indices(self) -> np.ndarray:
        """获取子路径结束索引"""
        if self.subpath_end_indices is None:
            self.subpath_end_indices = self.get_subpath_end_indices_from_points(self.get_points())
        return self.subpath_end_indices

    def get_subpaths_from_points(self, points: Vect3Array) -> list[Vect3Array]:
        """从点数组中获取子路径列表"""
        if len(points) == 0:
            return []
        end_indices = self.get_subpath_end_indices_from_points(points)
        start_indices = [0, *(end_indices[:-1] + 2)]
        return [points[i1:i2 + 1] for i1, i2 in zip(start_indices, end_indices)]

    def get_subpaths(self) -> list[Vect3Array]:
        """获取当前路径的子路径列表"""
        return self.get_subpaths_from_points(self.get_points())

    def get_nth_curve_points(self, n: int) -> Vect3Array:
        """获取第n条曲线的点"""
        assert n < self.get_num_curves()
        return self.get_points()[2 * n:2 * n + 3]

    def get_nth_curve_function(self, n: int) -> Callable[[float], Vect3]:
        """获取第n条曲线的贝塞尔函数"""
        return bezier(self.get_nth_curve_points(n))

    def get_num_curves(self) -> int:
        """获取曲线数量"""
        return self.get_num_points() // 2

    def quick_point_from_proportion(self, alpha: float) -> Vect3:
        """
        快速获取比例位置的点
        
        假设所有曲线长度相同，因此可能不准确
        """
        num_curves = self.get_num_curves()
        if num_curves == 0:
            return self.get_center()
        n, residue = integer_interpolate(0, num_curves, alpha)
        curve_func = self.get_nth_curve_function(n)
        return curve_func(residue)

    def curve_and_prop_of_partial_point(self, alpha) -> Tuple[int, float]:
        """
        获取比例位置所在的曲线索引和在该曲线上的比例
        
        Args:
            alpha: 沿整个路径的比例（0到1）
        
        Returns:
            曲线索引和在该曲线上的比例
        """
        if alpha == 0:
            return (0, 0.0)
        partials: list[float] = [0]
        for tup in self.get_bezier_tuples():
            if self.consider_points_equal(tup[0], tup[1]):
                # 不考虑空曲线
                arclen = 0
            else:
                # 用起点到终点的直线近似长度
                arclen = get_norm(tup[2] - tup[0])
            partials.append(partials[-1] + arclen)
        full = partials[-1]
        if full == 0:
            return len(partials), 1.0
        # 找到第一个部分长度大于alpha倍总长度的索引
        index = next(
            (i for i, x in enumerate(partials) if x >= full * alpha),
            len(partials) - 1  # 默认值
        )
        residue = float(inverse_interpolate(
            partials[index - 1] / full, partials[index] / full, alpha
        ))
        return index - 1, residue

    def point_from_proportion(self, alpha: float) -> Vect3:
        """
        获取沿路径比例位置的点
        
        Args:
            alpha: 沿路径的比例（0到1）
        """
        if alpha <= 0:
            return self.get_start()
        elif alpha >= 1:
            return self.get_end()
        if self.get_num_points() == 0:
            return self.get_center()
        index, residue = self.curve_and_prop_of_partial_point(alpha)
        return self.get_nth_curve_function(index)(residue)

    def get_anchors_and_handles(self) -> list[Vect3]:
        """
        获取锚点和控制点
        
        returns anchors1, handles, anchors2,
        其中 (anchors1[i], handles[i], anchors2[i])
        定义了第i条二次贝塞尔曲线
        """
        points = self.get_points()
        return [points[0:-1:2], points[1::2], points[2::2]]

    def get_start_anchors(self) -> Vect3Array:
        """获取起始锚点数组"""
        return self.get_points()[0:-1:2]

    def get_end_anchors(self) -> Vect3:
        """获取结束锚点数组"""
        return self.get_points()[2::2]

    def get_anchors(self) -> Vect3Array:
        """获取所有锚点（每隔一个点取一个）"""
        return self.get_points()[::2]

    def get_points_without_null_curves(self, atol: float = 1e-9) -> Vect3Array:
        """获取去除空曲线后的点数组"""
        new_points = [self.get_points()[0]]
        for tup in self.get_bezier_tuples():
            if get_norm(tup[1] - tup[0]) > atol or get_norm(tup[2] - tup[0]) > atol:
                new_points.append(tup[1:])
        return np.vstack(new_points)

    def get_arc_length(self, n_sample_points: int | None = None) -> float:
        """
        获取弧长
        
        Args:
            n_sample_points: 采样点数量，为None时使用近似计算
        """
        if n_sample_points is not None:
            points = np.array([
                self.quick_point_from_proportion(a)
                for a in np.linspace(0, 1, n_sample_points)
            ])
            return poly_line_length(points)
        points = self.get_points()
        inner_len = poly_line_length(points[::2])  # 锚点连线长度
        outer_len = poly_line_length(points)  # 所有点连线长度
        return interpolate(inner_len, outer_len, 1 / 3)  # 取两者的插值

    def get_area_vector(self) -> Vect3:
        """
        获取面积向量
        
        返回一个向量，其长度是由锚点形成的多边形的面积，
        方向根据右手定则垂直于多边形
        """
        if not self.has_points():
            return np.zeros(3)

        p0 = self.get_anchors()
        p1 = np.vstack([p0[1:], p0[0]])  # 下一个锚点，最后一个连接到第一个

        # 每个项遍历所有边 [(x0, y0, z0), (x1, y1, z1)]
        sums = p0 + p1
        diffs = p1 - p0
        return 0.5 * np.array([
            (sums[:, 1] * diffs[:, 2]).sum(),  # 累加 (y0 + y1)*(z1 - z0)
            (sums[:, 2] * diffs[:, 0]).sum(),  # 累加 (z0 + z1)*(x1 - x0)
            (sums[:, 0] * diffs[:, 1]).sum(),  # 累加 (x0 + x1)*(y1 - y0)
        ])

    def get_unit_normal(self, refresh: bool = False) -> Vect3:
        """
        获取单位法向量
        
        Args:
            refresh: 是否强制刷新
        """
        if self.get_num_points() < 3:
            return OUT

        if not self.needs_new_unit_normal and not refresh:
            return self.data["base_normal"][1, :]

        area_vect = self.get_area_vector()
        area = get_norm(area_vect)
        if area > 0:
            normal = area_vect / area
        else:
            p = self.get_points()
            normal = get_unit_normal(p[1] - p[0], p[2] - p[1])
        self.data["base_normal"][1::2] = normal
        self.needs_new_unit_normal = False
        return normal

    def refresh_unit_normal(self) -> Self:
        """标记需要刷新单位法向量"""
        self.needs_new_unit_normal = True
        return self

    def rotate(
        self,
        angle: float,
        axis: Vect3 = OUT,
        about_point: Vect3 | None = None,
        **kwargs
    ) -> Self:
        """旋转对象并刷新法向量"""
        super().rotate(angle, axis, about_point,** kwargs)
        for mob in self.get_family():
            mob.refresh_unit_normal()
        return self

    def ensure_positive_orientation(self, recurse=True) -> Self:
        """确保正向朝向（法向量z分量为正）"""
        for mob in self.get_family(recurse):
            if mob.get_unit_normal()[2] < 0:
                mob.reverse_points()
        return self

    # 对齐相关方法
    def align_points(self, vmobject: VMobject) -> Self:
        """对齐当前对象和另一个VMobject的点"""
        if self.get_num_points() == len(vmobject.get_points()):
            for mob in [self, vmobject]:
                mob.get_joint_angles()
            return self

        for mob in self, vmobject:
            # 如果没有点，添加一个到"中心"位置的点
            if not mob.has_points():
                mob.start_new_path(mob.get_center())

        # 确定子路径并对齐
        subpaths1 = self.get_subpaths()
        subpaths2 = vmobject.get_subpaths()
        for subpaths in [subpaths1, subpaths2]:
            subpaths.sort(key=lambda sp: -sum(
                get_norm(p2 - p1)
                for p1, p2 in zip(sp, sp[1:])
            ))
        n_subpaths = max(len(subpaths1), len(subpaths2))

        # 开始构建新的子路径
        new_subpaths1 = []
        new_subpaths2 = []

        def get_nth_subpath(path_list, n):
            if n >= len(path_list):
                return np.vstack([path_list[0][:-1], path_list[0][::-1]])
            return path_list[n]
        for n in range(n_subpaths):
            # 获取当前索引对应的子路径（子路径数量不足时复用第一个并反转）
            sp1 = get_nth_subpath(subpaths1, n)
            sp2 = get_nth_subpath(subpaths2, n)
            
            # 计算两个子路径的长度差，确定需要插入的曲线数量（确保长度一致）
            diff1 = max(0, (len(sp2) - len(sp1)) // 2)  # sp1需插入的曲线数
            diff2 = max(0, (len(sp1) - len(sp2)) // 2)  # sp2需插入的曲线数
            
            # 插入曲线使两个子路径长度匹配
            sp1 = self.insert_n_curves_to_point_list(diff1, sp1)
            sp2 = self.insert_n_curves_to_point_list(diff2, sp2)
            
            # 非第一个子路径时，添加中间锚点标记路径结束
            if n > 0:
                new_subpaths1.append(new_subpaths1[-1][-1])
                new_subpaths2.append(new_subpaths2[-1][-1])
            
            # 将处理后的子路径添加到新列表
            new_subpaths1.append(sp1)
            new_subpaths2.append(sp2)

        # 应用处理后的点数据到两个对象
        for mob, paths in [(self, new_subpaths1), (vmobject, new_subpaths2)]:
            new_points = np.vstack(paths)  # 合并所有子路径的点
            # 调整数据长度并保持顺序，设置新点并更新连接角
            mob.resize_points(len(new_points), resize_func=resize_preserving_order)
            mob.set_points(new_points)
            mob.get_joint_angles()
        return self

    def insert_n_curves(self, n: int, recurse: bool = True) -> Self:
        """
        向家族成员中插入指定数量的曲线
        
        参数:
            n: 需插入的曲线总数
            recurse: 是否递归处理子对象
        """
        for mob in self.get_family(recurse):
            # 仅对有曲线的对象操作
            if mob.get_num_curves() > 0:
                # 插入曲线并更新点数据
                new_points = mob.insert_n_curves_to_point_list(n, mob.get_points())
                mob.set_points(new_points)
        return self

    def insert_n_curves_to_point_list(self, n: int, points: Vect3Array) -> Vect3Array:
        """
        向点列表中插入指定数量的曲线（将现有曲线细分）
        
        参数:
            n: 需插入的曲线总数
            points: 原始点列表
        
        返回:
            插入曲线后的新点列表
        """
        # 若只有1个点，直接重复点以满足曲线格式（2n+1个点）
        if len(points) == 1:
            return np.repeat(points, 2 * n + 1, 0)

        # 提取所有贝塞尔曲线元组（每组3个点：起点、控制点、终点）
        bezier_tuples = list(self.get_bezier_tuples_from_points(points))
        atol = self.tolerance_for_point_equality
        # 计算每条曲线的"权重"（非空曲线用起点到终点的距离，空曲线权重为0）
        norms = [
            0 if get_norm(tup[1] - tup[0]) < atol else get_norm(tup[2] - tup[0])
            for tup in bezier_tuples
        ]
        
        # 计算每条曲线需插入的细分次数（按权重分配插入数量）
        ipc = np.zeros(len(bezier_tuples), dtype=int)  # ipc: insertions per curve
        for _ in range(n):
            index = np.argmax(norms)  # 选择权重最大的曲线
            ipc[index] += 1
            # 更新权重（插入后曲线权重按比例降低）
            norms[index] *= ipc[index] / (ipc[index] + 1)

        # 构建新点列表
        new_points = [points[0]]  # 起始点
        for tup, n_inserts in zip(bezier_tuples, ipc):
            # 将1条曲线细分为n_inserts+1条小曲线
            alphas = np.linspace(0, 1, n_inserts + 2)  # 细分比例（含起点和终点）
            for a1, a2 in zip(alphas, alphas[1:]):
                # 提取细分后的部分曲线点（跳过重复的起点）
                new_points.extend(partial_quadratic_bezier_points(tup, a1, a2)[1:])
        return np.vstack(new_points)

    def pointwise_become_partial(self, vmobject: VMobject, a: float, b: float) -> Self:
        """
        使当前对象成为目标对象的部分曲线（从比例a到b）
        
        参数:
            vmobject: 目标对象
            a: 起始比例（0-1）
            b: 结束比例（0-1）
        """
        assert isinstance(vmobject, VMobject)
        vm_points = vmobject.get_points()
        # 复用目标对象的连接角数据
        self.data["joint_angle"] = vmobject.data["joint_angle"]
        
        # 若比例覆盖整个对象，直接复用所有点
        if a <= 0 and b >= 1:
            self.set_points(vm_points, refresh=False)
            return self
        
        num_curves = vmobject.get_num_curves()
        # 部分曲线包含三部分：
        # 1. 起始段：某条曲线的末尾部分
        # 2. 中间段：完整匹配目标曲线
        # 3. 结束段：某条曲线的起始部分

        # 计算比例对应的曲线索引和在曲线内的残留比例
        lower_index, lower_residue = integer_interpolate(0, num_curves, a)
        upper_index, upper_residue = integer_interpolate(0, num_curves, b)
        # 计算对应点的索引（每条曲线占2个点：控制点+终点）
        i1 = 2 * lower_index
        i2 = 2 * lower_index + 3
        i3 = 2 * upper_index
        i4 = 2 * upper_index + 3

        new_points = vm_points.copy()
        # 若目标对象无曲线，将点置0
        if num_curves == 0:
            new_points[:] = 0
            return self
        
        # 情况1：起始和结束在同一条曲线内
        if lower_index == upper_index:
            # 提取该曲线的部分段
            tup = partial_quadratic_bezier_points(vm_points[i1:i2], lower_residue, upper_residue)
            new_points[:i1] = tup[0]       # 起始段前的点统一为部分段起点
            new_points[i1:i4] = tup        # 中间段替换为部分曲线
            new_points[i4:] = tup[2]       # 结束段后的点统一为部分段终点
        # 情况2：起始和结束在不同曲线内
        else:
            # 提取起始曲线的末尾部分
            low_tup = partial_quadratic_bezier_points(vm_points[i1:i2], lower_residue, 1)
            # 提取结束曲线的起始部分
            high_tup = partial_quadratic_bezier_points(vm_points[i3:i4], 0, upper_residue)
            
            new_points[0:i1] = low_tup[0]  # 起始段前的点统一为起始部分起点
            new_points[i1:i2] = low_tup    # 起始曲线替换为末尾部分
            # 中间段（i2:i3）保持原曲线不变
            new_points[i3:i4] = high_tup   # 结束曲线替换为起始部分
            new_points[i4:] = high_tup[2]  # 结束段后的点统一为结束部分终点
        
        # 无效段的连接角置0
        self.data["joint_angle"][:i1] = 0
        self.data["joint_angle"][i4:] = 0
        self.set_points(new_points, refresh=False)
        return self

    def get_subcurve(self, a: float, b: float) -> Self:
        """
        提取对象的部分曲线（从比例a到b）并返回新对象
        
        参数:
            a: 起始比例（0-1）
            b: 结束比例（0-1）
        
        返回:
            包含部分曲线的新VMobject实例
        """
        vmob = self.copy()  # 复制当前对象
        vmob.pointwise_become_partial(self, a, b)  # 截取部分曲线
        return vmob

    def get_outer_vert_indices(self) -> np.ndarray:
        """
        生成外部顶点索引，格式为 (0, 1, 2, 2, 3, 4, 4, 5, 6, ...)
        用于着色器渲染时确定顶点连接顺序
        """
        n_curves = self.get_num_curves()
        # 若索引长度不匹配曲线数量，重新生成
        if len(self.outer_vert_indices) != 3 * n_curves:
            # 生成特定规律的索引序列（每条曲线对应3个索引）
            self.outer_vert_indices = (np.arange(1, 3 * n_curves + 1) * 2) // 3
        return self.outer_vert_indices

    # 着色器相关数据（可能需要刷新）

    def get_triangulation(self) -> np.ndarray:
        """
        计算图形内部的三角剖分索引，用于着色器渲染
        
        返回:
            三角剖分的顶点索引数组
        """
        # 首先基于点数据直接生成三角形，再处理内部剖分
        points = self.get_points()
        # 若点数量不足，返回空索引
        if len(points) <= 1:
            return np.zeros(0, dtype='i4')

        # 获取单位法向量，确保法向量方向为OUT（z轴正方向）
        normal_vector = self.get_unit_normal()
        if not np.isclose(normal_vector, OUT).all():
            # 旋转点使法向量与OUT一致
            points = np.dot(points, z_to_vector(normal_vector))

        # 计算每条曲线的方向向量及朝向（凹凸性）
        v01s = points[1::2] - points[0:-1:2]  # 起点到控制点的向量
        v12s = points[2::2] - points[1::2]  # 控制点到终点的向量
        curve_orientations = np.sign(cross2d(v01s, v12s))  # 叉积符号判断朝向
        concave_parts = curve_orientations < 0  # 凹向部分标记

        # 筛选用于剖分的内部顶点索引
        indices = np.arange(len(points), dtype=int)
        inner_vert_indices = np.hstack([
            indices[0::2],  # 所有锚点
            indices[1::2][concave_parts]  # 凹向部分的控制点
        ])
        inner_vert_indices.sort()  # 排序索引

        # 确定子路径结束对应的锚点（用于多环剖分）
        end_indices = self.get_subpath_end_indices()
        counts = np.arange(1, len(inner_vert_indices) + 1)
        # 筛选锚点对应的计数，标记子路径结束
        rings = counts[inner_vert_indices % 2 == 0][end_indices // 2]

        # 执行耳切法三角剖分
        inner_verts = points[inner_vert_indices]
        inner_tri_indices = inner_vert_indices[
            earclip_triangulation(inner_verts, rings)
        ]

        # 移除无效三角形（顶点连续的三角形）
        iti = inner_tri_indices
        # 情况1：索引连续递增（如i, i+1, i+2）
        null1 = (iti[0::3] + 1 == iti[1::3]) & (iti[0::3] + 2 == iti[2::3])
        # 情况2：索引连续递减（如i, i-1, i-2）
        null2 = (iti[0::3] - 1 == iti[1::3]) & (iti[0::3] - 2 == iti[2::3])
        inner_tri_indices = iti[~(null1 | null2).repeat(3)]

        # 合并外部顶点索引和内部剖分索引
        ovi = self.get_outer_vert_indices()
        tri_indices = np.hstack([ovi, inner_tri_indices])
        return tri_indices

    def refresh_joint_angles(self) -> Self:
        """标记家族成员需更新连接角"""
        for mob in self.get_family():
            mob.needs_new_joint_angles = True
        return self

    def get_joint_angles(self, refresh: bool = False) -> np.ndarray:
        """
        计算顶点连接角（切线向量之间的夹角）
        
        说明:
            "joint product"是一个4维向量，包含连接点处切线向量的叉积和点积
        参数:
            refresh: 是否强制重新计算
        
        返回:
            连接角数组（每个顶点对应一个角度）
        """
        # 若无需刷新且数据存在，直接返回
        if not self.needs_new_joint_angles and not refresh:
            return self.data["joint_angle"][:, 0]
        # 若连接角数据被锁定，直接返回现有数据
        if "joint_angle" in self.locked_data_keys:
            return self.data["joint_angle"][:, 0]

        # 标记已更新，避免重复计算
        self.needs_new_joint_angles = False
        self._data_has_changed = True

        # 旋转点使法向量方向为z轴正方向（简化2D平面计算）
        points = self.get_points() @ rotation_between_vectors(OUT, self.get_unit_normal())
        # 点数量不足时返回默认数据
        if len(points) < 3:
            return self.data["joint_angle"][:, 0]

        # 提取锚点和控制点，计算切线向量
        a0, h, a1 = points[0:-1:2], points[1::2], points[2::2]
        a0_to_h = h - a0  # 锚点到控制点的向量（入切线）
        h_to_a1 = a1 - h  # 控制点到下一个锚点的向量（出切线）

        # 初始化入切线和出切线数组（与点数量一致）
        v_in = np.zeros(points.shape)
        v_out = np.zeros(points.shape)
        # 为控制点和锚点分配切线向量
        v_in[1::2] = a0_to_h    # 控制点的入切线 = 锚点到该控制点的向量
        v_in[2::2] = h_to_a1    # 锚点的入切线 = 前一个控制点到该锚点的向量
        v_out[0:-1:2] = a0_to_h # 锚点的出切线 = 该锚点到下一个控制点的向量
        v_out[1::2] = h_to_a1   # 控制点的出切线 = 该控制点到下一个锚点的向量

        # 处理闭合路径或标记非闭合路径的端点
        ends = self.get_subpath_end_indices()
        starts = [0, *(e + 2 for e in ends[:-1])]  # 子路径起始索引
        for start, end in zip(starts, ends):
            if start == end:
                continue
            # 闭合路径：端点的切线向量与起始点关联
            if (points[start] == points[end]).all():
                v_in[start] = v_out[end - 1]
                v_out[end] = v_in[start + 1]
            # 非闭合路径：端点的入切线=出切线（避免尖锐端点）
            else:
                v_in[start] = v_out[start]
                v_out[end] = v_in[end]

        # 计算切线向量之间的夹角（基于2D平面的角度差）
        angles_in = np.arctan2(v_in[:, 1], v_in[:, 0])  # 入切线的角度
        angles_out = np.arctan2(v_out[:, 1], v_out[:, 0])  # 出切线的角度
        angle_diffs = angles_out - angles_in  # 角度差
        # 调整角度差到[-PI, PI]范围
        angle_diffs[angle_diffs < -PI] += TAU
        angle_diffs[angle_diffs > PI] -= TAU
        # 保存连接角数据
        self.data["joint_angle"][:, 0] = angle_diffs
        return self.data["joint_angle"][:, 0]

    def lock_matching_data(self, vmobject1: VMobject, vmobject2: VMobject) -> Self:
        """
        锁定两个对象的匹配数据（确保连接角等数据同步）
        
        参数:
            vmobject1: 第一个对象
            vmobject2: 第二个对象
        """
        # 确保三个对象（当前+两个目标）的连接角已计算
        for mob in [self, vmobject1, vmobject2]:
            mob.get_joint_angles()
        # 调用父类方法锁定匹配数据
        super().lock_matching_data(vmobject1, vmobject2)
        return self

    def triggers_refresh(func: Callable):
        """
        装饰器：触发数据刷新（子路径索引、连接角、法向量）
        
        用于修改点或数据的方法，确保相关依赖数据同步更新
        """
        @wraps(func)
        def wrapper(self, *args, refresh=True, **kwargs):
            func(self, *args, **kwargs)
            if refresh:
                self.subpath_end_indices = None  # 重置子路径索引
                self.refresh_joint_angles()     # 刷新连接角
                self.refresh_unit_normal()      # 刷新法向量
            return self
        return wrapper

    @triggers_refresh
    def set_points(self, points: Vect3Array) -> Self:
        """设置点数据（装饰器触发刷新），确保点数量为奇数（锚点+控制点格式）"""
        assert len(points) == 0 or len(points) % 2 == 1
        return super().set_points(points)

    @triggers_refresh
    def append_points(self, points: Vect3Array) -> Self:
        """追加点数据（装饰器触发刷新），确保点数量为偶数（匹配控制点+终点格式）"""
        assert len(points) % 2 == 0
        return super().append_points(points)

    def reverse_points(self, recurse: bool = True) -> Self:
        """
        反转点的顺序（递归处理子对象）
        
        说明:
            反转后会重置被视为路径结束的锚点，并反转法向量方向
        参数:
            recurse: 是否递归处理子对象
        """
        for mob in self.get_family(recurse):
            if not mob.has_points():
                continue
            # 重置子路径结束锚点（交换控制点和锚点位置）
            inner_ends = mob.get_subpath_end_indices()[:-1]
            mob.data["point"][inner_ends + 1] = mob.data["point"][inner_ends + 2]
            # 反转法向量方向
            mob.data["base_normal"][1::2] *= -1
            # 重置子路径索引（需重新计算）
            self.subpath_end_indices = None
        return super().reverse_points()

    @triggers_refresh
    def set_data(self, data: np.ndarray) -> Self:
        """设置数据数组（装饰器触发刷新）"""
        return super().set_data(data)

    # TODO: 如何智能处理切线向量？
    @triggers_refresh
    def apply_function(
        self,
        function: Callable[[Vect3], Vect3],
        make_smooth: bool = False,** kwargs
    ) -> Self:
        """
        对所有点应用函数变换（装饰器触发刷新）
        
        参数:
            function: 点变换函数（输入3D点，输出3D点）
            make_smooth: 变换后是否自动平滑
            **kwargs: 传递给父类apply_function的参数
        """
        super().apply_function(function, **kwargs)
        # 若需要平滑，自动执行近似平滑处理
        if self.make_smooth_after_applying_functions or make_smooth:
            self.make_smooth(approx=True)
        return self

    @triggers_refresh
    def stretch(self, *args, **kwargs) -> Self:
        """拉伸对象（装饰器触发刷新）"""
        return super().stretch(*args, **kwargs)

    @triggers_refresh
    def apply_matrix(self, *args, **kwargs) -> Self:
        """应用矩阵变换（装饰器触发刷新）"""
        return super().apply_matrix(*args, **kwargs)

    def rotate(
        self,
        angle: float,
        axis: Vect3 = OUT,
        about_point: Vect3 | None = None,** kwargs
    ) -> Self:
        """
        旋转对象（重写父类方法，确保法向量同步刷新）
        
        参数:
            angle: 旋转角度（弧度）
            axis: 旋转轴（默认OUT，z轴正方向）
            about_point: 旋转中心点（默认None，绕自身中心）
            **kwargs: 传递给apply_points_function的参数
        """
        # 计算旋转矩阵的转置（用于点变换）
        rot_matrix_T = rotation_matrix_transpose(angle, axis)
        # 应用旋转变换
        self.apply_points_function(
            lambda points: np.dot(points, rot_matrix_T),
            about_point,
            **kwargs
        )
        # 刷新所有家族成员的法向量
        for mob in self.get_family():
            mob.get_unit_normal(refresh=True)
        return self

    def set_animating_status(self, is_animating: bool, recurse: bool = True):
        """
        设置动画状态（重写父类方法，确保连接角同步刷新）
        
        参数:
            is_animating: 是否处于动画中
            recurse: 是否递归处理子对象
        """
        super().set_animating_status(is_animating, recurse)
        # 刷新所有家族成员的连接角
        for submob in self.get_family(recurse):
            submob.get_joint_angles(refresh=True)
        return self

    # 着色器相关方法

    def init_shader_wrapper(self, ctx: Context):
        """
        初始化着色器包装器（用于渲染）
        
        参数:
            ctx: OpenGL上下文
        """
        self.shader_wrapper = VShaderWrapper(
            ctx=ctx,
            vert_data=self.data,          # 顶点数据
            mobject_uniforms=self.uniforms,  # 对象统一变量
            code_replacements=self.shader_code_replacements,  # 着色器代码替换
            stroke_behind=self.stroke_behind,  # 描边是否在填充后渲染
            depth_test=self.depth_test    # 是否启用深度测试
        )

    def refresh_shader_wrapper_id(self):
        """
        刷新着色器包装器ID（确保描边层级同步）
        
        当描边层级（stroke_behind）变化时，需更新着色器包装器
        """
        for submob in self.get_family():
            if submob.shader_wrapper is not None:
                submob.shader_wrapper.stroke_behind = submob.stroke_behind
        super().refresh_shader_wrapper_id()
        return self

    def get_shader_data(self) -> np.ndarray:
        """
        获取着色器所需的完整数据（确保连接角和法向量已更新）
        
        返回:
            包含顶点、颜色、连接角等数据的数组
        """
        # 确保连接角已计算
        self.get_joint_angles()
        # 统一法向量的基础点（与第一个点一致）
        self.data["base_normal"][0::2] = self.data["point"][0]
        return super().get_shader_data()

    def get_shader_vert_indices(self) -> Optional[np.ndarray]:
        """获取着色器渲染时的顶点索引（复用外部顶点索引）"""
        return self.get_outer_vert_indices()


class VGroup(Group, VMobject, Generic[SubVmobjectType]):
    """
    向量图形组类，用于管理多个VMobject实例
    
    继承Group和VMobject，支持向量图形特有的批量操作
    """
    def __init__(self, *vmobjects: SubVmobjectType | Iterable[SubVmobjectType], **kwargs):
        super().__init__(**kwargs)
        # 验证所有子对象均为VMobject类型
        if any(isinstance(vmob, Mobject) and not isinstance(vmob, VMobject) for vmob in vmobjects):
            raise Exception("Only VMobjects can be passed into VGroup")
        # 吸收传入的子对象
        self._ingest_args(*vmobjects)
        # 若有子对象，复用第一个子对象的统一变量（确保风格一致）
        if self.submobjects:
            self.uniforms.update(self.submobjects[0].uniforms)

    def __add__(self, other: VMobject) -> Self:
        """重载加法运算符，添加VMobject到组中"""
        assert isinstance(other, VMobject)
        return self.add(other)

    # 仅为了让代码检查工具（linter）识别VGroup[...]的索引访问（如VGroup()[0]）
    def __getitem__(self, index) -> SubVmobjectType:
        return super().__getitem__(index)


class VectorizedPoint(Point, VMobject):
    """
    向量化点类，结合Point和VMobject的特性
    
    用于需要向量图形属性（如颜色、描边）的点对象
    """
    def __init__(
        self,
        location: np.ndarray = ORIGIN,
        color: ManimColor = BLACK,
        fill_opacity: float = 0.0,
        stroke_width: float = 0.0,** kwargs
    ):
        # 初始化Point基类（位置属性）
        Point.__init__(self, location, **kwargs)
        # 初始化VMobject基类（向量图形属性）
        VMobject.__init__(
            self,
            color=color,
            fill_opacity=fill_opacity,
            stroke_width=stroke_width,
            **kwargs
        )
        # 设置点数据（仅包含一个位置点）
        self.set_points(np.array([location]))


class CurvesAsSubmobjects(VGroup):
    """
    将单个VMobject的每条曲线拆分为独立子对象的组
    
    用于对曲线进行逐段操作（如单独动画、着色）
    """
    def __init__(self, vmobject: VMobject, **kwargs):
        super().__init__(**kwargs)
        # 遍历目标对象的每条贝塞尔曲线，创建独立子对象
        for tup in vmobject.get_bezier_tuples():
            part = VMobject()
            part.set_points(tup)          # 为子对象设置单条曲线的点
            part.match_style(vmobject)    # 匹配目标对象的样式
            self.add(part)


class DashedVMobject(VMobject):
    """
    虚线对象，将目标VMobject转换为虚线样式
    
    通过截取目标对象的部分曲线实现虚线效果
    """
    def __init__(
        self,
        vmobject: VMobject,
        num_dashes: int = 15,          # 虚线段数量
        positive_space_ratio: float = 0.5,  # 实线占比（0-1）
        **kwargs
    ):
        super().__init__(**kwargs)

        if num_dashes > 0:
            # 生成单位区间的分割比例（含起点和终点）
            alphas = np.linspace(0, 1, num_dashes + 1)
            
            # 计算每条虚线的长度比例
            full_d_alpha = (1.0 / num_dashes)  # 每个虚线单元的总长度比例
            partial_d_alpha = full_d_alpha * positive_space_ratio  # 实线部分的长度比例
            
            # 重新缩放比例，确保最后一条虚线的终点与目标对象一致
            alphas /= (1 - full_d_alpha + partial_d_alpha)
            
            # 截取每条虚线段并添加到当前对象
            self.add(*[
                vmobject.get_subcurve(alpha, alpha + partial_d_alpha)
                for alpha in alphas[:-1]
            ])
        # 子对象家族关系已通过get_subcurve自动处理
        # 匹配目标对象的样式（不递归，避免重复处理）
        self.match_style(vmobject, recurse=False)


class VHighlight(VGroup):
    """
    高亮效果组，为目标对象添加多层描边高亮
    
    通过生成多层递增宽度的描边，实现发光或立体高亮效果
    """
    def __init__(
        self,
        vmobject: VMobject,
        n_layers: int = 5,                 # 高亮层数
        color_bounds: Tuple[ManimColor] = (GREY_C, GREY_E),  # 颜色渐变范围
        max_stroke_addition: float = 5.0,  # 最大描边宽度增量
    ):
        # 复制目标对象n_layers次（生成多层描边）
        outline = vmobject.replicate(n_layers)
        # 关闭填充（仅保留描边用于高亮）
        outline.set_fill(opacity=0)
        # 生成递增的描边宽度增量（从0到max_stroke_addition）
        added_widths = np.linspace(0, max_stroke_addition, n_layers + 1)[1:]
        # 生成颜色渐变（从color_bounds[0]到color_bounds[1]）
        colors = color_gradient(color_bounds, n_layers)
        
        # 反向遍历层级，为每层设置递增的描边宽度和对应的渐变颜色
        for part, added_width, color in zip(reversed(outline), added_widths, colors):
            for sm in part.family_members_with_points():
                sm.set_stroke(
                    width=sm.get_stroke_width() + added_width,  # 基础宽度+增量
                    color=color,
                )
        # 将所有高亮层添加到组中
        super().__init__(*outline)
