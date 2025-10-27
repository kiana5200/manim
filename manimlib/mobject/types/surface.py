from __future__ import annotations

import moderngl
import numpy as np

from manimlib.constants import GREY
from manimlib.constants import OUT
from manimlib.mobject.mobject import Mobject
from manimlib.utils.bezier import integer_interpolate
from manimlib.utils.bezier import interpolate
from manimlib.utils.bezier import inverse_interpolate
from manimlib.utils.images import get_full_raster_image_path
from manimlib.utils.iterables import listify
from manimlib.utils.iterables import resize_with_interpolation
from manimlib.utils.simple_functions import clip
from manimlib.utils.space_ops import normalize_along_axis
from manimlib.utils.space_ops import cross

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable, Iterable, Sequence, Tuple

    from manimlib.camera.camera import Camera
    from manimlib.typing import ManimColor, Vect3, Vect3Array, Self


class Surface(Mobject):
    """
    曲面基类，继承自Mobject，用于表示3D空间中的参数化曲面
    
    该类提供了曲面渲染、顶点计算、法向量计算等基础功能，
    子类可通过实现uv_func方法定义具体的曲面形状
    """
    # 渲染图元类型：三角形
    render_primitive: int = moderngl.TRIANGLES
    # 着色器文件夹
    shader_folder: str = "surface"
    # 数据类型定义，包含顶点坐标、法向量相关点和颜色信息
    data_dtype: np.dtype = np.dtype([
        ('point', np.float32, (3,)),          # 顶点坐标
        ('d_normal_point', np.float32, (3,)), # 用于计算法向量的偏移点
        ('rgba', np.float32, (4,)),           # 颜色和透明度
    ])
    # 类点数据的键名
    pointlike_data_keys = ['point', 'd_normal_point']

    def __init__(
        self,
        color: ManimColor = GREY,
        shading: Tuple[float, float, float] = (0.3, 0.2, 0.4),
        depth_test: bool = True,
        u_range: Tuple[float, float] = (0.0, 1.0),
        v_range: Tuple[float, float] = (0.0, 1.0),
        # 分辨率表示采样点数量，每个坐标方向的采样点数量比近似正方形的行列数多1
        resolution: Tuple[int, int] = (101, 101),
        prefered_creation_axis: int = 1,
        # 用于计算du和dv步长的微小量
        epsilon: float = 1e-3,
        # 用于确定法向量方向的曲面偏移量
        normal_nudge: float = 1e-3,** kwargs
    ):
        self.u_range = u_range          # 参数u的取值范围
        self.v_range = v_range          # 参数v的取值范围
        self.resolution = resolution    # 采样分辨率 (u方向点数, v方向点数)
        self.prefered_creation_axis = prefered_creation_axis  # 首选创建轴
        self.epsilon = epsilon          # 微小量，用于计算偏导数
        self.normal_nudge = normal_nudge  # 法向量计算的偏移量

        super().__init__(
            **kwargs,
            color=color,
            shading=shading,
            depth_test=depth_test,
        )
        self.compute_triangle_indices()  # 计算三角形索引

    def uv_func(self, u: float, v: float) -> tuple[float, float, float]:
        """
        参数化曲面函数，将(u, v)参数映射到3D空间中的点
        
        子类需要实现此方法来定义具体的曲面形状
        默认返回平面上的点 (u, v, 0)
        """
        return (u, v, 0.0)

    @Mobject.affects_data
    def init_points(self):
        """初始化曲面的顶点数据，包括顶点坐标和法向量"""
        dim = self.dim  # 维度，通常为3
        nu, nv = self.resolution  # u和v方向的采样点数量
        # 生成u和v方向的均匀采样点
        u_range = np.linspace(*self.u_range, nu)
        v_range = np.linspace(*self.v_range, nv)

        # 创建三个网格：
        # - 原始uv值生成的网格
        # - u方向偏移epsilon的网格
        # - v方向偏移epsilon的网格
        uv_grid = np.array([[[u, v] for v in v_range] for u in u_range])
        uv_plus_du = uv_grid.copy()
        uv_plus_du[:, :, 0] += self.epsilon  # u方向偏移
        uv_plus_dv = uv_grid.copy()
        uv_plus_dv[:, :, 1] += self.epsilon  # v方向偏移

        # 计算三个网格对应的3D点
        points, du_points, dv_points = [
            np.apply_along_axis(
                lambda p: self.uv_func(*p), 2, grid
            ).reshape((nu * nv, dim))
            for grid in (uv_grid, uv_plus_du, uv_plus_dv)
        ]
        # 计算法向量：通过偏导数叉乘得到
        crosses = cross(du_points - points, dv_points - points)
        normals = normalize_along_axis(crosses, 1)  # 归一化法向量

        # 设置顶点数据和法向量偏移点
        self.set_points(points)
        self.data['d_normal_point'] = points + self.normal_nudge * normals

    def uv_to_point(self, u, v):
        """
        将(u, v)参数映射到曲面上的点，使用双线性插值
        
        参数:
            u: 参数u的值
            v: 参数v的值
        返回:
            曲面上对应点的3D坐标
        """
        nu, nv = self.resolution
        # 将顶点数据重塑为网格形式
        uv_grid = np.reshape(self.get_points(), (nu, nv, self.dim))

        # 计算u和v在采样范围内的归一化位置
        alpha1 = clip(inverse_interpolate(*self.u_range[:2], u), 0, 1)
        alpha2 = clip(inverse_interpolate(*self.v_range[:2], v), 0, 1)
        scaled_u = alpha1 * (nu - 1)
        scaled_v = alpha2 * (nv - 1)
        # 找到周围的四个采样点
        u_int = int(scaled_u)
        v_int = int(scaled_v)
        u_int_plus = min(u_int + 1, nu - 1)
        v_int_plus = min(v_int + 1, nv - 1)

        # 四个相邻点
        a = uv_grid[u_int, v_int, :]
        b = uv_grid[u_int, v_int_plus, :]
        c = uv_grid[u_int_plus, v_int, :]
        d = uv_grid[u_int_plus, v_int_plus, :]

        # 双线性插值计算最终点
        u_res = scaled_u % 1
        v_res = scaled_v % 1
        return interpolate(
            interpolate(a, b, v_res),
            interpolate(c, d, v_res),
            u_res
        )

    def apply_points_function(self, *args, **kwargs) -> Self:
        """应用点变换函数后更新法向量"""
        super().apply_points_function(*args, **kwargs)
        self.get_unit_normals()
        return self

    def compute_triangle_indices(self) -> np.ndarray:
        """
        计算三角形索引数组，用于将顶点连接成三角形
        
        返回:
            三角形索引数组，每个三角形由3个顶点索引组成
        """
        # TODO: 如果曲面分辨率发生变化，需要确保调用此方法
        nu, nv = self.resolution
        if nu == 0 or nv == 0:
            self.triangle_indices = np.zeros(0, dtype=int)
            return self.triangle_indices
        # 创建索引网格
        index_grid = np.arange(nu * nv).reshape((nu, nv))
        # 每个小矩形分成两个三角形，共6个索引
        indices = np.zeros(6 * (nu - 1) * (nv - 1), dtype=int)
        indices[0::6] = index_grid[:-1, :-1].flatten()  # 左上
        indices[1::6] = index_grid[+1:, :-1].flatten()  # 左下
        indices[2::6] = index_grid[:-1, +1:].flatten()  # 右上
        indices[3::6] = index_grid[:-1, +1:].flatten()  # 右上
        indices[4::6] = index_grid[+1:, :-1].flatten()  # 左下
        indices[5::6] = index_grid[+1:, +1:].flatten()  # 右下
        self.triangle_indices = indices
        return self.triangle_indices

    def get_triangle_indices(self) -> np.ndarray:
        """获取三角形索引数组"""
        return self.triangle_indices

    def get_unit_normals(self) -> Vect3Array:
        """
        计算单位法向量数组
        
        TODO: 可以尝试使用相邻网格值计算更可靠的法向量
        """
        return normalize_along_axis(self.data['d_normal_point'] - self.data['point'], 1)

    @Mobject.affects_data
    def pointwise_become_partial(
        self,
        smobject: "Surface",
        a: float,
        b: float,
        axis: int | None = None
    ) -> Self:
        """
        部分地成为另一个曲面的一部分，用于动画过渡
        
        参数:
            smobject: 目标曲面
            a, b: 部分范围 [0, 1]
            axis: 操作轴，默认为首选创建轴
        """
        assert isinstance(smobject, Surface)
        if axis is None:
            axis = self.prefered_creation_axis
        if a <= 0 and b >= 1:
            self.match_points(smobject)
            return self

        nu, nv = smobject.resolution
        self.data['point'][:] = self.get_partial_points_array(
            smobject.data['point'], a, b,
            (nu, nv, 3),
            axis=axis
        )
        return self

    def get_partial_points_array(
        self,
        points: Vect3Array,
        a: float,
        b: float,
        resolution: Sequence[int],
        axis: int
    ) -> Vect3Array:
        """
        获取部分点数组，用于pointwise_become_partial方法
        
        参数:
            points: 原始点数组
            a, b: 部分范围 [0, 1]
            resolution: 分辨率 (nu, nv, 3)
            axis: 操作轴
        返回:
            部分点数组
        """
        if len(points) == 0:
            return points
        nu, nv = resolution[:2]
        points = points.reshape(resolution).copy()
        max_index = resolution[axis] - 1
        # 计算范围对应的索引
        lower_index, lower_residue = integer_interpolate(0, max_index, a)
        upper_index, upper_residue = integer_interpolate(0, max_index, b)
        
        if axis == 0:  # u方向
            # 处理下界
            points[:lower_index] = interpolate(
                points[lower_index],
                points[lower_index + 1],
                lower_residue
            )
            # 处理上界
            points[upper_index + 1:] = interpolate(
                points[upper_index],
                points[upper_index + 1],
                upper_residue
            )
        else:  # v方向
            shape = (nu, 1, resolution[2])
            # 处理下界
            points[:, :lower_index] = interpolate(
                points[:, lower_index],
                points[:, lower_index + 1],
                lower_residue
            ).reshape(shape)
            # 处理上界
            points[:, upper_index + 1:] = interpolate(
                points[:, upper_index],
                points[:, upper_index + 1],
                upper_residue
            ).reshape(shape)
        return points.reshape((nu * nv, *resolution[2:]))

    @Mobject.affects_data
    def sort_faces_back_to_front(self, vect: Vect3 = OUT) -> Self:
        """
        按指定方向向量对三角形面进行排序（从后到前），用于处理透明效果
        
        参数:
            vect: 排序参考方向向量
        """
        tri_is = self.triangle_indices
        points = self.get_points()

        # 计算每个三角形中心与方向向量的点积，用于排序
        dots = (points[tri_is[::3]] * vect).sum(1)
        indices = np.argsort(dots)
        # 按排序结果重新排列三角形索引
        for k in range(3):
            tri_is[k::3] = tri_is[k::3][indices]
        return self

    def always_sort_to_camera(self, camera: Camera) -> Self:
        """
        添加更新器，使曲面始终面向相机排序（处理透明效果）
        
        参数:
            camera: 相机对象
        """
        def updater(surface: Surface):
            vect = camera.get_location() - surface.get_center()
            surface.sort_faces_back_to_front(vect)
        self.add_updater(updater)
        return self

    def get_shader_vert_indices(self) -> np.ndarray:
        """获取着色器使用的顶点索引"""
        return self.get_triangle_indices()


class ParametricSurface(Surface):
    """
    参数化曲面类，继承自Surface
    
    可通过传入uv_func函数来定义任意参数化曲面
    """
    def __init__(
        self,
        uv_func: Callable[[float, float], Iterable[float]],
        u_range: tuple[float, float] = (0, 1),
        v_range: tuple[float, float] = (0, 1),
        **kwargs
    ):
        self.passed_uv_func = uv_func  # 传入的参数化函数
        super().__init__(u_range=u_range, v_range=v_range,** kwargs)

    def uv_func(self, u, v):
        """实现父类的uv_func，调用传入的参数化函数"""
        return self.passed_uv_func(u, v)


class SGroup(Surface):
    """
    曲面组类，用于组合多个曲面
    
    继承自Surface，但实际上作为容器使用
    """
    def __init__(
        self,
        *parametric_surfaces: Surface,
        **kwargs
    ):
        super().__init__(resolution=(0, 0), **kwargs)
        self.add(*parametric_surfaces)  # 添加曲面到组中

    def init_points(self):
        """重写初始化点方法，无需执行任何操作"""
        pass  # Needed?


class TexturedSurface(Surface):
    """
    带纹理的曲面类，继承自Surface
    
    可将图像纹理映射到曲面上
    """
    shader_folder: str = "textured_surface"  # 纹理着色器文件夹
    # 数据类型定义，包含纹理坐标和透明度
    data_dtype: Sequence[Tuple[str, type, Tuple[int]]] = [
        ('point', np.float32, (3,)),          # 顶点坐标
        ('d_normal_point', np.float32, (3,)), # 法向量相关点
        ('im_coords', np.float32, (2,)),      # 图像纹理坐标
        ('opacity', np.float32, (1,)),        # 透明度
    ]

    def __init__(
        self,
        uv_surface: Surface,
        image_file: str,
        dark_image_file: str | None = None,** kwargs
    ):
        if not isinstance(uv_surface, Surface):
            raise Exception("uv_surface must be of type Surface")
        # 设置纹理信息
        if dark_image_file is None:
            dark_image_file = image_file
            self.num_textures = 1  # 单纹理
        else:
            self.num_textures = 2  # 双纹理（明亮和黑暗版本）

        # 纹理文件路径
        texture_paths = {
            "LightTexture": get_full_raster_image_path(image_file),
            "DarkTexture": get_full_raster_image_path(dark_image_file),
        }

        self.uv_surface = uv_surface  # 基础曲面
        self.uv_func = uv_surface.uv_func  # 复用基础曲面的参数化函数
        self.u_range: Tuple[float, float] = uv_surface.u_range
        self.v_range: Tuple[float, float] = uv_surface.v_range
        self.resolution: Tuple[int, int] = uv_surface.resolution
        super().__init__(
            texture_paths=texture_paths,
            shading=tuple(uv_surface.shading),
            **kwargs
        )

    @Mobject.affects_data
    def init_points(self):
        """初始化带纹理曲面的顶点数据，包括纹理坐标"""
        surf = self.uv_surface
        nu, nv = surf.resolution
        self.resize_points(surf.get_num_points())
        self.resolution = surf.resolution
        # 复制基础曲面的顶点数据
        self.data['point'][:] = surf.data['point']
        self.data['d_normal_point'][:] = surf.data['d_normal_point']
        self.data['opacity'][:, 0] = surf.data["rgba"][:, 3]
        # 设置纹理坐标（图像坐标）
        self.data["im_coords"] = np.array([
            [u, v]
            for u in np.linspace(0, 1, nu)
            for v in np.linspace(1, 0, nv)  # 反转y方向，因为图像坐标系原点在左上角
        ])

    def init_uniforms(self):
        """初始化着色器 uniforms，设置纹理数量"""
        super().init_uniforms()
        self.uniforms["num_textures"] = self.num_textures

    @Mobject.affects_data
    def set_opacity(self, opacity: float | Iterable[float], recurse=True) -> Self:
        """设置透明度"""
        op_arr = np.array(listify(opacity))
        self.data["opacity"][:, 0] = resize_with_interpolation(op_arr, len(self.data))
        return self

    def set_color(
        self,
        color: ManimColor | Iterable[ManimColor] | None,
        opacity: float | Iterable[float] | None = None,
        recurse: bool = True
    ) -> Self:
        """设置颜色（主要用于设置透明度）"""
        if opacity is not None:
            self.set_opacity(opacity)
        return self

    def pointwise_become_partial(
        self,
        tsmobject: "TexturedSurface",
        a: float,
        b: float,
        axis: int = 1
    ) -> Self:
        """部分地成为另一个带纹理曲面的一部分，同时处理纹理坐标"""
        super().pointwise_become_partial(tsmobject, a, b, axis)
        im_coords = self.data["im_coords"]
        im_coords[:] = tsmobject.data["im_coords"]
        if a <= 0 and b >= 1:
            return self
        nu, nv = tsmobject.resolution
        # 更新纹理坐标的部分区域
        im_coords[:] = self.get_partial_points_array(
            im_coords, a, b, (nu, nv, 2), axis
        )
        return self