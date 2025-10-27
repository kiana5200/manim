from __future__ import annotations

import moderngl  # 现代OpenGL库，用于硬件加速渲染
import numpy as np  # 数值计算库，用于处理点坐标等数据

from manimlib.constants import GREY_C, YELLOW  # 导入默认颜色常量
from manimlib.constants import ORIGIN, NULL_POINTS  # 导入原点和空点常量
from manimlib.mobject.mobject import Mobject  # 基础图形对象类
from manimlib.mobject.types.point_cloud_mobject import PMobject  # 点云图形对象基类
from manimlib.utils.iterables import resize_with_interpolation  # 用于插值调整数组大小的工具函数

from typing import TYPE_CHECKING  # 用于类型检查的常量

if TYPE_CHECKING:
    # 类型提示相关导入（仅在类型检查时生效）
    import numpy.typing as npt
    from typing import Sequence, Tuple
    from manimlib.typing import ManimColor, Vect3, Vect3Array, Self


# 默认参数定义
DEFAULT_DOT_RADIUS = 0.05  # 默认点半径
DEFAULT_GLOW_DOT_RADIUS = 0.2  # 默认发光点半径
DEFAULT_GRID_HEIGHT = 6  # 默认网格高度
DEFAULT_BUFF_RATIO = 0.5  # 默认缓冲比例


class DotCloud(PMobject):
    """点云类，用于渲染大量点的集合"""
    shader_folder: str = "true_dot"  # 着色器文件夹
    render_primitive: int = moderngl.POINTS  # 渲染原语类型（OpenGL的点图元）
    # 数据类型定义：包含点坐标、半径和颜色信息
    data_dtype: Sequence[Tuple[str, type, Tuple[int]]] = [
        ('point', np.float32, (3,)),  # 点坐标（3D）
        ('radius', np.float32, (1,)),  # 点半径
        ('rgba', np.float32, (4,)),  # 点颜色（RGBA）
    ]

    def __init__(
        self,
        points: Vect3Array = NULL_POINTS,  # 点集合
        color: ManimColor = GREY_C,  # 点颜色
        opacity: float = 1.0,  # 不透明度
        radius: float = DEFAULT_DOT_RADIUS,  # 点半径
        glow_factor: float = 0.0,  # 发光因子
        anti_alias_width: float = 2.0,  # 抗锯齿宽度
        **kwargs
    ):
        self.radius = radius  # 存储半径
        self.glow_factor = glow_factor  # 存储发光因子
        self.anti_alias_width = anti_alias_width  # 存储抗锯齿宽度

        # 调用父类构造函数
        super().__init__(
            color=color,
            opacity=opacity,** kwargs
        )
        self.set_radius(self.radius)  # 设置点半径

        if points is not None:
            self.set_points(points)  # 设置点坐标

    def init_uniforms(self) -> None:
        """初始化着色器 uniforms 变量"""
        super().init_uniforms()  # 调用父类方法
        # 向着色器传递发光因子和抗锯齿宽度
        self.uniforms["glow_factor"] = self.glow_factor
        self.uniforms["anti_alias_width"] = self.anti_alias_width

    def to_grid(
        self,
        n_rows: int,  # 行数
        n_cols: int,  # 列数
        n_layers: int = 1,  # 层数（3D网格）
        buff_ratio: float | None = None,  # 缓冲比例（统一设置三个方向）
        h_buff_ratio: float = 1.0,  # 水平方向缓冲比例
        v_buff_ratio: float = 1.0,  # 垂直方向缓冲比例
        d_buff_ratio: float = 1.0,  # 深度方向缓冲比例
        height: float = DEFAULT_GRID_HEIGHT,  # 网格高度
    ) -> Self:
        """将点云排列成网格形状"""
        n_points = n_rows * n_cols * n_layers  # 总点数
        # 生成网格点坐标（先按索引生成，再转换为网格坐标）
        points = np.repeat(range(n_points), 3, axis=0).reshape((n_points, 3))
        points[:, 0] = points[:, 0] % n_cols  # X坐标（列索引）
        points[:, 1] = (points[:, 1] // n_cols) % n_rows  # Y坐标（行索引）
        points[:, 2] = points[:, 2] // (n_rows * n_cols)  # Z坐标（层索引）
        self.set_points(points.astype(float))  # 设置点坐标为浮点数

        # 如果指定了统一缓冲比例，则覆盖三个方向的缓冲比例
        if buff_ratio is not None:
            v_buff_ratio = buff_ratio
            h_buff_ratio = buff_ratio
            d_buff_ratio = buff_ratio

        radius = self.get_radius()  # 获取当前点半径
        ns = [n_cols, n_rows, n_layers]  # 三个方向的网格数量
        brs = [h_buff_ratio, v_buff_ratio, d_buff_ratio]  # 三个方向的缓冲比例
        self.set_radius(0)  # 临时将半径设为0，避免影响缩放计算
        # 按每个方向调整大小，考虑缓冲和半径
        for n, br, dim in zip(ns, brs, range(3)):
            self.rescale_to_fit(2 * radius * (1 + br) * (n - 1), dim, stretch=True)
        self.set_radius(radius)  # 恢复半径
        if height is not None:
            self.set_height(height)  # 设置网格高度
        self.center()  # 将网格居中
        return self

    @Mobject.affects_data
    def set_radii(self, radii: npt.ArrayLike) -> Self:
        """设置每个点的半径（支持不同点不同半径）"""
        n_points = self.get_num_points()  # 获取点数量
        radii = np.array(radii).reshape((len(radii), 1))  # 重塑半径数组形状
        # 插值调整半径数组大小以匹配点数量
        self.data["radius"][:] = resize_with_interpolation(radii, n_points)
        self.refresh_bounding_box()  # 刷新边界框
        return self

    def get_radii(self) -> np.ndarray:
        """获取所有点的半径"""
        return self.data["radius"]

    @Mobject.affects_data
    def set_radius(self, radius: float) -> Self:
        """设置所有点的统一半径"""
        # 如果有点数据则使用现有数据，否则使用默认数据
        data = self.data if self.get_num_points() > 0 else self._data_defaults
        data["radius"][:] = radius  # 设置所有点的半径
        self.refresh_bounding_box()  # 刷新边界框
        return self

    def get_radius(self) -> float:
        """获取点的最大半径（代表整体半径）"""
        return self.get_radii().max()

    def scale_radii(self, scale_factor: float) -> Self:
        """按比例缩放所有点的半径"""
        self.set_radius(scale_factor * self.get_radii())
        return self

    def set_glow_factor(self, glow_factor: float) -> Self:
        """设置发光因子"""
        self.uniforms["glow_factor"] = glow_factor
        return self

    def get_glow_factor(self) -> float:
        """获取当前发光因子"""
        return self.uniforms["glow_factor"]

    def compute_bounding_box(self) -> Vect3Array:
        """计算点云的边界框（考虑点半径）"""
        bb = super().compute_bounding_box()  # 调用父类方法获取基础边界框
        radius = self.get_radius()  # 获取点半径
        # 扩展边界框以包含点的半径
        bb[0] += np.full((3,), -radius)
        bb[2] += np.full((3,), radius)
        return bb

    def scale(
        self,
        scale_factor: float | npt.ArrayLike,
        scale_radii: bool = True,
        **kwargs
    ) -> Self:
        """缩放点云（可选择是否缩放点半径）"""
        super().scale(scale_factor,** kwargs)  # 调用父类方法缩放点坐标
        if scale_radii:
            self.set_radii(scale_factor * self.get_radii())  # 按比例缩放半径
        return self

    def make_3d(
        self,
        reflectiveness: float = 0.5,  # 反射率
        gloss: float = 0.1,  # 光泽度
        shadow: float = 0.2  # 阴影强度
    ) -> Self:
        """将点云设置为3D效果（添加光照和深度测试）"""
        self.set_shading(reflectiveness, gloss, shadow)  # 设置着色参数
        self.apply_depth_test()  # 应用深度测试
        return self


class TrueDot(DotCloud):
    """单个点类（继承自点云，仅包含一个点）"""
    def __init__(self, center: Vect3 = ORIGIN, **kwargs):
        # 调用父类构造函数，点集合设为单个中心点
        super().__init__(points=np.array([center]),** kwargs)


class GlowDots(DotCloud):
    """发光点云类（预设置发光相关参数）"""
    def __init__(
        self,
        points: Vect3Array = NULL_POINTS,
        color: ManimColor = YELLOW,  # 默认黄色
        radius: float = DEFAULT_GLOW_DOT_RADIUS,  # 默认发光点半径
        glow_factor: float = 2.0,  # 默认发光因子
        **kwargs,
    ):
        # 调用父类构造函数，设置发光相关默认参数
        super().__init__(
            points,
            color=color,
            radius=radius,
            glow_factor=glow_factor,** kwargs,
        )


class GlowDot(GlowDots):
    """单个发光点类（继承自发光点云，仅包含一个点）"""
    def __init__(self, center: Vect3 = ORIGIN, **kwargs):
        # 调用父类构造函数，点集合设为单个中心点
        super().__init__(points=np.array([center]),** kwargs)