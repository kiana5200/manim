from __future__ import annotations

import numpy as np

from manimlib.mobject.mobject import Mobject
from manimlib.utils.color import color_gradient
from manimlib.utils.color import color_to_rgba
from manimlib.utils.iterables import resize_with_interpolation

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable
    from manimlib.typing import ManimColor, Vect3, Vect3Array, Vect4Array, Self


class PMobject(Mobject):
    """点云基类，继承自Mobject，用于处理点集合的相关操作"""
    
    def set_points(self, points: Vect3Array):
        """
        设置点云的点集合
        
        参数:
            points: 三维点的数组，形状为(N, 3)
        """
        if len(points) == 0:
            points = np.zeros((0, 3))
        super().set_points(points)
        self.resize_points(len(points))
        return self

    def add_points(
        self,
        points: Vect3Array,
        rgbas: Vect4Array | None = None,
        color: ManimColor | None = None,
        opacity: float | None = None
    ) -> Self:
        """
        向点云中添加多个点
        
        参数:
            points: 要添加的三维点数组，形状为(N, 3)
            rgbas: 每个点对应的RGBA颜色数组，形状为(N, 4)，若为None则使用默认颜色
            color: 所有点的统一颜色，若指定则忽略rgbas
            opacity: 不透明度，若指定则应用于所有点
            
        说明:
            若指定color，则会为所有添加的点生成相同的RGBA值
        """
        self.append_points(points)
        # rgbas数组会随points一起调整大小
        if color is not None:
            if opacity is None:
                opacity = self.data["rgba"][-1, 3]
            rgbas = np.repeat(
                [color_to_rgba(color, opacity)],
                len(points),
                axis=0
            )
        if rgbas is not None:
            self.data["rgba"][-len(rgbas):] = rgbas
        return self

    def add_point(self, point: Vect3, rgba=None, color=None, opacity=None) -> Self:
        """
        向点云中添加单个点
        
        参数:
            point: 单个三维点，形状为(3,)
            rgba: 该点的RGBA颜色值，形状为(4,)
            color: 点的颜色
            opacity: 点的不透明度
        """
        rgbas = None if rgba is None else [rgba]
        self.add_points([point], rgbas, color, opacity)
        return self

    @Mobject.affects_data
    def set_color_by_gradient(self, *colors: ManimColor) -> Self:
        """
        按渐变颜色设置所有点的颜色
        
        参数:
            *colors: 渐变的颜色序列
        """
        self.data["rgba"][:] = np.array(list(map(
            color_to_rgba,
            color_gradient(colors, self.get_num_points())
        )))
        return self

    @Mobject.affects_data
    def match_colors(self, pmobject: PMobject) -> Self:
        """
        匹配另一个点云的颜色
        
        参数:
            pmobject: 要匹配颜色的点云对象
        """
        self.data["rgba"][:] = resize_with_interpolation(
            pmobject.data["rgba"], self.get_num_points()
        )
        return self

    @Mobject.affects_data
    def filter_out(self, condition: Callable[[np.ndarray], bool]) -> Self:
        """
        根据条件过滤掉点
        
        参数:
            condition: 过滤条件函数，输入点坐标，返回True则保留该点
        """
        for mob in self.family_members_with_points():
            # 对每个点应用条件，取反后作为索引筛选
            mob.data = mob.data[~np.apply_along_axis(condition, 1, mob.get_points())]
        return self

    @Mobject.affects_data
    def sort_points(self, function: Callable[[Vect3], None] = lambda p: p[0]) -> Self:
        """
        按指定函数对点数进行排序
        
        参数:
            function: 排序依据函数，输入点坐标，返回一个数值用于排序
                      默认按x坐标(p[0])排序
        """
        for mob in self.family_members_with_points():
            # 计算每个点的排序键值并获取排序索引
            indices = np.argsort(
                np.apply_along_axis(function, 1, mob.get_points())
            )
            mob.data[:] = mob.data[indices]
        return self

    @Mobject.affects_data
    def ingest_submobjects(self) -> Self:
        """将所有子对象的数据合并到当前对象"""
        self.data = np.vstack([
            sm.data for sm in self.get_family()
        ])
        return self

    def point_from_proportion(self, alpha: float) -> np.ndarray:
        """
        根据比例获取对应位置的点
        
        参数:
            alpha: 比例值，范围[0, 1]
            
        返回:
            对应位置的点坐标
        """
        index = alpha * (self.get_num_points() - 1)
        return self.get_points()[int(index)]

    @Mobject.affects_data
    def pointwise_become_partial(self, pmobject: PMobject, a: float, b: float) -> Self:
        """
        从另一个点云中截取部分点作为当前点云的数据
        
        参数:
            pmobject: 源点云对象
            a: 起始比例，范围[0, 1]
            b: 结束比例，范围[0, 1]
        """
        lower_index = int(a * pmobject.get_num_points())
        upper_index = int(b * pmobject.get_num_points())
        self.data = pmobject.data[lower_index:upper_index].copy()
        return self


class PGroup(PMobject):
    """点云组类，用于管理多个PMobject对象的集合"""
    
    def __init__(self, *pmobs: PMobject, **kwargs):
        """
        初始化点云组
        
        参数:
            *pmobs: 多个PMobject对象
            **kwargs: 传递给父类的参数
            
        异常:
            若有非PMobject类型的对象则抛出异常
        """
        if not all([isinstance(m, PMobject) for m in pmobs]):
            raise Exception("所有子对象必须是PMobject类型")
        super().__init__(** kwargs)
        self.add(*pmobs)