from __future__ import annotations

import numpy as np
import moderngl  # 现代OpenGL绑定库，用于图形渲染
from PIL import Image  # 用于处理图像文件

from manimlib.constants import DL, DR, UL, UR  # 导入角落位置常量（左下、右下、左上、右上）
from manimlib.mobject.mobject import Mobject  # 导入基础图形对象类
from manimlib.utils.bezier import inverse_interpolate  # 用于计算反向插值
from manimlib.utils.images import get_full_raster_image_path  # 获取图像的完整路径
from manimlib.utils.iterables import listify, resize_with_interpolation  # 处理可迭代对象的工具函数

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Sequence, Tuple
    from manimlib.typing import Vect3  # 类型提示：三维向量


class ImageMobject(Mobject):
    """图像基元类，用于在场景中显示图像"""
    shader_folder: str = "image"  # 着色器文件夹路径
    # 数据类型定义：顶点数据的结构
    data_dtype: Sequence[Tuple[str, type, Tuple[int]]] = [
        ('point', np.float32, (3,)),  # 顶点位置（三维）
        ('im_coords', np.float32, (2,)),  # 图像纹理坐标（二维）
        ('opacity', np.float32, (1,)),  # 不透明度
    ]
    render_primitive: int = moderngl.TRIANGLES  # 渲染图元类型：三角形

    def __init__(
        self,
        filename: str,
        height: float = 4.0,
        **kwargs
    ):
        """
        初始化图像对象
        
        参数:
            filename: 图像文件名称
            height: 图像显示高度（默认4.0）
            **kwargs: 传递给父类Mobject的其他参数
        """
        self.height = height
        # 获取图像的完整路径
        self.image_path = get_full_raster_image_path(filename)
        # 打开图像文件
        self.image = Image.open(self.image_path)
        # 调用父类构造函数，设置纹理路径
        super().__init__(texture_paths={"Texture": self.image_path},** kwargs)

    def init_data(self) -> None:
        """初始化顶点数据"""
        super().init_data(length=6)  # 6个顶点（组成两个三角形）
        # 设置顶点位置：两个三角形组成一个矩形（左上、左下、右上、右下、右上、左下）
        self.data["point"][:] = [UL, DL, UR, DR, UR, DL]
        # 设置纹理坐标：对应图像的四个角落
        self.data["im_coords"][:] = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 0), (0, 1)]
        # 设置初始不透明度
        self.data["opacity"][:] = self.opacity

    def init_points(self) -> None:
        """初始化顶点位置，设置图像宽高比例"""
        size = self.image.size  # 获取图像原始尺寸（宽, 高）
        # 根据原始宽高比设置宽度
        self.set_width(2 * size[0] / size[1], stretch=True)
        # 设置图像高度
        self.set_height(self.height)

    @Mobject.affects_data
    def set_opacity(self, opacity: float, recurse: bool = True):
        """
        设置图像不透明度
        
        参数:
            opacity: 不透明度值（0-1之间）
            recurse: 是否递归应用到子对象（默认True）
        """
        # 调整不透明度数据的尺寸并赋值
        self.data["opacity"][:, 0] = resize_with_interpolation(
            np.array(listify(opacity)),
            self.get_num_points()
        )
        return self

    def set_color(self, color, opacity=None, recurse=None):
        """
        图像对象不支持设置颜色，重写父类方法
        
        注：图像颜色由纹理本身决定，因此该方法不做任何操作
        """
        return self

    def point_to_rgb(self, point: Vect3) -> Vect3:
        """
        根据给定点的位置，获取图像对应位置的RGB颜色值
        
        参数:
            point: 要采样颜色的三维点坐标
            
        返回:
            归一化的RGB颜色值（0-1之间）
        """
        # 获取图像左上角和右下角的二维坐标
        x0, y0 = self.get_corner(UL)[:2]
        x1, y1 = self.get_corner(DR)[:2]
        # 计算点在图像范围内的归一化坐标（0-1之间）
        x_alpha = inverse_interpolate(x0, x1, point[0])
        y_alpha = inverse_interpolate(y0, y1, point[1])
        # 检查点是否在图像范围内
        if not (0 <= x_alpha <= 1) and (0 <= y_alpha <= 1):
            raise Exception("Cannot sample color from outside an image")  # 点在图像外时抛出异常

        # 获取图像像素尺寸
        pw, ph = self.image.size
        # 计算对应像素坐标并获取RGB值
        rgb = self.image.getpixel((
            int((pw - 1) * x_alpha),  # x方向像素坐标
            int((ph - 1) * y_alpha),  # y方向像素坐标
        ))[:3]  # 取前三个通道（RGB）
        # 将RGB值归一化到0-1范围并返回
        return np.array(rgb) / 255