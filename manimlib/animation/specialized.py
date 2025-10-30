# 从__future__模块导入annotations，用于支持 postponed evaluation of type annotations（延迟类型注释评估）
# 这允许在类型提示中使用尚未定义的类
from __future__ import annotations

# 从manimlib.animation.composition模块导入LaggedStart类
# LaggedStart用于创建一个动画组合，其中子动画按顺序延迟启动
from manimlib.animation.composition import LaggedStart

# 从manimlib.animation.transform模块导入Restore类
# Restore动画用于将一个对象恢复到之前保存的状态
from manimlib.animation.transform import Restore

# 从manimlib.constants模块导入BLACK和WHITE常量
# 这些是预定义的颜色常量，分别代表黑色和白色
from manimlib.constants import BLACK, WHITE

# 从manimlib.mobject.geometry模块导入Circle类
# Circle用于创建圆形几何对象
from manimlib.mobject.geometry import Circle

# 从manimlib.mobject.types.vectorized_mobject模块导入VGroup类
# VGroup是向量图形对象的容器，用于组合多个mobject（可移动对象）
from manimlib.mobject.types.vectorized_mobject import VGroup

# 从typing模块导入TYPE_CHECKING常量
# TYPE_CHECKING在运行时为False，仅在类型检查时为True，用于条件导入类型提示
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from manimlib.typing import ManimColor


class Broadcast(LaggedStart):
    """
    广播动画类，继承自LaggedStart（延迟启动的动画组合）
    用于创建从焦点向外扩散的圆形波纹效果
    """
    def __init__(
        self,
        focal_point: np.ndarray,  # 动画的焦点/中心点，numpy数组表示坐标
        small_radius: float = 0.0,  # 圆形初始半径
        big_radius: float = 5.0,    # 圆形最终半径
        n_circles: int = 5,         # 波纹的数量（圆的个数）
        start_stroke_width: float = 8.0,  # 圆形初始线宽
        color: ManimColor = WHITE,  # 波纹的颜色
        run_time: float = 3.0,      # 动画总时长
        lag_ratio: float = 0.2,     # 每个子动画之间的延迟比例
        remover: bool = True,       # 动画结束后是否移除图形
        **kwargs                    # 其他传递给父类的参数
    ):
        # 保存初始化参数为实例属性
        self.focal_point = focal_point
        self.small_radius = small_radius
        self.big_radius = big_radius
        self.n_circles = n_circles
        self.start_stroke_width = start_stroke_width
        self.color = color

        # 创建一个向量图形组，用于管理所有波纹圆形
        circles = VGroup()
        # 循环创建指定数量的圆形
        for x in range(n_circles):
            # 创建一个圆形，初始设置为黑色（不可见）且线宽为0
            circle = Circle(
                radius=big_radius,
                stroke_color=BLACK,
                stroke_width=0,
            )
            # 为圆形添加更新器，确保它始终保持在焦点位置
            circle.add_updater(lambda c: c.move_to(focal_point))
            # 保存圆形当前状态（大半径、黑色、线宽0的状态）
            circle.save_state()
            # 将圆形设置为初始大小（小半径）
            circle.set_width(small_radius * 2)  # 直径 = 半径*2
            # 设置圆形的初始可见样式（指定颜色和线宽）
            circle.set_stroke(color, start_stroke_width)
            # 将圆形添加到图形组中
            circles.add(circle)
        
        # 调用父类LaggedStart的初始化方法
        # 使用map函数为每个圆形创建Restore动画（恢复到之前保存的状态）
        super().__init__(
            *map(Restore, circles),  # 展开所有圆形的Restore动画作为参数
            run_time=run_time,       # 传递总时长
            lag_ratio=lag_ratio,     # 传递延迟比例
            remover=remover,         # 传递是否移除的参数
            **kwargs                 # 传递其他参数
        )
