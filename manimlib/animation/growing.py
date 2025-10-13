# 启用Python 3.7+的注解向前兼容支持，允许在类型注解中使用尚未定义的类
from __future__ import annotations

# 从Manim的变换动画模块导入Transform类，用于对象间的变换动画
from manimlib.animation.transform import Transform

# 导入类型检查相关模块
from typing import TYPE_CHECKING

# 条件导入，仅在类型检查时执行（运行时不生效）
# 用于解决循环导入问题，同时提供完整的类型提示支持
if TYPE_CHECKING:
    import numpy as np  # 导入numpy库，用于数值计算相关的类型提示
    
    from manimlib.mobject.geometry import Arrow  # 导入箭头图形类Arrow
    from manimlib.mobject.mobject import Mobject  # 导入基础可移动对象类Mobject
    from manimlib.typing import ManimColor  # 导入Manim颜色类型ManimColor


# 定义GrowFromPoint类，继承自Transform，用于实现物体从指定点生长出来的动画效果
class GrowFromPoint(Transform):
    # 初始化方法，设置从指定点生长的相关参数
    def __init__(
        self,
        mobject: Mobject,  # 要进行生长动画的Mobject对象
        point: np.ndarray,  # 生长的起始点（坐标）
        point_color: ManimColor = None,  # 生长点的颜色，默认为None（使用物体自身颜色）
        **kwargs  # 其他关键字参数，传递给父类Transform
    ):
        self.point = point  # 保存生长起始点到实例变量
        self.point_color = point_color  # 保存生长点颜色到实例变量
        # 调用父类Transform的初始化方法，传递必要参数
        super().__init__(mobject,** kwargs)

    # 创建目标状态的方法，定义动画结束时物体的状态
    def create_target(self) -> Mobject:
        # 返回原物体的副本作为目标状态（即生长结束时显示原物体的状态）
        return self.mobject.copy()

    # 创建起始状态的方法，定义动画开始时物体的状态
    def create_starting_mobject(self) -> Mobject:
        # 调用父类的方法获取基础起始状态
        start = super().create_starting_mobject()
        # 将起始状态的物体缩放到0（变成一个点）
        start.scale(0)
        # 将缩放后的点移动到指定的生长起始点
        start.move_to(self.point)
        # 如果指定了生长点颜色，则设置起始状态物体的颜色
        if self.point_color is not None:
            start.set_color(self.point_color)
        # 返回配置好的起始状态物体
        return start


# 定义GrowFromCenter类，继承自GrowFromPoint，实现从物体中心生长的动画
class GrowFromCenter(GrowFromPoint):
    # 初始化方法，设置从中心生长的参数
    def __init__(self, mobject: Mobject, **kwargs):
        # 计算物体的中心坐标作为生长起始点
        point = mobject.get_center()
        # 调用父类GrowFromPoint的初始化方法，传入物体和中心坐标
        super().__init__(mobject, point, **kwargs)


# 定义GrowFromEdge类，继承自GrowFromPoint，实现从物体边缘生长的动画
class GrowFromEdge(GrowFromPoint):
    # 初始化方法，设置从指定边缘生长的参数
    def __init__(self, mobject: Mobject, edge: np.ndarray, **kwargs):
        # 根据指定的边缘方向，获取物体边界框上的对应点作为生长起始点
        # edge参数通常是方向向量，如UP、DOWN、LEFT、RIGHT等
        point = mobject.get_bounding_box_point(edge)
        # 调用父类GrowFromPoint的初始化方法，传入物体和边缘点
        super().__init__(mobject, point, **kwargs)


# 定义GrowArrow类，继承自GrowFromPoint，专门实现箭头从起点生长的动画
class GrowArrow(GrowFromPoint):
    # 初始化方法，设置箭头生长的参数
    def __init__(self, arrow: Arrow, **kwargs):
        # 获取箭头的起点坐标作为生长起始点
        point = arrow.get_start()
        # 调用父类GrowFromPoint的初始化方法，传入箭头和其起点
        super().__init__(arrow, point, **kwargs)
