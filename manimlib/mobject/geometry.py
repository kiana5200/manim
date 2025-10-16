from __future__ import annotations

import math

import numpy as np

from manimlib.constants import DL, DOWN, DR, LEFT, ORIGIN, OUT, RIGHT, UL, UP, UR
from manimlib.constants import RED, BLACK, DEFAULT_MOBJECT_COLOR, DEFAULT_LIGHT_COLOR
from manimlib.constants import MED_SMALL_BUFF, SMALL_BUFF
from manimlib.constants import DEG, PI, TAU
from manimlib.mobject.mobject import Mobject
from manimlib.mobject.types.vectorized_mobject import DashedVMobject
from manimlib.mobject.types.vectorized_mobject import VGroup
from manimlib.mobject.types.vectorized_mobject import VMobject
from manimlib.utils.bezier import quadratic_bezier_points_for_arc
from manimlib.utils.iterables import adjacent_n_tuples
from manimlib.utils.iterables import adjacent_pairs
from manimlib.utils.simple_functions import clip
from manimlib.utils.simple_functions import fdiv
from manimlib.utils.space_ops import angle_between_vectors
from manimlib.utils.space_ops import angle_of_vector
from manimlib.utils.space_ops import cross2d
from manimlib.utils.space_ops import compass_directions
from manimlib.utils.space_ops import find_intersection
from manimlib.utils.space_ops import get_norm
from manimlib.utils.space_ops import normalize
from manimlib.utils.space_ops import rotate_vector
from manimlib.utils.space_ops import rotation_matrix_transpose
from manimlib.utils.space_ops import rotation_between_vectors

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Iterable, Optional
    from manimlib.typing import ManimColor, Vect3, Vect3Array, Self


DEFAULT_DOT_RADIUS = 0.08
DEFAULT_SMALL_DOT_RADIUS = 0.04
DEFAULT_DASH_LENGTH = 0.05
DEFAULT_ARROW_TIP_LENGTH = 0.35
DEFAULT_ARROW_TIP_WIDTH = 0.35

# 是否弃用？（注释：标记该类可能面临弃用）
class TipableVMobject(VMobject):
    """
    用于Arc（圆弧）和Line（线段）之间的共享功能。
    功能大致可分为以下几类：

        * 尖端的添加、创建与修改
            - add_tip方法调用create_tip，再将新尖端添加到
                TipableVMobject的子对象列表中
            - 样式与位置配置

        * 尖端检查
            - 布尔值检查TipableVMobject是否有尖端，以及是否有起始端尖端

        * 获取器（Getters）
            - 直接的访问方法，返回与TipableVMobject实例的尖端、长度等相关的信息
    """
    # 尖端默认配置：填充不透明度、描边宽度、尖端样式（0=三角形，1=内平滑，2=点）
    tip_config: dict = dict(
        fill_opacity=1.0,
        stroke_width=0.0,
        tip_style=0.0,  # triangle=0, inner_smooth=1, dot=2
    )

    # 尖端的添加、创建与修改
    def add_tip(self, at_start: bool = False, **kwargs) -> Self:
        """
        为TipableVMobject实例添加尖端，需注意：
        若添加的是“起始端尖端”，可能需要交换端点位置。
        """
        # 创建尖端（根据是否在起始端调整）
        tip = self.create_tip(at_start, **kwargs)
        # 根据尖端位置重置对象的端点
        self.reset_endpoints_based_on_tip(tip, at_start)
        # 为尖端分配属性（如标记起始端/末端尖端）
        self.asign_tip_attr(tip, at_start)  # 注：原代码可能存在拼写错误，应为assign_tip_attr
        # 让尖端颜色与对象描边颜色一致
        tip.set_color(self.get_stroke_color())
        # 将尖端添加为子对象
        self.add(tip)
        return self

    def create_tip(self, at_start: bool = False, **kwargs) -> ArrowTip:
        """
        为尖端设置样式、调整空间位置，并将新创建的尖端返回给调用者。
        """
        # 获取未定位的尖端（仅创建实例，未设置位置）
        tip = self.get_unpositioned_tip(**kwargs)
        # 根据是否在起始端，为尖端设置位置
        self.position_tip(tip, at_start)
        return tip

    def get_unpositioned_tip(self, **kwargs) -> ArrowTip:
        """
        返回已配置样式，但尚未设置空间位置的尖端。
        """
        # 合并默认尖端配置与传入配置（传入配置优先级更高）
        config = dict()
        config.update(self.tip_config)
        config.update(kwargs)
        # 创建并返回未定位的ArrowTip实例
        return ArrowTip(**config)

    def position_tip(self, tip: ArrowTip, at_start: bool = False) -> ArrowTip:
        # 通过锚点（尖端附着点）和控制点（确定切线方向）定位尖端
        if at_start:
            # 起始端尖端：锚点为对象起点，控制点为第一个手柄
            anchor = self.get_start()
            handle = self.get_first_handle()
        else:
            # 末端尖端：控制点为最后一个手柄，锚点为对象终点
            handle = self.get_last_handle()
            anchor = self.get_end()
    
        # 计算并旋转尖端，使其方向与切线方向一致
        # 向量（手柄-锚点）的角度 - π（补偿方向） - 尖端自身角度
        tip.rotate(angle_of_vector(handle - anchor) - PI - tip.get_angle())
        # 将尖端移动到锚点：让尖端的尖端点与锚点重合
        tip.shift(anchor - tip.get_tip_point())
        return tip

    def reset_endpoints_based_on_tip(self, tip: ArrowTip, at_start: bool) -> Self:
        # 若对象长度为0，put_start_and_end_on方法无效，直接返回
        if self.get_length() == 0:
            return self

        # 根据尖端位置调整对象的端点（确保尖端与对象端点对齐）
        if at_start:
            # 起始端尖端：对象起点设为尖端的基座，终点保持不变
            start = tip.get_base()
            end = self.get_end()
        else:
            # 末端尖端：对象起点保持不变，终点设为尖端的基座
            start = self.get_start()
            end = tip.get_base()
    
        # 更新对象的起点和终点
        self.put_start_and_end_on(start, end)
        return self

    def asign_tip_attr(self, tip: ArrowTip, at_start: bool) -> Self:
        # 注：原方法名存在拼写错误，正确应为“assign_tip_attr”
        # 将尖端实例分配到对应的属性（起始端尖端/末端尖端）
        if at_start:
            self.start_tip = tip
        else:
            self.tip = tip
        return self

    
    # 尖端检查（Checking for tips）
    def has_tip(self) -> bool:
        # 检查是否存在末端尖端：需同时有"tip"属性且该尖端在子对象中
        return hasattr(self, "tip") and self.tip in self

    def has_start_tip(self) -> bool:
        # 检查是否存在起始端尖端：需同时有"start_tip"属性且该尖端在子对象中
        return hasattr(self, "start_tip") and self.start_tip in self

    # 获取器（Getters）
    def pop_tips(self) -> VGroup:
        # 先记录对象当前的起点和终点（后续用于重置）
        start, end = self.get_start_and_end()
        # 初始化存储移除尖端的组
        result = VGroup()
    
        # 若有末端尖端，移除并添加到结果组
        if self.has_tip():
            result.add(self.tip)
            self.remove(self.tip)
        # 若有起始端尖端，移除并添加到结果组
        if self.has_start_tip():
            result.add(self.start_tip)
            self.remove(self.start_tip)
    
        # 重置对象的起点和终点（避免因移除尖端导致位置偏移）
        self.put_start_and_end_on(start, end)
        return result

    def get_tips(self) -> VGroup:
        """
        返回包含TipableVMObject实例所有尖端的VGroup（VMobject集合）。
        """
        result = VGroup()
        # 若有末端尖端，添加到结果组
        if hasattr(self, "tip"):
            result.add(self.tip)
        # 若有起始端尖端，添加到结果组
        if hasattr(self, "start_tip"):
            result.add(self.start_tip)
        return result

    def get_tip(self) -> ArrowTip:
        """返回TipableVMobject实例的（第一个）尖端，
        若不存在尖端则抛出异常。"""
        tips = self.get_tips()
        if len(tips) == 0:
            raise Exception("tip not found")  # 未找到尖端
        else:
            return tips[0]  # 返回第一个尖端（通常是末端尖端）

    def get_default_tip_length(self) -> float:
        # 返回默认尖端长度（依赖实例的"tip_length"属性）
        return self.tip_length

    def get_first_handle(self) -> Vect3:
        # 返回对象的第一个控制点（取点列表的第二个点，索引为1）
        return self.get_points()[1]

    def get_last_handle(self) -> Vect3:
        # 返回对象的最后一个控制点（取点列表的倒数第二个点）
        return self.get_points()[-2]

    def get_end(self) -> Vect3:
        # 若有末端尖端，返回尖端的起点作为对象的终点；否则调用父类方法
        if self.has_tip():
            return self.tip.get_start()
        else:
            return VMobject.get_end(self)

    def get_start(self) -> Vect3:
        # 若有起始端尖端，返回起始尖端的起点作为对象的起点；否则调用父类方法
        if self.has_start_tip():
            return self.start_tip.get_start()
        else:
            return VMobject.get_start(self)

    def get_length(self) -> float:
        # 获取对象的起点和终点，计算两点间的欧氏距离作为长度
        start, end = self.get_start_and_end()
        return get_norm(start - end)

class Arc(TipableVMobject):
    '''
    创建圆弧（Arc）对象。
    
    参数
    -----
    start_angle : float
        圆弧的起始角度，单位为弧度。（角度按逆时针方向计算）
    angle : float
        圆弧在圆心处对应的圆心角，单位为弧度。（角度按逆时针方向计算）
    radius : float
        圆弧的半径
    arc_center : array_like
        圆弧的圆心坐标
    
    示例 :
            arc = Arc(start_angle=TAU/4, angle=TAU/2, radius=3, arc_center=ORIGIN)
            arc = Arc(angle=TAU/4, radius=4.5, arc_center=(1,2,0), color=BLUE)
    
    返回
    -----
    out : Arc object
        符合指定参数的圆弧对象
    '''

    def __init__(
        self,
        start_angle: float = 0,
        angle: float = TAU / 4,
        radius: float = 1.0,
        n_components: Optional[int] = None,
        arc_center: Vect3 = ORIGIN,** kwargs
    ):
        # 调用父类TipableVMobject的初始化方法
        super().__init__(**kwargs)

        # 确定圆弧的贝塞尔曲线分段数：未指定时，按圆心角占全圆的比例计算（全圆默认16段）
        if n_components is None:
            # 全圆（TAU弧度）对应16段，按比例计算当前圆弧的分段数
            n_components = int(15 * (abs(angle) / TAU)) + 1

        # 1. 生成圆弧的二次贝塞尔曲线上的点
        self.set_points(quadratic_bezier_points_for_arc(angle, n_components))
        # 2. 绕原点旋转，使圆弧对齐到起始角度
        self.rotate(start_angle, about_point=ORIGIN)
        # 3. 绕原点缩放，使圆弧达到指定半径
        self.scale(radius, about_point=ORIGIN)
        # 4. 平移圆弧，使圆心移动到指定位置
        self.shift(arc_center)

    def get_arc_center(self) -> Vect3:
        """
        通过计算前两个锚点的法线，找到它们的交点，即为圆弧的圆心。
        """
        # 获取前两个锚点（a1、a2）和中间的控制点（h）
        a1, h, a2 = self.get_points()[:3]
        # 计算两个切线向量（控制点到锚点的向量）
        t1 = h - a1
        t2 = h - a2
        # 计算两个切线向量的法线（旋转90度，TAU/4弧度=90度）
        n1 = rotate_vector(t1, TAU / 4)
        n2 = rotate_vector(t2, TAU / 4)
        # 找到两条法线的交点，即为圆弧圆心
        return find_intersection(a1, n1, a2, n2)

    def get_start_angle(self) -> float:
        # 计算“圆弧起点 - 圆心”向量的角度，结果对TAU取余（确保在0~2π范围内）
        angle = angle_of_vector(self.get_start() - self.get_arc_center())
        return angle % TAU

    def get_stop_angle(self) -> float:
        # 计算“圆弧终点 - 圆心”向量的角度，结果对TAU取余（确保在0~2π范围内）
        angle = angle_of_vector(self.get_end() - self.get_arc_center())
        return angle % TAU

    def move_arc_center_to(self, point: Vect3) -> Self:
        # 计算当前圆心到目标点的偏移量，按该偏移量平移圆弧，使圆心移动到目标点
        self.shift(point - self.get_arc_center())
        return self


class ArcBetweenPoints(Arc):
    '''
    创建经过指定起点和终点的圆弧，圆弧在圆心处对应的圆心角由“angle”参数指定。
    
    参数
    -----
    start : array_like
        圆弧的起点坐标
    end : array_like
        圆弧的终点坐标
    angle : float
        圆弧在圆心处对应的圆心角，单位为弧度。（角度按逆时针方向计算）
    
    示例 :
            arc = ArcBetweenPoints(start=(0, 0, 0), end=(1, 2, 0), angle=TAU / 2)
            arc = ArcBetweenPoints(start=(-2, 3, 0), end=(1, 2, 0), angle=-TAU / 12, color=BLUE)
    
    返回
    -----
    out : ArcBetweenPoints object
        符合指定参数的圆弧对象
    '''

    def __init__(
        self,
        start: Vect3,
        end: Vect3,
        angle: float = TAU / 4,** kwargs
    ):
        # 调用父类Arc的初始化方法，传入圆心角和其他额外参数
        super().__init__(angle=angle, **kwargs)
        
        # 若圆心角为0，将圆弧简化为直线段（用两个角点定义：左到右的默认线段）
        if angle == 0:
            self.set_points_as_corners([LEFT, RIGHT])
        
        # 将圆弧的起点和终点定位到指定的坐标
        self.put_start_and_end_on(start, end)


class CurvedArrow(ArcBetweenPoints):
    '''
    创建经过指定起点和终点的弯曲箭头，箭头在圆心处对应的圆心角由“angle”参数指定。
    
    参数
    -----
    start_point : array_like
        弯曲箭头的起点坐标
    end_point : array_like
        弯曲箭头的终点坐标
    angle : float
        弯曲箭头在圆心处对应的圆心角，单位为弧度。（角度按逆时针方向计算）
    
    示例 :
            curvedArrow = CurvedArrow(start_point=(0, 0, 0), end_point=(1, 2, 0), angle=TAU/2)
            curvedArrow = CurvedArrow(start_point=(-2, 3, 0), end_point=(1, 2, 0), angle=-TAU/12, color=BLUE)
    
    返回
    -----
    out : CurvedArrow object
        符合指定参数的弯曲箭头对象
    '''

    def __init__(
        self,
        start_point: Vect3,
        end_point: Vect3,** kwargs
    ):
        # 调用父类ArcBetweenPoints的初始化方法，传入起点、终点和其他额外参数
        super().__init__(start_point, end_point, **kwargs)
        # 为弯曲箭头添加末端尖端（继承自TipableVMobject的方法）
        self.add_tip()


class CurvedDoubleArrow(CurvedArrow):
    '''
    创建经过指定起点和终点的双向弯曲箭头，箭头在圆心处对应的圆心角由“angle”参数指定。
    
    参数
    -----
    start_point : array_like
        双向弯曲箭头的起点坐标
    end_point : array_like
        双向弯曲箭头的终点坐标
    angle : float
        双向弯曲箭头在圆心处对应的圆心角，单位为弧度。（角度按逆时针方向计算）
    
    示例 :
            curvedDoubleArrow = CurvedDoubleArrow(start_point = (0, 0, 0), end_point = (1, 2, 0), angle = TAU/2)
            curvedDoubleArrow = CurvedDoubleArrow(start_point = (-2, 3, 0), end_point = (1, 2, 0), angle = -TAU/12, color = BLUE)
    
    返回
    -----
    out : CurvedDoubleArrow object
        符合指定参数的双向弯曲箭头对象
    '''

    def __init__(
        self,
        start_point: Vect3,
        end_point: Vect3,** kwargs
    ):
        # 调用父类CurvedArrow的初始化方法，传入起点、终点和其他额外参数
        super().__init__(start_point, end_point, **kwargs)
        # 为双向箭头添加起始端尖端（父类已添加末端尖端，此处补全双向）
        self.add_tip(at_start=True)

class Circle(Arc):
    '''
    创建圆形（Circle）对象。
    
    参数
    -----
    radius : float
        圆的半径
    arc_center : array_like
        圆的圆心坐标
    
    示例 :
            circle = Circle(radius=2, arc_center=(1,2,0))
            circle = Circle(radius=3.14, arc_center=2 * LEFT + UP, color=DARK_BLUE)
    
    返回
    -----
    out : Circle object
        符合指定参数的圆形对象
    '''

    def __init__(
        self,
        start_angle: float = 0,
        stroke_color: ManimColor = RED,** kwargs
    ):
        # 调用父类Arc的初始化方法，强制圆心角为TAU（2π，即完整圆周）
        super().__init__(
            start_angle,  # 圆的起始角度（默认0）
            TAU,          # 圆心角固定为2π，确保是完整圆形
            stroke_color=stroke_color,  # 描边颜色默认红色
            **kwargs      # 传递其他参数（如radius、arc_center等）
        )

    def surround(
        self,
        mobject: Mobject,
        dim_to_match: int = 0,
        stretch: bool = False,
        buff: float = MED_SMALL_BUFF
    ) -> Self:
        # 先让圆匹配目标Mobject的尺寸
        self.replace(mobject, dim_to_match, stretch)
        # 在宽度方向扩展：原宽度 + 2倍缓冲（左右各加buff）
        self.stretch((self.get_width() + 2 * buff) / self.get_width(), 0)
        # 在高度方向扩展：原高度 + 2倍缓冲（上下各加buff）
        self.stretch((self.get_height() + 2 * buff) / self.get_height(), 1)
        return self

    def point_at_angle(self, angle: float) -> Vect3:
        # 获取圆的起始角度
        start_angle = self.get_start_angle()
        # 将目标角度转换为比例（0~1），再通过比例获取圆上对应点
        return self.point_from_proportion(
            ((angle - start_angle) % TAU) / TAU
        )

    def get_radius(self) -> float:
        # 计算“圆的起点 - 圆心”向量的模长，即为半径
        return get_norm(self.get_start() - self.get_center())

class Dot(Circle):
    '''
    创建点（Dot）对象。点是一个填充白色、无边界、半径为DEFAULT_DOT_RADIUS的圆形。
    
    参数
    -----
    point : array_like
        点的中心坐标
    
    示例 :
            dot = Dot(point=(1, 2, 0))
    
    返回
    -----
    out : Dot object
        符合指定参数的点对象
    '''

    def __init__(
        self,
        point: Vect3 = ORIGIN,
        radius: float = DEFAULT_DOT_RADIUS,
        stroke_color: ManimColor = BLACK,
        stroke_width: float = 0.0,
        fill_opacity: float = 1.0,
        fill_color: ManimColor = DEFAULT_MOBJECT_COLOR,** kwargs
    ):
        # 调用父类Circle的初始化方法，配置点的专属样式
        super().__init__(
            arc_center=point,          # 点的中心坐标（对应圆的圆心）
            radius=radius,            # 点的半径（默认DEFAULT_DOT_RADIUS）
            stroke_color=stroke_color, # 描边颜色默认黑色
            stroke_width=stroke_width, # 描边宽度默认0（无边界）
            fill_opacity=fill_opacity, # 填充不透明度默认1（完全不透明）
            fill_color=fill_color,     # 填充颜色默认DEFAULT_MOBJECT_COLOR
            **kwargs                   # 传递其他额外参数
        )


class SmallDot(Dot):
    '''
    创建小点点（SmallDot）对象。小点点是一个填充白色、无边界、半径为DEFAULT_SMALL_DOT_RADIUS的圆形。
    
    参数
    -----
    point : array_like
        小点点的中心坐标
    
    示例 :
            smallDot = SmallDot(point=(1, 2, 0))
    
    返回
    -----
    out : SmallDot object
        符合指定参数的小点点对象
    '''

    def __init__(
        self,
        point: Vect3 = ORIGIN,
        radius: float = DEFAULT_SMALL_DOT_RADIUS,** kwargs
    ):
        # 调用父类Dot的初始化方法，核心差异是默认半径为DEFAULT_SMALL_DOT_RADIUS
        super().__init__(point, radius=radius, **kwargs)

class Ellipse(Circle):
    '''
    创建椭圆（Ellipse）对象。
    
    参数
    -----
    width : float
        椭圆的宽度
    height : float
        椭圆的高度
    arc_center : array_like
        椭圆的中心坐标
    
    示例 :
            ellipse = Ellipse(width=4, height=1, arc_center=(3, 3, 0))
            ellipse = Ellipse(width=2, height=5, arc_center=ORIGIN, color=BLUE)
    
    返回
    -----
    out : Ellipse object
        符合指定参数的椭圆对象
    '''

    def __init__(
        self,
        width: float = 2.0,
        height: float = 1.0,** kwargs
    ):
        # 调用父类Circle的初始化方法（继承圆形的完整闭合特性）
        super().__init__(**kwargs)
        # 按指定宽度拉伸椭圆（stretch=True允许非均匀拉伸）
        self.set_width(width, stretch=True)
        # 按指定高度拉伸椭圆（stretch=True确保高度独立于宽度）
        self.set_height(height, stretch=True)


class AnnularSector(VMobject):
    '''
    创建环形扇形（AnnularSector）对象，即两个同心圆的扇形区域之间的环形部分。
    
    参数
    -----
    inner_radius : float
        环形扇形的内半径
    outer_radius : float
        环形扇形的外半径
    start_angle : float
        环形扇形的起始角度（角度按逆时针方向计算）
    angle : float
        环形扇形在圆心处对应的圆心角（角度按逆时针方向计算）
    arc_center : array_like
        环形扇形的中心坐标
    
    示例 :
            annularSector = AnnularSector(inner_radius=1, outer_radius=2, angle=TAU/2, start_angle=TAU*3/4, arc_center=(1,-2,0))
    
    返回
    -----
    out : AnnularSector object
        符合指定参数的环形扇形对象
    '''

    def __init__(
        self,
        angle: float = TAU / 4,
        start_angle: float = 0.0,
        inner_radius: float = 1.0,
        outer_radius: float = 2.0,
        arc_center: Vect3 = ORIGIN,
        fill_color: ManimColor = DEFAULT_LIGHT_COLOR,
        fill_opacity: float = 1.0,
        stroke_width: float = 0.0,** kwargs,
    ):
        # 调用父类VMobject的初始化方法，配置填充和描边样式
        super().__init__(
            fill_color=fill_color,      # 填充颜色默认浅色调
            fill_opacity=fill_opacity,  # 填充不透明度默认1（完全不透明）
            stroke_width=stroke_width,  # 描边宽度默认0（无边界）
            **kwargs,                   # 传递其他额外参数
        )

        # 初始化内外圆弧：创建两个同心的圆弧，分别对应内半径和外半径
        inner_arc, outer_arc = [
            Arc(
                start_angle=start_angle,  # 共享起始角度
                angle=angle,              # 共享圆心角
                radius=radius,            # 分别使用内半径和外半径
                arc_center=arc_center,    # 共享圆心
            )
            for radius in (inner_radius, outer_radius)
        ]
        
        # 构建环形扇形的轮廓点集：
        # 1. 添加反转的内圆弧点（从终点到起点）
        self.set_points(inner_arc.get_points()[::-1])  # [::-1]实现点序反转
        # 2. 添加从内圆弧起点到外圆弧起点的线段
        self.add_line_to(outer_arc.get_points()[0])
        # 3. 添加外圆弧的点（从起点到终点）
        self.add_subpath(outer_arc.get_points())
        # 4. 添加从外圆弧终点到内圆弧终点的线段（闭合环形）
        self.add_line_to(inner_arc.get_points()[-1])

class Sector(AnnularSector):
    '''
    创建扇形（Sector）对象，即圆心角对应的扇形区域（可理解为内半径为0的环形扇形）。
    
    参数
    -----
    outer_radius : float
        扇形的半径（对应环形扇形的外半径）
    start_angle : float
        扇形的起始角度，单位为弧度。（角度按逆时针方向计算）
    angle : float
        扇形在圆心处对应的圆心角，单位为弧度。（角度按逆时针方向计算）
    arc_center : array_like
        扇形的中心坐标
    
    示例 :
            sector = Sector(outer_radius=1, start_angle=TAU/3, angle=TAU/2, arc_center=[0,3,0])
            sector = Sector(outer_radius=3, start_angle=TAU/4, angle=TAU/4, arc_center=ORIGIN, color=PINK)
    
    返回
    -----
    out : Sector object
        符合指定参数的扇形对象
    '''

    def __init__(
        self,
        angle: float = TAU / 4,
        radius: float = 1.0,** kwargs
    ):
        # 调用父类AnnularSector的初始化方法，核心是将内半径固定为0
        super().__init__(
            angle,              # 继承父类的圆心角参数
            inner_radius=0,     # 内半径设为0，使环形扇形退化为普通扇形
            outer_radius=radius,# 外半径作为扇形的实际半径
            **kwargs            # 传递其他额外参数（如start_angle、arc_center等）
        )

class Annulus(VMobject):
    '''
    创建圆环（Annulus）对象，即两个同心圆之间的环形区域（完整闭合，无圆心角限制）。
    
    参数
    -----
    inner_radius : float
        圆环的内半径
    outer_radius : float
        圆环的外半径
    arc_center : array_like
        圆环的中心坐标（与参数center功能一致）
    
    示例 :
            annulus = Annulus(inner_radius=2, outer_radius=3, arc_center=(1, -1, 0))
            annulus = Annulus(inner_radius=2, outer_radius=3, stroke_width=20, stroke_color=RED, fill_color=BLUE, arc_center=ORIGIN)
    
    返回
    -----
    out : Annulus object
        符合指定参数的圆环对象
    '''

    def __init__(
        self,
        inner_radius: float = 1.0,
        outer_radius: float = 2.0,
        fill_opacity: float = 1.0,
        stroke_width: float = 0.0,
        fill_color: ManimColor = DEFAULT_LIGHT_COLOR,
        center: Vect3 = ORIGIN,** kwargs,
    ):
        # 调用父类VMobject的初始化方法，配置填充与描边样式
        super().__init__(
            fill_color=fill_color,      # 填充颜色默认浅色调
            fill_opacity=fill_opacity,  # 填充不透明度默认1（完全不透明）
            stroke_width=stroke_width,  # 描边宽度默认0（无边界）
            **kwargs,                   # 传递其他额外参数
        )

        # 记录圆环外半径（便于后续复用）
        self.radius = outer_radius
        # 1. 生成外圆的二次贝塞尔曲线路径（圆心角TAU，完整圆形）
        outer_path = outer_radius * quadratic_bezier_points_for_arc(TAU)
        # 2. 生成内圆的二次贝塞尔曲线路径（圆心角-TAU，反向绘制以形成闭合环形）
        inner_path = inner_radius * quadratic_bezier_points_for_arc(-TAU)
        # 3. 将外圆路径添加为子路径
        self.add_subpath(outer_path)
        # 4. 将内圆路径添加为子路径（反向绘制确保环形区域闭合）
        self.add_subpath(inner_path)
        # 5. 将圆环平移到指定中心位置
        self.shift(center)


class Line(TipableVMobject):
    '''
    创建连接“起点”和“终点”的线段（Line）对象。
    
    参数
    -----
    start : array_like
        线段的起点坐标或Mobject对象
    end : array_like
        线段的终点坐标或Mobject对象
    buff : float
        线段与起点/终点之间的缓冲距离（默认0，无缓冲）
    path_arc : float
        线段的弯曲弧度（默认0，为直线；非0时呈弧形）
    
    示例 :
            line = Line((0, 0, 0), (3, 0, 0))
            line = Line((1, 2, 0), (-2, -3, 0), color=BLUE)
    
    返回
    -----
    out : Line object
        符合指定参数的线段对象
    '''

    def __init__(
        self,
        start: Vect3 | Mobject = LEFT,
        end: Vect3 | Mobject = RIGHT,
        buff: float = 0.0,
        path_arc: float = 0.0,** kwargs
    ):
        # 调用父类TipableVMobject的初始化方法（支持添加尖端）
        super().__init__(**kwargs)
        # 记录线段的弯曲弧度和缓冲距离
        self.path_arc = path_arc
        self.buff = buff
        # 处理起点和终点（若为Mobject则取对应边界点）
        self.set_start_and_end_attrs(start, end)
        # 根据起点、终点、缓冲和弧度生成线段的点集
        self.set_points_by_ends(self.start, self.end, buff, path_arc)

    def set_points_by_ends(
        self,
        start: Vect3,
        end: Vect3,
        buff: float = 0,
        path_arc: float = 0
    ) -> Self:
        # 清空现有点集，重新开始绘制路径
        self.clear_points()
        self.start_new_path(start)
        # 按指定弧度添加线段（0为直线，非0为弧形）
        self.add_arc_to(end, path_arc)

        # 应用缓冲距离：缩短线段两端，避免与起点/终点重叠
        if buff > 0:
            # 计算线段总长度
            length = self.get_arc_length()
            # 计算缓冲对应的比例（最大不超过0.5，避免线段反向）
            alpha = min(buff / length, 0.5)
            # 截取线段中间部分（去除两端缓冲区域）
            self.pointwise_become_partial(self, alpha, 1 - alpha)
        return self

    def set_path_arc(self, new_value: float) -> Self:
        # 更新线段的弯曲弧度，并重新初始化点集
        self.path_arc = new_value
        self.init_points()
        return self

    def set_start_and_end_attrs(self, start: Vect3 | Mobject, end: Vect3 | Mobject):
        # 初步获取起点和终点（若为Mobject则先取中心）
        rough_start = self.pointify(start)
        rough_end = self.pointify(end)
        # 计算起点到终点的单位向量（确定方向）
        vect = normalize(rough_end - rough_start)
        # 重新确定准确的起点和终点：若为Mobject，取沿方向的边界点（避免线段穿入Mobject内部）
        self.start = self.pointify(start, vect)
        self.end = self.pointify(end, -vect)

    def pointify(
        self,
        mob_or_point: Mobject | Vect3,
        direction: Vect3 | None = None
    ) -> Vect3:
        """
        将传入Line（或子类）的参数（坐标或Mobject）转换为3D点。
        """
        if isinstance(mob_or_point, Mobject):
            mob = mob_or_point
            # 无方向时，返回Mobject的中心
            if direction is None:
                return mob.get_center()
            # 有方向时，返回Mobject沿该方向的边界点
            else:
                return mob.get_continuous_bounding_box_point(direction)
        else:
            # 若为坐标，转换为符合当前维度的3D点（不足3维补0）
            point = mob_or_point
            result = np.zeros(self.dim)
            result[:len(point)] = point
            return result

    def put_start_and_end_on(self, start: Vect3, end: Vect3) -> Self:
        # 获取当前的起点和终点
        curr_start, curr_end = self.get_start_and_end()
        # 若当前起点和终点重合（零长度线段），直接用新端点重建线段
        if np.isclose(curr_start, curr_end).all():
            self.set_points_by_ends(start, end, buff=0, path_arc=self.path_arc)
            return self
        # 否则调用父类方法更新端点
        return super().put_start_and_end_on(start, end)

    def get_vector(self) -> Vect3:
        # 返回从起点到终点的向量
        return self.get_end() - self.get_start()

    def get_unit_vector(self) -> Vect3:
        # 返回单位化的方向向量（长度为1）
        return normalize(self.get_vector())

    def get_angle(self) -> float:
        # 返回线段与x轴正方向的夹角（弧度）
        return angle_of_vector(self.get_vector())

    def get_projection(self, point: Vect3) -> Vect3:
        """
        返回点在直线上的投影点
        """
        # 获取线段的单位方向向量
        unit_vect = self.get_unit_vector()
        # 获取线段起点
        start = self.get_start()
        # 计算点到起点的向量在单位方向向量上的投影长度，再计算投影点坐标
        return start + np.dot(point - start, unit_vect) * unit_vect

    def get_slope(self) -> float:
        # 返回线段的斜率（tan(角度)）
        return np.tan(self.get_angle())

    def set_angle(self, angle: float, about_point: Optional[Vect3] = None) -> Self:
        # 若未指定旋转中心，默认以线段起点为中心
        if about_point is None:
            about_point = self.get_start()
        # 计算目标角度与当前角度的差值，按该差值旋转线段
        self.rotate(
            angle - self.get_angle(),
            about_point=about_point,
        )
        return self

    def set_length(self, length: float, **kwargs):
        # 按目标长度与当前长度的比例缩放线段
        self.scale(length / self.get_length(),** kwargs)
        return self

    def get_arc_length(self) -> float:
        # 直线长度（向量模长）
        arc_len = get_norm(self.get_vector())
        # 若线段有弯曲弧度，按圆弧长度公式修正
        if self.path_arc > 0:
            arc_len *= self.path_arc / (2 * math.sin(self.path_arc / 2))
        return arc_len


class DashedLine(Line):
    '''
    创建连接“起点”和“终点”的虚线（DashedLine）对象。
    
    参数
    -----
    start : array_like
        虚线的起点坐标
    end : array_like
        虚线的终点坐标
    dash_length : float
        每个短线段（ dash ）的长度，默认值为 DEFAULT_DASH_LENGTH
    positive_space_ratio : float
        实线部分占“实线+空白”周期的比例，默认0.5（即实线与空白长度相等）
    
    示例 :
            line = DashedLine((0, 0, 0), (3, 0, 0))
            line = DashedLine((1, 2, 3), (4, 5, 6), dash_length=0.01)
    
    返回
    -----
    out : DashedLine object
        符合指定参数的虚线对象
    '''

    def __init__(
        self,
        start: Vect3 = LEFT,
        end: Vect3 = RIGHT,
        dash_length: float = DEFAULT_DASH_LENGTH,
        positive_space_ratio: float = 0.5,** kwargs
    ):
        # 调用父类Line的初始化方法，先创建完整线段
        super().__init__(start, end, **kwargs)

        # 计算需要的短线段数量
        num_dashes = self.calculate_num_dashes(dash_length, positive_space_ratio)
        # 用DashedVMobject将完整线段分割为虚线
        dashes = DashedVMobject(
            self,
            num_dashes=num_dashes,          # 短线段总数
            positive_space_ratio=positive_space_ratio  # 实线占比
        )
        # 清空原完整线段的点集，添加分割后的短线段
        self.clear_points()
        self.add(*dashes)

    def calculate_num_dashes(self, dash_length: float, positive_space_ratio: float) -> int:
        try:
            # 计算“实线+空白”的完整周期长度
            full_length = dash_length / positive_space_ratio
            # 按总长度除以周期长度，向上取整得到短线段数量
            return int(np.ceil(self.get_length() / full_length))
        except ZeroDivisionError:
            # 避免比例为0导致的错误，默认返回1个短线段
            return 1

    def get_start(self) -> Vect3:
        # 若有子对象（短线段），返回第一个短线段的起点；否则调用父类方法
        if len(self.submobjects) > 0:
            return self.submobjects[0].get_start()
        else:
            return Line.get_start(self)

    def get_end(self) -> Vect3:
        # 若有子对象（短线段），返回最后一个短线段的终点；否则调用父类方法
        if len(self.submobjects) > 0:
            return self.submobjects[-1].get_end()
        else:
            return Line.get_end(self)

    def get_start_and_end(self) -> Tuple[Vect3, Vect3]:
        # 直接调用自身的get_start和get_end方法，返回虚线的整体起点和终点
        return self.get_start(), self.get_end()

    def get_first_handle(self) -> Vect3:
        # 返回第一个短线段的第一个控制点（取其点列表的第二个点）
        return self.submobjects[0].get_points()[1]

    def get_last_handle(self) -> Vect3:
        # 返回最后一个短线段的最后一个控制点（取其点列表的倒数第二个点）
        return self.submobjects[-1].get_points()[-2]

class TangentLine(Line):
    '''
    创建与指定向量图形对象（VMobject）相切的线段（TangentLine）。
    
    参数
    -----
    vmob : VMobject object
        线段要与之相切的向量图形对象
    alpha : float
        图形对象轮廓上的点位参数，取值范围为0到1（包含0和1），对应图形周长的比例位置
    length : float
        切线的总长度，默认值为2
    d_alpha : float
        计算切线方向时所用的“微小步长”，默认值为1e-6，用于通过两点近似切线方向
    
    示例 :
            circle = Circle(arc_center=ORIGIN, radius=3, color=GREEN)
            tangentLine = TangentLine(vmob=circle, alpha=1/3, length=6, color=BLUE)
    
    返回
    -----
    out : TangentLine object
        符合指定参数的切线对象
    '''

    def __init__(
        self,
        vmob: VMobject,
        alpha: float,
        length: float = 2,
        d_alpha: float = 1e-6,** kwargs
    ):
        # 1. 计算两个极近的alpha值（围绕目标alpha），确保在0~1范围内
        a1 = clip(alpha - d_alpha, 0, 1)  # 目标alpha左侧微小步长的点位
        a2 = clip(alpha + d_alpha, 0, 1)  # 目标alpha右侧微小步长的点位
        
        # 2. 调用父类Line的初始化方法：以两个近点为端点创建线段，近似切线方向
        # pfp 是 point_from_proportion 的缩写，通过alpha值获取图形上的点
        super().__init__(vmob.pfp(a1), vmob.pfp(a2), **kwargs)
        
        # 3. 缩放线段，使其达到指定的总长度
        self.scale(length / self.get_length())

class Elbow(VMobject):
    '''
    创建肘形（Elbow）对象，即呈L形的图形，可用于连接或标记拐角。
    
    参数
    -----
    width : float
        肘形的整体宽度（控制L形的尺寸），默认值为0.2
    angle : float
        肘形与水平方向的夹角，单位为弧度，按逆时针方向计算，默认值为0（水平向右的L形）
    
    示例 :
            line = Elbow(width=2, angle=TAU/16)
    
    返回
    -----
    out : Elbow object
        符合指定参数的肘形对象
    '''

    def __init__(
        self,
        width: float = 0.2,
        angle: float = 0,** kwargs
    ):
        # 调用父类VMobject的初始化方法
        super().__init__(**kwargs)
        # 1. 以角点模式创建基础L形：从上方(UP)到右上(UR)再到右方(RIGHT)
        self.set_points_as_corners([UP, UR, RIGHT])
        # 2. 按指定宽度缩放L形，缩放中心为原点(ORIGIN)
        self.set_width(width, about_point=ORIGIN)
        # 3. 按指定角度旋转L形，旋转中心为原点(ORIGIN)
        self.rotate(angle, about_point=ORIGIN)

class StrokeArrow(Line):
    '''
    创建带尖端的描边箭头（StrokeArrow），尖端宽度与线段描边宽度关联，风格更统一。
    
    核心差异：相比普通Line加尖端，其尖端尺寸会随线段描边宽度自适应调整，无需额外配置尖端样式。
    '''

    def __init__(
        self,
        start: Vect3 | Mobject,
        end: Vect3 | Mobject,
        stroke_color: ManimColor = DEFAULT_LIGHT_COLOR,
        stroke_width: float = 5,
        buff: float = 0.25,
        tip_width_ratio: float = 5,          # 尖端宽度与线段描边宽度的比例
        tip_len_to_width: float = 0.0075,    # 尖端长度与“尖端宽度”的比例系数
        max_tip_length_to_length_ratio: float = 0.3,  # 尖端最长不超过线段总长的30%
        max_width_to_length_ratio: float = 8.0,       # 描边宽度与线段总长的最大比例
        **kwargs,
    ):
        # 记录箭头专属的尖端配置参数
        self.tip_width_ratio = tip_width_ratio
        self.tip_len_to_width = tip_len_to_width
        self.max_tip_length_to_length_ratio = max_tip_length_to_length_ratio
        self.max_width_to_length_ratio = max_width_to_length_ratio
        self.n_tip_points = 3  # 尖端由3个点构成（简化三角形尖端）
        self.original_stroke_width = stroke_width  # 记录原始描边宽度
        
        # 调用父类Line的初始化方法，传递基础线段参数
        super().__init__(
            start, end,
            stroke_color=stroke_color,
            stroke_width=stroke_width,
            buff=buff,** kwargs
        )

    def set_points_by_ends(
        self,
        start: Vect3,
        end: Vect3,
        buff: float = 0,
        path_arc: float = 0
    ) -> Self:
        # 1. 先调用父类方法，生成基础线段的点集
        super().set_points_by_ends(start, end, buff, path_arc)
        # 2. 插入尖端的锚点（确定尖端在何处衔接线段）
        self.insert_tip_anchor()
        # 3. 根据描边宽度创建匹配的尖端
        self.create_tip_with_stroke_width()
        return self

    def insert_tip_anchor(self) -> Self:
        # 记录线段当前的终点（后续用于定位尖端）
        prev_end = self.get_end()
        # 计算线段的总长度
        arc_len = self.get_arc_length()
        
        # 1. 计算理论尖端长度：尖端宽度（描边宽度×比例）×长度系数
        tip_len = self.get_stroke_width() * self.tip_width_ratio * self.tip_len_to_width
        
        # 2. 限制尖端长度：不超过线段总长的30%（避免尖端过长不协调）
        if tip_len >= self.max_tip_length_to_length_ratio * arc_len or arc_len == 0:
            alpha = self.max_tip_length_to_length_ratio  # 按比例截取
        else:
            alpha = tip_len / arc_len  # 按理论长度截取
        
        # 3. 调整线段主体长度，留出尖端位置
        if self.path_arc > 0 and self.buff > 0:
            self.insert_n_curves(10)  # 弧形线段时插入更多曲线段，保证平滑
        # 截取线段主体（去掉末端用于放尖端的部分）
        self.pointwise_become_partial(self, 0.0, 1.0 - alpha)
        # 添加线段主体新终点到原终点的连线（为尖端提供基础边）
        self.add_line_to(self.get_end())
        self.add_line_to(prev_end)
        
        self.n_tip_points = 3  # 重置尖端点数（确保后续创建尖端时用3点三角形）
        return self

    @Mobject.affects_data
    def create_tip_with_stroke_width(self) -> Self:
        # 若线段点数不足3（无法构成尖端），直接返回
        if self.get_num_points() < 3:
            return self
        # 1. 限制描边宽度：不超过“线段总长/最大比例”，避免过粗
        stroke_width = min(
            self.original_stroke_width,  # 原始设定的描边宽度
            self.max_width_to_length_ratio * self.get_length(),  # 按线段长度限制的最大宽度
        )
        # 2. 计算尖端宽度（描边宽度×比例系数）
        tip_width = self.tip_width_ratio * stroke_width
        ntp = self.n_tip_points  # 尖端的点数（默认3）
        
        # 3. 配置线段主体的描边宽度：除尖端外，保持统一宽度
        self.data['stroke_width'][:-ntp] = self.data['stroke_width'][0]
        # 4. 配置尖端的描边宽度：从尖端底部到顶端，宽度线性递减至0（形成三角形尖端）
        self.data['stroke_width'][-ntp:, 0] = tip_width * np.linspace(1, 0, ntp)
        return self

    def reset_tip(self) -> Self:
        # 重新通过起点、终点和弧度生成线段，间接重置尖端
        self.set_points_by_ends(
            self.get_start(), self.get_end(),
            path_arc=self.path_arc
        )
        return self

    def set_stroke(
        self,
        color: ManimColor | Iterable[ManimColor] | None = None,
        width: float | Iterable[float] | None = None,
        *args, **kwargs
    ) -> Self:
        # 1. 调用父类方法更新描边颜色和宽度
        super().set_stroke(color=color, width=width, *args, **kwargs)
        # 2. 更新原始描边宽度记录，确保后续尖端适配新宽度
        self.original_stroke_width = self.get_stroke_width()
        # 3. 若线段已有点集，重置尖端以匹配新描边
        if self.has_points():
            self.reset_tip()
        return self

    def _handle_scale_side_effects(self, scale_factor: float) -> Self:
        # 缩放后（比例非1），重置尖端以适配缩放后的线段尺寸
        if scale_factor != 1.0:
            self.reset_tip()
        return self


class Arrow(Line):
    '''
    创建带填充尖端的箭头（Arrow），支持直线或弧形路径，尖端样式可通过多个参数精细控制。
    
    参数
    ----------
    start : array_like
        箭头的起点坐标或Mobject对象
    end : array_like
        箭头的终点坐标或Mobject对象
    buff : float, optional
        箭头与起点/终点的缓冲距离，默认值为MED_SMALL_BUFF
    path_arc : float, optional
        箭头路径的弯曲弧度，默认0（直线箭头）；非0时为弧形箭头
    thickness : float, optional
        箭杆的基础厚度，影响箭杆宽度，默认值为3.0
    tip_width_ratio : float, optional
        尖端宽度与箭杆宽度的比例，默认值为5（尖端宽度是箭杆的5倍）
    tip_angle : float, optional
        尖端的夹角（两斜边形成的角），默认值为PI/3（60度）
    max_tip_length_to_length_ratio : float, optional
        尖端长度与箭头总长度的最大比例，默认0.5（尖端最长不超过总长度的50%）
    max_width_to_length_ratio : float, optional
        箭杆宽度与箭头总长度的最大比例，默认0.1（避免箭杆过粗）
    fill_color : ManimColor, optional
        箭头（含尖端）的填充颜色，默认值为DEFAULT_LIGHT_COLOR
    fill_opacity : float, optional
        填充不透明度，默认值为1.0（完全不透明）
    stroke_width : float, optional
        描边宽度，默认值为0.0（无描边，仅填充）
    **kwargs
        传递给父类Line的其他参数
    
    示例
    --------
    >>> arrow = Arrow((0, 0, 0), (3, 0, 0))
    >>> curved_arrow = Arrow(LEFT, RIGHT, path_arc=PI/4)
    >>> thick_arrow = Arrow(UP, DOWN, thickness=5.0, tip_width_ratio=3)
    
    返回
    -------
    Arrow
        符合指定参数的箭头对象
    '''

    # 厚度乘数：用于将thickness参数转换为实际箭杆宽度
    tickness_multiplier = 0.015

    def __init__(
        self,
        start: Vect3 | Mobject = LEFT,
        end: Vect3 | Mobject = LEFT,  # 注：此处示例默认值应为RIGHT，可能是笔误，保留原代码
        buff: float = MED_SMALL_BUFF,
        path_arc: float = 0,
        fill_color: ManimColor = DEFAULT_LIGHT_COLOR,
        fill_opacity: float = 1.0,
        stroke_width: float = 0.0,
        thickness: float = 3.0,
        tip_width_ratio: float = 5,
        tip_angle: float = PI / 3,
        max_tip_length_to_length_ratio: float = 0.5,
        max_width_to_length_ratio: float = 0.1,** kwargs,
    ):
        # 记录箭头专属的样式控制参数
        self.thickness = thickness
        self.tip_width_ratio = tip_width_ratio
        self.tip_angle = tip_angle
        self.max_tip_length_to_length_ratio = max_tip_length_to_length_ratio
        self.max_width_to_length_ratio = max_width_to_length_ratio
        
        # 调用父类Line的初始化方法，传递基础路径和样式参数
        super().__init__(
            start, end,
            fill_color=fill_color,
            fill_opacity=fill_opacity,
            stroke_width=stroke_width,
            buff=buff,
            path_arc=path_arc,
            **kwargs
        )

    def get_key_dimensions(self, length):
        # 1. 计算基础箭杆宽度：厚度 × 厚度乘数（将thickness参数转换为实际宽度）
        width = self.thickness * self.tickness_multiplier
        # 2. 限制箭杆宽度：不超过“最大宽长比”，避免过粗
        w_ratio = fdiv(self.max_width_to_length_ratio, fdiv(width, length))  # 宽长比是否超限
        if w_ratio < 1:  # 超限则按比例缩小宽度
            width *= w_ratio

        # 3. 计算基础尖端宽度：箭杆宽度 × 尖端宽度比例
        tip_width = self.tip_width_ratio * width
        # 4. 计算基础尖端长度：由尖端宽度和尖端夹角推导（三角形几何关系）
        tip_length = tip_width / (2 * np.tan(self.tip_angle / 2))
        # 5. 限制尖端长度：不超过“最大尖长比”，避免尖端过长
        t_ratio = fdiv(self.max_tip_length_to_length_ratio, fdiv(tip_length, length))  # 尖长比是否超限
        if t_ratio < 1:  # 超限则按比例缩小尖端长度和宽度
            tip_length *= t_ratio
            tip_width *= t_ratio

        return width, tip_width, tip_length

    def set_points_by_ends(
        self,
        start: Vect3,
        end: Vect3,
        buff: float = 0,
        path_arc: float = 0
    ) -> Self:
        # 计算起点到终点的向量及长度（避免零长度）
        vect = end - start
        length = max(get_norm(vect), 1e-8)  # More systematic min?
        unit_vect = normalize(vect)

        # 1. 获取箭杆宽度、尖端宽度、尖端长度（已按比例限制）
        width, tip_width, tip_length = self.get_key_dimensions(length - buff)

        # 2. 根据路径类型（直线/弧形）调整起点和终点（预留缓冲和尖端空间）
        if path_arc == 0:  # 直线箭头：直接沿向量方向偏移缓冲距离
            start = start + buff * unit_vect
            end = end - buff * unit_vect
        else:  # 弧形箭头：按圆弧半径计算偏移后的起点和终点
            R = length / 2 / math.sin(path_arc / 2)  # 圆弧半径
            midpoint = 0.5 * (start + end)  # 起点终点中点
            # 计算圆弧圆心（垂直于向量方向）
            center = midpoint + rotate_vector(0.5 * vect, PI / 2) / math.tan(path_arc / 2)
            # 按缓冲距离偏移起点和终点（沿圆弧切线方向）
            start = center + rotate_vector(start - center, buff / R)
            end = center + rotate_vector(end - center, -buff / R)
            # 调整圆弧角度（减去缓冲和尖端占用的角度）
            path_arc -= (2 * buff + tip_length) / R
        # 更新调整后的向量和长度
        vect = end - start
        length = get_norm(vect)

        # 3. 生成箭杆的轮廓点（分直线和弧形两种情况）
        if path_arc == 0:  # 直线箭杆：生成左右两侧的直线点
            # 左侧箭杆点（假设箭头初始朝左）
            points1 = (length - tip_length) * np.array([RIGHT, 0.5 * RIGHT, ORIGIN])
            points1 += width * UP / 2  # 上移半个箭杆宽度
            # 右侧箭杆点（与左侧对称，下移）
            points2 = points1[::-1] + width * DOWN
        else:  # 弧形箭杆：生成内外两侧的圆弧点
            # 生成圆弧的二次贝塞尔曲线点（正向和反向）
            points1 = quadratic_bezier_points_for_arc(path_arc)
            points2 = np.array(points1[::-1])
            # 缩放圆弧点到实际半径（外侧+半个宽度，内侧-半个宽度）
            points1 *= (R + width / 2)
            points2 *= (R - width / 2)
            # 旋转圆弧点，使其方向匹配箭头路径
            rot_T = rotation_matrix_transpose(PI / 2 - path_arc, OUT)
            for points in points1, points2:
                points[:] = np.dot(points, rot_T)
                points += R * DOWN  # 平移到正确位置
        # 注：此时箭杆点的方向和位置尚未匹配最终的起点终点，后续会调整

        # 4. 组装箭头轮廓：箭杆 + 尖端 + 闭合
        self.set_points(points1)  # 添加左侧/外侧箭杆点
        self.add_line_to(tip_width * UP / 2)  # 连接到尖端上顶点
        self.add_line_to(tip_length * LEFT)  # 连接到尖端顶点（朝左）
        self.tip_index = len(self.get_points()) - 1  # 记录尖端顶点位置
        self.add_line_to(tip_width * DOWN / 2)  # 连接到尖端下顶点
        self.add_line_to(points2[0])  # 连接到右侧/内侧箭杆起点
        self.add_subpath(points2)  # 添加右侧/内侧箭杆点
        self.add_line_to(points1[0])  # 闭合轮廓（回到左侧/外侧箭杆起点）

        # 5. 调整箭头的方向和位置，使其匹配目标起点和终点
        # 旋转箭头，使其方向与目标向量一致
        self.rotate(angle_of_vector(vect) - self.get_angle())
        # 微调旋转（处理3D方向，确保箭头垂直于向量平面）
        self.rotate(
            PI / 2 - np.arccos(normalize(vect)[2]),
            axis=rotate_vector(self.get_unit_vector(), -PI / 2),
        )
        # 平移箭头，使其起点与目标起点重合
        self.shift(start - self.get_start())
        return self

    def reset_points_around_ends(self) -> Self:
        # 基于当前的起点和终点，重新调用set_points_by_ends生成箭头轮廓
        # 作用：修改参数（如厚度、弧度）后，快速刷新箭头形状
        self.set_points_by_ends(
            self.get_start().copy(),  # 复制当前起点，避免原数据被修改
            self.get_end().copy(),    # 复制当前终点
            path_arc=self.path_arc    # 保留当前的路径弧度
        )
        return self

    def get_start(self) -> Vect3:
        # 箭头起点定义：箭杆两侧起始点的中点（而非单一顶点，确保与箭杆宽度匹配）
        points = self.get_points()
        return 0.5 * (points[0] + points[-3])  # points[0]为左侧起点，points[-3]为右侧起点

    def get_end(self) -> Vect3:
        # 箭头终点定义：尖端的顶点（由之前记录的tip_index定位）
        return self.get_points()[self.tip_index]

    def get_start_and_end(self):
        # 直接返回起点和终点的元组，简化外部调用
        return (self.get_start(), self.get_end())

    def put_start_and_end_on(self, start: Vect3, end: Vect3) -> Self:
        # 将箭头的起点和终点强制设置到指定位置，缓冲距离设为0（精准定位）
        self.set_points_by_ends(start, end, buff=0, path_arc=self.path_arc)
        return self

    def scale(self, *args, **kwargs) -> Self:
        # 重写缩放方法：缩放后调用reset_points_around_ends，确保箭头比例（如尖端宽度）重新适配
        super().scale(*args, **kwargs)
        self.reset_points_around_ends()
        return self

    def set_thickness(self, thickness: float) -> Self:
        # 修改箭杆厚度后，刷新箭头形状以适配新厚度
        self.thickness = thickness
        self.reset_points_around_ends()
        return self

    def set_path_arc(self, path_arc: float) -> Self:
        # 修改路径弧度（直线/弧形切换）后，刷新箭头形状以适配新弧度
        self.path_arc = path_arc
        self.reset_points_around_ends()
        return self

    def set_perpendicular_to_camera(self, camera_frame):
        # 调整箭头朝向，使其平面垂直于相机视角（避免3D场景中箭头显示为线）
        # 1. 计算相机到箭头中心的向量
        to_cam = camera_frame.get_implied_camera_location() - self.get_center()
        # 2. 获取箭头当前的法向量（垂直于箭头平面）
        normal = self.get_unit_normal()
        # 3. 获取箭头的方向向量（作为旋转轴，确保旋转时不改变箭头指向）
        axis = normalize(self.get_vector())
        # 4. 计算目标法向量（确保垂直于箭头方向，且朝向相机）
        trg_normal = to_cam - np.dot(to_cam, axis) * axis
        # 5. 生成旋转矩阵，将当前法向量旋转到目标法向量
        mat = rotation_between_vectors(normal, trg_normal)
        # 6. 应用旋转，旋转中心为箭头起点（避免起点偏移）
        self.apply_matrix(mat, about_point=self.get_start())
        return self


class Vector(Arrow):
    '''
    创建向量（Vector）对象，本质是起点固定为原点（ORIGIN）的箭头，用于表示数学中的向量。
    
    参数
    -----
    direction : array_like
        向量的方向坐标（同时决定向量的终点和长度），支持2D或3D坐标
    buff : float, optional
        向量与起点/终点的缓冲距离，默认值为0.0（因起点固定为原点，无额外缓冲需求）
    **kwargs
        传递给父类Arrow的其他参数，如thickness（厚度）、tip_angle（尖端夹角）等
    
    示例 :
            arrow = Vector(direction=LEFT)  # 向左的单位向量
            vector = Vector(direction=(3, 4, 0), thickness=2)  # 终点为(3,4,0)的3D向量
    
    返回
    -----
    out : Vector object
        符合指定参数的向量对象
    '''

    def __init__(
        self,
        direction: Vect3 = RIGHT,
        buff: float = 0.0,** kwargs
    ):
        # 处理2D坐标：若输入为2个值，自动补全z轴为0，转为3D坐标
        if len(direction) == 2:
            direction = np.hstack([direction, 0])
        # 调用父类Arrow的初始化方法，强制起点为原点，终点为direction参数
        super().__init__(ORIGIN, direction, buff=buff, **kwargs)

class CubicBezier(VMobject):
    '''
    创建三次贝塞尔曲线（CubicBezier）对象，通过四个控制点定义平滑的曲线形态。
    
    三次贝塞尔曲线由两类共四个控制点定义：两个锚点（起点和终点）用于确定曲线的端点，
    两个控制点用于控制曲线的弯曲方向和曲率，曲线会从起点出发、向控制点方向弯曲，最终到达终点。

    参数
    ----------
    a0 : array_like
        第一个锚点，即曲线的起点。
    h0 : array_like
        第一个控制点，用于控制曲线从起点（a0）出发时的方向和曲率。
    h1 : array_like
        第二个控制点，用于控制曲线向终点（a1）靠近时的方向和曲率。
    a1 : array_like
        第二个锚点，即曲线的终点。
    **kwargs
        传递给父类VMobject的额外参数，例如描边颜色（stroke_color）、描边宽度（stroke_width）、
        填充颜色（fill_color）、填充不透明度（fill_opacity）等。

    返回
    -------
    CubicBezier
        表示指定三次贝塞尔曲线的CubicBezier对象。

    '''

    def __init__(
        self,
        a0: Vect3,
        h0: Vect3,
        h1: Vect3,
        a1: Vect3,** kwargs
    ):
        # 调用父类VMobject的初始化方法，配置基础样式
        super().__init__(**kwargs)
        # 直接调用VMobject的方法，根据四个控制点生成三次贝塞尔曲线
        self.add_cubic_bezier_curve(a0, h0, h1, a1)


class Polygon(VMobject):
    '''
    通过连接指定顶点创建多边形（Polygon）对象，顶点按输入顺序依次连接，最后自动闭合。
    
    参数
    -----
    *vertices : array_like
        多边形的顶点坐标，支持传入多个2D或3D坐标，数量需≥3（构成闭合图形）
    
    示例 :
            triangle = Polygon((-3,0,0), (3,0,0), (0,3,0))  # 三角形（3个顶点）
            square = Polygon((1,1,0), (1,-1,0), (-1,-1,0), (-1,1,0))  # 正方形（4个顶点）
    
    返回
    -----
    out : Polygon object
        符合指定参数的多边形对象
    '''

    def __init__(
        self,
        *vertices: Vect3,** kwargs
    ):
        # 调用父类VMobject的初始化方法，配置填充、描边等基础样式
        super().__init__(**kwargs)
        # 将顶点按顺序转为角点路径，最后添加第一个顶点实现闭合（[*vertices, vertices[0]]）
        self.set_points_as_corners([*vertices, vertices[0]])

    def get_vertices(self) -> Vect3Array:
        # 返回多边形的所有顶点（取起点锚点，即角点路径的转折点）
        return self.get_start_anchors()

    def round_corners(self, radius: Optional[float] = None) -> Self:
        # 若未指定圆角半径，自动计算：取最短边长度的25%（避免圆角过大导致图形变形）
        if radius is None:
            verts = self.get_vertices()
            # 计算所有相邻顶点间的边长，取最小值
            min_edge_length = min(
                get_norm(v1 - v2)
                for v1, v2 in zip(verts, verts[1:])  # 遍历相邻顶点对
                if not np.isclose(v1, v2).all()     # 排除重合顶点（避免除以0）
            )
            radius = 0.25 * min_edge_length  # 自动半径 = 最短边的1/4
        
        # 1. 获取所有顶点，准备处理每个角
        vertices = self.get_vertices()
        arcs = []  # 存储每个角的圆弧（用于替换直角）
        
        # 2. 遍历每个顶点（v2），结合前一个顶点（v1）和后一个顶点（v3）计算圆角
        for v1, v2, v3 in adjacent_n_tuples(vertices, 3):
            # 计算顶点v2处的两个邻边方向（单位向量）
            vect1 = normalize(v2 - v1)  # v1→v2的方向
            vect2 = normalize(v3 - v2)  # v2→v3的方向
            # 计算两个邻边的夹角（决定圆角的弧度）
            angle = angle_between_vectors(vect1, vect2)
            
            # 计算“切角长度”：从顶点v2向两边截取的距离（确保圆弧平滑衔接）
            cut_off_length = radius * np.tan(angle / 2)
            # 确定圆弧方向：正半径凸向外侧，负半径凸向内侧（通过叉积判断）
            sign = float(np.sign(radius * cross2d(vect1, vect2)))
            
            # 创建顶点v2处的圆角圆弧：连接两个切角点
            arc = ArcBetweenPoints(
                v2 - vect1 * cut_off_length,  # 圆弧起点（v1→v2方向上的切角点）
                v2 + vect2 * cut_off_length,  # 圆弧终点（v2→v3方向上的切角点）
                angle=sign * angle,           # 圆弧的角度（含方向）
                n_components=2,               # 圆弧的细分组件数
            )
            arcs.append(arc)  # 收集当前顶点的圆弧

        # 3. 重新构建圆角多边形的轮廓
        self.clear_points()  # 清空原直角顶点的路径
        arcs = [arcs[-1], *arcs[:-1]]  # 调整圆弧顺序，确保闭合衔接
        
        # 4. 依次添加每个圆弧，并连接相邻圆弧的端点
        for arc1, arc2 in adjacent_pairs(arcs):
            self.add_subpath(arc1.get_points())  # 添加当前圆弧的点集
            self.add_line_to(arc2.get_start())   # 连接到下一个圆弧的起点
        return self

class Polyline(VMobject):
    '''
    创建折线（Polyline）对象，按输入顶点顺序连接线段，但**不自动闭合**（区别于Polygon）。
    
    参数
    -----
    *vertices : array_like
        折线的顶点坐标，支持传入多个2D或3D坐标，数量需≥2（构成至少一条线段）
    **kwargs
        传递给父类VMobject的额外参数，如描边颜色（stroke_color）、描边宽度（stroke_width）等
    
    示例 :
            polyline = Polyline((0,0,0), (2,1,0), (-1,3,0))  # 两段线段组成的折线，不闭合
    '''
    def __init__(
        self,
        *vertices: Vect3,** kwargs
    ):
        # 调用父类VMobject的初始化方法，配置基础样式
        super().__init__(**kwargs)
        # 按顶点顺序生成角点路径，仅连接顶点，不额外添加起点闭合
        self.set_points_as_corners(vertices)


class RegularPolygon(Polygon):
    '''
    创建正多边形（RegularPolygon）对象，所有边长相等、内角相等，默认居中显示。
    
    参数
    -----
    n : int
        正多边形的顶点数量（如n=3为正三角形，n=5为正五边形），默认值为6（正六边形）
    radius : float
        正多边形的外接圆半径（顶点到中心的距离），默认值为1.0
    start_angle : float | None, optional
        正多边形的起始角度（单位：弧度），按逆时针方向计算；
        默认为None，此时奇数边正多边形起始角度为0，偶数边为90度（DEG），确保图形端正
    **kwargs
        传递给父类Polygon的额外参数，如填充颜色（fill_color）、描边宽度（stroke_width）等
    
    示例 :
            pentagon = RegularPolygon(n=5, start_angle=30 * DEGREES)  # 起始角度30度的正五边形
            hexagon = RegularPolygon(n=6, radius=2, fill_color=BLUE)  # 外接圆半径2、蓝色填充的正六边形
    
    返回
    -----
    out : RegularPolygon object
        符合指定参数的正多边形对象
    '''

    def __init__(
        self,
        n: int = 6,
        radius: float = 1.0,
        start_angle: float | None = None,** kwargs
    ):
        # 处理默认起始角度：奇数边（n%2=1）为0，偶数边为90度，避免图形偏移
        if start_angle is None:
            start_angle = (n % 2) * 90 * DEG
        
        # 1. 计算起始顶点的向量：从中心沿起始角度指向第一个顶点
        start_vect = rotate_vector(radius * RIGHT, start_angle)
        # 2. 生成所有顶点：按等角度间隔分布在半径为radius的圆上（compass_directions实现均分）
        vertices = compass_directions(n, start_vect)
        # 3. 调用父类Polygon的初始化方法，传入顶点生成闭合正多边形
        super().__init__(*vertices, **kwargs)


class Triangle(RegularPolygon):
    '''
    创建正三角形（Triangle）对象，即顶点数固定为3的正多边形，默认居中显示，所有边长和内角均相等。
    
    参数
    -----
    start_angle : float, optional
        正三角形的起始角度（单位：弧度），按逆时针方向计算；
        未指定时默认遵循RegularPolygon规则，即奇数边起始角度为0，确保图形端正
    radius : float, optional
        正三角形的外接圆半径（顶点到中心的距离），默认值为1.0（继承自RegularPolygon）
    **kwargs
        传递给父类RegularPolygon的额外参数，如填充颜色（fill_color）、描边宽度（stroke_width）等
    
    示例 :
            triangle = Triangle(start_angle=45 * DEGREES)  # 起始角度45度的正三角形
            red_triangle = Triangle(radius=2, fill_color=RED, stroke_width=2)  # 半径2、红色填充的正三角形
    
    返回
    -----
    out : Triangle object
        符合指定参数的正三角形对象
    '''

    def __init__(self, **kwargs):
        # 调用父类RegularPolygon的初始化方法，强制顶点数n=3（固定为三角形）
        super().__init__(n=3, **kwargs)

class ArrowTip(Triangle):
    '''
    创建箭头尖端（ArrowTip）对象，基于正三角形扩展，支持多种尖端样式（三角形、平滑内侧、圆点），可灵活适配箭头。
    
    参数
    -----
    angle : float, optional
        尖端的旋转角度（单位：弧度），用于匹配箭头的方向，默认值为0
    width : float, optional
        尖端的底部宽度（垂直于尖端方向的尺寸），默认值为DEFAULT_ARROW_TIP_WIDTH
    length : float, optional
        尖端的长度（沿尖端方向的尺寸），默认值为DEFAULT_ARROW_TIP_LENGTH
    fill_opacity : float, optional
        尖端的填充不透明度，默认值为1.0（完全不透明）
    fill_color : ManimColor, optional
        尖端的填充颜色，默认值为DEFAULT_MOBJECT_COLOR
    stroke_width : float, optional
        尖端的描边宽度，默认值为0.0（无描边）
    tip_style : int, optional
        尖端样式，0为三角形（默认）、1为内侧平滑形、2为圆点形
    **kwargs
        传递给父类Triangle的额外参数
    '''

    def __init__(
        self,
        angle: float = 0,
        width: float = DEFAULT_ARROW_TIP_WIDTH,
        length: float = DEFAULT_ARROW_TIP_LENGTH,
        fill_opacity: float = 1.0,
        fill_color: ManimColor = DEFAULT_MOBJECT_COLOR,
        stroke_width: float = 0.0,
        tip_style: int = 0,  # triangle=0, inner_smooth=1, dot=2
        **kwargs
    ):
        # 调用父类Triangle的初始化方法，固定正三角形起始角度为0，配置填充和描边
        super().__init__(
            start_angle=0,
            fill_opacity=fill_opacity,
            fill_color=fill_color,
            stroke_width=stroke_width,** kwargs
        )
        
        # 1. 基础缩放：按宽度（高度）和长度（宽度）调整尖端尺寸
        self.set_height(width)
        self.set_width(length, stretch=True)
        
        # 2. 按样式调整尖端形态
        if tip_style == 1:  # 内侧平滑形：调整高度并移动中间点实现平滑效果
            self.set_height(length * 0.9, stretch=True)
            self.data["point"][4] += np.array([0.6 * length, 0, 0])  # 移动第5个点（三角形中间点）
        elif tip_style == 2:  # 圆点形：替换为圆点的点集，按长度一半调整大小
            h = length / 2
            self.set_points(Dot().set_width(h).get_points())
        
        # 3. 旋转尖端到指定角度，匹配箭头方向
        self.rotate(angle)

    def get_base(self) -> Vect3:
        # 返回尖端的底部中心点（按比例0.5取点，对应三角形底边中点）
        return self.point_from_proportion(0.5)

    def get_tip_point(self) -> Vect3:
        # 返回尖端的顶点（取点集中的第一个点，对应三角形的顶端）
        return self.get_points()[0]

    def get_vector(self) -> Vect3:
        # 返回从尖端底部中心指向顶点的向量（代表尖端的方向）
        return self.get_tip_point() - self.get_base()

    def get_angle(self) -> float:
        # 返回尖端方向与x轴正方向的夹角（单位：弧度）
        return angle_of_vector(self.get_vector())

    def get_length(self) -> float:
        # 返回尖端的实际长度（底部中心到顶点的距离）
        return get_norm(self.get_vector())


class Rectangle(Polygon):
    '''
    创建矩形（Rectangle）对象，基于Polygon实现，默认居中显示，支持通过宽高快速定义尺寸。
    
    参数
    -----
    width : float, optional
        矩形的宽度（水平方向尺寸），默认值为4.0
    height : float, optional
        矩形的高度（垂直方向尺寸），默认值为2.0
    **kwargs
        传递给父类Polygon的额外参数，如填充颜色（fill_color）、描边宽度（stroke_width）、颜色（color）等
    
    示例 :
            rectangle = Rectangle(width=3, height=4, color=BLUE)  # 宽3、高4、蓝色的矩形
            filled_rect = Rectangle(width=2, height=1, fill_color=GREEN, fill_opacity=0.8)  # 绿色半透明填充矩形
    
    返回
    -----
    out : Rectangle object
        符合指定参数的矩形对象
    '''

    def __init__(
        self,
        width: float = 4.0,
        height: float = 2.0,** kwargs
    ):
        # 1. 以默认角点（UR右上、UL左上、DL左下、DR右下）创建基础矩形轮廓
        super().__init__(UR, UL, DL, DR, **kwargs)
        # 2. 按指定宽度缩放矩形（stretch=True允许非均匀缩放，不强制保持宽高比）
        self.set_width(width, stretch=True)
        # 3. 按指定高度缩放矩形
        self.set_height(height, stretch=True)

    def surround(self, mobject, buff=SMALL_BUFF) -> Self:
        # 计算包围目标Mobject所需的形状：目标形状尺寸 + 2倍缓冲距离（确保不紧贴）
        target_shape = np.array(mobject.get_shape()) + 2 * buff
        # 按目标形状调整矩形尺寸
        self.set_shape(*target_shape)
        # 将矩形移动到与目标Mobject相同的位置（居中包围）
        self.move_to(mobject)
        return self


class Square(Rectangle):
    '''
    创建正方形（Square）对象，基于Rectangle实现，宽高强制相等，默认居中显示。
    
    参数
    -----
    side_length : float, optional
        正方形的边长（宽和高均等于该值），默认值为2.0
    **kwargs
        传递给父类Rectangle的额外参数，如填充颜色（fill_color）、描边宽度（stroke_width）、颜色（color）等
    
    示例 :
            square = Square(side_length=5, color=PINK)  # 边长5、粉色的正方形
            border_square = Square(side_length=3, stroke_color=BLACK, stroke_width=2, fill_opacity=0)  # 黑色边框、无填充的正方形
    
    返回
    -----
    out : Square object
        符合指定参数的正方形对象
    '''

    def __init__(self, side_length: float = 2.0, **kwargs):
        # 调用父类Rectangle的初始化方法，强制宽和高均为边长（确保正方形形态）
        super().__init__(side_length, side_length, **kwargs)

class RoundedRectangle(Rectangle):
    '''
    创建圆角矩形（RoundedRectangle）对象，基于Rectangle扩展，通过添加圆角半径参数实现四角圆润效果，默认居中显示。
    
    参数
    -----
    width : float, optional
        圆角矩形的宽度（水平方向尺寸），默认值为4.0（继承自Rectangle）
    height : float, optional
        圆角矩形的高度（垂直方向尺寸），默认值为2.0（继承自Rectangle）
    corner_radius : float, optional
        圆角矩形的拐角半径，半径越大拐角越圆润，默认值为0.5
    **kwargs
        传递给父类Rectangle的额外参数，如填充颜色（fill_color）、描边宽度（stroke_width）、颜色（color）等
    
    示例 :
            rRectangle = RoundedRectangle(width=3, height=4, corner_radius=1, color=BLUE)  # 宽3、高4、圆角1、蓝色的圆角矩形
            soft_rect = RoundedRectangle(width=2, height=1, corner_radius=0.8, fill_color=GREY)  # 高圆角、灰色填充的圆角矩形
    
    返回
    -----
    out : RoundedRectangle object
        符合指定参数的圆角矩形对象
    '''

    def __init__(
        self,
        width: float = 4.0,
        height: float = 2.0,
        corner_radius: float = 0.5,** kwargs
    ):
        # 1. 调用父类Rectangle的初始化方法，先创建基础矩形
        super().__init__(width, height, **kwargs)
        # 2. 调用Polygon类的round_corners方法，按指定半径添加圆角效果
        self.round_corners(corner_radius)
