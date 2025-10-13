# 导入Python未来版本的注解特性（支持字符串形式类名、泛型等灵活类型提示）
from __future__ import annotations

# 导入基础数学库（用于角度计算、三角函数等）
import math

# 导入数值计算库（核心用于向量运算、矩阵变换、数组处理）
import numpy as np

# 导入Manim核心常量（方向向量、颜色、缓冲距离、角度单位等）
from manimlib.constants import DL, DOWN, DR, LEFT, ORIGIN, OUT, RIGHT, UL, UP, UR  # 方向向量（如UP=(0,1,0)，LEFT=(-1,0,0)）
from manimlib.constants import RED, BLACK, DEFAULT_MOBJECT_COLOR, DEFAULT_LIGHT_COLOR  # 颜色常量（默认图形色、高亮色等）
from manimlib.constants import MED_SMALL_BUFF, SMALL_BUFF  # 缓冲距离（用于对象间间距控制，如SMALL_BUFF=0.1）
from manimlib.constants import DEG, PI, TAU  # 角度/圆周常量（DEG=π/180，TAU=2π，即一个圆周）

# 导入Manim基础图形类（向量图形基类、分组类、虚线类）
from manimlib.mobject.mobject import Mobject  # 所有可见对象的基类
from manimlib.mobject.types.vectorized_mobject import DashedVMobject  # 虚线向量图形类（如虚线、虚线圆弧）
from manimlib.mobject.types.vectorized_mobject import VGroup  # 向量图形分组类（批量管理VMobject，支持向量运算）
from manimlib.mobject.types.vectorized_mobject import VMobject  # 向量图形基类（支持路径绘制、填充、描边，如线段、圆弧）

# 导入Manim工具函数（贝塞尔曲线、迭代器、数值处理、空间运算）
from manimlib.utils.bezier import quadratic_bezier_points_for_arc  # 生成圆弧的二次贝塞尔曲线路径点
from manimlib.utils.iterables import adjacent_n_tuples  # 生成相邻n元组（如[1,2,3]→[(1,2), (2,3)]）
from manimlib.utils.iterables import adjacent_pairs  # 生成相邻对（同adjacent_n_tuples(n=2)，简化调用）
from manimlib.utils.simple_functions import clip  # 数值裁剪函数（将值限制在指定范围内，如clip(5, 0, 4)=4）
from manimlib.utils.simple_functions import fdiv  # 浮点除法函数（确保结果为float，如fdiv(3,2)=1.5）
from manimlib.utils.space_ops import angle_between_vectors  # 计算两个向量间的夹角（返回弧度值）
from manimlib.utils.space_ops import angle_of_vector  # 计算向量与x轴正方向的夹角（返回弧度值，逆时针为正）
from manimlib.utils.space_ops import cross2d  # 计算二维向量的叉积（返回标量，用于判断方向、是否共线）
from manimlib.utils.space_ops import compass_directions  # 获取罗盘方向向量列表（如[UP, RIGHT, DOWN, LEFT]）
from manimlib.utils.space_ops import find_intersection  # 计算两条线段的交点（返回交点坐标或None）
from manimlib.utils.space_ops import get_norm  # 计算向量的模长（即向量长度，如get_norm((3,4))=5）
from manimlib.utils.space_ops import normalize  # 归一化向量（将向量缩放至模长为1，方向不变）
from manimlib.utils.space_ops import rotate_vector  # 旋转向量（按指定角度旋转二维/三维向量，返回新向量）
from manimlib.utils.space_ops import rotation_matrix_transpose  # 旋转矩阵的转置（用于反向旋转，优化计算效率）
from manimlib.utils.space_ops import rotation_between_vectors  # 计算两个向量间的旋转矩阵（用于对象定向对齐）

# 导入类型提示相关模块（仅在类型检查时生效，不影响运行时）
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Iterable, Optional  # 可迭代对象类型、可选类型（允许为None）
    from manimlib.typing import ManimColor, Vect3, Vect3Array, Self  # Manim自定义类型：颜色、三维向量、三维向量数组、自身类型


# 图形默认参数（可在实例化时通过config覆盖）
DEFAULT_DOT_RADIUS = 0.08  # 普通点（Dot）的默认半径
DEFAULT_SMALL_DOT_RADIUS = 0.04  # 小点（如标记点）的默认半径
DEFAULT_DASH_LENGTH = 0.05  # 虚线（DashedVMobject）的默认线段长度
DEFAULT_ARROW_TIP_LENGTH = 0.35  # 箭头尖端的默认长度
DEFAULT_ARROW_TIP_WIDTH = 0.35  # 箭头尖端的默认宽度


# 可添加尖端的向量图形基类（用于Line、Arc等需要带箭头的图形）
# 注：注释中提到"Deprecate?"，表示该类可能在未来版本中被废弃
class TipableVMobject(VMobject):
    """
    用于Line（线段）和Arc（圆弧）等图形的共享基类，核心提供“添加尖端（如箭头）”的功能。
    功能可大致分为以下几类：

        1. 尖端的添加、创建与修改
            - add_tip() 方法调用 create_tip() 创建尖端，再将新尖端加入子对象列表
            - 支持尖端的样式配置（如填充透明度、线宽）和位置配置（如尖端朝向）

        2. 尖端存在性检查
            - 布尔值检查：判断当前图形是否带有尖端，或是否带有起始端尖端（双向箭头场景）

        3. 信息获取（Getters）
            - 简单的访问方法：返回尖端相关信息（如尖端对象列表）、图形长度等
    """
    # 尖端的默认配置（子类或实例化时可覆盖）
    tip_config: dict = dict(
        fill_opacity=1.0,  # 尖端填充透明度（1.0为完全不透明）
        stroke_width=0.0,  # 尖端描边线宽（0.0为无描边，仅填充）
        tip_style=0.0,     # 尖端样式（0=三角形，1=内部平滑型，2=点型）
    )

    # Adding, Creating, Modifying tips
    # 尖端（如箭头）的添加、创建与修改相关方法
    def add_tip(self, at_start: bool = False, **kwargs) -> Self:
        """
        为当前可添加尖端的向量图形（如线段、圆弧）添加尖端（默认箭头）。
        若指定“起始端添加”（at_start=True），会自动调整图形端点以匹配尖端位置。

        参数说明：
            at_start: 布尔值，True表示在图形起始端添加尖端，False表示在末端添加（默认）
            **kwargs: 关键字参数，用于覆盖尖端默认配置（如tip_style、fill_opacity）
        返回：添加尖端后的当前图形对象（支持链式调用，如line.add_tip().set_color(RED)）
        """
        # 1. 创建尖端对象（根据at_start和配置参数）
        tip = self.create_tip(at_start, **kwargs)
        # 2. 根据尖端位置调整图形的端点（确保尖端与图形衔接自然）
        self.reset_endpoints_based_on_tip(tip, at_start)
        # 3. 为尖端绑定属性（标记是“起始端尖端”还是“末端尖端”）
        self.asign_tip_attr(tip, at_start)
        # 4. 让尖端颜色与图形描边颜色一致（保持风格统一）
        tip.set_color(self.get_stroke_color())
        # 5. 将尖端作为子对象添加到当前图形中
        self.add(tip)
        return self

    def create_tip(self, at_start: bool = False, **kwargs) -> ArrowTip:
        """
        创建并返回一个配置好样式和位置的尖端对象（ArrowTip）。
        流程：先配置尖端样式（无位置），再根据图形端点和方向确定尖端位置。

        参数说明：
            at_start: 布尔值，指定尖端添加在起始端还是末端
            **kwargs: 覆盖尖端样式的配置参数
        返回：配置完成且位置正确的ArrowTip对象
        """
        # 1. 获取样式配置完成但未定位的尖端
        tip = self.get_unpositioned_tip(**kwargs)
        # 2. 根据图形端点和at_start参数，为尖端定位
        self.position_tip(tip, at_start)
        return tip

    def get_unpositioned_tip(self, **kwargs) -> ArrowTip:
        """
        创建并返回一个“仅配置样式、未指定位置”的尖端对象。
        配置优先级：实例kwargs > 类默认tip_config（确保灵活覆盖）。

        参数：**kwargs - 尖端样式配置（如fill_opacity、tip_style）
        返回：样式配置完成但未定位的ArrowTip对象
        """
        # 1. 初始化配置字典（基础为类默认tip_config）
        config = dict()
        config.update(self.tip_config)  # 合并类默认配置
        config.update(kwargs)  # 用传入参数覆盖默认配置
        # 2. 基于配置创建并返回ArrowTip对象（未定位）
        return ArrowTip(**config)

    def position_tip(self, tip: ArrowTip, at_start: bool = False) -> ArrowTip:
        """
        为尖端对象（ArrowTip）定位，确保其与图形端点对齐且方向匹配图形切线方向。
        核心逻辑：通过图形的端点和相邻控制点，计算尖端的旋转角度和位移。

        参数说明：
            tip: 待定位的ArrowTip对象
            at_start: 布尔值，指定尖端对应图形的起始端还是末端
        返回：定位完成的ArrowTip对象
        """
        # 确定尖端的“锚点”（图形端点）和“方向控制点”（相邻的路径点，用于计算切线方向）
        if at_start:
            anchor = self.get_start()  # 尖端锚点 = 图形起始端
            handle = self.get_first_handle()  # 方向控制点 = 起始端的下一个路径点
        else:
            handle = self.get_last_handle()  # 方向控制点 = 末端的前一个路径点
            anchor = self.get_end()  # 尖端锚点 = 图形末端

        # 1. 计算尖端旋转角度：匹配图形在锚点处的切线方向
        # - 向量(handle - anchor)的方向 = 图形在锚点处的切线方向
        # - 减去PI（180度）是因为ArrowTip默认朝向与切线方向相反，需翻转
        # - 减去tip.get_angle()是为了抵消尖端自身的初始角度偏移
        tip.rotate(angle_of_vector(handle - anchor) - PI - tip.get_angle())

        # 2. 移动尖端：让尖端的“尖端点”与图形锚点完全重合（确保衔接无缝）
        tip.shift(anchor - tip.get_tip_point())

        return tip

    def reset_endpoints_based_on_tip(self, tip: ArrowTip, at_start: bool) -> Self:
        """
        根据尖端位置调整图形的端点，避免图形与尖端重叠（确保视觉上衔接自然）。
        仅当图形长度非零时生效（长度为0时无法调整端点）。

        参数说明：
            tip: 已定位的ArrowTip对象
            at_start: 布尔值，标记尖端在起始端还是末端
        返回：端点调整后的当前图形对象
        """
        # 若图形长度为0（无实际路径），跳过调整（避免错误）
        if self.get_length() == 0:
            return self

        # 根据尖端位置，重新定义图形的起始端和末端
        if at_start:
            # 起始端添加尖端：图形新起始端 = 尖端的基部（避免与尖端重叠）
            new_start = tip.get_base()
            new_end = self.get_end()  # 末端保持不变
        else:
            new_start = self.get_start()  # 起始端保持不变
            # 末端添加尖端：图形新末端 = 尖端的基部
            new_end = tip.get_base()

        # 调用VMobject的方法，将图形端点重置为新计算的位置
        self.put_start_and_end_on(new_start, new_end)
        return self

    def asign_tip_attr(self, tip: ArrowTip, at_start: bool) -> Self:
        """
        为尖端绑定属性，在当前图形对象上标记尖端类型（起始端/末端），便于后续查询。

        参数说明：
            tip: 已添加的ArrowTip对象
            at_start: 布尔值，标记尖端在起始端还是末端
        返回：绑定属性后的当前图形对象
        """
        if at_start:
            # 起始端尖端：绑定为self.start_tip属性
            self.start_tip = tip
        else:
            # 末端尖端：绑定为self.tip属性
            self.tip = tip
        return self

    # 尖端存在性检查相关方法
    def has_tip(self) -> bool:
        """
        检查当前图形是否包含“末端尖端”（默认添加的尖端）。
        返回：布尔值，True表示有末端尖端，False表示无。
        逻辑：1. 检查是否有self.tip属性；2. 确认该尖端是当前图形的子对象（避免已移除的情况）
        """
        return hasattr(self, "tip") and self.tip in self

    def has_start_tip(self) -> bool:
        """
        检查当前图形是否包含“起始端尖端”（at_start=True添加的尖端）。
        返回：布尔值，True表示有起始端尖端，False表示无。
        逻辑：1. 检查是否有self.start_tip属性；2. 确认该尖端是当前图形的子对象
        """
        return hasattr(self, "start_tip") and self.start_tip in self

    # 尖端信息获取相关方法（Getters）
    def pop_tips(self) -> VGroup:
        """
        移除当前图形上所有尖端（起始端和末端），并将这些尖端组成VGroup返回。
        移除后会恢复图形原始的端点位置（避免因尖端调整导致的端点偏移）。

        返回：包含所有被移除尖端的VGroup对象（可用于后续复用）
        """
        # 先记录图形原始的端点位置（移除尖端后需恢复）
        original_start, original_end = self.get_start_and_end()
        # 初始化存储被移除尖端的VGroup
        removed_tips = VGroup()

        # 1. 移除末端尖端（若存在）
        if self.has_tip():
            removed_tips.add(self.tip)
            self.remove(self.tip)
        # 2. 移除起始端尖端（若存在）
        if self.has_start_tip():
            removed_tips.add(self.start_tip)
            self.remove(self.start_tip)

        # 恢复图形原始的端点位置（抵消之前因添加尖端导致的调整）
        self.put_start_and_end_on(original_start, original_end)
        return removed_tips

    def get_tips(self) -> VGroup:
        """
        获取当前图形上所有尖端（起始端和末端），组成VGroup返回（不移除尖端）。
        返回：包含所有尖端的VGroup对象（若无糖端则为空VGroup）
        """
        tips_group = VGroup()
        # 若有末端尖端，添加到组中
        if hasattr(self, "tip"):
            tips_group.add(self.tip)
        # 若有起始端尖端，添加到组中
        if hasattr(self, "start_tip"):
            tips_group.add(self.start_tip)
        return tips_group

    def get_tip(self) -> ArrowTip:
        """
        获取当前图形的“第一个尖端”（优先返回末端尖端，无则返回起始端尖端）。
        若图形无任何尖端，抛出异常（确保调用时存在尖端）。

        返回：当前图形的第一个ArrowTip对象
        异常：若无糖端，抛出Exception("tip not found")
        """
        all_tips = self.get_tips()
        if len(all_tips) == 0:
            raise Exception("tip not found")
        else:
            return all_tips[0]  # 返回第一个尖端（通常是末端尖端）

    def get_default_tip_length(self) -> float:
        """
        获取尖端的默认长度（依赖图形自身的tip_length属性）。
        返回：尖端默认长度（浮点值，如DEFAULT_ARROW_TIP_LENGTH=0.35）
        """
        return self.tip_length

    def get_first_handle(self) -> Vect3:
        """
        获取图形路径的“第一个控制点”（用于计算起始端切线方向，辅助尖端定位）。
        返回：第一个控制点的三维坐标（Vect3类型）
        逻辑：图形路径点列表的第2个点（索引1，第1个点是起始端）
        """
        return self.get_points()[1]

    def get_last_handle(self) -> Vect3:
        """
        获取图形路径的“最后一个控制点”（用于计算末端切线方向，辅助尖端定位）。
        返回：最后一个控制点的三维坐标（Vect3类型）
        逻辑：图形路径点列表的倒数第2个点（索引-2，最后1个点是末端）
        """
        return self.get_points()[-2]

    def get_end(self) -> Vect3:
        """
        获取图形的“实际末端”（若有末端尖端，返回尖端的基部；若无则返回图形原始末端）。
        返回：图形实际末端的三维坐标（Vect3类型）
        作用：确保后续基于“末端”的操作（如连接其他图形）能与尖端自然衔接
        """
        if self.has_tip():
            return self.tip.get_start()  # 有尖端时，末端 = 尖端基部
        else:
            return VMobject.get_end(self)  # 无尖端时，调用父类方法获取原始末端

    def get_start(self) -> Vect3:
        """
        获取图形的“实际起始端”（若有起始端尖端，返回尖端的基部；若无则返回图形原始起始端）。
        返回：图形实际起始端的三维坐标（Vect3类型）
        作用：确保后续基于“起始端”的操作能与尖端自然衔接
        """
        if self.has_start_tip():
            return self.start_tip.get_start()  # 有起始端尖端时，起始端 = 尖端基部
        else:
            return VMobject.get_start(self)  # 无尖端时，调用父类方法获取原始起始端

    def get_length(self) -> float:
        """
        计算图形的“实际长度”（基于实际起始端和实际末端的直线距离）。
        返回：图形实际长度（浮点值，单位与Manim坐标系一致）
        逻辑：1. 获取实际起始端和末端；2. 计算两点间距离（向量模长）
        """
        actual_start, actual_end = self.get_start_and_end()
        return get_norm(actual_start - actual_end)  # get_norm计算向量模长（距离）


class Arc(TipableVMobject):
    '''
    生成圆弧图形（继承自可添加尖端的向量图形基类TipableVMobject），支持自定义起始角度、弧度、半径和圆心。

    参数说明：
        start_angle : 浮点数，圆弧的起始角度（单位：弧度），默认0（沿x轴正方向）。角度按**逆时针**计算。
        angle : 浮点数，圆弧在圆心处对应的圆心角（单位：弧度），默认TAU/4（45度）。角度按**逆时针**计算，负值表示顺时针。
        radius : 浮点数，圆弧的半径，默认1.0（Manim坐标系单位）。
        n_components : 可选整数，圆弧的贝塞尔曲线分段数（控制圆弧平滑度）。默认根据圆心角自动计算（整圆16段，角度越小分段越少）。
        arc_center : 三维向量（Vect3），圆弧的圆心坐标，默认ORIGIN（原点(0,0,0)）。
        **kwargs : 关键字参数，用于传递VMobject的通用配置（如color、stroke_width、fill_opacity等）。

    示例：
        # 生成以原点为圆心、半径3、起始角45度（TAU/4）、圆心角90度（TAU/2）的圆弧
        arc = Arc(start_angle=TAU/4, angle=TAU/2, radius=3, arc_center=ORIGIN)
        # 生成以(1,2,0)为圆心、半径4.5、圆心角45度、蓝色的圆弧
        arc = Arc(angle=TAU/4, radius=4.5, arc_center=(1,2,0), color=BLUE)

    返回值：
        out : Arc对象，满足指定参数的圆弧图形。
    '''

    def __init__(
            self,
            start_angle: float = 0,
            angle: float = TAU / 4,
            radius: float = 1.0,
            n_components: Optional[int] = None,
            arc_center: Vect3 = ORIGIN,
            **kwargs
    ):
        # 调用父类TipableVMobject的初始化方法，传递通用配置（如颜色、线宽）
        super().__init__(**kwargs)

        # 1. 确定圆弧的贝塞尔曲线分段数（默认按圆心角自动计算，确保平滑度）
        if n_components is None:
            # 整圆（TAU）对应16段，其他角度按比例计算（最少1段）
            n_components = int(15 * (abs(angle) / TAU)) + 1

        # 2. 生成圆弧的二次贝塞尔曲线路径点（默认以原点为圆心、半径1、起始角0）
        arc_points = quadratic_bezier_points_for_arc(angle, n_components)
        self.set_points(arc_points)  # 将路径点设置到当前圆弧对象

        # 3. 旋转圆弧到指定的起始角度（绕原点旋转，因初始路径以原点为中心）
        self.rotate(start_angle, about_point=ORIGIN)

        # 4. 缩放圆弧到指定半径（绕原点缩放，保持圆心在原点）
        self.scale(radius, about_point=ORIGIN)

        # 5. 将圆弧平移到指定的圆心位置（从原点移至arc_center）
        self.shift(arc_center)

    def get_arc_center(self) -> Vect3:
        """
        计算并返回圆弧的圆心坐标（通过圆弧路径的前两个锚点和控制点的法线交点推导）。
        核心逻辑：利用圆弧路径的切线方向计算法线，两条法线的交点即为圆心。

        返回值：三维向量（Vect3），圆弧的圆心坐标。
        """
        # 获取圆弧路径的前3个点：第1个锚点（a1）、第1个控制点（h）、第2个锚点（a2）
        a1, h, a2 = self.get_points()[:3]

        # 计算切线向量：控制点到锚点的向量（反映圆弧在锚点处的切线方向）
        tangent1 = h - a1  # a1处的切线向量
        tangent2 = h - a2  # a2处的切线向量

        # 计算法线向量：将切线向量旋转90度（TAU/4弧度），得到指向圆心的法线
        normal1 = rotate_vector(tangent1, TAU / 4)
        normal2 = rotate_vector(tangent2, TAU / 4)

        # 两条法线的交点即为圆弧的圆心（调用find_intersection计算两直线交点）
        return find_intersection(a1, normal1, a2, normal2)

    def get_start_angle(self) -> float:
        """
        计算并返回圆弧的起始角度（相对于圆心，沿x轴正方向逆时针测量）。
        角度结果取模TAU（0~2π），确保范围在一个圆周内。

        返回值：浮点数，圆弧的起始角度（单位：弧度）。
        """
        # 起始点到圆心的向量（方向即为起始角度的方向）
        start_to_center = self.get_start() - self.get_arc_center()
        # 计算该向量与x轴正方向的夹角，取模TAU确保在0~2π范围内
        return angle_of_vector(start_to_center) % TAU

    def get_stop_angle(self) -> float:
        """
        计算并返回圆弧的终止角度（相对于圆心，沿x轴正方向逆时针测量）。
        角度结果取模TAU（0~2π），确保范围在一个圆周内。

        返回值：浮点数，圆弧的终止角度（单位：弧度）。
        """
        # 终止点到圆心的向量（方向即为终止角度的方向）
        end_to_center = self.get_end() - self.get_arc_center()
        # 计算该向量与x轴正方向的夹角，取模TAU确保在0~2π范围内
        return angle_of_vector(end_to_center) % TAU

    def move_arc_center_to(self, point: Vect3) -> Self:
        """
        将圆弧的圆心移动到指定位置（保持圆弧的形状和角度不变，仅平移整个圆弧）。

        参数：point - 三维向量（Vect3），目标圆心位置。

        返回值：当前Arc对象（支持链式调用，如arc.move_arc_center_to((2,3,0)).set_color(RED)）。
        """
        # 计算当前圆心到目标圆心的平移向量，沿该向量平移圆弧
        self.shift(point - self.get_arc_center())
        return self


class ArcBetweenPoints(Arc):
    '''
    生成经过两个指定点（起点和终点）的圆弧（继承自Arc类），圆心角由angle参数指定。
    核心特点：无需手动计算圆心，只需指定起点、终点和圆心角，自动匹配圆弧路径。

    参数说明：
        start : 三维向量（Vect3），圆弧的起始点坐标。
        end : 三维向量（Vect3），圆弧的终止点坐标。
        angle : 浮点数，圆弧在圆心处对应的圆心角（单位：弧度），默认TAU/4（45度）。
                正值表示**逆时针**圆弧，负值表示**顺时针**圆弧（用于区分两点间的两条可能圆弧）。
        **kwargs : 关键字参数，用于传递Arc/VMobject的通用配置（如color、stroke_width、radius等）。

    示例：
        # 生成经过(0,0,0)和(1,2,0)、圆心角90度（TAU/2）的圆弧
        arc = ArcBetweenPoints(start=(0, 0, 0), end=(1, 2, 0), angle=TAU / 2)
        # 生成经过(-2,3,0)和(1,2,0)、圆心角-15度（-TAU/12，顺时针）、蓝色的圆弧
        arc = ArcBetweenPoints(start=(-2, 3, 0), end=(1, 2, 0), angle=-TAU / 12, color=BLUE)

    返回值：
        out : ArcBetweenPoints对象，满足指定参数的圆弧图形。
    '''

    # 首先补充 ArcBetweenPoints 类 __init__ 方法的完整逻辑（承接上文），再解析后续类
    class ArcBetweenPoints(Arc):
        def __init__(
                self,
                start: Vect3,
                end: Vect3,
                angle: float = TAU / 4,
                **kwargs
        ):
            # 1. 调用父类 Arc 的初始化：先按默认参数（如半径1、圆心原点）创建基础圆弧
            # 注意：此时的半径、圆心会被后续 put_start_and_end_on 覆盖，仅保留圆心角 angle
            super().__init__(angle=angle, **kwargs)

            # 2. 特殊处理：若圆心角为0，无法形成圆弧，退化为线段
            # 先创建从 LEFT(-1,0,0) 到 RIGHT(1,0,0) 的基础线段，后续再调整到目标起点/终点
            if angle == 0:
                self.set_points_as_corners([LEFT, RIGHT])

            # 3. 核心逻辑：强制将圆弧的起点/终点对齐到目标位置，自动计算匹配的圆心和半径
            # put_start_and_end_on 是 VMobject 类的方法，此处 Arc 子类已重写，会根据圆弧特性调整
            self.put_start_and_end_on(start, end)

    class CurvedArrow(ArcBetweenPoints):
        '''
        生成带**单向箭头**的弯曲图形（继承自 ArcBetweenPoints），本质是在“经过两点的圆弧”基础上自动添加末端尖端，
        适用于需要弯曲指向的场景（如流程图分支、数学角度标注、路径引导）。

        参数说明：
            start_point : 三维向量（Vect3），弯曲箭头的起始坐标（如 (0,0,0)、LEFT+UP）。
            end_point : 三维向量（Vect3），弯曲箭头的终止坐标。
            angle : 浮点数，圆弧在圆心处的圆心角（单位：弧度），默认 TAU/4（45°）。
                    - 正值：圆弧**逆时针**弯曲（从起点到终点逆时针绕行）；
                    - 负值：圆弧**顺时针**弯曲（从起点到终点顺时针绕行）；
                    （用于区分两点间的两条可能圆弧，例如两点水平相距2，angle=TAU/2是上半圆，angle=-TAU/2是下半圆）。
            **kwargs : 关键字参数，传递图形样式配置（如 color=RED、stroke_width=2、tip_style=1 等）。

        示例：
            # 生成从 (0,0,0) 到 (1,2,0)、逆时针弯曲90°（TAU/2）的红色弯曲箭头
            curved_arrow = CurvedArrow(start_point=(0,0,0), end_point=(1,2,0), angle=TAU/2, color=RED)
            # 生成从 (-2,3,0) 到 (1,2,0)、顺时针弯曲15°（-TAU/12）、粗线宽的蓝色弯曲箭头
            curved_arrow = CurvedArrow(start_point=(-2,3,0), end_point=(1,2,0), angle=-TAU/12, color=BLUE, stroke_width=3)

        返回值：
            out : CurvedArrow 对象，带末端箭头的弯曲图形。
        '''

        def __init__(
                self,
                start_point: Vect3,
                end_point: Vect3,
                **kwargs
        ):
            # 1. 调用父类 ArcBetweenPoints 的初始化：生成经过两点的圆弧
            super().__init__(start=start_point, end=end_point, **kwargs)
            # 2. 自动添加末端尖端（箭头）：复用 TipableVMobject 的 add_tip 方法，默认 at_start=False（末端添加）
            self.add_tip()

    class CurvedDoubleArrow(CurvedArrow):
        '''
        生成带**双向箭头**的弯曲图形（继承自 CurvedArrow），在“单向弯曲箭头”基础上额外添加起始端尖端，
        适用于需要双向关联的场景（如两个对象的互相依赖、等价关系、双向数据流向）。

        参数说明：
            start_point : 三维向量（Vect3），双向弯曲箭头的起始坐标。
            end_point : 三维向量（Vect3），双向弯曲箭头的终止坐标。
            angle : 浮点数，圆弧的圆心角（单位：弧度），默认 TAU/4（45°），正负意义同 CurvedArrow。
            **kwargs : 关键字参数，传递图形样式配置（如 color=GREEN、tip_width=0.5 等）。

        示例：
            # 生成从 (0,0,0) 到 (1,2,0)、逆时针弯曲90°的双向弯曲箭头
            double_arrow = CurvedDoubleArrow(start_point=(0,0,0), end_point=(1,2,0), angle=TAU/2)
            # 生成从 (-2,3,0) 到 (1,2,0)、顺时针弯曲15°、绿色粗线的双向弯曲箭头
            double_arrow = CurvedDoubleArrow(start_point=(-2,3,0), end_point=(1,2,0), angle=-TAU/12, color=GREEN, stroke_width=3)

        返回值：
            out : CurvedDoubleArrow 对象，带双向箭头的弯曲图形。
        '''

        def __init__(
                self,
                start_point: Vect3,
                end_point: Vect3,
                **kwargs
        ):
            # 1. 调用父类 CurvedArrow 的初始化：生成带末端箭头的弯曲图形
            super().__init__(start_point=start_point, end_point=end_point, **kwargs)
            # 2. 额外添加起始端尖端（箭头）：指定 at_start=True（起始端添加），实现双向箭头
            self.add_tip(at_start=True)

    class Circle(Arc):
        '''
        生成圆形（继承自 Arc），本质是**圆心角为 TAU（2π，即 360°）的完整圆弧**，
        简化了圆形的创建流程（无需手动设置圆心角，直接指定半径和圆心即可）。

        参数说明：
            radius : 浮点数，圆形的半径（Manim 坐标系单位），默认 1.0。
            arc_center : 三维向量（Vect3），圆形的圆心坐标，默认 ORIGIN（原点 (0,0,0)）。
            **kwargs : 关键字参数，传递图形样式配置：
                    - color：边框颜色（如 color=DARK_BLUE）；
                    - fill_opacity：填充透明度（0=透明，1=完全填充，如 fill_opacity=0.5）；
                    - stroke_width：边框线宽（如 stroke_width=2）。

        示例：
            # 生成半径 2、圆心在 (1,2,0) 的黑色圆形
            circle1 = Circle(radius=2, arc_center=(1,2,0))
            # 生成半径 3.14、圆心在 (-2,1,0)（2*LEFT+UP）、深蓝色填充的圆形
            circle2 = Circle(radius=3.14, arc_center=2*LEFT+UP, color=DARK_BLUE, fill_opacity=1.0)

        返回值：
            out : Circle 对象，完整的圆形图形。
        '''

        def __init__(
                self,
                radius: float = 1.0,
                arc_center: Vect3 = ORIGIN,
                **kwargs
        ):
            # 调用父类 Arc 的初始化：强制将圆心角 angle 设为 TAU（360°），确保是完整圆形
            # 其他参数（半径、圆心）直接传递，样式配置通过 **kwargs 传递
            super().__init__(
                angle=TAU,  # 核心：固定圆心角为整圆
                radius=radius,  # 圆形半径
                arc_center=arc_center,  # 圆形圆心
                **kwargs  # 样式配置（如颜色、填充）
            )
            # 补充：若需确保圆形闭合（避免因精度问题导致的缺口），可添加此句（Arc 生成整圆时已默认闭合，此处为冗余保障）
            # self.make_closed()

    # 首先补充 Circle 类的三个核心方法（surround、point_at_angle、get_radius），再解析后续子类
    class Circle(Arc):
        # （承接上文 __init__ 方法，此处补充三个关键实例方法）
        def surround(
                self,
                mobject: Mobject,
                dim_to_match: int = 0,
                stretch: bool = False,
                buff: float = MED_SMALL_BUFF
        ) -> Self:
            """
            调整圆形大小和位置，使其“包围”指定的Mobject（如矩形、文本），并保留指定缓冲距离。
            常用于给对象添加圆形边框、高亮标记等场景。

            参数说明：
                mobject: 待包围的Mobject对象（如Rectangle、Text、VGroup）。
                dim_to_match: 优先匹配的维度（0=宽度，1=高度），默认0（先匹配宽度）。
                stretch: 布尔值，是否允许圆形拉伸为椭圆以完全匹配对象尺寸（False则保持圆形，取宽高中较大值）。
                buff: 圆形与对象之间的缓冲距离（空白间隙），默认MED_SMALL_BUFF（Manim预定义常量，约0.2）。

            返回值：调整后的当前Circle对象（支持链式调用，如circle.surround(text).set_color(RED)）。
            """
            # 1. 先将圆形替换为与目标对象尺寸匹配的形状（基础尺寸适配）
            self.replace(mobject, dim_to_match, stretch)
            # 2. 在宽度方向添加缓冲：将圆形宽度扩展为“原宽度 + 2*buff”（左右各留buff）
            self.stretch((self.get_width() + 2 * buff) / self.get_width(), 0)
            # 3. 在高度方向添加缓冲：将圆形高度扩展为“原高度 + 2*buff”（上下各留buff）
            self.stretch((self.get_height() + 2 * buff) / self.get_height(), 1)
            return self

        def point_at_angle(self, angle: float) -> Vect3:
            """
            获取圆形上“与x轴正方向成指定角度”的点的坐标（极坐标转直角坐标）。
            常用于获取圆形上特定方向的点（如时钟12点位置、3点位置）。

            参数：angle - 目标角度（单位：弧度），逆时针为正。
            返回值：三维向量（Vect3），圆形上对应角度的点坐标。
            """
            # 1. 获取圆形的起始角度（圆弧的起始方向，Circle类默认从0弧度开始，即x轴正方向）
            start_angle = self.get_start_angle()
            # 2. 计算目标角度相对于起始角度的比例（归一化到0~1，对应圆弧的0~100%长度）
            # 取模TAU确保角度在0~2π范围内，避免负角度或超范围角度导致错误
            proportion = ((angle - start_angle) % TAU) / TAU
            # 3. 根据比例获取圆形上的点（point_from_proportion是VMobject方法，按路径长度比例取点）
            return self.point_from_proportion(proportion)

        def get_radius(self) -> float:
            """
            计算并返回圆形的半径（通过“起始点到圆心的距离”推导，确保精度）。

            返回值：浮点数，圆形的半径（Manim坐标系单位）。
            """
            # 1. 获取圆形的起始点（Circle类默认起始点在(半径, 0, 0)，即x轴正方向）
            # 2. 获取圆形的圆心（arc_center，初始化时指定）
            # 3. 计算两点间距离（向量模长），即为半径
            return get_norm(self.get_start() - self.get_center())

    class Dot(Circle):
        '''
        生成“点”图形（继承自Circle），本质是“填充不描边”的小型圆形，
        常用于标记位置（如坐标点、顶点、交互点）、指示焦点等场景。

        参数说明：
            point : 三维向量（Vect3），点的中心坐标，默认ORIGIN（原点(0,0,0)）。
            radius : 浮点数，点的半径，默认DEFAULT_DOT_RADIUS（预定义常量，约0.08）。
            stroke_color : 描边颜色，默认BLACK（黑色），但stroke_width=0时不可见。
            stroke_width : 描边线宽，默认0.0（无描边，仅显示填充区域）。
            fill_opacity : 填充透明度，默认1.0（完全不透明，实心点）。
            fill_color : 填充颜色，默认DEFAULT_MOBJECT_COLOR（Manim默认色，通常为白色）。
            **kwargs : 关键字参数，传递Circle/VMobject的额外配置（如z_index=10，调整渲染层级）。

        示例：
            # 生成中心在(1,2,0)的默认大小点
            dot = Dot(point=(1, 2, 0))
            # 生成中心在LEFT+UP（(-1,1,0)）、红色、大半径的点
            dot = Dot(point=LEFT+UP, radius=0.2, fill_color=RED)

        返回值：
            out : Dot object，满足指定参数的“点”图形。
        '''

        def __init__(
                self,
                point: Vect3 = ORIGIN,
                radius: float = DEFAULT_DOT_RADIUS,
                stroke_color: ManimColor = BLACK,
                stroke_width: float = 0.0,
                fill_opacity: float = 1.0,
                fill_color: ManimColor = DEFAULT_MOBJECT_COLOR,
                **kwargs
        ):
            # 调用父类Circle的初始化方法，将圆形配置为“点”的样式
            super().__init__(
                arc_center=point,  # 点的中心 = 圆形的圆心
                radius=radius,  # 点的半径 = 圆形的半径
                stroke_color=stroke_color,  # 描边颜色（默认黑色，但线宽为0）
                stroke_width=stroke_width,  # 描边线宽（默认0，无描边）
                fill_opacity=fill_opacity,  # 填充透明度（默认1，实心）
                fill_color=fill_color,  # 填充颜色（默认白色）
                **kwargs  # 额外配置（如渲染层级）
            )

    class SmallDot(Dot):
        '''
        生成“小点”图形（继承自Dot），本质是半径更小的Dot，
        常用于标记次要位置（如辅助点、网格点、临时标记），避免遮挡主要对象。

        参数说明：
            point : 三维向量（Vect3），小点的中心坐标，默认ORIGIN（原点(0,0,0)）。
            radius : 浮点数，小点的半径，默认DEFAULT_SMALL_DOT_RADIUS（预定义常量，约0.04，是Dot的一半）。
            **kwargs : 关键字参数，传递Dot/Circle的额外配置（如fill_color=BLUE、fill_opacity=0.8）。

        示例：
            # 生成中心在(1,2,0)的默认大小小点
            small_dot = SmallDot(point=(1, 2, 0))
            # 生成中心在(3,0,0)、蓝色半透明的小点
            small_dot = SmallDot(point=(3,0,0), fill_color=BLUE, fill_opacity=0.5)

        返回值：
            out : SmallDot object，满足指定参数的“小点”图形。
        '''

        def __init__(
                self,
                point: Vect3 = ORIGIN,
                radius: float = DEFAULT_SMALL_DOT_RADIUS,
                **kwargs
        ):
            # 调用父类Dot的初始化方法，仅覆盖半径参数（使用更小的默认值）
            super().__init__(
                point=point,  # 小点的中心坐标
                radius=radius,  # 小点的半径（默认更小）
                **kwargs  # 额外配置（如颜色、透明度）
            )

    class Ellipse(Circle):
        '''
        生成“椭圆”图形（继承自Circle），本质是“拉伸后的圆形”，
        常用于需要非正圆形状的场景（如椭圆轨道、扁平图标、数据可视化中的椭圆标记）。

        参数说明：
            width : 浮点数，椭圆的宽度（水平方向直径），默认2.0（与Circle默认半径1.0的直径一致）。
            height : 浮点数，椭圆的高度（垂直方向直径），默认1.0（是宽度的一半，形成水平拉伸的椭圆）。
            **kwargs : 关键字参数，传递Circle/VMobject的额外配置（如arc_center=(2,3,0)、fill_color=YELLOW）。

        示例：
            # 生成宽度4、高度1、中心在(3,3,0)的椭圆
            ellipse = Ellipse(width=4, height=1, arc_center=(3, 3, 0))
            # 生成宽度2、高度5、中心在原点、蓝色的椭圆
            ellipse = Ellipse(width=2, height=5, arc_center=ORIGIN, color=BLUE)

        返回值：
            out : Ellipse object，满足指定参数的“椭圆”图形。
        '''

        def __init__(
                self,
                width: float = 2.0,
                height: float = 1.0,
                **kwargs
        ):
            # 1. 调用父类Circle的初始化方法，生成基础圆形（默认半径1.0，中心原点）
            super().__init__(**kwargs)
            # 2. 调整宽度：将圆形水平拉伸/压缩到指定width（stretch=True允许非均匀缩放）
            self.set_width(width, stretch=True)
            # 3. 调整高度：将圆形垂直拉伸/压缩到指定height
            self.set_height(height, stretch=True)
            # 注：set_width/set_height是VMobject方法，会自动计算缩放比例，确保最终尺寸匹配参数


class AnnularSector(VMobject):
    '''
    生成**环形扇区**（又称“圆环扇区”，即带内孔的扇形），形状类似“披萨的一块但中间挖空”，
    适用于表示角度范围、环形图表切片（如甜甜圈图的一部分）、环形标注等场景。

    参数说明：
        inner_radius : 浮点数，环形扇区的内半径（中间空心部分的半径），默认1.0。
        outer_radius : 浮点数，环形扇区的外半径（整体的最大半径），默认2.0。
        start_angle : 浮点数，环形扇区的起始角度（单位：弧度），默认0.0（x轴正方向），角度按**逆时针**计算。
        angle : 浮点数，环形扇区在圆心处的圆心角（单位：弧度），默认TAU/4（45°），角度按**逆时针**计算。
        arc_center : 三维向量（Vect3），环形扇区的圆心坐标，默认ORIGIN（原点(0,0,0)）。
        fill_color : 填充颜色，默认DEFAULT_LIGHT_COLOR（Manim预定义浅色，通常为浅灰）。
        fill_opacity : 填充透明度，默认1.0（完全不透明，实心填充）。
        stroke_width : 描边线宽，默认0.0（无描边，仅显示填充区域）。
        **kwargs : 关键字参数，传递VMobject的额外配置（如z_index=5、stroke_color=BLACK）。

    示例：
        # 生成内半径1、外半径2、圆心角90°（TAU/2）、起始角270°（TAU*3/4）、圆心在(1,-2,0)的环形扇区
        annular_sector = AnnularSector(inner_radius=1, outer_radius=2, angle=TAU/2, start_angle=TAU*3/4, arc_center=(1,-2,0))

    返回值：
        out : AnnularSector object，满足指定参数的环形扇区图形。
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
            stroke_width: float = 0.0,
            **kwargs,
    ):
        # 1. 调用父类VMobject的初始化，传递填充、描边等样式配置
        super().__init__(
            fill_color=fill_color,
            fill_opacity=fill_opacity,
            stroke_width=stroke_width,
            **kwargs,
        )

        # 2. 创建环形扇区的“内圆弧”和“外圆弧”（复用Arc类生成两段圆弧）
        inner_arc, outer_arc = [
            Arc(
                start_angle=start_angle,  # 内、外圆弧共享起始角度
                angle=angle,  # 内、外圆弧共享圆心角
                radius=radius,  # 分别使用内半径、外半径
                arc_center=arc_center,  # 内、外圆弧共享圆心
            )
            for radius in (inner_radius, outer_radius)  # 循环生成内、外圆弧
        ]

        # 3. 拼接环形扇区的完整路径（确保图形闭合，支持填充）
        # - 第一步：添加“反向的内圆弧”（从内圆弧终点到起点，形成内侧边界）
        self.set_points(inner_arc.get_points()[::-1])  # [::-1] 反转路径点顺序
        # - 第二步：添加“从内圆弧起点到外圆弧起点的线段”（连接内、外圆弧的起始端）
        self.add_line_to(outer_arc.get_points()[0])
        # - 第三步：添加“外圆弧”（从外圆弧起点到终点，形成外侧边界）
        self.add_subpath(outer_arc.get_points())  # add_subpath 追加子路径（保持连续性）
        # - 第四步：添加“从外圆弧终点到内圆弧终点的线段”（连接内、外圆弧的终止端，闭合图形）
        self.add_line_to(inner_arc.get_points()[-1])


class Sector(AnnularSector):
    '''
    生成**扇形**（即“实心扇形”，无内孔的环形扇区），形状类似“披萨的一块”，
    适用于表示角度、饼图切片、扇形标注（如三角函数中的单位圆扇形）等场景。

    参数说明：
        outer_radius : 浮点数，扇形的半径（整体最大半径），默认1.0（此处参数名用radius也可，通过**kwargs传递）。
        start_angle : 浮点数，扇形的起始角度（单位：弧度），默认0.0（x轴正方向），角度按**逆时针**计算。
        angle : 浮点数，扇形在圆心处的圆心角（单位：弧度），默认TAU/4（45°），角度按**逆时针**计算。
        arc_center : 三维向量（Vect3），扇形的圆心坐标，默认ORIGIN（原点(0,0,0)）。
        **kwargs : 关键字参数，传递AnnularSector/VMobject的配置（如fill_color=PINK、stroke_width=2）。

    示例：
        # 生成半径1、起始角120°（TAU/3）、圆心角90°（TAU/2）、圆心在(0,3,0)的扇形
        sector = Sector(outer_radius=1, start_angle=TAU/3, angle=TAU/2, arc_center=[0,3,0])
        # 生成半径3、起始角45°（TAU/4）、圆心角45°（TAU/4）、圆心在原点、粉色的扇形
        sector = Sector(outer_radius=3, start_angle=TAU/4, angle=TAU/4, arc_center=ORIGIN, color=PINK)

    返回值：
        out : Sector object，满足指定参数的扇形图形。
    '''

    def __init__(
            self,
            angle: float = TAU / 4,
            radius: float = 1.0,
            **kwargs
    ):
        # 核心逻辑：扇形 = 内半径为0的环形扇区（内孔消失，形成实心扇形）
        super().__init__(
            angle=angle,  # 继承环形扇区的圆心角
            inner_radius=0,  # 内半径设为0，消除内孔
            outer_radius=radius,  # 外半径设为扇形的半径
            **kwargs  # 传递其他配置（如起始角度、圆心、颜色）
        )


class Annulus(VMobject):
    '''
    生成**圆环**（即“完整的环形”，无角度缺口的环形扇区），形状类似“甜甜圈的外圈”，
    适用于表示环形边框、进度条背景、环形装饰元素等场景。

    参数说明：
        inner_radius : 浮点数，圆环的内半径（中间空心部分的半径），默认1.0。
        outer_radius : 浮点数，圆环的外半径（整体的最大半径），默认2.0。
        arc_center : 三维向量（Vect3），圆环的圆心坐标，默认ORIGIN（原点(0,0,0)）（参数名用center也可，初始化时统一为center）。
        fill_color : 填充颜色，默认DEFAULT_LIGHT_COLOR（Manim预定义浅色，通常为浅灰）。
        fill_opacity : 填充透明度，默认1.0（完全不透明，实心填充）。
        stroke_width : 描边线宽，默认0.0（无描边，仅显示填充区域）。
        **kwargs : 关键字参数，传递VMobject的额外配置（如stroke_color=RED、z_index=3）。

    示例：
        # 生成内半径2、外半径3、圆心在(1,-1,0)的圆环
        annulus = Annulus(inner_radius=2, outer_radius=3, arc_center=(1, -1, 0))
        # 生成内半径2、外半径3、描边线宽20、描边红色、填充蓝色、圆心在原点的圆环
        annulus = Annulus(inner_radius=2, outer_radius=3, stroke_width=20, stroke_color=RED, fill_color=BLUE, arc_center=ORIGIN)

    返回值：
        out : Annulus object，满足指定参数的圆环图形。
    '''

    def __init__(
            self,
            inner_radius: float = 1.0,
            outer_radius: float = 2.0,
            fill_opacity: float = 1.0,
            stroke_width: float = 0.0,
            fill_color: ManimColor = DEFAULT_LIGHT_COLOR,
            center: Vect3 = ORIGIN,
            **kwargs,
    ):
        # 1. 调用父类VMobject的初始化，传递填充、描边等样式配置
        super().__init__(
            fill_color=fill_color,
            fill_opacity=fill_opacity,
            stroke_width=stroke_width,
            **kwargs,
        )

        # 2. 记录外半径（供后续可能的方法调用，如get_radius）
        self.radius = outer_radius

        # 3. 生成圆环的“外圆弧路径”（整圆，顺时针方向，TAU=2π）
        # quadratic_bezier_points_for_arc(TAU) 生成整圆的贝塞尔路径点，乘以outer_radius缩放
        outer_path = outer_radius * quadratic_bezier_points_for_arc(TAU)
        # 4. 生成圆环的“内圆弧路径”（整圆，逆时针方向，-TAU确保与外圆弧方向相反，形成闭合区域）
        inner_path = inner_radius * quadratic_bezier_points_for_arc(-TAU)

        # 5. 拼接圆环的完整路径（外圆弧+内圆弧，确保填充区域为环形）
        self.add_subpath(outer_path)  # 先添加外圆弧路径
        self.add_subpath(inner_path)  # 再添加内圆弧路径

        # 6. 将圆环平移到指定的圆心位置（从原点移至center）
        self.shift(center)


class Line(TipableVMobject):
    '''
    Creates a line joining the points "start" and "end".
    Parameters
    -----
    start : array_like
        Starting point of the line
    end : array_like
        Ending point of the line
    Examples :
            line = Line((0, 0, 0), (3, 0, 0))
            line = Line((1, 2, 0), (-2, -3, 0), color=BLUE)
    Returns
    -----
    out : Line object
        A Line object satisfying the specified parameters
    '''

    # 首先补充 Line 类的完整实现（基于代码片段推导，继承自 TipableVMobject 或 VMobject，核心是“可带圆弧路径的线段”）
    class Line(TipableVMobject):  # 结合上下文，Line 继承自可添加尖端的 TipableVMobject，支持箭头
        """
        生成线段（支持直线路径或带圆弧的弯曲路径），是 Manim 中最基础的线性图形之一，
        可通过 add_tip() 添加箭头，适用于连接对象、指示方向、绘制坐标轴等场景。
        """

        def __init__(
                self,
                start: Vect3 | Mobject = LEFT,  # 起点：可传三维向量（如LEFT=(-1,0,0)）或Mobject（取边界点/中心点）
                end: Vect3 | Mobject = RIGHT,  # 终点：同起点，默认RIGHT=(1,0,0)
                buff: float = 0.0,  # 端点缓冲距离（线段与起点/终点对象的间隙，避免重叠）
                path_arc: float = 0.0,  # 路径圆弧角（单位：弧度），0为直线，正值逆时针弯曲，负值顺时针弯曲
                **kwargs  # 通用配置（如color、stroke_width、tip_style等）
        ):
            super().__init__(**kwargs)  # 调用父类（TipableVMobject）初始化，传递样式配置
            self.path_arc = path_arc  # 记录圆弧角，供后续修改路径使用
            self.buff = buff  # 记录缓冲距离
            self.set_start_and_end_attrs(start, end)  # 解析起点/终点（处理Mobject类型，确定最终坐标）
            # 根据解析后的起点、终点、缓冲距离、圆弧角，生成线段路径
            self.set_points_by_ends(self.start, self.end, buff, path_arc)

        def set_points_by_ends(
                self,
                start: Vect3,  # 解析后的起点坐标（三维向量）
                end: Vect3,  # 解析后的终点坐标（三维向量）
                buff: float = 0,  # 缓冲距离
                path_arc: float = 0  # 路径圆弧角
        ) -> Self:
            """根据起点、终点、缓冲和圆弧角，重新生成线段的路径点"""
            self.clear_points()  # 清除现有路径点，避免叠加
            self.start_new_path(start)  # 启动新路径，设置起点
            # 添加路径段：直线或圆弧（path_arc=0为直线，非0为圆弧）
            self.add_arc_to(end, path_arc)

            # 应用缓冲距离：若buff>0，从线段两端各裁剪buff长度（避免与其他对象重叠）
            if buff > 0:
                total_length = self.get_arc_length()  # 计算当前线段总长度
                # 裁剪比例：buff/总长度，最大0.5（避免裁剪后线段消失）
                alpha = min(buff / total_length, 0.5)
                # 保留线段的 [alpha, 1-alpha] 部分（即两端各裁去alpha比例）
                self.pointwise_become_partial(self, alpha, 1 - alpha)
            return self

        def set_path_arc(self, new_value: float) -> Self:
            """修改线段的路径圆弧角，并重绘路径"""
            self.path_arc = new_value  # 更新圆弧角
            self.init_points()  # 重新初始化路径（内部会调用set_points_by_ends）
            return self

        def set_start_and_end_attrs(self, start: Vect3 | Mobject, end: Vect3 | Mobject):
            """解析起点和终点：若为Mobject，计算其边界点；若为向量，直接使用"""
            # 1. 先获取起点和终点的“粗略坐标”（Mobject取中心，向量直接用）
            rough_start = self.pointify(start)
            rough_end = self.pointify(end)
            # 2. 计算起点到终点的单位向量（用于确定Mobject的“朝向边界点”）
            direction = normalize(rough_end - rough_start)  # 归一化，确保方向正确且模长为1
            # 3. 重新解析起点和终点：Mobject取“朝向终点/起点的边界点”，避免线段穿入对象内部
            self.start = self.pointify(start, direction)  # 起点：Mobject朝向终点的边界点
            self.end = self.pointify(end, -direction)  # 终点：Mobject朝向起点的边界点（-direction反向）

        def pointify(
                self,
                mob_or_point: Mobject | Vect3,  # 输入：Mobject或三维向量
                direction: Vect3 | None = None  # 方向向量（仅Mobject时使用，用于找边界点）
        ) -> Vect3:
            """
            将Line的输入（Mobject或向量）转换为三维坐标点，是Line类处理“灵活输入”的核心方法。
            """
            if isinstance(mob_or_point, Mobject):
                # 若输入是Mobject：
                mob = mob_or_point
                if direction is None:
                    # 无方向时，返回Mobject的中心点（默认 fallback）
                    return mob.get_center()
                else:
                    # 有方向时，返回Mobject“沿direction方向的最远边界点”（避免线段穿入对象）
                    return mob.get_continuous_bounding_box_point(direction)
            else:
                # 若输入是向量（如(1,2,0)、LEFT）：
                point = mob_or_point
                # 生成与Line维度匹配的三维向量（避免二维向量缺z轴的问题）
                result = np.zeros(self.dim)  # self.dim通常为3（Manim默认3D坐标系）
                result[:len(point)] = point  # 将输入向量的值赋给前n维（如二维向量补z=0）
                return result

        def put_start_and_end_on(self, start: Vect3, end: Vect3) -> Self:
            """强制将线段的起点和终点设置为指定坐标（重写父类方法，处理“零长度线段”特殊情况）"""
            curr_start, curr_end = self.get_start_and_end()  # 获取当前起点和终点
            # 特殊情况：当前线段长度为0（起点终点重合），直接调用set_points_by_ends重绘
            if np.isclose(curr_start, curr_end).all():
                self.set_points_by_ends(start, end, buff=0, path_arc=self.path_arc)
                return self
            # 普通情况：调用父类方法（VMobject的put_start_and_end_on）调整路径
            return super().put_start_and_end_on(start, end)

        def get_vector(self) -> Vect3:
            """获取线段的“方向向量”（终点 - 起点）"""
            return self.get_end() - self.get_start()

        def get_unit_vector(self) -> Vect3:
            """获取线段的“单位方向向量”（方向与get_vector一致，模长为1）"""
            return normalize(self.get_vector())

        def get_angle(self) -> float:
            """获取线段与x轴正方向的夹角（单位：弧度），逆时针为正"""
            return angle_of_vector(self.get_vector())

        def get_projection(self, point: Vect3) -> Vect3:
            """计算指定点在当前线段上的“正交投影点”（即点到线段的垂足）"""
            unit_vect = self.get_unit_vector()  # 线段的单位方向向量
            start = self.get_start()  # 线段起点
            # 投影公式：start + (point - start)在unit_vect上的投影长度 * unit_vect
            projection_length = np.dot(point - start, unit_vect)
            return start + projection_length * unit_vect

        def get_slope(self) -> float:
            """获取线段的斜率（二维平面内，y轴变化量/x轴变化量，即tan(角度)）"""
            return np.tan(self.get_angle())

        def set_angle(self, angle: float, about_point: Optional[Vect3] = None) -> Self:
            """将线段旋转到指定角度（绕指定点旋转，默认绕起点旋转）"""
            if about_point is None:
                about_point = self.get_start()  # 默认绕起点旋转
            # 旋转角度 = 目标角度 - 当前角度（仅旋转差值，避免过度旋转）
            rotate_angle = angle - self.get_angle()
            self.rotate(rotate_angle, about_point=about_point)  # 调用父类旋转方法
            return self

        def set_length(self, length: float, **kwargs) -> Self:
            """将线段缩放至指定长度（保持方向不变，默认绕起点缩放）"""
            current_length = self.get_length()  # 当前线段长度
            # 缩放比例 = 目标长度 / 当前长度（current_length=0时不缩放，避免除以零）
            scale_ratio = length / current_length if current_length != 0 else 1.0
            self.scale(scale_ratio, **kwargs)  # 调用父类缩放方法（**kwargs可传about_point等）
            return self

        def get_arc_length(self) -> float:
            """计算线段的实际路径长度（直线为两点距离，圆弧为圆弧长度）"""
            # 若为直线（path_arc=0），直接返回两点距离；若为圆弧，计算圆弧长度
            if self.path_arc == 0:
                return get_norm(self.get_end() - self.get_start())
            else:
                # 圆弧长度公式：半径 * 圆心角（需先计算圆弧半径）
                start, end = self.get_start(), self.get_end()
                chord_length = get_norm(end - start)  # 弦长（起点到终点的直线距离）
                radius = chord_length / (2 * np.sin(np.abs(self.path_arc) / 2))  # 圆弧半径
                return radius * np.abs(self.path_arc)  # 圆弧长度 = 半径 * 圆心角（绝对值确保为正）

    class DashedLine(Line):
        '''
        生成虚线线段（继承自Line类），本质是将Line分割为多个短线段，适用于表示辅助线、隐藏线、可选路径等场景。

        参数说明：
            start : 三维向量或Mobject，虚线的起点，默认LEFT=(-1,0,0)。
            end : 三维向量或Mobject，虚线的终点，默认RIGHT=(1,0,0)。
            dash_length : 浮点数，每个短线段（“ dash ”）的长度，默认DEFAULT_DASH_LENGTH=0.05。
            positive_space_ratio : 浮点数，“实线段长度占总周期的比例”（周期=实线长+空白长），默认0.5（实线与空白等长）。
            **kwargs : 关键字参数，传递Line的通用配置（如color、stroke_width、path_arc等）。

        示例：
            # 生成从(0,0,0)到(3,0,0)的默认虚线
            line = DashedLine((0, 0, 0), (3, 0, 0))
            # 生成从(1,2,3)到(4,5,6)、短线段长度0.01的虚线
            line = DashedLine((1, 2, 3), (4, 5, 6), dash_length=0.01)

        返回值：
            out : DashedLine object，满足指定参数的虚线线段。
        '''

        def __init__(
                self,
                start: Vect3 = LEFT,
                end: Vect3 = RIGHT,
                dash_length: float = DEFAULT_DASH_LENGTH,
                positive_space_ratio: float = 0.5,
                **kwargs
        ):
            # 1. 调用父类Line的初始化，生成完整的线段（后续会分割为虚线）
            super().__init__(start, end, **kwargs)

            # 2. 计算需要的短线段数量（根据总长度、单段长度、比例）
            num_dashes = self.calculate_num_dashes(dash_length, positive_space_ratio)
            # 3. 使用DashedVMobject将完整线段分割为虚线（核心工具类，处理虚线分割逻辑）
            dashes = DashedVMobject(
                self,  # 待分割的完整线段
                num_dashes=num_dashes,  # 短线段数量
                positive_space_ratio=positive_space_ratio  # 实线占比
            )

            # 4. 替换当前对象的路径：清除原线段，添加分割后的虚线
            self.clear_points()  # 清除完整线段的路径点
            self.add(*dashes)  # 添加所有短线段（*dashes解包列表）

        def calculate_num_dashes(self, dash_length: float, positive_space_ratio: float) -> int:
            """计算虚线的短线段数量（确保每个短线段长度接近dash_length）"""
            total_length = self.get_arc_length()  # 完整线段的总长度
            if total_length == 0:
                return 0  # 零长度线段无需分割
            # 虚线周期长度 = 短线段长度 / 实线占比（周期=实线长+空白长）
            cycle_length = dash_length / positive_space_ratio
            # 总周期数 = 总长度 / 周期长度，向上取整确保覆盖完整线段
            num_cycles = np.ceil(total_length / cycle_length)
            # 短线段数量 = 周期数（每个周期1段实线）
            return int(num_cycles)

    # 首先补充上文未完整解析的“虚线相关方法”（推测属于DashedVMobject或其衍生类），再解析后续类
    # 注：这些方法用于计算虚线分段、获取端点和控制点，核心服务于“虚线图形”的渲染逻辑
    def calculate_num_dashes(self, dash_length: float, positive_space_ratio: float) -> int:
        """
        计算虚线所需的“总段数”（含实线段和空白段），确保虚线均匀覆盖整个图形长度。

        参数说明：
            dash_length : 浮点数，单段实线的长度（如DEFAULT_DASH_LENGTH=0.05）。
            positive_space_ratio : 浮点数，实线长度占“实线+空白”周期的比例（如0.5表示“实线0.05+空白0.05”）。
        返回值：整数，虚线的总段数（确保覆盖图形全长，向上取整）。
        异常处理：若positive_space_ratio为0（无实线），默认返回1段（避免除以零错误）。
        """
        try:
            # 计算“实线+空白”的完整周期长度（单段实线长度 / 占比）
            full_cycle_length = dash_length / positive_space_ratio
            # 总段数 = 图形总长度 / 周期长度，向上取整（确保最后一段即使不足周期也显示）
            return int(np.ceil(self.get_length() / full_cycle_length))
        except ZeroDivisionError:
            # 若比例为0（无实线），返回1段（避免崩溃，实际可能显示空白）
            return 1

    def get_start(self) -> Vect3:
        """
        获取虚线图形的“起始点”（优先从子对象获取，无则调用Line类的默认逻辑）。
        适用于由多段子线段组成的虚线（如虚线圆弧、虚线折线）。

        返回值：三维向量（Vect3），虚线的起始点坐标。
        """
        if len(self.submobjects) > 0:
            # 若有子对象（如多段小线段），取第一个子对象的起始点
            return self.submobjects[0].get_start()
        else:
            # 无子类时，按Line类逻辑获取起始点（默认从路径点列表取第一个点）
            return Line.get_start(self)

    def get_end(self) -> Vect3:
        """
        获取虚线图形的“终止点”（优先从子对象获取，无则调用Line类的默认逻辑）。

        返回值：三维向量（Vect3），虚线的终止点坐标。
        """
        if len(self.submobjects) > 0:
            # 若有子对象，取最后一个子对象的终止点
            return self.submobjects[-1].get_end()
        else:
            # 无子类时，按Line类逻辑获取终止点（默认从路径点列表取最后一个点）
            return Line.get_end(self)

    def get_start_and_end(self) -> Tuple[Vect3, Vect3]:
        """
        便捷方法：同时获取虚线的起始点和终止点。

        返回值：元组（Tuple），格式为 (起始点坐标, 终止点坐标)。
        """
        return self.get_start(), self.get_end()

    def get_first_handle(self) -> Vect3:
        """
        获取虚线图形的“第一个控制点”（用于尖端定位、切线计算），从第一个子对象的路径点中提取。

        返回值：三维向量（Vect3），第一个控制点的坐标（子对象路径点列表的第2个点，索引1）。
        """
        return self.submobjects[0].get_points()[1]

    def get_last_handle(self) -> Vect3:
        """
        获取虚线图形的“最后一个控制点”，从最后一个子对象的路径点中提取。

        返回值：三维向量（Vect3），最后一个控制点的坐标（子对象路径点列表的倒数第2个点，索引-2）。
        """
        return self.submobjects[-1].get_points()[-2]

    class TangentLine(Line):
        '''
        生成“与指定向量图形（VMobject）相切的直线”，适用于几何演示（如圆的切线、曲线的切线）、
        导数几何意义可视化等场景。

        参数说明：
            vmob : VMobject对象，待求切线的目标图形（如Circle、Ellipse、Bezier曲线）。
            alpha : 浮点数，目标图形上的“比例位置”（范围0~1），0对应图形起点，1对应终点，0.5对应中点。
            length : 浮点数，切线的总长度，默认2（Manim坐标系单位）。
            d_alpha : 浮点数，计算切线方向的“微小偏移量”（用于数值求导），默认1e-6（确保精度）。
            **kwargs : 关键字参数，传递Line类的样式配置（如color=BLUE、stroke_width=2）。

        示例：
            # 生成与圆心在原点、半径3的绿色圆相切于alpha=1/3位置、长度6的蓝色切线
            circle = Circle(arc_center=ORIGIN, radius=3, color=GREEN)
            tangent_line = TangentLine(vmob=circle, alpha=1/3, length=6, color=BLUE)

        返回值：
            out : TangentLine object，与目标图形在指定位置相切的直线。
        '''

        def __init__(
                self,
                vmob: VMobject,
                alpha: float,
                length: float = 2,
                d_alpha: float = 1e-6,
                **kwargs
        ):
            # 1. 计算“微小偏移的两个位置”（alpha±d_alpha），确保在0~1范围内（避免超出图形路径）
            # clip函数：将值限制在[0,1]，防止alpha-d_alpha<0或alpha+d_alpha>1
            alpha1 = clip(alpha - d_alpha, 0, 1)
            alpha2 = clip(alpha + d_alpha, 0, 1)

            # 2. 获取目标图形上两个微小偏移位置的点（pfp = point_from_proportion，按比例取点）
            point1 = vmob.pfp(alpha1)
            point2 = vmob.pfp(alpha2)

            # 3. 调用父类Line的初始化：以两个点为基础创建直线（此时直线方向即为切线方向）
            super().__init__(point1, point2, **kwargs)

            # 4. 缩放直线到指定长度（基础直线长度是两点间距，按目标长度比例缩放）
            self.scale(length / self.get_length())

    class Elbow(VMobject):
        '''
        生成“L形折线”（类似手肘形状），适用于需要直角转弯的连接场景（如流程图中的直角分支、
        机械结构中的直角连接、坐标标注中的直角引线）。

        参数说明：
            width : 浮点数，L形的“臂长”（水平和垂直段的长度），默认0.2。
            angle : 浮点数，L形整体的旋转角度（单位：弧度），默认0（水平向右+垂直向上的标准L形），
                    角度按**逆时针**计算（如TAU/16=22.5°，表示L形逆时针旋转22.5°）。
            **kwargs : 关键字参数，传递VMobject的样式配置（如color=RED、stroke_width=3）。

        示例：
            # 生成臂长2、旋转22.5°（TAU/16）的L形折线
            elbow = Elbow(width=2, angle=TAU/16)

        返回值：
            out : Elbow object，满足指定参数的L形折线图形。
        '''

        def __init__(
                self,
                width: float = 0.2,
                angle: float = 0,
                **kwargs
        ):
            # 1. 调用父类VMobject的初始化，传递样式配置
            super().__init__(**kwargs)

            # 2. 创建基础L形路径：从UP(0,1,0)到UR(1,1,0)（水平向右），再到RIGHT(1,0,0)（垂直向下）
            # set_points_as_corners：按直角折线方式设置路径点（无贝塞尔曲线，纯折线）
            self.set_points_as_corners([UP, UR, RIGHT])

            # 3. 缩放L形到指定臂长：以原点为中心，将水平/垂直段长度调整为width
            self.set_width(width, about_point=ORIGIN)

            # 4. 旋转L形到指定角度：以原点为中心，按angle旋转整体
            self.rotate(angle, about_point=ORIGIN)

    class StrokeArrow(Line):
        '''
        生成“带渐变宽度箭头”的线段（继承自Line），核心特点是箭头尖端宽度随长度自适应，
        适用于需要强调箭头视觉层次的场景（如手绘风格动画、重点流程指向）。

        参数说明：
            start : 三维向量（Vect3）或Mobject对象，箭头的起始位置（若为Mobject则取其中心）。
            end : 三维向量（Vect3）或Mobject对象，箭头的终止位置（若为Mobject则取其中心）。
            stroke_color : 描边颜色，默认DEFAULT_LIGHT_COLOR（浅灰色）。
            stroke_width : 基础描边线宽，默认5（箭头根部宽度）。
            buff : 起始/终止位置与目标对象的缓冲距离（若start/end为Mobject），默认0.25。
            tip_width_ratio : 尖端宽度与根部宽度的比例，默认5（尖端更粗）。
            tip_len_to_width : 尖端长度与根部宽度的比例系数，默认0.0075（用于计算尖端长度）。
            max_tip_length_to_length_ratio : 尖端长度与箭头总长度的最大比例，默认0.3（避免尖端过长）。
            max_width_to_length_ratio : 最大宽度与箭头总长度的比例，默认8.0（避免宽度过大）。
            **kwargs : 关键字参数，传递Line类的额外配置（如z_index=10）。
        '''

        def __init__(
                self,
                start: Vect3 | Mobject,
                end: Vect3 | Mobject,
                stroke_color: ManimColor = DEFAULT_LIGHT_COLOR,
                stroke_width: float = 5,
                buff: float = 0.25,
                tip_width_ratio: float = 5,
                tip_len_to_width: float = 0.0075,
                max_tip_length_to_length_ratio: float = 0.3,
                max_width_to_length_ratio: float = 8.0,
                **kwargs,
        ):
            # 1. 保存箭头尖端的配置参数（供后续生成尖端时使用）
            self.tip_width_ratio = tip_width_ratio  # 尖端宽度比例
            self.tip_len_to_width = tip_len_to_width  # 尖端长度与根部宽度的系数
            self.max_tip_length_to_length_ratio = max_tip_length_to_length_ratio  # 尖端长度上限比例
            self.max_width_to_length_ratio = max_width_to_length_ratio  # 宽度上限比例
            self.n_tip_points = 3  # 尖端的顶点数量（默认3个点组成三角形尖端）
            self.original_stroke_width = stroke_width  # 保存原始线宽（避免缩放后丢失）

            # 2. 调用父类Line的初始化：创建基础线段，传递位置、颜色、线宽、缓冲距离等参数
            super().__init__(
                start, end,
                stroke_color=stroke_color,
                stroke_width=stroke_width,
                buff=buff,
                **kwargs
            )

            # 注：该类的核心“渐变宽度+尖端”逻辑通常在后续方法中实现（如create_tip、update_tip），
            # 此处仅完成基础参数初始化和线段创建，确保箭头主体与Line类兼容。

    class Arrow(Line):
        '''
        生成“实心箭头”（继承自Line），核心是带填充的箭头主体+三角形尖端，
        适用于大多数需要明确指向的场景（如流程图、数学推导、向量标注）。

        参数说明：
            start : 三维向量或Mobject，箭头起点（Mobject取边界点），默认LEFT=(-1,0,0)。
            end : 三维向量或Mobject，箭头终点（Mobject取边界点），默认LEFT（需手动指定，避免零长度）。
            buff : 箭头与起点/终点的缓冲距离，默认MED_SMALL_BUFF（约0.2）。
            path_arc : 箭头路径的圆弧角（弧度），0为直线，正值逆时针弯曲，默认0。
            fill_color : 箭头填充颜色，默认DEFAULT_LIGHT_COLOR（浅灰）。
            fill_opacity : 填充透明度，默认1.0（实心）。
            stroke_width : 描边线宽，默认0.0（无描边）。
            thickness : 箭头主体（箭杆）的厚度，默认3.0。
            tip_width_ratio : 尖端宽度与箭杆宽度的比例，默认5（尖端是箭杆的5倍宽）。
            tip_angle : 尖端的顶角（弧度），默认PI/3（60°）。
            max_tip_length_to_length_ratio : 尖端长度与箭头总长度的最大比例，默认0.5（避免尖端过长）。
            max_width_to_length_ratio : 箭杆宽度与箭头总长度的最大比例，默认0.1（避免箭杆过粗）。
            **kwargs : 传递Line类的额外配置（如z_index=5）。

        示例：
            >>> # 生成从(0,0,0)到(3,0,0)的默认箭头
            >>> arrow = Arrow((0, 0, 0), (3, 0, 0))
            >>> # 生成从LEFT到RIGHT、逆时针弯曲45°的箭头
            >>> curved_arrow = Arrow(LEFT, RIGHT, path_arc=PI/4)
            >>> # 生成从UP到DOWN、粗箭杆（thickness=5）、窄尖端（ratio=3）的箭头
            >>> thick_arrow = Arrow(UP, DOWN, thickness=5.0, tip_width_ratio=3)

        返回值：
            Arrow : 满足参数的实心箭头对象。
        '''
        # 厚度系数：将thickness参数转换为实际宽度（内部缩放用）
        tickness_multiplier = 0.015

        def __init__(
                self,
                start: Vect3 | Mobject = LEFT,
                end: Vect3 | Mobject = LEFT,
                buff: float = MED_SMALL_BUFF,
                path_arc: float = 0,
                fill_color: ManimColor = DEFAULT_LIGHT_COLOR,
                fill_opacity: float = 1.0,
                stroke_width: float = 0.0,
                thickness: float = 3.0,
                tip_width_ratio: float = 5,
                tip_angle: float = PI / 3,
                max_tip_length_to_length_ratio: float = 0.5,
                max_width_to_length_ratio: float = 0.1,
                **kwargs,
        ):
            # 保存箭头关键参数（用于后续计算尺寸）
            self.thickness = thickness
            self.tip_width_ratio = tip_width_ratio
            self.tip_angle = tip_angle
            self.max_tip_length_to_length_ratio = max_tip_length_to_length_ratio
            self.max_width_to_length_ratio = max_width_to_length_ratio
            # 调用父类Line初始化：传递填充、描边、位置等基础参数
            super().__init__(
                start, end,
                fill_color=fill_color,
                fill_opacity=fill_opacity,
                stroke_width=stroke_width,
                buff=buff,
                path_arc=path_arc, **kwargs
            )

        def get_key_dimensions(self, length):
            """计算箭头的关键尺寸（箭杆宽度、尖端宽度、尖端长度），含比例限制逻辑"""
            # 1. 计算基础箭杆宽度：厚度参数 × 厚度系数（转换为实际坐标单位）
            width = self.thickness * self.tickness_multiplier
            # 2. 限制箭杆宽度：不超过箭头长度 × 最大宽长比（避免过粗）
            w_ratio = fdiv(self.max_width_to_length_ratio, fdiv(width, length))  # 安全除法（防零）
            if w_ratio < 1:
                width *= w_ratio  # 按比例缩小宽度以满足限制

            # 3. 计算尖端宽度：箭杆宽度 × 尖端宽度比例
            tip_width = self.tip_width_ratio * width
            # 4. 计算理论尖端长度：由尖端宽度和顶角推导（三角形几何关系：tan(θ/2) = 半宽/长度）
            tip_length = tip_width / (2 * np.tan(self.tip_angle / 2))
            # 5. 限制尖端长度：不超过箭头长度 × 最大尖长比（避免尖端过长）
            t_ratio = fdiv(self.max_tip_length_to_length_ratio, fdiv(tip_length, length))
            if t_ratio < 1:
                tip_length *= t_ratio  # 按比例缩小尖端长度
                tip_width *= t_ratio  # 同步缩小尖端宽度（保持顶角不变）

            return width, tip_width, tip_length

        def set_points_by_ends(
                self,
                start: Vect3,
                end: Vect3,
                buff: float = 0,
                path_arc: float = 0
        ) -> Self:
            """
            核心方法：根据起点、终点生成箭头的完整路径（箭杆+尖端），支持直线和弯曲两种形态。
            """
            # 1. 计算起点到终点的向量及长度
            vect = end - start
            length = max(get_norm(vect), 1e-8)  # 确保长度不为0（避免除以零）
            unit_vect = normalize(vect)  # 单位方向向量（用于方向计算）

            # 2. 计算箭头关键尺寸（考虑缓冲后的有效长度）
            width, tip_width, tip_length = self.get_key_dimensions(length - buff)

            # 3. 调整起点和终点位置（应用缓冲距离，区分直线和弯曲路径）
            if path_arc == 0:
                # 直线箭头：起点向终点移动buff，终点向起点移动buff（预留尖端空间）
                start = start + buff * unit_vect
                end = end - buff * unit_vect
            else:
                # 弯曲箭头：基于圆弧半径计算缓冲后的起点和终点
                R = length / 2 / math.sin(path_arc / 2)  # 圆弧半径（由弦长公式推导）
                midpoint = 0.5 * (start + end)  # 起点终点的中点
                # 计算圆弧圆心（垂直于起点终点连线，向弯曲方向偏移）
                center = midpoint + rotate_vector(0.5 * vect, PI / 2) / math.tan(path_arc / 2)
                # 缓冲后的起点/终点：绕圆心旋转对应角度（弧长=Rθ → θ=buff/R）
                start = center + rotate_vector(start - center, buff / R)
                end = center + rotate_vector(end - center, -buff / R)
                # 调整圆弧角：减去缓冲和尖端占用的角度
                path_arc -= (2 * buff + tip_length) / R

            # 4. 重新计算调整后的向量和长度
            vect = end - start
            length = get_norm(vect)

            # 5. 生成箭杆路径（直线或弯曲）
            if path_arc == 0:
                # 直线箭杆：由上下两条平行线组成（形成矩形箭杆）
                # points1：上边缘路径（从起点到尖端根部，向上偏移width/2）
                points1 = (length - tip_length) * np.array([RIGHT, 0.5 * RIGHT, ORIGIN])
                points1 += width * UP / 2
                # points2：下边缘路径（与上边缘对称，向下偏移width/2）
                points2 = points1[::-1] + width * DOWN
            else:
                # 弯曲箭杆：由内外两条圆弧组成（形成环形箭杆）
                points1 = quadratic_bezier_points_for_arc(path_arc)  # 外圆弧路径点
                points2 = np.array(points1[::-1])  # 内圆弧路径点（反转方向确保闭合）
                # 缩放圆弧至实际半径（外圆R+width/2，内圆R-width/2）
                points1 *= (R + width / 2)
                points2 *= (R - width / 2)
                # 旋转圆弧以匹配箭头方向
                rot_T = rotation_matrix_transpose(PI / 2 - path_arc, OUT)  # 旋转矩阵（转置用于点旋转）
                for points in points1, points2:
                    points[:] = np.dot(points, rot_T)  # 应用旋转
                    points += R * DOWN  # 平移到正确位置

            # 6. 构建完整箭头路径（箭杆+尖端）
            self.set_points(points1)  # 先添加箭杆上边缘
            # 添加尖端：上顶点 → 尖端顶点 → 下顶点
            self.add_line_to(tip_width * UP / 2)  # 尖端上顶点
            self.add_line_to(tip_length * LEFT)  # 尖端顶点（箭头最前端）
            self.tip_index = len(self.get_points()) - 1  # 记录尖端顶点索引
            self.add_line_to(tip_width * DOWN / 2)  # 尖端下顶点
            self.add_line_to(points2[0])  # 连接到箭杆下边缘起点
            self.add_subpath(points2)  # 添加箭杆下边缘
            self.add_line_to(points1[0])  # 闭合路径（回到上边缘起点）

            # 7. 旋转和平移箭头以匹配目标方向和位置
            # 旋转箭头至与向量方向一致
            self.rotate(angle_of_vector(vect) - self.get_angle())
            # 微调旋转以确保在3D场景中垂直于相机视角
            self.rotate(
                PI / 2 - np.arccos(normalize(vect)[2]),
                axis=rotate_vector(self.get_unit_vector(), -PI / 2),
            )
            # 平移箭头使起点对齐目标起点
            self.shift(start - self.get_start())
            return self

        def reset_points_around_ends(self) -> Self:
            """重置箭头路径（基于当前起点和终点，重新计算尺寸和形状）"""
            self.set_points_by_ends(
                self.get_start().copy(),
                self.get_end().copy(),
                path_arc=self.path_arc
            )
            return self

        def get_start(self) -> Vect3:
            """获取箭头的实际起点（箭杆根部的中点，非几何路径的第一个点）"""
            points = self.get_points()
            return 0.5 * (points[0] + points[-3])  # 上边缘起点与下边缘终点的中点

        def get_end(self) -> Vect3:
            """获取箭头的实际终点（尖端顶点）"""
            return self.get_points()[self.tip_index]

        def get_start_and_end(self):
            """便捷方法：同时返回起点和终点"""
            return (self.get_start(), self.get_end())

        def put_start_and_end_on(self, start: Vect3, end: Vect3) -> Self:
            """强制将箭头的起点和终点设置为指定坐标（重写父类方法）"""
            self.set_points_by_ends(start, end, buff=0, path_arc=self.path_arc)
            return self

        def scale(self, *args, **kwargs) -> Self:
            """重写缩放方法：缩放后重置箭头路径（确保尖端比例正确）"""
            super().scale(*args, **kwargs)
            self.reset_points_around_ends()
            return self

        def set_thickness(self, thickness: float) -> Self:
            """设置箭杆厚度并重置路径（实时更新外观）"""
            self.thickness = thickness
            self.reset_points_around_ends()
            return self

        def set_path_arc(self, path_arc: float) -> Self:
            """设置箭头路径的圆弧角并重置路径（切换直线/弯曲形态）"""
            self.path_arc = path_arc
            self.reset_points_around_ends()
            return self

    class Arrow(Line):
        # （补充 Arrow 类的最后一个方法）
        def set_perpendicular_to_camera(self, camera_frame):
            """
            调整箭头方向，使其平面垂直于“箭头中心到相机”的连线（3D场景中优化视觉效果）。
            适用于确保箭头在3D视图中显示为正面，避免因视角问题导致的变形。

            参数：camera_frame - 相机帧对象（包含相机位置信息）
            返回：调整后的当前 Arrow 对象
            """
            # 1. 计算从箭头中心到相机位置的向量
            to_cam = camera_frame.get_implied_camera_location() - self.get_center()
            # 2. 获取箭头当前的单位法向量（垂直于箭头平面）
            normal = self.get_unit_normal()
            # 3. 获取箭头的方向向量（单位化）
            axis = normalize(self.get_vector())

            # 4. 计算目标法向量：将 to_cam 投影到垂直于箭头方向的平面（避免旋转影响箭头指向）
            trg_normal = to_cam - np.dot(to_cam, axis) * axis  # 移除沿箭头方向的分量
            # 5. 计算从当前法向量到目标法向量的旋转矩阵
            mat = rotation_between_vectors(normal, trg_normal)
            # 6. 应用旋转矩阵（绕箭头起点旋转，保持起点位置不变）
            self.apply_matrix(mat, about_point=self.get_start())
            return self

    class Vector(Arrow):
        '''
        生成“向量”（继承自Arrow），本质是起点固定在原点的箭头，
        适用于数学中的向量表示（如力、速度、位移向量）、坐标系轴标记等场景。

        参数说明：
            direction : 三维向量（或二维向量，自动补z=0），向量的方向和长度，默认RIGHT=(1,0,0)。
            buff : 向量与终点的缓冲距离，默认0.0（向量终点精确对齐direction坐标）。
            **kwargs : 传递Arrow类的配置（如color=RED、thickness=2）。

        示例：
            >>> # 生成方向向左（LEFT=(-1,0,0)）的向量
            >>> arrow = Vector(direction=LEFT)
            >>> # 生成方向为(3,4,0)、蓝色的向量（长度5，因3²+4²=5²）
            >>> vec = Vector(direction=(3,4,0), color=BLUE)

        返回值：
            out : Vector object，起点在原点、指向direction的向量箭头。
        '''

        def __init__(
                self,
                direction: Vect3 = RIGHT,
                buff: float = 0.0, **kwargs
        ):
            # 处理二维向量：自动补全z轴为0（确保在3D坐标系中正确显示）
            if len(direction) == 2:
                direction = np.hstack([direction, 0])  # 如(3,4) → (3,4,0)
            # 调用父类Arrow的初始化：起点固定为ORIGIN(0,0,0)，终点为direction
            super().__init__(ORIGIN, direction, buff=buff, **kwargs)

    class CubicBezier(VMobject):
        '''
        生成“三次贝塞尔曲线”（继承自VMobject），由两个锚点（起点和终点）和两个控制点定义，
        曲线形状受控制点影响，适用于绘制平滑曲线（如路径、波浪线、自定义形状边缘）。

        参数说明：
            a0 : 三维向量，第一个锚点（曲线起点）。
            h0 : 三维向量，第一个控制点（影响曲线从a0出发的方向和曲率）。
            h1 : 三维向量，第二个控制点（影响曲线向a1靠近的方向和曲率）。
            a1 : 三维向量，第二个锚点（曲线终点）。
            **kwargs : 传递VMobject的配置（如stroke_color=GREEN、stroke_width=2）。

        返回值：
            CubicBezier : 满足参数的三次贝塞尔曲线对象。
        '''

        def __init__(
                self,
                a0: Vect3,
                h0: Vect3,
                h1: Vect3,
                a1: Vect3, **kwargs
        ):
            super().__init__(**kwargs)
            # 添加三次贝塞尔曲线路径（VMobject的方法，自动计算曲线点）
            self.add_cubic_bezier_curve(a0, h0, h1, a1)

    class Polygon(VMobject):
        '''
        生成“多边形”（继承自VMobject），由指定顶点按顺序连接形成闭合图形，
        适用于绘制任意多边形（如三角形、五边形、不规则多边形）。

        参数说明：
            *vertices : 可变参数，多个三维向量，按顺序表示多边形的顶点（如(0,0,0), (1,0,0), (0,1,0)）。
            **kwargs : 传递VMobject的配置（如fill_color=YELLOW、stroke_width=1）。

        示例：
            >>> # 生成顶点为(-3,0,0)、(3,0,0)、(0,3,0)的三角形
            >>> triangle = Polygon((-3,0,0), (3,0,0), (0,3,0))

        返回值：
            out : Polygon object，满足顶点参数的闭合多边形。
        '''

        def __init__(
                self,
                *vertices: Vect3, **kwargs
        ):
            super().__init__(**kwargs)
            # 设置多边形路径：按顶点顺序连接，最后回到第一个顶点形成闭合
            self.set_points_as_corners([*vertices, vertices[0]])

        def get_vertices(self) -> Vect3Array:
            """获取多边形的所有顶点（不包含闭合的重复起点）"""
            return self.get_start_anchors()  # start_anchors 存储路径中的锚点（顶点）

        def round_corners(self, radius: Optional[float] = None) -> Self:
            """
            将多边形的直角顶点替换为圆弧（圆角），使边缘平滑。

            参数：radius - 圆角半径，默认自动计算（为最短边的1/4）。
            返回：圆角处理后的当前多边形对象。
            """
            if radius is None:
                # 自动计算半径：取最短边长度的1/4
                verts = self.get_vertices()
                min_edge_length = min(
                    get_norm(v1 - v2)
                    for v1, v2 in zip(verts, verts[1:])  # 遍历相邻顶点计算边长
                    if not np.isclose(v1, v2).all()  # 跳过重合顶点
                )
                radius = 0.25 * min_edge_length

            vertices = self.get_vertices()
            arcs = []  # 存储每个顶点替换后的圆弧
            # 遍历每个顶点及其前后顶点（v1→v2→v3，处理v2）
            for v1, v2, v3 in adjacent_n_tuples(vertices, 3):
                # 计算顶点两侧的单位向量（v1到v2，v3到v2）
                vect1 = normalize(v2 - v1)
                vect2 = normalize(v3 - v2)
                # 计算两向量的夹角（顶点的内角）
                angle = angle_between_vectors(vect1, vect2)
                # 计算从顶点到圆弧起点/终点的距离（确保圆弧与边相切）
                cut_off_length = radius * np.tan(angle / 2)
                # 确定圆弧方向（凸角/凹角）：由半径符号和向量叉积决定
                sign = float(np.sign(radius * cross2d(vect1, vect2)))
                # 创建顶点替换的圆弧（从v2-vect1*cut_off_length到v2+vect2*cut_off_length）
                arc = ArcBetweenPoints(
                    v2 - vect1 * cut_off_length,
                    v2 + vect2 * cut_off_length,
                    angle=sign * angle,
                    n_components=2,  # 圆弧分段数（2段足够平滑）
                )
                arcs.append(arc)

            # 重新构建多边形路径（用圆弧替换顶点）
            self.clear_points()
            # 调整圆弧顺序：确保从最后一个圆弧开始，形成闭合
            arcs = [arcs[-1], *arcs[:-1]]
            # 拼接所有圆弧路径
            for arc1, arc2 in adjacent_pairs(arcs):
                self.add_subpath(arc1.get_points())  # 添加当前圆弧
                self.add_line_to(arc2.get_start())  # 连接到下一个圆弧的起点
            return self

    class Polyline(VMobject):
        """
        生成“折线”（继承自VMobject），由指定顶点按顺序连接形成非闭合的线段序列，
        适用于绘制折线图、路径线、不闭合的多边形边框等。

        参数说明：
            *vertices : 可变参数，多个三维向量，按顺序表示折线的顶点。
            **kwargs : 传递VMobject的配置（如color=BLACK、stroke_width=2）。
        """

        def __init__(
                self,
                *vertices: Vect3, **kwargs
        ):
            super().__init__(**kwargs)
            # 设置折线路径：按顶点顺序连接，不闭合（与Polygon的区别）
            self.set_points_as_corners(vertices)

    class RegularPolygon(Polygon):
        '''
        生成“正多边形”（继承自Polygon），所有边和角相等，中心在原点，
        适用于绘制正三角形、正方形、正五边形等规则图形。

        参数说明：
            n : 整数，多边形的边数/顶点数，默认6（正六边形）。
            radius : 浮点数，顶点到中心的距离（外接圆半径），默认1.0。
            start_angle : 浮点数，第一个顶点的起始角度（弧度），默认：
                          - 奇数n：0弧度（x轴正方向）；
                          - 偶数n：90度（PI/2弧度，y轴正方向）。
            **kwargs : 传递Polygon的配置（如fill_color=PURPLE、stroke_width=1）。

        示例：
            >>> # 生成5边形，起始角度30度（30*DEGREES）
            >>> pentagon = RegularPolygon(n=5, start_angle=30 * DEGREES)

        返回值：
            out : RegularPolygon object，满足参数的正多边形。
        '''

        def __init__(
                self,
                n: int = 6,
                radius: float = 1.0,
                start_angle: float | None = None, **kwargs
        ):
            # 确定默认起始角度：奇数n从x轴开始，偶数n从y轴开始（视觉居中）
            if start_angle is None:
                start_angle = (n % 2) * 90 * DEG  # n为偶数时n%2=0 → 0；奇数时1 → 90°
            # 计算第一个顶点的方向向量（从中心出发的单位向量旋转start_angle）
            start_vect = rotate_vector(radius * RIGHT, start_angle)
            # 生成所有顶点：均匀分布在圆周上（调用compass_directions生成等分方向）
            vertices = compass_directions(n, start_vect)
            # 调用父类Polygon的初始化，传入所有顶点
            super().__init__(*vertices, **kwargs)

    class Triangle(RegularPolygon):
        '''
        生成“正三角形”（继承自RegularPolygon，n=3），
        适用于几何图形演示、标记重点等场景。

        参数说明：
            start_angle : 浮点数，第一个顶点的起始角度（弧度），默认0（x轴正方向）。
            **kwargs : 传递RegularPolygon的配置（如side_length=2、color=RED）。

        示例：
            >>> # 生成起始角度45度的正三角形
            >>> triangle = Triangle(start_angle=45 * DEGREES)

        返回值：
            out : Triangle object，正三角形对象。
        '''

        def __init__(self, **kwargs):
            # 调用父类RegularPolygon的初始化，固定n=3（三角形）
            super().__init__(n=3, **kwargs)

    class ArrowTip(Triangle):
        """
        生成“箭头尖端”（继承自Triangle），用于Arrow、Line等图形的尖端，
        支持多种样式（三角形、平滑型、点型）。
        """

        def __init__(
                self,
                angle: float = 0,  # 尖端旋转角度（弧度）
                width: float = DEFAULT_ARROW_TIP_WIDTH,  # 尖端基部宽度
                length: float = DEFAULT_ARROW_TIP_LENGTH,  # 尖端长度
                fill_opacity: float = 1.0,  # 填充透明度
                fill_color: ManimColor = DEFAULT_MOBJECT_COLOR,  # 填充颜色
                stroke_width: float = 0.0,  # 描边线宽（默认无）
                tip_style: int = 0,  # 尖端样式：0=三角形，1=内部平滑型，2=点型
                **kwargs
        ):
            super().__init__(
                start_angle=0,  # 三角形起始角度（后续会旋转）
                fill_opacity=fill_opacity,
                fill_color=fill_color,
                stroke_width=stroke_width, **kwargs
            )
            # 设置尖端尺寸：高度=宽度，宽度=长度（拉伸三角形至尖端比例）
            self.set_height(width)
            self.set_width(length, stretch=True)

            # 根据样式调整尖端形状
            if tip_style == 1:
                # 内部平滑型：调整高度和路径点，使尖端更圆润
                self.set_height(length * 0.9, stretch=True)
                self.data["point"][4] += np.array([0.6 * length, 0, 0])  # 调整中间点
            elif tip_style == 2:
                # 点型：替换为圆点（复用Dot的路径）
                h = length / 2
                self.set_points(Dot().set_width(h).get_points())

            # 旋转尖端至指定角度
            self.rotate(angle)

        def get_base(self) -> Vect3:
            """获取尖端的基部中心点（与箭杆连接的位置）"""
            return self.point_from_proportion(0.5)  # 路径中点

        def get_tip_point(self) -> Vect3:
            """获取尖端的顶点（最前端的点）"""
            return self.get_points()[0]  # 路径第一个点

        def get_vector(self) -> Vect3:
            """获取从基部到顶点的向量（尖端方向）"""
            return self.get_tip_point() - self.get_base()

        def get_angle(self) -> float:
            """获取尖端的朝向角度（与x轴正方向的夹角）"""
            return angle_of_vector(self.get_vector())

        def get_length(self) -> float:
            """获取尖端的长度（基部到顶点的距离）"""
            return get_norm(self.get_vector())

    class Rectangle(Polygon):
        '''
        生成“矩形”（继承自Polygon），由四个顶点（UR、UL、DL、DR）组成，
        适用于绘制方框、背景、按钮等矩形元素。

        参数说明：
            width : 浮点数，矩形宽度（水平方向），默认4.0。
            height : 浮点数，矩形高度（垂直方向），默认2.0。
            **kwargs : 传递Polygon的配置（如fill_color=GRAY、stroke_width=2）。

        示例：
            >>> # 生成宽3、高4、蓝色的矩形
            >>> rectangle = Rectangle(width=3, height=4, color=BLUE)

        返回值：
            out : Rectangle object，满足参数的矩形。
        '''

        def __init__(
                self,
                width: float = 4.0,
                height: float = 2.0, **kwargs
        ):
            # 调用父类Polygon的初始化，顶点为右上(UR)、左上(UL)、左下(DL)、右下(DR)
            super().__init__(UR, UL, DL, DR, **kwargs)
            # 缩放矩形至指定宽度和高度
            self.set_width(width, stretch=True)  # 水平方向拉伸至width
            self.set_height(height, stretch=True)  # 垂直方向拉伸至height

        def surround(self, mobject, buff=SMALL_BUFF) -> Self:
            """
            调整矩形大小和位置，使其包围指定Mobject，并保留缓冲距离。

            参数：
                mobject : 被包围的Mobject对象。
                buff : 矩形与对象之间的缓冲距离，默认SMALL_BUFF（0.1）。
            返回：调整后的当前矩形对象。
            """
            # 计算目标尺寸：对象的包围盒尺寸 + 2*buff（上下左右各留buff）
            target_shape = np.array(mobject.get_shape()) + 2 * buff
            self.set_shape(*target_shape)  # 调整矩形尺寸
            self.move_to(mobject)  # 移动矩形至对象中心
            return self

    class Square(Rectangle):
        '''
        生成“正方形”（继承自Rectangle，宽=高），
        适用于绘制方块、图标、等宽高的背景等。

        参数说明：
            side_length : 浮点数，正方形的边长，默认2.0。
            **kwargs : 传递Rectangle的配置（如color=PINK、fill_opacity=0.5）。

        示例：
            >>> # 生成边长5、粉色的正方形
            >>> square = Square(side_length=5, color=PINK)

        返回值：
            out : Square object，满足参数的正方形。
        '''

        def __init__(self, side_length: float = 2.0, **kwargs):
            # 调用父类Rectangle的初始化，宽和高均为side_length
            super().__init__(side_length, side_length, **kwargs)

    class RoundedRectangle(Rectangle):
        '''
        生成“圆角矩形”（继承自Rectangle），矩形的四个角为圆弧，
        适用于绘制按钮、卡片、柔和边框等元素。

        参数说明：
            width : 浮点数，矩形宽度，默认4.0。
            height : 浮点数，矩形高度，默认2.0。
            corner_radius : 浮点数，圆角半径，默认0.5。
            **kwargs : 传递Rectangle的配置（如fill_color=WHITE、stroke_width=1）。

        示例：
            >>> # 生成宽3、高4、圆角半径1、蓝色的圆角矩形
            >>> r_rect = RoundedRectangle(width=3, height=4, corner_radius=1, color=BLUE)

        返回值：
            out : RoundedRectangle object，满足参数的圆角矩形。
        '''

        def __init__(
                self,
                width: float = 4.0,
                height: float = 2.0,
                corner_radius: float = 0.5, **kwargs
        ):
            # 调用父类Rectangle的初始化，生成普通矩形
            super().__init__(width, height, **kwargs)
            # 对四个角进行圆角处理（复用Polygon的round_corners方法）
            self.round_corners(corner_radius)
