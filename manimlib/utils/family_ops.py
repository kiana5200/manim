# ManimGL Mobject 家族成员处理工具：提供提取 Mobject 家族成员和递归移除指定 Mobject 的功能，
# 用于管理复杂嵌套结构的 Mobject（如 VGroup 及其子对象），确保操作能深入到所有层级的子对象。


from __future__ import annotations

from typing import TYPE_CHECKING

# 仅静态类型检查阶段导入类型注解（避免循环依赖）
if TYPE_CHECKING:
    from typing import Iterable, List, Set, Tuple

    from manimlib.mobject.mobject import Mobject  # 基础图形对象类


def extract_mobject_family_members(
    mobject_list: Iterable[Mobject],
    exclude_pointless: bool = False
) -> list[Mobject]:
    """
    提取多个 Mobject 的所有家族成员（包括自身及所有层级的子对象），形成扁平列表。
    
    家族成员定义：一个 Mobject 的家族包括其自身、所有子对象（submobjects）、子对象的子对象等（递归所有层级）。
    
    参数：
        mobject_list : 待提取的 Mobject 可迭代对象（如列表、VGroup）；
        exclude_pointless : 是否排除无顶点数据的 Mobject（`has_points() == False`），
                            默认为 False（保留所有家族成员）。
    
    返回：list[Mobject] - 所有家族成员的扁平列表，按深度优先顺序排列。
    """
    return [
        sm  # 子对象（家族成员）
        for mob in mobject_list  # 遍历输入的每个 Mobject
        for sm in mob.get_family()  # 递归获取该 Mobject 的所有家族成员
        # 根据 exclude_pointless 决定是否保留无顶点数据的成员
        if (not exclude_pointless) or sm.has_points()
    ]


def recursive_mobject_remove(
    mobjects: List[Mobject],
    to_remove: Set[Mobject]
) -> Tuple[List[Mobject], bool]:
    """
    递归移除列表中包含指定 Mobject 家族成员的对象：若某个 Mobject 本身在 `to_remove` 中，则直接移除；
    若其子孙对象在 `to_remove` 中，则替换为不含这些子孙的新结构，确保所有层级的指定对象都被移除。
    
    返回值：
        - 处理后的 Mobject 列表（已移除所有指定对象及其家族成员）；
        - 布尔值，指示是否有对象被移除（True 表示有移除操作）。
    
    参数：
        mobjects : 待处理的 Mobject 列表；
        to_remove : 需移除的 Mobject 集合（包含其家族成员的对象都会被处理）。
    """
    result = []  # 存储处理后的 Mobject 列表
    found_in_list = False  # 标记是否在当前层级发现并移除了对象

    for mob in mobjects:
        # 情况1：当前 Mobject 本身在待移除集合中，直接跳过（不加入结果）
        if mob in to_remove:
            found_in_list = True
            continue

        # 情况2：递归处理当前 Mobject 的子对象
        sub_list, found_in_submobjects = recursive_mobject_remove(
            mob.submobjects, to_remove
        )

        if found_in_submobjects:
            # 子对象中存在被移除的元素：用处理后的子对象列表替换原有的子对象
            # 先复制当前 Mobject（避免修改原对象）
            new_mob = mob.copy()
            new_mob.submobjects = sub_list  # 更新子对象为处理后的列表
            result.append(new_mob)
            found_in_list = True  # 标记有移除操作
        else:
            # 子对象中无移除操作：直接保留当前 Mobject
            result.append(mob)

    return result, found_in_list