# ManimGL 调试与辅助工具函数：提供 Mobject 层级结构打印和索引标签生成功能，
# 便于开发过程中查看 Mobject 父子关系、快速定位子对象，是调试复杂动画的实用工具。


from __future__ import annotations

# 导入核心常量和类：黑色（用于标签描边）、日志工具、整数标签、矢量组
from manimlib.constants import BLACK
from manimlib.logger import log
from manimlib.mobject.numbers import Integer  # 整数文本 Mobject
from manimlib.mobject.types.vectorized_mobject import VGroup  # 矢量对象组

from typing import TYPE_CHECKING

# 仅静态类型检查阶段导入 Mobject 类型（避免循环依赖）
if TYPE_CHECKING:
    from manimlib.mobject.mobject import Mobject


def print_family(mobject: Mobject, n_tabs: int = 0) -> None:
    """
    递归打印 Mobject 的层级结构（含子对象），用于调试时查看对象父子关系和内存地址。
    
    输出格式：每个对象前的制表符数量表示其层级（父对象 0 个制表符，子对象 1 个，以此类推），
    后跟对象字符串表示和内存地址（id），便于区分不同实例。
    
    参数：
        mobject : 要打印层级的 Mobject（如 VGroup、Scene 中的所有对象）；
        n_tabs : 初始制表符数量（默认 0，用于递归时增加层级）。
    """
    # 打印当前对象：制表符（层级）+ 对象描述 + 内存地址
    log.debug("\t" * n_tabs + str(mobject) + " " + str(id(mobject)))
    # 递归打印所有子对象（层级+1）
    for submob in mobject.submobjects:
        print_family(submob, n_tabs + 1)


def index_labels(
    mobject: Mobject, 
    label_height: float = 0.15
) -> VGroup:
    """
    为 Mobject 的所有子对象生成索引标签（0, 1, 2...），直观标记子对象的序号，
    便于在动画中定位特定子对象（如 VGroup 中的第 n 个元素）。
    
    标签特性：
    - 以整数形式显示子对象索引（从 0 开始）；
    - 标签大小由 label_height 控制（默认 0.15 虚拟单位）；
    - 标签居中放置在对应子对象上；
    - 标签添加黑色描边（避免与背景混淆）。
    
    参数：
        mobject : 包含子对象的 Mobject（如 VGroup，需支持迭代访问子对象）；
        label_height : 标签的高度（虚拟单位，控制大小）。
    返回：VGroup，包含所有索引标签（按子对象顺序排列）。
    """
    labels = VGroup()  # 用于存储所有标签的矢量组
    # 遍历子对象，为每个子对象创建索引标签
    for n, submob in enumerate(mobject):
        label = Integer(n)  # 创建整数标签（显示索引 n）
        label.set_height(label_height)  # 设置标签高度
        label.move_to(submob)  # 将标签移动到子对象的位置（居中）
        label.set_backstroke(BLACK, 5)  # 添加黑色描边（宽度 5），增强可读性
        labels.add(label)  # 将标签添加到组中
    return labels