# 导入未来版本的注解特性，支持更灵活的类型提示写法
from __future__ import annotations

# 导入抽象基类相关模块，用于定义抽象类和抽象方法
from abc import ABC, abstractmethod
# 导入迭代工具模块，用于高效循环和组合
import itertools as it
# 导入正则表达式模块，用于字符串匹配和处理
import re
# 导入科学计算库 scipy 的优化和空间距离计算功能
# - linear_sum_assignment: 用于解决指派问题（匈牙利算法）
# - cdist: 用于计算两个集合中所有点对的距离
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

# 从 manim 库导入常用常量和类
# - DEFAULT_MOBJECT_COLOR: 默认图形颜色
# - log: manim 的日志工具
# - SVGMobject: 用于处理 SVG 图形的类
# - VMobject/VGroup: 矢量图形对象及其组合类
from manimlib.constants import DEFAULT_MOBJECT_COLOR
from manimlib.logger import log
from manimlib.mobject.svg.svg_mobject import SVGMobject
from manimlib.mobject.types.vectorized_mobject import VMobject
from manimlib.mobject.types.vectorized_mobject import VGroup

# 从 manim 库导入颜色处理的工具函数
# - color_to_hex/hex_to_int/int_to_hex: 颜色格式转换函数
from manimlib.utils.color import color_to_hex
from manimlib.utils.color import hex_to_int
from manimlib.utils.color import int_to_hex

# 导入类型检查相关模块（仅在类型检查时生效）
from typing import TYPE_CHECKING

# 若处于类型检查模式，导入需要的类型注解（避免运行时依赖）
if TYPE_CHECKING:
    from typing import Callable
    from manimlib.typing import ManimColor, Span, Selector


# 定义 StringMobject 类，继承自 SVGMobject 和 ABC（抽象基类）
# 它是 Tex 和 MarkupText 的共同抽象父类
class StringMobject(SVGMobject, ABC):
    """
    An abstract base class for `Tex` and `MarkupText`

    This class aims to optimize the logic of "slicing submobjects
    via substrings". This could be much clearer and more user-friendly
    than slicing through numerical indices explicitly.

    Users are expected to specify substrings in `isolate` parameter
    if they want to do anything with their corresponding submobjects.
    `isolate` parameter can be either a string, a `re.Pattern` object,
    or a 2-tuple containing integers or None, or a collection of the above.
    Note, substrings specified cannot *partly* overlap with each other.

    Each instance of `StringMobject` may generate 2 svg files.
    The additional one is generated with some color commands inserted,
    so that each submobject of the original `SVGMobject` will be labelled
    by the color of its paired submobject from the additional `SVGMobject`.
    """
    height = None  # 类属性，用于存储图形高度，默认为 None

    def __init__(
        self,
        string: str,
        fill_color: ManimColor = DEFAULT_MOBJECT_COLOR,
        fill_border_width: float = 0.5,
        stroke_color: ManimColor = DEFAULT_MOBJECT_COLOR,
        stroke_width: float = 0,
        base_color: ManimColor = DEFAULT_MOBJECT_COLOR,
        isolate: Selector = (),
        protect: Selector = (),
        # When set to true, only the labelled svg is
        # rendered, and its contents are used directly
        # for the body of this String Mobject
        use_labelled_svg: bool = False,
        **kwargs
    ):
        """
            string (str): 要渲染的原始字符串。
            fill_color (ManimColor): 填充颜色。
            fill_border_width (float): 填充边框宽度。
            stroke_color (ManimColor): 描边颜色。
            stroke_width (float): 描边宽度。
            base_color (ManimColor): 基础颜色。
            isolate (Selector): 需要隔离的子字符串选择器。
            protect (Selector): 需要保护的子字符串选择器。
            use_labelled_svg (bool): 是否直接使用带标签的SVG。
            **kwargs: 传递给父类 SVGMobject 的其他参数。
        """
        self.string = string
        self.base_color = base_color or DEFAULT_MOBJECT_COLOR
        self.isolate = isolate
        self.protect = protect
        self.use_labelled_svg = use_labelled_svg

        self.parse()  # 解析字符串，处理isolate和protect
        svg_string = self.get_svg_string()  # 获取SVG字符串
        super().__init__(svg_string=svg_string, **kwargs)  # 调用父类构造函数
        self.set_stroke(stroke_color, stroke_width)  # 设置描边
        self.set_fill(fill_color, border_width=fill_border_width)  # 设置填充
        self.labels = [submob.label for submob in self.submobjects]  # 提取子对象标签

    # 获取SVG字符串。
    # 参数: is_labelled (bool): 是否获取带标签的SVG。
    def get_svg_string(self, is_labelled: bool = False) -> str:
        # 确定是否需要带标签的内容
        content = self.get_content(is_labelled or self.use_labelled_svg)
        # 将内容转换为SVG字符串
        return self.get_svg_string_by_content(content)

    @abstractmethod
    def get_svg_string_by_content(self, content: str) -> str:
        """
        抽象方法,用于将处理后的内容(content)转换为SVG字符串。
        子类(如Tex、MarkupText)必须实现此方法以提供具体的转换逻辑。
    
        参数:
            content (str): 已处理的内容字符串（可能包含标签或特殊指令）。
        返回:
            str: SVG格式的字符串。
    """
        return ""

    def assign_labels_by_color(self, mobjects: list[VMobject]) -> None:
        """
        Assuming each mobject in the list `mobjects` has a fill color
        meant to represent a numerical label, this assigns those
        those numerical labels to each mobject as an attribute
        """
        # 获取需要被标记的文本片段总数
        labels_count = len(self.labelled_spans)
        # 如果只有一个需要被标记的片段，所有子对象都属于它，直接分配标签0
        if labels_count == 1:
            for mob in mobjects:
                mob.label = 0
            return

        unrecognizable_colors = []
        # 遍历每一个图形对象
        for mob in mobjects:
            # 将对象的填充颜色从ManimColor -> HEX字符串 -> 整数，这个整数就是标签
            label = hex_to_int(color_to_hex(mob.get_fill_color()))
            # 检查标签是否有效（即是否在预期的范围内）
            if label >= labels_count:
                # 如果颜色无法识别（标签超出范围），记录下来并给一个默认标签0
                unrecognizable_colors.append(label)
                label = 0
            # 将计算出的标签赋值给对象的 `label` 属性    
            mob.label = label

        # 如果存在无法识别的颜色标签，打印一条警告日志
        if unrecognizable_colors:
            log.warning(
                "Unrecognizable color labels detected (%s). " + \
                "The result could be unexpected.",
                ", ".join(
                    int_to_hex(color)
                    for color in unrecognizable_colors
                )
            )


    def mobjects_from_svg_string(self, svg_string: str) -> list[VMobject]:
        """
        从SVG字符串创建图形对象列表，并为它们分配标签。
        这是`StringMobject`实现“通过子字符串选择子对象”的核心方法。
        它通过比较“带颜色标签的SVG”和“普通SVG”来完成标签的匹配和分配。

        参数:
            svg_string (str): 用于创建图形对象的原始SVG字符串。

        返回:
            list[VMobject]: 一组带有`label`属性的图形对象。
        """
        # 1. 从原始SVG字符串创建基础图形对象
        submobs = super().mobjects_from_svg_string(svg_string)

        # 2. 直接使用带标签的SVG（快速路径）
        if self.use_labelled_svg:
            # This means submobjects are colored according to spans
            # 如果直接使用带标签的SVG，那么子对象的颜色已经编码了标签信息
            self.assign_labels_by_color(submobs)
            return submobs

        # Otherwise, submobs are not colored, so generate a new list
        # of submobject which are and use those for labels
        # 3. 标准流程：通过对比带标签和不带标签的SVG来分配标签
        # 将原始SVG创建的对象作为未标记的对象
        unlabelled_submobs = submobs
        # 4. 创建并解析带标签的SVG
        # 生成包含颜色标签的内容
        labelled_content = self.get_content(is_labelled=True)
        # 将带标签的内容转换为SVG字符串
        labelled_file = self.get_svg_string_by_content(labelled_content)
        # 从带标签的SVG字符串创建图形对象
        labelled_submobs = super().mobjects_from_svg_string(labelled_file)
        # 保存带标签和不带标签的子对象列表，方便后续使用
        self.labelled_submobs = labelled_submobs
        self.unlabelled_submobs = unlabelled_submobs

        # 5. 为带标签的子对象分配标签
        # 根据它们的颜色（编码了标签信息）来设置`label`属性
        self.assign_labels_by_color(labelled_submobs)
        # 6. 对齐和匹配两个SVG的子对象
        # 使用匈牙利算法，根据位置关系将带标签的子对象与未带标签的子对象进行最佳匹配
        self.rearrange_submobjects_by_positions(labelled_submobs, unlabelled_submobs)
        # 7. 将标签从带标签的子对象复制到未带标签的子对象
        for usm, lsm in zip(unlabelled_submobs, labelled_submobs):
            usm.label = lsm.label

        # 8. 错误处理：如果两个SVG的子对象数量不匹配
        if len(unlabelled_submobs) != len(labelled_submobs):
            log.warning(
                "Cannot align submobjects of the labelled svg " + \
                "to the original svg. Skip the labelling process."
            )
            # 如果无法对齐，为所有未标记的子对象分配默认标签0
            for usm in unlabelled_submobs:
                usm.label = 0
            return unlabelled_submobs

        # 9. 返回带有正确标签的、未带颜色标记的原始图形对象
        return unlabelled_submobs

    def rearrange_submobjects_by_positions(
        self, labelled_submobs: list[VMobject], unlabelled_submobs: list[VMobject],
    ) -> None:
        """
            根据位置关系重新排列带标签的子对象列表，使其与未带标签的子对象列表中的元素按位置一一对应。

        这个方法是`StringMobject`内部实现的关键一步。由于插入颜色标签后，SVG的渲染结构可能会发生变化，
        导致带标签和不带标签的两个SVG生成的子对象顺序不一致。此方法通过计算距离并使用匈牙利算法来找到最佳匹配。

        参数:
            labelled_submobs (list[VMobject]): 从带颜色标签的SVG中生成的子对象列表。
            unlabelled_submobs (list[VMobject]): 从原始SVG中生成的子对象列表。
        """

        """
        Rearrange `labeleled_submobjects` so that each submobject
        is labelled by the nearest one of `unlabelled_submobs`.
        The correctness cannot be ensured, since the svg may
        change significantly after inserting color commands.
        """
        # 如果带标签的子对象列表为空，则无需进行任何操作
        if len(labelled_submobs) == 0:
            return

        # 1. 粗略对齐
        # 创建一个临时的VGroup来包含所有带标签的子对象
        labelled_svg = VGroup(*labelled_submobs)
        # 将这个组合移动到与未带标签的子对象组合相同的位置和大小，实现初步对齐
        labelled_svg.replace(VGroup(*unlabelled_submobs))
        # 2. 计算距离矩阵
        # 使用scipy的cdist计算两个点集之间所有点对的距离
        # 行：未带标签的子对象的中心点
        # 列：带标签的子对象的中心点
        distance_matrix = cdist(
            [submob.get_center() for submob in unlabelled_submobs],
            [submob.get_center() for submob in labelled_submobs]
        )
        # 3. 求解最优匹配
        # 使用匈牙利算法（linear_sum_assignment）找到距离总和最小的匹配方案
        # 返回值中的 `indices` 是带标签子对象的索引，与未带标签子对象的顺序相对应
        _, indices = linear_sum_assignment(distance_matrix)
        # 4. 重新排列带标签的子对象列表
        # 根据找到的最优匹配索引，对带标签的子对象列表进行重新排序
        # 排序后，labelled_submobs[i] 应该与 unlabelled_submobs[i] 是同一个字符/元素
        labelled_submobs[:] = [labelled_submobs[index] for index in indices]

    # Toolkits

    def find_spans_by_selector(self, selector: Selector) -> list[Span]:
        """
        根据选择器在字符串中查找匹配的文本片段(span),返回它们的索引范围。

        选择器(selector)可以是字符串、正则表达式对象、索引元组，或它们的集合。
        """
        def find_spans_by_single_selector(sel):
            # 如果选择器是字符串，则精确匹配该子串
            if isinstance(sel, str):
                return [
                    match_obj.span()
                    for match_obj in re.finditer(re.escape(sel), self.string)
                ]
            # 如果选择器是正则表达式对象，则按模式匹配
            if isinstance(sel, re.Pattern):
                return [
                    match_obj.span()
                    for match_obj in sel.finditer(self.string)
                ]
            # 如果选择器是二元组，则按索引切片处理
            if isinstance(sel, tuple) and len(sel) == 2 and all(
                isinstance(index, int) or index is None
                for index in sel
            ):
                l = len(self.string)
                span = tuple(
                    default_index if index is None else
                    min(index, l) if index >= 0 else max(index + l, 0)
                    for index, default_index in zip(sel, (0, l))
                )
                return [span]
            return None

        # 处理单个选择器的情况
        result = find_spans_by_single_selector(selector)
        # 如果选择器不是单个有效类型，则视为选择器集合进行遍历
        if result is None:
            result = []
            for sel in selector:
                spans = find_spans_by_single_selector(sel)
                if spans is None:
                    raise TypeError(f"Invalid selector: '{sel}'")
                result.extend(spans)
        # 过滤掉无效的索引范围（起始位置大于结束位置）
        return list(filter(lambda span: span[0] <= span[1], result))

    @staticmethod
    def span_contains(span_0: Span, span_1: Span) -> bool:
        return span_0[0] <= span_1[0] and span_0[1] >= span_1[1]

    # Parsing

    def parse(self) -> None:
        """
        这是 `StringMobject` 的核心解析方法。它的主要任务是：
        1.  根据 `isolate` 和 `protect` 等选择器，在原始字符串中找到所有需要特殊处理的文本片段（spans）。
        2.  处理这些片段之间的嵌套和重叠关系，确保它们能被正确地识别和标记。
        3.  生成一个详细的指令列表（`inserted_items`），用于指导如何生成带颜色标签的SVG。
        4.  生成一个需要被标记的文本片段列表（`labelled_items`），每个片段都关联一个唯一的标签和可能的属性（如颜色）。

        这个过程非常复杂，因为它需要处理字符串中的各种语法结构（如括号、命令）和用户指定的选择器之间的交互。
        """
        # 辅助函数：根据索引范围(span)从原始字符串中获取子串
        def get_substr(span: Span) -> str:
            return self.string[slice(*span)]

        # 1. 收集所有需要处理的项目
        # 获取通过配置文件或特定语法定义的项目（例如 MarkupText 的 <color> 标签）
        configured_items = self.get_configured_items()
        # 找到所有需要隔离的文本片段的索引范围
        isolated_spans = self.find_spans_by_selector(self.isolate)
        # 找到所有需要保护的文本片段的索引范围
        protected_spans = self.find_spans_by_selector(self.protect)
        # 找到所有特殊命令（如自定义的颜色命令）的匹配项
        command_matches = self.get_command_matches(self.string)

        def get_key(category, i, flag):
            def get_span_by_category(category, i):
                if category == 0:
                    return configured_items[i][0]
                if category == 1:
                    return isolated_spans[i]
                if category == 2:
                    return protected_spans[i]
                return command_matches[i].span()

            index, paired_index = get_span_by_category(category, i)[::flag]
            return (
                index,
                flag * (2 if index != paired_index else -1),
                -paired_index,
                flag * category,
                flag * i
            )

        index_items = sorted([
            (category, i, flag)
            for category, item_length in enumerate((
                len(configured_items),
                len(isolated_spans),
                len(protected_spans),
                len(command_matches)
            ))
            for i in range(item_length)
            for flag in (1, -1)
        ], key=lambda t: get_key(*t))

        # 初始化用于跟踪解析状态的各种堆栈和列表
        inserted_items = []          # 最终生成的指令列表，用于生成带标签的SVG
        labelled_items = []          # 最终确定的、需要被标记的文本片段列表
        overlapping_spans = []       # 用于记录检测到的重叠片段
        level_mismatched_spans = []  # 用于记录因括号/命令嵌套问题无法处理的片段

        label = 1
        protect_level = 0
        bracket_stack = [0]
        bracket_count = 0
        open_command_stack = []
        open_stack = []
        # 遍历所有按顺序排列的“事件”
        for category, i, flag in index_items:
            # 第一部分：处理保护项（category=2）和命令项（category=3）
            if category >= 2:
                # 更新保护级别：开始事件（flag=1）提升级别，结束事件（flag=-1）降低级别
                protect_level += flag

                # 分支1：跳过保护项的开始事件，或所有保护项（category=2）的结束事件
                # （保护项仅用于控制级别，不直接生成标签）
                if flag == 1 or category == 2:
                    continue
                # 分支2：处理命令项（category=3）的结束事件
                # 1. 记录当前命令的索引和类型（0表示普通命令标记）
                inserted_items.append((i, 0))
                # 获取当前命令的匹配结果（如自定义颜色命令）
                command_match = command_matches[i]
                # 判断命令类型（1=开始命令，-1=结束命令，0=无效命令）
                command_flag = self.get_command_flag(command_match)

                # 子分支1：若为开始命令（如"\color{"），记录括号层级和命令位置
                if command_flag == 1:
                    bracket_count += 1
                    bracket_stack.append(bracket_count)
                    # 记录命令在inserted_items中的位置和索引，供结束命令匹配
                    open_command_stack.append((len(inserted_items), i))
                    continue
                # 子分支2：若为无效命令，跳过
                if command_flag == 0:
                    continue

                # 子分支3：若为结束命令（如"}"），匹配对应的开始命令
                # 弹出之前记录的开始命令位置和索引
                pos, i_ = open_command_stack.pop()
                bracket_stack.pop()  # 括号层级回退
                open_command_match = command_matches[i_]  # 获取开始命令的匹配结果
                # 从开始-结束命令对中提取属性（如颜色值）
                attr_dict = self.get_attr_dict_from_command_pair(
                    open_command_match, command_match
                )
                if attr_dict is None:  # 若属性无效，跳过
                    continue

                # 确定命令对包裹的文本片段范围（开始命令结束位置 ~ 结束命令开始位置）
                span = (open_command_match.end(), command_match.start())
                # 将该片段加入待标记列表（关联属性）
                labelled_items.append((span, attr_dict))
                # 在inserted_items中插入标签的开始和结束标记（用于生成带颜色的SVG）
                inserted_items.insert(pos, (label, 1))  # 标签开始（label标识，1=开始）
                inserted_items.insert(-1, (label, -1))  # 标签结束（label标识，-1=结束）
                label += 1  # 更新标签，确保下一个片段用新标签
                continue

            # 第二部分：处理配置项（category=0）和隔离项（category=1）
            # 分支1：若为片段开始事件（flag=1），记录当前状态到堆栈
            if flag == 1:
                open_stack.append((
                    len(inserted_items),  # 当前在inserted_items中的位置
                    category, i,          # 事件类型和索引
                    protect_level,        # 当前保护级别（用于后续校验）
                    bracket_stack.copy()  # 当前括号层级（深拷贝，用于后续校验）
                ))
                continue

            # 分支2：若为片段结束事件（flag=-1），匹配堆栈中记录的开始事件
            # 1. 获取当前片段的范围和属性（配置项有预设属性，隔离项无属性）
            span, attr_dict = configured_items[i] \
                if category == 0 else (isolated_spans[i], {})
            # 弹出堆栈中记录的开始事件状态
            pos, category_, i_, protect_level_, bracket_stack_ \
                = open_stack.pop()
            # 校验1：开始-结束事件的类型和索引是否匹配（防止片段交叉）
            if category_ != category or i_ != i:
                overlapping_spans.append(span)
                continue
            # 校验2：片段是否处于保护状态（保护状态下不标记）
            if protect_level_ or protect_level:
                continue
            # 校验3：片段开始和结束时的括号层级是否一致（防止跨括号标记）
            if bracket_stack_ != bracket_stack:
                level_mismatched_spans.append(span)  # 记录层级不匹配的片段
                continue
            # 所有校验通过：将片段加入待标记列表，并插入标签标记
            labelled_items.append((span, attr_dict))
            inserted_items.insert(pos, (label, 1))  # 标签开始
            inserted_items.append((label, -1))  # 标签结束
            label += 1

        # 收尾工作
        # 将整个字符串作为一个整体添加到标记列表的开头
        labelled_items.insert(0, ((0, len(self.string)), {}))
        # 为整个字符串添加开始和结束标签指令
        inserted_items.insert(0, (0, 1))
        inserted_items.append((0, -1))

        # 报告错误
        # 如果在解析过程中发现了重叠或无法处理的片段，打印警告日志
        if overlapping_spans:
            log.warning(
                "Partly overlapping substrings detected: %s",
                ", ".join(
                    f"'{get_substr(span)}'"
                    for span in overlapping_spans
                )
            )
        # 若存在括号层级不匹配的片段（如片段跨未闭合括号），打印警告日志
        if level_mismatched_spans:
            log.warning(
                "Cannot handle substrings: %s",
                ", ".join(
                    f"'{get_substr(span)}'"  # 用辅助函数get_substr获取片段对应的实际文本
                    for span in level_mismatched_spans
                )
            )

        def reconstruct_string(
            start_item: tuple[int, int],
            end_item: tuple[int, int],
            command_replace_func: Callable[[re.Match], str],
            command_insert_func: Callable[[int, int, dict[str, str]], str]
        ) -> str:
            """
            重构字符串的核心辅助函数。
            根据解析得到的inserted_items(指令列表），结合自定义的命令替换/插入逻辑，
            生成最终用于渲染SVG的字符串(可能包含颜色标签等特殊指令）。

            参数:
                start_item: 重构的起始指令(来自inserted_items的元素,格式为(i, flag))
                end_item: 重构的结束指令（格式同上）
                command_replace_func: 替换原始命令的函数(如将自定义命令替换为SVG支持的指令)
                command_insert_func: 插入标签命令的函数（如为标记片段插入颜色指令）
            返回:
                重构后的完整字符串
            """

            def get_edge_item(i: int, flag: int) -> tuple[Span, str]:
                """
                处理inserted_items中的单个指令,返回其对应的索引范围和替换/插入的字符串。
                指令分两类:命令指令(flag=0)和标签指令(flag≠0)。
                """
                # 1. 处理命令指令（flag=0，对应原始字符串中的特殊命令，如自定义颜色命令）
                if flag == 0:
                    match_obj = command_matches[i]  # 获取命令的正则匹配结果
                    return (
                        match_obj.span(),  # 返回命令在原始字符串中的索引范围（start, end）
                        command_replace_func(match_obj)  # 用自定义函数替换该命令
                    )
                # 2. 处理标签指令（flag≠0，对应待标记的片段，如isolate指定的片段）
                span, attr_dict = labelled_items[i] # 获取片段的索引范围和属性（如颜色）
                index = span[flag < 0]  # 根据flag方向取片段的起始（flag=1）或结束（flag=-1）索引
                return (
                    (index, index),  # 返回一个“空范围”（仅标记位置，不占用文本）
                    command_insert_func(i, flag, attr_dict)  # 插入标签命令（如颜色开始/结束指令）
                )

            # 步骤1：提取从start_item到end_item的所有指令，并处理为（索引范围，替换文本）对
            items = [
                get_edge_item(i, flag)
                for i, flag in inserted_items[slice(
                    inserted_items.index(start_item),  # 起始指令在inserted_items中的位置
                    inserted_items.index(end_item) + 1  # 结束指令位置（+1确保包含结束项）
                )]
            ]
            # 步骤2：提取原始字符串中“指令之间的文本片段”
            pieces = [
                get_substr((start, end))  # 用get_substr获取原始字符串中两个指令之间的文本
                for start, end in zip(
                    [interval_end for (_, interval_end), _ in items[:-1]],  # 前一个指令的结束索引
                    [interval_start for (interval_start, _), _ in items[1:]]  # 后一个指令的起始索引
                )
            ]
            # 步骤3：提取指令对应的“替换/插入文本”（排除首尾指令，仅保留中间指令）
            interval_pieces = [piece for _, piece in items[1:-1]]
            # 步骤4：拼接文本片段和指令文本，生成最终重构字符串
            # 用zip将“原始文本片段”和“指令文本”交替拼接，最后补空字符串确保长度匹配
            return "".join(it.chain(*zip(pieces, (*interval_pieces, ""))))

        # 存储解析结果：待标记片段的索引范围列表
        self.labelled_spans = [span for span, _ in labelled_items]
        # 将重构字符串的函数绑定为实例方法，供后续生成SVG时调用
        self.reconstruct_string = reconstruct_string

    def get_content(self, is_labelled: bool) -> str:
        content = self.reconstruct_string(
            (0, 1), (0, -1),
            self.replace_for_content,
            lambda label, flag, attr_dict: self.get_command_string(
                attr_dict,
                is_end=flag < 0,
                label_hex=int_to_hex(label) if is_labelled else None
            )
        )
        prefix, suffix = self.get_content_prefix_and_suffix(
            is_labelled=is_labelled
        )
        return "".join((prefix, content, suffix))

    @staticmethod
    @abstractmethod
    def get_command_matches(string: str) -> list[re.Match]:
        return []

    @staticmethod
    @abstractmethod
    def get_command_flag(match_obj: re.Match) -> int:
        return 0

    @staticmethod
    @abstractmethod
    def replace_for_content(match_obj: re.Match) -> str:
        return ""

    @staticmethod
    @abstractmethod
    def replace_for_matching(match_obj: re.Match) -> str:
        return ""

    @staticmethod
    @abstractmethod
    def get_attr_dict_from_command_pair(
        open_command: re.Match, close_command: re.Match,
    ) -> dict[str, str] | None:
        return None

    @abstractmethod
    def get_configured_items(self) -> list[tuple[Span, dict[str, str]]]:
        return []

    @staticmethod
    @abstractmethod
    def get_command_string(
        attr_dict: dict[str, str], is_end: bool, label_hex: str | None
    ) -> str:
        return ""

    @abstractmethod
    def get_content_prefix_and_suffix(
        self, is_labelled: bool
    ) -> tuple[str, str]:
        return "", ""

    # Selector

    def get_submob_indices_list_by_span(
        self, arbitrary_span: Span
    ) -> list[int]:
        return [
            submob_index
            for submob_index, label in enumerate(self.labels)
            if self.span_contains(arbitrary_span, self.labelled_spans[label])
        ]

    def get_specified_part_items(self) -> list[tuple[str, list[int]]]:
        return [
            (
                self.string[slice(*span)],
                self.get_submob_indices_list_by_span(span)
            )
            for span in self.labelled_spans[1:]
        ]

    def get_specified_substrings(self) -> list[str]:
        substrs = [
            self.string[slice(*span)]
            for span in self.labelled_spans[1:]
        ]
        # Use dict.fromkeys to remove duplicates while retaining order
        return list(dict.fromkeys(substrs).keys())

    def get_group_part_items(self) -> list[tuple[str, list[int]]]:
        if not self.labels:
            return []

        def get_neighbouring_pairs(vals):
            return list(zip(vals[:-1], vals[1:]))

        range_lens, group_labels = zip(*(
            (len(list(grouper)), val)
            for val, grouper in it.groupby(self.labels)
        ))
        submob_indices_lists = [
            list(range(*submob_range))
            for submob_range in get_neighbouring_pairs(
                [0, *it.accumulate(range_lens)]
            )
        ]
        labelled_spans = self.labelled_spans
        start_items = [
            (group_labels[0], 1),
            *(
                (curr_label, 1)
                if self.span_contains(
                    labelled_spans[prev_label], labelled_spans[curr_label]
                )
                else (prev_label, -1)
                for prev_label, curr_label in get_neighbouring_pairs(
                    group_labels
                )
            )
        ]
        end_items = [
            *(
                (curr_label, -1)
                if self.span_contains(
                    labelled_spans[next_label], labelled_spans[curr_label]
                )
                else (next_label, 1)
                for curr_label, next_label in get_neighbouring_pairs(
                    group_labels
                )
            ),
            (group_labels[-1], -1)
        ]
        group_substrs = [
            re.sub(r"\s+", "", self.reconstruct_string(
                start_item, end_item,
                self.replace_for_matching,
                lambda label, flag, attr_dict: ""
            ))
            for start_item, end_item in zip(start_items, end_items)
        ]
        return list(zip(group_substrs, submob_indices_lists))

    def get_submob_indices_lists_by_selector(
        self, selector: Selector
    ) -> list[list[int]]:
        return list(filter(
            lambda indices_list: indices_list,
            [
                self.get_submob_indices_list_by_span(span)
                for span in self.find_spans_by_selector(selector)
            ]
        ))

    def build_parts_from_indices_lists(
        self, indices_lists: list[list[int]]
    ) -> VGroup:
        return VGroup(*(
            VGroup(*(
                self.submobjects[submob_index]
                for submob_index in indices_list
            ))
            for indices_list in indices_lists
        ))

    def build_groups(self) -> VGroup:
        return self.build_parts_from_indices_lists([
            indices_list
            for _, indices_list in self.get_group_part_items()
        ])

    def select_parts(self, selector: Selector) -> VGroup:
        specified_substrings = self.get_specified_substrings()
        if isinstance(selector, (str, re.Pattern)) and selector not in specified_substrings:
            return self.select_unisolated_substring(selector)
        indices_list = self.get_submob_indices_lists_by_selector(selector)
        return self.build_parts_from_indices_lists(indices_list)

    def __getitem__(self, value: int | slice | Selector) -> VMobject:
        if isinstance(value, (int, slice)):
            return super().__getitem__(value)
        return self.select_parts(value)

    def select_part(self, selector: Selector, index: int = 0) -> VMobject:
        return self.select_parts(selector)[index]

    def substr_to_path_count(self, substr: str) -> int:
        return len(re.sub(r"\s", "", substr))

    def get_symbol_substrings(self):
        return list(re.sub(r"\s", "", self.string))

    def select_unisolated_substring(self, pattern: str | re.Pattern) -> VGroup:
        if isinstance(pattern, str):
            pattern = re.compile(re.escape(pattern))
        result = []
        for match in re.finditer(pattern, self.string):
            index = match.start()
            start = self.substr_to_path_count(self.string[:index])
            substr = match.group()
            end = start + self.substr_to_path_count(substr)
            result.append(self[start:end])
        return VGroup(*result)

    def set_parts_color(self, selector: Selector, color: ManimColor):
        self.select_parts(selector).set_color(color)
        return self

    def set_parts_color_by_dict(self, color_map: dict[Selector, ManimColor]):
        for selector, color in color_map.items():
            self.set_parts_color(selector, color)
        return self

    def get_string(self) -> str:
        return self.string
