# 导入未来版本的注解特性（支持更灵活的类型提示写法）
from __future__ import annotations

# 导入工具函数：reduce用于迭代计算，operator模块用于操作符函数
from functools import reduce
import operator as op
# 导入正则表达式模块，用于字符串匹配/处理
import re

# 从manim库导入常用常量和类：
# - BLACK/DEFAULT_MOBJECT_COLOR：默认颜色常量
# - SVGMobject：用于处理SVG图形的类
# - VGroup：用于组合多个图形对象的类
# - latex_to_svg：将LaTeX代码转换为SVG图形的工具函数
from manimlib.constants import BLACK, DEFAULT_MOBJECT_COLOR
from manimlib.mobject.svg.svg_mobject import SVGMobject
from manimlib.mobject.types.vectorized_mobject import VGroup
from manimlib.utils.tex_file_writing import latex_to_svg

# 导入类型检查相关模块（仅在类型检查时生效，不影响运行）
from typing import TYPE_CHECKING

# 若处于类型检查模式，导入需要的类型注解（避免运行时依赖）
if TYPE_CHECKING:
    from typing import Iterable, List, Dict
    from manimlib.typing import ManimColor


# 定义字体点数到manim单位的缩放因子：1个字体点对应0.001个manim单位
SCALE_FACTOR_PER_FONT_POINT = 0.001


# 定义 SingleStringTex 类，继承自 SVGMobject，用于将单个LaTeX字符串渲染为SVG图形对象
class SingleStringTex(SVGMobject):
    height: float | None = None  # 类属性，存储图形高度，默认为None（未指定）

    def __init__(
        self,
        tex_string: str,                  # 要渲染的LaTeX字符串（核心内容）
        height: float | None = None,      # 图形最终高度，优先级高于字体大小缩放
        fill_color: ManimColor = DEFAULT_MOBJECT_COLOR,  # 填充颜色，默认使用Manim默认色
        fill_opacity: float = 1.0,        # 填充不透明度，默认完全不透明
        stroke_width: float = 0,          # 描边宽度，默认无描边
        svg_default: dict = dict(fill_color=DEFAULT_MOBJECT_COLOR),  # SVG元素默认样式
        path_string_config: dict = dict(),# SVG路径配置参数
        font_size: int = 48,              # LaTeX渲染的字体大小（单位：磅）
        alignment: str = R"\centering",   # LaTeX内容对齐方式，默认居中
        math_mode: bool = True,           # 是否启用数学模式（如渲染公式），默认启用
        organize_left_to_right: bool = False,  # 是否按从左到右顺序排列子对象，默认不排列
        template: str = "",               # LaTeX模板文件路径，默认使用系统模板
        additional_preamble: str = "",    # 额外的LaTeX导言区代码（如加载宏包）
        **kwargs
    ):
        # 存储LaTeX渲染相关的关键参数为实例属性
        self.tex_string = tex_string
        self.svg_default = dict(svg_default)
        self.path_string_config = dict(path_string_config)
        self.font_size = font_size
        self.alignment = alignment
        self.math_mode = math_mode
        self.organize_left_to_right = organize_left_to_right
        self.template = template
        self.additional_preamble = additional_preamble

        # 调用父类 SVGMobject 构造函数，传递图形样式等参数初始化
        super().__init__(
            height=height,
            fill_color=fill_color,
            fill_opacity=fill_opacity,
            stroke_width=stroke_width,
            path_string_config=path_string_config,
            **kwargs
        )

        # 若未指定高度，按“字体大小×缩放因子”自动计算缩放比例
        if self.height is None:
            self.scale(SCALE_FACTOR_PER_FONT_POINT * self.font_size)
        # 若需要按从左到右排列子对象，执行排序逻辑
        if self.organize_left_to_right:
            self.organize_submobjects_left_to_right()

@property
def hash_seed(self) -> tuple:
    """
    定义属性 hash_seed，返回用于标识当前 SingleStringTex 实例的唯一元组（用于哈希计算）。
    包含类名、SVG默认样式、路径配置、LaTeX字符串等关键参数，确保相同配置生成相同哈希。
    """
    return (
        self.__class__.__name__,  # 当前类名（固定为 SingleStringTex）
        self.svg_default,         # SVG元素默认样式配置
        self.path_string_config,  # SVG路径配置参数
        self.tex_string,          # 核心LaTeX字符串
        self.alignment,           # LaTeX内容对齐方式
        self.math_mode,           # 是否启用数学模式
        self.template,            # LaTeX模板路径
        self.additional_preamble  # 额外LaTeX导言区代码
    )

def get_svg_string_by_content(self, content: str) -> str:
    """
    将传入的LaTeX内容（content）转换为SVG格式的字符串。
    调用manim的latex_to_svg工具函数，使用当前实例的模板和额外导言区配置。
    
    参数:
        content: 要转换的LaTeX内容字符串
    返回:
        对应的SVG格式字符串
    """
    return latex_to_svg(content, self.template, self.additional_preamble)

def get_tex_file_body(self, tex_string: str) -> str:
    """
    生成完整的LaTeX文件正文内容（用于后续转换为SVG）。
    处理LaTeX字符串，根据是否启用数学模式添加对应环境，再加上对齐方式配置。
    
    参数:
        tex_string: 原始LaTeX字符串
    返回:
        完整的LaTeX文件正文（含对齐方式、数学环境）
    """
    # 对原始LaTeX字符串进行特殊处理（如替换特殊字符，由modify_special_strings实现）
    new_tex = self.get_modified_expression(tex_string)
    # 若启用数学模式，给处理后的LaTeX内容包裹align*环境（用于公式对齐）
    if self.math_mode:
        new_tex = "\\begin{align*}\n" + new_tex + "\n\\end{align*}"
    # 拼接对齐方式（如\centering）和处理后的LaTeX内容，形成完整正文
    return self.alignment + "\n" + new_tex

def get_modified_expression(self, tex_string: str) -> str:
    """
    对LaTeX字符串进行预处理：去除首尾空白，并调用modify_special_strings处理特殊字符/格式。
    
    参数:
        tex_string: 原始LaTeX字符串
    返回:
        预处理后的LaTeX字符串
    """
    return self.modify_special_strings(tex_string.strip())  # strip()去除首尾空白

def modify_special_strings(self, tex: str) -> str:
    """
    对LaTeX字符串进行预处理，解决一些常见的语法问题，确保其能被正确渲染。
    """
    tex = tex.strip()

    # 1. 处理需要填充的特殊命令
    # 判断是否需要添加单个填充符（\quad）
    should_add_filler = reduce(op.or_, [
        tex == "\\over",        # 分数线需要有分子
        tex == "\\overline",    # 上划线需要有内容
        tex == "\\sqrt",        # 平方根需要有内容
        tex == "\\sqrt{",       # 平方根命令未闭合
        tex.endswith("_"),      # 下标的底未指定
        tex.endswith("^"),      # 上标的底未指定
        tex.endswith("dot"),    # 点命令未指定对象
    ])
    if should_add_filler:
        filler = "{\\quad}"  # 使用\quad作为填充
        tex += filler

    # 2. 处理需要双重填充的特殊命令
    should_add_double_filler = reduce(op.or_, [
        tex == "\\overset",     # \overset需要两个参数
    ])
    if should_add_double_filler:
        filler = "{\\quad}{\\quad}"  # 添加两个填充
        tex += filler

    # 3. 处理特殊的空白情况
    if tex == "\\substack":
        tex = "\\quad"  # \substack需要内容，用\quad替代
    if tex == "":
        tex = "\\quad"  # 空字符串用\quad替代

    # 4. 处理行首的换行符
    if tex.startswith("\\\\"):
        tex = tex.replace("\\\\", "\\quad\\\\")  # 在换行前添加\quad，避免文件以换行开头

    # 5. 平衡括号
    tex = self.balance_braces(tex)

    # 6. 处理不平衡的 \left 和 \right
    # 统计 \left 和 \right 的数量
    num_lefts, num_rights = [
        len([
            s for s in tex.split(substr)[1:]
            if s and s[0] in "(){}[]|.\\"
        ])
        for substr in ("\\left", "\\right")
    ]
    # 如果数量不相等，将 \left 和 \right 替换为 \big
    if num_lefts != num_rights:
        tex = tex.replace("\\left", "\\big")
        tex = tex.replace("\\right", "\\big")

    # 7. 处理不平衡的环境（如 array）
    for context in ["array"]:
        begin_in = ("\\begin{%s}" % context) in tex
        end_in = ("\\end{%s}" % context) in tex
        if begin_in ^ end_in:  # 如果只有开始或只有结束
            tex = ""  # 将内容清空，避免渲染错误

    return tex

def balance_braces(self, tex: str) -> str:
    """
    平衡LaTeX字符串中的花括号 `{` 和 `}`，使其成对出现，增强渲染的容错性。

    参数:
        tex (str): 需要处理的LaTeX字符串。

    返回:
        str: 经过平衡处理后的LaTeX字符串。
    """
    num_unclosed_brackets = 0
    # 遍历字符串中的每一个字符
    for i in range(len(tex)):
        # 如果当前字符前有反斜杠 \，说明是转义字符（如 \{}），跳过不处理
        if i > 0 and tex[i - 1] == "\\":
            continue
        char = tex[i]
        # 如果遇到左花括号 `{`，未闭合计数器加一
        if char == "{":
            num_unclosed_brackets += 1
        # 如果遇到右花括号 `}`
        elif char == "}":
            # 如果此时没有未闭合的左括号，说明多了一个右括号，在字符串开头补一个左括号
            if num_unclosed_brackets == 0:
                tex = "{" + tex
            # 否则，将未闭合计数器减一
            else:
                num_unclosed_brackets -= 1
    # 遍历结束后，如果还有未闭合的左括号，在字符串末尾添加相应数量的右括号来闭合它们
    tex += num_unclosed_brackets * "}"
    return tex


def get_tex(self) -> str:
    """
    获取用于创建此对象的原始LaTeX字符串。

    返回:
        str: 存储在实例中的原始LaTeX字符串。
    """
    return self.tex_string


def organize_submobjects_left_to_right(self):
    """
    按从左到右的顺序排列对象的子部分。
    对于确保复杂的LaTeX表达式（如分数、根号）的子元素在动画中以正确的顺序出现很有用。
    """
    # 根据每个子对象的x坐标（p[0]）进行排序，实现从左到右的排列
    self.sort(lambda p: p[0])
    return self


# 定义 OldTex 类，继承自 SingleStringTex，支持多段LaTeX字符串拆分、隔离特定内容及颜色映射
class OldTex(SingleStringTex):
    def __init__(
        self,
        *tex_strings: str,                # 可变参数，传入一个或多个LaTeX字符串（核心内容）
        arg_separator: str = "",          # 连接多段LaTeX字符串的分隔符，默认无分隔符
        isolate: List[str] = [],          # 需要单独隔离的子字符串列表（如特殊符号、关键词）
        tex_to_color_map: Dict[str, ManimColor] = {},  # LaTeX内容到颜色的映射字典（指定内容颜色）
        **kwargs                          # 传递给父类 SingleStringTex 的其他参数
    ):
        # 1. 拆分并隔离LaTeX字符串
        # 调用方法将多段字符串按“需隔离内容”拆分，得到拆分后的子字符串列表
        self.tex_strings = self.break_up_tex_strings(
            tex_strings,
            substrings_to_isolate=[*isolate, *tex_to_color_map.keys()]  # 需隔离的内容：用户指定+颜色映射的键
        )
        # 将拆分后的子字符串用分隔符连接，形成完整LaTeX字符串
        full_string = arg_separator.join(self.tex_strings)

        # 2. 调用父类构造函数初始化（渲染完整LaTeX字符串）
        super().__init__(full_string, **kwargs)
        
        # 3. 按拆分后的子字符串分割图形对象（让每个隔离内容成为独立子对象）
        self.break_up_by_substrings(self.tex_strings)
        # 4. 根据颜色映射字典，为指定LaTeX内容设置对应颜色
        self.set_color_by_tex_to_color_map(tex_to_color_map)

        # 5. 若需按从左到右排列子对象，执行排序
        if self.organize_left_to_right:
            self.organize_submobjects_left_to_right()

    def break_up_tex_strings(self, tex_strings: Iterable[str], substrings_to_isolate: List[str] = []) -> Iterable[str]:
        """
        将传入的LaTeX字符串按“需隔离的子字符串”拆分，返回拆分后的子字符串列表。
        （例如：若隔离"x"，则"ax+b"拆分为["a", "x", "+b"]）
        
        参数:
            tex_strings: 待拆分的原始LaTeX字符串集合
            substrings_to_isolate: 需要单独隔离的子字符串列表
        返回:
            拆分后、非空的子字符串列表
        """
        # 若无需隔离任何内容，直接返回原始字符串集合
        if len(substrings_to_isolate) == 0:
            return tex_strings
        
        # 1. 构建正则表达式模式（转义特殊字符，用|连接所有需隔离内容）
        patterns = (
            "({})".format(re.escape(ss))  # 转义子字符串中的正则特殊字符（如\、{等）
            for ss in substrings_to_isolate
        )
        pattern = "|".join(patterns)  # 正则模式：匹配任意一个需隔离的子字符串
        
        # 2. 遍历每个原始字符串，按正则模式拆分
        pieces = []
        for s in tex_strings:
            if pattern:
                # 按模式拆分字符串，结果包含分隔符（需隔离内容）
                pieces.extend(re.split(pattern, s))
            else:
                pieces.append(s)
        
        # 3. 过滤掉空字符串，返回非空拆分结果
        return list(filter(lambda s: s, pieces))

def break_up_by_substrings(self, tex_strings: Iterable[str]):
    """
    根据拆分后的LaTeX字符串列表，重新组织当前对象的子对象结构。
    它将渲染后的SVG图形的底层路径（submobjects）分组，使每个分组对应一个被隔离的LaTeX子字符串。
    这是实现“按内容上色”和“按内容动画”的关键步骤。

    参数:
        tex_strings (Iterable[str]): 一个包含了被隔离和未被隔离部分的LaTeX字符串列表。
    """
    # 如果列表中只有一个字符串，说明没有需要隔离的内容，
    # 则将整个对象作为一个单独的子对象处理。
    if len(list(tex_strings)) == 1:
        submob = self.copy()
        self.set_submobjects([submob])
        return self

    new_submobjects = []
    curr_index = 0
    # 遍历拆分后的每一个LaTeX子字符串
    for tex_string in tex_strings:
        tex_string = tex_string.strip()
        if len(tex_string) == 0:
            continue
        
        # 1. 创建一个临时的SingleStringTex对象
        # 这个对象本身不会被添加到场景中，它的作用是：
        # a. 验证这个LaTeX子字符串是否能被渲染。
        # b. 获取渲染后它应该包含多少个底层路径（submobjects）。
        sub_tex_mob = SingleStringTex(tex_string, math_mode=self.math_mode)
        num_submobs = len(sub_tex_mob)
        if num_submobs == 0:
            continue
        
        # 2. 从原始对象中截取对应的底层路径
        # 根据临时对象的子对象数量，从原始对象的完整路径列表中截取相应的部分。
        new_index = curr_index + num_submobs
        sub_tex_mob.set_submobjects(self.submobjects[curr_index:new_index])
        
        # 3. 将分组后的对象添加到新列表中
        new_submobjects.append(sub_tex_mob)
        curr_index = new_index

    # 4. 用新的、分组后的子对象列表替换原始的子对象列表
    # 从此以后，self.submobjects 中的每一个元素都是一个代表被隔离内容的 VGroup/SingleStringTex。
    self.set_submobjects(new_submobjects)
    return self

def get_parts_by_tex(
        self,
        tex: str,
        substring: bool = True,
        case_sensitive: bool = True
    ) -> VGroup:
    """
    根据指定的LaTeX字符串，从当前OldTex对象的子对象中筛选匹配的部分，返回组合后的VGroup。

    参数:
        tex (str): 用于匹配的目标LaTeX字符串（如"x"、"\\frac"）。
        substring (bool): 是否按子串匹配（True：目标在子对象LaTeX中即可；False：需完全一致），默认True。
        case_sensitive (bool): 是否区分大小写（True：严格区分大小写；False：忽略大小写），默认True。

    返回:
        VGroup: 所有匹配的子对象（SingleStringTex类型）组成的组合对象。
    """
    # 定义匹配规则的内部函数
    def test(tex1, tex2):
        # 若不区分大小写，先将两个字符串转为小写
        if not case_sensitive:
            tex1 = tex1.lower()
            tex2 = tex2.lower()
        # 按子串匹配或完全匹配判断
        if substring:
            return tex1 in tex2  # 子串匹配：tex1是tex2的一部分即算匹配
        else:
            return tex1 == tex2  # 完全匹配：两个字符串必须完全相同

    # 筛选子对象：仅保留SingleStringTex类型且符合匹配规则的子对象，组合成VGroup返回
    return VGroup(*filter(
        lambda m: isinstance(m, SingleStringTex) and test(tex, m.get_tex()),
        self.submobjects
    ))

def get_part_by_tex(self, tex: str, **kwargs) -> SingleStringTex | None:
    """
    根据指定的LaTeX字符串，获取当前OldTex对象中第一个匹配的子对象（SingleStringTex类型）。
    若没有匹配项，返回None。

    参数:
        tex (str): 用于匹配的目标LaTeX字符串。
        **kwargs: 传递给get_parts_by_tex的额外参数（如substring、case_sensitive）。

    返回:
        SingleStringTex | None: 第一个匹配的子对象，或None（无匹配时）。
    """
    # 先通过get_parts_by_tex获取所有匹配项
    all_parts = self.get_parts_by_tex(tex, **kwargs)
    # 返回第一个匹配项，若无匹配则返回None
    return all_parts[0] if all_parts else None

def set_color_by_tex(self, tex: str, color: ManimColor, **kwargs):
    """
    根据指定的LaTeX字符串，为当前OldTex对象中所有匹配的子对象设置颜色。

    参数:
        tex (str): 用于匹配的目标LaTeX字符串。
        color (ManimColor): 要设置的颜色（如RED、BLUE）。
        **kwargs: 传递给get_parts_by_tex的额外参数（如substring、case_sensitive）。

    返回:
        self: 返回自身，支持链式调用。
    """
    # 获取所有匹配的子对象，调用set_color设置颜色
    self.get_parts_by_tex(tex, **kwargs).set_color(color)
    return self

def set_color_by_tex_to_color_map(
        self,
        tex_to_color_map: dict[str, ManimColor],
        **kwargs
    ):
    """
    根据一个LaTeX字符串到颜色的映射字典，批量为对象中的特定部分设置颜色。

    参数:
        tex_to_color_map (dict): 一个形如 { "tex_string": color, ... } 的字典。
        **kwargs: 传递给 `set_color_by_tex` 的额外参数（如 `substring`）。

    返回:
        self: 返回自身，支持链式调用。
    """
    # 遍历字典中的每一个键值对（LaTeX字符串和对应的颜色）
    for tex, color in list(tex_to_color_map.items()):
        # 调用 set_color_by_tex 为匹配的部分设置颜色
        self.set_color_by_tex(tex, color, **kwargs)
    return self


def index_of_part(self, part: SingleStringTex, start: int = 0) -> int:
    """
    获取指定子对象在 `self.submobjects` 列表中的索引。

    参数:
        part (SingleStringTex): 要查找的子对象。
        start (int): 搜索的起始索引，默认为0。

    返回:
        int: 子对象的索引。
    """
    return self.submobjects.index(part, start)


def index_of_part_by_tex(self, tex: str, start: int = 0, **kwargs) -> int:
    """
    根据LaTeX字符串，获取其对应子对象在 `self.submobjects` 列表中的索引。

    参数:
        tex (str): 要查找的LaTeX字符串。
        start (int): 搜索的起始索引，默认为0。
        **kwargs: 传递给 `get_part_by_tex` 的额外参数。

    返回:
        int: 子对象的索引。
    """
    # 先用LaTeX字符串找到对应的子对象
    part = self.get_part_by_tex(tex, **kwargs)
    # 再获取该子对象的索引
    return self.index_of_part(part, start)


def slice_by_tex(
        self,
        start_tex: str | None = None,
        stop_tex: str | None = None,
        **kwargs
    ) -> VGroup:
    """
    根据起始和结束的LaTeX字符串，从当前对象中切分出一个子对象组合（VGroup）。

    参数:
        start_tex (str | None): 起始部分的LaTeX字符串。如果为None，则从开头开始。
        stop_tex (str | None): 结束部分的LaTeX字符串。如果为None，则切到结尾。
        **kwargs: 传递给 `index_of_part_by_tex` 的额外参数。

    返回:
        VGroup: 切分出的子对象组合。
    """
    # 确定切片的起始索引
    if start_tex is None:
        start_index = 0
    else:
        start_index = self.index_of_part_by_tex(start_tex, **kwargs)

    # 确定切片的结束索引并执行切片
    if stop_tex is None:
        # 如果没有结束字符串，返回从start_index到末尾的所有子对象
        return self[start_index:]
    else:
        # 如果有结束字符串，找到其索引并进行切片（不包含结束部分）
        stop_index = self.index_of_part_by_tex(stop_tex, start=start_index, **kwargs)
        return self[start_index:stop_index]


def sort_alphabetically(self) -> None:
    """
    将所有子对象按照其LaTeX字符串的字母顺序进行排序。
    """
    self.submobjects.sort(key=lambda m: m.get_tex())


def set_bstroke(self, color: ManimColor = BLACK, width: float = 4):
    """
    为对象设置一个“背景描边”，即一个比前景更粗、颜色不同的描边，常用于突出显示。

    参数:
        color (ManimColor): 背景描边的颜色，默认为黑色。
        width (float): 背景描边的宽度，默认为4。

    返回:
        self: 返回自身，支持链式调用。
    """
    self.set_stroke(color, width, background=True)
    return self


# 定义 OldTexText 类，继承自 OldTex，专门用于渲染普通文本（非数学公式）
class OldTexText(OldTex):
    def __init__(
        self,
        *tex_strings: str,                # 可变参数，传入一个或多个待渲染的文本字符串
        math_mode: bool = False,          # 数学模式开关，默认关闭（区别于父类OldTex的默认True）
        arg_separator: str = "",          # 连接多段文本的分隔符，默认无分隔符
        **kwargs                          # 传递给父类 OldTex 的其他参数（如颜色、字体大小等）
    ):
        # 调用父类 OldTex 的构造函数初始化
        # 核心差异：强制关闭数学模式，适配普通文本渲染场景（避免公式环境干扰）
        super().__init__(
            *tex_strings,
            math_mode=math_mode,           # 固定传递False，确保以普通文本模式渲染
            arg_separator=arg_separator,  # 传递文本分隔符
            **kwargs                       # 传递其他配置（如样式、隔离内容等）
        )
