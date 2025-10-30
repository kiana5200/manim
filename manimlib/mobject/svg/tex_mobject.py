# 导入Python 3.7+的特性：允许在类定义中直接引用类名作为类型注解（无需提前声明）
# 例如在 `class A: def func(self) -> A: ...` 中，`A` 无需提前定义即可作为返回类型
from __future__ import annotations

# 导入正则表达式模块，用于处理字符串的模式匹配、查找、替换等操作
# 常见场景：提取文本中的特定内容、验证字符串格式、批量替换字符等
import re

# 导入路径处理模块，提供面向对象的文件路径操作接口
# 相比传统的os.path，Path支持链式调用（如Path("a/b").joinpath("c")），且跨平台兼容性更好
# 常见场景：拼接路径、判断文件是否存在、创建目录、遍历文件夹等
from pathlib import Path

from manimlib.mobject.svg.string_mobject import StringMobject
from manimlib.mobject.types.vectorized_mobject import VGroup
from manimlib.mobject.types.vectorized_mobject import VMobject
from manimlib.utils.color import color_to_hex
from manimlib.utils.color import hex_to_int
from manimlib.utils.tex_file_writing import latex_to_svg
from manimlib.utils.tex import num_tex_symbols
from manimlib.logger import log

from typing import TYPE_CHECKING

# 仅在类型检查模式下导入类型注解（避免运行时循环依赖或冗余导入）
if TYPE_CHECKING:
    from manimlib.typing import ManimColor, Span, Selector, Self


# Tex对象的缩放因子：将SVG生成的图形尺寸适配Manim坐标系
TEX_MOB_SCALE_FACTOR = 0.001


class Tex(StringMobject):
    """
    用于渲染LaTeX公式的图形对象类，继承自处理字符串的`StringMobject`。
    核心功能是将LaTeX代码转换为SVG图形，并支持颜色映射、内容隔离等特性。
    """
    # 默认的LaTeX环境（用于多行文公式对齐，如多行等式）
    tex_environment: str = "align*"

    def __init__(
        self,
        *tex_strings: str,
        font_size: int = 48,
        alignment: str = R"\centering",
        template: str = "",
        additional_preamble: str = "",
        tex_to_color_map: dict = dict(),
        t2c: dict = dict(),
        isolate: Selector = [],
        use_labelled_svg: bool = True,
        **kwargs
    ):
        """
        初始化Tex对象，处理LaTeX字符串、配置渲染参数并调用父类初始化。

        参数:
            *tex_strings: 可变参数，传入一个或多个LaTeX字符串（会自动拼接并标记为隔离片段）。
            font_size: 渲染后的字体大小（默认48），最终通过缩放因子适配Manim坐标系。
            alignment: LaTeX环境内的对齐方式（默认居中，即`\centering`）。
            template: 自定义LaTeX模板（用于覆盖默认渲染模板，如引入特定宏包）。
            additional_preamble: 额外的LaTeX导言区代码（如`\usepackage{amsmath}`）。
            tex_to_color_map: LaTeX内容到颜色的映射字典（键为LaTeX片段，值为颜色）。
            t2c: `tex_to_color_map`的简写，功能完全一致，优先级低于前者（后者会覆盖前者）。
            isolate: 需要单独标记的LaTeX片段（用于后续精准选择或颜色设置）。
            use_labelled_svg: 是否使用带标签的SVG渲染（用于追踪片段位置，支持后续选择操作）。
            **kwargs: 传递给父类`StringMobject`的额外参数（如位置、透明度等）。
        """
        # 1. 处理多LaTeX字符串参数：拼接为单个字符串，并自动将每个原始字符串标记为隔离片段
        # Combine multi-string arg, but mark them to isolate
        if len(tex_strings) > 1:
            # 确保isolate是列表类型（若传入字符串/正则/元组，转为列表）
            if isinstance(isolate, (str, re.Pattern, tuple)):
                isolate = [isolate]
            # 将每个原始tex字符串添加到隔离列表（后续可单独选择或设置样式）
            isolate = [*isolate, *tex_strings]

        # 2. 拼接所有LaTeX字符串为单个字符串，并去除首尾空白
        tex_string = (" ".join(tex_strings)).strip()

        # Prevent from passing an empty string.
        # 3. 避免传入空字符串：若拼接后为空，用LaTeX换行符占位（防止渲染错误）
        if not tex_string.strip():
            tex_string = R"\\"  # LaTeX中的空行/换行标记

        # 4. 保存LaTeX相关配置参数
        self.tex_string = tex_string          # 最终用于渲染的LaTeX字符串
        self.alignment = alignment            # LaTeX环境内的对齐方式
        self.template = template              # 自定义LaTeX模板
        self.additional_preamble = additional_preamble  # 额外导言区代码
        # 合并颜色映射（t2c是简写，优先级低，会被tex_to_color_map覆盖）
        self.tex_to_color_map = dict(**t2c, **tex_to_color_map)

        # 5. 调用父类StringMobject的初始化方法（处理SVG渲染、隔离片段等核心逻辑）
        super().__init__(
            tex_string,
            use_labelled_svg=use_labelled_svg,
            isolate=isolate,
            **kwargs
        )

        # 6. 根据颜色映射为LaTeX片段设置颜色
        self.set_color_by_tex_to_color_map(self.tex_to_color_map)

        # 7. 按字体大小缩放图形（通过TEX_MOB_SCALE_FACTOR适配Manim坐标系）
        self.scale(TEX_MOB_SCALE_FACTOR * font_size)

        # 8. 保存字体大小（必须在缩放后赋值，确保记录的是最终生效的字体大小）
        self.font_size = font_size  # Important for this to go after the scale call

    def get_svg_string_by_content(self, content: str) -> str:
        """
        根据传入的LaTeX内容字符串，生成对应的SVG字符串。
        这是连接LaTeX渲染与SVG图形的核心方法，通过调用`latex_to_svg`工具函数实现转换。

        参数:
            content (str): 待渲染为SVG的LaTeX内容（通常是经过处理的完整LaTeX代码，含环境声明）。

        返回:
            str: 渲染后的SVG字符串，可直接用于后续解析为Manim图形对象。
        """
        # 调用外部LaTeX转SVG工具函数，传入模板、额外导言区和原始TeX字符串（用于简化处理）
        return latex_to_svg(content, self.template, self.additional_preamble, short_tex=self.tex_string)

    def _handle_scale_side_effects(self, scale_factor: float) -> Self:
        """
        处理缩放操作带来的副作用，核心是同步更新字体大小属性。
        当Tex对象被缩放时，需确保记录的`font_size`与实际显示尺寸一致，避免后续操作尺寸混乱。

        参数:
            scale_factor (float): 缩放因子（大于1放大，小于1缩小）。

        返回:
            Self: 处理后的Tex对象自身，支持链式调用。
        """
        if hasattr(self, "font_size"):
            self.font_size *= scale_factor
        return self

    # Parsing

    @staticmethod
    def get_command_matches(string: str) -> list[re.Match]:
        """
        静态方法：解析LaTeX字符串，提取其中的命令（如`\alpha`、`\frac`）和成对的花括号`{}`，
        返回正则匹配结果列表。核心是处理嵌套/相邻的花括号，确保每对`{`与`}`正确匹配。

        参数:
            string (str): 待解析的原始LaTeX字符串（如`\frac{1}{2} + \alpha`）。

        返回:
            list[re.Match]: 包含所有命令和成对花括号的正则匹配结果列表。

        异常:
            ValueError: 若LaTeX字符串中存在未闭合的`{`或多余的`}`，抛出语法错误。
        """
        # 正则表达式模式：匹配LaTeX命令（如\a、\frac）、左花括号{+（一个或多个）、右花括号}+（一个或多个）
        # re.X允许注释和空格，re.S使.匹配换行符
        # Lump together adjacent brace pairs
        pattern = re.compile(r"""
            (?P<command>\\(?:[a-zA-Z]+|.))  # 匹配命令：\ + 字母序列（如\frac）或单个字符（如\$）
            |(?P<open>{+)                   # 匹配一个或多个左花括号
            |(?P<close>}+)                  # 匹配一个或多个右花括号
        """, flags=re.X | re.S)
        result = []  # 存储最终的匹配结果
        open_stack = []  # 栈：记录未匹配的左花括号信息，元素为 (左花括号的起止位置, 插入结果列表的索引)

        # 遍历所有初始匹配结果（未处理花括号配对）
        for match_obj in pattern.finditer(string):
            # 情况1：匹配到左花括号 {+
            if match_obj.group("open"):
                # 记录左花括号的位置和当前结果列表长度（后续用于插入匹配的花括号对）
                open_stack.append((match_obj.span(), len(result)))
            # 情况2：匹配到右花括号 }+    
            elif match_obj.group("close"):
                close_start, close_end = match_obj.span()
                # 循环处理：直到右花括号完全匹配或栈为空（处理嵌套/多对花括号）
                while True:
                    # 若栈为空但有未匹配的右花括号，说明语法错误（缺少{）
                    if not open_stack:
                        raise ValueError("Missing '{' inserted")
                    # 弹出栈顶未匹配的左花括号信息
                    (open_start, open_end), index = open_stack.pop()
                    n = min(open_end - open_start, close_end - close_start)
                    result.insert(index, pattern.fullmatch(
                        string, pos=open_end - n, endpos=open_end
                    ))
                    result.append(pattern.fullmatch(
                        string, pos=close_start, endpos=close_start + n
                    ))
                    close_start += n
                    if close_start < close_end:
                        continue
                    open_end -= n
                    if open_start < open_end:
                        open_stack.append(((open_start, open_end), index))
                    break
            # 情况3：匹配到LaTeX命令（如\alpha、\frac），直接加入结果列表
            else:
                result.append(match_obj)
        # 遍历结束后，若栈不为空，说明存在未闭合的{，语法错误
        if open_stack:
            raise ValueError("Missing '}' inserted")
        return result

    @staticmethod
    def get_command_flag(match_obj: re.Match) -> int:
        """
        判断单个命令匹配结果的类型（开始命令/结束命令/无效命令）。
        用于在解析LaTeX时识别花括号的层次结构。

        参数:
            match_obj (re.Match): 单个命令的正则匹配结果。

        返回:
            int: 命令类型标记（1=开始命令，-1=结束命令，0=无效命令）。
        """
        if match_obj.group("open"):
            return 1  # 左花括号 {，表示开始
        if match_obj.group("close"):
            return -1  # 右花括号 }，表示结束
        return 0  # 其他命令（如 \alpha），既非开始也非结束

    @staticmethod
    def replace_for_content(match_obj: re.Match) -> str:
        """
        将命令匹配结果替换为适合生成SVG内容的字符串。
        对于Tex类，直接返回原始匹配内容即可，因为花括号和命令本身就是LaTeX语法的一部分。

        参数:
            match_obj (re.Match): 单个命令的正则匹配结果。

        返回:
            str: 替换后的字符串（即原始匹配内容）。
        """
        return match_obj.group()

    @staticmethod
    def replace_for_matching(match_obj: re.Match) -> str:
        """
        将命令匹配结果替换为适合“匹配逻辑”的字符串。
        此方法用于生成一个“规范化”的字符串，以便在比较文本片段时忽略花括号等结构字符。

        参数:
            match_obj (re.Match): 单个命令的正则匹配结果。

        返回:
            str: 替换后的字符串。
        """
        if match_obj.group("command"):
            return match_obj.group()  # 保留命令（如 \alpha）
        return ""  # 移除花括号 { 和 }，因为它们不影响文本内容的匹配

    @staticmethod
    def get_attr_dict_from_command_pair(
        open_command: re.Match, close_command: re.Match
    ) -> dict[str, str] | None:
        """
        从“开始命令-结束命令”对中提取属性字典。
        对于Tex类，此方法默认不提取任何属性，仅在特殊情况下返回空字典。

        参数:
            open_command (re.Match): 开始命令的正则匹配结果。
            close_command (re.Match): 结束命令的正则匹配结果。

        返回:
            dict[str, str] | None: 属性字典（若提取到属性）或None（未提取到）。
        """
        # 仅当开始命令的匹配内容长度 >= 2 时返回空字典
        # 这是一个占位逻辑，实际可能用于区分不同类型的命令对
        if len(open_command.group()) >= 2:
            return {}
        return None

    def get_configured_items(self) -> list[tuple[Span, dict[str, str]]]:
        """
        获取所有通过 `tex_to_color_map` 配置的文本片段及其对应的属性。
        该方法返回一个列表，其中每个元素是 (Span, 属性字典) 的元组。

        返回:
            list[tuple[Span, dict[str, str]]]: 配置项列表。
        """
        return [
            (span, {})  # 目前属性字典为空，可扩展为包含颜色或其他样式信息
            for selector in self.tex_to_color_map
            for span in self.find_spans_by_selector(selector)
        ]

    @staticmethod
    def get_color_command(rgb_hex: str) -> str:
        """
        将16进制RGB颜色字符串转换为LaTeX的 `\color[RGB]{...}` 命令。

        参数:
            rgb_hex (str): 16进制颜色字符串，如 "FF0080"。

        返回:
            str: LaTeX颜色命令，如 "\\color[RGB]{255, 0, 128}"。
        """
        # 将16进制颜色字符串转换为整数
        rgb = hex_to_int(rgb_hex)
        # 分解为R、G、B三个8位整数
        rg, b = divmod(rgb, 256)
        r, g = divmod(rg, 256)
        # 格式化并返回LaTeX颜色命令
        return f"\\color[RGB]{{{r}, {g}, {b}}}"

    @staticmethod
    def get_command_string(
        attr_dict: dict[str, str], is_end: bool, label_hex: str | None
    ) -> str:
        """
        生成用于标记文本片段的LaTeX命令字符串。
        当使用带标签的SVG渲染时，该方法用于在需要着色的文本片段前后插入颜色命令。

        参数:
            attr_dict (dict[str, str]): 属性字典（目前未使用）。
            is_end (bool): 是否为结束命令（True表示片段结束，False表示片段开始）。
            label_hex (str | None): 16进制颜色标签（用于生成颜色命令）。

        返回:
            str: LaTeX命令字符串。
        """
        # 如果没有颜色标签，不生成任何命令
        if label_hex is None:
            return ""
        # 如果是结束命令，返回双右花括号（结束颜色环境）
        if is_end:
            return "}}"
        # 如果是开始命令，返回双左花括号+颜色命令（开始颜色环境）
        return "{{" + Tex.get_color_command(label_hex)

    def get_content_prefix_and_suffix(
        self, is_labelled: bool
    ) -> tuple[str, str]:
        """
        生成用于包装LaTeX内容的前缀和后缀字符串。
        这些字符串包含环境声明、对齐方式和颜色设置等。

        参数:
            is_labelled (bool): 是否使用带标签的SVG渲染。

        返回:
            tuple[str, str]: (前缀字符串, 后缀字符串)。
        """
        prefix_lines = []
        suffix_lines = []
        # 如果不是带标签的渲染，应用基础颜色
        if not is_labelled:
            prefix_lines.append(self.get_color_command(
                color_to_hex(self.base_color)
            ))
        # 添加对齐命令
        if self.alignment:
            prefix_lines.append(self.alignment)
        # 添加LaTeX环境声明
        if self.tex_environment:
            prefix_lines.append(f"\\begin{{{self.tex_environment}}}")
            suffix_lines.append(f"\\end{{{self.tex_environment}}}")
        # 格式化前缀和后缀字符串（添加换行符）
        return (
            "".join([line + "\n" for line in prefix_lines]),
            "".join(["\n" + line for line in suffix_lines])
        )

    # Method alias

    def get_parts_by_tex(self, selector: Selector) -> VGroup:
        """
        根据选择器获取LaTeX片段对应的图形部分。
        这是 `select_parts` 方法的别名，使API更符合LaTeX使用习惯。

        参数:
            selector (Selector): 用于查找LaTeX片段的选择器。

        返回:
            VGroup: 一个包含所有匹配片段的VGroup。
        """
        return self.select_parts(selector)

    def get_part_by_tex(self, selector: Selector, index: int = 0) -> VMobject:
        """
        根据选择器获取指定索引的LaTeX片段对应的图形部分。
        这是 `select_part` 方法的别名。

        参数:
            selector (Selector): 用于查找LaTeX片段的选择器。
            index (int): 要返回的片段在匹配结果中的索引，默认为0。

        返回:
            VMobject: 匹配结果中指定索引的图形部分。
        """
        return self.select_part(selector, index)

    def set_color_by_tex(self, selector: Selector, color: ManimColor):
        """
        根据选择器为LaTeX片段设置颜色。
        这是 `set_parts_color` 方法的别名。

        参数:
            selector (Selector): 用于查找LaTeX片段的选择器。
            color (ManimColor): 要设置的颜色。

        返回:
            self: 返回自身，便于链式调用。
        """
        return self.set_parts_color(selector, color)

    def set_color_by_tex_to_color_map(
        self, color_map: dict[Selector, ManimColor]
    ):
        """
        根据一个字典批量为多个LaTeX片段设置颜色。
        这是 `set_parts_color_by_dict` 方法的别名。

        参数:
            color_map (dict[Selector, ManimColor]): 一个字典，键是选择器，值是对应的颜色。

        返回:
            self: 返回自身，便于链式调用。
        """
        return self.set_parts_color_by_dict(color_map)

    # 获取原始的LaTeX字符串。
    # 这是 `get_string` 方法的别名。
    def get_tex(self) -> str:
        return self.get_string()

    # 将LaTeX子字符串映射为SVG路径的数量。
    # 这个方法通过计算子字符串中包含的LaTeX符号数量来实现映射。
    # Specific to Tex
    def substr_to_path_count(self, substr: str) -> int:
        tex = self.get_tex()
        # 检查当前图形的子对象数量是否与LaTeX字符串中的符号数量匹配
        # 如果不匹配，打印警告信息（可能是由于复杂的宏或特殊符号导致的）
        if len(self) != num_tex_symbols(tex):
            log.warning(f"Estimated size of {tex} does not match true size")
        # 返回子字符串中包含的LaTeX符号数量
        return num_tex_symbols(substr)

    def get_symbol_substrings(self):
        """
        将LaTeX字符串拆分为单个符号或命令的列表。
        这对于精确控制和选择LaTeX公式中的特定部分非常有用。

        返回:
            list[str]: 包含LaTeX符号或命令的列表。
        """
        # 正则表达式模式，用于匹配LaTeX中的符号或命令
        pattern = "|".join((
            # Tex commands
            r"\\[a-zA-Z]+",  # 匹配LaTeX命令，如 \alpha, \frac
            # And most single characters, with these exceptions
            r"[^\^\{\}\s\_\$\\\&]",  # 匹配大多数单个字符，但排除特殊字符如^, {, }, 空格等
        ))
        # 使用正则表达式查找所有匹配项
        return re.findall(pattern, self.string)

    def make_number_changeable(
        self,
        value: float | int | str,
        index: int = 0,
        replace_all: bool = False,
        **config,
    ) -> VMobject:
        """
        查找并返回指定数字在LaTeX公式中的图形部分，使其可以被后续动画修改。
        这是创建数字变化动画的第一步。

        参数:
            value (float | int | str): 要查找的数字（可以是整数、浮点数或字符串形式）。
            index (int): 要查找的数字在所有匹配项中的索引，默认为0（第一个）。
            replace_all (bool): 是否替换所有匹配的数字，默认为False（仅替换指定索引的一个）。
            **config: 传递给返回的VMobject的其他配置参数。

        返回:
            VMobject: 代表指定数字的图形对象（通常是一个VGroup）。
        """
        # 1. 将输入的数字转换为字符串，以便进行文本匹配
        substr = str(value)
        # 2. 在当前Tex对象中查找所有匹配该数字字符串的部分
        parts = self.select_parts(substr)
        # 3. 处理未找到匹配项的情况
        if len(parts) == 0:
            log.warning(f"{value} not found in Tex.make_number_changeable call")
            return VMobject()  # 返回一个空的VMobject，避免后续出错
        # 4. 处理索引越界的情况
        if index > len(parts) - 1:
            log.warning(f"Requested {index}th occurance of {value}, but only {len(parts)} exist")
            return VMobject()  # 返回一个空的VMobject
        # 5. 根据replace_all标志决定返回单个还是多个匹配部分
        if not replace_all:
            parts = [parts[index]]  # 如果不替换所有，只保留指定索引的那一个

        # 6. 创建一个VGroup来包装找到的部分，并应用配置
        # 注意：原始代码似乎不完整，这里根据上下文补充了合理的返回逻辑
        from manimlib.mobject.numbers import DecimalNumber

        decimal_mobs = []
        for part in parts:
            # 1. 确定小数点后的位数
            if "." in substr:
                num_decimal_places = len(substr.split(".")[1])
            else:
                num_decimal_places = 0
            # 2. 创建DecimalNumber对象，用于动态显示数字
            decimal_mob = DecimalNumber(
                float(value),
                num_decimal_places=num_decimal_places,
                **config,
            )
            # 3. 用新创建的DecimalNumber替换原来的数字部分
            decimal_mob.replace(part)
            decimal_mob.match_style(part)
            # 4. 如果原数字由多个子对象组成，移除多余的部分
            if len(part) > 1:
                self.remove(*part[1:])
            # 5. 将原数字的第一个子对象替换为DecimalNumber
            self.replace_submobject(self.submobjects.index(part[0]), decimal_mob)
            decimal_mobs.append(decimal_mob)

            # Replace substr with something that looks like a tex command. This
            # is to ensure Tex.substr_to_path_count counts it correctly.
            # 6. 更新内部字符串表示，确保后续操作正确计数
            # 将原数字字符串替换为一个特殊的LaTeX命令占位符
            self.string = self.string.replace(substr, R"\decimalmob", 1)

        # 7. 根据replace_all标志返回适当的结果
        if replace_all:
            return VGroup(*decimal_mobs)
        return decimal_mobs[index]


# 用于显示普通文本的Tex子类。
# 它不使用任何特殊的LaTeX数学环境，适用于显示普通文本内容。
class TexText(Tex):
    tex_environment: str = ""
