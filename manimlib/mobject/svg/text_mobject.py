# 导入Python 3.7+的特性：允许在类定义中直接引用类名作为类型注解（无需提前声明）
from __future__ import annotations

# 导入上下文管理器相关工具，用于创建临时的、可进入/退出的运行环境
from contextlib import contextmanager

# 导入标准库中的os模块，用于与操作系统交互（如环境变量、文件路径等）
import os

# 导入pathlib模块，提供面向对象的文件路径操作接口
from pathlib import Path

# 导入正则表达式模块，用于处理字符串的模式匹配、查找、替换等操作
import re

# 导入临时文件模块，用于创建临时文件和目录
import tempfile

# 导入functools中的lru_cache装饰器，用于缓存函数的计算结果，提升性能
from functools import lru_cache

# 导入manimpango库，Manim用于文本渲染的核心库（基于Pango）
import manimpango

# 导入pygments库，用于代码语法高亮
import pygments

# 导入pygments的格式化器，用于将高亮后的代码转换为特定格式（如HTML、SVG等）
import pygments.formatters

# 导入pygments的词法分析器，用于识别不同编程语言的语法
import pygments.lexers

from manimlib.config import manim_config
from manimlib.constants import DEFAULT_PIXEL_WIDTH, FRAME_WIDTH
from manimlib.constants import NORMAL
from manimlib.logger import log
from manimlib.mobject.svg.string_mobject import StringMobject
from manimlib.utils.cache import cache_on_disk
from manimlib.utils.color import color_to_hex
from manimlib.utils.color import int_to_hex
from manimlib.utils.simple_functions import hash_string

from typing import TYPE_CHECKING

# 仅在类型检查模式下导入类型注解（避免运行时循环依赖或冗余导入）
if TYPE_CHECKING:
    from typing import Iterable

    from manimlib.mobject.types.vectorized_mobject import VGroup
    from manimlib.typing import ManimColor, Span, Selector


# 文本对象的缩放因子：将SVG生成的图形尺寸适配Manim坐标系
TEXT_MOB_SCALE_FACTOR = 0.0076
# 默认行间距缩放比例
DEFAULT_LINE_SPACING_SCALE = 0.6
# 确保画布足够大以容纳所有字形
DEFAULT_CANVAS_WIDTH = 16384
DEFAULT_CANVAS_HEIGHT = 16384


# 临时的对齐方式处理器
class _Alignment:
    """
    一个简单的辅助类，用于将字符串形式的对齐方式（如"LEFT"）转换为Manim内部使用的数值。
    """
    VAL_DICT = {
        "LEFT": 0,
        "CENTER": 1,
        "RIGHT": 2
    }

    def __init__(self, s: str):
        """
        初始化对齐方式处理器。

        参数:
            s (str): 对齐方式字符串，如"LEFT", "CENTER", "RIGHT"。
        """
        self.value = _Alignment.VAL_DICT[s.upper()]


@lru_cache(maxsize=128)
@cache_on_disk
def markup_to_svg(
    markup_str: str,
    justify: bool = False,
    indent: float = 0,
    alignment: str = "CENTER",
    line_width: float | None = None,
) -> str:
    """
    将Manim的Pango标记语言字符串转换为SVG字符串。
    这是文本渲染的核心步骤，使用manimpango库将带有样式的文本转换为可渲染的SVG图形。
    该函数通过`lru_cache`和`cache_on_disk`进行缓存，以避免对相同输入重复计算。

    参数:
        markup_str (str): 包含Pango标记的文本字符串，如"<b>Hello</b>"。
        justify (bool): 是否两端对齐文本。
        indent (float): 首行缩进量。
        alignment (str): 水平对齐方式，可选"LEFT", "CENTER", "RIGHT"。
        line_width (float | None): 文本框的最大宽度（以Manim单位计），None表示无限制。

    返回:
        str: 渲染后的SVG字符串。

    异常:
        ValueError: 如果输入的markup_str格式无效，则抛出异常。
    """
    # 1. 验证标记字符串的有效性
    validate_error = manimpango.MarkupUtils.validate(markup_str)
    if validate_error:
        raise ValueError(
            f"Invalid markup string \"{markup_str}\"\n" + \
            f"{validate_error}"
        )

    # 2. 准备manimpango所需的参数
    # `manimpango`库接口可能仍在开发中，以下代码是为了适配其接口
    alignment_obj = _Alignment(alignment)
    
    # 将Manim的逻辑宽度转换为Pango的像素宽度
    if line_width is None:
        pango_width = -1  # -1表示无宽度限制
    else:
        # FRAME_WIDTH 和 DEFAULT_PIXEL_WIDTH 是Manim的全局常量
        pango_width = line_width / FRAME_WIDTH * DEFAULT_PIXEL_WIDTH

    # 3. 使用manimpango将标记文本渲染为SVG文件
    # 创建一个临时文件路径用于保存SVG
    temp_file = Path(tempfile.gettempdir(), hash_string(markup_str)).with_suffix(".svg")
    
    manimpango.MarkupUtils.text2svg(
        text=markup_str,
        font="",                     # 字体样式已在markup_str中处理
        slant="NORMAL",              # 字体倾斜已在markup_str中处理
        weight="NORMAL",             # 字体粗细已在markup_str中处理
        size=1,                      # 字体大小已在markup_str中处理
        _=0,                         # 占位参数
        disable_liga=False,          # 是否禁用连字
        file_name=str(temp_file),    # 输出SVG文件路径
        START_X=0,                   # SVG画布起始X坐标
        START_Y=0,                   # SVG画布起始Y坐标
        width=DEFAULT_CANVAS_WIDTH,  # SVG画布宽度
        height=DEFAULT_CANVAS_HEIGHT,# SVG画布高度
        justify=justify,             # 是否两端对齐
        indent=indent,               # 首行缩进
        line_spacing=None,           # 行间距已在markup_str中处理
        alignment=alignment_obj,     # 水平对齐方式
        pango_width=pango_width      # Pango文本框宽度
    )

    # 4. 读取并清理
    # 读取临时SVG文件的内容
    result = temp_file.read_text()
    # 删除临时文件
    os.remove(temp_file)
    
    return result


class MarkupText(StringMobject):
    """
    支持Pango标记语言的文本图形类，继承自`StringMobject`。
    核心功能是通过Pango标记（如`<b>`加粗、`<i>`斜体）定义文本样式，并将其渲染为Manim图形对象。
    参考Pango标记文档：https://docs.gtk.org/Pango/pango_markup.html
    """
    # 1. 预定义Pango标记标签与对应样式的映射表
    # 键：Pango标记标签（如"b"）；值：标签对应的样式配置（如字体粗细）
    MARKUP_TAGS = {
        "b": {"font_weight": "bold"},               # 加粗标签
        "big": {"font_size": "larger"},             # 放大标签
        "i": {"font_style": "italic"},              # 斜体标签
        "s": {"strikethrough": "true"},             # 删除线标签
        "sub": {"baseline_shift": "subscript", "font_scale": "subscript"},  # 下标标签
        "sup": {"baseline_shift": "superscript", "font_scale": "superscript"},  # 上标标签
        "small": {"font_size": "smaller"},          # 缩小标签
        "tt": {"font_family": "monospace"},         # 等宽字体标签
        "u": {"underline": "single"},               # 下划线标签
    }

    # 2. 特殊字符转义映射表（避免标记语法冲突）
    # 键：原始特殊字符；值：Pango标记中的转义字符
    MARKUP_ENTITY_DICT = {
        "<": "&lt;",    # 小于号转义
        ">": "&gt;",    # 大于号转义
        "&": "&amp;",   # 和号转义
        "\"": "&quot;", # 双引号转义
        "'": "&apos;"   # 单引号转义
    }

    def __init__(
        self,
        text: str,
        font_size: int = 48,
        height: float | None = None,
        justify: bool = False,
        indent: float = 0,
        alignment: str = "",
        line_width: float | None = None,
        font: str = "",
        slant: str = NORMAL,
        weight: str = NORMAL,
        gradient: Iterable[ManimColor] | None = None,
        line_spacing_height: float | None = None,
        text2color: dict = {},
        text2font: dict = {},
        text2gradient: dict = {},
        text2slant: dict = {},
        text2weight: dict = {},
        # For convenience, one can use shortened names
        # 简写参数（优先级高于完整参数，方便快速配置）
        lsh: float | None = None,  # 行间距高度简写（覆盖line_spacing_height）
        t2c: dict = {},            # 文本-颜色映射简写（覆盖text2color）
        t2f: dict = {},            # 文本-字体映射简写（覆盖text2font）
        t2g: dict = {},            # 文本-渐变映射简写（覆盖text2gradient）
        t2s: dict = {},            # 文本-倾斜映射简写（覆盖text2slant）
        t2w: dict = {},            # 文本-粗细映射简写（覆盖text2weight）
        global_config: dict = {},  # 全局文本样式配置
        local_configs: dict = {},  # 局部文本样式配置
        disable_ligatures: bool = True,  # 是否禁用字体连字（如"fi"连写）
        isolate: Selector = re.compile(r"\w+", re.U),  # 文本片段隔离规则（默认匹配单词）
        **kwargs
    ):
        # 初始化MarkupText对象，处理文本内容、样式配置并调用父类初始化。
        # 1. 读取全局文本配置（Manim的全局配置，如默认字体、对齐方式）
        text_config = manim_config.text
        # 2. 保存基础文本与样式参数
        self.text = text                      # 原始文本（含标记）
        self.font_size = font_size            # 基础字体大小
        self.justify = justify                # 两端对齐开关
        self.indent = indent                  # 首行缩进
        # 对齐方式：优先使用传入值，否则用全局配置
        self.alignment = alignment or text_config.alignment
        self.line_width = line_width          # 文本框最大宽度
        # 字体：优先使用传入值，否则用全局配置
        self.font = font or text_config.font
        self.slant = slant                    # 字体倾斜
        self.weight = weight                  # 字体粗细

        # 3. 处理简写参数（简写优先级高于完整参数）
        self.lsh = line_spacing_height or lsh  # 行间距高度
        self.t2c = text2color or t2c          # 文本-颜色映射
        self.t2f = text2font or t2f          # 文本-字体映射
        self.t2g = text2gradient or t2g      # 文本-渐变映射
        self.t2s = text2slant or t2s          # 文本-倾斜映射
        self.t2w = text2weight or t2w        # 文本-粗细映射

        # 4. 保存其他配置
        self.global_config = global_config    # 全局样式配置
        self.local_configs = local_configs    # 局部样式配置
        self.disable_ligatures = disable_ligatures  # 禁用连字开关
        self.isolate = isolate                # 文本片段隔离规则

        # 5. 调用父类`StringMobject`的初始化方法（处理SVG渲染核心逻辑）
        super().__init__(text, height=height, **kwargs)

        # 6. 处理渐变与颜色配置
        # 警告：当前不支持从SVG解析渐变，需手动调用set_color_by_gradient
        if self.t2g:
            log.warning("""
                Manim currently cannot parse gradient from svg.
                Please set gradient via `set_color_by_gradient`.
            """)
        # 应用全局渐变
        if gradient:
            self.set_color_by_gradient(*gradient)
        # 应用文本-颜色映射
        if self.t2c:
            self.set_color_by_text_to_color_map(self.t2c)

        # 7. 自适应缩放（仅当未指定固定高度时）
        if height is None:
            self.scale(TEXT_MOB_SCALE_FACTOR)

    def get_svg_string_by_content(self, content: str) -> str:
        """
        根据传入的标记文本内容，生成对应的SVG字符串。
        这是连接标记文本渲染与SVG图形的核心方法，通过调用`markup_to_svg`工具函数实现转换。

        参数:
            content (str): 待渲染为SVG的标记文本内容（通常是经过处理的完整Pango标记代码）。

        返回:
            str: 渲染后的SVG字符串，可直接用于后续解析为Manim图形对象。
        """
        # 保存当前处理的内容，以备后续使用
        self.content = content
        # 调用外部标记文本转SVG工具函数，传入当前对象的样式配置
        return markup_to_svg(
            content,
            justify=self.justify,
            indent=self.indent,
            alignment=self.alignment,
            line_width=self.line_width
        )

    # Toolkits

    @staticmethod
    # 对标记文本中的特殊字符进行转义，防止其被解析为标记语法。
    # 例如，将"<"转换为"&lt;"，确保它被显示为普通字符而不是标签的开始。
    def escape_markup_char(substr: str) -> str:
        return MarkupText.MARKUP_ENTITY_DICT.get(substr, substr)

    @staticmethod
    # 将标记文本中已转义的特殊字符还原为原始字符。
    # 例如，将"&lt;"转换回"<"。
    def unescape_markup_char(substr: str) -> str:
        # 创建一个反转义映射字典（原始字典的键值互换）
        return {
            v: k
            for k, v in MarkupText.MARKUP_ENTITY_DICT.items()
        }.get(substr, substr)

    # Parsing

    @staticmethod
    def get_command_matches(string: str) -> list[re.Match]:
        pattern = re.compile(r"""
            (?P<tag>
                <
                (?P<close_slash>/)?
                (?P<tag_name>\w+)\s*
                (?P<attr_list>(?:\w+\s*\=\s*(?P<quot>["']).*?(?P=quot)\s*)*)
                (?P<elision_slash>/)?
                >
            )
            |(?P<passthrough>
                <\?.*?\?>|<!--.*?-->|<!\[CDATA\[.*?\]\]>|<!DOCTYPE.*?>
            )
            |(?P<entity>&(?P<unicode>\#(?P<hex>x)?)?(?P<content>.*?);)
            |(?P<char>[>"'])
        """, flags=re.X | re.S)
        return list(pattern.finditer(string))

    @staticmethod
    def get_command_flag(match_obj: re.Match) -> int:
        if match_obj.group("tag"):
            if match_obj.group("close_slash"):
                return -1
            if not match_obj.group("elision_slash"):
                return 1
        return 0

    @staticmethod
    def replace_for_content(match_obj: re.Match) -> str:
        if match_obj.group("tag"):
            return ""
        if match_obj.group("char"):
            return MarkupText.escape_markup_char(match_obj.group("char"))
        return match_obj.group()

    @staticmethod
    def replace_for_matching(match_obj: re.Match) -> str:
        if match_obj.group("tag") or match_obj.group("passthrough"):
            return ""
        if match_obj.group("entity"):
            if match_obj.group("unicode"):
                base = 10
                if match_obj.group("hex"):
                    base = 16
                return chr(int(match_obj.group("content"), base))
            return MarkupText.unescape_markup_char(match_obj.group("entity"))
        return match_obj.group()

    @staticmethod
    def get_attr_dict_from_command_pair(
        open_command: re.Match, close_command: re.Match
    ) -> dict[str, str] | None:
        pattern = r"""
            (?P<attr_name>\w+)
            \s*\=\s*
            (?P<quot>["'])(?P<attr_val>.*?)(?P=quot)
        """
        tag_name = open_command.group("tag_name")
        if tag_name == "span":
            return {
                match_obj.group("attr_name"): match_obj.group("attr_val")
                for match_obj in re.finditer(
                    pattern, open_command.group("attr_list"), re.S | re.X
                )
            }
        return MarkupText.MARKUP_TAGS.get(tag_name, {})

    # 获取所有通过文本映射（如text2color）和局部配置定义的文本片段及其样式属性。
    # 该方法返回一个列表，其中每个元素是 (Span, 属性字典) 的元组，用于后续应用样式。
    def get_configured_items(self) -> list[tuple[Span, dict[str, str]]]:
        return [
            # 第一部分：处理文本到样式的映射（颜色、字体、倾斜、粗细）
            *(
                (span, {key: val})
                for t2x_dict, key in (
                    (self.t2c, "foreground"),     # 文本-颜色映射 -> Pango的"foreground"属性
                    (self.t2f, "font_family"),    # 文本-字体映射 -> Pango的"font_family"属性
                    (self.t2s, "font_style"),     # 文本-倾斜映射 -> Pango的"font_style"属性
                    (self.t2w, "font_weight")     # 文本-粗细映射 -> Pango的"font_weight"属性
                )
                for selector, val in t2x_dict.items()  # 遍历映射字典中的每个选择器和值
                for span in self.find_spans_by_selector(selector)  # 查找选择器匹配的所有文本片段（Span）
            ),
            # 第二部分：处理局部配置（更灵活的样式定义）
            *(
                (span, local_config)
                for selector, local_config in self.local_configs.items()  # 遍历局部配置字典
                for span in self.find_spans_by_selector(selector)  # 查找选择器匹配的所有文本片段（Span）
            )
        ]

    @staticmethod
    def get_command_string(
        attr_dict: dict[str, str], is_end: bool, label_hex: str | None
    ) -> str:
        if is_end:
            return "</span>"

        if label_hex is not None:
            converted_attr_dict = {"foreground": label_hex}
            for key, val in attr_dict.items():
                if key in (
                    "background", "bgcolor",
                    "underline_color", "overline_color", "strikethrough_color"
                ):
                    converted_attr_dict[key] = "black"
                elif key not in ("foreground", "fgcolor", "color"):
                    converted_attr_dict[key] = val
        else:
            converted_attr_dict = attr_dict.copy()
        attrs_str = " ".join([
            f"{key}='{val}'"
            for key, val in converted_attr_dict.items()
        ])
        return f"<span {attrs_str}>"

    def get_content_prefix_and_suffix(
        self, is_labelled: bool
    ) -> tuple[str, str]:
        """
        生成用于包装标记文本内容的前缀和后缀字符串。
        这些字符串包含全局样式属性（如颜色、字体、大小）和布局设置（如行高），
        并根据Pango版本决定是否支持某些高级属性。

        参数:
            is_labelled (bool): 是否使用带标签的SVG渲染。

        返回:
            tuple[str, str]: (前缀字符串, 后缀字符串)。
        """
        # 1. 构建全局属性字典
        # 这些属性将应用于整个文本块
        global_attr_dict = {
            "foreground": color_to_hex(self.base_color),  # 文本颜色（转为十六进制）
            "font_family": self.font,                     # 字体族
            "font_style": self.slant,                     # 字体样式（如NORMAL, ITALIC）
            "font_weight": self.weight,                   # 字体粗细（如NORMAL, BOLD）
            # 字体大小：Pango使用"十二点活字"单位，需要特殊转换
            "font_size": str(round(self.font_size * 1024)),
        }
        # `line_height` attribute is supported since Pango 1.50.
        # 2. 处理行高（line_height）属性（Pango 1.50+支持）
        pango_version = manimpango.pango_version()
        if tuple(map(int, pango_version.split("."))) < (1, 50):
            # 如果Pango版本低于1.50，不支持line_height属性
            if self.lsh is not None:
                log.warning(
                    "Pango version %s found (< 1.50), "
                    "unable to set `line_height` attribute",
                    pango_version
                )
        else:
            # 如果Pango版本足够新，计算并设置行高
            line_spacing_scale = self.lsh or DEFAULT_LINE_SPACING_SCALE
            global_attr_dict["line_height"] = str(
                ((line_spacing_scale) + 1) * 0.6
            )
        # 3. 处理连字（ligatures）禁用
        if self.disable_ligatures:
            # 通过字体特性（font_features）禁用各种连字
            global_attr_dict["font_features"] = "liga=0,dlig=0,clig=0,hlig=0"

        # 4. 应用全局配置覆盖
        # 如果有额外的全局配置，用它们覆盖默认值
        global_attr_dict.update(self.global_config)
        # 5. 生成前缀和后缀命令字符串
        # 使用get_command_string方法将属性字典转换为Pango标记语言的命令
        return tuple(
            self.get_command_string(
                global_attr_dict,
                is_end=is_end,
                # 如果是带标签的渲染，提供一个占位标签（颜色为0，即黑色）
                label_hex=int_to_hex(0) if is_labelled else None
            )
            for is_end in (False, True)  # 分别生成前缀（is_end=False）和后缀（is_end=True）
        )

    # Method alias

    # 根据选择器获取文本片段对应的图形部分。
    # 这是 `select_parts` 方法的别名，使API更符合文本操作习惯。
    def get_parts_by_text(self, selector: Selector) -> VGroup:
        return self.select_parts(selector)

    # 根据选择器获取指定索引的文本片段对应的图形部分。
    # 这是 `select_part` 方法的别名。
    def get_part_by_text(self, selector: Selector, **kwargs) -> VGroup:
        return self.select_part(selector, **kwargs)

    # 根据选择器为文本片段设置颜色。
    # 这是 `set_parts_color` 方法的别名。
    def set_color_by_text(self, selector: Selector, color: ManimColor):
        return self.set_parts_color(selector, color)

    # # 根据一个字典批量为多个文本片段设置颜色。
    # 这是 `set_parts_color_by_dict` 方法的别名。
    # color_map (dict[Selector, ManimColor]): 一个字典，键是选择器，值是对应的颜色。
    def set_color_by_text_to_color_map(
        self, color_map: dict[Selector, ManimColor]
    ):
        return self.set_parts_color_by_dict(color_map)

    # 获取原始的文本字符串。
    # 这是 `get_string` 方法的别名。
    def get_text(self) -> str:
        return self.get_string()


class Text(MarkupText):
    def __init__(
        self,
        text: str,
        # For backward compatibility
        isolate: Selector = (re.compile(r"\w+", re.U), re.compile(r"\S+", re.U)),
        use_labelled_svg: bool = True,
        path_string_config: dict = dict(
            use_simple_quadratic_approx=True,
        ),
        **kwargs
    ):
        super().__init__(
            text,
            isolate=isolate,
            use_labelled_svg=use_labelled_svg,
            path_string_config=path_string_config,
            **kwargs
        )

    @staticmethod
    def get_command_matches(string: str) -> list[re.Match]:
        """
        静态方法：解析文本字符串，提取其中需要转义的特殊字符（<、>、&、"、'），
        返回这些字符的正则匹配结果列表。核心是识别可能与Pango标记语法冲突的字符，为后续转义做准备。
        """
        # 正则表达式模式：匹配Pango标记中需要转义的5个特殊字符
        pattern = re.compile(r"""[<>&"']""")
        # 查找所有匹配的字符，返回匹配结果列表
        return list(pattern.finditer(string))

    @staticmethod
    # 静态方法：判断特殊字符匹配结果的“命令类型标记”。
    def get_command_flag(match_obj: re.Match) -> int:
        return 0

    @staticmethod
    # 静态方法：将特殊字符匹配结果替换为适合生成SVG内容的“转义后字符”。
    # 目的是避免特殊字符被Pango解析为标记语法，确保其作为普通文本显示。
    def replace_for_content(match_obj: re.Match) -> str:
        return Text.escape_markup_char(match_obj.group())

    @staticmethod
    # 静态方法：将特殊字符匹配结果替换为适合“文本片段匹配”的字符串。
    # 由于匹配逻辑需要基于原始字符（而非转义后字符），因此直接返回原始匹配内容。
    def replace_for_matching(match_obj: re.Match) -> str:
        return match_obj.group()


class Code(MarkupText):
    """
    代码高亮显示类：继承自MarkupText，利用Pygments库实现代码语法高亮，
    支持指定编程语言、代码风格、字体等，最终生成带Pango标记的高亮文本。
    """
    def __init__(
        self,
        code: str,  # 待高亮显示的代码字符串（如Python代码文本）
        font: str = "Consolas",  # 代码显示字体，默认等宽字体Consolas（适合代码）
        font_size: int = 24,  # 字体大小，默认24
        lsh: float = 1.0,  # 行间距系数（Line Spacing Height），默认1.0（正常间距）
        fill_color: ManimColor = None,  # 文本填充色，默认None（使用代码风格默认颜色）
        stroke_color: ManimColor = None,  # 文本描边色，默认None（无描边）
        language: str = "python",  # 代码所属编程语言，默认Python（用于语法解析）
        # Visit https://pygments.org/demo/ to have a preview of more styles.（访问该链接预览更多代码风格）
        code_style: str = "monokai",  # 代码高亮风格，默认monokai（深色高对比度风格）
        **kwargs
    ):
        # 根据指定编程语言获取Pygments的语法解析器（lexer）
        lexer = pygments.lexers.get_lexer_by_name(language)
        # 创建Pygments的PangoMarkup格式器，用于生成带标记的高亮文本（适配Manim的MarkupText）
        formatter = pygments.formatters.PangoMarkupFormatter(
            style=code_style
        )
        # 对输入代码进行语法高亮处理，生成Pango标记字符串
        markup = pygments.highlight(code, lexer, formatter)
        # 移除标记中的<tt>和</tt>标签（PangoMarkup格式器可能生成，Manim无需该标签）
        markup = re.sub(r"</?tt>", "", markup)
        # 调用父类MarkupText的初始化方法，传入处理后的高亮标记文本及样式参数
        super().__init__(
            markup,
            font=font,
            font_size=font_size,
            lsh=lsh,
            stroke_color=stroke_color,
            fill_color=fill_color,
            **kwargs
        )


@contextmanager
def register_font(font_file: str | Path):
    """
    临时注册字体文件的上下文管理器：在with语句块内，将指定字体文件添加到Pango的字体搜索路径，
    块结束后自动注销该字体，避免全局字体环境污染。
    
    搜索字体文件的顺序：
    1. 绝对路径（直接查找指定的完整路径）
    2. 下载目录（若绝对路径不存在，尝试在下载目录中查找）

    参数
    ----------
    font_file : str | Path
        待注册的字体文件路径（如"./fonts/CustomFont.ttf"）

    示例
    --------
    使用with语句临时注册字体，在块内可使用该字体创建文本：
    .. code-block:: python
        with register_font("path/to/font_file.ttf"):
           a = Text("Hello", font="Custom Font Name")  # "Custom Font Name"为字体实际名称

    异常
    ------
    FileNotFoundError:
        若指定的字体文件在所有搜索路径中均不存在时抛出
    AttributeError:
        在macOS系统上使用不兼容的ManimPango版本（低于v0.2.3）时抛出

    说明
    -----
    该方法注册的字体同样适用于CairoText类。
    .. important ::
        macOS系统需确保ManimPango版本≥v0.2.3，旧版本使用该方法会抛出AttributeError。
    """

    # 将输入的字体路径解析为绝对路径（统一路径格式，避免相对路径问题）
    file_path = Path(font_file).resolve()
    # 检查字体文件是否存在，不存在则抛出文件未找到异常
    if not file_path.exists():
        error = f"Can't find {font_file}."  # 错误信息：提示找不到指定字体文件
        raise FileNotFoundError(error)
    try:
        # 注册字体文件：调用manimpango的register_font方法，断言注册成功（失败则触发AssertionError）
        assert manimpango.register_font(str(file_path))
        yield  # 进入上下文管理器的代码块（with语句内的代码在此处执行）
    finally:
        # 无论块内代码是否抛出异常，最终都会注销该字体，恢复字体环境
        manimpango.unregister_font(str(file_path))
