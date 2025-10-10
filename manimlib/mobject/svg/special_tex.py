# 导入未来版本的注解特性，支持更灵活的类型提示语法
from __future__ import annotations

# 从manim库导入常用常量：
# - 间距相关：MED_SMALL_BUFF（中小编号间距）、MED_LARGE_BUFF（中大号间距）、SMALL_BUFF（小编号间距）
# - 颜色相关：DEFAULT_MOBJECT_COLOR（默认图形颜色）、GREY_C（C色阶灰色）
# - 方向向量：DOWN（下）、LEFT（左）、RIGHT（右）、UP（上）
# - 框架相关：FRAME_WIDTH（场景框架宽度）
from manimlib.constants import MED_SMALL_BUFF, DEFAULT_MOBJECT_COLOR, GREY_C
from manimlib.constants import DOWN, LEFT, RIGHT, UP
from manimlib.constants import FRAME_WIDTH
from manimlib.constants import MED_LARGE_BUFF, SMALL_BUFF

# 从manim库导入常用图形类：
# - Line：线段图形类（用于绘制直线）
# - VGroup：向量图形组合类（用于管理多个图形对象）
# - TexText：LaTeX文本渲染类（用于渲染普通文本，非公式模式）
from manimlib.mobject.geometry import Line
from manimlib.mobject.types.vectorized_mobject import VGroup
from manimlib.mobject.svg.tex_mobject import TexText


# 导入类型检查相关模块（仅在类型检查时生效，不影响代码运行）
from typing import TYPE_CHECKING

# 若处于类型检查模式，导入需要的类型注解（避免运行时依赖）
if TYPE_CHECKING:
    from manimlib.typing import ManimColor, Vect3  # Manim颜色类型、3D向量类型


# 定义 BulletedList 类，继承自 VGroup，用于创建带项目符号的列表
class BulletedList(VGroup):
    def __init__(
        self,
        *items: str,                 # 可变参数，传入一个或多个列表项的字符串
        buff: float = MED_LARGE_BUFF,  # 列表项之间的垂直间距，默认使用中等偏大的间距
        aligned_edge: Vect3 = LEFT,   # 列表项的对齐边缘，默认左对齐
        **kwargs                     # 传递给 TexText 的其他参数（如颜色、字体大小等）
    ):
        # 1. 构建 LaTeX 列表代码
        # 为每个列表项字符串添加 \item 命令
        labelled_content = [R"\item " + item for item in items]
        # 将所有 \item 项组合成一个完整的 LaTeX itemize 环境字符串
        tex_string = "\n".join([
            R"\begin{itemize}",   # 列表开始
            *labelled_content,   # 所有带 \item 的列表项
            R"\end{itemize}"     # 列表结束
        ])

        # 2. 创建并隔离列表项
        # 使用 TexText 渲染 LaTeX 列表，并将每个带 \item 的列表项隔离为独立的子对象
        tex_text = TexText(tex_string, isolate=labelled_content, **kwargs)
        # 提取并生成每个被隔离的列表项（包含项目符号和文本）
        lines = (tex_text.select_part(part) for part in labelled_content)

        # 3. 组合并排列列表项
        # 调用父类 VGroup 的构造函数，将所有列表项作为子对象添加进来
        super().__init__(*lines)

        # 4. 按垂直方向排列
        # 将所有列表项沿垂直方向（DOWN）排列，设置间距和对齐方式
        self.arrange(DOWN, buff=buff, aligned_edge=aligned_edge)

def fade_all_but(self, index: int, opacity: float = 0.25, scale_factor=0.7) -> None:
    """
    将列表中除指定索引项外的所有项变暗并缩小，以突出显示指定项。

    参数:
        index (int): 要突出显示的列表项的索引。
        opacity (float): 非突出显示项的不透明度。
        scale_factor (float): 非突出显示项的缩放比例。
    """
    # 1. 计算最大项目符号高度
    # 遍历所有子对象（列表项），找到项目符号（子对象的第一个元素）的最大高度
    max_dot_height = max([item[0].get_height() for item in self.submobjects])

    # 2. 遍历并设置每个列表项的样式
    for i, part in enumerate(self.submobjects):
        # 计算目标项目符号高度：指定项使用最大高度，其他项按比例缩小
        trg_dot_height = (1.0 if i == index else scale_factor) * max_dot_height

        # 设置填充不透明度：指定项完全不透明，其他项半透明
        part.set_fill(opacity=(1.0 if i == index else opacity))

        # 缩放列表项，使其项目符号达到目标高度，并保持左边缘对齐
        part.scale(trg_dot_height / part[0].get_height(), about_edge=LEFT)


class TexTextFromPresetString(TexText):
    """
    一个从预设字符串创建的 TexText 子类。
    通过继承并设置 `tex` 和 `default_color` 类属性，可以快速创建特定文本的实例。
    """
    tex: str = ""  # 预设的 LaTeX 文本字符串，子类应覆盖此属性
    default_color: ManimColor = DEFAULT_MOBJECT_COLOR  # 预设的默认颜色

    def __init__(self, **kwargs):
        """
        初始化方法。

        参数:
            **kwargs: 传递给父类 TexText 的参数。如果提供了 `color`,
                      则覆盖 `default_color`；否则使用 `default_color`。
        """
        super().__init__(
            self.tex,  # 使用预设的 tex 字符串
            color=kwargs.pop("color", self.default_color),  # 处理颜色参数
            **kwargs   # 传递其他所有参数
        )


# 定义 Title 类，继承自 TexText，用于创建带下划线的标题文本
class Title(TexText):
    def __init__(
        self,
        *text_parts: str,                # 可变参数，标题文本内容（可多段）
        font_size: int = 72,             # 标题字体大小，默认72
        include_underline: bool = True,  # 是否包含下划线，默认True
        underline_width: float = FRAME_WIDTH - 2,  # 下划线宽度，默认略小于场景宽度
        match_underline_width_to_text: bool = False,  # 是否让下划线宽度匹配文本宽度，默认False
        underline_buff: float = SMALL_BUFF,  # 文本与下划线之间的垂直间距，默认小间距
        underline_style: dict = dict(stroke_width=2, stroke_color=GREY_C),  # 下划线样式字典
        **kwargs                         # 传递给父类 TexText 的其他参数
    ):
        # 1. 创建并定位标题文本
        # 调用父类 TexText 构造函数创建标题文本对象
        super().__init__(*text_parts, font_size=font_size, **kwargs)
        # 将标题移动到场景顶部边缘，并保留一定缓冲距离
        self.to_edge(UP, buff=MED_SMALL_BUFF)

        # 2. 创建并配置下划线（如果需要）
        if include_underline:
            # 创建一条水平线作为下划线
            underline = Line(LEFT, RIGHT, **underline_style)
            # 将下划线定位到标题下方，保持指定间距
            underline.next_to(self, DOWN, buff=underline_buff)

            # 根据配置调整下划线宽度
            if match_underline_width_to_text:
                # 让下划线宽度与标题文本完全一致
                underline.match_width(self)
            else:
                # 使用预设的固定宽度
                underline.set_width(underline_width)

            # 将下划线添加到标题组合中，并作为属性保存以便后续访问
            self.add(underline)
            self.underline = underline
