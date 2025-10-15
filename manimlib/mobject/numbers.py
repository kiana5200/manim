# 从__future__导入annotations，用于支持延迟类型注解评估（Python 3.7+特性）
from __future__ import annotations
# 从functools导入lru_cache，用于函数结果的缓存（记忆化）
from functools import lru_cache

# 导入numpy库并简写为np，用于数值计算
import numpy as np

# 从manimlib.constants导入方向常量：下、左、右、上
from manimlib.constants import DOWN, LEFT, RIGHT, UP
# 从manimlib.constants导入默认的Mobject颜色常量
from manimlib.constants import DEFAULT_MOBJECT_COLOR
# 从manimlib.mobject.svg.tex_mobject导入Tex类，用于创建LaTeX文本对象
from manimlib.mobject.svg.tex_mobject import Tex
# 从manimlib.mobject.svg.text_mobject导入Text类，用于创建普通文本对象
from manimlib.mobject.svg.text_mobject import Text
# 从manimlib.mobject.types.vectorized_mobject导入VMobject类，矢量图形基类
from manimlib.mobject.types.vectorized_mobject import VMobject
# 从manimlib.utils.paths导入straight_path函数，用于生成直线路径
from manimlib.utils.paths import straight_path
# 从manimlib.utils.bezier导入interpolate函数，用于贝塞尔插值计算
from manimlib.utils.bezier import interpolate

# 从typing模块导入TYPE_CHECKING常量，用于条件性导入类型提示
from typing import TYPE_CHECKING

# 如果处于类型检查阶段（非运行时）
if TYPE_CHECKING:
    # 从typing模块导入需要的类型提示：TypeVar（类型变量）、Callable（可调用对象）
    from typing import TypeVar, Callable
    # 从manimlib.mobject.mobject导入Mobject类，所有Manim对象的基类
    from manimlib.mobject.mobject import Mobject
    # 从manimlib.typing导入特定的类型提示：Manim颜色类型、三维向量、Self类型
    from manimlib.typing import ManimColor, Vect3, Self

    # 定义类型变量T，约束为VMobject的子类
    T = TypeVar("T", bound=VMobject)


# 使用lru_cache装饰器缓存函数结果，避免重复计算相同输入的结果
@lru_cache()
def char_to_cahced_mob(char: str,** text_config):
    # 如果字符中包含反斜杠（通常是LaTeX命令）
    if "\\" in char:
        # 返回使用Tex创建的LaTeX文本对象，传入配置参数
        return Tex(char,** text_config)
    else:
        # 否则返回使用Text创建的普通文本对象，传入配置参数
        return Text(char,** text_config)


class DecimalNumber(VMobject):
    """
    十进制数字类，继承自VMobject，用于创建带格式的数字文本对象
    """
    def __init__(
        self,
        number: float | complex = 0,  # 要显示的数字，默认为0
        color: ManimColor = DEFAULT_MOBJECT_COLOR,  # 数字颜色
        stroke_width: float = 0,  # 描边宽度
        fill_opacity: float = 1.0,  # 填充不透明度
        fill_border_width: float = 0.5,  # 填充边框宽度
        num_decimal_places: int = 2,  # 小数位数，默认为2
        min_total_width: Optional[int] = 0,  # 最小总宽度，可选
        include_sign: bool = False,  # 是否包含符号
        group_with_commas: bool = True,  # 是否用逗号分组
        digit_buff_per_font_unit: float = 0.001,  # 每个字体单位的数字间距
        show_ellipsis: bool = False,  # 是否显示省略号
        unit: str | None = None,  # 单位，默认为None
        include_background_rectangle: bool = False,  # 是否包含背景矩形
        edge_to_fix: Vect3 = LEFT,  # 固定的边
        font_size: float = 48,  # 字体大小
        text_config: dict = dict(),  # 文本配置，不在这里传递font_size
        **kwargs
    ):
        # 初始化实例变量
        self.num_decimal_places = num_decimal_places
        self.include_sign = include_sign
        self.group_with_commas = group_with_commas
        self.min_total_width = min_total_width
        self.digit_buff_per_font_unit = digit_buff_per_font_unit
        self.show_ellipsis = show_ellipsis
        self.unit = unit
        self.include_background_rectangle = include_background_rectangle
        self.edge_to_fix = edge_to_fix
        self.font_size = font_size
        self.text_config = dict(text_config)

        # 调用父类VMobject的初始化方法
        super().__init__(
            color=color,
            stroke_width=stroke_width,
            fill_opacity=fill_opacity,
            fill_border_width=fill_border_width,** kwargs
        )

        # 从数字设置子对象
        self.set_submobjects_from_number(number)
        # 初始化颜色
        self.init_colors()

    def set_submobjects_from_number(self, number: float | complex) -> None:
        """从数字设置子对象"""
        # 保存数字
        self.number = number
        # 获取数字的字符串表示
        self.num_string = self.get_num_string(number)

        # 创建子对象模板列表
        submob_templates = list(map(self.char_to_mob, self.num_string))
        # 如果需要显示省略号，添加省略号
        if self.show_ellipsis:
            dots = self.char_to_mob("...")
            dots.arrange(RIGHT, buff=2 * dots[0].get_width())
            submob_templates.append(dots)
        # 如果有单位，添加单位
        if self.unit is not None:
            submob_templates.append(self.char_to_mob(self.unit))

        # 设置内部子对象
        font_size = self.get_font_size()
        # 如果子对象数量与现有子对象数量相同，直接替换
        if len(submob_templates) == len(self.submobjects):
            for sm, smt in zip(self.submobjects, submob_templates):
                sm.become(smt)
                sm.scale(font_size / smt.font_size)
        else:
            # 否则创建新的子对象
            self.set_submobjects([
                smt.copy().scale(font_size / smt.font_size)
                for smt in submob_templates
            ])

        # 计算数字间距并排列子对象
        digit_buff = self.digit_buff_per_font_unit * font_size
        self.arrange(RIGHT, buff=digit_buff, aligned_edge=DOWN)

        # 处理特殊字符的对齐
        for i, c in enumerate(self.num_string):
            if c == "–" and len(self.num_string) > i + 1:
                self[i].align_to(self[i + 1], UP)
                self[i].shift(self[i + 1].get_height() * DOWN / 2)
            elif c == ",":
                self[i].shift(self[i].get_height() * DOWN / 2)
        if self.unit and self.unit.startswith("^"):
            self[-1].align_to(self, UP)

        # 如果需要，添加背景矩形
        if self.include_background_rectangle:
            self.add_background_rectangle()

    def get_num_string(self, number: float | complex) -> str:
        """获取数字的字符串表示"""
        if isinstance(number, complex):
            formatter = self.get_complex_formatter()
        else:
            formatter = self.get_formatter()
        # 如果小数位数为0且是浮点数，转换为整数
        if self.num_decimal_places == 0 and isinstance(number, float):
            number = int(number)
        num_string = formatter.format(number)

        # 处理四舍五入后为0的负数
        rounded_num = np.round(number, self.num_decimal_places)
        if num_string.startswith("-") and rounded_num == 0:
            if self.include_sign:
                num_string = "+" + num_string[1:]
            else:
                num_string = num_string[1:]
        # 替换负号为长破折号
        num_string = num_string.replace("-", "–")
        return num_string

    def char_to_mob(self, char: str) -> Text:
        """将字符转换为文本对象"""
        return char_to_cahced_mob(char,** self.text_config)

    def interpolate(
        self,
        mobject1: Mobject,
        mobject2: Mobject,
        alpha: float,
        path_func: Callable[[np.ndarray, np.ndarray, float], np.ndarray] = straight_path
    ) -> Self:
        """插值方法，用于动画过渡"""
        super().interpolate(mobject1, mobject2, alpha, path_func)
        # 处理字体大小的插值
        if hasattr(mobject1, "font_size") and hasattr(mobject2, "font_size"):
            self.font_size = interpolate(mobject1.font_size, mobject2.font_size, alpha)
        return self

    def get_font_size(self) -> float:
        """获取字体大小"""
        return self.font_size

    def get_formatter(self, **kwargs) -> str:
        """
        获取格式化字符串，配置基于实例属性，可通过关键字参数覆盖
        相关关键字：
        - include_sign
        - group_with_commas
        - num_decimal_places
        - field_name
        """
        config = dict([
            (attr, getattr(self, attr))
            for attr in [
                "include_sign",
                "group_with_commas",
                "num_decimal_places",
                "min_total_width",
            ]
        ])
        config.update(kwargs)
        ndp = config["num_decimal_places"]
        return "".join([
            "{",
            config.get("field_name", ""),
            ":",
            "+" if config["include_sign"] else "",
            "0" + str(config.get("min_total_width", "")) if config.get("min_total_width") else "",
            "," if config["group_with_commas"] else "",
            f".{ndp}f" if ndp > 0 else "d",
            "}",
        ])

    def get_complex_formatter(self, **kwargs) -> str:
        """获取复数的格式化字符串"""
        return "".join([
            self.get_formatter(field_name="0.real"),
            self.get_formatter(field_name="0.imag", include_sign=True),
            "i"
        ])

    def get_tex(self):
        """获取LaTeX表示"""
        return self.num_string

    def set_value(self, number: float | complex) -> Self:
        """设置数字的值"""
        move_to_point = self.get_edge_center(self.edge_to_fix)
        style = self.family_members_with_points()[0].get_style()
        self.set_submobjects_from_number(number)
        self.move_to(move_to_point, self.edge_to_fix)
        self.set_style(**style)
        for submob in self.get_family():
            submob.uniforms.update(self.uniforms)
        return self

    def _handle_scale_side_effects(self, scale_factor: float) -> Self:
        """处理缩放的副作用"""
        self.font_size *= scale_factor
        return self

    def get_value(self) -> float | complex:
        """获取数字的值"""
        return self.number

    def increment_value(self, delta_t: float | complex = 1) -> Self:
        """增加数字的值"""
        self.set_value(self.get_value() + delta_t)
        return self

class Integer(DecimalNumber):
    def __init__(
        self,
        number: int = 0,
        num_decimal_places: int = 0,
        **kwargs,
    ):
        super().__init__(number, num_decimal_places=num_decimal_places, **kwargs)

    def get_value(self) -> int:
        return int(np.round(super().get_value()))
