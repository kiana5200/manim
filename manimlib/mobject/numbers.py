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
    # 定义DecimalNumber类，继承自VMobject
    def __init__(
        self,
        number: float | complex = 0,  # 要显示的数字，默认为0，可为浮点数或复数
        color: ManimColor = DEFAULT_MOBJECT_COLOR,  # 数字颜色，默认使用DEFAULT_MOBJECT_COLOR
        stroke_width: float = 0,  # 描边宽度，默认为0
        fill_opacity: float = 1.0,  # 填充不透明度，默认为1.0
        fill_border_width: float = 0.5,  # 填充边框宽度，默认为0.5
        num_decimal_places: int = 2,  # 小数位数，默认为2位
        min_total_width: Optional[int] = 0,  # 最小总宽度，可选，默认为0
        include_sign: bool = False,  # 是否包含符号，默认为False
        group_with_commas: bool = True,  # 是否用逗号分组，默认为True
        digit_buff_per_font_unit: float = 0.001,  # 每个字体单位的数字间距，默认为0.001
        show_ellipsis: bool = False,  # 是否显示省略号，默认为False
        unit: str | None = None,  # 单位，默认为None
        include_background_rectangle: bool = False,  # 是否包含背景矩形，默认为False
        edge_to_fix: Vect3 = LEFT,  # 固定的边，默认为LEFT方向
        font_size: float = 48,  # 字体大小，默认为48
        text_config: dict = dict(),  # 文本配置字典，默认为空字典
        **kwargs
    ):
        self.num_decimal_places = num_decimal_places  # 存储小数位数
        self.include_sign = include_sign  # 存储是否包含符号的布尔值
        self.group_with_commas = group_with_commas  # 存储是否用逗号分组的布尔值
        self.min_total_width = min_total_width  # 存储最小总宽度
        self.digit_buff_per_font_unit = digit_buff_per_font_unit  # 存储每个字体单位的数字间距
        self.show_ellipsis = show_ellipsis  # 存储是否显示省略号的布尔值
        self.unit = unit  # 存储单位
        self.include_background_rectangle = include_background_rectangle  # 存储是否包含背景矩形的布尔值
        self.edge_to_fix = edge_to_fix  # 存储固定的边
        self.font_size = font_size  # 存储字体大小
        self.text_config = dict(text_config)  # 存储文本配置字典

        super().__init__(  # 调用父类VMobject的初始化方法
            color=color,
            stroke_width=stroke_width,
            fill_opacity=fill_opacity,
            fill_border_width=fill_border_width,** kwargs
        )

        self.set_submobjects_from_number(number)  # 从数字设置子对象
        self.init_colors()  # 初始化颜色

    def set_submobjects_from_number(self, number: float | complex) -> None:
        # 定义从数字设置子对象的方法
        self.number = number  # 存储传入的数字
        self.num_string = self.get_num_string(number)  # 获取数字的字符串表示

        submob_templates = list(map(self.char_to_mob, self.num_string))  # 将数字字符串的每个字符转换为子对象模板
        if self.show_ellipsis:  # 如果需要显示省略号
            dots = self.char_to_mob("...")  # 创建省略号子对象
            dots.arrange(RIGHT, buff=2 * dots[0].get_width())  # 排列省略号子对象
            submob_templates.append(dots)  # 将省略号添加到子对象模板列表
        if self.unit is not None:  # 如果有单位
            submob_templates.append(self.char_to_mob(self.unit))  # 将单位转换为子对象并添加到列表

        font_size = self.get_font_size()  # 获取字体大小
        if len(submob_templates) == len(self.submobjects):  # 如果子对象模板数量与现有子对象数量相同
            for sm, smt in zip(self.submobjects, submob_templates):  # 遍历现有子对象和模板
                sm.become(smt)  # 让现有子对象变成模板子对象
                sm.scale(font_size / smt.font_size)  # 按字体大小比例缩放
        else:  # 如果数量不同
            self.set_submobjects([  # 设置新的子对象列表
                smt.copy().scale(font_size / smt.font_size)  # 复制模板并按比例缩放
                for smt in submob_templates
            ])

        digit_buff = self.digit_buff_per_font_unit * font_size  # 计算数字间距
        self.arrange(RIGHT, buff=digit_buff, aligned_edge=DOWN)  # 排列子对象

        for i, c in enumerate(self.num_string):  # 遍历数字字符串的每个字符和索引
            if c == "–" and len(self.num_string) > i + 1:  # 如果字符是长破折号且不是最后一个字符
                self[i].align_to(self[i + 1], UP)  # 与下一个字符顶部对齐
                self[i].shift(self[i + 1].get_height() * DOWN / 2)  # 向下移动半个高度
            elif c == ",":  # 如果字符是逗号
                self[i].shift(self[i].get_height() * DOWN / 2)  # 向下移动半个高度
        if self.unit and self.unit.startswith("^"):  # 如果有单位且单位以^开头
            self[-1].align_to(self, UP)  # 最后一个子对象（单位）与整体顶部对齐

        if self.include_background_rectangle:  # 如果需要包含背景矩形
            self.add_background_rectangle()  # 添加背景矩形

    def get_num_string(self, number: float | complex) -> str:
        # 定义获取数字字符串表示的方法
        if isinstance(number, complex):  # 如果是复数
            formatter = self.get_complex_formatter()  # 获取复数格式化器
        else:  # 否则是浮点数或整数
            formatter = self.get_formatter()  # 获取普通格式化器
        if self.num_decimal_places == 0 and isinstance(number, float):  # 如果小数位数为0且是浮点数
            number = int(number)  # 转换为整数
        num_string = formatter.format(number)  # 格式化数字为字符串

        rounded_num = np.round(number, self.num_decimal_places)  # 对数字进行四舍五入
        if num_string.startswith("-") and rounded_num == 0:  # 如果字符串以负号开头且四舍五入后为0
            if self.include_sign:  # 如果需要包含符号
                num_string = "+" + num_string[1:]  # 替换负号为正号
            else:  # 否则
                num_string = num_string[1:]  # 去掉负号
        num_string = num_string.replace("-", "–")  # 替换负号为长破折号
        return num_string  # 返回处理后的数字字符串

    def char_to_mob(self, char: str) -> Text:
        # 定义将字符转换为文本对象的方法
        return char_to_cahced_mob(char, **self.text_config)  # 调用缓存的字符转文本对象函数

    def interpolate(
        self,
        mobject1: Mobject,
        mobject2: Mobject,
        alpha: float,
        path_func: Callable[[np.ndarray, np.ndarray, float], np.ndarray] = straight_path
    ) -> Self:
        # 定义插值方法，用于动画过渡
        super().interpolate(mobject1, mobject2, alpha, path_func)  # 调用父类的插值方法
        if hasattr(mobject1, "font_size") and hasattr(mobject2, "font_size"):  # 如果两个对象都有字体大小属性
            self.font_size = interpolate(mobject1.font_size, mobject2.font_size, alpha)  # 插值计算字体大小
        return self  # 返回自身

    def get_font_size(self) -> float:
        # 定义获取字体大小的方法
        return self.font_size  # 返回字体大小

    def get_formatter(self, **kwargs) -> str:
        # 定义获取格式化字符串的方法，可通过关键字参数覆盖配置
        config = dict([  # 构建配置字典
            (attr, getattr(self, attr))
            for attr in [
                "include_sign",
                "group_with_commas",
                "num_decimal_places",
                "min_total_width",
            ]
        ])
        config.update(kwargs)  # 更新配置字典
        ndp = config["num_decimal_places"]  # 获取小数位数
        return "".join([  # 拼接格式化字符串
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
        # 定义获取复数格式化字符串的方法
        return "".join([  # 拼接复数格式化字符串
            self.get_formatter(field_name="0.real"),  # 实部格式化
            self.get_formatter(field_name="0.imag", include_sign=True),  # 虚部格式化（包含符号）
            "i"  # 虚数单位
        ])

    def get_tex(self):
        # 定义获取LaTeX表示的方法
        return self.num_string  # 返回数字字符串

    def set_value(self, number: float | complex) -> Self:
        # 定义设置数字值的方法
        move_to_point = self.get_edge_center(self.edge_to_fix)  # 获取要固定边的中心
        style = self.family_members_with_points()[0].get_style()  # 获取样式
        self.set_submobjects_from_number(number)  # 从新数字设置子对象
        self.move_to(move_to_point, self.edge_to_fix)  # 移动到固定边的中心
        self.set_style(**style)  # 设置样式
        for submob in self.get_family():  # 遍历所有相关子对象
            submob.uniforms.update(self.uniforms)  # 更新统一属性
        return self  # 返回自身

    def _handle_scale_side_effects(self, scale_factor: float) -> Self:
        # 定义处理缩放副作用的方法
        self.font_size *= scale_factor  # 按缩放因子调整字体大小
        return self  # 返回自身

    def get_value(self) -> float | complex:
        # 定义获取数字值的方法
        return self.number  # 返回存储的数字

    def increment_value(self, delta_t: float | complex = 1) -> Self:
        # 定义增加数字值的方法，默认增加1
        self.set_value(self.get_value() + delta_t)  # 设置为当前值加增量后的值
        return self  # 返回自身

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
