# 从__future__导入annotations，支持在类型提示中使用尚未定义的类
from __future__ import annotations

# 导入numpy用于数值计算和矩阵操作
import numpy as np

# 从manimlib.constants导入方向常量和原点常量
from manimlib.constants import DOWN, LEFT, RIGHT, ORIGIN
# 从manimlib.constants导入角度单位常量（度）
from manimlib.constants import DEG
# 从manimlib.mobject.numbers导入DecimalNumber用于显示数字
from manimlib.mobject.numbers import DecimalNumber
# 从manimlib.mobject.svg.tex_mobject导入Tex用于显示LaTeX公式
from manimlib.mobject.svg.tex_mobject import Tex
# 从manimlib.mobject.types.vectorized_mobject导入VGroup用于组合多个可移动对象
from manimlib.mobject.types.vectorized_mobject import VGroup
# 从manimlib.mobject.types.vectorized_mobject导入VMobject基类
from manimlib.mobject.types.vectorized_mobject import VMobject

# 导入类型检查相关模块
from typing import TYPE_CHECKING

# 如果是类型检查阶段（不执行实际代码）
if TYPE_CHECKING:
    # 导入所需的类型提示
    from typing import Sequence, Union, Optional
    from manimlib.typing import ManimColor, Vect3, VectNArray, Self

    # 定义矩阵类型的类型别名
    # 字符串矩阵类型：由字符串序列或字符串数组组成
    StringMatrixType = Union[Sequence[Sequence[str]], np.ndarray[int, np.dtype[np.str_]]]
    # 浮点矩阵类型：由浮点数序列或数值数组组成
    FloatMatrixType = Union[Sequence[Sequence[float]], VectNArray]
    # VMobject矩阵类型：由VMobject对象组成的二维序列
    VMobjectMatrixType = Sequence[Sequence[VMobject]]
    # 通用矩阵类型：可以是上述任何一种矩阵类型
    GenericMatrixType = Union[FloatMatrixType, StringMatrixType, VMobjectMatrixType]


class Matrix(VMobject):
    """矩阵类，继承自VMobject，用于在Manim中创建和显示各种类型的矩阵"""
    
    def __init__(
        self,
        matrix: GenericMatrixType,  # 矩阵数据，可以是数字矩阵、字符串矩阵或VMobject矩阵
        v_buff: float = 0.5,  # 垂直方向的缓冲距离（行与行之间的间距）
        h_buff: float = 0.5,  # 水平方向的缓冲距离（列与列之间的间距）
        bracket_h_buff: float = 0.2,  # 括号与矩阵内容之间的水平缓冲距离
        bracket_v_buff: float = 0.25,  # 括号在垂直方向超出矩阵内容的缓冲距离
        height: float | None = None,  # 矩阵整体高度（不包括括号的垂直缓冲），None则自动计算
        element_config: dict = dict(),  # 矩阵元素的配置参数（如颜色、字体等）
        element_alignment_corner: Vect3 = DOWN,  # 矩阵元素的对齐参考角（默认为底部）
        ellipses_row: Optional[int] = None,  # 需要放置省略号的行索引（可选）
        ellipses_col: Optional[int] = None,  # 需要放置省略号的列索引（可选）
    ):
        """初始化矩阵对象，处理输入数据并构建矩阵的视觉元素"""
        # 调用父类VMobject的初始化方法
        super().__init__()

        # 创建由可移动对象(VMobject)组成的矩阵
        # 将输入的原始矩阵数据转换为Manim可显示的对象矩阵
        self.mob_matrix = self.create_mobject_matrix(
            matrix, v_buff, h_buff, element_alignment_corner,
            **element_config
        )

        # 创建用于管理元素的辅助组，便于后续操作（如动画）
        n_cols = len(self.mob_matrix[0])  # 获取矩阵的列数
        # 所有元素的列表（按行优先顺序排列）
        self.elements = [elem for row in self.mob_matrix for elem in row]
        # 按列分组的元素集合（每列一个VGroup）
        self.columns = VGroup(*(
            VGroup(*(row[i] for row in self.mob_matrix))
            for i in range(n_cols)
        ))
        # 按行分组的元素集合（每行一个VGroup）
        self.rows = VGroup(*(VGroup(*row) for row in self.mob_matrix))
        
        # 如果指定了高度，则调整行的整体高度（考虑括号的垂直缓冲）
        if height is not None:
            self.rows.set_height(height - 2 * bracket_v_buff)
            
        # 创建矩阵的括号（左右两侧的括号）
        self.brackets = self.create_brackets(self.rows, bracket_v_buff, bracket_h_buff)
        self.ellipses = []  # 存储省略号对象的列表（用于表示大型矩阵的省略部分）

        # 将元素和括号添加到当前矩阵对象中
        self.add(*self.elements)
        self.add(*self.brackets)
        self.center()  # 将矩阵整体居中显示

        # 根据指定的位置添加省略号（如果需要）
        self.swap_entries_for_ellipses(
            ellipses_row,
            ellipses_col,
        )

    def copy(self, deep: bool = False):
        """复制矩阵对象，确保复制后的对象包含所有必要的属性
        
        Args:
            deep: 是否深度复制，True则复制所有子对象，False则仅复制引用
        """
        # 调用父类的copy方法获取基础复制对象
        result = super().copy(deep)
        
        # 获取当前对象和复制对象的所有子对象（家族成员）
        self_family = self.get_family()
        copy_family = result.get_family()
        
        # 复制特定属性（elements和ellipses），确保引用正确指向复制后的对象
        for attr in ["elements", "ellipses"]:
            setattr(result, attr, [
                # 在复制对象的家族中找到对应原始对象的位置，并引用复制后的对象
                copy_family[self_family.index(mob)]
                for mob in getattr(self, attr)
            ])
        return result

def create_mobject_matrix(
        self,
        matrix: GenericMatrixType,
        v_buff: float,
        h_buff: float,
        aligned_corner: Vect3,** element_config
    ) -> VMobjectMatrixType:
    """
    将输入的矩阵数据转换为由Manim可移动对象(VMobject)组成的矩阵，并排列它们的位置
    
    Args:
        matrix: 原始矩阵数据，可以是数字、字符串或VMobject的二维序列
        v_buff: 垂直方向的缓冲距离（行间距）
        h_buff: 水平方向的缓冲距离（列间距）
        aligned_corner: 元素对齐的参考角
        **element_config: 传递给元素的配置参数
        
    Returns:
        由VMobject组成的矩阵
    """
    # 将矩阵中的每个元素转换为VMobject
    mob_matrix = [
        [
            # 调用element_to_mobject方法将单个元素转换为VMobject
            self.element_to_mobject(element,** element_config)
            for element in row
        ]
        for row in matrix
    ]
    
    # 计算所有元素中的最大宽度和最大高度，用于统一对齐
    max_width = max(elem.get_width() for row in mob_matrix for elem in row)
    max_height = max(elem.get_height() for row in mob_matrix for elem in row)
    
    # 计算每行和每列之间的步长（包含缓冲距离）
    x_step = (max_width + h_buff) * RIGHT  # 水平方向步长（向右）
    y_step = (max_height + v_buff) * DOWN  # 垂直方向步长（向下）
    
    # 排列矩阵中每个元素的位置
    for i, row in enumerate(mob_matrix):
        for j, elem in enumerate(row):
            # 根据元素所在的行(i)和列(j)计算位置，并按指定角对齐
            elem.move_to(i * y_step + j * x_step, aligned_corner)
    
    return mob_matrix

def element_to_mobject(self, element, **config) -> VMobject:
    """
    将单个元素转换为对应的Manim可移动对象(VMobject)
    
    Args:
        element: 要转换的元素，可以是VMobject、数字或字符串
        **config: 传递给生成的VMobject的配置参数
        
    Returns:
        转换后的VMobject
    """
    if isinstance(element, VMobject):
        # 如果元素已经是VMobject，直接返回
        return element
    elif isinstance(element, float | complex):
        # 如果是数字（浮点数或复数），转换为DecimalNumber
        return DecimalNumber(element,** config)
    else:
        # 其他类型（通常是字符串）转换为Tex对象
        return Tex(str(element), **config)

def create_brackets(self, rows, v_buff: float, h_buff: float) -> VGroup:
    """
    创建矩阵两侧的括号
    
    Args:
        rows: 包含矩阵所有行的VGroup
        v_buff: 括号在垂直方向超出矩阵内容的距离
        h_buff: 括号与矩阵内容之间的水平距离
        
    Returns:
        包含左右括号的VGroup
    """
    # 创建包含矩阵括号的Tex对象
    # 使用LaTeX语法生成与矩阵行数匹配的括号
    brackets = Tex("".join((
        R"\left[\begin{array}{c}",  # 左括号和矩阵环境开始
        *len(rows) * [R"\quad \\"],  # 为每行添加一个占位符
        R"\end{array}\right]",      # 矩阵环境结束和右括号
    )))
    
    # 调整括号高度以匹配矩阵内容高度（加上垂直缓冲）
    brackets.set_height(rows.get_height() + v_buff)
    
    # 将括号分为左半部分和右半部分
    l_bracket = brackets[:len(brackets) // 2]  # 左括号
    r_bracket = brackets[len(brackets) // 2:]  # 右括号
    
    # 定位左右括号到矩阵的两侧
    l_bracket.next_to(rows, LEFT, h_buff)  # 左括号放在矩阵左侧
    r_bracket.next_to(rows, RIGHT, h_buff) # 右括号放在矩阵右侧
    
    # 返回包含左右括号的VGroup
    return VGroup(l_bracket, r_bracket)

def get_column(self, index: int):
    """
    获取矩阵中指定索引的列
    
    Args:
        index: 列的索引（从0开始）
        
    Returns:
        包含该列所有元素的VGroup
        
    Raises:
        IndexError: 如果索引超出有效范围
    """
    # 检查索引是否有效
    if not 0 <= index < len(self.columns):
        raise IndexError(f"索引 {index} 超出范围，矩阵共有 {len(self.columns)} 列")
    return self.columns[index]

def get_row(self, index: int):
    """
    获取矩阵中指定索引的行
    
    Args:
        index: 行的索引（从0开始）
        
    Returns:
        包含该行所有元素的VGroup
        
    Raises:
        IndexError: 如果索引超出有效范围
    """
    # 检查索引是否有效
    if not 0 <= index < len(self.rows):
        raise IndexError(f"索引 {index} 超出范围，矩阵共有 {len(self.rows)} 行")
    return self.rows[index]

def get_columns(self) -> VGroup:
    """
    获取矩阵所有列的集合
    
    Returns:
        包含所有列的VGroup
    """
    return self.columns

def get_rows(self) -> VGroup:
    """
    获取矩阵所有行的集合
    
    Returns:
        包含所有行的VGroup
    """
    return self.rows

def set_column_colors(self, *colors: ManimColor) -> Self:
    """
    为矩阵的列依次设置颜色
    
    Args:
        *colors: 为各列设置的颜色，与列按顺序对应
        
    Returns:
        矩阵对象本身（支持方法链调用）
    """
    columns = self.get_columns()
    # 为每列设置对应的颜色
    for color, column in zip(colors, columns):
        column.set_color(color)
    return self

def add_background_to_entries(self) -> Self:
    """
    为矩阵中的每个元素添加背景矩形
    
    Returns:
        矩阵对象本身（支持方法链调用）
    """
    # 为每个元素添加背景矩形
    for mob in self.get_entries():
        mob.add_background_rectangle()
    return self

def swap_entry_for_dots(self, entry, dots):
    """
    将矩阵中的某个元素替换为省略号
    
    Args:
        entry: 要被替换的矩阵元素（VMobject）
        dots: 用于替换的省略号对象（通常是Tex对象）
    """
    # 将省略号移动到被替换元素的位置
    dots.move_to(entry)
    # 用省略号替换原元素
    entry.become(dots)
    # 如果原元素在elements列表中，将其移除
    if entry in self.elements:
        self.elements.remove(entry)
    # 如果原元素不在ellipses列表中，将其添加
    if entry not in self.ellipses:
        self.ellipses.append(entry)

def swap_entries_for_ellipses(
    self,
    row_index: Optional[int] = None,
    col_index: Optional[int] = None,
    height_ratio: float = 0.65,
    width_ratio: float = 0.4
):
    """
    在指定行或列位置用省略号替换元素，用于表示大型矩阵的省略部分
    
    Args:
        row_index: 要替换为垂直省略号的行索引（可选）
        col_index: 要替换为水平省略号的列索引（可选）
        height_ratio: 垂直省略号与行高的比例
        width_ratio: 水平省略号与列宽的比例
        
    Returns:
        矩阵对象本身（支持方法链调用）
    """
    rows = self.get_rows()
    cols = self.get_columns()

    # 计算平均行高和垂直省略号的高度
    avg_row_height = rows.get_height() / len(rows)
    vdots_height = height_ratio * avg_row_height

    # 计算平均列宽和水平省略号的宽度
    avg_col_width = cols.get_width() / len(cols)
    hdots_width = width_ratio * avg_col_width

    # 检查行索引和列索引是否有效
    use_vdots = row_index is not None and -len(rows) <= row_index < len(rows)
    use_hdots = col_index is not None and -len(cols) <= col_index < len(cols)

    # 如果行索引有效，替换该行所有元素为垂直省略号
    if use_vdots:
        for column in cols:
            # 创建垂直省略号
            dots = Tex(R"\vdots")
            dots.set_height(vdots_height)
            # 替换元素
            self.swap_entry_for_dots(column[row_index], dots)
    
    # 如果列索引有效，替换该列所有元素为水平省略号
    if use_hdots:
        for row in rows:
            # 创建水平省略号
            dots = Tex(R"\hdots")
            dots.set_width(hdots_width)
            # 替换元素
            self.swap_entry_for_dots(row[col_index], dots)
    
    # 如果同时使用了水平和垂直省略号，将交叉处的省略号旋转45度
    if use_vdots and use_hdots:
        rows[row_index][col_index].rotate(-45 * DEG)
    
    return self

def get_mob_matrix(self) -> VMobjectMatrixType:
    """
    获取由可移动对象组成的矩阵
    
    Returns:
        由VMobject组成的二维列表
    """
    return self.mob_matrix

def get_entries(self) -> VGroup:
    """
    获取矩阵中所有元素的集合
    
    Returns:
        包含所有矩阵元素的VGroup
    """
    return VGroup(*self.elements)

def get_brackets(self) -> VGroup:
    """
    获取矩阵的括号
    
    Returns:
        包含左右括号的VGroup
    """
    return VGroup(*self.brackets)

def get_ellipses(self) -> VGroup:
    """
    获取矩阵中的所有省略号
    
    Returns:
        包含所有省略号的VGroup
    """
    return VGroup(*self.ellipses)



class DecimalMatrix(Matrix):
    """
    用于显示十进制数字的矩阵子类
    """
    def __init__(
        self,
        matrix: FloatMatrixType,  # 浮点型矩阵数据
        num_decimal_places: int = 2,  # 保留的小数位数
        decimal_config: dict = dict(),  # 十进制数字的配置参数
        **config  # 传递给父类的其他参数
    ):
        self.float_matrix = matrix  # 存储原始浮点矩阵
        # 调用父类的初始化方法
        super().__init__(
            matrix,
            element_config=dict(
                num_decimal_places=num_decimal_places,** decimal_config
            ),
            **config
        )

    def element_to_mobject(self, element, **decimal_config) -> DecimalNumber:
        """
        将元素转换为DecimalNumber对象（重写父类方法）
        
        Args:
            element: 要转换的数字元素
            **decimal_config: DecimalNumber的配置参数
            
        Returns:
            转换后的DecimalNumber对象
        """
        return DecimalNumber(element, **decimal_config)


class IntegerMatrix(DecimalMatrix):
    """
    用于显示整数的矩阵子类（DecimalMatrix的子类）
    """
    def __init__(
        self,
        matrix: FloatMatrixType,  # 整数矩阵数据（可以是浮点形式）
        num_decimal_places: int = 0,  # 小数位数固定为0
        decimal_config: dict = dict(),  # 数字的配置参数
        **config  # 传递给父类的其他参数
    ):
        # 调用父类的初始化方法，强制小数位数为0
        super().__init__(matrix, num_decimal_places, decimal_config, **config)


class TexMatrix(Matrix):
    """
    用于显示LaTeX字符串的矩阵子类
    """
    def __init__(
        self,
        matrix: StringMatrixType,  # 字符串矩阵数据（包含LaTeX代码）
        tex_config: dict = dict(),  # Tex对象的配置参数
        **config,  # 传递给父类的其他参数
    ):
        # 调用父类的初始化方法
        super().__init__(
            matrix,
            element_config=tex_config,** config
        )


class MobjectMatrix(Matrix):
    """
    用于显示由Manim可移动对象(VMobject)组成的矩阵子类
    """
    def __init__(
        self,
        group: VGroup,  # 包含要组成矩阵的VMobject的VGroup
        n_rows: int | None = None,  # 矩阵的行数（可选）
        n_cols: int | None = None,  # 矩阵的列数（可选）
        height: float = 4.0,  # 矩阵的高度
        element_alignment_corner=ORIGIN,  # 元素对齐的参考点
        **config,  # 传递给父类的其他参数
    ):
        # 计算行数和列数的默认值
        n_mobs = len(group)
        if n_rows is None:
            # 如果未指定行数，根据列数或元素总数的平方根计算
            n_rows = int(np.sqrt(n_mobs)) if n_cols is None else n_mobs // n_cols
        if n_cols is None:
            # 如果未指定列数，根据行数计算
            n_cols = n_mobs // n_rows

        # 检查元素数量是否足够
        if len(group) < n_rows * n_cols:
            raise Exception("输入到MobjectMatrix的元素数量必须至少为n_rows * n_cols")

        # 构建由VMobject组成的矩阵
        mob_matrix = [
            [group[n * n_cols + k] for k in range(n_cols)]
            for n in range(n_rows)
        ]
        
        # 更新配置参数
        config.update(
            height=height,
            element_alignment_corner=element_alignment_corner,
        )
        
        # 调用父类的初始化方法
        super().__init__(mob_matrix, **config)

    def element_to_mobject(self, element: VMobject, **config) -> VMobject:
        """
        直接返回VMobject（重写父类方法，因为元素已经是VMobject）
        
        Args:
            element: 要转换的VMobject
            **config: 配置参数（此处未使用）
            
        Returns:
            原VMobject
        """
        return element