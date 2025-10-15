from __future__ import annotations

import numpy as np

from manimlib.constants import DOWN, LEFT, RIGHT, ORIGIN
from manimlib.constants import DEG
from manimlib.mobject.numbers import DecimalNumber
from manimlib.mobject.svg.tex_mobject import Tex
from manimlib.mobject.types.vectorized_mobject import VGroup
from manimlib.mobject.types.vectorized_mobject import VMobject

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Sequence, Union, Optional
    from manimlib.typing import ManimColor, Vect3, VectNArray, Self

    StringMatrixType = Union[Sequence[Sequence[str]], np.ndarray[int, np.dtype[np.str_]]]
    FloatMatrixType = Union[Sequence[Sequence[float]], VectNArray]
    VMobjectMatrixType = Sequence[Sequence[VMobject]]
    GenericMatrixType = Union[FloatMatrixType, StringMatrixType, VMobjectMatrixType]


class Matrix(VMobject):
    def __init__(
        self,
        matrix: GenericMatrixType,
        v_buff: float = 0.5,
        h_buff: float = 0.5,
        bracket_h_buff: float = 0.2,
        bracket_v_buff: float = 0.25,
        height: float | None = None,
        element_config: dict = dict(),
        element_alignment_corner: Vect3 = DOWN,
        ellipses_row: Optional[int] = None,
        ellipses_col: Optional[int] = None,
    ):
        """
        矩阵可包含数字、Tex字符串或Mobject对象
        """
        # 调用父类VMobject初始化
        super().__init__()

        # 将输入矩阵转换为Mobject矩阵（每个元素转为Mobject并排版）
        self.mob_matrix = self.create_mobject_matrix(
            matrix, v_buff, h_buff, element_alignment_corner,
            **element_config
        )

        # 为矩阵元素创建便于操作的组
        n_cols = len(self.mob_matrix[0])  # 获取矩阵列数
        # 所有元素的扁平组（按行遍历）
        self.elements = [elem for row in self.mob_matrix for elem in row]
        # 按列分组（每列元素组成一个VGroup）
        self.columns = VGroup(*(
            VGroup(*(row[i] for row in self.mob_matrix))
            for i in range(n_cols)
        ))
        # 按行分组（每行元素组成一个VGroup）
        self.rows = VGroup(*(VGroup(*row) for row in self.mob_matrix))
        
        # 若指定矩阵高度，调整行组高度（扣除括号垂直间距）
        if height is not None:
            self.rows.set_height(height - 2 * bracket_v_buff)
        
        # 创建矩阵的左右括号
        self.brackets = self.create_brackets(self.rows, bracket_v_buff, bracket_h_buff)
        # 初始化椭圆占位符列表（用于省略矩阵部分元素）
        self.ellipses = []

        # 将元素和括号添加到矩阵中
        self.add(*self.elements)
        self.add(*self.brackets)
        # 矩阵整体居中
        self.center()

        # 按需在指定行列位置添加椭圆占位符（替换原有元素）
        self.swap_entries_for_ellipses(
            ellipses_row,
            ellipses_col,
        )


    def copy(self, deep: bool = False):
        # 调用父类的copy方法创建副本
        result = super().copy(deep)
        # 获取当前对象和副本的所有子对象家族
        self_family = self.get_family()
        copy_family = result.get_family()
        # 复制"elements"和"ellipses"属性，确保引用副本中的对应对象
        for attr in ["elements", "ellipses"]:
            setattr(result, attr, [
                copy_family[self_family.index(mob)]  # 找到副本中对应的对象
                for mob in getattr(self, attr)
            ])
        return result

    def create_mobject_matrix(
        self,
        matrix: GenericMatrixType,
        v_buff: float,
        h_buff: float,
        aligned_corner: Vect3,
        **element_config
    ) -> VMobjectMatrixType:
        """
        创建并排列Mobject矩阵
        """
        # 将矩阵元素转换为Mobject并保持原矩阵结构
        mob_matrix = [
            [
                self.element_to_mobject(element, **element_config)
                for element in row
            ]
            for row in matrix
        ]
        # 计算所有元素的最大宽度和高度，用于统一对齐
        max_width = max(elem.get_width() for row in mob_matrix for elem in row)
        max_height = max(elem.get_height() for row in mob_matrix for elem in row)
        # 计算列间距（最大宽度+水平缓冲）和行间距（最大高度+垂直缓冲）
        x_step = (max_width + h_buff) * RIGHT
        y_step = (max_height + v_buff) * DOWN
        # 按行列位置排列每个元素
        for i, row in enumerate(mob_matrix):
            for j, elem in enumerate(row):
                elem.move_to(i * y_step + j * x_step, aligned_corner)
        return mob_matrix

    def element_to_mobject(self, element, **config) -> VMobject:
        # 根据元素类型转换为对应的Mobject
        if isinstance(element, VMobject):
            # 若已是Mobject，直接返回
            return element
        elif isinstance(element, float | complex):
            # 数字类型转换为DecimalNumber
            return DecimalNumber(element, **config)
        else:
            # 其他类型转换为Tex对象
            return Tex(str(element), **config)

    def create_brackets(self, rows, v_buff: float, h_buff: float) -> VGroup:
        # 创建矩阵的左右方括号（使用Tex生成）
        brackets = Tex("".join((
            R"\left[\begin{array}{c}",  # 左括号和起始环境
            *len(rows) * [R"\quad \\"],  # 每行一个占位符
            R"\end{array}\right]",      # 结束环境和右括号
        )))
        # 调整括号高度（行组高度+垂直缓冲）
        brackets.set_height(rows.get_height() + v_buff)
        # 分割左右括号
        l_bracket = brackets[:len(brackets) // 2]
        r_bracket = brackets[len(brackets) // 2:]
        # 定位左括号到行组左侧，保持水平缓冲
        l_bracket.next_to(rows, LEFT, h_buff)
        # 定位右括号到行组右侧，保持水平缓冲
        r_bracket.next_to(rows, RIGHT, h_buff)
        return VGroup(l_bracket, r_bracket)

    def get_column(self, index: int):
        # 检查列索引是否在有效范围内，超出则抛出索引错误
        if not 0 <= index < len(self.columns):
            raise IndexError(f"Index {index} out of bound for matrix with {len(self.columns)} columns")
        return self.columns[index]

    def get_row(self, index: int):
        # 检查行索引是否在有效范围内，超出则抛出索引错误
        if not 0 <= index < len(self.rows):
            raise IndexError(f"Index {index} out of bound for matrix with {len(self.rows)} rows")
        return self.rows[index]

    def get_columns(self) -> VGroup:
        # 返回所有列组成的VGroup
        return self.columns

    def get_rows(self) -> VGroup:
        # 返回所有行组成的VGroup
        return self.rows

    def set_column_colors(self, *colors: ManimColor) -> Self:
        columns = self.get_columns()
        # 为每列设置对应的颜色（颜色与列按顺序匹配）
        for color, column in zip(colors, columns):
            column.set_color(color)
        return self

    def add_background_to_entries(self) -> Self:
        # 为矩阵所有元素添加背景矩形（调用元素的add_background_rectangle方法）
        for mob in self.get_entries():
            mob.add_background_rectangle()
        return self

    def swap_entry_for_dots(self, entry, dots):
        # 将省略号（dots）移动到目标元素（entry）的位置
        dots.move_to(entry)
        entry.become(dots)
        # 若目标元素在elements列表中，将其移除
        if entry in self.elements:
            self.elements.remove(entry)
        # 若目标元素不在ellipses列表中，将其添加（标记为省略号）
        if entry not in self.ellipses:
            self.ellipses.append(entry)

    def swap_entries_for_ellipses(
        self,
        row_index: Optional[int] = None,
        col_index: Optional[int] = None,
        height_ratio: float = 0.65,
        width_ratio: float = 0.4
    ):
        # 获取矩阵的行组和列组
        rows = self.get_rows()
        cols = self.get_columns()

        # 计算平均行高和垂直省略号（vdots）的高度
        avg_row_height = rows.get_height() / len(rows)
        vdots_height = height_ratio * avg_row_height

        # 计算平均列宽和水平省略号（hdots）的宽度
        avg_col_width = cols.get_width() / len(cols)
        hdots_width = width_ratio * avg_col_width

        # 判断是否需要添加垂直省略号和水平省略号（索引需在有效范围内）
        use_vdots = row_index is not None and -len(rows) <= row_index < len(rows)
        use_hdots = col_index is not None and -len(cols) <= col_index < len(cols)

        # 若需添加垂直省略号：遍历每列，替换指定行的元素为垂直省略号
        if use_vdots:
            for column in cols:
                # Add vdots
                dots = Tex(R"\vdots")
                dots.set_height(vdots_height)
                # 替换列中指定行的元素为省略号
                self.swap_entry_for_dots(column[row_index], dots)
        # 若需添加水平省略号：遍历每行，替换指定列的元素为水平省略号
        if use_hdots:
            for row in rows:
                # Add hdots
                dots = Tex(R"\hdots")
                dots.set_width(hdots_width)
                self.swap_entry_for_dots(row[col_index], dots)
        # 若同时添加两种省略号：将交叉位置的省略号旋转45度（避免重叠）
        if use_vdots and use_hdots:
            rows[row_index][col_index].rotate(-45 * DEG)
        return self

    def get_mob_matrix(self) -> VMobjectMatrixType:
        # 返回由Mobject组成的矩阵（保持行列结构）
        return self.mob_matrix

    def get_entries(self) -> VGroup:
        # 返回所有矩阵元素组成的VGroup
        return VGroup(*self.elements)

    def get_brackets(self) -> VGroup:
        # 返回矩阵左右括号组成的VGroup
        return VGroup(*self.brackets)

    def get_ellipses(self) -> VGroup:
        # 返回所有省略号元素组成的VGroup
        return VGroup(*self.ellipses)


class DecimalMatrix(Matrix):
    def __init__(
        self,
        matrix: FloatMatrixType,
        num_decimal_places: int = 2,
        decimal_config: dict = dict(),** config
    ):
        # 存储原始浮点数矩阵
        self.float_matrix = matrix
        # 调用父类Matrix的初始化方法
        super().__init__(
            matrix,
            # 配置矩阵元素：默认保留2位小数，合并自定义小数配置
            element_config=dict(
                num_decimal_places=num_decimal_places,
                **decimal_config
            ),
            # 传递其他额外配置
            **config
        )

    def element_to_mobject(self, element, **decimal_config) -> DecimalNumber:
        # 重写父类方法：强制将所有元素转换为DecimalNumber（仅处理数字）
        return DecimalNumber(element, **decimal_config)

class IntegerMatrix(DecimalMatrix):
    def __init__(
        self,
        matrix: FloatMatrixType,
        num_decimal_places: int = 0,
        decimal_config: dict = dict(),** config
    ):
        # 调用父类DecimalMatrix的初始化方法，强制小数位数为0（显示整数）
        super().__init__(matrix, num_decimal_places, decimal_config, **config)


class TexMatrix(Matrix):
    def __init__(
        self,
        matrix: StringMatrixType,
        tex_config: dict = dict(),
        **config,
    ):
        # 调用父类Matrix的初始化方法，元素配置使用Tex相关参数
        super().__init__(
            matrix,
            element_config=tex_config,  # 传递Tex对象的配置（如字体、颜色等）
            **config
        )


class MobjectMatrix(Matrix):
    def __init__(
        self,
        group: VGroup,
        n_rows: int | None = None,
        n_cols: int | None = None,
        height: float = 4.0,
        element_alignment_corner=ORIGIN,** config,
    ):
        # 获取VGroup中Mobject的总数量
        n_mobs = len(group)
        
        # 推导行列数：无指定时，优先按平方根推测，或根据已知行列数计算
        if n_rows is None:
            n_rows = int(np.sqrt(n_mobs)) if n_cols is None else n_mobs // n_cols
        if n_cols is None:
            n_cols = n_mobs // n_rows
        
        # 检查VGroup中元素数量是否足够填充指定行列数的矩阵
        if len(group) < n_rows * n_cols:
            raise Exception("MobjectMatrix的输入必须至少包含n_rows * n_cols个元素")

        # 将VGroup中的元素按行列数重新组织为矩阵结构
        mob_matrix = [
            [group[n * n_cols + k] for k in range(n_cols)]  # 第n行的n_cols个元素
            for n in range(n_rows)
        ]
        
        # 更新配置：添加高度和元素对齐角参数
        config.update(
            height=height,
            element_alignment_corner=element_alignment_corner,
        )
        
        # 调用父类Matrix的初始化方法
        super().__init__(mob_matrix,  **config)

    def element_to_mobject(self, element: VMobject, **config) -> VMobject:
        # 重写父类方法：Mobject元素无需转换，直接返回原对象
        return element