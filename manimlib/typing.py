# 从typing模块导入TYPE_CHECKING常量，用于类型检查时的条件导入
from typing import TYPE_CHECKING

# 仅在类型检查阶段执行以下代码（运行时不执行）
if TYPE_CHECKING:
    # 从typing模块导入常用类型，用于类型注解
    from typing import Union, Tuple, Annotated, Literal, Iterable, Dict
    # 从colour库导入Color类，用于颜色相关类型注解
    from colour import Color
    # 导入numpy并简写为np，用于数组相关类型注解
    import numpy as np
    # 导入re模块，用于正则表达式相关类型注解
    import re

    # 尝试从typing_extensions导入Self类型（Python 3.11+标准库typing包含Self，低版本需扩展库）
    try:
        from typing_extensions import Self
    except ImportError:
        from typing import Self

    # 常用类型的缩写定义

    # 表示Manim颜色类型，可接受字符串（颜色名/十六进制）、Color对象或None
    ManimColor = Union[str, Color, None]
    # 表示范围指定器，可为（起始值, 结束值, 步长）或（起始值, 结束值）的元组
    RangeSpecifier = Tuple[float, float, float] | Tuple[float, float]


    # 表示跨度（如字符串切片的起止索引），为两个整数的元组
    Span = tuple[int, int]
    # 表示单个选择器，支持字符串、正则表达式对象或（起始索引, 结束索引）元组（索引可为None）
    SingleSelector = Union[
        str,
        re.Pattern,
        tuple[Union[int, None], Union[int, None]],
    ]
    # 表示选择器，可为单个选择器或多个选择器的可迭代对象
    Selector = Union[SingleSelector, Iterable[SingleSelector]]

    # 表示统一字典，键为字符串，值支持浮点数、布尔值、numpy数组或元组
    UniformDict = Dict[str, float | bool | np.ndarray | tuple]

    # 以下是numpy数组的各种别名，用于指定特定形状的数组
    # 理论上，这些注解可用于运行时检查数组大小，但目前仅用于增强可读性
    # 以便在numpy未来增强类型检查时提供更好的支持

    # 浮点型数组（int为维度元组类型，np.float64为数据类型）
    FloatArray = np.ndarray[int, np.dtype[np.float64]]
    # 2维向量（使用Annotated标记形状为2）
    Vect2 = Annotated[FloatArray, Literal[2]]
    # 3维向量（形状为3）
    Vect3 = Annotated[FloatArray, Literal[3]]
    # 4维向量（形状为4）
    Vect4 = Annotated[FloatArray, Literal[4]]
    # N维向量（形状为"N"，表示任意长度的一维数组）
    VectN = Annotated[FloatArray, Literal["N"]]
    # 3x3矩阵（形状为3x3）
    Matrix3x3 = Annotated[FloatArray, Literal[3, 3]]
    # 向量数组（形状为"N×1"，表示N个1维向量的集合）
    VectArray = Annotated[FloatArray, Literal["N", 1]]
    # 2维向量数组（形状为"N×2"）
    Vect2Array = Annotated[FloatArray, Literal["N", 2]]
    # 3维向量数组（形状为"N×3"）
    Vect3Array = Annotated[FloatArray, Literal["N", 3]]
    # 4维向量数组（形状为"N×4"）
    Vect4Array = Annotated[FloatArray, Literal["N", 4]]
    # N维向量数组（形状为"N×M"，表示N个M维向量的集合）
    VectNArray = Annotated[FloatArray, Literal["N", "M"]]
