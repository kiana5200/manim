from typing import TYPE_CHECKING

# 仅在静态类型检查阶段（如 mypy 校验）执行以下导入，运行时不生效
# 避免导入未安装依赖（如 typing_extensions）或冗余模块，不影响程序性能
if TYPE_CHECKING:
    # 导入标准库核心类型，用于组合复杂注解
    from typing import Union, Tuple, Annotated, Literal, Iterable, Dict
    # 导入第三方库类型：colour.Color 用于颜色处理，numpy 用于数组/向量
    from colour import Color
    import numpy as np
    import re

    # 兼容 Python 版本：Python 3.11+ 原生支持 Self，低版本通过 typing_extensions 导入
    try:
        from typing_extensions import Self
    except ImportError:
        from typing import Self

    # ------------------------------ 常用基础类型别名 ------------------------------
    # 颜色类型：支持字符串（如"red"/"#FF0000"）、colour.Color 对象、None（使用默认颜色）
    ManimColor = Union[str, Color, None]
    # 范围指定类型：支持（起始, 结束, 步长）或（起始, 结束）两种格式，用于数值范围定义
    RangeSpecifier = Tuple[float, float, float] | Tuple[float, float]

    # ------------------------------ 选择器类型（文本/元素选择） ------------------------------
    # 单个选择器：支持三种选择方式
    # 1. 字符串：精确匹配（如文本字符"a"、元素名称"square"）
    # 2. 正则表达式：模糊匹配（如 re.compile(r"\d+") 匹配数字）
    # 3. 整数元组：索引范围（如(0,3)表示第0-3个元素，(None,2)表示前2个元素）
    SingleSelector = Union[
        str,
        re.Pattern,
        tuple[Union[int, None], Union[int, None]],
    ]
    # 选择器：支持单个选择器或多个选择器的可迭代对象（如列表、元组）
    Selector = Union[SingleSelector, Iterable[SingleSelector]]

    # ------------------------------ 统一字典类型（配置/属性传递） ------------------------------
    # 键为字符串，值为框架常用类型（浮点数/布尔值/数组/元组），避免字典值类型混乱
    UniformDict = Dict[str, float | bool | np.ndarray | tuple]

    # ------------------------------ 向量/矩阵类型（增强可读性） ------------------------------
    # 基于 numpy.ndarray 的形状注解，仅用于静态检查和代码可读性，不涉及运行时维度校验
    # 核心作用：明确数组维度，避免类型混淆（如 2D 向量 vs 3D 向量）

    # 任意维度的浮点数组（通用浮点型数组）
    FloatArray = np.ndarray[int, np.dtype[np.float64]]
    # 2维向量（[x, y]）：用于2D场景的位置、方向向量（如 RIGHT=(1,0)）
    Vect2 = Annotated[FloatArray, Literal[2]]
    # 3维向量（[x, y, z]）：用于3D场景的位置、方向向量（如 OUT=(0,0,1)）
    Vect3 = Annotated[FloatArray, Literal[3]]
    # 4维向量（[x, y, z, w]）：用于齐次坐标、RGBA颜色向量
    Vect4 = Annotated[FloatArray, Literal[4]]
    # N维向量（任意长度）：用于动态长度的向量数据
    VectN = Annotated[FloatArray, Literal["N"]]
    # 3x3矩阵：用于3D场景的变换（旋转、缩放、平移矩阵）
    Matrix3x3 = Annotated[FloatArray, Literal[3, 3]]
    # 单列向量数组（N×1）：多个单维向量的集合
    VectArray = Annotated[FloatArray, Literal["N", 1]]
    # 2维向量数组（N×2）：多个2D向量的集合（如多个2D点的坐标）
    Vect2Array = Annotated[FloatArray, Literal["N", 2]]
    # 3维向量数组（N×3）：多个3D向量的集合（如3D点云、顶点坐标）
    Vect3Array = Annotated[FloatArray, Literal["N", 3]]
    # 4维向量数组（N×4）：多个4D向量的集合（如齐次坐标集合）
    Vect4Array = Annotated[FloatArray, Literal["N", 4]]
    # N×M 向量数组：任意维度的向量集合（通用多维向量组）
    VectNArray = Annotated[FloatArray, Literal["N", "M"]]