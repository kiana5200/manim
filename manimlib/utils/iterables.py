# ManimGL 通用工具函数库：提供列表处理、数组操作、数据转换等基础功能，
# 涵盖去重、合并、批量分组、随机打乱、数组缩放等常用操作，是框架各模块的基础依赖。


from __future__ import annotations

from colour import Color  # 颜色对象（用于哈希处理）

import numpy as np
import random  # 随机数工具（用于打乱列表）

from typing import TYPE_CHECKING

# 仅静态类型检查阶段导入类型注解
if TYPE_CHECKING:
    from typing import Callable, Iterable, Sequence, TypeVar

    T = TypeVar("T")  # 通用类型变量（任意类型）
    S = TypeVar("S")  # 通用类型变量（用于属性值）


# ------------------------------ 列表去重与更新 ------------------------------
def remove_list_redundancies(lst: Sequence[T]) -> list[T]:
    """
    移除列表中的重复元素，保留最后一次出现的元素，同时维持其余元素的顺序。
    
    实现逻辑：通过反转列表 → 用 dict.fromkeys 去重（保留首次出现，即原列表的末次出现）→ 再反转回来。
    
    参数：lst - 待去重的序列（如列表、元组）
    返回：去重后的列表（保留顺序和末次出现的重复元素）。
    示例：[1, 2, 2, 3, 2] → [1, 3, 2]
    """
    return list(reversed(dict.fromkeys(reversed(lst))))


def list_update(l1: Iterable[T], l2: Iterable[T]) -> list[T]:
    """
    合并两个列表并去重，保持元素顺序，l2 中的元素优先级高于 l1（即 l2 中的重复元素覆盖 l1 中的）。
    
    等价于集合的 update 操作，但保留顺序，适用于需要维持插入顺序的场景。
    
    参数：
        l1 : 原始列表；
        l2 : 待合并的列表（元素优先级更高）。
    返回：合并去重后的列表。
    示例：l1=[1,2], l2=[2,3] → [1,2,3]
    """
    return remove_list_redundancies([*l1, *l2])  # 拼接后去重，l2元素在后，保留末次出现


def list_difference_update(l1: Iterable[T], l2: Iterable[T]) -> list[T]:
    """
    从 l1 中移除所有在 l2 中出现的元素，保留 l1 的原始顺序。
    
    等价于集合的 difference_update 操作，但保留顺序。
    
    参数：
        l1 : 原始列表；
        l2 : 待移除元素的列表。
    返回：过滤后的 l1 列表。
    示例：l1=[1,2,3], l2=[2] → [1,3]
    """
    return [e for e in l1 if e not in l2]


# ------------------------------ 相邻元素组合 ------------------------------
def adjacent_n_tuples(objects: Sequence[T], n: int) -> zip[tuple[T, ...]]:
    """
    生成序列中相邻 n 个元素的元组迭代器，支持循环环绕（最后一个元素与第一个元素相邻）。
    
    参数：
        objects : 输入序列；
        n : 每个元组的元素个数。
    返回：zip 对象，包含相邻 n 元素的元组。
    示例：objects=[1,2,3], n=2 → (1,2), (2,3), (3,1)
    """
    return zip(*[
        [*objects[k:], *objects[:k]]  # 对每个 k 偏移序列，实现循环
        for k in range(n)
    ])


def adjacent_pairs(objects: Sequence[T]) -> zip[tuple[T, T]]:
    """
    生成序列中相邻元素的 pairs 迭代器（n=2 的相邻元组），循环环绕。
    
    参数：objects - 输入序列
    返回：zip 对象，包含相邻元素对。
    示例：objects=[1,2,3] → (1,2), (2,3), (3,1)
    """
    return adjacent_n_tuples(objects, 2)  # 调用 n=2 的相邻元组函数


# ------------------------------ 按属性批量分组 ------------------------------
def batch_by_property(
    items: Iterable[T],
    property_func: Callable[[T], S]
) -> list[tuple[T, S]]:
    """
    按元素属性批量分组：将连续具有相同属性的元素分为一组，保留原始顺序，
    适用于需要按连续相同属性处理的场景（如连续同色的图形批量设置动画）。
    
    参数：
        items : 待分组的元素迭代器；
        property_func : 提取元素属性的函数（如 lambda x: x.color）。
    返回：列表，每个元素为 (批量元素列表, 属性值) 的元组。
    示例：items=[1,1,2,2,3], property_func=lambda x:x → [([1,1],1), ([2,2],2), ([3],3)]
    """
    batch_prop_pairs = []
    curr_batch = []  # 当前批次的元素
    curr_prop = None  # 当前批次的属性值

    for item in items:
        prop = property_func(item)  # 计算当前元素的属性
        if prop != curr_prop:
            # 属性变化：保存上一批次（非空时）
            if len(curr_batch) > 0:
                batch_prop_pairs.append((curr_batch, curr_prop))
            # 初始化新批次
            curr_prop = prop
            curr_batch = [item]
        else:
            # 属性不变：加入当前批次
            curr_batch.append(item)

    # 处理最后一批次
    if len(curr_batch) > 0:
        batch_prop_pairs.append((curr_batch, curr_prop))

    return batch_prop_pairs


# ------------------------------ 类型转换与随机化 ------------------------------
def listify(obj: object) -> list:
    """
    将任意对象转换为列表：
    - 字符串特殊处理（返回单元素列表，避免被拆分为字符）；
    - 可迭代对象（如元组、集合）转换为列表；
    - 非可迭代对象返回单元素列表。
    
    参数：obj - 任意对象
    返回：列表（确保输入对象被包裹为列表形式）。
    示例："abc" → ["abc"]；(1,2) → [1,2]；5 → [5]
    """
    if isinstance(obj, str):
        return [obj]  # 字符串不拆分
    try:
        return list(obj)  # 可迭代对象转为列表
    except TypeError:
        return [obj]  # 非可迭代对象返回单元素列表


def shuffled(iterable: Iterable) -> list:
    """
    随机打乱可迭代对象的元素顺序，返回新列表（不修改原对象）。
    
    参数：iterable - 待打乱的可迭代对象
    返回：打乱后的列表。
    """
    as_list = list(iterable)  # 转换为列表
    random.shuffle(as_list)   # 原地打乱
    return as_list


# ------------------------------ 数组缩放与匹配 ------------------------------
def resize_array(nparray: np.ndarray, length: int) -> np.ndarray:
    """
    调整数组长度（第一维度），超出部分循环填充，不足部分截断。
    
    参数：
        nparray : 输入 numpy 数组；
        length : 目标长度。
    返回：调整后的数组（第一维度长度为 length）。
    示例：nparray=[[1],[2]], length=3 → [[1],[2],[1]]
    """
    if len(nparray) == length:
        return nparray
    return np.resize(nparray, (length, *nparray.shape[1:]))  # 利用 numpy  resize 循环填充


def resize_preserving_order(nparray: np.ndarray, length: int) -> np.ndarray:
    """
    调整数组长度，通过均匀采样保留原始元素顺序（不插值，仅选择现有元素）。
    
    参数：
        nparray : 输入数组；
        length : 目标长度。
    返回：采样后的数组（长度为 length，元素来自原数组）。
    示例：nparray=[0,1,2], length=5 → [0,0,1,2,2]（近似均匀采样）
    """
    if len(nparray) == 0:
        return np.resize(nparray, length)
    if len(nparray) == length:
        return nparray
    # 计算采样索引（均匀分布在原数组长度范围内）
    indices = np.arange(length) * len(nparray) // length
    return nparray[indices]


def resize_with_interpolation(nparray: np.ndarray, length: int) -> np.ndarray:
    """
    调整数组长度，通过线性插值生成新元素，保持数据趋势（适用于平滑过渡）。
    
    参数：
        nparray : 输入数组（1D 或更高维，第一维度为长度）；
        length : 目标长度。
    返回：插值后的数组（长度为 length）。
    示例：nparray=[0,2], length=3 → [0,1,2]（线性插值）
    """
    if len(nparray) == length:
        return nparray
    # 原数组为单元素或常量数组：直接重复填充
    if len(nparray) == 1 or array_is_constant(nparray):
        return nparray[:1].repeat(length, axis=0)
    if length == 0:
        return np.zeros((0, *nparray.shape[1:]))  # 空数组
    # 生成均匀分布的连续索引（0 到 len-1）
    cont_indices = np.linspace(0, len(nparray) - 1, length)
    # 对每个索引进行线性插值
    return np.array([
        (1 - a) * nparray[lh] + a * nparray[rh]
        for ci in cont_indices
        for lh, rh, a in [(int(ci), int(np.ceil(ci)), ci % 1)]  # lh=左索引，rh=右索引，a=插值比例
    ])


def make_even(
    iterable_1: Sequence[T],
    iterable_2: Sequence[S]
) -> tuple[Sequence[T], Sequence[S]]:
    """
    调整两个序列的长度至相同（取较长者的长度），通过均匀采样保持各自的元素分布。
    
    参数：
        iterable_1 : 第一个序列；
        iterable_2 : 第二个序列。
    返回：长度相同的两个序列（元组）。
    示例：iterable_1=[1,2], iterable_2=[3] → ([1,2], [3,3])
    """
    len1 = len(iterable_1)
    len2 = len(iterable_2)
    if len1 == len2:
        return iterable_1, iterable_2
    new_len = max(len1, len2)  # 目标长度为较长序列的长度
    # 对两个序列分别进行均匀采样
    return (
        [iterable_1[(n * len1) // new_len] for n in range(new_len)],
        [iterable_2[(n * len2) // new_len] for n in range(new_len)]
    )


# ------------------------------ 数组判断与运算 ------------------------------
def arrays_match(arr1: np.ndarray, arr2: np.ndarray) -> bool:
    """
    判断两个 numpy 数组是否完全相同（形状相同且所有元素相等）。
    
    参数：arr1, arr2 - 待比较的数组
    返回：布尔值，True 表示数组完全相同。
    """
    return arr1.shape == arr2.shape and (arr1 == arr2).all()


def array_is_constant(arr: np.ndarray) -> bool:
    """
    判断数组是否为常量数组（所有元素都与第一个元素相等）。
    
    参数：arr - 输入数组
    返回：布尔值，True 表示所有元素相同。
    """
    return len(arr) > 0 and (arr == arr[0]).all()


def cartesian_product(*arrays: np.ndarray) -> np.ndarray:
    """
    计算多个数组的笛卡尔积（所有可能的元素组合）。
    
    来源：https://stackoverflow.com/a/11146645
    参数：*arrays - 任意数量的 numpy 数组
    返回：2D 数组，每行是一个元素组合。
    示例：arrays=[[1,2], [3,4]] → [[1,3], [1,4], [2,3], [2,4]]
    """
    la = len(arrays)
    dtype = np.result_type(*arrays)  # 确定结果数据类型
    # 创建多维数组存储笛卡尔积
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a  # 填充每个维度的元素
    # 重塑为 2D 数组（组合数 × 数组个数）
    return arr.reshape(-1, la)


# ------------------------------ 通用哈希函数 ------------------------------
def hash_obj(obj: object) -> int:
    """
    通用对象哈希函数：支持字典、集合、列表、元组、Color 对象等，生成唯一哈希值，
    解决 Python 内置 hash 对可变对象（如列表）不支持的问题。
    
    参数：obj - 任意对象
    返回：整数哈希值。
    """
    if isinstance(obj, dict):
        # 字典：排序键值对后哈希（确保键顺序不影响结果）
        return hash(tuple(sorted([
            (hash_obj(k), hash_obj(v)) for k, v in obj.items()
        ])))
    if isinstance(obj, set):
        # 集合：排序元素后哈希（确保元素顺序不影响结果）
        return hash(tuple(sorted(hash_obj(e) for e in obj)))
    if isinstance(obj, (tuple, list)):
        # 列表/元组：哈希每个元素的哈希值组成的元组
        return hash(tuple(hash_obj(e) for e in obj))
    if isinstance(obj, Color):
        # 颜色对象：哈希其 RGB 值（避免不同实例但同颜色的哈希差异）
        return hash(obj.get_rgb())
    # 其他对象：使用内置 hash
    return hash(obj)