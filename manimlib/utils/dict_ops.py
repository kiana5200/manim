import itertools as it
import numpy as np


def merge_dicts_recursively(*dicts):
    """
    递归合并多个字典：创建一个包含所有输入字典键的新字典，键的值由列表中最后出现该键的字典决定（后出现的字典优先级更高），
    当键对应的值也是字典时，会递归合并这些子字典（而非直接覆盖），适用于深层配置合并场景（如默认配置与用户配置的融合）。

    核心逻辑：
    1. 遍历所有输入字典的键值对（按字典顺序，后出现的字典键值对会覆盖前面的，除非值是字典）；
    2. 若键已存在于结果中，且新旧值均为字典，则递归合并这两个子字典；
    3. 其他情况（键不存在，或值非字典），直接用新值覆盖旧值。

    参数：*dicts - 任意数量的字典（如 dict1, dict2, dict3）

    返回：合并后的新字典，包含所有输入字典的键，深层嵌套字典会被递归合并。

    示例：
        dict1 = {"a": 1, "b": {"x": 10}}
        dict2 = {"b": {"y": 20}, "c": 3}
        合并结果：{"a": 1, "b": {"x": 10, "y": 20}, "c": 3}
    """
    result = dict()
    # 链式遍历所有字典的键值对（顺序为输入字典的顺序，后出现的字典键值对靠后）
    all_items = it.chain(*[d.items() for d in dicts])
    for key, value in all_items:
        # 若键已在结果中，且新旧值均为字典，则递归合并
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts_recursively(result[key], value)
        else:
            # 否则直接赋值（后出现的键值对会覆盖前面的）
            result[key] = value
    return result