# ManimGL 磁盘缓存工具：提供基于磁盘的函数结果缓存机制，用于存储计算成本高的中间结果（如 Tex 渲染、复杂图形生成），
# 避免重复计算，显著提升重复运行时的效率。


from __future__ import annotations

import os
from diskcache import Cache  # 磁盘缓存库（提供持久化键值存储）
from contextlib import contextmanager  # 上下文管理器工具（暂未使用）
from functools import wraps  # 函数装饰器工具（用于包装缓存函数）

# 导入 Manim 工具函数：获取缓存目录、字符串哈希
from manimlib.utils.directories import get_cache_dir
from manimlib.utils.simple_functions import hash_string

from typing import TYPE_CHECKING

# 仅静态类型检查阶段导入类型变量
if TYPE_CHECKING:
    T = TypeVar('T')  # 通用类型变量（用于标注缓存函数的返回值类型）


# ------------------------------ 缓存配置与初始化 ------------------------------
CACHE_SIZE = 1e9  # 缓存最大容量：1GB（超过后自动清理旧缓存）
# 初始化磁盘缓存对象：存储路径为 Manim 的缓存目录（由 get_cache_dir() 确定）
_cache = Cache(get_cache_dir(), size_limit=CACHE_SIZE)


# ------------------------------ 缓存装饰器 ------------------------------
def cache_on_disk(func: Callable[..., T]) -> Callable[..., T]:
    """
    磁盘缓存装饰器：将函数的输入参数与返回结果关联并存储到磁盘，
    下次以相同参数调用时直接返回缓存结果，避免重复计算。
    
    核心逻辑：
    1. 对函数名、位置参数、关键字参数进行哈希，生成唯一缓存键；
    2. 检查缓存中是否存在该键对应的结果，存在则直接返回；
    3. 不存在则执行函数，将结果存入缓存后返回。
    
    参数：func - 待缓存的函数（如 Tex 渲染函数、复杂图形生成函数）
    返回：包装后的函数（具备缓存功能）。
    """
    @wraps(func)  # 保留原函数的元信息（如名称、文档字符串）
    def wrapper(*args, **kwargs):
        # 生成缓存键：将函数名、参数转换为字符串后哈希（确保唯一）
        key = hash_string(f"{func.__name__}{args}{kwargs}")
        # 尝试从缓存中获取结果
        value = _cache.get(key)
        if value is None:
            # 缓存未命中：执行函数并将结果存入缓存
            value = func(*args, **kwargs)
            _cache.set(key, value)
        return value  # 返回缓存结果或新计算的结果
    return wrapper


# ------------------------------ 缓存管理函数 ------------------------------
def clear_cache():
    """
    清空所有磁盘缓存：删除缓存目录中的所有内容，适用于缓存数据过期或损坏的情况（如修改了 Tex 模板后需要重新渲染）。
    """
    _cache.clear()