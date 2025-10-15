# 从__future__导入annotations，支持在类型提示中使用尚未定义的类
from __future__ import annotations

import copy
from functools import wraps
import itertools as it
import os
import pickle
import random
import sys

import moderngl  # 现代OpenGL绑定库，用于渲染
import numbers
import numpy as np

# 从manimlib.constants导入各种常量
from manimlib.constants import DEFAULT_MOBJECT_TO_EDGE_BUFF  # 对象到边缘的默认缓冲距离
from manimlib.constants import DEFAULT_MOBJECT_TO_MOBJECT_BUFF  # 对象之间的默认缓冲距离
from manimlib.constants import DOWN, IN, LEFT, ORIGIN, OUT, RIGHT, UP  # 方向向量常量
from manimlib.constants import FRAME_X_RADIUS, FRAME_Y_RADIUS  # 帧的X/Y半径
from manimlib.constants import MED_SMALL_BUFF  # 中等偏小的缓冲距离
from manimlib.constants import TAU  # 2π常量（360度）
from manimlib.constants import DEFAULT_MOBJECT_COLOR  # 默认对象颜色

# 导入事件处理相关模块
from manimlib.event_handler import EVENT_DISPATCHER  # 事件调度器
from manimlib.event_handler.event_listner import EventListener  # 事件监听器
from manimlib.event_handler.event_type import EventType  # 事件类型

from manimlib.logger import log  # 日志工具
from manimlib.shader_wrapper import ShaderWrapper  # 着色器包装器

# 导入颜色处理工具函数
from manimlib.utils.color import color_gradient  # 颜色渐变
from manimlib.utils.color import color_to_rgb  # 颜色转RGB
from manimlib.utils.color import get_colormap_list  # 获取颜色映射列表
from manimlib.utils.color import rgb_to_hex  # RGB转十六进制

# 导入可迭代对象处理工具函数
from manimlib.utils.iterables import arrays_match  # 数组匹配检查
from manimlib.utils.iterables import array_is_constant  # 数组是否为常量
from manimlib.utils.iterables import batch_by_property  # 按属性批量处理
from manimlib.utils.iterables import list_update  # 列表更新
from manimlib.utils.iterables import listify  # 转换为列表
from manimlib.utils.iterables import resize_array  # 调整数组大小
from manimlib.utils.iterables import resize_preserving_order  # 保持顺序调整大小
from manimlib.utils.iterables import resize_with_interpolation  # 插值调整大小

# 导入贝塞尔曲线和插值工具
from manimlib.utils.bezier import integer_interpolate  # 整数插值
from manimlib.utils.bezier import interpolate  # 插值

# 导入路径和空间操作工具
from manimlib.utils.paths import straight_path  # 直线路径
from manimlib.utils.shaders import get_colormap_code  # 获取颜色映射代码
from manimlib.utils.space_ops import angle_of_vector  # 向量角度计算
from manimlib.utils.space_ops import get_norm  # 求范数
from manimlib.utils.space_ops import rotation_matrix_transpose  # 旋转矩阵转置

# 类型检查相关导入
from typing import TYPE_CHECKING
from typing import TypeVar, Generic, Iterable
SubmobjectType = TypeVar('SubmobjectType', bound='Mobject')  # 子对象类型变量


if TYPE_CHECKING:
    # 类型提示定义
    from typing import Callable, Iterator, Union, Tuple, Optional, Any
    import numpy.typing as npt
    from manimlib.typing import ManimColor, Vect3, Vect4Array, Vect3Array, UniformDict, Self
    from moderngl.context import Context

    T = TypeVar('T')
    # 更新器类型定义：基于时间的更新器和非时间更新器
    TimeBasedUpdater = Callable[["Mobject", float], "Mobject" | None]
    NonTimeUpdater = Callable[["Mobject"], "Mobject" | None]
    Updater = Union[TimeBasedUpdater, NonTimeUpdater]


class Mobject(object):
    """
    所有可渲染数学对象的基类，提供基础的渲染和动画功能
    """
    dim: int = 3  # 维度，默认为3D
    shader_folder: str = ""  # 着色器文件夹路径
    render_primitive: int = moderngl.TRIANGLE_STRIP  # 渲染图元类型，默认为三角形带
    # 数据类型定义，必须与顶点着色器的属性匹配
    data_dtype: np.dtype = np.dtype([
        ('point', np.float32, (3,)),  # 点坐标（x,y,z）
        ('rgba', np.float32, (4,)),   # 颜色和透明度（r,g,b,a）
    ])
    aligned_data_keys = ['point']  # 需要对齐的数据键
    pointlike_data_keys = ['point']  # 类点数据键

    def __init__(
        self,
        color: ManimColor = DEFAULT_MOBJECT_COLOR,  # 对象颜色，默认为白色
        opacity: float = 1.0,  # 不透明度，1.0为完全不透明
        shading: Tuple[float, float, float] = (0.0, 0.0, 0.0),  # 着色参数
        # 纹理路径字典
        texture_paths: dict[str, str] | None = None,
        # 如果为True，对象不会随相机位置旋转
        is_fixed_in_frame: bool = False,
        depth_test: bool = False,  # 是否启用深度测试
        z_index: int = 0,  # z轴索引，用于渲染排序
    ):
        self.color = color
        self.opacity = opacity
        self.shading = shading
        self.texture_paths = texture_paths or {}
        self.depth_test = depth_test
        self.z_index = z_index

        # 内部状态变量
        self.submobjects: list[Mobject] = []  # 子对象列表
        self.parents: list[Mobject] = []  # 父对象列表
        self.family: list[Mobject] | None = [self]  # 对象家族（包括自身和所有子对象）
        self.locked_data_keys: set[str] = set()  # 锁定的数据键（不可修改）
        self.const_data_keys: set[str] = set()  # 常量数据键（不参与插值）
        self.locked_uniform_keys: set[str] = set()  # 锁定的统一变量键
        self.saved_state = None  # 保存的状态
        self.target = None  # 目标状态（用于动画）
        self.bounding_box: Vect3Array = np.zeros((3, 3))  #  bounding box（边界框）
        self.shader_wrapper: Optional[ShaderWrapper] = None  # 着色器包装器
        self._is_animating: bool = False  # 是否正在动画中
        self._needs_new_bounding_box: bool = True  # 是否需要更新边界框
        self._data_has_changed: bool = True  # 数据是否已更改（用于渲染优化）
        self.shader_code_replacements: dict[str, str] = dict()  # 着色器代码替换字典

        # 初始化各种组件
        self.init_data()  # 初始化数据
        self.init_uniforms()  # 初始化统一变量
        self.init_updaters()  # 初始化更新器
        self.init_event_listners()  # 初始化事件监听器
        self.init_points()  # 初始化点
        self.init_colors()  # 初始化颜色

        # 应用深度测试设置
        if self.depth_test:
            self.apply_depth_test()
        # 如果需要固定在帧中
        if is_fixed_in_frame:
            self.fix_in_frame()

    def __str__(self):
    #返回对象的字符串表示形式，默认为类名
        return self.__class__.__name__

def __add__(self, other: Mobject) -> Mobject:
    """
    重载加法运算符，用于将两个Mobject组合成一个组
    
    Args:
        other: 要添加的另一个Mobject
        
    Returns:
        包含当前对象和other的组对象
        
    Raises:
        AssertionError: 如果other不是Mobject实例
    """
    assert isinstance(other, Mobject)
    return self.get_group_class()(self, other)

def __mul__(self, other: int) -> Mobject:
    """
    重载乘法运算符，用于复制当前对象指定次数
    
    Args:
        other: 复制次数（整数）
        
    Returns:
        包含多个当前对象副本的组
        
    Raises:
        AssertionError: 如果other不是整数
    """
    assert isinstance(other, int)
    return self.replicate(other)

def init_data(self, length: int = 0):
    """
    初始化对象的数据数组
    
    Args:
        length: 数据数组的初始长度
    """
    # 创建指定长度的空数据数组，数据类型由类定义的data_dtype决定
    self.data = np.zeros(length, dtype=self.data_dtype)
    # 存储数据默认值（用于后续扩展数组时使用）
    self._data_defaults = np.ones(1, dtype=self.data.dtype)

def init_uniforms(self):
    """初始化着色器的统一变量（uniforms）"""
    self.uniforms: UniformDict = {
        "is_fixed_in_frame": 0.0,  # 是否固定在帧中（0表示否，1表示是）
        "shading": np.array(self.shading, dtype=float),  # 着色参数
        "clip_plane": np.zeros(4),  # 裁剪平面参数
    }

def init_colors(self):
    """初始化对象的颜色和透明度"""
    self.set_color(self.color, self.opacity)

def init_points(self):
    """
    初始化对象的点数据
    
    通常在子类中实现，除非有意留空
    """
    pass

def set_uniforms(self, uniforms: dict) -> Self:
    """
    设置着色器的统一变量
    
    Args:
        uniforms: 包含统一变量键值对的字典
        
    Returns:
        对象本身（支持方法链）
    """
    for key, value in uniforms.items():
        # 如果是numpy数组，创建副本避免外部修改影响内部状态
        if isinstance(value, np.ndarray):
            value = value.copy()
        self.uniforms[key] = value
    return self

@property
def animate(self) -> _AnimationBuilder | Self:
    """
    动画构建器属性，用于创建动画
    
    通过Mobject.animate.method()调用的方法可以传递给Scene.play()，
    相当于调用ApplyMethod(mobject.method)
    
    借鉴自https://github.com/ManimCommunity/manim/
    """
    return _AnimationBuilder(self)

@property
def always(self) -> _UpdaterBuilder:
    """
    更新器构建器属性，用于创建持续性更新
    
    通过mobject.always.method(*args, **kwargs)调用的方法
    将在每一帧都被调用
    """
    return _UpdaterBuilder(self)

@property
def f_always(self) -> _FunctionalUpdaterBuilder:
    """
    功能性更新器构建器属性，类似always但参数是生成器函数
    
    通过mobject.f_always.method(func1, func2, ...)调用的方法
    将在每一帧使用生成器函数的返回值作为参数调用原方法
    """
    return _FunctionalUpdaterBuilder(self)

def note_changed_data(self, recurse_up: bool = True) -> Self:
    """
    标记数据已更改，触发重新渲染
    
    Args:
        recurse_up: 是否向上递归通知父对象
        
    Returns:
        对象本身（支持方法链）
    """
    self._data_has_changed = True
    if recurse_up:
        # 通知所有父对象数据已更改
        for mob in self.parents:
            mob.note_changed_data()
    return self

@staticmethod
def affects_data(func: Callable[..., T]) -> Callable[..., T]:
    """
    装饰器：标记修改数据的方法，自动触发数据更改通知
    
    Args:
        func: 要装饰的方法
        
    Returns:
        包装后的方法
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        # 调用方法后标记数据已更改
        self.note_changed_data()
        return result
    return wrapper

@staticmethod
def affects_family_data(func: Callable[..., T]) -> Callable[..., T]:
    """
    装饰器：标记修改家族数据的方法，自动触发家族所有对象的数据更改通知
    
    Args:
        func: 要装饰的方法
        
    Returns:
        包装后的方法
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        # 通知家族中所有有数据点的对象数据已更改
        for mob in self.family_members_with_points():
            mob.note_changed_data()
        return result
    return wrapper

@affects_data
def set_data(self, data: np.ndarray) -> Self:
    """
    设置对象的完整数据数组
    
    Args:
        data: 新的数据数组，必须与对象的数据类型匹配
        
    Returns:
        对象本身（支持方法链）
        
    Raises:
        AssertionError: 如果数据类型不匹配
    """
    assert data.dtype == self.data.dtype
    self.resize_points(len(data))
    self.data[:] = data
    return self

@affects_data
def resize_points(
    self,
    new_length: int,
    resize_func: Callable[[np.ndarray, int], np.ndarray] = resize_array
) -> Self:
    """
    调整点数据的长度
    
    Args:
        new_length: 新的长度
        resize_func: 用于调整数组大小的函数
        
    Returns:
        对象本身（支持方法链）
    """
    if new_length == 0:
        if len(self.data) > 0:
            # 保存当前数据的第一个元素作为默认值
            self._data_defaults[:1] = self.data[:1]
    elif self.get_num_points() == 0:
        # 如果当前没有数据，使用默认值初始化
        self.data = self._data_defaults.copy()

    # 调整数据数组大小
    self.data = resize_func(self.data, new_length)
    # 刷新边界框
    self.refresh_bounding_box()
    return self

@affects_data
def set_points(self, points: Vect3Array | list[Vect3]) -> Self:
    """
    设置对象的点坐标数据
    
    Args:
        points: 新的点坐标数组或列表
        
    Returns:
        对象本身（支持方法链）
    """
    # 调整点数量并保持顺序
    self.resize_points(len(points), resize_func=resize_preserving_order)
    # 设置点坐标
    self.data["point"][:] = points
    return self

@affects_data
def append_points(self, new_points: Vect3Array) -> Self:
    """
    向对象添加新的点坐标
    
    Args:
        new_points: 要添加的点坐标数组
        
    Returns:
        对象本身（支持方法链）
    """
    n = self.get_num_points()
    # 调整点数量以容纳新点
    self.resize_points(n + len(new_points))
    # 新点的其他数据默认使用最后一个现有点的值
    self.data[n:] = self.data[n - 1]
    # 设置新点的坐标
    self.data["point"][n:] = new_points
    # 刷新边界框
    self.refresh_bounding_box()
    return self

@affects_family_data
def reverse_points(self) -> Self:
    """
    反转家族中所有对象的点顺序
    
    Returns:
        对象本身（支持方法链）
    """
    for mob in self.get_family():
        mob.data[:] = mob.data[::-1]
    return self

@affects_family_data
def apply_points_function(
    self,
    func: Callable[[np.ndarray], np.ndarray],
    about_point: Vect3 | None = None,
    about_edge: Vect3 = ORIGIN,
    works_on_bounding_box: bool = False
) -> Self:
    """
    对家族中所有对象的点数据应用变换函数
    
    Args:
        func: 要应用于点数据的变换函数
        about_point: 变换的参考点，None则使用about_edge计算
        about_edge: 用于计算参考点的边界框边缘
        works_on_bounding_box: 函数是否直接作用于边界框
        
    Returns:
        对象本身（支持方法链）
    """
    # 如果未指定参考点，使用边界框的指定边缘点作为参考点
    if about_point is None and about_edge is not None:
        about_point = self.get_bounding_box_point(about_edge)

    # 对家族中所有对象应用变换
    for mob in self.get_family():
        # 获取所有类点数据（通常包括point字段）
        arrs = [mob.data[key] for key in mob.pointlike_data_keys if mob.has_points()]
        # 如果需要处理边界框，将边界框也加入处理列表
        if works_on_bounding_box:
            arrs.append(mob.get_bounding_box())

        # 对每个数组应用变换函数
        for arr in arrs:
            if about_point is None:
                arr[:] = func(arr)
            else:
                # 围绕参考点进行变换：先平移到原点，应用变换，再平移回原位置
                arr[:] = func(arr - about_point) + about_point

    # 根据是否直接处理边界框决定如何刷新边界框
    if not works_on_bounding_box:
        self.refresh_bounding_box(recurse_down=True)
    else:
        for parent in self.parents:
            parent.refresh_bounding_box()
    return self

@affects_data
def match_points(self, mobject: Mobject) -> Self:
    """
    匹配另一个Mobject的点数据
    
    Args:
        mobject: 要匹配的Mobject
        
    Returns:
        对象本身（支持方法链）
    """
    # 调整点数量以匹配目标对象
    self.resize_points(len(mobject.data), resize_func=resize_preserving_order)
    # 复制所有类点数据
    for key in self.pointlike_data_keys:
        self.data[key][:] = mobject.data[key]
    return self

    # Others related to points

def get_points(self) -> Vect3Array:
    # 返回当前Mobject的点数据
    return self.data["point"]

def clear_points(self) -> Self:
    # 清除所有点数据（通过将点数量调整为0）
    self.resize_points(0)
    # 返回自身以支持链式调用
    return self

def get_num_points(self) -> int:
    # 返回点的数量
    return len(self.get_points())

def get_all_points(self) -> Vect3Array:
    # 如果存在子对象
    if self.submobjects:
        # 垂直堆叠所有家族成员（包括自身和所有子对象）的点数据
        return np.vstack([sm.get_points() for sm in self.get_family()])
    else:
        # 如果没有子对象，直接返回自身的点数据
        return self.get_points()

def has_points(self) -> bool:
    # 检查是否包含任何点数据
    return len(self.get_points()) > 0

def get_bounding_box(self) -> Vect3Array:
    # 如果需要更新边界框
    if self._needs_new_bounding_box:
        # 计算新的边界框并更新
        self.bounding_box[:] = self.compute_bounding_box()
        # 标记为不需要更新
        self._needs_new_bounding_box = False
    # 返回当前边界框
    return self.bounding_box

def compute_bounding_box(self) -> Vect3Array:
    # 收集自身的点数据和所有子对象的边界框
    all_points = np.vstack([
        self.get_points(),
        *(
            mob.get_bounding_box()
            for mob in self.get_family()[1:]  # 从家族中排除自身
            if mob.has_points()  # 只包含有数据点的对象
        )
    ])
    # 如果没有任何点数据，返回零矩阵
    if len(all_points) == 0:
        return np.zeros((3, self.dim))
    else:
        # 计算最小点、最大点和中点
        mins = all_points.min(0)  # 各维度最小值
        maxs = all_points.max(0)  # 各维度最大值
        mids = (mins + maxs) / 2  # 各维度中点
        # 返回包含最小点、中点和最大点的边界框
        return np.array([mins, mids, maxs])

def refresh_bounding_box(
    self,
    recurse_down: bool = False,
    recurse_up: bool = True
) -> Self:
    # 标记家族中所有对象需要更新边界框（如果需要向下递归）
    for mob in self.get_family(recurse_down):
        mob._needs_new_bounding_box = True
    # 如果需要向上递归，通知所有父对象刷新边界框
    if recurse_up:
        for parent in self.parents:
            parent.refresh_bounding_box()
    # 返回自身以支持链式调用
    return self

def are_points_touching(
    self,
    points: Vect3Array,
    buff: float = 0
) -> np.ndarray:
    # 获取当前对象的边界框
    bb = self.get_bounding_box()
    # 计算考虑缓冲值的最小和最大边界
    mins = (bb[0] - buff)
    maxs = (bb[2] + buff)
    # 检查每个点是否在边界范围内（返回布尔数组）
    return ((points >= mins) * (points <= maxs)).all(1)

def is_point_touching(
    self,
    point: Vect3,
    buff: float = 0
) -> bool:
    # 检查单个点是否在边界范围内（将点转换为二维数组后调用are_points_touching）
    return self.are_points_touching(np.array(point, ndmin=2), buff)[0]

def is_touching(self, mobject: Mobject, buff: float = 1e-2) -> bool:
    # 获取当前对象和目标对象的边界框
    bb1 = self.get_bounding_box()
    bb2 = mobject.get_bounding_box()
    # 检查两个边界框是否有重叠（没有分离）
    return not any((
        (bb2[2] < bb1[0] - buff).any(),  # 目标对象右边界在当前对象左边界左侧
        (bb2[0] > bb1[2] + buff).any(),  # 目标对象左边界在当前对象右边界右侧
    ))

    # Family matters

def __getitem__(self, value: int | slice) -> Mobject:
    # 支持通过索引或切片访问子对象
    if isinstance(value, slice):
        # 如果是切片，创建一个包含对应子对象的组
        GroupClass = self.get_group_class()
        return GroupClass(*self.split().__getitem__(value))
    # 如果是索引，直接返回对应子对象
    return self.split().__getitem__(value)

def __iter__(self) -> Iterator[Self]:
    # 支持迭代子对象
    return iter(self.split())

def __len__(self) -> int:
    # 返回子对象的数量
    return len(self.split())

def split(self) -> list[Self]:
    # 返回所有子对象的列表
    return self.submobjects

@affects_data
def note_changed_family(self, only_changed_order=False) -> Self:
    # 标记家族关系已更改（清除缓存的家族列表）
    self.family = None
    # 如果不只是顺序改变，更新相关状态
    if not only_changed_order:
        self.refresh_has_updater_status()
        self.refresh_bounding_box()
    # 通知所有父对象家族关系已更改
    for parent in self.parents:
        parent.note_changed_family()
    # 返回自身以支持链式调用
    return self

def get_family(self, recurse: bool = True) -> list[Mobject]:
    # 如果不需要递归，只返回自身
    if not recurse:
        return [self]
    # 如果家族列表未缓存，重新构建
    if self.family is None:
        # 递归获取所有子对象的家族成员
        sub_families = (sm.get_family() for sm in self.submobjects)
        # 构建包含自身和所有子对象家族成员的列表
        self.family = [self, *it.chain(*sub_families)]
    # 返回完整的家族成员列表
    return self.family

def family_members_with_points(self) -> list[Mobject]:
    # 返回家族中所有有数据的成员（数据不为空的Mobject）
    return [m for m in self.get_family() if len(m.data) > 0]

def get_ancestors(self, extended: bool = False) -> list[Mobject]:
    """
    返回父对象、祖父对象等祖先对象。
    结果顺序应为从层级结构中较高的成员到较低的成员。

    如果extended设为True，将包含所有家族成员的祖先，
    例如子对象的其他父对象
    """
    ancestors = []
    # 需要处理的对象列表（如果extended为True，则包含所有家族成员）
    to_process = list(self.get_family(recurse=extended))
    # 排除自身及子对象（避免循环引用）
    excluded = set(to_process)
    # 遍历处理所有需要处理的对象
    while to_process:
        # 取出最后一个对象并获取其所有父对象
        for p in to_process.pop().parents:
            # 如果父对象不在排除列表中
            if p not in excluded:
                # 添加到祖先列表
                ancestors.append(p)
                # 将该父对象加入待处理列表，以便查找其上一级祖先
                to_process.append(p)
    # 反转列表，使层级最高的祖先排在前面
    ancestors.reverse()
    # 去除重复项同时保持顺序
    return list(dict.fromkeys(ancestors))

def add(self, *mobjects: Mobject) -> Self:
    # 检查是否尝试添加自身，不允许自包含
    if self in mobjects:
        raise Exception("Mobject cannot contain self")
    # 遍历所有要添加的对象
    for mobject in mobjects:
        # 如果对象不在子对象列表中，则添加
        if mobject not in self.submobjects:
            self.submobjects.append(mobject)
        # 如果当前对象不在该子对象的父列表中，则添加
        if self not in mobject.parents:
            mobject.parents.append(self)
    # 通知家族关系已更改
    self.note_changed_family()
    # 返回自身以支持链式调用
    return self

def remove(
    self,
    *to_remove: Mobject,
    reassemble: bool = True,
    recurse: bool = True
) -> Self:
    # 遍历当前对象家族中的所有成员（根据recurse参数决定是否递归）
    for parent in self.get_family(recurse):
        # 遍历所有要移除的对象
        for child in to_remove:
            # 如果子对象在当前父对象的子列表中，则移除
            if child in parent.submobjects:
                parent.submobjects.remove(child)
            # 如果当前父对象在子对象的父列表中，则移除
            if parent in child.parents:
                child.parents.remove(parent)
        # 如果需要重新组装，通知家族关系已更改
        if reassemble:
            parent.note_changed_family()
    # 返回自身以支持链式调用
    return self

def clear(self) -> Self:
    # 移除所有子对象（不递归）
    self.remove(*self.submobjects, recurse=False)
    # 返回自身以支持链式调用
    return self

def add_to_back(self, *mobjects: Mobject) -> Self:
    # 将新对象添加到子对象列表的前面（显示在后方）
    self.set_submobjects(list_update(mobjects, self.submobjects))
    # 返回自身以支持链式调用
    return self

def replace_submobject(self, index: int, new_submob: Mobject) -> Self:
    # 获取指定索引处的旧子对象
    old_submob = self.submobjects[index]
    # 如果当前对象在旧子对象的父列表中，则移除
    if self in old_submob.parents:
        old_submob.parents.remove(self)
    # 替换子对象列表中指定索引处的对象
    self.submobjects[index] = new_submob
    # 将当前对象添加到新子对象的父列表中
    new_submob.parents.append(self)
    # 通知家族关系已更改
    self.note_changed_family()
    # 返回自身以支持链式调用
    return self

def insert_submobject(self, index: int, new_submob: Mobject) -> Self:
    # 在指定索引处插入新子对象
    self.submobjects.insert(index, new_submob)
    # 通知家族关系已更改
    self.note_changed_family()
    # 返回自身以支持链式调用
    return self

def set_submobjects(self, submobject_list: list[Mobject]) -> Self:
    # 如果子对象列表没有变化，则直接返回
    if self.submobjects == submobject_list:
        return self
    # 清除现有子对象
    self.clear()
    # 添加新的子对象列表
    self.add(*submobject_list)
    # 返回自身以支持链式调用
    return self

def digest_mobject_attrs(self) -> Self:
    """
    确保所有作为Mobject类型的属性都包含在子对象列表中。
    """
    # 获取所有类型为Mobject的属性值
    mobject_attrs = [x for x in list(self.__dict__.values()) if isinstance(x, Mobject)]
    # 更新子对象列表，添加所有Mobject类型的属性
    self.set_submobjects(list_update(self.submobjects, mobject_attrs))
    # 返回自身以支持链式调用
    return self

# 子对象排列方法

def arrange(
    self,
    direction: Vect3 = RIGHT,
    center: bool = True,** kwargs
) -> Self:
    # 遍历子对象列表，将每个对象排列在前一个对象的指定方向
    for m1, m2 in zip(self.submobjects, self.submobjects[1:]):
        m2.next_to(m1, direction, **kwargs)
    # 如果需要居中，将整个组居中
    if center:
        self.center()
    # 返回自身以支持链式调用
    return self

def arrange_in_grid(
    self,
    n_rows: int | None = None,
    n_cols: int | None = None,
    buff: float | None = None,
    h_buff: float | None = None,
    v_buff: float | None = None,
    buff_ratio: float | None = None,
    h_buff_ratio: float = 0.5,
    v_buff_ratio: float = 0.5,
    aligned_edge: Vect3 = ORIGIN,
    fill_rows_first: bool = True
) -> Self:
    # 获取子对象列表
    submobs = self.submobjects
    # 子对象数量
    n_submobs = len(submobs)
    # 如果未指定行数，则根据列数或平方根计算行数
    if n_rows is None:
        n_rows = int(np.sqrt(n_submobs)) if n_cols is None else n_submobs // n_cols
    # 如果未指定列数，则根据行数计算列数
    if n_cols is None:
        n_cols = n_submobs // n_rows

    # 处理缓冲参数
    if buff is not None:
        h_buff = buff
        v_buff = buff
    else:
        if buff_ratio is not None:
            v_buff_ratio = buff_ratio
            h_buff_ratio = buff_ratio
        # 如果未指定水平缓冲，则使用第一个子对象宽度的比例
        if h_buff is None:
            h_buff = h_buff_ratio * self[0].get_width()
        # 如果未指定垂直缓冲，则使用第一个子对象高度的比例
        if v_buff is None:
            v_buff = v_buff_ratio * self[0].get_height()

    # 计算网格中每个单元格的宽度和高度（包含缓冲）
    x_unit = h_buff + max([sm.get_width() for sm in submobs])
    y_unit = v_buff + max([sm.get_height() for sm in submobs])

    # 遍历每个子对象并放置在网格中
    for index, sm in enumerate(submobs):
        # 根据填充顺序计算x和y坐标
        if fill_rows_first:
            # 先填充行，再填充列
            x, y = index % n_cols, index // n_cols
        else:
            # 先填充列，再填充行
            x, y = index // n_rows, index % n_rows
        # 将子对象移动到原点并对齐到指定边缘
        sm.move_to(ORIGIN, aligned_edge)
        # 根据计算的坐标偏移子对象
        sm.shift(x * x_unit * RIGHT + y * y_unit * DOWN)
    # 将整个网格居中
    self.center()
    # 返回自身以支持链式调用
    return self

def arrange_to_fit_dim(self, length: float, dim: int, about_edge=ORIGIN) -> Self:
    # 获取参考点（基于指定边缘）
    ref_point = self.get_bounding_box_point(about_edge)
    # 子对象数量
    n_submobs = len(self.submobjects)
    # 如果子对象数量小于等于1，则无需排列
    if n_submobs <= 1:
        return
    # 计算所有子对象在指定维度上的总长度
    total_length = sum(sm.length_over_dim(dim) for sm in self.submobjects)
    # 计算每个子对象之间的缓冲
    buff = (length - total_length) / (n_submobs - 1)
    # 创建指定维度的单位向量
    vect = np.zeros(self.dim)
    vect[dim] = 1
    # 初始位置
    x = 0
    # 遍历子对象并设置位置
    for submob in self.submobjects:
        submob.set_coord(x, dim, -vect)
        # 更新下一个子对象的位置（当前长度+缓冲）
        x += submob.length_over_dim(dim) + buff
    # 将整个组移动回参考点（保持对齐）
    self.move_to(ref_point, about_edge)
    # 返回自身以支持链式调用
    return self

def arrange_to_fit_width(self, width: float, about_edge=ORIGIN) -> Self:
    # 调用arrange_to_fit_dim方法，指定宽度方向(0维)进行排列调整
    return self.arrange_to_fit_dim(width, 0, about_edge)

def arrange_to_fit_height(self, height: float, about_edge=ORIGIN) -> Self:
    # 调用arrange_to_fit_dim方法，指定高度方向(1维)进行排列调整
    return self.arrange_to_fit_dim(height, 1, about_edge)

def arrange_to_fit_depth(self, depth: float, about_edge=ORIGIN) -> Self:
    # 调用arrange_to_fit_dim方法，指定深度方向(2维)进行排列调整
    return self.arrange_to_fit_dim(depth, 2, about_edge)

def sort(
    self,
    point_to_num_func: Callable[[np.ndarray], float] = lambda p: p[0],
    submob_func: Callable[[Mobject]] | None = None
) -> Self:
    # 如果提供了子对象排序函数，则使用该函数对submobjects进行排序
    if submob_func is not None:
        self.submobjects.sort(key=submob_func)
    # 否则使用默认的中心点排序函数（按x坐标排序）
    else:
        self.submobjects.sort(key=lambda m: point_to_num_func(m.get_center()))
    # 通知家族成员顺序已改变（仅顺序改变）
    self.note_changed_family(only_changed_order=True)
    return self

def shuffle(self, recurse: bool = False) -> Self:
    # 如果需要递归打乱，则对每个子对象也执行shuffle
    if recurse:
        for submob in self.submobjects:
            submob.shuffle(recurse=True)
    # 随机打乱当前对象的submobjects顺序
    random.shuffle(self.submobjects)
    # 通知家族成员顺序已改变（仅顺序改变）
    self.note_changed_family(only_changed_order=True)
    return self

def reverse_submobjects(self) -> Self:
    # 反转submobjects列表的顺序
    self.submobjects.reverse()
    # 通知家族成员顺序已改变（仅顺序改变）
    self.note_changed_family(only_changed_order=True)
    return self

# 复制和序列化相关方法

@staticmethod
def stash_mobject_pointers(func: Callable[..., T]) -> Callable[..., T]:
    # 装饰器：用于在序列化/深拷贝前暂存不需要复制的Mobject指针属性
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # 不需要复制的属性列表
        uncopied_attrs = ["parents", "target", "saved_state"]
        # 用于暂存属性值的字典
        stash = dict()
        # 暂存并清空指定属性
        for attr in uncopied_attrs:
            if hasattr(self, attr):
                value = getattr(self, attr)
                stash[attr] = value
                # 根据属性类型设置空值（列表类型用空列表，其他用None）
                null_value = [] if isinstance(value, list) else None
                setattr(self, attr, null_value)
        # 执行被装饰的函数（序列化/深拷贝）
        result = func(self, *args, **kwargs)
        # 恢复暂存的属性值
        self.__dict__.update(stash)
        return result
    return wrapper

@stash_mobject_pointers
def serialize(self) -> bytes:
    # 使用pickle序列化当前对象，返回字节流
    return pickle.dumps(self)

def deserialize(self, data: bytes) -> Self:
    # 从字节流反序列化对象，并替换当前对象的内容
    self.become(pickle.loads(data))
    return self

@stash_mobject_pointers
def deepcopy(self) -> Self:
    # 创建当前对象的深拷贝
    return copy.deepcopy(self)

def copy(self, deep: bool = False) -> Self:
    # 如果需要深拷贝，调用deepcopy方法
    if deep:
        return self.deepcopy()

    # 否则进行浅拷贝
    result = copy.copy(self)

    # 重置父对象、目标和保存状态
    result.parents = []
    result.target = None
    result.saved_state = None

    # 对uniforms中的numpy数组进行拷贝，其他值保持引用
    result.uniforms = {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in self.uniforms.items()
    }

    # 直接复制子对象列表（不使用add方法以避免额外检查）
    result.submobjects = [sm.copy() for sm in self.submobjects]
    # 更新子对象的父指针为当前复制出的对象
    for sm in result.submobjects:
        sm.parents = [result]
    # 构建家族成员列表
    result.family = [result, *it.chain(*(sm.get_family() for sm in result.submobjects))]

    # 复制更新器列表
    result.updaters = list(self.updaters)
    # 标记数据已更改
    result._data_has_changed = True
    # 重置着色器包装器
    result.shader_wrapper = None

    # 处理其他属性的拷贝
    family = self.get_family()
    for attr, value in self.__dict__.items():
        # 如果属性是家族中的Mobject，替换为对应拷贝对象
        if isinstance(value, Mobject) and value is not self:
            if value in family:
                setattr(result, attr, result.family[family.index(value)])
        # 如果属性是numpy数组，进行拷贝
        elif isinstance(value, np.ndarray):
            setattr(result, attr, value.copy())
    return result

def generate_target(self, use_deepcopy: bool = False) -> Self:
    # 生成当前对象的目标对象（用于动画过渡等场景）
    # 使用指定的复制方式（深拷贝或浅拷贝）创建副本作为目标对象
    self.target = self.copy(deep=use_deepcopy)
    # 将当前对象的保存状态同步到目标对象
    self.target.saved_state = self.saved_state
    return self.target

def save_state(self, use_deepcopy: bool = False) -> Self:
    # 保存当前对象的状态，用于后续恢复
    # 使用指定的复制方式创建当前状态的副本
    self.saved_state = self.copy(deep=use_deepcopy)
    # 将当前的目标对象引用同步到保存的状态中
    self.saved_state.target = self.target
    return self

def restore(self) -> Self:
    # 从之前保存的状态恢复对象
    # 检查是否有保存的状态，没有则抛出异常
    if not hasattr(self, "saved_state") or self.saved_state is None:
        raise Exception("Trying to restore without having saved")
    # 使当前对象变成保存状态的副本
    self.become(self.saved_state)
    return self

def become(self, mobject: Mobject, match_updaters=False) -> Self:
    """
    编辑所有数据和子对象，使其与另一个mobject完全相同
    """
    # 对齐两个对象的家族结构（确保子对象层级一致）
    self.align_family(mobject)
    # 获取两个对象的家族成员列表
    family1 = self.get_family()
    family2 = mobject.get_family()
    # 逐个同步家族成员的属性
    for sm1, sm2 in zip(family1, family2):
        sm1.set_data(sm2.data)               # 同步数据
        sm1.set_uniforms(sm2.uniforms)       # 同步 uniforms
        sm1.bounding_box[:] = sm2.bounding_box  # 同步边界框
        sm1.shader_folder = sm2.shader_folder  # 同步着色器文件夹
        sm1.texture_paths = sm2.texture_paths  # 同步纹理路径
        sm1.depth_test = sm2.depth_test      # 同步深度测试设置
        sm1.render_primitive = sm2.render_primitive  # 同步渲染图元
        sm1._needs_new_bounding_box = sm2._needs_new_bounding_box  # 同步边界框更新标记
    # 确保命名的家族成员引用正确传递
    for attr, value in list(mobject.__dict__.items()):
        if isinstance(value, Mobject) and value in family2:
            # 将属性引用映射到当前对象家族中对应的成员
            setattr(self, attr, family1[family2.index(value)])
    # 如果需要，同步更新器
    if match_updaters:
        self.match_updaters(mobject)
    return self

def looks_identical(self, mobject: Mobject) -> bool:
    # 检查当前对象与另一个对象是否看起来完全相同
    # 获取包含点数据的家族成员
    fam1 = self.family_members_with_points()
    fam2 = mobject.family_members_with_points()
    # 家族成员数量不同则不相同
    if len(fam1) != len(fam2):
        return False
    # 逐个检查家族成员的属性
    for m1, m2 in zip(fam1, fam2):
        # 点数量不同则不相同
        if m1.get_num_points() != m2.get_num_points():
            return False
        # 数据类型不同则不相同
        if not m1.data.dtype == m2.data.dtype:
            return False
        # 检查数据中每个字段是否接近
        for key in m1.data.dtype.names:
            if not np.isclose(m1.data[key], m2.data[key]).all():
                return False
        # 检查uniforms的键是否一致
        if set(m1.uniforms).difference(m2.uniforms):
            return False
        # 检查每个uniform的值是否接近
        for key in m1.uniforms:
            value1 = m1.uniforms[key]
            value2 = m2.uniforms[key]
            # 数组大小不同则不相同
            if isinstance(value1, np.ndarray) and isinstance(value2, np.ndarray) and not value1.size == value2.size:
                return False
            # 值不接近则不相同
            if not np.isclose(value1, value2).all():
                return False
    # 所有检查通过，认为相同
    return True

def has_same_shape_as(self, mobject: Mobject) -> bool:
    # 检查当前对象与另一个对象是否具有相同的形状（忽略位置和大小差异）
    # 通过中心化和归一化高度来标准化点集
    points1, points2 = (
        (m.get_all_points() - m.get_center()) / m.get_height()
        for m in (self, mobject)
    )
    # 点数量不同则形状不同
    if len(points1) != len(points2):
        return False
    # 检查标准化后的点是否接近（容差为宽度的1%）
    return bool(np.isclose(points1, points2, atol=self.get_width() * 1e-2).all())

    # Creating new Mobjects from this one

    def replicate(self, n: int) -> Self:
        group_class = self.get_group_class()
        return group_class(*(self.copy() for _ in range(n)))

def replicate(self, n: int) -> Self:
    # 获取当前对象的组类（用于创建新的组）
    group_class = self.get_group_class()
    # 创建n个当前对象的副本，并使用组类包装返回
    return group_class(*(self.copy() for _ in range(n)))

def get_grid(
    self,
    n_rows: int,
    n_cols: int,
    height: float | None = None,
    width: float | None = None,
    group_by_rows: bool = False,
    group_by_cols: bool = False,** kwargs
) -> Self:
    """
    返回一个包含当前对象多个副本的新对象，这些副本排列成网格状
    """
    # 计算网格中对象的总数
    total = n_rows * n_cols
    # 复制当前对象total次，形成网格的基础元素
    grid = self.replicate(total)
    # 如果按列分组，设置填充方式为先填充列
    if group_by_cols:
        kwargs["fill_rows_first"] = False
    # 将复制的对象排列成指定行列的网格
    grid.arrange_in_grid(n_rows, n_cols, **kwargs)
    # 如果指定了高度，设置网格的总高度
    if height is not None:
        grid.set_height(height)
    # 如果指定了宽度，设置网格的总宽度（这里原文可能笔误，应该是set_width）
    if width is not None:
        grid.set_height(width)

    # 获取组类用于后续分组
    group_class = self.get_group_class()
    # 如果按行分组，将网格按行分割并包装成组
    if group_by_rows:
        return group_class(*(grid[n:n + n_cols] for n in range(0, total, n_cols)))
    # 如果按列分组，将网格按列分割并包装成组
    elif group_by_cols:
        return group_class(*(grid[n:n + n_rows] for n in range(0, total, n_rows)))
    # 否则直接返回排列好的网格
    else:
        return grid

    # Updating

def init_updaters(self):
    # 初始化更新器相关属性
    self.updaters: list[Updater] = list()  # 存储更新器函数的列表
    self._has_updaters_in_family: Optional[bool] = False  # 标记家族中是否有更新器
    self.updating_suspended: bool = False  # 标记更新是否被暂停

def update(self, dt: float = 0, recurse: bool = True) -> Self:
    # 如果没有更新器或更新被暂停，则直接返回
    if not self.has_updaters() or self.updating_suspended:
        return self
    # 如果需要递归更新，先更新所有子对象
    if recurse:
        for submob in self.submobjects:
            submob.update(dt, recurse)
    # 执行当前对象的所有更新器
    for updater in self.updaters:
        # 检查更新器是否接受dt参数，如果是则传入时间增量
        if "dt" in updater.__code__.co_varnames:
            updater(self, dt=dt)
        # 否则直接调用更新器
        else:
            updater(self)
    return self

def get_updaters(self) -> list[Updater]:
    # 返回当前对象的更新器列表
    return self.updaters

def add_updater(self, update_func: Updater, call: bool = True) -> Self:
    # 向更新器列表添加新的更新函数
    self.updaters.append(update_func)
    # 如果需要，立即调用一次更新（时间增量为0）
    if call:
        self.update(dt=0)
    # 刷新更新器状态标记
    self.refresh_has_updater_status()
    # 执行一次更新
    self.update()
    return self

def insert_updater(self, update_func: Updater, index=0):
    # 在指定位置插入更新函数
    self.updaters.insert(index, update_func)
    # 刷新更新器状态标记
    self.refresh_has_updater_status()
    return self

def remove_updater(self, update_func: Updater) -> Self:
    # 移除所有匹配的更新函数
    while update_func in self.updaters:
        self.updaters.remove(update_func)
    # 刷新更新器状态标记
    self.refresh_has_updater_status()
    return self

def clear_updaters(self, recurse: bool = True) -> Self:
    # 清除家族中所有对象的更新器（根据recurse参数决定是否递归）
    for mob in self.get_family(recurse):
        mob.updaters = []
        mob._has_updaters_in_family = False
    # 刷新祖先的更新器状态标记
    for parent in self.get_ancestors():
        parent._has_updaters_in_family = False
    return self

def match_updaters(self, mobject: Mobject) -> Self:
    # 将当前对象的更新器设置为与目标对象相同
    self.updaters = list(mobject.updaters)
    # 刷新更新器状态标记
    self.refresh_has_updater_status()
    return self

def suspend_updating(self, recurse: bool = True) -> Self:
    # 暂停当前对象的更新
    self.updating_suspended = True
    # 如果需要递归，暂停所有子对象的更新
    if recurse:
        for submob in self.submobjects:
            submob.suspend_updating(recurse)
    return self

def resume_updating(self, recurse: bool = True, call_updater: bool = True) -> Self:
    # 恢复当前对象的更新
    self.updating_suspended = False
    # 如果需要递归，恢复所有子对象的更新
    if recurse:
        for submob in self.submobjects:
            submob.resume_updating(recurse)
    # 恢复所有父对象的更新（不递归，不立即调用更新器）
    for parent in self.parents:
        parent.resume_updating(recurse=False, call_updater=False)
    # 如果需要，立即调用一次更新
    if call_updater:
        self.update(dt=0, recurse=recurse)
    return self

def has_updaters(self) -> bool:
    # 检查是否有更新器（包括子对象的）
    if self._has_updaters_in_family is None:
        # 重新计算并保存状态：如果自身有更新器或任何子对象有更新器，则返回True
        self._has_updaters_in_family = bool(self.updaters) or any(
            sm.has_updaters() for sm in self.submobjects
        )
    return self._has_updaters_in_family

def refresh_has_updater_status(self) -> Self:
    # 刷新更新器状态标记（设为None将触发重新计算）
    self._has_updaters_in_family = None
    # 递归刷新所有父对象的状态
    for parent in self.parents:
        parent.refresh_has_updater_status()
    return self

# 检查是否标记为静态或用于相机

def is_changing(self) -> bool:
    # 检查对象是否正在变化（处于动画中或有更新器）
    return self._is_animating or self.has_updaters()

def set_animating_status(self, is_animating: bool, recurse: bool = True) -> Self:
    # 设置对象的动画状态（是否正在动画中）
    # 对家族成员和祖先应用此状态
    for mob in (*self.get_family(recurse), *self.get_ancestors()):
        mob._is_animating = is_animating
    return self

# 变换操作

def shift(self, vector: Vect3) -> Self:
    # 平移对象：将所有点加上指定向量
    self.apply_points_function(
        lambda points: points + vector,
        about_edge=None,
        works_on_bounding_box=True,  # 同时更新边界框
    )
    return self

def scale(
    self,
    scale_factor: float | npt.ArrayLike,
    min_scale_factor: float = 1e-8,
    about_point: Vect3 | None = None,
    about_edge: Vect3 = ORIGIN
) -> Self:
    """
    默认行为是围绕对象的中心缩放。
    about_edge参数可以是一个向量，表示围绕对象的哪个边缘缩放，
    例如，mob.scale(about_edge = RIGHT) 围绕mob.get_right()缩放。
    此外，如果指定了about_point，则围绕该点进行缩放。
    """
    # 确保缩放因子不小于最小缩放因子
    if isinstance(scale_factor, numbers.Number):
        scale_factor = max(scale_factor, min_scale_factor)
    else:
        scale_factor = np.array(scale_factor).clip(min=min_scale_factor)
    # 应用缩放变换到所有点
    self.apply_points_function(
        lambda points: scale_factor * points,
        about_point=about_point,
        about_edge=about_edge,
        works_on_bounding_box=True,  # 同时更新边界框
    )
    # 处理缩放带来的副作用（子类可能需要重写此方法）
    for mob in self.get_family():
        mob._handle_scale_side_effects(scale_factor)
    return self

def _handle_scale_side_effects(self, scale_factor):
    # 处理缩放带来的副作用
    # 供子类（如DecimalNumber）重写，在尺寸改变时进行其他调整
    pass

def stretch(self, factor: float, dim: int, **kwargs) -> Self:
    # 在指定维度上拉伸对象
    def func(points):
        points[:, dim] *= factor  # 仅在指定维度上应用拉伸因子
        return points
    # 应用拉伸变换
    self.apply_points_function(func, works_on_bounding_box=True,** kwargs)
    return self

def rotate_about_origin(self, angle: float, axis: Vect3 = OUT) -> Self:
    # 围绕原点旋转对象
    return self.rotate(angle, axis, about_point=ORIGIN)

def rotate(
    self,
    angle: float,
    axis: Vect3 = OUT,
    about_point: Vect3 | None = None,
    **kwargs
) -> Self:
    # 计算旋转矩阵的转置（用于点的旋转）
    rot_matrix_T = rotation_matrix_transpose(angle, axis)
    # 应用旋转变换：将点与旋转矩阵相乘
    self.apply_points_function(
        lambda points: np.dot(points, rot_matrix_T),
        about_point,** kwargs
    )
    return self

def flip(self, axis: Vect3 = UP, **kwargs) -> Self:
    # 翻转对象：本质是围绕指定轴旋转180度（TAU/2，TAU=2π）
    return self.rotate(TAU / 2, axis, **kwargs)

def apply_function(self, function: Callable[[np.ndarray], np.ndarray], **kwargs) -> Self:
    # 对对象的每个点应用自定义函数（默认围绕原点，而非对象中心）
    # 如果未指定变换中心点，默认使用原点
    if len(kwargs) == 0:
        kwargs["about_point"] = ORIGIN
    # 调用点处理函数，对每个点执行自定义function
    self.apply_points_function(
        lambda points: np.array([function(p) for p in points]),
        **kwargs
    )
    return self

def apply_function_to_position(self, function: Callable[[np.ndarray], np.ndarray]) -> Self:
    # 对对象的位置（中心点）应用自定义函数，移动对象到新位置
    # 计算新中心点：将原中心点传入function
    self.move_to(function(self.get_center()))
    return self

def apply_function_to_submobject_positions(
    self,
    function: Callable[[np.ndarray], np.ndarray]
) -> Self:
    # 对所有子对象的位置（中心点）应用自定义函数
    for submob in self.submobjects:
        submob.apply_function_to_position(function)
    return self

def apply_matrix(self, matrix: npt.ArrayLike, **kwargs) -> Self:
    # 对对象应用矩阵变换（默认围绕原点，而非对象中心）
    # 如果未指定变换中心点（about_point/about_edge），默认使用原点
    if ("about_point" not in kwargs) and ("about_edge" not in kwargs):
        kwargs["about_point"] = ORIGIN
    # 创建与对象维度匹配的单位矩阵（确保矩阵维度正确）
    full_matrix = np.identity(self.dim)
    matrix = np.array(matrix)
    # 将输入矩阵嵌入到单位矩阵中（处理低维矩阵适配高维对象）
    full_matrix[:matrix.shape[0], :matrix.shape[1]] = matrix
    # 应用矩阵变换：点与矩阵转置相乘（符合点的变换规则）
    self.apply_points_function(
        lambda points: np.dot(points, full_matrix.T),
        **kwargs
    )
    return self

def apply_complex_function(self, function: Callable[[complex], complex], **kwargs) -> Self:
    # 对对象应用复变函数（仅作用于XY平面，Z轴保持不变）
    def R3_func(point):
        # 提取点的XY坐标，转换为复数
        x, y, z = point
        xy_complex = function(complex(x, y))
        # 返回变换后的3D点（Z轴不变）
        return [
            xy_complex.real,  # 复数实部作为新X坐标
            xy_complex.imag,  # 复数虚部作为新Y坐标
            z                 # 保持Z坐标不变
        ]
    # 调用3D点处理函数执行变换
    return self.apply_function(R3_func, **kwargs)

def wag(
    self,
    direction: Vect3 = RIGHT,
    axis: Vect3 = DOWN,
    wag_factor: float = 1.0
) -> Self:
    # 使对象产生"摆动"效果（沿指定方向，按轴的分布梯度偏移）
    # 遍历所有包含点数据的家族成员
    for mob in self.family_members_with_points():
        # 计算每个点在指定轴上的投影值（用于生成偏移梯度）
        alphas = np.dot(mob.get_points(), np.transpose(axis))
        # 归一化投影值到[0,1]区间
        alphas -= min(alphas)
        alphas /= max(alphas)
        # 应用摆动因子调整偏移梯度（指数级调整，控制摆动幅度分布）
        alphas = alphas**wag_factor
        # 计算每个点的偏移向量：梯度值 × 摆动方向
        mob.set_points(mob.get_points() + np.dot(
            alphas.reshape((len(alphas), 1)),  # 梯度值转为列向量
            np.array(direction).reshape((1, mob.dim))  # 摆动方向转为行向量
        ))
    return self

# 定位相关方法

def center(self) -> Self:
    # 将对象移动到原点（通过平移抵消当前中心点坐标）
    self.shift(-self.get_center())
    return self

def align_on_border(
    self,
    direction: Vect3,
    buff: float = DEFAULT_MOBJECT_TO_EDGE_BUFF
) -> Self:
    """
    方向参数只需是2D平面中指向边缘或角落的向量即可。
    """
    # 计算目标边界点：根据方向向量的符号，获取帧的边缘/角落坐标
    target_point = np.sign(direction) * (FRAME_X_RADIUS, FRAME_Y_RADIUS, 0)
    # 获取对象上与目标方向对应的边界点（需对齐的点）
    point_to_align = self.get_bounding_box_point(direction)
    # 计算平移量：目标点 - 待对齐点 - 边界间距（避免紧贴边缘）
    shift_val = target_point - point_to_align - buff * np.array(direction)
    # 过滤无效维度的平移（仅保留方向向量非零的维度）
    shift_val = shift_val * abs(np.sign(direction))
    # 执行平移，将对象对齐到边界
    self.shift(shift_val)
    return self

def to_corner(
    self,
    corner: Vect3 = LEFT + DOWN,
    buff: float = DEFAULT_MOBJECT_TO_EDGE_BUFF
) -> Self:
    # 将对象移动到帧的角落（复用align_on_border，方向为角落向量）
    return self.align_on_border(corner, buff)

def to_edge(
    self,
    edge: Vect3 = LEFT,
    buff: float = DEFAULT_MOBJECT_TO_EDGE_BUFF
) -> Self:
    # 将对象移动到帧的边缘（复用align_on_border，方向为边缘向量）
    return self.align_on_border(edge, buff)

def next_to(
    self,
    mobject_or_point: Mobject | Vect3,
    direction: Vect3 = RIGHT,
    buff: float = DEFAULT_MOBJECT_TO_MOBJECT_BUFF,
    aligned_edge: Vect3 = ORIGIN,
    submobject_to_align: Mobject | None = None,
    index_of_submobject_to_align: int | slice | None = None,
    coor_mask: Vect3 = np.array([1, 1, 1]),
) -> Self:
    # 处理目标为Mobject的情况（获取目标对齐点）
    if isinstance(mobject_or_point, Mobject):
        mob = mobject_or_point
        # 根据子对象索引指定目标对齐器（子对象级对齐）
        if index_of_submobject_to_align is not None:
            target_aligner = mob[index_of_submobject_to_align]
        # 无索引则使用目标对象本身作为对齐器
        else:
            target_aligner = mob
        # 计算目标对齐点：目标对齐器的"aligned_edge + direction"边界点
        target_point = target_aligner.get_bounding_box_point(
            aligned_edge + direction
        )
    # 处理目标为点的情况（直接使用该点作为目标对齐点）
    else:
        target_point = mobject_or_point

    # 确定当前对象的对齐器（子对象级对齐）
    if submobject_to_align is not None:
        aligner = submobject_to_align
    elif index_of_submobject_to_align is not None:
        aligner = self[index_of_submobject_to_align]
    # 无指定则使用当前对象本身作为对齐器
    else:
        aligner = self

    # 计算当前对象的待对齐点：对齐器的"aligned_edge - direction"边界点
    point_to_align = aligner.get_bounding_box_point(aligned_edge - direction)

    # 计算平移量：(目标点 - 待对齐点 + 间距) × 坐标掩码（过滤无效维度）
    self.shift((target_point - point_to_align + buff * direction) * coor_mask)
    return self

def shift_onto_screen(self, **kwargs) -> Self:
    # 定义屏幕在X、Y轴的半长（用于计算屏幕边界）
    space_lengths = [FRAME_X_RADIUS, FRAME_Y_RADIUS]
    # 遍历四个方向（上、下、左、右），检查对象是否超出屏幕边界
    for vect in UP, DOWN, LEFT, RIGHT:
        # 确定当前方向对应的维度（X轴或Y轴）
        dim = np.argmax(np.abs(vect))
        # 从参数中获取边界间距，默认使用全局默认值
        buff = kwargs.get("buff", DEFAULT_MOBJECT_TO_EDGE_BUFF)
        # 计算当前方向的最大允许坐标（屏幕边界 - 间距）
        max_val = space_lengths[dim] - buff
        # 获取对象在当前方向上的边缘中心点
        edge_center = self.get_edge_center(vect)
        # 如果边缘中心点超出最大允许坐标，将对象对齐到该方向的屏幕边缘
        if np.dot(edge_center, vect) > max_val:
            self.to_edge(vect, **kwargs)
    return self

def is_off_screen(self) -> bool:
    # 检查对象左边缘是否超出屏幕右边界
    if self.get_left()[0] > FRAME_X_RADIUS:
        return True
    # 检查对象右边缘是否超出屏幕左边界
    if self.get_right()[0] < -FRAME_X_RADIUS:
        return True
    # 检查对象下边缘是否超出屏幕上边界
    if self.get_bottom()[1] > FRAME_Y_RADIUS:
        return True
    # 检查对象上边缘是否超出屏幕下边界
    if self.get_top()[1] < -FRAME_Y_RADIUS:
        return True
    # 所有边界均未超出，返回False
    return False

def stretch_about_point(self, factor: float, dim: int, point: Vect3) -> Self:
    # 围绕指定点在指定维度上拉伸对象（复用stretch方法，指定about_point参数）
    return self.stretch(factor, dim, about_point=point)

def stretch_in_place(self, factor: float, dim: int) -> Self:
    # 原地拉伸对象（当前已冗余，直接调用stretch即可，默认围绕对象中心拉伸）
    return self.stretch(factor, dim)

def rescale_to_fit(self, length: float, dim: int, stretch: bool = False, **kwargs) -> Self:
    # 获取对象在指定维度上的当前长度
    old_length = self.length_over_dim(dim)
    # 若当前长度为0，无需缩放，直接返回
    if old_length == 0:
        return self
    # 计算缩放因子：目标长度 / 当前长度
    scale_factor = length / old_length
    # 若指定拉伸模式，在指定维度上单独拉伸
    if stretch:
        self.stretch(scale_factor, dim,** kwargs)
    # 否则进行整体缩放（所有维度按同一比例缩放）
    else:
        self.scale(scale_factor, **kwargs)
    return self

def stretch_to_fit_width(self, width: float, **kwargs) -> Self:
    # 拉伸对象以适配指定宽度（X轴，维度0，强制拉伸模式）
    return self.rescale_to_fit(width, 0, stretch=True, **kwargs)

def stretch_to_fit_height(self, height: float, **kwargs) -> Self:
    # 拉伸对象以适配指定高度（Y轴，维度1，强制拉伸模式）
    return self.rescale_to_fit(height, 1, stretch=True, **kwargs)

def stretch_to_fit_depth(self, depth: float, **kwargs) -> Self:
    # 拉伸对象以适配指定深度（Z轴，维度2，强制拉伸模式）
    return self.rescale_to_fit(depth, 2, stretch=True, **kwargs)

def set_width(self, width: float, stretch: bool = False, **kwargs) -> Self:
    # 设置对象宽度（X轴，维度0），可选择拉伸或整体缩放
    return self.rescale_to_fit(width, 0, stretch=stretch, **kwargs)

def set_height(self, height: float, stretch: bool = False, **kwargs) -> Self:
    # 设置对象高度（Y轴，维度1），可选择拉伸或整体缩放
    return self.rescale_to_fit(height, 1, stretch=stretch, **kwargs)

def set_depth(self, depth: float, stretch: bool = False, **kwargs) -> Self:
    # 设置对象深度（Z轴，维度2），可选择拉伸或整体缩放
    return self.rescale_to_fit(depth, 2, stretch=stretch, **kwargs)

def set_max_width(self, max_width: float, **kwargs) -> Self:
    # 仅当对象当前宽度超过最大允许宽度时，将宽度设置为最大值
    if self.get_width() > max_width:
        self.set_width(max_width,** kwargs)
    return self

def set_max_height(self, max_height: float, **kwargs) -> Self:
    # 仅当对象当前高度超过最大允许高度时，将高度设置为最大值
    if self.get_height() > max_height:
        self.set_height(max_height, **kwargs)
    return self

def set_max_depth(self, max_depth: float, **kwargs) -> Self:
    # 仅当对象当前深度超过最大允许深度时，将深度设置为最大值
    if self.get_depth() > max_depth:
        self.set_depth(max_depth,** kwargs)
    return self

def set_min_width(self, min_width: float, **kwargs) -> Self:
    # 仅当对象当前宽度小于最小允许宽度时，将宽度设置为最小值
    if self.get_width() < min_width:
        self.set_width(min_width, **kwargs)
    return self

def set_min_height(self, min_height: float, **kwargs) -> Self:
    # 仅当对象当前高度小于最小允许高度时，将高度设置为最小值
    if self.get_height() < min_height:
        self.set_height(min_height,** kwargs)
    return self

def set_min_depth(self, min_depth: float, **kwargs) -> Self:
    # 仅当对象当前深度小于最小允许深度时，将深度设置为最小值
    if self.get_depth() < min_depth:
        self.set_depth(min_depth, **kwargs)
    return self

def set_shape(
    self,
    width: Optional[float] = None,
    height: Optional[float] = None,
    depth: Optional[float] = None,** kwargs
) -> Self:
    # 若指定了宽度，拉伸对象以适配该宽度
    if width is not None:
        self.set_width(width, stretch=True, **kwargs)
    # 若指定了高度，拉伸对象以适配该高度
    if height is not None:
        self.set_height(height, stretch=True, **kwargs)
    # 若指定了深度，拉伸对象以适配该深度
    if depth is not None:
        self.set_depth(depth, stretch=True, **kwargs)
    return self

def set_coord(self, value: float, dim: int, direction: Vect3 = ORIGIN) -> Self:
    # 获取对象在指定维度、指定方向上的当前坐标
    curr = self.get_coord(dim, direction)
    # 创建维度与对象匹配的零向量（用于存储平移量）
    shift_vect = np.zeros(self.dim)
    # 计算指定维度的平移量：目标值 - 当前值
    shift_vect[dim] = value - curr
    # 执行平移，将对象指定维度的坐标设置为目标值
    self.shift(shift_vect)
    return self

def set_x(self, x: float, direction: Vect3 = ORIGIN) -> Self:
    # 设置对象X轴坐标（维度0），复用set_coord方法
    return self.set_coord(x, 0, direction)

def set_y(self, y: float, direction: Vect3 = ORIGIN) -> Self:
    # 设置对象Y轴坐标（维度1），复用set_coord方法
    return self.set_coord(y, 1, direction)

def set_z(self, z: float, direction: Vect3 = ORIGIN) -> Self:
    # 设置对象Z轴坐标（维度2），复用set_coord方法
    return self.set_coord(z, 2, direction)

def set_z_index(self, z_index: int) -> Self:
    # 设置对象的Z索引（用于控制渲染层级，Z索引高的对象优先渲染）
    self.z_index = z_index
    return self

def space_out_submobjects(self, factor: float = 1.5, **kwargs) -> Self:
    # 先整体缩放当前对象（放大factor倍），拉开子对象间距
    self.scale(factor,** kwargs)
    # 再将每个子对象缩小回原尺寸（1/factor倍），保持子对象大小不变
    for submob in self.submobjects:
        submob.scale(1. / factor)
    return self

def move_to(
    self,
    point_or_mobject: Mobject | Vect3,
    aligned_edge: Vect3 = ORIGIN,
    coor_mask: Vect3 = np.array([1, 1, 1])
) -> Self:
    # 处理目标为Mobject的情况：获取目标对象指定对齐边缘的边界点
    if isinstance(point_or_mobject, Mobject):
        target = point_or_mobject.get_bounding_box_point(aligned_edge)
    # 处理目标为点的情况：直接使用该点作为目标点
    else:
        target = point_or_mobject
    # 获取当前对象指定对齐边缘的边界点（待对齐的点）
    point_to_align = self.get_bounding_box_point(aligned_edge)
    # 计算平移量：(目标点 - 待对齐点) × 坐标掩码（过滤不需要平移的维度）
    self.shift((target - point_to_align) * coor_mask)
    return self

def replace(self, mobject: Mobject, dim_to_match: int = 0, stretch: bool = False) -> Self:
    # 若目标对象无点数据且无子对象（空对象），将当前对象缩放到0（隐藏）
    if not mobject.get_num_points() and not mobject.submobjects:
        self.scale(0)
        return self
    # 拉伸模式：按目标对象的每个维度单独拉伸当前对象
    if stretch:
        for i in range(self.dim):
            self.rescale_to_fit(mobject.length_over_dim(i), i, stretch=True)
    # 非拉伸模式：仅按指定维度缩放当前对象（保持原比例）
    else:
        self.rescale_to_fit(
            mobject.length_over_dim(dim_to_match),  # 目标对象指定维度的长度
            dim_to_match,                          # 待匹配的维度
            stretch=False                          # 禁用拉伸（整体缩放）
        )
    # 将当前对象平移到目标对象的中心点位置
    self.shift(mobject.get_center() - self.get_center())
    return self

def surround(
    self,
    mobject: Mobject,
    dim_to_match: int = 0,
    stretch: bool = False,
    buff: float = MED_SMALL_BUFF
) -> Self:
    # 先调用replace方法，使当前对象与目标对象尺寸、位置初步匹配
    self.replace(mobject, dim_to_match, stretch)
    # 获取目标对象指定维度的长度
    length = mobject.length_over_dim(dim_to_match)
    # 缩放当前对象，在目标对象基础上增加指定间距（实现"包围"效果）
    self.scale((length + buff) / length)
    return self

def put_start_and_end_on(self, start: Vect3, end: Vect3) -> Self:
    # 获取当前对象的起始点和终止点（如线段的两端）
    curr_start, curr_end = self.get_start_and_end()
    # 计算当前对象起始点到终止点的向量
    curr_vect = curr_end - curr_start
    # 若当前向量为零向量（闭合图形），抛出异常（无法定位闭合图形的端点）
    if np.all(curr_vect == 0):
        raise Exception("Cannot position endpoints of closed loop")
    # 计算目标起始点到目标终止点的向量
    target_vect = end - start

    # 1. 缩放当前对象：使当前向量长度匹配目标向量长度（围绕当前起始点缩放）
    self.scale(
        get_norm(target_vect) / get_norm(curr_vect),  # 缩放因子：目标长度/当前长度
        about_point=curr_start,                       # 围绕当前起始点缩放（避免起始点偏移）
    )

    # 2. 2D平面旋转：使当前向量在XY平面的角度匹配目标向量
    self.rotate(
        angle_of_vector(target_vect) - angle_of_vector(curr_vect),  # 旋转角度差
    )

    # 3. 3D空间旋转：调整Z轴方向的角度，使当前向量完全匹配目标向量的3D方向
    self.rotate(
        # 计算Z轴方向的角度差：当前向量的Z角 - 目标向量的Z角
        np.arctan2(curr_vect[2], get_norm(curr_vect[:2])) - np.arctan2(target_vect[2], get_norm(target_vect[:2])),
        axis=np.array([-target_vect[1], target_vect[0], 0]),  # 旋转轴：垂直于目标向量的XY平面法向量
    )

    # 4. 平移当前对象：将当前起始点移动到目标起始点
    self.shift(start - self.get_start())
    return self
    # Color functions

@affects_family_data  # 装饰器：标记该方法会修改家族成员的数据，需触发相关更新
def set_rgba_array(
    self,
    rgba_array: npt.ArrayLike,
    name: str = "rgba",
    recurse: bool = False
) -> Self:
    # 遍历当前对象的家族成员（根据recurse决定是否递归子对象）
    for mob in self.get_family(recurse):
        # 若当前成员有数据点，使用其数据；否则使用默认数据模板
        data = mob.data if mob.get_num_points() > 0 else mob._data_defaults
        # 将RGBA数组赋值到指定名称的字段（默认是"rgba"字段，存储颜色和透明度）
        data[name][:] = rgba_array
    return self

def set_color_by_rgba_func(
    self,
    func: Callable[[Vect3Array], Vect4Array],
    recurse: bool = True
) -> Self:
    """
    输入函数需接收3D坐标点数组，输出对应的RGBA颜色数组（每个点对应一个RGBA值）
    """
    # 遍历家族成员，为每个成员设置颜色
    for mob in self.get_family(recurse):
        # 1. 获取成员的所有3D点；2. 用自定义函数计算每个点的RGBA；3. 应用RGBA数组
        mob.set_rgba_array(func(mob.get_points()))
    return self

def set_color_by_rgb_func(
    self,
    func: Callable[[Vect3Array], Vect3Array],
    opacity: float = 1,
    recurse: bool = True
) -> Self:
    """
    输入函数需接收3D坐标点数组，输出对应的RGB颜色数组（每个点对应一个RGB值）
    """
    # 遍历家族成员，为每个成员设置颜色和透明度
    for mob in self.get_family(recurse):
        # 获取成员的所有3D点
        points = mob.get_points()
        # 创建与点数量匹配的透明度数组（所有点使用统一透明度）
        opacity_array = np.ones((points.shape[0], 1)) * opacity
        # 1. 用自定义函数计算RGB；2. 将RGB与透明度拼接成RGBA；3. 应用RGBA数组
        mob.set_rgba_array(np.hstack((func(points), opacity_array)))
    return self

@affects_family_data  # 装饰器：标记该方法会修改家族成员的数据
def set_rgba_array_by_color(
    self,
    color: ManimColor | Iterable[ManimColor] | None = None,
    opacity: float | Iterable[float] | None = None,
    name: str = "rgba",
    recurse: bool = True
) -> Self:
    # 遍历家族成员，为每个成员设置RGBA
    for mob in self.get_family(recurse):
        # 选择成员的数据（有数据点则用自身数据，否则用默认模板）
        data = mob.data if mob.has_points() > 0 else mob._data_defaults
        
        # 处理颜色：若指定了颜色
        if color is not None:
            # 1. 将颜色（或颜色列表）转为RGB数组；2. 用listify统一格式（处理单个颜色/颜色列表）
            rgbs = np.array(list(map(color_to_rgb, listify(color))))
            # 若颜色数量大于1，插值调整颜色数组长度以匹配数据长度（实现渐变等效果）
            if 1 < len(rgbs):
                rgbs = resize_with_interpolation(rgbs, len(data))
            # 将RGB值赋值到RGBA字段的前3列（RGB通道）
            data[name][:, :3] = rgbs
        
        # 处理透明度：若指定了透明度
        if opacity is not None:
            # 若透明度不是单个数值（如透明度列表），插值调整长度以匹配数据长度
            if not isinstance(opacity, (float, int, np.floating)):
                opacity = resize_with_interpolation(np.array(opacity), len(data))
            # 将透明度赋值到RGBA字段的第4列（Alpha通道）
            data[name][:, 3] = opacity
    return self

def set_color(
    self,
    color: ManimColor | Iterable[ManimColor] | None,
    opacity: float | Iterable[float] | None = None,
    recurse: bool = True
) -> Self:
    # 先为当前对象设置颜色和透明度（不递归，避免重复处理）
    self.set_rgba_array_by_color(color, opacity, recurse=False)
    # 若需要递归，单独调用子对象的set_color（允许子对象重写该方法实现自定义逻辑）
    if recurse:
        for submob in self.submobjects:
            submob.set_color(color, recurse=True)
    return self

def set_opacity(
    self,
    opacity: float | Iterable[float] | None,
    recurse: bool = True
) -> Self:
    # 先为当前对象设置透明度（颜色设为None，仅修改Alpha通道，不递归）
    self.set_rgba_array_by_color(color=None, opacity=opacity, recurse=False)
    # 若需要递归，调用子对象的set_opacity
    if recurse:
        for submob in self.submobjects:
            submob.set_opacity(opacity, recurse=True)
    return self

def get_color(self) -> str:
    # 获取当前对象的颜色：1. 取RGBA字段第一个点的RGB值；2. 转为十六进制颜色字符串
    return rgb_to_hex(self.data["rgba"][0, :3])

def get_opacity(self) -> float:
    # 获取当前对象的透明度：取RGBA字段第一个点的Alpha值（转为Python浮点数）
    return float(self.data["rgba"][0, 3])

def get_opacities(self) -> float:
    # 获取所有点的透明度：返回RGBA字段的第4列（所有点的Alpha通道）
    return self.data["rgba"][:, 3]

def set_color_by_gradient(self, *colors: ManimColor) -> Self:
    # 若当前对象有数据点，直接为其设置渐变颜色（通过set_color处理多颜色插值）
    if self.has_points():
        self.set_color(colors)
    # 若当前对象无数据点（如组对象），为其子对象设置渐变颜色
    else:
        self.set_submobject_colors_by_gradient(*colors)
    return self

def set_submobject_colors_by_gradient(self, *colors: ManimColor) -> Self:
    # 检查颜色数量：至少需要1个颜色
    if len(colors) == 0:
        raise Exception("Need at least one color")
    # 若只有1个颜色，直接为所有子对象设置统一颜色
    elif len(colors) == 1:
        return self.set_color(*colors)

    # 获取当前对象的子对象列表（用于逐个分配渐变颜色）
    mobs = self.submobjects
    # 生成与子对象数量匹配的渐变颜色数组（插值计算中间色）
    new_colors = color_gradient(colors, len(mobs))

    # 为每个子对象分配对应的渐变颜色
    for mob, color in zip(mobs, new_colors):
        mob.set_color(color)
    return self

def fade(self, darkness: float = 0.5, recurse: bool = True) -> Self:
    # 实现"褪色"效果：透明度 = 1 - 暗度（darkness越大，透明度越低，越暗）
    self.set_opacity(1.0 - darkness, recurse=recurse)

def get_shading(self) -> np.ndarray:
    # 获取当前对象的着色参数（从uniforms中读取"shading"字段，控制光照反射效果）
    return self.uniforms["shading"]

def set_shading(
    self,
    reflectiveness: float | None = None,
    gloss: float | None = None,
    shadow: float | None = None,
    recurse: bool = True
) -> Self:
    """
    - 反射度（reflectiveness）越大：物体朝向光源的面越亮
    - 阴影度（shadow）越大：物体背向光源的面越暗
    - 光泽度（gloss）：控制物体表面光泽感，使反光区域更集中
    """
    # 遍历家族成员，为每个成员设置着色参数
    for mob in self.get_family(recurse):
        # 获取当前成员的着色参数（避免直接修改原数组，先读取）
        shading = mob.uniforms["shading"]
        # 按顺序更新反射度、光泽度、阴影度（仅更新非None的参数）
        for i, value in enumerate([reflectiveness, gloss, shadow]):
            if value is not None:
                shading[i] = value
        # 将更新后的着色参数写回uniforms（不递归，避免重复处理）
        mob.set_uniform(shading=shading, recurse=False)
    return self

def get_reflectiveness(self) -> float:
    # 获取着色参数中的反射度：shading数组第0位存储反射度
    return self.get_shading()[0]

def get_gloss(self) -> float:
    # 获取着色参数中的光泽度：shading数组第1位存储光泽度
    return self.get_shading()[1]

def get_shadow(self) -> float:
    # 获取着色参数中的阴影度：shading数组第2位存储阴影度
    return self.get_shading()[2]

def set_reflectiveness(self, reflectiveness: float, recurse: bool = True) -> Self:
    # 单独设置反射度：复用set_shading方法，仅传递reflectiveness参数
    self.set_shading(reflectiveness=reflectiveness, recurse=recurse)
    return self

def set_gloss(self, gloss: float, recurse: bool = True) -> Self:
    # 单独设置光泽度：复用set_shading方法，仅传递gloss参数
    self.set_shading(gloss=gloss, recurse=recurse)
    return self

def set_shadow(self, shadow: float, recurse: bool = True) -> Self:
    # 单独设置阴影度：复用set_shading方法，仅传递shadow参数
    self.set_shading(shadow=shadow, recurse=recurse)
    return self

# 背景矩形相关方法

def add_background_rectangle(
    self,
    color: ManimColor | None = None,
    opacity: float = 1.0,** kwargs
) -> Self:
    # 延迟导入BackgroundRectangle（避免循环导入）
    from manimlib.mobject.shape_matchers import BackgroundRectangle
    # 创建背景矩形对象：尺寸匹配当前对象，应用指定颜色和透明度
    self.background_rectangle = BackgroundRectangle(
        self, color=color,
        fill_opacity=opacity,
        **kwargs  # 传递额外参数（如边框、圆角等）
    )
    # 将背景矩形添加到当前对象的最底层（避免遮挡其他内容）
    self.add_to_back(self.background_rectangle)
    return self

def add_background_rectangle_to_submobjects(self, **kwargs) -> Self:
    # 为当前对象的所有子对象添加背景矩形
    for submobject in self.submobjects:
        submobject.add_background_rectangle(**kwargs)
    return self

def add_background_rectangle_to_family_members_with_points(self, **kwargs) -> Self:
    # 为家族中所有包含点数据的成员添加背景矩形（排除空对象）
    for mob in self.family_members_with_points():
        mob.add_background_rectangle(**kwargs)
    return self

# 获取属性相关方法（Getters）

def get_bounding_box_point(self, direction: Vect3) -> Vect3:
    # 根据方向向量获取边界框上的对应点（如边缘、角落）
    bb = self.get_bounding_box()  # 获取边界框（格式：[左下后, 中心, 右上前]）
    # 计算边界框索引：方向向量符号转为0/2索引（对应边界框的两个端点）
    indices = (np.sign(direction) + 1).astype(int)
    # 按维度提取边界框点：每个维度取对应索引的坐标
    return np.array([
        bb[indices[i]][i] for i in range(3)
    ])

def get_edge_center(self, direction: Vect3) -> Vect3:
    # 获取指定方向的边缘中心点（复用边界框点计算逻辑，部分场景下边缘中心即边界框点）
    return self.get_bounding_box_point(direction)

def get_corner(self, direction: Vect3) -> Vect3:
    # 获取指定方向的角落点（复用边界框点计算逻辑，方向向量为对角时对应角落）
    return self.get_bounding_box_point(direction)

def get_all_corners(self):
    # 获取边界框的所有8个角落点（3D边界框有2^3=8个角落）
    bb = self.get_bounding_box()
    # 生成所有维度的索引组合（0和2分别对应边界框的两个端点）
    return np.array([
        [bb[indices[-i + 1]][i] for i in range(3)]
        for indices in it.product([0, 2], repeat=3)
    ])

def get_center(self) -> Vect3:
    # 获取对象中心：边界框的第1位存储中心坐标
    return self.get_bounding_box()[1]

def get_center_of_mass(self) -> Vect3:
    # 获取质心：所有点坐标的平均值（适用于不规则形状）
    return self.get_all_points().mean(0)

def get_boundary_point(self, direction: Vect3) -> Vect3:
    # 获取对象在指定方向上的最外点（区别于边界框，基于实际点数据）
    all_points = self.get_all_points()  # 获取所有点数据
    # 计算每个点相对于中心的方向向量
    boundary_directions = all_points - self.get_center()
    # 归一化方向向量（消除距离影响，仅保留方向）
    norms = np.linalg.norm(boundary_directions, axis=1)
    boundary_directions /= np.repeat(norms, 3).reshape((len(norms), 3))
    # 找到与目标方向最匹配的点（点积最大的点）
    index = np.argmax(np.dot(boundary_directions, np.array(direction).T))
    return all_points[index]

def get_continuous_bounding_box_point(self, direction: Vect3) -> Vect3:
    # 获取边界框在任意连续方向上的对应点（支持非轴对齐方向）
    dl, center, ur = self.get_bounding_box()  # 边界框：左下后、中心、右上前
    corner_vect = (ur - center)  # 中心到右上前的向量（半长向量）
    # 计算方向向量在各维度上的比例，找到最大比例维度，确保点在边界框上
    return center + direction / np.max(np.abs(np.true_divide(
        direction, corner_vect,
        out=np.zeros(len(direction)),  # 避免除零，分母为0时输出0
        where=((corner_vect) != 0)     # 仅在分母非零时计算
    )))

def get_top(self) -> Vect3:
    # 获取对象顶部点（UP方向的边缘中心）
    return self.get_edge_center(UP)

def get_bottom(self) -> Vect3:
    # 获取对象底部点（DOWN方向的边缘中心）
    return self.get_edge_center(DOWN)

def get_right(self) -> Vect3:
    # 获取对象右侧点（RIGHT方向的边缘中心）
    return self.get_edge_center(RIGHT)

def get_left(self) -> Vect3:
    # 获取对象左侧点（LEFT方向的边缘中心）
    return self.get_edge_center(LEFT)

def get_zenith(self) -> Vect3:
    # 获取对象最高点（OUT方向的边缘中心，3D场景中朝外）
    return self.get_edge_center(OUT)

def get_nadir(self) -> Vect3:
    # 获取对象最低点（IN方向的边缘中心，3D场景中朝内）
    return self.get_edge_center(IN)

def length_over_dim(self, dim: int) -> float:
    # 计算指定维度上的长度（边界框在该维度的跨度）
    bb = self.get_bounding_box()
    return abs((bb[2] - bb[0])[dim])  # bb[2]-bb[0]是维度跨度，取绝对值

def get_width(self) -> float:
    # 获取宽度（X轴维度，dim=0）
    return self.length_over_dim(0)

def get_height(self) -> float:
    # 获取高度（Y轴维度，dim=1）
    return self.length_over_dim(1)

def get_depth(self) -> float:
    # 获取深度（Z轴维度，dim=2）
    return self.length_over_dim(2)

def get_shape(self) -> Tuple[float]:
    # 获取对象的三维尺寸（宽、高、深）
    return tuple(self.length_over_dim(dim) for dim in range(3))

def get_coord(self, dim: int, direction: Vect3 = ORIGIN) -> float:
    """
    通用化的坐标获取方法，可替代get_x、get_y、get_z
    """
    # 获取指定维度、指定方向上的坐标（从边界框点中提取对应维度值）
    return self.get_bounding_box_point(direction)[dim]

def get_x(self, direction=ORIGIN) -> float:
    # 获取X轴坐标（dim=0），复用get_coord
    return self.get_coord(0, direction)

def get_y(self, direction=ORIGIN) -> float:
    # 获取Y轴坐标（dim=1），复用get_coord
    return self.get_coord(1, direction)

def get_z(self, direction=ORIGIN) -> float:
    # 获取Z轴坐标（dim=2），复用get_coord
    return self.get_coord(2, direction)

def get_start(self) -> Vect3:
    # 检查对象是否有数据点，无点则抛出异常
    self.throw_error_if_no_points()
    # 返回第一个点的副本（避免外部修改原数据）
    return self.get_points()[0].copy()

def get_end(self) -> Vect3:
    # 检查对象是否有数据点，无点则抛出异常
    self.throw_error_if_no_points()
    # 返回最后一个点的副本
    return self.get_points()[-1].copy()

def get_start_and_end(self) -> tuple[Vect3, Vect3]:
    # 检查对象是否有数据点，无点则抛出异常
    self.throw_error_if_no_points()
    points = self.get_points()
    # 返回起始点和终止点的副本
    return (points[0].copy(), points[-1].copy())

def point_from_proportion(self, alpha: float) -> Vect3:
    # 根据比例alpha获取点（alpha∈[0,1]，0对应起点，1对应终点）
    points = self.get_points()
    # 计算整数索引和小数比例（如alpha=0.3，n=5 → i=1，subalpha=0.5）
    i, subalpha = integer_interpolate(0, len(points) - 1, alpha)
    # 在第i个点和第i+1个点之间插值
    return interpolate(points[i], points[i + 1], subalpha)

def pfp(self, alpha):
    """point_from_proportion的缩写，快速调用"""
    return self.point_from_proportion(alpha)

def get_pieces(self, n_pieces: int) -> Group:
    # 将对象分割为n_pieces个连续部分，返回包含这些部分的组
    # 创建空副本作为模板（清除子对象，保留基础属性）
    template = self.copy()
    template.set_submobjects([])
    # 生成分割比例（0到1之间的n_pieces+1个均匀点）
    alphas = np.linspace(0, 1, n_pieces + 1)
    # 逐个创建分割部分，拼接成组
    return Group(*[
        template.copy().pointwise_become_partial(
            self, a1, a2  # 每个部分对应原对象的[a1,a2]比例区间
        )
        for a1, a2 in zip(alphas[:-1], alphas[1:])
    ])

def get_z_index_reference_point(self) -> Vect3:
    # TODO：z_index_group的默认定义位置可优化
    # 获取Z索引参考点（用于确定渲染层级，默认使用自身或z_index_group的中心）
    z_index_group = getattr(self, "z_index_group", self)
    return z_index_group.get_center()

# 匹配其他对象属性的方法

def match_color(self, mobject: Mobject) -> Self:
    # 匹配目标对象的颜色
    return self.set_color(mobject.get_color())

def match_style(self, mobject: Mobject) -> Self:
    # 匹配目标对象的整体样式（颜色、透明度、着色参数）
    self.set_color(mobject.get_color())          # 匹配颜色
    self.set_opacity(mobject.get_opacity())      # 匹配透明度
    self.set_shading(*mobject.get_shading())     # 匹配着色参数（反射度、光泽度、阴影度）
    return self

def match_dim_size(self, mobject: Mobject, dim: int,** kwargs) -> Self:
    # 匹配目标对象在指定维度上的尺寸
    return self.rescale_to_fit(
        mobject.length_over_dim(dim),  # 目标对象指定维度的长度
        dim,                           # 待匹配的维度
        **kwargs                       # 传递额外参数（如缩放中心点）
    )

def match_width(self, mobject: Mobject, **kwargs) -> Self:
    # 匹配目标对象的宽度（X轴维度，dim=0），复用match_dim_size方法
    # **kwargs传递额外参数（如缩放中心点、是否拉伸等）
    return self.match_dim_size(mobject, 0,** kwargs)

def match_height(self, mobject: Mobject, **kwargs) -> Self:
    # 匹配目标对象的高度（Y轴维度，dim=1），复用match_dim_size方法
    return self.match_dim_size(mobject, 1, **kwargs)

def match_depth(self, mobject: Mobject, **kwargs) -> Self:
    # 匹配目标对象的深度（Z轴维度，dim=2），复用match_dim_size方法
    return self.match_dim_size(mobject, 2,** kwargs)

def match_coord(
    self,
    mobject_or_point: Mobject | Vect3,
    dim: int,
    direction: Vect3 = ORIGIN
) -> Self:
    # 处理目标为Mobject的情况：获取目标对象在指定维度和方向上的坐标
    if isinstance(mobject_or_point, Mobject):
        coord = mobject_or_point.get_coord(dim, direction)
    # 处理目标为点的情况：直接提取点在指定维度上的坐标值
    else:
        coord = mobject_or_point[dim]
    # 将当前对象在指定维度和方向上的坐标设置为目标坐标
    return self.set_coord(coord, dim=dim, direction=direction)

def match_x(
    self,
    mobject_or_point: Mobject | Vect3,
    direction: Vect3 = ORIGIN
) -> Self:
    # 匹配目标对象/X轴坐标（dim=0），复用match_coord方法
    return self.match_coord(mobject_or_point, 0, direction)

def match_y(
    self,
    mobject_or_point: Mobject | Vect3,
    direction: Vect3 = ORIGIN
) -> Self:
    # 匹配目标对象/Y轴坐标（dim=1），复用match_coord方法
    return self.match_coord(mobject_or_point, 1, direction)

def match_z(
    self,
    mobject_or_point: Mobject | Vect3,
    direction: Vect3 = ORIGIN
) -> Self:
    # 匹配目标对象/Z轴坐标（dim=2），复用match_coord方法
    return self.match_coord(mobject_or_point, 2, direction)

def align_to(
    self,
    mobject_or_point: Mobject | Vect3,
    direction: Vect3 = ORIGIN
) -> Self:
    """
    示例：
    1. mob1.align_to(mob2, UP) → 垂直移动mob1，使其顶部边缘与mob2的顶部边缘对齐
    2. mob1.align_to(mob2, RIGHT) → 水平移动mob1，使其中心与mob2的中心在垂直方向对齐
    """
    # 处理目标为Mobject的情况：获取目标对象在指定方向上的边界框点
    if isinstance(mobject_or_point, Mobject):
        point = mobject_or_point.get_bounding_box_point(direction)
    # 处理目标为点的情况：直接使用该点作为对齐目标点
    else:
        point = mobject_or_point

    # 遍历所有维度，仅对方向向量非零的维度执行对齐（即仅在目标方向上调整）
    for dim in range(self.dim):
        if direction[dim] != 0:
            self.set_coord(point[dim], dim, direction)
    return self

def get_group_class(self):
    # 返回当前对象对应的组类（默认是Group，用于创建包含自身的组）
    return Group

# 对齐相关方法（Alignment）

def is_aligned_with(self, mobject: Mobject) -> bool:
    # 检查当前对象是否与目标对象对齐（需满足数据长度和子对象数量一致）
    # 1. 检查数据长度是否相同（点数据数量一致）
    if len(self.data) != len(mobject.data):
        return False
    # 2. 检查子对象数量是否相同
    if len(self.submobjects) != len(mobject.submobjects):
        return False
    # 3. 递归检查所有子对象是否对齐
    return all(
        sm1.is_aligned_with(sm2)
        for sm1, sm2 in zip(self.submobjects, mobject.submobjects)
    )

def align_data_and_family(self, mobject: Mobject) -> Self:
    # 同时对齐家族结构和数据（先对齐家族，再对齐数据）
    self.align_family(mobject)  # 对齐子对象数量和层级
    self.align_data(mobject)    # 对齐点数据长度
    return self

def align_data(self, mobject: Mobject) -> Self:
    # 对齐当前对象与目标对象的家族成员数据（点数量匹配）
    # 遍历双方家族成员，逐个对齐点数据
    for mob1, mob2 in zip(self.get_family(), mobject.get_family()):
        mob1.align_points(mob2)
    return self

def align_points(self, mobject: Mobject) -> Self:
    # 对齐两个对象的点数据长度（统一调整为两者中的最大长度）
    # 计算双方点数量的最大值
    max_len = max(self.get_num_points(), mobject.get_num_points())
    # 对两个对象分别调整点数量，使用保序插值确保形状不变
    for mob in (self, mobject):
        mob.resize_points(max_len, resize_func=resize_preserving_order)
    return self

def align_family(self, mobject: Mobject) -> Self:
    # 对齐当前对象与目标对象的家族结构（子对象数量匹配）
    mob1 = self       # 当前对象
    mob2 = mobject    # 目标对象
    n1 = len(mob1)    # 当前对象的子对象数量
    n2 = len(mob2)    # 目标对象的子对象数量

    # 若子对象数量不同，为较少的一方添加空的子对象以补全数量
    if n1 != n2:
        mob1.add_n_more_submobjects(max(0, n2 - n1))  # 给mob1补全子对象
        mob2.add_n_more_submobjects(max(0, n1 - n2))  # 给mob2补全子对象

    # 递归对齐双方的子对象（确保层级结构一致）
    for sm1, sm2 in zip(mob1.submobjects, mob2.submobjects):
        sm1.align_family(sm2)
    return self

def push_self_into_submobjects(self) -> Self:
    # 将当前对象自身转为子对象（原对象变为空容器，包含自身副本）
    # 1. 创建当前对象的副本（保留原属性和数据）
    copy = self.copy()
    # 2. 清空副本的子对象（避免循环引用）
    copy.set_submobjects([])
    # 3. 清空当前对象的点数据（变为空容器）
    self.resize_points(0)
    # 4. 将副本添加为当前对象的子对象
    self.add(copy)
    return self

def add_n_more_submobjects(self, n: int) -> Self:
    # 如果不需要添加子对象（n=0），直接返回
    if n == 0:
        return self

    # 获取当前子对象的数量
    curr = len(self.submobjects)
    # 如果当前没有子对象，创建n个空点对象作为子对象
    if curr == 0:
        # 创建一个空的参考对象，点数据设为当前对象的中心点
        null_mob = self.copy()
        null_mob.set_points([self.get_center()])
        # 设置n个空对象副本作为子对象
        self.set_submobjects([
            null_mob.copy() for k in range(n)
        ])
        return self

    # 计算目标子对象总数（当前数量 + 需添加数量）
    target = curr + n
    # 计算每个现有子对象需要重复的次数（均匀分配新增数量）
    repeat_indices = (np.arange(target) * curr) // target
    split_factors = [
        (repeat_indices == i).sum() for i in range(curr)
    ]

    # 生成新的子对象列表
    new_submobs = []
    for submob, sf in zip(self.submobjects, split_factors):
        # 添加原有的子对象
        new_submobs.append(submob)
        # 添加sf-1个不可见的副本（补全数量）
        for k in range(1, sf):
            new_submobs.append(submob.invisible_copy())
    # 更新子对象列表
    self.set_submobjects(new_submobs)
    return self

def invisible_copy(self) -> Self:
    # 创建当前对象的不可见副本（透明度设为0）
    return self.copy().set_opacity(0)

    # Interpolate

def interpolate(
    self,
    mobject1: Mobject,
    mobject2: Mobject,
    alpha: float,
    path_func: Callable[[np.ndarray, np.ndarray, float], np.ndarray] = straight_path
) -> Self:
    # 确定需要插值的数据字段（排除锁定的数据键）
    keys = [k for k in self.data.dtype.names if k not in self.locked_data_keys]
    # 如果有需要更新的数据，标记数据已更改
    if keys:
        self.note_changed_data()

    # 对每个需要插值的数据字段执行插值
    for key in keys:
        md1 = mobject1.data[key]  # 第一个对象的数据
        md2 = mobject2.data[key]  # 第二个对象的数据

        # 处理常量数据字段（所有点共享同一值）
        if key in self.const_data_keys:
            md1 = md1[0]  # 取第一个点的值作为代表
            md2 = md2[0]

        # 对点类数据使用路径函数插值（如位置、颜色等）
        if key in self.pointlike_data_keys:
            self.data[key] = path_func(md1, md2, alpha)
        # 对其他数据使用线性插值
        else:
            self.data[key] = (1 - alpha) * md1 + alpha * md2

    # 对uniforms执行插值（排除锁定的uniform键）
    for key in self.uniforms:
        if key in self.locked_uniform_keys:
            continue  # 跳过锁定的uniform
        # 确保两个对象都有该uniform才进行插值
        if key not in mobject1.uniforms or key not in mobject2.uniforms:
            continue
        self.uniforms[key] = (1 - alpha) * mobject1.uniforms[key] + alpha * mobject2.uniforms[key]

    # 对边界框执行插值
    self.bounding_box[:] = path_func(mobject1.bounding_box, mobject2.bounding_box, alpha)
    return self

def pointwise_become_partial(self, mobject, a, b) -> Self:
    """
    按点设置当前对象，使其成为目标对象的一部分
    输入参数0 <= a < b <= 1 确定要成为目标对象的哪一部分
    """
    # 需在子类中实现具体逻辑
    return self

# 数据锁定相关方法

def lock_data(self, keys: Iterable[str]) -> Self:
    """
    为加速某些动画（尤其是变换动画），可以标记哪些数据
    在动画过程中不会改变，这样插值时可以跳过这些数据，
    也避免不必要地读取到shader_wrapper中
    """
    # 如果有更新器，不锁定数据（更新器可能会修改数据）
    if self.has_updaters():
        return self
    # 记录锁定的数据键
    self.locked_data_keys = set(keys)
    return self

def lock_uniforms(self, keys: Iterable[str]) -> Self:
    # 如果有更新器，不锁定uniforms
    if self.has_updaters():
        return self
    # 记录锁定的uniform键
    self.locked_uniform_keys = set(keys)
    return self

def lock_matching_data(self, mobject1: Mobject, mobject2: Mobject) -> Self:
    # 遍历当前对象、mobject1、mobject2的家族成员三元组
    tuples = zip(
        self.get_family(),
        mobject1.get_family(),
        mobject2.get_family(),
    )
    for sm, sm1, sm2 in tuples:
        # 确保三者的数据类型一致，否则跳过
        if not sm.data.dtype == sm1.data.dtype == sm2.data.dtype:
            continue
        # 锁定在两个对象中相同的数据字段
        sm.lock_data(
            key for key in sm.data.dtype.names
            if arrays_match(sm1.data[key], sm2.data[key])
        )
        # 锁定在两个对象中相同的uniform字段
        sm.lock_uniforms(
            key for key in self.uniforms
            if all(listify(mobject1.uniforms.get(key, 0) == mobject2.uniforms.get(key, 0)))
        )
        # 标记常量数据字段（非锁定且在三个对象中均为常量）
        sm.const_data_keys = set(
            key for key in sm.data.dtype.names
            if key not in sm.locked_data_keys
            if all(
                array_is_constant(mob.data[key])
                for mob in (sm, sm1, sm2)
            )
        )
    return self

def unlock_data(self) -> Self:
    # 递归解锁当前对象家族中所有成员的数据锁定状态
    for mob in self.get_family():
        mob.locked_data_keys = set()       # 清空锁定的数据字段集合（解锁数据）
        mob.const_data_keys = set()        # 清空常量数据字段集合（取消常量标记）
        mob.locked_uniform_keys = set()    # 清空锁定的Uniform字段集合（解锁Uniform）
    return self

# 涉及Shader Uniform（着色器统一变量）的操作方法

@staticmethod
def affects_shader_info_id(func: Callable[..., T]) -> Callable[..., T]:
    # 装饰器：标记被装饰的方法会影响Shader信息ID，执行后需刷新Shader包装器ID
    @wraps(func)  # 保留原函数的元数据（如名称、文档字符串）
    def wrapper(self, *args, **kwargs):
        # 先执行原函数逻辑
        result = func(self, *args, **kwargs)
        # 刷新Shader包装器ID（确保Shader识别更新后的状态）
        self.refresh_shader_wrapper_id()
        return result
    return wrapper

@affects_shader_info_id  # 应用装饰器，修改Uniform后刷新Shader ID
def set_uniform(self, recurse: bool = True, **new_uniforms) -> Self:
    # 为家族成员设置Shader Uniform（统一变量）
    # recurse=True表示递归应用到所有子对象
    for mob in self.get_family(recurse):
        mob.uniforms.update(new_uniforms)  # 更新Uniform字典
    return self

@affects_shader_info_id  # 应用装饰器，修改后刷新Shader ID
def fix_in_frame(self, recurse: bool = True) -> Self:
    # 将对象固定在屏幕坐标系中（不随相机移动而变化）
    # 通过设置is_fixed_in_frame为1.0实现（Shader会根据该值调整渲染逻辑）
    self.set_uniform(recurse, is_fixed_in_frame=1.0)
    return self

@affects_shader_info_id  # 应用装饰器，修改后刷新Shader ID
def unfix_from_frame(self, recurse: bool = True) -> Self:
    # 取消对象在屏幕坐标系的固定（恢复随相机移动）
    # 设置is_fixed_in_frame为0.0
    self.set_uniform(recurse, is_fixed_in_frame=0.0)
    return self

def is_fixed_in_frame(self) -> bool:
    # 判断对象是否固定在屏幕坐标系中
    # 从Uniform中读取is_fixed_in_frame值，转为布尔类型
    return bool(self.uniforms["is_fixed_in_frame"])

@affects_shader_info_id  # 应用装饰器，修改后刷新Shader ID
def apply_depth_test(self, recurse: bool = True) -> Self:
    # 为家族成员启用深度测试（3D渲染中避免后绘制的物体遮挡先绘制的物体）
    for mob in self.get_family(recurse):
        mob.depth_test = True  # 开启深度测试标记
    return self

@affects_shader_info_id  # 应用装饰器，修改后刷新Shader ID
def deactivate_depth_test(self, recurse: bool = True) -> Self:
    # 为家族成员禁用深度测试
    for mob in self.get_family(recurse):
        mob.depth_test = False  # 关闭深度测试标记
    return self

def set_clip_plane(
    self,
    vect: Vect3 | None = None,
    threshold: float | None = None,
    recurse=True
) -> Self:
    # 设置裁剪平面（用于裁剪物体的特定区域，仅渲染平面一侧的部分）
    for submob in self.get_family(recurse):
        # 如果指定了平面法向量（vect），更新裁剪平面的前3个分量（法向量）
        if vect is not None:
            submob.uniforms["clip_plane"][:3] = vect
        # 如果指定了阈值（threshold），更新裁剪平面的第4个分量（距离原点的距离）
        if threshold is not None:
            submob.uniforms["clip_plane"][3] = threshold
    return self

def deactivate_clip_plane(self) -> Self:
    # 禁用裁剪平面（将裁剪平面参数全部设为0，Shader会跳过裁剪逻辑）
    self.uniforms["clip_plane"][:] = 0
    return self

# Shader代码操作方法

@affects_data  # 装饰器：标记修改Shader代码会影响数据状态
def replace_shader_code(self, old: str, new: str) -> Self:
    # 替换家族成员的Shader代码片段（用于自定义着色逻辑）
    for mob in self.get_family():
        # 记录代码替换规则（旧片段→新片段）
        mob.shader_code_replacements[old] = new
        # 清空现有Shader包装器（后续需重新初始化以应用新代码）
        mob.shader_wrapper = None
    return self

def set_color_by_code(self, glsl_code: str) -> Self:
    """
    通过GLSL代码片段自定义对象颜色
    代码运行上下文包含以下变量：
    - vec4 color：当前颜色（需修改该变量实现颜色自定义）
    - vec3 point：顶点的3D坐标
    - vec3 unit_normal：顶点的单位法向量
    """
    # 替换Shader中预留的颜色函数插入位置，注入自定义GLSL代码
    self.replace_shader_code(
        "///// INSERT COLOR FUNCTION HERE /////",  # Shader中的预留标记
        glsl_code  # 自定义GLSL颜色逻辑
    )
    return self

def set_color_by_xyz_func(
    self,
    glsl_snippet: str,
    min_value: float = -5.0,
    max_value: float = 5.0,
    colormap: str = "viridis"
) -> Self:
    """
    通过GLSL表达式（基于x、y、z坐标）生成颜色
    输入的GLSL表达式需返回float类型，用于映射到颜色映射表
    """
    # TODO：实现一个通过修改点数据而非Shader代码的版本
    # 将GLSL表达式中的x/y/z替换为point.x/point.y/point.z（适配Shader顶点数据）
    for char in "xyz":
        glsl_snippet = glsl_snippet.replace(char, "point." + char)
    # 获取指定颜色映射表（如viridis）的RGB列表
    rgb_list = get_colormap_list(colormap)
    # 注入颜色计算代码：将GLSL表达式结果映射到颜色映射表
    self.set_color_by_code(
        "color.rgb = float_to_color({}, {}, {}, {});".format(
            glsl_snippet,          # 基于坐标的GLSL表达式（返回float）
            float(min_value),      # 表达式结果的最小值（用于归一化）
            float(max_value),      # 表达式结果的最大值（用于归一化）
            get_colormap_code(rgb_list)  # 颜色映射表的GLSL代码
        )
    )
    return self

# Shader数据相关方法

def init_shader_wrapper(self, ctx: Context):
    # 初始化Shader包装器（连接Python数据与GPU Shader的桥梁）
    self.shader_wrapper = ShaderWrapper(
        ctx=ctx,  # 图形上下文（如OpenGL上下文）
        vert_data=self.data,  # 顶点数据（如位置、颜色）
        shader_folder=self.shader_folder,  # Shader文件所在文件夹路径
        mobject_uniforms=self.uniforms,  # 对象的Uniform变量
        texture_paths=self.texture_paths,  # 纹理文件路径列表
        depth_test=self.depth_test,  # 是否启用深度测试
        render_primitive=self.render_primitive,  # 渲染图元（如三角形、线段）
        code_replacements=self.shader_code_replacements,  # Shader代码替换规则
    )

def refresh_shader_wrapper_id(self):
    # 刷新家族成员的Shader包装器ID（确保Shader识别状态更新）
    for submob in self.get_family():
        if submob.shader_wrapper is not None:
            # 同步深度测试状态到Shader包装器
            submob.shader_wrapper.depth_test = submob.depth_test
            # 刷新Shader包装器的ID（标记状态已变）
            submob.shader_wrapper.refresh_id()
    # 标记当前对象及其所有祖先的数据已变更（触发后续重渲染）
    for mob in (self, *self.get_ancestors()):
        mob._data_has_changed = True
    return self

def get_shader_wrapper(self, ctx: Context) -> ShaderWrapper:
    # 获取Shader包装器，若未初始化则先创建
    if self.shader_wrapper is None:
        self.init_shader_wrapper(ctx)
    return self.shader_wrapper

def get_shader_wrapper_list(self, ctx: Context) -> list[ShaderWrapper]:
    # 获取家族中所有含点数据成员的Shader包装器列表（按Shader ID分组）
    # 筛选出有顶点数据的家族成员（排除空对象）
    family = self.family_members_with_points()
    # 按Shader ID分组（相同Shader的对象共享一个包装器，优化渲染效率）
    batches = batch_by_property(family, lambda sm: sm.get_shader_wrapper(ctx).get_id())

    result = []
    for submobs, sid in batches:
        # 取组内第一个对象的Shader包装器（同组共享）
        shader_wrapper = submobs[0].shader_wrapper
        # 收集组内所有对象的Shader数据（顶点数据等）
        data_list = [sm.get_shader_data() for sm in submobs]
        # 将数据读入Shader包装器（准备批量渲染）
        shader_wrapper.read_in(data_list)
        result.append(shader_wrapper)
    return result

    def get_shader_data(self) -> np.ndarray:
        indices = self.get_shader_vert_indices()
        if indices is not None:
            return self.data[indices]
        else:
            return self.data

    def get_uniforms(self):
        return self.uniforms

    def get_shader_vert_indices(self) -> Optional[np.ndarray]:
        return None

    def render(self, ctx: Context, camera_uniforms: dict):
        if self._data_has_changed:
            self.shader_wrappers = self.get_shader_wrapper_list(ctx)
            self._data_has_changed = False
        for shader_wrapper in self.shader_wrappers:
            shader_wrapper.update_program_uniforms(camera_uniforms)
            shader_wrapper.pre_render()
            shader_wrapper.render()

    # Event Handlers
    """
        Event handling follows the Event Bubbling model of DOM in javascript.
        Return false to stop the event bubbling.
        To learn more visit https://www.quirksmode.org/js/events_order.html

        Event Callback Argument is a callable function taking two arguments:
            1. Mobject
            2. EventData
    """

    def init_event_listners(self):
        self.event_listners: list[EventListener] = []

    def add_event_listner(
        self,
        event_type: EventType,
        event_callback: Callable[[Mobject, dict[str]]]
    ):
        event_listner = EventListener(self, event_type, event_callback)
        self.event_listners.append(event_listner)
        EVENT_DISPATCHER.add_listner(event_listner)
        return self

    def remove_event_listner(
        self,
        event_type: EventType,
        event_callback: Callable[[Mobject, dict[str]]]
    ):
        event_listner = EventListener(self, event_type, event_callback)
        while event_listner in self.event_listners:
            self.event_listners.remove(event_listner)
        EVENT_DISPATCHER.remove_listner(event_listner)
        return self

    def clear_event_listners(self, recurse: bool = True):
        self.event_listners = []
        if recurse:
            for submob in self.submobjects:
                submob.clear_event_listners(recurse=recurse)
        return self

    def get_event_listners(self):
        return self.event_listners

    def get_family_event_listners(self):
        return list(it.chain(*[sm.get_event_listners() for sm in self.get_family()]))

    def get_has_event_listner(self):
        return any(
            mob.get_event_listners()
            for mob in self.get_family()
        )

    def add_mouse_motion_listner(self, callback):
        self.add_event_listner(EventType.MouseMotionEvent, callback)

    def remove_mouse_motion_listner(self, callback):
        self.remove_event_listner(EventType.MouseMotionEvent, callback)

    def add_mouse_press_listner(self, callback):
        self.add_event_listner(EventType.MousePressEvent, callback)

    def remove_mouse_press_listner(self, callback):
        self.remove_event_listner(EventType.MousePressEvent, callback)

    def add_mouse_release_listner(self, callback):
        self.add_event_listner(EventType.MouseReleaseEvent, callback)

    def remove_mouse_release_listner(self, callback):
        self.remove_event_listner(EventType.MouseReleaseEvent, callback)

    def add_mouse_drag_listner(self, callback):
        self.add_event_listner(EventType.MouseDragEvent, callback)

    def remove_mouse_drag_listner(self, callback):
        self.remove_event_listner(EventType.MouseDragEvent, callback)

    def add_mouse_scroll_listner(self, callback):
        self.add_event_listner(EventType.MouseScrollEvent, callback)

    def remove_mouse_scroll_listner(self, callback):
        self.remove_event_listner(EventType.MouseScrollEvent, callback)

    def add_key_press_listner(self, callback):
        self.add_event_listner(EventType.KeyPressEvent, callback)

    def remove_key_press_listner(self, callback):
        self.remove_event_listner(EventType.KeyPressEvent, callback)

    def add_key_release_listner(self, callback):
        self.add_event_listner(EventType.KeyReleaseEvent, callback)

    def remove_key_release_listner(self, callback):
        self.remove_event_listner(EventType.KeyReleaseEvent, callback)

    # Errors

    def throw_error_if_no_points(self):
        if not self.has_points():
            message = "Cannot call Mobject.{} " +\
                      "for a Mobject with no points"
            caller_name = sys._getframe(1).f_code.co_name
            raise Exception(message.format(caller_name))


class Group(Mobject, Generic[SubmobjectType]):
    def __init__(self, *mobjects: SubmobjectType | Iterable[SubmobjectType], **kwargs):
        super().__init__(**kwargs)
        self._ingest_args(*mobjects)

    def _ingest_args(self, *args: Mobject | Iterable[Mobject]):
        if len(args) == 0:
            return
        if all(isinstance(mob, Mobject) for mob in args):
            self.add(*args)
        elif isinstance(args[0], Iterable):
            self.add(*args[0])
        else:
            raise Exception(f"Invalid argument to Group of type {type(args[0])}")

    def __add__(self, other: Mobject | Group) -> Self:
        assert isinstance(other, Mobject)
        return self.add(other)

    # This is just here to make linters happy with references to things like Group(...)[0]
    def __getitem__(self, index) -> SubmobjectType:
        return super().__getitem__(index)


class Point(Mobject):
    def __init__(
        self,
        location: Vect3 = ORIGIN,
        artificial_width: float = 1e-6,
        artificial_height: float = 1e-6,
        **kwargs
    ):
        self.artificial_width = artificial_width
        self.artificial_height = artificial_height
        super().__init__(**kwargs)
        self.set_location(location)

    def get_width(self) -> float:
        return self.artificial_width

    def get_height(self) -> float:
        return self.artificial_height

    def get_location(self) -> Vect3:
        return self.get_points()[0].copy()

    def get_bounding_box_point(self, *args, **kwargs) -> Vect3:
        return self.get_location()

    def set_location(self, new_loc: npt.ArrayLike) -> Self:
        self.set_points(np.array(new_loc, ndmin=2, dtype=float))
        return self


class _AnimationBuilder:
    def __init__(self, mobject: Mobject):
        self.mobject = mobject
        self.overridden_animation = None
        self.mobject.generate_target()
        self.is_chaining = False
        self.methods: list[Callable] = []
        self.anim_args = {}
        self.can_pass_args = True

    def __getattr__(self, method_name: str):
        method = getattr(self.mobject.target, method_name)
        self.methods.append(method)
        has_overridden_animation = hasattr(method, "_override_animate")

        if (self.is_chaining and has_overridden_animation) or self.overridden_animation:
            raise NotImplementedError(
                "Method chaining is currently not supported for " + \
                "overridden animations"
            )

        def update_target(*method_args, **method_kwargs):
            if has_overridden_animation:
                self.overridden_animation = method._override_animate(
                    self.mobject, *method_args, **method_kwargs
                )
            else:
                method(*method_args, **method_kwargs)
            return self

        self.is_chaining = True
        return update_target

    def __call__(self, **kwargs):
        return self.set_anim_args(**kwargs)

    def __dir__(self) -> list[str]:
        """
        Extend attribute list of _AnimationBuilder object to include mobject attributes
        for better autocompletion in the IPython terminal when using interactive mode.
        """
        methods = super().__dir__()
        mobject_methods = [
            attr for attr in dir(self.mobject)
            if not attr.startswith('_')
        ]
        return sorted(set(methods+mobject_methods))

    def set_anim_args(self, **kwargs):
        '''
        You can change the args of :class:`~manimlib.animation.transform.Transform`, such as

        - ``run_time``
        - ``time_span``
        - ``rate_func``
        - ``lag_ratio``
        - ``path_arc``
        - ``path_func``

        and so on.
        '''

        if not self.can_pass_args:
            raise ValueError(
                "Animation arguments can only be passed by calling ``animate`` " + \
                "or ``set_anim_args`` and can only be passed once",
            )

        self.anim_args = kwargs
        self.can_pass_args = False
        return self

    def build(self):
        from manimlib.animation.transform import _MethodAnimation

        if self.overridden_animation:
            return self.overridden_animation

        return _MethodAnimation(self.mobject, self.methods, **self.anim_args)


def override_animate(method):
    def decorator(animation_method):
        method._override_animate = animation_method
        return animation_method

    return decorator


class _UpdaterBuilder:
    def __init__(self, mobject: Mobject):
        self.mobject = mobject

    def __getattr__(self, method_name: str):
        def add_updater(*method_args, **method_kwargs):
            self.mobject.add_updater(
                lambda m: getattr(m, method_name)(*method_args, **method_kwargs)
            )
            return self
        return add_updater


class _FunctionalUpdaterBuilder:
    def __init__(self, mobject: Mobject):
        self.mobject = mobject

    def __getattr__(self, method_name: str):
        def add_updater(*method_args, **method_kwargs):
            self.mobject.add_updater(
                lambda m: getattr(m, method_name)(
                    *(arg() for arg in method_args),
                    **{
                        key: value()
                        for key, value in method_kwargs.items()
                    }
                )
            )
            return self
        return add_updater
