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
        fam1 = self.family_members_with_points()
        fam2 = mobject.family_members_with_points()
        if len(fam1) != len(fam2):
            return False
        for m1, m2 in zip(fam1, fam2):
            if m1.get_num_points() != m2.get_num_points():
                return False
            if not m1.data.dtype == m2.data.dtype:
                return False
            for key in m1.data.dtype.names:
                if not np.isclose(m1.data[key], m2.data[key]).all():
                    return False
            if set(m1.uniforms).difference(m2.uniforms):
                return False
            for key in m1.uniforms:
                value1 = m1.uniforms[key]
                value2 = m2.uniforms[key]
                if isinstance(value1, np.ndarray) and isinstance(value2, np.ndarray) and not value1.size == value2.size:
                    return False
                if not np.isclose(value1, value2).all():
                    return False
        return True

    def has_same_shape_as(self, mobject: Mobject) -> bool:
        # Normalize both point sets by centering and making height 1
        points1, points2 = (
            (m.get_all_points() - m.get_center()) / m.get_height()
            for m in (self, mobject)
        )
        if len(points1) != len(points2):
            return False
        return bool(np.isclose(points1, points2, atol=self.get_width() * 1e-2).all())

    # Creating new Mobjects from this one

    def replicate(self, n: int) -> Self:
        group_class = self.get_group_class()
        return group_class(*(self.copy() for _ in range(n)))

    def get_grid(
        self,
        n_rows: int,
        n_cols: int,
        height: float | None = None,
        width: float | None = None,
        group_by_rows: bool = False,
        group_by_cols: bool = False,
        **kwargs
    ) -> Self:
        """
        Returns a new mobject containing multiple copies of this one
        arranged in a grid
        """
        total = n_rows * n_cols
        grid = self.replicate(total)
        if group_by_cols:
            kwargs["fill_rows_first"] = False
        grid.arrange_in_grid(n_rows, n_cols, **kwargs)
        if height is not None:
            grid.set_height(height)
        if width is not None:
            grid.set_height(width)

        group_class = self.get_group_class()
        if group_by_rows:
            return group_class(*(grid[n:n + n_cols] for n in range(0, total, n_cols)))
        elif group_by_cols:
            return group_class(*(grid[n:n + n_rows] for n in range(0, total, n_rows)))
        else:
            return grid

    # Updating

    def init_updaters(self):
        self.updaters: list[Updater] = list()
        self._has_updaters_in_family: Optional[bool] = False
        self.updating_suspended: bool = False

    def update(self, dt: float = 0, recurse: bool = True) -> Self:
        if not self.has_updaters() or self.updating_suspended:
            return self
        if recurse:
            for submob in self.submobjects:
                submob.update(dt, recurse)
        for updater in self.updaters:
            # This is hacky, but if an updater takes dt as an arg,
            # it will be passed the change in time from here
            if "dt" in updater.__code__.co_varnames:
                updater(self, dt=dt)
            else:
                updater(self)
        return self

    def get_updaters(self) -> list[Updater]:
        return self.updaters

    def add_updater(self, update_func: Updater, call: bool = True) -> Self:
        self.updaters.append(update_func)
        if call:
            self.update(dt=0)
        self.refresh_has_updater_status()
        self.update()
        return self

    def insert_updater(self, update_func: Updater, index=0):
        self.updaters.insert(index, update_func)
        self.refresh_has_updater_status()
        return self

    def remove_updater(self, update_func: Updater) -> Self:
        while update_func in self.updaters:
            self.updaters.remove(update_func)
        self.refresh_has_updater_status()
        return self

    def clear_updaters(self, recurse: bool = True) -> Self:
        for mob in self.get_family(recurse):
            mob.updaters = []
            mob._has_updaters_in_family = False
        for parent in self.get_ancestors():
            parent._has_updaters_in_family = False
        return self

    def match_updaters(self, mobject: Mobject) -> Self:
        self.updaters = list(mobject.updaters)
        self.refresh_has_updater_status()
        return self

    def suspend_updating(self, recurse: bool = True) -> Self:
        self.updating_suspended = True
        if recurse:
            for submob in self.submobjects:
                submob.suspend_updating(recurse)
        return self

    def resume_updating(self, recurse: bool = True, call_updater: bool = True) -> Self:
        self.updating_suspended = False
        if recurse:
            for submob in self.submobjects:
                submob.resume_updating(recurse)
        for parent in self.parents:
            parent.resume_updating(recurse=False, call_updater=False)
        if call_updater:
            self.update(dt=0, recurse=recurse)
        return self

    def has_updaters(self) -> bool:
        if self._has_updaters_in_family is None:
            # Recompute and save
            self._has_updaters_in_family = bool(self.updaters) or any(
                sm.has_updaters() for sm in self.submobjects
            )
        return self._has_updaters_in_family

    def refresh_has_updater_status(self) -> Self:
        self._has_updaters_in_family = None
        for parent in self.parents:
            parent.refresh_has_updater_status()
        return self

    # Check if mark as static or not for camera

    def is_changing(self) -> bool:
        return self._is_animating or self.has_updaters()

    def set_animating_status(self, is_animating: bool, recurse: bool = True) -> Self:
        for mob in (*self.get_family(recurse), *self.get_ancestors()):
            mob._is_animating = is_animating
        return self

    # Transforming operations

    def shift(self, vector: Vect3) -> Self:
        self.apply_points_function(
            lambda points: points + vector,
            about_edge=None,
            works_on_bounding_box=True,
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
        Default behavior is to scale about the center of the mobject.
        The argument about_edge can be a vector, indicating which side of
        the mobject to scale about, e.g., mob.scale(about_edge = RIGHT)
        scales about mob.get_right().

        Otherwise, if about_point is given a value, scaling is done with
        respect to that point.
        """
        if isinstance(scale_factor, numbers.Number):
            scale_factor = max(scale_factor, min_scale_factor)
        else:
            scale_factor = np.array(scale_factor).clip(min=min_scale_factor)
        self.apply_points_function(
            lambda points: scale_factor * points,
            about_point=about_point,
            about_edge=about_edge,
            works_on_bounding_box=True,
        )
        for mob in self.get_family():
            mob._handle_scale_side_effects(scale_factor)
        return self

    def _handle_scale_side_effects(self, scale_factor):
        # In case subclasses, such as DecimalNumber, need to make
        # any other changes when the size gets altered
        pass

    def stretch(self, factor: float, dim: int, **kwargs) -> Self:
        def func(points):
            points[:, dim] *= factor
            return points
        self.apply_points_function(func, works_on_bounding_box=True, **kwargs)
        return self

    def rotate_about_origin(self, angle: float, axis: Vect3 = OUT) -> Self:
        return self.rotate(angle, axis, about_point=ORIGIN)

    def rotate(
        self,
        angle: float,
        axis: Vect3 = OUT,
        about_point: Vect3 | None = None,
        **kwargs
    ) -> Self:
        rot_matrix_T = rotation_matrix_transpose(angle, axis)
        self.apply_points_function(
            lambda points: np.dot(points, rot_matrix_T),
            about_point,
            **kwargs
        )
        return self

    def flip(self, axis: Vect3 = UP, **kwargs) -> Self:
        return self.rotate(TAU / 2, axis, **kwargs)

    def apply_function(self, function: Callable[[np.ndarray], np.ndarray], **kwargs) -> Self:
        # Default to applying matrix about the origin, not mobjects center
        if len(kwargs) == 0:
            kwargs["about_point"] = ORIGIN
        self.apply_points_function(
            lambda points: np.array([function(p) for p in points]),
            **kwargs
        )
        return self

    def apply_function_to_position(self, function: Callable[[np.ndarray], np.ndarray]) -> Self:
        self.move_to(function(self.get_center()))
        return self

    def apply_function_to_submobject_positions(
        self,
        function: Callable[[np.ndarray], np.ndarray]
    ) -> Self:
        for submob in self.submobjects:
            submob.apply_function_to_position(function)
        return self

    def apply_matrix(self, matrix: npt.ArrayLike, **kwargs) -> Self:
        # Default to applying matrix about the origin, not mobjects center
        if ("about_point" not in kwargs) and ("about_edge" not in kwargs):
            kwargs["about_point"] = ORIGIN
        full_matrix = np.identity(self.dim)
        matrix = np.array(matrix)
        full_matrix[:matrix.shape[0], :matrix.shape[1]] = matrix
        self.apply_points_function(
            lambda points: np.dot(points, full_matrix.T),
            **kwargs
        )
        return self

    def apply_complex_function(self, function: Callable[[complex], complex], **kwargs) -> Self:
        def R3_func(point):
            x, y, z = point
            xy_complex = function(complex(x, y))
            return [
                xy_complex.real,
                xy_complex.imag,
                z
            ]
        return self.apply_function(R3_func, **kwargs)

    def wag(
        self,
        direction: Vect3 = RIGHT,
        axis: Vect3 = DOWN,
        wag_factor: float = 1.0
    ) -> Self:
        for mob in self.family_members_with_points():
            alphas = np.dot(mob.get_points(), np.transpose(axis))
            alphas -= min(alphas)
            alphas /= max(alphas)
            alphas = alphas**wag_factor
            mob.set_points(mob.get_points() + np.dot(
                alphas.reshape((len(alphas), 1)),
                np.array(direction).reshape((1, mob.dim))
            ))
        return self

    # Positioning methods

    def center(self) -> Self:
        self.shift(-self.get_center())
        return self

    def align_on_border(
        self,
        direction: Vect3,
        buff: float = DEFAULT_MOBJECT_TO_EDGE_BUFF
    ) -> Self:
        """
        Direction just needs to be a vector pointing towards side or
        corner in the 2d plane.
        """
        target_point = np.sign(direction) * (FRAME_X_RADIUS, FRAME_Y_RADIUS, 0)
        point_to_align = self.get_bounding_box_point(direction)
        shift_val = target_point - point_to_align - buff * np.array(direction)
        shift_val = shift_val * abs(np.sign(direction))
        self.shift(shift_val)
        return self

    def to_corner(
        self,
        corner: Vect3 = LEFT + DOWN,
        buff: float = DEFAULT_MOBJECT_TO_EDGE_BUFF
    ) -> Self:
        return self.align_on_border(corner, buff)

    def to_edge(
        self,
        edge: Vect3 = LEFT,
        buff: float = DEFAULT_MOBJECT_TO_EDGE_BUFF
    ) -> Self:
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
        if isinstance(mobject_or_point, Mobject):
            mob = mobject_or_point
            if index_of_submobject_to_align is not None:
                target_aligner = mob[index_of_submobject_to_align]
            else:
                target_aligner = mob
            target_point = target_aligner.get_bounding_box_point(
                aligned_edge + direction
            )
        else:
            target_point = mobject_or_point
        if submobject_to_align is not None:
            aligner = submobject_to_align
        elif index_of_submobject_to_align is not None:
            aligner = self[index_of_submobject_to_align]
        else:
            aligner = self
        point_to_align = aligner.get_bounding_box_point(aligned_edge - direction)
        self.shift((target_point - point_to_align + buff * direction) * coor_mask)
        return self

    def shift_onto_screen(self, **kwargs) -> Self:
        space_lengths = [FRAME_X_RADIUS, FRAME_Y_RADIUS]
        for vect in UP, DOWN, LEFT, RIGHT:
            dim = np.argmax(np.abs(vect))
            buff = kwargs.get("buff", DEFAULT_MOBJECT_TO_EDGE_BUFF)
            max_val = space_lengths[dim] - buff
            edge_center = self.get_edge_center(vect)
            if np.dot(edge_center, vect) > max_val:
                self.to_edge(vect, **kwargs)
        return self

    def is_off_screen(self) -> bool:
        if self.get_left()[0] > FRAME_X_RADIUS:
            return True
        if self.get_right()[0] < -FRAME_X_RADIUS:
            return True
        if self.get_bottom()[1] > FRAME_Y_RADIUS:
            return True
        if self.get_top()[1] < -FRAME_Y_RADIUS:
            return True
        return False

    def stretch_about_point(self, factor: float, dim: int, point: Vect3) -> Self:
        return self.stretch(factor, dim, about_point=point)

    def stretch_in_place(self, factor: float, dim: int) -> Self:
        # Now redundant with stretch
        return self.stretch(factor, dim)

    def rescale_to_fit(self, length: float, dim: int, stretch: bool = False, **kwargs) -> Self:
        old_length = self.length_over_dim(dim)
        if old_length == 0:
            return self
        if stretch:
            self.stretch(length / old_length, dim, **kwargs)
        else:
            self.scale(length / old_length, **kwargs)
        return self

    def stretch_to_fit_width(self, width: float, **kwargs) -> Self:
        return self.rescale_to_fit(width, 0, stretch=True, **kwargs)

    def stretch_to_fit_height(self, height: float, **kwargs) -> Self:
        return self.rescale_to_fit(height, 1, stretch=True, **kwargs)

    def stretch_to_fit_depth(self, depth: float, **kwargs) -> Self:
        return self.rescale_to_fit(depth, 2, stretch=True, **kwargs)

    def set_width(self, width: float, stretch: bool = False, **kwargs) -> Self:
        return self.rescale_to_fit(width, 0, stretch=stretch, **kwargs)

    def set_height(self, height: float, stretch: bool = False, **kwargs) -> Self:
        return self.rescale_to_fit(height, 1, stretch=stretch, **kwargs)

    def set_depth(self, depth: float, stretch: bool = False, **kwargs) -> Self:
        return self.rescale_to_fit(depth, 2, stretch=stretch, **kwargs)

    def set_max_width(self, max_width: float, **kwargs) -> Self:
        if self.get_width() > max_width:
            self.set_width(max_width, **kwargs)
        return self

    def set_max_height(self, max_height: float, **kwargs) -> Self:
        if self.get_height() > max_height:
            self.set_height(max_height, **kwargs)
        return self

    def set_max_depth(self, max_depth: float, **kwargs) -> Self:
        if self.get_depth() > max_depth:
            self.set_depth(max_depth, **kwargs)
        return self

    def set_min_width(self, min_width: float, **kwargs) -> Self:
        if self.get_width() < min_width:
            self.set_width(min_width, **kwargs)
        return self

    def set_min_height(self, min_height: float, **kwargs) -> Self:
        if self.get_height() < min_height:
            self.set_height(min_height, **kwargs)
        return self

    def set_min_depth(self, min_depth: float, **kwargs) -> Self:
        if self.get_depth() < min_depth:
            self.set_depth(min_depth, **kwargs)
        return self

    def set_shape(
        self,
        width: Optional[float] = None,
        height: Optional[float] = None,
        depth: Optional[float] = None,
        **kwargs
    ) -> Self:
        if width is not None:
            self.set_width(width, stretch=True, **kwargs)
        if height is not None:
            self.set_height(height, stretch=True, **kwargs)
        if depth is not None:
            self.set_depth(depth, stretch=True, **kwargs)
        return self

    def set_coord(self, value: float, dim: int, direction: Vect3 = ORIGIN) -> Self:
        curr = self.get_coord(dim, direction)
        shift_vect = np.zeros(self.dim)
        shift_vect[dim] = value - curr
        self.shift(shift_vect)
        return self

    def set_x(self, x: float, direction: Vect3 = ORIGIN) -> Self:
        return self.set_coord(x, 0, direction)

    def set_y(self, y: float, direction: Vect3 = ORIGIN) -> Self:
        return self.set_coord(y, 1, direction)

    def set_z(self, z: float, direction: Vect3 = ORIGIN) -> Self:
        return self.set_coord(z, 2, direction)

    def set_z_index(self, z_index: int) -> Self:
        self.z_index = z_index
        return self

    def space_out_submobjects(self, factor: float = 1.5, **kwargs) -> Self:
        self.scale(factor, **kwargs)
        for submob in self.submobjects:
            submob.scale(1. / factor)
        return self

    def move_to(
        self,
        point_or_mobject: Mobject | Vect3,
        aligned_edge: Vect3 = ORIGIN,
        coor_mask: Vect3 = np.array([1, 1, 1])
    ) -> Self:
        if isinstance(point_or_mobject, Mobject):
            target = point_or_mobject.get_bounding_box_point(aligned_edge)
        else:
            target = point_or_mobject
        point_to_align = self.get_bounding_box_point(aligned_edge)
        self.shift((target - point_to_align) * coor_mask)
        return self

    def replace(self, mobject: Mobject, dim_to_match: int = 0, stretch: bool = False) -> Self:
        if not mobject.get_num_points() and not mobject.submobjects:
            self.scale(0)
            return self
        if stretch:
            for i in range(self.dim):
                self.rescale_to_fit(mobject.length_over_dim(i), i, stretch=True)
        else:
            self.rescale_to_fit(
                mobject.length_over_dim(dim_to_match),
                dim_to_match,
                stretch=False
            )
        self.shift(mobject.get_center() - self.get_center())
        return self

    def surround(
        self,
        mobject: Mobject,
        dim_to_match: int = 0,
        stretch: bool = False,
        buff: float = MED_SMALL_BUFF
    ) -> Self:
        self.replace(mobject, dim_to_match, stretch)
        length = mobject.length_over_dim(dim_to_match)
        self.scale((length + buff) / length)
        return self

    def put_start_and_end_on(self, start: Vect3, end: Vect3) -> Self:
        curr_start, curr_end = self.get_start_and_end()
        curr_vect = curr_end - curr_start
        if np.all(curr_vect == 0):
            raise Exception("Cannot position endpoints of closed loop")
        target_vect = end - start
        self.scale(
            get_norm(target_vect) / get_norm(curr_vect),
            about_point=curr_start,
        )
        self.rotate(
            angle_of_vector(target_vect) - angle_of_vector(curr_vect),
        )
        self.rotate(
            np.arctan2(curr_vect[2], get_norm(curr_vect[:2])) - np.arctan2(target_vect[2], get_norm(target_vect[:2])),
            axis=np.array([-target_vect[1], target_vect[0], 0]),
        )
        self.shift(start - self.get_start())
        return self

    # Color functions

    @affects_family_data
    def set_rgba_array(
        self,
        rgba_array: npt.ArrayLike,
        name: str = "rgba",
        recurse: bool = False
    ) -> Self:
        for mob in self.get_family(recurse):
            data = mob.data if mob.get_num_points() > 0 else mob._data_defaults
            data[name][:] = rgba_array
        return self

    def set_color_by_rgba_func(
        self,
        func: Callable[[Vect3Array], Vect4Array],
        recurse: bool = True
    ) -> Self:
        """
        Func should take in a point in R3 and output an rgba value
        """
        for mob in self.get_family(recurse):
            mob.set_rgba_array(func(mob.get_points()))
        return self

    def set_color_by_rgb_func(
        self,
        func: Callable[[Vect3Array], Vect3Array],
        opacity: float = 1,
        recurse: bool = True
    ) -> Self:
        """
        Func should take in a point in R3 and output an rgb value
        """
        for mob in self.get_family(recurse):
            points = mob.get_points()
            opacity = np.ones((points.shape[0], 1)) * opacity
            mob.set_rgba_array(np.hstack((func(points), opacity)))
        return self

    @affects_family_data
    def set_rgba_array_by_color(
        self,
        color: ManimColor | Iterable[ManimColor] | None = None,
        opacity: float | Iterable[float] | None = None,
        name: str = "rgba",
        recurse: bool = True
    ) -> Self:
        for mob in self.get_family(recurse):
            data = mob.data if mob.has_points() > 0 else mob._data_defaults
            if color is not None:
                rgbs = np.array(list(map(color_to_rgb, listify(color))))
                if 1 < len(rgbs):
                    rgbs = resize_with_interpolation(rgbs, len(data))
                data[name][:, :3] = rgbs
            if opacity is not None:
                if not isinstance(opacity, (float, int, np.floating)):
                    opacity = resize_with_interpolation(np.array(opacity), len(data))
                data[name][:, 3] = opacity
        return self

    def set_color(
        self,
        color: ManimColor | Iterable[ManimColor] | None,
        opacity: float | Iterable[float] | None = None,
        recurse: bool = True
    ) -> Self:
        self.set_rgba_array_by_color(color, opacity, recurse=False)
        # Recurse to submobjects differently from how set_rgba_array_by_color
        # in case they implement set_color differently
        if recurse:
            for submob in self.submobjects:
                submob.set_color(color, recurse=True)
        return self

    def set_opacity(
        self,
        opacity: float | Iterable[float] | None,
        recurse: bool = True
    ) -> Self:
        self.set_rgba_array_by_color(color=None, opacity=opacity, recurse=False)
        if recurse:
            for submob in self.submobjects:
                submob.set_opacity(opacity, recurse=True)
        return self

    def get_color(self) -> str:
        return rgb_to_hex(self.data["rgba"][0, :3])

    def get_opacity(self) -> float:
        return float(self.data["rgba"][0, 3])

    def get_opacities(self) -> float:
        return self.data["rgba"][:, 3]

    def set_color_by_gradient(self, *colors: ManimColor) -> Self:
        if self.has_points():
            self.set_color(colors)
        else:
            self.set_submobject_colors_by_gradient(*colors)
        return self

    def set_submobject_colors_by_gradient(self, *colors: ManimColor) -> Self:
        if len(colors) == 0:
            raise Exception("Need at least one color")
        elif len(colors) == 1:
            return self.set_color(*colors)

        # mobs = self.family_members_with_points()
        mobs = self.submobjects
        new_colors = color_gradient(colors, len(mobs))

        for mob, color in zip(mobs, new_colors):
            mob.set_color(color)
        return self

    def fade(self, darkness: float = 0.5, recurse: bool = True) -> Self:
        self.set_opacity(1.0 - darkness, recurse=recurse)

    def get_shading(self) -> np.ndarray:
        return self.uniforms["shading"]

    def set_shading(
        self,
        reflectiveness: float | None = None,
        gloss: float | None = None,
        shadow: float | None = None,
        recurse: bool = True
    ) -> Self:
        """
        Larger reflectiveness makes things brighter when facing the light
        Larger shadow makes faces opposite the light darker
        Makes parts bright where light gets reflected toward the camera
        """
        for mob in self.get_family(recurse):
            shading = mob.uniforms["shading"]
            for i, value in enumerate([reflectiveness, gloss, shadow]):
                if value is not None:
                    shading[i] = value
            mob.set_uniform(shading=shading, recurse=False)
        return self

    def get_reflectiveness(self) -> float:
        return self.get_shading()[0]

    def get_gloss(self) -> float:
        return self.get_shading()[1]

    def get_shadow(self) -> float:
        return self.get_shading()[2]

    def set_reflectiveness(self, reflectiveness: float, recurse: bool = True) -> Self:
        self.set_shading(reflectiveness=reflectiveness, recurse=recurse)
        return self

    def set_gloss(self, gloss: float, recurse: bool = True) -> Self:
        self.set_shading(gloss=gloss, recurse=recurse)
        return self

    def set_shadow(self, shadow: float, recurse: bool = True) -> Self:
        self.set_shading(shadow=shadow, recurse=recurse)
        return self

    # Background rectangle

    def add_background_rectangle(
        self,
        color: ManimColor | None = None,
        opacity: float = 1.0,
        **kwargs
    ) -> Self:
        from manimlib.mobject.shape_matchers import BackgroundRectangle
        self.background_rectangle = BackgroundRectangle(
            self, color=color,
            fill_opacity=opacity,
            **kwargs
        )
        self.add_to_back(self.background_rectangle)
        return self

    def add_background_rectangle_to_submobjects(self, **kwargs) -> Self:
        for submobject in self.submobjects:
            submobject.add_background_rectangle(**kwargs)
        return self

    def add_background_rectangle_to_family_members_with_points(self, **kwargs) -> Self:
        for mob in self.family_members_with_points():
            mob.add_background_rectangle(**kwargs)
        return self

    # Getters

    def get_bounding_box_point(self, direction: Vect3) -> Vect3:
        bb = self.get_bounding_box()
        indices = (np.sign(direction) + 1).astype(int)
        return np.array([
            bb[indices[i]][i]
            for i in range(3)
        ])

    def get_edge_center(self, direction: Vect3) -> Vect3:
        return self.get_bounding_box_point(direction)

    def get_corner(self, direction: Vect3) -> Vect3:
        return self.get_bounding_box_point(direction)

    def get_all_corners(self):
        bb = self.get_bounding_box()
        return np.array([
            [bb[indices[-i + 1]][i] for i in range(3)]
            for indices in it.product([0, 2], repeat=3)
        ])

    def get_center(self) -> Vect3:
        return self.get_bounding_box()[1]

    def get_center_of_mass(self) -> Vect3:
        return self.get_all_points().mean(0)

    def get_boundary_point(self, direction: Vect3) -> Vect3:
        all_points = self.get_all_points()
        boundary_directions = all_points - self.get_center()
        norms = np.linalg.norm(boundary_directions, axis=1)
        boundary_directions /= np.repeat(norms, 3).reshape((len(norms), 3))
        index = np.argmax(np.dot(boundary_directions, np.array(direction).T))
        return all_points[index]

    def get_continuous_bounding_box_point(self, direction: Vect3) -> Vect3:
        dl, center, ur = self.get_bounding_box()
        corner_vect = (ur - center)
        return center + direction / np.max(np.abs(np.true_divide(
            direction, corner_vect,
            out=np.zeros(len(direction)),
            where=((corner_vect) != 0)
        )))

    def get_top(self) -> Vect3:
        return self.get_edge_center(UP)

    def get_bottom(self) -> Vect3:
        return self.get_edge_center(DOWN)

    def get_right(self) -> Vect3:
        return self.get_edge_center(RIGHT)

    def get_left(self) -> Vect3:
        return self.get_edge_center(LEFT)

    def get_zenith(self) -> Vect3:
        return self.get_edge_center(OUT)

    def get_nadir(self) -> Vect3:
        return self.get_edge_center(IN)

    def length_over_dim(self, dim: int) -> float:
        bb = self.get_bounding_box()
        return abs((bb[2] - bb[0])[dim])

    def get_width(self) -> float:
        return self.length_over_dim(0)

    def get_height(self) -> float:
        return self.length_over_dim(1)

    def get_depth(self) -> float:
        return self.length_over_dim(2)

    def get_shape(self) -> Tuple[float]:
        return tuple(self.length_over_dim(dim) for dim in range(3))

    def get_coord(self, dim: int, direction: Vect3 = ORIGIN) -> float:
        """
        Meant to generalize get_x, get_y, get_z
        """
        return self.get_bounding_box_point(direction)[dim]

    def get_x(self, direction=ORIGIN) -> float:
        return self.get_coord(0, direction)

    def get_y(self, direction=ORIGIN) -> float:
        return self.get_coord(1, direction)

    def get_z(self, direction=ORIGIN) -> float:
        return self.get_coord(2, direction)

    def get_start(self) -> Vect3:
        self.throw_error_if_no_points()
        return self.get_points()[0].copy()

    def get_end(self) -> Vect3:
        self.throw_error_if_no_points()
        return self.get_points()[-1].copy()

    def get_start_and_end(self) -> tuple[Vect3, Vect3]:
        self.throw_error_if_no_points()
        points = self.get_points()
        return (points[0].copy(), points[-1].copy())

    def point_from_proportion(self, alpha: float) -> Vect3:
        points = self.get_points()
        i, subalpha = integer_interpolate(0, len(points) - 1, alpha)
        return interpolate(points[i], points[i + 1], subalpha)

    def pfp(self, alpha):
        """Abbreviation for point_from_proportion"""
        return self.point_from_proportion(alpha)

    def get_pieces(self, n_pieces: int) -> Group:
        template = self.copy()
        template.set_submobjects([])
        alphas = np.linspace(0, 1, n_pieces + 1)
        return Group(*[
            template.copy().pointwise_become_partial(
                self, a1, a2
            )
            for a1, a2 in zip(alphas[:-1], alphas[1:])
        ])

    def get_z_index_reference_point(self) -> Vect3:
        # TODO, better place to define default z_index_group?
        z_index_group = getattr(self, "z_index_group", self)
        return z_index_group.get_center()

    # Match other mobject properties

    def match_color(self, mobject: Mobject) -> Self:
        return self.set_color(mobject.get_color())

    def match_style(self, mobject: Mobject) -> Self:
        self.set_color(mobject.get_color())
        self.set_opacity(mobject.get_opacity())
        self.set_shading(*mobject.get_shading())
        return self

    def match_dim_size(self, mobject: Mobject, dim: int, **kwargs) -> Self:
        return self.rescale_to_fit(
            mobject.length_over_dim(dim), dim,
            **kwargs
        )

    def match_width(self, mobject: Mobject, **kwargs) -> Self:
        return self.match_dim_size(mobject, 0, **kwargs)

    def match_height(self, mobject: Mobject, **kwargs) -> Self:
        return self.match_dim_size(mobject, 1, **kwargs)

    def match_depth(self, mobject: Mobject, **kwargs) -> Self:
        return self.match_dim_size(mobject, 2, **kwargs)

    def match_coord(
        self,
        mobject_or_point: Mobject | Vect3,
        dim: int,
        direction: Vect3 = ORIGIN
    ) -> Self:
        if isinstance(mobject_or_point, Mobject):
            coord = mobject_or_point.get_coord(dim, direction)
        else:
            coord = mobject_or_point[dim]
        return self.set_coord(coord, dim=dim, direction=direction)

    def match_x(
        self,
        mobject_or_point: Mobject | Vect3,
        direction: Vect3 = ORIGIN
    ) -> Self:
        return self.match_coord(mobject_or_point, 0, direction)

    def match_y(
        self,
        mobject_or_point: Mobject | Vect3,
        direction: Vect3 = ORIGIN
    ) -> Self:
        return self.match_coord(mobject_or_point, 1, direction)

    def match_z(
        self,
        mobject_or_point: Mobject | Vect3,
        direction: Vect3 = ORIGIN
    ) -> Self:
        return self.match_coord(mobject_or_point, 2, direction)

    def align_to(
        self,
        mobject_or_point: Mobject | Vect3,
        direction: Vect3 = ORIGIN
    ) -> Self:
        """
        Examples:
        mob1.align_to(mob2, UP) moves mob1 vertically so that its
        top edge lines ups with mob2's top edge.

        mob1.align_to(mob2, alignment_vect = RIGHT) moves mob1
        horizontally so that it's center is directly above/below
        the center of mob2
        """
        if isinstance(mobject_or_point, Mobject):
            point = mobject_or_point.get_bounding_box_point(direction)
        else:
            point = mobject_or_point

        for dim in range(self.dim):
            if direction[dim] != 0:
                self.set_coord(point[dim], dim, direction)
        return self

    def get_group_class(self):
        return Group

    # Alignment

    def is_aligned_with(self, mobject: Mobject) -> bool:
        if len(self.data) != len(mobject.data):
            return False
        if len(self.submobjects) != len(mobject.submobjects):
            return False
        return all(
            sm1.is_aligned_with(sm2)
            for sm1, sm2 in zip(self.submobjects, mobject.submobjects)
        )

    def align_data_and_family(self, mobject: Mobject) -> Self:
        self.align_family(mobject)
        self.align_data(mobject)
        return self

    def align_data(self, mobject: Mobject) -> Self:
        for mob1, mob2 in zip(self.get_family(), mobject.get_family()):
            mob1.align_points(mob2)
        return self

    def align_points(self, mobject: Mobject) -> Self:
        max_len = max(self.get_num_points(), mobject.get_num_points())
        for mob in (self, mobject):
            mob.resize_points(max_len, resize_func=resize_preserving_order)
        return self

    def align_family(self, mobject: Mobject) -> Self:
        mob1 = self
        mob2 = mobject
        n1 = len(mob1)
        n2 = len(mob2)
        if n1 != n2:
            mob1.add_n_more_submobjects(max(0, n2 - n1))
            mob2.add_n_more_submobjects(max(0, n1 - n2))
        # Recurse
        for sm1, sm2 in zip(mob1.submobjects, mob2.submobjects):
            sm1.align_family(sm2)
        return self

    def push_self_into_submobjects(self) -> Self:
        copy = self.copy()
        copy.set_submobjects([])
        self.resize_points(0)
        self.add(copy)
        return self

    def add_n_more_submobjects(self, n: int) -> Self:
        if n == 0:
            return self

        curr = len(self.submobjects)
        if curr == 0:
            # If empty, simply add n point mobjects
            null_mob = self.copy()
            null_mob.set_points([self.get_center()])
            self.set_submobjects([
                null_mob.copy()
                for k in range(n)
            ])
            return self
        target = curr + n
        repeat_indices = (np.arange(target) * curr) // target
        split_factors = [
            (repeat_indices == i).sum()
            for i in range(curr)
        ]
        new_submobs = []
        for submob, sf in zip(self.submobjects, split_factors):
            new_submobs.append(submob)
            for k in range(1, sf):
                new_submobs.append(submob.invisible_copy())
        self.set_submobjects(new_submobs)
        return self

    def invisible_copy(self) -> Self:
        return self.copy().set_opacity(0)

    # Interpolate

    def interpolate(
        self,
        mobject1: Mobject,
        mobject2: Mobject,
        alpha: float,
        path_func: Callable[[np.ndarray, np.ndarray, float], np.ndarray] = straight_path
    ) -> Self:
        keys = [k for k in self.data.dtype.names if k not in self.locked_data_keys]
        if keys:
            self.note_changed_data()
        for key in keys:
            md1 = mobject1.data[key]
            md2 = mobject2.data[key]
            if key in self.const_data_keys:
                md1 = md1[0]
                md2 = md2[0]
            if key in self.pointlike_data_keys:
                self.data[key] = path_func(md1, md2, alpha)
            else:
                self.data[key] = (1 - alpha) * md1 + alpha * md2

        for key in self.uniforms:
            if key in self.locked_uniform_keys:
                continue
            if key not in mobject1.uniforms or key not in mobject2.uniforms:
                continue
            self.uniforms[key] = (1 - alpha) * mobject1.uniforms[key] + alpha * mobject2.uniforms[key]
        self.bounding_box[:] = path_func(mobject1.bounding_box, mobject2.bounding_box, alpha)
        return self

    def pointwise_become_partial(self, mobject, a, b) -> Self:
        """
        Set points in such a way as to become only
        part of mobject.
        Inputs 0 <= a < b <= 1 determine what portion
        of mobject to become.
        """
        # To be implemented in subclass
        return self

    # Locking data

    def lock_data(self, keys: Iterable[str]) -> Self:
        """
        To speed up some animations, particularly transformations,
        it can be handy to acknowledge which pieces of data
        won't change during the animation so that calls to
        interpolate can skip this, and so that it's not
        read into the shader_wrapper objects needlessly
        """
        if self.has_updaters():
            return self
        self.locked_data_keys = set(keys)
        return self

    def lock_uniforms(self, keys: Iterable[str]) -> Self:
        if self.has_updaters():
            return self
        self.locked_uniform_keys = set(keys)
        return self

    def lock_matching_data(self, mobject1: Mobject, mobject2: Mobject) -> Self:
        tuples = zip(
            self.get_family(),
            mobject1.get_family(),
            mobject2.get_family(),
        )
        for sm, sm1, sm2 in tuples:
            if not sm.data.dtype == sm1.data.dtype == sm2.data.dtype:
                continue
            sm.lock_data(
                key for key in sm.data.dtype.names
                if arrays_match(sm1.data[key], sm2.data[key])
            )
            sm.lock_uniforms(
                key for key in self.uniforms
                if all(listify(mobject1.uniforms.get(key, 0) == mobject2.uniforms.get(key, 0)))
            )
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
        for mob in self.get_family():
            mob.locked_data_keys = set()
            mob.const_data_keys = set()
            mob.locked_uniform_keys = set()
        return self

    # Operations touching shader uniforms

    @staticmethod
    def affects_shader_info_id(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            self.refresh_shader_wrapper_id()
            return result
        return wrapper

    @affects_shader_info_id
    def set_uniform(self, recurse: bool = True, **new_uniforms) -> Self:
        for mob in self.get_family(recurse):
            mob.uniforms.update(new_uniforms)
        return self

    @affects_shader_info_id
    def fix_in_frame(self, recurse: bool = True) -> Self:
        self.set_uniform(recurse, is_fixed_in_frame=1.0)
        return self

    @affects_shader_info_id
    def unfix_from_frame(self, recurse: bool = True) -> Self:
        self.set_uniform(recurse, is_fixed_in_frame=0.0)
        return self

    def is_fixed_in_frame(self) -> bool:
        return bool(self.uniforms["is_fixed_in_frame"])

    @affects_shader_info_id
    def apply_depth_test(self, recurse: bool = True) -> Self:
        for mob in self.get_family(recurse):
            mob.depth_test = True
        return self

    @affects_shader_info_id
    def deactivate_depth_test(self, recurse: bool = True) -> Self:
        for mob in self.get_family(recurse):
            mob.depth_test = False
        return self

    def set_clip_plane(
        self,
        vect: Vect3 | None = None,
        threshold: float | None = None,
        recurse=True
    ) -> Self:
        for submob in self.get_family(recurse):
            if vect is not None:
                submob.uniforms["clip_plane"][:3] = vect
            if threshold is not None:
                submob.uniforms["clip_plane"][3] = threshold
        return self

    def deactivate_clip_plane(self) -> Self:
        self.uniforms["clip_plane"][:] = 0
        return self

    # Shader code manipulation

    @affects_data
    def replace_shader_code(self, old: str, new: str) -> Self:
        for mob in self.get_family():
            mob.shader_code_replacements[old] = new
            mob.shader_wrapper = None
        return self

    def set_color_by_code(self, glsl_code: str) -> Self:
        """
        Takes a snippet of code and inserts it into a
        context which has the following variables:
        vec4 color, vec3 point, vec3 unit_normal.
        The code should change the color variable
        """
        self.replace_shader_code(
            "///// INSERT COLOR FUNCTION HERE /////",
            glsl_code
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
        Pass in a glsl expression in terms of x, y and z which returns
        a float.
        """
        # TODO, add a version of this which changes the point data instead
        # of the shader code
        for char in "xyz":
            glsl_snippet = glsl_snippet.replace(char, "point." + char)
        rgb_list = get_colormap_list(colormap)
        self.set_color_by_code(
            "color.rgb = float_to_color({}, {}, {}, {});".format(
                glsl_snippet,
                float(min_value),
                float(max_value),
                get_colormap_code(rgb_list)
            )
        )
        return self

    # For shader data

    def init_shader_wrapper(self, ctx: Context):
        self.shader_wrapper = ShaderWrapper(
            ctx=ctx,
            vert_data=self.data,
            shader_folder=self.shader_folder,
            mobject_uniforms=self.uniforms,
            texture_paths=self.texture_paths,
            depth_test=self.depth_test,
            render_primitive=self.render_primitive,
            code_replacements=self.shader_code_replacements,
        )

    def refresh_shader_wrapper_id(self):
        for submob in self.get_family():
            if submob.shader_wrapper is not None:
                submob.shader_wrapper.depth_test = submob.depth_test
                submob.shader_wrapper.refresh_id()
        for mob in (self, *self.get_ancestors()):
            mob._data_has_changed = True
        return self

    def get_shader_wrapper(self, ctx: Context) -> ShaderWrapper:
        if self.shader_wrapper is None:
            self.init_shader_wrapper(ctx)
        return self.shader_wrapper

    def get_shader_wrapper_list(self, ctx: Context) -> list[ShaderWrapper]:
        family = self.family_members_with_points()
        batches = batch_by_property(family, lambda sm: sm.get_shader_wrapper(ctx).get_id())

        result = []
        for submobs, sid in batches:
            shader_wrapper = submobs[0].shader_wrapper
            data_list = [sm.get_shader_data() for sm in submobs]
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
