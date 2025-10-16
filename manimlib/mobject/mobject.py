# 导入Python 3.7+的类型提示向后兼容支持，允许在类定义中使用尚未完全定义的类作为类型注解
from __future__ import annotations

# 导入复制模块，用于对象的深拷贝和浅拷贝操作
import copy
# 导入functools中的wraps装饰器，用于装饰器定义时保留被装饰函数的元信息（如函数名、文档字符串）
from functools import wraps
# 导入itertools模块并简写为it，用于高效创建和操作迭代器（如循环、排列组合等）
import itertools as it
# 导入os模块，用于与操作系统交互（如文件路径操作、环境变量访问等）
import os
# 导入pickle模块，用于对象的序列化和反序列化（将对象保存到文件或从文件加载对象）
import pickle
# 导入random模块，用于生成随机数（如随机选择、随机打乱等）
import random
# 导入sys模块，用于访问Python解释器的相关变量和功能（如命令行参数、系统路径等）
import sys

# 导入moderngl模块，用于访问现代OpenGL功能，支持高性能图形渲染（如着色器、纹理、顶点缓冲等）
import moderngl
# 导入numbers模块，用于检查对象是否为数值类型（如int、float、complex等）
import numbers
# 导入numpy模块并简写为np，用于高性能数值计算和数组操作（如矩阵运算、数据存储等）
import numpy as np

from manimlib.constants import DEFAULT_MOBJECT_TO_EDGE_BUFF
from manimlib.constants import DEFAULT_MOBJECT_TO_MOBJECT_BUFF
from manimlib.constants import DOWN, IN, LEFT, ORIGIN, OUT, RIGHT, UP
from manimlib.constants import FRAME_X_RADIUS, FRAME_Y_RADIUS
from manimlib.constants import MED_SMALL_BUFF
from manimlib.constants import TAU
from manimlib.constants import DEFAULT_MOBJECT_COLOR
from manimlib.event_handler import EVENT_DISPATCHER
from manimlib.event_handler.event_listner import EventListener
from manimlib.event_handler.event_type import EventType
from manimlib.logger import log
from manimlib.shader_wrapper import ShaderWrapper
from manimlib.utils.color import color_gradient
from manimlib.utils.color import color_to_rgb
from manimlib.utils.color import get_colormap_list
from manimlib.utils.color import rgb_to_hex
from manimlib.utils.iterables import arrays_match
from manimlib.utils.iterables import array_is_constant
from manimlib.utils.iterables import batch_by_property
from manimlib.utils.iterables import list_update
from manimlib.utils.iterables import listify
from manimlib.utils.iterables import resize_array
from manimlib.utils.iterables import resize_preserving_order
from manimlib.utils.iterables import resize_with_interpolation
from manimlib.utils.bezier import integer_interpolate
from manimlib.utils.bezier import interpolate
from manimlib.utils.paths import straight_path
from manimlib.utils.shaders import get_colormap_code
from manimlib.utils.space_ops import angle_of_vector
from manimlib.utils.space_ops import get_norm
from manimlib.utils.space_ops import rotation_matrix_transpose

from typing import TYPE_CHECKING
from typing import TypeVar, Generic, Iterable
# 定义泛型类型变量SubmobjectType，限定为Mobject类及其子类，用于指定子对象的类型
SubmobjectType = TypeVar('SubmobjectType', bound='Mobject')


# 仅在类型检查时导入相关模块和类型，避免运行时循环导入或不必要依赖
if TYPE_CHECKING:
    from typing import Callable, Iterator, Union, Tuple, Optional, Any
    import numpy.typing as npt
    # 导入Manim相关自定义类型：颜色、3D向量、4D向量数组、3D向量数组、统一变量字典、自身类型
    from manimlib.typing import ManimColor, Vect3, Vect4Array, Vect3Array, UniformDict, Self
    from moderngl.context import Context  # 导入ModernGL的上下文类型

    T = TypeVar('T')  # 通用泛型类型变量，用于泛型函数/类
    # 基于时间的更新器类型：接收Mobject实例和时间参数，返回Mobject或None
    TimeBasedUpdater = Callable[["Mobject", float], "Mobject" | None]
    # 非时间更新器类型：仅接收Mobject实例，返回Mobject或None
    NonTimeUpdater = Callable[["Mobject"], "Mobject" | None]
    # 更新器类型：是基于时间的更新器或非时间更新器的联合类型
    Updater = Union[TimeBasedUpdater, NonTimeUpdater]


class Mobject(object):
    """
    数学图形对象基类（Mathematical Object）：Manim中所有可视化图形的核心父类，
    定义了图形的基础属性（如颜色、透明度）、数据结构（如顶点、颜色数据）及核心初始化逻辑，
    子类（如矩形、文本、线段）均基于此类扩展。
    """
    # 图形维度：默认3（支持3D渲染）
    dim: int = 3
    # 着色器文件夹路径：子类可指定自定义着色器所在文件夹，默认空字符串（使用默认着色器）
    shader_folder: str = ""
    # 渲染图元类型：默认使用ModernGL的三角形带（TRIANGLE_STRIP），用于指定图形的渲染方式
    render_primitive: int = moderngl.TRIANGLE_STRIP
    # 数据类型：与顶点着色器（vert shader）的属性匹配，定义顶点数据的存储结构
    data_dtype: np.dtype = np.dtype([
        ('point', np.float32, (3,)),  # 顶点坐标：3个float32类型值（x, y, z）
        ('rgba', np.float32, (4,)),   # 顶点颜色：4个float32类型值（r, g, b, a）
    ])
    # 需对齐的数据键：指定需要内存对齐的数据字段，默认仅'point'（优化渲染性能）
    aligned_data_keys = ['point']
    # 类点数据键：指定表示"点"的数据集，默认仅'point'（用于点相关操作，如平移、旋转）
    pointlike_data_keys = ['point']

    def __init__(
        self,
        color: ManimColor = DEFAULT_MOBJECT_COLOR,  # 图形颜色，默认使用全局默认图形颜色
        opacity: float = 1.0,  # 透明度，默认1.0（完全不透明）
        shading: Tuple[float, float, float] = (0.0, 0.0, 0.0),  # 着色参数，默认(0,0,0)（无额外着色）
        # 纹理路径字典：键为纹理名称，值为纹理文件路径，默认None（无纹理）
        texture_paths: dict[str, str] | None = None,
        # 是否固定在画面中：为True时，图形不随相机旋转/平移而移动，默认False
        is_fixed_in_frame: bool = False,
        depth_test: bool = False,  # 是否启用深度测试：为True时，按Z轴深度决定渲染层级，默认False
        z_index: int = 0,  # Z轴索引：用于控制图形的渲染顺序，值越大越靠后渲染，默认0
    ):
        # 初始化图形的基础样式属性
        self.color = color
        self.opacity = opacity
        self.shading = shading
        self.texture_paths = texture_paths
        self.depth_test = depth_test
        self.z_index = z_index

        # 初始化图形的内部状态属性
        self.submobjects: list[Mobject] = []  # 存储当前图形的子对象列表
        self.parents: list[Mobject] = []     # 存储当前图形的父对象列表
        self.family: list[Mobject] | None = [self]  # 存储图形的家族（自身及所有子对象），默认仅包含自身
        self.locked_data_keys: set[str] = set()  # 锁定的数据键：这些数据字段不允许被修改
        self.const_data_keys: set[str] = set()   # 常量数据键：这些数据字段视为常量，不参与更新
        self.locked_uniform_keys: set[str] = set()  # 锁定的统一变量键：这些着色器统一变量不允许被修改
        self.saved_state = None  # 保存的状态：用于暂存图形的某个状态（如位置、颜色）
        self.target = None       # 目标状态：用于动画过渡的目标状态
        # 包围盒：3x3的3D向量数组，存储图形的轴对齐包围盒（AABB），默认全零
        self.bounding_box: Vect3Array = np.zeros((3, 3))
        self.shader_wrapper: Optional[ShaderWrapper] = None  # 着色器包装器：管理图形的着色器实例，默认None
        self._is_animating: bool = False  # 动画状态标记：是否正在执行动画，默认False
        self._needs_new_bounding_box: bool = True  # 包围盒更新标记：是否需要重新计算包围盒，默认True
        self._data_has_changed: bool = True  # 数据变更标记：顶点/颜色数据是否已变更，默认True
        # 着色器代码替换字典：用于动态替换着色器代码中的特定内容，默认空字典
        self.shader_code_replacements: dict[str, str] = dict()

        # 调用核心初始化方法，完成数据、统一变量、更新器等关键组件的初始化
        self.init_data()          # 初始化顶点、颜色等核心数据
        self.init_uniforms()      # 初始化着色器所需的统一变量（如投影矩阵、颜色）
        self.init_updaters()      # 初始化更新器：用于实时更新图形状态（如位置、颜色）
        self.init_event_listners()# 初始化事件监听器：用于响应鼠标、键盘等交互事件
        self.init_points()        # 初始化顶点：设置图形的初始顶点位置
        self.init_colors()        # 初始化颜色：设置图形的初始颜色（含透明度）

        # 根据参数启用深度测试
        if self.depth_test:
            self.apply_depth_test()
        # 根据参数将图形固定在画面中
        if is_fixed_in_frame:
            self.fix_in_frame()

    def __str__(self):
        """
        字符串表示方法：返回当前图形对象的类名（如"Rectangle"、"Text"），
        用于print()或字符串拼接时的默认显示。
        """
        return self.__class__.__name__

    def __add__(self, other: Mobject) -> Mobject:
        """
        加法运算符重载：将当前图形与另一个Mobject实例合并为一个Group（组），
        仅支持两个Mobject实例相加，否则抛出断言错误。
        
        参数
        -----
        other : Mobject
            待合并的另一个图形对象
        
        返回
        -----
        Mobject
            包含当前图形和other的Group实例
        
        断言
        -----
        断言other是Mobject实例，否则报错
        """
        assert isinstance(other, Mobject)
        # 调用当前图形的组类（默认Group），将两个图形合并为组
        return self.get_group_class()(self, other)

    def __mul__(self, other: int) -> Mobject:
        """
        乘法运算符重载：将当前图形复制指定次数（other次），生成包含多个复制图形的Group，
        仅支持与整数相乘，否则抛出断言错误。
        
        参数
        -----
        other : int
            复制次数（需为非负整数）
        
        返回
        -----
        Mobject
            包含other个当前图形复制体的Group实例
        
        断言
        -----
        断言other是整数，否则报错
        """
        assert isinstance(other, int)
        # 调用replicate方法复制图形，返回包含复制体的Group
        return self.replicate(other)

    def init_data(self, length: int = 0):
        """
        初始化图形的核心数据：创建空的顶点数据数组（按data_dtype格式），
        并初始化数据默认值（用于后续填充新顶点数据）。
        
        参数
        -----
        length : int, optional
            初始数据数组的长度（即顶点数量），默认0（空数组）
        """
        # 创建指定长度的空数据数组， dtype匹配图形的核心数据类型
        self.data = np.zeros(length, dtype=self.data_dtype)
        # 初始化数据默认值：1个元素的数组，用于后续生成新顶点时提供默认值
        self._data_defaults = np.ones(1, dtype=self.data_dtype)

    def init_uniforms(self):
        """
        初始化着色器统一变量：创建存储统一变量的字典，包含固定在画面标记、着色参数、裁剪平面等
        基础统一变量，这些变量会传递给着色器以控制渲染效果。
        """
        self.uniforms: UniformDict = {
            "is_fixed_in_frame": 0.0,  # 是否固定在画面：0.0为否，1.0为是（默认否）
            "shading": np.array(self.shading, dtype=float),  # 着色参数：转换为float数组传入着色器
            "clip_plane": np.zeros(4),  # 裁剪平面参数：默认全零（无裁剪）
        }

    def init_colors(self):
        """
        初始化图形颜色：调用set_color方法，将图形的初始颜色和透明度应用到所有顶点，
        确保图形创建时即拥有设定的颜色样式。
        """
        self.set_color(self.color, self.opacity)

    def init_points(self):
        """
        初始化图形顶点：子类需重写此方法以定义特定图形的顶点位置（如矩形的四个顶点），
        父类中仅作为占位，无实际逻辑。
        """
        # Typically implemented in subclass, unless purposefully left blank（通常在子类中实现，除非有意留空）
        pass

    def set_uniforms(self, uniforms: dict) -> Self:
        """
        设置着色器统一变量：批量更新uniforms字典中的键值对，若值为numpy数组则先复制，
        避免外部数组修改影响内部统一变量，支持链式调用。
        
        参数
        -----
        uniforms : dict
            待更新的统一变量字典，键为统一变量名，值为对应数据（如数值、数组）
        
        返回
        -----
        Self
            当前图形对象自身（支持链式调用，如mobj.set_uniforms(...).scale(...)）
        """
        for key, value in uniforms.items():
            # 若值为numpy数组，复制一份以避免外部修改影响
            if isinstance(value, np.ndarray):
                value = value.copy()
            self.uniforms[key] = value
        return self

    @property
    def animate(self) -> _AnimationBuilder | Self:
        """
        动画构建器属性：通过mobject.animate.method()的形式调用图形方法时，
        会生成对应的动画对象（可传入Scene.play()播放），而非直接执行方法。
        
        示例：
        scene.play(square.animate.scale(2))  # 播放正方形放大2倍的动画
        
        来源：
        Borrowed from https://github.com/ManimCommunity/manim/
        
        返回
        -----
        _AnimationBuilder | Self
            动画构建器实例，用于构建方法对应的动画
        """
        return _AnimationBuilder(self)

    @property
    def always(self) -> _UpdaterBuilder:
        """
        实时更新器构建器属性：通过mobject.always.method(*args)的形式调用图形方法时，
        会将该方法注册为实时更新器，在每一帧自动执行，确保图形状态实时更新。
        
        示例：
        square.always.move_to(mouse_point)  # 每帧让正方形跟随鼠标位置
        
        返回
        -----
        _UpdaterBuilder
            更新器构建器实例，用于注册实时执行的方法
        """
        return _UpdaterBuilder(self)

    @property
    def f_always(self) -> _FunctionalUpdaterBuilder:
        """
        函数式实时更新器构建器属性：与always类似，但支持传入返回值为目标类型的函数作为参数，
        每帧会先执行函数获取参数值，再调用图形方法，适用于参数需动态计算的场景。
        
        示例：
        def get_rand_color(): return random.choice([RED, BLUE])
        square.f_always.set_color(get_rand_color)  # 每帧随机设置正方形颜色
        
        返回
        -----
        _FunctionalUpdaterBuilder
            函数式更新器构建器实例，用于注册参数为函数的实时方法
        """
        return _FunctionalUpdaterBuilder(self)

    def note_changed_data(self, recurse_up: bool = True) -> Self:
        """
        标记数据已变更：设置_data_has_changed标记为True（通知渲染系统数据需更新），
        并可选择向上父级父对象传播此标记，确保父对象也知晓数据变更。
        
        参数
        -----
        recurse_up : bool, optional
            是否向上递归通知父对象，默认True（通知所有父对象）
        """
        self._data_has_changed = True  # 标记数据已变更
        # 若需要向上递归，遍历所有父对象并调用其note_changed_data方法
        if recurse_up:
            for mob in self.parents:
                mob.note_changed_data()
        return self

    @staticmethod
    def affects_data(func: Callable[..., T]) -> Callable[..., T]:
        """
        装饰器：用于标记修改图形数据的方法，被装饰的方法执行后会自动调用note_changed_data，
        确保数据变更被正确标记。
        
        参数
        -----
        func : Callable[..., T]
            被装饰的方法（通常是修改顶点、颜色等数据的方法）
        
        返回
        -----
        Callable[..., T]
            包装后的方法，执行原方法后自动标记数据变更
        """
        @wraps(func)  # 保留原方法的元信息（如函数名、文档字符串）
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)  # 执行原方法
            self.note_changed_data()  # 标记数据已变更
            return result
        return wrapper

    @staticmethod
    def affects_family_data(func: Callable[..., T]) -> Callable[..., T]:
        """
        装饰器：用于标记修改家族数据的方法，被装饰的方法执行后会通知家族中所有含顶点的成员
        标记数据变更，适用于修改会影响子对象数据的方法。
        
        参数
        -----
        func : Callable[..., T]
            被装饰的方法（通常是影响子对象数据的方法）
        
        返回
        -----
        Callable[..., T]
            包装后的方法，执行原方法后自动标记家族所有成员的数据变更
        """
        @wraps(func)  # 保留原方法的元信息
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)  # 执行原方法
            # 遍历家族中所有含顶点的成员，标记其数据已变更
            for mob in self.family_members_with_points():
                mob.note_changed_data()
            return result
        return wrapper

    # Only these methods should directly affect points（仅以下方法应直接修改顶点数据）

    @affects_data
    def set_data(self, data: np.ndarray) -> Self:
        """
        直接设置图形的核心数据：替换整个数据数组（需与图形的数据类型匹配），
        自动调整顶点数量并标记数据变更。
        
        参数
        -----
        data : np.ndarray
            新的核心数据数组，dtype必须与self.data_dtype一致
        
        断言
        -----
        断言输入数据的dtype与图形的data_dtype一致，否则报错
        """
        assert data.dtype == self.data.dtype  # 确保数据类型匹配
        self.resize_points(len(data))  # 调整顶点数量以匹配新数据长度
        self.data[:] = data  # 替换数据内容
        return self

    @affects_data
    def resize_points(
        self,
        new_length: int,
        resize_func: Callable[[np.ndarray, int], np.ndarray] = resize_array
    ) -> Self:
        """
        调整顶点数量：按新长度调整数据数组大小，可指定自定义调整函数，
        自动更新包围盒并标记数据变更。
        
        参数
        -----
        new_length : int
            新的顶点数量（数据数组长度）
        resize_func : Callable, optional
            调整数组大小的函数，默认使用resize_array（保留原有数据并填充默认值）
        """
        if new_length == 0:
            # 若新长度为0且当前数据非空，保存当前第一个数据作为默认值
            if len(self.data) > 0:
                self._data_defaults[:1] = self.data[:1]
        elif self.get_num_points() == 0:
            # 若当前无顶点且新长度非0，用默认值初始化数据
            self.data = self._data_defaults.copy()

        # 调用调整函数调整数据数组长度
        self.data = resize_func(self.data, new_length)
        self.refresh_bounding_box()  # 重新计算包围盒
        return self

    @affects_data
    def set_points(self, points: Vect3Array | list[Vect3]) -> Self:
        """
        设置顶点坐标：替换图形的所有顶点坐标，自动调整顶点数量并保持其他数据字段不变，
        标记数据变更。
        
        参数
        -----
        points : Vect3Array | list[Vect3]
            新的顶点坐标集合（3D向量数组或列表）
        """
        # 调整顶点数量，使用保持顺序的调整函数
        self.resize_points(len(points), resize_func=resize_preserving_order)
        # 替换数据中的"point"字段（顶点坐标）
        self.data["point"][:] = points
        return self

    @affects_data
    def append_points(self, new_points: Vect3Array) -> Self:
        """
        追加顶点坐标：在现有顶点后添加新顶点，新顶点的非坐标数据（如颜色）默认继承最后一个旧顶点，
        自动更新包围盒并标记数据变更。
        
        参数
        -----
        new_points : Vect3Array
            待追加的新顶点坐标集合（3D向量数组）
        """
        n = self.get_num_points()  # 获取当前顶点数量
        self.resize_points(n + len(new_points))  # 调整顶点数量以容纳新顶点
        # 新顶点的非坐标数据默认继承最后一个旧顶点的属性
        self.data[n:] = self.data[n - 1]
        # 替换新顶点的坐标数据
        self.data["point"][n:] = new_points
        self.refresh_bounding_box()  # 重新计算包围盒
        return self

    @affects_family_data
    def reverse_points(self) -> Self:
        """
        反转顶点顺序：反转当前图形及所有子对象的顶点数据顺序（影响渲染顺序），
        通知家族所有成员标记数据变更。
        """
        # 遍历家族中所有对象，反转其数据顺序
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
        对家族所有成员的点数据应用函数：遍历当前图形及子对象，对所有“类点数据”（如顶点坐标）
        执行指定函数（如平移、旋转），支持围绕特定点或边界点运算，可选择是否作用于包围盒。
        
        参数
        -----
        func : Callable[[np.ndarray], np.ndarray]
            作用于点数据的函数，输入为点数组，输出为处理后的点数组（如lambda x: x*2实现缩放）
        about_point : Vect3 | None, optional
            运算围绕的中心点，默认None（直接作用于点）
        about_edge : Vect3, optional
            若未指定about_point，围绕图形的某个边界点运算（如RIGHT表示右边界中心），默认ORIGIN
        works_on_bounding_box : bool, optional
            函数是否直接作用于包围盒而非原始点数据，默认False（作用于原始点）
            当前图形对象自身（支持链式调用）
        """
        # 若未指定about_point但指定了about_edge，获取对应边界点作为中心点
        if about_point is None and about_edge is not None:
            about_point = self.get_bounding_box_point(about_edge)

        # 遍历家族中所有图形对象（自身及子对象）
        for mob in self.get_family():
            # 收集当前对象的所有“类点数据”数组（如point字段），仅处理有顶点的对象
            arrs = [mob.data[key] for key in mob.pointlike_data_keys if mob.has_points()]
            # 若需作用于包围盒，将包围盒数组也加入待处理列表
            if works_on_bounding_box:
                arrs.append(mob.get_bounding_box())

            # 对每个点数组应用函数
            for arr in arrs:
                if about_point is None:
                    # 无中心点：直接对数组执行函数
                    arr[:] = func(arr)
                else:
                    # 有中心点：先将点平移到中心点坐标系，执行函数后再平移回去
                    arr[:] = func(arr - about_point) + about_point

        # 根据是否作用于包围盒，决定如何更新包围盒
        if not works_on_bounding_box:
            # 作用于原始点：递归更新自身及子对象的包围盒
            self.refresh_bounding_box(recurse_down=True)
        else:
            # 作用于包围盒：仅更新父对象的包围盒（子对象包围盒已直接修改）
            for parent in self.parents:
                parent.refresh_bounding_box()
        return self

    @affects_data
    def match_points(self, mobject: Mobject) -> Self:
        """
        匹配目标图形的点数据：调整当前图形的顶点数量以匹配目标图形，
        并复制目标图形的所有“类点数据”（如顶点坐标），使当前图形与目标图形点结构一致。
        
        参数
        -----
        mobject : Mobject
            目标图形对象，需与当前图形的“类点数据”字段匹配
        """
        # 调整当前图形顶点数量，与目标图形一致，保持点顺序
        self.resize_points(len(mobject.data), resize_func=resize_preserving_order)
        # 复制目标图形的所有“类点数据”到当前图形
        for key in self.pointlike_data_keys:
            self.data[key][:] = mobject.data[key]
        return self

    # Others related to points（其他与点数据相关的方法）

    def get_points(self) -> Vect3Array:
        """
        获取当前图形的顶点坐标数组：直接返回data字段中的“point”数据，
        即图形的所有顶点3D坐标。
        
        返回
        -----
        Vect3Array
            顶点坐标数组，形状为(N, 3)，N为顶点数量
        """
        return self.data["point"]

    def clear_points(self) -> Self:
        """
        清空当前图形的顶点数据：将顶点数量调整为0，删除所有顶点，
        标记数据变更。
        """
        self.resize_points(0)
        return self

    def get_num_points(self) -> int:
        """
        获取当前图形的顶点数量：通过顶点坐标数组的长度计算。
        
        返回
        -----
        int
            顶点数量（≥0）
        """
        return len(self.get_points())

    def get_all_points(self) -> Vect3Array:
        """
        获取家族所有成员的顶点坐标：若当前图形有子对象，垂直堆叠自身及所有子对象的顶点数组；
        若无子对象，直接返回自身顶点数组。
        
        返回
        -----
        Vect3Array
            所有顶点坐标的合并数组，形状为(M, 3)，M为家族总顶点数量
        """
        if self.submobjects:
            # 有子对象：递归获取家族所有成员的顶点，垂直堆叠
            return np.vstack([sm.get_points() for sm in self.get_family()])
        else:
            # 无子对象：返回自身顶点数组
            return self.get_points()

    def has_points(self) -> bool:
        """
        判断当前图形是否有顶点数据：通过顶点数量是否大于0判断。
        
        返回
        -----
        bool
            True表示有顶点，False表示无顶点
        """
        return len(self.get_points()) > 0

    def get_bounding_box(self) -> Vect3Array:
        """
        获取当前图形的包围盒：若包围盒需更新（_needs_new_bounding_box为True），
        先调用compute_bounding_box计算新包围盒并缓存，否则直接返回缓存的包围盒。
        
        返回
        -----
        Vect3Array
            包围盒数组，形状为(3, 3)，存储轴对齐包围盒的最小值、最大值等信息
        """
        if self._needs_new_bounding_box:
            # 计算新包围盒并更新缓存
            self.bounding_box[:] = self.compute_bounding_box()
            self._needs_new_bounding_box = False  # 标记包围盒已更新，无需再计算
        return self.bounding_box

    def compute_bounding_box(self) -> Vect3Array:
        """
        计算图形的包围盒：合并自身顶点及所有子对象的包围盒，生成轴对齐的包围盒（AABB），
        包含最小值、中点、最大值三组坐标，用于碰撞检测、定位等场景。
        
        返回
        -----
        Vect3Array
            包围盒数组，形状为(3, dim)（dim为图形维度，默认3），
            分别存储[最小值坐标, 中点坐标, 最大值坐标]
        """
        # 收集所有用于计算包围盒的点：1. 自身顶点；2. 所有子对象的包围盒顶点
        all_points = np.vstack([
            self.get_points(),  # 自身的顶点数组
            *(
                mob.get_bounding_box()  # 遍历家族中除自身外的成员，获取其包围盒
                for mob in self.get_family()[1:]
                if mob.has_points()  # 仅处理有顶点的子对象
            )
        ])
        # 若没有任何点（自身及子对象均无顶点），返回全零包围盒
        if len(all_points) == 0:
            return np.zeros((3, self.dim))
        else:
            # 计算所有点在各维度的最小值和最大值
            mins = all_points.min(0)  # 各维度最小值（包围盒左下角）
            maxs = all_points.max(0)  # 各维度最大值（包围盒右上角）
            mids = (mins + maxs) / 2  # 各维度中点（包围盒中心）
            # 组合为包围盒数组返回
            return np.array([mins, mids, maxs])

    def refresh_bounding_box(
        self,
        recurse_down: bool = False,
        recurse_up: bool = True
    ) -> Self:
        """
        强制刷新包围盒：标记当前图形及指定范围的家族成员需要重新计算包围盒，
        支持向下递归子对象和向上递归父对象，确保包围盒数据同步。
        
        参数
        -----
        recurse_down : bool, optional
            是否向下递归标记所有子对象，默认False（仅标记自身）
        recurse_up : bool, optional
            是否向上递归通知父对象刷新，默认True（父对象同步刷新）
        """
        # 遍历指定范围的家族成员（自身或含子对象），标记需重新计算包围盒
        for mob in self.get_family(recurse_down):
            mob._needs_new_bounding_box = True
        # 若需向上递归，通知所有父对象执行刷新（父对象的包围盒依赖当前对象）
        if recurse_up:
            for parent in self.parents:
                parent.refresh_bounding_box()
        return self

    def are_points_touching(
        self,
        points: Vect3Array,
        buff: float = 0
    ) -> np.ndarray:
        """
        批量判断多个点是否在包围盒内（含缓冲）：检查每个点是否在“包围盒±缓冲”的范围内，
        返回与输入点数量一致的布尔数组，标记每个点是否命中。
        
        参数
        -----
        points : Vect3Array
            待判断的点数组，形状为(N, 3)（N为点数量）
        buff : float, optional
            包围盒的缓冲距离（扩大或缩小包围盒范围），默认0（无缓冲）
        
        返回
        -----
        np.ndarray
            布尔数组，形状为(N,)，True表示对应点在包围盒内（含缓冲）
        """
        bb = self.get_bounding_box()  # 获取当前图形的包围盒
        mins = (bb[0] - buff)  # 包围盒最小值 - 缓冲（扩大下限）
        maxs = (bb[2] + buff)  # 包围盒最大值 + 缓冲（扩大上限）
        # 检查每个点是否在所有维度都满足“≥mins且≤maxs”，返回批量结果
        return ((points >= mins) * (points <= maxs)).all(1)

    def is_point_touching(
        self,
        point: Vect3,
        buff: float = 0
    ) -> bool:
        """
        判断单个点是否在包围盒内（含缓冲）：调用批量判断方法，传入单个点并返回对应结果，
        简化单点命中检测的调用流程。
        
        参数
        -----
        point : Vect3
            待判断的单个3D点（如np.array([1,2,3])）
        buff : float, optional
            包围盒的缓冲距离，默认0（无缓冲）
        
        返回
        -----
        bool
            True表示点在包围盒内（含缓冲），False表示不在
        """
        # 将单点转换为2D数组（适配are_points_touching的输入格式），取第一个结果
        return self.are_points_touching(np.array(point, ndmin=2), buff)[0]

    def is_touching(self, mobject: Mobject, buff: float = 1e-2) -> bool:
        """
        判断当前图形与目标图形是否碰撞（基于包围盒）：检查两个图形的包围盒是否有重叠（含缓冲），
        若不存在“目标图形完全在当前图形左侧/右侧”等完全分离情况，则判定为碰撞。
        
        参数
        -----
        mobject : Mobject
            目标图形对象（待判断是否与当前图形碰撞）
        buff : float, optional
            包围盒的碰撞缓冲，默认1e-2（避免浮点精度导致的误判）
        
        返回
        -----
        bool
            True表示两个图形包围盒重叠（碰撞），False表示完全分离（无碰撞）
        """
        bb1 = self.get_bounding_box()  # 当前图形的包围盒
        bb2 = mobject.get_bounding_box()  # 目标图形的包围盒
        # 检查是否存在完全分离的情况：若均不满足，则判定为碰撞
        return not any((
            (bb2[2] < bb1[0] - buff).any(),  # 目标图形的最大值 < 当前图形最小值-缓冲（目标在左侧）
            (bb2[0] > bb1[2] + buff).any(),  # 目标图形的最小值 > 当前图形最大值+缓冲（目标在右侧）
        ))

    # Family matters

    def __getitem__(self, value: int | slice) -> Mobject:
        """
        索引访问魔法方法：支持通过整数索引或切片获取拆分后的子对象，
        切片时自动将结果封装为当前图形的组类（如Group）。
        
        参数
        -----
        value : int | slice
            整数索引（如0、1）或切片（如1:3），用于指定获取的子对象范围
        
        返回
        -----
        Mobject
            单个子对象（整数索引时）或包含多个子对象的组（切片时）
        """
        if isinstance(value, slice):
            # 切片访问：获取组类，将切片结果封装为组
            GroupClass = self.get_group_class()
            return GroupClass(*self.split().__getitem__(value))
        # 整数索引：直接返回对应子对象
        return self.split().__getitem__(value)

    def __iter__(self) -> Iterator[Self]:
        """
        迭代器魔法方法：使图形对象可迭代，迭代时返回拆分后的子对象序列。
        
        返回
        -----
        Iterator[Self]
            子对象的迭代器
        """
        return iter(self.split())

    def __len__(self) -> int:
        """
        长度魔法方法：返回图形对象拆分后的子对象数量。
        
        返回
        -----
        int
            子对象数量（≥0）
        """
        return len(self.split())

    def split(self) -> list[Self]:
        """
        拆分图形为子对象列表：默认返回存储的子对象列表（submobjects），
        子类可重写以实现自定义拆分逻辑（如按顶点拆分）。
        
        返回
        -----
        list[Self]
            子对象列表
        """
        return self.submobjects

    @affects_data
    def note_changed_family(self, only_changed_order=False) -> Self:
        """
        标记家族结构已变更：清空缓存的家族列表（family），根据需要刷新更新器状态和包围盒，
        并向上通知父对象同步更新家族信息。
        
        参数
        -----
        only_changed_order : bool, optional
            仅子对象顺序变更时为True（无需刷新更新器和包围盒），默认False
        """
        self.family = None  # 清空家族缓存，触发后续重建
        if not only_changed_order:
            self.refresh_has_updater_status()  # 刷新更新器状态
            self.refresh_bounding_box()  # 刷新包围盒
        # 向上通知所有父对象家族结构已变更
        for parent in self.parents:
            parent.note_changed_family()
        return self

    def get_family(self, recurse: bool = True) -> list[Mobject]:
        """
        获取家族成员列表：包含自身及所有子对象（递归），结果缓存以避免重复计算，
        支持非递归模式（仅返回自身）。
        
        参数
        -----
        recurse : bool, optional
            是否递归包含所有子对象，默认True（完整家族）
        
        返回
        -----
        list[Mobject]
            家族成员列表，自身始终在首位， followed by submobjects and their families
        """
        if not recurse:
            return [self]  # 非递归模式：仅返回自身
        # 若家族缓存为空，重建家族列表
        if self.family is None:
            # 递归获取所有子对象的家族，扁平化为单列表
            sub_families = (sm.get_family() for sm in self.submobjects)
            self.family = [self, *it.chain(*sub_families)]  # 自身 + 所有子对象家族
        return self.family

    def family_members_with_points(self) -> list[Mobject]:
        """
        获取家族中含顶点数据的成员：过滤家族列表，仅保留有顶点数据（data非空）的成员。
        
        返回
        -----
        list[Mobject]
            含顶点数据的家族成员列表
        """
        return [m for m in self.get_family() if len(m.data) > 0]

    def get_ancestors(self, extended: bool = False) -> list[Mobject]:
        """
        获取所有祖先对象：包括父对象、祖父对象等，按层级从高到低排序，
        支持扩展模式（包含所有子对象的祖先）。
        
        参数
        -----
        extended : bool, optional
            为True时包含所有家族成员的祖先，默认False（仅当前图形的祖先）
        
        返回
        -----
        list[Mobject]
            祖先对象列表，层级高的在前，无重复
        
        说明
        -----
        Order of result should be from higher members of the hierarchy down.
        （结果顺序为层级从高到低）
        
        If extended is set to true, it includes the ancestors of all family members,
        e.g. any other parents of a submobject
        （extended为True时，包含所有家族成员的祖先，如子对象的其他父对象）
        """
        ancestors = []
        # 待处理对象：extended为True时包含所有家族成员，否则仅自身
        to_process = list(self.get_family(recurse=extended))
        excluded = set(to_process)  # 排除自身及家族成员，避免循环引用
        # 广度优先遍历所有父对象
        while to_process:
            for p in to_process.pop().parents:
                if p not in excluded:
                    ancestors.append(p)
                    to_process.append(p)
                    excluded.add(p)  # 加入排除集，避免重复处理
        # 反转列表，使层级最高的祖先排在前面
        ancestors.reverse()
        # 去重并保留顺序（使用字典键的唯一性）
        return list(dict.fromkeys(ancestors))

    def add(self, *mobjects: Mobject) -> Self:
        """
        向当前图形添加子对象：将一个或多个Mobject实例添加为子对象，
        建立“父-子”关联（当前图形为父，传入对象为子），并更新家族结构。
        
        参数
        -----
        *mobjects : Mobject
            待添加的子对象（一个或多个Mobject实例）
        
        异常
        -----
        Exception
            若尝试添加自身为子对象，抛出“无法包含自身”的异常
        """
        # 禁止添加自身为子对象，避免循环引用
        if self in mobjects:
            raise Exception("Mobject cannot contain self")
        # 遍历每个待添加的子对象
        for mobject in mobjects:
            # 若子对象不在当前图形的子对象列表中，添加到列表
            if mobject not in self.submobjects:
                self.submobjects.append(mobject)
            # 若当前图形不在子对象的父对象列表中，添加到父列表
            if self not in mobject.parents:
                mobject.parents.append(self)
        # 标记家族结构已变更，触发缓存更新
        self.note_changed_family()
        return self

    def remove(
        self,
        *to_remove: Mobject,
        reassemble: bool = True,
        recurse: bool = True
    ) -> Self:
        """
        从当前图形移除子对象：递归或非递归地从自身及家族成员中移除指定子对象，
        解除“父-子”关联，可选择是否重建家族结构。
        
        参数
        -----
        *to_remove : Mobject
            待移除的子对象（一个或多个Mobject实例）
        reassemble : bool, optional
            移除后是否重建家族结构（更新缓存），默认True
        recurse : bool, optional
            是否递归遍历家族成员（自身及所有子对象）进行移除，默认True
        """
        # 遍历目标家族成员（recurse=True时含子对象，否则仅自身）
        for parent in self.get_family(recurse):
            # 遍历每个待移除的子对象
            for child in to_remove:
                # 若子对象在当前父对象的子列表中，移除
                if child in parent.submobjects:
                    parent.submobjects.remove(child)
                # 若当前父对象在子对象的父列表中，移除
                if parent in child.parents:
                    child.parents.remove(parent)
            # 若需要重建家族结构，标记父对象家族变更
            if reassemble:
                parent.note_changed_family()
        return self

    def clear(self) -> Self:
        """
        清空当前图形的所有子对象：调用remove方法移除自身所有子对象，
        不递归处理子对象的子对象，仅清空当前图形的直接子对象。
        """
        # 移除自身所有子对象，recurse=False表示不递归处理子对象的子对象
        self.remove(*self.submobjects, recurse=False)
        return self

    def add_to_back(self, *mobjects: Mobject) -> Self:
        """
        将子对象添加到最底层（渲染顺序最前）：先保留现有子对象，再将新子对象添加到子列表头部，
        确保新子对象渲染时位于现有子对象下方（底层），并更新家族结构。
        
        参数
        -----
        *mobjects : Mobject
            待添加到底层的子对象（一个或多个Mobject实例）
        """
        # 调用list_update函数，将新子对象添加到现有子列表头部（底层）
        self.set_submobjects(list_update(mobjects, self.submobjects))
        return self

    def replace_submobject(self, index: int, new_submob: Mobject) -> Self:
        """
        替换指定索引的子对象：移除子列表中指定索引的旧子对象，添加新子对象，
        解除旧关联、建立新关联，并更新家族结构。
        
        参数
        -----
        index : int
            待替换子对象在子列表中的索引（从0开始）
        new_submob : Mobject
            用于替换的新子对象
        """
        # 获取指定索引的旧子对象
        old_submob = self.submobjects[index]
        # 解除当前图形与旧子对象的父-子关联
        if self in old_submob.parents:
            old_submob.parents.remove(self)
        # 替换子列表中指定索引的子对象
        self.submobjects[index] = new_submob
        # 建立当前图形与新子对象的父-子关联
        new_submob.parents.append(self)
        # 标记家族结构已变更，触发缓存更新
        self.note_changed_family()
        return self
    
    def insert_submobject(self, index: int, new_submob: Mobject) -> Self:
        """
        在指定索引插入子对象：将新子对象插入到子列表的指定位置，
        建立“父-子”关联并更新家族结构，不影响其他子对象的顺序。
        
        参数
        -----
        index : int
            插入位置的索引（从0开始，0表示插入到最前，len(submobjects)表示插入到最后）
        new_submob : Mobject
            待插入的新子对象
        """
        # 在子列表指定索引处插入新子对象
        self.submobjects.insert(index, new_submob)
        # 建立当前图形与新子对象的父-子关联
        new_submob.parents.append(self)
        # 标记家族结构已变更，触发缓存更新
        self.note_changed_family()
        return self

    def set_submobjects(self, submobject_list: list[Mobject]) -> Self:
        """
        批量设置子对象：清空当前所有子对象，再添加新的子对象列表，
        完全替换原有子对象集合，仅在列表不同时执行操作。
        
        参数
        -----
        submobject_list : list[Mobject]
            新的子对象列表（替换原有子列表）
        """
        # 若新列表与当前子列表一致，直接返回（避免重复操作）
        if self.submobjects == submobject_list:
            return self
        self.clear()  # 清空当前所有子对象
        self.add(*submobject_list)  # 添加新的子对象列表
        return self

    def digest_mobject_attrs(self) -> Self:
        """
        整合Mobject类型属性为子对象：遍历当前图形的所有属性，
        将类型为Mobject的属性添加到子列表中，确保这些属性被纳入家族管理。
        
        说明
        -----
        Ensures all attributes which are mobjects are included
        in the submobjects list.
        （确保所有Mobject类型的属性都被包含在子对象列表中）
        """
        # 筛选出所有值为Mobject实例的属性
        mobject_attrs = [x for x in list(self.__dict__.values()) if isinstance(x, Mobject)]
        # 更新子列表：保留原有子对象，添加新的Mobject属性（去重）
        self.set_submobjects(list_update(self.submobjects, mobject_attrs))
        return self

    # Submobject organization（子对象排列相关方法）

    def arrange(
        self,
        direction: Vect3 = RIGHT,
        center: bool = True,
        **kwargs
    ) -> Self:
        """
        按指定方向排列子对象：将子列表中的子对象按顺序沿指定方向（如RIGHT、DOWN）排列，
        相邻子对象自动保持指定间距（通过**kwargs传递buff参数），可选择是否整体居中。
        
        参数
        -----
        direction : Vect3, optional
            排列方向，默认RIGHT（水平向右），可传入DOWN（垂直向下）等方向向量
        center : bool, optional
            排列后是否将整个子对象组居中，默认True（居中）
        **kwargs
            传递给next_to方法的参数，如buff（子对象间间距）
        """
        # 遍历相邻的子对象对（第1个与第2个、第2个与第3个...）
        for m1, m2 in zip(self.submobjects, self.submobjects[1:]):
            # 将后一个子对象（m2）移动到前一个子对象（m1）的指定方向旁
            m2.next_to(m1, direction, **kwargs)
        # 若需居中，将整个当前图形（含所有子对象）居中
        if center:
            self.center()
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
        """
        将子对象排列为网格：按指定行数/列数将子对象排列成网格，自动计算网格单元大小和间距，
        支持按行填充或按列填充，可自定义间距比例和对齐方式。
        
        参数
        -----
        n_rows : int | None, optional
            网格行数，默认None（自动根据列数或子对象数量计算）
        n_cols : int | None, optional
            网格列数，默认None（自动根据行数或子对象数量计算）
        buff : float | None, optional
            统一的水平/垂直间距，设置后覆盖h_buff和v_buff，默认None
        h_buff : float | None, optional
            水平间距（列之间），默认None（按h_buff_ratio和第一个子对象宽度计算）
        v_buff : float | None, optional
            垂直间距（行之间），默认None（按v_buff_ratio和第一个子对象高度计算）
        buff_ratio : float | None, optional
            统一的间距比例（相对于子对象尺寸），设置后覆盖h_buff_ratio和v_buff_ratio，默认None
        h_buff_ratio : float, optional
            水平间距比例（相对于子对象最大宽度），默认0.5（即间距为最大宽度的0.5倍）
        v_buff_ratio : float, optional
            垂直间距比例（相对于子对象最大高度），默认0.5（即间距为最大高度的0.5倍）
        aligned_edge : Vect3, optional
            子对象在网格单元中的对齐边缘，默认ORIGIN（中心对齐）
        fill_rows_first : bool, optional
            是否按行优先填充（先填满一行再填下一行），默认True；False为按列优先填充
        """
        submobs = self.submobjects  # 获取子对象列表
        n_submobs = len(submobs)    # 子对象总数
        # 自动计算行数/列数（若未指定）
        if n_rows is None:
            # 若未指定行数：列数也未指定则按平方根估算，否则按子对象总数//列数计算
            n_rows = int(np.sqrt(n_submobs)) if n_cols is None else n_submobs // n_cols
        if n_cols is None:
            # 若未指定列数：按子对象总数//行数计算
            n_cols = n_submobs // n_rows

        # 处理间距参数（优先级：buff > buff_ratio > 单独h_buff/v_buff）
        if buff is not None:
            h_buff = buff
            v_buff = buff
        else:
            if buff_ratio is not None:
                v_buff_ratio = buff_ratio
                h_buff_ratio = buff_ratio
            # 若未指定水平间距：按比例和子对象最大宽度计算
            if h_buff is None:
                h_buff = h_buff_ratio * max([sm.get_width() for sm in submobs])
            # 若未指定垂直间距：按比例和子对象最大高度计算
            if v_buff is None:
                v_buff = v_buff_ratio * max([sm.get_height() for sm in submobs])

        # 计算网格单元大小（子对象最大尺寸 + 对应间距）
        x_unit = h_buff + max([sm.get_width() for sm in submobs])  # 水平单元（列宽+水平间距）
        y_unit = v_buff + max([sm.get_height() for sm in submobs]) # 垂直单元（行高+垂直间距）

        # 遍历每个子对象，计算并设置其在网格中的位置
        for index, sm in enumerate(submobs):
            # 按行优先/列优先计算子对象的网格坐标（x：列索引，y：行索引）
            if fill_rows_first:
                x, y = index % n_cols, index // n_cols  # 行优先：索引%列数=列号，索引//列数=行号
            else:
                x, y = index // n_rows, index % n_rows  # 列优先：索引//行数=列号，索引%行数=行号
            # 将子对象按指定边缘对齐到原点（网格单元的基准点）
            sm.move_to(ORIGIN, aligned_edge)
            # 按网格坐标偏移子对象（水平方向×x_unit，垂直方向×y_unit，向下为正）
            sm.shift(x * x_unit * RIGHT + y * y_unit * DOWN)
        # 将整个网格组居中
        self.center()
        return self

    def arrange_to_fit_dim(self, length: float, dim: int, about_edge=ORIGIN) -> Self:
        """
        按指定维度适配长度排列子对象：在指定维度（如水平0、垂直1）上，
        根据目标总长度自动计算子对象间距，使所有子对象紧凑排列后总长度恰好匹配目标值，
        并保持排列中心与原始基准边缘对齐。
        
        参数
        -----
        length : float
            目标总长度（子对象总尺寸 + 间距总和）
        dim : int
            目标维度，0=水平（x轴）、1=垂直（y轴）、2=深度（z轴）
        about_edge : Vect3, optional
            排列的基准边缘（如LEFT表示以左侧为基准），默认ORIGIN（中心）
        """
        # 获取排列的基准点（基于指定边缘的包围盒点）
        ref_point = self.get_bounding_box_point(about_edge)
        n_submobs = len(self.submobjects)  # 子对象数量
        # 子对象数量≤1时无需排列，直接返回
        if n_submobs <= 1:
            return self

        # 计算所有子对象在目标维度上的总尺寸（不含间距）
        total_length = sum(sm.length_over_dim(dim) for sm in self.submobjects)
        # 计算子对象间的均匀间距（总长度 - 子对象总尺寸）/（子对象数量-1）
        buff = (length - total_length) / (n_submobs - 1)
        # 创建维度方向向量（仅目标维度为1，其他为0）
        vect = np.zeros(self.dim)
        vect[dim] = 1

        x = 0  # 记录当前子对象在目标维度上的起始位置
        for submob in self.submobjects:
            # 按当前位置设置子对象在目标维度上的坐标（对齐负方向边缘）
            submob.set_coord(x, dim, -vect)
            # 更新下一个子对象的起始位置（当前子对象尺寸 + 间距）
            x += submob.length_over_dim(dim) + buff
        
        # 移动整个组，使排列后的基准边缘与原始基准点对齐
        self.move_to(ref_point, about_edge)
        return self

    def arrange_to_fit_width(self, width: float, about_edge=ORIGIN) -> Self:
        """
        按目标宽度适配排列子对象：调用维度0（水平方向）的适配排列方法，
        使子对象水平排列后的总宽度恰好匹配目标宽度。
        
        参数
        -----
        width : float
            目标水平总宽度
        about_edge : Vect3, optional
            水平排列的基准边缘，默认ORIGIN
        """
        return self.arrange_to_fit_dim(width, 0, about_edge)

    def arrange_to_fit_height(self, height: float, about_edge=ORIGIN) -> Self:
        """
        按目标高度适配排列子对象：调用维度1（垂直方向）的适配排列方法，
        使子对象垂直排列后的总高度恰好匹配目标高度。
        
        参数
        -----
        height : float
            目标垂直总高度
        about_edge : Vect3, optional
            垂直排列的基准边缘，默认ORIGIN
        """
        return self.arrange_to_fit_dim(height, 1, about_edge)

    def arrange_to_fit_depth(self, depth: float, about_edge=ORIGIN) -> Self:
        """
        按目标深度适配排列子对象：调用维度2（深度方向）的适配排列方法，
        使子对象沿深度排列后的总深度恰好匹配目标深度。
        
        参数
        -----
        depth : float
            目标深度总长度
        about_edge : Vect3, optional
            深度排列的基准边缘，默认ORIGIN
        """
        return self.arrange_to_fit_dim(depth, 2, about_edge)

    def sort(
        self,
        point_to_num_func: Callable[[np.ndarray], float] = lambda p: p[0],
        submob_func: Callable[[Mobject], float] | None = None
    ) -> Self:
        """
        对子对象排序：支持两种排序方式，一是通过子对象中心坐标的函数映射值排序，
        二是通过自定义子对象属性函数排序，排序后仅更新子对象顺序，不改变位置。
        
        参数
        -----
        point_to_num_func : Callable[[np.ndarray], float], optional
            坐标映射函数，输入子对象中心坐标，输出排序用数值（默认按x轴坐标排序）
        submob_func : Callable[[Mobject], float] | None, optional
            子对象自定义函数，输入子对象，输出排序用数值（优先级高于point_to_num_func），默认None
        """
        if submob_func is not None:
            # 按自定义子对象函数排序
            self.submobjects.sort(key=submob_func)
        else:
            # 按子对象中心坐标的映射值排序（默认x轴坐标）
            self.submobjects.sort(key=lambda m: point_to_num_func(m.get_center()))
        # 仅子对象顺序变更，标记家族变更时无需刷新更新器和包围盒
        self.note_changed_family(only_changed_order=True)
        return self

    def shuffle(self, recurse: bool = False) -> Self:
        """
        随机打乱子对象顺序：支持递归打乱（同时打乱子对象的子对象），
        仅改变子对象顺序，不改变位置，打乱后更新家族顺序缓存。
        
        参数
        -----
        recurse : bool, optional
            是否递归打乱子对象的子对象，默认False（仅打乱当前层子对象）
        """
        if recurse:
            # 递归打乱所有子对象的子对象
            for submob in self.submobjects:
                submob.shuffle(recurse=True)
        # 随机打乱当前层子对象顺序
        random.shuffle(self.submobjects)
        # 仅顺序变更，标记家族变更时无需刷新更新器和包围盒
        self.note_changed_family(only_changed_order=True)
        return self

    def reverse_submobjects(self) -> Self:
        """
        反转子对象顺序：将当前层子对象列表反转（如[1,2,3]变为[3,2,1]），
        仅改变顺序不改变位置，更新家族顺序缓存。
        """
        # 反转子对象列表
        self.submobjects.reverse()
        # 仅顺序变更，标记家族变更时无需刷新更新器和包围盒
        self.note_changed_family(only_changed_order=True)
        return self
    
# 复制与序列化相关方法

    @staticmethod
    def stash_mobject_pointers(func: Callable[..., T]) -> Callable[..., T]:
        """
        装饰器：在执行目标函数前暂存并清空与其他Mobject相关的引用（如父对象、目标状态），
        避免序列化或深拷贝时出现循环引用问题，函数执行后恢复暂存的引用。
        
        参数
        -----
        func : Callable[..., T]
            需要处理Mobject引用的函数（如序列化、深拷贝）
        
        返回
        -----
        Callable[..., T]
            包装后的函数，自动处理引用暂存与恢复
        """
        @wraps(func)  # 保留原函数元信息
        def wrapper(self, *args, **kwargs):
            # 无需复制的属性列表（包含其他Mobject引用，可能导致循环依赖）
            uncopied_attrs = ["parents", "target", "saved_state"]
            stash = dict()  # 用于暂存属性值的字典
            
            # 暂存并清空指定属性
            for attr in uncopied_attrs:
                if hasattr(self, attr):
                    value = getattr(self, attr)
                    stash[attr] = value  # 保存原始值
                    # 根据属性类型设置临时空值（列表用空列表，其他用None）
                    null_value = [] if isinstance(value, list) else None
                    setattr(self, attr, null_value)
            
            # 执行目标函数（如序列化、深拷贝）
            result = func(self, *args, **kwargs)
            
            # 恢复暂存的属性值
            self.__dict__.update(stash)
            return result
        return wrapper

    @stash_mobject_pointers
    def serialize(self) -> bytes:
        """
        序列化当前图形对象：使用pickle将对象转换为字节流，序列化前自动处理
        可能导致循环引用的Mobject引用，确保序列化可正常完成。
        
        返回
        -----
        bytes
            序列化后的字节流数据
        """
        return pickle.dumps(self)

    def deserialize(self, data: bytes) -> Self:
        """
        反序列化字节流为当前图形对象：使用pickle加载字节流数据，
        并将加载结果应用到当前对象（覆盖现有状态）。
        
        参数
        -----
        data : bytes
            序列化后的字节流数据（通常由serialize方法生成）
        
        返回
        -----
        Self
            当前图形对象自身（状态已更新为反序列化结果）
        """
        self.become(pickle.loads(data))  # 用加载的对象替换当前对象状态
        return self

    @stash_mobject_pointers
    def deepcopy(self) -> Self:
        """
        创建当前图形对象的深拷贝：使用copy.deepcopy生成完全独立的副本，
        所有子对象和数据均被深度复制，无共享引用。
        """
        return copy.deepcopy(self)

    def copy(self, deep: bool = False) -> Self:
        """
        创建当前图形对象的副本：支持浅拷贝（默认）和深拷贝，浅拷贝时
        基础属性和子对象引用被复制，但子对象本身不深度复制；深拷贝等同于deepcopy。
        
        参数
        -----
        deep : bool, optional
            若为True则执行深拷贝，否则执行浅拷贝，默认False
        
        返回
        -----
        Self
            拷贝得到的新图形对象
        """
        if deep:
            return self.deepcopy()  # 深拷贝直接调用deepcopy方法

        # 浅拷贝：使用copy.copy复制基础属性
        result = copy.copy(self)

        # 重置与其他对象的关联（避免副本与原对象共享父/子关系）
        result.parents = []
        result.target = None
        result.saved_state = None

        # 复制统一变量：numpy数组需显式拷贝，避免共享内存
        result.uniforms = {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in self.uniforms.items()
        }

        # 复制子对象（递归浅拷贝），直接修改家族列表（跳过add方法的额外检查）
        result.submobjects = [sm.copy() for sm in self.submobjects]
        # 建立副本与子对象的父-子关联
        for sm in result.submobjects:
            sm.parents = [result]
        # 重建副本的家族列表（自身 + 所有子对象的家族）
        result.family = [result, *it.chain(*(sm.get_family() for sm in result.submobjects))]

        # 复制更新器列表（直接引用，不深度复制）
        result.updaters = list(self.updaters)
        # 标记数据已变更，确保渲染时更新
        result._data_has_changed = True
        # 重置着色器包装器（副本需重新创建着色器实例）
        result.shader_wrapper = None

        # 处理其他属性：若属性是原家族中的Mobject，替换为副本家族中对应的对象
        family = self.get_family()
        for attr, value in self.__dict__.items():
            if isinstance(value, Mobject) and value is not self:
                if value in family:
                    # 找到副本家族中对应位置的对象并替换
                    setattr(result, attr, result.family[family.index(value)])
            elif isinstance(value, np.ndarray):
                # numpy数组显式拷贝，避免共享内存
                setattr(result, attr, value.copy())
        return result

    def generate_target(self, use_deepcopy: bool = False) -> Self:
        """
        生成当前图形对象的目标状态副本：创建一个副本作为动画过渡的目标状态，
        存储在target属性中，便于后续动画使用（如MoveToTarget）。
        
        参数
        -----
        use_deepcopy : bool, optional
            若为True则使用深拷贝生成目标，否则使用浅拷贝，默认False
        """
        # 创建当前对象的副本作为目标
        self.target = self.copy(deep=use_deepcopy)
        # 复制保存的状态到目标（确保状态一致性）
        self.target.saved_state = self.saved_state
        return self.target

    def save_state(self, use_deepcopy: bool = False) -> Self:
        """
        保存当前图形对象的状态：创建当前对象的副本（浅拷贝或深拷贝），
        存储到`saved_state`属性中，同时保留目标状态（target），便于后续恢复。
        
        参数
        -----
        use_deepcopy : bool, optional
            若为True则深拷贝保存状态（副本与原对象完全独立），否则浅拷贝，默认False
        """
        # 拷贝当前对象作为保存的状态
        self.saved_state = self.copy(deep=use_deepcopy)
        # 保留当前对象的目标状态（确保恢复时目标状态也一致）
        self.saved_state.target = self.target
        return self

    def restore(self) -> Self:
        """
        恢复到之前保存的状态：将`saved_state`中存储的状态应用到当前对象，
        若未保存过状态则抛出异常。
        
        异常
        -----
        Exception
            若未调用过`save_state`或`saved_state`为None，抛出“无保存状态可恢复”的异常
        """
        # 检查是否有保存的状态
        if not hasattr(self, "saved_state") or self.saved_state is None:
            raise Exception("Trying to restore without having saved")
        # 调用become方法，将当前对象变为保存状态的副本
        self.become(self.saved_state)
        return self

    def become(self, mobject: Mobject, match_updaters=False) -> Self:
        """
        使当前对象与目标对象完全一致：复制目标对象的所有数据（顶点、颜色）、子对象、
        统一变量及渲染相关属性，可选择是否同步更新器，实现对象状态的完全替换。
        
        说明
        -----
        Edit all data and submobjects to be identical to another mobject
        （编辑所有数据和子对象，使其与另一个mobject完全相同）
        
        参数
        -----
        mobject : Mobject
            目标对象（当前对象将复制其所有状态）
        match_updaters : bool, optional
            若为True则同步目标对象的更新器列表，否则保持当前更新器，默认False
        """
        # 对齐两个对象的家族结构（确保子对象数量和层级匹配）
        self.align_family(mobject)
        # 获取当前对象和目标对象的家族成员列表
        family1 = self.get_family()
        family2 = mobject.get_family()

        # 遍历两个家族的成员，逐个复制状态
        for sm1, sm2 in zip(family1, family2):
            sm1.set_data(sm2.data)  # 复制顶点、颜色等核心数据
            sm1.set_uniforms(sm2.uniforms)  # 复制着色器统一变量
            sm1.bounding_box[:] = sm2.bounding_box  # 复制包围盒
            # 复制渲染相关属性
            sm1.shader_folder = sm2.shader_folder
            sm1.texture_paths = sm2.texture_paths
            sm1.depth_test = sm2.depth_test
            sm1.render_primitive = sm2.render_primitive
            sm1._needs_new_bounding_box = sm2._needs_new_bounding_box

        # 处理目标对象中命名的Mobject属性（如self.rect = Rectangle()），替换为当前家族对应的成员
        for attr, value in list(mobject.__dict__.items()):
            if isinstance(value, Mobject) and value in family2:
                # 找到当前家族中与目标属性对应的成员，赋值给当前对象的同名属性
                setattr(self, attr, family1[family2.index(value)])

        # 若需要同步更新器，复制目标对象的更新器列表
        if match_updaters:
            self.match_updaters(mobject)
        return self

    def looks_identical(self, mobject: Mobject) -> bool:
        """
        判断当前对象与目标对象是否外观完全一致：检查两者家族中含顶点的成员数量、
        顶点数据类型、各字段数值（顶点坐标、颜色）及统一变量是否完全匹配（允许浮点误差）。
        
        参数
        -----
        mobject : Mobject
            待比较的目标对象
        
        返回
        -----
        bool
            True表示两者外观完全一致，False表示存在差异
        """
        # 获取两个对象家族中含顶点数据的成员
        fam1 = self.family_members_with_points()
        fam2 = mobject.family_members_with_points()

        # 1. 检查含顶点的家族成员数量是否一致
        if len(fam1) != len(fam2):
            return False

        # 2. 逐个比较家族成员的细节
        for m1, m2 in zip(fam1, fam2):
            # 检查顶点数量是否一致
            if m1.get_num_points() != m2.get_num_points():
                return False
            # 检查数据类型是否一致
            if not m1.data.dtype == m2.data.dtype:
                return False
            # 检查每个数据字段（如point、rgba）的数值是否接近（允许浮点误差）
            for key in m1.data.dtype.names:
                if not np.isclose(m1.data[key], m2.data[key]).all():
                    return False
            # 检查统一变量的键是否完全一致（无额外或缺失的键）
            if set(m1.uniforms).difference(m2.uniforms):
                return False
            # 检查每个统一变量的数值是否接近（允许浮点误差）
            for key in m1.uniforms:
                value1 = m1.uniforms[key]
                value2 = m2.uniforms[key]
                # 若为numpy数组，先检查尺寸是否一致
                if isinstance(value1, np.ndarray) and isinstance(value2, np.ndarray) and not value1.size == value2.size:
                    return False
                # 检查数值是否接近
                if not np.isclose(value1, value2).all():
                    return False

        # 所有检查通过，外观一致
        return True

    def has_same_shape_as(self, mobject: Mobject) -> bool:
        """
        判断当前对象与目标对象是否形状相同：通过将两者的所有顶点归一化（居中、高度缩放为1），
        比较归一化后的顶点坐标是否接近（允许基于宽度的微小误差），忽略位置、大小差异。
        
        参数
        -----
        mobject : Mobject
            待比较形状的目标对象
        
        返回
        -----
        bool
            True表示形状相同，False表示形状不同
        """
        # 归一化顶点：1. 减去自身中心（居中）；2. 除以自身高度（高度缩放为1）
        points1, points2 = (
            (m.get_all_points() - m.get_center()) / m.get_height()
            for m in (self, mobject)
        )
        # 1. 先检查归一化后的顶点数量是否一致
        if len(points1) != len(points2):
            return False
        # 2. 检查所有顶点坐标是否接近（误差容忍度为当前对象宽度的1%）
        return bool(np.isclose(points1, points2, atol=self.get_width() * 1e-2).all())

    # Creating new Mobjects from this one（基于当前对象创建新Mobject的方法）

    def replicate(self, n: int) -> Self:
        """
        复制当前对象指定次数：创建n个当前对象的浅拷贝，将其封装为当前对象的组类（如Group），
        生成包含多个副本的新组。
        
        参数
        -----
        n : int
            复制次数（需为非负整数）
        
        返回
        -----
        Self
            包含n个当前对象副本的组实例
        """
        group_class = self.get_group_class()  # 获取当前对象对应的组类
        # 生成n个副本，传入组类创建新组
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
        基于当前对象创建网格布局的副本组：生成n_rows×n_cols个当前对象的副本，
        按网格排列，支持按行/按列分组，可指定整体高度/宽度。
        
        说明
        -----
        Returns a new mobject containing multiple copies of this one arranged in a grid
        （返回包含多个当前对象副本的新Mobject，副本按网格排列）
        
        参数
        -----
        n_rows : int
            网格行数
        n_cols : int
            网格列数
        height : float | None, optional
            网格整体高度，设置后缩放网格以匹配该高度，默认None（不强制缩放）
        width : float | None, optional
            网格整体宽度，设置后缩放网格以匹配该宽度，默认None（不强制缩放）
        group_by_rows : bool, optional
            若为True，将每行副本封装为独立子组，最终返回组的组，默认False
        group_by_cols : bool, optional
            若为True，将每列副本封装为独立子组，最终返回组的组，默认False（与group_by_rows互斥）
        **kwargs
            传递给arrange_in_grid的参数（如buff、aligned_edge等）
        """
        total = n_rows * n_cols  # 副本总数
        grid = self.replicate(total)  # 生成指定数量的副本组

        # 若按列分组，设置网格按列优先填充
        if group_by_cols:
            kwargs["fill_rows_first"] = False
        # 将副本组按网格排列
        grid.arrange_in_grid(n_rows, n_cols, **kwargs)

        # 若指定整体高度/宽度，缩放网格以匹配
        if height is not None:
            grid.set_height(height)
        if width is not None:
            grid.set_width(width)  # 原代码中误写为set_height，此处修正为set_width

        group_class = self.get_group_class()
        # 按行分组：将每n_cols个副本分为一组（一行）
        if group_by_rows:
            return group_class(*(grid[n:n + n_cols] for n in range(0, total, n_cols)))
        # 按列分组：将每n_rows个副本分为一组（一列）
        elif group_by_cols:
            return group_class(*(grid[n:n + n_rows] for n in range(0, total, n_rows)))
        # 不分组：直接返回网格排列的副本组
        else:
            return grid

    # Updating

    def init_updaters(self):
        """
        初始化更新器相关属性：创建空的更新器列表，设置更新器状态标记和暂停标记，
        为后续添加、执行更新器做准备。
        """
        self.updaters: list[Updater] = list()  # 存储所有更新器的列表（支持时间/非时间更新器）
        self._has_updaters_in_family: Optional[bool] = False  # 家族是否含更新器的标记，默认False
        self.updating_suspended: bool = False  # 更新暂停标记，默认False（允许更新）

    def update(self, dt: float = 0, recurse: bool = True) -> Self:
        """
        执行更新逻辑：递归执行自身及子对象的所有更新器，根据更新器是否接收`dt`参数传递时间差，
        仅在允许更新且存在更新器时执行。
        
        参数
        -----
        dt : float, optional
            时间差（自上次更新到当前的时间间隔），默认0（适用于非时间依赖的更新器）
        recurse : bool, optional
            是否递归执行子对象的更新器，默认True（递归执行）
        """
        # 若无可执行的更新器或更新已暂停，直接返回
        if not self.has_updaters() or self.updating_suspended:
            return self
        # 递归执行所有子对象的更新
        if recurse:
            for submob in self.submobjects:
                submob.update(dt, recurse)
        # 遍历自身所有更新器，根据参数签名决定是否传递dt
        for updater in self.updaters:
            # 检查更新器函数是否包含"dt"参数（通过函数参数名判断）
            if "dt" in updater.__code__.co_varnames:
                updater(self, dt=dt)  # 传递时间差dt给更新器
            else:
                updater(self)  # 不传递dt，仅传入当前对象
        return self

    def get_updaters(self) -> list[Updater]:
        """
        获取当前对象的所有更新器列表：直接返回存储的更新器列表，用于查看或后续操作（如移除）。
        """
        return self.updaters

    def add_updater(self, update_func: Updater, call: bool = True) -> Self:
        """
        向当前对象添加更新器：将更新器函数加入列表，可选择立即执行一次更新，
        并刷新家族的更新器状态标记。
        
        参数
        -----
        update_func : Updater
            待添加的更新器函数（需符合TimeBasedUpdater或NonTimeUpdater的签名）
        call : bool, optional
            添加后是否立即调用一次更新（dt=0），默认True（立即执行）
        """
        self.updaters.append(update_func)  # 将更新器加入列表
        # 若需要立即执行，调用update方法（dt=0）
        if call:
            self.update(dt=0)
        # 刷新家族的更新器状态标记（告知父对象当前对象含更新器）
        self.refresh_has_updater_status()
        self.update()  # 再次执行更新，确保状态同步
        return self

    def insert_updater(self, update_func: Updater, index=0):
        """
        在指定索引插入更新器：将更新器函数插入到更新器列表的指定位置（默认插入到首位），
        刷新家族的更新器状态标记，用于调整更新器执行顺序。
        
        参数
        -----
        update_func : Updater
            待插入的更新器函数
        index : int, optional
            插入位置的索引，默认0（插入到列表首位，优先执行）
        """
        self.updaters.insert(index, update_func)  # 在指定索引插入更新器
        self.refresh_has_updater_status()  # 刷新家族的更新器状态标记
        return self

    def remove_updater(self, update_func: Updater) -> Self:
        """
        从当前对象移除更新器：删除更新器列表中所有匹配的更新器函数（支持重复移除），
        刷新家族的更新器状态标记，停止该更新器的执行。
        
        参数
        -----
        update_func : Updater
            待移除的更新器函数
        """
        # 循环移除所有匹配的更新器（处理重复添加的情况）
        while update_func in self.updaters:
            self.updaters.remove(update_func)
        self.refresh_has_updater_status()  # 刷新家族的更新器状态标记
        return self

    def clear_updaters(self, recurse: bool = True) -> Self:
        """
        清空更新器：递归或非递归地删除自身及家族成员的所有更新器，
        重置更新器状态标记，并同步更新祖先对象的更新器状态。
        
        参数
        -----
        recurse : bool, optional
            是否递归清空子对象的更新器，默认True（清空所有家族成员）
        """
        # 遍历目标家族成员（recurse=True时含子对象），清空更新器并重置状态
        for mob in self.get_family(recurse):
            mob.updaters = []  # 清空更新器列表
            mob._has_updaters_in_family = False  # 重置家族更新器状态标记
        # 遍历所有祖先对象，重置其家族更新器状态标记
        for parent in self.get_ancestors():
            parent._has_updaters_in_family = False
        return self

    def match_updaters(self, mobject: Mobject) -> Self:
        """
        同步目标对象的更新器：将当前对象的更新器列表替换为目标对象的更新器列表，
        并刷新家族的更新器状态标记，使当前对象与目标对象的更新逻辑一致。
        
        参数
        -----
        mobject : Mobject
            目标对象（提供待同步的更新器列表）
        """
        # 复制目标对象的更新器列表（浅拷贝，共享更新器函数引用）
        self.updaters = list(mobject.updaters)
        # 刷新家族更新器状态标记，确保父对象感知更新器变化
        self.refresh_has_updater_status()
        return self

    def suspend_updating(self, recurse: bool = True) -> Self:
        """
        暂停更新：设置更新暂停标记为True，禁止当前对象执行更新器，
        可选择递归暂停所有子对象的更新，暂停后`update`方法将不生效。
        
        参数
        -----
        recurse : bool, optional
            是否递归暂停子对象的更新，默认True（暂停所有家族成员）
        """
        self.updating_suspended = True  # 暂停当前对象更新
        # 若递归，暂停所有子对象的更新
        if recurse:
            for submob in self.submobjects:
                submob.suspend_updating(recurse)
        return self

    def resume_updating(self, recurse: bool = True, call_updater: bool = True) -> Self:
        """
        恢复更新：设置更新暂停标记为False，允许当前对象执行更新器，
        可选择递归恢复子对象更新、通知祖先对象恢复，并可立即执行一次更新。
        
        参数
        -----
        recurse : bool, optional
            是否递归恢复子对象的更新，默认True（恢复所有家族成员）
        call_updater : bool, optional
            恢复后是否立即执行一次更新（dt=0），默认True（立即同步状态）
        """
        self.updating_suspended = False  # 恢复当前对象更新
        # 若递归，恢复所有子对象的更新
        if recurse:
            for submob in self.submobjects:
                submob.resume_updating(recurse)
        # 通知所有祖先对象恢复更新（不递归、不立即执行，避免重复触发）
        for parent in self.parents:
            parent.resume_updating(recurse=False, call_updater=False)
        # 若需要，恢复后立即执行一次更新，同步当前状态
        if call_updater:
            self.update(dt=0, recurse=recurse)
        return self

    def has_updaters(self) -> bool:
        """
        判断当前对象或其家族是否含更新器：若状态标记未缓存，先检查自身更新器列表
        及所有子对象的更新器状态，缓存结果后返回；若已缓存，直接返回缓存值。
        
        返回
        -----
        bool
            True表示当前对象或其家族含更新器，False表示无任何更新器
        """
        # 若状态标记未缓存（为None），重新计算并缓存
        if self._has_updaters_in_family is None:
            # 自身有更新器 或 任一子对象有更新器，即判定为含更新器
            self._has_updaters_in_family = bool(self.updaters) or any(
                sm.has_updaters() for sm in self.submobjects
            )
        return self._has_updaters_in_family

    def refresh_has_updater_status(self) -> Self:
        """
        刷新更新器状态标记：重置当前对象的家族更新器状态标记（设为None），
        并向上通知所有父对象同步刷新，确保更新器状态的准确性。
        """
        self._has_updaters_in_family = None  # 重置当前对象的状态标记（触发后续重新计算）
        # 向上通知所有父对象刷新状态标记，确保家族层级的状态同步
        for parent in self.parents:
            parent.refresh_has_updater_status()
        return self
    
    # 标记对象是否为静态（供相机判断是否需要实时渲染）

    def is_changing(self) -> bool:
        """
        判断对象是否处于动态变化中：若对象正在执行动画（_is_animating为True）或包含更新器，
        则判定为动态变化，相机需实时渲染；否则为静态，可优化渲染性能。
        
        返回
        -----
        bool
            True表示对象动态变化中，False表示对象静态无变化
        """
        return self._is_animating or self.has_updaters()

    def set_animating_status(self, is_animating: bool, recurse: bool = True) -> Self:
        """
        设置对象的动画状态标记：递归或非递归地更新自身、家族成员及祖先对象的_is_animating标记，
        用于告知系统对象是否正在执行动画。
        
        参数
        -----
        is_animating : bool
            动画状态，True表示正在执行动画，False表示动画结束
        recurse : bool, optional
            是否递归更新家族成员（自身及子对象）的动画状态，默认True
        """
        # 遍历家族成员（含/不含子对象）和所有祖先对象，统一设置动画状态
        for mob in (*self.get_family(recurse), *self.get_ancestors()):
            mob._is_animating = is_animating
        return self

    # 变换操作（平移、缩放等）

    def shift(self, vector: Vect3) -> Self:
        """
        平移对象：沿指定3D向量移动对象（如RIGHT表示向右、UP表示向上），
        直接修改顶点坐标和包围盒，所有家族成员同步平移。
        
        参数
        -----
        vector : Vect3
            平移向量（如np.array([1,0,0])表示沿x轴正方向平移1单位）
        """
        self.apply_points_function(
            lambda points: points + vector,  # 平移逻辑：顶点坐标 + 平移向量
            about_edge=None,  # 无需围绕特定边缘，直接平移
            works_on_bounding_box=True,  # 同时更新包围盒（避免后续重复计算）
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
        缩放对象：按指定缩放因子放大/缩小对象，支持围绕特定点或边缘缩放，
        并限制最小缩放因子避免对象消失，缩放后同步处理家族成员的副作用（如纹理、字体）。
        
        说明
        -----
        Default behavior is to scale about the center of the mobject.
        The argument about_edge can be a vector, indicating which side of
        the mobject to scale about, e.g., mob.scale(about_edge = RIGHT)
        scales about mob.get_right().
        Otherwise, if about_point is given a value, scaling is done with
        respect to that point.
        （默认围绕对象中心缩放；about_edge可指定围绕边缘缩放，如RIGHT表示围绕右边缘；
        若指定about_point，则围绕该点缩放）
        
        参数
        -----
        scale_factor : float | npt.ArrayLike
            缩放因子，单个数值表示等比例缩放（如2表示放大2倍），数组表示各轴独立缩放（如[2,1,1]表示x轴放大2倍）
        min_scale_factor : float, optional
            最小缩放因子，避免缩放后对象过小或消失，默认1e-8
        about_point : Vect3 | None, optional
            缩放中心点，默认None（优先使用about_edge）
        about_edge : Vect3, optional
            缩放围绕的边缘（如LEFT、TOP），默认ORIGIN（围绕中心）
        """
        # 处理缩放因子：确保不小于最小缩放因子，避免对象消失
        if isinstance(scale_factor, numbers.Number):
            scale_factor = max(scale_factor, min_scale_factor)
        else:
            scale_factor = np.array(scale_factor).clip(min=min_scale_factor)
        
        # 调用通用点处理方法执行缩放逻辑
        self.apply_points_function(
            lambda points: scale_factor * points,  # 缩放逻辑：顶点坐标 × 缩放因子
            about_point=about_point,  # 围绕指定点缩放
            about_edge=about_edge,    # 围绕指定边缘缩放（about_point为None时生效）
            works_on_bounding_box=True,  # 同时更新包围盒
        )
        
        # 处理缩放带来的副作用（如纹理缩放、字体大小调整等，由子类实现_handle_scale_side_effects）
        for mob in self.get_family():
            mob._handle_scale_side_effects(scale_factor)
        return self

    def _handle_scale_side_effects(self, scale_factor):
        """
        处理缩放带来的副作用：空实现，供子类（如DecimalNumber）重写，
        用于在缩放时调整额外属性（如字体大小、纹理比例等）。
        
        说明
        -----
        In case subclasses, such as DecimalNumber, need to make
        any other changes when the size gets altered
        （供子类如DecimalNumber在尺寸改变时做额外调整）
        """
        pass

    def stretch(self, factor: float, dim: int, **kwargs) -> Self:
        """
        沿指定维度拉伸对象：仅在指定维度（如x轴0、y轴1）上按比例拉伸，
        其他维度保持不变，可指定拉伸围绕的点或边缘。
        
        参数
        -----
        factor : float
            拉伸因子（>1表示拉长，0<factor<1表示压缩）
        dim : int
            拉伸维度，0=x轴、1=y轴、2=z轴
        **kwargs
            传递给apply_points_function的参数（如about_point、about_edge）
        """
        # 定义拉伸函数：仅指定维度的坐标乘以拉伸因子
        def func(points):
            points[:, dim] *= factor
            return points
        # 应用拉伸函数到所有点和包围盒
        self.apply_points_function(func, works_on_bounding_box=True,** kwargs)
        return self

    def rotate_about_origin(self, angle: float, axis: Vect3 = OUT) -> Self:
        """
        围绕原点旋转对象：调用rotate方法，指定旋转中心为原点（ORIGIN），
        简化围绕世界坐标系原点旋转的操作。
        
        参数
        -----
        angle : float
            旋转角度（弧度制，如TAU/4表示90度）
        axis : Vect3, optional
            旋转轴，默认OUT（垂直屏幕向外，即z轴正方向）
        """
        return self.rotate(angle, axis, about_point=ORIGIN)

    def rotate(
        self,
        angle: float,
        axis: Vect3 = OUT,
        about_point: Vect3 | None = None,
        **kwargs
    ) -> Self:
        """
        旋转对象：围绕指定轴和中心点旋转对象，通过旋转矩阵实现3D空间中的旋转，
        支持任意轴和中心点，适用于各种旋转动画。
        
        参数
        -----
        angle : float
            旋转角度（弧度制）
        axis : Vect3, optional
            旋转轴向量（如OUT为z轴、RIGHT为x轴），默认OUT
        about_point : Vect3 | None, optional
            旋转中心点，默认None（围绕对象自身中心）
        **kwargs
            传递给apply_points_function的其他参数
        """
        # 计算旋转矩阵的转置（用于后续点乘实现旋转）
        rot_matrix_T = rotation_matrix_transpose(angle, axis)
        # 应用旋转函数：点坐标 × 旋转矩阵转置 = 旋转后的坐标
        self.apply_points_function(
            lambda points: np.dot(points, rot_matrix_T),
            about_point,** kwargs
        )
        return self

    def flip(self, axis: Vect3 = UP, **kwargs) -> Self:
        """
        翻转对象：通过旋转180度（TAU/2弧度）实现对象翻转，等价于绕指定轴旋转半圈，
        可指定翻转围绕的中心点。
        
        参数
        -----
        axis : Vect3, optional
            翻转轴（如UP为y轴，水平翻转；RIGHT为x轴，垂直翻转），默认UP
        **kwargs
            传递给rotate方法的参数（如about_point）
        """
        return self.rotate(TAU / 2, axis,** kwargs)  # TAU/2 = π，即180度旋转

    def apply_function(self, function: Callable[[np.ndarray], np.ndarray], **kwargs) -> Self:
        """
        对对象的每个顶点应用自定义函数：将函数逐个应用到所有顶点坐标，
        支持指定变换围绕的中心点（默认围绕原点），适用于自定义几何变换。
        
        参数
        -----
        function : Callable[[np.ndarray], np.ndarray]
            顶点变换函数，输入单个顶点坐标（3D数组），输出变换后的坐标
        **kwargs
            传递给apply_points_function的参数（如about_point），默认about_point=ORIGIN
        """
        # 默认为围绕原点变换，而非对象中心
        if len(kwargs) == 0:
            kwargs["about_point"] = ORIGIN
        # 应用函数到所有顶点：逐个处理每个点
        self.apply_points_function(
            lambda points: np.array([function(p) for p in points]),** kwargs
        )
        return self

    def apply_function_to_position(self, function: Callable[[np.ndarray], np.ndarray]) -> Self:
        """
        对对象位置应用自定义函数：计算对象中心坐标经函数变换后的新位置，
        将对象整体移动到新位置（不改变对象自身形状和朝向）。
        
        参数
        -----
        function : Callable[[np.ndarray], np.ndarray]
            位置变换函数，输入对象中心坐标（3D数组），输出新位置坐标
        """
        # 计算新位置：函数作用于当前中心坐标
        self.move_to(function(self.get_center()))
        return self

    def apply_function_to_submobject_positions(
        self,
        function: Callable[[np.ndarray], np.ndarray]
    ) -> Self:
        """
        对所有子对象位置应用自定义函数：递归对每个子对象调用apply_function_to_position，
        仅改变子对象的位置，不影响父对象自身位置，适用于批量调整子对象布局。
        
        参数
        -----
        function : Callable[[np.ndarray], np.ndarray]
            位置变换函数，输入子对象中心坐标，输出新位置坐标
        """
        for submob in self.submobjects:
            submob.apply_function_to_position(function)
        return self

    def apply_matrix(self, matrix: npt.ArrayLike, **kwargs) -> Self:
        """
        对对象应用矩阵变换：将指定矩阵扩展为与对象维度匹配的单位矩阵，
        通过矩阵乘法实现线性变换（如旋转、缩放、剪切），默认围绕原点变换。
        
        参数
        -----
        matrix : npt.ArrayLike
            变换矩阵（如2x2矩阵用于2D变换、3x3矩阵用于3D变换），维度可小于对象维度（将自动补全为单位矩阵）
        **kwargs
            传递给apply_points_function的参数（如about_point指定变换中心点）
        """
        # 若未指定变换中心点（about_point或about_edge），默认围绕原点变换
        if ("about_point" not in kwargs) and ("about_edge" not in kwargs):
            kwargs["about_point"] = ORIGIN
        
        # 创建与对象维度匹配的单位矩阵（基础矩阵，确保变换维度正确）
        full_matrix = np.identity(self.dim)
        # 将输入矩阵转换为numpy数组，并填充到单位矩阵的左上角（补全维度）
        matrix = np.array(matrix)
        full_matrix[:matrix.shape[0], :matrix.shape[1]] = matrix
        
        # 应用矩阵变换：顶点坐标 × 矩阵转置（确保变换方向正确）
        self.apply_points_function(
            lambda points: np.dot(points, full_matrix.T),
            **kwargs
        )
        return self

    def apply_complex_function(self, function: Callable[[complex], complex], **kwargs) -> Self:
        """
        对对象应用复变函数变换：将对象的2D顶点坐标（x,y）视为复数（x+yi），
        执行复变函数运算后转换回3D坐标（z轴保持不变），实现复杂2D变形。
        
        参数
        -----
        function : Callable[[complex], complex]
            复变函数（如lambda z: z**2实现平方变换、lambda z: np.exp(z)实现指数变换）
        **kwargs
            传递给apply_function的参数（如about_point指定变换中心点）
        """
        # 定义3D坐标变换函数：提取x,y组成复数，执行复变函数后还原为3D坐标
        def R3_func(point):
            x, y, z = point
            xy_complex = function(complex(x, y))  # 将(x,y)转为复数并执行函数
            return [
                xy_complex.real,  # 复数实部作为新x坐标
                xy_complex.imag,  # 复数虚部作为新y坐标
                z                 # z坐标保持不变
            ]
        # 调用apply_function执行3D坐标变换
        return self.apply_function(R3_func, **kwargs)

    def wag(self, direction: Vect3 = RIGHT, axis: Vect3 = DOWN, wag_factor: float = 1.0) -> Self:
        """
        使对象产生“摆动”变形：沿指定轴计算顶点的权重系数，按系数沿摆动方向偏移顶点，
        实现类似“弯曲”“摆动”的非线性变形（如旗帜飘动、叶子摆动）。
        
        参数
        -----
        direction : Vect3, optional
            摆动偏移方向（如RIGHT表示水平摆动、UP表示垂直摆动），默认RIGHT
        axis : Vect3, optional
            计算权重的参考轴（如DOWN表示沿y轴向下渐变权重），默认DOWN
        wag_factor : float, optional
            摆动幅度系数（>1增强幅度梯度，<1减弱幅度梯度），默认1.0
        """
        # 遍历家族中所有含顶点数据的成员，逐个执行摆动变形
        for mob in self.family_members_with_points():
            # 1. 计算每个顶点沿参考轴的投影值（作为权重基础）
            alphas = np.dot(mob.get_points(), np.transpose(axis))
            # 2. 归一化权重：将投影值映射到[0,1]区间
            alphas -= min(alphas)  # 平移到最小值为0
            alphas /= max(alphas)  # 缩放最大值为1（避免除零，因mob有顶点则max≥min）
            # 3. 调整权重梯度：通过幂运算改变摆动幅度的分布
            alphas = alphas**wag_factor
            # 4. 计算顶点偏移量：权重 × 摆动方向向量（确保每个顶点偏移量不同）
            offsets = np.dot(
                alphas.reshape((len(alphas), 1)),  # 权重数组（N,1）
                np.array(direction).reshape((1, mob.dim))  # 摆动方向（1,3）
            )
            # 5. 应用偏移到顶点，实现摆动变形
            mob.set_points(mob.get_points() + offsets)
        return self

    # 定位相关方法（控制对象在画面中的位置与对齐）

    def center(self) -> Self:
        """
        将对象居中：计算对象的中心坐标，沿相反方向平移对象，使其中心与世界坐标系原点（ORIGIN）重合，
        实现画面居中效果。
        """
        # 平移向量 = -中心坐标（将中心移至原点）
        self.shift(-self.get_center())
        return self

    def align_on_border(
        self,
        direction: Vect3,
        buff: float = DEFAULT_MOBJECT_TO_EDGE_BUFF
    ) -> Self:
        """
        将对象对齐到画面边界：根据指定方向（如LEFT、TOP+RIGHT）计算画面边界目标点，
        平移对象使其边界点与目标点对齐，并保留指定缓冲距离。
        
        说明
        -----
        Direction just needs to be a vector pointing towards side or
        corner in the 2d plane.
        （方向只需是指向2D平面中边缘或角落的向量，如LEFT表示左边缘，LEFT+UP表示左上角落）
        
        参数
        -----
        direction : Vect3
            对齐方向向量（如LEFT、RIGHT、UP+DOWN不合法，需指向单一边缘/角落）
        buff : float, optional
            对象与画面边界的缓冲距离，默认使用全局默认边缘缓冲（DEFAULT_MOBJECT_TO_EDGE_BUFF）
        """
        # 计算画面边界目标点：取方向向量的正负符号 × 画面半宽/半高（仅2D平面，z轴为0）
        target_point = np.sign(direction) * (FRAME_X_RADIUS, FRAME_Y_RADIUS, 0)
        # 获取对象需要对齐的边界点（与方向对应的包围盒点）
        point_to_align = self.get_bounding_box_point(direction)
        # 计算平移向量：目标点 - 对齐点 - 缓冲距离×方向（确保缓冲方向正确）
        shift_val = target_point - point_to_align - buff * np.array(direction)
        # 修正平移向量：仅在方向向量非零的维度生效（避免无关维度偏移）
        shift_val = shift_val * abs(np.sign(direction))
        # 执行平移，完成边界对齐
        self.shift(shift_val)
        return self

    def to_corner(
        self,
        corner: Vect3 = LEFT + DOWN,
        buff: float = DEFAULT_MOBJECT_TO_EDGE_BUFF
    ) -> Self:
        """
        将对象对齐到画面角落：调用align_on_border方法，默认对齐到左下角落（LEFT+DOWN），
        简化角落对齐的调用流程。
        
        参数
        -----
        corner : Vect3, optional
            目标角落方向（如LEFT+UP表示左上、RIGHT+DOWN表示右下），默认LEFT+DOWN（左下）
        buff : float, optional
            对象与角落的缓冲距离，默认使用全局默认边缘缓冲
        """
        return self.align_on_border(corner, buff)

    def to_edge(
        self,
        edge: Vect3 = LEFT,
        buff: float = DEFAULT_MOBJECT_TO_EDGE_BUFF
    ) -> Self:
        """
        将对象对齐到画面边缘：调用align_on_border方法，默认对齐到左边缘（LEFT），
        简化边缘对齐的调用流程。
        
        参数
        -----
        edge : Vect3, optional
            目标边缘方向（如LEFT、RIGHT、UP、DOWN），默认LEFT（左边缘）
        buff : float, optional
            对象与边缘的缓冲距离，默认使用全局默认边缘缓冲
        """
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
        """
        将对象放置在目标（对象或点）的指定方向旁：支持子对象级对齐、指定轴向生效，
        自动计算平移距离并保留缓冲，是对象间相对定位的核心方法。
        
        参数
        -----
        mobject_or_point : Mobject | Vect3
            目标参考物，可为另一个Mobject实例或3D坐标点
        direction : Vect3, optional
            相对于目标的放置方向（如RIGHT表示在目标右侧、UP表示在目标上方），默认RIGHT
        buff : float, optional
            对象与目标的缓冲距离，默认使用全局默认对象间缓冲（DEFAULT_MOBJECT_TO_MOBJECT_BUFF）
        aligned_edge : Vect3, optional
            对齐边缘（如LEFT表示对象左边缘与目标右边缘对齐），默认ORIGIN（中心对齐）
        submobject_to_align : Mobject | None, optional
            用于对齐的子对象（当前对象的子对象），默认None（使用当前对象自身）
        index_of_submobject_to_align : int | slice | None, optional
            用于对齐的子对象索引（替代submobject_to_align），默认None
        coor_mask : Vect3, optional
            坐标掩码（如[1,0,1]表示仅x和z轴生效，y轴不偏移），默认[1,1,1]（全轴生效）
        """
        # 1. 计算目标参考点（根据目标是对象还是点）
        if isinstance(mobject_or_point, Mobject):
            mob = mobject_or_point
            # 确定目标对象中用于对齐的子对象（按索引或默认自身）
            if index_of_submobject_to_align is not None:
                target_aligner = mob[index_of_submobject_to_align]
            else:
                target_aligner = mob
            # 目标点 = 目标对齐物的“对齐边缘+方向”对应的包围盒点（如目标右边缘）
            target_point = target_aligner.get_bounding_box_point(
                aligned_edge + direction
            )
        else:
            # 目标是点时，直接使用该点作为目标点
            target_point = mobject_or_point

        # 2. 确定当前对象中用于对齐的部分（子对象或自身）
        if submobject_to_align is not None:
            aligner = submobject_to_align
        elif index_of_submobject_to_align is not None:
            aligner = self[index_of_submobject_to_align]
        else:
            aligner = self
        # 当前对齐点 = 对齐物的“对齐边缘-方向”对应的包围盒点（如当前左边缘）
        point_to_align = aligner.get_bounding_box_point(aligned_edge - direction)

        # 3. 计算平移向量并执行平移（应用坐标掩码，限制生效轴）
        shift_vector = (target_point - point_to_align + buff * direction) * coor_mask
        self.shift(shift_vector)
        return self

    def shift_onto_screen(self, **kwargs) -> Self:
        """
        将对象平移到屏幕内：检查对象的上下左右边缘是否超出屏幕范围，
        若超出则将对应边缘对齐到屏幕边缘（保留缓冲），确保对象完全显示在屏幕内。
        
        参数
        -----
        **kwargs
            传递给to_edge方法的参数，如buff（对象与屏幕边缘的缓冲距离）
        """
        # 屏幕在x、y轴的半长度（用于判断是否超出屏幕）
        space_lengths = [FRAME_X_RADIUS, FRAME_Y_RADIUS]
        # 遍历四个方向（上、下、左、右），逐个检查是否超出屏幕
        for vect in UP, DOWN, LEFT, RIGHT:
            # 确定当前方向对应的维度（x轴0或y轴1）
            dim = np.argmax(np.abs(vect))
            # 获取缓冲距离（默认使用全局默认边缘缓冲）
            buff = kwargs.get("buff", DEFAULT_MOBJECT_TO_EDGE_BUFF)
            # 屏幕在当前维度的最大有效范围（半长度 - 缓冲）
            max_val = space_lengths[dim] - buff
            # 获取对象在当前方向的边缘中心坐标
            edge_center = self.get_edge_center(vect)
            # 若边缘中心超出最大有效范围，将对象对齐到对应屏幕边缘
            if np.dot(edge_center, vect) > max_val:
                self.to_edge(vect, **kwargs)
        return self

    def is_off_screen(self) -> bool:
        """
        判断对象是否在屏幕外：检查对象的左右上下边缘是否完全超出屏幕范围，
        只要有一个方向完全超出，即判定为在屏幕外。
        
        返回
        -----
        bool
            True表示对象完全在屏幕外，False表示对象部分或全部在屏幕内
        """
        # 左边缘 > 屏幕右边界 → 完全在右侧屏幕外
        if self.get_left()[0] > FRAME_X_RADIUS:
            return True
        # 右边缘 < 屏幕左边界 → 完全在左侧屏幕外
        if self.get_right()[0] < -FRAME_X_RADIUS:
            return True
        # 下边缘 > 屏幕上边界 → 完全在上侧屏幕外
        if self.get_bottom()[1] > FRAME_Y_RADIUS:
            return True
        # 上边缘 < 屏幕下边界 → 完全在下侧屏幕外
        if self.get_top()[1] < -FRAME_Y_RADIUS:
            return True
        # 所有方向均未完全超出 → 在屏幕内
        return False

    def stretch_about_point(self, factor: float, dim: int, point: Vect3) -> Self:
        """
        围绕指定点沿维度拉伸对象：调用stretch方法，显式指定拉伸围绕的中心点，
        简化“定点拉伸”的调用流程（如围绕鼠标位置拉伸对象）。
        
        参数
        -----
        factor : float
            拉伸因子（>1拉长，0<factor<1压缩）
        dim : int
            拉伸维度（0=x轴、1=y轴、2=z轴）
        point : Vect3
            拉伸围绕的中心点（如鼠标坐标、对象顶点）
        """
        return self.stretch(factor, dim, about_point=point)

    def stretch_in_place(self, factor: float, dim: int) -> Self:
        """
        原位拉伸对象：仅调用stretch方法，无额外逻辑，当前已冗余（与stretch功能一致），
        保留该方法用于向后兼容。
        
        参数
        -----
        factor : float
            拉伸因子
        dim : int
            拉伸维度
        """
        # Now redundant with stretch（当前与stretch方法冗余）
        return self.stretch(factor, dim)

    def rescale_to_fit(self, length: float, dim: int, stretch: bool = False, **kwargs) -> Self:
        """
        缩放/拉伸对象以适配目标长度：根据stretch参数选择“等比例缩放”或“指定维度拉伸”，
        使对象在目标维度上的长度恰好匹配指定值（如将文本宽度适配为10单位）。
        
        参数
        -----
        length : float
            目标维度长度（如目标宽度、目标高度）
        dim : int
            目标维度（0=x轴、1=y轴、2=z轴）
        stretch : bool, optional
            若为True则沿指定维度拉伸（不保持宽高比），False则等比例缩放（保持宽高比），默认False
        **kwargs
            传递给stretch或scale方法的参数（如about_point指定缩放/拉伸中心点）
        """
        # 获取对象在目标维度上的当前长度
        old_length = self.length_over_dim(dim)
        # 若当前长度为0（无顶点），无需调整，直接返回
        if old_length == 0:
            return self
        # 计算缩放/拉伸因子（目标长度 / 当前长度）
        factor = length / old_length
        # 按参数选择拉伸或缩放
        if stretch:
            self.stretch(factor, dim,** kwargs)
        else:
            self.scale(factor, **kwargs)
        return self

    def stretch_to_fit_width(self, width: float, **kwargs) -> Self:
        """
        拉伸对象以适配目标宽度：调用rescale_to_fit方法，指定维度0（x轴）和stretch=True，
        沿水平方向拉伸对象，使其宽度恰好匹配目标值（不保持宽高比）。
        
        参数
        -----
        width : float
            目标宽度
        **kwargs
            传递给stretch方法的参数（如about_point指定拉伸中心点）
        """
        return self.rescale_to_fit(width, 0, stretch=True, **kwargs)

    def stretch_to_fit_height(self, height: float, **kwargs) -> Self:
        """
        拉伸对象以适配目标高度：调用rescale_to_fit方法，指定维度1（y轴）和stretch=True，
        沿垂直方向拉伸对象，使其高度恰好匹配目标值（不保持宽高比）。
        
        参数
        -----
        height : float
            目标高度
        **kwargs
            传递给stretch方法的参数（如about_point指定拉伸中心点）
        """
        return self.rescale_to_fit(height, 1, stretch=True, **kwargs)
    
    def stretch_to_fit_depth(self, depth: float, **kwargs) -> Self:
        """
        拉伸对象以适配目标深度：调用rescale_to_fit方法，指定维度2（z轴）和stretch=True，
        沿深度方向拉伸对象，使其深度恰好匹配目标值（不保持宽高比，仅3D对象生效）。
        
        参数
        -----
        depth : float
            目标深度
        **kwargs
            传递给stretch方法的参数（如about_point指定拉伸中心点）
        """
        return self.rescale_to_fit(depth, 2, stretch=True,** kwargs)

    def set_width(self, width: float, stretch: bool = False, **kwargs) -> Self:
        """
        设置对象宽度：调用rescale_to_fit方法，指定维度0（x轴），根据stretch参数选择
        “等比例缩放”或“水平拉伸”，使对象宽度恰好匹配目标值（是rescale_to_fit的宽度专用简化版）。
        
        参数
        -----
        width : float
            目标宽度
        stretch : bool, optional
            为True时水平拉伸（不保持宽高比），False时等比例缩放（保持宽高比），默认False
        **kwargs
            传递给stretch或scale方法的参数（如about_point指定缩放/拉伸中心点）
        """
        return self.rescale_to_fit(width, 0, stretch=stretch, **kwargs)

    def set_height(self, height: float, stretch: bool = False, **kwargs) -> Self:
        """
        设置对象高度：调用rescale_to_fit方法，指定维度1（y轴），根据stretch参数选择
        “等比例缩放”或“垂直拉伸”，使对象高度恰好匹配目标值（是rescale_to_fit的高度专用简化版）。
        
        参数
        -----
        height : float
            目标高度
        stretch : bool, optional
            为True时垂直拉伸（不保持宽高比），False时等比例缩放（保持宽高比），默认False
        **kwargs
            传递给stretch或scale方法的参数（如about_point指定缩放/拉伸中心点）
        """
        return self.rescale_to_fit(height, 1, stretch=stretch,** kwargs)

    def set_depth(self, depth: float, stretch: bool = False, **kwargs) -> Self:
        """
        设置对象深度：调用rescale_to_fit方法，指定维度2（z轴），根据stretch参数选择
        “等比例缩放”或“深度拉伸”，使对象深度恰好匹配目标值（仅3D对象生效，是rescale_to_fit的深度专用简化版）。
        
        参数
        -----
        depth : float
            目标深度
        stretch : bool, optional
            为True时深度拉伸（不保持宽高比），False时等比例缩放（保持宽高比），默认False
        **kwargs
            传递给stretch或scale方法的参数（如about_point指定缩放/拉伸中心点）
        """
        return self.rescale_to_fit(depth, 2, stretch=stretch, **kwargs)

    def set_max_width(self, max_width: float, **kwargs) -> Self:
        """
        设置对象最大宽度：仅当对象当前宽度超过max_width时，调用set_width将宽度缩放到max_width，
        宽度未超限时不做调整（用于限制对象最大尺寸，避免溢出）。
        
        参数
        -----
        max_width : float
            最大允许宽度
        **kwargs
            传递给set_width方法的参数（如stretch、about_point）
        """
        if self.get_width() > max_width:
            self.set_width(max_width, **kwargs)
        return self

    def set_max_height(self, max_height: float, **kwargs) -> Self:
        """
        设置对象最大高度：仅当对象当前高度超过max_height时，调用set_height将高度缩放到max_height，
        高度未超限时不做调整（用于限制对象最大尺寸）。
        
        参数
        -----
        max_height : float
            最大允许高度
        **kwargs
            传递给set_height方法的参数（如stretch、about_point）
        """
        if self.get_height() > max_height:
            self.set_height(max_height,** kwargs)
        return self

    def set_max_depth(self, max_depth: float, **kwargs) -> Self:
        """
        设置对象最大深度：仅当对象当前深度超过max_depth时，调用set_depth将深度缩放到max_depth，
        深度未超限时不做调整（仅3D对象生效，用于限制最大深度）。
        
        参数
        -----
        max_depth : float
            最大允许深度
        **kwargs
            传递给set_depth方法的参数（如stretch、about_point）
        """
        if self.get_depth() > max_depth:
            self.set_depth(max_depth, **kwargs)
        return self

    def set_min_width(self, min_width: float, **kwargs) -> Self:
        """
        设置对象最小宽度：仅当对象当前宽度小于min_width时，调用set_width将宽度放大到min_width，
        宽度未小于时不做调整（用于保证对象最小显示尺寸）。
        
        参数
        -----
        min_width : float
            最小允许宽度
        **kwargs
            传递给set_width方法的参数（如stretch、about_point）
        """
        if self.get_width() < min_width:
            self.set_width(min_width,** kwargs)
        return self

    def set_min_height(self, min_height: float, **kwargs) -> Self:
        """
        设置对象最小高度：仅当对象当前高度小于min_height时，调用set_height将高度放大到min_height，
        高度未小于时不做调整（用于保证对象最小显示尺寸）。
        
        参数
        -----
        min_height : float
            最小允许高度
        **kwargs
            传递给set_height方法的参数（如stretch、about_point）
        """
        if self.get_height() < min_height:
            self.set_height(min_height, **kwargs)
        return self

    def set_min_depth(self, min_depth: float, **kwargs) -> Self:
        """
        设置对象最小深度：仅当对象当前深度小于min_depth时，调用set_depth将深度放大到min_depth，
        深度未小于时不做调整（仅3D对象生效，用于保证最小深度）。
        
        参数
        -----
        min_depth : float
            最小允许深度
        **kwargs
            传递给set_depth方法的参数（如stretch、about_point）
        """
        if self.get_depth() < min_depth:
            self.set_depth(min_depth,** kwargs)
        return self

    def set_shape(
        self,
        width: Optional[float] = None,
        height: Optional[float] = None,
        depth: Optional[float] = None,
        **kwargs
    ) -> Self:
        """
        同时设置对象的宽、高、深：分别选指定宽度、高度、深度中的一个或多个，
        通过拉伸（不保持宽高比）将对象调整到目标尺寸，未指定的维度保持不变。
        
        参数
        -----
        width : Optional[float], optional
            目标宽度，为None时不调整宽度，默认None
        height : Optional[float], optional
            目标高度，为None时不调整高度，默认None
        depth : Optional[float], optional
            目标深度，为None时不调整深度，默认None
        **kwargs
            传递给set_width/set_height/set_depth的参数（如about_point指定拉伸中心）
        """
        # 分别设置指定的维度（均使用拉伸模式，不保持比例）
        if width is not None:
            self.set_width(width, stretch=True,** kwargs)
        if height is not None:
            self.set_height(height, stretch=True, **kwargs)
        if depth is not None:
            self.set_depth(depth, stretch=True,** kwargs)
        return self

    def set_coord(self, value: float, dim: int, direction: Vect3 = ORIGIN) -> Self:
        """
        设置对象在指定维度的坐标：计算当前坐标与目标坐标的差值，沿该维度平移对象，
        使指定方向的点（如中心、左边缘）在目标维度上的坐标恰好为目标值。
        
        参数
        -----
        value : float
            目标坐标值
        dim : int
            目标维度（0=x轴、1=y轴、2=z轴）
        direction : Vect3, optional
            参考方向（如LEFT表示左边缘、ORIGIN表示中心），默认ORIGIN
        """
        # 获取对象在指定维度和方向上的当前坐标
        curr = self.get_coord(dim, direction)
        # 计算平移向量：仅目标维度有差值，其他维度为0
        shift_vect = np.zeros(self.dim)
        shift_vect[dim] = value - curr
        # 执行平移，将坐标设置为目标值
        self.shift(shift_vect)
        return self

    def set_x(self, x: float, direction: Vect3 = ORIGIN) -> Self:
        """
        设置对象在x轴的坐标：调用set_coord方法，指定维度0（x轴），
        使对象指定方向的点（如左边缘、中心）的x坐标为目标值。
        
        参数
        -----
        x : float
            目标x坐标值
        direction : Vect3, optional
            参考方向（如LEFT表示左边缘x坐标），默认ORIGIN（中心x坐标）
        """
        return self.set_coord(x, 0, direction)

    def set_y(self, y: float, direction: Vect3 = ORIGIN) -> Self:
        """
        设置对象在y轴的坐标：调用set_coord方法，指定维度1（y轴），
        使对象指定方向的点（如上边缘、中心）的y坐标为目标值。
        
        参数
        -----
        y : float
            目标y坐标值
        direction : Vect3, optional
            参考方向（如UP表示上边缘y坐标），默认ORIGIN（中心y坐标）
        """
        return self.set_coord(y, 1, direction)

    def set_z(self, z: float, direction: Vect3 = ORIGIN) -> Self:
        """
        设置对象在z轴的坐标：调用set_coord方法，指定维度2（z轴），
        使对象指定方向的点的z坐标为目标值（主要影响3D渲染层级）。
        
        参数
        -----
        z : float
            目标z坐标值
        direction : Vect3, optional
            参考方向，默认ORIGIN（中心z坐标）
        """
        return self.set_coord(z, 2, direction)

    def set_z_index(self, z_index: int) -> Self:
        """
        设置对象的z-index（渲染层级）：修改对象的z_index属性，
        控制多个对象的前后渲染顺序（值越大越靠上）。
        
        参数
        -----
        z_index : int
            渲染层级值（整数，可正可负，默认0）
        """
        self.z_index = z_index
        return self

    def space_out_submobjects(self, factor: float = 1.5, **kwargs) -> Self:
        """
        增加子对象间的间距：先整体放大当前对象（含所有子对象），再将每个子对象缩小回原尺寸，
        通过“整体放大-子对象还原”的差值实现子对象间距扩大，支持指定缩放中心点。
        """
        self.scale(factor,** kwargs)  # 整体放大当前对象（子对象随父对象一起放大）
        # 逐个将子对象缩小到原尺寸（1/整体放大倍数），仅保留间距扩大效果
        for submob in self.submobjects:
            submob.scale(1. / factor)
        return self

    def move_to(
        self,
        point_or_mobject: Mobject | Vect3,
        aligned_edge: Vect3 = ORIGIN,
        coor_mask: Vect3 = np.array([1, 1, 1])
    ) -> Self:
        """
        将对象移动到目标位置：根据目标（点或对象）计算平移向量，使对象的指定边缘（如中心、左边缘）
        与目标的对应边缘对齐，支持限制生效的坐标轴。
        
        参数
        -----
        point_or_mobject : Mobject | Vect3
            目标位置，可为3D坐标点或另一个Mobject实例（使用其包围盒边缘）
        aligned_edge : Vect3, optional
            对齐边缘（如LEFT表示对象左边缘与目标左边缘对齐），默认ORIGIN（中心对齐）
        coor_mask : Vect3, optional
            坐标掩码（如[1,0,1]表示仅x、z轴移动，y轴固定），默认[1,1,1]（全轴移动）
        """
        # 确定目标点：若为对象，取其指定边缘的包围盒点；若为点，直接使用
        if isinstance(point_or_mobject, Mobject):
            target = point_or_mobject.get_bounding_box_point(aligned_edge)
        else:
            target = point_or_mobject
        # 获取当前对象需要对齐的边缘点
        point_to_align = self.get_bounding_box_point(aligned_edge)
        # 计算平移向量（目标点 - 对齐点），应用坐标掩码后执行平移
        self.shift((target - point_to_align) * coor_mask)
        return self

    def replace(self, mobject: Mobject, dim_to_match: int = 0, stretch: bool = False) -> Self:
        """
        替换目标对象的位置与尺寸：将当前对象调整为目标对象的尺寸（等比例或拉伸），
        并移动到目标对象的位置，实现“替换”目标对象的视觉效果。
        
        参数
        -----
        mobject : Mobject
            被替换的目标对象（提供尺寸和位置参考）
        dim_to_match : int, optional
            等比例缩放时的参考维度（0=x轴、1=y轴、2=z轴），默认0（按宽度匹配）
        stretch : bool, optional
            为True时按目标对象的各维度单独拉伸（不保持宽高比），False时按参考维度等比例缩放，默认False
        """
        # 若目标对象无顶点且无子对象（空对象），将当前对象缩放到0（隐藏）
        if not mobject.get_num_points() and not mobject.submobjects:
            self.scale(0)
            return self
        
        # 调整当前对象尺寸以匹配目标对象
        if stretch:
            # 拉伸模式：按目标对象的每个维度单独调整（不保持宽高比）
            for i in range(self.dim):
                self.rescale_to_fit(mobject.length_over_dim(i), i, stretch=True)
        else:
            # 等比例模式：按参考维度匹配，其他维度按比例缩放（保持宽高比）
            self.rescale_to_fit(
                mobject.length_over_dim(dim_to_match),
                dim_to_match,
                stretch=False
            )
        
        # 将当前对象移动到目标对象的中心位置
        self.shift(mobject.get_center() - self.get_center())
        return self
    
    def surround(self,
        mobject: Mobject,
        dim_to_match: int = 0,
        stretch: bool = False,
        buff: float = MED_SMALL_BUFF
    ) -> Self:
        """
        包围目标对象：先将当前对象调整为与目标对象尺寸匹配（等比例或拉伸），
        再按目标对象尺寸加缓冲距离放大，最终实现当前对象包裹目标对象且保留指定间距的效果。
        """
        # 第一步：调用replace方法，使当前对象尺寸匹配目标对象并移动到目标对象位置
        self.replace(mobject, dim_to_match, stretch)
        # 第二步：计算放大比例 =（目标对象参考维度长度 + 缓冲距离）/ 目标对象参考维度长度
        length = mobject.length_over_dim(dim_to_match)
        scale_factor = (length + buff) / length
        # 第三步：按计算的比例放大当前对象，实现包围效果
        self.scale(scale_factor)
        return self

    def put_start_and_end_on(self, start: Vect3, end: Vect3) -> Self:
        """
        固定对象的起点和终点到目标位置：通过缩放、旋转（2D+3D）和平移，
        将对象的起始端点（curr_start）移动到目标起点（start），终止端点（curr_end）移动到目标终点（end），
        适用于线段、箭头等有明确起止方向的对象。        
        
        异常
        -----
        Exception
            若对象是闭合回路（起点与终点重合，curr_vect为0向量），抛出“无法定位闭合回路端点”的异常
        """
        # 1. 获取当前对象的起点和终点，计算当前起止向量
        curr_start, curr_end = self.get_start_and_end()
        curr_vect = curr_end - curr_start
        # 若当前起止向量为0（闭合回路），无法定位端点，抛出异常
        if np.all(curr_vect == 0):
            raise Exception("Cannot position endpoints of closed loop")
        
        # 2. 计算目标起止向量
        target_vect = end - start

        # 3. 缩放对象：使当前对象的长度匹配目标起止向量的长度（围绕当前起点缩放，避免起点偏移）
        self.scale(
            get_norm(target_vect) / get_norm(curr_vect),  # 缩放比例 = 目标长度 / 当前长度
            about_point=curr_start,
        )

        # 4. 2D平面旋转：使当前对象的平面方向匹配目标向量的平面方向（绕z轴旋转）
        self.rotate(
            angle_of_vector(target_vect) - angle_of_vector(curr_vect),  # 旋转角度 = 目标角度 - 当前角度
        )

        # 5. 3D空间旋转：调整对象在z轴方向的倾斜，匹配目标向量的3D方向（绕垂直于目标向量的轴旋转）
        self.rotate(
            # 旋转角度 = 当前向量的z轴倾斜角 - 目标向量的z轴倾斜角
            np.arctan2(curr_vect[2], get_norm(curr_vect[:2])) - np.arctan2(target_vect[2], get_norm(target_vect[:2])),
            axis=np.array([-target_vect[1], target_vect[0], 0]),  # 旋转轴：垂直于目标向量的平面轴
        )

        # 6. 平移对象：将对象起点从curr_start移动到目标起点start
        self.shift(start - self.get_start())
        return self

    # 颜色相关方法（控制对象的RGBA颜色与透明度）

    @affects_family_data
    def set_rgba_array(
        self,
        rgba_array: npt.ArrayLike,
        name: str = "rgba",
        recurse: bool = False
    ) -> Self:
        """
        批量设置家族成员的RGBA颜色数组：将指定的RGBA数组（含红、绿、蓝、透明度通道）
        赋值给对象自身或其家族成员的颜色数据字段，支持自定义颜色字段名。
        
        参数
        -----
        rgba_array : npt.ArrayLike
            RGBA颜色数组，形状需与对象顶点数量匹配（如(N,4)，N为顶点数，4个通道分别对应RGBA）
        name : str, optional
            颜色数据在对象data中的字段名，默认"rgba"（标准颜色字段）
        recurse : bool, optional
            是否递归设置所有子对象的颜色，默认False（仅设置当前对象）
        """
        # 遍历目标家族成员（自身或含子对象）
        for mob in self.get_family(recurse):
            # 若对象有顶点，使用其现有data；若无顶点，使用默认数据模板
            data = mob.data if mob.get_num_points() > 0 else mob._data_defaults
            # 将RGBA数组赋值到指定颜色字段（直接修改数据，确保颜色实时更新）
            data[name][:] = rgba_array
        return self

    def set_color_by_rgba_func(
        self,
        func: Callable[[Vect3Array], Vect4Array],
        recurse: bool = True
    ) -> Self:
        """
        通过函数动态设置RGBA颜色：对每个家族成员的顶点坐标应用自定义函数，
        由函数返回对应顶点的RGBA颜色，实现基于位置的动态着色（如渐变、纹理）。
        
        说明
        -----
        Func should take in a point in R3 and output an rgba value
        （函数需接收3D空间中的顶点坐标，输出对应的RGBA颜色值）
        """
        # 遍历家族中所有成员，逐个通过函数设置颜色
        for mob in self.get_family(recurse):
            # 1. 获取当前成员的所有顶点坐标
            # 2. 将顶点坐标传入函数，生成对应的RGBA颜色数组
            # 3. 调用set_rgba_array应用生成的颜色数组
            mob.set_rgba_array(func(mob.get_points()))
        return self

    def set_color_by_rgb_func(
        self,
        func: Callable[[Vect3Array], Vect3Array],
        opacity: float = 1,
        recurse: bool = True
    ) -> Self:
        """
        通过函数动态设置RGB颜色并统一控制透明度：对顶点坐标应用自定义函数生成RGB颜色，
        再拼接统一的透明度通道（RGBA），实现“颜色动态、透明度固定”的着色效果。
        
        说明
        -----
        Func should take in a point in R3 and output an rgb value
        （函数需接收3D空间中的顶点坐标，输出对应的RGB颜色值）
        """
        # 遍历家族中所有成员，逐个设置颜色
        for mob in self.get_family(recurse):
            # 1. 获取当前成员的顶点坐标
            points = mob.get_points()
            # 2. 生成与顶点数量匹配的统一透明度数组（N,1）
            opacity_array = np.ones((points.shape[0], 1)) * opacity
            # 3. 函数生成RGB数组 → 拼接透明度数组 → 得到RGBA数组
            rgba_array = np.hstack((func(points), opacity_array))
            # 4. 应用RGBA数组设置颜色
            mob.set_rgba_array(rgba_array)
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
