# 从__future__导入annotations，支持Python 3.7及以下版本的类型提示语法（延迟类型解析）
from __future__ import annotations

# 导入itertools库并简写为it，用于处理迭代器（如生成相邻对、笛卡尔积等）
import itertools as it

# 导入numpy库并简写为np，用于数值计算、数组操作（如向量运算、矩阵处理）
import numpy as np
# 从scipy.integrate导入solve_ivp，用于求解常微分方程组（可用于模拟运动、变化过程）
from scipy.integrate import solve_ivp

# 从manimlib.constants导入帧尺寸常量，控制画面宽高
from manimlib.constants import FRAME_HEIGHT, FRAME_WIDTH
# 从manimlib.constants导入默认图形颜色常量
from manimlib.constants import DEFAULT_MOBJECT_COLOR
# 从manimlib.animation.indication导入VShowPassingFlash，用于创建闪烁指示动画
from manimlib.animation.indication import VShowPassingFlash
# 从manimlib.mobject.types.vectorized_mobject导入向量图形组合类
from manimlib.mobject.types.vectorized_mobject import VGroup
from manimlib.mobject.types.vectorized_mobject import VMobject
# 从manimlib.utils.bezier导入贝塞尔曲线相关函数，用于插值计算
from manimlib.utils.bezier import interpolate
from manimlib.utils.bezier import inverse_interpolate
# 从manimlib.utils.color导入颜色处理函数，用于生成颜色映射、渐变色列表
from manimlib.utils.color import get_colormap_list
from manimlib.utils.color import get_color_map
# 从manimlib.utils.iterables导入可迭代对象处理函数，用于生成笛卡尔积
from manimlib.utils.iterables import cartesian_product
# 从manimlib.utils.rate_functions导入线性速率函数，用于控制动画速度
from manimlib.utils.rate_functions import linear
# 从manimlib.utils.space_ops导入空间运算函数，用于计算向量的模（长度）
from manimlib.utils.space_ops import get_norm

# 导入typing模块中的TYPE_CHECKING常量，用于条件性导入类型提示（仅类型检查时生效，不影响运行）
from typing import TYPE_CHECKING

# 若处于类型检查阶段（非运行时），导入所需的类型提示类/工具
if TYPE_CHECKING:
    from typing import Callable, Iterable, Sequence, TypeVar, Tuple, Optional
    # 从manimlib.typing导入Manim自定义类型，用于类型注解（颜色、向量等）
    from manimlib.typing import ManimColor, Vect3, VectN, VectArray, Vect3Array, Vect4Array

    # 从manimlib.mobject.coordinate_systems导入坐标系类，用于类型注解
    from manimlib.mobject.coordinate_systems import CoordinateSystem
    # 从manimlib.mobject.mobject导入Mobject基类，用于类型注解
    from manimlib.mobject.mobject import Mobject

    # 定义类型变量T，限定为Mobject的子类，用于泛型类型注解
    T = TypeVar("T")


#### Delete these two ###
def get_vectorized_rgb_gradient_function(
    min_value: T,
    max_value: T,
    color_map: str
) -> Callable[[VectN], Vect3Array]:
    """
    创建**向量化的RGB渐变函数**，支持批量将数值映射为对应颜色
    
    参数:
        min_value: 数值范围的最小值（对应渐变的起始颜色）
        max_value: 数值范围的最大值（对应渐变的结束颜色）
        color_map: 颜色映射名称（如"viridis"、"coolwarm"，需符合Manim支持的色表）
    
    返回:
        向量化函数：输入数值数组，输出对应RGB颜色数组（每个颜色为[R,G,B]格式）
    """
    # 根据颜色映射名称，获取对应的RGB颜色列表，并转换为numpy数组
    rgbs = np.array(get_colormap_list(color_map))

    # 定义内部向量化函数，实现数值到颜色的映射
    def func(values):
        # 1. 将输入数值归一化到[0,1]区间：计算每个数值在min-max范围内的相对位置
        alphas = inverse_interpolate(
            min_value, max_value, np.array(values)
        )
        # 2. 裁剪归一化结果：确保超出min-max的数值也能映射到有效颜色（<min取0，>max取1）
        alphas = np.clip(alphas, 0, 1)
        # 3. 缩放归一化值：将[0,1]映射到颜色列表的索引范围（如色表有100种颜色，缩放为[0,99]）
        scaled_alphas = alphas * (len(rgbs) - 1)
        # 4. 计算当前颜色的索引（向下取整，获取基础颜色）
        indices = scaled_alphas.astype(int)
        # 5. 计算下一个颜色的索引（避免超出色表长度，用clip限制最大值）
        next_indices = np.clip(indices + 1, 0, len(rgbs) - 1)
        # 6. 计算两个颜色间的插值比例（scaled_alphas的小数部分，决定两种颜色的混合程度）
        inter_alphas = scaled_alphas % 1
        # 7. 重塑插值比例数组：使其与RGB颜色维度匹配（每个比例重复3次，对应R/G/B通道）
        inter_alphas = inter_alphas.repeat(3).reshape((len(indices), 3))
        # 8. 两种颜色的线性插值：生成最终的RGB颜色数组
        result = interpolate(rgbs[indices], rgbs[next_indices], inter_alphas)
        return result

    # 返回向量化的颜色映射函数
    return func


def get_rgb_gradient_function(
    min_value: T,
    max_value: T,
    color_map: str
) -> Callable[[float], Vect3]:
    """
    创建**单值RGB渐变函数**，将单个数值映射为对应颜色（基于向量化函数封装）
    
    参数:
        min_value: 数值范围的最小值
        max_value: 数值范围的最大值
        color_map: 颜色映射名称
    
    返回:
        单值函数：输入单个数值，输出对应RGB颜色（[R,G,B]格式）
    """
    # 先获取向量化的渐变函数
    vectorized_func = get_vectorized_rgb_gradient_function(min_value, max_value, color_map)
    # 封装为单值函数：输入单个数值→转为数组传入向量化函数→取第一个结果（仅一个元素）
    return lambda value: vectorized_func(np.array([value]))[0]


#### 以下为向量场相关工具函数 ####
def ode_solution_points(function, state0, time, dt=0.01):
    """
    求解常微分方程（ODE），生成运动轨迹的坐标点
    
    参数:
        function: ODE函数，输入当前状态（如位置/速度），输出状态的变化率（如速度/加速度）
        state0: 初始状态（如初始位置[x0,y0]或初始位置+速度[x0,y0,vx0,vy0]）
        time: 模拟总时长（从t=0到t=time）
        dt: 时间步长，默认0.01（步长越小，轨迹越精确）
    
    返回:
        轨迹点数组：shape=(总步数, 状态维度)，每一行是对应时间点的状态（如位置）
    """
    # 调用scipy的solve_ivp求解ODE
    solution = solve_ivp(
        lambda t, state: function(state),  # ODE函数（忽略时间t，适用于自治系统）
        t_span=(0, time),                  # 时间范围
        y0=state0,                         # 初始状态
        t_eval=np.arange(0, time, dt)      # 输出时间点（按dt间隔生成）
    )
    # 转换结果格式：将solve_ivp返回的"行=状态维度，列=时间步"转为"行=时间步，列=状态维度"
    return solution.y.T


def move_along_vector_field(
    mobject: Mobject,
    func: Callable[[Vect3], Vect3]
) -> Mobject:
    """
    让单个图形（Mobject）沿向量场运动（添加实时更新器）
    
    参数:
        mobject: 要运动的图形（如点、圆、正方形）
        func: 向量场函数，输入图形的中心坐标，输出对应位置的向量（运动方向和速度）
    
    返回:
        添加了更新器的图形对象（会自动沿向量场运动）
    """
    # 为图形添加更新器：每帧根据向量场更新位置
    mobject.add_updater(
        lambda m, dt: m.shift(
            func(m.get_center()) * dt  # 位移 = 向量场强度 × 时间步长（dt是每帧时间）
        )
    )
    return mobject


def move_submobjects_along_vector_field(
    mobject: Mobject,
    func: Callable[[Vect3], Vect3]
) -> Mobject:
    """
    让组合图形（Mobject）的所有子图形，分别沿向量场运动（添加实时更新器）
    
    参数:
        mobject: 组合图形（如VGroup，包含多个子图形）
        func: 向量场函数，输入子图形中心坐标，输出对应位置的向量
    
    返回:
        添加了更新器的组合图形对象（子图形各自沿向量场运动）
    """
    # 定义更新逻辑：遍历所有子图形，计算并更新位置
    def apply_nudge(mob, dt):
        for submob in mob:
            # 获取子图形中心的x、y坐标（忽略z轴，适用于2D场景）
            x, y = submob.get_center()[:2]
            # 检查子图形是否在画面内（避免运动到屏幕外）
            if abs(x) < FRAME_WIDTH and abs(y) < FRAME_HEIGHT:
                # 子图形位移 = 向量场强度 × 时间步长
                submob.shift(func(submob.get_center()) * dt)

    # 为组合图形添加更新器
    mobject.add_updater(apply_nudge)
    return mobject


def move_points_along_vector_field(
    mobject: Mobject,
    func: Callable[[float, float], Iterable[float]],
    coordinate_system: CoordinateSystem
) -> Mobject:
    """
    让图形的每个顶点，沿指定坐标系下的向量场运动（支持自定义坐标系，如极坐标）
    
    参数:
        mobject: 要运动的图形（如多边形，顶点位置会更新）
        func: 向量场函数（在目标坐标系下），输入坐标系内坐标（如极径r、极角θ），输出向量分量
        coordinate_system: 目标坐标系（如PolarCoordinateSystem，提供坐标转换功能）
    
    返回:
        添加了更新器的图形对象（顶点沿自定义坐标系的向量场运动）
    """
    # 简化变量名：目标坐标系
    cs = coordinate_system
    # 获取坐标系原点（用于后续向量计算）
    origin = cs.get_origin()

    # 定义更新逻辑：遍历图形每个顶点，按向量场更新位置
    def apply_nudge(mob, dt):
        mob.apply_function(
            lambda p:  # p：图形的某个顶点（笛卡尔坐标）
            p + (  # 顶点最终位移 = 原始位置 + 向量场位移
                cs.c2p(*func(*cs.p2c(p))) - origin  # 1. 笛卡尔坐标→目标坐标系坐标→向量场函数→目标坐标系向量→笛卡尔坐标向量
            ) * dt  # 2. 向量 × 时间步长（dt），得到每帧位移
        )
    # 为图形添加更新器
    mobject.add_updater(apply_nudge)
    return mobject


def get_sample_coords(
    coordinate_system: CoordinateSystem,
    density: float = 1.0
) -> it.product[tuple[Vect3, ...]]:
    """
    在指定坐标系中生成均匀采样的坐标点集合
    
    参数:
        coordinate_system: 坐标系对象（如笛卡尔坐标系、极坐标系等）
        density: 采样密度，值越大采样点越密集（默认1.0）
    
    返回:
        采样点数组，每个元素是一个坐标点的坐标值
    """
    # 存储每个维度的采样范围
    ranges = []
    # 获取坐标系所有维度的范围参数（最小值、最大值、步长）
    for range_args in coordinate_system.get_all_ranges():
        _min, _max, step = range_args
        # 根据密度调整步长：密度越大，步长越小（采样点越密集）
        step /= density
        # 生成当前维度的采样点序列（从_min到_max，步长为调整后的step）
        ranges.append(np.arange(_min, _max + step, step))
    # 使用笛卡尔积生成所有维度的采样点组合，并转换为numpy数组
    return np.array(list(it.product(*ranges)))


def vectorize(pointwise_function: Callable[[Tuple], Tuple]):
    """
    将逐点处理的函数转换为支持数组输入的向量化函数
    
    参数:
        pointwise_function: 接收单个坐标点（元组形式）并返回处理结果的函数
    
    返回:
        向量化函数：接收坐标点数组，返回对应处理结果的数组
    """
    # 定义向量化函数
    def v_func(coords_array: VectArray) -> VectArray:
        # 遍历输入数组中的每个坐标点，应用原始逐点函数，最后转换为numpy数组
        return np.array([pointwise_function(*coords) for coords in coords_array])

    # 返回向量化函数
    return v_func


# Mobjects


class VectorField(VMobject):
    """向量场类，继承自VMobject，用于在Manim中生成并显示2D/3D向量场（箭头网格）"""
    
    def __init__(
        self,
        # 向量化向量场函数：输入坐标点数组，输出对应位置的向量数组
        func: Callable[[VectArray], VectArray],
        # 关联的坐标系（如Axes/NumberPlane），用于坐标转换
        coordinate_system: CoordinateSystem,
        # 自定义采样点坐标数组（可选，未提供则自动生成）
        sample_coords: Optional[VectArray] = None,
        # 采样密度（值越大，箭头分布越密集），默认2.0
        density: float = 2.0,
        # 向量大小范围（用于颜色映射归一化），可选
        magnitude_range: Optional[Tuple[float, float]] = None,
        # 向量统一颜色（指定后忽略颜色映射），可选
        color: Optional[ManimColor] = None,
        # 默认颜色映射名称（3Blue1Brown风格），默认"3b1b_colormap"
        color_map_name: Optional[str] = "3b1b_colormap",
        # 自定义颜色映射函数（输入向量大小序列，输出RGBA数组），可选
        color_map: Optional[Callable[[Sequence[float]], Vect4Array]] = None,
        # 线条不透明度，默认1.0
        stroke_opacity: float = 1.0,
        # 向量主体线条宽度，默认3
        stroke_width: float = 3,
        # 箭头尖端宽度与主体宽度的比例，默认4
        tip_width_ratio: float = 4,
        # 箭头尖端长度与尖端宽度的比例，默认0.01
        tip_len_to_width: float = 0.01,
        # 向量显示的最大长度（可选，None则自动计算）
        max_vect_len: float | None = None,
        # 最大向量长度与采样步长的比例（自动计算时用），默认0.8
        max_vect_len_to_step_size: float = 0.8,
        # 是否使用平面线条样式，默认False
        flat_stroke: bool = False,
        # 向量大小到不透明度的映射函数（待完善）
        norm_to_opacity_func=None,
        **kwargs  # 其他传递给父类VMobject的参数
    ):
        # 存储向量场核心函数
        self.func = func
        # 存储关联的坐标系
        self.coordinate_system = coordinate_system
        # 存储主体线条宽度
        self.stroke_width = stroke_width
        # 存储箭头尖端宽度比例
        self.tip_width_ratio = tip_width_ratio
        # 存储箭头尖端长度比例
        self.tip_len_to_width = tip_len_to_width
        # 存储不透明度映射函数
        self.norm_to_opacity_func = norm_to_opacity_func

        # 处理采样点：使用自定义采样点或自动生成
        if sample_coords is not None:
            self.sample_coords = sample_coords
        else:
            # 调用工具函数生成坐标系内的均匀采样点
            self.sample_coords = get_sample_coords(coordinate_system, density)
        # 将采样点从坐标系坐标转换为全局坐标
        self.update_sample_points()

        # 计算向量显示的最大长度
        if max_vect_len is None:
            # 自动计算：根据采样点间距和比例系数
            step_size = get_norm(self.sample_points[1] - self.sample_points[0])
            self.max_displayed_vect_len = max_vect_len_to_step_size * step_size
        else:
            # 手动指定：结合坐标系单位大小（适配不同坐标轴缩放）
            self.max_displayed_vect_len = max_vect_len * coordinate_system.x_axis.get_unit_size()

        # 准备颜色映射的向量大小范围
        if magnitude_range is None:
            # 自动计算：向量场在所有采样点的最大大小
            max_value = max(map(get_norm, func(self.sample_coords)))
            magnitude_range = (0, max_value)
        self.magnitude_range = magnitude_range

        # 配置颜色映射逻辑
        if color is not None:
            # 若指定统一颜色，禁用颜色映射
            self.color_map = None
        else:
            # 使用自定义颜色映射或默认颜色映射
            self.color_map = color_map or get_color_map(color_map_name)

        # 初始化箭头各部分的基础宽度数组
        self.init_base_stroke_width_array(len(self.sample_coords))

        # 调用父类VMobject的构造函数，初始化向量场图形属性
        super().__init__(
            stroke_opacity=stroke_opacity,
            flat_stroke=flat_stroke,** kwargs
        )
        # 设置向量线条的颜色和宽度
        self.set_stroke(color, stroke_width)
        # 生成并更新所有向量箭头的显示
        self.update_vectors()

    def init_points(self):
        """初始化向量场的顶点数据（每个箭头用8个顶点定义）"""
        # 采样点数量 = 箭头数量
        n_samples = len(self.sample_coords)
        # 总顶点数 = 8*箭头数 - 1（最后一个箭头无需额外间隔点）
        self.set_points(np.zeros((8 * n_samples - 1, 3)))
        # 设置箭头各顶点的连接方式为"无特殊关节"
        self.set_joint_type('no_joint')

    def get_sample_points(
        self,
        center: np.ndarray,
        width: float,
        height: float,
        depth: float,
        x_density: float,
        y_density: float,
        z_density: float
    ) -> np.ndarray:
        """在指定3D区域内生成均匀采样点（支持局部采样）"""
        # 计算区域中心到角落的半长向量
        to_corner = np.array([width / 2, height / 2, depth / 2])
        # 计算各维度的采样间隔（密度越大，间隔越小）
        spacings = 1.0 / np.array([x_density, y_density, z_density])
        # 调整半长向量，确保为采样间隔的整数倍（避免边缘采样点错位）
        to_corner = spacings * (to_corner / spacings).astype(int)
        # 计算区域的下/上边界坐标
        lower_corner = center - to_corner
        upper_corner = center + to_corner + spacings
        # 生成各维度采样点序列，并用笛卡尔积组合为3D采样点数组
        return cartesian_product(*(
            np.arange(low, high, space)
            for low, high, space in zip(lower_corner, upper_corner, spacings)
        ))

    def init_base_stroke_width_array(self, n_sample_points):
        """初始化箭头各部分的基础宽度比例数组（控制箭头形状）"""
        # 基础宽度数组：长度 = 8*箭头数 - 1，默认值1（与主体宽度一致）
        arr = np.ones(8 * n_sample_points - 1)
        # 箭头尖端部分宽度比例：尖端起点（4::8）→ 尖端中点（5::8）→ 尖端终点（6::8）
        arr[4::8] = self.tip_width_ratio       # 尖端起点：宽度=主体*比例
        arr[5::8] = self.tip_width_ratio * 0.5 # 尖端中点：宽度=主体*比例*0.5（渐变）
        arr[6::8] = 0                          # 尖端终点：宽度=0（收尖）
        arr[7::8] = 0                          # 箭头间间隔点：宽度=0（避免连在一起）
        # 存储基础宽度比例数组
        self.base_stroke_width_array = arr

    def set_sample_coords(self, sample_coords: VectArray):
        """更新采样点坐标（外部调用，用于动态调整采样区域）"""
        self.sample_coords = sample_coords
        return self

    def set_stroke(self, color=None, width=None, opacity=None, behind=None, flat=None, recurse=True):
        """重写父类方法：统一设置线条样式（颜色/宽度/不透明度）"""
        # 调用父类set_stroke设置颜色、不透明度等（宽度单独处理）
        super().set_stroke(color, None, opacity, behind, flat, recurse)
        # 若指定宽度，调用set_stroke_width更新
        if width is not None:
            self.set_stroke_width(float(width))
        return self

    def set_stroke_width(self, width: float):
        """设置向量线条宽度（应用基础宽度比例数组）"""
        if self.get_num_points() > 0:
            # 计算最终宽度：主体宽度 × 基础比例数组（实现箭头尖端渐变）
            self.get_stroke_widths()[:] = width * self.base_stroke_width_array
            # 更新存储的主体宽度
            self.stroke_width = width
        return self

    def update_sample_points(self):
        """将采样点从坐标系坐标（如Axes的x/y）转换为全局坐标"""
        # coordinate_system.c2p：坐标系坐标→全局坐标；sample_coords.T：转置为行向量便于传入
        self.sample_points = self.coordinate_system.c2p(*self.sample_coords.T)

    def update_vectors(self):
        """核心方法：计算并绘制所有向量箭头（位置、长度、颜色、宽度）"""
        # 计算箭头尖端的实际宽度和长度
        tip_width = self.tip_width_ratio * self.stroke_width
        tip_len = self.tip_len_to_width * tip_width

        # 1. 计算坐标系内的向量输出（未转换为全局坐标）
        outputs = self.func(self.sample_coords)
        # 计算每个向量的大小（用于颜色映射和长度限制）
        output_norms = np.linalg.norm(outputs, axis=1)[:, np.newaxis]

        # 2. 将向量转换为全局坐标下的向量（排除坐标系原点偏移）
        # coordinate_system.c2p转换向量→减去坐标系原点→得到全局坐标系下的向量
        out_vects = self.coordinate_system.c2p(*outputs.T) - self.coordinate_system.get_origin()
        # 计算全局向量的大小
        out_vect_norms = np.linalg.norm(out_vects, axis=1)[:, np.newaxis]
        # 计算全局向量的单位向量（用于确定箭头方向）
        unit_outputs = np.zeros_like(out_vects)
        # 安全除法：避免除以0（where指定非零位置才计算）
        np.true_divide(out_vects, out_vect_norms, out=unit_outputs, where=(out_vect_norms > 0))

        # 3. 计算箭头的显示长度（限制最大长度，避免重叠）
        max_len = self.max_displayed_vect_len
        if max_len < np.inf:
            # 用双曲正切函数平滑限制长度（接近max_len时增长放缓）
            drawn_norms = max_len * np.tanh(out_vect_norms / max_len)
        else:
            # 不限制长度，使用原始向量大小
            drawn_norms = out_vect_norms

        # 4. 计算箭头主体终点到尖端起点的距离（排除尖端长度）
        # clip确保距离非负（短向量可能无明显尖端）
        dist_to_head_base = np.clip(drawn_norms - tip_len, 0, np.inf)

        # 5. 设置箭头所有顶点的位置（每个箭头8个顶点）
        points = self.get_points()
        points[0::8] = self.sample_points                  # 顶点0：箭头起点
        points[2::8] = self.sample_points + dist_to_head_base * unit_outputs  # 顶点2：箭头主体终点（尖端起点）
        points[4::8] = points[2::8]                        # 顶点4：尖端起点（与顶点2重合，用于宽度过渡）
        points[6::8] = self.sample_points + drawn_norms * unit_outputs        # 顶点6：箭头尖端终点
        # 顶点1/3/5：中间过渡点（取相邻顶点中点，实现平滑线条）
        for i in (1, 3, 5):
            points[i::8] = 0.5 * (points[i - 1::8] + points[i + 1::8])
        points[7::8] = points[6:-1:8]                      # 顶点7：箭头间间隔点（与前一个箭头尖端终点重合）

        # 6. 调整箭头各部分的线条宽度（应用基础比例+长度缩放）
        # 基础宽度数组×主体宽度：得到原始宽度分布
        width_arr = self.stroke_width * self.base_stroke_width_array
        # 宽度缩放系数：短向量尖端更窄（clip限制在0-1，避免负宽度）
        width_scalars = np.clip(drawn_norms / tip_len, 0, 1)
        # 扩展系数数组：每个箭头对应8个系数（最后一个箭头少1个）
        width_scalars = np.repeat(width_scalars, 8)[:-1]
        # 应用最终宽度：缩放系数×原始宽度分布
        self.get_stroke_widths()[:] = width_scalars * width_arr

        # 7. 调整箭头颜色（若启用颜色映射）
        if self.color_map is not None:
            self.get_stroke_colors()  # 确保颜色数组长度匹配顶点数
            low, high = self.magnitude_range
            # 向量大小归一化：(实际大小-最小值)/(最大值-最小值)→[0,1]
            normalized_norms = inverse_interpolate(low, high, np.repeat(output_norms, 8)[:-1])
            # 颜色映射：归一化大小→RGBA颜色（取前3通道，忽略透明度）
            self.data['stroke_rgba'][:, :3] = self.color_map(normalized_norms)[:, :3]

        # 8. 调整箭头不透明度（若启用不透明度映射）
        if self.norm_to_opacity_func is not None:
            # 扩展向量大小数组，匹配顶点数
            extended_norms = np.repeat(output_norms, 8)[:-1]
            # 应用不透明度映射函数
            self.get_stroke_opacities()[:] = self.norm_to_opacity_func(extended_norms)

        # 通知Manim图形系统：向量场数据已更新，需要重新渲染
        self.note_changed_data()
        return self

class TimeVaryingVectorField(VectorField):
    """时变向量场类，继承自VectorField，支持向量场随时间动态变化"""
    
    def __init__(
        self,
        # 时变向量场函数：输入坐标点数组和时间，输出对应时间的向量数组
        time_func: Callable[[VectArray, float], VectArray],
        coordinate_system: CoordinateSystem,  # 关联的坐标系
        **kwargs  # 传递给父类VectorField的参数（如采样密度、颜色等）
    ):
        # 初始化时间变量（初始时刻为0）
        self.time = 0

        # 定义适配父类的静态向量场函数：固定时间参数，仅接收坐标
        def func(coords):
            return time_func(coords, self.time)

        # 调用父类VectorField的构造函数，传入适配后的静态函数
        super().__init__(func, coordinate_system,** kwargs)
        # 添加时间更新器：每帧根据时间步长dt更新时间
        self.add_updater(lambda m, dt: m.increment_time(dt))
        # 设置向量场始终随时间更新（确保每帧重新计算向量）
        self.always.update_vectors()

    def increment_time(self, dt):
        """时间递增方法：每帧更新时间变量"""
        self.time += dt  # 时间 = 当前时间 + 帧时间步长dt


class StreamLines(VGroup):
    """流线类，继承自VGroup，用于绘制向量场的流线（粒子运动轨迹）"""
    
    def __init__(
        self,
        func: Callable[[VectArray], VectArray],  # 静态向量场函数（输入坐标，输出向量）
        coordinate_system: CoordinateSystem,    # 关联的坐标系
        density: float = 1.0,                   # 流线密度（值越大，流线越多）
        n_repeats: int = 1,                     # 流线重复生成次数（增加密度）
        noise_factor: float | None = None,      # 采样点随机偏移量（避免流线重叠）
        # 流线绘制参数
        solution_time: float = 3,               # 流线模拟总时长
        dt: float = 0.05,                       # 模拟时间步长（步长越小，轨迹越平滑）
        arc_len: float = 3,                     # 流线弧长（暂未启用）
        max_time_steps: int = 200,              # 最大模拟步数（暂未启用）
        n_samples_per_line: int = 10,           # 每条流线的采样点数（暂未启用）
        cutoff_norm: float = 15,                # 向量大小阈值（暂未启用）
        # 样式参数
        stroke_width: float = 1.0,              # 流线宽度
        stroke_color: ManimColor = DEFAULT_MOBJECT_COLOR,  # 流线统一颜色
        stroke_opacity: float = 1,              # 流线不透明度
        color_by_magnitude: bool = True,        # 是否按向量大小着色（True则启用颜色映射）
        magnitude_range: Tuple[float, float] = (0, 2.0),  # 颜色映射的向量大小范围
        taper_stroke_width: bool = False,       # 是否让流线宽度渐变（两端窄、中间宽）
        color_map: str = "3b1b_colormap",       # 颜色映射名称
        **kwargs  # 传递给父类VGroup的参数
    ):
        # 调用父类VGroup的构造函数
        super().__init__(**kwargs)
        # 存储核心参数
        self.func = func
        self.coordinate_system = coordinate_system
        self.density = density
        self.n_repeats = n_repeats
        self.noise_factor = noise_factor
        self.solution_time = solution_time
        self.dt = dt
        self.arc_len = arc_len
        self.max_time_steps = max_time_steps
        self.n_samples_per_line = n_samples_per_line
        self.cutoff_norm = cutoff_norm
        self.stroke_width = stroke_width
        self.stroke_color = stroke_color
        self.stroke_opacity = stroke_opacity
        self.color_by_magnitude = color_by_magnitude
        self.magnitude_range = magnitude_range
        self.taper_stroke_width = taper_stroke_width
        self.color_map = color_map

        # 绘制所有流线
        self.draw_lines()
        # 初始化流线样式（颜色、宽度等）
        self.init_style()

    def point_func(self, points: Vect3Array) -> Vect3:
        """点级向量场函数：输入全局坐标点，输出对应全局向量（适配坐标系）"""
        # 1. 全局坐标→坐标系坐标（如Axes的x/y坐标）
        in_coords = np.array(self.coordinate_system.p2c(points)).T
        # 2. 调用向量场函数，得到坐标系内的向量
        out_coords = self.func(in_coords)
        # 3. 获取坐标系原点（用于抵消偏移）
        origin = self.coordinate_system.get_origin()
        # 4. 坐标系向量→全局向量（减去原点偏移）
        return self.coordinate_system.c2p(*out_coords.T) - origin

    def draw_lines(self) -> None:
        """核心方法：生成并绘制所有流线（基于ODE求解粒子运动轨迹）"""
        # 存储所有流线的列表
        lines = []

        # 遍历所有采样点（每个采样点对应一条流线）
        for coords in self.get_sample_coords():
            # 1. 求解ODE：计算粒子从当前采样点出发的运动轨迹（坐标系内坐标）
            solution_coords = ode_solution_points(
                self.func,          # 向量场函数（决定粒子运动方向）
                coords,             # 初始采样点（坐标系内）
                self.solution_time, # 模拟总时长
                self.dt             # 时间步长
            )
            # 2. 创建流线对象（VMobject）
            line = VMobject()
            # 3. 将轨迹从坐标系坐标转换为全局坐标，并设置为流线的平滑顶点
            line.set_points_smoothly(self.coordinate_system.c2p(*solution_coords.T))
            # 4. 存储流线的模拟时间（暂未启用）
            line.virtual_time = self.solution_time
            # 5. 将流线添加到列表
            lines.append(line)
        # 6. 将所有流线设置为当前对象的子图形
        self.set_submobjects(lines)

    def get_sample_coords(self):
        """生成流线的初始采样点（支持随机偏移，避免流线重叠）"""
        # 简化变量名：关联的坐标系
        cs = self.coordinate_system
        # 1. 生成坐标系内的均匀采样点
        sample_coords = get_sample_coords(cs, self.density)

        # 2. 计算采样点的随机偏移量（默认基于坐标系单位大小和密度）
        noise_factor = self.noise_factor
        if noise_factor is None:
            noise_factor = (cs.x_axis.get_unit_size() / self.density) * 0.5

        # 3. 生成最终采样点：重复n_repeats次 + 随机偏移（增加密度并避免重叠）
        return np.array([
            coords + noise_factor * np.random.random(coords.shape)
            for n in range(self.n_repeats)  # 重复生成n_repeats次
            for coords in sample_coords     # 遍历初始均匀采样点
        ])

    def init_style(self) -> None:
        """初始化所有流线的样式（颜色和宽度）"""
        # 1. 按向量大小着色（启用颜色映射）
        if self.color_by_magnitude:
            # 创建向量化的颜色映射函数：输入向量大小，输出RGB颜色
            values_to_rgbs = get_vectorized_rgb_gradient_function(
                *self.magnitude_range,  # 颜色映射的向量大小范围
                self.color_map,         # 颜色映射名称
            )
            cs = self.coordinate_system
            # 遍历每条流线，为其设置渐变颜色
            for line in self.submobjects:
                # 计算流线每个顶点处的向量大小
                norms = [
                    get_norm(self.func(cs.p2c(point)))  # 全局坐标→坐标系坐标→向量场函数→向量大小
                    for point in line.get_points()      # 遍历流线的所有顶点
                ]
                # 根据向量大小生成对应的RGB颜色
                rgbs = values_to_rgbs(norms)
                # 构建RGBA数组（添加透明度通道）
                rgbas = np.zeros((len(rgbs), 4))
                rgbas[:, :3] = rgbs          # 前3通道：RGB颜色
                rgbas[:, 3] = self.stroke_opacity  # 第4通道：透明度
                # 为流线设置颜色数组
                line.set_rgba_array(rgbas, "stroke_rgba")
        # 2. 统一颜色（不启用颜色映射）
        else:
            self.set_stroke(self.stroke_color, opacity=self.stroke_opacity)

        # 3. 设置流线宽度（支持渐变宽度）
        if self.taper_stroke_width:
            # 渐变宽度：两端0 → 中间self.stroke_width（通过列表指定渐变趋势）
            width = [0, self.stroke_width, 0]
        else:
            # 固定宽度：所有位置均为self.stroke_width
            width = self.stroke_width
        # 应用宽度设置
        self.set_stroke(width=width)


class AnimatedStreamLines(VGroup):
    """动态流线类，继承自VGroup，用于为StreamLines添加流动动画效果"""
    
    def __init__(
        self,
        stream_lines: StreamLines,  # 静态流线对象（StreamLines实例）
        lag_range: float = 4,       # 流线动画的时间延迟范围（随机延迟，避免同步）
        rate_multiple: float = 1.0, # 动画速率倍数（值越大动画越快）
        line_anim_config: dict = dict(
            rate_func=linear,       # 动画速率函数（默认线性）
            time_width=1.0,         # 动画时间宽度（控制流动效果范围）
        ),
        **kwargs                    # 传递给父类VGroup的参数
    ):
        # 调用父类VGroup的构造函数
        super().__init__(** kwargs)
        # 存储原始流线对象
        self.stream_lines = stream_lines

        # 为每条流线创建动画
        for line in stream_lines:
            # 创建"流动闪烁"动画：模拟流线的流动效果
            line.anim = VShowPassingFlash(
                line,  # 动画作用的流线
                # 动画运行时间 = 流线虚拟时间 / 速率倍数
                run_time=line.virtual_time / rate_multiple,
                **line_anim_config,  # 传入动画配置参数
            )
            # 初始化动画（设置起始状态）
            line.anim.begin()
            # 为每条流线设置随机初始时间（产生错落有致的动画效果）
            line.time = -lag_range * np.random.random()
            # 将动画关联的图形添加到当前组合中
            self.add(line.anim.mobject)

        # 添加更新器：每帧更新动画状态
        self.add_updater(lambda m, dt: m.update(dt))

    def update(self, dt: float = 0) -> None:
        """更新方法：每帧更新所有流线的动画状态"""
        # 简化变量名：原始流线对象
        stream_lines = self.stream_lines
        # 遍历每条流线，更新其动画进度
        for line in stream_lines:
            # 累加时间（当前帧的时间步长）
            line.time += dt
            # 计算调整后的时间：确保非负并在动画周期内循环
            adjusted_time = max(line.time, 0) % line.anim.run_time
            # 更新动画进度（范围0-1）
            line.anim.update(adjusted_time / line.anim.run_time)