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
    def __init__(
        self,
        # Vectorized function: Takes in an array of coordinates, returns an array of outputs.
        func: Callable[[VectArray], VectArray],
        # Typically a set of Axes or NumberPlane
        coordinate_system: CoordinateSystem,
        sample_coords: Optional[VectArray] = None,
        density: float = 2.0,
        magnitude_range: Optional[Tuple[float, float]] = None,
        color: Optional[ManimColor] = None,
        color_map_name: Optional[str] = "3b1b_colormap",
        color_map: Optional[Callable[[Sequence[float]], Vect4Array]] = None,
        stroke_opacity: float = 1.0,
        stroke_width: float = 3,
        tip_width_ratio: float = 4,
        tip_len_to_width: float = 0.01,
        max_vect_len: float | None = None,
        max_vect_len_to_step_size: float = 0.8,
        flat_stroke: bool = False,
        norm_to_opacity_func=None,  # TODO, check on this
        **kwargs
    ):
        self.func = func
        self.coordinate_system = coordinate_system
        self.stroke_width = stroke_width
        self.tip_width_ratio = tip_width_ratio
        self.tip_len_to_width = tip_len_to_width
        self.norm_to_opacity_func = norm_to_opacity_func

        # Search for sample_points
        if sample_coords is not None:
            self.sample_coords = sample_coords
        else:
            self.sample_coords = get_sample_coords(coordinate_system, density)
        self.update_sample_points()

        if max_vect_len is None:
            step_size = get_norm(self.sample_points[1] - self.sample_points[0])
            self.max_displayed_vect_len = max_vect_len_to_step_size * step_size
        else:
            self.max_displayed_vect_len = max_vect_len * coordinate_system.x_axis.get_unit_size()

        # Prepare the color map
        if magnitude_range is None:
            max_value = max(map(get_norm, func(self.sample_coords)))
            magnitude_range = (0, max_value)

        self.magnitude_range = magnitude_range

        if color is not None:
            self.color_map = None
        else:
            self.color_map = color_map or get_color_map(color_map_name)

        self.init_base_stroke_width_array(len(self.sample_coords))

        super().__init__(
            stroke_opacity=stroke_opacity,
            flat_stroke=flat_stroke,
            **kwargs
        )
        self.set_stroke(color, stroke_width)
        self.update_vectors()

    def init_points(self):
        n_samples = len(self.sample_coords)
        self.set_points(np.zeros((8 * n_samples - 1, 3)))
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
        to_corner = np.array([width / 2, height / 2, depth / 2])
        spacings = 1.0 / np.array([x_density, y_density, z_density])
        to_corner = spacings * (to_corner / spacings).astype(int)
        lower_corner = center - to_corner
        upper_corner = center + to_corner + spacings
        return cartesian_product(*(
            np.arange(low, high, space)
            for low, high, space in zip(lower_corner, upper_corner, spacings)
        ))

    def init_base_stroke_width_array(self, n_sample_points):
        arr = np.ones(8 * n_sample_points - 1)
        arr[4::8] = self.tip_width_ratio
        arr[5::8] = self.tip_width_ratio * 0.5
        arr[6::8] = 0
        arr[7::8] = 0
        self.base_stroke_width_array = arr

    def set_sample_coords(self, sample_coords: VectArray):
        self.sample_coords = sample_coords
        return self

    def set_stroke(self, color=None, width=None, opacity=None, behind=None, flat=None, recurse=True):
        super().set_stroke(color, None, opacity, behind, flat, recurse)
        if width is not None:
            self.set_stroke_width(float(width))
        return self

    def set_stroke_width(self, width: float):
        if self.get_num_points() > 0:
            self.get_stroke_widths()[:] = width * self.base_stroke_width_array
            self.stroke_width = width
        return self

    def update_sample_points(self):
        self.sample_points = self.coordinate_system.c2p(*self.sample_coords.T)

    def update_vectors(self):
        tip_width = self.tip_width_ratio * self.stroke_width
        tip_len = self.tip_len_to_width * tip_width

        # Outputs in the coordinate system
        outputs = self.func(self.sample_coords)
        output_norms = np.linalg.norm(outputs, axis=1)[:, np.newaxis]

        # Corresponding vector values in global coordinates
        out_vects = self.coordinate_system.c2p(*outputs.T) - self.coordinate_system.get_origin()
        out_vect_norms = np.linalg.norm(out_vects, axis=1)[:, np.newaxis]
        unit_outputs = np.zeros_like(out_vects)
        np.true_divide(out_vects, out_vect_norms, out=unit_outputs, where=(out_vect_norms > 0))

        # How long should the arrows be drawn, in global coordinates
        max_len = self.max_displayed_vect_len
        if max_len < np.inf:
            drawn_norms = max_len * np.tanh(out_vect_norms / max_len)
        else:
            drawn_norms = out_vect_norms

        # What's the distance from the base of an arrow to
        # the base of its head?
        dist_to_head_base = np.clip(drawn_norms - tip_len, 0, np.inf)  # Mixing units!

        # Set all points
        points = self.get_points()
        points[0::8] = self.sample_points
        points[2::8] = self.sample_points + dist_to_head_base * unit_outputs
        points[4::8] = points[2::8]
        points[6::8] = self.sample_points + drawn_norms * unit_outputs
        for i in (1, 3, 5):
            points[i::8] = 0.5 * (points[i - 1::8] + points[i + 1::8])
        points[7::8] = points[6:-1:8]

        # Adjust stroke widths
        width_arr = self.stroke_width * self.base_stroke_width_array
        width_scalars = np.clip(drawn_norms / tip_len, 0, 1)
        width_scalars = np.repeat(width_scalars, 8)[:-1]
        self.get_stroke_widths()[:] = width_scalars * width_arr

        # Potentially adjust opacity and color
        if self.color_map is not None:
            self.get_stroke_colors()  # Ensures the array is updated to appropriate length
            low, high = self.magnitude_range
            self.data['stroke_rgba'][:, :3] = self.color_map(
                inverse_interpolate(low, high, np.repeat(output_norms, 8)[:-1])
            )[:, :3]

        if self.norm_to_opacity_func is not None:
            self.get_stroke_opacities()[:] = self.norm_to_opacity_func(
                np.repeat(output_norms, 8)[:-1]
            )

        self.note_changed_data()
        return self


class TimeVaryingVectorField(VectorField):
    def __init__(
        self,
        # Takes in an array of points and a float for time
        time_func: Callable[[VectArray, float], VectArray],
        coordinate_system: CoordinateSystem,
        **kwargs
    ):
        self.time = 0

        def func(coords):
            return time_func(coords, self.time)

        super().__init__(func, coordinate_system, **kwargs)
        self.add_updater(lambda m, dt: m.increment_time(dt))
        self.always.update_vectors()

    def increment_time(self, dt):
        self.time += dt


class StreamLines(VGroup):
    def __init__(
        self,
        func: Callable[[VectArray], VectArray],
        coordinate_system: CoordinateSystem,
        density: float = 1.0,
        n_repeats: int = 1,
        noise_factor: float | None = None,
        # Config for drawing lines
        solution_time: float = 3,
        dt: float = 0.05,
        arc_len: float = 3,
        max_time_steps: int = 200,
        n_samples_per_line: int = 10,
        cutoff_norm: float = 15,
        # Style info
        stroke_width: float = 1.0,
        stroke_color: ManimColor = DEFAULT_MOBJECT_COLOR,
        stroke_opacity: float = 1,
        color_by_magnitude: bool = True,
        magnitude_range: Tuple[float, float] = (0, 2.0),
        taper_stroke_width: bool = False,
        color_map: str = "3b1b_colormap",
        **kwargs
    ):
        super().__init__(**kwargs)
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

        self.draw_lines()
        self.init_style()

    def point_func(self, points: Vect3Array) -> Vect3:
        in_coords = np.array(self.coordinate_system.p2c(points)).T
        out_coords = self.func(in_coords)
        origin = self.coordinate_system.get_origin()
        return self.coordinate_system.c2p(*out_coords.T) - origin

    def draw_lines(self) -> None:
        lines = []

        # Todo, it feels like coordinate system should just have
        # the ODE solver built into it, no?
        lines = []
        for coords in self.get_sample_coords():
            solution_coords = ode_solution_points(self.func, coords, self.solution_time, self.dt)
            line = VMobject()
            line.set_points_smoothly(self.coordinate_system.c2p(*solution_coords.T))
            # TODO, account for arc length somehow?
            line.virtual_time = self.solution_time
            lines.append(line)
        self.set_submobjects(lines)

    def get_sample_coords(self):
        cs = self.coordinate_system
        sample_coords = get_sample_coords(cs, self.density)

        noise_factor = self.noise_factor
        if noise_factor is None:
            noise_factor = (cs.x_axis.get_unit_size() / self.density) * 0.5

        return np.array([
            coords + noise_factor * np.random.random(coords.shape)
            for n in range(self.n_repeats)
            for coords in sample_coords
        ])

    def init_style(self) -> None:
        if self.color_by_magnitude:
            values_to_rgbs = get_vectorized_rgb_gradient_function(
                *self.magnitude_range, self.color_map,
            )
            cs = self.coordinate_system
            for line in self.submobjects:
                norms = [
                    get_norm(self.func(cs.p2c(point)))
                    for point in line.get_points()
                ]
                rgbs = values_to_rgbs(norms)
                rgbas = np.zeros((len(rgbs), 4))
                rgbas[:, :3] = rgbs
                rgbas[:, 3] = self.stroke_opacity
                line.set_rgba_array(rgbas, "stroke_rgba")
        else:
            self.set_stroke(self.stroke_color, opacity=self.stroke_opacity)

        if self.taper_stroke_width:
            width = [0, self.stroke_width, 0]
        else:
            width = self.stroke_width
        self.set_stroke(width=width)


class AnimatedStreamLines(VGroup):
    def __init__(
        self,
        stream_lines: StreamLines,
        lag_range: float = 4,
        rate_multiple: float = 1.0,
        line_anim_config: dict = dict(
            rate_func=linear,
            time_width=1.0,
        ),
        **kwargs
    ):
        super().__init__(**kwargs)
        self.stream_lines = stream_lines

        for line in stream_lines:
            line.anim = VShowPassingFlash(
                line,
                run_time=line.virtual_time / rate_multiple,
                **line_anim_config,
            )
            line.anim.begin()
            line.time = -lag_range * np.random.random()
            self.add(line.anim.mobject)

        self.add_updater(lambda m, dt: m.update(dt))

    def update(self, dt: float = 0) -> None:
        stream_lines = self.stream_lines
        for line in stream_lines:
            line.time += dt
            adjusted_time = max(line.time, 0) % line.anim.run_time
            line.anim.update(adjusted_time / line.anim.run_time)
