# 从 __future__ 导入 annotations，支持 Python 3.7+ 的前向类型注解
from __future__ import annotations

# 导入数学相关模块
import math
# 导入警告模块，用于处理警告信息
import warnings

# 导入 numpy 并简写为 np，用于数值计算和数组操作
import numpy as np
# 从 scipy 的空间变换模块导入 Rotation，用于处理旋转操作
from scipy.spatial.transform import Rotation

# 从 manimlib 的常量模块导入角度单位常量（度和弧度）
from manimlib.constants import DEG, RADIANS
# 导入帧形状常量（用于定义场景的帧尺寸）
from manimlib.constants import FRAME_SHAPE
# 导入方向向量常量（下、左、原点、外、右、上）
from manimlib.constants import DOWN, LEFT, ORIGIN, OUT, RIGHT, UP
# 导入圆周率常量
from manimlib.constants import PI
# 导入基础图形对象类 Mobject
from manimlib.mobject.mobject import Mobject
# 从空间操作工具模块导入归一化函数
from manimlib.utils.space_ops import normalize
# 从简单函数工具模块导入剪辑函数（用于限制值的范围）
from manimlib.utils.simple_functions import clip

# 导入类型检查相关模块
from typing import TYPE_CHECKING

# 如果是类型检查阶段（而非实际运行）
if TYPE_CHECKING:
    # 从 manimlib 的类型模块导入三维向量类型 Vect3（仅用于类型注解）
    from manimlib.typing import Vect3


# 定义 CameraFrame 类，继承自基础图形类 Mobject，用于控制相机的帧参数与视角
class CameraFrame(Mobject):
    # 构造方法，初始化相机帧的基础属性
    def __init__(
        self,
        frame_shape: tuple[float, float] = FRAME_SHAPE,  # 相机帧的宽高比，默认使用全局帧形状常量
        center_point: Vect3 = ORIGIN,  # 相机帧的中心点，默认在坐标原点
        fovy: float = 45 * DEG,  # y轴方向的视场角，默认45度（转为弧度值）
        euler_axes: str = "zxz",  # 欧拉角的旋转轴顺序，默认"zxz"（先绕z轴、再x轴、最后z轴）
        z_index=-1,  # 渲染层级，-1确保相机帧在场景中优先渲染
        **kwargs,  # 传递给父类 Mobject 的其他关键字参数
    ):
        # 调用父类构造方法，设置渲染层级和其他参数
        super().__init__(z_index=z_index, **kwargs)

        # 初始化相机 uniforms（着色器统一变量）：姿态用四元数存储，初始为单位四元数（无旋转）
        self.uniforms["orientation"] = Rotation.identity().as_quat()
        # 初始化 uniforms 中的视场角，存储y轴方向的视场角
        self.uniforms["fovy"] = fovy

        # 存储相机的默认姿态（初始无旋转）
        self.default_orientation = Rotation.identity()
        # 初始化视图矩阵（4x4矩阵，用于坐标变换），初始为单位矩阵
        self.view_matrix = np.identity(4)
        # 存储4x4单位矩阵，用于后续矩阵运算复用
        self.id4x4 = np.identity(4)
        # 相机位置，初始设为OUT方向（垂直屏幕向外），后续会通过set_points更新
        self.camera_location = OUT
        # 存储欧拉角的旋转轴顺序
        self.euler_axes = euler_axes

        # 为相机帧设置特征点：原点（中心点）、左、右、下、上，共5个点
        self.set_points(np.array([ORIGIN, LEFT, RIGHT, DOWN, UP]))
        # 按帧形状设置宽度，允许拉伸以匹配目标宽高比
        self.set_width(frame_shape[0], stretch=True)
        # 按帧形状设置高度，允许拉伸以匹配目标宽高比
        self.set_height(frame_shape[1], stretch=True)
        # 将相机帧移动到指定的中心点
        self.move_to(center_point)

    # 设置相机的姿态（旋转状态），接收Rotation对象并转为四元数存储
    def set_orientation(self, rotation: Rotation):
        # 将Rotation对象的四元数赋值给uniforms的orientation（用[:]确保修改原数组）
        self.uniforms["orientation"][:] = rotation.as_quat()
        return self  # 返回自身以支持链式调用

    # 获取当前相机的姿态，从uniforms的四元数转为Rotation对象
    def get_orientation(self):
        return Rotation.from_quat(self.uniforms["orientation"])

    # 将当前姿态设为默认姿态，后续可通过to_default_state恢复
    def make_orientation_default(self):
        self.default_orientation = self.get_orientation()
        return self  # 返回自身以支持链式调用

    # 恢复相机到默认状态：重置形状、中心点和姿态
    def to_default_state(self):
        self.set_shape(*FRAME_SHAPE)  # 重置帧形状为全局默认
        self.center()  # 重置中心点到原点
        self.set_orientation(self.default_orientation)  # 恢复默认姿态
        return self  # 返回自身以支持链式调用

    # 获取相机当前的欧拉角（按指定的旋转轴顺序）
    def get_euler_angles(self) -> np.ndarray:
        # 先获取当前姿态的Rotation对象
        orientation = self.get_orientation()
        # 若姿态为单位四元数（无旋转），直接返回全零欧拉角
        if np.isclose(orientation.as_quat(), [0, 0, 0, 1]).all():
            return np.zeros(3)
        # 捕获并忽略scipy在特定旋转下发出的UserWarning（如接近奇异点时）
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)  # 忽略UserWarning
            # 从Rotation对象获取欧拉角，反转顺序以匹配Manim的角度定义
            angles = orientation.as_euler(self.euler_axes)[::-1]
        # 处理"zxz"轴顺序下的万向节锁问题（旋转轴重合导致自由度丢失）
        if self.euler_axes == "zxz":
            # 若第二个角度（phi）接近0，合并第一和第三个角度，重置第三个角度为0
            if np.isclose(angles[1], 0, atol=1e-2):
                angles[0] = angles[0] + angles[2]
                angles[2] = 0
            # 若第二个角度接近PI（180度），调整第一和第三个角度，重置第三个角度为0
            if np.isclose(angles[1], PI, atol=1e-2):
                angles[0] = angles[0] - angles[2]
                angles[2] = 0
        return angles  # 返回处理后的欧拉角（theta, phi, gamma）

    # 获取欧拉角中的第一个角度（theta，通常对应绕z轴旋转）
    def get_theta(self):
        return self.get_euler_angles()[0]

    # 获取欧拉角中的第二个角度（phi，通常对应绕x轴旋转）
    def get_phi(self):
        return self.get_euler_angles()[1]

    # 获取欧拉角中的第三个角度（gamma，通常对应绕z轴二次旋转）
    def get_gamma(self):
        return self.get_euler_angles()[2]

    # 计算相机的缩放比例：当前高度与默认帧高度的比值
    def get_scale(self):
        return self.get_height() / FRAME_SHAPE[1]

    # 获取相机旋转矩阵的转置（即逆矩阵，因旋转矩阵是正交矩阵，逆=转置）
    def get_inverse_camera_rotation_matrix(self):
        return self.get_orientation().as_matrix().T

    # 获取视图矩阵（4x4），用于将世界坐标转换为相机本地坐标
    def get_view_matrix(self, refresh=False):
        """
        返回4x4仿射变换矩阵，用于将世界空间中的点
        映射到相机的内部坐标系统
        """
        # 若相机数据（位置、姿态、缩放）有变化，重新计算视图矩阵
        if self._data_has_changed:
            # 初始化平移矩阵（单位矩阵基础上修改平移部分）
            shift = self.id4x4.copy()
            # 初始化旋转矩阵（单位矩阵基础上修改旋转部分）
            rotation = self.id4x4.copy()

            # 获取当前相机的缩放比例
            scale = self.get_scale()
            # 平移矩阵：将世界坐标原点平移到相机中心点的反方向（消除相机位置偏移）
            shift[:3, 3] = -self.get_center()
            # 旋转矩阵：上三角3x3部分设为相机旋转的逆矩阵（对齐相机坐标系）
            rotation[:3, :3] = self.get_inverse_camera_rotation_matrix()
            # 矩阵乘法：旋转矩阵 × 平移矩阵，结果存入视图矩阵（out参数避免创建新数组）
            np.dot(rotation, shift, out=self.view_matrix)
            # 若缩放比例有效（>0），对视图矩阵的前3行（空间坐标）应用缩放
            if scale > 0:
                self.view_matrix[:3, :4] /= scale

        return self.view_matrix  # 返回计算后的视图矩阵

    # 获取视图矩阵的逆矩阵，用于将相机本地坐标转换回世界坐标
    def get_inv_view_matrix(self):
        return np.linalg.inv(self.get_view_matrix())

    # 重写interpolate方法，添加@Mobject.affects_data装饰器
    # 标记该方法会修改相机数据，触发视图矩阵等参数的重新计算
    @Mobject.affects_data
    def interpolate(self, *args, **kwargs):
        super().interpolate(*args, **kwargs)  # 调用父类的插值方法

    # 重写rotate方法，实现相机绕指定轴的旋转，添加@Mobject.affects_data装饰器
    @Mobject.affects_data
    def rotate(self, angle: float, axis: np.ndarray = OUT, **kwargs):
        # 根据旋转角度和轴创建Rotation对象（先归一化轴向量确保方向正确）
        rot = Rotation.from_rotvec(angle * normalize(axis))
        # 新姿态 = 新旋转 × 当前姿态（右乘，实现相对旋转）
        self.set_orientation(rot * self.get_orientation())
        return self  # 返回自身以支持链式调用

    # 设置相机的欧拉角，支持单独修改某一个角度
    def set_euler_angles(
        self,
        theta: float | None = None,  # 第一个欧拉角（theta），None表示不修改
        phi: float | None = None,    # 第二个欧拉角（phi），None表示不修改
        gamma: float | None = None,  # 第三个欧拉角（gamma），None表示不修改
        units: float = RADIANS       # 角度单位，默认弧度（RADIANS），可选角度（DEG）
    ):
        # 先获取当前的欧拉角（作为基础值）
        eulers = self.get_euler_angles()  # 顺序：theta, phi, gamma
        # 遍历三个角度，若传入非None值，则用新值（转换为指定单位）更新
        for i, var in enumerate([theta, phi, gamma]):
            if var is not None:
                eulers[i] = var * units
        # 若所有角度都为0，直接设为单位旋转（无旋转）
        if all(eulers == 0):
            rot = Rotation.identity()
        else:
            # 从欧拉角创建Rotation对象，注意反转角度顺序以匹配scipy的轴顺序要求
            rot = Rotation.from_euler(self.euler_axes, eulers[::-1])
        # 应用新姿态
        self.set_orientation(rot)
        return self  # 返回自身以支持链式调用

    # 增量修改相机的欧拉角（在当前角度基础上增加指定值）
    def increment_euler_angles(
        self,
        dtheta: float = 0,  # theta角的增量，默认0
        dphi: float = 0,    # phi角的增量，默认0
        dgamma: float = 0,  # gamma角的增量，默认0
        units: float = RADIANS  # 角度单位，默认弧度
    ):
        # 获取当前欧拉角
        angles = self.get_euler_angles()
        # 计算新角度：当前角度 + 增量（转换为指定单位）
        new_angles = angles + np.array([dtheta, dphi, dgamma]) * units

        # 限制phi角的范围（避免相机翻转或视角异常）
        if self.euler_axes == "zxz":
            # "zxz"轴顺序下，phi角范围限制为0到PI（0到180度）
            new_angles[1] = clip(new_angles[1], 0, PI)
        elif self.euler_axes == "zxy":
            # "zxy"轴顺序下，phi角范围限制为-PI/2到PI/2（-90到90度）
            new_angles[1] = clip(new_angles[1], -PI / 2, PI / 2)

        # 根据新角度创建Rotation对象，反转顺序以匹配scipy要求
        new_rot = Rotation.from_euler(self.euler_axes, new_angles[::-1])
        # 应用新姿态
        self.set_orientation(new_rot)
        return self  # 返回自身以支持链式调用

    # 设置欧拉角的旋转轴顺序（如"zxz"、"zxy"）
    def set_euler_axes(self, seq: str):
        self.euler_axes = seq

    # 快捷方法：设置欧拉角（默认角度单位为度）、中心点和高度
    def reorient(
        self,
        theta_degrees: float | None = None,  # theta角（度），None不修改
        phi_degrees: float | None = None,    # phi角（度），None不修改
        gamma_degrees: float | None = None,  # gamma角（度），None不修改
        center: Vect3 | tuple[float, float, float] | None = None,  # 新中心点，None不修改
        height: float | None = None  # 新高度，None不修改
    ):
        """
        set_euler_angles的快捷方法，默认接收角度单位为度
        """
        # 调用set_euler_angles，指定单位为度（DEG）
        self.set_euler_angles(theta_degrees, phi_degrees, gamma_degrees, units=DEG)
        # 若指定了新中心点，移动相机帧到该点
        if center is not None:
            self.move_to(np.array(center))
        # 若指定了新高度，设置相机帧的高度
        if height is not None:
            self.set_height(height)
        return self  # 返回自身以支持链式调用

    # 单独设置theta角（调用set_euler_angles）
    def set_theta(self, theta: float):
        return self.set_euler_angles(theta=theta)

    # 单独设置phi角（调用set_euler_angles）
    def set_phi(self, phi: float):
        return self.set_euler_angles(phi=phi)

    # 单独设置gamma角（调用set_euler_angles）
    def set_gamma(self, gamma: float):
        return self.set_euler_angles(gamma=gamma)

    # 增量修改theta角（调用increment_euler_angles）
    def increment_theta(self, dtheta: float, units=RADIANS):
        self.increment_euler_angles(dtheta=dtheta, units=units)
        return self  # 返回自身以支持链式调用

    # 增量修改phi角（调用increment_euler_angles）
    def increment_phi(self, dphi: float, units=RADIANS):
        self.increment_euler_angles(dphi=dphi, units=units)
        return self  # 返回自身以支持链式调用

    # 增量修改gamma角（调用increment_euler_angles）
    def increment_gamma(self, dgamma: float, units=RADIANS):
        self.increment_euler_angles(dgamma=dgamma, units=units)
        return self  # 返回自身以支持链式调用

    # 为相机添加环绕旋转更新器：按指定角速度持续旋转theta角
    def add_ambient_rotation(self, angular_speed=1 * DEG):
        # 添加更新器：每帧根据时间增量dt更新theta角（角速度×时间=角度增量）
        self.add_updater(lambda m, dt: m.increment_theta(angular_speed * dt))
        return self  # 返回自身以支持链式调用

    # 设置相机的焦距（根据焦距反推视场角），添加@Mobject.affects_data装饰器
    @Mobject.affects_data
    def set_focal_distance(self, focal_distance: float):
        # 公式推导：fovy = 2 * arctan(0.5*帧高度 / 焦距)，确保视场角与焦距匹配
        self.uniforms["fovy"] = 2 * math.atan(0.5 * self.get_height() / focal_distance)
        return self  # 返回自身以支持链式调用

    # 直接设置相机的视场角（y轴方向），添加@Mobject.affects_data装饰器
    @Mobject.affects_data
    def set_field_of_view(self, field_of_view: float):
        self.uniforms["fovy"] = field_of_view
        return self  # 返回自身以支持链式调用

    # 获取相机帧的当前形状（宽，高）
    def get_shape(self):
        return (self.get_width(), self.get_height())

    # 获取相机帧的宽高比（宽度/高度）
    def get_aspect_ratio(self):
        width, height = self.get_shape()
        return width / height

    # 获取相机帧的中心点（依赖set_points设置的第一个点为原点）
    def get_center(self) -> np.ndarray:
        # 假设set_points设置的第一个点是中心点
        return self.get_points()[0]

    # 获取相机帧的宽度（右特征点x坐标 - 左特征点x坐标）
    def get_width(self) -> float:
        points = self.get_points()
        return points[2, 0] - points[1, 0]  # 第3个点（RIGHT）x - 第2个点（LEFT）x

    # 获取相机帧的高度（上特征点y坐标 - 下特征点y坐标）
    def get_height(self) -> float:
        points = self.get_points()
        return points[4, 1] - points[3, 1]  # 第5个点（UP）y - 第4个点（DOWN）y

    # 根据当前视场角和帧高度计算焦距
    def get_focal_distance(self) -> float:
        # 公式推导：焦距 = 0.5*帧高度 / tan(0.5*视场角)
        return 0.5 * self.get_height() / math.tan(0.5 * self.uniforms["fovy"])

    # 获取当前的视场角（y轴方向）
    def get_field_of_view(self) -> float:
        return self.uniforms["fovy"]

    # 计算相机在世界空间中的隐含位置（基于姿态和焦距）
    def get_implied_camera_location(self) -> np.ndarray:
        # 若相机数据有变化，重新计算相机位置
        if self._data_has_changed:
            # 获取相机旋转逆矩阵的第三行（对应相机的"向前"方向）
            to_camera = self.get_inverse_camera_rotation_matrix()[2]
            # 获取当前焦距
            dist = self.get_focal_distance()
            # 相机位置 = 相机中心点 + 焦距 × 向前方向（确保相机在帧前方）
            self.camera_location = self.get_center() + dist * to_camera
        return self.camera_location  # 返回计算后的相机位置

    # 将世界空间中的点转换为相机固定帧坐标（相对/绝对模式）
    def to_fixed_frame_point(self, point: Vect3, relative: bool = False):
        # 获取视图矩阵
        view = self.get_view_matrix()
        # 构造4D点：相对模式下w=0（仅方向，无平移），绝对模式下w=1（含平移）
        point4d = [*point, 0 if relative else 1]
        # 矩阵乘法：4D点 × 视图矩阵转置（适配行向量），取前3维作为结果
        return np.dot(point4d, view.T)[:3]

    # 将相机固定帧坐标转换为世界空间中的点（相对/绝对模式）
    def from_fixed_frame_point(self, point: Vect3, relative: bool = False):
        # 获取视图矩阵的逆矩阵
        inv_view = self.get_inv_view_matrix()
        # 构造4D点：相对模式下w=0，绝对模式下w=1
        point4d = [*point, 0 if relative else 1]
        # 矩阵乘法：4D点 × 逆视图矩阵转置，取前3维作为结果
        return np.dot(point4d, inv_view.T)[:3]