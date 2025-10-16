# 从 __future__ 导入 annotations，支持 Python 3.7+ 的前向类型注解（如在类型提示中直接使用未定义的类名）
from __future__ import annotations

# 导入 moderngl 库，用于简化 OpenGL 现代版本（3.3+）的渲染操作
import moderngl
# 导入 numpy 并简写为 np，用于数值计算、数组处理（如存储顶点数据、颜色数据）
import numpy as np
# 导入 OpenGL.GL 并简写为 gl，用于直接调用 OpenGL 底层函数（兼容旧版接口）
import OpenGL.GL as gl
# 从 PIL（Python Imaging Library）导入 Image 类，用于图像的读取、保存与像素操作
from PIL import Image

# 从 manimlib 的相机模块导入 CameraFrame 类，该类封装了相机的帧参数、姿态与视角控制
from manimlib.camera.camera_frame import CameraFrame
# 从 manimlib 常量模块导入默认黑色（用于背景色等）
from manimlib.constants import BLACK
# 导入默认分辨率（如 (1920, 1080)，定义渲染图像的像素尺寸）
from manimlib.constants import DEFAULT_RESOLUTION
# 导入默认帧高度（相机可视区域的高度，用于计算视角与缩放）
from manimlib.constants import FRAME_HEIGHT
# 导入默认帧宽度（相机可视区域的宽度，与高度共同定义帧的宽高比）
from manimlib.constants import FRAME_WIDTH
# 从 manimlib 的图形对象模块导入基础图形类 Mobject（所有可渲染对象的父类）
from manimlib.mobject.mobject import Mobject
# 导入 Point 类（表示单个点的图形对象，用于标记位置等）
from manimlib.mobject.mobject import Point
# 从颜色工具模块导入 color_to_rgba 函数，用于将颜色（如字符串、RGB值）转换为 RGBA 格式（含透明度）
from manimlib.utils.color import color_to_rgba

# 导入类型检查相关模块，用于支持条件性的类型注解（仅在类型检查时生效，不影响运行）
from typing import TYPE_CHECKING

# 判断是否处于类型检查阶段（如使用 mypy 等工具检查代码时）
if TYPE_CHECKING:
    # 从 typing 模块导入 Optional 类型，用于表示参数/变量可为指定类型或 None
    from typing import Optional
    # 从 manimlib 的类型模块导入 ManimColor（所有支持的颜色类型的统称）和 Vect3（三维向量类型）
    from manimlib.typing import ManimColor, Vect3
    # 导入 Window 类（窗口管理类，用于创建渲染窗口、处理用户输入），避免循环导入
    from manimlib.window import Window


# 定义 Camera 类，负责管理渲染上下文、帧缓冲、相机参数及图像捕获
class Camera(object):
    # 构造方法，初始化相机的核心参数与渲染相关组件
    def __init__(
        self,
        window: Optional[Window] = None,  # 关联的窗口对象（None表示无窗口，仅后台渲染）
        background_image: Optional[str] = None,  # 背景图像路径（None表示无背景图）
        frame_config: dict = dict(),  # 传递给CameraFrame的配置参数（如帧形状、视场角）
        resolution=DEFAULT_RESOLUTION,  # 渲染分辨率（如(1920,1080)），决定图像像素尺寸
        fps: int = 30,  # 渲染帧率（帧/秒），用于视频生成
        background_color: ManimColor = BLACK,  # 背景色，默认黑色
        background_opacity: float = 1.0,  # 背景透明度（0=完全透明，1=完全不透明）
        max_allowable_norm: float = FRAME_WIDTH,  # 向量图形顶点的最大允许模长，超限时会缩放
        image_mode: str = "RGBA",  # 图像模式，默认RGBA（含透明度通道）
        n_channels: int = 4,  # 图像通道数（RGBA对应4通道）
        pixel_array_dtype: type = np.uint8,  # 像素数组的数据类型（默认uint8，取值0-255）
        light_source_position: Vect3 = np.array([-10, 10, 10]),  # 光源位置（3D场景用）
        samples: int = 0,  # 多重采样数（抗锯齿用，0=无采样，3D场景建议设>0如4）
    ):
        # 存储窗口对象
        self.window = window
        # 存储背景图像路径
        self.background_image = background_image
        # 存储默认渲染分辨率（后续可能重命名，暂用此名）
        self.default_pixel_shape = resolution
        # 存储帧率
        self.fps = fps
        # 存储顶点最大允许模长
        self.max_allowable_norm = max_allowable_norm
        # 存储图像模式
        self.image_mode = image_mode
        # 存储图像通道数
        self.n_channels = n_channels
        # 存储像素数组数据类型
        self.pixel_array_dtype = pixel_array_dtype
        # 存储光源位置
        self.light_source_position = light_source_position
        # 存储多重采样数
        self.samples = samples

        # 计算RGB通道的最大值（如uint8对应255）
        self.rgb_max_val: float = np.iinfo(self.pixel_array_dtype).max
        # 将背景色+透明度转换为RGBA列表（每个值0-1）
        self.background_rgba: list[float] = list(color_to_rgba(
            background_color, background_opacity
        ))
        # 存储着色器统一变量（后续传递给OpenGL程序）
        self.uniforms = dict()
        # 初始化CameraFrame（相机帧，控制视角、姿态）
        self.init_frame(**frame_config)
        # 初始化OpenGL上下文（moderngl核心对象）
        self.init_context()
        # 初始化帧缓冲（FBO，用于存储渲染结果）
        self.init_fbo()
        # 初始化光源（3D场景光照计算用）
        self.init_light_source()

    # 初始化CameraFrame，创建相机帧对象
    def init_frame(self, **config) -> None:
        self.frame = CameraFrame(**config)

    # 初始化OpenGL上下文（moderngl.Context），配置渲染状态
    def init_context(self) -> None:
        # 若无窗口，创建独立上下文（用于后台渲染，无界面）
        if self.window is None:
            self.ctx: moderngl.Context = moderngl.create_standalone_context()
        # 若有窗口，使用窗口的上下文（用于实时显示）
        else:
            self.ctx: moderngl.Context = self.window.ctx

        # 启用"点大小控制"（允许通过着色器设置点的像素大小）
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        # 启用"混合模式"（处理透明物体的叠加渲染）
        self.ctx.enable(moderngl.BLEND)

    # 初始化帧缓冲（FBO），创建用于文件输出和绘制的缓冲对象
    def init_fbo(self) -> None:
        # 创建用于文件（图像/视频）输出的FBO，支持多重采样（抗锯齿）
        self.fbo_for_files = self.get_fbo(self.samples)

        # 创建用于绘制的FBO，禁用多重采样（用于快速绘制，如实时预览）
        self.draw_fbo = self.get_fbo(samples=0)

        # 若无窗口，窗口FBO设为None，默认使用文件输出FBO
        if self.window is None:
            self.window_fbo = None
            self.fbo = self.fbo_for_files
        # 若有窗口，获取窗口的默认FBO，默认使用窗口FBO
        else:
            self.window_fbo = self.ctx.detect_framebuffer()
            self.fbo = self.window_fbo

        # 激活当前使用的FBO（后续渲染操作将写入此FBO）
        self.fbo.use()

    # 初始化光源，创建表示光源位置的Point对象
    def init_light_source(self) -> None:
        self.light_source = Point(self.light_source_position)

    # 切换是否使用窗口FBO（用于实时显示与后台渲染的切换）
    def use_window_fbo(self, use: bool = True):
        # 断言窗口对象存在（否则无法切换到窗口FBO）
        assert self.window is not None
        # 若use=True，切换到窗口FBO；否则切换到文件输出FBO
        if use:
            self.fbo = self.window_fbo
        else:
            self.fbo = self.fbo_for_files

    # ------------------------------
    # 与帧缓冲（FBO）相关的方法
    # ------------------------------
    # 创建并返回帧缓冲（FBO），包含颜色附件和深度附件
    def get_fbo(
        self,
        samples: int = 0  # 该FBO的多重采样数
    ) -> moderngl.Framebuffer:
        return self.ctx.framebuffer(
            # 颜色附件：创建纹理，用于存储RGB/RGBA颜色数据
            color_attachments=self.ctx.texture(
                self.default_pixel_shape,  # 纹理尺寸（与渲染分辨率一致）
                components=self.n_channels,  # 纹理通道数（与图像通道数一致）
                samples=samples,  # 纹理的多重采样数
            ),
            # 深度附件：创建深度渲染缓冲，用于3D场景的深度测试
            depth_attachment=self.ctx.depth_renderbuffer(
                self.default_pixel_shape,  # 深度缓冲尺寸（与渲染分辨率一致）
                samples=samples  # 深度缓冲的多重采样数
            )
        )

    # 清空当前FBO的颜色和深度缓冲，填充背景色
    def clear(self) -> None:
        # 清空当前FBO，使用背景RGBA色
        self.fbo.clear(*self.background_rgba)
        # 若有窗口，同时清空窗口的FBO
        if self.window:
            self.window.clear(*self.background_rgba)

    # 将源FBO的内容复制到目标FBO（使用OpenGL的Blit操作）
    def blit(self, src_fbo, dst_fbo):
        """
        使用Blit操作在FBO之间复制像素块
        """
        # 绑定源FBO为"读取FBO"（从该FBO读取像素）
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, src_fbo.glo)
        # 绑定目标FBO为"绘制FBO"（将像素写入该FBO）
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, dst_fbo.glo)
        # 执行Blit：复制颜色缓冲，使用线性插值（抗锯齿）
        gl.glBlitFramebuffer(
            *src_fbo.viewport,  # 源FBO的视口（复制区域）
            *dst_fbo.viewport,  # 目标FBO的视口（粘贴区域）
            gl.GL_COLOR_BUFFER_BIT,  # 复制颜色缓冲
            gl.GL_LINEAR  # 缩放时使用线性插值
        )

    # 从当前FBO获取原始像素数据（字节流）
    def get_raw_fbo_data(self, dtype: str = 'f1') -> bytes:
        # 将当前FBO的内容复制到draw_fbo（确保无多重采样，数据格式统一）
        self.blit(self.fbo, self.draw_fbo)
        # 从draw_fbo读取像素数据，返回字节流
        return self.draw_fbo.read(
            viewport=self.draw_fbo.viewport,  # 读取的视口（默认全尺寸）
            components=self.n_channels,  # 读取的通道数
            dtype=dtype,  # 数据类型（如'f1'=float32）
        )

    # 从当前FBO获取PIL.Image对象（可直接保存为图片文件）
    def get_image(self) -> Image.Image:
        return Image.frombytes(
            'RGBA',  # 图像模式（与Camera的image_mode对应）
            self.get_pixel_shape(),  # 图像尺寸（像素宽×高）
            self.get_raw_fbo_data(),  # 原始像素字节流
            'raw',  # 数据格式标识
            'RGBA',  # 输入数据的通道顺序
            0,  # 每行数据的字节间隔（0=无间隔）
            -1  # 垂直方向翻转（因OpenGL纹理原点在左下角，PIL在左上角）
        )

    # 从当前FBO获取numpy像素数组（用于数值处理）
    def get_pixel_array(self) -> np.ndarray:
        # 获取float32格式的原始像素数据
        raw = self.get_raw_fbo_data(dtype='f4')
        # 将字节流转换为float32的一维数组
        flat_arr = np.frombuffer(raw, dtype='f4')
        # 重塑为三维数组：(高度, 宽度, 通道数)，并垂直翻转（匹配PIL坐标）
        arr = flat_arr.reshape([*reversed(self.draw_fbo.size), self.n_channels])
        arr = arr[::-1]
        # 将float32值（0-1）转换为目标数据类型（如uint8的0-255）
        return (self.rgb_max_val * arr).astype(self.pixel_array_dtype)

    # 获取当前FBO的纹理对象（可能用于后续渲染复用，暂保留该方法）
    def get_texture(self) -> moderngl.Texture:
        texture = self.ctx.texture(
            size=self.fbo.size,  # 纹理尺寸（与FBO一致）
            components=4,  # 纹理通道数（固定为4，RGBA）
            data=self.get_raw_fbo_data(),  # 纹理数据（来自FBO）
            dtype='f4'  # 数据类型（float32）
        )
        return texture

    # ------------------------------
    # 获取相机属性的方法
    # ------------------------------
    # 计算每个像素对应的世界空间宽度（用于像素与世界坐标的转换）
    def get_pixel_size(self) -> float:
        return self.frame.get_width() / self.get_pixel_shape()[0]

    # 获取当前FBO的像素尺寸（宽×高）
    def get_pixel_shape(self) -> tuple[int, int]:
        return self.fbo.size

    # 获取当前FBO的像素宽度
    def get_pixel_width(self) -> int:
        return self.get_pixel_shape()[0]

    # 获取当前FBO的像素高度
    def get_pixel_height(self) -> int:
        return self.get_pixel_shape()[1]

    # 获取当前FBO的宽高比（宽度/高度）
    def get_aspect_ratio(self):
        pw, ph = self.get_pixel_shape()
        return pw / ph

    # 获取相机帧的高度（世界空间中的高度）
    def get_frame_height(self) -> float:
        return self.frame.get_height()

    # 获取相机帧的宽度（世界空间中的宽度）
    def get_frame_width(self) -> float:
        return self.frame.get_width()

    # 获取相机帧的形状（宽×高，世界空间）
    def get_frame_shape(self) -> tuple[float, float]:
        return (self.get_frame_width(), self.get_frame_height())

    # 获取相机帧的中心点（世界空间坐标）
    def get_frame_center(self) -> np.ndarray:
        return self.frame.get_center()

    # 获取相机在世界空间中的隐含位置（基于帧的姿态和焦距）
    def get_location(self) -> tuple[float, float, float]:
        return self.frame.get_implied_camera_location()

    # 调整相机帧的形状，使其宽高比与像素宽高比匹配
    def resize_frame_shape(self, fixed_dimension: bool = False) -> None:
        """
        调整帧形状以匹配像素宽高比，fixed_dimension决定固定哪个维度：
        - fixed_dimension=False：固定帧宽度，调整高度
        - fixed_dimension=True：固定帧高度，调整宽度
        """
        # 获取当前帧的高度和宽度
        frame_height = self.get_frame_height()
        frame_width = self.get_frame_width()
        # 获取当前像素的宽高比
        aspect_ratio = self.get_aspect_ratio()
        # 根据fixed_dimension调整帧尺寸
        if not fixed_dimension:
            # 固定宽度，计算新高度（宽度/像素宽高比）
            frame_height = frame_width / aspect_ratio
        else:
            # 固定高度，计算新宽度（高度×像素宽高比）
            frame_width = aspect_ratio * frame_height
        # 应用新的帧高度和宽度（允许拉伸以匹配）
        self.frame.set_height(frame_height, stretch=True)
        self.frame.set_width(frame_width, stretch=True)

    # ------------------------------
    # 渲染相关方法
    # ------------------------------
    # 捕获并渲染指定的图形对象（将对象绘制到当前FBO）
    def capture(self, *mobjects: Mobject) -> None:
        # 清空当前FBO（填充背景色）
        self.clear()
        # 刷新着色器统一变量（确保使用最新的相机参数）
        self.refresh_uniforms()
        # 激活当前FBO（确保渲染写入正确的缓冲）
        self.fbo.use()
        # 遍历所有图形对象，调用其render方法进行渲染
        for mobject in mobjects:
            mobject.render(self.ctx, self.uniforms)

        # 若有窗口，交换窗口的前后缓冲（显示渲染结果）
        if self.window:
            self.window.swap_buffers()
            # 若当前FBO不是窗口FBO，将渲染结果复制到窗口FBO并再次交换缓冲
            if self.fbo is not self.window_fbo:
                self.blit(self.fbo, self.window_fbo)
                self.window.swap_buffers()

    # 刷新着色器统一变量（将相机参数更新到uniforms字典，供OpenGL程序使用）
    def refresh_uniforms(self) -> None:
        # 获取相机帧对象
        frame = self.frame
        # 获取相机的视图矩阵（世界→相机坐标转换）
        view_matrix = frame.get_view_matrix()
        # 获取光源的世界位置
        light_pos = self.light_source.get_location()
        # 获取相机的世界位置
        cam_pos = self.frame.get_implied_camera_location()

        # 更新uniforms字典，包含渲染所需的核心参数
        self.uniforms.update(
            view=tuple(view_matrix.T.flatten()),  # 视图矩阵（转置后展平为元组）
            frame_scale=frame.get_scale(),  # 相机帧的缩放比例
            frame_rescale_factors=(  # 帧的重缩放因子（用于坐标归一化）
                2.0 / FRAME_WIDTH,  # x轴归一化因子（将FRAME_WIDTH范围转为-1~1）
                2.0 / FRAME_HEIGHT,  # y轴归一化因子（将FRAME_HEIGHT范围转为-1~1）
                frame.get_scale() / frame.get_focal_distance(),  # z轴缩放因子（3D用）
            ),
            pixel_size=self.get_pixel_size(),  # 像素对应的世界宽度
            camera_position=tuple(cam_pos),  # 相机世界位置（元组格式）
            light_position=tuple(light_pos),  # 光源世界位置（元组格式）
        )


# 定义ThreeDCamera类，继承自Camera，用于3D场景
# 主要为了兼容旧场景代码，默认启用4倍多重采样（提升3D场景抗锯齿效果）
class ThreeDCamera(Camera):
    def __init__(self, samples: int = 4, **kwargs):
        # 调用父类构造方法，默认设置samples=4（3D场景抗锯齿）
        super().__init__(samples=samples, **kwargs)
