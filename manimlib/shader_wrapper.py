from __future__ import annotations

import copy
import os
import re

import OpenGL.GL as gl
import moderngl
import numpy as np
from functools import lru_cache

from manimlib.config import parse_cli
from manimlib.config import manim_config
from manimlib.utils.shaders import get_shader_code_from_file
from manimlib.utils.shaders import get_shader_program
from manimlib.utils.shaders import image_path_to_texture
from manimlib.utils.shaders import set_program_uniform

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional, Tuple, Iterable
    from manimlib.typing import UniformDict
    from moderngl.vertex_array import VertexArray
    from moderngl.framebuffer import Framebuffer

# Mobjects that should be rendered with
# the same shader will be organized and
# clumped together based on keeping track
# of a dict holding all the relevant information
# to that shader


class ShaderWrapper(object):
    def __init__(
        self,
        ctx: moderngl.context.Context,
        vert_data: np.ndarray,
        shader_folder: Optional[str] = None,
        mobject_uniforms: Optional[UniformDict] = None,  # A dictionary mapping names of uniform variables
        texture_paths: Optional[dict[str, str]] = None,  # A dictionary mapping names to filepaths for textures.
        depth_test: bool = False,
        render_primitive: int = moderngl.TRIANGLE_STRIP,
        code_replacements: dict[str, str] = dict(),
    ):
        self.ctx = ctx
        self.vert_data = vert_data
        self.vert_attributes = vert_data.dtype.names
        self.shader_folder = shader_folder
        self.depth_test = depth_test
        self.render_primitive = render_primitive
        self.texture_paths = texture_paths or dict()

        self.program_uniform_mirror: UniformDict = dict()
        self.bind_to_mobject_uniforms(mobject_uniforms or dict())

        self.init_program_code()
        for old, new in code_replacements.items():
            self.replace_code(old, new)
        self.init_program()
        self.init_textures()
        self.init_vertex_objects()
        self.refresh_id()

    def __deepcopy__(self, memo):
        # Don't allow deepcopies, e.g. if the mobject with this ShaderWrapper as an
        # attribute gets copies. Returning None means the parent object with this ShaderWrapper
        # as an attribute should smoothly handle this case.
        return None

    def init_program_code(self) -> None:
        def get_code(name: str) -> str | None:
            return get_shader_code_from_file(
                os.path.join(self.shader_folder, f"{name}.glsl")
            )

        self.program_code: dict[str, str | None] = {
            "vertex_shader": get_code("vert"),
            "geometry_shader": get_code("geom"),
            "fragment_shader": get_code("frag"),
        }

    def init_program(self):
        if not self.shader_folder:
            self.program = None
            self.vert_format = None
            self.programs = []
            return
        self.program = get_shader_program(self.ctx, **self.program_code)
        self.vert_format = moderngl.detect_format(self.program, self.vert_attributes)
        self.programs = [self.program]

    def init_textures(self):
        self.texture_names_to_ids = dict()
        self.textures = []
        for name, path in self.texture_paths.items():
            self.add_texture(name, image_path_to_texture(path, self.ctx))

    def init_vertex_objects(self):
        self.vbo = None
        self.vaos = []

    def add_texture(self, name: str, texture: moderngl.Texture):
        max_units = self.ctx.info['GL_MAX_TEXTURE_IMAGE_UNITS']
        if len(self.textures) >= max_units:
            raise ValueError(f"Unable to use more than {max_units} textures for a program")
        # The position in the list determines its id
        self.texture_names_to_ids[name] = len(self.textures)
        self.textures.append(texture)

    def bind_to_mobject_uniforms(self, mobject_uniforms: UniformDict):
        self.mobject_uniforms = mobject_uniforms

    def get_id(self) -> int:
        return self.id

    def refresh_id(self) -> None:
        self.id = hash("".join(map(str, [
            "".join(map(str, self.program_code.values())),
            self.mobject_uniforms,
            self.depth_test,
            self.render_primitive,
            self.texture_paths,
        ])))

    def replace_code(self, old: str, new: str) -> None:
        code_map = self.program_code
        for name in code_map:
            if code_map[name] is None:
                continue
            code_map[name] = re.sub(old, new, code_map[name])
        self.init_program()
        self.refresh_id()

    # ------------------------------ 着色器上下文与渲染控制方法 ------------------------------
    def use_clip_plane(self):
        """
        判断是否启用裁剪平面：检查统一变量中是否包含有效的裁剪平面参数（非全零）。
        
        裁剪平面用于在 3D 场景中隐藏部分几何体（如截面效果），仅当 `mobject_uniforms` 中
        存在 "clip_plane" 且值不全为零时启用。
        
        返回：布尔值，True 表示启用裁剪平面。
        """
        if "clip_plane" not in self.mobject_uniforms:
            return False
        # 检查裁剪平面参数是否非全零（全零表示无效裁剪）
        return any(self.mobject_uniforms["clip_plane"])

    def set_ctx_depth_test(self, enable: bool = True) -> None:
        """
        控制 OpenGL 上下文的深度测试开关：启用时，GPU 会根据深度缓冲区判断像素遮挡关系（3D 必备），
        禁用时，像素按绘制顺序覆盖（2D 常用）。
        
        参数：enable - 是否启用深度测试（True=启用，False=禁用）。
        """
        if enable:
            self.ctx.enable(moderngl.DEPTH_TEST)  # 启用深度测试
        else:
            self.ctx.disable(moderngl.DEPTH_TEST)  # 禁用深度测试

    def set_ctx_clip_plane(self, enable: bool = True) -> None:
        """
        控制 OpenGL 裁剪平面开关：启用时，使用 `clip_plane` 统一变量定义的平面裁剪几何体。
        
        参数：enable - 是否启用裁剪平面（True=启用，False=禁用）。
        """
        if enable:
            # 启用第一个裁剪平面（GL_CLIP_DISTANCE0 对应着色器中的 gl_ClipDistance[0]）
            gl.glEnable(gl.GL_CLIP_DISTANCE0)


# ------------------------------ 顶点数据加载与 GPU 对象管理 ------------------------------
    def read_in(self, data_list: Iterable[np.ndarray]):
        """
        加载顶点数据到 GPU：将多个顶点数据数组合并，更新顶点缓冲区（VBO）和顶点数组（VAO），
        确保 GPU 中的数据与 Mobject 的几何信息同步。
        
        逻辑：
        1. 计算总数据长度，空数据时清空 VBO；
        2. 合并输入数据到 `vert_data`（复用现有数组或创建新数组）；
        3. 根据数据大小判断是否需要重建 VBO（数据大小变化时）；
        4. 新建 VBO 或更新现有 VBO 数据，并生成对应的 VAO。
        
        参数：data_list - 待合并的顶点数据数组列表（每个数组需与 `vert_data` 同结构）。
        """
        # 计算总数据长度（所有数组的元素数之和）
        total_len = sum(map(len, data_list))
        if total_len == 0:
            # 空数据：清空现有 VBO
            if self.vbo is not None:
                self.vbo.clear()
            return

        # 合并数据到 vert_data（复用现有数组或创建新数组）
        if len(self.vert_data) != total_len:
            self.vert_data = np.concatenate(data_list)  # 数据长度变化：创建新数组
        else:
            np.concatenate(data_list, out=self.vert_data)  # 数据长度不变：复用现有数组

        # 计算总数据大小（字节数）
        total_size = self.vert_data.itemsize * total_len
        # 数据大小变化或无 VBO 时：释放旧 VBO 并新建
        if self.vbo is not None and self.vbo.size != total_size:
            self.release()  # 释放旧 VBO 和 VAO（vbo 会被设为 None）
        if self.vbo is None:
            self.vbo = self.ctx.buffer(self.vert_data)  # 创建新 VBO（将数据上传到 GPU）
            self.generate_vaos()  # 生成 VAO（关联 VBO 与着色器属性）
        else:
            self.vbo.write(self.vert_data)  # 数据大小不变：直接更新 VBO 数据

    def generate_vaos(self):
        """
        生成顶点数组对象（VAO）：VAO 是 VBO 与着色器属性之间的桥梁，记录顶点数据如何传递给着色器，
        每个着色器程序对应一个 VAO（存储在 `vaos` 列表中）。
        """
        self.vaos = [
            self.ctx.vertex_array(
                program=program,  # 关联的着色器程序
                # 绑定 VBO 与顶点属性：(VBO, 属性格式, 属性名1, 属性名2, ...)
                content=[(self.vbo, self.vert_format, *self.vert_attributes)],
                mode=self.render_primitive  # 渲染图元类型（如三角形带、三角形列表）
            )
            for program in self.programs  # 为每个着色器程序生成 VAO
        ]


# ------------------------------ 渲染流程控制方法 ------------------------------
    def pre_render(self):
        """
        渲染前准备工作：启用深度测试/裁剪平面，绑定纹理到对应的纹理单元，
        确保渲染环境正确配置。
        """
        # 启用/禁用深度测试（根据自身配置）
        self.set_ctx_depth_test(self.depth_test)
        # 启用/禁用裁剪平面（根据是否有有效裁剪参数）
        self.set_ctx_clip_plane(self.use_clip_plane())
        # 绑定所有纹理到对应的纹理单元（ID=索引，如 0→GL_TEXTURE0，1→GL_TEXTURE1）
        for tid, texture in enumerate(self.textures):
            texture.use(tid)

    def render(self):
        """
        执行渲染：遍历所有顶点数组对象（VAO），调用其 `render` 方法，
        触发 GPU 绘制几何图形（基于绑定的 VBO 数据和着色器程序）。
        """
        for vao in self.vaos:
            vao.render()  # 绘制 VAO 关联的几何数据

    def update_program_uniforms(self, camera_uniforms: UniformDict):
        """
        更新着色器程序的统一变量：合并 Mobject 自身、相机、纹理的统一变量，
        传递给着色器（如变换矩阵、颜色、纹理单元 ID），确保渲染参数实时生效。
        
        参数：camera_uniforms - 相机相关的统一变量（如视图矩阵、投影矩阵）。
        """
        for program in self.programs:
            if program is None:
                continue
            # 依次更新 Mobject 自身、相机、纹理的统一变量
            for uniforms in [self.mobject_uniforms, camera_uniforms, self.texture_names_to_ids]:
                for name, value in uniforms.items():
                    set_program_uniform(program, name, value)  # 设置着色器变量


# ------------------------------ 资源释放方法 ------------------------------
    def release(self):
        """
        释放顶点相关 GPU 资源（VBO 和 VAO）：避免 GPU 内存泄漏，
        调用后需重新调用 `read_in` 生成新资源。
        """
        for obj in (self.vbo, *self.vaos):
            if obj is not None:
                obj.release()  # 释放 GPU 资源
        self.init_vertex_objects()  # 重置 VBO 和 VAO 为空

    def release_textures(self):
        """
        释放纹理资源：释放所有加载的纹理对象，清空纹理列表和映射，
        适用于 Mobject 销毁或纹理更新时。
        """
        for texture in self.textures:
            texture.release()  # 释放纹理 GPU 资源
            del texture  # 移除引用
        self.textures = []
        self.texture_names_to_ids = dict()


# ------------------------------ 二次贝塞尔曲线专用着色器包装类 ------------------------------
class VShaderWrapper(ShaderWrapper):
    """
    二次贝塞尔曲线着色器包装类：继承自 `ShaderWrapper`，专为处理二次贝塞尔曲线（如平滑边缘、填充）设计，
    支持描边（stroke）、填充（fill）、深度（depth）三种渲染模式，使用专用着色器目录和纹理。
    """
    def __init__(
        self,
        ctx: moderngl.context.Context,
        vert_data: np.ndarray,
        shader_folder: Optional[str] = None,
        mobject_uniforms: Optional[UniformDict] = None,
        texture_paths: Optional[dict[str, str]] = None,
        depth_test: bool = False,
        render_primitive: int = moderngl.TRIANGLES,  # 二次贝塞尔曲线默认用三角形列表渲染
        code_replacements: dict[str, str] = dict(),
        stroke_behind: bool = False,  # 描边是否在填充之后渲染（控制层级）
    ):
        self.stroke_behind = stroke_behind  # 描边层级控制
        # 调用父类构造函数初始化基础属性
        super().__init__(
            ctx=ctx,
            vert_data=vert_data,
            shader_folder=shader_folder,
            mobject_uniforms=mobject_uniforms,
            texture_paths=texture_paths,
            depth_test=depth_test,
            render_primitive=render_primitive,
            code_replacements=code_replacements,
        )
        # 创建填充画布（离屏渲染用帧缓冲）并添加为纹理
        self.fill_canvas = VShaderWrapper.get_fill_canvas(self.ctx)
        # 添加颜色纹理和深度纹理（用于二次贝塞尔曲线的填充和描边计算）
        self.add_texture('Texture', self.fill_canvas[0].color_attachments[0])
        self.add_texture('DepthTexture', self.fill_canvas[2].color_attachments[0])

    def init_program_code(self) -> None:
        """
        初始化二次贝塞尔曲线专用着色器代码：从 "quadratic_bezier" 目录读取描边、填充、深度三种模式的
        顶点/几何/片段着色器代码，存储在 `program_code` 字典中（键格式："{模式}_{着色器类型}"）。
        """
        self.program_code = {
            f"{vtype}_{name}": get_shader_code_from_file(
                os.path.join("quadratic_bezier", f"{vtype}", f"{name}.glsl")
            )
            for vtype in ["stroke", "fill", "depth"]  # 三种渲染模式
            for name in ["vert", "geom", "frag"]      # 三种着色器类型
        }
    # ------------------------------ VShaderWrapper 专用初始化方法 ------------------------------
    def init_program(self):
        """
        初始化二次贝塞尔曲线专用着色器程序：创建四种着色器程序（描边、填充、填充边框、填充深度），
        并定义每种程序对应的顶点属性格式和属性名，适配二次贝塞尔曲线的复杂渲染需求（如平滑边缘、填充计算）。
        """
        # 1. 创建四种着色器程序（针对不同渲染阶段）
        # 描边程序：处理曲线边缘的绘制
        self.stroke_program = get_shader_program(
            self.ctx,
            vertex_shader=self.program_code["stroke_vert"],
            geometry_shader=self.program_code["stroke_geom"],
            fragment_shader=self.program_code["stroke_frag"],
        )
        # 填充程序：处理曲线内部区域的填充
        self.fill_program = get_shader_program(
            self.ctx,
            vertex_shader=self.program_code["fill_vert"],
            geometry_shader=self.program_code["fill_geom"],
            fragment_shader=self.program_code["fill_frag"],
        )
        # 填充边框程序：基于描边程序修改片段着色器，实现填充边缘的半透明效果
        self.fill_border_program = get_shader_program(
            self.ctx,
            vertex_shader=self.program_code["stroke_vert"],
            geometry_shader=self.program_code["stroke_geom"],
            fragment_shader=self.program_code["stroke_frag"].replace(
                "// MODIFY FRAG COLOR",  # 替换标记
                "frag_color.a *= 0.95; frag_color.rgb *= frag_color.a;",  # 半透明处理
            )
        )
        # 填充深度程序：处理填充区域的深度信息（用于遮挡计算）
        self.fill_depth_program = get_shader_program(
            self.ctx,
            vertex_shader=self.program_code["depth_vert"],
            geometry_shader=self.program_code["depth_geom"],
            fragment_shader=self.program_code["depth_frag"],
        )
        # 存储所有程序（用于后续渲染和统一变量更新）
        self.programs = [self.stroke_program, self.fill_program, self.fill_border_program, self.fill_depth_program]

        # 2. 定义顶点属性格式（描述顶点数据在内存中的布局，供 GPU 解析）
        # 描边顶点格式：3f(点坐标) + 4f(描边颜色RGBA) + 1f(描边宽度) + 1f(连接角) + 16x(预留16字节) + 3f(单位法向量) + 4x(预留4字节)
        self.stroke_vert_format = '3f 4f 1f 1f 16x 3f 4x'
        self.stroke_vert_attributes = ['point', 'stroke_rgba', 'stroke_width', 'joint_angle', 'unit_normal']

        # 填充顶点格式：3f(点坐标) + 24x(预留24字节) + 4f(填充颜色RGBA) + 3f(基础法向量) + 4x(预留4字节)
        self.fill_vert_format = '3f 24x 4f 3f 4x'
        self.fill_vert_attributes = ['point', 'fill_rgba', 'base_normal']

        # 填充边框顶点格式：3f(点坐标) + 20x(预留20字节) + 1f(连接角) + 4f(描边颜色RGBA) + 3f(单位法向量) + 1f(描边宽度)
        self.fill_border_vert_format = '3f 20x 1f 4f 3f 1f'
        self.fill_border_vert_attributes = ['point', 'joint_angle', 'stroke_rgba', 'unit_normal', 'stroke_width']

        # 填充深度顶点格式：3f(点坐标) + 40x(预留40字节) + 3f(基础法向量) + 4x(预留4字节)
        self.fill_depth_vert_format = '3f 40x 3f 4x'
        self.fill_depth_vert_attributes = ['point', 'base_normal']

    def init_vertex_objects(self):
        """
        初始化二次贝塞尔曲线专用顶点对象：为四种渲染程序分别创建顶点数组对象（VAO）的占位符，
        后续在 `generate_vaos` 中根据 VBO 数据和属性格式生成具体 VAO。
        """
        self.vbo = None  # 顶点缓冲区对象（存储所有顶点数据）
        # 四种渲染程序对应的 VAO 占位符
        self.stroke_vao = None
        self.fill_vao = None
        self.fill_border_vao = None
        self.fill_depth_vao = None
        self.vaos = []  # 存储所有 VAO 的列表

    def generate_vaos(self):
        """
        为四种着色器程序生成对应的顶点数组对象（VAO）：关联 VBO 数据与着色器属性，
        确保每种渲染程序能正确解析顶点数据（如描边程序读取描边颜色，填充程序读取填充颜色）。
        """
        # 描边 VAO：关联描边程序、VBO、描边属性格式
        self.stroke_vao = self.ctx.vertex_array(
            program=self.stroke_program,
            content=[(self.vbo, self.stroke_vert_format, *self.stroke_vert_attributes)],
            mode=self.render_primitive,
        )
        # 填充 VAO：关联填充程序、VBO、填充属性格式
        self.fill_vao = self.ctx.vertex_array(
            program=self.fill_program,
            content=[(self.vbo, self.fill_vert_format, *self.fill_vert_attributes)],
            mode=self.render_primitive,
        )
        # 填充边框 VAO：关联填充边框程序、VBO、填充边框属性格式
        self.fill_border_vao = self.ctx.vertex_array(
            program=self.fill_border_program,
            content=[(self.vbo, self.fill_border_vert_format, *self.fill_border_vert_attributes)],
            mode=self.render_primitive,
        )
        # 填充深度 VAO：关联填充深度程序、VBO、填充深度属性格式
        self.fill_depth_vao = self.ctx.vertex_array(
            program=self.fill_depth_program,
            content=[(self.vbo, self.fill_depth_vert_format, *self.fill_depth_vert_attributes)],
            mode=self.render_primitive,
        )
        # 汇总所有 VAO 到列表
        self.vaos = [self.stroke_vao, self.fill_vao, self.fill_border_vao, self.fill_depth_vao]


# ------------------------------ 渲染控制与配置更新 ------------------------------
    def set_backstroke(self, value: bool = True):
        """
        设置描边层级：控制描边是在填充之前还是之后渲染（影响视觉效果，如描边是否被填充覆盖）。
        
        参数：value - True 表示描边在填充之后渲染（显示在上方），False 表示相反。
        """
        self.stroke_behind = value

    def refresh_id(self):
        """
        刷新配置 ID：在父类 ID 基础上添加描边层级（stroke_behind），确保层级变化时 ID 也变化，
        避免不同层级配置的着色器被错误共享。
        """
        super().refresh_id()  # 先调用父类生成基础 ID
        self.id = hash(str(self.id) + str(self.stroke_behind))  # 追加描边层级信息


# ------------------------------ 二次贝塞尔曲线渲染方法 ------------------------------
    def render_stroke(self):
        """
        渲染描边：使用描边 VAO 绘制曲线边缘，仅在 VAO 有效时执行（避免空数据渲染错误）。
        """
        if self.stroke_vao is None:
            return
        self.stroke_vao.render()  # 触发描边绘制

    def render_fill(self):
        """
        渲染填充（核心逻辑）：通过离屏渲染计算填充区域（解决复杂形状的 alpha 混合问题），
        步骤包括：填充区域计算 → 深度信息处理 → 边框绘制 → 最终合成到主帧缓冲。
        """
        if self.fill_vao is None:
            return  # 填充 VAO 无效时直接返回

        # 保存当前帧缓冲（主屏幕帧缓冲），获取填充画布的三个帧缓冲
        original_fbo = self.ctx.fbo
        fill_tx_fbo, fill_tx_vao, depth_tx_fbo = self.fill_canvas  # 离屏渲染用帧缓冲

        # ------------------------------ 步骤1：离屏渲染填充区域 ------------------------------
        # 切换到填充纹理帧缓冲（离屏渲染，避免直接绘制到主屏幕）
        fill_tx_fbo.clear()  # 清空离屏帧缓冲
        fill_tx_fbo.use()

        # 暂存深度测试状态，渲染填充时禁用深度测试（避免填充被错误遮挡）
        apply_depth_test = bool(gl.glGetBooleanv(gl.GL_DEPTH_TEST))  # 获取当前深度测试状态
        self.ctx.disable(moderngl.DEPTH_TEST)

        # 设置特殊混合模式：通过正负方向三角形的 alpha 抵消，计算正确的填充区域（解决自相交形状）
        gl.glBlendFuncSeparate(
            gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA,  # RGB 通道混合函数
            gl.GL_ONE_MINUS_DST_ALPHA, gl.GL_ONE           # Alpha 通道混合函数
        )
        self.fill_vao.render()  # 渲染填充区域到离屏帧缓冲

        # ------------------------------ 步骤2：处理深度信息（可选） ------------------------------
        if apply_depth_test:
            self.ctx.enable(moderngl.DEPTH_TEST)  # 恢复深度测试
            depth_tx_fbo.clear(1.0)  # 清空深度纹理帧缓冲
            depth_tx_fbo.use()  # 切换到深度纹理帧缓冲

            # 设置混合模式：取最小值（确保深度值正确叠加）
            gl.glBlendFunc(gl.GL_ONE, gl.GL_ONE)
            gl.glBlendEquation(gl.GL_MIN)
            self.fill_depth_vao.render()  # 渲染深度信息到离屏帧缓冲

        # ------------------------------ 步骤3：渲染填充边框（离屏） ------------------------------
        # 设置混合模式：取最大值（确保边框 alpha 不被覆盖）
        gl.glBlendFunc(gl.GL_ONE, gl.GL_ONE)
        gl.glBlendEquation(gl.GL_MAX)
        self.fill_border_vao.render()  # 渲染填充边框到离屏帧缓冲

        # ------------------------------ 步骤4：合成到主屏幕 ------------------------------
        original_fbo.use()  # 切换回主帧缓冲
        # 设置标准混合模式：alpha 混合（确保填充区域正确叠加到主场景）
        gl.glBlendFunc(gl.GL_ONE, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glBlendEquation(gl.GL_FUNC_ADD)
        fill_tx_vao.render()  # 将离屏渲染的填充结果绘制到主屏幕

        # ------------------------------ 步骤5：恢复混合状态 ------------------------------
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)  # 恢复默认混合模式

    # ------------------------------ 静态方法：创建共享填充画布（离屏渲染核心） ------------------------------
    @lru_cache
    @staticmethod
    def get_fill_canvas(ctx: moderngl.Context) -> Tuple[Framebuffer, VertexArray, Framebuffer]:
        """
        创建二次贝塞尔曲线填充用的共享离屏渲染画布：所有 VShaderWrapper 实例共享同一画布（通过 lru_cache 缓存），
        解决复杂形状填充的 alpha 混合问题（如自相交曲线、不规则多边形），核心原理是通过离屏渲染计算像素的“环绕数”，
        再将结果合成到主屏幕。
        
        返回值：Tuple[Framebuffer, VertexArray, Framebuffer]
            - 第一个 Framebuffer：填充颜色离屏帧缓冲（存储填充区域的颜色信息）；
            - VertexArray：简单四边形 VAO（用于将离屏纹理绘制到主屏幕）；
            - 第二个 Framebuffer：深度信息离屏帧缓冲（存储填充区域的深度信息）。
        """
        # 获取相机分辨率（主屏幕渲染分辨率）
        size = manim_config.camera.resolution
        double_size = (2 * size[0], 2 * size[1])  # 填充纹理尺寸：2倍分辨率（抗锯齿，提升填充精度）

        # 1. 创建离屏纹理（浮点型 dtype 确保 alpha 可正负，支持环绕数计算）
        # 填充颜色纹理：4通道（RGBA），f2=16位浮点（支持负 alpha 值，用于抵消计算）
        fill_texture = ctx.texture(size=double_size, components=4, dtype='f2')
        # 深度信息纹理：1通道（仅深度值），f4=32位浮点（高精度深度存储）
        depth_texture = ctx.texture(size=size, components=1, dtype='f4')

        # 2. 创建离屏帧缓冲（绑定纹理，用于离屏渲染）
        fill_texture_fbo = ctx.framebuffer(fill_texture)  # 颜色离屏帧缓冲
        depth_texture_fbo = ctx.framebuffer(depth_texture)  # 深度离屏帧缓冲

        # 3. 定义简单四边形着色器（用于将离屏纹理合成到主屏幕）
        # 顶点着色器：将纹理坐标转为屏幕坐标（映射到全屏四边形）
        simple_vert = '''
            #version 330

            in vec2 texcoord;  // 输入纹理坐标（0-1 范围）
            out vec2 uv;       // 输出纹理坐标（传递给片段着色器）

            void main() {
                // 纹理坐标 (0,0)→(1,1) 转为 NDC 坐标 (-1,-1)→(1,1)（屏幕全屏）
                gl_Position = vec4((2.0 * texcoord - 1.0), 0.0, 1.0);
                uv = texcoord;
            }
        '''
        # 片段着色器：调整离屏纹理的 alpha 值（还原环绕数计算结果），并读取深度信息
        alpha_adjust_frag = '''
            #version 330

            uniform sampler2D Texture;       // 填充颜色纹理（离屏渲染结果）
            uniform sampler2D DepthTexture;  // 深度信息纹理

            in vec2 uv;  // 纹理坐标（从顶点着色器传入）
            out vec4 color;  // 最终输出颜色

            void main() {
                // 采样离屏填充纹理的颜色
                color = texture(Texture, uv);
                if(color.a == 0) discard;  // alpha=0 表示无填充，丢弃该像素

                // 还原环绕数计算结果：负 alpha 转为正常透明度（核心逻辑）
                if(color.a < 0){
                    color.a = -color.a / (1.0 - color.a);  // 负 alpha 转为正透明度
                    color.rgb *= (color.a - 1);           // 调整 RGB 颜色（抵消离屏渲染时的混合影响）
                }

                // 抵消填充着色器中的缩放（修正颜色亮度）
                color *= 1.06;

                // 读取深度纹理，设置当前像素的深度值（确保遮挡关系正确）
                gl_FragDepth = texture(DepthTexture, uv)[0];
            }
        '''
        # 创建简单四边形着色器程序
        fill_program = ctx.program(
            vertex_shader=simple_vert,
            fragment_shader=alpha_adjust_frag,
        )

        # 4. 创建简单四边形 VAO（用于绘制全屏纹理）
        # 四边形顶点坐标（纹理坐标 0-1，对应屏幕全屏）
        verts = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        simple_vbo = ctx.buffer(verts.astype('f4').tobytes())  # 顶点缓冲区（float32 类型）
        # 简单 VAO：关联着色器、VBO，模式为三角形带（TRIANGLE_STRIP，4个顶点绘制2个三角形组成四边形）
        fill_texture_vao = ctx.simple_vertex_array(
            fill_program, simple_vbo, 'texcoord',
            mode=moderngl.TRIANGLE_STRIP
        )

        # 返回离屏帧缓冲、四边形 VAO、深度帧缓冲
        return (fill_texture_fbo, fill_texture_vao, depth_texture_fbo)

    # ------------------------------ 核心渲染方法（整合描边与填充） ------------------------------
    def render(self):
        """
        统一渲染入口：根据描边层级（stroke_behind）决定描边和填充的渲染顺序，
        确保视觉层级符合预期（描边在填充之上或之下）。
        """
        if self.stroke_behind:
            # stroke_behind=True：先渲染描边，再渲染填充（填充覆盖描边，描边在下方）
            self.render_stroke()
            self.render_fill()
        else:
            # stroke_behind=False（默认）：先渲染填充，再渲染描边（描边覆盖填充，描边在上方）
            self.render_fill()
            self.render_stroke()