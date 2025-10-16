from __future__ import annotations

import copy
import os
import re

# 导入 OpenGL 和 ModernGL 相关依赖：用于 GPU 渲染、着色器编程、顶点/纹理对象管理
import OpenGL.GL as gl
import moderngl
import numpy as np
from functools import lru_cache  # 缓存装饰器（暂未使用）

# 导入 Manim 配置和着色器工具：用于读取着色器代码、创建着色器程序、纹理处理
from manimlib.config import parse_cli
from manimlib.config import manim_config
from manimlib.utils.shaders import get_shader_code_from_file  # 从文件读取着色器代码
from manimlib.utils.shaders import get_shader_program       # 创建 ModernGL 着色器程序
from manimlib.utils.shaders import image_path_to_texture    # 图片路径转 ModernGL 纹理
from manimlib.utils.shaders import set_program_uniform       # 给着色器程序设置统一变量（暂未使用）

from typing import TYPE_CHECKING

# 仅静态类型检查阶段导入类型注解
if TYPE_CHECKING:
    from typing import Optional, Tuple, Iterable
    from manimlib.typing import UniformDict  # 统一字典类型：键为字符串，值为框架常用类型
    from moderngl.vertex_array import VertexArray  # ModernGL 顶点数组对象（用于渲染）
    from moderngl.framebuffer import Framebuffer  # ModernGL 帧缓冲对象（用于离屏渲染）


class ShaderWrapper(object):
    """
    着色器包装类：封装 ModernGL 着色器相关的核心组件（着色器程序、顶点数据、纹理、顶点数组），
    为 Mobject 提供统一的 GPU 渲染接口，负责将 Mobject 的几何数据与着色器逻辑绑定，
    支持深度测试、纹理加载、着色器代码替换等功能。
    
    核心作用：将 Mobject 的“几何信息”（顶点、颜色）与“渲染样式”（着色器代码）解耦，
    相同着色器配置的 Mobject 可共享该包装类，提升渲染效率。
    """
    def __init__(
        self,
        ctx: moderngl.context.Context,  # ModernGL OpenGL 上下文（核心渲染环境）
        vert_data: np.ndarray,          # 顶点数据数组（结构化数组，含顶点坐标、颜色等属性）
        shader_folder: Optional[str] = None,  # 着色器文件目录（存放 vert.glsl/frag.glsl 等）
        mobject_uniforms: Optional[UniformDict] = None,  # Mobject 相关统一变量（传递给着色器的全局参数）
        texture_paths: Optional[dict[str, str]] = None,  # 纹理路径字典（键：纹理名称，值：图片文件路径）
        depth_test: bool = False,       # 是否启用深度测试（3D 场景必需，避免遮挡错误）
        render_primitive: int = moderngl.TRIANGLE_STRIP,  # 渲染图元类型（如三角形带、三角形列表）
        code_replacements: dict[str, str] = dict(),  # 着色器代码替换规则（动态修改着色器逻辑）
    ):
        self.ctx = ctx  # 保存 OpenGL 上下文
        self.vert_data = vert_data  # 保存顶点数据（Mobject 的核心几何信息）
        self.vert_attributes = vert_data.dtype.names  # 顶点属性名称（如 "position"、"color"、"uv"）
        self.shader_folder = shader_folder  # 着色器目录
        self.depth_test = depth_test  # 深度测试开关
        self.render_primitive = render_primitive  # 渲染图元类型
        self.texture_paths = texture_paths or dict()  # 纹理路径（默认空字典）

        self.program_uniform_mirror: UniformDict = dict()  # 统一变量镜像（同步着色器与 Python 端变量）
        # 绑定 Mobject 统一变量（如缩放系数、旋转矩阵等传递给着色器的参数）
        self.bind_to_mobject_uniforms(mobject_uniforms or dict())

        # 初始化着色器代码、程序、纹理、顶点对象
        self.init_program_code()
        # 应用着色器代码替换（动态修改着色器逻辑，如替换宏定义）
        for old, new in code_replacements.items():
            self.replace_code(old, new)
        self.init_program()  # 创建着色器程序
        self.init_textures()  # 加载纹理
        self.init_vertex_objects()  # 初始化顶点缓冲区（VBO）和顶点数组（VAO）
        self.refresh_id()  # 生成当前着色器配置的唯一 ID（用于共享和缓存）

    def __deepcopy__(self, memo):
        """
        禁止深度拷贝：当 Mobject 被深拷贝时，着色器包装类不跟随拷贝（返回 None），
        避免 GPU 资源（如纹理、顶点数组）被重复创建，提升性能并避免资源泄漏。
        """
        return None

    def init_program_code(self) -> None:
        """
        从着色器目录读取顶点着色器、几何着色器、片段着色器代码，存储到 `program_code` 字典。
        
        着色器文件命名规则：顶点着色器=vert.glsl、几何着色器=geom.glsl、片段着色器=frag.glsl，
        若文件不存在，对应着色器代码为 None（使用 ModernGL 默认逻辑）。
        """
        # 内部辅助函数：读取指定名称的着色器文件
        def get_code(name: str) -> str | None:
            return get_shader_code_from_file(
                os.path.join(self.shader_folder, f"{name}.glsl")
            )

        # 存储三种着色器的代码（None 表示未提供该着色器）
        self.program_code: dict[str, str | None] = {
            "vertex_shader": get_code("vert"),    # 顶点着色器（处理顶点位置、变换）
            "geometry_shader": get_code("geom"),  # 几何着色器（可选，处理图元生成/修改）
            "fragment_shader": get_code("frag"),  # 片段着色器（处理像素颜色、纹理采样）
        }

    def init_program(self):
        """
        基于读取的着色器代码创建 ModernGL 着色器程序，并检测顶点属性格式。
        
        若未指定着色器目录（shader_folder=None），则不创建程序（self.program=None），
        适用于无需自定义着色器的简单 Mobject。
        """
        if not self.shader_folder:
            self.program = None  # 无着色器程序
            self.vert_format = None  # 无顶点属性格式
            self.programs = []  # 着色器程序列表（兼容多程序场景）
            return
        # 创建着色器程序（传入三种着色器代码）
        self.program = get_shader_program(self.ctx, **self.program_code)
        # 检测顶点属性格式（根据着色器输入和顶点数据属性名自动匹配）
        self.vert_format = moderngl.detect_format(self.program, self.vert_attributes)
        self.programs = [self.program]  # 存储为列表（便于后续扩展多程序渲染）

    def init_textures(self):
        """
        加载纹理路径字典中的所有图片，转换为 ModernGL 纹理对象，存储纹理名称与 ID 的映射。
        
        纹理 ID 为纹理在列表中的索引（0、1、2...），对应着色器中的纹理单元（GL_TEXTURE0、GL_TEXTURE1...），
        超过 GPU 支持的最大纹理单元数会抛出异常。
        """
        self.texture_names_to_ids = dict()  # 纹理名称 → 纹理单元 ID 映射
        self.textures = []  # 存储 ModernGL 纹理对象的列表
        # 遍历纹理路径，加载每个纹理
        for name, path in self.texture_paths.items():
            self.add_texture(name, image_path_to_texture(path, self.ctx))

    def init_vertex_objects(self):
        """
        初始化顶点相关 GPU 对象：顶点缓冲区（VBO）和顶点数组（VAO），
        后续会在 Mobject 渲染时根据顶点数据动态创建具体对象，此处初始化空值。
        """
        self.vbo = None  # 顶点缓冲区对象（存储顶点数据到 GPU）
        self.vaos = []   # 顶点数组对象（关联 VBO 和着色器属性，用于渲染）

    def add_texture(self, name: str, texture: moderngl.Texture):
        """
        新增纹理到着色器包装类：检查 GPU 支持的最大纹理单元数，避免超出限制。
        
        参数：
            name : 纹理名称（与着色器中的统一变量名对应）；
            texture : ModernGL 纹理对象（由 image_path_to_texture 生成）。
        """
        # 获取 GPU 支持的最大纹理单元数（如 16、32，不同显卡可能不同）
        max_units = self.ctx.info['GL_MAX_TEXTURE_IMAGE_UNITS']
        if len(self.textures) >= max_units:
            raise ValueError(f"Unable to use more than {max_units} textures for a program")
        # 纹理 ID 为其在列表中的索引（对应着色器纹理单元）
        self.texture_names_to_ids[name] = len(self.textures)
        self.textures.append(texture)

    def bind_to_mobject_uniforms(self, mobject_uniforms: UniformDict):
        """
        绑定 Mobject 相关的统一变量（如变换矩阵、颜色系数等），这些变量会传递给着色器程序，
        控制渲染效果（如 Mobject 缩放、旋转、颜色叠加）。
        
        参数：mobject_uniforms - 统一变量字典（键为着色器中的变量名，值为对应的数值/矩阵）。
        """
        self.mobject_uniforms = mobject_uniforms

    def get_id(self) -> int:
        """
        获取当前着色器配置的唯一 ID（用于判断两个 Mobject 是否可共享同一着色器包装类）。
        
        返回：int - 哈希值（基于着色器代码、统一变量、纹理路径等配置生成）。
        """
        return self.id

    def refresh_id(self) -> None:
        """
        刷新着色器配置的唯一 ID：基于着色器代码、统一变量、深度测试开关、渲染图元、纹理路径生成哈希值，
        配置变更时需调用此方法更新 ID，确保共享逻辑正确。
        """
        self.id = hash("".join(map(str, [
            "".join(map(str, self.program_code.values())),  # 着色器代码（核心标识）
            self.mobject_uniforms,  # 统一变量
            self.depth_test,        # 深度测试开关
            self.render_primitive,  # 渲染图元类型
            self.texture_paths,     # 纹理路径
        ])))

    def replace_code(self, old: str, new: str) -> None:
        """
        动态替换着色器代码中的内容（如替换宏定义、条件编译语句），支持运行时修改着色器逻辑，
        替换后需重新初始化着色器程序并更新配置 ID。
        
        参数：
            old : 待替换的字符串（如 "#define USE_TEXTURE 0"）；
            new : 替换后的字符串（如 "#define USE_TEXTURE 1"）。
        """
        code_map = self.program_code
        # 遍历三种着色器代码，执行替换
        for name in code_map:
            if code_map[name] is None:
                continue
            code_map[name] = re.sub(old, new, code_map[name])
        # 重新初始化着色器程序（应用新代码）
        self.init_program()
        # 更新配置 ID（代码变更导致配置变化）
        self.refresh_id()

    # Changing context
    def use_clip_plane(self):
        if "clip_plane" not in self.mobject_uniforms:
            return False
        return any(self.mobject_uniforms["clip_plane"])

    def set_ctx_depth_test(self, enable: bool = True) -> None:
        if enable:
            self.ctx.enable(moderngl.DEPTH_TEST)
        else:
            self.ctx.disable(moderngl.DEPTH_TEST)

    def set_ctx_clip_plane(self, enable: bool = True) -> None:
        if enable:
            gl.glEnable(gl.GL_CLIP_DISTANCE0)

    # Adding data

    def read_in(self, data_list: Iterable[np.ndarray]):
        total_len = sum(map(len, data_list))
        if total_len == 0:
            if self.vbo is not None:
                self.vbo.clear()
            return

        # If possible, read concatenated data into existing list
        if len(self.vert_data) != total_len:
            self.vert_data = np.concatenate(data_list)
        else:
            np.concatenate(data_list, out=self.vert_data)

        # Either create new vbo, or read data into it
        total_size = self.vert_data.itemsize * total_len
        if self.vbo is not None and self.vbo.size != total_size:
            self.release()  # This sets vbo to be None
        if self.vbo is None:
            self.vbo = self.ctx.buffer(self.vert_data)
            self.generate_vaos()
        else:
            self.vbo.write(self.vert_data)

    def generate_vaos(self):
        # Vertex array object
        self.vaos = [
            self.ctx.vertex_array(
                program=program,
                content=[(self.vbo, self.vert_format, *self.vert_attributes)],
                mode=self.render_primitive,
            )
            for program in self.programs
        ]

    # Related to data and rendering
    def pre_render(self):
        self.set_ctx_depth_test(self.depth_test)
        self.set_ctx_clip_plane(self.use_clip_plane())
        for tid, texture in enumerate(self.textures):
            texture.use(tid)

    def render(self):
        for vao in self.vaos:
            vao.render()

    def update_program_uniforms(self, camera_uniforms: UniformDict):
        for program in self.programs:
            if program is None:
                continue
            for uniforms in [self.mobject_uniforms, camera_uniforms, self.texture_names_to_ids]:
                for name, value in uniforms.items():
                    set_program_uniform(program, name, value)

    def release(self):
        for obj in (self.vbo, *self.vaos):
            if obj is not None:
                obj.release()
        self.init_vertex_objects()

    def release_textures(self):
        for texture in self.textures:
            texture.release()
            del texture
        self.textures = []
        self.texture_names_to_ids = dict()


class VShaderWrapper(ShaderWrapper):
    def __init__(
        self,
        ctx: moderngl.context.Context,
        vert_data: np.ndarray,
        shader_folder: Optional[str] = None,
        mobject_uniforms: Optional[UniformDict] = None,  # A dictionary mapping names of uniform variables
        texture_paths: Optional[dict[str, str]] = None,  # A dictionary mapping names to filepaths for textures.
        depth_test: bool = False,
        render_primitive: int = moderngl.TRIANGLES,
        code_replacements: dict[str, str] = dict(),
        stroke_behind: bool = False,
    ):
        self.stroke_behind = stroke_behind
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
        self.fill_canvas = VShaderWrapper.get_fill_canvas(self.ctx)
        self.add_texture('Texture', self.fill_canvas[0].color_attachments[0])
        self.add_texture('DepthTexture', self.fill_canvas[2].color_attachments[0])

    def init_program_code(self) -> None:
        self.program_code = {
            f"{vtype}_{name}": get_shader_code_from_file(
                os.path.join("quadratic_bezier", f"{vtype}", f"{name}.glsl")
            )
            for vtype in ["stroke", "fill", "depth"]
            for name in ["vert", "geom", "frag"]
        }

    def init_program(self):
        self.stroke_program = get_shader_program(
            self.ctx,
            vertex_shader=self.program_code["stroke_vert"],
            geometry_shader=self.program_code["stroke_geom"],
            fragment_shader=self.program_code["stroke_frag"],
        )
        self.fill_program = get_shader_program(
            self.ctx,
            vertex_shader=self.program_code["fill_vert"],
            geometry_shader=self.program_code["fill_geom"],
            fragment_shader=self.program_code["fill_frag"],
        )
        self.fill_border_program = get_shader_program(
            self.ctx,
            vertex_shader=self.program_code["stroke_vert"],
            geometry_shader=self.program_code["stroke_geom"],
            fragment_shader=self.program_code["stroke_frag"].replace(
                "// MODIFY FRAG COLOR",
                "frag_color.a *= 0.95; frag_color.rgb *= frag_color.a;",
            )
        )
        self.fill_depth_program = get_shader_program(
            self.ctx,
            vertex_shader=self.program_code["depth_vert"],
            geometry_shader=self.program_code["depth_geom"],
            fragment_shader=self.program_code["depth_frag"],
        )
        self.programs = [self.stroke_program, self.fill_program, self.fill_border_program, self.fill_depth_program]

        # Full vert format looks like this (total of 4x23 = 92 bytes):
        # point 3
        # stroke_rgba 4
        # stroke_width 1
        # joint_angle 1
        # fill_rgba 4
        # base_normal 3
        # fill_border_width 1
        self.stroke_vert_format = '3f 4f 1f 1f 16x 3f 4x'
        self.stroke_vert_attributes = ['point', 'stroke_rgba', 'stroke_width', 'joint_angle', 'unit_normal']

        self.fill_vert_format = '3f 24x 4f 3f 4x'
        self.fill_vert_attributes = ['point', 'fill_rgba', 'base_normal']

        self.fill_border_vert_format = '3f 20x 1f 4f 3f 1f'
        self.fill_border_vert_attributes = ['point', 'joint_angle', 'stroke_rgba', 'unit_normal', 'stroke_width']

        self.fill_depth_vert_format = '3f 40x 3f 4x'
        self.fill_depth_vert_attributes = ['point', 'base_normal']

    def init_vertex_objects(self):
        self.vbo = None
        self.stroke_vao = None
        self.fill_vao = None
        self.fill_border_vao = None
        self.vaos = []

    def generate_vaos(self):
        self.stroke_vao = self.ctx.vertex_array(
            program=self.stroke_program,
            content=[(self.vbo, self.stroke_vert_format, *self.stroke_vert_attributes)],
            mode=self.render_primitive,
        )
        self.fill_vao = self.ctx.vertex_array(
            program=self.fill_program,
            content=[(self.vbo, self.fill_vert_format, *self.fill_vert_attributes)],
            mode=self.render_primitive,
        )
        self.fill_border_vao = self.ctx.vertex_array(
            program=self.fill_border_program,
            content=[(self.vbo, self.fill_border_vert_format, *self.fill_border_vert_attributes)],
            mode=self.render_primitive,
        )
        self.fill_depth_vao = self.ctx.vertex_array(
            program=self.fill_depth_program,
            content=[(self.vbo, self.fill_depth_vert_format, *self.fill_depth_vert_attributes)],
            mode=self.render_primitive,
        )
        self.vaos = [self.stroke_vao, self.fill_vao, self.fill_border_vao, self.fill_depth_vao]

    def set_backstroke(self, value: bool = True):
        self.stroke_behind = value

    def refresh_id(self):
        super().refresh_id()
        self.id = hash(str(self.id) + str(self.stroke_behind))

    # Rendering
    def render_stroke(self):
        if self.stroke_vao is None:
            return
        self.stroke_vao.render()

    def render_fill(self):
        if self.fill_vao is None:
            return

        original_fbo = self.ctx.fbo
        fill_tx_fbo, fill_tx_vao, depth_tx_fbo = self.fill_canvas

        # Render to a separate texture, due to strange alpha compositing
        # for the blended winding calculation
        fill_tx_fbo.clear()
        fill_tx_fbo.use()

        # Be sure not to apply depth test while rendering fill
        # but set it back to where it was after
        apply_depth_test = bool(gl.glGetBooleanv(gl.GL_DEPTH_TEST))
        self.ctx.disable(moderngl.DEPTH_TEST)

        # With this blend function, the effect of blending alpha a with
        # -a / (1 - a) cancels out, so we can cancel positively and negatively
        # oriented triangles
        gl.glBlendFuncSeparate(
            gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA,
            gl.GL_ONE_MINUS_DST_ALPHA, gl.GL_ONE
        )
        self.fill_vao.render()

        if apply_depth_test:
            self.ctx.enable(moderngl.DEPTH_TEST)
            depth_tx_fbo.clear(1.0)
            depth_tx_fbo.use()
            gl.glBlendFunc(gl.GL_ONE, gl.GL_ONE)
            gl.glBlendEquation(gl.GL_MIN)
            self.fill_depth_vao.render()

        # Now add border, just taking the max alpha
        gl.glBlendFunc(gl.GL_ONE, gl.GL_ONE)
        gl.glBlendEquation(gl.GL_MAX)
        self.fill_border_vao.render()

        # Take the texture we were just drawing to, and render it to
        # the main scene. Account for how alphas have been premultiplied
        original_fbo.use()
        gl.glBlendFunc(gl.GL_ONE, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glBlendEquation(gl.GL_FUNC_ADD)
        fill_tx_vao.render()

        # Return to original blending state
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    # Static method returning one shared value across all VShaderWrappers
    @lru_cache
    @staticmethod
    def get_fill_canvas(ctx: moderngl.Context) -> Tuple[Framebuffer, VertexArray, Framebuffer]:
        """
        Because VMobjects with fill are rendered in a funny way, using
        alpha blending to effectively compute the winding number around
        each pixel, they need to be rendered to a separate texture, which
        is then composited onto the ordinary frame buffer.

        This returns a texture, loaded into a frame buffer, and a vao
        which can display that texture as a simple quad onto a screen,
        along with the rgb value which is meant to be discarded.
        """
        size = manim_config.camera.resolution
        double_size = (2 * size[0], 2 * size[1])

        # Important to make sure dtype is floating point (not fixed point)
        # so that alpha values can be negative and are not clipped
        fill_texture = ctx.texture(size=double_size, components=4, dtype='f2')
        # Use another one to keep track of depth
        depth_texture = ctx.texture(size=size, components=1, dtype='f4')

        fill_texture_fbo = ctx.framebuffer(fill_texture)
        depth_texture_fbo = ctx.framebuffer(depth_texture)

        simple_vert = '''
            #version 330

            in vec2 texcoord;
            out vec2 uv;

            void main() {
                gl_Position = vec4((2.0 * texcoord - 1.0), 0.0, 1.0);
                uv = texcoord;
            }
        '''
        alpha_adjust_frag = '''
            #version 330

            uniform sampler2D Texture;
            uniform sampler2D DepthTexture;

            in vec2 uv;
            out vec4 color;

            void main() {
                color = texture(Texture, uv);
                if(color.a == 0) discard;

                if(color.a < 0){
                    color.a = -color.a / (1.0 - color.a);
                    color.rgb *= (color.a - 1);
                }

                // Counteract scaling in fill frag
                color *= 1.06;

                gl_FragDepth = texture(DepthTexture, uv)[0];
            }
        '''
        fill_program = ctx.program(
            vertex_shader=simple_vert,
            fragment_shader=alpha_adjust_frag,
        )

        verts = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        simple_vbo = ctx.buffer(verts.astype('f4').tobytes())
        fill_texture_vao = ctx.simple_vertex_array(
            fill_program, simple_vbo, 'texcoord',
            mode=moderngl.TRIANGLE_STRIP
        )

        return (fill_texture_fbo, fill_texture_vao, depth_texture_fbo)

    def render(self):
        if self.stroke_behind:
            self.render_stroke()
            self.render_fill()
        else:
            self.render_fill()
            self.render_stroke()
