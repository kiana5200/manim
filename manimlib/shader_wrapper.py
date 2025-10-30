# 从__future__模块导入annotations，用于支持 postponed类型注解评估
from __future__ import annotations

# 导入copy模块，用于对象的复制操作
import copy
# 导入os模块，用于与操作系统交互（如文件路径处理）
import os
# 导入re模块，用于正则表达式操作
import re

# 导入OpenGL.GL模块并简写为gl，用于底层OpenGL操作
import OpenGL.GL as gl
# 导入moderngl库，用于现代OpenGL封装，简化 shader 编程等操作
import moderngl
# 导入numpy库并简写为np，用于数值计算和数组操作
import numpy as np
# 从functools模块导入lru_cache装饰器，用于缓存函数调用结果，提高性能
from functools import lru_cache

# 从manimlib.config模块导入parse_cli函数，用于解析命令行参数
from manimlib.config import parse_cli
# 从manimlib.config模块导入manim_config对象，用于访问Manim的配置信息
from manimlib.config import manim_config
# 从manimlib.utils.shaders模块导入get_shader_code_from_file函数，用于从文件中获取着色器代码
from manimlib.utils.shaders import get_shader_code_from_file
# 从manimlib.utils.shaders模块导入get_shader_program函数，用于创建着色器程序
from manimlib.utils.shaders import get_shader_program
# 从manimlib.utils.shaders模块导入image_path_to_texture函数，用于将图像路径转换为纹理对象
from manimlib.utils.shaders import image_path_to_texture
# 从manimlib.utils.shaders模块导入set_program_uniform函数，用于设置着色器程序的 uniforms 变量
from manimlib.utils.shaders import set_program_uniform

# 从typing模块导入TYPE_CHECKING常量，用于条件性导入类型提示（仅在类型检查时生效）
from typing import TYPE_CHECKING

# 如果处于类型检查阶段（非运行时）
if TYPE_CHECKING:
    # 从typing模块导入Optional、Tuple、Iterable类型，用于类型注解
    from typing import Optional, Tuple, Iterable
    # 从manimlib.typing模块导入UniformDict类型，用于表示uniform变量的字典类型
    from manimlib.typing import UniformDict
    # 从moderngl.vertex_array模块导入VertexArray类型，用于顶点数组对象的类型注解
    from moderngl.vertex_array import VertexArray
    # 从moderngl.framebuffer模块导入Framebuffer类型，用于帧缓冲对象的类型注解
    from moderngl.framebuffer import Framebuffer

# Mobjects that should be rendered with
# the same shader will be organized and
# clumped together based on keeping track
# of a dict holding all the relevant information
# to that shader


class ShaderWrapper(object):
    """着色器包装器类，用于管理着色器程序、顶点数据、纹理等渲染相关资源"""
    
    def __init__(
        self,
        ctx: moderngl.context.Context,  # moderngl上下文对象，用于创建和管理OpenGL资源
        vert_data: np.ndarray,  # 顶点数据数组，包含顶点属性信息
        shader_folder: Optional[str] = None,  # 着色器文件所在文件夹路径，可选
        mobject_uniforms: Optional[UniformDict] = None,  # 映射uniform变量名的字典（注释补充：用于传递给着色器的变量）
        texture_paths: Optional[dict[str, str]] = None,  # 纹理名称到文件路径的映射字典
        depth_test: bool = False,  # 是否启用深度测试
        render_primitive: int = moderngl.TRIANGLE_STRIP,  # 渲染图元类型（默认三角形带）
        code_replacements: dict[str, str] = dict(),  # 着色器代码中的替换规则（键为旧内容，值为新内容）
    ):
        self.ctx = ctx  # 保存moderngl上下文
        self.vert_data = vert_data  # 保存顶点数据
        self.vert_attributes = vert_data.dtype.names  # 从顶点数据 dtype 中获取顶点属性名称
        self.shader_folder = shader_folder  # 保存着色器文件夹路径
        self.depth_test = depth_test  # 保存深度测试启用状态
        self.render_primitive = render_primitive  # 保存渲染图元类型
        self.texture_paths = texture_paths or dict()  # 初始化纹理路径字典（默认为空字典）

        self.program_uniform_mirror: UniformDict = dict()  # 用于镜像着色器程序中的uniform变量
        self.bind_to_mobject_uniforms(mobject_uniforms or dict())  # 绑定mobject的uniform变量（默认为空字典）

        self.init_program_code()  # 初始化着色器程序代码
        # 应用代码替换规则
        for old, new in code_replacements.items():
            self.replace_code(old, new)
        self.init_program()  # 初始化着色器程序
        self.init_textures()  # 初始化纹理
        self.init_vertex_objects()  # 初始化顶点相关对象（VBO、VAO等）
        self.refresh_id()  # 刷新当前对象的唯一标识

    def __deepcopy__(self, memo):
        """重写深拷贝方法，禁止深拷贝
        
        说明：如果包含此ShaderWrapper的mobject被复制，返回None意味着父对象可以平滑处理这种情况
        """
        return None

    def init_program_code(self) -> None:
        """初始化着色器程序代码（顶点、几何、片段着色器）"""
        def get_code(name: str) -> str | None:
            """从文件中获取指定名称的着色器代码"""
            return get_shader_code_from_file(
                os.path.join(self.shader_folder, f"{name}.glsl")  # 拼接着色器文件路径（name对应vert/geom/frag）
            )

        # 存储三种着色器的代码（顶点着色器、几何着色器、片段着色器）
        self.program_code: dict[str, str | None] = {
            "vertex_shader": get_code("vert"),  # 读取vert.glsl作为顶点着色器代码
            "geometry_shader": get_code("geom"),  # 读取geom.glsl作为几何着色器代码
            "fragment_shader": get_code("frag"),  # 读取frag.glsl作为片段着色器代码
        }

    def init_program(self):
        """初始化着色器程序"""
        if not self.shader_folder:
            # 如果没有指定着色器文件夹，初始化空程序相关属性
            self.program = None
            self.vert_format = None
            self.programs = []
            return
        # 创建着色器程序（传入三种着色器代码）
        self.program = get_shader_program(self.ctx, **self.program_code)
        # 检测顶点格式（根据程序和顶点属性名称）
        self.vert_format = moderngl.detect_format(self.program, self.vert_attributes)
        self.programs = [self.program]  # 存储程序的列表（可能用于多程序场景）

    def init_textures(self):
        """初始化纹理资源"""
        self.texture_names_to_ids = dict()  # 纹理名称到纹理单元ID的映射
        self.textures = []  # 存储纹理对象的列表
        # 遍历纹理路径字典，添加每个纹理
        for name, path in self.texture_paths.items():
            self.add_texture(name, image_path_to_texture(path, self.ctx))  # 将图像路径转换为纹理并添加

    def init_vertex_objects(self):
        """初始化顶点相关对象（VBO和VAO）"""
        self.vbo = None  # 顶点缓冲对象（Vertex Buffer Object）
        self.vaos = []  # 顶点数组对象（Vertex Array Object）列表

    def add_texture(self, name: str, texture: moderngl.Texture):
        """添加纹理并管理纹理单元ID
        
        Args:
            name: 纹理名称（用于在着色器中引用）
            texture: moderngl纹理对象
        """
        # 获取上下文支持的最大纹理单元数量
        max_units = self.ctx.info['GL_MAX_TEXTURE_IMAGE_UNITS']
        # 检查是否超过最大纹理单元限制
        if len(self.textures) >= max_units:
            raise ValueError(f"程序最多只能使用 {max_units} 个纹理")
        # 纹理在列表中的位置即为其纹理单元ID
        self.texture_names_to_ids[name] = len(self.textures)
        self.textures.append(texture)  # 添加纹理到列表

    def bind_to_mobject_uniforms(self, mobject_uniforms: UniformDict):
        """绑定mobject的uniform变量字典"""
        self.mobject_uniforms = mobject_uniforms

    def get_id(self) -> int:
        """获取当前对象的唯一标识ID"""
        return self.id

    def refresh_id(self) -> None:
        """刷新当前对象的唯一标识ID（基于关键属性的哈希值）"""
        self.id = hash("".join(map(str, [
            "".join(map(str, self.program_code.values())),  # 着色器代码的哈希部分
            self.mobject_uniforms,  # uniform变量的哈希部分
            self.depth_test,  # 深度测试状态的哈希部分
            self.render_primitive,  # 渲染图元类型的哈希部分
            self.texture_paths,  # 纹理路径的哈希部分
        ])))

    def replace_code(self, old: str, new: str) -> None:
        """替换着色器代码中的指定内容
        
        Args:
            old: 要替换的旧字符串
            new: 替换后的新字符串
        """
        code_map = self.program_code
        # 遍历三种着色器代码，执行替换
        for name in code_map:
            if code_map[name] is None:
                continue  # 跳过空代码
            code_map[name] = re.sub(old, new, code_map[name])  # 正则替换
        self.init_program()  # 重新初始化程序（使代码替换生效）
        self.refresh_id()  # 刷新ID（因为代码已改变）

    # 上下文状态设置相关方法
    def use_clip_plane(self):
        """判断是否需要使用裁剪平面"""
        if "clip_plane" not in self.mobject_uniforms:
            return False  # 没有裁剪平面uniform变量，返回False
        # 检查裁剪平面参数是否非全零（非全零则需要启用）
        return any(self.mobject_uniforms["clip_plane"])

    def set_ctx_depth_test(self, enable: bool = True) -> None:
        """设置上下文的深度测试状态
        
        Args:
            enable: 是否启用深度测试
        """
        if enable:
            self.ctx.enable(moderngl.DEPTH_TEST)  # 启用深度测试
        else:
            self.ctx.disable(moderngl.DEPTH_TEST)  # 禁用深度测试

    def set_ctx_clip_plane(self, enable: bool = True) -> None:
        """设置上下文的裁剪平面状态（启用GL_CLIP_DISTANCE0）
        
        Args:
            enable: 是否启用裁剪平面
        """
        if enable:
            gl.glEnable(gl.GL_CLIP_DISTANCE0)  # 启用第一个裁剪平面

    # 数据添加相关方法
    def read_in(self, data_list: Iterable[np.ndarray]):
        """读取并合并顶点数据列表到当前顶点数据中
        
        Args:
            data_list: 顶点数据数组的可迭代对象
        """
        total_len = sum(map(len, data_list))  # 计算所有数据的总长度
        if total_len == 0:
            # 如果总长度为0，清空现有VBO
            if self.vbo is not None:
                self.vbo.clear()
            return

        # 若现有顶点数据长度与总长度不匹配，重新拼接数据
        if len(self.vert_data) != total_len:
            self.vert_data = np.concatenate(data_list)
        else:
            # 长度匹配时直接写入现有数组（避免重新分配内存）
            np.concatenate(data_list, out=self.vert_data)

        # 计算总数据大小（字节数）
        total_size = self.vert_data.itemsize * total_len
        # 若VBO不存在或大小不匹配，释放旧VBO并创建新的
        if self.vbo is not None and self.vbo.size != total_size:
            self.release()  # 释放资源（会将vbo设为None）
        if self.vbo is None:
            self.vbo = self.ctx.buffer(self.vert_data)  # 创建新VBO并写入数据
            self.generate_vaos()  # 生成VAO（顶点数组对象）
        else:
            self.vbo.write(self.vert_data)  # 向现有VBO写入数据

    def generate_vaos(self):
        """生成顶点数组对象（VAO）
        
        为每个着色器程序创建对应的VAO，关联VBO、顶点格式和顶点属性
        """
        self.vaos = [
            self.ctx.vertex_array(
                program=program,  # 关联的着色器程序
                # 内容元组：(VBO, 顶点格式, 顶点属性列表)
                content=[(self.vbo, self.vert_format, *self.vert_attributes)],
                mode=self.render_primitive,  # 渲染图元类型
            )
            for program in self.programs  # 为每个程序生成VAO
        ]

    # 与数据和渲染相关的方法
    def pre_render(self):
        """渲染前的准备工作"""
        # 设置深度测试状态
        self.set_ctx_depth_test(self.depth_test)
        # 设置裁剪平面状态
        self.set_ctx_clip_plane(self.use_clip_plane())
        # 绑定所有纹理到对应的纹理单元
        for tid, texture in enumerate(self.textures):
            texture.use(tid)

    def render(self):
        """执行渲染操作"""
        # 遍历所有VAO并渲染
        for vao in self.vaos:
            vao.render()

    def update_program_uniforms(self, camera_uniforms: UniformDict):
        """更新着色器程序中的uniform变量
        
        Args:
            camera_uniforms: 相机相关的uniform变量字典
        """
        for program in self.programs:
            if program is None:
                continue
            # 合并多种uniform来源（mobject自身、相机、纹理单元映射）
            for uniforms in [self.mobject_uniforms, camera_uniforms, self.texture_names_to_ids]:
                for name, value in uniforms.items():
                    # 为程序设置uniform变量
                    set_program_uniform(program, name, value)

    def release(self):
        """释放顶点相关资源（VBO和VAO）"""
        for obj in (self.vbo, *self.vaos):
            if obj is not None:
                obj.release()  # 释放资源
        self.init_vertex_objects()  # 重新初始化顶点对象（置空）

    def release_textures(self):
        """释放纹理资源"""
        for texture in self.textures:
            texture.release()  # 释放纹理
            del texture  # 删除引用
        self.textures = []  # 清空纹理列表
        self.texture_names_to_ids = dict()  # 清空纹理映射


class VShaderWrapper(ShaderWrapper):
    """专门用于处理矢量图形（VMobject）的着色器包装器"""
    
    def __init__(
        self,
        ctx: moderngl.context.Context,
        vert_data: np.ndarray,
        shader_folder: Optional[str] = None,
        mobject_uniforms: Optional[UniformDict] = None,  # 映射uniform变量名的字典
        texture_paths: Optional[dict[str, str]] = None,  # 纹理名称到文件路径的映射
        depth_test: bool = False,
        render_primitive: int = moderngl.TRIANGLES,  # 渲染图元默认三角形
        code_replacements: dict[str, str] = dict(),  # 着色器代码替换规则
        stroke_behind: bool = False,  # 描边是否在填充后面渲染
    ):
        self.stroke_behind = stroke_behind  # 保存描边渲染顺序标志
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
        # 获取填充画布（帧缓冲和VAO）
        self.fill_canvas = VShaderWrapper.get_fill_canvas(self.ctx)
        # 添加填充纹理和深度纹理
        self.add_texture('Texture', self.fill_canvas[0].color_attachments[0])
        self.add_texture('DepthTexture', self.fill_canvas[2].color_attachments[0])

    def init_program_code(self) -> None:
        """初始化着色器程序代码（针对描边、填充、深度三种类型）"""
        self.program_code = {
            # 生成键格式："{类型}_{着色器类型}"（如"stroke_vert"）
            f"{vtype}_{name}": get_shader_code_from_file(
                os.path.join("quadratic_bezier", f"{vtype}", f"{name}.glsl")
            )
            for vtype in ["stroke", "fill", "depth"]  # 三种类型：描边、填充、深度
            for name in ["vert", "geom", "frag"]  # 三种着色器：顶点、几何、片段
        }

    def init_program(self):
        """初始化各类着色器程序"""
        # 描边程序
        self.stroke_program = get_shader_program(
            self.ctx,
            vertex_shader=self.program_code["stroke_vert"],
            geometry_shader=self.program_code["stroke_geom"],
            fragment_shader=self.program_code["stroke_frag"],
        )
        # 填充程序
        self.fill_program = get_shader_program(
            self.ctx,
            vertex_shader=self.program_code["fill_vert"],
            geometry_shader=self.program_code["fill_geom"],
            fragment_shader=self.program_code["fill_frag"],
        )
        # 填充边框程序（基于描边程序修改片段着色器）
        self.fill_border_program = get_shader_program(
            self.ctx,
            vertex_shader=self.program_code["stroke_vert"],
            geometry_shader=self.program_code["stroke_geom"],
            fragment_shader=self.program_code["stroke_frag"].replace(
                "// MODIFY FRAG COLOR",
                "frag_color.a *= 0.95; frag_color.rgb *= frag_color.a;",  # 调整边框透明度
            )
        )
        # 填充深度程序
        self.fill_depth_program = get_shader_program(
            self.ctx,
            vertex_shader=self.program_code["depth_vert"],
            geometry_shader=self.program_code["depth_geom"],
            fragment_shader=self.program_code["depth_frag"],
        )
        # 存储所有程序
        self.programs = [self.stroke_program, self.fill_program, self.fill_border_program, self.fill_depth_program]

        # 顶点格式说明（总4x23=92字节）：
        # point 3（点坐标）
        # stroke_rgba 4（描边颜色）
        # stroke_width 1（描边宽度）
        # joint_angle 1（连接角）
        # fill_rgba 4（填充颜色）
        # base_normal 3（基础法向量）
        # fill_border_width 1（填充边框宽度）
        
        # 描边顶点格式（16x表示16字节填充，用于内存对齐）
        self.stroke_vert_format = '3f 4f 1f 1f 16x 3f 4x'
        self.stroke_vert_attributes = ['point', 'stroke_rgba', 'stroke_width', 'joint_angle', 'unit_normal']

        # 填充顶点格式
        self.fill_vert_format = '3f 24x 4f 3f 4x'
        self.fill_vert_attributes = ['point', 'fill_rgba', 'base_normal']

        # 填充边框顶点格式
        self.fill_border_vert_format = '3f 20x 1f 4f 3f 1f'
        self.fill_border_vert_attributes = ['point', 'joint_angle', 'stroke_rgba', 'unit_normal', 'stroke_width']

        # 填充深度顶点格式
        self.fill_depth_vert_format = '3f 40x 3f 4x'
        self.fill_depth_vert_attributes = ['point', 'base_normal']

    def init_vertex_objects(self):
        """初始化顶点相关对象（VBO和各类VAO）"""
        self.vbo = None  # 顶点缓冲对象
        self.stroke_vao = None  # 描边VAO
        self.fill_vao = None  # 填充VAO
        self.fill_border_vao = None  # 填充边框VAO
        self.vaos = []  # VAO列表

    def generate_vaos(self):
        """生成各类VAO（与对应程序和顶点格式关联）"""
        # 描边VAO
        self.stroke_vao = self.ctx.vertex_array(
            program=self.stroke_program,
            content=[(self.vbo, self.stroke_vert_format, *self.stroke_vert_attributes)],
            mode=self.render_primitive,
        )
        # 填充VAO
        self.fill_vao = self.ctx.vertex_array(
            program=self.fill_program,
            content=[(self.vbo, self.fill_vert_format, *self.fill_vert_attributes)],
            mode=self.render_primitive,
        )
        # 填充边框VAO
        self.fill_border_vao = self.ctx.vertex_array(
            program=self.fill_border_program,
            content=[(self.vbo, self.fill_border_vert_format, *self.fill_border_vert_attributes)],
            mode=self.render_primitive,
        )
        # 填充深度VAO
        self.fill_depth_vao = self.ctx.vertex_array(
            program=self.fill_depth_program,
            content=[(self.vbo, self.fill_depth_vert_format, *self.fill_depth_vert_attributes)],
            mode=self.render_primitive,
        )
        # 汇总所有VAO
        self.vaos = [self.stroke_vao, self.fill_vao, self.fill_border_vao, self.fill_depth_vao]

    def set_backstroke(self, value: bool = True):
        """设置描边是否在填充后面渲染"""
        self.stroke_behind = value

    def refresh_id(self):
        """刷新唯一标识ID（包含描边顺序标志）"""
        super().refresh_id()  # 调用父类方法
        self.id = hash(str(self.id) + str(self.stroke_behind))  # 叠加描边顺序的哈希

    # 渲染相关方法
    def render_stroke(self):
        """渲染描边"""
        if self.stroke_vao is None:
            return
        self.stroke_vao.render()

    def render_fill(self):
        """渲染填充（包含特殊的混合计算）"""
        if self.fill_vao is None:
            return

        # 保存原始帧缓冲
        original_fbo = self.ctx.fbo
        # 获取填充画布组件（填充帧缓冲、填充VAO、深度帧缓冲）
        fill_tx_fbo, fill_tx_vao, depth_tx_fbo = self.fill_canvas

        # 渲染到独立纹理（用于处理复杂的alpha混合计算，如缠绕数算法）
        fill_tx_fbo.clear()  # 清空填充帧缓冲
        fill_tx_fbo.use()  # 切换到填充帧缓冲

        # 保存当前深度测试状态并禁用（填充渲染不需要深度测试）
        apply_depth_test = bool(gl.glGetBooleanv(gl.GL_DEPTH_TEST))
        self.ctx.disable(moderngl.DEPTH_TEST)

        # 设置混合函数：用于正负方向三角形的颜色抵消（实现缠绕数计算）
        gl.glBlendFuncSeparate(
            gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA,
            gl.GL_ONE_MINUS_DST_ALPHA, gl.GL_ONE
        )
        self.fill_vao.render()  # 渲染填充

        # 如果需要深度测试，渲染深度信息
        if apply_depth_test:
            self.ctx.enable(moderngl.DEPTH_TEST)
            depth_tx_fbo.clear(1.0)  # 清空深度帧缓冲
            depth_tx_fbo.use()  # 切换到深度帧缓冲
            gl.glBlendFunc(gl.GL_ONE, gl.GL_ONE)  # 设置混合函数
            gl.glBlendEquation(gl.GL_MIN)  # 设置混合方程（取最小值）
            self.fill_depth_vao.render()  # 渲染深度

        # 添加边框（取最大alpha值）
        gl.glBlendFunc(gl.GL_ONE, gl.GL_ONE)
        gl.glBlendEquation(gl.GL_MAX)  # 混合方程（取最大值）
        self.fill_border_vao.render()  # 渲染填充边框

        # 将填充纹理渲染到主场景
        original_fbo.use()  # 切换回原始帧缓冲
        gl.glBlendFunc(gl.GL_ONE, gl.GL_ONE_MINUS_SRC_ALPHA)  # 恢复混合函数
        gl.glBlendEquation(gl.GL_FUNC_ADD)  # 恢复混合方程
        fill_tx_vao.render()  # 渲染填充纹理到主场景

        # 恢复原始混合状态
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    @lru_cache  # 缓存结果，所有实例共享同一画布
    @staticmethod
    def get_fill_canvas(ctx: moderngl.Context) -> Tuple[Framebuffer, VertexArray, Framebuffer]:
        """
        静态方法：创建填充渲染所需的画布资源（共享于所有VShaderWrapper实例）
        
        说明：带填充的矢量图形需要特殊渲染方式（通过alpha混合计算像素缠绕数），
        因此需要先渲染到独立纹理，再合成到主帧缓冲。
        
        返回：(填充帧缓冲, 填充纹理VAO, 深度帧缓冲)
        """
        size = manim_config.camera.resolution  # 获取相机分辨率
        double_size = (2 * size[0], 2 * size[1])  # 双倍大小纹理（用于抗锯齿）

        # 创建填充纹理（4通道，浮点类型以支持负alpha值）
        fill_texture = ctx.texture(size=double_size, components=4, dtype='f2')
        # 创建深度纹理（1通道，浮点类型）
        depth_texture = ctx.texture(size=size, components=1, dtype='f4')

        # 创建帧缓冲并关联纹理
        fill_texture_fbo = ctx.framebuffer(fill_texture)
        depth_texture_fbo = ctx.framebuffer(depth_texture)

        # 简单顶点着色器（用于将纹理渲染到屏幕）
        simple_vert = '''
            #version 330

            in vec2 texcoord;
            out vec2 uv;

            void main() {
                gl_Position = vec4((2.0 * texcoord - 1.0), 0.0, 1.0);  // 转换纹理坐标到NDC
                uv = texcoord;  // 传递纹理坐标
            }
        '''
        # alpha调整片段着色器（处理填充纹理的alpha值转换）
        alpha_adjust_frag = '''
            #version 330

            uniform sampler2D Texture;  // 填充纹理
            uniform sampler2D DepthTexture;  // 深度纹理

            in vec2 uv;  // 纹理坐标
            out vec4 color;  // 输出颜色

            void main() {
                color = texture(Texture, uv);  // 采样填充纹理
                if(color.a == 0) discard;  // 丢弃透明像素

                // 处理负alpha值（缠绕数计算结果转换）
                if(color.a < 0){
                    color.a = -color.a / (1.0 - color.a);
                    color.rgb *= (color.a - 1);
                }

                // 抵消填充片段着色器中的缩放
                color *= 1.06;

                // 设置深度值
                gl_FragDepth = texture(DepthTexture, uv)[0];
            }
        '''
        # 创建纹理渲染程序
        fill_program = ctx.program(
            vertex_shader=simple_vert,
            fragment_shader=alpha_adjust_frag,
        )

        # 创建全屏四边形顶点数据（纹理坐标）
        verts = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        simple_vbo = ctx.buffer(verts.astype('f4').tobytes())  # 创建VBO
        # 创建纹理渲染VAO
        fill_texture_vao = ctx.simple_vertex_array(
            fill_program, simple_vbo, 'texcoord',  # 关联程序、VBO和属性
            mode=moderngl.TRIANGLE_STRIP  # 三角形带方式渲染
        )

        return (fill_texture_fbo, fill_texture_vao, depth_texture_fbo)

    def render(self):
        """执行渲染（根据描边顺序标志决定渲染顺序）"""
        if self.stroke_behind:
            # 先渲染描边，再渲染填充（描边在下层）
            self.render_stroke()
            self.render_fill()
        else:
            # 先渲染填充，再渲染描边（描边在上层）
            self.render_fill()
            self.render_stroke()
