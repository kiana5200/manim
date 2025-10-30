# ManimGL LaTeX 渲染工具：负责将 LaTeX 代码转换为 SVG 图形，支持自定义模板、预导言和编译选项，
# 是处理数学公式、复杂文本渲染的核心模块，为 TexMobject 提供底层支持。


from __future__ import annotations

import os
import re
import yaml
import subprocess
from functools import lru_cache  # 内存缓存，优化重复渲染

from pathlib import Path
import tempfile  # 临时文件管理，用于中间编译过程

from manimlib.utils.cache import cache_on_disk  # 磁盘缓存，持久化渲染结果
from manimlib.config import manim_config  # 全局配置
from manimlib.config import get_manim_dir  # Manim 安装目录
from manimlib.logger import log  # 日志工具
from manimlib.utils.simple_functions import hash_string  # 字符串哈希，用于缓存键生成


def get_tex_template_config(template_name: str) -> dict[str, str]:
    """
    获取 LaTeX 模板配置（编译器和预导言）：从 tex_templates.yml 中读取指定模板的配置，
    支持自定义编译方式（如 latex/xelatex）和预导言（宏包导入、格式设置）。
    
    参数：template_name - 模板名称（如 "default"、"ctex" 用于中文）
    返回：dict - 包含 "compiler"（编译器）和 "preamble"（预导言）的配置字典。
    异常处理：模板不存在时警告并 fallback 到 "default" 模板。
    """
    # 标准化模板名称（替换空格为下划线，转为小写）
    name = template_name.replace(" ", "_").lower()
    # 模板配置文件路径（位于 Manim 安装目录下）
    template_path = os.path.join(get_manim_dir(), "manimlib", "tex_templates.yml")
    # 读取模板配置
    with open(template_path, encoding="utf-8") as tex_templates_file:
        templates_dict = yaml.safe_load(tex_templates_file)
    # 模板不存在时的容错处理
    if name not in templates_dict:
        log.warning(f"Cannot recognize template {name}, falling back to 'default'.")
        name = "default"
    return templates_dict[name]


@lru_cache
def get_tex_config(template: str = "") -> tuple[str, str]:
    """
    获取 LaTeX 编译配置：返回指定模板的编译器和预导言（带内存缓存）。
    
    参数：template - 模板名称（默认使用配置文件中的 tex.template）
    返回：tuple - (编译器名称, 预导言字符串)。
    """
    template = template or manim_config.tex.template  # 优先使用传入模板，否则用全局配置
    config = get_tex_template_config(template)
    return config["compiler"], config["preamble"]


def get_full_tex(content: str, preamble: str = "") -> str:
    """
    生成完整的 LaTeX 源代码：将内容和预导言组合为可编译的 LaTeX 文档结构，
    使用 standalone 类确保输出为紧凑的预览图。
    
    参数：
        content - LaTeX 内容（如数学公式 "$a + b$"）；
        preamble - 预导言（宏包导入、格式设置等）。
    返回：str - 完整的 LaTeX 代码字符串。
    """
    return "\n\n".join((
        "\\documentclass[preview]{standalone}",  # 文档类：预览模式，紧凑输出
        preamble,                               # 预导言
        "\\begin{document}",                    # 文档开始
        content,                                # 核心内容
        "\\end{document}"                       # 文档结束
    )) + "\n"


@lru_cache(maxsize=128)
def latex_to_svg(
    latex: str,
    template: str = "",
    additional_preamble: str = "",
    short_tex: str = "",
    show_message_during_execution: bool = True,
) -> str:
    """
    将 LaTeX 代码转换为 SVG 字符串（带内存缓存，最多缓存 128 个结果）。
    
    流程：
    1. 确定编译器和预导言（结合模板和额外预导言）；
    2. 生成完整 LaTeX 代码；
    3. 调用 full_tex_to_svg 编译并转换为 SVG。
    
    参数：
        latex - LaTeX 内容字符串；
        template - 模板名称；
        additional_preamble - 额外的预导言（补充宏包等）；
        short_tex - 简化的显示文本（用于进度提示）；
        show_message_during_execution - 是否显示编译进度。
    返回：str - SVG 源代码。
    异常：LaTeX 编译失败时抛出 LatexError。
    """
    # 显示编译进度信息
    if show_message_during_execution:
        # 截断过长的文本，避免提示信息过宽
        message = f"Writing {(short_tex or latex)[:70]}..."
    else:
        message = ""

    # 获取编译器和基础预导言
    compiler, preamble = get_tex_config(template)
    # 合并基础预导言和额外预导言
    preamble = "\n".join([preamble, additional_preamble])
    # 生成完整 LaTeX 代码
    full_tex = get_full_tex(latex, preamble)
    # 编译并转换为 SVG
    return full_tex_to_svg(full_tex, compiler, message)


@cache_on_disk
def full_tex_to_svg(full_tex: str, compiler: str = "latex", message: str = "") -> str:
    """
    将完整 LaTeX 代码编译为 SVG 字符串（带磁盘缓存，持久化重复结果）。
    
    核心步骤：
    1. 创建临时目录，写入 LaTeX 源文件；
    2. 调用 latex/xelatex 编译生成 DVI/XDV 文件；
    3. 使用 dvisvgm 将 DVI/XDV 转换为 SVG；
    4. 返回 SVG 字符串，缓存结果避免重复编译。
    
    参数：
        full_tex - 完整的 LaTeX 代码；
        compiler - 编译器（"latex" 或 "xelatex"）；
        message - 编译进度提示信息。
    返回：str - SVG 源代码。
    异常：编译失败时抛出 LatexError，不支持的编译器抛出 NotImplementedError。
    """
    if message:
        print(message, end="\r")  # 显示进度，覆盖当前行

    # 确定中间文件扩展名（latex 生成 .dvi，xelatex 生成 .xdv）
    if compiler == "latex":
        dvi_ext = ".dvi"
    elif compiler == "xelatex":
        dvi_ext = ".xdv"
    else:
        raise NotImplementedError(f"Compiler '{compiler}' is not implemented")

    # 使用临时目录存储中间文件（编译过程产生的 .tex, .dvi, .log 等）
    with tempfile.TemporaryDirectory() as temp_dir:
        tex_path = Path(temp_dir, "working").with_suffix(".tex")  # LaTeX 源文件路径
        dvi_path = tex_path.with_suffix(dvi_ext)                  # 编译输出的 DVI/XDV 路径

        # 写入 LaTeX 源代码到临时文件
        tex_path.write_text(full_tex)

        # 执行 LaTeX 编译命令
        process = subprocess.run(
            [
                compiler,                                  # 编译器可执行文件
                *(['-no-pdf'] if compiler == "xelatex" else []),  # xelatex 需禁用 PDF 输出
                "-interaction=batchmode",                  # 静默模式，不输出交互信息
                "-halt-on-error",                          # 出错时停止编译
                f"-output-directory={temp_dir}",           # 输出目录为临时目录
                tex_path                                   # 输入的 LaTeX 文件
            ],
            capture_output=True,  # 捕获 stdout 和 stderr
            text=True             # 输出为字符串类型
        )

        # 处理编译错误
        if process.returncode != 0:
            error_str = ""
            log_path = tex_path.with_suffix(".log")  # 编译日志文件路径
            if log_path.exists():
                content = log_path.read_text()
                # 从日志中提取错误信息（匹配 "! " 开头的错误行）
                error_match = re.search(r"(?<=\n! ).*\n.*\n", content)
                if error_match:
                    error_str = error_match.group()
            # 抛出包含错误信息的异常
            raise LatexError(error_str or "LaTeX compilation failed")

        # 使用 dvisvgm 将 DVI/XDV 转换为 SVG，并捕获输出
        process = subprocess.run(
            [
                "dvisvgm",     # dvisvgm 可执行文件（DVI 转 SVG 工具）
                dvi_path,      # 输入的 DVI/XDV 文件
                "-n",          # 不嵌入字体（使用 SVG 路径绘制文本）
                "-v", "0",     # 日志级别 0（静默模式）
                "--stdout",    # 输出到标准输出（而非文件）
            ],
            capture_output=True  # 捕获输出的 SVG 数据
        )

        # 解码 SVG 结果（字节流 → 字符串）
        result = process.stdout.decode('utf-8')

    # 清除进度提示
    if message:
        print(" " * len(message), end="\r")

    return result


class LatexError(Exception):
    """LaTeX 编译错误异常：当 LaTeX 代码无法成功编译时抛出，包含错误信息。"""
    pass