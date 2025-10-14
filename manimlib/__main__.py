#!/usr/bin/env python
# 指定脚本解释器为系统默认的Python解释器，使脚本可直接执行

# 从addict库导入Dict类，这是一个支持属性访问的字典扩展类
from addict import Dict

# 从manimlib库导入版本号变量
from manimlib import __version__
# 从manimlib.config模块导入manim_config配置对象，包含Manim的全局配置
from manimlib.config import manim_config
# 从manimlib.config模块导入parse_cli函数，用于解析命令行参数
from manimlib.config import parse_cli
# 导入manimlib.extract_scene模块，用于从脚本中提取场景
import manimlib.extract_scene
# 从manimlib.utils.cache模块导入clear_cache函数，用于清除缓存
from manimlib.utils.cache import clear_cache
# 从manimlib.window模块导入Window类，用于创建可视化窗口
from manimlib.window import Window


# 从IPython.terminal.embed模块导入KillEmbedded类，用于终止IPython嵌入环境
from IPython.terminal.embed import KillEmbedded


# 从typing模块导入TYPE_CHECKING常量，用于条件类型检查
from typing import TYPE_CHECKING
# 如果处于类型检查阶段（非运行时），执行以下导入
if TYPE_CHECKING:
    # 从argparse模块导入Namespace类，用于类型注解（表示命令行参数解析结果）
    from argparse import Namespace


# ManimGL 库的“入口程序”代码，负责**解析命令行参数、初始化运行环境、循环执行场景**，是用户通过命令行启动动画渲染的核心逻辑。


def run_scenes():
    """
    核心功能：循环运行场景，并检测“场景重新加载”请求（如交互式调试中触发重载）。
    负责初始化窗口、配置参数，以及处理场景运行中的特殊异常（如主动重载、中断）。
    
    逻辑步骤：
    1. **配置隔离**：创建 `scene_config` 字典（复制全局 `manim_config.scene`），避免修改全局配置影响其他模块；
    2. **窗口初始化**：若命令行配置“显示窗口”（`run_config.show_in_window=True`），创建可复用的 `Window` 对象，
       并将窗口实例传入 `scene_config`（供后续场景渲染使用，避免重复创建窗口）；
    3. **场景循环执行**：
        a. 尝试调用 `manimlib.extract_scene.main()` 从用户代码中提取场景（根据配置的文件、场景名）；
        b. 遍历提取到的所有场景，调用 `scene.run()` 执行每个场景（渲染动画、播放或导出）；
        c. 若执行成功，跳出循环并返回（正常结束）；
    4. **异常处理**：
        - `KillEmbedded` 异常：由交互式命令（如 `reload_scene()`）触发，表示请求重新加载场景，
          捕获后不终止循环，回到步骤3重新提取并运行场景；
        - `KeyboardInterrupt` 异常：用户按下 `Ctrl+C` 中断程序，捕获后跳出循环，正常退出。
    """
    # 1. 复制全局场景配置，避免修改全局变量
    scene_config = Dict(manim_config.scene)
    run_config = manim_config.run  # 获取运行配置（如是否显示窗口、导出格式）

    # 2. 初始化窗口（仅当需要显示窗口时）
    if run_config.show_in_window:
        window = Window(**manim_config.window)  # 根据窗口配置（如大小、标题）创建实例
        scene_config.update(window=window)      # 将窗口传入场景配置，供场景使用

    # 3. 循环执行场景（支持重新加载）
    while True:
        try:
            # 从用户代码中提取场景（blocking调用，若用户代码有交互式逻辑会在此阻塞）
            # extract_scene.main() 负责解析用户脚本、找到指定的Scene子类
            scenes = manimlib.extract_scene.main(scene_config, run_config)
            # 遍历所有提取到的场景，依次执行
            for scene in scenes:
                scene.run()  # 执行场景（渲染+播放/导出）
            return  # 所有场景执行完成，正常退出循环
        except KillEmbedded:
            # 捕获“重新加载”异常（如用户在IPython交互中调用reload_scene()）
            # 不做处理，直接回到循环开头重新提取场景
            pass
        except KeyboardInterrupt:
            # 捕获用户中断（Ctrl+C），跳出循环退出
            break


def main():
    """
    ManimGL 的“主入口函数”：负责启动程序、打印版本信息、解析命令行参数、触发场景运行。
    是命令行调用 `manimgl` 时首先执行的函数。
    
    逻辑步骤：
    1. **打印版本**：在终端打印 ManimGL 版本号（绿色高亮，提升可读性）；
    2. **解析命令行参数**：调用 `parse_cli()` 函数解析用户输入的命令行参数（如 `-o` 导出、`-w` 显示窗口），
       参数解析结果会更新到全局 `manim_config` 中；
    3. **特殊参数处理**：
        - 若用户仅输入 `-v/--version` 且未指定脚本文件（`args.file is None`），仅打印版本后退出；
        - 若用户输入 `--clear-cache`，调用 `clear_cache()` 清理渲染缓存（如临时SVG、帧文件）；
    4. **启动场景运行**：调用 `run_scenes()` 进入场景执行逻辑。
    """
    # 1. 打印版本号（ANSI 转义码 \033[32m 设为绿色，\033[0m 重置颜色）
    print(f"ManimGL \033[32mv{__version__}\033[0m")

    # 2. 解析命令行参数（如 manimgl example.py SquareScene -w -o）
    args = parse_cli()

    # 3. 处理特殊参数
    # 仅请求版本信息且无脚本文件，打印版本后退出
    if args.version and args.file is None:
        return
    # 清理缓存（用户指定 --clear-cache 时）
    if args.clear_cache:
        clear_cache()

    # 4. 启动场景运行逻辑
    run_scenes()


# 4. 程序入口：若该文件被直接执行（而非作为模块导入），调用 main() 启动
if __name__ == "__main__":
    main()