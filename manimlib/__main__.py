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


def run_scenes():
    """
    Runs the scenes in a loop and detects when a scene reload is requested.
    """
    # Create a new dict to be able to upate without
    # altering global configuration
    scene_config = Dict(manim_config.scene)
    run_config = manim_config.run

    if run_config.show_in_window:
        # Create a reusable window
        window = Window(**manim_config.window)
        scene_config.update(window=window)

    while True:
        try:
            # Blocking call since a scene may init an IPython shell()
            scenes = manimlib.extract_scene.main(scene_config, run_config)
            for scene in scenes:
                scene.run()
            return
        except KillEmbedded:
            # Requested via the `exit_raise` IPython runline magic
            # by means of the reload_scene() command
            pass
        except KeyboardInterrupt:
            break


def main():
    """
    Main entry point for ManimGL.
    """
    print(f"ManimGL \033[32mv{__version__}\033[0m")

    args = parse_cli()
    if args.version and args.file is None:
        return
    if args.clear_cache:
        clear_cache()

    run_scenes()


if __name__ == "__main__":
    main()
