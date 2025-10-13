#!/usr/bin/env python
# 指定脚本解释器为系统中的python

from addict import Dict
# 从addict库导入Dict类，用于创建可以通过属性访问的字典

from manimlib import __version__
# 从manimlib库导入版本号变量

from manimlib.config import manim_config
# 从manimlib的config模块导入全局配置对象manim_config

from manimlib.config import parse_cli
# 从manimlib的config模块导入解析命令行参数的函数parse_cli

import manimlib.extract_scene
# 导入manimlib的extract_scene模块，用于提取和处理场景

from manimlib.utils.cache import clear_cache
# 从manimlib的工具缓存模块导入清除缓存的函数

from manimlib.window import Window
# 从manimlib的window模块导入Window类，用于创建显示窗口


from IPython.terminal.embed import KillEmbedded
# 从IPython终端嵌入模块导入KillEmbedded异常类，用于处理IPython嵌入中断


from typing import TYPE_CHECKING
# 从typing模块导入TYPE_CHECKING常量，用于类型检查条件判断

if TYPE_CHECKING:
    # 仅在类型检查时执行以下导入，不影响运行时
    from argparse import Namespace
    # 导入Namespace类型用于类型注解


def run_scenes():
    """
    Runs the scenes in a loop and detects when a scene reload is requested.
    循环运行场景并检测是否需要重新加载场景
    """
    # Create a new dict to be able to update without
    # altering global configuration
    # 创建新的字典以在不改变全局配置的情况下进行更新
    scene_config = Dict(manim_config.scene)
    run_config = manim_config.run

    if run_config.show_in_window:
        # 如果配置为在窗口中显示
        # Create a reusable window
        # 创建可重用的窗口
        window = Window(** manim_config.window)
        scene_config.update(window=window)
        # 将窗口对象添加到场景配置中

    while True:
        # 无限循环，用于支持场景重新加载
        try:
            # Blocking call since a scene may init an IPython shell()
            # 阻塞调用，因为场景可能会初始化IPython shell
            scenes = manimlib.extract_scene.main(scene_config, run_config)
            # 提取并获取场景列表
            
            for scene in scenes:
                # 遍历场景列表
                scene.run()
                # 运行每个场景
            return
            # 所有场景运行完毕后退出循环
        except KillEmbedded:
            # 捕获KillEmbedded异常（当请求重新加载场景时）
            # Requested via the `exit_raise` IPython runline magic
            # by means of the reload_scene() command
            # 通过IPython的`exit_raise`魔术命令或reload_scene()命令触发
            pass
            # 忽略异常，继续循环以重新加载场景
        except KeyboardInterrupt:
            # 捕获键盘中断（通常是Ctrl+C）
            break
            # 跳出循环，结束程序


def main():
    """
    Main entry point for ManimGL.
    ManimGL的主入口点
    """
    print(f"ManimGL \033[32mv{__version__}\033[0m")
    # 打印ManimGL版本号，使用ANSI转义码使版本号显示为绿色

    args = parse_cli()
    # 解析命令行参数并将结果存储在args中

    if args.version and args.file is None:
        # 如果命令行参数指定了版本且未指定文件
        return
        # 直接返回，不执行后续操作（只显示版本号）

    if args.clear_cache:
        # 如果命令行参数指定了清除缓存
        clear_cache()
        # 执行清除缓存操作

    run_scenes()
    # 调用run_scenes()函数运行场景


if __name__ == "__main__":
    # 如果当前模块作为主程序运行
    main()
    # 调用main()函数，启动程序
