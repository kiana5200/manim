# 从__future__模块导入annotations特性，用于支持更灵活的类型注解（如字符串形式的类型引用）
from __future__ import annotations

# 导入builtins模块，该模块包含Python的内置函数和类型（如int、str、print等）
import builtins
# 导入importlib模块，用于动态导入其他模块
import importlib
# 导入os模块，提供与操作系统交互的功能（如文件路径操作）
import os
# 导入sys模块，提供与Python解释器交互的功能（如命令行参数、系统路径）
import sys
# 导入sysconfig模块，用于获取Python的配置信息（如安装路径、编译选项等）
import sysconfig

# 从manimlib.config模块导入manim_config对象，该对象存储Manim库的配置信息
from manimlib.config import manim_config
# 从manimlib.logger模块导入log对象，用于Manim库的日志记录功能
from manimlib.logger import log

# 定义Module类型别名，指向importlib.util.types模块中的ModuleType类型，用于类型注解
Module = importlib.util.types.ModuleType


class ModuleLoader:
    """
    用于从文件加载模块并处理其导入的工具类。

    该类的大部分功能仅用于重新加载（reload）功能，
    而 `get_module` 方法是导入模块的主要入口。
    """

    @staticmethod
    def get_module(file_name: str | None, is_during_reload=False) -> Module | None:
        """
        从文件导入模块并返回该模块。

        在重新加载期间（当用户在 IPython  shell 中调用 `reload()` 时），
        我们还会跟踪已导入的模块并重新加载它们（否则它们会被缓存）。
        有关重新加载参数的设置，请参见 reload_manager。

        注意：重新加载模块时，`exec_module()` 会被调用两次：
        1. 在 exec_module_and_track_imports 中跟踪导入的模块
        2. 在此处，使用重新加载后的相关导入模块再次实际执行该模块
        """
        if file_name is None:
            return None

        # 将文件名转换为模块名（替换路径分隔符为点，移除 .py 后缀）
        module_name = file_name.replace(os.sep, ".").replace(".py", "")
        # 根据文件路径创建模块规格
        spec = importlib.util.spec_from_file_location(module_name, file_name)
        # 从规格创建模块对象
        module = importlib.util.module_from_spec(spec)

        if is_during_reload:
            # 执行模块并跟踪其导入的模块
            imported_modules = ModuleLoader._exec_module_and_track_imports(spec, module)
            # 用于跟踪已重新加载的模块，避免重复加载
            reloaded_modules_tracker = set()
            # 重新加载导入的模块
            ModuleLoader._reload_modules(imported_modules, reloaded_modules_tracker)

        # 执行模块（首次加载或重新加载时的二次执行）
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _exec_module_and_track_imports(spec, module: Module) -> set[str]:
        """
        执行给定的模块（导入它）并返回在其执行过程中导入的所有模块。

        实现方式是将内置的 __import__ 函数替换为一个自定义函数，
        该函数会跟踪导入的模块。最后，恢复原始的 __import__ 内置函数。
        """
        imported_modules: set[str] = set()
        # 保存原始的 __import__ 函数
        original_import = builtins.__import__

        def tracked_import(name, globals=None, locals=None, fromlist=(), level=0):
            """
            自定义的 __import__ 函数，功能与原始函数完全相同，
            但会通过将模块名添加到集合中来跟踪导入的模块。
            """
            # 调用原始导入函数
            result = original_import(name, globals, locals, fromlist, level)
            # 记录导入的模块名
            imported_modules.add(name)
            return result

        # 替换内置的 __import__ 为自定义的跟踪函数
        builtins.__import__ = tracked_import

        try:
            module_name = module.__name__
            log.debug('Reloading module "%s"', module_name)

            # 执行模块，此时会触发 tracked_import 跟踪导入
            spec.loader.exec_module(module)
        finally:
            # 无论是否发生异常，都恢复原始的 __import__ 函数
            builtins.__import__ = original_import

        return imported_modules

    @staticmethod
    def _reload_modules(modules: set[str], reloaded_modules_tracker: set[str]):
        """
        在给定的模块中，重新加载那些尚未导入的模块。

        我们会跳过非用户定义的模块（参见 `is_user_defined_module()`）。
        """
        for mod in modules:
            # 如果模块已在跟踪集中，跳过
            if mod in reloaded_modules_tracker:
                continue

            # 如果不是用户定义的模块，跳过
            if not ModuleLoader._is_user_defined_module(mod):
                continue

            # 从系统模块中获取该模块
            module = sys.modules[mod]
            # 深度重新加载模块（包括其依赖的模块）
            ModuleLoader._deep_reload(module, reloaded_modules_tracker)

            # 将已重新加载的模块加入跟踪集
            reloaded_modules_tracker.add(mod)

    @staticmethod
    def _is_user_defined_module(mod: str) -> bool:
        """
        判断给定的模块是否为用户定义的模块。

        一个模块被视为用户定义的模块，如果：
        - 它不是标准库的一部分
        - 并且它不是外部库（site-packages 或 dist-packages 中的库）
        """
        # 如果模块未被导入，返回 False
        if mod not in sys.modules:
            return False

        # 如果是内置模块，返回 False
        if mod in sys.builtin_module_names:
            return False

        # 获取模块对象
        module = sys.modules[mod]
        # 获取模块的文件路径
        module_path = getattr(module, "__file__", None)
        if module_path is None:
            return False
        # 转换为绝对路径
        module_path = os.path.abspath(module_path)

        # 如果是外部库（位于 site-packages 或 dist-packages），返回 False
        if "site-packages" in module_path or "dist-packages" in module_path:
            return False

        # 如果是标准库，返回 False
        standard_lib_path = sysconfig.get_path("stdlib")
        if module_path.startswith(standard_lib_path):
            return False

        # 其余情况视为用户定义的模块
        return True

    @staticmethod
    def _deep_reload(module: Module, reloaded_modules_tracker: set[str]):
        """
        递归重新加载给定模块所导入的模块。

        仅重新加载用户定义的模块，参见 `is_user_defined_module()`。
        """
        # 检查是否需要忽略 manimlib 模块的重新加载
        ignore_manimlib_modules = manim_config.ignore_manimlib_modules_on_reload
        if ignore_manimlib_modules and module.__name__.startswith("manimlib"):
            return
        # 不重新加载 manimlib 的配置模块（避免修改全局配置）
        if module.__name__.startswith("manimlib.config"):
            return

        # 如果模块没有 __dict__ 属性（不可变模块），直接返回
        if not hasattr(module, "__dict__"):
            return

        # 防止同一模块被多次重新加载
        if module.__name__ in reloaded_modules_tracker:
            return
        reloaded_modules_tracker.add(module.__name__)

        # 递归处理所有导入的模块
        for _attr_name, attr_value in module.__dict__.items():
            # 如果属性是模块对象
            if isinstance(attr_value, Module):
                # 若是用户定义的模块，递归重新加载
                if ModuleLoader._is_user_defined_module(attr_value.__name__):
                    ModuleLoader._deep_reload(attr_value, reloaded_modules_tracker)

            # 处理属于类或函数的模块（例如 `from custom_module import CustomClass` 这种情况）
            elif hasattr(attr_value, "__module__"):
                attr_module_name = attr_value.__module__
                # 若是用户定义的模块，递归重新加载
                if ModuleLoader._is_user_defined_module(attr_module_name):
                    attr_module = sys.modules[attr_module_name]
                    ModuleLoader._deep_reload(attr_module, reloaded_modules_tracker)

        # 重新加载当前模块
        log.debug('Reloading module "%s"', module.__name__)
        importlib.reload(module)