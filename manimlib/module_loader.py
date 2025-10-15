# ManimGL 模块加载与重载工具类，负责**从用户脚本文件加载模块、跟踪依赖模块、支持深度重载**，
# 核心解决“代码修改后无需重启框架”的问题（如交互式调试时更新脚本），是提升开发效率的关键组件。


class ModuleLoader:
    """
    模块加载工具类：提供模块加载、依赖跟踪、深度重载功能，
    核心用于处理用户脚本及其依赖模块的动态更新（如 `--autoreload` 模式或交互式 `reload()` 命令）。
    """

    @staticmethod
    def get_module(file_name: str | None, is_during_reload=False) -> Module | None:
        """
        主入口：从指定脚本文件加载模块，支持重载时跟踪并更新依赖模块。
        
        核心逻辑：
        1. 模块路径处理：将文件路径转换为模块名（如 "example.py" → "example"，"dir/example.py" → "dir.example"）；
        2. 模块规范创建：通过 `importlib.util.spec_from_file_location` 生成模块规范（关联文件与模块名）；
        3. 模块实例化：基于规范创建空模块实例；
        4. 重载时处理：若处于重载模式（`is_during_reload=True`），先跟踪模块依赖并深度重载，再执行模块；
        5. 模块执行：调用规范加载器执行模块（运行脚本代码，定义场景类等）。
        
        参数：
            file_name : 脚本文件路径（如 "example.py"），None 则返回 None；
            is_during_reload : 是否处于重载过程（True 时需跟踪依赖模块）。
        返回：加载完成的模块实例，或 None（无文件路径时）。
        """
        if file_name is None:
            return None

        # 1. 文件路径转模块名（替换路径分隔符为点，移除 .py 后缀）
        module_name = file_name.replace(os.sep, ".").replace(".py", "")
        # 2. 生成模块规范（关联模块名与文件路径）
        spec = importlib.util.spec_from_file_location(module_name, file_name)
        # 3. 创建空模块实例
        module = importlib.util.module_from_spec(spec)

        # 4. 重载模式：跟踪依赖模块并深度重载
        if is_during_reload:
            # 跟踪模块执行时导入的所有依赖
            imported_modules = ModuleLoader._exec_module_and_track_imports(spec, module)
            # 深度重载所有用户定义的依赖模块
            reloaded_modules_tracker = set()
            ModuleLoader._reload_modules(imported_modules, reloaded_modules_tracker)

        # 5. 执行模块（运行脚本代码，定义场景类等）
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _exec_module_and_track_imports(spec, module: Module) -> set[str]:
        """
        执行模块并跟踪其导入的所有依赖模块：通过替换内置 `__import__` 函数实现依赖记录。
        
        核心逻辑：
        1. 保存原始 `__import__` 函数（避免影响其他模块）；
        2. 定义自定义 `tracked_import` 函数：执行原始导入逻辑，同时记录导入的模块名；
        3. 替换内置 `__import__` 为自定义函数，执行模块；
        4. 恢复原始 `__import__` 函数，返回记录的依赖模块名集合。
        
        参数：
            spec : 模块规范（含加载器）；
            module : 待执行的模块实例。
        返回：模块执行过程中导入的所有依赖模块名集合（如 {"numpy", "manimlib", "custom_utils"}）。
        """
        imported_modules: set[str] = set()
        original_import = builtins.__import__  # 保存原始导入函数

        # 自定义导入函数：记录导入的模块名
        def tracked_import(name, globals=None, locals=None, fromlist=(), level=0):
            result = original_import(name, globals, locals, fromlist, level)  # 执行原始导入
            imported_modules.add(name)  # 记录导入的模块名
            return result

        # 替换内置 __import__ 为自定义函数
        builtins.__import__ = tracked_import

        try:
            module_name = module.__name__
            log.debug('Reloading module "%s"', module_name)
            # 执行模块（运行脚本代码，触发依赖导入）
            spec.loader.exec_module(module)
        finally:
            builtins.__import__ = original_import  # 恢复原始导入函数

        return imported_modules

    @staticmethod
    def _reload_modules(modules: set[str], reloaded_modules_tracker: set[str]):
        """
        重载指定模块集合中的“用户自定义模块”（跳过标准库和第三方库），避免重复重载。
        
        核心逻辑：
        1. 遍历待重载模块名，跳过已重载的模块（通过 tracker 去重）；
        2. 检查模块是否为用户自定义（`_is_user_defined_module`），非用户模块则跳过；
        3. 从 `sys.modules` 获取模块实例，调用 `_deep_reload` 深度重载；
        4. 将重载后的模块加入 tracker，避免重复处理。
        
        参数：
            modules : 待重载的模块名集合；
            reloaded_modules_tracker : 已重载模块名的跟踪集合（去重用）。
        """
        for mod_name in modules:
            # 跳过已重载的模块
            if mod_name in reloaded_modules_tracker:
                continue

            # 跳过非用户自定义模块（标准库、第三方库）
            if not ModuleLoader._is_user_defined_module(mod_name):
                continue

            # 从系统模块缓存中获取模块实例
            module = sys.modules[mod_name]
            # 深度重载模块及其依赖
            ModuleLoader._deep_reload(module, reloaded_modules_tracker)

            # 标记为已重载
            reloaded_modules_tracker.add(mod_name)

    @staticmethod
    def _is_user_defined_module(mod_name: str) -> bool:
        """
        判断模块是否为“用户自定义模块”（非标准库、非第三方库），避免重载系统核心模块。
        
        判断条件（需同时满足）：
        1. 模块已加载到 `sys.modules` 中；
        2. 模块不是内置模块（如 `sys`、`os`）；
        3. 模块文件路径不存在或不包含“site-packages”/“dist-packages”（排除第三方库）；
        4. 模块文件路径不位于标准库目录下（排除系统模块）。
        
        参数：mod_name : 模块名（如 "example"、"custom_utils"）
        返回：布尔值，True 表示是用户自定义模块。
        """
        # 模块未加载，跳过
        if mod_name not in sys.modules:
            return False

        # 内置模块（如 sys、os），跳过
        if mod_name in sys.builtin_module_names:
            return False

        module = sys.modules[mod_name]
        module_path = getattr(module, "__file__", None)  # 获取模块文件路径
        if module_path is None:
            return False  # 无文件路径（如内置模块），跳过

        module_path = os.path.abspath(module_path)

        # 第三方库（位于 site-packages 或 dist-packages），跳过
        if "site-packages" in module_path or "dist-packages" in module_path:
            return False

        # 标准库（位于系统标准库目录），跳过
        standard_lib_path = sysconfig.get_path("stdlib")
        if module_path.startswith(standard_lib_path):
            return False

        # 满足所有条件，是用户自定义模块
        return True

    @staticmethod
    def _deep_reload(module: Module, reloaded_modules_tracker: set[str]):
        """
        深度重载模块：递归重载模块的所有用户自定义依赖模块，再重载模块本身，
        确保代码修改能完全生效（如用户脚本依赖的 `custom_utils.py` 修改后也能更新）。
        
        核心逻辑：
        1. 跳过保护模块：Manim 核心模块（除 config 外）和 config 模块（避免全局配置被重置）；
        2. 跳过非容器模块：无 `__dict__` 的模块（无法遍历属性）；
        3. 去重处理：已重载的模块不再处理；
        4. 递归重载依赖：遍历模块属性，若属性是用户模块或来自用户模块，递归重载；
        5. 重载当前模块：调用 `importlib.reload` 重载模块本身。
        
        参数：
            module : 待深度重载的模块实例；
            reloaded_modules_tracker : 已重载模块名的跟踪集合（去重用）。
        """
        # 1. 跳过 Manim 核心模块（避免框架自身逻辑被破坏）
        ignore_manimlib_modules = manim_config.ignore_manimlib_modules_on_reload
        if ignore_manimlib_modules and module.__name__.startswith("manimlib"):
            return
        # 跳过 config 模块（避免全局配置被重置）
        if module.__name__.startswith("manimlib.config"):
            return

        # 2. 非容器模块（无 __dict__，无法遍历属性），跳过
        if not hasattr(module, "__dict__"):
            return

        # 3. 已重载，跳过
        mod_name = module.__name__
        if mod_name in reloaded_modules_tracker:
            return
        reloaded_modules_tracker.add(mod_name)  # 标记为待重载

        # 4. 递归重载模块的所有用户自定义依赖
        for _attr_name, attr_value in module.__dict__.items():
            # 情况1：属性本身是模块
            if isinstance(attr_value, Module):
                if ModuleLoader._is_user_defined_module(attr_value.__name__):
                    ModuleLoader._deep_reload(attr_value, reloaded_modules_tracker)
            # 情况2：属性来自某个模块（如类、函数，通过 __module__ 关联）
            elif hasattr(attr_value, "__module__"):
                attr_mod_name = attr_value.__module__
                if ModuleLoader._is_user_defined_module(attr_mod_name):
                    attr_module = sys.modules[attr_mod_name]
                    ModuleLoader._deep_reload(attr_module, reloaded_modules_tracker)

        # 5. 重载当前模块（更新代码）
        log.debug('Reloading module "%s"', mod_name)
        importlib.reload(module)