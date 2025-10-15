# ManimGL 日志配置模块，负责初始化框架的日志系统，使用 `rich` 库提供美观的终端日志输出，
# 统一日志入口（`log` 对象），确保框架运行中的信息、警告、错误能清晰呈现给用户。


# ------------------------------ 1. 导入依赖与导出声明 ------------------------------
import logging
from rich.logging import RichHandler  # RichHandler 提供彩色、结构化的终端日志输出

# 声明模块导出内容：仅导出 `log` 对象，供其他模块统一使用（避免重复创建日志实例）
__all__ = ["log"]


# ------------------------------ 2. 日志系统初始化 ------------------------------
# 日志输出格式：仅保留日志消息（不含默认的时间、模块名等冗余信息，通过 RichHandler 自动补充结构化格式）
FORMAT = "%(message)s"

# 初始化 logging 系统
logging.basicConfig(
    level=logging.WARNING,  # 默认日志级别：WARNING（仅显示警告及以上级别日志，如 ERROR、CRITICAL）
    format=FORMAT,          # 日志消息格式
    datefmt="[%X]",         # 时间格式：[时:分:秒]（如 [14:30:45]），由 RichHandler 嵌入到日志输出中
    handlers=[RichHandler()]# 使用 RichHandler 处理日志输出，支持彩色文本、层级缩进、终端高亮
)


# ------------------------------ 3. 创建框架专用日志对象 ------------------------------
# 创建名为 "manimgl" 的日志实例，作为整个框架的统一日志入口
# 其他模块通过 `from manimlib.logger import log` 导入后，可调用 log.info()、log.warning() 等方法输出日志
log = logging.getLogger("manimgl")