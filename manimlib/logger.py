# 导入logging模块，用于日志记录功能
import logging

# 从rich.logging导入RichHandler，用于美化日志输出格式
from rich.logging import RichHandler

# 定义模块的公开接口，仅导出"log"对象
__all__ = ["log"]


# 定义日志消息的格式（仅包含消息内容）
FORMAT = "%(message)s"
# 配置logging的基本设置
logging.basicConfig(
    level=logging.WARNING,  # 设置日志级别为WARNING（只记录WARNING及以上级别日志）
    format=FORMAT,          # 使用定义的格式
    datefmt="[%X]",         # 日期时间格式（[%H:%M:%S]）
    handlers=[RichHandler()]  # 使用RichHandler处理日志，提供更美观的输出
)

# 创建名为"manimgl"的日志记录器实例，供manimgl库使用
log = logging.getLogger("manimgl")