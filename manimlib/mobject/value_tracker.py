# 从未来版本导入注解功能，允许在类型注解中使用尚未定义的类
from __future__ import annotations

# 导入numpy库，用于数值计算
import numpy as np
# 从manimlib库导入Mobject基类，所有可显示对象的基类
from manimlib.mobject.mobject import Mobject
# 从manimlib库导入listify工具函数，用于将输入转换为列表
from manimlib.utils.iterables import listify

# 导入类型检查相关模块
from typing import TYPE_CHECKING

# 条件导入，仅在类型检查时生效，避免循环导入问题
if TYPE_CHECKING:
    # 从manimlib的类型定义中导入Self类型，用于方法返回自身类型注解
    from manimlib.typing import Self


class ValueTracker(Mobject):
    """
    不用于显示。相反，其位置编码了一些数字，
    通常被其他动画或持续动画用于其更新函数，
    通过将其视为mobject，它仍然可以像其他任何对象一样被动画化和操作。
    """
    # 定义值的类型为numpy的float64
    value_type: type = np.float64

    def __init__(
        self,
        value: float | complex | np.ndarray = 0,
        **kwargs
    ):
        # 初始化追踪的值，默认为0
        self.value = value
        # 调用父类Mobject的初始化方法，传入其他关键字参数
        super().__init__(** kwargs)

    def init_uniforms(self) -> None:
        # 调用父类的init_uniforms方法，初始化 uniforms
        super().init_uniforms()
        # 将追踪的值转换为numpy数组，并存储在uniforms字典中
        self.uniforms["value"] = np.array(
            listify(self.value),  # 将值转换为列表形式
            dtype=self.value_type,  # 使用指定的数据类型
        )

    def get_value(self) -> float | complex | np.ndarray:
        # 从uniforms中获取值
        result = self.uniforms["value"]
        # 如果结果只有一个元素，返回单个值，否则返回整个数组
        if len(result) == 1:
            return result[0]
        return result

    def set_value(self, value: float | complex | np.ndarray) -> Self:
        # 设置追踪的值，更新uniforms中的存储
        self.uniforms["value"][:] = value
        # 返回自身实例，支持方法链式调用
        return self

    def increment_value(self, d_value: float | complex) -> None:
        # 增加追踪的值，在当前值基础上加上d_value
        self.set_value(self.get_value() + d_value)


class ExponentialValueTracker(ValueTracker):
    """
    操作方式与ValueTracker类似，但它将值编码为位置坐标的指数，
    这改变了插值行为
    """

    def get_value(self) -> float | complex:
        # 返回指数值，基于父类的get_value结果计算指数
        return np.exp(ValueTracker.get_value(self))

    def set_value(self, value: float | complex):
        # 设置值时存储其对数，使得插值时按指数方式变化
        return ValueTracker.set_value(self, np.log(value))


class ComplexValueTracker(ValueTracker):
    # 复数值追踪器，值的类型为numpy的complex128
    value_type: type = np.complex128