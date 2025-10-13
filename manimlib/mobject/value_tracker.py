from __future__ import annotations

import numpy as np
from manimlib.mobject.mobject import Mobject
from manimlib.utils.iterables import listify

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from manimlib.typing import Self  # 仅在类型检查时导入Self类型注解


class ValueTracker(Mobject):
    """
    一个特殊的Mobject，它不用于在屏幕上显示，而是用于“跟踪”一个数值。
    通过将数值存储在其内部状态中，它可以像其他Mobject一样被动画化和操作。
    这使得它成为连接动画和动态数据的理想桥梁。

    例如，一个动画可以通过改变ValueTracker的值来间接控制另一个对象的属性（如位置、颜色等）。
    """
    # 定义存储值的数据类型，默认为64位浮点数
    value_type: type = np.float64

    def __init__(
        self,
        value: float | complex | np.ndarray = 0,
        **kwargs
    ):
        """
        初始化ValueTracker。

        参数:
            value: 初始跟踪的值，可以是浮点数、复数或NumPy数组。
            **kwargs: 传递给父类Mobject的额外参数。
        """
        self.value = value  # 临时存储值，在init_uniforms中会被正确处理
        super().__init__(**kwargs)

    def init_uniforms(self) -> None:
        """
        初始化WebGL相关的uniform变量，这是Mobject生命周期的一部分。
        在这里，我们将跟踪的值存储在一个NumPy数组中，以便于后续的动画和更新。
        """
        super().init_uniforms()
        # 将跟踪的值转换为NumPy数组，并指定数据类型
        self.uniforms["value"] = np.array(
            listify(self.value),
            dtype=self.value_type,
        )

    def get_value(self) -> float | complex | np.ndarray:
        """
        获取当前跟踪的值。

        返回:
            当前的值。如果数组长度为1，则返回单个元素，否则返回整个数组。
        """
        result = self.uniforms["value"]
        if len(result) == 1:
            return result[0]
        return result

    def set_value(self, value: float | complex | np.ndarray) -> Self:
        """
        设置跟踪的值。

        参数:
            value: 新的值。

        返回:
            ValueTracker对象本身，便于链式调用。
        """
        # 使用切片赋值[:]来修改数组内容，而不是替换整个数组对象
        self.uniforms["value"][:] = value
        return self

    def increment_value(self, d_value: float | complex) -> None:
        """
        增加跟踪的值。

        参数:
            d_value: 要增加的量。
        """
        self.set_value(self.get_value() + d_value)


class ExponentialValueTracker(ValueTracker):
    """
    指数值跟踪器，是ValueTracker的子类。
    它在内部存储值的对数，因此在动画插值时，值的变化是指数级的。
    这对于需要实现平滑增长或衰减的动画非常有用（如人口增长、放射性衰变）。
    """

    def get_value(self) -> float | complex:
        """
        获取当前的值。
        通过对内部存储的对数进行指数运算来恢复原始值。
        """
        return np.exp(ValueTracker.get_value(self))

    def set_value(self, value: float | complex):
        """
        设置跟踪的值。
        将传入的值取对数后，存储在内部。
        """
        return ValueTracker.set_value(self, np.log(value))


class ComplexValueTracker(ValueTracker):
    """
    复数值跟踪器，是ValueTracker的子类。
    它专门用于跟踪复数，通过将内部存储值的数据类型设置为`np.complex128`来实现。
    """
    # 将数据类型覆盖为128位复数
    value_type: type = np.complex128