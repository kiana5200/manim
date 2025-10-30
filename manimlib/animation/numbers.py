# 从__future__导入annotations，支持在类型注解中使用尚未定义的类
from __future__ import annotations

# 从manimlib.animation.animation模块导入Animation类，作为动画的基类
from manimlib.animation.animation import Animation
# 从manimlib.mobject.numbers模块导入DecimalNumber类，用于数字显示
from manimlib.mobject.numbers import DecimalNumber
# 从manimlib.utils.bezier导入interpolate函数，用于插值计算
from manimlib.utils.bezier import interpolate
# 从manimlib.utils.simple_functions导入clip函数，用于值的范围限制
from manimlib.utils.simple_functions import clip

# 从typing模块导入TYPE_CHECKING，用于条件导入类型提示
from typing import TYPE_CHECKING

# 如果是类型检查阶段（非运行时执行），则导入所需的类型提示
if TYPE_CHECKING:
    # 从typing模块导入Callable，用于标注可调用对象类型
    from typing import Callable



"""
用于实现数字的动态变化效果
核心功能：
1.接收一个DecimalNumber对象（用于显示数字）和一个更新函数
2动画过程中，根据当前进度alpha（0 到 1）计算对应的数字值
3.通过更新函数可以灵活定义数字的变化规律（如线性增长、指数变化等）
4.实时更新数字显示，实现平滑的数字变化动画
"""
# 定义ChangingDecimal类，继承自Animation，用于实现数字的动态变化动画
class ChangingDecimal(Animation):
    def __init__(
        self,
        decimal_mob: DecimalNumber,  # 要进行动态变化的DecimalNumber对象（数字显示物体）
        number_update_func: Callable[[float], float],
        # 数字更新函数，接收一个alpha值（0到1），返回对应时刻的数字值
        suspend_mobject_updating: bool = False,  # 是否暂停物体的更新，默认不暂停
        **kwargs  # 其他关键字参数，传递给父类Animation
    ):
        # 断言确保输入的decimal_mob确实是DecimalNumber类型
        assert isinstance(decimal_mob, DecimalNumber)
        # 保存数字更新函数到实例变量
        self.number_update_func = number_update_func
        # 调用父类Animation的初始化方法，传递参数
        super().__init__(
            decimal_mob,
            suspend_mobject_updating=suspend_mobject_updating,** kwargs
        )
        # 显式将数字物体赋值给self.mobject（确保引用正确）
        self.mobject = decimal_mob

    # 插值方法，更新数字的显示值
    def interpolate_mobject(self, alpha: float) -> None:
        # 计算考虑时间跨度的实际alpha值（处理动画的开始和结束时间）
        true_alpha = self.time_spanned_alpha(alpha)
        # 调用数字更新函数，传入当前alpha值，获取新的数字值
        new_value = self.number_update_func(true_alpha)
        # 更新数字物体的显示值
        self.mobject.set_value(new_value)


"""
ChangeDecimalToValue类是ChangingDecimal的简化版，专门用于实现数字从当前值到目标值的平滑过渡
核心特点：
1.无需手动定义更新函数，只需指定目标数值
2.自动获取数字当前值作为起始值
3.使用interpolate函数实现线性插值，确保数值变化平滑自然
"""
# 定义ChangeDecimalToValue类，继承自ChangingDecimal，用于实现数字从当前值平滑过渡到目标值的动画
class ChangeDecimalToValue(ChangingDecimal):
    def __init__(
        self,
        decimal_mob: DecimalNumber,  # 要进行数值变化的DecimalNumber对象
        target_number: float | complex,  # 目标数值（可以是浮点数或复数）
        **kwargs  # 其他关键字参数，传递给父类
    ):
        # 获取数字对象当前的数值作为起始值
        start_number = decimal_mob.number
        # 调用父类ChangingDecimal的初始化方法
        super().__init__(
            decimal_mob,
            # 定义数字更新函数：使用插值函数实现从起始值到目标值的平滑过渡
            # a是alpha值（0到1），interpolate函数根据a计算中间值
            lambda a: interpolate(start_number, target_number, a),** kwargs
        )



"""
CountInFrom类是ChangingDecimal的子类，专门实现 "从指定数值计数到当前值" 的动画效果
核心特点：
1.与ChangeDecimalToValue方向相反：从源数值过渡到物体当前的数值
2.源数值默认为 0，可自定义（如从 5 计数到当前的 10）
3.使用clip(a, 0, 1)确保插值比例始终在有效范围内，避免数值异常
4.通过interpolate函数实现平滑的数值过渡
"""
# 定义CountInFrom类，继承自ChangingDecimal，用于实现数字从指定源数值递增/递减到当前值的动画
class CountInFrom(ChangingDecimal):
    def __init__(
        self,
        decimal_mob: DecimalNumber,  # 要进行计数动画的DecimalNumber对象
        source_number: float | complex = 0,  # 起始源数值，默认为0
        **kwargs  # 其他关键字参数，传递给父类
    ):
        # 获取数字对象当前的数值作为目标值（动画结束时的数值）
        start_number = decimal_mob.get_value()
        # 调用父类ChangingDecimal的初始化方法
        super().__init__(
            decimal_mob,
            # 定义数字更新函数：
            # 使用interpolate函数实现从源数值到目标值的平滑过渡
            # clip(a, 0, 1)确保alpha值始终在[0,1]范围内，避免超出边界
            lambda a: interpolate(source_number, start_number, clip(a, 0, 1)),** kwargs
        )
