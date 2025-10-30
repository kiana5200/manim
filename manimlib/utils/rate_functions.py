# ManimGL 动画速率函数库：提供多种预定义的时间插值曲线（速率函数），
# 控制动画从开始到结束的进度变化节奏（如匀速、加速、减速、往返等），
# 是实现流畅自然动画效果的核心工具。


from __future__ import annotations

import numpy as np

# 导入贝塞尔曲线工具：用于自定义速率曲线
from manimlib.utils.bezier import bezier

from typing import TYPE_CHECKING

# 仅静态类型检查阶段导入类型注解
if TYPE_CHECKING:
    from typing import Callable  # 函数类型注解


# ------------------------------ 基础速率函数 ------------------------------
def linear(t: float) -> float:
    """
    线性速率函数：动画进度与时间成正比（匀速运动），是最基础的速率曲线。
    
    参数：t - 时间参数（0→开始，1→结束）
    返回：进度值（0→1，与 t 相等）。
    """
    return t


def smooth(t: float) -> float:
    """
    平滑速率函数：在 t=0 和 t=1 处的一阶、二阶导数均为 0，实现“缓入缓出”效果，
    动画开始和结束时速度较慢，中间加速，视觉上更自然（类似物理运动的加速减速）。
    
    等价于贝塞尔曲线 bezier([0, 0, 0, 1, 1, 1]) 的结果。
    
    参数：t - 时间参数（0→1）
    返回：进度值（0→1，平滑过渡）。
    """
    s = 1 - t
    return (t**3) * (10 * s * s + 5 * s * t + t * t)


# ------------------------------ 快速进入/退出速率函数 ------------------------------
def rush_into(t: float) -> float:
    """
    快速进入速率函数：动画前半段快速加速到最大速度，后半段保持缓出，
    适用于物体“冲进来”的效果。
    
    参数：t - 时间参数（0→1）
    返回：进度值（0→1，前半段快速推进）。
    """
    return 2 * smooth(0.5 * t)  # 对前半段时间应用 smooth 并放大


def rush_from(t: float) -> float:
    """
    快速退出速率函数：动画开始时已有初速度，前半段保持缓入，后半段快速减速，
    适用于物体“冲出去”的效果。
    
    参数：t - 时间参数（0→1）
    返回：进度值（0→1，后半段快速推进）。
    """
    return 2 * smooth(0.5 * (t + 1)) - 1  # 对偏移后的时间应用 smooth 并调整范围


def slow_into(t: float) -> float:
    """
    缓慢进入速率函数：基于平方根曲线，开始时缓慢加速，逐渐变快，
    适用于“轻柔进入”的动画（如物体从远处慢慢靠近）。
    
    参数：t - 时间参数（0→1）
    返回：进度值（0→1，初期缓慢）。
    """
    return np.sqrt(1 - (1 - t) * (1 - t))  # 平方根曲线实现缓入


# ------------------------------ 复合速率函数 ------------------------------
def double_smooth(t: float) -> float:
    """
    双平滑速率函数：前半段和后半段分别应用 smooth 函数，整体呈现“缓入-缓出-缓入-缓出”的复合效果，
    适用于需要中间停顿感的动画。
    
    参数：t - 时间参数（0→1）
    返回：进度值（0→1，两段平滑过渡）。
    """
    if t < 0.5:
        return 0.5 * smooth(2 * t)  # 前半段：0→0.5，应用 smooth
    else:
        return 0.5 * (1 + smooth(2 * t - 1))  # 后半段：0.5→1，应用 smooth


def there_and_back(t: float) -> float:
    """
    往返速率函数：动画先从起点到终点（t=0→0.5），再从终点返回起点（t=0.5→1），
    适用于“去而复返”的效果（如物体移动到某处后返回）。
    
    参数：t - 时间参数（0→1）
    返回：进度值（0→1→0，平滑往返）。
    """
    new_t = 2 * t if t < 0.5 else 2 * (1 - t)  # 前半段映射到 0→1，后半段映射到 1→0
    return smooth(new_t)  # 对映射后的时间应用 smooth


def there_and_back_with_pause(t: float, pause_ratio: float = 1. / 3) -> float:
    """
    带停顿的往返速率函数：先到终点（t=0→0.5-停顿比例/2），停顿一段时间，再返回起点，
    适用于需要在终点短暂停留的动画（如物体到达后停留再返回）。
    
    参数：
        t - 时间参数（0→1）；
        pause_ratio - 停顿时间占总时间的比例（默认 1/3）。
    返回：进度值（0→1→1→0，中间有停顿）。
    """
    a = 2. / (1. - pause_ratio)  # 缩放因子，确保运动阶段的速率正常
    if t < 0.5 - pause_ratio / 2:
        return smooth(a * t)  # 前往终点阶段
    elif t < 0.5 + pause_ratio / 2:
        return 1  # 停顿阶段（保持在终点）
    else:
        return smooth(a - a * t)  # 返回起点阶段


# ------------------------------ 带惯性/超调的速率函数 ------------------------------
def running_start(t: float, pull_factor: float = -0.5) -> float:
    """
    启动惯性速率函数：动画开始前有“预拉动”效果（进度略低于 0），模拟物理中的惯性启动，
    适用于需要体现惯性的动画（如物体从静止开始加速，有轻微回退再前进）。
    
    参数：
        t - 时间参数（0→1）；
        pull_factor - 预拉动强度（负值表示回退，绝对值越大效果越明显）。
    返回：进度值（先略低于 0，再平滑过渡到 1）。
    """
    return bezier([0, 0, pull_factor, pull_factor, 1, 1, 1])(t)  # 贝塞尔曲线实现惯性效果


def overshoot(t: float, pull_factor: float = 1.5) -> float:
    """
    超调速率函数：动画超过终点后再回调到终点，模拟物理中的弹性超调，
    适用于需要活泼感的动画（如物体到达后轻微超出再弹回）。
    
    参数：
        t - 时间参数（0→1）；
        pull_factor - 超调强度（>1 表示超过终点，值越大超调越多）。
    返回：进度值（0→pull_factor→1，带超调效果）。
    """
    return bezier([0, 0, pull_factor, pull_factor, 1, 1])(t)  # 贝塞尔曲线实现超调


# ------------------------------ 速率函数修饰器 ------------------------------
def not_quite_there(
    func: Callable[[float], float] = smooth,
    proportion: float = 0.7
) -> Callable[[float], float]:
    """
    不完全到达修饰器：将速率函数的最大进度限制在指定比例（如 0.7），
    适用于“未完成”的动画（如物体移动到目标位置的 70% 处停止）。
    
    参数：
        func - 基础速率函数（默认 smooth）；
        proportion - 最大进度比例（0→1）。
    返回：修饰后的速率函数。
    """
    def result(t):
        return proportion * func(t)  # 缩放基础函数的结果
    return result


def wiggle(t: float, wiggles: float = 2) -> float:
    """
    摆动速率函数：在往返运动的基础上叠加正弦摆动，模拟物体抖动效果，
    适用于“摇晃”“颤抖”类动画（如选中的物体轻微抖动）。
    
    参数：
        t - 时间参数（0→1）；
        wiggles - 摆动次数（默认 2 次）。
    返回：带摆动的进度值（围绕 0→1→0 波动）。
    """
    return there_and_back(t) * np.sin(wiggles * np.pi * t)  # 往返函数 × 正弦摆动


def squish_rate_func(
    func: Callable[[float], float],
    a: float = 0.4,
    b: float = 0.6
) -> Callable[[float], float]:
    """
    压缩速率函数：将基础速率函数的有效区间压缩到 [a, b]，
    - t < a 时进度保持为 func(0)（起始值）；
    - t > b 时进度保持为 func(1)（结束值）；
    - a ≤ t ≤ b 时按比例应用基础函数。
    适用于需要“延迟开始”或“提前结束”的动画（如物体等待一段时间后再运动）。
    
    参数：
        func - 基础速率函数；
        a - 有效区间起始点（0→1）；
        b - 有效区间结束点（a < b ≤1）。
    返回：压缩后的速率函数。
    """
    def result(t):
        if a == b:
            return a  # 区间为空时返回固定值
        elif t < a:
            return func(0)  # 早于有效区间：保持起始进度
        elif t > b:
            return func(1)  # 晚于有效区间：保持结束进度
        else:
            return func((t - a) / (b - a))  # 有效区间内：比例缩放应用基础函数
    return result


def lingering(t: float) -> float:
    """
    延迟速率函数：基于 squish_rate_func，有效区间为 [0, 0.8]，
    动画在前 80% 时间内完成，最后 20% 时间保持静止，适用于“停留”效果。
    
    参数：t - 时间参数（0→1）
    返回：进度值（0→1 在前 80% 时间完成，之后保持 1）。
    """
    return squish_rate_func(lambda t: t, 0, 0.8)(t)  # 线性函数压缩到 [0, 0.8]


def exponential_decay(t: float, half_life: float = 0.1) -> float:
    """
    指数衰减速率函数：进度随时间按指数规律趋近于 1，模拟阻尼衰减效果，
    适用于“逐渐停止”的动画（如物体受阻力减速，最终静止）。
    
    参数：
        t - 时间参数（0→1）；
        half_life - 半衰期（值越小，衰减越快，默认 0.1）。
    返回：进度值（0→1，初期快速增长，逐渐趋于平缓）。
    """
    return 1 - np.exp(-t / half_life)  # 指数衰减公式


# 总结：这些速率函数通过控制动画进度随时间的变化规律，实现了多样化的运动效果，
# 是 Manim 动画系统的核心组成部分，可通过 Animation 类的 rate_func 参数指定，
# 如 Square().animate(run_time=2, rate_func=smooth).shift(RIGHT) 实现平滑移动。