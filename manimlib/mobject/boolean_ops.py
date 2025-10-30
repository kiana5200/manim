# 从未来版本导入注解功能，允许在类型注解中使用尚未定义的类
from __future__ import annotations

# 导入numpy库，用于数值计算和数组操作
import numpy as np
# 导入pathops库，用于处理路径的布尔运算
import pathops

# 从manimlib库导入VMobject类，这是可矢量化的图形对象基类
from manimlib.mobject.types.vectorized_mobject import VMobject


# 2D图形对象之间的布尔运算
# 代码借鉴自 https://github.com/ManimCommunity/manim/

def _convert_vmobject_to_skia_path(vmobject: VMobject) -> pathops.Path:
    """将VMobject对象转换为skia路径对象"""
    # 创建一个空的skia路径对象
    path = pathops.Path()
    # 遍历所有包含点数据的子对象
    for submob in vmobject.family_members_with_points():
        # 遍历子对象的所有子路径
        for subpath in submob.get_subpaths():
            # 将点转换为贝塞尔曲线元组
            quads = vmobject.get_bezier_tuples_from_points(subpath)
            # 获取路径的起始点
            start = subpath[0]
            # 移动到起始点
            path.moveTo(*start[:2])  # 只取x和y坐标
            # 遍历所有贝塞尔曲线段并添加到路径
            for p0, p1, p2 in quads:
                # 添加二次贝塞尔曲线
                path.quadTo(*p1[:2], *p2[:2])
            # 如果路径是闭合的（起点和终点相同）
            if vmobject.consider_points_equal(subpath[0], subpath[-1]):
                # 闭合路径
                path.close()
    # 返回转换后的skia路径
    return path


def _convert_skia_path_to_vmobject(
    path: pathops.Path,
    vmobject: VMobject
) -> VMobject:
    """将skia路径对象转换回VMobject对象"""
    # 获取路径操作动词的枚举类
    PathVerb = pathops.PathVerb
    # 记录当前路径的起始点（初始化一个3D点，z坐标为0）
    current_path_start = np.array([0.0, 0.0, 0.0])
    # 遍历路径中的所有操作和点
    for path_verb, points in path:
        # 如果是闭合路径操作
        if path_verb == PathVerb.CLOSE:
            # 添加一条线连接到当前路径的起始点，闭合路径
            vmobject.add_line_to(current_path_start)
        else:
            # 将点转换为3D坐标（添加z=0）
            points = np.hstack((np.array(points), np.zeros((len(points), 1))))
            # 如果是移动操作
            if path_verb == PathVerb.MOVE:
                # 遍历所有移动点
                for point in points:
                    # 更新当前路径的起始点
                    current_path_start = point
                    # 开始新的路径
                    vmobject.start_new_path(point)
            # 如果是三次贝塞尔曲线操作
            elif path_verb == PathVerb.CUBIC:
                # 添加三次贝塞尔曲线
                vmobject.add_cubic_bezier_curve_to(*points)
            # 如果是直线操作
            elif path_verb == PathVerb.LINE:
                # 添加直线到指定点
                vmobject.add_line_to(points[0])
            # 如果是二次贝塞尔曲线操作
            elif path_verb == PathVerb.QUAD:
                # 添加二次贝塞尔曲线
                vmobject.add_quadratic_bezier_curve_to(*points)
            else:
                # 抛出异常处理不支持的路径操作
                raise Exception(f"Unsupported: {path_verb}")
    # 反转点的顺序并返回VMobject对象
    return vmobject.reverse_points()


class Union(VMobject):
    """用于计算多个VMobject的并集的类"""
    def __init__(self, *vmobjects: VMobject, **kwargs):
        # 检查输入的VMobject数量是否至少为2个
        if len(vmobjects) < 2:
            raise ValueError("Union操作至少需要2个图形对象。")
        # 调用父类VMobject的初始化方法
        super().__init__(** kwargs)
        # 创建一个输出路径对象
        outpen = pathops.Path()
        # 将所有输入的VMobject转换为skia路径
        paths = [
            _convert_vmobject_to_skia_path(vmobject)
            for vmobject in vmobjects
        ]
        # 执行并集布尔运算
        pathops.union(paths, outpen.getPen())
        # 将运算结果转换回VMobject并赋值给当前实例
        _convert_skia_path_to_vmobject(outpen, self)


class Difference(VMobject):
    """用于    用于计算两个图形对象的差集的类
    差集结果为：第一个对象中减去与第二个对象重叠的部分
    """
    def __init__(self, subject: VMobject, clip: VMobject, **kwargs):
        # 调用父类VMobject的初始化方法，传入关键字参数
        super().__init__(** kwargs)
        # 创建一个输出路径对象，用于存储运算结果
        outpen = pathops.Path()
        # 执行差集布尔运算：subject减去与clip重叠的部分
        pathops.difference(
            [_convert_vmobject_to_skia_path(subject)],  # 被减的主体图形
            [_convert_vmobject_to_skia_path(clip)],     # 用于裁剪的图形
            outpen.getPen(),                            # 存储结果的画笔
        )
        # 将运算得到的skia路径转换回VMobject并赋值给当前实例
        _convert_skia_path_to_vmobject(outpen, self)


class Intersection(VMobject):
    """
    用于计算多个图形对象交集的类
    交集结果为：所有图形对象共同重叠的部分
    """
    def __init__(self, *vmobjects: VMobject, **kwargs):
        # 检查输入的图形对象数量是否至少为2个
        if len(vmobjects) < 2:
            raise ValueError("Intersection操作至少需要2个图形对象。")
        # 调用父类VMobject的初始化方法，传入关键字参数
        super().__init__(** kwargs)
        # 创建一个输出路径对象，用于存储运算结果
        outpen = pathops.Path()
        # 先计算前两个图形对象的交集
        pathops.intersection(
            [_convert_vmobject_to_skia_path(vmobjects[0])],  # 第一个图形
            [_convert_vmobject_to_skia_path(vmobjects[1])],  # 第二个图形
            outpen.getPen(),                                 # 存储结果的画笔
        )
        # 初始化新的输出路径为前两个图形的交集结果
        new_outpen = outpen
        # 遍历剩余的图形对象，逐步计算与当前交集结果的交集
        for _i in range(2, len(vmobjects)):
            # 创建新的路径对象存储中间结果
            new_outpen = pathops.Path()
            # 计算当前交集结果与下一个图形的交集
            pathops.intersection(
                [outpen],                                   # 之前的交集结果
                [_convert_vmobject_to_skia_path(vmobjects[_i])],  # 下一个图形
                new_outpen.getPen(),                        # 存储新结果的画笔
            )
            # 更新输出路径为新的交集结果
            outpen = new_outpen
        # 将最终的交集结果转换回VMobject并赋值给当前实例
        _convert_skia_path_to_vmobject(outpen, self)


class Exclusion(VMobject):
    """
    用于计算多个图形对象异或(排斥)的类
    异或结果为：所有图形对象的并集减去它们的交集（即只属于其中一个对象的部分）
    """
    def __init__(self, *vmobjects: VMobject, **kwargs):
        # 检查输入的图形对象数量是否至少为2个
        if len(vmobjects) < 2:
            raise ValueError("Exclusion操作至少需要2个图形对象。")
        # 调用父类VMobject的初始化方法，传入关键字参数
        super().__init__(** kwargs)
        # 创建一个输出路径对象，用于存储运算结果
        outpen = pathops.Path()
        # 先计算前两个图形对象的异或
        pathops.xor(
            [_convert_vmobject_to_skia_path(vmobjects[0])],  # 第一个图形
            [_convert_vmobject_to_skia_path(vmobjects[1])],  # 第二个图形
            outpen.getPen(),                                 # 存储结果的画笔
        )
        # 初始化新的输出路径为前两个图形的异或结果
        new_outpen = outpen
        # 遍历剩余的图形对象，逐步计算与当前异或结果的异或
        for _i in range(2, len(vmobjects)):
            # 创建新的路径对象存储中间结果
            new_outpen = pathops.Path()
            # 计算当前异或结果与下一个图形的异或
            pathops.xor(
                [outpen],                                   # 之前的异或结果
                [_convert_vmobject_to_skia_path(vmobjects[_i])],  # 下一个图形
                new_outpen.getPen(),                        # 存储新结果的画笔
            )
            # 更新输出路径为新的异或结果
            outpen = new_outpen
        # 将最终的异或结果转换回VMobject并赋值给当前实例
        _convert_skia_path_to_vmobject(outpen, self)
