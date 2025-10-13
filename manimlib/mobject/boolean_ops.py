from __future__ import annotations

import numpy as np
import pathops

from manimlib.mobject.types.vectorized_mobject import VMobject


# Boolean operations between 2D mobjects
# Borrowed from https://github.com/ManimCommunity/manim/

def _convert_vmobject_to_skia_path(vmobject: VMobject) -> pathops.Path:
    """
    将Manim的矢量图形对象（VMobject）转换为Skia路径对象（pathops.Path）。
    核心是解析VMobject的子路径、贝塞尔曲线数据，映射为Skia支持的绘制指令（移动、二次贝塞尔、闭合），生成可用于Skia渲染或路径运算的路径对象。

    参数:
        vmobject (VMobject): 待转换的Manim矢量图形对象（如Rectangle、Tex、自定义图形等）。

    返回:
        pathops.Path: 转换后的Skia路径对象，包含完整的绘制指令和坐标信息。
    """
    # 1. 初始化一个空的Skia路径对象
    skia_path = pathops.Path()

    # 2. 遍历VMobject的所有「含顶点的家族成员」
    # family_members_with_points()：获取VMobject及其所有子对象中，包含顶点数据的对象（过滤无顶点的辅助对象）
    for submob in vmobject.family_members_with_points():
        # 3. 遍历当前子对象的所有子路径（一个图形可能由多个独立子路径组成，如带孔图形）
        for subpath in submob.get_subpaths():
            # 4. 将子路径的顶点转换为「二次贝塞尔曲线元组」
            # Manim内部会将折线、曲线统一处理为二次贝塞尔曲线（p0=起点, p1=控制点, p2=终点）
            bezier_quads = vmobject.get_bezier_tuples_from_points(subpath)
            
            # 5. 提取子路径的起始点，执行Skia的「移动到」指令
            # Skia路径需从起始点开始，且仅需2D坐标（忽略Manim的z轴）
            start_point = subpath[0]
            skia_path.moveTo(*start_point[:2])  # 取x、y坐标，丢弃z坐标（[:2]）
            
            # 6. 遍历所有二次贝塞尔曲线元组，执行Skia的「二次贝塞尔到」指令
            for p0, p1, p2 in bezier_quads:
                # p0：当前段起点（与上一段终点重合，可省略）；p1：控制点；p2：当前段终点
                # Skia的quadTo需传入「控制点坐标、终点坐标」，均取2D值
                skia_path.quadTo(*p1[:2], *p2[:2])
            
            # 7. 判断子路径是否闭合（起点与终点是否重合），若闭合则执行Skia的「闭合路径」指令
            # consider_points_equal：Manim的工具方法，判断两个顶点是否视为同一位置（处理浮点误差）
            if vmobject.consider_points_equal(subpath[0], subpath[-1]):
                skia_path.close()  # 闭合路径：自动连接终点到起点

    # 8. 返回转换后的完整Skia路径对象
    return skia_path


def _convert_skia_path_to_vmobject(
    path: pathops.Path,
    vmobject: VMobject
) -> VMobject:
    """
    将Skia路径（pathops.Path）转换为Manim的矢量图形对象（VMobject）。
    核心是解析Skia路径中的绘制指令（如移动、直线、贝塞尔曲线），并映射为Manim对应的图形绘制方法，最终生成可在Manim中渲染的矢量图形。

    参数:
        path (pathops.Path): Skia格式的路径对象，包含一系列绘制指令（动词）和坐标点。
        vmobject (VMobject): 待填充路径数据的Manim矢量图形对象（如Rectangle、Circle等）。

    返回:
        VMobject: 已填充Skia路径数据的Manim矢量图形对象。
    """
    # 从pathops库导入路径指令枚举（表示不同的绘制动作，如移动、直线、闭合等）
    PathVerb = pathops.PathVerb
    # 记录当前路径的起始点（用于闭合路径时回到起点），初始化为3D坐标（Manim默认坐标系）
    current_path_start = np.array([0.0, 0.0, 0.0])

    # 遍历Skia路径中的每一组「指令-坐标点」对
    for path_verb, points in path:
        # 1. 处理「闭合路径」指令（PathVerb.CLOSE）
        if path_verb == PathVerb.CLOSE:
            # 绘制一条从当前点回到路径起始点的直线，完成闭合
            vmobject.add_line_to(current_path_start)
        
        # 2. 处理非闭合指令（移动、直线、贝塞尔曲线等）
        else:
            # 将Skia的2D坐标点（x,y）转换为Manim的3D坐标（x,y,0）
            # np.hstack：在水平方向拼接数组；np.zeros((len(points), 1))：生成与点数量一致的z轴（0）列
            points = np.hstack((np.array(points), np.zeros((len(points), 1))))

            # 2.1 处理「移动到」指令（PathVerb.MOVE）：开始新路径的起点
            if path_verb == PathVerb.MOVE:
                for point in points:
                    # 更新当前路径的起始点为移动目标点
                    current_path_start = point
                    # 在Manim中启动新路径（后续绘制从该点开始）
                    vmobject.start_new_path(point)
            
            # 2.2 处理「三次贝塞尔曲线」指令（PathVerb.CUBIC）：需要2个控制点+1个终点
            elif path_verb == PathVerb.CUBIC:
                # Manim的add_cubic_bezier_curve_to需要依次传入「控制点1、控制点2、终点」
                # 由于points已转换为3D数组，直接解包传入即可
                vmobject.add_cubic_bezier_curve_to(*points)
            
            # 2.3 处理「直线」指令（PathVerb.LINE）：从当前点绘制到目标点
            elif path_verb == PathVerb.LINE:
                # points中仅含1个目标点，直接传入绘制直线
                vmobject.add_line_to(points[0])
            
            # 2.4 处理「二次贝塞尔曲线」指令（PathVerb.QUAD）：需要1个控制点+1个终点
            elif path_verb == PathVerb.QUAD:
                # Manim的add_quadratic_bezier_curve_to需要依次传入「控制点、终点」
                vmobject.add_quadratic_bezier_curve_to(*points)
            
            # 2.5 处理未支持的指令（抛出异常提示）
            else:
                raise Exception(f"Unsupported Skia path verb: {path_verb}")

    # 反转路径的点顺序（修正Skia与Manim路径方向差异，确保图形填充/描边正确）
    return vmobject.reverse_points()


class Union(VMobject):
    def __init__(self, *vmobjects: VMobject, **kwargs):
        if len(vmobjects) < 2:
            raise ValueError("At least 2 mobjects needed for Union.")
        super().__init__(**kwargs)
        outpen = pathops.Path()
        paths = [
            _convert_vmobject_to_skia_path(vmobject)
            for vmobject in vmobjects
        ]
        pathops.union(paths, outpen.getPen())
        _convert_skia_path_to_vmobject(outpen, self)


class Difference(VMobject):
    """
    用于实现两个矢量图形对象（VMobject）「差集（Difference）运算」的复合图形类。
    差集运算的规则是：从第一个图形（主体）中减去与第二个图形（裁剪图形）重叠的部分，仅保留主体中不重叠的区域。
    核心依赖Skia的路径运算（pathops）实现差集逻辑，并将结果转换为Manim可渲染的VMobject。
    """
    def __init__(self, subject: VMobject, clip: VMobject, **kwargs):
        """
        初始化Difference对象，对主体图形和裁剪图形执行差集运算。

        参数:
            subject (VMobject): 主体图形，差集运算的被减数（将从中减去重叠部分）。
            clip (VMobject): 裁剪图形，差集运算的减数（用于减去重叠部分）。
            **kwargs: 传递给父类VMobject的额外参数（如填充色、边框宽度、透明度等）。
        """
        # 1. 调用父类VMobject的初始化方法，初始化基础图形属性
        super().__init__(**kwargs)

        # 2. 初始化Skia路径对象，用于存储差集运算的结果
        outpen = pathops.Path()

        # 3. 执行差集运算
        # - 先将Manim的VMobject转换为Skia的Path对象
        # - 使用pathops.difference执行差集运算，结果写入outpen
        pathops.difference(
            [_convert_vmobject_to_skia_path(subject)],  # 主体图形的Skia路径（被减数）
            [_convert_vmobject_to_skia_path(clip)],     # 裁剪图形的Skia路径（减数）
            outpen.getPen(),                            # 用于接收运算结果的Skia画笔
        )

        # 4. 将最终的Skia差集结果路径，转换回Manim的VMobject（填充当前Difference对象）
        _convert_skia_path_to_vmobject(outpen, self)


class Intersection(VMobject):
    def __init__(self, *vmobjects: VMobject, **kwargs):
        if len(vmobjects) < 2:
            raise ValueError("At least 2 mobjects needed for Intersection.")
        super().__init__(**kwargs)
        outpen = pathops.Path()
        pathops.intersection(
            [_convert_vmobject_to_skia_path(vmobjects[0])],
            [_convert_vmobject_to_skia_path(vmobjects[1])],
            outpen.getPen(),
        )
        new_outpen = outpen
        for _i in range(2, len(vmobjects)):
            new_outpen = pathops.Path()
            pathops.intersection(
                [outpen],
                [_convert_vmobject_to_skia_path(vmobjects[_i])],
                new_outpen.getPen(),
            )
            outpen = new_outpen
        _convert_skia_path_to_vmobject(outpen, self)


class Exclusion(VMobject):
    """
    用于实现多个矢量图形对象（VMobject）「异或（XOR）运算」的复合图形类。
    异或运算的规则是：保留所有图形的轮廓，但仅填充「仅被一个图形覆盖」的区域（即重叠区域会被剔除，非重叠区域保留）。
    核心依赖Skia的路径运算（pathops）实现多图形的异或逻辑，并将结果转换为Manim可渲染的VMobject。
    """
    def __init__(self, *vmobjects: VMobject, **kwargs):
        """
        初始化Exclusion对象，对传入的多个VMobject执行异或运算。

        参数:
            *vmobjects (VMobject): 可变参数，传入至少2个待执行异或运算的VMobject（如Rectangle、Circle等）。
            **kwargs: 传递给父类VMobject的额外参数（如填充色、边框宽度、透明度等）。

        异常:
            ValueError: 若传入的VMobject数量少于2个，抛出异常（异或运算至少需要2个操作数）。
        """
        # 1. 校验输入：异或运算至少需要2个VMobject
        if len(vmobjects) < 2:
            raise ValueError("At least 2 mobjects needed for Exclusion.")
        
        # 2. 调用父类VMobject的初始化方法，初始化基础图形属性
        super().__init__(**kwargs)

        # 3. 初始化Skia路径对象，用于存储异或运算的中间结果和最终结果
        outpen = pathops.Path()

        # 4. 第一步：对前2个VMobject执行异或运算
        # - 先将2个Manim的VMobject转换为Skia的Path对象
        # - 使用pathops.xor执行异或运算，结果写入outpen
        pathops.xor(
            [_convert_vmobject_to_skia_path(vmobjects[0])],  # 第一个图形的Skia路径
            [_convert_vmobject_to_skia_path(vmobjects[1])],  # 第二个图形的Skia路径
            outpen.getPen(),  # 用于接收运算结果的Skia画笔
        )

        # 5. 后续步骤：对剩余的VMobject依次执行异或运算（累积结果）
        # 若传入超过2个图形，需将前一步的结果与下一个图形继续异或
        new_outpen = outpen  # 临时变量，存储每一步的新结果
        for _i in range(2, len(vmobjects)):
            new_outpen = pathops.Path()  # 重置临时路径，准备接收新运算结果
            pathops.xor(
                [outpen],  # 上一步异或运算的结果（累积路径）
                [_convert_vmobject_to_skia_path(vmobjects[_i])],  # 当前待运算图形的Skia路径
                new_outpen.getPen(),  # 接收新结果的画笔
            )
            outpen = new_outpen  # 更新累积结果，为下一步运算做准备

        # 6. 将最终的Skia异或结果路径，转换回Manim的VMobject（填充当前Exclusion对象）
        _convert_skia_path_to_vmobject(outpen, self)
