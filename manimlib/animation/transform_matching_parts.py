# 从__future__模块导入annotations，支持类型注释的延迟评估
# 允许在类型提示中使用尚未定义的类或函数
from __future__ import annotations

# 导入itertools模块并简写为it，用于创建迭代器和处理迭代相关操作
import itertools as it
# 从difflib模块导入SequenceMatcher，用于字符串序列的相似性比较
from difflib import SequenceMatcher

# 从manimlib.animation.composition导入AnimationGroup
# AnimationGroup用于将多个动画组合在一起同时或按顺序播放
from manimlib.animation.composition import AnimationGroup
# 从manimlib.animation.fading导入FadeInFromPoint
# FadeInFromPoint是从指定点淡入的动画效果
from manimlib.animation.fading import FadeInFromPoint
# 从manimlib.animation.fading导入FadeOutToPoint
# FadeOutToPoint是向指定点淡出的动画效果
from manimlib.animation.transform import Transform
# 从manimlib.mobject.mobject导入Mobject
# Mobject是所有可移动对象的基类
from manimlib.mobject.mobject import Mobject
# 从manimlib.mobject.types.vectorized_mobject导入VMobject
# VMobject是向量图形对象的基类，支持更复杂的图形操作
from manimlib.mobject.svg.string_mobject import StringMobject
# StringMobject用于处理字符串的可移动对象

# 从typing模块导入TYPE_CHECKING常量
# 用于在类型检查阶段执行特定代码，运行时不执行
from typing import TYPE_CHECKING

# 条件判断：仅在类型检查时执行以下代码块
if TYPE_CHECKING:
    # 从typing模块导入Iterable类型，用于标注可迭代对象
    from typing import Iterable
    # 从manimlib.scene.scene导入Scene类
    # Scene是所有场景的基类，用于组织和播放动画
    from manimlib.scene.scene import Scene


# 定义TransformMatchingParts类，继承自AnimationGroup
# 用于实现两个复杂物体（包含多个子部分）之间的智能匹配变换
class TransformMatchingParts(AnimationGroup):
    def __init__(
        self,
        source: Mobject,  # 源物体（变换的起始物体）
        target: Mobject,  # 目标物体（变换的结束物体）
        matched_pairs: Iterable[tuple[Mobject, Mobject]] = [],  # 预定义的匹配对子部分
        match_animation: type = Transform,  # 用于匹配部分的动画类型
        mismatch_animation: type = Transform,  # 用于不匹配部分的动画类型
        run_time: float = 2,  # 动画总运行时间
        lag_ratio: float = 0,  # 子动画之间的延迟比例
        **kwargs,  # 其他动画配置参数
    ):
        self.source = source  # 存储源物体
        self.target = target  # 存储目标物体
        self.match_animation = match_animation  # 存储匹配动画类型
        self.mismatch_animation = mismatch_animation  # 存储不匹配动画类型
        self.anim_config = dict(** kwargs)  # 存储动画配置参数

        # 逐步将逐步构建从源物体部分到目标物体部分的变换列表
        # 这两个列表跟踪到目前为止已处理的部分
        self.source_pieces = source.family_members_with_points()  # 源物体的所有带点的子部分
        self.target_pieces = target.family_members_with_points()  # 目标物体的所有带点的子部分
        self.anims = []  # 存储所有子动画

        # 处理预定义的匹配对
        for pair in matched_pairs:
            self.add_transform(*pair)

        # 匹配所有形状相同的子部分对
        for pair in self.find_pairs_with_matching_shapes(self.source_pieces, self.target_pieces):
            self.add_transform(*pair)

        # 最后，处理不匹配的部分
        # 处理源物体中未匹配的部分：从目标中心淡出
        for source_piece in self.source_pieces:
            # 检查该部分是否已在某个动画中
            if any([source_piece in anim.mobject.get_family() for anim in self.anims]):
                continue
            self.anims.append(FadeOutToPoint(
                source_piece, target.get_center(),
                **self.anim_config
            ))
        
        # 处理目标物体中未匹配的部分：从源中心淡入
        for target_piece in self.target_pieces:
            if any([target_piece in anim.mobject.get_family() for anim in self.anims]):
                continue
            self.anims.append(FadeInFromPoint(
                target_piece, source.get_center(),
                **self.anim_config
            ))

        # 调用父类AnimationGroup的初始化方法，组合所有子动画
        super().__init__(
            *self.anims,
            run_time=run_time,
            lag_ratio=lag_ratio,
        )

    # 添加源子部分到目标子部分的变换动画
    def add_transform(
        self,
        source: Mobject,  # 源子部分
        target: Mobject,  # 目标子部分
    ):
        # 获取源和目标子部分的所有带点家族成员
        new_source_pieces = source.family_members_with_points()
        new_target_pieces = target.family_members_with_points()
        
        # 不处理空的源或目标部分
        if len(new_source_pieces) == 0 or len(new_target_pieces) == 0:
            return
        
        # 检查这些部分是否尚未被处理
        source_is_new = all(char in self.source_pieces for char in new_source_pieces)
        target_is_new = all(char in self.target_pieces for char in new_target_pieces)
        if not source_is_new or not target_is_new:
            return

        # 根据形状是否匹配选择相应的动画类型
        transform_type = self.mismatch_animation 
        if source.has_same_shape_as(target):
            transform_type = self.match_animation

        # 创建变换动画并添加到动画列表
        self.anims.append(transform_type(source, target, **self.anim_config))
        
        # 从待处理列表中移除已处理的部分
        for char in new_source_pieces:
            self.source_pieces.remove(char)
        for char in new_target_pieces:
            self.target_pieces.remove(char)

    # 寻找形状匹配的子部分对
    def find_pairs_with_matching_shapes(
        self,
        chars1: list[Mobject],  # 第一组子部分
        chars2: list[Mobject]  # 第二组子部分
    ) -> list[tuple[Mobject, Mobject]]:
        result = []
        # 检查所有可能的组合，寻找形状匹配的对子
        for char1, char2 in it.product(chars1, chars2):
            if char1.has_same_shape_as(char2):
                result.append((char1, char2))
        return result

    # 从场景中清理动画资源
    def clean_up_from_scene(self, scene: Scene) -> None:
        super().clean_up_from_scene(scene)  # 调用父类清理方法
        scene.remove(self.mobject)  # 移除源物体
        scene.add(self.target)  # 添加目标物体到场景


# 定义TransformMatchingShapes类，继承自TransformMatchingParts
# 作为TransformMatchingParts的别名存在，功能完全相同
class TransformMatchingShapes(TransformMatchingParts):
    """Alias for TransformMatchingParts"""
    pass


# 定义TransformMatchingStrings类，继承自TransformMatchingParts
# 专门用于字符串（StringMobject）之间的匹配变换动画
class TransformMatchingStrings(TransformMatchingParts):
    def __init__(
        self,
        source: StringMobject,  # 源字符串物体
        target: StringMobject,  # 目标字符串物体
        matched_keys: Iterable[str] = [],  # 预定义的匹配字符键（字符串索引或子串）
        key_map: dict[str, str] = dict(),  # 字符映射关系（源字符→目标字符）
        matched_pairs: Iterable[tuple[VMobject, VMobject]] = [],  # 预定义的匹配对子部分
        **kwargs,  # 其他传递给父类的参数
    ):
        # 组合所有匹配对：预定义的匹配对 + 自动计算的匹配块
        matched_pairs = [
            *matched_pairs,  # 原有匹配对
            # 自动计算的匹配块（基于字符、子串匹配）
            *self.matching_blocks(source, target, matched_keys, key_map),
        ]

        # 调用父类TransformMatchingParts的初始化方法
        super().__init__(
            source, target,
            matched_pairs=matched_pairs,** kwargs,
        )

    # 计算源字符串和目标字符串之间的匹配块（连续匹配的子串）
    def matching_blocks(
        self,
        source: StringMobject,  # 源字符串
        target: StringMobject,  # 目标字符串
        matched_keys: Iterable[str],  # 预定义匹配键
        key_map: dict[str, str]  # 字符映射关系
    ) -> list[tuple[VMobject, VMobject]]:
        # 获取源和目标字符串的符号子串列表
        syms1 = source.get_symbol_substrings()
        syms2 = target.get_symbol_substrings()
        
        # 计算每个子串对应的路径数量（用于索引计算）
        counts1 = list(map(source.substr_to_path_count, syms1))
        counts2 = list(map(target.substr_to_path_count, syms2))

        # 开始处理用户指定的匹配
        # 添加matched_keys中指定的匹配对（源和目标使用相同的键）
        blocks = [(source[key], target[key]) for key in matched_keys]
        # 添加key_map中指定的映射关系（源键→目标键）
        blocks += [(source[key1], target[key2]) for key1, key2 in key_map.items()]

        # 将已匹配的部分标记为"Null"，避免重复匹配
        for sub_source, sub_target in blocks:
            # 标记源字符串中已匹配的部分
            for i in range(len(syms1)):
                if source[i] in sub_source.family_members_with_points():
                    syms1[i] = "Null1"
            # 标记目标字符串中已匹配的部分
            for j in range(len(syms2)):
                if target[j] in sub_target.family_members_with_points():
                    syms2[j] = "Null2"

        # 自动匹配最长的连续相同子串
        while True:
            # 使用序列匹配器寻找最长匹配子序列
            matcher = SequenceMatcher(None, syms1, syms2)
            match = matcher.find_longest_match(0, len(syms1), 0, len(syms2))
            
            # 如果没有找到匹配，退出循环
            if match.size == 0:
                break

            # 计算源字符串中匹配块的起始索引和大小
            i1 = sum(counts1[:match.a])
            size = sum(counts1[match.a:match.a + match.size])
            # 计算目标字符串中匹配块的起始索引
            i2 = sum(counts2[:match.b])

            # 添加这个匹配块到列表
            blocks.append((source[i1:i1 + size], target[i2:i2 + size]))

            # 将已匹配的部分标记为"Null"，避免重复匹配
            for i in range(match.size):
                syms1[match.a + i] = "Null1"
                syms2[match.b + i] = "Null2"

        return blocks


# 定义TransformMatchingTex类，继承自TransformMatchingStrings
# 作为TransformMatchingStrings的别名，专门用于Tex字符串的匹配变换
class TransformMatchingTex(TransformMatchingStrings):
    """Alias for TransformMatchingStrings"""
    pass
