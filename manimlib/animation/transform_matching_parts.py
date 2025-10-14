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


class TransformMatchingParts(AnimationGroup):
    def __init__(
        self,
        source: Mobject,
        target: Mobject,
        matched_pairs: Iterable[tuple[Mobject, Mobject]] = [],
        match_animation: type = Transform,
        mismatch_animation: type = Transform,
        run_time: float = 2,
        lag_ratio: float = 0,
        **kwargs,
    ):
        self.source = source
        self.target = target
        self.match_animation = match_animation
        self.mismatch_animation = mismatch_animation
        self.anim_config = dict(**kwargs)

        # We will progressively build up a list of transforms
        # from pieces in source to those in target. These
        # two lists keep track of which pieces are accounted
        # for so far
        self.source_pieces = source.family_members_with_points()
        self.target_pieces = target.family_members_with_points()
        self.anims = []

        for pair in matched_pairs:
            self.add_transform(*pair)

        # Match any pairs with the same shape
        for pair in self.find_pairs_with_matching_shapes(self.source_pieces, self.target_pieces):
            self.add_transform(*pair)

        # Finally, account for mismatches
        for source_piece in self.source_pieces:
            if any([source_piece in anim.mobject.get_family() for anim in self.anims]):
                continue
            self.anims.append(FadeOutToPoint(
                source_piece, target.get_center(),
                **self.anim_config
            ))
        for target_piece in self.target_pieces:
            if any([target_piece in anim.mobject.get_family() for anim in self.anims]):
                continue
            self.anims.append(FadeInFromPoint(
                target_piece, source.get_center(),
                **self.anim_config
            ))

        super().__init__(
            *self.anims,
            run_time=run_time,
            lag_ratio=lag_ratio,
        )

    def add_transform(
        self,
        source: Mobject,
        target: Mobject,
    ):
        new_source_pieces = source.family_members_with_points()
        new_target_pieces = target.family_members_with_points()
        if len(new_source_pieces) == 0 or len(new_target_pieces) == 0:
            # Don't animate null sorces or null targets
            return
        source_is_new = all(char in self.source_pieces for char in new_source_pieces)
        target_is_new = all(char in self.target_pieces for char in new_target_pieces)
        if not source_is_new or not target_is_new:
            return

        transform_type = self.mismatch_animation 
        if source.has_same_shape_as(target):
            transform_type = self.match_animation

        self.anims.append(transform_type(source, target, **self.anim_config))
        for char in new_source_pieces:
            self.source_pieces.remove(char)
        for char in new_target_pieces:
            self.target_pieces.remove(char)

    def find_pairs_with_matching_shapes(
        self,
        chars1: list[Mobject],
        chars2: list[Mobject]
    ) -> list[tuple[Mobject, Mobject]]:
        result = []
        for char1, char2 in it.product(chars1, chars2):
            if char1.has_same_shape_as(char2):
                result.append((char1, char2))
        return result

    def clean_up_from_scene(self, scene: Scene) -> None:
        super().clean_up_from_scene(scene)
        scene.remove(self.mobject)
        scene.add(self.target)


class TransformMatchingShapes(TransformMatchingParts):
    """Alias for TransformMatchingParts"""
    pass


class TransformMatchingStrings(TransformMatchingParts):
    def __init__(
        self,
        source: StringMobject,
        target: StringMobject,
        matched_keys: Iterable[str] = [],
        key_map: dict[str, str] = dict(),
        matched_pairs: Iterable[tuple[VMobject, VMobject]] = [],
        **kwargs,
    ):
        matched_pairs = [
            *matched_pairs,
            *self.matching_blocks(source, target, matched_keys, key_map),
        ]

        super().__init__(
            source, target,
            matched_pairs=matched_pairs,
            **kwargs,
        )

    def matching_blocks(
        self,
        source: StringMobject,
        target: StringMobject,
        matched_keys: Iterable[str],
        key_map: dict[str, str]
    ) -> list[tuple[VMobject, VMobject]]:
        syms1 = source.get_symbol_substrings()
        syms2 = target.get_symbol_substrings()
        counts1 = list(map(source.substr_to_path_count, syms1))
        counts2 = list(map(target.substr_to_path_count, syms2))

        # Start with user specified matches
        blocks = [(source[key], target[key]) for key in matched_keys]
        blocks += [(source[key1], target[key2]) for key1, key2 in key_map.items()]

        # Nullify any intersections with those matches in the two symbol lists
        for sub_source, sub_target in blocks:
            for i in range(len(syms1)):
                if source[i] in sub_source.family_members_with_points():
                    syms1[i] = "Null1"
            for j in range(len(syms2)):
                if target[j] in sub_target.family_members_with_points():
                    syms2[j] = "Null2"

        # Group together longest matching substrings
        while True:
            matcher = SequenceMatcher(None, syms1, syms2)
            match = matcher.find_longest_match(0, len(syms1), 0, len(syms2))
            if match.size == 0:
                break

            i1 = sum(counts1[:match.a])
            i2 = sum(counts2[:match.b])
            size = sum(counts1[match.a:match.a + match.size])

            blocks.append((source[i1:i1 + size], target[i2:i2 + size]))

            for i in range(match.size):
                syms1[match.a + i] = "Null1"
                syms2[match.b + i] = "Null2"

        return blocks


class TransformMatchingTex(TransformMatchingStrings):
    """Alias for TransformMatchingStrings"""
    pass
