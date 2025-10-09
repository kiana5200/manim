# 从__future__模块导入 annotations，用于支持延迟类型注解解析（Python 3.7+ 特性）
from __future__ import annotations

# 导入 itertools 模块并简写为 it，用于创建迭代器和处理迭代相关操作
import itertools as it
# 导入 numpy 库并简写为 np，用于数值计算和数组操作
import numpy as np
# 导入 pyperclip 模块，用于实现剪贴板功能（复制粘贴）
import pyperclip
# 从 IPython 核心模块导入 get_ipython 函数，用于获取当前 IPython 解释器实例
from IPython.core.getipython import get_ipython
# 从 pyglet.window 模块导入 key 并简写为 PygletWindowKeys，用于处理窗口按键事件
from pyglet.window import key as PygletWindowKeys

# 从 manimlib.animation.fading 模块导入 FadeIn 类，用于实现淡入动画效果
from manimlib.animation.fading import FadeIn
# 从 manimlib.config 模块导入 manim_config，用于访问 Manim 的配置参数
from manimlib.config import manim_config
# 从 manimlib.constants 模块导入方向向量常量：左下、下、右下、左、原点、右、左上、上、右上
from manimlib.constants import DL, DOWN, DR, LEFT, ORIGIN, RIGHT, UL, UP, UR
# 从 manimlib.constants 模块导入帧尺寸常量：帧宽度、帧高度、小间距
from manimlib.constants import FRAME_WIDTH, FRAME_HEIGHT, SMALL_BUFF
# 从 manimlib.constants 模块导入圆周率常量
from manimlib.constants import PI
# 从 manimlib.constants 模块导入角度单位常量（度）
from manimlib.constants import DEG
# 从 manimlib.constants 模块导入颜色常量：Manim 颜色字典、白色、灰色A、灰色C
from manimlib.constants import MANIM_COLORS, WHITE, GREY_A, GREY_C
# 从 manimlib.mobject.geometry 模块导入几何图形类：线段
from manimlib.mobject.geometry import Line
# 从 manimlib.mobject.geometry 模块导入几何图形类：矩形
from manimlib.mobject.geometry import Rectangle
# 从 manimlib.mobject.geometry 模块导入几何图形类：正方形
from manimlib.mobject.geometry import Square
# 从 manimlib.mobject.mobject 模块导入组合对象类：Group（用于组合多个可移动对象）
from manimlib.mobject.mobject import Group
# 从 manimlib.mobject.mobject 模块导入基础可移动对象类：Mobject（所有可移动对象的基类）
from manimlib.mobject.mobject import Mobject
# 从 manimlib.mobject.numbers 模块导入十进制数字类：DecimalNumber（用于显示数字）
from manimlib.mobject.numbers import DecimalNumber
# 从 manimlib.mobject.svg.tex_mobject 模块导入 Tex 类（用于显示 LaTeX 公式）
from manimlib.mobject.svg.tex_mobject import Tex
# 从 manimlib.mobject.svg.text_mobject 模块导入 Text 类（用于显示文本）
from manimlib.mobject.svg.text_mobject import Text
# 从 manimlib.mobject.types.dot_cloud 模块导入点云类：DotCloud（用于显示点的集合）
from manimlib.mobject.types.dot_cloud import DotCloud
# 从 manimlib.mobject.types.vectorized_mobject 模块导入向量组合类：VGroup
from manimlib.mobject.types.vectorized_mobject import VGroup
# 从 manimlib.mobject.types.vectorized_mobject 模块导入向量高亮类：VHighlight
from manimlib.mobject.types.vectorized_mobject import VHighlight
# 从 manimlib.mobject.types.vectorized_mobject 模块导入向量可移动对象类：VMobject
from manimlib.mobject.types.vectorized_mobject import VMobject
# 从 manimlib.scene.scene 模块导入场景基类：Scene（所有场景的父类）
from manimlib.scene.scene import Scene
# 从 manimlib.scene.scene 模块导入场景状态类：SceneState（用于管理场景状态）
from manimlib.scene.scene import SceneState
# 从 manimlib.utils.family_ops 模块导入提取对象家族成员的函数
from manimlib.utils.family_ops import extract_mobject_family_members
# 从 manimlib.utils.space_ops 模块导入计算向量范数（长度）的函数
from manimlib.utils.space_ops import get_norm
# 从 manimlib.utils.tex_file_writing 模块导入 LaTeX 错误类（用于处理 LaTeX 相关错误）
from manimlib.utils.tex_file_writing import LatexError

# 从 typing 模块导入 TYPE_CHECKING 常量，用于条件性导入类型注解（仅在类型检查时生效）
from typing import TYPE_CHECKING

# 如果处于类型检查阶段（非运行时执行），导入特定的类型注解
if TYPE_CHECKING:
    # 从 manimlib.typing 模块导入三维向量类型注解
    from manimlib.typing import Vect3

# 从配置中获取"选择"操作的按键绑定，用于在交互模式中选择对象
SELECT_KEY = manim_config.key_bindings.select

# 从配置中获取"取消选择"操作的按键绑定，用于在交互模式中取消选择对象
UNSELECT_KEY = manim_config.key_bindings.unselect

# 从配置中获取"抓取"操作的按键绑定，用于在交互模式中自由拖动对象
GRAB_KEY = manim_config.key_bindings.grab

# 从配置中获取"X轴抓取"操作的按键绑定，用于在交互模式中仅沿X轴拖动对象
X_GRAB_KEY = manim_config.key_bindings.x_grab

# 从配置中获取"Y轴抓取"操作的按键绑定，用于在交互模式中仅沿Y轴拖动对象
Y_GRAB_KEY = manim_config.key_bindings.y_grab

# 定义所有抓取相关的按键列表，包含自由抓取和轴向抓取的按键
GRAB_KEYS = [GRAB_KEY, X_GRAB_KEY, Y_GRAB_KEY]

# 从配置中获取"调整大小"操作的按键绑定（标注为TODO，可能尚未完全实现）
RESIZE_KEY = manim_config.key_bindings.resize  # TODO

# 从配置中获取"颜色调整"操作的按键绑定，用于在交互模式中修改对象颜色
COLOR_KEY = manim_config.key_bindings.color

# 从配置中获取"信息显示"操作的按键绑定，用于在交互模式中显示对象信息
INFORMATION_KEY = manim_config.key_bindings.information

# 从配置中获取"光标控制"操作的按键绑定，用于在交互模式中控制光标行为
CURSOR_KEY = manim_config.key_bindings.cursor

# For keyboard interactions

ARROW_SYMBOLS: list[int] = [
    PygletWindowKeys.LEFT,
    PygletWindowKeys.UP,
    PygletWindowKeys.RIGHT,
    PygletWindowKeys.DOWN,
]

ALL_MODIFIERS = PygletWindowKeys.MOD_CTRL | PygletWindowKeys.MOD_COMMAND | PygletWindowKeys.MOD_SHIFT

# Note, a lot of the functionality here is still buggy and very much a work in progress.


class InteractiveScene(Scene):
    """
    To select mobjects on screen, hold ctrl and move the mouse to highlight a region,
    or just tap ctrl to select the mobject under the cursor.

    Pressing command + t will toggle between modes where you either select top level
    mobjects part of the scene, or low level pieces.

    Hold 'g' to grab the selection and move it around
    Hold 'h' to drag it constrained in the horizontal direction
    Hold 'v' to drag it constrained in the vertical direction
    Hold 't' to resize selection, adding 'shift' to resize with respect to a corner

    Command + 'c' copies the ids of selections to clipboard
    Command + 'v' will paste either:
        - The copied mobject
        - A Tex mobject based on copied LaTeX
        - A Text mobject based on copied Text
    Command + 'z' restores selection back to its original state
    Command + 's' saves the selected mobjects to file
    """
    corner_dot_config = dict(
        color=WHITE,
        radius=0.05,
        glow_factor=2.0,
    )
    selection_rectangle_stroke_color = WHITE
    selection_rectangle_stroke_width = 1.0
    palette_colors = MANIM_COLORS
    selection_nudge_size = 0.05
    cursor_location_config = dict(
        font_size=24,
        fill_color=GREY_C,
        num_decimal_places=3,
    )
    time_label_config = dict(
        font_size=24,
        fill_color=GREY_C,
        num_decimal_places=1,
    )
    crosshair_width = 0.2
    crosshair_style = dict(
        stroke_color=GREY_A,
        stroke_width=[3, 0, 3],
    )

    def setup(self):
        self.selection = Group()
        self.selection_highlight = self.get_selection_highlight()
        self.selection_rectangle = self.get_selection_rectangle()
        self.crosshair = self.get_crosshair()
        self.information_label = self.get_information_label()
        self.color_palette = self.get_color_palette()
        self.unselectables = [
            self.selection,
            self.selection_highlight,
            self.selection_rectangle,
            self.crosshair,
            self.information_label,
            self.camera.frame
        ]
        self.select_top_level_mobs = True
        self.regenerate_selection_search_set()

        self.is_selecting = False
        self.is_grabbing = False

        self.add(self.selection_highlight)

    def get_selection_rectangle(self):
        rect = Rectangle(
            stroke_color=self.selection_rectangle_stroke_color,
            stroke_width=self.selection_rectangle_stroke_width,
        )
        rect.fix_in_frame()
        rect.fixed_corner = ORIGIN
        rect.add_updater(self.update_selection_rectangle)
        return rect

    def update_selection_rectangle(self, rect: Rectangle):
        p1 = rect.fixed_corner
        p2 = self.frame.to_fixed_frame_point(self.mouse_point.get_center())
        rect.set_points_as_corners([
            p1, np.array([p2[0], p1[1], 0]),
            p2, np.array([p1[0], p2[1], 0]),
            p1,
        ])
        return rect

    def get_selection_highlight(self):
        result = Group()
        result.tracked_mobjects = []
        result.add_updater(self.update_selection_highlight)
        return result

    def update_selection_highlight(self, highlight: Mobject):
        if set(highlight.tracked_mobjects) == set(self.selection):
            return

        # Otherwise, refresh contents of highlight
        highlight.tracked_mobjects = list(self.selection)
        highlight.set_submobjects([
            self.get_highlight(mob)
            for mob in self.selection
        ])
        try:
            index = min((
                i for i, mob in enumerate(self.mobjects)
                for sm in self.selection
                if sm in mob.get_family()
            ))
            self.mobjects.remove(highlight)
            self.mobjects.insert(index - 1, highlight)
        except ValueError:
            pass

    def get_crosshair(self):
        lines = VMobject().replicate(2)
        lines[0].set_points([LEFT, ORIGIN, RIGHT])
        lines[1].set_points([UP, ORIGIN, DOWN])
        crosshair = VGroup(*lines)

        crosshair.set_width(self.crosshair_width)
        crosshair.set_style(**self.crosshair_style)
        crosshair.set_animating_status(True)
        crosshair.fix_in_frame()
        return crosshair

    def get_color_palette(self):
        palette = VGroup(*(
            Square(fill_color=color, fill_opacity=1, side_length=1)
            for color in self.palette_colors
        ))
        palette.set_stroke(width=0)
        palette.arrange(RIGHT, buff=0.5)
        palette.set_width(FRAME_WIDTH - 0.5)
        palette.to_edge(DOWN, buff=SMALL_BUFF)
        palette.fix_in_frame()
        return palette

    def get_information_label(self):
        loc_label = VGroup(*(
            DecimalNumber(**self.cursor_location_config)
            for n in range(3)
        ))

        def update_coords(loc_label):
            for mob, coord in zip(loc_label, self.mouse_point.get_location()):
                mob.set_value(coord)
            loc_label.arrange(RIGHT, buff=loc_label.get_height())
            loc_label.to_corner(DR, buff=SMALL_BUFF)
            loc_label.fix_in_frame()
            return loc_label

        loc_label.add_updater(update_coords)

        time_label = DecimalNumber(0, **self.time_label_config)
        time_label.to_corner(DL, buff=SMALL_BUFF)
        time_label.fix_in_frame()
        time_label.add_updater(lambda m, dt: m.increment_value(dt))

        return VGroup(loc_label, time_label)

    # Overrides
    def get_state(self):
        return SceneState(self, ignore=[
            self.selection_highlight,
            self.selection_rectangle,
            self.crosshair,
        ])

    def restore_state(self, scene_state: SceneState):
        super().restore_state(scene_state)
        self.mobjects.insert(0, self.selection_highlight)

    def add(self, *mobjects: Mobject):
        super().add(*mobjects)
        self.regenerate_selection_search_set()

    def remove(self, *mobjects: Mobject):
        super().remove(*mobjects)
        self.regenerate_selection_search_set()

    def remove_all_except(self, *mobjects_to_keep : Mobject):
        super().remove_all_except(*mobjects_to_keep)
        self.regenerate_selection_search_set()

    # Related to selection

    def toggle_selection_mode(self):
        self.select_top_level_mobs = not self.select_top_level_mobs
        self.refresh_selection_scope()
        self.regenerate_selection_search_set()

    def get_selection_search_set(self) -> list[Mobject]:
        return self.selection_search_set

    def regenerate_selection_search_set(self):
        selectable = list(filter(
            lambda m: m not in self.unselectables,
            self.mobjects
        ))
        if self.select_top_level_mobs:
            self.selection_search_set = selectable
        else:
            self.selection_search_set = [
                submob
                for mob in selectable
                for submob in mob.family_members_with_points()
            ]

    def refresh_selection_scope(self):
        curr = list(self.selection)
        if self.select_top_level_mobs:
            self.selection.set_submobjects([
                mob
                for mob in self.mobjects
                if any(sm in mob.get_family() for sm in curr)
            ])
            self.selection.refresh_bounding_box(recurse_down=True)
        else:
            self.selection.set_submobjects(
                extract_mobject_family_members(
                    curr, exclude_pointless=True,
                )
            )

    def get_corner_dots(self, mobject: Mobject) -> Mobject:
        dots = DotCloud(**self.corner_dot_config)
        radius = float(self.corner_dot_config["radius"])
        if mobject.get_depth() < 1e-2:
            vects = [DL, UL, UR, DR]
        else:
            vects = np.array(list(it.product(*3 * [[-1, 1]])))
        dots.add_updater(lambda d: d.set_points([
            mobject.get_corner(v) + v * radius
            for v in vects
        ]))
        return dots

    def get_highlight(self, mobject: Mobject) -> Mobject:
        if isinstance(mobject, VMobject) and mobject.has_points() and not self.select_top_level_mobs:
            length = max([mobject.get_height(), mobject.get_width()])
            result = VHighlight(
                mobject,
                max_stroke_addition=min([50 * length, 10]),
            )
            result.add_updater(lambda m: m.replace(mobject, stretch=True))
            return result
        elif isinstance(mobject, DotCloud):
            return Mobject()
        else:
            return self.get_corner_dots(mobject)

    def add_to_selection(self, *mobjects: Mobject):
        mobs = list(filter(
            lambda m: m not in self.unselectables and m not in self.selection,
            mobjects
        ))
        if len(mobs) == 0:
            return
        self.selection.add(*mobs)
        for mob in mobs:
            mob.set_animating_status(True)

    def toggle_from_selection(self, *mobjects: Mobject):
        for mob in mobjects:
            if mob in self.selection:
                self.selection.remove(mob)
                mob.set_animating_status(False)
                mob.refresh_bounding_box()
            else:
                self.add_to_selection(mob)

    def clear_selection(self):
        for mob in self.selection:
            mob.set_animating_status(False)
            mob.refresh_bounding_box()
        self.selection.set_submobjects([])

    def disable_interaction(self, *mobjects: Mobject):
        for mob in mobjects:
            for sm in mob.get_family():
                self.unselectables.append(sm)
        self.regenerate_selection_search_set()

    def enable_interaction(self, *mobjects: Mobject):
        for mob in mobjects:
            for sm in mob.get_family():
                if sm in self.unselectables:
                    self.unselectables.remove(sm)

    # Functions for keyboard actions

    def copy_selection(self):
        names = []
        shell = get_ipython()
        for mob in self.selection:
            name = str(id(mob))
            if shell is None:
                continue
            for key, value in shell.user_ns.items():
                if mob is value:
                    name = key
            names.append(name)
        pyperclip.copy(", ".join(names))

    def paste_selection(self):
        clipboard_str = pyperclip.paste()
        # Try pasting a mobject
        try:
            ids = map(int, clipboard_str.split(","))
            mobs = map(self.id_to_mobject, ids)
            mob_copies = [m.copy() for m in mobs if m is not None]
            self.clear_selection()
            self.play(*(
                FadeIn(mc, run_time=0.5, scale=1.5)
                for mc in mob_copies
            ))
            self.add_to_selection(*mob_copies)
            return
        except ValueError:
            pass
        # Otherwise, treat as tex or text
        if set("\\^=+").intersection(clipboard_str):  # Proxy to text for LaTeX
            try:
                new_mob = Tex(clipboard_str)
            except LatexError:
                return
        else:
            new_mob = Text(clipboard_str)
        self.clear_selection()
        self.add(new_mob)
        self.add_to_selection(new_mob)

    def delete_selection(self):
        self.remove(*self.selection)
        self.clear_selection()

    def enable_selection(self):
        self.is_selecting = True
        self.add(self.selection_rectangle)
        self.selection_rectangle.fixed_corner = self.frame.to_fixed_frame_point(
            self.mouse_point.get_center()
        )

    def gather_new_selection(self):
        self.is_selecting = False
        if self.selection_rectangle in self.mobjects:
            self.remove(self.selection_rectangle)
            additions = []
            for mob in reversed(self.get_selection_search_set()):
                if self.selection_rectangle.is_touching(mob):
                    additions.append(mob)
                    if self.selection_rectangle.get_arc_length() < 1e-2:
                        break
            self.toggle_from_selection(*additions)

    def prepare_grab(self):
        mp = self.mouse_point.get_center()
        self.mouse_to_selection = mp - self.selection.get_center()
        self.is_grabbing = True

    def prepare_resizing(self, about_corner=False):
        center = self.selection.get_center()
        mp = self.mouse_point.get_center()
        if about_corner:
            self.scale_about_point = self.selection.get_corner(center - mp)
        else:
            self.scale_about_point = center
        self.scale_ref_vect = mp - self.scale_about_point
        self.scale_ref_width = self.selection.get_width()
        self.scale_ref_height = self.selection.get_height()

    def toggle_color_palette(self):
        if len(self.selection) == 0:
            return
        if self.color_palette not in self.mobjects:
            self.save_state()
            self.add(self.color_palette)
        else:
            self.remove(self.color_palette)

    def display_information(self, show=True):
        if show:
            self.add(self.information_label)
        else:
            self.remove(self.information_label)

    def group_selection(self):
        group = self.get_group(*self.selection)
        self.add(group)
        self.clear_selection()
        self.add_to_selection(group)

    def ungroup_selection(self):
        pieces = []
        for mob in list(self.selection):
            self.remove(mob)
            pieces.extend(list(mob))
        self.clear_selection()
        self.add(*pieces)
        self.add_to_selection(*pieces)

    def nudge_selection(self, vect: np.ndarray, large: bool = False):
        nudge = self.selection_nudge_size
        if large:
            nudge *= 10
        self.selection.shift(nudge * vect)

    # Key actions
    def on_key_press(self, symbol: int, modifiers: int) -> None:
        super().on_key_press(symbol, modifiers)
        char = chr(symbol)
        if char == SELECT_KEY and (modifiers & ALL_MODIFIERS) == 0:
            self.enable_selection()
        if char == UNSELECT_KEY:
            self.clear_selection()
        elif char in GRAB_KEYS and (modifiers & ALL_MODIFIERS) == 0:
            self.prepare_grab()
        elif char == RESIZE_KEY and (modifiers & PygletWindowKeys.MOD_SHIFT):
            self.prepare_resizing(about_corner=((modifiers & PygletWindowKeys.MOD_SHIFT) > 0))
        elif symbol == PygletWindowKeys.LSHIFT:
            if self.window.is_key_pressed(ord("t")):
                self.prepare_resizing(about_corner=True)
        elif char == COLOR_KEY and (modifiers & ALL_MODIFIERS) == 0:
            self.toggle_color_palette()
        elif char == INFORMATION_KEY and (modifiers & ALL_MODIFIERS) == 0:
            self.display_information()
        elif char == "c" and (modifiers & (PygletWindowKeys.MOD_COMMAND | PygletWindowKeys.MOD_CTRL)):
            self.copy_selection()
        elif char == "v" and (modifiers & (PygletWindowKeys.MOD_COMMAND | PygletWindowKeys.MOD_CTRL)):
            self.paste_selection()
        elif char == "x" and (modifiers & (PygletWindowKeys.MOD_COMMAND | PygletWindowKeys.MOD_CTRL)):
            self.copy_selection()
            self.delete_selection()
        elif symbol == PygletWindowKeys.BACKSPACE:
            self.delete_selection()
        elif char == "a" and (modifiers & (PygletWindowKeys.MOD_COMMAND | PygletWindowKeys.MOD_CTRL)):
            self.clear_selection()
            self.add_to_selection(*self.mobjects)
        elif char == "g" and (modifiers & (PygletWindowKeys.MOD_COMMAND | PygletWindowKeys.MOD_CTRL)):
            self.group_selection()
        elif char == "g" and (modifiers & (PygletWindowKeys.MOD_COMMAND | PygletWindowKeys.MOD_CTRL | PygletWindowKeys.MOD_SHIFT)):
            self.ungroup_selection()
        elif char == "t" and (modifiers & (PygletWindowKeys.MOD_COMMAND | PygletWindowKeys.MOD_CTRL)):
            self.toggle_selection_mode()
        elif char == "d" and (modifiers & PygletWindowKeys.MOD_SHIFT):
            self.copy_frame_positioning()
        elif char == "c" and (modifiers & PygletWindowKeys.MOD_SHIFT):
            self.copy_cursor_position()
        elif symbol in ARROW_SYMBOLS:
            self.nudge_selection(
                vect=[LEFT, UP, RIGHT, DOWN][ARROW_SYMBOLS.index(symbol)],
                large=(modifiers & PygletWindowKeys.MOD_SHIFT),
            )
        # Adding crosshair
        if char == CURSOR_KEY:
            if self.crosshair in self.mobjects:
                self.remove(self.crosshair)
            else:
                self.add(self.crosshair)
        if char == SELECT_KEY:
            self.add(self.crosshair)

        # Conditions for saving state
        if char in [GRAB_KEY, X_GRAB_KEY, Y_GRAB_KEY, RESIZE_KEY]:
            self.save_state()

    def on_key_release(self, symbol: int, modifiers: int) -> None:
        super().on_key_release(symbol, modifiers)
        if chr(symbol) == SELECT_KEY:
            self.gather_new_selection()
        if chr(symbol) in GRAB_KEYS:
            self.is_grabbing = False
        elif chr(symbol) == INFORMATION_KEY:
            self.display_information(False)
        elif symbol == PygletWindowKeys.LSHIFT and self.window.is_key_pressed(ord(RESIZE_KEY)):
            self.prepare_resizing(about_corner=False)

    # Mouse actions
    def handle_grabbing(self, point: Vect3):
        diff = point - self.mouse_to_selection
        if self.window.is_key_pressed(ord(GRAB_KEY)):
            self.selection.move_to(diff)
        elif self.window.is_key_pressed(ord(X_GRAB_KEY)):
            self.selection.set_x(diff[0])
        elif self.window.is_key_pressed(ord(Y_GRAB_KEY)):
            self.selection.set_y(diff[1])

    def handle_resizing(self, point: Vect3):
        if not hasattr(self, "scale_about_point"):
            return
        vect = point - self.scale_about_point
        if self.window.is_key_pressed(PygletWindowKeys.LCTRL):
            for i in (0, 1):
                scalar = vect[i] / self.scale_ref_vect[i]
                self.selection.rescale_to_fit(
                    scalar * [self.scale_ref_width, self.scale_ref_height][i],
                    dim=i,
                    about_point=self.scale_about_point,
                    stretch=True,
                )
        else:
            scalar = get_norm(vect) / get_norm(self.scale_ref_vect)
            self.selection.set_width(
                scalar * self.scale_ref_width,
                about_point=self.scale_about_point
            )

    def handle_sweeping_selection(self, point: Vect3):
        mob = self.point_to_mobject(
            point,
            search_set=self.get_selection_search_set(),
            buff=SMALL_BUFF
        )
        if mob is not None:
            self.add_to_selection(mob)

    def choose_color(self, point: Vect3):
        # Search through all mobject on the screen, not just the palette
        to_search = [
            sm
            for mobject in self.mobjects
            for sm in mobject.family_members_with_points()
            if mobject not in self.unselectables
        ]
        mob = self.point_to_mobject(point, to_search)
        if mob is not None:
            self.selection.set_color(mob.get_color())
        self.remove(self.color_palette)

    def on_mouse_motion(self, point: Vect3, d_point: Vect3) -> None:
        super().on_mouse_motion(point, d_point)
        self.crosshair.move_to(self.frame.to_fixed_frame_point(point))
        if self.is_grabbing:
            self.handle_grabbing(point)
        elif self.window.is_key_pressed(ord(RESIZE_KEY)):
            self.handle_resizing(point)
        elif self.window.is_key_pressed(ord(SELECT_KEY)) and self.window.is_key_pressed(PygletWindowKeys.LSHIFT):
            self.handle_sweeping_selection(point)

    def on_mouse_drag(
        self,
        point: Vect3,
        d_point: Vect3,
        buttons: int,
        modifiers: int
    ) -> None:
        super().on_mouse_drag(point, d_point, buttons, modifiers)
        self.crosshair.move_to(self.frame.to_fixed_frame_point(point))

    def on_mouse_release(self, point: Vect3, button: int, mods: int) -> None:
        super().on_mouse_release(point, button, mods)
        if self.color_palette in self.mobjects:
            self.choose_color(point)
        else:
            self.clear_selection()

    # Copying code to recreate state
    def copy_frame_positioning(self):
        frame = self.frame
        center = frame.get_center()
        height = frame.get_height()
        angles = frame.get_euler_angles()

        call = f"reorient("
        theta, phi, gamma = (angles / DEG).astype(int)
        call += f"{theta}, {phi}, {gamma}"
        if any(center != 0):
            call += f", {tuple(np.round(center, 2))}"
        if height != FRAME_HEIGHT:
            call += ", {:.2f}".format(height)
        call += ")"
        pyperclip.copy(call)

    def copy_cursor_position(self):
        pyperclip.copy(str(tuple(self.mouse_point.get_center().round(2))))
