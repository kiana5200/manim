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

# 定义方向键符号列表，包含左右上下四个方向键
ARROW_SYMBOLS: list[int] = [
    PygletWindowKeys.LEFT,    # 左方向键
    PygletWindowKeys.UP,      # 上方向键
    PygletWindowKeys.RIGHT,   # 右方向键
    PygletWindowKeys.DOWN,    # 下方向键
]

# 定义所有修饰键的组合，包括Ctrl、Command和Shift键
# 使用按位或操作符组合这些修饰键的常量值
ALL_MODIFIERS = PygletWindowKeys.MOD_CTRL | PygletWindowKeys.MOD_COMMAND | PygletWindowKeys.MOD_SHIFT


class InteractiveScene(Scene):
    """
    交互场景类，继承自基础场景类Scene
    
    交互操作说明：
    - 要选择屏幕上的图形对象(mobjects)，按住Ctrl键并移动鼠标以框选区域，
      或者直接点击Ctrl键选择光标下的图形对象
    - 按下Command + t可以切换选择模式：选择场景中的顶级图形对象
      或选择低级别的图形组件
    - 按住'g'键可以抓取选中的对象并移动它们
    - 按住'h'键可以沿水平方向拖动选中的对象
    - 按住'v'键可以沿垂直方向拖动选中的对象
    - 按住't'键可以调整选中对象的大小，按住Shift键可以相对于角点调整大小
    
    快捷键功能：
    - Command + 'c'：将选中对象的ID复制到剪贴板
    - Command + 'v'：粘贴内容，可以是：
        - 复制的图形对象
        - 基于复制的LaTeX代码创建的Tex图形对象
        - 基于复制的文本创建的Text图形对象
    - Command + 'z'：将选中的对象恢复到原始状态
    - Command + 's'：将选中的对象保存到文件
    """
    # 定义选中区域角点的样式配置
    corner_dot_config = dict(
        color=WHITE,           # 颜色为白色
        radius=0.05,           # 半径为0.05
        glow_factor=2.0,       # 发光系数为2.0
    )
    # 选择矩形框的描边颜色
    selection_rectangle_stroke_color = WHITE
    # 选择矩形框的描边宽度
    selection_rectangle_stroke_width = 1.0
    # 调色板颜色，使用MANIM_COLORS预定义颜色集
    palette_colors = MANIM_COLORS
    # 选中对象的微调大小
    selection_nudge_size = 0.05
    # 光标位置显示的配置
    cursor_location_config = dict(
        font_size=24,          # 字体大小为24
        fill_color=GREY_C,     # 填充颜色为灰色C
        num_decimal_places=3,  # 保留3位小数
    )
    # 时间标签的配置
    time_label_config = dict(
        font_size=24,          # 字体大小为24
        fill_color=GREY_C,     # 填充颜色为灰色C
        num_decimal_places=1,  # 保留1位小数
    )
    # 十字准星的宽度
    crosshair_width = 0.2
    # 十字准星的样式配置
    crosshair_style = dict(
        stroke_color=GREY_A,   # 描边颜色为灰色A
        stroke_width=[3, 0, 3],# 描边宽度，可能用于不同部分的宽度设置
    )

    def setup(self):
    """初始化交互场景的各种组件和状态"""
    # 创建一个组来管理当前选中的所有图形对象
    self.selection = Group()
    # 获取用于高亮显示选中对象的组件
    self.selection_highlight = self.get_selection_highlight()
    # 获取用于框选区域的矩形组件
    self.selection_rectangle = self.get_selection_rectangle()
    # 获取十字准星组件
    self.crosshair = self.get_crosshair()
    # 获取信息标签组件（可能显示光标位置、时间等信息）
    self.information_label = self.get_information_label()
    # 获取颜色选择面板
    self.color_palette = self.get_color_palette()
    
    # 定义不可被选中的对象列表，包括各种交互组件和相机帧
    self.unselectables = [
        self.selection,               # 选中组本身
        self.selection_highlight,     # 高亮显示组件
        self.selection_rectangle,     # 选择矩形框
        self.crosshair,               # 十字准星
        self.information_label,       # 信息标签
        self.camera.frame             # 相机帧
    ]
    
    # 设置选择模式：True表示只选择顶级图形对象，False可能包括子对象
    self.select_top_level_mobs = True
    # 重新生成用于选择的搜索集合（可能是场景中可被选中的对象列表）
    self.regenerate_selection_search_set()

    # 初始化状态变量：是否正在框选
    self.is_selecting = False
    # 初始化状态变量：是否正在拖动选中的对象
    self.is_grabbing = False

    # 将高亮显示组件添加到场景中
    self.add(self.selection_highlight)

def get_selection_rectangle(self):
    """创建并返回用于框选的矩形组件"""
    rect = Rectangle(
        # 使用预定义的描边颜色
        stroke_color=self.selection_rectangle_stroke_color,
        # 使用预定义的描边宽度
        stroke_width=self.selection_rectangle_stroke_width,
    )
    # 将矩形固定在场景帧中，不受相机移动影响
    rect.fix_in_frame()
    # 设置矩形的固定角点（初始为原点）
    rect.fixed_corner = ORIGIN
    # 为矩形添加更新器，使其跟随鼠标移动更新位置和大小
    rect.add_updater(self.update_selection_rectangle)
    return rect

def update_selection_rectangle(self, rect: Rectangle):
    """更新选择矩形的位置和大小，使其从固定角点延伸到当前鼠标位置"""
    # 获取矩形的固定角点（框选的起点）
    p1 = rect.fixed_corner
    # 将当前鼠标位置转换为固定帧坐标
    p2 = self.frame.to_fixed_frame_point(self.mouse_point.get_center())
    # 根据两个角点设置矩形的四个顶点，形成矩形
    rect.set_points_as_corners([
        p1,                          # 起点（固定角）
        np.array([p2[0], p1[1], 0]), # 起点右侧水平点
        p2,                          # 终点（鼠标位置）
        np.array([p1[0], p2[1], 0]), # 起点上方垂直点
        p1,                          # 回到起点，闭合矩形
    ])
    return rect

def get_selection_highlight(self):
    """创建并返回用于高亮显示选中对象的组"""
    result = Group()
    # 存储被跟踪的图形对象（即选中的对象）
    result.tracked_mobjects = []
    # 添加更新器，当选中对象变化时更新高亮显示
    result.add_updater(self.update_selection_highlight)
    return result

def update_selection_highlight(self, highlight: Mobject):
    """更新高亮显示组件，使其与当前选中的对象保持一致"""
    # 如果跟踪的对象与当前选中的对象相同，则无需更新
    if set(highlight.tracked_mobjects) == set(self.selection):
        return

    # 否则，刷新高亮显示的内容
    # 更新跟踪的对象列表为当前选中的对象
    highlight.tracked_mobjects = list(self.selection)
    # 为每个选中的对象创建高亮显示，并设置为高亮组的子对象
    highlight.set_submobjects([
        self.get_highlight(mob) for mob in self.selection
    ])
    
    try:
        # 找到选中对象在场景对象列表中的位置，用于正确排序显示层级
        index = min((
            i for i, mob in enumerate(self.mobjects)
            for sm in self.selection
            if sm in mob.get_family()
        ))
        # 将高亮显示组件移动到选中对象的下方，避免遮挡
        self.mobjects.remove(highlight)
        self.mobjects.insert(index - 1, highlight)
    except ValueError:
        # 如果找不到对应位置，则不做处理
        pass
def get_crosshair(self):
    """创建并返回十字准星组件，用于显示鼠标在场景中的位置"""
    # 创建一个向量对象组，并复制出2个线条（水平和垂直）
    lines = VMobject().replicate(2)
    # 设置第一条线为水平线：从左到原点再到右
    lines[0].set_points([LEFT, ORIGIN, RIGHT])
    # 设置第二条线为垂直线：从上到原点再到下
    lines[1].set_points([UP, ORIGIN, DOWN])
    # 将两条线组合成十字准星
    crosshair = VGroup(*lines)

    # 设置十字准星的宽度
    crosshair.set_width(self.crosshair_width)
    # 应用预定义的十字准星样式（颜色、线宽等）
    crosshair.set_style(** self.crosshair_style)
    # 设置十字准星为可动画状态
    crosshair.set_animating_status(True)
    # 将十字准星固定在场景帧中，不受相机移动影响
    crosshair.fix_in_frame()
    return crosshair

def get_color_palette(self):
    """创建并返回颜色选择面板，用于快速选择对象颜色"""
    # 创建一个包含多个颜色方块的组，每个方块对应一种预定义颜色
    palette = VGroup(*(
        # 为每种颜色创建一个填充方块，无描边
        Square(fill_color=color, fill_opacity=1, side_length=1)
        for color in self.palette_colors  # 使用预定义的调色板颜色
    ))
    # 去除所有方块的描边
    palette.set_stroke(width=0)
    # 将颜色方块水平排列，间距为0.5
    palette.arrange(RIGHT, buff=0.5)
    # 设置整个调色板的宽度为场景宽度减0.5，适应场景
    palette.set_width(FRAME_WIDTH - 0.5)
    # 将调色板放置在场景底部边缘，保留小间距
    palette.to_edge(DOWN, buff=SMALL_BUFF)
    # 将调色板固定在场景帧中，不受相机移动影响
    palette.fix_in_frame()
    return palette

def get_information_label(self):
    """创建并返回信息标签组件，显示鼠标坐标和时间信息"""
    # 创建3个十进制数字显示组件，用于显示X、Y、Z坐标
    loc_label = VGroup(*(
        DecimalNumber(**self.cursor_location_config)  # 使用预定义的光标位置样式
        for n in range(3)
    ))

    def update_coords(loc_label):
        """更新坐标标签的回调函数，实时显示鼠标当前位置"""
        # 遍历坐标标签和鼠标位置的三维坐标值
        for mob, coord in zip(loc_label, self.mouse_point.get_location()):
            # 更新数字显示为当前坐标值
            mob.set_value(coord)
        # 将三个坐标标签水平排列，间距为标签高度
        loc_label.arrange(RIGHT, buff=loc_label.get_height())
        # 将坐标标签放置在场景右下角，保留小间距
        loc_label.to_corner(DR, buff=SMALL_BUFF)
        # 固定坐标标签在场景帧中
        loc_label.fix_in_frame()
        return loc_label

    # 为坐标标签添加更新器，使其实时更新
    loc_label.add_updater(update_coords)

    # 创建时间标签，初始值为0，使用预定义的时间标签样式
    time_label = DecimalNumber(0,** self.time_label_config)
    # 将时间标签放置在场景左下角，保留小间距
    time_label.to_corner(DL, buff=SMALL_BUFF)
    # 固定时间标签在场景帧中
    time_label.fix_in_frame()
    # 为时间标签添加更新器，每帧增加流逝的时间（dt）
    time_label.add_updater(lambda m, dt: m.increment_value(dt))

    # 将坐标标签和时间标签组合成一个信息标签组并返回
    return VGroup(loc_label, time_label)

    # 重写父类Scene的方法（Overrides）
def get_state(self):
    """
    重写父类方法，获取当前场景状态
    作用：保存场景状态时，排除交互相关的临时组件（避免这些组件被序列化或恢复）
    """
    # 创建场景状态对象，忽略高亮框、选择矩形、十字准星这三个交互组件
    return SceneState(self, ignore=[
        self.selection_highlight,  # 选中对象的高亮显示组件
        self.selection_rectangle,  # 鼠标框选的矩形组件
        self.crosshair,            # 十字准星组件
    ])

def restore_state(self, scene_state: SceneState):
    """
    重写父类方法，恢复场景状态
    作用：从保存的状态恢复后，重新将高亮组件添加到场景最底层（避免被其他对象遮挡）
    """
    # 先调用父类的状态恢复方法，完成基础场景对象的恢复
    super().restore_state(scene_state)
    # 将高亮组件插入到场景对象列表的第一个位置（最底层），确保能正常显示
    self.mobjects.insert(0, self.selection_highlight)

def add(self, *mobjects: Mobject):
    """
    重写父类方法，向场景添加图形对象
    扩展：添加对象后重新生成可选择对象集合（确保新添加的对象能被选中）
    """
    # 调用父类的add方法，完成对象的基础添加逻辑
    super().add(*mobjects)
    # 重新生成可选择对象集合（更新场景中可被选中的对象列表）
    self.regenerate_selection_search_set()

def remove(self, *mobjects: Mobject):
    """
    重写父类方法，从场景移除图形对象
    扩展：移除对象后重新生成可选择对象集合（确保已移除的对象不再被选中）
    """
    # 调用父类的remove方法，完成对象的基础移除逻辑
    super().remove(*mobjects)
    # 重新生成可选择对象集合（更新场景中可被选中的对象列表）
    self.regenerate_selection_search_set()

def remove_all_except(self, *mobjects_to_keep : Mobject):
    """
    重写父类方法，移除场景中除指定对象外的所有对象
    扩展：移除后重新生成可选择对象集合（确保剩余对象能正常被选中）
    """
    # 调用父类的remove_all_except方法，保留指定对象并移除其他对象
    super().remove_all_except(*mobjects_to_keep)
    # 重新生成可选择对象集合（更新场景中可被选中的对象列表）
    self.regenerate_selection_search_set()


# 与选择功能相关的方法（Related to selection）
def toggle_selection_mode(self):
    """
    切换选择模式（顶级对象/低级组件）
    作用：在“选择场景顶级对象”和“选择对象的子组件”之间切换
    """
    # 反转选择模式标志（True ↔ False）
    self.select_top_level_mobs = not self.select_top_level_mobs
    # 刷新当前选中对象的范围（适配新的选择模式）
    self.refresh_selection_scope()
    # 重新生成可选择对象集合（适配新的选择模式）
    self.regenerate_selection_search_set()

def get_selection_search_set(self) -> list[Mobject]:
    """
    获取当前可选择的对象集合
    返回：根据选择模式筛选后的可选择对象列表
    """
    return self.selection_search_set

def regenerate_selection_search_set(self):
    """
    重新生成可选择对象集合
    逻辑：1. 先过滤掉不可选中的对象；2. 根据选择模式决定是否包含子组件
    """
    # 第一步：筛选场景中所有“可选中”的对象（排除unselectables列表中的组件）
    selectable = list(filter(
        lambda m: m not in self.unselectables,  # 排除不可选中的组件
        self.mobjects  # 场景中所有对象
    ))
    
    # 第二步：根据选择模式决定可选择对象的范围
    if self.select_top_level_mobs:
        # 顶级模式：可选择对象 = 筛选后的顶级对象（不包含子组件）
        self.selection_search_set = selectable
    else:
        # 低级模式：可选择对象 = 所有顶级对象的子组件（包含嵌套的子对象）
        self.selection_search_set = [
            submob  # 子组件
            for mob in selectable  # 遍历每个可选中的顶级对象
            for submob in mob.family_members_with_points()  # 获取该对象的所有带坐标的子组件
        ]

def refresh_selection_scope(self):
    """
    刷新当前选中对象的范围
    作用：切换选择模式后，调整已选中的对象（确保符合新模式的层级规则）
    """
    # 保存当前选中的对象列表（临时变量）
    curr = list(self.selection)
    
    if self.select_top_level_mobs:
        # 切换到顶级模式：将选中的子组件替换为其所属的顶级对象
        self.selection.set_submobjects([
            mob  # 顶级对象
            for mob in self.mobjects  # 遍历场景中所有对象
            # 筛选条件：该顶级对象的家族中包含当前选中的子组件
            if any(sm in mob.get_family() for sm in curr)
        ])
        # 刷新选中对象组的边界框（递归处理子对象，确保边界框准确）
        self.selection.refresh_bounding_box(recurse_down=True)
    else:
        # 切换到低级模式：提取当前选中对象的所有子组件（排除无意义的空对象）
        self.selection.set_submobjects(
            extract_mobject_family_members(
                curr,  # 当前选中的对象
                exclude_pointless=True,  # 排除无坐标、无意义的子对象
            )
        )

def get_corner_dots(self, mobject: Mobject) -> Mobject:
    """
    为指定对象创建“角点标记点”
    作用：选中对象时，在对象的四个角（或三维顶点）显示小 dots，用于标识选中状态
    参数：mobject - 需要添加角点标记的图形对象
    返回：包含所有角点标记的DotCloud对象
    """
    # 创建角点标记的集合（使用预定义的角点样式配置）
    dots = DotCloud(**self.corner_dot_config)
    # 获取角点标记的半径（从配置中提取）
    radius = float(self.corner_dot_config["radius"])
    
    # 判断对象是2D还是3D（通过深度值判断：深度<0.01视为2D）
    if mobject.get_depth() < 1e-2:
        # 2D对象：使用四个对角方向（左下、左上、右上、右下）
        vects = [DL, UL, UR, DR]
    else:
        # 3D对象：生成8个三维顶点方向（x/y/z轴各±1的组合）
        vects = np.array(list(it.product(*3 * [[-1, 1]])))
    
    # 为角点标记添加更新器：实时跟随对象的位置变化
    dots.add_updater(lambda d: d.set_points([
        # 计算每个角点的位置：对象的角点坐标 + 方向*半径（避免标记与对象重叠）
        mobject.get_corner(v) + v * radius
        for v in vects
    ]))
    return dots

def get_highlight(self, mobject: Mobject) -> Mobject:
    """
    为指定对象创建“选中高亮效果”
    作用：根据对象类型生成不同的高亮样式（2D子组件用边框高亮，其他用角点标记）
    参数：mobject - 需要添加高亮的图形对象
    返回：高亮效果对应的图形对象
    """
    # 情况1：2D向量对象（有坐标）且处于“低级选择模式”→ 用边框高亮
    if isinstance(mobject, VMobject) and mobject.has_points() and not self.select_top_level_mobs:
        # 计算对象的最大尺寸（高度和宽度中的较大值）
        length = max([mobject.get_height(), mobject.get_width()])
        # 创建边框高亮对象（VHighlight）
        result = VHighlight(
            mobject,  # 目标对象
            # 计算最大边框宽度：50*对象尺寸（但不超过10，避免过大）
            max_stroke_addition=min([50 * length, 10]),
        )
        # 添加更新器：实时跟随目标对象的位置/形状变化
        result.add_updater(lambda m: m.replace(mobject, stretch=True))
        return result
    
    # 情况2：DotCloud对象（本身是点集合）→ 不添加高亮（避免重复标记）
    elif isinstance(mobject, DotCloud):
        return Mobject()  # 返回空对象（无高亮）
    
    # 情况3：其他对象（顶级对象、3D对象等）→ 用角点标记高亮
    else:
        return self.get_corner_dots(mobject)

def add_to_selection(self, *mobjects: Mobject):
    """
    将对象添加到选中集合
    作用：筛选出“可选中且未被选中”的对象，加入选中组并标记为“可动画”
    参数：*mobjects - 需要添加到选中集合的一个或多个图形对象
    """
    # 筛选条件：1. 不在不可选中列表；2. 不在已选中列表
    mobs = list(filter(
        lambda m: m not in self.unselectables and m not in self.selection,
        mobjects
    ))
    
    # 若没有符合条件的对象，直接返回（避免无效操作）
    if len(mobs) == 0:
        return
    
    # 将筛选后的对象添加到选中组
    self.selection.add(*mobs)
    # 标记这些对象为“可动画”（确保后续操作能触发动画效果）
    for mob in mobs:
        mob.set_animating_status(True)

def toggle_from_selection(self, *mobjects: Mobject):
    """
    切换对象的选中状态（选中↔未选中）
    作用：对每个对象，如果已选中则移除，未选中则添加
    参数：*mobjects - 需要切换选中状态的一个或多个图形对象
    """
    for mob in mobjects:
        if mob in self.selection:
            # 已选中：从选中组移除，标记为“不可动画”，刷新边界框
            self.selection.remove(mob)
            mob.set_animating_status(False)
            mob.refresh_bounding_box()
        else:
            # 未选中：添加到选中组（复用add_to_selection的筛选逻辑）
            self.add_to_selection(mob)

def clear_selection(self):
    """
    清空当前选中集合
    作用：移除所有选中对象，恢复其初始状态（不可动画+刷新边界框）
    """
    for mob in self.selection:
        # 标记对象为“不可动画”
        mob.set_animating_status(False)
        # 刷新对象的边界框（确保后续渲染准确）
        mob.refresh_bounding_box()
    # 清空选中组的子对象（彻底清空选中状态）
    self.selection.set_submobjects([])

def disable_interaction(self, *mobjects: Mobject):
    """
    禁用指定对象的交互（使其不可选中）
    作用：将对象及其所有子组件加入“不可选中列表”，并更新可选择集合
    参数：*mobjects - 需要禁用交互的一个或多个图形对象
    """
    for mob in mobjects:
        # 遍历对象的所有子组件（包含嵌套子对象）
        for sm in mob.get_family():
            # 将子组件加入不可选中列表
            self.unselectables.append(sm)
    # 重新生成可选择对象集合（确保禁用的对象不再出现在可选择列表中）
    self.regenerate_selection_search_set()

def enable_interaction(self, *mobjects: Mobject):
    """
    启用指定对象的交互（使其可选中）
    作用：将对象及其所有子组件从“不可选中列表”移除（恢复可选中状态）
    参数：*mobjects - 需要启用交互的一个或多个图形对象
    """
    for mob in mobjects:
        # 遍历对象的所有子组件（包含嵌套子对象）
        for sm in mob.get_family():
            # 如果子组件在不可选中列表中，将其移除
            if sm in self.unselectables:
                self.unselectables.remove(sm)
    # 注：此处未调用regenerate_selection_search_set()，需确保外部调用时手动更新，或后续操作触发更新

    # Functions for keyboard actions

    def copy_selection(self):
    """
    将当前选中对象的标识复制到剪贴板
    逻辑：优先获取对象在IPython环境中的变量名，若无则用对象ID，最终以逗号分隔存入剪贴板
    """
    # 存储选中对象标识的列表（变量名或ID）
    names = []
    # 获取当前IPython交互环境（用于查找对象对应的变量名）
    shell = get_ipython()
    
    # 遍历每个选中的对象
    for mob in self.selection:
        # 默认用对象的ID作为标识（转为字符串）
        name = str(id(mob))
        # 若存在IPython环境（非脚本运行模式）
        if shell is None:
            continue
        # 遍历IPython用户命名空间中的所有变量（key=变量名，value=变量值）
        for key, value in shell.user_ns.items():
            # 若变量值就是当前对象（内存地址一致），则用变量名作为标识
            if mob is value:
                name = key
        # 将当前对象的标识加入列表
        names.append(name)
    
    # 将所有标识用逗号连接成字符串，复制到系统剪贴板
    pyperclip.copy(", ".join(names))

def paste_selection(self):
    """
    从剪贴板粘贴内容，生成对应图形对象并添加到场景
    优先级：1. 粘贴已复制的图形对象（通过ID）；2. 粘贴LaTeX代码生成Tex对象；3. 粘贴普通文本生成Text对象
    """
    # 获取剪贴板中的字符串内容
    clipboard_str = pyperclip.paste()
    
    # 尝试第一种情况：粘贴已复制的图形对象（剪贴板内容为对象ID列表）
    try:
        # 将剪贴板字符串按逗号分割，转为整数ID列表
        ids = map(int, clipboard_str.split(","))
        # 根据ID查找对应的图形对象（需依赖self.id_to_mobject方法）
        mobs = map(self.id_to_mobject, ids)
        # 对找到的非空对象创建副本（避免修改原对象）
        mob_copies = [m.copy() for m in mobs if m is not None]
        
        # 清空当前选中状态
        self.clear_selection()
        # 播放淡入动画：新粘贴的对象从1.5倍缩放大小淡入，持续0.5秒
        self.play(*(
            FadeIn(mc, run_time=0.5, scale=1.5)
            for mc in mob_copies
        ))
        # 将新粘贴的对象加入选中集合
        self.add_to_selection(*mob_copies)
        return  # 粘贴成功，直接返回
    except ValueError:
        # 若剪贴板内容不是整数ID（触发ValueError），则进入后续粘贴逻辑
        pass
    
    # 尝试第二种情况：粘贴LaTeX代码（通过判断是否包含LaTeX特征字符）
    if set("\\^=+").intersection(clipboard_str):  # 特征字符：反斜杠、 caret、等号、加号（LaTeX常用）
        try:
            # 用剪贴板内容创建Tex对象（渲染LaTeX公式）
            new_mob = Tex(clipboard_str)
        except LatexError:
            # 若LaTeX语法错误，直接返回（粘贴失败）
            return
    # 第三种情况：粘贴普通文本
    else:
        # 用剪贴板内容创建Text对象（渲染普通文本）
        new_mob = Text(clipboard_str)
    
    # 清空当前选中状态，添加新对象并选中
    self.clear_selection()
    self.add(new_mob)
    self.add_to_selection(new_mob)

def delete_selection(self):
    """
    删除当前选中的对象
    逻辑：先从场景中移除选中对象，再清空选中状态
    """
    # 从场景中移除所有选中对象
    self.remove(*self.selection)
    # 清空选中集合（重置选中状态）
    self.clear_selection()

def enable_selection(self):
    """
    启用框选功能
    逻辑：标记“正在框选”状态，添加框选矩形到场景，并初始化矩形的固定角点（当前鼠标位置）
    """
    # 设置“正在框选”状态为True
    self.is_selecting = True
    # 将框选矩形添加到场景（开始显示框选区域）
    self.add(self.selection_rectangle)
    # 计算当前鼠标位置在“固定帧”中的坐标（不受相机移动影响）
    fixed_mouse_pos = self.frame.to_fixed_frame_point(
        self.mouse_point.get_center()
    )
    # 将框选矩形的固定角点设为当前鼠标位置（框选起点）
    self.selection_rectangle.fixed_corner = fixed_mouse_pos

def gather_new_selection(self):
    """
    完成框选并获取选中对象
    逻辑：结束框选状态，移除框选矩形，根据矩形区域筛选可选中对象并切换其选中状态
    """
    # 标记“正在框选”状态为False（结束框选）
    self.is_selecting = False
    
    # 若框选矩形仍在场景中（避免重复操作）
    if self.selection_rectangle in self.mobjects:
        # 从场景中移除框选矩形（隐藏框选区域）
        self.remove(self.selection_rectangle)
        # 存储本次框选要添加的对象列表
        additions = []
        
        # 反向遍历可选择对象集合（确保先选中上层对象，避免被下层对象覆盖）
        for mob in reversed(self.get_selection_search_set()):
            # 若当前对象与框选矩形有接触（包含或重叠）
            if self.selection_rectangle.is_touching(mob):
                additions.append(mob)
                # 若框选矩形的弧长极小（接近点击而非拖拽框选），只选一个对象后退出循环
                if self.selection_rectangle.get_arc_length() < 1e-2:
                    break
        
        # 切换这些对象的选中状态（已选中→取消，未选中→选中）
        self.toggle_from_selection(*additions)

def prepare_grab(self):
    """
    准备拖拽选中对象
    逻辑：计算鼠标与选中对象中心的偏移量，标记“正在拖拽”状态（确保拖拽时对象跟随鼠标）
    """
    # 获取当前鼠标在场景中的中心坐标
    mp = self.mouse_point.get_center()
    # 计算鼠标与选中对象中心的偏移量（用于后续拖拽时保持相对位置）
    self.mouse_to_selection = mp - self.selection.get_center()
    # 标记“正在拖拽”状态为True
    self.is_grabbing = True

def prepare_resizing(self, about_corner=False):
    """
    准备调整选中对象的大小
    逻辑：根据是否“相对于角点缩放”，确定缩放基准点、参考向量和初始尺寸（用于后续计算缩放比例）
    参数：about_corner - 布尔值，True表示相对于角点缩放，False表示相对于中心缩放
    """
    # 获取选中对象的中心坐标
    center = self.selection.get_center()
    # 获取当前鼠标在场景中的中心坐标
    mp = self.mouse_point.get_center()
    
    if about_corner:
        # 相对于角点缩放：计算缩放基准点（选中对象上远离鼠标的角点）
        # 逻辑：中心 - 鼠标位置 → 方向向量，对应对象的对角点
        self.scale_about_point = self.selection.get_corner(center - mp)
    else:
        # 相对于中心缩放：缩放基准点为选中对象的中心
        self.scale_about_point = center
    
    # 计算“缩放参考向量”（鼠标位置 - 缩放基准点）
    self.scale_ref_vect = mp - self.scale_about_point
    # 记录选中对象当前的宽度和高度（作为缩放前的初始尺寸）
    self.scale_ref_width = self.selection.get_width()
    self.scale_ref_height = self.selection.get_height()

def toggle_color_palette(self):
    """
    切换颜色调色板的显示/隐藏状态
    逻辑：仅当有对象被选中时生效，显示时保存场景状态，隐藏时直接移除
    """
    # 若当前无选中对象，直接返回（无需显示调色板）
    if len(self.selection) == 0:
        return
    
    # 若调色板未在场景中（当前隐藏）
    if self.color_palette not in self.mobjects:
        # 保存当前场景状态（便于后续恢复）
        self.save_state()
        # 将调色板添加到场景（显示调色板）
        self.add(self.color_palette)
    else:
        # 若调色板已在场景中（当前显示），从场景中移除（隐藏调色板）
        self.remove(self.color_palette)

def display_information(self, show=True):
    """
    控制信息标签（坐标+时间）的显示/隐藏
    参数：show - 布尔值，True显示信息标签，False隐藏
    """
    if show:
        # 显示：将信息标签添加到场景
        self.add(self.information_label)
    else:
        # 隐藏：从场景中移除信息标签
        self.remove(self.information_label)

def group_selection(self):
    """
    将当前选中的多个对象组合成一个Group对象
    逻辑：创建Group并添加选中对象，移除原对象，选中新组合的Group
    """
    # 创建Group对象，包含所有当前选中的对象（需依赖self.get_group方法）
    group = self.get_group(*self.selection)
    # 将新创建的Group添加到场景
    self.add(group)
    # 清空原有的选中状态
    self.clear_selection()
    # 将新Group加入选中集合（后续操作针对整个Group）
    self.add_to_selection(group)

def ungroup_selection(self):
    """
    将当前选中的Group对象拆分为单个子对象
    逻辑：移除原Group，提取所有子对象并添加到场景，选中这些子对象
    """
    # 存储拆分后的子对象列表
    pieces = []
    
    # 遍历当前选中的每个对象（预期为Group）
    for mob in list(self.selection):
        # 从场景中移除原Group对象
        self.remove(mob)
        # 提取Group中的所有子对象，加入pieces列表
        pieces.extend(list(mob))
    
    # 清空原有的选中状态
    self.clear_selection()
    # 将拆分后的子对象添加到场景
    self.add(*pieces)
    # 将这些子对象加入选中集合（后续操作针对单个子对象）
    self.add_to_selection(*pieces)

def nudge_selection(self, vect: np.ndarray, large: bool = False):
    """
    微调当前选中对象的位置
    逻辑：根据方向向量和微调幅度，移动选中对象（支持普通微调与大步微调）
    参数：
        vect - 三维numpy数组，代表微调的方向（如RIGHT、UP等）
        large - 布尔值，True表示大步微调（幅度×10），False表示普通微调
    """
    # 获取基础微调幅度（从类配置中读取）
    nudge = self.selection_nudge_size
    # 若为大步微调，幅度扩大10倍
    if large:
        nudge *= 10
    # 按照“幅度×方向”移动选中对象
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
