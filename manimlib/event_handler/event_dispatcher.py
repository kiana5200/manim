# 从 __future__ 导入 annotations，支持 Python 3.7+ 的前向类型注解
# 允许在类型提示中直接使用尚未定义的类名，提升代码可读性
from __future__ import annotations

# 导入 numpy 并简写为 np，用于后续可能的数值计算（如存储鼠标坐标等）
import numpy as np

# 从 manimlib 的事件处理器模块导入 EventListener 类
# 该类是事件监听的基类，用于定义“监听特定事件并执行对应逻辑”的行为
from manimlib.event_handler.event_listner import EventListener
# 从 manimlib 的事件处理器模块导入 EventType 类/枚举
# 该类定义了 Manim 支持的事件类型（如鼠标点击、键盘按下、窗口resize等），用于标识事件类别
from manimlib.event_handler.event_type import EventType

# 定义 EventDispatcher 类，作为事件调度中枢，管理监听器与事件分发
class EventDispatcher(object):
    # 构造方法，初始化事件监听器存储结构与交互状态变量
    def __init__(self):
        # 初始化事件监听器字典：key为EventType（事件类型），value为对应类型的监听器列表
        # 为每一种EventType预创建空列表，避免后续添加监听器时键不存在
        self.event_listners: dict[
            EventType, list[EventListener]
        ] = {
            event_type: []
            for event_type in EventType
        }
        # 存储当前鼠标位置（3D坐标，默认原点）
        self.mouse_point = np.array((0., 0., 0.))
        # 存储鼠标拖拽时的位置（3D坐标，默认原点）
        self.mouse_drag_point = np.array((0., 0., 0.))
        # 存储当前按下的键盘按键集合（元素为按键的symbol编码，int类型）
        self.pressed_keys: set[int] = set()
        # 存储当前可拖拽对象的监听器列表（仅在鼠标按下时筛选，释放时清空）
        self.draggable_object_listners: list[EventListener] = []

    # 添加事件监听器：将监听器注册到对应事件类型的列表中
    def add_listner(self, event_listner: EventListener):
        # 断言传入的监听器是EventListener实例，确保类型正确
        assert isinstance(event_listner, EventListener)
        # 根据监听器自身的event_type，将其添加到字典对应的列表中
        self.event_listners[event_listner.event_type].append(event_listner)
        return self  # 返回自身以支持链式调用（如dispatcher.add_listner().add_listner()）

    # 移除事件监听器：从对应事件类型的列表中删除指定监听器
    def remove_listner(self, event_listner: EventListener):
        # 断言传入的监听器是EventListener实例，确保类型正确
        assert isinstance(event_listner, EventListener)
        try:
            # 循环删除列表中的监听器（处理同一监听器被多次添加的情况）
            while event_listner in self.event_listners[event_listner.event_type]:
                self.event_listners[event_listner.event_type].remove(event_listner)
        except:
            # 捕获所有异常（如监听器不在列表中），不抛出错误（原注释中已注释掉抛错逻辑）
            pass
        return self  # 返回自身以支持链式调用

    # 事件分发核心方法：根据事件类型，将事件数据传递给对应监听器并执行回调
    def dispatch(self, event_type: EventType, **event_data):
        # 1. 根据事件类型更新交互状态（鼠标位置、按键状态、可拖拽监听器）
        # 若为鼠标移动事件，更新当前鼠标位置（从事件数据中获取point参数）
        if event_type == EventType.MouseMotionEvent:
            self.mouse_point = event_data["point"]
        # 若为鼠标拖拽事件，更新拖拽时的鼠标位置
        elif event_type == EventType.MouseDragEvent:
            self.mouse_drag_point = event_data["point"]
        # 若为键盘按下事件，将按键symbol添加到已按下集合
        elif event_type == EventType.KeyPressEvent:
            self.pressed_keys.add(event_data["symbol"])  # 注：未处理组合键（Modifiers）
        # 若为键盘释放事件，从已按下集合中移除该按键symbol
        elif event_type == EventType.KeyReleaseEvent:
            self.pressed_keys.difference_update({event_data["symbol"]})  # 注：未处理组合键
        # 若为鼠标按下事件，筛选可拖拽的监听器（仅MouseDragEvent类型且鼠标点在其mobject上）
        elif event_type == EventType.MousePressEvent:
            self.draggable_object_listners = [
                listner
                for listner in self.event_listners[EventType.MouseDragEvent]
                if listner.mobject.is_point_touching(self.mouse_point)
            ]
        # 若为鼠标释放事件，清空可拖拽监听器列表（结束拖拽状态）
        elif event_type == EventType.MouseReleaseEvent:
            self.draggable_object_listners = []

        # 初始化事件传播标志：None表示无特殊控制，False表示终止事件传播
        propagate_event = None

        # 2. 根据事件类型分发事件到对应监听器
        # 情况1：鼠标拖拽事件——仅分发给筛选出的可拖拽监听器
        if event_type == EventType.MouseDragEvent:
            for listner in self.draggable_object_listners:
                # 断言监听器类型正确（防御性检查）
                assert isinstance(listner, EventListener)
                # 执行监听器的回调函数，传入其关联的mobject和事件数据
                propagate_event = listner.callback(listner.mobject, event_data)
                # 若回调返回False，终止事件传播（不再分发给后续监听器）
                if propagate_event is not None and propagate_event is False:
                    return propagate_event

        # 情况2：其他鼠标事件（如点击、移动）——分发给对应类型监听器，且鼠标点在其mobject上
        elif event_type.value.startswith('mouse'):
            for listner in self.event_listners[event_type]:
                # 仅当监听器的mobject包含当前鼠标点时，才执行回调
                if listner.mobject.is_point_touching(self.mouse_point):
                    propagate_event = listner.callback(
                        listner.mobject, event_data)
                    # 若回调返回False，终止事件传播
                    if propagate_event is not None and propagate_event is False:
                        return propagate_event

        # 情况3：键盘事件（按下/释放）——分发给所有对应类型的监听器
        elif event_type.value.startswith('key'):
            for listner in self.event_listners[event_type]:
                # 执行所有键盘监听器的回调（无需位置判断）
                propagate_event = listner.callback(listner.mobject, event_data)
                # 若回调返回False，终止事件传播
                if propagate_event is not None and propagate_event is False:
                    return propagate_event

        # 返回事件传播标志（供上层判断是否继续处理）
        return propagate_event

    # 获取所有事件监听器的总数量（遍历字典中所有列表的长度并求和）
    def get_listners_count(self) -> int:
        return sum([len(value) for key, value in self.event_listners.items()])

    # 获取当前鼠标位置（返回3D坐标数组）
    def get_mouse_point(self) -> np.ndarray:
        return self.mouse_point

    # 获取当前鼠标拖拽位置（返回3D坐标数组）
    def get_mouse_drag_point(self) -> np.ndarray:
        return self.mouse_drag_point

    # 检查指定按键是否处于按下状态（传入按键symbol，返回布尔值）
    def is_key_pressed(self, symbol: int) -> bool:
        return (symbol in self.pressed_keys)

    # 重载 += 运算符，使其等同于add_listner（简化监听器添加语法：dispatcher += listner）
    __iadd__ = add_listner
    # 重载 -= 运算符，使其等同于remove_listner（简化监听器移除语法：dispatcher -= listner）
    __isub__ = remove_listner
    # 重载 () 运算符，使其等同于dispatch（简化事件分发语法：dispatcher(event_type, **data)）
    __call__ = dispatch
    # 重载 len() 函数，使其返回监听器总数量（简化计数语法：len(dispatcher)）
    __len__ = get_listners_count
