# LaTeX 命令与符号数量映射表：定义常见 LaTeX 命令对应的视觉符号数量，
# 用于辅助统计公式中实际显示的符号总数（忽略纯格式命令），是 num_tex_symbols 函数的核心参考数据。

TEX_TO_SYMBOL_COUNT = {
    # 空格与间距命令（不产生可见符号）
    R"\!": 0,    # 负thinspace
    R"\,": 0,    # thinspace
    R"\-": 0,    # 连字符（仅格式）
    R"\/": 0,    # 中等空格
    R"\:": 0,    # 中等空格
    R"\;": 0,    # 厚空格
    R"\>": 0,    # 右对齐空格
    # 特殊字符命令（部分不常用或格式相关）
    R"\aa": 0,   # 带圈a（不常用）
    R"\AA": 0,   # 大写带圈A
    R"\ae": 0,   #  æ 字符
    R"\AE": 0,   #  Æ 字符
    # 三角函数与反三角函数（按字母数计数）
    R"\arccos": 6,  # arccos（6个字母）
    R"\arcsin": 6,  # arcsin（6个字母）
    R"\arctan": 6,  # arctan（6个字母）
    R"\arg": 3,     # arg（3个字母）
    # 文档格式命令（不影响符号显示）
    R"\author": 0,      # 作者信息
    R"\bf": 0,          # 粗体（格式）
    R"\bibliography": 0, # 参考文献
    R"\bibliographystyle": 0, # 参考文献格式
    # 大小括号命令（仅格式，符号数由内容决定）
    R"\big": 0,     # 大括号前缀
    R"\Big": 0,     # 更大括号前缀
    # 大型运算符（按视觉复杂度计数）
    R"\bigodot": 4,   # 大圆圈乘（视觉上4个符号单元）
    R"\bigoplus": 5,  # 大圆圈加（视觉上5个符号单元）
    # 文档布局命令
    R"\bigskip": 0,    # 大间距
    R"\bmod": 3,       # 模运算（b mod，3个符号）
    R"\boldmath": 0,   # 粗体数学模式
    # 分数格式命令
    R"\bottomfraction": 2, # 底部分数占比（格式参数）
    # 特殊符号
    R"\bowtie": 2,     # 领结符号（视觉上2个单元）
    # 字体样式命令
    R"\cal": 0,        # 花体字体
    # 省略号
    R"\cdots": 3,      # 中间省略号（3个点）
    # 对齐命令
    R"\centering": 0,  # 居中对齐
    # 引用命令
    R"\cite": 2,       # 引用标记（如 [1]，计2个符号）
    # 关系符号
    R"\cong": 2,       # 全等符号（视觉上2个单元）
    # 目录相关命令
    R"\contentsline": 0, # 目录行
    # 三角函数
    R"\cos": 3,        # cos（3个字母）
    R"\cosh": 4,       # cosh（4个字母）
    R"\cot": 3,        # cot（3个字母）
    R"\coth": 4,       # coth（4个字母）
    R"\csc": 3,        # csc（3个字母）
    # 日期命令
    R"\date": 0,       # 日期
    # 浮动体格式命令
    R"\dblfloatpagefraction": 2, # 双栏浮动体占比
    R"\dbltopfraction": 2,       # 双栏顶部浮动体占比
    # 省略号
    R"\ddots": 3,      # 对角线省略号（3个点）
    # 数学命令
    R"\deg": 3,        # deg（3个字母）
    R"\det": 3,        # det（3个字母）
    R"\dim": 3,        # dim（3个字母）
    # 显示样式命令
    R"\displaystyle": 0, # 行间公式样式
    # 运算符
    R"\div": 2,        # 除号（视觉上2个单元）
    R"\doteq": 2,      # 约等于（2个单元）
    # 填充命令
    R"\dotfill": 0,    # 点填充
    R"\dots": 3,       # 省略号（3个点）
    # 强调命令
    R"\emph": 0,       # 强调（格式）
    # 指数函数
    R"\exp": 3,        # exp（3个字母）
    # 盒子命令
    R"\fbox": 4,       # 带框盒子（边框计4个单元）
    # 浮动体参数
    R"\floatpagefraction": 2, # 浮动体占页比例
    R"\flushbottom": 0,       # 底部对齐
    # 字体大小命令
    R"\footnotesize": 0, # 小字体
    # 脚注命令
    R"\footnotetext": 0, # 脚注文本
    # 框架命令
    R"\frame": 2,      # 框架（2个单元）
    R"\framebox": 4,   # 带框盒子（4个单元）
    # 排版命令
    R"\fussy": 0,      # 严格排版
    # 最大公约数
    R"\gcd": 3,        # gcd（3个字母）
    # 幽灵命令（占位）
    R"\ghost": 0,      # 幽灵文本（不显示）
    # 词汇表命令
    R"\glossary": 0,   # 词汇表
    # 水平填充
    R"\hfill": 0,      # 水平填充
    # 同态
    R"\hom": 3,        # hom（3个字母）
    # 箭头符号
    R"\hookleftarrow": 2,  # 带钩左箭头（2个单元）
    R"\hookrightarrow": 2, # 带钩右箭头（2个单元）
    # 水平规则填充
    R"\hrulefill": 0,  # 水平线填充
    # 字体大小
    R"\huge": 0,       # 极大字体
    R"\Huge": 0,       # 最大字体
    # 断字命令
    R"\hyphenation": 0, # 断字规则
    # 等价符号
    R"\iff": 2,        # 等价（2个单元）
    # 虚部
    R"\Im": 2,         # Im（2个字母）
    # 索引命令
    R"\index": 0,      # 索引
    # 下确界
    R"\inf": 3,        # inf（3个字母）
    # 斜体命令
    R"\it": 0,         # 斜体
    # 核
    R"\ker": 3,        # ker（3个字母）
    # 左标记命令
    R"\l": 0,          # 左标记
    R"\L": 0,          # 大写左标记
    # 标签命令
    R"\label": 0,      # 标签
    # 字体大小
    R"\large": 0,      # 大字体
    R"\Large": 0,      # 较大字体
    R"\LARGE": 0,      # 很大字体
    # 省略号
    R"\ldots": 3,      # 底线省略号（3个点）
    # 左对齐公式
    R"\lefteqn": 0,    # 左对齐公式
    # 左括号
    R"\left": 0,       # 左括号前缀
    # 对数
    R"\lg": 2,         # lg（2个字母）
    R"\lim": 3,        # lim（3个字母）
    R"\liminf": 6,     # liminf（6个字母）
    R"\limsup": 6,     # limsup（6个字母）
    # 换行命令
    R"\linebreak": 0,  # 强制换行
    # 自然对数
    R"\ln": 2,         # ln（2个字母）
    R"\log": 3,        # log（3个字母）
    # 长箭头
    R"\longleftarrow": 2,      # 长左箭头（2个单元）
    R"\Longleftarrow": 2,      # 很长左箭头（2个单元）
    R"\longleftrightarrow": 2, # 长左右箭头（2个单元）
    R"\Longleftrightarrow": 2, # 很长左右箭头（2个单元）
    R"\longmapsto": 3,         # 长映射箭头（3个单元）
    R"\longrightarrow": 2,     # 长右箭头（2个单元）
    R"\Longrightarrow": 2,     # 很长右箭头（2个单元）
    # 盒子命令
    R"\makebox": 0,    # 自定义盒子
    # 映射箭头
    R"\mapsto": 2,     # 映射箭头（2个单元）
    # 右标记
    R"\markright": 0,  # 右标记
    # mathds字体
    R"\mathds": 0,     # mathds字体
    # 最大值
    R"\max": 3,        # max（3个字母）
    # 文本盒子
    R"\mbox": 0,       # 文本盒子
    # 中等间距
    R"\medskip": 0,    # 中等间距
    # 最小值
    R"\min": 3,        # min（3个字母）
    # 数学斜体
    R"\mit": 0,        # 数学斜体
    # 满足符号
    R"\models": 2,     # 满足符号（2个单元）
    # 不等号
    R"\ne": 2,         # 不等（2个单元）
    R"\neq": 2,        # 不等（2个单元）
    # 换行命令
    R"\newline": 0,    # 换行
    # 无缩进
    R"\noindent": 0,   # 无缩进
    # 不换行
    R"\nolinebreak": 0, # 不换行
    # 无编号
    R"\nonumber": 0,   # 公式无编号
    # 不换页
    R"\nopagebreak": 0, # 不换页
    # 边注位置
    R"\normalmarginpar": 0, # 正常边注位置
    # 正常字体
    R"\normalsize": 0, # 正常字体大小
    # 不属于符号
    R"\notin": 2,      # 不属于（2个单元）
    # 特殊字符
    R"\o": 0,          #  ø 字符
    R"\O": 0,          #  Ø 字符
    # 回车命令
    R"\obeycr": 0,     # 服从回车
    # 特殊字符
    R"\oe": 0,         #  œ 字符
    R"\OE": 0,         #  Œ 字符
    # 上括号
    R"\overbrace": 4,  # 上括号（4个单元）
    # 换页命令
    R"\pagebreak": 0,  # 强制换页
    # 页码样式
    R"\pagenumbering": 0, # 页码样式
    # 页码引用
    R"\pageref": 2,    # 页码引用（如 [5]，2个单元）
    # 模运算
    R"\pmod": 5,       # p mod（5个符号）
    # 概率
    R"\Pr": 2,         # Pr（2个字母）
    # 保护命令
    R"\protect": 0,    # 保护命令
    # 间距命令
    R"\qquad": 0,      # 双quad间距
    R"\quad": 0,       # quad间距
    # 底部对齐
    R"\raggedbottom": 0, # 底部不齐
    # 左对齐
    R"\raggedleft": 0, # 左对齐
    # 右对齐
    R"\raggedright": 0, # 右对齐
    # 实部
    R"\Re": 2,         # Re（2个字母）
    # 引用命令
    R"\ref": 2,        # 引用（如 [3]，2个单元）
    # 恢复回车
    R"\restorecr": 0,  # 恢复回车
    # 反向边注
    R"\reversemarginpar": 0, # 反向边注
    # 右括号
    R"\right": 0,      # 右括号前缀
    # 罗马字体
    R"\rm": 0,         # 罗马字体
    # 小大写字体
    R"\sc": 0,         # 小大写字体
    # 字体大小
    R"\scriptscriptstyle": 0, # 极小字体
    R"\scriptsize": 0,        # 超小字体
    R"\scriptstyle": 0,       # 脚本字体
    # 正割函数
    R"\sec": 3,        # sec（3个字母）
    # 无衬线字体
    R"\sf": 0,         # 无衬线字体
    # 短堆叠
    R"\shortstack": 0, # 短堆叠
    # 正弦函数
    R"\sin": 3,        # sin（3个字母）
    R"\sinh": 4,       # sinh（4个字母）
    # 倾斜字体
    R"\sl": 0,         # 倾斜字体
    # 宽松排版
    R"\sloppy": 0,     # 宽松排版
    # 字体大小
    R"\small": 0,      # 小字体
    R"\Small": 0,      # 较小字体
    R"\smallskip": 0,  # 小间距
    # 平方根
    R"\sqrt": 2,       # 根号（2个单元）
    # 特殊字符
    R"\ss": 0,         #  ß 字符
    # 上确界
    R"\sup": 3,        # sup（3个字母）
    # 正切函数
    R"\tan": 3,        # tan（3个字母）
    R"\tanh": 4,       # tanh（4个字母）
    # 文本模式
    R"\text": 0,       # 文本模式
    R"\textbf": 0,     # 粗体文本
    # 文本占比
    R"\textfraction": 2, # 文本占页比例
    # 文本样式
    R"\textstyle": 0,  # 行内公式样式
    # 线条粗细
    R"\thicklines": 0, # 粗线条
    R"\thinlines": 0,  # 细线条
    R"\thinspace": 0,  # 细空格
    # 字体大小
    R"\tiny": 0,       # 极小字体
    # 标题命令
    R"\title": 0,      # 标题
    # 日期命令
    R"\today": 15,     # 今天日期（约15个字符）
    # 顶部占比
    R"\topfraction": 2, # 顶部浮动体占比
    # 打字机字体
    R"\tt": 0,         # 打字机字体
    # 输出命令
    R"\typeout": 0,    # 输出到终端
    # 非粗体数学
    R"\unboldmath": 0, # 非粗体数学模式
    # 下括号
    R"\underbrace": 6, # 下括号（6个单元）
    # 下划线
    R"\underline": 0,  # 下划线（格式）
    # 取值命令
    R"\value": 0,      # 取值
    # 垂直省略号
    R"\vdots": 3,      # 垂直省略号（3个点）
    # 垂直线
    R"\vline": 0       # 垂直线（格式）
}