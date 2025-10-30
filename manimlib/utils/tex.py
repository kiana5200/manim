# ManimGL LaTeX 符号计数工具：用于精确统计 LaTeX 表达式中视觉呈现的符号数量，
# 忽略格式控制命令和环境声明，为文本排版、尺寸调整提供量化依据。


from __future__ import annotations

import re
from functools import lru_cache  # 缓存装饰器，优化重复计算

# 导入预定义的 LaTeX 命令与符号数量映射表
from manimlib.utils.tex_to_symbol_count import TEX_TO_SYMBOL_COUNT


@lru_cache
def num_tex_symbols(tex: str) -> int:
    """
    计算 LaTeX 代码中实际显示的符号数量（排除格式命令和环境）。
    
    核心逻辑：
    1. 移除不影响显示的 LaTeX 环境（如 \phantom、\begin 等）；
    2. 区分普通字符与 LaTeX 命令，分别计数；
    3. 普通字符排除格式控制符（^、_ 等），命令参考预定义映射表计数。
    
    参数：tex - LaTeX 源代码字符串
    返回：int - 视觉符号总数（带缓存，相同输入仅计算一次）。
    """
    # 先移除 LaTeX 环境相关代码
    tex = remove_tex_environments(tex)

    # 正则表达式：匹配各类 LaTeX 命令
    commands_pattern = r"""
        (?P<sqrt>\\sqrt\[[0-9]+\])|    # 带参数的开方命令（如 \sqrt[3]）
        (?P<escaped_brace>\\[{}])|      \# 转义的花括号（\{ 或 \}）
        (?P<cmd>\\[a-zA-Z!,-/:;<>]+)    \# 常规命令（如 \alpha、\sum）
    """

    total = 0  # 总符号数
    pos = 0    # 当前处理位置

    # 遍历所有匹配的命令
    for match in re.finditer(commands_pattern, tex, re.VERBOSE):
        # 统计当前命令前的普通字符（排除格式控制符）
        total += sum(
            1 for c in tex[pos:match.start()] 
            if c not in "^{} \n\t_$\\&"  # 这些字符不显示为独立符号
        )

        # 根据命令类型累加符号数
        if match.group("sqrt"):
            # 开方命令 \sqrt[n]：长度减去固定前缀 "\sqrt["（5个字符）
            total += len(match.group()) - 5
        elif match.group("escaped_brace"):
            # 转义花括号（\{ 或 \}）计为1个符号
            total += 1
        else:
            # 常规命令：按映射表计数，默认1个符号
            total += TEX_TO_SYMBOL_COUNT.get(match.group(), 1)

        pos = match.end()  # 更新处理位置

    # 统计剩余字符中的有效符号
    total += sum(
        1 for c in tex[pos:] 
        if c not in "^{} \n\t_$\\&"
    )

    return total


def remove_tex_environments(tex: str) -> str:
    """
    移除 LaTeX 代码中不影响视觉呈现的环境声明和占位符。
    
    处理内容：
    - \phantom{...}：移除占位文本（不显示）；
    - \begin{...}、\end{...}：移除环境声明（仅控制格式）。
    
    参数：tex - 原始 LaTeX 代码
    返回：净化后的 LaTeX 代码（仅保留显示相关内容）。
    """
    # 移除 \phantom{...} 及其内容（占位符不显示）
    tex = re.sub(r"\\phantom\{[^}]*\}", "", tex)
    # 移除 \begin 和 \end 环境命令（如 \begin{align}、\end{equation}）
    tex = re.sub(r"\\(begin|end)(\{\w+\})?(\{\w+\})?(\[\w+\])?", "", tex)
    return tex