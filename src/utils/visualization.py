# -*- coding: utf-8 -*-
"""
可视化工具函数
==============
包含matplotlib配置、图表生成等功能
"""

import matplotlib
import matplotlib.pyplot as plt
import platform


def setup_matplotlib_chinese():
    """
    配置matplotlib支持中文显示
    按优先级尝试多种中文字体，确保跨平台兼容性
    """
    from matplotlib import font_manager
    
    # 根据操作系统选择合适的中文字体
    system = platform.system()
    
    if system == 'Windows':
        font_list = ['Microsoft YaHei', 'SimHei', 'SimSun', 'FangSong', 'KaiTi']
    elif system == 'Darwin':  # macOS
        font_list = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Hiragino Sans GB']
    else:  # Linux
        font_list = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 
                     'Droid Sans Fallback', 'AR PL UMing CN']
    
    # 添加通用后备字体
    font_list.extend(['DejaVu Sans', 'Arial Unicode MS'])
    
    # 检查可用字体
    available_fonts = set([f.name for f in font_manager.fontManager.ttflist])
    
    # 选择第一个可用的字体
    selected_font = None
    for font in font_list:
        if font in available_fonts:
            selected_font = font
            break
    
    if selected_font:
        matplotlib.rcParams['font.sans-serif'] = [selected_font] + font_list
        print(f"使用中文字体: {selected_font}")
    else:
        matplotlib.rcParams['font.sans-serif'] = font_list
        print("警告: 未找到理想中文字体，可能出现显示问题")
    
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    # 刷新字体缓存
    font_manager._load_fontmanager(try_read_cache=False)


def create_figure(figsize=(12, 8), title=None):
    """
    创建标准化的matplotlib图表
    
    Args:
        figsize: 图表尺寸
        title: 图表标题
    
    Returns:
        fig, ax: matplotlib图表对象
    """
    fig, ax = plt.subplots(figsize=figsize)
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    return fig, ax


def save_figure(fig, filepath, dpi=300, bbox_inches='tight'):
    """
    保存图表到文件
    
    Args:
        fig: matplotlib图表对象
        filepath: 保存路径
        dpi: 分辨率
        bbox_inches: 边界设置
    """
    fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
    print(f"图表已保存: {filepath}")
