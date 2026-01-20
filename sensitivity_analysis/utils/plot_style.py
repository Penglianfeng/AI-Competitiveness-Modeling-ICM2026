# -*- coding: utf-8 -*-
"""
统一绑定绘图风格模块
Unified Plot Style Module for Sensitivity Analysis
===================================================

本模块提供统一的绑定绘图风格，确保所有敏感性分析图表保持一致的视觉效果。

Features:
- 学术风格配色方案 (colorblind-friendly)
- 统一的字体大小设置
- 国家标识颜色映射
- 高分辨率输出 (300 DPI)

Author: AI Modeling Assistant
Date: January 2026
"""

import matplotlib.pyplot as plt
import matplotlib
import platform
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# 常量定义
# =============================================================================

# 默认图像尺寸和分辨率
FIGSIZE_DEFAULT: Tuple[int, int] = (10, 6)
FIGSIZE_WIDE: Tuple[int, int] = (14, 6)
FIGSIZE_TALL: Tuple[int, int] = (10, 10)
FIGSIZE_SQUARE: Tuple[int, int] = (8, 8)
DPI_DEFAULT: int = 300

# 字体大小配置
FONT_SIZE_TITLE: int = 14
FONT_SIZE_LABEL: int = 12
FONT_SIZE_TICK: int = 10
FONT_SIZE_LEGEND: int = 10
FONT_SIZE_ANNOTATION: int = 9

# 国家颜色映射 (colorblind-friendly palette)
COUNTRY_COLORS: Dict[str, str] = {
    'USA': '#1f77b4',      # 蓝色 - 美国
    'CHN': '#d62728',      # 红色 - 中国
    'GBR': '#2ca02c',      # 绿色 - 英国
    'DEU': '#ff7f0e',      # 橙色 - 德国
    'KOR': '#9467bd',      # 紫色 - 韩国
    'JPN': '#8c564b',      # 棕色 - 日本
    'FRA': '#e377c2',      # 粉色 - 法国
    'CAN': '#7f7f7f',      # 灰色 - 加拿大
    'ARE': '#bcbd22',      # 黄绿色 - 阿联酋
    'IND': '#17becf',      # 青色 - 印度
}

# 国家名称映射 (中英文)
COUNTRY_NAMES: Dict[str, Dict[str, str]] = {
    'USA': {'en': 'United States', 'cn': '美国', 'short': 'USA'},
    'CHN': {'en': 'China', 'cn': '中国', 'short': 'CHN'},
    'GBR': {'en': 'United Kingdom', 'cn': '英国', 'short': 'UK'},
    'DEU': {'en': 'Germany', 'cn': '德国', 'short': 'DEU'},
    'KOR': {'en': 'South Korea', 'cn': '韩国', 'short': 'KOR'},
    'JPN': {'en': 'Japan', 'cn': '日本', 'short': 'JPN'},
    'FRA': {'en': 'France', 'cn': '法国', 'short': 'FRA'},
    'CAN': {'en': 'Canada', 'cn': '加拿大', 'short': 'CAN'},
    'ARE': {'en': 'UAE', 'cn': '阿联酋', 'short': 'UAE'},
    'IND': {'en': 'India', 'cn': '印度', 'short': 'IND'},
}

# 标准国家顺序
COUNTRIES_ORDER: List[str] = ['USA', 'CHN', 'GBR', 'DEU', 'KOR', 'JPN', 'FRA', 'CAN', 'ARE', 'IND']

# 指标颜色映射 (用于权重龙卷风图)
INDICATOR_COLORS: Dict[str, str] = {
    'A': '#1f77b4',  # 算力与基础设施 - 蓝色
    'B': '#ff7f0e',  # 人才与教育 - 橙色
    'C': '#2ca02c',  # 科研产出 - 绿色
    'D': '#d62728',  # 开源生态 - 红色
    'E': '#9467bd',  # 产业与资本 - 紫色
    'F': '#8c564b',  # 治理准备度 - 棕色
}

# 学术配色方案
ACADEMIC_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


# =============================================================================
# 字体配置
# =============================================================================

def setup_chinese_font() -> str:
    """
    配置matplotlib支持中文显示
    Configure matplotlib for Chinese character support
    
    Returns:
        str: 选中的字体名称
    """
    from matplotlib import font_manager
    
    system = platform.system()
    
    if system == 'Windows':
        font_list = ['Microsoft YaHei', 'SimHei', 'SimSun', 'FangSong', 'KaiTi']
    elif system == 'Darwin':  # macOS
        font_list = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Hiragino Sans GB']
    else:  # Linux
        font_list = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC',
                     'Droid Sans Fallback', 'AR PL UMing CN']
    
    # 添加通用后备字体
    font_list.extend(['DejaVu Sans', 'Arial Unicode MS', 'Arial'])
    
    available_fonts = set([f.name for f in font_manager.fontManager.ttflist])
    
    selected_font = None
    for font in font_list:
        if font in available_fonts:
            selected_font = font
            break
    
    if selected_font:
        matplotlib.rcParams['font.sans-serif'] = [selected_font] + font_list
    else:
        matplotlib.rcParams['font.sans-serif'] = font_list
    
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    return selected_font or 'default'


def setup_plot_style(use_chinese: bool = False) -> None:
    """
    Setup unified plot style for all sensitivity analysis figures
    
    Args:
        use_chinese: Whether to enable Chinese font support (default: False for English-only)
    """
    # Configure fonts - use English fonts by default
    if use_chinese:
        setup_chinese_font()
    else:
        # Use standard English fonts
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['axes.unicode_minus'] = False
    
    # Use seaborn style as base
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 自定义参数
    custom_params = {
        # 图像尺寸
        'figure.figsize': FIGSIZE_DEFAULT,
        'figure.dpi': 100,
        'savefig.dpi': DPI_DEFAULT,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        
        # 字体大小
        'font.size': FONT_SIZE_TICK,
        'axes.titlesize': FONT_SIZE_TITLE,
        'axes.titleweight': 'bold',
        'axes.labelsize': FONT_SIZE_LABEL,
        'xtick.labelsize': FONT_SIZE_TICK,
        'ytick.labelsize': FONT_SIZE_TICK,
        'legend.fontsize': FONT_SIZE_LEGEND,
        
        # 网格
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'grid.color': '#cccccc',
        
        # 边框
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 1.0,
        
        # 颜色
        'axes.prop_cycle': plt.cycler(color=ACADEMIC_PALETTE),
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        
        # 图例
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '#cccccc',
        
        # 线条
        'lines.linewidth': 2.0,
        'lines.markersize': 6,
    }
    
    matplotlib.rcParams.update(custom_params)


def get_country_colors(countries: Optional[List[str]] = None) -> Dict[str, str]:
    """
    获取国家颜色映射
    
    Args:
        countries: 国家代码列表，None 则返回全部
    
    Returns:
        Dict[str, str]: 国家代码到颜色的映射
    """
    if countries is None:
        return COUNTRY_COLORS.copy()
    return {c: COUNTRY_COLORS.get(c, '#333333') for c in countries}


def get_country_name(code: str, lang: str = 'en') -> str:
    """
    获取国家名称
    
    Args:
        code: 国家代码
        lang: 语言 ('en', 'cn', 'short')
    
    Returns:
        str: 国家名称
    """
    if code in COUNTRY_NAMES:
        return COUNTRY_NAMES[code].get(lang, code)
    return code


# =============================================================================
# 图形创建与保存
# =============================================================================

def create_figure(
    nrows: int = 1,
    ncols: int = 1,
    figsize: Optional[Tuple[int, int]] = None,
    **kwargs
) -> Tuple[plt.Figure, np.ndarray]:
    """
    创建符合统一风格的图形
    Create figure with unified style
    
    Args:
        nrows: 子图行数
        ncols: 子图列数
        figsize: 图形尺寸，None 则自动计算
        **kwargs: 传递给 plt.subplots 的其他参数
    
    Returns:
        Tuple[Figure, Axes]: matplotlib Figure 和 Axes 对象
    """
    if figsize is None:
        # 自动计算尺寸
        base_width = 5 * ncols
        base_height = 4 * nrows
        figsize = (min(base_width, 16), min(base_height, 12))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    
    return fig, axes


def save_figure(
    fig: plt.Figure,
    filepath: Path,
    dpi: int = DPI_DEFAULT,
    close_fig: bool = True
) -> None:
    """
    保存图形到文件
    Save figure to file with standard settings
    
    Args:
        fig: matplotlib Figure 对象
        filepath: 保存路径
        dpi: 分辨率
        close_fig: 是否在保存后关闭图形
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(
        filepath,
        dpi=dpi,
        bbox_inches='tight',
        pad_inches=0.1,
        facecolor='white',
        edgecolor='none'
    )
    
    if close_fig:
        plt.close(fig)
    
    print(f"  📊 图片已保存: {filepath}")


# =============================================================================
# 辅助绑定函数
# =============================================================================

def add_ranking_annotations(
    ax: plt.Axes,
    x_positions: np.ndarray,
    y_values: np.ndarray,
    rankings: np.ndarray,
    fontsize: int = FONT_SIZE_ANNOTATION
) -> None:
    """
    为柱状图添加排名标注
    
    Args:
        ax: matplotlib Axes 对象
        x_positions: X 轴位置
        y_values: Y 轴值（柱高）
        rankings: 排名数组
        fontsize: 字体大小
    """
    for x, y, rank in zip(x_positions, y_values, rankings):
        ax.annotate(
            f'#{int(rank)}',
            xy=(x, y),
            xytext=(0, 3),
            textcoords='offset points',
            ha='center',
            va='bottom',
            fontsize=fontsize,
            fontweight='bold'
        )


def add_confidence_band(
    ax: plt.Axes,
    x: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    color: str,
    alpha: float = 0.2,
    label: Optional[str] = None
) -> None:
    """
    添加置信区间阴影带
    
    Args:
        ax: matplotlib Axes 对象
        x: X 轴数据
        y_mean: Y 轴均值
        y_std: Y 轴标准差
        color: 颜色
        alpha: 透明度
        label: 图例标签
    """
    ax.fill_between(
        x,
        y_mean - 1.96 * y_std,
        y_mean + 1.96 * y_std,
        color=color,
        alpha=alpha,
        label=label
    )


def create_colorbar(
    fig: plt.Figure,
    ax: plt.Axes,
    mappable,
    label: str = '',
    orientation: str = 'vertical'
) -> plt.colorbar:
    """
    创建统一风格的颜色条
    
    Args:
        fig: matplotlib Figure 对象
        ax: matplotlib Axes 对象
        mappable: 可映射对象（如 imshow 返回值）
        label: 颜色条标签
        orientation: 方向 ('vertical' 或 'horizontal')
    
    Returns:
        colorbar 对象
    """
    cbar = fig.colorbar(mappable, ax=ax, orientation=orientation, pad=0.02)
    cbar.set_label(label, fontsize=FONT_SIZE_LABEL)
    cbar.ax.tick_params(labelsize=FONT_SIZE_TICK)
    return cbar


# =============================================================================
# 初始化
# =============================================================================

# 模块加载时自动设置绑定风格
setup_plot_style()

if __name__ == '__main__':
    # 测试绑定风格
    print("Testing plot style setup...")
    
    fig, ax = create_figure()
    
    # 测试绑定
    x = np.linspace(0, 10, 100)
    for i, (country, color) in enumerate(list(COUNTRY_COLORS.items())[:5]):
        y = np.sin(x + i * 0.5) + i * 0.5
        ax.plot(x, y, color=color, label=get_country_name(country))
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_title('Test Plot with Unified Style')
    ax.legend()
    
    plt.show()
