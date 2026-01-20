# -*- coding: utf-8 -*-
"""
SA-2: 权重龙卷风图分析
Weight Tornado Chart Analysis (Problem 1 & 2)
=============================================

目的：识别对最终排名影响最大的指标权重
Methods:
1. 逐一将每个指标权重在 [w_i - 20%, w_i + 20%] 范围内变化
2. 记录中国/美国综合得分的变化区间
3. 按影响幅度排序

输出：
- fig_sa2_weight_tornado.png: 双向水平条形图 (Tornado Chart)

Author: AI Modeling Assistant
Date: January 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
import logging
from scipy.stats import rankdata

# 设置随机种子
np.random.seed(42)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

# =============================================================================
# 路径配置
# =============================================================================

def get_base_path() -> Path:
    """动态获取项目根目录"""
    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        if (parent / 'configs').exists() or (parent / 'outputs').exists():
            return parent
    return Path.cwd()

BASE_PATH = get_base_path()
DATA_PATH = BASE_PATH / 'outputs' / 'problem1_2'
OUTPUT_PATH = BASE_PATH / 'outputs' / 'sensitivity_analysis'
FIGURES_PATH = OUTPUT_PATH / 'figures'
TABLES_PATH = OUTPUT_PATH / 'tables'

# =============================================================================
# 常量定义
# =============================================================================

COUNTRIES = ['USA', 'CHN', 'GBR', 'DEU', 'KOR', 'JPN', 'FRA', 'CAN', 'ARE', 'IND']
COUNTRY_NAMES_EN = {
    'USA': 'United States', 'CHN': 'China', 'GBR': 'United Kingdom', 'DEU': 'Germany',
    'KOR': 'South Korea', 'JPN': 'Japan', 'FRA': 'France', 'CAN': 'Canada',
    'ARE': 'UAE', 'IND': 'India'
}

# Indicator column names and English names
INDICATOR_INFO = {
    'A1_Hardware_Compute_log': {'name': 'A1 Hardware Compute', 'category': 'A', 'en': 'Hardware Compute'},
    'A2_Energy_IDC_log': {'name': 'A2 Energy/IDC', 'category': 'A', 'en': 'Energy/IDC'},
    'A3_Connectivity_norm': {'name': 'A3 Connectivity', 'category': 'A', 'en': 'Connectivity'},
    'B1_Talent_Stock_log': {'name': 'B1 Talent Stock', 'category': 'B', 'en': 'Talent Stock'},
    'B3_STEM_Supply_norm': {'name': 'B3 STEM Supply', 'category': 'B', 'en': 'STEM Supply'},
    'C1_Research_Qty_log': {'name': 'C1 Research Quantity', 'category': 'C', 'en': 'Research Quantity'},
    'C2_High_Impact_Res_log': {'name': 'C2 High Impact Research', 'category': 'C', 'en': 'High Impact Research'},
    'D1_GitHub_Activity_log': {'name': 'D1 GitHub Activity', 'category': 'D', 'en': 'GitHub Activity'},
    'D3_OpenSource_Impact_log': {'name': 'D3 OpenSource Impact', 'category': 'D', 'en': 'OpenSource Impact'},
    'E1_Vertical_VC_log': {'name': 'E1 Vertical VC', 'category': 'E', 'en': 'Vertical VC'},
    'E2_Capital_Flow_log': {'name': 'E2 Capital Flow', 'category': 'E', 'en': 'Capital Flow'},
    'E3_Ind_Adoption_norm': {'name': 'E3 Industry Adoption', 'category': 'E', 'en': 'Industry Adoption'},
    'F1_Gov_RD_Exp_norm': {'name': 'F1 Gov R&D', 'category': 'F', 'en': 'Gov R&D'},
    'F2_IP_Protection_log': {'name': 'F2 IP Protection', 'category': 'F', 'en': 'IP Protection'},
}

INDICATOR_COLS = list(INDICATOR_INFO.keys())

# Category colors
CATEGORY_COLORS = {
    'A': '#1f77b4',  # Infrastructure - Blue
    'B': '#ff7f0e',  # Talent & Education - Orange
    'C': '#2ca02c',  # Research Output - Green
    'D': '#d62728',  # OpenSource - Red
    'E': '#9467bd',  # Industry & Capital - Purple
    'F': '#8c564b',  # Governance - Brown
}

# 权重扰动范围
WEIGHT_PERTURBATION = 0.20  # ±20%


# =============================================================================
# 数据加载
# =============================================================================

def load_topsis_data() -> pd.DataFrame:
    """加载TOPSIS评分数据"""
    data_file = DATA_PATH / 'topsis_scores.csv'
    
    if not data_file.exists():
        raise FileNotFoundError(f"找不到TOPSIS数据文件: {data_file}")
    
    df = pd.read_csv(data_file)
    df = df[df['Country'].isin(COUNTRIES)]
    
    return df


def load_dematel_weights() -> Dict[str, float]:
    """加载DEMATEL权重"""
    weights_file = DATA_PATH / 'dematel_weights.csv'
    
    if weights_file.exists():
        weights_df = pd.read_csv(weights_file)
        weights = dict(zip(weights_df['Indicator'], weights_df['Weight']))
        return weights
    else:
        # 等权重 fallback
        return {col: 1.0 / len(INDICATOR_COLS) for col in INDICATOR_COLS}


# =============================================================================
# TOPSIS 计算
# =============================================================================

def normalize_matrix(X: np.ndarray) -> np.ndarray:
    """向量归一化"""
    norms = np.sqrt((X ** 2).sum(axis=0))
    norms = np.where(norms == 0, 1e-10, norms)
    return X / norms


def calculate_topsis_scores(
    data: pd.DataFrame,
    indicator_cols: List[str],
    weights: Dict[str, float]
) -> np.ndarray:
    """计算TOPSIS综合得分"""
    available_cols = [col for col in indicator_cols if col in data.columns]
    X = data[available_cols].values.astype(float)
    
    # 处理缺失值
    col_medians = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        nan_mask = np.isnan(X[:, j])
        X[nan_mask, j] = col_medians[j] if not np.isnan(col_medians[j]) else 0
    
    # 归一化
    X_norm = normalize_matrix(X)
    
    # 加权
    w = np.array([weights.get(col, 1.0 / len(available_cols)) for col in available_cols])
    w = w / w.sum()
    X_weighted = X_norm * w
    
    # 理想解
    ideal_best = X_weighted.max(axis=0)
    ideal_worst = X_weighted.min(axis=0)
    
    # 距离
    d_best = np.sqrt(((X_weighted - ideal_best) ** 2).sum(axis=1))
    d_worst = np.sqrt(((X_weighted - ideal_worst) ** 2).sum(axis=1))
    
    # TOPSIS得分
    scores = d_worst / (d_best + d_worst + 1e-10)
    
    return scores


# =============================================================================
# 权重敏感性分析
# =============================================================================

def weight_sensitivity_analysis(
    df: pd.DataFrame,
    base_weights: Dict[str, float],
    target_country: str = 'CHN',
    perturbation: float = WEIGHT_PERTURBATION
) -> pd.DataFrame:
    """
    权重敏感性分析 - 单因素扰动法
    
    Args:
        df: 最新年份的数据
        base_weights: 基准权重
        target_country: 目标国家
        perturbation: 扰动幅度
    
    Returns:
        pd.DataFrame: 各指标的敏感性分析结果
    """
    # 获取最新年份数据
    latest_year = df['Year'].max()
    df_latest = df[df['Year'] == latest_year].copy()
    df_latest = df_latest.sort_values('Country').reset_index(drop=True)
    
    countries = df_latest['Country'].tolist()
    target_idx = countries.index(target_country) if target_country in countries else 0
    
    # 基准得分
    available_cols = [col for col in INDICATOR_COLS if col in df_latest.columns]
    base_scores = calculate_topsis_scores(df_latest, available_cols, base_weights)
    base_score = base_scores[target_idx]
    
    results = []
    
    for indicator in available_cols:
        if indicator not in base_weights:
            continue
        
        base_w = base_weights[indicator]
        
        # 权重减少
        weights_low = base_weights.copy()
        weights_low[indicator] = base_w * (1 - perturbation)
        scores_low = calculate_topsis_scores(df_latest, available_cols, weights_low)
        score_low = scores_low[target_idx]
        
        # 权重增加
        weights_high = base_weights.copy()
        weights_high[indicator] = base_w * (1 + perturbation)
        scores_high = calculate_topsis_scores(df_latest, available_cols, weights_high)
        score_high = scores_high[target_idx]
        
        # 计算影响
        impact_low = score_low - base_score
        impact_high = score_high - base_score
        total_range = abs(impact_high - impact_low)
        
        results.append({
            'Indicator': indicator,
            'Indicator_Name': INDICATOR_INFO.get(indicator, {}).get('name', indicator),
            'Category': INDICATOR_INFO.get(indicator, {}).get('category', 'X'),
            'Base_Weight': base_w,
            'Score_Low': score_low,
            'Score_Base': base_score,
            'Score_High': score_high,
            'Impact_Low': impact_low,
            'Impact_High': impact_high,
            'Total_Range': total_range
        })
    
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('Total_Range', ascending=True)  # 升序排列，绑定时最大的在上
    
    return result_df


def dual_country_sensitivity(
    df: pd.DataFrame,
    base_weights: Dict[str, float],
    perturbation: float = WEIGHT_PERTURBATION
) -> Dict[str, pd.DataFrame]:
    """
    对中国和美国进行双重敏感性分析
    
    Args:
        df: 数据
        base_weights: 基准权重
        perturbation: 扰动幅度
    
    Returns:
        Dict: {'CHN': df_chn, 'USA': df_usa}
    """
    return {
        'CHN': weight_sensitivity_analysis(df, base_weights, 'CHN', perturbation),
        'USA': weight_sensitivity_analysis(df, base_weights, 'USA', perturbation)
    }


# =============================================================================
# 可视化函数
# =============================================================================

def plot_weight_tornado(
    sensitivity_results: Dict[str, pd.DataFrame],
    output_path: Path
) -> None:
    """
    绑定权重龙卷风图
    
    Args:
        sensitivity_results: 敏感性分析结果 {'CHN': df, 'USA': df}
        output_path: 输出路径
    """
    from .utils.plot_style import (
        setup_plot_style, save_figure,
        FONT_SIZE_TITLE, FONT_SIZE_LABEL, FONT_SIZE_TICK, FONT_SIZE_LEGEND
    )
    
    setup_plot_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    
    countries_plot = [('CHN', 'China'), ('USA', 'United States')]
    
    for ax, (country_code, country_name) in zip(axes, countries_plot):
        df_sens = sensitivity_results[country_code]
        
        n_indicators = len(df_sens)
        y_pos = np.arange(n_indicators)
        
        # Get colors
        colors = [CATEGORY_COLORS.get(cat, '#999999') for cat in df_sens['Category']]
        
        # Baseline score
        base_score = df_sens['Score_Base'].iloc[0]
        
        # Draw bidirectional bar chart
        # Left side (weight decrease impact)
        bars_low = ax.barh(y_pos, df_sens['Impact_Low'], 
                          color=colors, alpha=0.7, edgecolor='white',
                          label='Weight -20%')
        
        # Right side (weight increase impact)
        bars_high = ax.barh(y_pos, df_sens['Impact_High'], 
                           color=colors, alpha=1.0, edgecolor='black',
                           label='Weight +20%', hatch='//')
        
        # Baseline reference line
        ax.axvline(x=0, color='black', linewidth=2, linestyle='-')
        
        # Set Y-axis labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_sens['Indicator_Name'], fontsize=FONT_SIZE_TICK)
        
        # Set title and labels
        ax.set_xlabel(f'Score Change (vs. base {base_score:.4f})', 
                      fontsize=FONT_SIZE_LABEL)
        ax.set_title(f'{country_name} Weight Sensitivity Analysis',
                    fontsize=FONT_SIZE_TITLE, fontweight='bold')
        
        # 添加影响幅度标注
        for i, (_, row) in enumerate(df_sens.iterrows()):
            # 在条形末端添加数值
            if row['Impact_Low'] != 0:
                ax.annotate(f"{row['Impact_Low']:.4f}",
                           xy=(row['Impact_Low'], i),
                           xytext=(-5, 0) if row['Impact_Low'] < 0 else (5, 0),
                           textcoords='offset points',
                           ha='right' if row['Impact_Low'] < 0 else 'left',
                           va='center', fontsize=7)
            if row['Impact_High'] != 0:
                ax.annotate(f"{row['Impact_High']:.4f}",
                           xy=(row['Impact_High'], i),
                           xytext=(5, 0) if row['Impact_High'] > 0 else (-5, 0),
                           textcoords='offset points',
                           ha='left' if row['Impact_High'] > 0 else 'right',
                           va='center', fontsize=7)
        
        ax.grid(axis='x', alpha=0.3)
        ax.set_axisbelow(True)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#1f77b4', label='A: Infrastructure'),
            Patch(facecolor='#ff7f0e', label='B: Talent & Education'),
            Patch(facecolor='#2ca02c', label='C: Research Output'),
            Patch(facecolor='#d62728', label='D: OpenSource'),
            Patch(facecolor='#9467bd', label='E: Industry & Capital'),
            Patch(facecolor='#8c564b', label='F: Governance'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', 
                 fontsize=FONT_SIZE_LEGEND - 1, ncol=2)
    
    plt.tight_layout()
    
    # 保存图形
    save_figure(fig, output_path / 'fig_sa2_weight_tornado.png')


def plot_combined_sensitivity(
    sensitivity_results: Dict[str, pd.DataFrame],
    output_path: Path
) -> None:
    """
    绘制组合敏感性比较图
    
    Args:
        sensitivity_results: 敏感性分析结果
        output_path: 输出路径
    """
    from .utils.plot_style import (
        setup_plot_style, save_figure,
        FONT_SIZE_TITLE, FONT_SIZE_LABEL, FONT_SIZE_TICK
    )
    
    setup_plot_style()
    
    # 合并两国的敏感性数据
    df_chn = sensitivity_results['CHN'].copy()
    df_usa = sensitivity_results['USA'].copy()
    
    df_chn['Total_Range_CHN'] = df_chn['Total_Range']
    df_usa['Total_Range_USA'] = df_usa['Total_Range']
    
    df_merged = df_chn[['Indicator', 'Indicator_Name', 'Category', 'Total_Range_CHN']].merge(
        df_usa[['Indicator', 'Total_Range_USA']], on='Indicator'
    )
    
    # 按平均影响排序
    df_merged['Avg_Range'] = (df_merged['Total_Range_CHN'] + df_merged['Total_Range_USA']) / 2
    df_merged = df_merged.sort_values('Avg_Range', ascending=False)
    
    # 绘图
    fig, ax = plt.subplots(figsize=(12, 8))
    
    n_indicators = len(df_merged)
    y_pos = np.arange(n_indicators)
    width = 0.35
    
    # 获取颜色
    colors = [CATEGORY_COLORS.get(cat, '#999999') for cat in df_merged['Category']]
    
    bars1 = ax.barh(y_pos - width/2, df_merged['Total_Range_CHN'], width,
                    color='#d62728', alpha=0.8, label='China')
    bars2 = ax.barh(y_pos + width/2, df_merged['Total_Range_USA'], width,
                    color='#1f77b4', alpha=0.8, label='USA')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_merged['Indicator_Name'], fontsize=FONT_SIZE_TICK)
    ax.set_xlabel('Score Change Range', fontsize=FONT_SIZE_LABEL)
    ax.set_title('USA vs China Weight Sensitivity Comparison',
                fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax.legend(fontsize=FONT_SIZE_TICK)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, output_path / 'fig_sa2_weight_comparison.png')


# =============================================================================
# 主函数
# =============================================================================

def run_weight_tornado(output_dir: Optional[Path] = None) -> Dict:
    """
    运行权重龙卷风图分析
    
    Args:
        output_dir: 输出目录
    
    Returns:
        Dict: 分析结果
    """
    logger.info("=" * 60)
    logger.info("SA-2: 权重龙卷风图分析 / Weight Tornado Chart Analysis")
    logger.info("=" * 60)
    
    # 设置输出路径
    if output_dir is None:
        output_dir = OUTPUT_PATH
    output_dir = Path(output_dir)
    figures_path = output_dir / 'figures'
    tables_path = output_dir / 'tables'
    
    figures_path.mkdir(parents=True, exist_ok=True)
    tables_path.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    logger.info("加载数据...")
    df = load_topsis_data()
    weights = load_dematel_weights()
    
    # 运行敏感性分析
    logger.info("运行权重敏感性分析...")
    sensitivity_results = dual_country_sensitivity(df, weights)
    
    # 生成可视化
    logger.info("生成可视化图表...")
    plot_weight_tornado(sensitivity_results, figures_path)
    plot_combined_sensitivity(sensitivity_results, figures_path)
    
    # 保存结果表格
    logger.info("保存结果表格...")
    for country, df_sens in sensitivity_results.items():
        df_sens.to_csv(tables_path / f'sa2_weight_sensitivity_{country}.csv',
                       index=False, encoding='utf-8-sig')
    
    # 识别关键指标
    df_chn = sensitivity_results['CHN'].sort_values('Total_Range', ascending=False)
    df_usa = sensitivity_results['USA'].sort_values('Total_Range', ascending=False)
    
    logger.info("\n关键发现:")
    logger.info(f"  中国最敏感指标: {df_chn.iloc[0]['Indicator_Name']} (影响幅度: {df_chn.iloc[0]['Total_Range']:.4f})")
    logger.info(f"  美国最敏感指标: {df_usa.iloc[0]['Indicator_Name']} (影响幅度: {df_usa.iloc[0]['Total_Range']:.4f})")
    
    logger.info("SA-2 分析完成!")
    logger.info(f"  📊 图表: {figures_path}")
    logger.info(f"  📋 表格: {tables_path}")
    
    return {
        'sensitivity_results': sensitivity_results,
        'key_indicators_chn': df_chn.head(5)['Indicator_Name'].tolist(),
        'key_indicators_usa': df_usa.head(5)['Indicator_Name'].tolist()
    }


if __name__ == '__main__':
    results = run_weight_tornado()
