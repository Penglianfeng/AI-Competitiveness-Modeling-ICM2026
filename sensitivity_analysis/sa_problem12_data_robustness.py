# -*- coding: utf-8 -*-
"""
SA-1: 数据口径敏感性分析
Data Robustness Sensitivity Analysis (Problem 1 & 2)
=====================================================

目的：检验2025年外推数据对排名的影响
Methods:
1. 对比"含2025外推值" vs "仅用2024真实值"两套数据的排名差异
2. 对外推值添加噪声 ε ~ N(0, σ²)，σ ∈ {5%, 10%, 15%}
3. 统计各国排名变化的频率分布

输出：
- fig_sa1_data_robustness.png: 分组柱状图 + 误差棒
- sa_summary_problem12.csv: 汇总表格

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

# 指标列名（用于TOPSIS计算）
INDICATOR_COLS = [
    'A1_Hardware_Compute_log', 'A2_Energy_IDC_log', 'A3_Connectivity_norm',
    'B1_Talent_Stock_log', 'B3_STEM_Supply_norm',
    'C1_Research_Qty_log', 'C2_High_Impact_Res_log',
    'D1_GitHub_Activity_log', 'D3_OpenSource_Impact_log',
    'E1_Vertical_VC_log', 'E2_Capital_Flow_log', 'E3_Ind_Adoption_norm',
    'F1_Gov_RD_Exp_norm', 'F2_IP_Protection_log'
]

# 噪声水平
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.15]  # 0%, 5%, 10%, 15%
N_SIMULATIONS = 500  # Monte Carlo 模拟次数


# =============================================================================
# 数据加载与预处理
# =============================================================================

def load_topsis_data() -> pd.DataFrame:
    """
    加载TOPSIS评分数据
    
    Returns:
        pd.DataFrame: 包含所有年份和国家的TOPSIS数据
    """
    data_file = DATA_PATH / 'topsis_scores.csv'
    
    if not data_file.exists():
        logger.error(f"数据文件不存在: {data_file}")
        raise FileNotFoundError(f"找不到TOPSIS数据文件: {data_file}")
    
    df = pd.read_csv(data_file)
    
    # 过滤目标国家
    df = df[df['Country'].isin(COUNTRIES)]
    
    logger.info(f"加载数据: {len(df)} 条记录, 年份范围: {df['Year'].min()}-{df['Year'].max()}")
    
    return df


def load_dematel_weights() -> Dict[str, float]:
    """
    加载DEMATEL权重
    
    Returns:
        Dict[str, float]: 指标权重字典
    """
    weights_file = DATA_PATH / 'dematel_weights.csv'
    
    if weights_file.exists():
        weights_df = pd.read_csv(weights_file)
        weights = dict(zip(weights_df['Indicator'], weights_df['Weight']))
        logger.info(f"加载DEMATEL权重: {len(weights)} 个指标")
        return weights
    else:
        # 等权重 fallback
        logger.warning("未找到DEMATEL权重文件，使用等权重")
        return {col: 1.0 / len(INDICATOR_COLS) for col in INDICATOR_COLS}


# =============================================================================
# TOPSIS 计算
# =============================================================================

def normalize_matrix(X: np.ndarray) -> np.ndarray:
    """
    向量归一化
    
    Args:
        X: 决策矩阵 (n_samples, n_features)
    
    Returns:
        归一化后的矩阵
    """
    norms = np.sqrt((X ** 2).sum(axis=0))
    norms = np.where(norms == 0, 1e-10, norms)
    return X / norms


def calculate_topsis_scores(
    data: pd.DataFrame,
    indicator_cols: List[str],
    weights: Dict[str, float]
) -> np.ndarray:
    """
    计算TOPSIS综合得分
    
    Args:
        data: 包含指标数据的DataFrame
        indicator_cols: 指标列名列表
        weights: 权重字典
    
    Returns:
        np.ndarray: TOPSIS综合得分
    """
    # 提取指标矩阵
    available_cols = [col for col in indicator_cols if col in data.columns]
    X = data[available_cols].values.astype(float)
    
    # 处理缺失值（使用列中位数填充）
    col_medians = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        nan_mask = np.isnan(X[:, j])
        X[nan_mask, j] = col_medians[j] if not np.isnan(col_medians[j]) else 0
    
    # 归一化
    X_norm = normalize_matrix(X)
    
    # 加权
    w = np.array([weights.get(col, 1.0 / len(available_cols)) for col in available_cols])
    w = w / w.sum()  # 归一化权重
    X_weighted = X_norm * w
    
    # 理想解与负理想解（假设所有指标都是效益型）
    ideal_best = X_weighted.max(axis=0)
    ideal_worst = X_weighted.min(axis=0)
    
    # 计算距离
    d_best = np.sqrt(((X_weighted - ideal_best) ** 2).sum(axis=1))
    d_worst = np.sqrt(((X_weighted - ideal_worst) ** 2).sum(axis=1))
    
    # TOPSIS得分
    scores = d_worst / (d_best + d_worst + 1e-10)
    
    return scores


def calculate_rankings(scores: np.ndarray) -> np.ndarray:
    """
    计算排名（得分越高排名越靠前，1为最高）
    
    Args:
        scores: TOPSIS得分数组
    
    Returns:
        np.ndarray: 排名数组
    """
    return rankdata(-scores, method='ordinal')


# =============================================================================
# 敏感性分析核心函数
# =============================================================================

def compare_2024_vs_2025(df: pd.DataFrame, weights: Dict[str, float]) -> Dict:
    """
    对比2024真实值 vs 2025外推值的排名差异
    
    Args:
        df: 完整数据
        weights: 权重字典
    
    Returns:
        Dict: 对比结果
    """
    # 获取2024和2025年数据
    df_2024 = df[df['Year'] == 2024].copy()
    df_2025 = df[df['Year'] == 2025].copy()
    
    if len(df_2024) == 0 or len(df_2025) == 0:
        # 如果没有2024或2025数据，尝试使用最近两年
        years = sorted(df['Year'].unique())
        if len(years) >= 2:
            df_2024 = df[df['Year'] == years[-2]].copy()
            df_2025 = df[df['Year'] == years[-1]].copy()
            logger.info(f"使用 {years[-2]} 和 {years[-1]} 年数据进行对比")
    
    # 按国家排序以确保对应
    df_2024 = df_2024.sort_values('Country').reset_index(drop=True)
    df_2025 = df_2025.sort_values('Country').reset_index(drop=True)
    
    # 计算TOPSIS得分
    available_cols = [col for col in INDICATOR_COLS if col in df_2024.columns]
    
    scores_2024 = calculate_topsis_scores(df_2024, available_cols, weights)
    scores_2025 = calculate_topsis_scores(df_2025, available_cols, weights)
    
    # 计算排名
    rankings_2024 = calculate_rankings(scores_2024)
    rankings_2025 = calculate_rankings(scores_2025)
    
    # 排名变化
    rank_changes = rankings_2025 - rankings_2024
    
    return {
        'countries': df_2024['Country'].tolist(),
        'scores_2024': scores_2024,
        'scores_2025': scores_2025,
        'rankings_2024': rankings_2024,
        'rankings_2025': rankings_2025,
        'rank_changes': rank_changes
    }


def monte_carlo_noise_simulation(
    df: pd.DataFrame,
    weights: Dict[str, float],
    noise_level: float,
    n_simulations: int = N_SIMULATIONS
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Monte Carlo噪声模拟
    
    对2025年数据添加噪声，模拟数据不确定性对排名的影响
    
    Args:
        df: 2025年数据
        weights: 权重字典
        noise_level: 噪声水平 (标准差比例)
        n_simulations: 模拟次数
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - rankings_all: (n_simulations, n_countries) 所有模拟的排名
            - scores_all: (n_simulations, n_countries) 所有模拟的得分
    """
    df_2025 = df[df['Year'] == df['Year'].max()].copy()
    df_2025 = df_2025.sort_values('Country').reset_index(drop=True)
    
    available_cols = [col for col in INDICATOR_COLS if col in df_2025.columns]
    X_original = df_2025[available_cols].values.astype(float)
    
    n_countries = len(df_2025)
    rankings_all = np.zeros((n_simulations, n_countries))
    scores_all = np.zeros((n_simulations, n_countries))
    
    for sim in range(n_simulations):
        # 添加噪声
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, X_original.shape)
            X_noisy = X_original * (1 + noise)
            X_noisy = np.maximum(X_noisy, 0)  # 确保非负
        else:
            X_noisy = X_original.copy()
        
        # 创建带噪声的DataFrame
        df_noisy = df_2025.copy()
        df_noisy[available_cols] = X_noisy
        
        # 计算TOPSIS得分和排名
        scores = calculate_topsis_scores(df_noisy, available_cols, weights)
        rankings = calculate_rankings(scores)
        
        rankings_all[sim] = rankings
        scores_all[sim] = scores
    
    return rankings_all, scores_all


def analyze_ranking_stability(rankings_all: np.ndarray, countries: List[str]) -> pd.DataFrame:
    """
    分析排名稳定性
    
    Args:
        rankings_all: (n_simulations, n_countries) 所有模拟的排名
        countries: 国家列表
    
    Returns:
        pd.DataFrame: 排名稳定性统计
    """
    n_countries = len(countries)
    
    results = []
    for i, country in enumerate(countries):
        country_rankings = rankings_all[:, i]
        results.append({
            'Country': country,
            'Country_Name': COUNTRY_NAMES_EN.get(country, country),
            'Mean_Rank': np.mean(country_rankings),
            'Std_Rank': np.std(country_rankings),
            'Min_Rank': np.min(country_rankings),
            'Max_Rank': np.max(country_rankings),
            'Median_Rank': np.median(country_rankings),
            'Mode_Rank': pd.Series(country_rankings).mode()[0],
            'Rank_Range': np.max(country_rankings) - np.min(country_rankings)
        })
    
    return pd.DataFrame(results)


# =============================================================================
# 可视化函数
# =============================================================================

def plot_data_robustness(
    comparison_result: Dict,
    noise_results: Dict[float, Tuple[np.ndarray, np.ndarray]],
    output_path: Path
) -> None:
    """
    绑定数据口径敏感性分析图
    
    Args:
        comparison_result: 2024 vs 2025对比结果
        noise_results: 不同噪声水平的Monte Carlo结果
        output_path: 输出路径
    """
    # 导入绑定风格
    from .utils.plot_style import (
        setup_plot_style, save_figure, create_figure,
        COUNTRY_COLORS, get_country_name, FONT_SIZE_TITLE,
        FONT_SIZE_LABEL, FONT_SIZE_TICK, FONT_SIZE_LEGEND
    )
    
    setup_plot_style()
    
    countries = comparison_result['countries']
    n_countries = len(countries)
    
    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # ===== 子图1: 2024 vs 2025 排名对比 =====
    ax1 = axes[0]
    x = np.arange(n_countries)
    width = 0.35
    
    colors_2024 = [COUNTRY_COLORS.get(c, '#999999') for c in countries]
    
    bars1 = ax1.bar(x - width/2, comparison_result['rankings_2024'], width,
                    label='2024 (Actual)', color=colors_2024, alpha=0.7, edgecolor='black')
    bars2 = ax1.bar(x + width/2, comparison_result['rankings_2025'], width,
                    label='2025 (Extrapolated)', color=colors_2024, alpha=1.0, edgecolor='black', hatch='//')
    
    ax1.set_xlabel('Country', fontsize=FONT_SIZE_LABEL)
    ax1.set_ylabel('Ranking (1=Highest)', fontsize=FONT_SIZE_LABEL)
    ax1.set_title('2024 Actual vs 2025 Extrapolated Rankings',
                  fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([get_country_name(c, 'short') for c in countries], 
                        rotation=45, ha='right', fontsize=FONT_SIZE_TICK)
    ax1.legend(fontsize=FONT_SIZE_LEGEND)
    ax1.invert_yaxis()  # 排名1在上
    ax1.set_ylim(n_countries + 0.5, 0.5)
    ax1.grid(axis='y', alpha=0.3)
    
    # 添加排名变化标注
    for i, (change, bar2) in enumerate(zip(comparison_result['rank_changes'], bars2)):
        if change != 0:
            color = 'green' if change < 0 else 'red'  # 排名下降（数值变小）是好事
            symbol = '↑' if change < 0 else '↓'
            ax1.annotate(f'{symbol}{abs(change)}',
                        xy=(bar2.get_x() + bar2.get_width()/2, bar2.get_height()),
                        xytext=(0, -15), textcoords='offset points',
                        ha='center', va='top', color=color, fontweight='bold',
                        fontsize=FONT_SIZE_TICK)
    
    # ===== Subplot 2: Noise sensitivity grouped bar chart + error bars =====
    ax2 = axes[1]
    
    noise_labels = ['0%\n(Baseline)', '5%', '10%', '15%']
    n_noise_levels = len(NOISE_LEVELS)
    width = 0.8 / n_countries
    
    for i, country in enumerate(countries):
        mean_ranks = []
        std_ranks = []
        
        for noise_level in NOISE_LEVELS:
            rankings_all, _ = noise_results[noise_level]
            country_rankings = rankings_all[:, i]
            mean_ranks.append(np.mean(country_rankings))
            std_ranks.append(np.std(country_rankings))
        
        x_pos = np.arange(n_noise_levels) + i * width - (n_countries - 1) * width / 2
        
        ax2.bar(x_pos, mean_ranks, width * 0.9,
                yerr=std_ranks, capsize=2,
                color=COUNTRY_COLORS.get(country, '#999999'),
                alpha=0.8, label=get_country_name(country, 'short'),
                edgecolor='white', linewidth=0.5)
    
    ax2.set_xlabel('Noise Level (std)', fontsize=FONT_SIZE_LABEL)
    ax2.set_ylabel('Mean Rank +/- Std', fontsize=FONT_SIZE_LABEL)
    ax2.set_title('Ranking Stability under Different Noise Levels',
                  fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax2.set_xticks(np.arange(n_noise_levels))
    ax2.set_xticklabels(noise_labels, fontsize=FONT_SIZE_TICK)
    ax2.legend(loc='upper left', ncol=2, fontsize=FONT_SIZE_LEGEND - 1)
    ax2.invert_yaxis()
    ax2.set_ylim(n_countries + 0.5, 0.5)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图形
    save_figure(fig, output_path / 'fig_sa1_data_robustness.png')


# =============================================================================
# 主函数
# =============================================================================

def run_data_robustness(output_dir: Optional[Path] = None) -> Dict:
    """
    运行数据口径敏感性分析
    
    Args:
        output_dir: 输出目录，None则使用默认路径
    
    Returns:
        Dict: 分析结果
    """
    logger.info("=" * 60)
    logger.info("SA-1: 数据口径敏感性分析 / Data Robustness Analysis")
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
    
    # 1. 对比 2024 vs 2025
    logger.info("对比 2024真实值 vs 2025外推值...")
    comparison_result = compare_2024_vs_2025(df, weights)
    
    # 2. Monte Carlo 噪声模拟
    logger.info("运行 Monte Carlo 噪声模拟...")
    noise_results = {}
    for noise_level in NOISE_LEVELS:
        logger.info(f"  噪声水平: {noise_level*100:.0f}%")
        rankings_all, scores_all = monte_carlo_noise_simulation(
            df, weights, noise_level, N_SIMULATIONS
        )
        noise_results[noise_level] = (rankings_all, scores_all)
    
    # 3. 分析排名稳定性
    logger.info("分析排名稳定性...")
    stability_results = {}
    for noise_level in NOISE_LEVELS:
        rankings_all, _ = noise_results[noise_level]
        stability_df = analyze_ranking_stability(rankings_all, comparison_result['countries'])
        stability_df['Noise_Level'] = f"{noise_level*100:.0f}%"
        stability_results[noise_level] = stability_df
    
    # 合并稳定性结果
    all_stability = pd.concat(stability_results.values(), ignore_index=True)
    
    # 4. 生成可视化
    logger.info("生成可视化图表...")
    plot_data_robustness(comparison_result, noise_results, figures_path)
    
    # 5. Save summary tables
    logger.info("Saving summary tables...")
    
    # Save comparison results
    comparison_df = pd.DataFrame({
        'Country': comparison_result['countries'],
        'Country_EN': [COUNTRY_NAMES_EN.get(c, c) for c in comparison_result['countries']],
        'Score_2024': comparison_result['scores_2024'],
        'Score_2025': comparison_result['scores_2025'],
        'Rank_2024': comparison_result['rankings_2024'],
        'Rank_2025': comparison_result['rankings_2025'],
        'Rank_Change': comparison_result['rank_changes']
    })
    comparison_df.to_csv(tables_path / 'sa1_2024_vs_2025_comparison.csv', 
                         index=False, encoding='utf-8-sig')
    
    # Save stability analysis
    all_stability.to_csv(tables_path / 'sa1_ranking_stability.csv', 
                         index=False, encoding='utf-8-sig')
    
    # Summary statistics
    summary_stats = {
        'Analysis_Item': ['Data Robustness Analysis'],
        'N_Simulations': [N_SIMULATIONS],
        'Noise_Levels': [', '.join([f'{n*100:.0f}%' for n in NOISE_LEVELS])],
        'Max_Rank_Change_5pct': [stability_results[0.05]['Rank_Range'].max()],
        'Max_Rank_Change_10pct': [stability_results[0.10]['Rank_Range'].max()],
        'Max_Rank_Change_15pct': [stability_results[0.15]['Rank_Range'].max()],
        'Most_Unstable_Country': [stability_results[0.15].loc[
            stability_results[0.15]['Std_Rank'].idxmax(), 'Country_Name'
        ]],
        'Most_Stable_Country': [stability_results[0.15].loc[
            stability_results[0.15]['Std_Rank'].idxmin(), 'Country_Name'
        ]]
    }
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(tables_path / 'sa_summary_problem12.csv', 
                      index=False, encoding='utf-8-sig')
    
    logger.info("SA-1 Analysis Complete!")
    logger.info(f"  Figures: {figures_path / 'fig_sa1_data_robustness.png'}")
    logger.info(f"  Tables: {tables_path}")
    
    return {
        'comparison': comparison_result,
        'noise_results': noise_results,
        'stability_results': stability_results,
        'summary': summary_stats
    }


if __name__ == '__main__':
    results = run_data_robustness()
