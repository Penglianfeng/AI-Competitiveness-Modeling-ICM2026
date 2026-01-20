# -*- coding: utf-8 -*-
"""
SA-4: 预算/约束敏感性分析
Budget and Constraint Sensitivity Analysis (Problem 4)
======================================================

目的：分析预算约束和不同场景对最优分配的影响
Methods:
1. 多场景预算分析（紧缩、基准、宽松）
2. Pareto前沿分析
3. 领域权重敏感性

输出：
- fig_sa5_budget_scenarios.png: 分场景堆叠面积图
- fig_sa6_pareto_frontier.png: Pareto前沿图

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
from scipy.optimize import minimize, differential_evolution
from dataclasses import dataclass, field

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
OUTPUT_PATH = BASE_PATH / 'outputs' / 'sensitivity_analysis'
FIGURES_PATH = OUTPUT_PATH / 'figures'
TABLES_PATH = OUTPUT_PATH / 'tables'

# =============================================================================
# 常量定义
# =============================================================================

EPS = 1e-10
N_YEARS = 10  # 2026-2035
N_SECTORS = 6  # A-F 六个领域

YEAR_START = 2026
YEAR_END = 2035
YEARS = list(range(YEAR_START, YEAR_END + 1))

# 领域名称
SECTOR_NAMES_CN = ['基建(A)', '人才(B)', '科研(C)', '开源(D)', '产业(E)', '治理(F)']
SECTOR_NAMES_EN = ['Infrastructure', 'Talent', 'Research', 'OpenSource', 'Industry', 'Governance']

# 领域颜色
SECTOR_COLORS = {
    'A': '#1f77b4',  # 蓝色 - 基建
    'B': '#ff7f0e',  # 橙色 - 人才
    'C': '#2ca02c',  # 绿色 - 科研
    'D': '#d62728',  # 红色 - 开源
    'E': '#9467bd',  # 紫色 - 产业
    'F': '#8c564b',  # 棕色 - 治理
}

# 场景配置
BUDGET_SCENARIOS = {
    'tight': {
        'name_cn': '紧缩情景',
        'name_en': 'Tight Budget',
        'total_budget': 6000.0,  # 亿元
        'max_annual': 1200.0,
        'color': '#e74c3c'  # 红色
    },
    'baseline': {
        'name_cn': '基准情景',
        'name_en': 'Baseline',
        'total_budget': 10000.0,
        'max_annual': 2000.0,
        'color': '#3498db'  # 蓝色
    },
    'loose': {
        'name_cn': '宽松情景',
        'name_en': 'Loose Budget',
        'total_budget': 15000.0,
        'max_annual': 3000.0,
        'color': '#27ae60'  # 绿色
    }
}


# =============================================================================
# OCBAM-II 模型参数
# =============================================================================

@dataclass
class BudgetSensitivityParams:
    """预算敏感性分析参数"""
    
    # 初始状态 (2025年归一化后的指标值)
    X_2025: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.4, 0.3, 0.2, 0.4, 0.6]))
    
    # 折旧与增长参数
    depreciation_rate_A: float = 0.35
    talent_lag: int = 4
    ces_rho: float = 0.33
    ces_alpha: float = 0.4
    
    # 效率参数
    infra_efficiency: float = 0.0008
    talent_efficiency: float = 0.001
    research_efficiency: float = 0.001
    opensource_efficiency: float = 0.002
    industry_leverage: float = 3.0
    governance_efficiency: float = 0.02
    
    # TOPSIS权重
    topsis_weights: np.ndarray = field(default_factory=lambda: np.array([0.15, 0.2, 0.25, 0.15, 0.15, 0.1]))
    
    # 罚函数系数
    penalty_smooth_coef: float = 0.0005
    penalty_budget_coef: float = 1000.0


# =============================================================================
# 简化OCBAM模拟器
# =============================================================================

def simulate_ocbam(
    U_matrix: np.ndarray,
    params: BudgetSensitivityParams,
    total_budget: float
) -> Tuple[np.ndarray, float]:
    """
    简化的OCBAM模拟
    
    Args:
        U_matrix: 决策变量矩阵 (N_YEARS, N_SECTORS)
        params: 模型参数
        total_budget: 总预算约束
    
    Returns:
        state_history: 状态历史 (N_YEARS+1, N_SECTORS)
        final_score: 最终综合得分
    """
    state_history = np.zeros((N_YEARS + 1, N_SECTORS))
    state_history[0] = params.X_2025.copy()
    
    # 历史人才投入（用于滞后效应）
    talent_history = np.zeros(params.talent_lag)
    talent_history[:] = 100.0  # 假设历史平均值
    
    for t in range(N_YEARS):
        X_current = state_history[t].copy()
        U_current = U_matrix[t, :]
        
        X_next = X_current.copy()
        
        # A - 基建（折旧 + 投资）
        depreciation = params.depreciation_rate_A * X_current[0]
        X_next[0] = X_current[0] * (1 - params.depreciation_rate_A) + \
                    params.infra_efficiency * U_current[0]
        
        # B - 人才（滞后效应）
        lagged_investment = talent_history[0] if t >= params.talent_lag else 100.0
        talent_boost = params.talent_efficiency * lagged_investment
        X_next[1] = X_current[1] * (1 + 0.02) + talent_boost * 0.1
        
        # 更新人才投入历史
        talent_history = np.roll(talent_history, -1)
        talent_history[-1] = U_current[1]
        
        # C - 科研（CES生产函数简化）
        A_contrib = X_current[0] ** params.ces_alpha
        B_contrib = X_current[1] ** (1 - params.ces_alpha)
        ces_output = A_contrib * B_contrib
        X_next[2] = X_current[2] * 0.95 + params.research_efficiency * U_current[2] + 0.02 * ces_output
        
        # D - 开源
        X_next[3] = X_current[3] * (1 + 0.03) + params.opensource_efficiency * U_current[3]
        
        # E - 产业
        leverage_effective = params.industry_leverage / (1 + 0.001 * U_current[4])
        X_next[4] = X_current[4] * 0.98 + 0.0002 * U_current[4] * leverage_effective
        
        # F - 治理
        X_next[5] = X_current[5] * 0.99 + params.governance_efficiency * U_current[5] * 0.001
        
        # 限制状态变量范围
        X_next = np.clip(X_next, 0.01, 2.0)
        state_history[t + 1] = X_next
    
    # 计算最终综合得分
    final_state = state_history[-1]
    final_score = np.sum(params.topsis_weights * final_state)
    
    # 添加预算惩罚
    total_spent = np.sum(U_matrix)
    if total_spent > total_budget:
        budget_penalty = params.penalty_budget_coef * (total_spent - total_budget) / total_budget
        final_score -= budget_penalty
    
    return state_history, final_score


def objective_function(
    U_flat: np.ndarray,
    params: BudgetSensitivityParams,
    total_budget: float,
    max_annual: float
) -> float:
    """
    优化目标函数（最小化负得分）
    
    Args:
        U_flat: 展平的决策变量
        params: 模型参数
        total_budget: 总预算
        max_annual: 年度最大投入
    
    Returns:
        负得分（用于最小化）
    """
    U_matrix = U_flat.reshape(N_YEARS, N_SECTORS)
    
    # 年度约束违反惩罚
    annual_totals = U_matrix.sum(axis=1)
    annual_penalty = np.sum(np.maximum(0, annual_totals - max_annual)) * 10
    
    _, final_score = simulate_ocbam(U_matrix, params, total_budget)
    
    # 平滑惩罚（减少年度波动）
    smooth_penalty = 0
    for s in range(N_SECTORS):
        smooth_penalty += np.sum(np.diff(U_matrix[:, s]) ** 2) * params.penalty_smooth_coef
    
    return -(final_score - annual_penalty - smooth_penalty)


def optimize_allocation(
    params: BudgetSensitivityParams,
    total_budget: float,
    max_annual: float
) -> Tuple[np.ndarray, float]:
    """
    优化预算分配
    
    Args:
        params: 模型参数
        total_budget: 总预算
        max_annual: 年度最大投入
    
    Returns:
        optimal_U: 最优分配矩阵
        optimal_score: 最优得分
    """
    n_vars = N_YEARS * N_SECTORS
    
    # 初始猜测：单一均分解在该问题的数值尺度下很容易成为“假收敛点”。
    # 采用多启动（均分 + 启发式 + 随机扰动）选取最优解。
    avg_annual = total_budget / N_YEARS
    avg_sector = avg_annual / N_SECTORS
    U_uniform = np.full(n_vars, avg_sector)
    
    # 边界约束
    bounds = [(0, max_annual / N_SECTORS * 2) for _ in range(n_vars)]
    
    # 线性约束：
    # 1) 总预算：强制用满（等式约束）
    # 2) 年度预算：每年总投入不超过 max_annual（不等式约束）
    from scipy.optimize import LinearConstraint
    A = np.zeros((1 + N_YEARS, n_vars))
    A[0, :] = 1.0
    for t in range(N_YEARS):
        A[1 + t, t * N_SECTORS:(t + 1) * N_SECTORS] = 1.0

    lb = np.zeros(1 + N_YEARS)
    ub = np.zeros(1 + N_YEARS)
    lb[0] = total_budget
    ub[0] = total_budget
    lb[1:] = 0.0
    ub[1:] = max_annual
    linear_constraints = LinearConstraint(A, lb, ub)

    def _solve(x0: np.ndarray):
        return minimize(
            objective_function,
            x0,
            args=(params, total_budget, max_annual),
            method='SLSQP',
            bounds=bounds,
            constraints=[linear_constraints],
            # 注意：eps 为有限差分步长；模型系数量级较小，需适当增大以获得稳定梯度。
            options={'maxiter': 800, 'ftol': 1e-9, 'eps': 1.0, 'disp': False}
        )

    # 启发式领域占比（用于打破对称性；并非最终方案，只是更好的起点）
    # 直觉：科研/开源边际收益更高，治理较低；人才有滞后但依然需要一定投入。
    p_heuristic = np.array([0.12, 0.18, 0.26, 0.22, 0.16, 0.06], dtype=float)
    p_heuristic = p_heuristic / p_heuristic.sum()
    U_heuristic = np.tile(avg_annual * p_heuristic, (N_YEARS, 1)).reshape(-1)

    # 随机扰动（围绕启发式占比，生成若干可行初值）
    rng = np.random.default_rng(42)
    x0_candidates = [
        U_uniform,
        U_heuristic,
    ]
    for _ in range(4):
        noise = rng.normal(0.0, 0.05, size=N_SECTORS)
        p = np.clip(p_heuristic * (1.0 + noise), 0.01, None)
        p = p / p.sum()
        x0_candidates.append(np.tile(avg_annual * p, (N_YEARS, 1)).reshape(-1))

    best_result = None
    best_fun = np.inf
    for idx, x0 in enumerate(x0_candidates):
        res = _solve(x0)
        if not res.success:
            logger.debug(f"SLSQP start {idx} not converged: status={res.status}, fun={res.fun}, msg={res.message}")
        if res.fun < best_fun:
            best_fun = res.fun
            best_result = res

    result = best_result
    if result is None:
        raise RuntimeError("Optimization failed: no candidate produced a result")

    if not result.success:
        logger.warning(
            f"Optimization did not fully converge (success={result.success}, status={result.status}). "
            f"Message: {result.message}" 
        )
    
    optimal_U = result.x.reshape(N_YEARS, N_SECTORS)
    optimal_score = -result.fun
    
    return optimal_U, optimal_score


# =============================================================================
# 多场景分析
# =============================================================================

def run_scenario_analysis(params: BudgetSensitivityParams) -> Dict[str, Dict]:
    """
    运行多场景预算分析
    
    Args:
        params: 基准参数
    
    Returns:
        Dict: 各场景结果
    """
    results = {}
    
    for scenario_name, scenario_config in BUDGET_SCENARIOS.items():
        logger.info(f"  Optimizing {scenario_config['name_en']} (Budget: {scenario_config['total_budget']:.0f})...")
        
        optimal_U, optimal_score = optimize_allocation(
            params,
            scenario_config['total_budget'],
            scenario_config['max_annual']
        )
        
        state_history, _ = simulate_ocbam(optimal_U, params, scenario_config['total_budget'])
        
        results[scenario_name] = {
            'config': scenario_config,
            'optimal_U': optimal_U,
            'optimal_score': optimal_score,
            'state_history': state_history,
            'annual_totals': optimal_U.sum(axis=1),
            'sector_totals': optimal_U.sum(axis=0)
        }
        
        logger.info(f"    Optimal Score: {optimal_score:.4f}")
    
    return results


# =============================================================================
# Pareto 前沿分析
# =============================================================================

def pareto_frontier_analysis(
    params: BudgetSensitivityParams,
    n_points: int = 20
) -> Dict[str, np.ndarray]:
    """
    计算 Pareto 前沿
    
    目标1: 最大化最终综合得分
    目标2: 最小化总投入
    
    Args:
        params: 模型参数
        n_points: 前沿点数
    
    Returns:
        Dict: Pareto前沿数据
    """
    # 预算范围
    budget_range = np.linspace(5000, 20000, n_points)
    
    scores = []
    budgets = []
    allocations = []
    
    for budget in budget_range:
        max_annual = budget / N_YEARS * 1.5
        
        try:
            optimal_U, optimal_score = optimize_allocation(params, budget, max_annual)
            actual_budget = np.sum(optimal_U)
            
            scores.append(optimal_score)
            budgets.append(actual_budget)
            allocations.append(optimal_U)
        except Exception as e:
            logger.warning(f"Budget {budget:.0f} optimization failed: {e}")
            continue
    
    # 计算Pareto前沿
    scores = np.array(scores)
    budgets = np.array(budgets)
    
    # 识别Pareto最优点
    is_pareto = np.ones(len(scores), dtype=bool)
    for i in range(len(scores)):
        for j in range(len(scores)):
            if i != j:
                # j 支配 i 如果 j 有更高得分且更低预算
                if scores[j] >= scores[i] and budgets[j] <= budgets[i]:
                    if scores[j] > scores[i] or budgets[j] < budgets[i]:
                        is_pareto[i] = False
                        break
    
    return {
        'scores': scores,
        'budgets': budgets,
        'is_pareto': is_pareto,
        'allocations': allocations
    }


def analyze_budget_elasticity(
    scenario_results: Dict[str, Dict]
) -> pd.DataFrame:
    """
    分析预算弹性
    
    Args:
        scenario_results: 场景分析结果
    
    Returns:
        DataFrame: 弹性分析结果
    """
    elasticity_data = []
    
    scenarios = list(scenario_results.keys())
    for i in range(len(scenarios) - 1):
        s1, s2 = scenarios[i], scenarios[i + 1]
        r1, r2 = scenario_results[s1], scenario_results[s2]
        
        budget_change = (r2['config']['total_budget'] - r1['config']['total_budget']) / r1['config']['total_budget']
        score_change = (r2['optimal_score'] - r1['optimal_score']) / (r1['optimal_score'] + EPS)
        
        elasticity = score_change / (budget_change + EPS)
        
        elasticity_data.append({
            'Scenario_From': s1,
            'Scenario_To': s2,
            'Budget_Change_Pct': budget_change * 100,
            'Score_Change_Pct': score_change * 100,
            'Elasticity': elasticity,
            'Marginal_Return': score_change / budget_change if budget_change != 0 else 0
        })
    
    return pd.DataFrame(elasticity_data)


# =============================================================================
# 可视化函数
# =============================================================================

def plot_budget_scenarios(
    scenario_results: Dict[str, Dict],
    output_path: Path
) -> None:
    """
    绘制分场景堆叠面积图
    
    Args:
        scenario_results: 场景分析结果
        output_path: 输出路径
    """
    from .utils.plot_style import (
        setup_plot_style, save_figure,
        FONT_SIZE_TITLE, FONT_SIZE_LABEL, FONT_SIZE_TICK, FONT_SIZE_LEGEND
    )
    
    setup_plot_style()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, (scenario_name, results) in enumerate(scenario_results.items()):
        ax = axes[idx]
        config = results['config']
        optimal_U = results['optimal_U']
        
        # 堆叠面积图
        years = np.array(YEARS)
        
        # 准备堆叠数据
        bottom = np.zeros(N_YEARS)
        colors = [SECTOR_COLORS[chr(65 + i)] for i in range(N_SECTORS)]
        
        for s in range(N_SECTORS):
            ax.fill_between(years, bottom, bottom + optimal_U[:, s],
                           color=colors[s], alpha=0.8,
                           label=SECTOR_NAMES_EN[s])
            bottom += optimal_U[:, s]
        
        # Add annual total investment line
        ax.plot(years, bottom, 'k--', linewidth=1.5, label='Annual Total')
        
        ax.set_xlabel('Year', fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel('Investment (100M CNY)', fontsize=FONT_SIZE_LABEL)
        ax.set_title(f"{config['name_en']}\n"
                    f"Total Budget: {config['total_budget']:.0f} | Score: {results['optimal_score']:.3f}",
                    fontsize=FONT_SIZE_TITLE, fontweight='bold')
        
        if idx == 0:
            ax.legend(loc='upper left', fontsize=FONT_SIZE_LEGEND - 1, ncol=2)
        
        ax.set_xlim(YEAR_START, YEAR_END)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, output_path / 'fig_sa5_budget_scenarios.png')


def plot_pareto_frontier(
    pareto_data: Dict[str, np.ndarray],
    output_path: Path
) -> None:
    """
    绘制Pareto前沿图
    
    Args:
        pareto_data: Pareto前沿数据
        output_path: 输出路径
    """
    from .utils.plot_style import (
        setup_plot_style, save_figure,
        FONT_SIZE_TITLE, FONT_SIZE_LABEL, FONT_SIZE_TICK
    )
    
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scores = pareto_data['scores']
    budgets = pareto_data['budgets']
    is_pareto = pareto_data['is_pareto']
    
    # Plot all points
    ax.scatter(budgets[~is_pareto], scores[~is_pareto],
               c='gray', alpha=0.5, s=50, label='Dominated Solutions')
    
    # Plot Pareto frontier points
    pareto_budgets = budgets[is_pareto]
    pareto_scores = scores[is_pareto]
    
    # Sort for line connection
    sort_idx = np.argsort(pareto_budgets)
    pareto_budgets = pareto_budgets[sort_idx]
    pareto_scores = pareto_scores[sort_idx]
    
    ax.scatter(pareto_budgets, pareto_scores,
               c='#e74c3c', s=100, zorder=5, label='Pareto Optimal')
    ax.plot(pareto_budgets, pareto_scores,
            'r-', linewidth=2, alpha=0.7)
    
    # Annotate key scenarios
    for scenario_name, config in BUDGET_SCENARIOS.items():
        budget_level = config['total_budget']
        # Find closest point
        closest_idx = np.argmin(np.abs(budgets - budget_level))
        ax.annotate(f"{config['name_en']}\n({budgets[closest_idx]:.0f})",
                   xy=(budgets[closest_idx], scores[closest_idx]),
                   xytext=(20, 20), textcoords='offset points',
                   fontsize=9, ha='left',
                   arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=config['color'], alpha=0.3))
    
    ax.set_xlabel('Total Budget (100M CNY)', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('Composite Score', fontsize=FONT_SIZE_LABEL)
    ax.set_title('Budget-Performance Pareto Frontier',
                fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax.legend(loc='lower right', fontsize=FONT_SIZE_TICK)
    ax.grid(True, alpha=0.3)
    
    # Add marginal utility region annotations
    if len(pareto_budgets) > 2:
        # High efficiency zone
        mid_idx = len(pareto_budgets) // 2
        ax.axvspan(pareto_budgets[0], pareto_budgets[mid_idx],
                   alpha=0.1, color='green', label='High Marginal Utility')
        ax.axvspan(pareto_budgets[mid_idx], pareto_budgets[-1],
                   alpha=0.1, color='orange', label='Diminishing Returns')
    
    plt.tight_layout()
    save_figure(fig, output_path / 'fig_sa6_pareto_frontier.png')


def plot_sector_allocation_comparison(
    scenario_results: Dict[str, Dict],
    output_path: Path
) -> None:
    """
    绘制各场景领域分配对比图
    
    Args:
        scenario_results: 场景分析结果
        output_path: 输出路径
    """
    from .utils.plot_style import (
        setup_plot_style, save_figure,
        FONT_SIZE_TITLE, FONT_SIZE_LABEL, FONT_SIZE_TICK
    )
    
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    scenarios = list(scenario_results.keys())
    n_scenarios = len(scenarios)
    
    x = np.arange(N_SECTORS)
    width = 0.25
    
    for i, scenario_name in enumerate(scenarios):
        results = scenario_results[scenario_name]
        config = results['config']
        sector_totals = results['sector_totals']
        
        # Normalize to percentage
        sector_pct = sector_totals / np.sum(sector_totals) * 100
        
        bars = ax.bar(x + i * width - width, sector_pct, width,
                     color=config['color'], alpha=0.8,
                     label=f"{config['name_en']}")
    
    ax.set_xlabel('Investment Sector', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('Allocation Percentage (%)', fontsize=FONT_SIZE_LABEL)
    ax.set_title('Sector Allocation Comparison by Scenario',
                fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(SECTOR_NAMES_EN, fontsize=FONT_SIZE_TICK)
    ax.legend(fontsize=FONT_SIZE_TICK)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_figure(fig, output_path / 'fig_sa5_sector_allocation.png')


# =============================================================================
# 主函数
# =============================================================================

def run_budget_sensitivity(output_dir: Optional[Path] = None) -> Dict:
    """
    运行预算/约束敏感性分析
    
    Args:
        output_dir: 输出目录
    
    Returns:
        Dict: 分析结果
    """
    logger.info("=" * 60)
    logger.info("SA-4: Budget/Constraint Sensitivity Analysis")
    logger.info("=" * 60)
    
    # Set output paths
    if output_dir is None:
        output_dir = OUTPUT_PATH
    output_dir = Path(output_dir)
    figures_path = output_dir / 'figures'
    tables_path = output_dir / 'tables'
    
    figures_path.mkdir(parents=True, exist_ok=True)
    tables_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize parameters
    params = BudgetSensitivityParams()
    
    # Multi-scenario analysis
    logger.info("Running multi-scenario budget optimization...")
    scenario_results = run_scenario_analysis(params)
    
    # Pareto frontier analysis
    logger.info("Computing Pareto frontier...")
    pareto_data = pareto_frontier_analysis(params, n_points=15)
    
    # Budget elasticity analysis
    logger.info("Analyzing budget elasticity...")
    elasticity_df = analyze_budget_elasticity(scenario_results)
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    plot_budget_scenarios(scenario_results, figures_path)
    plot_pareto_frontier(pareto_data, figures_path)
    plot_sector_allocation_comparison(scenario_results, figures_path)
    
    # Save result tables
    logger.info("Saving result tables...")
    
    # Save optimal allocations for each scenario
    for scenario_name, results in scenario_results.items():
        df_allocation = pd.DataFrame(
            results['optimal_U'],
            index=YEARS,
            columns=SECTOR_NAMES_EN
        )
        df_allocation.index.name = 'Year'
        df_allocation.to_csv(
            tables_path / f'sa4_optimal_allocation_{scenario_name}.csv',
            encoding='utf-8-sig'
        )
    
    # Save elasticity analysis results
    elasticity_df.to_csv(tables_path / 'sa4_budget_elasticity.csv', 
                         index=False, encoding='utf-8-sig')
    
    # Save Pareto frontier data
    pareto_df = pd.DataFrame({
        'Budget': pareto_data['budgets'],
        'Score': pareto_data['scores'],
        'Is_Pareto_Optimal': pareto_data['is_pareto']
    })
    pareto_df.to_csv(tables_path / 'sa4_pareto_frontier.csv', 
                     index=False, encoding='utf-8-sig')
    
    # Key findings
    logger.info("\nKey Findings:")
    
    baseline_score = scenario_results['baseline']['optimal_score']
    tight_score = scenario_results['tight']['optimal_score']
    loose_score = scenario_results['loose']['optimal_score']
    
    logger.info(f"  Baseline Score: {baseline_score:.4f}")
    logger.info(f"  Tight Budget Score: {tight_score:.4f} (Change: {(tight_score/baseline_score-1)*100:+.1f}%)")
    logger.info(f"  Loose Budget Score: {loose_score:.4f} (Change: {(loose_score/baseline_score-1)*100:+.1f}%)")
    
    if len(elasticity_df) > 0:
        avg_elasticity = elasticity_df['Elasticity'].mean()
        logger.info(f"  Average Budget Elasticity: {avg_elasticity:.3f}")
    
    # Count Pareto optimal solutions
    n_pareto = np.sum(pareto_data['is_pareto'])
    logger.info(f"  Number of Pareto Optimal Solutions: {n_pareto}")
    
    logger.info("SA-4 Analysis Complete!")
    logger.info(f"  Figures: {figures_path}")
    logger.info(f"  Tables: {tables_path}")
    
    return {
        'scenario_results': scenario_results,
        'pareto_data': pareto_data,
        'elasticity_df': elasticity_df
    }


if __name__ == '__main__':
    results = run_budget_sensitivity()
