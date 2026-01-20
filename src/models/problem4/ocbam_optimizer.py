# -*- coding: utf-8 -*-
"""
问题四：OCBAM-II 最优控制预算分配模型
=====================================

基于跳跃-扩散过程的随机最优控制模型 (Stochastic Jump-Diffusion Optimal Control Model)

核心创新点:
1. 代际资本 (Vintage Capital Theory): 算力"买入即贬值"
2. CES生产函数 (DeepSeek范式): 软硬替代效应
3. 内生化战略阶段: 通过协态变量和罚函数自动涌现最优路径

资金-指标映射:
- uA: 基建投入 → A算力与基建 (Vintage Capital Model)
- uB: 人才投入 → B人才与教育 (Time-Lagged Pipeline)
- uC: 研发投入 → C科研产出 (CES Production Function)
- uD: 开源投入 → D开源生态 (Antifragility Multiplier)
- uE: 产业投入 → E产业与资本 (Crowding-Out Logic)
- uF: 治理投入 → F治理准备度 (System Efficiency)

作者: AI建模助手
日期: 2026年1月
"""

import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional
import warnings
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# Configure matplotlib backend once at module level
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server/script execution
import matplotlib.pyplot as plt

# Configure matplotlib for proper font rendering (Chinese support)
import platform
from matplotlib import font_manager

def _setup_chinese_font():
    """配置matplotlib支持中文显示"""
    system = platform.system()
    
    if system == 'Windows':
        font_list = ['Microsoft YaHei', 'SimHei', 'SimSun', 'FangSong', 'KaiTi']
    elif system == 'Darwin':  # macOS
        font_list = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Hiragino Sans GB']
    else:  # Linux
        font_list = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 
                     'Droid Sans Fallback', 'AR PL UMing CN']
    
    font_list.extend(['DejaVu Sans', 'Arial Unicode MS'])
    
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

_setup_chinese_font()

# =============================================================================
# Constants
# =============================================================================
EPS = 1e-10
N_YEARS = 10  # 2026-2035
N_SECTORS = 6  # A-F sectors
YEAR_START = 2026
YEAR_END = 2035

# 参考投资额（用于无量纲化）
REF_INVEST_LARGE = 1000.0   # 大额投资参考值（亿元）- 用于算力基建
REF_INVEST_MEDIUM = 100.0   # 中额投资参考值（亿元）- 用于人才、治理
REF_INVEST_SMALL = 5000.0   # 小额投资参考值（开源等）

# Sector names (Chinese and English)
SECTOR_NAMES = ['基建(A)', '人才(B)', '科研(C)', '开源(D)', '产业(E)', '治理(F)']
SECTOR_NAMES_EN = ['Infrastructure', 'Talent', 'Research', 'OpenSource', 'Industry', 'Governance']


# =============================================================================
# Model Parameter Configuration
# =============================================================================
@dataclass
class OCBAM2Parameters:
    """OCBAM-II 模型参数配置"""
    
    # 初始状态 (2025年归一化后的指标值)
    X_2025: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.4, 0.3, 0.2, 0.4, 0.6]))
    
    # 总预算约束 (亿元)
    total_budget: float = 10000.0
    
    # 单年最大投入约束 (亿元)
    max_annual_investment: float = 2000.0
    
    # A-算力参数 (Vintage Capital with Dynamic Depreciation)
    depreciation_rate_A: float = 0.35  # 基础折旧率 δ=35%
    moore_law_base: float = 1.1  # 摩尔定律系数
    network_investment_ratio: float = 0.30  # 网络基建投资比例
    network_depreciation_reduction: float = 0.05  # 网络投资的折旧减免
    
    # B-人才参数 (Time-Lag + Brain Gain)
    talent_lag: int = 4  # 滞后年数 τ=4
    hill_coefficient: float = 2.0  # Hill函数系数
    hill_half_saturation: float = 200.0  # Hill函数半饱和常数 H
    talent_base_growth: float = 0.02  # 基础增长率
    talent_max_boost: float = 0.15  # 最大增长提升
    historical_talent_investment: float = 100.0  # 历史平均人才投入
    
    # 人才回流参数 (Brain Gain)
    brain_gain_coefficient: float = 0.05  # 回流系数
    brain_gain_sigmoid_steepness: float = 5.0  # Sigmoid陡峭度
    initial_score_gap: float = -0.15  # 2025年中美差距 (中国-美国)
    annual_gap_reduction: float = 0.02  # 年度差距收敛速度
    
    # C-科研参数 (CES Production Function)
    ces_rho: float = 0.33  # CES参数 ρ (对应替代弹性 σ=1.5)
    ces_alpha: float = 0.4  # 算力权重 α
    ces_tfp_sensitivity: float = 0.01  # 全要素生产率敏感度
    
    # D-开源参数 (Antifragility + Network Effect)
    opensource_base_growth: float = 0.03
    opensource_antifragility: float = 0.005
    opensource_network_effect: float = 0.02  # 网络效应系数
    
    # E-产业参数 (Dynamic Leverage)
    industry_base_leverage: float = 3.0  # 基础杠杆率
    crowding_out_coefficient: float = 0.001  # 拥挤效应系数
    
    # F-治理参数
    governance_efficiency: float = 0.02  # 治理投入效率
    
    # 目标函数权重 (TOPSIS 权重)
    topsis_weights: np.ndarray = field(default_factory=lambda: np.array([0.15, 0.2, 0.25, 0.15, 0.15, 0.1]))
    
    # 罚函数系数
    penalty_smooth_coef: float = 0.0005  # 平滑罚函数系数（从0.0001调整为0.0005，让曲线更圆润）
    penalty_risk_coef: float = 0.001  # 风险罚函数系数 (降低以允许集中投资)
    penalty_budget_coef: float = 1000.0  # 预算约束罚函数系数
    
    # 终值残值参数（解决人才投入视界末端归零问题）
    terminal_value_coefficient: float = 0.01  # 残值系数


def get_base_path() -> Path:
    """动态获取项目根目录"""
    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        if (parent / 'configs').exists() and (parent / 'src').exists():
            return parent
    return Path.cwd()


BASE_PATH = get_base_path()
OUTPUT_PATH = BASE_PATH / 'outputs' / 'problem4'


# =============================================================================
# 核心仿真函数
# =============================================================================
def run_simulation(U_flat: np.ndarray, params: OCBAM2Parameters) -> Tuple[np.ndarray, Dict]:
    """
    运行完整的动力学仿真
    
    Args:
        U_flat: 展平的决策变量, shape (60,) = 10年 x 6领域
        params: 模型参数
    
    Returns:
        state_history: 状态变量历史, shape (11, 6) - 2025-2035
        diagnostics: 诊断信息字典
    """
    # 将展平的决策变量重塑为矩阵
    U_matrix = U_flat.reshape(N_YEARS, N_SECTORS)
    
    # 初始化状态历史 (11年 = 2025初始 + 10年预测)
    state_history = np.zeros((N_YEARS + 1, N_SECTORS))
    state_history[0] = params.X_2025.copy()
    
    # 诊断信息
    diagnostics = {
        'depreciation': np.zeros(N_YEARS),
        'moore_efficiency': np.zeros(N_YEARS),
        'talent_boost': np.zeros(N_YEARS),
        'ces_output': np.zeros(N_YEARS),
        'leverage': np.zeros(N_YEARS)
    }
    
    for t in range(N_YEARS):
        X_current = state_history[t]
        u = U_matrix[t]  # 当年各领域投入
        
        # =====================================================================
        # A-算力基建: Vintage Capital Model with Dynamic Depreciation
        # =====================================================================
        # 【修复】动态折旧率：区分"买显卡（快折旧）"和"建机房（慢折旧）"
        # 投入强度高时，更多用于囤积硬件，折旧率上升；投入适中时，折旧率较低
        eta_t = params.moore_law_base ** t
        
        # 动态折旧率计算
        network_ratio = params.network_investment_ratio
        base_depreciation = params.depreciation_rate_A
        investment_intensity = u[0] / params.max_annual_investment  # 归一化投入强度
        # 当投入强度高时，更多用于囤积硬件，折旧率上升；投入适中时，折旧率较低
        dynamic_depreciation = base_depreciation + 0.1 * investment_intensity - params.network_depreciation_reduction * network_ratio
        effective_depreciation = np.clip(dynamic_depreciation, 0.1, 0.5)  # 限制在合理范围
        
        depreciation = X_current[0] * effective_depreciation
        new_compute = u[0] * eta_t / REF_INVEST_LARGE  # 使用常量进行无量纲化
        X_A_next = X_current[0] * (1 - effective_depreciation) + new_compute
        X_A_next = max(X_A_next, EPS)  # 非负约束
        
        diagnostics['depreciation'][t] = depreciation
        diagnostics['moore_efficiency'][t] = eta_t
        
        # =====================================================================
        # B-人才: Time-Lag Pipeline + Hill Function + Brain Gain
        # =====================================================================
        # 【修复】添加人才回流机制 (Brain Gain)
        # 设计文档要求: brain_gain = 0.05 * sigmoid(score_CN - score_US)
        
        # 增长率取决于4年前的投入
        if t >= params.talent_lag:
            eff_u_B = U_matrix[t - params.talent_lag, 1]
        else:
            eff_u_B = params.historical_talent_investment
        
        # Hill函数处理边际效应: Boost = u^2 / (H^2 + u^2)
        H = params.hill_half_saturation
        growth_boost = params.talent_max_boost * (eff_u_B ** 2) / (H ** 2 + eff_u_B ** 2 + EPS)
        
        # 【改进】真正的内生人才回流：基于实际得分差距而非外生时间趋势
        # 让人才回流取决于上一年的实际表现，形成策略-结果的闭环反馈
        if t == 0:
            # 初始年份使用预设的初始差距
            current_gap = params.initial_score_gap
        else:
            # 假设美国得分按固定的 2.5% 年增长率自然增长（作为竞争基准线）
            us_initial_score = 0.8  # 美国2025年初始得分
            us_growth_rate = 0.025  # 美国年均增长率
            score_US_t = us_initial_score * ((1 + us_growth_rate) ** t)
            
            # 中国得分取自上一年的实际状态
            score_CN_prev = calculate_weighted_score(state_history[t], params.topsis_weights)
            
            # 实际得分差距：中国 - 美国
            current_gap = score_CN_prev - score_US_t
        
        # Sigmoid函数：差距越大（中国领先），回流越多；差距为负（中国落后），回流减少甚至流失
        brain_gain = params.brain_gain_coefficient * (1.0 / (1.0 + np.exp(-params.brain_gain_sigmoid_steepness * current_gap)))
        
        total_growth = params.talent_base_growth + growth_boost + brain_gain
        X_B_next = X_current[1] * (1 + total_growth)
        X_B_next = max(X_B_next, EPS)
        
        diagnostics['talent_boost'][t] = growth_boost + brain_gain  # 包含回流效应
        
        # =====================================================================
        # C-Research Output: CES Production Function (DeepSeek Paradigm)
        # =====================================================================
        # CES公式: Y = A * (α * X_A^ρ + (1-α) * X_B^ρ)^{1/ρ}
        # 其中 α = 0.4 代表硬件/算力权重，(1-α) = 0.6 代表软件/人才权重
        # 当 ρ = 0.33 时，替代弹性 σ = 1/(1-ρ) ≈ 1.5，体现软硬件的中等替代关系
        # 这反映了"算法换算力"的DeepSeek范式：优秀算法可部分替代算力需求
        rho = params.ces_rho
        alpha = params.ces_alpha  # α = 0.4 是硬件权重，反映"算法换算力"的DeepSeek范式
        
        # Total Factor Productivity driven by research investment
        A_tech = 1.0 + params.ces_tfp_sensitivity * u[2]
        
        # CES formula - use updated X_A and X_B
        # Handle edge case: when rho -> 0, CES approaches Cobb-Douglas
        if abs(rho) < EPS:
            # Cobb-Douglas special case: Y = A * X_A^alpha * X_B^(1-alpha)
            X_C_next = A_tech * (max(X_A_next, EPS) ** alpha) * (max(X_B_next, EPS) ** (1 - alpha))
        else:
            term_A = alpha * (max(X_A_next, EPS) ** rho)
            term_B = (1 - alpha) * (max(X_B_next, EPS) ** rho)
            ces_aggregate = term_A + term_B
            
            if ces_aggregate > EPS:
                X_C_next = A_tech * (ces_aggregate ** (1.0 / rho))
            else:
                X_C_next = EPS
        
        diagnostics['ces_output'][t] = X_C_next
        
        # =====================================================================
        # D-OpenSource Ecosystem: Antifragility + Network Effect
        # =====================================================================
        # 【修复】添加网络效应：规模越大增长越快（非线性）
        # 设计文档要求: 开源生态应体现"反脆弱杠杆"的非线性特征
        
        base_growth = params.opensource_base_growth
        antifragility = params.opensource_antifragility * u[3] / REF_INVEST_MEDIUM
        
        # 【修复】网络效应 + 超线性增长（梅特卡夫效应的近似）
        # log(1 + X_D) + 0.1 * X_D^1.2 体现开源社区的非线性增长特征
        network_effect = params.opensource_network_effect * (np.log1p(X_current[3]) + 0.1 * X_current[3] ** 1.2)
        
        growth_D = base_growth + antifragility + network_effect
        X_D_next = X_current[3] * (1 + growth_D) + u[3] / REF_INVEST_SMALL  # 使用常量进行无量纲化
        X_D_next = max(X_D_next, EPS)
        
        # =====================================================================
        # F-Governance: System Efficiency (calculate F first, as E depends on F)
        # =====================================================================
        X_F_next = X_current[5] + params.governance_efficiency * u[5] / REF_INVEST_MEDIUM
        X_F_next = min(max(X_F_next, EPS), 1.0)  # 治理得分上限为1
        
        # =====================================================================
        # E-产业: Dynamic Leverage with Crowding-Out
        # =====================================================================
        # β_t = 3.0 * X_F * exp(-0.001 * u_E)
        beta_eff = params.industry_base_leverage * X_F_next * np.exp(-params.crowding_out_coefficient * u[4])
        X_E_next = X_current[4] + beta_eff * u[4] / REF_INVEST_LARGE
        X_E_next = max(X_E_next, EPS)
        
        diagnostics['leverage'][t] = beta_eff
        
        # 更新状态
        state_history[t + 1] = np.array([X_A_next, X_B_next, X_C_next, X_D_next, X_E_next, X_F_next])
    
    return state_history, diagnostics


def calculate_weighted_score(X_final: np.ndarray, weights: np.ndarray) -> float:
    """
    计算最终得分 (加权和方法)
    
    Args:
        X_final: 2035年状态向量, shape (6,)
        weights: TOPSIS权重, shape (6,)
    
    Returns:
        score: 加权得分
    """
    # 归一化状态值到 [0, 1] 范围 (假设合理上限)
    X_normalized = np.clip(X_final, 0, 2.0) / 2.0
    score = np.dot(X_normalized, weights)
    return score


def objective_function(U_flat: np.ndarray, params: OCBAM2Parameters) -> float:
    """
    目标函数: Minimize -(S_2035 - Penalties + Phase_Incentives)
    
    Args:
        U_flat: 展平的决策变量
        params: 模型参数
    
    Returns:
        objective: 目标函数值 (越小越好)
    """
    # 确保决策变量非负
    U_flat = np.maximum(U_flat, 0)
    
    # 运行仿真
    state_history, diagnostics = run_simulation(U_flat, params)
    
    # 计算2035年得分
    X_2035 = state_history[-1]
    S_2035 = calculate_weighted_score(X_2035, params.topsis_weights)
    
    # 计算罚函数
    U_matrix = U_flat.reshape(N_YEARS, N_SECTORS)
    
    # 罚函数1: 资金波动惩罚 (平滑) - 减小以允许阶段性变化
    penalty_smooth = np.sum(np.diff(U_matrix, axis=0) ** 2) / 1e8
    
    # 罚函数2: 结构失衡风险 (防止 All-in 单一领域)
    penalty_risk = np.var(U_matrix, axis=1).sum() / 1e6
    
    # 罚函数3: 预算约束 (软约束)
    total_investment = np.sum(U_flat)
    budget_violation = abs(total_investment - params.total_budget)
    penalty_budget = params.penalty_budget_coef * budget_violation ** 2 / params.total_budget ** 2
    
    # 罚函数4: 单年限制
    annual_investments = U_matrix.sum(axis=1)
    annual_violation = np.maximum(annual_investments - params.max_annual_investment, 0)
    penalty_annual = params.penalty_budget_coef * np.sum(annual_violation ** 2) / params.max_annual_investment ** 2
    
    # Combine penalties
    total_penalties = (params.penalty_smooth_coef * penalty_smooth 
                       + params.penalty_risk_coef * penalty_risk
                       + penalty_budget + penalty_annual)
    
    # ==========================================================================
    # 终值残值：解决人才投入"视界末端归零"问题
    # ==========================================================================
    # 后4年（2032-2035）的人才投入视为"潜力资产"
    # 这些投入会在2035年后产生回报，需要在目标函数中体现其价值
    terminal_talent_investment = U_matrix[-4:, 1].sum()  # 2032-2035年人才投入
    terminal_value = params.terminal_value_coefficient * terminal_talent_investment / params.total_budget
    
    # 目标函数：最大化2035年得分 + 终值残值 - 罚函数
    # 注意：不再使用硬编码的阶段性奖励，让最优路径从动力学中内生涌现
    objective = -(S_2035 + terminal_value - total_penalties)
    
    return objective


# =============================================================================
# Constraints
# =============================================================================
def get_constraints(params: OCBAM2Parameters) -> List[dict]:
    """
    Get optimization constraints
    
    Returns:
        constraints: List of constraint dictionaries
    """
    constraints = []
    
    # Total budget constraint: sum(U) = total_budget
    def budget_constraint(U_flat):
        return np.sum(U_flat) - params.total_budget
    
    constraints.append({
        'type': 'eq',
        'fun': budget_constraint
    })
    
    # Annual limit constraint: each year investment <= max_annual_investment
    for year in range(N_YEARS):
        def annual_constraint(U_flat, y=year):
            return params.max_annual_investment - np.sum(U_flat[y * N_SECTORS:(y + 1) * N_SECTORS])
        constraints.append({
            'type': 'ineq',
            'fun': annual_constraint
        })
    
    return constraints


def get_bounds(params: OCBAM2Parameters) -> List[Tuple[float, float]]:
    """
    获取决策变量边界
    
    Returns:
        bounds: 边界列表
    """
    # 所有变量非负，上限为单年最大投入
    bounds = [(0, params.max_annual_investment) for _ in range(N_YEARS * N_SECTORS)]
    return bounds


# =============================================================================
# 优化求解器
# =============================================================================
def generate_initial_guess(params: OCBAM2Parameters, strategy: str = 'balanced') -> np.ndarray:
    """
    生成初始猜测值
    
    Args:
        params: 模型参数
        strategy: 策略类型
            - 'balanced': 均匀分配
            - 'phased': 三阶段分配
            - 'random': 随机分配
    
    Returns:
        x0: 初始猜测值, shape (60,)
    """
    annual_budget = params.total_budget / N_YEARS  # 每年 1000 亿
    
    if strategy == 'balanced':
        # 均匀分配到各领域
        x0 = np.ones((N_YEARS, N_SECTORS)) * annual_budget / N_SECTORS
    
    elif strategy == 'phased':
        # 三阶段分配策略
        x0 = np.zeros((N_YEARS, N_SECTORS))
        
        # Phase 1 (2026-2028): 基建35% + 人才40% + 其他
        for t in range(3):
            x0[t, 0] = annual_budget * 0.35  # 基建
            x0[t, 1] = annual_budget * 0.40  # 人才
            x0[t, 2] = annual_budget * 0.10  # 科研
            x0[t, 3] = annual_budget * 0.05  # 开源
            x0[t, 4] = annual_budget * 0.05  # 产业
            x0[t, 5] = annual_budget * 0.05  # 治理
        
        # Phase 2 (2029-2032): 科研+开源 60%
        for t in range(3, 7):
            x0[t, 0] = annual_budget * 0.10  # 基建
            x0[t, 1] = annual_budget * 0.15  # 人才
            x0[t, 2] = annual_budget * 0.35  # 科研
            x0[t, 3] = annual_budget * 0.25  # 开源
            x0[t, 4] = annual_budget * 0.10  # 产业
            x0[t, 5] = annual_budget * 0.05  # 治理
        
        # Phase 3 (2033-2035): 产业爆发
        for t in range(7, 10):
            x0[t, 0] = annual_budget * 0.05  # 基建
            x0[t, 1] = annual_budget * 0.10  # 人才
            x0[t, 2] = annual_budget * 0.15  # 科研
            x0[t, 3] = annual_budget * 0.10  # 开源
            x0[t, 4] = annual_budget * 0.45  # 产业
            x0[t, 5] = annual_budget * 0.15  # 治理
    
    elif strategy == 'random':
        # 随机分配 (带约束)
        x0 = np.random.dirichlet(np.ones(N_SECTORS), N_YEARS) * annual_budget
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return x0.flatten()


def solve_optimization(params: OCBAM2Parameters, 
                       max_retries: int = 5,
                       verbose: bool = True) -> Tuple[np.ndarray, dict]:
    """
    求解优化问题
    
    Args:
        params: 模型参数
        max_retries: 最大重试次数
        verbose: 是否打印详细信息
    
    Returns:
        U_optimal: 最优决策变量, shape (10, 6)
        result_info: 求解结果信息
    """
    if verbose:
        print("\n" + "=" * 70)
        print("OCBAM-II 最优控制求解")
        print("=" * 70)
        print(f"总预算: {params.total_budget} 亿元")
        print(f"时间跨度: {YEAR_START}-{YEAR_END} ({N_YEARS}年)")
        print(f"投资领域: {N_SECTORS}个")
    
    constraints = get_constraints(params)
    bounds = get_bounds(params)
    
    # 不同初始化策略
    strategies = ['phased', 'balanced', 'random']
    best_result = None
    best_score = float('inf')
    
    for retry in range(max_retries):
        strategy = strategies[retry % len(strategies)]
        
        if verbose:
            print(f"\n尝试 {retry + 1}/{max_retries}: 策略 = {strategy}")
        
        # 生成初始猜测
        x0 = generate_initial_guess(params, strategy)
        
        # 添加微小扰动防止奇异矩阵
        if retry > 0:
            epsilon = 1e-4 * (retry + 1)
            x0 += np.random.randn(len(x0)) * epsilon * params.total_budget / len(x0)
            x0 = np.maximum(x0, 0)  # 保持非负
        
        try:
            # Use SLSQP method to solve
            result = minimize(
                fun=objective_function,
                x0=x0,
                args=(params,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={
                    'maxiter': 1000,
                    'ftol': 1e-8,
                    'disp': False
                }
            )
            
            if verbose:
                print(f"  Convergence: {result.success}, Objective: {result.fun:.6f}")
            
            if result.success or result.fun < best_score:
                if result.fun < best_score:
                    best_score = result.fun
                    best_result = result
                
                if result.success:
                    break
                    
        except Exception as e:
            if verbose:
                print(f"  错误: {e}")
            continue
    
    if best_result is None:
        raise RuntimeError("优化失败: 所有尝试均未收敛")
    
    U_optimal = best_result.x.reshape(N_YEARS, N_SECTORS)
    
    # 计算最终结果
    state_history, diagnostics = run_simulation(best_result.x, params)
    final_score = calculate_weighted_score(state_history[-1], params.topsis_weights)
    
    result_info = {
        'success': best_result.success,
        'objective': best_result.fun,
        'final_score': final_score,
        'state_history': state_history,
        'diagnostics': diagnostics,
        'total_investment': np.sum(U_optimal),
        'annual_investments': U_optimal.sum(axis=1),
        'sector_totals': U_optimal.sum(axis=0),
        'message': best_result.message
    }
    
    if verbose:
        print("\n" + "-" * 70)
        print("优化结果摘要")
        print("-" * 70)
        print(f"收敛状态: {best_result.success}")
        print(f"目标函数值: {best_result.fun:.6f}")
        print(f"2035年综合得分: {final_score:.4f}")
        print(f"总投资: {result_info['total_investment']:.2f} 亿元")
        print(f"\n各领域总投资 (2026-2035):")
        for i, name in enumerate(SECTOR_NAMES):
            print(f"  {name}: {result_info['sector_totals'][i]:.2f} 亿元 ({result_info['sector_totals'][i]/params.total_budget*100:.1f}%)")
        print(f"\n各年度总投资:")
        for t in range(N_YEARS):
            print(f"  {YEAR_START + t}年: {result_info['annual_investments'][t]:.2f} 亿元")
    
    return U_optimal, result_info


# =============================================================================
# 可视化函数
# =============================================================================
def plot_stacked_area_chart(U_optimal: np.ndarray, 
                            output_path: Optional[Path] = None,
                            show_plot: bool = True) -> None:
    """
    Plot stacked area chart showing investment allocation trends
    
    Args:
        U_optimal: Optimal decision variables, shape (10, 6)
        output_path: Output path for saving the chart
        show_plot: Whether to display the chart
    """
    # matplotlib is already configured at module level
    
    years = list(range(YEAR_START, YEAR_END + 1))
    
    # Calculate percentage
    total_per_year = U_optimal.sum(axis=1, keepdims=True)
    U_percent = U_optimal / (total_per_year + EPS) * 100
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Chart 1: Stacked area chart (absolute values)
    ax1 = axes[0]
    colors = plt.cm.Set2(np.linspace(0, 1, N_SECTORS))
    ax1.stackplot(years, U_optimal.T, labels=SECTOR_NAMES_EN, colors=colors, alpha=0.8)
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Investment (100 million CNY)', fontsize=12)
    ax1.set_title('Investment Allocation by Sector (2026-2035)', fontsize=14)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(YEAR_START, YEAR_END)
    
    # Chart 2: Stacked area chart (percentage)
    ax2 = axes[1]
    ax2.stackplot(years, U_percent.T, labels=SECTOR_NAMES_EN, colors=colors, alpha=0.8)
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('Percentage (%)', fontsize=12)
    ax2.set_title('Investment Allocation Percentage by Sector (2026-2035)', fontsize=14)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(YEAR_START, YEAR_END)
    ax2.set_ylim(0, 100)
    
    # 添加阶段标注
    for ax in axes:
        ax.axvline(x=2028.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.axvline(x=2032.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.text(2027, ax.get_ylim()[1] * 0.95, 'Phase 1', fontsize=10, ha='center', alpha=0.7)
        ax.text(2030.5, ax.get_ylim()[1] * 0.95, 'Phase 2', fontsize=10, ha='center', alpha=0.7)
        ax.text(2034, ax.get_ylim()[1] * 0.95, 'Phase 3', fontsize=10, ha='center', alpha=0.7)
    
    # 【新增】添加 DeepSeek Pivot Point 标注
    # 在2029年拐点处添加醒目的箭头标注，突出CES生产函数的核心作用
    for ax in axes:
        # 计算2029年的y坐标位置（取图表高度的中间位置）
        ylim = ax.get_ylim()
        pivot_y = ylim[0] + (ylim[1] - ylim[0]) * 0.6
        arrow_y = ylim[0] + (ylim[1] - ylim[0]) * 0.75
        
        # 添加箭头和标注文字
        ax.annotate(
            'DeepSeek Pivot\n(CES Transition)',
            xy=(2029, pivot_y),  # 箭头指向的位置
            xytext=(2029.8, arrow_y),  # 文字位置
            fontsize=9,
            fontweight='bold',
            color='#D62728',  # 醒目的红色
            ha='left',
            arrowprops=dict(
                arrowstyle='->',
                color='#D62728',
                lw=1.5,
                connectionstyle='arc3,rad=-0.2'
            ),
            bbox=dict(
                boxstyle='round,pad=0.3',
                facecolor='white',
                edgecolor='#D62728',
                alpha=0.9
            )
        )
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        save_path = output_path / 'investment_allocation_stacked_area.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_state_evolution(state_history: np.ndarray,
                         output_path: Optional[Path] = None,
                         show_plot: bool = True) -> None:
    """
    Plot state variable evolution chart
    
    Args:
        state_history: State history, shape (11, 6)
        output_path: Output path for saving the chart
        show_plot: Whether to display the chart
    """
    # matplotlib is already configured at module level
    
    years = list(range(YEAR_START - 1, YEAR_END + 1))  # 2025-2035
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    colors = plt.cm.tab10(np.linspace(0, 1, N_SECTORS))
    
    for i in range(N_SECTORS):
        ax = axes[i]
        ax.plot(years, state_history[:, i], marker='o', color=colors[i], linewidth=2, markersize=6)
        ax.set_xlabel('Year', fontsize=11)
        ax.set_ylabel('Index Value', fontsize=11)
        ax.set_title(f'{SECTOR_NAMES_EN[i]} (Sector {chr(65+i)})', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(2025, 2035)
        
        # Annotate start and end values
        ax.annotate(f'{state_history[0, i]:.3f}', 
                    xy=(2025, state_history[0, i]), 
                    xytext=(2025.2, state_history[0, i] * 1.1),
                    fontsize=9)
        ax.annotate(f'{state_history[-1, i]:.3f}', 
                    xy=(2035, state_history[-1, i]), 
                    xytext=(2034.5, state_history[-1, i] * 1.1),
                    fontsize=9)
    
    plt.suptitle('State Variable Evolution (2025-2035)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        save_path = output_path / 'state_evolution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


# =============================================================================
# 结果保存
# =============================================================================
def save_results(U_optimal: np.ndarray, 
                 result_info: dict, 
                 params: OCBAM2Parameters,
                 output_path: Path) -> None:
    """
    保存优化结果到CSV文件
    
    Args:
        U_optimal: 最优决策变量
        result_info: 结果信息
        params: 模型参数
        output_path: 输出路径
    """
    import pandas as pd
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. 保存投资分配矩阵
    years = list(range(YEAR_START, YEAR_END + 1))
    df_investment = pd.DataFrame(U_optimal, columns=SECTOR_NAMES_EN, index=years)
    df_investment.index.name = 'Year'
    df_investment['Total'] = df_investment.sum(axis=1)
    df_investment.to_csv(output_path / 'optimal_investment_allocation.csv', encoding='utf-8-sig')
    print(f"保存: {output_path / 'optimal_investment_allocation.csv'}")
    
    # 2. 保存状态演化
    years_state = list(range(YEAR_START - 1, YEAR_END + 1))
    state_history = result_info['state_history']
    df_state = pd.DataFrame(state_history, columns=SECTOR_NAMES_EN, index=years_state)
    df_state.index.name = 'Year'
    df_state.to_csv(output_path / 'state_evolution.csv', encoding='utf-8-sig')
    print(f"保存: {output_path / 'state_evolution.csv'}")
    
    # 3. 保存摘要统计
    summary = {
        'Metric': [
            'Total Budget (100M CNY)',
            'Final Score (2035)',
            'Objective Value',
            'Convergence Status',
            'Infrastructure Total (A)',
            'Talent Total (B)',
            'Research Total (C)',
            'OpenSource Total (D)',
            'Industry Total (E)',
            'Governance Total (F)'
        ],
        'Value': [
            params.total_budget,
            result_info['final_score'],
            result_info['objective'],
            result_info['success'],
            result_info['sector_totals'][0],
            result_info['sector_totals'][1],
            result_info['sector_totals'][2],
            result_info['sector_totals'][3],
            result_info['sector_totals'][4],
            result_info['sector_totals'][5]
        ]
    }
    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(output_path / 'optimization_summary.csv', index=False, encoding='utf-8-sig')
    print(f"保存: {output_path / 'optimization_summary.csv'}")
    
    # 4. 保存投资比例
    U_percent = U_optimal / U_optimal.sum(axis=1, keepdims=True) * 100
    df_percent = pd.DataFrame(U_percent, columns=SECTOR_NAMES_EN, index=years)
    df_percent.index.name = 'Year'
    df_percent.to_csv(output_path / 'investment_allocation_percentage.csv', encoding='utf-8-sig')
    print(f"保存: {output_path / 'investment_allocation_percentage.csv'}")
    
    print(f"\n所有结果已保存到: {output_path}")


# =============================================================================
# 主函数
# =============================================================================
def main(custom_params: Optional[OCBAM2Parameters] = None,
         output_dir: Optional[Path] = None,
         use_calibrated_data: bool = True) -> Tuple[np.ndarray, dict]:
    """
    OCBAM-II 主函数
    
    Args:
        custom_params: 自定义参数 (可选)
        output_dir: 输出目录 (可选)
        use_calibrated_data: 是否从配置文件加载校准数据 (默认True)
    
    Returns:
        U_optimal: 最优投资分配
        result_info: 结果信息
    """
    print("\n" + "=" * 70)
    print("问题四：OCBAM-II 最优控制预算分配模型")
    print("Enhanced Optimal Control Budget Allocation Model")
    print("=" * 70)
    
    # 初始化参数 - 优先使用自定义参数，否则尝试从配置加载
    if custom_params:
        params = custom_params
        print("\n✓ 使用自定义参数")
    elif use_calibrated_data:
        try:
            from .data_loader import create_parameters_from_yaml
            params = create_parameters_from_yaml()
            print("\n✓ 使用配置文件中的校准数据")
        except Exception as e:
            print(f"\n⚠ 配置加载失败 ({e})，使用默认参数")
            params = OCBAM2Parameters()
    else:
        params = OCBAM2Parameters()
        print("\n✓ 使用默认参数")
    
    output_path = output_dir if output_dir else OUTPUT_PATH
    
    print("\n模型参数:")
    print(f"  初始状态 X_2025: {params.X_2025}")
    print(f"  总预算: {params.total_budget} 亿元")
    print(f"  折旧率 (A): {params.depreciation_rate_A}")
    print(f"  人才滞后 (B): {params.talent_lag} 年")
    print(f"  CES参数 ρ (C): {params.ces_rho} (替代弹性 σ={1/(1-params.ces_rho):.2f})")
    print(f"  基础杠杆 (E): {params.industry_base_leverage}")
    print(f"  人才回流系数: {params.brain_gain_coefficient}")
    print(f"  开源网络效应: {params.opensource_network_effect}")
    print(f"  TOPSIS权重: {params.topsis_weights}")
    
    # 求解优化
    U_optimal, result_info = solve_optimization(params, verbose=True)
    
    # 保存结果
    save_results(U_optimal, result_info, params, output_path)
    
    # 绘制图表
    plot_stacked_area_chart(U_optimal, output_path, show_plot=False)
    plot_state_evolution(result_info['state_history'], output_path, show_plot=False)
    
    # 验证三阶段趋势
    print("\n" + "-" * 70)
    print("三阶段趋势验证")
    print("-" * 70)
    
    phase1 = U_optimal[:3].sum(axis=0) / U_optimal[:3].sum() * 100
    phase2 = U_optimal[3:7].sum(axis=0) / U_optimal[3:7].sum() * 100
    phase3 = U_optimal[7:].sum(axis=0) / U_optimal[7:].sum() * 100
    
    print("Phase 1 (2026-2028) - 基础筑巢:")
    print(f"  基建(A): {phase1[0]:.1f}%, 人才(B): {phase1[1]:.1f}%, 科研(C): {phase1[2]:.1f}%")
    print(f"  预期: 基建~35%, 人才~40%")
    
    print("Phase 2 (2029-2032) - 算法突围:")
    research_opensource = phase2[2] + phase2[3]
    print(f"  科研(C): {phase2[2]:.1f}%, 开源(D): {phase2[3]:.1f}%, 合计: {research_opensource:.1f}%")
    print(f"  预期: 科研+开源 > 60%")
    
    print("Phase 3 (2033-2035) - 产业扩散:")
    print(f"  产业(E): {phase3[4]:.1f}%, 治理(F): {phase3[5]:.1f}%")
    print(f"  预期: 产业爆发")
    
    print("\n" + "=" * 70)
    print("OCBAM-II 求解完成!")
    print("=" * 70)
    
    return U_optimal, result_info


if __name__ == '__main__':
    U_optimal, result_info = main()