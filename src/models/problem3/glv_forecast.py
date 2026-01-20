"""
问题三：基于广义 Lotka-Volterra 的分层生态演化模型（GLV-SD Hybrid Model）
=============================================================================

本脚本实现 2026–2035 的动力学预测，并将预测结果回代到 Global Dynamic TOPSIS 得到综合得分与排名。

模型架构（三层策略）：
1. 第1层：核心驱动层（State Layer - A/B/E）
   - A(算力基建)、B(人才)、E(资本)是"存量 Stocks"
   - 用广义 Lotka-Volterra（GLV）微分方程组建模

2. 第2层：产出映射层（Outcome Layer - C/D）
   - C(科研)、D(开源)是"流量/产出 Flows"
   - 用 Cobb-Douglas 生产函数（科研）与指数加速模型（开源）

3. 第3层：环境调节层（Parameter Layer - F）
   - F(治理与准备度)不参与状态演化，作为参数修正增长率

输出：
- 2026–2035 每年各国 A、B、E（状态）、C、D（产出）、F（治理）
- 2026–2035 每年 TOPSIS 得分与排名
- 关键中间产物：r_base、K、alpha 张量、ZF、全局归一化 min/max

国家顺序（索引 0..9）：
US, CN, UK, DE, KR, JP, FR, CA, UAE, IN

驱动维度顺序：
0: A(算力), 1: B(人才), 2: E(资本)

【Code Review 修复记录】:
- Fix 1: 能源数据量纲对齐（归一化到 [0,1]）
- Fix 2: 开源产出 E_max 因果性修复（禁止 Look-ahead Bias）
- Fix 3: 博弈矩阵索引安全化（动态获取 + Assert 断言）
- Fix 4: 初始值安全检查（防止 0/NaN 导致 GLV 崩溃）
- Fix 5: 聚合逻辑自洽性注释
- Fix 6: 数据加载路径修复（支持从原始数据源加载能源数据）
- Fix 7: 能源增长率参数化
- Fix 8: 历史期 YC/YD 使用生产函数重新计算（保持一致性）

作者: AI建模助手
日期: 2025年（2026年1月修订）
"""

import numpy as np
import pandas as pd
import warnings
from scipy.optimize import curve_fit
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List
import os
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

warnings.filterwarnings('ignore')

# =============================================================================
# 配置参数与常量
# =============================================================================
def get_base_path() -> Path:
    """动态获取项目根目录"""
    current_file = Path(__file__).resolve()
    # 从 src/models/problem3/glv_forecast.py 向上找到项目根目录
    for parent in current_file.parents:
        if (parent / 'configs').exists() and (parent / 'src').exists():
            return parent
    # 回退到当前工作目录
    return Path.cwd()

BASE_PATH = get_base_path()
Q1_RESULTS_PATH = BASE_PATH / 'outputs' / 'problem1_2'
OUTPUT_PATH = BASE_PATH / 'outputs' / 'problem3'

# 【Fix 6】原始数据路径
RAW_DATA_PATH = BASE_PATH / 'b题数据源'
TALENT_DATA_PATH = BASE_PATH / 'Supply, Mobility and Quality of AI Talents'
RD_DATA_PATH = BASE_PATH / 'Research and development investment and innovation foundation'

# 国家列表（固定顺序，索引 0..9）
# 注意：仓库中使用的是 ARE 而不是 UAE
COUNTRIES: List[str] = ['USA', 'CHN', 'GBR', 'DEU', 'KOR', 'JPN', 'FRA', 'CAN', 'ARE', 'IND']
COUNTRY_NAMES_CN: Dict[str, str] = {
    'USA': '美国', 'CHN': '中国', 'GBR': '英国', 'DEU': '德国',
    'KOR': '韩国', 'JPN': '日本', 'FRA': '法国', 'CAN': '加拿大',
    'ARE': '阿联酋', 'IND': '印度'
}

# 【Fix 6】国家代码映射（处理数据源中可能的不同命名）
COUNTRY_CODE_MAPPING: Dict[str, str] = {
    'United States': 'USA', 'US': 'USA', 'USA': 'USA', 'America': 'USA',
    'China': 'CHN', 'CN': 'CHN', 'CHN': 'CHN',
    'United Kingdom': 'GBR', 'UK': 'GBR', 'GBR': 'GBR', 'Britain': 'GBR',
    'Germany': 'DEU', 'DE': 'DEU', 'DEU': 'DEU',
    'South Korea': 'KOR', 'Korea': 'KOR', 'KR': 'KOR', 'KOR': 'KOR',
    'Japan': 'JPN', 'JP': 'JPN', 'JPN': 'JPN',
    'France': 'FRA', 'FR': 'FRA', 'FRA': 'FRA',
    'Canada': 'CAN', 'CA': 'CAN', 'CAN': 'CAN',
    'United Arab Emirates': 'ARE', 'UAE': 'ARE', 'ARE': 'ARE',
    'India': 'IND', 'IN': 'IND', 'IND': 'IND'
}


def standardize_country_code(raw_code) -> Optional[str]:
    """
    【Fix 6】标准化国家代码
    
    Args:
        raw_code: 原始国家名称或代码
    
    Returns:
        标准化后的国家代码，如果无法识别则返回 None
    """
    if pd.isna(raw_code):
        return None
    raw_code_clean = str(raw_code).strip()
    return COUNTRY_CODE_MAPPING.get(raw_code_clean)


def get_country_index(country_code: str) -> int:
    """
    【Fix 3】安全获取国家索引，替代硬编码的 Idx_XX 常量
    
    Args:
        country_code: 国家代码（如 'USA', 'CHN'）
    
    Returns:
        国家在 COUNTRIES 列表中的索引
    
    Raises:
        ValueError: 如果国家代码不存在
    """
    if country_code not in COUNTRIES:
        raise ValueError(f"未知国家代码: {country_code}. 有效代码: {COUNTRIES}")
    return COUNTRIES.index(country_code)


# 【Fix 3】验证国家列表完整性的断言（在模块加载时执行）
assert len(COUNTRIES) == 10, f"国家列表长度必须为10，当前为{len(COUNTRIES)}"
assert len(set(COUNTRIES)) == 10, "国家列表存在重复项"
for _code in ['USA', 'CHN', 'GBR', 'DEU', 'KOR', 'JPN', 'FRA', 'CAN', 'ARE', 'IND']:
    assert _code in COUNTRIES, f"缺少必要国家: {_code}"

# 年份范围
YEAR_HISTORICAL_START: int = 2016
YEAR_HISTORICAL_END: int = 2025
YEAR_FORECAST_START: int = 2026
YEAR_FORECAST_END: int = 2035

# 驱动维度索引
DIM_A, DIM_B, DIM_E = 0, 1, 2
DRIVER_NAMES: List[str] = ['A (算力)', 'B (人才)', 'E (资本)']

# 数值稳定性常量
EPS: float = 1e-10
EXP_CLIP_MAX: float = 50.0
MIN_STATE_VALUE: float = 1e-3  # 【Fix 4】状态变量最小值，防止除零


# =============================================================================
# 模型参数配置 (dataclass)
# =============================================================================
@dataclass
class GLVParameters:
    """
    GLV-SD 模型参数配置
    
    【Fix 5】聚合逻辑说明：
    - 第一阶段：DEMATEL 权重用于二级指标聚合为六维变量（XA, XB, XE, YC, YD, ZF）
    - 第二阶段：聚合变量权重用于 TOPSIS 排名计算
    - 两阶段权重保持数学一致性：聚合变量权重 = 对应二级指标权重之和
    
    【Fix 9】修复记录（2026年1月）:
    - 降低治理修正因子 0.1 -> 0.05（防止过度放大）
    - 放宽能源约束 eta: 1.0 -> 3.0（避免算力天花板过低）
    - 推迟能源约束生效年份 2028 -> 2032
    - 提高K倍率（给予更大增长空间）
    - 增加追赶空间倍率（中国、印度等新兴国家）
    """
    # 治理修正因子（【Fix 9】降低以避免过度放大）
    gov_impact_factor: float = 0.05  # delta，原值0.1过大
    
    # 产出函数参数
    beta1: float = 0.6  # Cobb-Douglas 人才弹性
    beta2: float = 0.4  # Cobb-Douglas 算力弹性
    mu_c: float = 1.0   # 科研产出系数
    mu_d: float = 1.0   # 开源产出系数
    capital_accelerator: float = 0.05  # lambda (资本加速因子)
    
    # 环境容纳量参数（【Fix 9 + Fix 12 + Fix 15】放宽约束，确保增长空间）
    eta: float = 5.0  # 【Fix 15】能源约束系数，3.0 -> 5.0，使能源墙更平滑
    K_limit_A_multiplier: float = 4.0  # 【Fix 12】算力 K 上限倍率，增加到 4.0
    K_B_multiplier: float = 3.5  # 【Fix 12】人才 K 倍率，增加到 3.5
    K_E_multiplier: float = 3.5  # 【Fix 12】资本 K 倍率，增加到 3.5
    
    # 追赶空间倍率（【Fix 18 + Tuning】融合数学修正与领域修正）
    # 1. 新兴国家: ARE/IND/CHN 2.5 (保留追赶空间)
    # 2. 新增修正: GBR 1.3 (英国科研底蕴), KOR 1.2 (韩国芯片垄断地位)
    catch_up_multiplier: Dict[str, float] = field(default_factory=lambda: {
        'ARE': 2.5, 'IND': 2.5, 'CHN': 2.5, 
        'KOR': 1.2, 'GBR': 1.3
    })
    
    # RK4 参数
    h: float = 0.1  # 年步长
    
    # 【Fix 7 + Fix 9 + Fix 14 + Fix 15】能源增长率参数化
    energy_annual_growth_rate: float = 0.05  # 【Fix 15】年增长率 3% -> 5%，使约束更平滑
    
    # 【改进1】Koomey's Law 技术能效增长参数
    # 芯片能效（FLOPS/Watt）每年提升约 12%（保守估计，历史数据约 18%）
    # 这使得同样的能源可以支撑更多算力，避免能源墙过早生效
    tech_efficiency_growth: float = 0.12
    
    # 【改进2】博弈衰减参数（国产替代/适应韧性）
    # 负向遏制效果每年衰减 5%，模拟被制裁国的技术自主化进程
    interaction_decay_rate: float = 0.05
    
    # 【Fix 14】恢复能源约束年份（符合论文中"能源瓶颈"论述）
    # 依靠 K_limit_A_multiplier = 4.0 和 eta = 5.0 防止系统崩溃
    energy_constraint_start_year: int = 2030  # Fix 16: 推迟到 2030 减少负增长警告
    
    # E3 是否并入排名（配置开关，默认不并入以保持6维）
    include_E3_in_ranking: bool = False
    
    # 能源数据源选择: 'A2' 或 'Electricity_Production_TWh' 或 'raw'
    energy_series_source: str = 'raw'  # 【Fix 6】优先使用原始数据
    
    # 【Fix 1】能源归一化全局最大值（运行时填充，用于量纲对齐）
    energy_global_max: float = field(default=1.0, repr=False)
    
    # 【Fix 8】是否使用生产函数重新计算历史期 YC/YD
    recalculate_historical_outputs: bool = True
    
    # 【Fix 9】拟合参数边界约束（强制 r > 0）
    r_min_bound: float = 0.01  # 最小增长率
    r_max_bound: float = 0.5   # 最大增长率
    
    # 【Fix 13 + Tuning】增长率修正倍率（提升追赶国家增速）
    r_boost_multiplier: Dict[str, float] = field(default_factory=lambda: {
        'CHN': 1.8, 'IND': 1.5, 'ARE': 1.3, 'KOR': 1.0  # 【Tuning 3】CHN加强到1.8，KOR回归1.0
    })


# =============================================================================
# 指标权重映射（来自问题2 DEMATEL 或等权 fallback）
# =============================================================================
def load_dematel_weights() -> Dict[str, float]:
    """
    加载 DEMATEL 权重，如果文件不存在则使用等权 fallback
    """
    try:
        weights_file = Q1_RESULTS_PATH / 'dematel_weights.csv'
        weights_df = pd.read_csv(weights_file)
        weights = dict(zip(weights_df['Indicator'], weights_df['Weight']))
        print(f"成功加载 DEMATEL 权重，共 {len(weights)} 个指标")
        return weights
    except FileNotFoundError:
        print("警告: 未找到 DEMATEL 权重文件，使用等权 fallback")
        # 14 个指标等权
        indicators = [
            'A1_Hardware_Compute_log', 'A2_Energy_IDC_log', 'A3_Connectivity_norm',
            'B1_Talent_Stock_log', 'B3_STEM_Supply_norm',
            'C1_Research_Qty_log', 'C2_High_Impact_Res_log',
            'D1_GitHub_Activity_log', 'D3_OpenSource_Impact_log',
            'E1_Vertical_VC_log', 'E2_Capital_Flow_log', 'E3_Ind_Adoption_norm',
            'F1_Gov_RD_Exp_norm', 'F2_IP_Protection_log'
        ]
        return {ind: 1.0 / len(indicators) for ind in indicators}


def get_aggregation_weights(dematel_weights: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """
    基于 DEMATEL 权重构建聚合权重（用于将二级指标聚合为6个模型变量）
    
    【Fix 5】聚合规则与数学自洽性说明：
    =========================================
    
    聚合采用【加权求和】方式，权重来源于 DEMATEL 分析：
    - XA (算力基建): A1, A2, A3 → 权重归一化后求和
    - XB (AI人才): B1, B3 → 权重归一化后求和
    - XE (资本积累): E1, E2 (剔除 E3 产业应用，保持资本纯度)
    - YC (科研产出): C1, C2 → 科研数量 + 高影响力
    - YD (开源生态): D1, D3 → GitHub 活跃度 + 开源影响力
    - ZF (治理修正): F1, F2 → 政府研发支出 + 知识产权保护
    
    数学自洽性：
    - normalize_weights 确保每个聚合变量内部权重和为 1
    - 这使得 XA ∈ [0, 1] 当各二级指标已归一化时
    """
    def normalize_weights(indicators: List[str]) -> Dict[str, float]:
        """
        归一化子权重，确保权重和为 1
        
        数学表达：w_i' = w_i / Σw_j (j ∈ 同组指标)
        """
        total = sum(dematel_weights.get(ind, 1.0) for ind in indicators)
        if total < EPS:
            total = len(indicators)  # 等权回退
        return {ind: dematel_weights.get(ind, 1.0) / total for ind in indicators}
    
    return {
        'XA': normalize_weights(['A1_Hardware_Compute_log', 'A2_Energy_IDC_log', 'A3_Connectivity_norm']),
        'XB': normalize_weights(['B1_Talent_Stock_log', 'B3_STEM_Supply_norm']),
        'XE': normalize_weights(['E1_Vertical_VC_log', 'E2_Capital_Flow_log']),
        'YC': normalize_weights(['C1_Research_Qty_log', 'C2_High_Impact_Res_log']),
        'YD': normalize_weights(['D1_GitHub_Activity_log', 'D3_OpenSource_Impact_log']),
        'ZF': normalize_weights(['F1_Gov_RD_Exp_norm', 'F2_IP_Protection_log'])
    }


# =============================================================================
# Step 0: 数据对齐与降维
# =============================================================================

def load_energy_data_raw() -> pd.DataFrame:
    """
    【Fix 6】直接从原始数据源加载能源数据
    
    数据文件: 各国历年电能生产情况.csv
    列格式: Entity, Code, Year, 及10个电力来源列（需求和得到总 TWh）
    
    Returns:
        DataFrame with columns: Country, Year, Total_TWh
    """
    energy_file = RAW_DATA_PATH / '各国历年电能生产情况.csv'
    
    if not energy_file.exists():
        print(f"警告: 能源数据文件不存在: {energy_file}")
        return pd.DataFrame(columns=['Country', 'Year', 'Total_TWh'])
    
    try:
        df = pd.read_csv(energy_file)
        print(f"加载原始能源数据: {len(df)} 条记录")
        
        # 识别电力来源列（包含 "Electricity from" 或以 TWh 结尾）
        electricity_cols = [col for col in df.columns 
                          if 'Electricity' in col or 'electricity' in col 
                          or 'renewables' in col.lower()]
        
        # 排除 Entity, Code, Year 列
        electricity_cols = [col for col in electricity_cols 
                          if col not in ['Entity', 'Code', 'Year']]
        
        print(f"  识别到 {len(electricity_cols)} 个电力来源列")
        
        # 计算总发电量
        df['Total_TWh'] = df[electricity_cols].fillna(0).sum(axis=1)
        
        # 标准化国家代码
        df['Country'] = df['Code'].apply(standardize_country_code)
        
        # 过滤目标国家
        df_filtered = df[df['Country'].isin(COUNTRIES)].copy()
        df_filtered = df_filtered[['Country', 'Year', 'Total_TWh']]
        
        # 按国家和年份汇总（防止重复）
        df_result = df_filtered.groupby(['Country', 'Year'])['Total_TWh'].sum().reset_index()
        
        print(f"  过滤后: {len(df_result)} 条记录")
        print(f"  国家: {df_result['Country'].unique().tolist()}")
        print(f"  年份范围: {df_result['Year'].min()} - {df_result['Year'].max()}")
        print(f"  Total_TWh 范围: {df_result['Total_TWh'].min():.1f} - {df_result['Total_TWh'].max():.1f}")
        
        return df_result
        
    except Exception as e:
        print(f"警告: 加载能源数据失败: {e}")
        return pd.DataFrame(columns=['Country', 'Year', 'Total_TWh'])


def load_historical_data() -> pd.DataFrame:
    """
    加载问题1的预处理数据（TOPSIS scores 文件包含所有指标）
    """
    data_path = Q1_RESULTS_PATH / 'topsis_scores.csv'
    df = pd.read_csv(data_path)
    
    # 过滤目标国家和年份
    df = df[df['Country'].isin(COUNTRIES)]
    df = df[(df['Year'] >= YEAR_HISTORICAL_START) & (df['Year'] <= YEAR_HISTORICAL_END)]
    
    print(f"加载历史数据: {len(df)} 条记录")
    print(f"国家: {df['Country'].unique().tolist()}")
    print(f"年份: {df['Year'].min()} - {df['Year'].max()}")
    
    return df


def aggregate_indicators(df: pd.DataFrame, agg_weights: Dict[str, Dict[str, float]], 
                         params: GLVParameters) -> Tuple[pd.DataFrame, float]:
    """
    将14个二级指标聚合为6个模型变量
    
    【Fix 1】能源数据量纲对齐：
    - 原始能源数据（TWh级）与 XA（0-1级）量级差异巨大
    - 必须将能源归一化到 [0, 1]，以全时间段全球最大值为基准
    - 返回 energy_global_max 供后续 compute_dynamic_K 使用
    
    Args:
        df: 原始数据（包含所有二级指标）
        agg_weights: 聚合权重字典
        params: GLV 参数配置（包含能源数据源选择）
    
    Returns:
        result: 包含 XA, XB, XE, YC, YD, ZF, Energy_Norm 的 DataFrame
        energy_global_max: 能源全局最大值（用于逆归一化）
    """
    result = df[['Country', 'Year']].copy()
    
    # 首先处理缺失数据：对每个国家，使用前向填充 + 后向填充
    df_filled = df.copy()
    indicator_cols = [col for col in df.columns if col.endswith('_log') or col.endswith('_norm')]
    
    print("\n处理缺失数据（前向/后向填充）...")
    for country in COUNTRIES:
        country_mask = df_filled['Country'] == country
        for col in indicator_cols:
            # 获取该国家该列的数据
            country_data = df_filled.loc[country_mask, col].copy()
            # 检查是否有零值（可能是由于 log(0+1)=0 导致的缺失）
            if col.endswith('_log'):
                # 对于 log 列，0 值可能表示缺失，用插值填充
                zero_mask = country_data <= EPS
                if zero_mask.sum() > 0 and zero_mask.sum() < len(country_data):
                    # 用非零值的中位数填充
                    non_zero_median = country_data[~zero_mask].median()
                    if pd.notna(non_zero_median) and non_zero_median > 0:
                        country_data.loc[zero_mask] = non_zero_median
                        df_filled.loc[country_mask, col] = country_data
                        if zero_mask.sum() > 3:  # 只在有多个缺失时打印
                            print(f"  {country} {col}: 填充 {zero_mask.sum()} 个零值 -> {non_zero_median:.4f}")
            # 前向/后向填充
            df_filled.loc[country_mask, col] = df_filled.loc[country_mask, col].ffill().bfill()
    
    # 聚合每个模型变量
    for var_name, weights in agg_weights.items():
        result[var_name] = 0.0
        for indicator, weight in weights.items():
            if indicator in df_filled.columns:
                # 使用已有的 _norm 或 _log 列（这些已经是归一化后的值）
                result[var_name] += weight * df_filled[indicator].fillna(0)
            else:
                print(f"警告: 指标 {indicator} 不存在于数据中")
    
    # ==========================================================================
    # 【Fix 1 + Fix 6】能源数据量纲对齐 - 优先使用原始数据源
    # ==========================================================================
    print(f"\n能源数据源选择: {params.energy_series_source}")
    energy_raw: pd.Series
    
    if params.energy_series_source == 'raw':
        # 【Fix 6】优先从原始数据源加载
        energy_df = load_energy_data_raw()
        if len(energy_df) > 0:
            # 合并能源数据到结果
            energy_merged = df[['Country', 'Year']].merge(
                energy_df, on=['Country', 'Year'], how='left'
            )
            energy_raw = energy_merged['Total_TWh'].fillna(energy_merged['Total_TWh'].median())
            print("  【Fix 6】使用原始数据源 (各国历年电能生产情况.csv)")
        else:
            # 回退到 A2 数据
            print("  原始数据加载失败，回退到 A2 指标")
            if 'A2_Energy_IDC_log' in df.columns:
                energy_raw = np.exp(df['A2_Energy_IDC_log'].fillna(0)) - 1
            else:
                energy_raw = pd.Series([1000] * len(df))
    elif params.energy_series_source == 'Electricity_Production_TWh' and 'Electricity_Production_TWh' in df.columns:
        energy_raw = df['Electricity_Production_TWh'].fillna(df['Electricity_Production_TWh'].median())
        print("  使用 Electricity_Production_TWh 作为能源数据源")
    elif params.energy_series_source == 'A2' or 'A2_Energy_IDC' in df.columns:
        if 'A2_Energy_IDC' in df.columns:
            energy_raw = df['A2_Energy_IDC'].fillna(df['A2_Energy_IDC'].median())
            print("  使用 A2_Energy_IDC 作为能源数据源")
        elif 'A2_Energy_IDC_log' in df.columns:
            energy_raw = np.exp(df['A2_Energy_IDC_log'].fillna(0)) - 1
            print("  使用 A2_Energy_IDC_log (反对数) 作为能源数据源")
        else:
            energy_raw = pd.Series([1000] * len(df))
            print("  警告: 未找到能源数据，使用默认值 1000")
    else:
        energy_raw = pd.Series([1000] * len(df))
        print("  警告: 未找到能源数据，使用默认值 1000")
    
    # 【Fix 1 核心】归一化能源到 [0, 1]
    energy_global_max = float(energy_raw.max())
    if energy_global_max < EPS:
        energy_global_max = 1.0  # 防止除零
    
    result['Energy'] = energy_raw.values  # 保留原始值（用于调试）
    result['Energy_Norm'] = energy_raw.values / energy_global_max  # 归一化值
    
    print(f"  【Fix 1】能源归一化: global_max={energy_global_max:.2f} TWh")
    print(f"  Energy_Norm: min={result['Energy_Norm'].min():.4f}, max={result['Energy_Norm'].max():.4f}")
    
    print(f"\n聚合后的模型变量:")
    for var in ['XA', 'XB', 'XE', 'YC', 'YD', 'ZF']:
        print(f"  {var}: min={result[var].min():.4f}, max={result[var].max():.4f}, mean={result[var].mean():.4f}")
    
    return result, energy_global_max


def build_state_tensor(agg_df: pd.DataFrame, params: GLVParameters) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    构建状态张量 (Year, Country, 3) 用于 A, B, E
    
    【Fix 1】使用归一化后的能源数据 Energy_Norm
    【Fix 8】同时返回历史期的 YC、YD（来自聚合数据或生产函数重算）
    
    Returns:
        state_tensor: shape (n_years, 10, 3)
        gov_scores: shape (10,) - 2025年的治理分数
        energy_series_norm: shape (n_years, 10) - 归一化能源时间序列 [0, 1]
        historical_YC: shape (n_years, 10) - 历史期科研产出
        historical_YD: shape (n_years, 10) - 历史期开源产出
    """
    years = sorted(agg_df['Year'].unique())
    n_years = len(years)
    n_countries = len(COUNTRIES)
    
    state_tensor = np.zeros((n_years, n_countries, 3))
    energy_series_norm = np.zeros((n_years, n_countries))  # 【Fix 1】使用归一化值
    historical_YC_raw = np.zeros((n_years, n_countries))  # 【Fix 8】历史 YC
    historical_YD_raw = np.zeros((n_years, n_countries))  # 【Fix 8】历史 YD
    
    for t_idx, year in enumerate(years):
        year_data = agg_df[agg_df['Year'] == year]
        for c_idx, country in enumerate(COUNTRIES):
            country_data = year_data[year_data['Country'] == country]
            if len(country_data) > 0:
                row = country_data.iloc[0]
                state_tensor[t_idx, c_idx, DIM_A] = row['XA']
                state_tensor[t_idx, c_idx, DIM_B] = row['XB']
                state_tensor[t_idx, c_idx, DIM_E] = row['XE']
                # 【Fix 1】使用归一化能源值
                energy_series_norm[t_idx, c_idx] = row['Energy_Norm']
                # 【Fix 8】收集历史期的 YC、YD
                historical_YC_raw[t_idx, c_idx] = row['YC']
                historical_YD_raw[t_idx, c_idx] = row['YD']
            else:
                # 如果缺失数据，使用前一年的值或默认值
                if t_idx > 0:
                    state_tensor[t_idx, c_idx, :] = state_tensor[t_idx - 1, c_idx, :]
                    energy_series_norm[t_idx, c_idx] = energy_series_norm[t_idx - 1, c_idx]
                    historical_YC_raw[t_idx, c_idx] = historical_YC_raw[t_idx - 1, c_idx]
                    historical_YD_raw[t_idx, c_idx] = historical_YD_raw[t_idx - 1, c_idx]
    
    # 【Fix 8】决定是否使用生产函数重新计算历史 YC/YD
    if params.recalculate_historical_outputs:
        print("\n【Fix 8】使用生产函数重新计算历史期 YC/YD...")
        historical_YC, historical_YD = compute_historical_outputs(state_tensor, params)
    else:
        print("\n使用聚合数据中的历史 YC/YD...")
        historical_YC = historical_YC_raw
        historical_YD = historical_YD_raw
    
    # ==========================================================================
    # 【Fix 10】状态张量归一化到 [0, 1]
    # ==========================================================================
    # 这是解决"嫌疑人A：数据量级未对齐"的核心修复
    # 原始 XA/XB/XE 是 log 值（约 3-10），需归一化以匹配 GLV 模型假设
    print("\n【Fix 10】状态张量归一化...")
    state_min = np.zeros(3)
    state_max = np.zeros(3)
    for k in range(3):
        state_min[k] = state_tensor[:, :, k].min()
        state_max[k] = state_tensor[:, :, k].max()
        state_range = state_max[k] - state_min[k]
        if state_range > EPS:
            state_tensor[:, :, k] = (state_tensor[:, :, k] - state_min[k]) / state_range
        print(f"  dim={k}: 原始范围 [{state_min[k]:.4f}, {state_max[k]:.4f}] -> [0, 1]")
    
    # 保存归一化参数（用于后续反归一化或 TOPSIS 计算）
    state_norm_params = {'min': state_min, 'max': state_max}
    
    # 获取 2025 年的治理分数 ZF
    data_2025 = agg_df[agg_df['Year'] == YEAR_HISTORICAL_END]
    gov_scores = np.zeros(n_countries)
    for c_idx, country in enumerate(COUNTRIES):
        country_data = data_2025[data_2025['Country'] == country]
        if len(country_data) > 0:
            gov_scores[c_idx] = country_data.iloc[0]['ZF']
    
    # 归一化 gov_scores 到 [0, 1]
    gov_min, gov_max = gov_scores.min(), gov_scores.max()
    if gov_max > gov_min:
        gov_scores = (gov_scores - gov_min) / (gov_max - gov_min)
    
    print(f"\n状态张量形状: {state_tensor.shape}")
    print(f"  【Fix 10】归一化后 X 范围: [{state_tensor.min():.4f}, {state_tensor.max():.4f}]")
    print(f"能源序列形状 (归一化): {energy_series_norm.shape}")
    print(f"  Energy_Norm 2025 范围: [{energy_series_norm[-1].min():.4f}, {energy_series_norm[-1].max():.4f}]")
    print(f"治理分数 (归一化): {gov_scores}")
    print(f"【Fix 8】历史 YC 形状: {historical_YC.shape}, YD 形状: {historical_YD.shape}")
    
    return state_tensor, gov_scores, energy_series_norm, historical_YC, historical_YD, state_norm_params


# =============================================================================
# Step 2: 博弈矩阵 α（Interaction Injection）
# =============================================================================
def build_interaction_matrix() -> np.ndarray:
    """
    构建博弈矩阵 alpha，形状 (3, 10, 10)
    
    【Fix 3】索引安全化：
    - 废除硬编码索引常量（Idx_CN 等）
    - 改用 get_country_index() 动态获取
    - 添加 Assert 断言确保国家顺序正确
    
    不训练，直接硬编码注入，避免 10 点数据过拟合
    
    规则：
    - alpha[k][i][j] 表示国家 j 对国家 i 在维度 k 上的影响
    - 正值表示促进（共生），负值表示抑制（竞争）
    """
    alpha = np.zeros((3, 10, 10))
    
    # 【Fix 3】动态获取国家索引
    Idx_US = get_country_index('USA')
    Idx_CN = get_country_index('CHN')
    Idx_UK = get_country_index('GBR')
    Idx_IN = get_country_index('IND')
    Idx_UAE = get_country_index('ARE')
    Idx_CA = get_country_index('CAN')
    
    # 【Fix 3】断言验证索引正确性
    assert Idx_US == 0, f"USA 索引应为 0，实际为 {Idx_US}"
    assert Idx_CN == 1, f"CHN 索引应为 1，实际为 {Idx_CN}"
    assert Idx_UK == 2, f"GBR 索引应为 2，实际为 {Idx_UK}"
    assert Idx_IN == 9, f"IND 索引应为 9，实际为 {Idx_IN}"
    assert Idx_UAE == 8, f"ARE 索引应为 8，实际为 {Idx_UAE}"
    assert Idx_CA == 7, f"CAN 索引应为 7，实际为 {Idx_CA}"
    
    print(f"\n【Fix 3】博弈矩阵索引验证通过:")
    print(f"  USA={Idx_US}, CHN={Idx_CN}, GBR={Idx_UK}, IND={Idx_IN}, ARE={Idx_UAE}, CAN={Idx_CA}")
    
    # 【Fix 9】博弈系数全面降低（原值杀伤力过强）
    # 原则：|alpha| < 0.1 以确保博弈项不超过Logistic增长项
    
    # A算力层：美国遏制中国（【Fix 9】-0.3 -> -0.08）
    alpha[DIM_A, Idx_CN, Idx_US] = -0.08  # 美国对中国算力的抑制（大幅降低）
    
    # A算力层：中国自主可控反哺（【Fix 9】-0.1 -> -0.03）
    alpha[DIM_A, Idx_US, Idx_CN] = -0.03  # 中国对美国的竞争（降低）
    
    # B人才层：印度反哺美国（人才流动）（【Fix 9】降低）
    alpha[DIM_B, Idx_US, Idx_IN] = 0.08   # 印度人才流向美国
    alpha[DIM_B, Idx_UK, Idx_IN] = 0.05   # 印度人才流向英国
    
    # B人才层：中国吸引海外人才
    alpha[DIM_B, Idx_CN, Idx_US] = 0.03  # 部分人才回流
    
    # E资本层：阿联酋资本溢出（对所有国家）
    alpha[DIM_E, :, Idx_UAE] = 0.1       # UAE 资本外溢
    alpha[DIM_E, Idx_UAE, Idx_UAE] = 0   # 对角线为 0
    
    # E资本层：美国资本影响
    alpha[DIM_E, Idx_UK, Idx_US] = 0.05  # 美国资本流向英国
    alpha[DIM_E, Idx_CA, Idx_US] = 0.08  # 美国资本流向加拿大
    alpha[DIM_E, Idx_IN, Idx_US] = 0.06  # 美国资本流向印度
    
    # E资本层：中国对外投资
    alpha[DIM_E, Idx_UAE, Idx_CN] = 0.05  # 中国资本流向阿联酋
    
    # 【Fix 17】韩国半导体交互 (新增 - 地缘政治现实修正)
    Idx_KR = get_country_index('KOR')
    Idx_JP = get_country_index('JPN')
    
    # 韩国HBM芯片对美国算力的正向支撑 (三星/SK海力士)
    alpha[DIM_A, Idx_US, Idx_KR] = 0.05   
    # 韩国芯片对中国的出口 (受限但仍有渠道)
    alpha[DIM_A, Idx_CN, Idx_KR] = 0.03   
    
    # 日韩半导体竞争 (设备与材料层面)
    alpha[DIM_A, Idx_KR, Idx_JP] = -0.02
    alpha[DIM_A, Idx_JP, Idx_KR] = -0.02
    
    # 确保对角线为 0
    for k in range(3):
        np.fill_diagonal(alpha[k], 0)
    
    print(f"\n博弈矩阵构建完成，形状: {alpha.shape}")
    print(f"  A层非零元素: {np.count_nonzero(alpha[DIM_A])}")
    print(f"  B层非零元素: {np.count_nonzero(alpha[DIM_B])}")
    print(f"  E层非零元素: {np.count_nonzero(alpha[DIM_E])}")
    
    return alpha


# =============================================================================
# Step 5: 参数训练（两阶段法）- 阶段1: 拟合 r_base 与 K
# =============================================================================
def logistic_curve(t, x0, r, K):
    """
    Logistic 增长曲线
    
    x(t) = K / (1 + ((K - x0) / x0) * exp(-r * t))
    """
    # 避免除零和溢出
    x0 = np.maximum(x0, EPS)
    K = np.maximum(K, x0 + EPS)
    ratio = (K - x0) / x0
    exp_term = np.clip(-r * t, -EXP_CLIP_MAX, EXP_CLIP_MAX)
    return K / (1 + ratio * np.exp(exp_term))


def fit_logistic_parameters(state_tensor: np.ndarray, years: np.ndarray,
                            params: GLVParameters) -> Tuple[np.ndarray, np.ndarray]:
    """
    对每个国家、每个指标拟合 Logistic 增长参数
    
    Args:
        state_tensor: shape (n_years, 10, 3)
        years: 年份数组
        params: GLV 参数配置（包含 K 倍率）
    
    Returns:
        r_base: shape (10, 3) - 基础增长率
        K_fitted: shape (10, 3) - 拟合得到的环境容纳量
    """
    n_countries = state_tensor.shape[1]
    n_dims = state_tensor.shape[2]
    
    # K 倍率数组: [K_limit_A_multiplier, K_B_multiplier, K_E_multiplier]
    K_multipliers = np.array([params.K_limit_A_multiplier, params.K_B_multiplier, params.K_E_multiplier])
    
    r_base = np.zeros((n_countries, n_dims))
    K_fitted = np.zeros((n_countries, n_dims))
    
    t_data = years - years[0]  # 相对时间
    
    print("\n拟合 Logistic 参数...")
    print(f"  K 倍率: A={params.K_limit_A_multiplier}, B={params.K_B_multiplier}, E={params.K_E_multiplier}")
    print(f"  【Fix 9】r 边界约束: [{params.r_min_bound}, {params.r_max_bound}]")
    
    negative_r_count = 0  # 统计负增长率拟合次数
    
    for c_idx in range(n_countries):
        for k in range(n_dims):
            x_data = state_tensor[:, c_idx, k]
            x0 = max(x_data[0], EPS)  # 防止初始值为0
            x_final = max(x_data[-1], EPS)
            x_max = max(x_data.max(), EPS)
            
            # 初始猜测：使用配置的 K 倍率
            r_init = 0.1
            K_init = max(x_max * K_multipliers[k], x0 * 1.5)
            
            # 【Fix 12】确保 K 的下界至少比 x_max 大 10%，留出增长空间
            K_lower_bound = x_max * 1.1
            
            try:
                # 【Fix 9】curve_fit 拟合，添加严格边界约束
                popt, _ = curve_fit(
                    lambda t, r, K: logistic_curve(t, x0, r, K),
                    t_data,
                    x_data,
                    p0=[r_init, K_init],
                    # 【Fix 9 + Fix 12】r 必须 > r_min_bound，K > x_max * 1.1
                    bounds=([params.r_min_bound, K_lower_bound], [params.r_max_bound, x_max * 10]),
                    maxfev=5000
                )
                r_base[c_idx, k] = popt[0]
                K_fitted[c_idx, k] = popt[1]
                
                # 【Fix 9】额外检查：如果拟合出的 r 过小，给出警告
                if popt[0] < 0.02:
                    print(f"  【Fix 9】提示: {COUNTRIES[c_idx]} dim={k} 拟合 r={popt[0]:.4f} 较小")
                    
            except (RuntimeError, ValueError) as e:
                # 拟合失败，使用回退策略（基于历史趋势计算增长率）
                # 计算历史年均增长率作为默认值
                if x_final > x0 and x0 > EPS:
                    r_default = np.log(x_final / x0) / (len(years) - 1)
                    r_default = np.clip(r_default, params.r_min_bound, params.r_max_bound)
                else:
                    r_default = 0.05
                r_base[c_idx, k] = r_default
                K_fitted[c_idx, k] = x_max * K_multipliers[k]
                print(f"  警告: {COUNTRIES[c_idx]} dim={k} 拟合失败，使用历史增长率 r={r_default:.4f}")
    
    # 【Fix 9】检查是否存在异常低的 r 值
    if (r_base < 0.02).any():
        low_r_count = (r_base < 0.02).sum()
        print(f"  【Fix 9】注意: {low_r_count} 个拟合 r 值 < 0.02，可能导致增长缓慢")
    
    # 打印拟合结果
    print("\n拟合结果 (r_base):")
    print(f"{'国家':<8} {'A(算力)':<12} {'B(人才)':<12} {'E(资本)':<12}")
    print("-" * 50)
    for c_idx, country in enumerate(COUNTRIES):
        print(f"{COUNTRY_NAMES_CN[country]:<8} {r_base[c_idx, 0]:<12.4f} {r_base[c_idx, 1]:<12.4f} {r_base[c_idx, 2]:<12.4f}")
    
    print("\n拟合结果 (K):")
    print(f"{'国家':<8} {'A(算力)':<12} {'B(人才)':<12} {'E(资本)':<12}")
    print("-" * 50)
    for c_idx, country in enumerate(COUNTRIES):
        print(f"{COUNTRY_NAMES_CN[country]:<8} {K_fitted[c_idx, 0]:<12.4f} {K_fitted[c_idx, 1]:<12.4f} {K_fitted[c_idx, 2]:<12.4f}")
    
    return r_base, K_fitted


# =============================================================================
# Step 2 & 4: GLV-SD 动力学方程与 RK4 求解器
# =============================================================================
def compute_dynamic_K(t: float, X_current: np.ndarray, K_base: np.ndarray, 
                       energy_2025_norm: np.ndarray, params: GLVParameters) -> np.ndarray:
    """
    计算动态环境容纳量 K(t)
    
    【改进1】引入 Koomey's Law 技术能效增长：
    - 芯片能效每年提升约 15%，使同样能源支撑更多算力
    - 公式：K_limit_A = eta * efficiency_multiplier * energy_t_norm * K_base_A
    
    【Fix 1】能源数据已归一化到 [0, 1]，与 XA 量级匹配
    
    Args:
        t: 当前时间（年份，如 2026.0）
        X_current: 当前状态，shape (10, 3)
        K_base: 基础 K 值，shape (10, 3)
        energy_2025_norm: 2025年的归一化能源数据 [0, 1]
        params: 模型参数
    
    Returns:
        K_t: 当前时刻的 K 值，shape (10, 3)
    """
    K_t = K_base.copy()
    
    # 【Fix 7】算力层的能源约束（参数化年份后生效）
    if t > params.energy_constraint_start_year:
        dt = t - YEAR_HISTORICAL_END
        
        # 1. 能源供给增长（原有逻辑）
        energy_growth = (1 + params.energy_annual_growth_rate) ** dt
        
        # 2. 【改进1】技术能效增长 (Koomey's Law)
        # 随着时间推移，同样的电能可以支撑更多的算力
        # 历史数据：芯片能效每 1.57 年翻倍，约年化 18%
        # 这里使用保守估计 15%
        efficiency_multiplier = (1 + params.tech_efficiency_growth) ** dt
        
        # 3. 能源供给归一化值（不再截断在 1.0）
        energy_t_norm = energy_2025_norm * energy_growth
        
        # 4. 综合约束：K_limit = 基础能效 * 技术进步倍率 * 能源供给 * 原始K
        # 技术进步使得"能源墙"随时间后移，符合历史趋势
        K_limit_A = params.eta * efficiency_multiplier * energy_t_norm * K_base[:, DIM_A]
        
        # 取最小值：受物理极限或能源极限的短板制约
        # 略微放宽 K_base 上限（2.0 倍），允许技术突破
        K_t[:, DIM_A] = np.minimum(K_base[:, DIM_A] * 2.0, K_limit_A)
    
    return K_t


def derivative_function(t: float, X_current: np.ndarray, r_base: np.ndarray, 
                        K_base: np.ndarray, alpha: np.ndarray, gov_scores: np.ndarray,
                        energy_2025: np.ndarray, params: GLVParameters) -> np.ndarray:
    """
    GLV-SD 动力学方程的导数函数
    
    【改进2】引入博弈系数的适应性衰减：
    - 负向遏制效果随时间衰减，模拟国产替代和技术适应
    - 衰减公式：decay_factor = exp(-decay_rate * dt)
    
    dX_{i,k}/dt = r^adj_{i,k} * X_{i,k} * (1 - X_{i,k}/K_{i,k}(t)) 
                 + X_{i,k} * sum_{j≠i} alpha_{ij,k}(t) * X_{j,k} / K_{i,k}(t)
    
    其中：
    r^adj_{i,k} = r^base_{i,k} * (1 + delta * Z_{i,F})
    alpha_{ij,k}(t) = alpha_{ij,k} * decay_factor  (当 alpha < 0 时)
    
    Args:
        t: 当前时间
        X_current: 当前状态，shape (10, 3)
        r_base: 基础增长率，shape (10, 3)
        K_base: 基础 K 值，shape (10, 3)
        alpha: 博弈矩阵，shape (3, 10, 10)
        gov_scores: 治理分数，shape (10,)
        energy_2025: 2025年能源数据
        params: 模型参数
    
    Returns:
        dXdt: 导数，shape (10, 3)
    """
    n_countries, n_dims = X_current.shape
    
    # 计算动态 K
    K_t = compute_dynamic_K(t, X_current, K_base, energy_2025, params)
    
    # 治理修正增长率
    # r^adj = r^base * (1 + delta * Z_F)
    r_adj = r_base * (1 + params.gov_impact_factor * gov_scores.reshape(-1, 1))
    
    # 【改进2】计算博弈系数的时间衰减因子
    # 假设负向遏制效果（如芯片禁令）随时间衰减，模拟国产替代进程
    if t > YEAR_HISTORICAL_END:
        dt = t - YEAR_HISTORICAL_END
        decay_factor = np.exp(-params.interaction_decay_rate * dt)
    else:
        decay_factor = 1.0
    
    dXdt = np.zeros_like(X_current)
    
    for k in range(n_dims):
        X_k = X_current[:, k]  # shape (10,)
        K_k = K_t[:, k]        # shape (10,)
        r_k = r_adj[:, k]      # shape (10,)
        
        # 【改进2】创建当前时刻的博弈矩阵副本，应用衰减
        alpha_k = alpha[k].copy()  # shape (10, 10)
        # 只对负值（遏制效应）应用衰减，正值（合作效应）保持不变
        negative_mask = alpha_k < 0
        alpha_k[negative_mask] *= decay_factor
        
        # 内生 Logistic 增长项
        logistic_term = r_k * X_k * (1 - X_k / (K_k + EPS))
        
        # 国际竞合博弈项（使用衰减后的 alpha）
        # X_{i,k} * sum_{j≠i} alpha_{ij,k}(t) * X_{j,k} / K_{i,k}
        interaction_sum = alpha_k @ X_k  # shape (10,)
        interaction_term = X_k * interaction_sum / (K_k + EPS)
        
        dXdt[:, k] = logistic_term + interaction_term
    
    return dXdt


def rk4_step(t: float, X: np.ndarray, h: float, r_base: np.ndarray, 
             K_base: np.ndarray, alpha: np.ndarray, gov_scores: np.ndarray,
             energy_2025: np.ndarray, params: GLVParameters) -> np.ndarray:
    """
    四阶 Runge-Kutta 单步
    """
    k1 = derivative_function(t, X, r_base, K_base, alpha, gov_scores, energy_2025, params)
    k2 = derivative_function(t + h/2, X + h/2 * k1, r_base, K_base, alpha, gov_scores, energy_2025, params)
    k3 = derivative_function(t + h/2, X + h/2 * k2, r_base, K_base, alpha, gov_scores, energy_2025, params)
    k4 = derivative_function(t + h, X + h * k3, r_base, K_base, alpha, gov_scores, energy_2025, params)
    
    X_next = X + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    # 非负约束
    X_next = np.maximum(X_next, 0)
    
    return X_next


def run_glv_simulation(X_2025: np.ndarray, r_base: np.ndarray, K_base: np.ndarray,
                       alpha: np.ndarray, gov_scores: np.ndarray, 
                       energy_2025: np.ndarray, params: GLVParameters) -> np.ndarray:
    """
    运行 GLV 仿真，从 2026 预测到 2035
    
    Args:
        X_2025: 2025年状态，shape (10, 3)
        r_base, K_base, alpha, gov_scores: 模型参数
        energy_2025: 2025年能源数据
        params: GLV 参数配置
    
    Returns:
        predicted_states: shape (10, 10, 3) - (年份, 国家, 维度)
    """
    h = params.h
    n_years = YEAR_FORECAST_END - YEAR_FORECAST_START + 1
    steps_per_year = int(1.0 / h)
    
    predicted_states = np.zeros((n_years, 10, 3))
    
    X_current = X_2025.copy()
    t = float(YEAR_FORECAST_START)
    
    print(f"\n运行 GLV 仿真 (h={h}, 每年 {steps_per_year} 步)...")
    
    # 【Fix 9】调试：跟踪负增长警告
    negative_growth_warnings = []
    
    for year_idx in range(n_years):
        # 记录整数年结果
        predicted_states[year_idx] = X_current.copy()
        
        # 【Fix 9】检测年度变化，警告负增长
        if year_idx > 0:
            delta = predicted_states[year_idx] - predicted_states[year_idx - 1]
            for c_idx in range(10):
                for k in range(3):
                    if delta[c_idx, k] < -0.01:  # 显著负增长
                        year = YEAR_FORECAST_START + year_idx
                        negative_growth_warnings.append(
                            f"  ⚠️ {year}年 {COUNTRIES[c_idx]} dim={k}: "
                            f"Δ={delta[c_idx, k]:.4f}"
                        )
        
        # 运行一年的仿真
        for _ in range(steps_per_year):
            X_current = rk4_step(t, X_current, h, r_base, K_base, alpha, 
                                 gov_scores, energy_2025, params)
            t += h
    
    print(f"仿真完成，输出形状: {predicted_states.shape}")
    
    # 【Fix 9】打印负增长警告（限制前10条）
    if negative_growth_warnings:
        print(f"\n【Fix 9】检测到 {len(negative_growth_warnings)} 个负增长警告:")
        for warn in negative_growth_warnings[:10]:
            print(warn)
        if len(negative_growth_warnings) > 10:
            print(f"  ... 及其他 {len(negative_growth_warnings) - 10} 个警告")
    else:
        print("\n【Fix 9】✓ 未检测到显著负增长")
    
    return predicted_states


# =============================================================================
# Step 3: 产出映射函数
# =============================================================================
def compute_historical_outputs(X_states: np.ndarray, params: GLVParameters) -> Tuple[np.ndarray, np.ndarray]:
    """
    【Fix 8】计算历史期的产出变量 YC 和 YD（使用生产函数）
    【改进3】Y_D 使用 tanh 软饱和，与 compute_outputs 保持一致
    
    Args:
        X_states: 历史状态矩阵，shape (n_years, 10, 3)
        params: 模型参数
    
    Returns:
        Y_C: 历史科研产出，shape (n_years, 10)
        Y_D: 历史开源产出，shape (n_years, 10)
    """
    n_years, n_countries, _ = X_states.shape
    
    A = X_states[:, :, DIM_A]
    B = X_states[:, :, DIM_B]
    E = X_states[:, :, DIM_E]
    
    # 安全值
    A_safe = np.maximum(A, EPS)
    B_safe = np.maximum(B, EPS)
    
    # 科研产出 (Cobb-Douglas)
    Y_C = params.mu_c * np.power(B_safe, params.beta1) * np.power(A_safe, params.beta2)
    
    # 开源产出（【改进3】使用 tanh 软饱和，与 compute_outputs 一致）
    Y_D = np.zeros((n_years, n_countries))
    E_cummax = np.zeros(n_years)
    
    for t in range(n_years):
        E_cummax[t] = np.maximum(E[:t+1, :].max(), EPS)
        ratio = E[t, :] / E_cummax[t]
        
        # 【改进3】使用 Tanh 软饱和替代纯指数，防止数值爆炸
        # Tanh 在 0 附近近似线性，在无穷大处趋于 1
        # 当 ratio 很高时，增益被锁定在 +100%（即 2 倍），而不是无限倍
        multiplier = 1.0 + np.tanh(params.capital_accelerator * ratio * 2.0)
        Y_D[t, :] = params.mu_d * B_safe[t, :] * multiplier
    
    print(f"  【Fix 8】历史 YC 范围: [{Y_C.min():.4f}, {Y_C.max():.4f}]")
    print(f"  【Fix 8 + 改进3】历史 YD 范围 (tanh): [{Y_D.min():.4f}, {Y_D.max():.4f}]")
    
    return Y_C, Y_D


def compute_outputs(X_states: np.ndarray, params: GLVParameters,
                    E_cummax_2025: float = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算产出变量 C (科研) 和 D (开源)
    
    【改进3】开源产出使用 S 型软饱和：
    - 原逻辑：exp(lambda * ratio) → 当 ratio 较大时容易爆炸
    - 新逻辑：1 + tanh(lambda * ratio) → 输出范围 [0, 2]，有理论天花板
    
    [Fix 2] 因果性修复：
    - 开源产出 Y_D 的分母 E_max 必须是逐年累积最大值
    - 禁止使用未来（2035年）的全局最大值，杜绝 Look-ahead Bias
    
    Args:
        X_states: 状态矩阵，shape (n_years, 10, 3)
        params: 模型参数
        E_cummax_2025: 历史期资本累积最大值（可选，用于预测期）
    
    Returns:
        Y_C: 科研产出，shape (n_years, 10)
        Y_D: 开源产出，shape (n_years, 10)
    """
    n_years, n_countries, _ = X_states.shape
    
    A = X_states[:, :, DIM_A]
    B = X_states[:, :, DIM_B]
    E = X_states[:, :, DIM_E]
    
    # 科研产出 (Cobb-Douglas) - 保持不变
    A_safe = np.maximum(A, EPS)
    B_safe = np.maximum(B, EPS)
    Y_C = params.mu_c * np.power(B_safe, params.beta1) * np.power(A_safe, params.beta2)
    
    # ==========================================================================
    # [Fix 2 + Fix 14 + 改进3] 开源产出（因果性修复 + 历史记忆 + S型软饱和）
    # ==========================================================================
    Y_D = np.zeros((n_years, n_countries))
    
    # 【Fix 14 核心】初始化为历史最大值，而不是 0
    if E_cummax_2025 is not None and E_cummax_2025 > EPS:
        current_cummax = E_cummax_2025
        print(f"  【Fix 14】E_cummax 初始化为历史值: {current_cummax:.4f}")
    else:
        current_cummax = np.maximum(E[0, :].max(), EPS)
        print(f"  【Fix 14】E_cummax 初始化为首年值: {current_cummax:.4f}")
    
    E_cummax_arr = np.zeros(n_years)  # 用于打印日志
    for t in range(n_years):
        # 更新当前的全局最大值（包含历史记忆）
        current_max_t = E[t, :].max()
        if current_max_t > current_cummax:
            current_cummax = current_max_t
        E_cummax_arr[t] = current_cummax
        
        # 计算资本比率
        ratio = E[t, :] / current_cummax  # ratio ∈ [0, 1]
        
        # 【改进3】使用 Tanh 软饱和替代纯指数，防止数值爆炸
        # Tanh 在 0 附近近似线性，在无穷大处趋于 1
        # 当 ratio 很高时，增益被锁定在 +100%（即 2 倍），而不是无限倍
        # 公式: multiplier = 1.0 + tanh(capital_accelerator * ratio * 2.0)
        multiplier = 1.0 + np.tanh(params.capital_accelerator * ratio * 2.0)
        
        Y_D[t, :] = params.mu_d * B_safe[t, :] * multiplier
    
    print(f"  [Fix 2] E_cummax 逐年累积: {E_cummax_arr}")
    print(f"  【改进3】Y_D 使用 tanh 软饱和: 范围 [{Y_D.min():.4f}, {Y_D.max():.4f}]")
    
    return Y_C, Y_D


# =============================================================================
# Step 6: 排名（回代 Global TOPSIS）
# =============================================================================
def global_normalize(historical_agg: pd.DataFrame, historical_state: np.ndarray,
                     historical_YC: np.ndarray, historical_YD: np.ndarray,
                     predicted_states: np.ndarray,
                     Y_C: np.ndarray, Y_D: np.ndarray, gov_scores: np.ndarray,
                     params: GLVParameters,
                     state_norm_params: Dict = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    全局归一化（2016-2035 全局 min/max）
    
    【Fix 8】历史期 YC/YD 使用传入的生产函数计算结果，而非聚合值
    【Fix 10】预测状态需要反归一化回原始量级，以便与历史数据一致进行 TOPSIS 计算
    
    Args:
        historical_agg: 历史聚合数据（用于 XA, XB, XE, ZF）
        historical_state: 历史状态张量，shape (n_years_hist, 10, 3) - 已归一化到[0,1]
        historical_YC: 历史科研产出（生产函数计算），shape (n_years_hist, 10)
        historical_YD: 历史开源产出（生产函数计算），shape (n_years_hist, 10)
        predicted_states: 预测状态，shape (10, 10, 3) - 已归一化到[0,1]
        Y_C, Y_D: 预测产出
        gov_scores: 治理分数
        params: 模型参数
        state_norm_params: 状态归一化参数（min/max），用于反归一化
    
    Returns:
        historical_norm: 历史数据归一化结果
        forecast_norm: 预测数据归一化结果
        global_minmax: 全局 min/max 字典
    """
    # 提取历史数据的聚合变量
    years_hist = sorted(historical_agg['Year'].unique())
    n_years_hist = len(years_hist)
    n_years_forecast = predicted_states.shape[0]
    n_countries = len(COUNTRIES)
    
    # 【Fix 10】将预测状态反归一化回原始量级
    if state_norm_params is not None:
        predicted_states_raw = np.zeros_like(predicted_states)
        for k in range(3):
            state_min = state_norm_params['min'][k]
            state_max = state_norm_params['max'][k]
            state_range = state_max - state_min
            predicted_states_raw[:, :, k] = predicted_states[:, :, k] * state_range + state_min
        print(f"  【Fix 10】预测状态反归一化: 范围 [{predicted_states_raw.min():.4f}, {predicted_states_raw.max():.4f}]")
    else:
        predicted_states_raw = predicted_states
    
    # 构建历史矩阵 (years_hist, countries, 6)
    # 变量顺序: [XA, XB, YC, YD, XE, ZF]
    historical_matrix = np.zeros((n_years_hist, n_countries, 6))
    
    for t_idx, year in enumerate(years_hist):
        year_data = historical_agg[historical_agg['Year'] == year]
        for c_idx, country in enumerate(COUNTRIES):
            country_data = year_data[year_data['Country'] == country]
            if len(country_data) > 0:
                row = country_data.iloc[0]
                historical_matrix[t_idx, c_idx, 0] = row['XA']  # 算力
                historical_matrix[t_idx, c_idx, 1] = row['XB']  # 人才
                # 【Fix 8】使用生产函数计算的 YC/YD
                historical_matrix[t_idx, c_idx, 2] = historical_YC[t_idx, c_idx]  # 科研
                historical_matrix[t_idx, c_idx, 3] = historical_YD[t_idx, c_idx]  # 开源
                historical_matrix[t_idx, c_idx, 4] = row['XE']  # 资本
                historical_matrix[t_idx, c_idx, 5] = row['ZF']  # 治理
    
    # 构建预测矩阵 (n_years_forecast, countries, 6)
    # 【Fix 10】使用反归一化后的预测状态
    forecast_matrix = np.zeros((n_years_forecast, n_countries, 6))
    forecast_matrix[:, :, 0] = predicted_states_raw[:, :, DIM_A]  # 算力
    forecast_matrix[:, :, 1] = predicted_states_raw[:, :, DIM_B]  # 人才
    forecast_matrix[:, :, 2] = Y_C                                # 科研
    forecast_matrix[:, :, 3] = Y_D                                # 开源
    forecast_matrix[:, :, 4] = predicted_states_raw[:, :, DIM_E]  # 资本
    # 治理分数在预测期间保持 2025 年的值（或可设置微小变化）
    for t_idx in range(n_years_forecast):
        forecast_matrix[t_idx, :, 5] = gov_scores
    
    # 合并历史和预测数据计算全局 min/max
    all_data = np.concatenate([
        historical_matrix.reshape(-1, 6),
        forecast_matrix.reshape(-1, 6)
    ], axis=0)
    
    global_min = all_data.min(axis=0)
    global_max = all_data.max(axis=0)
    global_range = global_max - global_min
    global_range[global_range < EPS] = EPS  # 避免除零
    
    global_minmax = {
        'min': global_min,
        'max': global_max,
        'range': global_range,
        'variables': ['XA', 'XB', 'YC', 'YD', 'XE', 'ZF']
    }
    
    print(f"\n全局归一化参数 (2016-2035):")
    for i, var in enumerate(global_minmax['variables']):
        print(f"  {var}: min={global_min[i]:.4f}, max={global_max[i]:.4f}")
    
    # 归一化
    historical_norm = (historical_matrix - global_min) / global_range
    forecast_norm = (forecast_matrix - global_min) / global_range
    
    return historical_norm, forecast_norm, global_minmax


def compute_topsis_scores(normalized_data: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    计算 TOPSIS 得分（理想解固定为 1，负理想解固定为 0）
    
    D^+_{i,t} = sqrt(sum_j w_j * (v_{i,j,t} - 1)^2)
    D^-_{i,t} = sqrt(sum_j w_j * (v_{i,j,t} - 0)^2)
    S_{i,t} = D^-_{i,t} / (D^+_{i,t} + D^-_{i,t})
    
    Args:
        normalized_data: 归一化数据，shape (n_years, n_countries, n_dims)
        weights: 权重，shape (n_dims,)
    
    Returns:
        scores: TOPSIS 得分，shape (n_years, n_countries)
    """
    n_years, n_countries, n_dims = normalized_data.shape
    
    # 加权矩阵
    V = normalized_data * weights
    
    # 理想解 = 1，负理想解 = 0（加权后）
    A_plus = weights  # shape (n_dims,)
    A_minus = np.zeros(n_dims)
    
    # 计算距离
    D_plus = np.sqrt(np.sum((V - A_plus) ** 2, axis=2))
    D_minus = np.sqrt(np.sum((V - A_minus) ** 2, axis=2))
    
    # TOPSIS 得分
    scores = D_minus / (D_plus + D_minus + EPS)
    
    return scores


def generate_rankings(scores: np.ndarray) -> np.ndarray:
    """
    根据得分生成排名（得分越高排名越靠前）
    
    Args:
        scores: shape (n_years, n_countries)
    
    Returns:
        rankings: shape (n_years, n_countries)
    """
    n_years, n_countries = scores.shape
    rankings = np.zeros_like(scores, dtype=int)
    
    for t in range(n_years):
        # 按得分降序排名
        rankings[t] = n_countries - np.argsort(np.argsort(scores[t]))
    
    return rankings


# =============================================================================
# 输出函数
# =============================================================================
def save_results(output_dir: Path, r_base: np.ndarray, K_fitted: np.ndarray,
                 predicted_states: np.ndarray, Y_C: np.ndarray, Y_D: np.ndarray,
                 gov_scores: np.ndarray, forecast_scores: np.ndarray, 
                 forecast_rankings: np.ndarray, global_minmax: Dict,
                 alpha: np.ndarray, params: GLVParameters):
    """
    保存所有结果到 CSV 文件
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    forecast_years = list(range(YEAR_FORECAST_START, YEAR_FORECAST_END + 1))
    
    # 1. 保存拟合参数 r_base 和 K
    params_data = []
    for c_idx, country in enumerate(COUNTRIES):
        params_data.append({
            'Country': country,
            'Country_CN': COUNTRY_NAMES_CN[country],
            'r_A': r_base[c_idx, DIM_A],
            'r_B': r_base[c_idx, DIM_B],
            'r_E': r_base[c_idx, DIM_E],
            'K_A': K_fitted[c_idx, DIM_A],
            'K_B': K_fitted[c_idx, DIM_B],
            'K_E': K_fitted[c_idx, DIM_E]
        })
    pd.DataFrame(params_data).to_csv(output_dir / 'glv_params_rK.csv', index=False, encoding='utf-8-sig')
    print(f"保存: {output_dir / 'glv_params_rK.csv'}")
    
    # 2. 保存预测状态 (A, B, E)
    states_data = []
    for t_idx, year in enumerate(forecast_years):
        for c_idx, country in enumerate(COUNTRIES):
            states_data.append({
                'Year': year,
                'Country': country,
                'Country_CN': COUNTRY_NAMES_CN[country],
                'XA_Compute': predicted_states[t_idx, c_idx, DIM_A],
                'XB_Talent': predicted_states[t_idx, c_idx, DIM_B],
                'XE_Capital': predicted_states[t_idx, c_idx, DIM_E]
            })
    pd.DataFrame(states_data).to_csv(output_dir / 'glv_simulated_states_2026_2035.csv', 
                                      index=False, encoding='utf-8-sig')
    print(f"保存: {output_dir / 'glv_simulated_states_2026_2035.csv'}")
    
    # 3. 保存产出 (C, D, F)
    outputs_data = []
    for t_idx, year in enumerate(forecast_years):
        for c_idx, country in enumerate(COUNTRIES):
            outputs_data.append({
                'Year': year,
                'Country': country,
                'Country_CN': COUNTRY_NAMES_CN[country],
                'YC_Research': Y_C[t_idx, c_idx],
                'YD_OpenSource': Y_D[t_idx, c_idx],
                'ZF_Governance': gov_scores[c_idx]
            })
    pd.DataFrame(outputs_data).to_csv(output_dir / 'glv_outputs_2026_2035.csv', 
                                       index=False, encoding='utf-8-sig')
    print(f"保存: {output_dir / 'glv_outputs_2026_2035.csv'}")
    
    # 4. 保存 TOPSIS 得分与排名
    topsis_data = []
    for t_idx, year in enumerate(forecast_years):
        for c_idx, country in enumerate(COUNTRIES):
            topsis_data.append({
                'Year': year,
                'Country': country,
                'Country_CN': COUNTRY_NAMES_CN[country],
                'TOPSIS_Score': forecast_scores[t_idx, c_idx],
                'Rank': forecast_rankings[t_idx, c_idx]
            })
    pd.DataFrame(topsis_data).to_csv(output_dir / 'topsis_forecast_scores_2026_2035.csv', 
                                      index=False, encoding='utf-8-sig')
    print(f"保存: {output_dir / 'topsis_forecast_scores_2026_2035.csv'}")
    
    # 5. 保存全局归一化参数
    minmax_df = pd.DataFrame({
        'Variable': global_minmax['variables'],
        'Global_Min': global_minmax['min'],
        'Global_Max': global_minmax['max']
    })
    minmax_df.to_csv(output_dir / 'global_minmax_2016_2035.csv', index=False, encoding='utf-8-sig')
    print(f"保存: {output_dir / 'global_minmax_2016_2035.csv'}")
    
    # 6. 保存博弈矩阵 alpha
    alpha_data = []
    dim_names = ['A_Compute', 'B_Talent', 'E_Capital']
    for k in range(3):
        for i in range(10):
            for j in range(10):
                if alpha[k, i, j] != 0:
                    alpha_data.append({
                        'Dimension': dim_names[k],
                        'Target_Country': COUNTRIES[i],
                        'Source_Country': COUNTRIES[j],
                        'Alpha_Value': alpha[k, i, j]
                    })
    pd.DataFrame(alpha_data).to_csv(output_dir / 'alpha_interaction_matrix.csv', 
                                     index=False, encoding='utf-8-sig')
    print(f"保存: {output_dir / 'alpha_interaction_matrix.csv'}")
    
    print(f"\n所有结果已保存到: {output_dir}")


def plot_forecast_rankings(forecast_years: List[int], forecast_scores: np.ndarray, 
                           forecast_rankings: np.ndarray, output_dir: Path):
    """
    绘制预测排名趋势图
    """
    output_dir = Path(output_dir)
    
    # Country names in English for plotting
    COUNTRY_NAMES_EN = {
        'USA': 'USA', 'CHN': 'China', 'GBR': 'UK', 'DEU': 'Germany',
        'KOR': 'South Korea', 'JPN': 'Japan', 'FRA': 'France', 'CAN': 'Canada',
        'ARE': 'UAE', 'IND': 'India'
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. 得分趋势图
    ax1 = axes[0]
    for c_idx, country in enumerate(COUNTRIES):
        ax1.plot(forecast_years, forecast_scores[:, c_idx], 
                 marker='o', label=COUNTRY_NAMES_EN[country], linewidth=2)
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('TOPSIS Score', fontsize=12)
    ax1.set_title('AI Competitiveness Score Forecast (2026-2035)', fontsize=14)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.grid(True, alpha=0.3)
    
    # 2. 排名趋势图
    ax2 = axes[1]
    for c_idx, country in enumerate(COUNTRIES):
        ax2.plot(forecast_years, forecast_rankings[:, c_idx], 
                 marker='s', label=COUNTRY_NAMES_EN[country], linewidth=2)
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('Rank', fontsize=12)
    ax2.set_title('AI Competitiveness Ranking Forecast (2026-2035)', fontsize=14)
    ax2.invert_yaxis()  # 排名越小越好
    ax2.set_yticks(range(1, 11))
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = output_dir / 'forecast_trends.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"保存图片: {save_path}")
    plt.close()


def print_ranking_summary(forecast_years: List[int], forecast_scores: np.ndarray, 
                          forecast_rankings: np.ndarray):
    """
    打印排名摘要
    """
    print("\n" + "=" * 80)
    print("2026-2035 AI 竞争力预测排名")
    print("=" * 80)
    
    for t_idx, year in enumerate(forecast_years):
        print(f"\n{year}年排名:")
        # 按得分排序
        sorted_indices = np.argsort(forecast_scores[t_idx])[::-1]
        for rank, c_idx in enumerate(sorted_indices, 1):
            country = COUNTRIES[c_idx]
            score = forecast_scores[t_idx, c_idx]
            print(f"  {rank:>2}. {COUNTRY_NAMES_CN[country]:<6} ({country:<3}): {score:.4f}")


# =============================================================================
# 主函数
# =============================================================================
def validate_initial_state(X_2025: np.ndarray, min_value: float = MIN_STATE_VALUE) -> np.ndarray:
    """
    【Fix 4】验证并修复初始状态
    
    检查 X_2025 是否包含 0、NaN 或 Inf，如有则进行最小值植入
    
    Args:
        X_2025: 2025年初始状态，shape (10, 3)
        min_value: 最小允许值（默认 1e-3）
    
    Returns:
        X_2025_safe: 安全的初始状态
    """
    X_safe = X_2025.copy()
    
    # 检查 NaN
    nan_mask = np.isnan(X_safe)
    if nan_mask.any():
        nan_count = nan_mask.sum()
        print(f"  【Fix 4】警告: 发现 {nan_count} 个 NaN 值，替换为 {min_value}")
        X_safe[nan_mask] = min_value
    
    # 检查 Inf
    inf_mask = np.isinf(X_safe)
    if inf_mask.any():
        inf_count = inf_mask.sum()
        print(f"  【Fix 4】警告: 发现 {inf_count} 个 Inf 值，替换为 {min_value}")
        X_safe[inf_mask] = min_value
    
    # 检查零值或负值
    zero_mask = X_safe <= 0
    if zero_mask.any():
        zero_count = zero_mask.sum()
        print(f"  【Fix 4】警告: 发现 {zero_count} 个零/负值，替换为 {min_value}")
        X_safe = np.maximum(X_safe, min_value)
    
    # 最终断言
    assert not np.isnan(X_safe).any(), "初始状态仍包含 NaN"
    assert not np.isinf(X_safe).any(), "初始状态仍包含 Inf"
    assert (X_safe > 0).all(), "初始状态仍包含非正值"
    
    print(f"  【Fix 4】初始状态验证通过: min={X_safe.min():.6f}, max={X_safe.max():.6f}")
    
    return X_safe


def main():
    """
    问题三完整求解流程：GLV-SD 预测 + TOPSIS 排名
    
    【Code Review 修复清单】:
    - Fix 1: 能源数据量纲对齐（归一化到 [0,1]）
    - Fix 2: 开源产出 E_max 因果性修复（禁止 Look-ahead Bias）
    - Fix 3: 博弈矩阵索引安全化（动态获取 + Assert 断言）
    - Fix 4: 初始值安全检查（防止 0/NaN 导致 GLV 崩溃）
    - Fix 5: 聚合逻辑自洽性注释
    """
    print("\n" + "=" * 80)
    print("问题三：基于广义 Lotka-Volterra 的分层生态演化模型（GLV-SD）")
    print("=" * 80)
    
    # 初始化参数
    params = GLVParameters()
    print(f"\n模型参数配置:")
    print(f"  治理修正因子 (delta): {params.gov_impact_factor}")
    print(f"  Cobb-Douglas 弹性: beta1={params.beta1}, beta2={params.beta2}")
    print(f"  资本加速因子 (lambda): {params.capital_accelerator}")
    print(f"  RK4 步长 (h): {params.h}")
    print(f"  E3 是否并入排名: {params.include_E3_in_ranking}")
    
    # ==========================================================================
    # Step 0: 数据加载与聚合
    # ==========================================================================
    print("\n" + "-" * 60)
    print("Step 0: 数据加载与聚合")
    print("-" * 60)
    
    # 加载 DEMATEL 权重
    dematel_weights = load_dematel_weights()
    agg_weights = get_aggregation_weights(dematel_weights)
    
    print("\n聚合权重:")
    for var, weights in agg_weights.items():
        print(f"  {var}: {weights}")
    
    # 加载历史数据
    historical_df = load_historical_data()
    
    # 【鲁棒性检查】确保关键国家数据存在
    for critical_country in ['USA', 'CHN', 'GBR', 'KOR']:
        if critical_country not in historical_df['Country'].values:
            print(f"⚠️ 警告: 历史数据中缺失 {critical_country}，可能影响模型准确性")
    
    # 【Fix 1】聚合指标（返回 energy_global_max）
    historical_agg, energy_global_max = aggregate_indicators(historical_df, agg_weights, params)
    params.energy_global_max = energy_global_max  # 存储到参数中
    
    # 【Fix 8 + Fix 10】构建状态张量（同时返回历史 YC/YD 和归一化参数）
    state_tensor, gov_scores, energy_series_norm, historical_YC, historical_YD, state_norm_params = build_state_tensor(historical_agg, params)
    
    # ==========================================================================
    # Step 2.3: 构建博弈矩阵
    # ==========================================================================
    print("\n" + "-" * 60)
    print("Step 2.3: 构建博弈矩阵")
    print("-" * 60)
    
    alpha = build_interaction_matrix()
    
    # ==========================================================================
    # Step 5: 参数拟合
    # ==========================================================================
    print("\n" + "-" * 60)
    print("Step 5: 参数拟合 (Logistic curve_fit)")
    print("-" * 60)
    
    years_array = np.array(list(range(YEAR_HISTORICAL_START, YEAR_HISTORICAL_END + 1)))
    r_base, K_fitted = fit_logistic_parameters(state_tensor, years_array, params)
    
    # 【Fix 4】检查 K 值是否有效
    if (K_fitted <= 0).any():
        print("  【Fix 4】警告: K 值存在非正值，进行修复")
        K_fitted = np.maximum(K_fitted, MIN_STATE_VALUE)
    
    # 调整 K 值：对 UAE/IND/CHN/KOR 给予更高的追赶空间
    for country, multiplier in params.catch_up_multiplier.items():
        c_idx = COUNTRIES.index(country)
        K_fitted[c_idx, :] *= multiplier
        print(f"调整 {country} 的 K 值 (×{multiplier})")
    
    # 【Fix 13】调整增长率：提升追赶国家增速
    for country, multiplier in params.r_boost_multiplier.items():
        c_idx = COUNTRIES.index(country)
        r_base[c_idx, :] = np.minimum(r_base[c_idx, :] * multiplier, params.r_max_bound)
        print(f"调整 {country} 的 r 值 (×{multiplier})")
    
    # ==========================================================================
    # 【Tuning】上帝视角参数修正 (Reality Alignment)
    # ==========================================================================
    print("\n【Tuning】执行上帝视角参数修正...")
    
    # 1. 美国 (USA)：作为创新源头，给予最低增速保障
    try:
        Idx_US = COUNTRIES.index('USA')
        # 强制美国所有维度的 r 至少为 0.08 (8%基础增长)，防止因历史数据波动导致的"躺平"
        original_r_us = r_base[Idx_US, :].copy()
        r_base[Idx_US, :] = np.maximum(r_base[Idx_US, :], 0.08)
        # 美国的上限 K 额外抬高 20% (技术突破红利)
        K_fitted[Idx_US, :] *= 1.2
        print(f"  USA: r修正 {original_r_us} -> {r_base[Idx_US, :]}, K值扩大 1.2x")
    except ValueError:
        pass

    # 2. 中国 (CHN)：确保算力和资本的天花板足够高，允许后期反超
    try:
        Idx_CN = COUNTRIES.index('CHN')
        # 算力(A) 和 资本(E) 的 K 值额外扩大 1.5 倍
        K_fitted[Idx_CN, DIM_A] *= 1.5 
        K_fitted[Idx_CN, DIM_E] *= 1.5
        print(f"  CHN: 算力与资本 K 值额外扩大 1.5x (允许长期追赶)")
    except ValueError:
        pass
    
    # ==========================================================================
    # Step 4: RK4 仿真
    # ==========================================================================
    print("\n" + "-" * 60)
    print("Step 4: RK4 数值仿真 (2026-2035)")
    print("-" * 60)
    
    # 获取 2025 年状态作为初始值
    X_2025_raw = state_tensor[-1]  # shape (10, 3)
    energy_2025_norm = energy_series_norm[-1]  # shape (10,) - 归一化值
    
    # 【Fix 9】数据量级检查（嫌疑人A排查）
    print("\n【Fix 9】数据量级检查:")
    print(f"  X_2025 范围: [{X_2025_raw.min():.6f}, {X_2025_raw.max():.6f}]")
    print(f"  X_2025 均值: {X_2025_raw.mean():.6f}")
    if X_2025_raw.max() > 10:
        print("  ⚠️ 警告: 状态值过大 (>10)，可能导致博弈项爆炸！")
    elif X_2025_raw.max() < 0.01:
        print("  ⚠️ 警告: 状态值过小 (<0.01)，可能导致数值精度问题！")
    else:
        print("  ✓ 状态值在合理范围 [0.01, 10]")
    
    # 打印各国2025年状态值
    print("\n  2025年各国状态值:")
    for c_idx, country in enumerate(COUNTRIES):
        print(f"    {COUNTRY_NAMES_CN[country]}: A={X_2025_raw[c_idx, 0]:.4f}, "
              f"B={X_2025_raw[c_idx, 1]:.4f}, E={X_2025_raw[c_idx, 2]:.4f}")
    
    # 【Fix 4】验证并修复初始状态
    print("\n验证初始状态...")
    X_2025 = validate_initial_state(X_2025_raw)
    
    # 【Fix 9】检查 K 值与 X 的关系（嫌疑人B排查）
    print("\n【Fix 9】K值与初始状态检查:")
    for c_idx, country in enumerate(COUNTRIES[:5]):  # 只显示前5个
        for k, dim_name in enumerate(['A', 'B', 'E']):
            x_val = X_2025[c_idx, k]
            k_val = K_fitted[c_idx, k]
            ratio = x_val / k_val if k_val > 0 else 0
            status = "⚠️ X≈K" if ratio > 0.8 else "✓"
            print(f"  {COUNTRY_NAMES_CN[country]}-{dim_name}: X={x_val:.4f}, K={k_val:.4f}, X/K={ratio:.2f} {status}")
    
    # 运行仿真
    predicted_states = run_glv_simulation(X_2025, r_base, K_fitted, alpha, 
                                          gov_scores, energy_2025_norm, params)
    
    # ==========================================================================
    # Step 3: 计算产出
    # ==========================================================================
    print("\n" + "-" * 60)
    print("Step 3: 计算产出 (Cobb-Douglas & 指数加速)")
    print("-" * 60)
    
    # 【Fix 11】先将预测状态反归一化，再计算产出（确保 YC/YD 与历史期量级一致）
    predicted_states_raw = np.zeros_like(predicted_states)
    for k in range(3):
        state_min = state_norm_params['min'][k]
        state_max = state_norm_params['max'][k]
        state_range = state_max - state_min
        predicted_states_raw[:, :, k] = predicted_states[:, :, k] * state_range + state_min
    
    print(f"  【Fix 11】反归一化后预测状态: 范围 [{predicted_states_raw.min():.4f}, {predicted_states_raw.max():.4f}]")
    
    # ==========================================================================
    # 【Fix 14】计算历史最大资本积累（传入 compute_outputs）
    # ==========================================================================
    # 还原历史资本数据 (2016-2025) —— 注意 state_tensor 已归一化，需反归一化
    state_min_E = state_norm_params['min'][DIM_E]
    state_max_E = state_norm_params['max'][DIM_E]
    state_range_E = state_max_E - state_min_E
    
    historical_E_raw = state_tensor[:, :, DIM_E] * state_range_E + state_min_E
    E_cummax_2025 = float(historical_E_raw.max())
    print(f"【Fix 14】历史最大资本积累 (2016-2025): {E_cummax_2025:.4f}")
    
    # 调用修复后的函数，传入 E_cummax_2025（保持历史记忆）
    Y_C, Y_D = compute_outputs(predicted_states_raw, params, E_cummax_2025=E_cummax_2025)
    
    print(f"科研产出 YC: min={Y_C.min():.4f}, max={Y_C.max():.4f}")
    print(f"开源产出 YD: min={Y_D.min():.4f}, max={Y_D.max():.4f}")
    
    # ==========================================================================
    # Step 6: TOPSIS 排名
    # ==========================================================================
    print("\n" + "-" * 60)
    print("Step 6: TOPSIS 排名 (全局归一化)")
    print("-" * 60)
    
    # 【Fix 8 + Fix 11】全局归一化（直接传入反归一化后的预测状态）
    historical_norm, forecast_norm, global_minmax = global_normalize(
        historical_agg, state_tensor, historical_YC, historical_YD,
        predicted_states_raw, Y_C, Y_D, gov_scores, params,
        state_norm_params=None  # 【Fix 11】已经反归一化，无需再次处理
    )
    
    # 【Fix 5】计算权重（基于 DEMATEL 或等权）
    # 变量顺序: [XA, XB, YC, YD, XE, ZF]
    # 聚合变量权重 = 对应二级指标权重之和（数学自洽性）
    var_to_indicator = {
        'XA': ['A1_Hardware_Compute_log', 'A2_Energy_IDC_log', 'A3_Connectivity_norm'],
        'XB': ['B1_Talent_Stock_log', 'B3_STEM_Supply_norm'],
        'YC': ['C1_Research_Qty_log', 'C2_High_Impact_Res_log'],
        'YD': ['D1_GitHub_Activity_log', 'D3_OpenSource_Impact_log'],
        'XE': ['E1_Vertical_VC_log', 'E2_Capital_Flow_log'],
        'ZF': ['F1_Gov_RD_Exp_norm', 'F2_IP_Protection_log']
    }
    
    # 计算每个聚合变量的权重（对应二级指标权重之和）
    agg_var_weights = []
    for var in ['XA', 'XB', 'YC', 'YD', 'XE', 'ZF']:
        w = sum(dematel_weights.get(ind, 1/14) for ind in var_to_indicator[var])
        agg_var_weights.append(w)
    agg_var_weights = np.array(agg_var_weights)
    agg_var_weights /= agg_var_weights.sum()  # 归一化
    
    # 【Fix 19】权重温和干预 (替代原激进方案)
    # 原方案: XA*1.5, YC*1.3, YD*0.8 → 过度偏向硬科技
    # 新方案: XA*1.2, YC*1.1, YD*0.9 → 保持相对平衡
    print("\n【Tuning】执行权重温和干预...")
    # 索引对应：0:XA, 1:XB, 2:YC, 3:YD, 4:XE, 5:ZF
    agg_var_weights[0] *= 1.2  # 算力基建 XA 权重小幅提升 20%
    agg_var_weights[2] *= 1.1  # 科研产出 YC 权重小幅提升 10%
    agg_var_weights[3] *= 0.9  # 开源生态 YD 权重小幅降低 10%
    agg_var_weights /= agg_var_weights.sum()  # 重新归一化
    
    print(f"\n聚合变量权重 (干预后):")
    for i, var in enumerate(['XA', 'XB', 'YC', 'YD', 'XE', 'ZF']):
        print(f"  {var}: {agg_var_weights[i]:.4f}")
    
    # 计算 TOPSIS 得分
    forecast_scores = compute_topsis_scores(forecast_norm, agg_var_weights)
    forecast_rankings = generate_rankings(forecast_scores)
    
    # 【Fix 9】计算历史期 TOPSIS 得分，用于连续性校验
    historical_scores = compute_topsis_scores(historical_norm, agg_var_weights)
    score_2025 = historical_scores[-1]  # 2025年得分
    score_2026 = forecast_scores[0]     # 2026年得分
    
    # 连续性校验
    print("\n【Fix 9】2025→2026 TOPSIS 得分连续性校验:")
    jump_ratio = np.abs(score_2026 - score_2025) / (score_2025 + EPS)
    discontinuity_count = 0
    
    for c_idx, country in enumerate(COUNTRIES):
        jump = jump_ratio[c_idx]
        score_diff = score_2026[c_idx] - score_2025[c_idx]
        direction = "↑" if score_diff > 0 else "↓"
        status = "⚠️" if jump > 0.15 else "✓"
        if jump > 0.15:
            discontinuity_count += 1
        print(f"  {COUNTRY_NAMES_CN[country]}: 2025={score_2025[c_idx]:.4f} → "
              f"2026={score_2026[c_idx]:.4f} ({direction}{abs(score_diff):.4f}, "
              f"跳变{jump*100:.1f}%) {status}")
    
    if discontinuity_count > 0:
        print(f"\n  ⚠️ 警告: {discontinuity_count} 个国家存在 >15% 的得分跳变，建议检查模型参数！")
    else:
        print(f"\n  ✓ 所有国家的2025→2026得分跳变均 <15%，连续性良好")
    
    # ==========================================================================
    # 输出结果
    # ==========================================================================
    print("\n" + "-" * 60)
    print("保存结果")
    print("-" * 60)
    
    save_results(OUTPUT_PATH, r_base, K_fitted, predicted_states, Y_C, Y_D,
                 gov_scores, forecast_scores, forecast_rankings, global_minmax,
                 alpha, params)
    
    # 绘制趋势图
    forecast_years = list(range(YEAR_FORECAST_START, YEAR_FORECAST_END + 1))
    plot_forecast_rankings(forecast_years, forecast_scores, forecast_rankings, OUTPUT_PATH)
    
    # 打印排名摘要
    print_ranking_summary(forecast_years, forecast_scores, forecast_rankings)
    
    print("\n" + "=" * 80)
    print("问题三求解完成！")
    print("=" * 80)
    
    return {
        'params': params,
        'r_base': r_base,
        'K_fitted': K_fitted,
        'alpha': alpha,
        'predicted_states': predicted_states,
        'Y_C': Y_C,
        'Y_D': Y_D,
        'gov_scores': gov_scores,
        'forecast_scores': forecast_scores,
        'forecast_rankings': forecast_rankings,
        'global_minmax': global_minmax
    }


if __name__ == '__main__':
    results = main()