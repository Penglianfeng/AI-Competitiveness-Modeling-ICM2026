"""
问题一：指标构建与分析 - 完整复现代码 (数据源修复版)
=================================================
本代码实现了以下内容：
1. Step 0: 数据预处理 (Data Preprocessing) - 修复数据加载与指标构建
2. Step 1: DEMATEL 因果权重计算 - 优化相关矩阵融合逻辑
3. Step 2: Global Dynamic TOPSIS 评价
4. Step 3: DEA-Malmquist 效率分析
5. Step 4: 稳健性检验 (Monte Carlo权重扰动/多方法对比)
6. Step 5: 2025年贡献分解与短板诊断

指标体系：
- A 算力与数字基础设施 (A1, A2, A3)
- B 人才与教育 (B1, B3)
- C 科研产出与前沿影响 (C1, C2)
- D 开源生态与工程化能力 (D1, D3)
- E 产业转化与资本活力 (E1, E2, E3)
- F 治理与采用准备度 (F1, F2)

数据源修复 (v2.0):
- A1: 使用TOP500 Rmax份额 + 芯片贸易指数 (load_top500_data + load_chip_trade_data)
- A2: 电力生产 + 安全服务器密度 (各国历年电能生产情况.csv + World Bank)
- B1: AI人才专项数据 + 研究人员密度 (combined_data.csv + World Bank)
- B3: CS Rankings得分 + 高等教育入学率 (load_cs_rankings_data + World Bank)
- E1: GenAI_VC + AI_Compute_VC (恢复完整定义)
- E2: AI_VC总量 + 跨境投资 (load_crossborder_vc)
- E3: 行业VC总额 + 高科技出口 (load_industry_vc_data + World Bank)

作者: AI建模助手
日期: 2025年
版本: 2.0 (数据源修复版)
"""

import numpy as np
import pandas as pd
import warnings
from scipy.stats import mstats, spearmanr, rankdata
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import matplotlib
warnings.filterwarnings('ignore')


def nan_to_col_median(X: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    使用列中位数填充NaN，避免将缺失数据视为0导致的评分偏低问题。
    
    背景：2024-2025年数据往往存在延迟发布，使用0填充会导致这些年份
    的TOPSIS得分集体下跌，与AI快速发展的现实不符。
    
    Args:
        X: 决策矩阵 (n_samples, n_features)
        eps: 最小值阈值，防止除零错误
        
    Returns:
        填充后的矩阵，NaN用列中位数替代
    """
    X = np.array(X, dtype=float, copy=True)
    
    # 计算每列的中位数（忽略NaN）
    col_medians = np.nanmedian(X, axis=0)
    
    # 如果某列全是NaN，使用eps作为默认值
    col_medians = np.where(np.isnan(col_medians), eps, col_medians)
    
    # 找到所有NaN的位置
    nan_mask = np.isnan(X)
    
    # 用列中位数填充NaN
    for j in range(X.shape[1]):
        X[nan_mask[:, j], j] = col_medians[j]
    
    # 确保所有值都 >= eps（避免除零）
    return np.maximum(X, eps)


def setup_matplotlib_chinese():
    """
    配置matplotlib支持中文显示
    按优先级尝试多种中文字体，确保跨平台兼容性
    """
    import platform
    
    # 根据操作系统选择合适的中文字体
    system = platform.system()
    
    if system == 'Windows':
        # Windows系统常见中文字体
        font_list = ['Microsoft YaHei', 'SimHei', 'SimSun', 'FangSong', 'KaiTi']
    elif system == 'Darwin':  # macOS
        font_list = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Hiragino Sans GB']
    else:  # Linux
        font_list = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 
                     'Droid Sans Fallback', 'AR PL UMing CN']
    
    # 添加通用后备字体
    font_list.extend(['DejaVu Sans', 'Arial Unicode MS'])
    
    # 检查可用字体
    from matplotlib import font_manager
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
        # 如果没有找到合适的字体，尝试使用系统默认
        matplotlib.rcParams['font.sans-serif'] = font_list
        print("警告: 未找到理想中文字体，可能出现显示问题")
    
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    # 刷新字体缓存
    font_manager._load_fontmanager(try_read_cache=False)


# =============================================================================
# 配置参数
# =============================================================================
# 目标国家列表
TARGET_COUNTRIES = ['USA', 'CHN', 'GBR', 'DEU', 'KOR', 'JPN', 'FRA', 'CAN', 'ARE', 'IND']
COUNTRY_NAMES_CN = {
    'USA': 'United States', 'CHN': 'China', 'GBR': 'United Kingdom', 'DEU': 'Germany',
    'KOR': 'South Korea', 'JPN': 'Japan', 'FRA': 'France', 'CAN': 'Canada',
    'ARE': 'UAE', 'IND': 'India'
}

# 分析年份范围
YEAR_START = 2016
YEAR_END = 2025

# 数据文件路径 - 自动检测运行环境
import os
import platform
from pathlib import Path

def get_base_path():
    """自动检测项目根目录"""
    # 从当前脚本位置推断 (src/models/problem1_2/solution.py -> 项目根目录)
    script_dir = Path(__file__).resolve().parent
    # 向上查找直到找到包含 'configs' 目录的路径
    for parent in [script_dir] + list(script_dir.parents):
        if (parent / 'configs').exists():
            return str(parent)
    
    # CI/CD 环境 (GitHub Actions)
    if platform.system() != 'Windows':
        return '/home/runner/work/Who-will-win-the-global-AI-race/Who-will-win-the-global-AI-race'
    
    # Windows 本地环境
    return r'D:\Who-will-win-the-global-AI-race'

BASE_PATH = get_base_path()

# 数据路径配置
MASTER_TABLE_PATH = os.path.join(BASE_PATH, 'b题数据源', 'preprocessed', 'master_table_o_award.csv')
AI_TALENT_PATH = os.path.join(BASE_PATH, 'Supply, Mobility and Quality of AI Talents', 'preprocessed', 'ai_talent_preprocessed.csv')
RD_DATA_PATH = os.path.join(BASE_PATH, 'Research and development investment and innovation foundation', 'preprocessed', 'rd_innovation_preprocessed.csv')
WORLD_BANK_PATH = os.path.join(BASE_PATH, 'Research and development investment and innovation foundation', 'World Bank Data', 'AI_development_indicators.csv')

# 输出目录
OUTPUT_DIR = os.path.join(BASE_PATH, 'outputs', 'problem1_2')

# 国家名称标准化映射（增强版）
COUNTRY_MAPPING = {
    # 英文名称变体
    'United States': 'USA', 'US': 'USA', 'America': 'USA', 'United States of America': 'USA',
    'China': 'CHN', 'CN': 'CHN', 'PRC': 'CHN', "China (People's Republic of)": 'CHN',
    'United Kingdom': 'GBR', 'UK': 'GBR', 'Britain': 'GBR', 'Great Britain': 'GBR',
    'Germany': 'DEU', 'DE': 'DEU',
    'South Korea': 'KOR', 'Korea': 'KOR', 'Republic of Korea': 'KOR', 'Rep. of Korea': 'KOR',
    'Japan': 'JPN', 'JP': 'JPN',
    'France': 'FRA', 'FR': 'FRA',
    'Canada': 'CAN', 'CA': 'CAN',
    'United Arab Emirates': 'ARE', 'UAE': 'ARE',
    'India': 'IND', 'IN': 'IND',
    # ISO 3166-1 alpha-3 codes (已经是标准格式)
    'USA': 'USA', 'CHN': 'CHN', 'GBR': 'GBR', 'DEU': 'DEU', 
    'KOR': 'KOR', 'JPN': 'JPN', 'FRA': 'FRA', 'CAN': 'CAN', 
    'ARE': 'ARE', 'IND': 'IND',
    # 其他常见变体
    'Singapore': 'SGP', 'Switzerland': 'CHE', 'Italy': 'ITA', 'Finland': 'FIN',
}


# =============================================================================
# 新增数据加载模块 (按指标体系要求修复)
# =============================================================================

def load_top500_data():
    """
    加载TOP500超算数据，计算各国Rmax份额
    
    数据源: b题数据源/TOP500_*.xls 或 TOP500_*.xlsx (2000-2025年)
    指标: A1 硬件算力规模
    
    Returns:
        pd.DataFrame: 包含 Country, Year, Rmax_Total, Rmax_Share 列
    """
    import glob
    
    top500_dir = os.path.join(BASE_PATH, 'b题数据源')
    
    # 查找所有TOP500文件 (支持 .xls 和 .xlsx 格式)
    xls_files = glob.glob(os.path.join(top500_dir, 'TOP500_*.xls'))
    xlsx_files = glob.glob(os.path.join(top500_dir, 'TOP500_*.xlsx'))
    all_files = xls_files + xlsx_files
    
    if not all_files:
        print("警告: 未找到TOP500数据文件，A1指标将使用备用方案")
        return pd.DataFrame(columns=['Country', 'Year', 'Rmax_Total', 'Rmax_Share'])
    
    results = []
    
    for file_path in all_files:
        try:
            # 从文件名提取年份 (格式: TOP500_202511.xlsx)
            filename = os.path.basename(file_path)
            year_str = filename.replace('TOP500_', '').replace('.xlsx', '').replace('.xls', '')
            year = int(year_str[:4])  # 取前4位作为年份
            
            # 跳过分析年份范围之外的数据
            if year < YEAR_START or year > YEAR_END:
                continue
            
            # 读取Excel文件
            df = pd.read_excel(file_path)
            
            # 确保有必要的列
            if 'Country' not in df.columns or 'Rmax [TFlop/s]' not in df.columns:
                # 尝试其他可能的列名
                rmax_col = None
                for col in df.columns:
                    if 'Rmax' in col:
                        rmax_col = col
                        break
                if rmax_col is None:
                    continue
            else:
                rmax_col = 'Rmax [TFlop/s]'
            
            # 标准化国家名称
            df['Country_Std'] = df['Country'].map(COUNTRY_MAPPING).fillna(df['Country'])
            
            # 按国家汇总Rmax
            country_rmax = df.groupby('Country_Std')[rmax_col].sum().reset_index()
            country_rmax.columns = ['Country', 'Rmax_Total']
            
            # 计算全球总Rmax和各国份额
            global_rmax = country_rmax['Rmax_Total'].sum()
            country_rmax['Rmax_Share'] = country_rmax['Rmax_Total'] / global_rmax if global_rmax > 0 else 0
            country_rmax['Year'] = year
            
            # 只保留目标国家
            country_rmax = country_rmax[country_rmax['Country'].isin(TARGET_COUNTRIES)]
            results.append(country_rmax)
            
        except Exception as e:
            print(f"警告: 加载 {file_path} 失败: {e}")
            continue
    
    if not results:
        return pd.DataFrame(columns=['Country', 'Year', 'Rmax_Total', 'Rmax_Share'])
    
    result_df = pd.concat(results, ignore_index=True)
    print(f"TOP500数据加载完成: {len(result_df)} 条记录, 年份 {result_df['Year'].min()}-{result_df['Year'].max()}")
    
    return result_df


def load_chip_trade_data():
    """
    加载AI芯片和半导体进出口数据
    
    数据源: b题数据源/AI芯片和半导体及相关产品进出口数据.csv
    用于: A1 硬件算力规模的辅助指标
    
    Returns:
        pd.DataFrame: 包含 Country, Year, Chip_Import, Chip_Export, Chip_Index 列
    """
    chip_file = os.path.join(BASE_PATH, 'b题数据源', 'AI芯片和半导体及相关产品进出口数据.csv')
    
    if not os.path.exists(chip_file):
        print("警告: 未找到芯片贸易数据文件")
        return pd.DataFrame(columns=['Country', 'Year', 'Chip_Import', 'Chip_Export', 'Chip_Index'])
    
    try:
        df = pd.read_csv(chip_file)
        
        # 标准化国家代码
        if 'reporterISO' in df.columns:
            df['Country'] = df['reporterISO'].map(COUNTRY_MAPPING).fillna(df['reporterISO'])
        elif 'reporterDesc' in df.columns:
            df['Country'] = df['reporterDesc'].map(COUNTRY_MAPPING)
        
        if 'refYear' in df.columns:
            df['Year'] = df['refYear']
        
        # 分离进口和出口数据
        # 使用 primaryValue 作为贸易额 (或 cifvalue/fobvalue)
        value_col = 'primaryValue' if 'primaryValue' in df.columns else 'fobvalue'
        
        # 按国家、年份、进出口类型汇总
        if 'flowCode' in df.columns:
            imports = df[df['flowCode'] == 'M'].groupby(['Country', 'Year'])[value_col].sum().reset_index()
            imports.columns = ['Country', 'Year', 'Chip_Import']
            
            exports = df[df['flowCode'] == 'X'].groupby(['Country', 'Year'])[value_col].sum().reset_index()
            exports.columns = ['Country', 'Year', 'Chip_Export']
            
            result = imports.merge(exports, on=['Country', 'Year'], how='outer')
            result = result.fillna(0)
        else:
            # 备用方案：直接汇总
            result = df.groupby(['Country', 'Year'])[value_col].sum().reset_index()
            result.columns = ['Country', 'Year', 'Chip_Index']
            result['Chip_Import'] = result['Chip_Index'] / 2
            result['Chip_Export'] = result['Chip_Index'] / 2
        
        # 计算芯片贸易指数 (出口为主，反映产业能力)
        if 'Chip_Index' not in result.columns:
            result['Chip_Index'] = 0.3 * result['Chip_Import'] + 0.7 * result['Chip_Export']
        
        # 过滤年份和国家
        result = result[
            (result['Country'].isin(TARGET_COUNTRIES)) & 
            (result['Year'] >= YEAR_START) & 
            (result['Year'] <= YEAR_END)
        ]
        
        print(f"芯片贸易数据加载完成: {len(result)} 条记录")
        return result
        
    except Exception as e:
        print(f"警告: 加载芯片贸易数据失败: {e}")
        return pd.DataFrame(columns=['Country', 'Year', 'Chip_Import', 'Chip_Export', 'Chip_Index'])


def load_cs_rankings_data():
    """
    加载CS Rankings数据，统计各国进入排名的顶尖AI院系数量
    
    数据源: b题数据源/20XX_AI领域大学计算机排名.csv (2000-2025年共26个文件)
    指标: B3 STEM教育供给实力
    
    Returns:
        pd.DataFrame: 包含 Country, Year, Top_AI_Depts, CS_Score 列
    """
    cs_dir = os.path.join(BASE_PATH, 'b题数据源')
    
    results = []
    
    for year in range(2000, 2026):
        file_path = os.path.join(cs_dir, f'{year}_AI领域大学计算机排名.csv')
        
        if not os.path.exists(file_path):
            continue
        
        if year < YEAR_START or year > YEAR_END:
            continue
            
        try:
            df = pd.read_csv(file_path)
            
            # 标准化国家名称 (Country/Region 列)
            country_col = 'Country/Region' if 'Country/Region' in df.columns else 'Country'
            df['Country_Std'] = df[country_col].map(COUNTRY_MAPPING).fillna(df[country_col])
            
            # 获取Count列 (论文几何平均数) - 更精确的匹配逻辑
            count_col = None
            for col in df.columns:
                # 精确匹配 "Count" 开头但不是 "Country"，或包含 "Geometric mean"
                if (col.startswith('Count') and not col.startswith('Country')) or 'Geometric mean' in col:
                    count_col = col
                    break
            
            if count_col is None:
                # 尝试使用第一个数值列 (排除序号列)
                for col in df.columns:
                    if df[col].dtype in ['float64', 'int64'] and col not in ['序号', 'Rank']:
                        count_col = col
                        break
            
            if count_col is None:
                continue
            
            # 转换为数值并处理缺失值
            df['Count_Numeric'] = pd.to_numeric(df[count_col], errors='coerce').fillna(0)
            
            # 按国家统计：顶尖院系数量 & 总Count得分
            country_stats = df.groupby('Country_Std').agg(
                Top_AI_Depts=('Count_Numeric', 'count'),
                CS_Total_Count=('Count_Numeric', 'sum')
            ).reset_index()
            country_stats.columns = ['Country', 'Top_AI_Depts', 'CS_Total_Count']
            
            # 计算加权得分 (考虑院系数量和质量)
            country_stats['CS_Score'] = country_stats['Top_AI_Depts'] * 0.4 + country_stats['CS_Total_Count'] * 0.6
            country_stats['Year'] = year
            
            # 只保留目标国家
            country_stats = country_stats[country_stats['Country'].isin(TARGET_COUNTRIES)]
            results.append(country_stats)
            
        except Exception as e:
            print(f"警告: 加载 {file_path} 失败: {e}")
            continue
    
    if not results:
        return pd.DataFrame(columns=['Country', 'Year', 'Top_AI_Depts', 'CS_Score'])
    
    result_df = pd.concat(results, ignore_index=True)
    print(f"CS Rankings数据加载完成: {len(result_df)} 条记录, 年份 {result_df['Year'].min()}-{result_df['Year'].max()}")
    
    return result_df


def load_industry_vc_data():
    """
    加载20+行业VC数据并聚合
    
    数据源: b题数据源/各国历年在人工智能领域*行业的风险投资（百万美元）.csv
    指标: E3 行业渗透与技术变现
    
    Returns:
        pd.DataFrame: 包含 Country, Year, Industry_VC_Total, Industry_Count 列
    """
    import glob
    
    vc_dir = os.path.join(BASE_PATH, 'b题数据源')
    
    # 匹配所有行业VC文件
    pattern = os.path.join(vc_dir, '各国历年在人工智能领域*行业的风险投资（百万美元）.csv')
    files = glob.glob(pattern)
    
    # 补充匹配Agriculture (没有"行业"关键词)
    agri_file = os.path.join(vc_dir, '各国历年在人工智能领域Agriculture的风险投资（百万美元）.csv')
    if os.path.exists(agri_file) and agri_file not in files:
        files.append(agri_file)
    
    if not files:
        print("警告: 未找到行业VC数据文件")
        return pd.DataFrame(columns=['Country', 'Year', 'Industry_VC_Total', 'Industry_Count'])
    
    all_data = []
    industries = []
    
    for file_path in files:
        try:
            # 提取行业名称
            filename = os.path.basename(file_path)
            industry = filename.replace('各国历年在人工智能领域', '').replace('行业的风险投资（百万美元）.csv', '').replace('的风险投资（百万美元）.csv', '')
            industries.append(industry)
            
            df = pd.read_csv(file_path)
            
            # 标准化列名
            if 'Country' in df.columns:
                df['Country_Std'] = df['Country'].map(COUNTRY_MAPPING).fillna(df['Country'])
            else:
                continue
            
            # 获取投资额列
            value_col = 'Sum_of_deals' if 'Sum_of_deals' in df.columns else None
            for col in df.columns:
                if 'Sum' in col or 'sum' in col or 'Value' in col:
                    value_col = col
                    break
            
            if value_col is None:
                continue
            
            # 标记行业
            df['Industry'] = industry
            df['VC_Amount'] = pd.to_numeric(df[value_col], errors='coerce').fillna(0)
            
            # 只保留目标国家和年份
            year_col = 'Year' if 'Year' in df.columns else 'year'
            df = df.rename(columns={year_col: 'Year'})
            
            df = df[
                (df['Country_Std'].isin(TARGET_COUNTRIES)) & 
                (df['Year'] >= YEAR_START) & 
                (df['Year'] <= YEAR_END)
            ]
            
            all_data.append(df[['Country_Std', 'Year', 'Industry', 'VC_Amount']])
            
        except Exception as e:
            print(f"警告: 加载 {file_path} 失败: {e}")
            continue
    
    if not all_data:
        return pd.DataFrame(columns=['Country', 'Year', 'Industry_VC_Total', 'Industry_Count'])
    
    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.rename(columns={'Country_Std': 'Country'})
    
    # 按国家和年份汇总
    result = combined.groupby(['Country', 'Year']).agg({
        'VC_Amount': 'sum',
        'Industry': 'nunique'
    }).reset_index()
    result.columns = ['Country', 'Year', 'Industry_VC_Total', 'Industry_Count']
    
    print(f"行业VC数据加载完成: {len(result)} 条记录, 覆盖 {len(industries)} 个行业")
    
    return result


def load_ai_talent_combined_data():
    """
    加载AI专门人才数据
    
    数据源: Supply, Mobility and Quality of AI Talents/ai_talent_data_v2/processed/combined_data.csv
    指标: B1 AI人才存量
    
    Returns:
        pd.DataFrame: 包含 Country, Year, AI_Researchers_Per_Million 列
    """
    talent_file = os.path.join(
        BASE_PATH, 
        'Supply, Mobility and Quality of AI Talents', 
        'ai_talent_data_v2', 
        'processed', 
        'combined_data.csv'
    )
    
    if not os.path.exists(talent_file):
        print(f"警告: 未找到AI人才数据文件: {talent_file}")
        return pd.DataFrame(columns=['Country', 'Year', 'AI_Researchers_Per_Million'])
    
    try:
        df = pd.read_csv(talent_file)
        
        # 标准化国家代码
        if 'country_code' in df.columns:
            df['Country'] = df['country_code'].str.upper()
        
        if 'year' in df.columns:
            df['Year'] = df['year']
        
        # 获取研究人员数据 (每百万人研究人员数)
        if 'value' in df.columns and 'indicator_code' in df.columns:
            researcher_df = df[df['indicator_code'] == 'SP.POP.SCIE.RD.P6'].copy()
            researcher_df = researcher_df.rename(columns={'value': 'AI_Researchers_Per_Million'})
            researcher_df = researcher_df[['Country', 'Year', 'AI_Researchers_Per_Million']]
        else:
            researcher_df = df[['Country', 'Year']].copy()
            researcher_df['AI_Researchers_Per_Million'] = 0
        
        # 过滤目标国家和年份
        researcher_df = researcher_df[
            (researcher_df['Country'].isin(TARGET_COUNTRIES)) & 
            (researcher_df['Year'] >= YEAR_START) & 
            (researcher_df['Year'] <= YEAR_END)
        ]
        
        # 处理缺失值 - 使用前后年份插值
        researcher_df['AI_Researchers_Per_Million'] = pd.to_numeric(
            researcher_df['AI_Researchers_Per_Million'], errors='coerce'
        )
        
        print(f"AI人才数据加载完成: {len(researcher_df)} 条记录")
        return researcher_df
        
    except Exception as e:
        print(f"警告: 加载AI人才数据失败: {e}")
        return pd.DataFrame(columns=['Country', 'Year', 'AI_Researchers_Per_Million'])


def load_crossborder_vc():
    """
    加载跨境AI投资数据
    
    数据源:
    - 各国历年在人工智能领域对外的风险投资（国家间）（百万美元）.csv
    - 各国历年在AI计算领域对外的风险投资（国家间）（百万美元）.csv
    - 各国历年在生成式人工智能领域对外的风险投资（国家间）（百万美元）.csv
    
    指标: E2 资本流动与总量
    
    Returns:
        pd.DataFrame: 包含 Country, Year, Crossborder_VC_Total, Crossborder_AI_Compute, Crossborder_GenAI 列
    """
    vc_dir = os.path.join(BASE_PATH, 'b题数据源')
    
    files = {
        'total': '各国历年在人工智能领域对外的风险投资（国家间）（百万美元）.csv',
        'compute': '各国历年在AI计算领域对外的风险投资（国家间）（百万美元）.csv',
        'genai': '各国历年在生成式人工智能领域对外的风险投资（国家间）（百万美元）.csv'
    }
    
    all_results = {}
    
    for key, filename in files.items():
        file_path = os.path.join(vc_dir, filename)
        
        if not os.path.exists(file_path):
            continue
        
        try:
            df = pd.read_csv(file_path)
            
            # 这是投资来源国数据 (Country source -> Country destination)
            # 按投资来源国汇总对外投资总额
            source_col = 'Country source' if 'Country source' in df.columns else 'Country_source'
            value_col = 'Sum_of_deals' if 'Sum_of_deals' in df.columns else None
            year_col = 'year' if 'year' in df.columns else 'Year'
            
            for col in df.columns:
                if 'Sum' in col or 'sum' in col:
                    value_col = col
                    break
            
            if value_col is None or source_col not in df.columns:
                continue
            
            # 标准化国家代码
            df['Country'] = df[source_col].map(COUNTRY_MAPPING).fillna(df[source_col])
            df['Year'] = df[year_col]
            df['VC_Value'] = pd.to_numeric(df[value_col], errors='coerce').fillna(0)
            
            # 按国家和年份汇总
            result = df.groupby(['Country', 'Year'])['VC_Value'].sum().reset_index()
            result = result[
                (result['Country'].isin(TARGET_COUNTRIES)) & 
                (result['Year'] >= YEAR_START) & 
                (result['Year'] <= YEAR_END)
            ]
            
            all_results[key] = result
            
        except Exception as e:
            print(f"警告: 加载 {filename} 失败: {e}")
            continue
    
    # 合并所有跨境VC数据
    if 'total' in all_results:
        final = all_results['total'].rename(columns={'VC_Value': 'Crossborder_VC_Total'})
    else:
        final = pd.DataFrame(columns=['Country', 'Year', 'Crossborder_VC_Total'])
    
    if 'compute' in all_results:
        compute_df = all_results['compute'].rename(columns={'VC_Value': 'Crossborder_AI_Compute'})
        if len(final) > 0:
            final = final.merge(compute_df, on=['Country', 'Year'], how='outer')
        else:
            final = compute_df
    
    if 'genai' in all_results:
        genai_df = all_results['genai'].rename(columns={'VC_Value': 'Crossborder_GenAI'})
        if len(final) > 0:
            final = final.merge(genai_df, on=['Country', 'Year'], how='outer')
        else:
            final = genai_df
    
    final = final.fillna(0)
    
    print(f"跨境VC数据加载完成: {len(final)} 条记录")
    return final


def validate_data_sources(df):
    """
    验证所有数据源是否正确加载
    """
    print("\n" + "=" * 60)
    print("数据源验证报告")
    print("=" * 60)
    
    checks = {
        'A1': ('Rmax_Share', 'TOP500数据'),
        'A2': ('A2_Energy_IDC', '电力+服务器数据'),
        'B1': ('AI_Researchers_Per_Million', 'AI人才数据'),
        'B3': ('Top_AI_Depts', 'CS Rankings数据'),
        'E1': ('E1_Vertical_VC', 'AI垂直领域VC'),
        'E2': ('Crossborder_VC_Total', '跨境投资数据'),
        'E3': ('Industry_VC_Total', '行业VC数据'),
    }
    
    results = {}
    for indicator, (col, source) in checks.items():
        if col in df.columns:
            coverage = (df[col] > 0).mean() * 100
            non_null = df[col].notna().mean() * 100
            # 使用ASCII友好的状态标记
            if coverage > 30:
                status = "[OK]"
            elif coverage > 10:
                status = "[WARN]"
            else:
                status = "[FAIL]"
        else:
            coverage = 0
            non_null = 0
            status = "[FAIL]"
        
        results[indicator] = {'coverage': coverage, 'non_null': non_null, 'status': status}
        print(f"{status} {indicator} ({source}): {coverage:.1f}% 有效数据, {non_null:.1f}% 非空")
    
    return results


# =============================================================================
# Step 0: 数据预处理 (Data Preprocessing)
# =============================================================================

def load_and_merge_data():
    """
    加载并合并所有数据源
    Returns: 合并后的DataFrame，包含所有指标
    """
    print("=" * 60)
    print("Step 0: 数据加载与合并")
    print("=" * 60)
    
    # 1. 加载主表 (b题数据源)
    print("\n1. 加载主表数据...")
    master_df = pd.read_csv(MASTER_TABLE_PATH)
    master_df['Country'] = master_df['Country'].str.upper()
    
    # 2. 加载AI人才数据
    print("2. 加载AI人才数据...")
    talent_df = pd.read_csv(AI_TALENT_PATH)
    talent_df['country_code'] = talent_df['country_code'].str.upper()
    talent_df = talent_df.rename(columns={'country_code': 'Country', 'year': 'Year'})
    
    # 3. 加载研发创新数据
    print("3. 加载研发创新数据...")
    rd_df = pd.read_csv(RD_DATA_PATH)
    rd_df['country_code'] = rd_df['country_code'].str.upper()
    rd_df = rd_df.rename(columns={'country_code': 'Country', 'year': 'Year'})
    
    # 4. 加载世界银行数据
    print("4. 加载世界银行数据...")
    wb_df = pd.read_csv(WORLD_BANK_PATH, encoding='utf-8')
    # 国家名称映射
    country_mapping = {
        'United States': 'USA', 'China': 'CHN', 'United Kingdom': 'GBR',
        'Germany': 'DEU', 'Korea, Rep.': 'KOR', 'Japan': 'JPN',
        'France': 'FRA', 'Canada': 'CAN', 'United Arab Emirates': 'ARE',
        'India': 'IND', 'Korea': 'KOR', 'South Korea': 'KOR'
    }
    wb_df['Country'] = wb_df['国家'].map(country_mapping)
    wb_df = wb_df.rename(columns={'年份': 'Year'})
    
    # 过滤目标国家和年份
    master_df = master_df[
        (master_df['Country'].isin(TARGET_COUNTRIES)) & 
        (master_df['Year'] >= YEAR_START) & 
        (master_df['Year'] <= YEAR_END)
    ]
    
    talent_df = talent_df[
        (talent_df['Country'].isin(TARGET_COUNTRIES)) & 
        (talent_df['Year'] >= YEAR_START) & 
        (talent_df['Year'] <= YEAR_END)
    ]
    
    rd_df = rd_df[
        (rd_df['Country'].isin(TARGET_COUNTRIES)) & 
        (rd_df['Year'] >= YEAR_START) & 
        (rd_df['Year'] <= YEAR_END)
    ]
    
    wb_df = wb_df[
        (wb_df['Country'].isin(TARGET_COUNTRIES)) & 
        (wb_df['Year'] >= YEAR_START) & 
        (wb_df['Year'] <= YEAR_END)
    ]
    
    # 合并数据
    print("\n5. 合并基础数据...")
    merged_df = master_df.copy()
    
    # 合并人才数据
    talent_cols = ['Country', 'Year', 'researchers_per_million', 'tertiary_gross_enrollment_pct', 
                   'stem_graduates_pct', 'AI_Talent_Index']
    available_talent_cols = ['Country', 'Year'] + [c for c in talent_cols[2:] if c in talent_df.columns]
    talent_subset = talent_df[available_talent_cols].drop_duplicates(['Country', 'Year'])
    merged_df = merged_df.merge(talent_subset, on=['Country', 'Year'], how='left', suffixes=('', '_talent'))
    
    # 合并研发数据
    rd_cols = ['Country', 'Year', 'rd_expenditure_pct_gdp', 'patent_applications_resident', 
               'high_tech_exports_usd', 'fixed_broadband_per_100', 'secure_internet_servers_per_million',
               'Innovation_Foundation_Index']
    available_rd_cols = ['Country', 'Year'] + [c for c in rd_cols[2:] if c in rd_df.columns]
    rd_subset = rd_df[available_rd_cols].drop_duplicates(['Country', 'Year'])
    merged_df = merged_df.merge(rd_subset, on=['Country', 'Year'], how='left', suffixes=('', '_rd'))
    
    # =========================================================================
    # 6. 加载新增数据源 (按指标体系修复要求)
    # =========================================================================
    print("\n6. 加载新增数据源...")
    
    # 6.1 TOP500超算数据 (A1)
    print("   6.1 加载TOP500数据 (A1 硬件算力)...")
    top500_df = load_top500_data()
    if len(top500_df) > 0:
        merged_df = merged_df.merge(top500_df[['Country', 'Year', 'Rmax_Total', 'Rmax_Share']], 
                                     on=['Country', 'Year'], how='left')
    else:
        merged_df['Rmax_Total'] = 0
        merged_df['Rmax_Share'] = 0
    
    # 6.2 芯片贸易数据 (A1辅助)
    print("   6.2 加载芯片贸易数据 (A1 辅助指标)...")
    chip_df = load_chip_trade_data()
    if len(chip_df) > 0:
        merged_df = merged_df.merge(chip_df[['Country', 'Year', 'Chip_Index']], 
                                     on=['Country', 'Year'], how='left')
    else:
        merged_df['Chip_Index'] = 0
    
    # 6.3 CS Rankings数据 (B3)
    print("   6.3 加载CS Rankings数据 (B3 STEM教育)...")
    cs_df = load_cs_rankings_data()
    if len(cs_df) > 0:
        merged_df = merged_df.merge(cs_df[['Country', 'Year', 'Top_AI_Depts', 'CS_Score']], 
                                     on=['Country', 'Year'], how='left')
    else:
        merged_df['Top_AI_Depts'] = 0
        merged_df['CS_Score'] = 0
    
    # 6.4 行业VC数据 (E3)
    print("   6.4 加载行业VC数据 (E3 行业渗透)...")
    industry_vc_df = load_industry_vc_data()
    if len(industry_vc_df) > 0:
        merged_df = merged_df.merge(industry_vc_df[['Country', 'Year', 'Industry_VC_Total', 'Industry_Count']], 
                                     on=['Country', 'Year'], how='left')
    else:
        merged_df['Industry_VC_Total'] = 0
        merged_df['Industry_Count'] = 0
    
    # 6.5 AI人才专项数据 (B1)
    print("   6.5 加载AI人才专项数据 (B1 人才存量)...")
    ai_talent_df = load_ai_talent_combined_data()
    if len(ai_talent_df) > 0:
        merged_df = merged_df.merge(ai_talent_df[['Country', 'Year', 'AI_Researchers_Per_Million']], 
                                     on=['Country', 'Year'], how='left')
    else:
        merged_df['AI_Researchers_Per_Million'] = np.nan
    
    # 6.6 跨境VC数据 (E2)
    print("   6.6 加载跨境VC数据 (E2 资本流动)...")
    crossborder_df = load_crossborder_vc()
    if len(crossborder_df) > 0:
        crossborder_cols = ['Country', 'Year'] + [c for c in crossborder_df.columns if 'Crossborder' in c]
        merged_df = merged_df.merge(crossborder_df[crossborder_cols], 
                                     on=['Country', 'Year'], how='left')
    else:
        merged_df['Crossborder_VC_Total'] = 0
    
    # 填充缺失值
    fill_cols = ['Rmax_Share', 'Chip_Index', 'Top_AI_Depts', 'CS_Score', 
                 'Industry_VC_Total', 'Industry_Count', 'Crossborder_VC_Total']
    for col in fill_cols:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].fillna(0)
    
    print(f"\n数据合并完成！共 {len(merged_df)} 条记录")
    print(f"国家数: {merged_df['Country'].nunique()}")
    print(f"年份范围: {merged_df['Year'].min()} - {merged_df['Year'].max()}")
    
    return merged_df


def winsorize_data(data, columns, limits=(0.01, 0.01)):
    """
    异常值处理 (Winsorization)
    对极端值进行缩尾处理
    """
    df = data.copy()
    for col in columns:
        if col in df.columns:
            valid_data = df[col].dropna()
            if len(valid_data) > 0:
                df[col] = mstats.winsorize(df[col].fillna(df[col].median()), limits=limits)
    return df


def log_transform(data, columns):
    """
    对数变换: X' = ln(X + 1)
    """
    df = data.copy()
    for col in columns:
        if col in df.columns:
            df[f'{col}_log'] = np.log1p(df[col].fillna(0).clip(lower=0))
    return df


def minmax_normalize_series(series):
    """
    对单个Series进行Min-Max归一化到[0, 1]范围
    用于减少重复代码
    """
    s_min = series.min()
    s_max = series.max()
    return (series - s_min) / (s_max - s_min + 1e-10)


def min_max_normalize(data, columns, feature_range=(0.01, 1)):
    """
    Min-Max 归一化到指定范围
    """
    df = data.copy()
    min_val, max_val = feature_range
    for col in columns:
        if col in df.columns:
            col_min = df[col].min()
            col_max = df[col].max()
            if col_max > col_min:
                df[f'{col}_norm'] = (df[col] - col_min) / (col_max - col_min) * (max_val - min_val) + min_val
            else:
                df[f'{col}_norm'] = min_val
    return df


def preprocess_indicators(data):
    """
    完整的数据预处理流程 (修复版 - 严格对齐指标体系)
    
    修复内容:
    - A1: 使用TOP500 Rmax份额 + 芯片贸易指数 (非电力+服务器)
    - A2: 电力 + 安全服务器组合
    - B1: AI人才专项数据 + 研究人员密度
    - B3: CS Rankings得分 + 高等教育入学率
    - E1: GenAI_VC + AI_Compute_VC (恢复完整定义)
    - E2: AI_VC总量 + 跨境投资
    - E3: 行业VC总额 + 高科技出口
    
    注意: min-max归一化下限设为0.01以避免0值导致数值不稳定
    """
    print("\n" + "=" * 60)
    print("数据预处理 (修复版 - 严格对齐指标体系)")
    print("=" * 60)
    
    df = data.copy()
    
    # =========================================================================
    # 1. A1: 硬件算力规模 (TOP500 Rmax份额 + 芯片贸易)
    # 数据源: TOP500_*.xls/xlsx, AI芯片和半导体及相关产品进出口数据.csv
    # =========================================================================
    print("\n1. 构建A1 (硬件算力规模)...")
    
    # 使用TOP500 Rmax份额作为主要指标
    rmax_share = df['Rmax_Share'].fillna(0) if 'Rmax_Share' in df.columns else pd.Series(0, index=df.index)
    rmax_share_norm = minmax_normalize_series(rmax_share)
    
    # 芯片贸易指数作为辅助
    chip_index = df['Chip_Index'].fillna(0) if 'Chip_Index' in df.columns else pd.Series(0, index=df.index)
    chip_index_norm = minmax_normalize_series(chip_index)
    
    # 如果TOP500数据可用，加权组合；否则使用备用方案
    if rmax_share.sum() > 0:
        print("   使用TOP500 Rmax份额 (70%) + 芯片贸易指数 (30%)")
        df['A1_Hardware_Compute'] = 0.7 * rmax_share_norm * 100 + 0.3 * chip_index_norm * 100
    else:
        # 备用方案: 使用电力+服务器作为代理
        print("   警告: TOP500数据不可用，使用电力+服务器作为备用方案")
        electricity = df['Electricity_Production_TWh'].fillna(0)
        servers = df['secure_internet_servers_per_million'].fillna(0) if 'secure_internet_servers_per_million' in df.columns else 0
        elec_norm = minmax_normalize_series(electricity)
        srv_norm = minmax_normalize_series(pd.Series(servers) if isinstance(servers, (int, float)) else servers)
        df['A1_Hardware_Compute'] = 0.6 * elec_norm * 100 + 0.4 * srv_norm * 100
    
    df['A1_Hardware_Compute'] = df['A1_Hardware_Compute'].clip(lower=1)
    
    # =========================================================================
    # 2. A2: 能源与数据中心支撑 (电力 + 安全服务器)
    # 数据源: 各国历年电能生产情况.csv, World Bank [安全服务器]
    # =========================================================================
    print("2. 构建A2 (能源与数据中心)...")
    
    electricity = df['Electricity_Production_TWh'].fillna(0)
    elec_norm = minmax_normalize_series(electricity)
    
    if 'secure_internet_servers_per_million' in df.columns:
        servers = df['secure_internet_servers_per_million'].fillna(0)
        srv_norm = minmax_normalize_series(servers)
        df['A2_Energy_IDC'] = 0.6 * elec_norm * 100 + 0.4 * srv_norm * 100
    else:
        df['A2_Energy_IDC'] = electricity
    
    # =========================================================================
    # 3. A3: 数字连接基础 (宽带连接率)
    # 数据源: World Bank [固定宽带订阅数]
    # =========================================================================
    print("3. 构建A3 (数字连接基础)...")
    if 'fixed_broadband_per_100' in df.columns:
        df['A3_Connectivity'] = df['fixed_broadband_per_100'].fillna(20)
    else:
        df['A3_Connectivity'] = 30
    
    # =========================================================================
    # 4. B1: AI人才存量 (AI人才专项 + 研究人员密度)
    # 数据源: combined_data.csv, World Bank [研究人员数]
    # =========================================================================
    print("4. 构建B1 (AI人才存量)...")
    
    # 优先使用AI人才专项数据
    if 'AI_Researchers_Per_Million' in df.columns and df['AI_Researchers_Per_Million'].notna().sum() > 0:
        ai_talent = df['AI_Researchers_Per_Million'].fillna(0)
        ai_talent_norm = minmax_normalize_series(ai_talent)
    else:
        ai_talent_norm = pd.Series(0, index=df.index)
    
    # 通用研究人员密度
    if 'researchers_per_million' in df.columns:
        researchers = df['researchers_per_million'].fillna(0)
        researchers_norm = minmax_normalize_series(researchers)
    else:
        researchers_norm = pd.Series(0.5, index=df.index)
    
    # 加权组合
    if ai_talent_norm.sum() > 0:
        df['B1_Talent_Stock'] = 0.6 * ai_talent_norm * 100 + 0.4 * researchers_norm * 100
    else:
        df['B1_Talent_Stock'] = researchers_norm * 100
    
    df['B1_Talent_Stock'] = df['B1_Talent_Stock'].clip(lower=1)
    
    # =========================================================================
    # 5. B3: STEM教育供给 (CS Rankings + 高等教育入学率)
    # 数据源: 20XX_AI领域大学计算机排名.csv, World Bank [高等教育入学率]
    # =========================================================================
    print("5. 构建B3 (STEM教育供给)...")
    
    # CS Rankings得分
    if 'CS_Score' in df.columns and df['CS_Score'].sum() > 0:
        cs_score = df['CS_Score'].fillna(0)
        cs_norm = minmax_normalize_series(cs_score)
    elif 'Top_AI_Depts' in df.columns and df['Top_AI_Depts'].sum() > 0:
        cs_score = df['Top_AI_Depts'].fillna(0)
        cs_norm = minmax_normalize_series(cs_score)
    else:
        cs_norm = pd.Series(0, index=df.index)
    
    # 高等教育入学率
    if 'tertiary_gross_enrollment_pct' in df.columns:
        tertiary = df['tertiary_gross_enrollment_pct'].fillna(50)
        tertiary_norm = minmax_normalize_series(tertiary)
    else:
        tertiary_norm = pd.Series(0.5, index=df.index)
    
    # 加权组合
    if cs_norm.sum() > 0:
        print("   使用CS Rankings (70%) + 高等教育入学率 (30%)")
        df['B3_STEM_Supply'] = 0.7 * cs_norm * 100 + 0.3 * tertiary_norm * 100
    else:
        print("   警告: CS Rankings数据不可用，仅使用高等教育入学率")
        df['B3_STEM_Supply'] = tertiary_norm * 100
    
    # =========================================================================
    # 6. C1: AI学术产出总量
    # 数据源: 各国历年人工智能出版物数量.csv
    # =========================================================================
    print("6. 构建C1 (AI学术产出)...")
    df['C1_Research_Qty'] = df['AI_Publications'].fillna(0) if 'AI_Publications' in df.columns else 0
    
    # =========================================================================
    # 7. C2: 高影响力科研产出
    # 数据源: 各国历年人工智能高影响力出版物数量.csv
    # =========================================================================
    print("7. 构建C2 (高影响力科研)...")
    df['C2_High_Impact_Res'] = df['AI_High_Impact_Publications'].fillna(0) if 'AI_High_Impact_Publications' in df.columns else 0
    
    # =========================================================================
    # 8. D1: GitHub活跃度
    # 数据源: 各国历年在GitHub上的项目数.csv
    # =========================================================================
    print("8. 构建D1 (GitHub活跃度)...")
    df['D1_GitHub_Activity'] = df['GitHub_AI_Projects'].fillna(0) if 'GitHub_AI_Projects' in df.columns else 0
    
    # =========================================================================
    # 9. D3: 开源项目影响力
    # 数据源: 各国历年在GitHub上的高影响力项目数.csv
    # =========================================================================
    print("9. 构建D3 (开源影响力)...")
    df['D3_OpenSource_Impact'] = df['GitHub_High_Impact_Projects'].fillna(0) if 'GitHub_High_Impact_Projects' in df.columns else 0
    
    # =========================================================================
    # 10. E1: AI垂直领域融资 (GenAI_VC + AI_Compute_VC)
    # 数据源: 各国历年对生成式人工智能初创企业的风险投资.csv
    #         各国历年对AI计算初创企业的风险投资.csv
    # 修复: 恢复完整定义，共线性通过VIF检验处理
    # =========================================================================
    print("10. 构建E1 (AI垂直领域融资)...")
    
    genai_vc = df['GenAI_VC_Investment_Constant2020'].fillna(0) if 'GenAI_VC_Investment_Constant2020' in df.columns else 0
    compute_vc = df['AI_Compute_VC_Investment_Constant2020'].fillna(0) if 'AI_Compute_VC_Investment_Constant2020' in df.columns else 0
    
    df['E1_Vertical_VC'] = genai_vc + compute_vc
    print("   E1 = GenAI_VC + AI_Compute_VC (恢复完整定义)")
    
    # =========================================================================
    # 11. E2: 资本流动与总量 (AI_VC总量 + 跨境投资)
    # 数据源: 各国历年在人工智能领域所有行业的风险投资.csv
    #         各国历年在人工智能领域对外的风险投资（国家间）.csv
    # =========================================================================
    print("11. 构建E2 (资本流动总量)...")
    
    total_ai_vc = df['AI_VC_Investment_Constant2020'].fillna(0) if 'AI_VC_Investment_Constant2020' in df.columns else 0
    crossborder_vc = df['Crossborder_VC_Total'].fillna(0) if 'Crossborder_VC_Total' in df.columns else 0
    
    df['E2_Capital_Flow'] = total_ai_vc + crossborder_vc
    print("   E2 = Total_AI_VC + Crossborder_VC")
    
    # =========================================================================
    # 12. E3: 行业渗透与技术变现 (行业VC总额 + 高科技出口)
    # 数据源: 各国历年在人工智能领域*行业的风险投资.csv (20+文件)
    #         World Bank [高科技出口占比]
    # =========================================================================
    print("12. 构建E3 (行业渗透)...")
    
    # 行业VC总额
    if 'Industry_VC_Total' in df.columns and df['Industry_VC_Total'].sum() > 0:
        industry_vc = df['Industry_VC_Total'].fillna(0)
        industry_vc_norm = minmax_normalize_series(industry_vc)
    else:
        industry_vc_norm = pd.Series(0, index=df.index)
    
    # 高科技出口
    if 'high_tech_exports_usd' in df.columns:
        high_tech = df['high_tech_exports_usd'].fillna(0)
        high_tech_norm = minmax_normalize_series(high_tech)
    else:
        high_tech_norm = pd.Series(0, index=df.index)
    
    # 加权组合
    if industry_vc_norm.sum() > 0:
        print("   使用行业VC总额 (60%) + 高科技出口 (40%)")
        df['E3_Ind_Adoption'] = 0.6 * industry_vc_norm * 100 + 0.4 * high_tech_norm * 100
    elif high_tech_norm.sum() > 0:
        df['E3_Ind_Adoption'] = high_tech_norm * 100
    else:
        # 备用方案
        df['E3_Ind_Adoption'] = df['AI_VC_Investment_Constant2020'].fillna(0) * 0.1 if 'AI_VC_Investment_Constant2020' in df.columns else 0
    
    # =========================================================================
    # 13. F1: 研发投入意愿
    # 数据源: World Bank [R&D支出占GDP比例]
    # =========================================================================
    print("13. 构建F1 (研发投入)...")
    if 'rd_expenditure_pct_gdp' in df.columns:
        df['F1_Gov_RD_Exp'] = df['rd_expenditure_pct_gdp'].fillna(1.5)
    else:
        df['F1_Gov_RD_Exp'] = 1.5
    
    # =========================================================================
    # 14. F2: 知识产权保护
    # 数据源: World Bank [专利申请数]
    # =========================================================================
    print("14. 构建F2 (知识产权)...")
    if 'patent_applications_resident' in df.columns:
        df['F2_IP_Protection'] = df['patent_applications_resident'].fillna(0)
    else:
        df['F2_IP_Protection'] = 1000
    
    # =========================================================================
    # 15. 数据变换与归一化
    # =========================================================================
    print("\n15. 异常值处理 (Winsorization)...")
    winsorize_cols = ['A1_Hardware_Compute', 'E1_Vertical_VC', 'E2_Capital_Flow', 'E3_Ind_Adoption']
    df = winsorize_data(df, winsorize_cols, limits=(0.01, 0.01))
    
    print("16. 对数变换...")
    log_cols = ['A1_Hardware_Compute', 'A2_Energy_IDC', 'B1_Talent_Stock', 
                'C1_Research_Qty', 'C2_High_Impact_Res', 'D1_GitHub_Activity', 
                'D3_OpenSource_Impact', 'E1_Vertical_VC', 'E2_Capital_Flow', 
                'F2_IP_Protection']
    df = log_transform(df, log_cols)
    
    print("17. Min-Max 归一化 (下限0.01避免数值不稳定)...")
    minmax_cols = ['A3_Connectivity', 'B3_STEM_Supply', 'E3_Ind_Adoption', 'F1_Gov_RD_Exp']
    df = min_max_normalize(df, minmax_cols, feature_range=(0.01, 1))
    
    print("\n" + "=" * 60)
    print("预处理完成！指标定义说明:")
    print("=" * 60)
    print("   A1 (硬件算力): TOP500 Rmax份额×0.7 + 芯片贸易×0.3")
    print("   A2 (能源IDC): 电力×0.6 + 安全服务器×0.4")
    print("   B1 (人才存量): AI人才×0.6 + 研究人员×0.4")
    print("   B3 (STEM教育): CS Rankings×0.7 + 高等教育率×0.3")
    print("   E1 (垂直VC): GenAI_VC + AI_Compute_VC")
    print("   E2 (资本流动): Total_AI_VC + Crossborder_VC")
    print("   E3 (行业渗透): 行业VC×0.6 + 高科技出口×0.4")
    
    # 验证数据源
    validate_data_sources(df)
    
    return df


# =============================================================================
# Step 1: DEMATEL 因果权重计算
# =============================================================================

def build_indicator_matrix(data, indicator_cols):
    """
    构建指标数据矩阵
    """
    # 确保只使用存在的列
    valid_cols = [col for col in indicator_cols if col in data.columns]
    
    # 按国家和年份聚合，取均值
    grouped = data.groupby(['Country', 'Year'])[valid_cols].mean().reset_index()
    
    return grouped, valid_cols


def build_expert_causal_matrix(indicator_cols):
    """
    根据 NIS-AI (国家人工智能创新系统) 框架构建专家因果图
    
    叙事逻辑：
    - 源头因子 (Cause/Input): A (算力基建), F1 (R&D投入), E1/E2 (资本活力)
    - 中介因子 (Mediator/Process): B (人才), C (科研), D (开源)
    - 结果因子 (Effect/Output): E3 (产业转化), C2/D3 (前沿影响)
    
    影响强度：0=无影响, 1=弱, 2=中, 3=强, 4=极强
    """
    # 指标简称映射
    indicator_map = {
        'A1_Hardware_Compute_log': 'A1',
        'A2_Energy_IDC_log': 'A2',
        'A3_Connectivity_norm': 'A3',
        'B1_Talent_Stock_log': 'B1',
        'B3_STEM_Supply_norm': 'B3',
        'C1_Research_Qty_log': 'C1',
        'C2_High_Impact_Res_log': 'C2',
        'D1_GitHub_Activity_log': 'D1',
        'D3_OpenSource_Impact_log': 'D3',
        'E1_Vertical_VC_log': 'E1',
        'E2_Capital_Flow_log': 'E2',
        'E3_Ind_Adoption_norm': 'E3',
        'F1_Gov_RD_Exp_norm': 'F1',
        'F2_IP_Protection_log': 'F2'
    }
    
    # 定义专家因果关系矩阵
    # 行影响列 (Z[i,j] = i对j的影响强度)
    causal_relations = {
        # A1 (算力) 驱动: 科研、开源、产业
        ('A1', 'C1'): 4, ('A1', 'C2'): 3, ('A1', 'D1'): 3, ('A1', 'D3'): 2, ('A1', 'E3'): 3,
        
        # A2 (能源基建) 支撑算力
        ('A2', 'A1'): 3, ('A2', 'D1'): 2,
        
        # A3 (连接) 支撑协作
        ('A3', 'D1'): 2, ('A3', 'B1'): 1,
        
        # B1 (人才存量) 是核心枢纽
        ('B1', 'C1'): 4, ('B1', 'C2'): 4, ('B1', 'D1'): 3, ('B1', 'D3'): 3, ('B1', 'E3'): 2,
        
        # B3 (STEM教育) 培养人才
        ('B3', 'B1'): 4, ('B3', 'C1'): 2,
        
        # C1 (科研产出) -> 高影响力产出
        ('C1', 'C2'): 4, ('C1', 'D1'): 2, ('C1', 'F2'): 2,
        
        # C2 (高影响力科研) -> 产业转化
        ('C2', 'E3'): 3, ('C2', 'F2'): 3,
        
        # D1 (GitHub活跃度) -> 高影响力项目
        ('D1', 'D3'): 4, ('D1', 'E3'): 2,
        
        # D3 (开源影响力) -> 产业转化
        ('D3', 'E3'): 3,
        
        # E1 (垂直VC) 驱动创新
        ('E1', 'A1'): 3, ('E1', 'D1'): 3, ('E1', 'E3'): 2,
        
        # E2 (资本流动) 是系统能量
        ('E2', 'E1'): 3, ('E2', 'B1'): 2, ('E2', 'C1'): 2, ('E2', 'D1'): 2,
        
        # E3 (产业转化) 反馈资本
        ('E3', 'E1'): 2, ('E3', 'E2'): 2,
        
        # F1 (R&D投入) 是源头驱动
        ('F1', 'A1'): 3, ('F1', 'B1'): 3, ('F1', 'C1'): 4, ('F1', 'E2'): 2,
        
        # F2 (知识产权) 保护创新
        ('F2', 'E1'): 2, ('F2', 'E3'): 2
    }
    
    # 构建矩阵
    valid_cols = [col for col in indicator_cols if col in indicator_map]
    n = len(valid_cols)
    Z = np.zeros((n, n))
    
    for i, col_i in enumerate(valid_cols):
        for j, col_j in enumerate(valid_cols):
            key_i = indicator_map.get(col_i)
            key_j = indicator_map.get(col_j)
            if key_i and key_j and (key_i, key_j) in causal_relations:
                Z[i, j] = causal_relations[(key_i, key_j)]
    
    return Z, valid_cols


def calculate_dematel_weights(data, indicator_cols):
    """
    DEMATEL (决策试验与评价实验室法) 权重计算 (优化版)
    
    基于 NIS-AI 框架的专家因果图计算权重，数据相关性作为校准。
    
    优化内容:
    - 使用Spearman相关替代Pearson，减少共线性噪声
    - 对相关强度做阈值截断 (|r|<0.2设为0)
    - 仅在专家图存在边的位置引入数据校准
    
    步骤:
    1. 建立直接影响矩阵 Z: 基于专家因果图 + 数据相关性混合
    2. 归一化: N = Z / max(行和)
    3. 计算总关系矩阵: T = N * (I - N)^(-1)
    4. 计算中心度权重: w_j = (R_j + C_j) / sum(R_k + C_k)
    """
    print("\n" + "=" * 60)
    print("Step 1: DEMATEL 因果权重计算 (优化版)")
    print("=" * 60)
    
    # 构建指标矩阵
    grouped, valid_cols = build_indicator_matrix(data, indicator_cols)
    
    # 提取数值矩阵
    X = grouped[valid_cols].values
    n_indicators = len(valid_cols)
    
    # 处理缺失值（使用列中位数填充，避免2024-2025年数据缺失导致的评分偏低）
    X = nan_to_col_median(X)
    
    # 1. 获取专家因果矩阵
    print("\n1. 构建专家因果矩阵 (基于NIS-AI框架)...")
    expert_Z, expert_cols = build_expert_causal_matrix(indicator_cols)
    
    # 2. 计算数据驱动的相关系数矩阵 (使用Spearman替代Pearson)
    print("2. 计算数据驱动的相关系数矩阵 (Spearman + 阈值截断)...")
    
    # 使用Spearman相关，减少极端值和共线性的影响
    n_samples, n_cols = X.shape
    spearman_corr = np.zeros((n_cols, n_cols))
    for i in range(n_cols):
        for j in range(n_cols):
            if i != j:
                rho, _ = spearmanr(X[:, i], X[:, j])
                spearman_corr[i, j] = rho if not np.isnan(rho) else 0
    
    # 取绝对值并应用阈值截断 (|r|<0.2 → 0，抑制弱相关噪声)
    data_Z = np.abs(spearman_corr)
    data_Z[data_Z < 0.2] = 0  # 阈值截断
    np.fill_diagonal(data_Z, 0)
    
    # 3. 混合专家矩阵和数据矩阵 (专家权重0.7, 数据权重0.3)
    # 优化: 仅在专家图存在边的位置，用数据相关去"校准边强度"
    print("3. 混合专家因果矩阵与数据相关矩阵 (专家主导+数据校准)...")
    expert_weight = 0.7
    data_weight = 0.3
    
    # 归一化专家矩阵到 [0, 1]
    if expert_Z.max() > 0:
        expert_Z_norm = expert_Z / expert_Z.max()
    else:
        expert_Z_norm = expert_Z
    
    # 创建混合矩阵 (仅在专家有边处引入数据校准)
    Z = np.zeros_like(expert_Z_norm)
    for i in range(n_indicators):
        for j in range(n_indicators):
            if expert_Z[i, j] > 0:
                # 专家有边: 混合专家判断和数据相关
                Z[i, j] = expert_weight * expert_Z_norm[i, j] + data_weight * data_Z[i, j]
            else:
                # 专家无边: 仅当数据相关强(>0.5)才引入弱边
                if data_Z[i, j] > 0.5:
                    Z[i, j] = data_weight * data_Z[i, j] * 0.3  # 降权处理
    
    np.fill_diagonal(Z, 0)  # 对角线设为0
    
    # 4. 归一化
    print("4. 归一化直接影响矩阵...")
    row_sums = Z.sum(axis=1)
    max_row_sum = max(row_sums.max(), 1e-10)
    N = Z / max_row_sum
    
    # 5. 计算总关系矩阵 T = N * (I - N)^(-1)
    print("5. 计算总关系矩阵 T...")
    I = np.eye(n_indicators)
    try:
        T = N @ np.linalg.inv(I - N)
    except np.linalg.LinAlgError:
        # 如果矩阵奇异，使用伪逆
        T = N @ np.linalg.pinv(I - N)
    
    # 6. 计算影响度和被影响度
    print("6. 计算影响度 (R) 和被影响度 (C)...")
    R = T.sum(axis=1)  # 行和：影响度
    C = T.sum(axis=0)  # 列和：被影响度
    
    # 7. 计算中心度 (R + C) 和原因度 (R - C)
    centrality = R + C  # 中心度：重要程度
    causality = R - C   # 原因度：驱动/结果
    
    # 8. 计算权重
    print("7. 计算中心度权重...")
    weights = centrality / centrality.sum()
    
    # 输出结果
    print("\n" + "-" * 60)
    print("DEMATEL 分析结果:")
    print("-" * 60)
    print(f"{'指标':<30} {'影响度R':<12} {'被影响度C':<12} {'中心度R+C':<12} {'原因度R-C':<12} {'权重':<10}")
    print("-" * 60)
    
    results = []
    for i, col in enumerate(valid_cols):
        results.append({
            'Indicator': col,
            'R': R[i],
            'C': C[i],
            'Centrality': centrality[i],
            'Causality': causality[i],
            'Weight': weights[i]
        })
        print(f"{col:<30} {R[i]:<12.4f} {C[i]:<12.4f} {centrality[i]:<12.4f} {causality[i]:<12.4f} {weights[i]:<10.4f}")
    
    print("-" * 60)
    
    # 判断因果类型
    print("\n因果分析:")
    for i, col in enumerate(valid_cols):
        if causality[i] > 0:
            print(f"  {col}: 原因因子 (驱动其他指标)")
        else:
            print(f"  {col}: 结果因子 (被其他指标驱动)")
    
    return dict(zip(valid_cols, weights)), T, pd.DataFrame(results)


# =============================================================================
# Step 2: Global Dynamic TOPSIS 评价
# =============================================================================

def global_topsis(data, indicator_cols, weights, years):
    """
    基于全局理想解的动态 TOPSIS 评价
    
    步骤:
    1. 构建全局决策矩阵 (所有年份堆叠)
    2. 全局归一化
    3. 加权决策矩阵
    4. 确定全局正负理想解
    5. 计算距离与得分
    """
    print("\n" + "=" * 60)
    print("Step 2: Global Dynamic TOPSIS 评价")
    print("=" * 60)
    
    # 获取有效列
    valid_cols = [col for col in indicator_cols if col in data.columns]
    
    # 1. 构建全局决策矩阵
    print("\n1. 构建全局决策矩阵...")
    all_data = []
    for year in years:
        year_data = data[data['Year'] == year].copy()
        if len(year_data) > 0:
            all_data.append(year_data)
    
    if not all_data:
        print("警告: 没有有效数据!")
        return None
    
    global_df = pd.concat(all_data, ignore_index=True)
    
    # 提取决策矩阵（使用列中位数填充NaN，避免最新年份数据缺失导致的评分偏低）
    X = global_df[valid_cols].values
    X = nan_to_col_median(X)
    
    # 2. 全局归一化 (正向指标)
    print("2. 全局归一化...")
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1  # 避免除零
    X_norm = (X - X_min) / X_range
    
    # 3. 加权决策矩阵
    print("3. 构建加权决策矩阵...")
    w = np.array([weights.get(col, 1/len(valid_cols)) for col in valid_cols])
    V = X_norm * w
    
    # 4. 确定全局正负理想解
    print("4. 确定全局正负理想解...")
    A_plus = V.max(axis=0)   # 正理想解
    A_minus = V.min(axis=0)  # 负理想解
    
    print(f"   正理想解 A+: {A_plus[:5]}...")
    print(f"   负理想解 A-: {A_minus[:5]}...")
    
    # 5. 计算距离与得分
    print("5. 计算距离与得分...")
    D_plus = np.sqrt(((V - A_plus) ** 2).sum(axis=1))
    D_minus = np.sqrt(((V - A_minus) ** 2).sum(axis=1))
    
    # TOPSIS 得分
    scores = D_minus / (D_plus + D_minus + 1e-10)
    
    # 添加得分到数据框
    global_df['TOPSIS_Score'] = scores
    
    # 6. 按年份输出排名
    print("\n" + "-" * 60)
    print("各年度 AI 竞争力排名 (TOPSIS)")
    print("-" * 60)
    
    results_by_year = {}
    for year in sorted(years):
        year_scores = global_df[global_df['Year'] == year][['Country', 'TOPSIS_Score']].copy()
        if len(year_scores) > 0:
            year_scores = year_scores.sort_values('TOPSIS_Score', ascending=False)
            year_scores['Rank'] = range(1, len(year_scores) + 1)
            year_scores['Country_CN'] = year_scores['Country'].map(COUNTRY_NAMES_CN)
            results_by_year[year] = year_scores
            
            print(f"\n{year}年排名:")
            for _, row in year_scores.head(10).iterrows():
                print(f"  {row['Rank']:>2}. {row['Country_CN']:<6} ({row['Country']:<3}): {row['TOPSIS_Score']:.4f}")
    
    return global_df, results_by_year


# =============================================================================
# Step 3: DEA-Malmquist 效率分析
# =============================================================================

def solve_dea(X_inputs, Y_outputs, k):
    """
    求解 DEA-CCR 模型 (基于产出导向)
    
    max θ
    s.t. Σλ_j * x_ij ≤ x_ik  (投入约束)
         Σλ_j * y_rj ≥ θ * y_rk  (产出约束)
         λ_j ≥ 0
    
    Args:
        X_inputs: 投入矩阵 (n_samples, n_inputs)
        Y_outputs: 产出矩阵 (n_samples, n_outputs)
        k: 待评价 DMU 的索引
    
    Returns:
        theta: 效率值 (距离函数的倒数)
    """
    n_samples, n_inputs = X_inputs.shape
    _, n_outputs = Y_outputs.shape
    
    # 决策变量: [θ, λ_1, ..., λ_n]
    # 目标: max θ -> min -θ
    c = np.zeros(1 + n_samples)
    c[0] = -1  # 最大化 θ
    
    # 不等式约束: A_ub @ x <= b_ub
    # 投入约束: Σλ_j * x_ij ≤ x_ik
    A_ub_inputs = np.hstack([np.zeros((n_inputs, 1)), X_inputs.T])
    b_ub_inputs = X_inputs[k, :]
    
    # 产出约束: θ * y_rk - Σλ_j * y_rj ≤ 0
    A_ub_outputs = np.hstack([Y_outputs[k:k+1, :].T, -Y_outputs.T])
    b_ub_outputs = np.zeros(n_outputs)
    
    A_ub = np.vstack([A_ub_inputs, A_ub_outputs])
    b_ub = np.hstack([b_ub_inputs, b_ub_outputs])
    
    # 变量边界
    bounds = [(1e-6, None)] + [(0, None)] * n_samples  # θ > 0, λ ≥ 0
    
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if result.success:
            theta = result.x[0]
            return 1 / theta if theta > 0 else 1
        else:
            return 1
    except Exception:
        return 1


def solve_dea_cross_period(X_frontier, Y_frontier, x_eval, y_eval):
    """
    求解跨期 DEA 模型 - 在一个时期的前沿面下评价另一个时期的DMU
    
    这是计算 Malmquist 指数所需的跨期距离函数
    
    Args:
        X_frontier: 前沿面投入矩阵 (n_samples, n_inputs)
        Y_frontier: 前沿面产出矩阵 (n_samples, n_outputs)
        x_eval: 待评价DMU的投入向量
        y_eval: 待评价DMU的产出向量
    
    Returns:
        theta: 距离函数值
    """
    n_samples, n_inputs = X_frontier.shape
    _, n_outputs = Y_frontier.shape
    
    # 决策变量: [θ, λ_1, ..., λ_n]
    c = np.zeros(1 + n_samples)
    c[0] = -1  # 最大化 θ
    
    # 投入约束: Σλ_j * x_ij ≤ x_eval_i
    A_ub_inputs = np.hstack([np.zeros((n_inputs, 1)), X_frontier.T])
    b_ub_inputs = x_eval
    
    # 产出约束: θ * y_eval_r - Σλ_j * y_rj ≤ 0
    A_ub_outputs = np.hstack([y_eval.reshape(-1, 1), -Y_frontier.T])
    b_ub_outputs = np.zeros(n_outputs)
    
    A_ub = np.vstack([A_ub_inputs, A_ub_outputs])
    b_ub = np.hstack([b_ub_inputs, b_ub_outputs])
    
    bounds = [(1e-6, None)] + [(0, None)] * n_samples
    
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if result.success:
            theta = result.x[0]
            return 1 / theta if theta > 0 else 1
        else:
            return 1
    except Exception:
        return 1


def calculate_malmquist(data, input_cols, output_cols, years):
    """
    计算 DEA-Malmquist 生产率指数
    
    MPI = EC × TC
    - EC (Efficiency Change): 效率变化
    - TC (Technological Change): 技术进步
    """
    print("\n" + "=" * 60)
    print("Step 3: DEA-Malmquist 效率分析")
    print("=" * 60)
    
    # 获取有效列
    valid_input_cols = [col for col in input_cols if col in data.columns]
    valid_output_cols = [col for col in output_cols if col in data.columns]
    
    if len(valid_input_cols) == 0 or len(valid_output_cols) == 0:
        print("警告: 投入或产出变量不足!")
        return None
    
    print(f"\n投入变量 ({len(valid_input_cols)}): {valid_input_cols}")
    print(f"产出变量 ({len(valid_output_cols)}): {valid_output_cols}")
    
    countries = data['Country'].unique()
    years = sorted([y for y in years if y in data['Year'].values])
    
    results = []
    
    for i, year in enumerate(years[:-1]):
        year_t = year
        year_t1 = years[i + 1]
        
        # 获取两年的数据
        data_t = data[data['Year'] == year_t].copy()
        data_t1 = data[data['Year'] == year_t1].copy()
        
        if len(data_t) == 0 or len(data_t1) == 0:
            continue
        
        # 构建投入产出矩阵
        X_t = data_t[valid_input_cols].values
        Y_t = data_t[valid_output_cols].values
        X_t1 = data_t1[valid_input_cols].values
        Y_t1 = data_t1[valid_output_cols].values
        
        # 处理缺失值和负值
        X_t = np.nan_to_num(X_t, nan=0.01).clip(min=0.01)
        Y_t = np.nan_to_num(Y_t, nan=0.01).clip(min=0.01)
        X_t1 = np.nan_to_num(X_t1, nan=0.01).clip(min=0.01)
        Y_t1 = np.nan_to_num(Y_t1, nan=0.01).clip(min=0.01)
        
        countries_t = data_t['Country'].values
        countries_t1 = data_t1['Country'].values
        
        # 对每个国家计算 Malmquist 指数
        for country in countries:
            if country in countries_t and country in countries_t1:
                k_t = np.where(countries_t == country)[0][0]
                k_t1 = np.where(countries_t1 == country)[0][0]
                
                try:
                    # 计算四个距离函数
                    # D^t(x^t, y^t): 在t期前沿面下评价t期
                    D_t_t = solve_dea(X_t, Y_t, k_t)
                    # D^{t+1}(x^{t+1}, y^{t+1}): 在t+1期前沿面下评价t+1期
                    D_t1_t1 = solve_dea(X_t1, Y_t1, k_t1)
                    
                    # 跨期距离函数 (使用正确的DEA计算)
                    # D^t(x^{t+1}, y^{t+1}): 在t期前沿面下评价t+1期的数据
                    x_t1 = X_t1[k_t1, :]
                    y_t1 = Y_t1[k_t1, :]
                    D_t_t1 = solve_dea_cross_period(X_t, Y_t, x_t1, y_t1)
                    
                    # D^{t+1}(x^t, y^t): 在t+1期前沿面下评价t期的数据
                    x_t = X_t[k_t, :]
                    y_t = Y_t[k_t, :]
                    D_t1_t = solve_dea_cross_period(X_t1, Y_t1, x_t, y_t)
                    
                    # 效率变化 EC = D^{t+1}(x^{t+1}, y^{t+1}) / D^t(x^t, y^t)
                    EC = D_t1_t1 / D_t_t if D_t_t > 0 else 1
                    
                    # 技术进步 TC = sqrt[(D^t(x^{t+1}, y^{t+1}) / D^{t+1}(x^{t+1}, y^{t+1})) * 
                    #                    (D^t(x^t, y^t) / D^{t+1}(x^t, y^t))]
                    if D_t1_t1 > 0 and D_t1_t > 0:
                        TC = np.sqrt((D_t_t1 / D_t1_t1) * (D_t_t / D_t1_t))
                    else:
                        TC = 1
                    
                    # Malmquist 指数 MPI = EC × TC
                    MPI = EC * TC
                    
                    results.append({
                        'Country': country,
                        'Period': f'{year_t}-{year_t1}',
                        'Year_Start': year_t,
                        'Year_End': year_t1,
                        'EC': EC,
                        'TC': TC,
                        'MPI': MPI
                    })
                except Exception as e:
                    results.append({
                        'Country': country,
                        'Period': f'{year_t}-{year_t1}',
                        'Year_Start': year_t,
                        'Year_End': year_t1,
                        'EC': 1,
                        'TC': 1,
                        'MPI': 1
                    })
    
    if not results:
        print("警告: 无法计算 Malmquist 指数!")
        return None
    
    results_df = pd.DataFrame(results)
    
    # 输出汇总
    print("\n" + "-" * 60)
    print("Malmquist 生产率指数汇总 (按国家)")
    print("-" * 60)
    
    summary = results_df.groupby('Country').agg({
        'EC': 'mean',
        'TC': 'mean', 
        'MPI': 'mean'
    }).round(4)
    
    summary['Country_CN'] = summary.index.map(COUNTRY_NAMES_CN)
    summary = summary.sort_values('MPI', ascending=False)
    
    print(f"\n{'国家':<10} {'效率变化EC':<12} {'技术进步TC':<12} {'全要素生产率MPI':<15}")
    print("-" * 60)
    for country, row in summary.iterrows():
        cn_name = COUNTRY_NAMES_CN.get(country, country)
        ec_interpretation = '进步' if row['EC'] > 1 else '退步'
        tc_interpretation = '进步' if row['TC'] > 1 else '退步'
        mpi_interpretation = '进步' if row['MPI'] > 1 else '退步'
        print(f"{cn_name:<10} {row['EC']:<12.4f} {row['TC']:<12.4f} {row['MPI']:<15.4f}")
    
    return results_df, summary


# =============================================================================
# 指标相关性分析
# =============================================================================

def analyze_indicator_correlations(data, indicator_cols, save_path=None):
    """
    分析指标之间的内在相关性
    探索因素之间的相互作用和相互影响
    """
    print("\n" + "=" * 60)
    print("指标相关性分析 - 探索内在关联")
    print("=" * 60)
    
    # 获取有效列
    valid_cols = [col for col in indicator_cols if col in data.columns]
    
    # 构建数据矩阵
    grouped = data.groupby(['Country', 'Year'])[valid_cols].mean().reset_index()
    X = grouped[valid_cols].values
    X = nan_to_col_median(X)
    
    # 计算相关系数矩阵
    corr_matrix = pd.DataFrame(X, columns=valid_cols).corr()
    
    # 分析结果
    print("\n" + "-" * 60)
    print("一、高度相关指标对 (|r| > 0.7):")
    print("-" * 60)
    
    high_corr_pairs = []
    for i in range(len(valid_cols)):
        for j in range(i+1, len(valid_cols)):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) > 0.7:
                high_corr_pairs.append((valid_cols[i], valid_cols[j], corr))
                direction = "正相关" if corr > 0 else "负相关"
                print(f"  {valid_cols[i][:25]:<25} <-> {valid_cols[j][:25]:<25}: {corr:.4f} ({direction})")
    
    print("\n" + "-" * 60)
    print("二、指标分组分析 (基于相关性聚类):")
    print("-" * 60)
    
    # 简化指标名称映射
    short_names = {
        'A1_Hardware_Compute_log': 'A1 Compute',
        'A2_Energy_IDC_log': 'A2 Energy',
        'A3_Connectivity_norm': 'A3 Connect',
        'B1_Talent_Stock_log': 'B1 Talent',
        'B3_STEM_Supply_norm': 'B3 STEM',
        'C1_Research_Qty_log': 'C1 Research',
        'C2_High_Impact_Res_log': 'C2 High-Impact',
        'D1_GitHub_Activity_log': 'D1 GitHub',
        'D3_OpenSource_Impact_log': 'D3 OpenSource',
        'E1_Vertical_VC_log': 'E1 Vertical VC',
        'E2_Capital_Flow_log': 'E2 Capital',
        'E3_Ind_Adoption_norm': 'E3 Adoption',
        'F1_Gov_RD_Exp_norm': 'F1 R&D Exp',
        'F2_IP_Protection_log': 'F2 IP Protect'
    }
    
    # 分析因素间的相互作用关系
    print("\n【因素相互作用分析】")
    print("\n1. 投入因素群 (驱动力):")
    input_factors = ['A1_Hardware_Compute_log', 'A2_Energy_IDC_log', 'E1_Vertical_VC_log', 
                     'E2_Capital_Flow_log', 'F1_Gov_RD_Exp_norm']
    for f in input_factors:
        if f in valid_cols:
            avg_corr = corr_matrix.loc[f, [c for c in valid_cols if c != f]].mean()
            print(f"   - {short_names.get(f, f)}: 平均相关度 {avg_corr:.4f}")
    
    print("\n2. 人力资本因素群 (中介):")
    human_factors = ['B1_Talent_Stock_log', 'B3_STEM_Supply_norm']
    for f in human_factors:
        if f in valid_cols:
            avg_corr = corr_matrix.loc[f, [c for c in valid_cols if c != f]].mean()
            print(f"   - {short_names.get(f, f)}: 平均相关度 {avg_corr:.4f}")
    
    print("\n3. 创新产出因素群 (结果):")
    output_factors = ['C1_Research_Qty_log', 'C2_High_Impact_Res_log', 'D1_GitHub_Activity_log', 
                      'D3_OpenSource_Impact_log', 'E3_Ind_Adoption_norm']
    for f in output_factors:
        if f in valid_cols:
            avg_corr = corr_matrix.loc[f, [c for c in valid_cols if c != f]].mean()
            print(f"   - {short_names.get(f, f)}: 平均相关度 {avg_corr:.4f}")
    
    # 绘制相关性热力图
    if save_path:
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 简化列名用于显示
        display_cols = [short_names.get(c, c[:15]) for c in valid_cols]
        
        im = ax.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1)
        
        ax.set_xticks(np.arange(len(valid_cols)))
        ax.set_yticks(np.arange(len(valid_cols)))
        ax.set_xticklabels(display_cols, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(display_cols, fontsize=9)
        
        # 添加数值标注
        for i in range(len(valid_cols)):
            for j in range(len(valid_cols)):
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                             ha='center', va='center', fontsize=7,
                             color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black')
        
        ax.set_title('AI Development Indicator Correlation Matrix', fontsize=14)
        plt.colorbar(im, ax=ax, label='Correlation')
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n相关性热力图已保存: {save_path}")
        plt.close()
    
    # 返回相关性总结
    summary = {
        'correlation_matrix': corr_matrix,
        'high_corr_pairs': high_corr_pairs,
        'indicator_groups': {
            'input_factors': input_factors,
            'human_capital': human_factors,
            'output_factors': output_factors
        }
    }
    
    return summary


def print_factor_interaction_analysis(dematel_results, corr_summary):
    """
    打印因素相互作用的详细分析报告
    """
    print("\n" + "=" * 80)
    print("指标因素相互作用与影响机制分析报告")
    print("=" * 80)
    
    print("""
【一、指标体系构建说明】

本研究基于"能力(Capability) + 准备度(Readiness)"的统一框架构建国家AI竞争力指标体系。
共设置6个一级指标，14个二级指标，涵盖AI发展的全生命周期。

一级指标:
  A. 算力与数字基础设施 (Compute & Infrastructure) - 硬指标
  B. 人才与教育 (Talent & Skills) - 关键中介
  C. 科研产出与前沿影响 (Research & Frontier Impact) - 硬指标
  D. 开源生态与工程化能力 (Open-source & Engineering) - 硬指标
  E. 产业转化与资本活力 (Adoption, Startups & Capital) - 关键产出
  F. 治理与采用准备度 (Governance & Readiness) - 政策环境

【二、因素间因果关系分析 (基于DEMATEL)】
""")
    
    # 分析原因因子
    cause_factors = dematel_results[dematel_results['Causality'] > 0].sort_values('Causality', ascending=False)
    effect_factors = dematel_results[dematel_results['Causality'] <= 0].sort_values('Causality')
    
    print("\n原因因子 (驱动AI发展的源头):")
    print("-" * 60)
    for _, row in cause_factors.iterrows():
        indicator = row['Indicator']
        short_name = indicator.replace('_log', '').replace('_norm', '')
        print(f"  * {short_name}: 原因度={row['Causality']:.4f}, 中心度={row['Centrality']:.4f}")
        if 'F1' in indicator:
            print("    -> 研发投入是整个系统的能量源头，驱动基建和人才培养")
        elif 'A1' in indicator or 'A2' in indicator:
            print("    -> 算力基建是AI能力的物理底座，支撑科研和工程化")
        elif 'E1' in indicator or 'E2' in indicator:
            print("    -> 资本活力提供发展动力，促进技术转化和人才聚集")
        elif 'B1' in indicator:
            print("    -> 人才存量是核心枢纽，连接投入与产出")
    
    print("\n结果因子 (AI发展的成果表现):")
    print("-" * 60)
    for _, row in effect_factors.iterrows():
        indicator = row['Indicator']
        short_name = indicator.replace('_log', '').replace('_norm', '')
        print(f"  * {short_name}: 原因度={row['Causality']:.4f}, 中心度={row['Centrality']:.4f}")
        if 'E3' in indicator:
            print("    -> 产业转化是最终经济产出，受所有因素综合影响")
        elif 'C1' in indicator or 'C2' in indicator:
            print("    -> 科研产出是知识创造的体现，依赖人才和资金")
        elif 'D1' in indicator or 'D3' in indicator:
            print("    -> 开源生态是技术扩散的载体，需要人才和基建支撑")
    
    print("""
【三、相互作用机制总结】

1. 正反馈循环 (马太效应):
   投入增加 -> 人才聚集 -> 科研产出增加 -> 吸引更多资本 -> 投入继续增加
   
2. 时滞效应:
   当年的投入 (算力、资金、教育) 需要 2-3 年才能转化为可见的产出
   
3. 涌现效应:
   算力 + 数据 + 算法的结合产生 1+1>>2 的非线性增长
   
4. 关键节点:
   - 人才 (B1) 是系统的核心枢纽，连接所有要素
   - 算力 (A1) 是硬实力的关键载体
   - 资本 (E2) 是系统运转的燃料
""")


# =============================================================================
# 可视化函数
# =============================================================================

def plot_dematel_results(dematel_results, save_path=None):
    """
    DEMATEL 分析结果可视化
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. 因果关系图
    ax1 = axes[0]
    centrality = dematel_results['Centrality'].values
    causality = dematel_results['Causality'].values
    labels = dematel_results['Indicator'].values
    
    colors = ['red' if c > 0 else 'blue' for c in causality]
    ax1.scatter(centrality, causality, c=colors, s=100, alpha=0.7)
    
    for i, label in enumerate(labels):
        ax1.annotate(label, (centrality[i], causality[i]), fontsize=8, ha='center')
    
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax1.set_xlabel('Centrality (R + C)', fontsize=12)
    ax1.set_ylabel('Causality (R - C)', fontsize=12)
    ax1.set_title('DEMATEL Cause-Effect Diagram', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # 2. 权重柱状图
    ax2 = axes[1]
    weights = dematel_results['Weight'].values
    y_pos = np.arange(len(labels))
    
    ax2.barh(y_pos, weights, color='steelblue', alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.set_xlabel('Weight', fontsize=12)
    ax2.set_title('DEMATEL Indicator Weights', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存: {save_path}")
    
    plt.close()


def plot_topsis_rankings(results_by_year, save_path=None):
    """
    TOPSIS 排名趋势图
    """
    # 收集所有国家的排名数据
    countries = list(COUNTRY_NAMES_CN.keys())
    years = sorted(results_by_year.keys())
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for country in countries:
        scores = []
        valid_years = []
        for year in years:
            if year in results_by_year:
                year_data = results_by_year[year]
                country_score = year_data[year_data['Country'] == country]['TOPSIS_Score']
                if len(country_score) > 0:
                    scores.append(country_score.values[0])
                    valid_years.append(year)
        
        if scores:
            ax.plot(valid_years, scores, marker='o', label=COUNTRY_NAMES_CN.get(country, country), linewidth=2)
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('TOPSIS Score', fontsize=12)
    ax.set_title('AI Competitiveness Score Trends (2016-2025)', fontsize=14)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存: {save_path}")
    
    plt.close()


def plot_malmquist_summary(malmquist_summary, save_path=None):
    """
    Malmquist 指数可视化
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    countries = malmquist_summary.index
    x = np.arange(len(countries))
    width = 0.25
    
    ax.bar(x - width, malmquist_summary['EC'], width, label='EC (Efficiency Change)', alpha=0.8)
    ax.bar(x, malmquist_summary['TC'], width, label='TC (Technological Change)', alpha=0.8)
    ax.bar(x + width, malmquist_summary['MPI'], width, label='MPI (Total Productivity)', alpha=0.8)
    
    ax.axhline(y=1, color='red', linestyle='--', linewidth=1, label='Benchmark (=1)')
    
    ax.set_xlabel('Country', fontsize=12)
    ax.set_ylabel('Index Value', fontsize=12)
    ax.set_title('DEA-Malmquist Productivity Index by Country', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([COUNTRY_NAMES_CN.get(c, c) for c in countries], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存: {save_path}")
    
    plt.close()


# =============================================================================
# Step 4: 稳健性检验 (Robustness Tests)
# =============================================================================

def run_topsis_with_weights(data, indicator_cols, weights, year=2025):
    """
    使用给定权重运行TOPSIS评价，返回特定年份的排名
    """
    valid_cols = [col for col in indicator_cols if col in data.columns]
    year_data = data[data['Year'] == year].copy()
    
    if len(year_data) == 0:
        return None
    
    X = year_data[valid_cols].values
    X = nan_to_col_median(X)  # 使用列中位数填充，避免最新年份数据缺失导致评分偏低
    
    # 全局归一化
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1
    X_norm = (X - X_min) / X_range
    
    # 加权决策矩阵
    w = np.array([weights.get(col, 1/len(valid_cols)) for col in valid_cols])
    V = X_norm * w
    
    # 理想解
    A_plus = V.max(axis=0)
    A_minus = V.min(axis=0)
    
    # 距离与得分
    D_plus = np.sqrt(((V - A_plus) ** 2).sum(axis=1))
    D_minus = np.sqrt(((V - A_minus) ** 2).sum(axis=1))
    scores = D_minus / (D_plus + D_minus + 1e-10)
    
    year_data = year_data.copy()
    year_data['TOPSIS_Score'] = scores
    year_data = year_data.sort_values('TOPSIS_Score', ascending=False)
    year_data['Rank'] = range(1, len(year_data) + 1)
    
    return year_data[['Country', 'TOPSIS_Score', 'Rank']]


def robustness_weight_perturbation(data, indicator_cols, base_weights, n_simulations=1000, 
                                    perturbation_pct=0.1, year=2025, output_path=None, 
                                    random_seed=42):
    """
    稳健性检验: Monte Carlo 权重扰动分析
    
    对DEMATEL权重做随机扰动（每个权重乘以 1±perturbation_pct 的均匀随机数），
    每次扰动后归一化权重和为1，重复N次，统计各国获得第1/第2/第3的频率。
    
    Args:
        data: 预处理后的数据
        indicator_cols: 指标列名
        base_weights: 基准权重字典
        n_simulations: 模拟次数 (默认1000)
        perturbation_pct: 扰动幅度 (默认±10%)
        year: 目标年份 (默认2025)
        output_path: 输出文件路径
        random_seed: 随机种子用于可复现性 (默认42)
    
    Returns:
        robustness_df: 稳健性统计结果DataFrame
    """
    print("\n" + "=" * 60)
    print(f"Step 4.1: Monte Carlo 权重扰动分析 (N={n_simulations})")
    print("=" * 60)
    
    # 设置随机种子以确保结果可复现
    rng = np.random.default_rng(random_seed)
    
    valid_cols = [col for col in indicator_cols if col in data.columns]
    base_w = np.array([base_weights.get(col, 1/len(valid_cols)) for col in valid_cols])
    
    # 存储每次模拟的排名结果
    rank_records = {country: [] for country in TARGET_COUNTRIES}
    first_place = {country: 0 for country in TARGET_COUNTRIES}
    second_place = {country: 0 for country in TARGET_COUNTRIES}
    third_place = {country: 0 for country in TARGET_COUNTRIES}
    top_two = {country: 0 for country in TARGET_COUNTRIES}
    
    print(f"\n运行 {n_simulations} 次权重扰动模拟 (seed={random_seed})...")
    
    for i in range(n_simulations):
        # 生成扰动权重: w_new = w_base * (1 + uniform(-pert, +pert))
        perturbations = 1 + rng.uniform(-perturbation_pct, perturbation_pct, len(base_w))
        perturbed_w = base_w * perturbations
        perturbed_w = perturbed_w / perturbed_w.sum()  # 归一化
        
        # 构建扰动权重字典
        perturbed_weights = dict(zip(valid_cols, perturbed_w))
        
        # 运行TOPSIS
        result = run_topsis_with_weights(data, indicator_cols, perturbed_weights, year)
        
        if result is not None:
            for _, row in result.iterrows():
                country = row['Country']
                rank = row['Rank']
                if country in rank_records:
                    rank_records[country].append(rank)
                    if rank == 1:
                        first_place[country] += 1
                    elif rank == 2:
                        second_place[country] += 1
                    elif rank == 3:
                        third_place[country] += 1
                    if rank <= 2:
                        top_two[country] += 1
    
    # 计算统计结果
    results = []
    for country in TARGET_COUNTRIES:
        ranks = rank_records.get(country, [])
        if ranks:
            results.append({
                'Country': country,
                'Country_CN': COUNTRY_NAMES_CN.get(country, country),
                'Mean_Rank': np.mean(ranks),
                'Std_Rank': np.std(ranks),
                'Min_Rank': np.min(ranks),
                'Max_Rank': np.max(ranks),
                'First_Place_Freq': first_place[country] / n_simulations,
                'Second_Place_Freq': second_place[country] / n_simulations,
                'Third_Place_Freq': third_place[country] / n_simulations,
                'Top_Two_Freq': top_two[country] / n_simulations
            })
    
    robustness_df = pd.DataFrame(results).sort_values('Mean_Rank')
    
    # 输出结果
    print("\n" + "-" * 80)
    print(f"2025年稳健性统计 (Monte Carlo, N={n_simulations}, 扰动±{perturbation_pct*100:.0f}%)")
    print("-" * 80)
    print(f"{'国家':<8} {'平均排名':<10} {'排名标准差':<12} {'第1频率':<10} {'第2频率':<10} {'前二频率':<10}")
    print("-" * 80)
    for _, row in robustness_df.iterrows():
        print(f"{row['Country_CN']:<8} {row['Mean_Rank']:<10.2f} {row['Std_Rank']:<12.3f} "
              f"{row['First_Place_Freq']:<10.3f} {row['Second_Place_Freq']:<10.3f} "
              f"{row['Top_Two_Freq']:<10.3f}")
    
    # 关键结论
    usa_first = robustness_df[robustness_df['Country'] == 'USA']['First_Place_Freq'].values[0] if len(robustness_df[robustness_df['Country'] == 'USA']) > 0 else 0
    chn_first = robustness_df[robustness_df['Country'] == 'CHN']['First_Place_Freq'].values[0] if len(robustness_df[robustness_df['Country'] == 'CHN']) > 0 else 0
    usa_top2 = robustness_df[robustness_df['Country'] == 'USA']['Top_Two_Freq'].values[0] if len(robustness_df[robustness_df['Country'] == 'USA']) > 0 else 0
    chn_top2 = robustness_df[robustness_df['Country'] == 'CHN']['Top_Two_Freq'].values[0] if len(robustness_df[robustness_df['Country'] == 'CHN']) > 0 else 0
    deu_third = robustness_df[robustness_df['Country'] == 'DEU']['Third_Place_Freq'].values[0] if len(robustness_df[robustness_df['Country'] == 'DEU']) > 0 else 0
    
    print("\n【稳健性结论】")
    print(f"  * 美国(USA)第1名频率: {usa_first:.1%}")
    print(f"  * 中国(CHN)第1名频率: {chn_first:.1%}")
    print(f"  * 美国稳居前二频率: {usa_top2:.1%}")
    print(f"  * 中国稳居前二频率: {chn_top2:.1%}")
    print(f"  * 德国(DEU)第3名频率: {deu_third:.1%}")
    
    # 判断排名稳定性
    if usa_top2 > 0.95 and chn_top2 > 0.95:
        print(f"  -> 中美前二地位非常稳定 (>95%)")
    elif usa_top2 > 0.8 and chn_top2 > 0.8:
        print(f"  -> 中美前二地位较为稳定 (>80%)")
    else:
        print(f"  -> 排名存在一定不确定性")
    
    # 保存结果
    if output_path:
        robustness_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n稳健性结果已保存: {output_path}")
    
    return robustness_df


def robustness_multi_method_comparison(data, indicator_cols, dematel_weights, year=2025, output_path=None):
    """
    稳健性检验: 多权重方法对比
    
    比较等权TOPSIS、熵权TOPSIS与DEMATEL权重TOPSIS的2025排名一致性。
    
    Args:
        data: 预处理后的数据
        indicator_cols: 指标列名
        dematel_weights: DEMATEL权重
        year: 目标年份
        output_path: 输出路径
    
    Returns:
        comparison_df: 对比结果DataFrame
    """
    print("\n" + "=" * 60)
    print("Step 4.2: 多权重方法对比 (等权/熵权/DEMATEL)")
    print("=" * 60)
    
    valid_cols = [col for col in indicator_cols if col in data.columns]
    year_data = data[data['Year'] == year].copy()
    
    if len(year_data) == 0:
        print("警告: 无目标年份数据!")
        return None
    
    X = year_data[valid_cols].values
    X = nan_to_col_median(X)  # 使用列中位数填充
    
    # 1. 等权权重
    equal_weights = {col: 1/len(valid_cols) for col in valid_cols}
    
    # 2. 熵权法计算权重
    print("\n1. 计算熵权权重...")
    # 归一化
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1
    X_norm = (X - X_min) / X_range + 1e-10
    
    # 计算比例矩阵
    P = X_norm / X_norm.sum(axis=0)
    P[P == 0] = 1e-10
    
    # 计算熵值
    k = 1 / np.log(len(X))
    E = -k * np.sum(P * np.log(P + 1e-10), axis=0)
    
    # 计算权重
    D = 1 - E
    entropy_w = D / D.sum()
    entropy_weights = dict(zip(valid_cols, entropy_w))
    
    print("   熵权前5位:")
    sorted_entropy = sorted(entropy_weights.items(), key=lambda x: x[1], reverse=True)
    for col, w in sorted_entropy[:5]:
        print(f"     {col}: {w:.4f}")
    
    # 3. 运行三种方法的TOPSIS
    print("\n2. 运行三种权重方法的TOPSIS...")
    
    result_equal = run_topsis_with_weights(data, indicator_cols, equal_weights, year)
    result_entropy = run_topsis_with_weights(data, indicator_cols, entropy_weights, year)
    result_dematel = run_topsis_with_weights(data, indicator_cols, dematel_weights, year)
    
    # 4. 合并结果
    comparison = result_dematel[['Country']].copy()
    comparison['DEMATEL_Rank'] = result_dematel['Rank'].values
    comparison['DEMATEL_Score'] = result_dematel['TOPSIS_Score'].values
    
    # 合并等权结果
    eq_dict = dict(zip(result_equal['Country'], result_equal['Rank']))
    comparison['Equal_Rank'] = comparison['Country'].map(eq_dict)
    eq_score_dict = dict(zip(result_equal['Country'], result_equal['TOPSIS_Score']))
    comparison['Equal_Score'] = comparison['Country'].map(eq_score_dict)
    
    # 合并熵权结果
    en_dict = dict(zip(result_entropy['Country'], result_entropy['Rank']))
    comparison['Entropy_Rank'] = comparison['Country'].map(en_dict)
    en_score_dict = dict(zip(result_entropy['Country'], result_entropy['TOPSIS_Score']))
    comparison['Entropy_Score'] = comparison['Country'].map(en_score_dict)
    
    comparison['Country_CN'] = comparison['Country'].map(COUNTRY_NAMES_CN)
    
    # 5. 计算Spearman相关系数
    print("\n3. 计算排名一致性 (Spearman相关)...")
    
    rho_de_eq, _ = spearmanr(comparison['DEMATEL_Rank'], comparison['Equal_Rank'])
    rho_de_en, _ = spearmanr(comparison['DEMATEL_Rank'], comparison['Entropy_Rank'])
    rho_eq_en, _ = spearmanr(comparison['Equal_Rank'], comparison['Entropy_Rank'])
    
    print(f"   DEMATEL vs 等权: ρ = {rho_de_eq:.4f}")
    print(f"   DEMATEL vs 熵权: ρ = {rho_de_en:.4f}")
    print(f"   等权 vs 熵权: ρ = {rho_eq_en:.4f}")
    
    # 6. 输出对比表
    print("\n" + "-" * 80)
    print(f"2025年排名对比 (三种权重方法)")
    print("-" * 80)
    print(f"{'国家':<10} {'DEMATEL':<12} {'等权':<12} {'熵权':<12}")
    print("-" * 80)
    for _, row in comparison.iterrows():
        print(f"{row['Country_CN']:<10} {row['DEMATEL_Rank']:<12} {row['Equal_Rank']:<12.0f} {row['Entropy_Rank']:<12.0f}")
    
    print("\n【排名一致性结论】")
    avg_rho = (rho_de_eq + rho_de_en + rho_eq_en) / 3
    if avg_rho > 0.9:
        print(f"  -> 三种方法排名高度一致 (平均rho={avg_rho:.3f})")
    elif avg_rho > 0.7:
        print(f"  -> 三种方法排名较为一致 (平均rho={avg_rho:.3f})")
    else:
        print(f"  -> 三种方法排名存在差异 (平均rho={avg_rho:.3f})，建议综合考虑")
    
    # 保存结果
    if output_path:
        comparison['Spearman_DEMATEL_Equal'] = rho_de_eq
        comparison['Spearman_DEMATEL_Entropy'] = rho_de_en
        comparison['Spearman_Equal_Entropy'] = rho_eq_en
        comparison.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n方法对比结果已保存: {output_path}")
    
    return comparison


# =============================================================================
# Step 5: 2025年贡献分解与短板诊断
# =============================================================================

def calculate_contribution_breakdown(data, indicator_cols, weights, year=2025, output_path=None):
    """
    计算2025年各国各指标的贡献分解
    
    贡献 = 权重 × 全局归一化后的指标值
    
    输出:
    - 每个国家的Top3贡献指标、Bottom3短板指标
    - contribution_breakdown_2025.csv
    """
    print("\n" + "=" * 60)
    print("Step 5: 2025年贡献分解与短板诊断")
    print("=" * 60)
    
    valid_cols = [col for col in indicator_cols if col in data.columns]
    year_data = data[data['Year'] == year].copy()
    
    if len(year_data) == 0:
        print("警告: 无目标年份数据!")
        return None
    
    X = year_data[valid_cols].values
    X = nan_to_col_median(X)  # 使用列中位数填充
    countries = year_data['Country'].values
    
    # 全局归一化
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1
    X_norm = (X - X_min) / X_range
    
    # 计算贡献矩阵
    w = np.array([weights.get(col, 1/len(valid_cols)) for col in valid_cols])
    contribution = X_norm * w  # 每个元素的贡献
    
    # 简化指标名称
    short_names = {
        'A1_Hardware_Compute_log': 'A1 Compute Infra',
        'A2_Energy_IDC_log': 'A2 Energy/IDC',
        'A3_Connectivity_norm': 'A3 Connectivity',
        'B1_Talent_Stock_log': 'B1 Talent Stock',
        'B3_STEM_Supply_norm': 'B3 STEM Supply',
        'C1_Research_Qty_log': 'C1 Research Output',
        'C2_High_Impact_Res_log': 'C2 High-Impact Res',
        'D1_GitHub_Activity_log': 'D1 GitHub Activity',
        'D3_OpenSource_Impact_log': 'D3 OpenSource Impact',
        'E1_Vertical_VC_log': 'E1 Vertical VC',
        'E2_Capital_Flow_log': 'E2 Capital Flow',
        'E3_Ind_Adoption_norm': 'E3 Industry Adoption',
        'F1_Gov_RD_Exp_norm': 'F1 R&D Investment',
        'F2_IP_Protection_log': 'F2 IP Protection'
    }
    
    # 构建贡献DataFrame
    contribution_df = pd.DataFrame(contribution, columns=valid_cols)
    contribution_df['Country'] = countries
    contribution_df['Country_CN'] = contribution_df['Country'].map(COUNTRY_NAMES_CN)
    contribution_df['Total_Score'] = contribution.sum(axis=1)
    
    # 输出每个国家的贡献分析
    print("\n" + "-" * 80)
    print("2025年各国贡献分解 (Top3优势指标 & Bottom3短板指标)")
    print("-" * 80)
    
    analysis_results = []
    
    for i, country in enumerate(countries):
        country_cn = COUNTRY_NAMES_CN.get(country, country)
        contribs = list(zip(valid_cols, contribution[i]))
        contribs_sorted = sorted(contribs, key=lambda x: x[1], reverse=True)
        
        top3 = contribs_sorted[:3]
        bottom3 = contribs_sorted[-3:]
        
        print(f"\n{country_cn} ({country}):")
        print(f"  总得分: {contribution[i].sum():.4f}")
        print("  优势 (Top3):")
        for col, val in top3:
            print(f"    - {short_names.get(col, col)}: {val:.4f}")
        print("  短板 (Bottom3):")
        for col, val in bottom3:
            print(f"    - {short_names.get(col, col)}: {val:.4f}")
        
        analysis_results.append({
            'Country': country,
            'Country_CN': country_cn,
            'Total_Score': contribution[i].sum(),
            'Top1_Indicator': short_names.get(top3[0][0], top3[0][0]),
            'Top1_Value': top3[0][1],
            'Top2_Indicator': short_names.get(top3[1][0], top3[1][0]),
            'Top2_Value': top3[1][1],
            'Top3_Indicator': short_names.get(top3[2][0], top3[2][0]),
            'Top3_Value': top3[2][1],
            'Bottom1_Indicator': short_names.get(bottom3[0][0], bottom3[0][0]),
            'Bottom1_Value': bottom3[0][1],
            'Bottom2_Indicator': short_names.get(bottom3[1][0], bottom3[1][0]),
            'Bottom2_Value': bottom3[1][1],
            'Bottom3_Indicator': short_names.get(bottom3[2][0], bottom3[2][0]),
            'Bottom3_Value': bottom3[2][1]
        })
    
    analysis_df = pd.DataFrame(analysis_results).sort_values('Total_Score', ascending=False)
    
    # 特别分析德国第三的贡献来源
    if 'DEU' in countries:
        print("\n" + "-" * 80)
        print("【德国(DEU)第三名分析】")
        print("-" * 80)
        deu_idx = list(countries).index('DEU')
        deu_contribs = list(zip(valid_cols, contribution[deu_idx]))
        deu_sorted = sorted(deu_contribs, key=lambda x: x[1], reverse=True)
        
        print("德国主要贡献来源 (按贡献排序):")
        for j, (col, val) in enumerate(deu_sorted[:5]):
            pct = val / contribution[deu_idx].sum() * 100
            print(f"  {j+1}. {short_names.get(col, col)}: {val:.4f} ({pct:.1f}%)")
    
    # 保存结果
    if output_path:
        # 保存详细贡献分解
        contribution_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n贡献分解已保存: {output_path}")
    
    return analysis_df, contribution_df


def plot_contribution_radar(contribution_df, countries_to_plot=['USA', 'CHN', 'DEU', 'GBR', 'JPN'], 
                            save_path=None):
    """
    绘制中美德英日的贡献条形图
    """
    print("\n生成贡献分析图...")
    
    valid_cols = [col for col in contribution_df.columns if col.endswith('_log') or col.endswith('_norm')]
    
    # 简化名称
    short_names = {
        'A1_Hardware_Compute_log': 'A1 Compute',
        'A2_Energy_IDC_log': 'A2 Energy',
        'A3_Connectivity_norm': 'A3 Connect',
        'B1_Talent_Stock_log': 'B1 Talent',
        'B3_STEM_Supply_norm': 'B3 STEM',
        'C1_Research_Qty_log': 'C1 Research',
        'C2_High_Impact_Res_log': 'C2 High-Impact',
        'D1_GitHub_Activity_log': 'D1 GitHub',
        'D3_OpenSource_Impact_log': 'D3 OpenSource',
        'E1_Vertical_VC_log': 'E1 Vertical VC',
        'E2_Capital_Flow_log': 'E2 Capital',
        'E3_Ind_Adoption_norm': 'E3 Adoption',
        'F1_Gov_RD_Exp_norm': 'F1 R&D',
        'F2_IP_Protection_log': 'F2 IP'
    }
    
    # 准备数据
    plot_data = contribution_df[contribution_df['Country'].isin(countries_to_plot)]
    
    if len(plot_data) == 0:
        print("警告: 无绘图数据!")
        return
    
    fig, axes = plt.subplots(len(countries_to_plot), 1, figsize=(12, 3*len(countries_to_plot)))
    
    if len(countries_to_plot) == 1:
        axes = [axes]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(valid_cols)))
    
    for idx, country in enumerate(countries_to_plot):
        ax = axes[idx]
        country_data = plot_data[plot_data['Country'] == country]
        
        if len(country_data) == 0:
            continue
        
        values = country_data[valid_cols].values[0]
        labels = [short_names.get(col, col[:8]) for col in valid_cols]
        
        bars = ax.barh(range(len(labels)), values, color=colors)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('Contribution Value', fontsize=10)
        ax.set_title(f'{COUNTRY_NAMES_CN.get(country, country)} ({country}) - Indicator Contribution', fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')
        
        # 添加数值标注
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2, 
                   f'{val:.3f}', va='center', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"贡献分析图已保存: {save_path}")
    
    plt.close()


# =============================================================================
# 主函数
# =============================================================================

def main():
    """
    问题一完整求解流程 (结构增强版)
    
    包含:
    - Step 0: 数据预处理 (解决A1/E1/E2共线问题)
    - Step 1: DEMATEL权重计算 (优化相关矩阵融合)
    - Step 2: Global Dynamic TOPSIS评价
    - Step 3: DEA-Malmquist效率分析
    - Step 4: 稳健性检验 (Monte Carlo + 多方法对比)
    - Step 5: 2025年贡献分解与短板诊断
    """
    # 配置matplotlib中文支持
    setup_matplotlib_chinese()
    
    print("\n" + "=" * 80)
    print("问题一：指标构建与分析 - 完整求解 (结构增强版)")
    print("=" * 80)
    print("\n关键改进:")
    print("  1. A1改用基础设施指数(电力+服务器+宽带)，消除与E2共线")
    print("  2. E1仅使用GenAI_VC，避免compute VC重复计分")
    print("  3. DEMATEL使用Spearman相关+阈值截断，抑制共线噪声")
    print("  4. 新增Monte Carlo稳健性检验和贡献分解")
    
    # Step 0: 数据加载与预处理
    raw_data = load_and_merge_data()
    processed_data = preprocess_indicators(raw_data)
    
    # 定义指标列 (使用预处理后的变量)
    indicator_cols = [
        'A1_Hardware_Compute_log',
        'A2_Energy_IDC_log', 
        'A3_Connectivity_norm',
        'B1_Talent_Stock_log',
        'B3_STEM_Supply_norm',
        'C1_Research_Qty_log',
        'C2_High_Impact_Res_log',
        'D1_GitHub_Activity_log',
        'D3_OpenSource_Impact_log',
        'E1_Vertical_VC_log',
        'E2_Capital_Flow_log',
        'E3_Ind_Adoption_norm',
        'F1_Gov_RD_Exp_norm',
        'F2_IP_Protection_log'
    ]
    
    # Step 1: DEMATEL 权重计算
    weights, T_matrix, dematel_results = calculate_dematel_weights(processed_data, indicator_cols)
    
    # Step 2: Global Dynamic TOPSIS
    years = list(range(YEAR_START, YEAR_END + 1))
    topsis_data, topsis_results = global_topsis(processed_data, indicator_cols, weights, years)
    
    # Step 3: DEA-Malmquist 分析
    # 定义 DEA 投入产出变量
    input_cols = [
        'A1_Hardware_Compute_log',  # 算力基建
        'B1_Talent_Stock_log',       # 人力投入
        'F1_Gov_RD_Exp_norm'         # 资金投入
    ]
    output_cols = [
        'C2_High_Impact_Res_log',    # 科研产出
        'D3_OpenSource_Impact_log',  # 技术产出
        'E3_Ind_Adoption_norm'       # 经济产出
    ]
    
    malmquist_results, malmquist_summary = calculate_malmquist(processed_data, input_cols, output_cols, years)
    
    # 可视化
    print("\n" + "=" * 60)
    print("生成可视化图表")
    print("=" * 60)
    
    output_dir = OUTPUT_DIR  # 使用配置中定义的输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    plot_dematel_results(dematel_results, f'{output_dir}/dematel_analysis.png')
    if topsis_results:
        plot_topsis_rankings(topsis_results, f'{output_dir}/topsis_trends.png')
    if malmquist_summary is not None:
        plot_malmquist_summary(malmquist_summary, f'{output_dir}/malmquist_analysis.png')
    
    # 保存结果
    print("\n" + "=" * 60)
    print("保存结果文件")
    print("=" * 60)
    
    dematel_results.to_csv(f'{output_dir}/dematel_weights.csv', index=False, encoding='utf-8-sig')
    print(f"DEMATEL 权重已保存: {output_dir}/dematel_weights.csv")
    
    if topsis_data is not None:
        topsis_data.to_csv(f'{output_dir}/topsis_scores.csv', index=False, encoding='utf-8-sig')
        print(f"TOPSIS 得分已保存: {output_dir}/topsis_scores.csv")
    
    if malmquist_results is not None:
        malmquist_results.to_csv(f'{output_dir}/malmquist_index.csv', index=False, encoding='utf-8-sig')
        print(f"Malmquist 指数已保存: {output_dir}/malmquist_index.csv")
    
    # Step 4: 相关性分析
    corr_summary = analyze_indicator_correlations(
        processed_data, indicator_cols, 
        save_path=f'{output_dir}/correlation_heatmap.png'
    )
    
    # 保存相关性矩阵
    corr_summary['correlation_matrix'].to_csv(f'{output_dir}/correlation_matrix.csv', encoding='utf-8-sig')
    print(f"相关性矩阵已保存: {output_dir}/correlation_matrix.csv")
    
    # 检查共线性改善情况
    print("\n" + "=" * 60)
    print("共线性检查 (修正后)")
    print("=" * 60)
    corr_mat = corr_summary['correlation_matrix']
    high_corr_found = False
    for i in range(len(corr_mat.columns)):
        for j in range(i+1, len(corr_mat.columns)):
            r = abs(corr_mat.iloc[i, j])
            if r > 0.98:
                print(f"  警告: {corr_mat.columns[i]} vs {corr_mat.columns[j]}: r={r:.4f}")
                high_corr_found = True
    if not high_corr_found:
        print("  [OK] 未发现 |r|>0.98 的同源指标对，共线性问题已改善!")
    
    # 打印因素相互作用分析报告
    print_factor_interaction_analysis(dematel_results, corr_summary)
    
    # =========================================================================
    # Step 4.1: 稳健性检验 - Monte Carlo 权重扰动
    # =========================================================================
    robustness_mc = robustness_weight_perturbation(
        processed_data, indicator_cols, weights,
        n_simulations=1000,  # 1000次模拟
        perturbation_pct=0.1,  # ±10%扰动
        year=2025,
        output_path=f'{output_dir}/robustness_2025_weight_perturbation.csv'
    )
    
    # =========================================================================
    # Step 4.2: 稳健性检验 - 多权重方法对比
    # =========================================================================
    robustness_methods = robustness_multi_method_comparison(
        processed_data, indicator_cols, weights,
        year=2025,
        output_path=f'{output_dir}/robustness_rank_method_comparison.csv'
    )
    
    # =========================================================================
    # Step 5: 2025年贡献分解与短板诊断
    # =========================================================================
    contribution_analysis, contribution_df = calculate_contribution_breakdown(
        processed_data, indicator_cols, weights,
        year=2025,
        output_path=f'{output_dir}/contribution_breakdown_2025.csv'
    )
    
    # 绘制贡献分析图
    plot_contribution_radar(
        contribution_df, 
        countries_to_plot=['USA', 'CHN', 'DEU', 'GBR', 'JPN'],
        save_path=f'{output_dir}/contribution_2025.png'
    )
    
    # =========================================================================
    # 输出 2025 年排名及稳健性总结
    # =========================================================================
    print("\n" + "=" * 80)
    print("2025 年 AI 竞争力最终排名")
    print("=" * 80)
    if 2025 in topsis_results:
        final_ranking = topsis_results[2025].copy()
        print(f"\n{'排名':<6} {'国家':<10} {'代码':<6} {'TOPSIS得分':<12}")
        print("-" * 40)
        for _, row in final_ranking.iterrows():
            print(f"{row['Rank']:<6} {row['Country_CN']:<10} {row['Country']:<6} {row['TOPSIS_Score']:.4f}")
    
    # =========================================================================
    # 最终总结
    # =========================================================================
    print("\n" + "=" * 80)
    print("结构增强版改进总结")
    print("=" * 80)
    print("""
【A1/E1/E2 新定义说明】
  * A1 (算力基建): 电力*0.5 + 安全服务器*0.3 + 宽带*0.2 的基础设施指数
                   不再使用VC代理，与E2完全解耦
  * E1 (垂直VC): 仅使用 GenAI_VC_Investment，不含 Compute_VC
  * E2 (资本总量): AI_VC_Investment 总投资 (compute VC 仅在此处计入一次)

【稳健性结论】
  见上方 Monte Carlo 分析输出，统计了中美第1/第2频率
  
【贡献分解说明】
  见上方各国 Top3 优势指标 / Bottom3 短板指标输出
  德国第三名主要贡献来源已在贡献分解部分详述
""")
    
    print("\n" + "=" * 80)
    print("问题一求解完成！(结构增强版)")
    print("=" * 80)
    
    return {
        'processed_data': processed_data,
        'dematel_results': dematel_results,
        'weights': weights,
        'topsis_results': topsis_results,
        'malmquist_results': malmquist_results,
        'malmquist_summary': malmquist_summary,
        'correlation_summary': corr_summary,
        'robustness_mc': robustness_mc,
        'robustness_methods': robustness_methods,
        'contribution_analysis': contribution_analysis
    }


if __name__ == '__main__':
    results = main()
