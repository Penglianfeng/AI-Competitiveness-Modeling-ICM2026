# -*- coding: utf-8 -*-
"""
华数杯 B题 - AI人才数据预处理脚本
============================================
针对 Supply, Mobility and Quality of AI Talents 文件夹
的数据进行预处理，生成可与AI主表合并的标准化数据

数据特点：
- AI人才供给：研究人员密度、技术人员密度
- 人才培养：高等教育入学率、STEM毕业生比例、学位完成率
- 教育投入：教育支出占GDP比例、高等教育生均支出
- 人口基础：总人口、劳动年龄人口占比
- 来源：World Bank、UNESCO UIS
- 主要为比例指标和密度指标（无需大规模对数变换）

输出：
- ai_talent_preprocessed.csv: 预处理后的宽表
- 可直接与主AI数据表按Country+Year合并

作者: 华数杯参赛队
日期: 2026-01-17
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.interpolate import CubicSpline
from datetime import datetime
import warnings
import json

warnings.filterwarnings('ignore')

# ============================================================================
# 全局配置
# ============================================================================

BASE_DIR = Path(r"d:\华数杯\Supply, Mobility and Quality of AI Talents")
MERGED_DATA_DIR = BASE_DIR / "merged_wide"
OUTPUT_DIR = BASE_DIR / "preprocessed"
OUTPUT_DIR.mkdir(exist_ok=True)

# 目标国家（与主数据集一致）
TARGET_COUNTRIES = ['USA', 'CHN', 'GBR', 'DEU', 'FRA', 'CAN', 'JPN', 'KOR', 'ARE', 'IND']

# 国家中文名
COUNTRY_CN = {
    'USA': '美国', 'CHN': '中国', 'GBR': '英国', 'DEU': '德国',
    'FRA': '法国', 'CAN': '加拿大', 'JPN': '日本', 'KOR': '韩国',
    'ARE': '阿联酋', 'IND': '印度'
}

# 目标年份（与AI主数据对齐: 2016-2025）
TARGET_YEARS = list(range(2016, 2026))

# 指标配置：类型、单位、是否对数变换、描述
INDICATOR_CONFIG = {
    # 密度型指标
    'researchers_per_million': {
        'type': 'density', 'unit': 'per million', 'log_transform': False,
        'description': '每百万人研究人员数', 'priority': 'high'
    },
    'researchers_per_million_fte': {
        'type': 'density', 'unit': 'per million', 'log_transform': False,
        'description': '每百万人研究人员数(FTE)', 'priority': 'medium'
    },
    'technicians_per_million': {
        'type': 'density', 'unit': 'per million', 'log_transform': False,
        'description': '每百万人研发技术人员数', 'priority': 'medium'
    },
    
    # 比例指标（百分比）
    'tertiary_gross_enrollment_pct': {
        'type': 'ratio', 'unit': '%', 'log_transform': False,
        'description': '高等教育毛入学率', 'priority': 'high'
    },
    'tertiary_female_share_pct': {
        'type': 'ratio', 'unit': '%', 'log_transform': False,
        'description': '高等教育女性占比', 'priority': 'low'
    },
    'education_expenditure_pct_gdp': {
        'type': 'ratio', 'unit': '%', 'log_transform': False,
        'description': '教育支出占GDP比例', 'priority': 'high'
    },
    'tertiary_spend_per_student_pct_gdp_pc': {
        'type': 'ratio', 'unit': '%', 'log_transform': False,
        'description': '高等教育生均支出占人均GDP比例', 'priority': 'medium'
    },
    'rd_expenditure_pct_gdp': {
        'type': 'ratio', 'unit': '%', 'log_transform': False,
        'description': 'R&D支出占GDP比例', 'priority': 'high'
    },
    'pop_15_64_pct': {
        'type': 'ratio', 'unit': '%', 'log_transform': False,
        'description': '15-64岁劳动年龄人口占比', 'priority': 'high'
    },
    'stem_graduates_pct': {
        'type': 'ratio', 'unit': '%', 'log_transform': False,
        'description': 'STEM毕业生占比', 'priority': 'high'
    },
    'tertiary_completion_25_34_pct': {
        'type': 'ratio', 'unit': '%', 'log_transform': False,
        'description': '25-34岁高等教育完成率', 'priority': 'medium'
    },
    
    # 绝对数量指标（需要对数变换）
    'population_total': {
        'type': 'count', 'unit': 'count', 'log_transform': True,
        'description': '总人口', 'priority': 'high'
    },
    'tertiary_enrollment_total': {
        'type': 'count', 'unit': 'count', 'log_transform': True,
        'description': '高等教育在校生总数', 'priority': 'medium'
    }
}


# ============================================================================
# 工具函数
# ============================================================================

def holt_winters_forecast(series: pd.Series, periods: int = 2) -> np.ndarray:
    """
    Holt-Winters指数平滑外推
    用于尾部缺失（如预测2024-2025年数据）
    """
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        
        clean_series = series.dropna()
        if len(clean_series) < 4:
            return linear_extrapolate(clean_series, periods)
        
        try:
            # 人才数据通常平稳增长，使用加法趋势+阻尼
            model = ExponentialSmoothing(
                clean_series.values,
                trend='add',
                seasonal=None,
                damped_trend=True
            )
            fitted = model.fit(optimized=True)
            forecast = fitted.forecast(periods)
            
            # 对于比例指标，确保在合理范围内
            return np.clip(forecast, 0, 200)  # 入学率可能超过100%
        except:
            return linear_extrapolate(clean_series, periods)
    except ImportError:
        return linear_extrapolate(series.dropna(), periods)


def linear_extrapolate(series: pd.Series, periods: int) -> np.ndarray:
    """简单线性外推"""
    if len(series) < 2:
        return np.array([series.iloc[-1]] * periods) if len(series) > 0 else np.array([np.nan] * periods)
    
    x = np.arange(len(series))
    y = series.values
    slope, intercept, _, _, _ = stats.linregress(x, y)
    
    future_x = np.arange(len(series), len(series) + periods)
    forecast = slope * future_x + intercept
    return forecast


def cubic_spline_interpolate(df: pd.DataFrame, country: str, 
                             year_col: str, value_col: str) -> pd.DataFrame:
    """
    三次样条插值填补中间缺失值
    """
    country_data = df[df['country_code'] == country].copy()
    country_data = country_data.sort_values(year_col)
    
    if len(country_data) < 4:
        return df
    
    valid_mask = country_data[value_col].notna()
    if valid_mask.sum() < 4:
        return df
    
    years_valid = country_data.loc[valid_mask, year_col].values
    values_valid = country_data.loc[valid_mask, value_col].values
    
    try:
        cs = CubicSpline(years_valid, values_valid)
        
        missing_mask = country_data[value_col].isna()
        if missing_mask.any():
            missing_years = country_data.loc[missing_mask, year_col].values
            # 只插值范围内的年份
            for year in missing_years:
                if years_valid.min() <= year <= years_valid.max():
                    idx = country_data[country_data[year_col] == year].index[0]
                    interpolated_value = float(cs(year))
                    # 确保插值结果合理
                    if value_col.endswith('_pct') and interpolated_value < 0:
                        interpolated_value = 0
                    df.loc[idx, value_col] = interpolated_value
    except Exception as e:
        pass
    
    return df


def log_transform_column(series: pd.Series, check_skewness: bool = True) -> tuple:
    """
    对数变换（log1p）
    返回: (变换后的series, 是否变换, 原始偏度)
    """
    clean = series.dropna()
    if len(clean) < 10:
        return series, False, np.nan
    
    skewness = clean.skew()
    
    if check_skewness and abs(skewness) > 1.5:
        transformed = np.log1p(series.clip(lower=0))
        return transformed, True, skewness
    
    return series, False, skewness


# ============================================================================
# 主预处理类
# ============================================================================

class AITalentPreprocessor:
    """AI人才数据预处理器"""
    
    def __init__(self):
        self.processing_log = []
        self.master_df = None
        
    def log(self, message: str):
        """记录处理日志"""
        self.processing_log.append({
            'timestamp': datetime.now().isoformat(),
            'message': message
        })
        print(f"  📝 {message}")
    
    def load_source_data(self) -> pd.DataFrame:
        """加载源数据"""
        print("\n" + "=" * 80)
        print("📂 1. 加载数据")
        print("=" * 80)
        
        source_file = MERGED_DATA_DIR / "ai_talent_wide.csv"
        if not source_file.exists():
            raise FileNotFoundError(f"找不到源数据文件: {source_file}")
        
        df = pd.read_csv(source_file)
        self.log(f"已加载 {len(df)} 行, {len(df.columns)} 列")
        
        # 筛选目标国家
        df = df[df['country_code'].isin(TARGET_COUNTRIES)]
        self.log(f"筛选目标国家后: {len(df)} 行")
        
        # 显示年份范围
        self.log(f"原始年份范围: {df['year'].min()}-{df['year'].max()}")
        
        return df
    
    def create_base_framework(self) -> pd.DataFrame:
        """创建主表基础框架（确保所有国家-年份组合都存在）"""
        print("\n" + "=" * 80)
        print("🏗️ 2. 创建基础框架")
        print("=" * 80)
        
        rows = []
        for country in TARGET_COUNTRIES:
            for year in TARGET_YEARS:
                rows.append({
                    'country_code': country,
                    'year': year,
                    'country_cn': COUNTRY_CN.get(country, country)
                })
        
        framework = pd.DataFrame(rows)
        self.log(f"创建 {len(framework)} 行基础框架 ({len(TARGET_COUNTRIES)}国 × {len(TARGET_YEARS)}年)")
        
        return framework
    
    def merge_with_framework(self, framework: pd.DataFrame, 
                             source_df: pd.DataFrame) -> pd.DataFrame:
        """将源数据合并到基础框架"""
        # 获取指标列（排除元数据列）
        meta_cols = ['country_code', 'year', 'country_cn', 'country_en']
        indicator_cols = [c for c in source_df.columns if c not in meta_cols]
        
        # 筛选目标年份
        source_df = source_df[source_df['year'].isin(TARGET_YEARS)]
        
        # 合并
        merge_cols = ['country_code', 'year'] + indicator_cols
        available_cols = [c for c in merge_cols if c in source_df.columns]
        
        result = framework.merge(
            source_df[available_cols],
            on=['country_code', 'year'],
            how='left'
        )
        
        self.log(f"已合并 {len(indicator_cols)} 个指标列")
        
        return result
    
    def interpolate_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """插值处理中间缺失值（三次样条）"""
        print("\n" + "=" * 80)
        print("🔧 3. 缺失值插值（三次样条）")
        print("=" * 80)
        
        meta_cols = ['country_code', 'year', 'country_cn']
        indicator_cols = [c for c in df.columns if c not in meta_cols]
        
        interpolated_count = 0
        
        for col in indicator_cols:
            for country in TARGET_COUNTRIES:
                country_mask = df['country_code'] == country
                country_data = df[country_mask].sort_values('year')
                
                # 检查是否有中间缺失（两端有值，中间缺失）
                values = country_data[col].values
                try:
                    valid_indices = np.where(~pd.isna(values))[0]
                except:
                    continue
                
                if len(valid_indices) < 4:
                    continue
                
                # 检测中间缺失
                first_valid = valid_indices[0]
                last_valid = valid_indices[-1]
                
                has_middle_missing = False
                for i in range(first_valid + 1, last_valid):
                    if pd.isna(values[i]):
                        has_middle_missing = True
                        break
                
                if has_middle_missing:
                    df = cubic_spline_interpolate(df, country, 'year', col)
                    interpolated_count += 1
        
        self.log(f"完成 {interpolated_count} 次三次样条插值")
        
        return df
    
    def extrapolate_tail_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """外推尾部缺失值（2024-2025）"""
        print("\n" + "=" * 80)
        print("📈 4. 尾部缺失外推（Holt-Winters）")
        print("=" * 80)
        
        meta_cols = ['country_code', 'year', 'country_cn']
        indicator_cols = [c for c in df.columns if c not in meta_cols]
        
        extrapolated_count = 0
        
        for col in indicator_cols:
            for country in TARGET_COUNTRIES:
                country_mask = df['country_code'] == country
                country_data = df[country_mask].sort_values('year')
                
                # 检查哪些年份需要外推
                missing_years = []
                for year in [2023, 2024, 2025]:
                    year_data = country_data[country_data['year'] == year]
                    if len(year_data) == 0 or pd.isna(year_data[col].values[0]):
                        missing_years.append(year)
                
                if not missing_years:
                    continue
                
                # 获取历史数据进行外推
                min_missing_year = min(missing_years)
                historical = country_data[country_data['year'] < min_missing_year][col].dropna()
                if len(historical) < 3:
                    continue
                
                # 外推
                periods = len(missing_years)
                try:
                    forecast = holt_winters_forecast(historical, periods)
                    
                    for i, year in enumerate(missing_years):
                        idx = df[(df['country_code'] == country) & (df['year'] == year)].index
                        if len(idx) > 0:
                            df.loc[idx[0], col] = forecast[i]
                            extrapolated_count += 1
                except Exception as e:
                    continue
        
        self.log(f"完成 {extrapolated_count} 个值的Holt-Winters外推")
        
        return df
    
    def apply_log_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """对绝对数量指标应用对数变换"""
        print("\n" + "=" * 80)
        print("📊 5. 对数变换（仅绝对数量指标）")
        print("=" * 80)
        
        transform_cols = [col for col, config in INDICATOR_CONFIG.items() 
                         if config.get('log_transform', False) and col in df.columns]
        
        for col in transform_cols:
            transformed, did_transform, skewness = log_transform_column(df[col])
            if did_transform:
                df[f'{col}_log'] = transformed
                self.log(f"{col}: 偏度={skewness:.2f}, 已添加对数变换列")
            else:
                # 对于人口等，即使偏度不高也做对数变换
                if col in ['population_total', 'tertiary_enrollment_total']:
                    df[f'{col}_log'] = np.log1p(df[col].clip(lower=0))
                    self.log(f"{col}: 强制对数变换（绝对数量指标）")
        
        return df
    
    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加衍生特征"""
        print("\n" + "=" * 80)
        print("🔨 6. 构建衍生特征")
        print("=" * 80)
        
        # 按国家排序以计算滞后和增长率
        df = df.sort_values(['country_code', 'year'])
        
        # 1. 年增长率
        growth_cols = ['researchers_per_million', 'tertiary_gross_enrollment_pct', 
                       'rd_expenditure_pct_gdp', 'education_expenditure_pct_gdp']
        for col in growth_cols:
            if col in df.columns:
                growth_col = f'{col}_YoY_Growth'
                df[growth_col] = df.groupby('country_code')[col].pct_change() * 100
                self.log(f"已计算 {col} 年增长率")
        
        # 2. 3年移动平均（平滑波动）
        ma_cols = ['researchers_per_million', 'tertiary_gross_enrollment_pct']
        for col in ma_cols:
            if col in df.columns:
                ma_col = f'{col}_MA3'
                df[ma_col] = df.groupby('country_code')[col].transform(
                    lambda x: x.rolling(window=3, min_periods=2).mean()
                )
                self.log(f"已计算 {col} 3年移动平均")
        
        # 3. 滞后特征（用于与AI产出的因果分析）
        lag_cols = ['researchers_per_million', 'tertiary_gross_enrollment_pct', 
                    'education_expenditure_pct_gdp', 'stem_graduates_pct']
        for col in lag_cols:
            if col in df.columns:
                for lag in [1, 2, 3]:
                    lag_col = f'{col}_lag{lag}'
                    df[lag_col] = df.groupby('country_code')[col].shift(lag)
                self.log(f"已为 {col} 添加1-3年滞后特征")
        
        # 4. 人才综合指数
        talent_components = ['researchers_per_million', 'tertiary_gross_enrollment_pct',
                            'rd_expenditure_pct_gdp', 'education_expenditure_pct_gdp']
        available_components = [c for c in talent_components if c in df.columns]
        
        if len(available_components) >= 3:
            # Min-Max标准化
            for col in available_components:
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df[f'{col}_normalized'] = (df[col] - min_val) / (max_val - min_val)
            
            # 加权平均
            norm_cols = [f'{c}_normalized' for c in available_components]
            df['AI_Talent_Index'] = df[norm_cols].mean(axis=1)
            self.log(f"已计算AI人才综合指数（基于{len(available_components)}个指标）")
            
            # 删除中间标准化列
            df = df.drop(columns=norm_cols, errors='ignore')
        
        # 5. 研究人员增长动力指标（研究人员增速 vs 教育投入）
        if 'researchers_per_million' in df.columns and 'education_expenditure_pct_gdp' in df.columns:
            df['researcher_growth_efficiency'] = (
                df['researchers_per_million_YoY_Growth'] / 
                df['education_expenditure_pct_gdp'].replace(0, np.nan)
            )
            self.log("已计算研究人员增长效率指标")
        
        # 6. 人才密度（研究人员/劳动年龄人口）
        if 'researchers_per_million' in df.columns and 'pop_15_64_pct' in df.columns:
            df['researcher_density_adjusted'] = (
                df['researchers_per_million'] * df['pop_15_64_pct'] / 100
            )
            self.log("已计算劳动人口调整后的研究人员密度")
        
        return df
    
    def handle_sparse_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理稀疏指标"""
        print("\n" + "=" * 80)
        print("🔍 7. 稀疏指标处理")
        print("=" * 80)
        
        meta_cols = ['country_code', 'year', 'country_cn']
        indicator_cols = [c for c in df.columns if c not in meta_cols 
                         and not c.endswith('_normalized') 
                         and not c.endswith('_YoY_Growth')
                         and not c.endswith('_MA3')
                         and not c.endswith('_lag1')
                         and not c.endswith('_lag2')
                         and not c.endswith('_lag3')
                         and not c.endswith('_log')]
        
        sparse_indicators = []
        for col in indicator_cols:
            coverage = df[col].notna().sum() / len(df) * 100
            if coverage < 20:
                sparse_indicators.append((col, coverage))
        
        if sparse_indicators:
            self.log(f"发现 {len(sparse_indicators)} 个稀疏指标 (覆盖率<20%):")
            for col, coverage in sparse_indicators:
                self.log(f"   • {col}: {coverage:.1f}% 覆盖率")
            
            # 不删除，但标记
            self.log("   ℹ️ 稀疏指标已保留，建模时请考虑降权或剔除")
        else:
            self.log("无稀疏指标")
        
        return df
    
    def validate_output(self, df: pd.DataFrame) -> dict:
        """验证输出数据质量"""
        print("\n" + "=" * 80)
        print("✅ 8. 输出验证")
        print("=" * 80)
        
        validation = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'countries': df['country_code'].nunique(),
            'years': df['year'].nunique(),
            'year_range': (int(df['year'].min()), int(df['year'].max())),
            'missing_summary': {}
        }
        
        # 检查每个指标的缺失率
        meta_cols = ['country_code', 'year', 'country_cn']
        indicator_cols = [c for c in df.columns if c not in meta_cols]
        
        for col in indicator_cols:
            missing_pct = df[col].isna().sum() / len(df) * 100
            validation['missing_summary'][col] = missing_pct
        
        # 打印验证结果
        print(f"   总行数: {validation['total_rows']}")
        print(f"   总列数: {validation['total_columns']}")
        print(f"   国家数: {validation['countries']}")
        print(f"   年份范围: {validation['year_range'][0]}-{validation['year_range'][1]}")
        
        # 检查核心指标覆盖
        core_indicators = ['researchers_per_million', 'tertiary_gross_enrollment_pct',
                          'rd_expenditure_pct_gdp', 'pop_15_64_pct']
        print(f"\n   核心指标覆盖率:")
        for col in core_indicators:
            if col in validation['missing_summary']:
                coverage = 100 - validation['missing_summary'][col]
                status = "✅" if coverage > 70 else "⚠️" if coverage > 40 else "❌"
                print(f"      {status} {col}: {coverage:.1f}%")
        
        # 检查高缺失率指标
        high_missing = {k: v for k, v in validation['missing_summary'].items() if v > 50}
        if high_missing:
            print(f"\n   ⚠️ 高缺失率指标 (>50%):")
            for col, pct in sorted(high_missing.items(), key=lambda x: -x[1])[:5]:
                print(f"      • {col}: {pct:.1f}%")
        
        return validation
    
    def save_output(self, df: pd.DataFrame, validation: dict):
        """保存预处理结果"""
        print("\n" + "=" * 80)
        print("💾 9. 保存输出")
        print("=" * 80)
        
        # 保存CSV
        csv_path = OUTPUT_DIR / "ai_talent_preprocessed.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        self.log(f"CSV已保存: {csv_path}")
        
        # 保存Excel
        excel_path = OUTPUT_DIR / "ai_talent_preprocessed.xlsx"
        df.to_excel(excel_path, index=False)
        self.log(f"Excel已保存: {excel_path}")
        
        # 保存列说明
        column_desc = {
            'country_code': '国家代码（ISO 3166-1 alpha-3）',
            'year': '年份',
            'country_cn': '国家中文名',
            'researchers_per_million': '每百万人研究人员数',
            'researchers_per_million_fte': '每百万人研究人员数(FTE)',
            'technicians_per_million': '每百万人研发技术人员数',
            'tertiary_gross_enrollment_pct': '高等教育毛入学率(%)',
            'education_expenditure_pct_gdp': '教育支出占GDP比例(%)',
            'rd_expenditure_pct_gdp': 'R&D支出占GDP比例(%)',
            'pop_15_64_pct': '15-64岁人口占比(%)',
            'stem_graduates_pct': 'STEM毕业生占比(%)',
            'tertiary_completion_25_34_pct': '25-34岁高等教育完成率(%)',
            'population_total': '总人口',
            'population_total_log': '总人口(对数)',
            'tertiary_enrollment_total': '高等教育在校生总数',
            'tertiary_enrollment_total_log': '高等教育在校生总数(对数)',
            '*_YoY_Growth': '年同比增长率(%)',
            '*_MA3': '3年移动平均',
            '*_lag1/2/3': '滞后1/2/3年特征',
            'AI_Talent_Index': 'AI人才综合指数(0-1标准化)',
            'researcher_density_adjusted': '劳动人口调整后的研究人员密度'
        }
        
        desc_path = OUTPUT_DIR / "column_descriptions.json"
        with open(desc_path, 'w', encoding='utf-8') as f:
            json.dump(column_desc, f, ensure_ascii=False, indent=2)
        self.log(f"列说明已保存: {desc_path}")
        
        # 保存处理日志
        log_path = OUTPUT_DIR / "preprocessing_log.json"
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.processing_log, f, ensure_ascii=False, indent=2)
        self.log(f"处理日志已保存: {log_path}")
        
        # 保存验证报告
        validation_path = OUTPUT_DIR / "validation_report.json"
        
        def convert(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            return obj
        
        validation_converted = {k: convert(v) if not isinstance(v, dict) 
                               else {kk: convert(vv) for kk, vv in v.items()} 
                               for k, v in validation.items()}
        
        with open(validation_path, 'w', encoding='utf-8') as f:
            json.dump(validation_converted, f, ensure_ascii=False, indent=2)
        self.log(f"验证报告已保存: {validation_path}")
    
    def run(self):
        """执行完整预处理流程"""
        print("=" * 100)
        print("🎓 华数杯 B题 - AI人才数据预处理")
        print("=" * 100)
        print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 1. 加载数据
        source_df = self.load_source_data()
        
        # 2. 创建基础框架
        framework = self.create_base_framework()
        
        # 3. 合并数据
        df = self.merge_with_framework(framework, source_df)
        
        # 4. 插值中间缺失
        df = self.interpolate_missing(df)
        
        # 5. 外推尾部缺失
        df = self.extrapolate_tail_missing(df)
        
        # 6. 对数变换
        df = self.apply_log_transform(df)
        
        # 7. 衍生特征
        df = self.add_derived_features(df)
        
        # 8. 处理稀疏指标
        df = self.handle_sparse_indicators(df)
        
        # 9. 验证
        validation = self.validate_output(df)
        
        # 10. 保存
        self.save_output(df, validation)
        
        self.master_df = df
        
        print("\n" + "=" * 100)
        print("✅ AI人才数据预处理完成!")
        print("=" * 100)
        
        # 显示最终统计
        print(f"\n📊 输出统计:")
        print(f"   - 维度: {df.shape[0]} 行 × {df.shape[1]} 列")
        print(f"   - 国家: {', '.join(TARGET_COUNTRIES)}")
        print(f"   - 年份: {TARGET_YEARS[0]}-{TARGET_YEARS[-1]}")
        print(f"   - 输出目录: {OUTPUT_DIR}")
        
        return df


# ============================================================================
# 主程序
# ============================================================================

def main():
    """主函数"""
    preprocessor = AITalentPreprocessor()
    df = preprocessor.run()
    
    # 显示数据预览
    print("\n📋 输出数据预览（前5行，核心列）:")
    core_cols = ['country_code', 'year', 'researchers_per_million', 
                 'tertiary_gross_enrollment_pct', 'rd_expenditure_pct_gdp', 
                 'AI_Talent_Index']
    display_cols = [c for c in core_cols if c in df.columns]
    print(df[display_cols].head().to_string())


if __name__ == "__main__":
    main()
