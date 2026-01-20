# -*- coding: utf-8 -*-
"""
华数杯 B题 - R&D与创新基础数据预处理脚本
============================================
针对 Research and development investment and innovation foundation 文件夹
的数据进行预处理，生成可与AI主表合并的标准化数据

数据特点：
- R&D支出占比、研究人员密度、专利申请、高等教育指标
- 来源：UNESCO UIS、World Bank
- 主要为比例指标和存量指标（无需大规模对数变换和通胀调整）

输出：
- rd_innovation_preprocessed.csv: 预处理后的宽表
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

BASE_DIR = Path(r"d:\华数杯\Research and development investment and innovation foundation")
MERGED_DATA_DIR = BASE_DIR / "merged_data"
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

# 目标年份（与AI数据对齐）
TARGET_YEARS = list(range(2016, 2026))

# 指标分类及处理策略
INDICATOR_CONFIG = {
    # 比例指标（百分比），无需对数变换
    'rd_expenditure_pct_gdp': {
        'type': 'ratio', 'unit': '%', 'log_transform': False,
        'description': 'R&D支出占GDP比例'
    },
    'rd_expenditure_pct_gdp_wb': {
        'type': 'ratio', 'unit': '%', 'log_transform': False,
        'description': 'R&D支出占GDP比例(World Bank)'
    },
    'higher_edu_enrollment_rate': {
        'type': 'ratio', 'unit': '%', 'log_transform': False,
        'description': '高等教育毛入学率'
    },
    'bachelor_degree_pct': {
        'type': 'ratio', 'unit': '%', 'log_transform': False,
        'description': '学士学位人口比例'
    },
    'master_degree_pct': {
        'type': 'ratio', 'unit': '%', 'log_transform': False,
        'description': '硕士学位人口比例'
    },
    'phd_degree_pct': {
        'type': 'ratio', 'unit': '%', 'log_transform': False,
        'description': '博士学位人口比例'
    },
    'internet_users_pct': {
        'type': 'ratio', 'unit': '%', 'log_transform': False,
        'description': '互联网用户比例'
    },
    'ict_service_exports_pct': {
        'type': 'ratio', 'unit': '%', 'log_transform': False,
        'description': 'ICT服务出口占比'
    },
    'high_tech_exports_pct': {
        'type': 'ratio', 'unit': '%', 'log_transform': False,
        'description': '高科技出口占制成品出口比例'
    },
    'govt_edu_expenditure_pct_gdp': {
        'type': 'ratio', 'unit': '%', 'log_transform': False,
        'description': '政府教育支出占GDP比例'
    },
    'higher_edu_expenditure_pct': {
        'type': 'ratio', 'unit': '%', 'log_transform': False,
        'description': '高等教育支出占教育支出比例'
    },
    'labor_force_higher_edu_pct': {
        'type': 'ratio', 'unit': '%', 'log_transform': False,
        'description': '高等教育劳动力比例'
    },
    
    # 密度指标（每百万人/每百人）
    'researchers_per_million': {
        'type': 'density', 'unit': 'per million', 'log_transform': False,
        'description': '每百万人研究人员数'
    },
    'researchers_per_million_wb': {
        'type': 'density', 'unit': 'per million', 'log_transform': False,
        'description': '每百万人研究人员数(World Bank)'
    },
    'RESEARCHERS_PER_MILLION': {
        'type': 'density', 'unit': 'per million', 'log_transform': False,
        'description': '每百万人研究人员数(UIS)'
    },
    'fixed_broadband_per_100': {
        'type': 'density', 'unit': 'per 100', 'log_transform': False,
        'description': '每百人固定宽带订阅数'
    },
    'mobile_subscriptions_per_100': {
        'type': 'density', 'unit': 'per 100', 'log_transform': False,
        'description': '每百人移动电话订阅数'
    },
    'secure_internet_servers_per_million': {
        'type': 'density', 'unit': 'per million', 'log_transform': False,
        'description': '每百万人安全互联网服务器数'
    },
    
    # 绝对数量指标（可能需要对数变换）
    'patent_applications_resident': {
        'type': 'count', 'unit': 'count', 'log_transform': True,
        'description': '居民专利申请数'
    },
    'patent_applications_nonresident': {
        'type': 'count', 'unit': 'count', 'log_transform': True,
        'description': '非居民专利申请数'
    },
    'high_tech_exports_usd': {
        'type': 'monetary', 'unit': 'USD', 'log_transform': True,
        'description': '高科技产品出口（美元）'
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
            # R&D比例数据通常平稳增长，使用加法趋势
            model = ExponentialSmoothing(
                clean_series.values,
                trend='add',
                seasonal=None,
                damped_trend=True
            )
            fitted = model.fit(optimized=True)
            forecast = fitted.forecast(periods)
            
            # 对于比例指标，确保在合理范围内
            return np.clip(forecast, 0, 100)
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
        return country_data
    
    valid_mask = country_data[value_col].notna()
    if valid_mask.sum() < 4:
        return country_data
    
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
                    interpolated_value = cs(year)
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
    
    if check_skewness and abs(skewness) > 2:
        transformed = np.log1p(series.clip(lower=0))
        return transformed, True, skewness
    
    return series, False, skewness


# ============================================================================
# 主预处理类
# ============================================================================

class RDDataPreprocessor:
    """R&D与创新基础数据预处理器"""
    
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
    
    def load_merged_data(self) -> pd.DataFrame:
        """加载合并后的宽表数据"""
        print("\n" + "=" * 80)
        print("📂 1. 加载数据")
        print("=" * 80)
        
        merged_file = MERGED_DATA_DIR / "rd_innovation_wide.csv"
        if not merged_file.exists():
            raise FileNotFoundError(f"找不到合并数据文件: {merged_file}")
        
        df = pd.read_csv(merged_file)
        self.log(f"已加载 {len(df)} 行, {len(df.columns)} 列")
        
        # 筛选目标国家
        df = df[df['country_code'].isin(TARGET_COUNTRIES)]
        self.log(f"筛选目标国家后: {len(df)} 行")
        
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
        # 获取指标列
        indicator_cols = [c for c in source_df.columns 
                         if c not in ['country_code', 'year', 'country_cn', 'country_en']]
        
        # 筛选目标年份
        source_df = source_df[source_df['year'].isin(TARGET_YEARS)]
        
        # 合并
        result = framework.merge(
            source_df[['country_code', 'year'] + indicator_cols],
            on=['country_code', 'year'],
            how='left'
        )
        
        self.log(f"已合并 {len(indicator_cols)} 个指标列")
        
        return result
    
    def interpolate_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """插值处理中间缺失值"""
        print("\n" + "=" * 80)
        print("🔧 3. 缺失值插值（三次样条）")
        print("=" * 80)
        
        indicator_cols = [c for c in df.columns 
                         if c not in ['country_code', 'year', 'country_cn']]
        
        interpolated_count = 0
        
        for col in indicator_cols:
            for country in TARGET_COUNTRIES:
                country_mask = df['country_code'] == country
                country_data = df[country_mask].sort_values('year')
                
                # 检查是否有中间缺失（两端有值，中间缺失）
                values = country_data[col].values
                valid_indices = np.where(~np.isnan(values.astype(float)))[0]
                
                if len(valid_indices) < 4:
                    continue
                
                # 检测中间缺失
                first_valid = valid_indices[0]
                last_valid = valid_indices[-1]
                
                for i in range(first_valid + 1, last_valid):
                    if np.isnan(float(values[i])):
                        # 有中间缺失，进行插值
                        df = cubic_spline_interpolate(df, country, 'year', col)
                        interpolated_count += 1
                        break
        
        self.log(f"完成 {interpolated_count} 次三次样条插值")
        
        return df
    
    def extrapolate_tail_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """外推尾部缺失值（2024-2025）"""
        print("\n" + "=" * 80)
        print("📈 4. 尾部缺失外推（Holt-Winters）")
        print("=" * 80)
        
        indicator_cols = [c for c in df.columns 
                         if c not in ['country_code', 'year', 'country_cn']]
        
        extrapolated_count = 0
        
        for col in indicator_cols:
            for country in TARGET_COUNTRIES:
                country_mask = df['country_code'] == country
                country_data = df[country_mask].sort_values('year')
                
                # 检查2024-2025是否缺失
                val_2024 = country_data[country_data['year'] == 2024][col].values
                val_2025 = country_data[country_data['year'] == 2025][col].values
                
                missing_years = []
                if len(val_2024) == 0 or pd.isna(val_2024[0]):
                    missing_years.append(2024)
                if len(val_2025) == 0 or pd.isna(val_2025[0]):
                    missing_years.append(2025)
                
                if not missing_years:
                    continue
                
                # 获取历史数据进行外推
                historical = country_data[country_data['year'] < min(missing_years)][col].dropna()
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
                self.log(f"{col}: 偏度={skewness:.2f}, 无需对数变换")
        
        return df
    
    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加衍生特征"""
        print("\n" + "=" * 80)
        print("🔨 6. 构建衍生特征")
        print("=" * 80)
        
        # 按国家排序以计算滞后和增长率
        df = df.sort_values(['country_code', 'year'])
        
        # 1. 年增长率
        growth_cols = ['rd_expenditure_pct_gdp', 'researchers_per_million', 'higher_edu_enrollment_rate']
        for col in growth_cols:
            if col in df.columns:
                growth_col = f'{col}_YoY_Growth'
                df[growth_col] = df.groupby('country_code')[col].pct_change() * 100
                self.log(f"已计算 {col} 年增长率")
        
        # 2. 3年移动平均（平滑波动）
        ma_cols = ['rd_expenditure_pct_gdp', 'researchers_per_million']
        for col in ma_cols:
            if col in df.columns:
                ma_col = f'{col}_MA3'
                df[ma_col] = df.groupby('country_code')[col].transform(
                    lambda x: x.rolling(window=3, min_periods=2).mean()
                )
                self.log(f"已计算 {col} 3年移动平均")
        
        # 3. 滞后特征（用于与AI产出的因果分析）
        lag_cols = ['rd_expenditure_pct_gdp', 'researchers_per_million', 
                    'higher_edu_enrollment_rate', 'patent_applications_resident']
        for col in lag_cols:
            if col in df.columns:
                for lag in [1, 2, 3]:
                    lag_col = f'{col}_lag{lag}'
                    df[lag_col] = df.groupby('country_code')[col].shift(lag)
                self.log(f"已为 {col} 添加1-3年滞后特征")
        
        # 4. 综合创新指数（可选）
        # 标准化后加权平均
        innovation_components = ['rd_expenditure_pct_gdp', 'researchers_per_million', 
                                 'higher_edu_enrollment_rate', 'internet_users_pct']
        available_components = [c for c in innovation_components if c in df.columns]
        
        if len(available_components) >= 3:
            # Min-Max标准化
            for col in available_components:
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df[f'{col}_normalized'] = (df[col] - min_val) / (max_val - min_val)
            
            # 加权平均
            norm_cols = [f'{c}_normalized' for c in available_components]
            df['Innovation_Foundation_Index'] = df[norm_cols].mean(axis=1)
            self.log(f"已计算创新基础综合指数（基于{len(available_components)}个指标）")
            
            # 删除中间标准化列
            df = df.drop(columns=norm_cols)
        
        # 5. 专利强度（居民/非居民比）
        if 'patent_applications_resident' in df.columns and 'patent_applications_nonresident' in df.columns:
            df['patent_intensity_ratio'] = (
                df['patent_applications_resident'] / 
                df['patent_applications_nonresident'].replace(0, np.nan)
            )
            self.log("已计算专利强度比（居民/非居民）")
        
        return df
    
    def handle_structural_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理结构性缺失（特别是新兴国家）"""
        print("\n" + "=" * 80)
        print("🌍 7. 结构性缺失处理")
        print("=" * 80)
        
        # 检查阿联酋和印度的覆盖情况
        for country in ['ARE', 'IND']:
            country_data = df[df['country_code'] == country]
            indicator_cols = [c for c in df.columns 
                             if c not in ['country_code', 'year', 'country_cn']]
            
            missing_counts = {}
            for col in indicator_cols:
                missing = country_data[col].isna().sum()
                if missing > 0:
                    missing_counts[col] = missing
            
            if missing_counts:
                self.log(f"{country} ({COUNTRY_CN[country]}): {len(missing_counts)} 个指标有缺失")
                # 这里可以添加更复杂的处理逻辑
                # 例如使用相似国家数据估算
        
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
        indicator_cols = [c for c in df.columns 
                         if c not in ['country_code', 'year', 'country_cn']]
        
        for col in indicator_cols:
            missing_pct = df[col].isna().sum() / len(df) * 100
            validation['missing_summary'][col] = missing_pct
        
        # 打印验证结果
        print(f"   总行数: {validation['total_rows']}")
        print(f"   总列数: {validation['total_columns']}")
        print(f"   国家数: {validation['countries']}")
        print(f"   年份范围: {validation['year_range'][0]}-{validation['year_range'][1]}")
        
        # 检查高缺失率指标
        high_missing = {k: v for k, v in validation['missing_summary'].items() if v > 30}
        if high_missing:
            print(f"\n   ⚠️ 高缺失率指标 (>30%):")
            for col, pct in sorted(high_missing.items(), key=lambda x: -x[1])[:5]:
                print(f"      • {col}: {pct:.1f}%")
        
        return validation
    
    def save_output(self, df: pd.DataFrame, validation: dict):
        """保存预处理结果"""
        print("\n" + "=" * 80)
        print("💾 9. 保存输出")
        print("=" * 80)
        
        # 保存CSV
        csv_path = OUTPUT_DIR / "rd_innovation_preprocessed.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        self.log(f"CSV已保存: {csv_path}")
        
        # 保存Excel
        excel_path = OUTPUT_DIR / "rd_innovation_preprocessed.xlsx"
        df.to_excel(excel_path, index=False)
        self.log(f"Excel已保存: {excel_path}")
        
        # 保存列说明
        column_desc = {
            'country_code': '国家代码（ISO 3166-1 alpha-3）',
            'year': '年份',
            'country_cn': '国家中文名',
            'rd_expenditure_pct_gdp': 'R&D支出占GDP比例(%)',
            'researchers_per_million': '每百万人研究人员数',
            'higher_edu_enrollment_rate': '高等教育毛入学率(%)',
            'patent_applications_resident': '居民专利申请数',
            'patent_applications_nonresident': '非居民专利申请数',
            'high_tech_exports_pct': '高科技出口占制成品出口比例(%)',
            'internet_users_pct': '互联网用户比例(%)',
            '*_YoY_Growth': '年同比增长率(%)',
            '*_MA3': '3年移动平均',
            '*_lag1/2/3': '滞后1/2/3年特征',
            '*_log': '对数变换值(log1p)',
            'Innovation_Foundation_Index': '创新基础综合指数(0-1标准化)'
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
        # 转换numpy类型
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
        print("🔬 华数杯 B题 - R&D与创新基础数据预处理")
        print("=" * 100)
        print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 1. 加载数据
        source_df = self.load_merged_data()
        
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
        
        # 8. 处理结构性缺失
        df = self.handle_structural_missing(df)
        
        # 9. 验证
        validation = self.validate_output(df)
        
        # 10. 保存
        self.save_output(df, validation)
        
        self.master_df = df
        
        print("\n" + "=" * 100)
        print("✅ R&D与创新基础数据预处理完成!")
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
    preprocessor = RDDataPreprocessor()
    df = preprocessor.run()
    
    # 显示数据预览
    print("\n📋 输出数据预览（前5行）:")
    print(df.head().to_string())


if __name__ == "__main__":
    main()
