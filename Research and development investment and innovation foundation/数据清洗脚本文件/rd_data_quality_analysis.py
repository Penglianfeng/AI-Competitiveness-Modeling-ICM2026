# -*- coding: utf-8 -*-
"""
华数杯 B题 - R&D与创新基础数据质量分析脚本
============================================
针对 Research and development investment and innovation foundation 文件夹
的数据进行深度质量分析

数据特点：
- R&D支出、研究人员、专利、高等教育等创新基础指标
- 来源：UNESCO UIS、World Bank
- 主要为比例指标和存量指标（与AI投资等流量指标不同）

作者: 华数杯参赛队
日期: 2026-01-17
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import warnings
import json
warnings.filterwarnings('ignore')

# ===================== 配置 =====================
BASE_DIR = Path(r"d:\华数杯\Research and development investment and innovation foundation")
UIS_DATA_DIR = BASE_DIR / "uis_rd_data"
WB_DATA_DIR = BASE_DIR / "World Bank Data"
MERGED_DATA_DIR = BASE_DIR / "merged_data"
OUTPUT_DIR = BASE_DIR / "preprocessed"
OUTPUT_DIR.mkdir(exist_ok=True)

# 目标国家（与主数据集保持一致）
TARGET_COUNTRIES = {
    'USA': '美国', 'CHN': '中国', 'GBR': '英国', 'DEU': '德国', 
    'FRA': '法国', 'CAN': '加拿大', 'JPN': '日本', 'KOR': '韩国', 
    'ARE': '阿联酋', 'IND': '印度'
}

# 目标年份范围（比AI数据范围更宽，用于趋势分析）
TARGET_YEARS = list(range(2010, 2026))
FOCUS_YEARS = list(range(2016, 2026))  # 重点分析年份（与AI数据对齐）

# 指标分类（用于确定处理方式）
INDICATOR_CATEGORIES = {
    'ratio_pct': [  # 比例/百分比指标，无需对数变换
        'rd_expenditure_pct_gdp',
        'bachelor_degree_pct',
        'master_degree_pct', 
        'phd_degree_pct',
        'internet_users_pct',
        'ict_service_exports_pct',
        'high_tech_exports_pct',
        'govt_edu_expenditure_pct_gdp',
        'higher_edu_expenditure_pct',
        'labor_force_higher_edu_pct'
    ],
    'count_intensive': [  # 密度型存量指标
        'researchers_per_million',
        'fixed_broadband_per_100',
        'mobile_subscriptions_per_100',
        'secure_internet_servers_per_million'
    ],
    'count_absolute': [  # 绝对数量指标，可能需要对数变换
        'patent_applications_resident',
        'patent_applications_nonresident',
        'high_tech_exports_usd'
    ],
    'enrollment': [  # 入学率指标
        'higher_edu_enrollment_rate'
    ]
}


class RDDataQualityAnalyzer:
    """R&D与创新基础数据质量分析器"""
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.recommendations = []
        self.analysis_results = {}
        
    def run_full_analysis(self):
        """执行完整分析"""
        print("=" * 100)
        print("🔬 华数杯 B题 - R&D与创新基础数据质量分析报告")
        print("=" * 100)
        print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"数据目录: {BASE_DIR}")
        print()
        
        # 1. 数据源概览
        self.analyze_data_sources()
        
        # 2. 合并数据集分析
        self.analyze_merged_dataset()
        
        # 3. 时间覆盖度分析
        self.analyze_temporal_coverage()
        
        # 4. 国家覆盖度分析  
        self.analyze_country_coverage()
        
        # 5. 缺失值模式分析
        self.analyze_missing_patterns()
        
        # 6. 数值分布分析
        self.analyze_value_distributions()
        
        # 7. 2024-2025年数据可用性
        self.analyze_recent_data_availability()
        
        # 8. 与主数据集时间对齐分析
        self.analyze_alignment_with_ai_data()
        
        # 9. 生成综合报告
        self.generate_report()
        
        return self.analysis_results
    
    def analyze_data_sources(self):
        """分析数据源结构"""
        print("\n" + "=" * 80)
        print("📂 1. 数据源概览")
        print("=" * 80)
        
        sources = {}
        
        # UIS数据
        if UIS_DATA_DIR.exists():
            raw_files = list((UIS_DATA_DIR / "raw").glob("*.csv")) if (UIS_DATA_DIR / "raw").exists() else []
            processed_files = list((UIS_DATA_DIR / "processed").glob("*.csv")) if (UIS_DATA_DIR / "processed").exists() else []
            sources['UIS (UNESCO Institute for Statistics)'] = {
                'raw_files': [f.name for f in raw_files],
                'processed_files': [f.name for f in processed_files]
            }
            print(f"\n📁 UIS数据:")
            print(f"   - 原始文件: {len(raw_files)} 个")
            for f in raw_files:
                print(f"      • {f.name}")
            print(f"   - 处理后文件: {len(processed_files)} 个")
        
        # World Bank数据
        if WB_DATA_DIR.exists():
            wb_files = list(WB_DATA_DIR.glob("*.csv"))
            sources['World Bank'] = {
                'files': [f.name for f in wb_files]
            }
            print(f"\n📁 World Bank数据:")
            print(f"   - 数据文件: {len(wb_files)} 个")
            for f in wb_files:
                df = pd.read_csv(f, nrows=1)
                print(f"      • {f.name} ({len(df.columns)} 列)")
        
        # 合并数据
        if MERGED_DATA_DIR.exists():
            merged_files = list(MERGED_DATA_DIR.glob("*.csv"))
            sources['Merged Data'] = {
                'files': [f.name for f in merged_files]
            }
            print(f"\n📁 合并数据:")
            for f in merged_files:
                df = pd.read_csv(f)
                print(f"      • {f.name} ({len(df)} 行, {len(df.columns)} 列)")
        
        self.analysis_results['data_sources'] = sources
    
    def analyze_merged_dataset(self):
        """分析合并后的宽表数据集"""
        print("\n" + "=" * 80)
        print("📊 2. 合并数据集分析 (rd_innovation_wide.csv)")
        print("=" * 80)
        
        merged_file = MERGED_DATA_DIR / "rd_innovation_wide.csv"
        if not merged_file.exists():
            print("⚠️ 合并数据集不存在!")
            return
        
        df = pd.read_csv(merged_file)
        
        print(f"\n基本信息:")
        print(f"   - 总行数: {len(df)}")
        print(f"   - 总列数: {len(df.columns)}")
        print(f"   - 国家数: {df['country_code'].nunique()}")
        print(f"   - 年份范围: {df['year'].min()} - {df['year'].max()}")
        
        # 指标列分析
        indicator_cols = [c for c in df.columns if c not in ['country_code', 'year', 'country_cn', 'country_en']]
        print(f"\n📈 指标列 ({len(indicator_cols)} 个):")
        
        for col in indicator_cols:
            non_null = df[col].notna().sum()
            non_null_pct = non_null / len(df) * 100
            if non_null > 0:
                mean_val = df[col].mean()
                std_val = df[col].std()
                print(f"   • {col}: {non_null_pct:.1f}% 非空, 均值={mean_val:.2f}, 标准差={std_val:.2f}")
        
        self.analysis_results['merged_dataset'] = {
            'rows': len(df),
            'columns': len(df.columns),
            'countries': df['country_code'].nunique(),
            'year_range': (int(df['year'].min()), int(df['year'].max())),
            'indicators': indicator_cols
        }
    
    def analyze_temporal_coverage(self):
        """分析时间覆盖度"""
        print("\n" + "=" * 80)
        print("📅 3. 时间覆盖度分析")
        print("=" * 80)
        
        merged_file = MERGED_DATA_DIR / "rd_innovation_wide.csv"
        if not merged_file.exists():
            return
            
        df = pd.read_csv(merged_file)
        
        # 各指标的时间覆盖
        indicator_cols = [c for c in df.columns if c not in ['country_code', 'year', 'country_cn', 'country_en']]
        
        coverage = {}
        print(f"\n各指标时间覆盖:")
        print("-" * 70)
        
        for col in indicator_cols[:15]:  # 显示主要指标
            valid_years = df[df[col].notna()]['year'].unique()
            if len(valid_years) > 0:
                min_year = int(min(valid_years))
                max_year = int(max(valid_years))
                coverage[col] = {
                    'min_year': min_year,
                    'max_year': max_year,
                    'years_count': len(valid_years)
                }
                # 检查2016-2025覆盖
                focus_coverage = len([y for y in valid_years if y in FOCUS_YEARS])
                print(f"   {col[:40]:40s}: {min_year}-{max_year}, 2016-2025覆盖: {focus_coverage}/10")
                
                if max_year < 2024:
                    self.warnings.append(f"⚠️ {col}: 最新数据仅到{max_year}年，缺少近期数据")
        
        self.analysis_results['temporal_coverage'] = coverage
    
    def analyze_country_coverage(self):
        """分析各国数据覆盖情况"""
        print("\n" + "=" * 80)
        print("🌍 4. 各国数据覆盖度分析")
        print("=" * 80)
        
        merged_file = MERGED_DATA_DIR / "rd_innovation_wide.csv"
        if not merged_file.exists():
            return
            
        df = pd.read_csv(merged_file)
        indicator_cols = [c for c in df.columns if c not in ['country_code', 'year', 'country_cn', 'country_en']]
        
        print(f"\n各国指标非空率:")
        print("-" * 70)
        
        country_coverage = {}
        for country in TARGET_COUNTRIES.keys():
            country_data = df[df['country_code'] == country]
            if len(country_data) == 0:
                print(f"   ❌ {country} ({TARGET_COUNTRIES[country]}): 无数据")
                self.issues.append(f"🚨 {country} ({TARGET_COUNTRIES[country]}) 无数据")
                continue
            
            # 计算非空率
            non_null_rates = {}
            for col in indicator_cols:
                non_null = country_data[col].notna().sum()
                non_null_pct = non_null / len(country_data) * 100
                non_null_rates[col] = non_null_pct
            
            avg_rate = np.mean(list(non_null_rates.values()))
            country_coverage[country] = {
                'avg_coverage': avg_rate,
                'years': len(country_data),
                'detail': non_null_rates
            }
            
            status = "✅" if avg_rate > 70 else "⚠️" if avg_rate > 40 else "❌"
            print(f"   {status} {country} ({TARGET_COUNTRIES[country]:4s}): 平均覆盖率 {avg_rate:.1f}%, {len(country_data)}年数据")
            
            if avg_rate < 40:
                self.warnings.append(f"⚠️ {country}: 数据覆盖率仅{avg_rate:.1f}%，需特殊处理")
        
        # 特别关注阿联酋和印度
        print(f"\n🔍 新兴国家数据详情:")
        for country in ['ARE', 'IND']:
            if country in country_coverage:
                detail = country_coverage[country]['detail']
                low_coverage = [k for k, v in detail.items() if v < 30]
                if low_coverage:
                    print(f"   {country} 低覆盖指标 (<30%): {low_coverage[:5]}")
        
        self.analysis_results['country_coverage'] = country_coverage
    
    def analyze_missing_patterns(self):
        """分析缺失值模式"""
        print("\n" + "=" * 80)
        print("🔍 5. 缺失值模式分析")
        print("=" * 80)
        
        merged_file = MERGED_DATA_DIR / "rd_innovation_wide.csv"
        if not merged_file.exists():
            return
            
        df = pd.read_csv(merged_file)
        indicator_cols = [c for c in df.columns if c not in ['country_code', 'year', 'country_cn', 'country_en']]
        
        print(f"\n缺失值统计:")
        print("-" * 70)
        
        missing_summary = []
        for col in indicator_cols:
            missing = df[col].isna().sum()
            missing_pct = missing / len(df) * 100
            if missing > 0:
                missing_summary.append({
                    'indicator': col,
                    'missing_count': missing,
                    'missing_pct': missing_pct
                })
        
        # 按缺失率排序
        missing_summary.sort(key=lambda x: x['missing_pct'], reverse=True)
        
        print(f"{'指标':<45} {'缺失数':>8} {'缺失率':>8}")
        print("-" * 65)
        for item in missing_summary[:15]:
            print(f"{item['indicator'][:44]:<45} {item['missing_count']:>8} {item['missing_pct']:>7.1f}%")
        
        # 分析缺失模式类型
        print(f"\n缺失模式分类:")
        
        # 1. 尾部缺失（最新年份缺失）
        tail_missing = []
        for col in indicator_cols:
            recent_data = df[df['year'] >= 2022][col]
            if recent_data.isna().all():
                tail_missing.append(col)
        if tail_missing:
            print(f"   📍 尾部缺失 (2022+无数据): {len(tail_missing)} 个指标")
            self.recommendations.append(f"💡 {len(tail_missing)}个指标缺少2022年后数据，建议Holt-Winters外推")
        
        # 2. 头部缺失（早期年份缺失）
        head_missing = []
        for col in indicator_cols:
            early_data = df[df['year'] <= 2012][col]
            if early_data.isna().all():
                head_missing.append(col)
        if head_missing:
            print(f"   📍 头部缺失 (2012前无数据): {len(head_missing)} 个指标")
        
        # 3. 随机缺失
        random_missing = [col for col in indicator_cols 
                         if col not in tail_missing and col not in head_missing 
                         and df[col].isna().sum() > 0]
        if random_missing:
            print(f"   📍 随机/结构性缺失: {len(random_missing)} 个指标")
            self.recommendations.append(f"💡 {len(random_missing)}个指标有中间缺失，建议三次样条插值")
        
        self.analysis_results['missing_patterns'] = {
            'tail_missing': tail_missing,
            'head_missing': head_missing,
            'random_missing': random_missing
        }
    
    def analyze_value_distributions(self):
        """分析数值分布特征"""
        print("\n" + "=" * 80)
        print("📈 6. 数值分布分析")
        print("=" * 80)
        
        merged_file = MERGED_DATA_DIR / "rd_innovation_wide.csv"
        if not merged_file.exists():
            return
            
        df = pd.read_csv(merged_file)
        indicator_cols = [c for c in df.columns if c not in ['country_code', 'year', 'country_cn', 'country_en']]
        
        print(f"\n分布特征分析:")
        print("-" * 85)
        print(f"{'指标':<35} {'偏度':>8} {'峰度':>8} {'变异系数':>10} {'建议对数':>10}")
        print("-" * 85)
        
        distribution_analysis = []
        for col in indicator_cols:
            values = df[col].dropna()
            if len(values) < 10:
                continue
            
            skewness = values.skew()
            kurtosis = values.kurtosis()
            cv = values.std() / values.mean() if values.mean() != 0 else np.nan
            
            # 判断是否需要对数变换
            # 对于R&D数据，主要是比例指标，一般不需要对数变换
            # 只有绝对数量指标（如专利数）可能需要
            need_log = False
            if col in INDICATOR_CATEGORIES.get('count_absolute', []):
                if abs(skewness) > 2 or cv > 2:
                    need_log = True
            
            distribution_analysis.append({
                'indicator': col,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'cv': cv,
                'need_log': need_log
            })
            
            log_mark = "✅" if need_log else "❌"
            print(f"{col[:34]:<35} {skewness:>8.2f} {kurtosis:>8.2f} {cv:>10.2f} {log_mark:>10}")
        
        # 统计需要对数变换的指标
        need_log_count = sum(1 for d in distribution_analysis if d['need_log'])
        print(f"\n📝 结论: {need_log_count} 个指标建议对数变换（主要是绝对数量指标）")
        print("   💡 注意: R&D数据多为比例指标，与AI投资数据不同，大部分不需要对数变换")
        
        self.analysis_results['distribution_analysis'] = distribution_analysis
    
    def analyze_recent_data_availability(self):
        """分析2024-2025年数据可用性"""
        print("\n" + "=" * 80)
        print("🎯 7. 2024-2025年数据可用性分析（关键！）")
        print("=" * 80)
        
        merged_file = MERGED_DATA_DIR / "rd_innovation_wide.csv"
        if not merged_file.exists():
            return
            
        df = pd.read_csv(merged_file)
        indicator_cols = [c for c in df.columns if c not in ['country_code', 'year', 'country_cn', 'country_en']]
        
        # 2024年数据
        data_2024 = df[df['year'] == 2024]
        has_2024 = []
        missing_2024 = []
        
        for col in indicator_cols:
            if data_2024[col].notna().any():
                has_2024.append(col)
            else:
                missing_2024.append(col)
        
        print(f"\n2024年数据:")
        print(f"   ✅ 有数据: {len(has_2024)} 个指标")
        print(f"   ❌ 缺失: {len(missing_2024)} 个指标")
        
        if missing_2024:
            print(f"   缺失指标: {missing_2024[:5]}{'...' if len(missing_2024) > 5 else ''}")
        
        # 2025年数据
        data_2025 = df[df['year'] == 2025]
        if len(data_2025) == 0:
            print(f"\n2025年数据:")
            print(f"   ❌ 无2025年数据记录")
            self.issues.append("🚨 R&D数据集无2025年数据，需要外推预测")
        else:
            has_2025 = [col for col in indicator_cols if data_2025[col].notna().any()]
            print(f"\n2025年数据:")
            print(f"   有数据: {len(has_2025)} 个指标")
        
        self.analysis_results['recent_availability'] = {
            'has_2024': has_2024,
            'missing_2024': missing_2024,
            'has_2025': len(data_2025) > 0
        }
        
        if len(missing_2024) > 0:
            self.recommendations.append(f"💡 {len(missing_2024)}个指标缺少2024年数据，建议时间序列外推")
    
    def analyze_alignment_with_ai_data(self):
        """分析与AI数据集的时间对齐情况"""
        print("\n" + "=" * 80)
        print("🔗 8. 与主AI数据集时间对齐分析")
        print("=" * 80)
        
        merged_file = MERGED_DATA_DIR / "rd_innovation_wide.csv"
        if not merged_file.exists():
            return
        
        df = pd.read_csv(merged_file)
        
        # AI数据集目标年份是2016-2025
        print(f"\nAI数据集目标范围: 2016-2025")
        print(f"R&D数据集范围: {df['year'].min()}-{df['year'].max()}")
        
        # 检查2016-2025的覆盖
        indicator_cols = [c for c in df.columns if c not in ['country_code', 'year', 'country_cn', 'country_en']]
        
        alignment_issues = []
        for col in indicator_cols:
            col_data = df[df[col].notna()]
            covered_focus = [y for y in FOCUS_YEARS if y in col_data['year'].values]
            if len(covered_focus) < len(FOCUS_YEARS):
                missing_years = [y for y in FOCUS_YEARS if y not in covered_focus]
                alignment_issues.append({
                    'indicator': col,
                    'missing_years': missing_years
                })
        
        if alignment_issues:
            print(f"\n⚠️ {len(alignment_issues)} 个指标在2016-2025期间有缺失:")
            for issue in alignment_issues[:10]:
                print(f"   • {issue['indicator']}: 缺失 {issue['missing_years']}")
            
            # 按缺失的年份分类
            missing_2025_count = sum(1 for i in alignment_issues if 2025 in i['missing_years'])
            missing_2024_count = sum(1 for i in alignment_issues if 2024 in i['missing_years'])
            
            print(f"\n   缺失2025年: {missing_2025_count} 个指标")
            print(f"   缺失2024年: {missing_2024_count} 个指标")
        else:
            print(f"\n✅ 所有指标在2016-2025期间数据完整")
        
        self.analysis_results['alignment_issues'] = alignment_issues
    
    def generate_report(self):
        """生成综合报告"""
        print("\n" + "=" * 100)
        print("📋 综合分析报告与建议")
        print("=" * 100)
        
        # 问题汇总
        print("\n🚨 关键问题:")
        if self.issues:
            for issue in self.issues:
                print(f"   {issue}")
        else:
            print("   ✅ 无关键问题")
        
        # 警告汇总
        print(f"\n⚠️ 警告 ({len(self.warnings)} 条):")
        for w in list(set(self.warnings))[:10]:
            print(f"   {w}")
        
        # 建议汇总
        print(f"\n💡 预处理建议:")
        for rec in list(set(self.recommendations)):
            print(f"   {rec}")
        
        # 与b题数据源脚本的差异
        print(f"\n📝 与AI数据预处理脚本的关键差异:")
        print("   1. 数据类型: R&D数据多为比例指标，无需大规模对数变换")
        print("   2. 货币处理: 无需CPI通胀调整（无大量美元投资数据）")
        print("   3. 时间范围: 数据从2010年开始，时间序列更长，可用于趋势分析")
        print("   4. 缺失处理: 尾部缺失（2024-2025）需外推，中间缺失可插值")
        print("   5. 国家异质性: 阿联酋/印度数据仍需特殊关注")
        
        # 保存报告
        report_content = f"""# R&D与创新基础数据质量分析报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 数据概览

- 数据来源: UNESCO UIS, World Bank
- 时间范围: 2010-2024
- 目标国家: {list(TARGET_COUNTRIES.keys())}
- 重点年份: 2016-2025（与AI数据对齐）

## 关键问题

{chr(10).join(['- ' + i for i in self.issues]) if self.issues else '无关键问题'}

## 警告

{chr(10).join(['- ' + w for w in list(set(self.warnings))])}

## 预处理建议

{chr(10).join(['- ' + r for r in list(set(self.recommendations))])}

## 与AI数据预处理的差异

| 方面 | AI数据(b题数据源) | R&D创新数据 |
|------|------------------|-------------|
| 数据类型 | 流量(投资、发表) | 存量/比例(R&D占比) |
| 对数变换 | 必须（偏度>5） | 仅绝对数量指标 |
| 通胀调整 | 必须（多年美元数据） | 不需要 |
| PPP调整 | 投资类需要 | 已有PPP版本 |
| 2025外推 | 部分指标需要 | 多数指标需要 |

## 预处理清单

### 1. 时间维度处理
- [ ] 2024-2025年缺失数据外推（Holt-Winters）
- [ ] 中间年份缺失插值（三次样条）
- [ ] 与AI数据2016-2025对齐

### 2. 缺失值处理
- [ ] 尾部缺失：时间序列外推
- [ ] 随机缺失：三次样条插值
- [ ] 结构性缺失（阿联酋等）：标记并考虑降权

### 3. 特征工程
- [ ] 创建年增长率特征
- [ ] 创建3年移动平均
- [ ] 创建与AI数据的滞后关联特征

### 4. 数据对齐
- [ ] 国家代码标准化（ISO 3166-1 alpha-3）
- [ ] 年份范围统一为2016-2025
- [ ] 输出格式与主表一致
"""
        
        report_path = OUTPUT_DIR / "rd_data_quality_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"\n✅ 详细报告已保存至: {report_path}")
        
        # 保存分析结果JSON
        results_path = OUTPUT_DIR / "rd_analysis_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            # 转换numpy类型
            def convert_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_types(i) for i in obj]
                return obj
            
            json.dump(convert_types(self.analysis_results), f, ensure_ascii=False, indent=2)
        
        print(f"✅ 分析结果JSON已保存至: {results_path}")


def main():
    """主函数"""
    analyzer = RDDataQualityAnalyzer()
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
