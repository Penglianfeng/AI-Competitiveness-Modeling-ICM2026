# -*- coding: utf-8 -*-
"""
华数杯 B题 - AI人才数据质量分析脚本
============================================
针对 Supply, Mobility and Quality of AI Talents 文件夹
的数据进行深度质量分析

数据特点：
- AI人才供给：研究人员密度、技术人员密度
- 人才培养：高等教育入学率、STEM毕业生比例、学位完成率
- 教育投入：教育支出占GDP比例、高等教育生均支出
- 人口基础：总人口、劳动年龄人口占比
- 来源：World Bank、UNESCO UIS

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
BASE_DIR = Path(r"d:\华数杯\Supply, Mobility and Quality of AI Talents")
MERGED_DATA_DIR = BASE_DIR / "merged_wide"
PROCESSED_DIR = BASE_DIR / "ai_talent_data_v2" / "processed"
OUTPUT_DIR = BASE_DIR / "preprocessed"
OUTPUT_DIR.mkdir(exist_ok=True)

# 目标国家（与主数据集保持一致）
TARGET_COUNTRIES = {
    'USA': '美国', 'CHN': '中国', 'GBR': '英国', 'DEU': '德国', 
    'FRA': '法国', 'CAN': '加拿大', 'JPN': '日本', 'KOR': '韩国', 
    'ARE': '阿联酋', 'IND': '印度'
}

# 目标年份范围
TARGET_YEARS = list(range(2015, 2026))
FOCUS_YEARS = list(range(2016, 2026))  # 重点分析年份（与AI主数据对齐）

# 指标分类（用于确定处理方式）
INDICATOR_CATEGORIES = {
    'density': [  # 密度型指标（每百万人）
        'researchers_per_million',
        'researchers_per_million_fte',
        'technicians_per_million',
    ],
    'ratio_pct': [  # 比例/百分比指标，无需对数变换
        'tertiary_gross_enrollment_pct',
        'tertiary_female_share_pct',
        'education_expenditure_pct_gdp',
        'tertiary_spend_per_student_pct_gdp_pc',
        'rd_expenditure_pct_gdp',
        'pop_15_64_pct',
        'stem_graduates_pct',
        'tertiary_completion_25_34_pct',
    ],
    'count_absolute': [  # 绝对数量指标，可能需要对数变换
        'population_total',
        'tertiary_enrollment_total',
    ]
}

# 指标中文名称
INDICATOR_CN = {
    'researchers_per_million': '每百万人研究人员数',
    'researchers_per_million_fte': '每百万人研究人员数(FTE)',
    'technicians_per_million': '每百万人研发技术人员数',
    'tertiary_gross_enrollment_pct': '高等教育毛入学率(%)',
    'tertiary_female_share_pct': '高等教育女性占比(%)',
    'education_expenditure_pct_gdp': '教育支出占GDP比例(%)',
    'tertiary_spend_per_student_pct_gdp_pc': '高等教育生均支出占人均GDP比例(%)',
    'rd_expenditure_pct_gdp': 'R&D支出占GDP比例(%)',
    'pop_15_64_pct': '15-64岁人口占比(%)',
    'stem_graduates_pct': 'STEM毕业生占比(%)',
    'tertiary_completion_25_34_pct': '25-34岁高等教育完成率(%)',
    'population_total': '总人口',
    'tertiary_enrollment_total': '高等教育在校生总数',
}


class AITalentQualityAnalyzer:
    """AI人才数据质量分析器"""
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.recommendations = []
        self.analysis_results = {}
        
    def run_full_analysis(self):
        """执行完整分析"""
        print("=" * 100)
        print("🎓 华数杯 B题 - AI人才数据质量分析报告")
        print("=" * 100)
        print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"数据目录: {BASE_DIR}")
        print()
        
        # 1. 数据源概览
        self.analyze_data_sources()
        
        # 2. 主数据集分析
        self.analyze_main_dataset()
        
        # 3. 时间覆盖度分析
        self.analyze_temporal_coverage()
        
        # 4. 国家覆盖度分析  
        self.analyze_country_coverage()
        
        # 5. 缺失值模式分析
        self.analyze_missing_patterns()
        
        # 6. 数值分布分析
        self.analyze_value_distributions()
        
        # 7. 2023-2025年数据可用性
        self.analyze_recent_data_availability()
        
        # 8. 指标相关性分析
        self.analyze_indicator_correlations()
        
        # 9. 与主AI数据集时间对齐分析
        self.analyze_alignment_with_ai_data()
        
        # 10. 生成综合报告
        self.generate_report()
        
        return self.analysis_results
    
    def load_main_data(self) -> pd.DataFrame:
        """加载主数据集"""
        main_file = MERGED_DATA_DIR / "ai_talent_wide.csv"
        if not main_file.exists():
            raise FileNotFoundError(f"主数据集不存在: {main_file}")
        return pd.read_csv(main_file)
    
    def analyze_data_sources(self):
        """分析数据源结构"""
        print("\n" + "=" * 80)
        print("📂 1. 数据源概览")
        print("=" * 80)
        
        sources = {}
        
        # 原始数据
        raw_dir = BASE_DIR / "ai_talent_data_v2" / "raw"
        if raw_dir.exists():
            raw_files = list(raw_dir.glob("*.csv"))
            sources['raw'] = [f.name for f in raw_files]
            print(f"\n📁 原始数据 ({raw_dir}):")
            for f in raw_files:
                df = pd.read_csv(f, nrows=5)
                print(f"   • {f.name}: {len(df.columns)} 列")
        
        # 处理后数据
        if PROCESSED_DIR.exists():
            processed_files = list(PROCESSED_DIR.glob("*.csv"))
            sources['processed'] = [f.name for f in processed_files]
            print(f"\n📁 处理后数据 ({PROCESSED_DIR}):")
            for f in processed_files:
                df = pd.read_csv(f)
                print(f"   • {f.name}: {len(df)} 行, {len(df.columns)} 列")
        
        # 合并宽表
        if MERGED_DATA_DIR.exists():
            merged_files = list(MERGED_DATA_DIR.glob("*.csv"))
            sources['merged'] = [f.name for f in merged_files]
            print(f"\n📁 合并宽表 ({MERGED_DATA_DIR}):")
            for f in merged_files:
                df = pd.read_csv(f)
                print(f"   • {f.name}: {len(df)} 行, {len(df.columns)} 列")
        
        self.analysis_results['data_sources'] = sources
    
    def analyze_main_dataset(self):
        """分析主数据集结构"""
        print("\n" + "=" * 80)
        print("📊 2. 主数据集分析 (ai_talent_wide.csv)")
        print("=" * 80)
        
        df = self.load_main_data()
        
        print(f"\n基本信息:")
        print(f"   - 总行数: {len(df)}")
        print(f"   - 总列数: {len(df.columns)}")
        print(f"   - 国家数: {df['country_code'].nunique()}")
        print(f"   - 年份范围: {df['year'].min()} - {df['year'].max()}")
        
        # 列信息
        print(f"\n📋 列名:")
        meta_cols = ['country_code', 'country_cn', 'country_en', 'year']
        indicator_cols = [c for c in df.columns if c not in meta_cols]
        
        print(f"   元数据列: {meta_cols}")
        print(f"\n   指标列 ({len(indicator_cols)} 个):")
        for col in indicator_cols:
            non_null = df[col].notna().sum()
            non_null_pct = non_null / len(df) * 100
            cn_name = INDICATOR_CN.get(col, col)
            print(f"      • {col}: {non_null_pct:.1f}% 非空 ({cn_name})")
        
        self.analysis_results['main_dataset'] = {
            'rows': len(df),
            'columns': len(df.columns),
            'countries': df['country_code'].unique().tolist(),
            'year_range': (int(df['year'].min()), int(df['year'].max())),
            'indicators': indicator_cols
        }
    
    def analyze_temporal_coverage(self):
        """分析时间覆盖度"""
        print("\n" + "=" * 80)
        print("📅 3. 时间覆盖度分析")
        print("=" * 80)
        
        df = self.load_main_data()
        meta_cols = ['country_code', 'country_cn', 'country_en', 'year']
        indicator_cols = [c for c in df.columns if c not in meta_cols]
        
        coverage = {}
        print(f"\n各指标时间覆盖:")
        print("-" * 80)
        print(f"{'指标':<40} {'起始年':<8} {'结束年':<8} {'2016-2025覆盖':<15}")
        print("-" * 80)
        
        for col in indicator_cols:
            valid_data = df[df[col].notna()]
            if len(valid_data) > 0:
                min_year = int(valid_data['year'].min())
                max_year = int(valid_data['year'].max())
                valid_years = valid_data['year'].unique()
                
                coverage[col] = {
                    'min_year': min_year,
                    'max_year': max_year,
                    'years_count': len(valid_years)
                }
                
                # 检查2016-2025覆盖
                focus_coverage = len([y for y in valid_years if y in FOCUS_YEARS])
                coverage_str = f"{focus_coverage}/10"
                
                print(f"   {col[:38]:<40} {min_year:<8} {max_year:<8} {coverage_str:<15}")
                
                if max_year < 2023:
                    self.warnings.append(f"⚠️ {col}: 最新数据仅到{max_year}年")
        
        self.analysis_results['temporal_coverage'] = coverage
    
    def analyze_country_coverage(self):
        """分析各国数据覆盖情况"""
        print("\n" + "=" * 80)
        print("🌍 4. 各国数据覆盖度分析")
        print("=" * 80)
        
        df = self.load_main_data()
        meta_cols = ['country_code', 'country_cn', 'country_en', 'year']
        indicator_cols = [c for c in df.columns if c not in meta_cols]
        
        print(f"\n各国指标非空率:")
        print("-" * 70)
        
        country_coverage = {}
        for country in TARGET_COUNTRIES.keys():
            country_data = df[df['country_code'] == country]
            if len(country_data) == 0:
                print(f"   ❌ {country} ({TARGET_COUNTRIES[country]}): 无数据")
                self.issues.append(f"🚨 {country} 无数据")
                continue
            
            # 计算各指标非空率
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
            
            status = "✅" if avg_rate > 60 else "⚠️" if avg_rate > 30 else "❌"
            print(f"   {status} {country} ({TARGET_COUNTRIES[country]:4s}): "
                  f"平均覆盖率 {avg_rate:.1f}%, {len(country_data)}年数据")
            
            if avg_rate < 40:
                self.warnings.append(f"⚠️ {country}: 数据覆盖率仅{avg_rate:.1f}%")
        
        # 特别关注低覆盖率指标
        print(f"\n🔍 低覆盖率指标详情:")
        for country in ['ARE', 'IND', 'CHN']:
            if country in country_coverage:
                detail = country_coverage[country]['detail']
                low_coverage = [(k, v) for k, v in detail.items() if v < 30]
                if low_coverage:
                    print(f"   {country} ({TARGET_COUNTRIES[country]}):")
                    for col, pct in sorted(low_coverage, key=lambda x: x[1])[:3]:
                        cn_name = INDICATOR_CN.get(col, col)
                        print(f"      • {cn_name}: {pct:.1f}%")
        
        self.analysis_results['country_coverage'] = country_coverage
    
    def analyze_missing_patterns(self):
        """分析缺失值模式"""
        print("\n" + "=" * 80)
        print("🔍 5. 缺失值模式分析")
        print("=" * 80)
        
        df = self.load_main_data()
        meta_cols = ['country_code', 'country_cn', 'country_en', 'year']
        indicator_cols = [c for c in df.columns if c not in meta_cols]
        
        print(f"\n缺失值统计:")
        print("-" * 70)
        
        missing_summary = []
        for col in indicator_cols:
            missing = df[col].isna().sum()
            missing_pct = missing / len(df) * 100
            cn_name = INDICATOR_CN.get(col, col)
            missing_summary.append({
                'indicator': col,
                'indicator_cn': cn_name,
                'missing_count': missing,
                'missing_pct': missing_pct
            })
        
        # 按缺失率排序
        missing_summary.sort(key=lambda x: x['missing_pct'], reverse=True)
        
        print(f"{'指标':<35} {'缺失数':>10} {'缺失率':>10}")
        print("-" * 60)
        for item in missing_summary:
            print(f"{item['indicator'][:34]:<35} {item['missing_count']:>10} "
                  f"{item['missing_pct']:>9.1f}%")
        
        # 分析缺失模式类型
        print(f"\n缺失模式分类:")
        
        tail_missing = []  # 尾部缺失
        sparse_indicators = []  # 稀疏指标
        
        for col in indicator_cols:
            # 尾部缺失检测
            recent_data = df[df['year'] >= 2023][col]
            if recent_data.isna().sum() / len(recent_data) > 0.8:
                tail_missing.append(col)
            
            # 稀疏指标检测（覆盖率<30%）
            if df[col].notna().sum() / len(df) < 0.3:
                sparse_indicators.append(col)
        
        if tail_missing:
            print(f"   📍 尾部缺失 (2023+数据稀少): {len(tail_missing)} 个指标")
            for col in tail_missing:
                cn_name = INDICATOR_CN.get(col, col)
                print(f"      • {cn_name}")
            self.recommendations.append(f"💡 {len(tail_missing)}个指标缺少2023年后数据，建议Holt-Winters外推")
        
        if sparse_indicators:
            print(f"   📍 稀疏指标 (覆盖率<30%): {len(sparse_indicators)} 个")
            for col in sparse_indicators:
                cn_name = INDICATOR_CN.get(col, col)
                print(f"      • {cn_name}")
            self.warnings.append(f"⚠️ {len(sparse_indicators)}个指标覆盖率低于30%，建模时考虑剔除或降权")
        
        self.analysis_results['missing_patterns'] = {
            'summary': missing_summary,
            'tail_missing': tail_missing,
            'sparse_indicators': sparse_indicators
        }
    
    def analyze_value_distributions(self):
        """分析数值分布特征"""
        print("\n" + "=" * 80)
        print("📈 6. 数值分布分析")
        print("=" * 80)
        
        df = self.load_main_data()
        meta_cols = ['country_code', 'country_cn', 'country_en', 'year']
        indicator_cols = [c for c in df.columns if c not in meta_cols]
        
        print(f"\n分布特征分析:")
        print("-" * 90)
        print(f"{'指标':<30} {'最小值':>12} {'最大值':>12} {'偏度':>8} {'建议对数':>10}")
        print("-" * 90)
        
        distribution_analysis = []
        for col in indicator_cols:
            values = df[col].dropna()
            if len(values) < 10:
                continue
            
            min_val = values.min()
            max_val = values.max()
            skewness = values.skew()
            
            # 判断是否需要对数变换
            need_log = False
            if col in INDICATOR_CATEGORIES.get('count_absolute', []):
                if abs(skewness) > 2:
                    need_log = True
            
            distribution_analysis.append({
                'indicator': col,
                'min': min_val,
                'max': max_val,
                'skewness': skewness,
                'need_log': need_log
            })
            
            log_mark = "✅ 是" if need_log else "❌ 否"
            print(f"   {col[:28]:<30} {min_val:>12.2f} {max_val:>12.2f} "
                  f"{skewness:>8.2f} {log_mark:>10}")
        
        # 结论
        need_log_count = sum(1 for d in distribution_analysis if d['need_log'])
        print(f"\n📝 结论:")
        print(f"   • {need_log_count} 个指标建议对数变换（绝对数量指标：人口、在校生数）")
        print(f"   • 密度指标和比例指标通常不需要对数变换")
        
        self.analysis_results['distribution_analysis'] = distribution_analysis
    
    def analyze_recent_data_availability(self):
        """分析2023-2025年数据可用性"""
        print("\n" + "=" * 80)
        print("🎯 7. 2023-2025年数据可用性分析（关键！）")
        print("=" * 80)
        
        df = self.load_main_data()
        meta_cols = ['country_code', 'country_cn', 'country_en', 'year']
        indicator_cols = [c for c in df.columns if c not in meta_cols]
        
        availability = {}
        
        for year in [2023, 2024, 2025]:
            year_data = df[df['year'] == year]
            if len(year_data) == 0:
                print(f"\n{year}年数据: ❌ 无记录")
                availability[year] = {'has_data': False, 'indicators': []}
                continue
            
            has_data = []
            missing_data = []
            
            for col in indicator_cols:
                if year_data[col].notna().any():
                    has_data.append(col)
                else:
                    missing_data.append(col)
            
            availability[year] = {
                'has_data': True,
                'indicators_with_data': has_data,
                'indicators_missing': missing_data
            }
            
            print(f"\n{year}年数据:")
            print(f"   ✅ 有数据: {len(has_data)} 个指标")
            if has_data:
                print(f"      {', '.join([INDICATOR_CN.get(c, c)[:10] for c in has_data[:5]])}")
            print(f"   ❌ 缺失: {len(missing_data)} 个指标")
            if missing_data:
                print(f"      {', '.join([INDICATOR_CN.get(c, c)[:10] for c in missing_data[:5]])}")
        
        # 2025年特别说明
        if 2025 not in [int(y) for y in df['year'].unique()]:
            self.issues.append("🚨 AI人才数据集无2025年数据，需要外推预测")
            print(f"\n⚠️ 2025年数据完全缺失，建议使用时间序列外推")
        
        self.analysis_results['recent_availability'] = availability
    
    def analyze_indicator_correlations(self):
        """分析指标相关性"""
        print("\n" + "=" * 80)
        print("🔗 8. 核心指标相关性分析")
        print("=" * 80)
        
        df = self.load_main_data()
        
        # 选择核心指标进行相关性分析
        core_indicators = [
            'researchers_per_million',
            'tertiary_gross_enrollment_pct',
            'education_expenditure_pct_gdp',
            'rd_expenditure_pct_gdp',
            'pop_15_64_pct'
        ]
        
        available_core = [c for c in core_indicators if c in df.columns]
        
        if len(available_core) >= 3:
            corr_matrix = df[available_core].corr()
            
            print(f"\n核心指标相关系数矩阵:")
            print("-" * 70)
            
            # 打印简化的相关性
            for i, col1 in enumerate(available_core):
                for j, col2 in enumerate(available_core):
                    if i < j:
                        corr = corr_matrix.loc[col1, col2]
                        cn1 = INDICATOR_CN.get(col1, col1)[:15]
                        cn2 = INDICATOR_CN.get(col2, col2)[:15]
                        strength = "强" if abs(corr) > 0.7 else "中" if abs(corr) > 0.4 else "弱"
                        print(f"   {cn1} ↔ {cn2}: {corr:.3f} ({strength})")
            
            self.analysis_results['correlations'] = corr_matrix.to_dict()
        else:
            print("   核心指标数量不足，无法进行相关性分析")
    
    def analyze_alignment_with_ai_data(self):
        """分析与主AI数据集的时间对齐情况"""
        print("\n" + "=" * 80)
        print("🔄 9. 与主AI数据集时间对齐分析")
        print("=" * 80)
        
        df = self.load_main_data()
        
        print(f"\n主AI数据集目标范围: 2016-2025")
        print(f"AI人才数据集实际范围: {df['year'].min()}-{df['year'].max()}")
        
        # 检查2016-2025覆盖
        meta_cols = ['country_code', 'country_cn', 'country_en', 'year']
        indicator_cols = [c for c in df.columns if c not in meta_cols]
        
        alignment_issues = []
        
        for col in indicator_cols:
            col_data = df[df[col].notna()]
            covered_years = [int(y) for y in col_data['year'].unique()]
            missing_focus = [y for y in FOCUS_YEARS if y not in covered_years]
            
            if missing_focus:
                alignment_issues.append({
                    'indicator': col,
                    'indicator_cn': INDICATOR_CN.get(col, col),
                    'missing_years': missing_focus
                })
        
        if alignment_issues:
            print(f"\n⚠️ {len(alignment_issues)} 个指标在2016-2025期间有缺失:")
            for issue in alignment_issues[:8]:
                missing_str = ', '.join(map(str, issue['missing_years']))
                print(f"   • {issue['indicator_cn'][:25]}: 缺失 [{missing_str}]")
            
            # 统计
            missing_2025 = sum(1 for i in alignment_issues if 2025 in i['missing_years'])
            missing_2024 = sum(1 for i in alignment_issues if 2024 in i['missing_years'])
            missing_2023 = sum(1 for i in alignment_issues if 2023 in i['missing_years'])
            
            print(f"\n   年份缺失统计:")
            print(f"      缺失2025年: {missing_2025} 个指标")
            print(f"      缺失2024年: {missing_2024} 个指标")
            print(f"      缺失2023年: {missing_2023} 个指标")
            
            self.recommendations.append(f"💡 需要外推{missing_2025}个指标的2025年数据")
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
        
        # AI人才数据特点说明
        print(f"\n📝 AI人才数据特点分析:")
        print("   1. 数据类型: 以密度指标和比例指标为主，多数不需要对数变换")
        print("   2. 时间范围: 2015-2024，需要补充2025年数据")
        print("   3. 缺失模式: 2023-2024年多指标缺失，需外推；部分指标稀疏")
        print("   4. 关键指标: researchers_per_million, tertiary_gross_enrollment_pct")
        print("   5. 稀疏指标: stem_graduates_pct, tertiary_completion_25_34_pct 覆盖率低")
        
        # 与R&D数据对比
        print(f"\n📊 与R&D创新数据的关系:")
        print("   • 重叠指标: rd_expenditure_pct_gdp, researchers_per_million")
        print("   • 互补指标: 人才数据侧重教育培养，R&D数据侧重创新投入产出")
        print("   • 合并建议: 按country_code+year合并，保留各自特有指标")
        
        # 保存报告
        report_content = f"""# AI人才数据质量分析报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 数据概览

- 数据来源: World Bank, UNESCO UIS
- 时间范围: 2015-2024
- 目标国家: {list(TARGET_COUNTRIES.keys())}
- 重点年份: 2016-2025（与AI主数据对齐）
- 核心指标: 研究人员密度、高等教育入学率、教育支出、R&D支出等

## 关键问题

{chr(10).join(['- ' + i for i in self.issues]) if self.issues else '无关键问题'}

## 警告

{chr(10).join(['- ' + w for w in list(set(self.warnings))])}

## 预处理建议

{chr(10).join(['- ' + r for r in list(set(self.recommendations))])}

## 指标覆盖率

| 指标 | 非空率 | 建议处理 |
|------|--------|---------|
| pop_15_64_pct | 100% | 直接使用 |
| population_total | 100% | 对数变换 |
| tertiary_gross_enrollment_pct | 89% | 插值补全 |
| rd_expenditure_pct_gdp | 76% | 插值+外推 |
| researchers_per_million | 66% | 插值+外推 |
| education_expenditure_pct_gdp | 63% | 插值+外推 |
| stem_graduates_pct | 9% | 考虑剔除或特殊处理 |
| tertiary_completion_25_34_pct | 7% | 考虑剔除或特殊处理 |

## 预处理清单

### 1. 时间维度处理
- [ ] 2023-2025年缺失数据外推（Holt-Winters指数平滑）
- [ ] 中间年份缺失插值（三次样条）
- [ ] 与AI主数据2016-2025对齐

### 2. 缺失值处理
- [ ] 尾部缺失：时间序列外推
- [ ] 中间缺失：三次样条插值
- [ ] 稀疏指标：标记并考虑降权或剔除

### 3. 特征工程
- [ ] 年增长率特征
- [ ] 3年移动平均（平滑波动）
- [ ] 人才综合指数（标准化加权）
- [ ] 与AI产出的滞后关联特征

### 4. 数据标准化
- [ ] 绝对数量指标对数变换（population_total, tertiary_enrollment_total）
- [ ] 国家代码标准化（ISO 3166-1 alpha-3）
- [ ] 输出格式与主表一致
"""
        
        report_path = OUTPUT_DIR / "ai_talent_quality_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"\n✅ 详细报告已保存至: {report_path}")
        
        # 保存分析结果JSON
        results_path = OUTPUT_DIR / "ai_talent_analysis_results.json"
        
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
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(convert_types(self.analysis_results), f, ensure_ascii=False, indent=2)
        
        print(f"✅ 分析结果JSON已保存至: {results_path}")


def main():
    """主函数"""
    analyzer = AITalentQualityAnalyzer()
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
