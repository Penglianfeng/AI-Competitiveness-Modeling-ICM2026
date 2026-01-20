# -*- coding: utf-8 -*-
"""
华数杯 B题 - 数据质量深度分析报告
Deep Data Quality Analysis for O-Award Level

本脚本针对O奖策略的6大要求进行全面检查：
1. 时间维度对齐与插补
2. 国家实体异质性处理
3. 货币与通胀标准化
4. 异常值与长尾分布处理
5. 滞后效应特征工程
6. 统一颗粒度
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ===================== 配置 =====================
DATA_DIR = Path(r"d:\华数杯\b题数据源")
OUTPUT_DIR = DATA_DIR / "preprocessed"
OUTPUT_DIR.mkdir(exist_ok=True)

# 目标国家（含中英文映射）
TARGET_COUNTRIES = {
    'USA': '美国', 'CHN': '中国', 'GBR': '英国', 'DEU': '德国', 
    'FRA': '法国', 'CAN': '加拿大', 'JPN': '日本', 'KOR': '韩国', 
    'ARE': '阿联酋', 'IND': '印度'
}
TARGET_YEARS = list(range(2016, 2026))

# 国家代码别名映射（处理不同数据源的命名差异）
COUNTRY_ALIASES = {
    'United States': 'USA', 'United States of America': 'USA', 'US': 'USA',
    'China': 'CHN', "People's Republic of China": 'CHN', 'PRC': 'CHN',
    'United Kingdom': 'GBR', 'UK': 'GBR', 'Great Britain': 'GBR',
    'Germany': 'DEU', 'Deutschland': 'DEU',
    'France': 'FRA',
    'Canada': 'CAN',
    'Japan': 'JPN',
    'South Korea': 'KOR', 'Korea, Republic of': 'KOR', 'Republic of Korea': 'KOR', 'Korea': 'KOR',
    'United Arab Emirates': 'ARE', 'UAE': 'ARE',
    'India': 'IND'
}

# 数据集分类
DATASET_CATEGORIES = {
    '出版物数据': [
        '各国历年人工智能出版物数量.csv',
        '各国历年人工智能出版物百分比.csv',
        '各国历年人工智能高影响力出版物数量.csv',
        '各国历年人工智能高影响力出版物百分比.csv',
        '各国历年人工智能Article数量.csv',
        '各国历年人工智能Book数量.csv',
        '各国历年人工智能Dataset数量.csv',
        '各国历年人工智能Dissertation数量.csv',
    ],
    '投资数据': [
        '各国历年在人工智能领域所有行业的风险投资（百万美元）.csv',
        '各国历年对生成式人工智能初创企业的风险投资（百万美元）.csv',
        '各国历年对AI计算初创企业的风险投资（百万美元）.csv',
        '各国历年在人工智能领域对外的风险投资（国家间）（百万美元）.csv',
    ],
    '基础设施数据': [
        '各国历年电能生产情况.csv',
    ],
    '人才与排名数据': [
        f'{year}_AI领域大学计算机排名.csv' for year in range(2000, 2026)
    ],
    'GitHub数据': [
        '各国历年在GitHub上的项目数.csv',
    ]
}


class DataQualityAnalyzer:
    """数据质量深度分析器"""
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.recommendations = []
        self.dataset_reports = {}
    
    def analyze_all_datasets(self):
        """分析所有数据集"""
        print("=" * 100)
        print("🔍 华数杯 B题 - 数据质量深度分析报告")
        print("=" * 100)
        print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"数据目录: {DATA_DIR}")
        print()
        
        # 获取所有CSV文件
        all_csv_files = list(DATA_DIR.glob("*.csv"))
        print(f"📂 发现 {len(all_csv_files)} 个CSV文件")
        print()
        
        # 1. 时间维度分析
        self.analyze_temporal_coverage(all_csv_files)
        
        # 2. 国家覆盖度分析
        self.analyze_country_coverage(all_csv_files)
        
        # 3. 2025年数据可用性分析（关键！）
        self.analyze_2025_data_availability(all_csv_files)
        
        # 4. 阿联酋等新兴国家数据分析
        self.analyze_emerging_countries_data()
        
        # 5. 数值分布分析（长尾检测）
        self.analyze_value_distributions()
        
        # 6. 货币数据分析
        self.analyze_currency_data()
        
        # 7. 数据颗粒度分析
        self.analyze_granularity()
        
        # 8. 生成综合报告
        self.generate_comprehensive_report()
        
        return self.dataset_reports
    
    def analyze_temporal_coverage(self, csv_files):
        """分析时间维度覆盖度"""
        print("\n" + "=" * 80)
        print("📅 1. 时间维度覆盖度分析")
        print("=" * 80)
        
        temporal_summary = []
        
        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path, nrows=1000)  # 读取前1000行快速分析
                
                # 识别年份列
                year_col = None
                for col in df.columns:
                    col_lower = col.lower()
                    if col_lower in ['year', 'date', 'quarter', '年份']:
                        year_col = col
                        break
                
                if year_col is None:
                    # 检查是否有年份在列名中
                    year_cols = [c for c in df.columns if str(c).isdigit() and 2000 <= int(c) <= 2030]
                    if year_cols:
                        continue  # 宽格式数据
                    continue
                
                # 分析年份范围
                years = df[year_col].dropna().unique()
                
                # 尝试提取年份（处理季度数据如 "2023 Q1"）
                def extract_year(val):
                    try:
                        if isinstance(val, (int, float)):
                            return int(val)
                        val_str = str(val)
                        for i in range(2000, 2030):
                            if str(i) in val_str:
                                return i
                        return None
                    except:
                        return None
                
                extracted_years = [extract_year(y) for y in years]
                extracted_years = [y for y in extracted_years if y is not None]
                
                if extracted_years:
                    min_year = min(extracted_years)
                    max_year = max(extracted_years)
                    
                    # 检查2016-2025覆盖度
                    target_coverage = [y for y in TARGET_YEARS if y in extracted_years]
                    missing_years = [y for y in TARGET_YEARS if y not in extracted_years]
                    
                    temporal_summary.append({
                        '文件名': file_path.name,
                        '最早年份': min_year,
                        '最新年份': max_year,
                        '覆盖2016-2025': f"{len(target_coverage)}/10",
                        '缺失年份': missing_years if missing_years else '无'
                    })
                    
                    if 2025 not in extracted_years:
                        self.warnings.append(f"⚠️ {file_path.name}: 缺失2025年数据")
                    if 2024 not in extracted_years:
                        self.warnings.append(f"⚠️ {file_path.name}: 缺失2024年数据")
                        
            except Exception as e:
                continue
        
        if temporal_summary:
            summary_df = pd.DataFrame(temporal_summary)
            print(summary_df.to_string(index=False))
            
            # 统计尾部缺失情况
            print("\n📊 尾部缺失统计（需要外推）:")
            missing_2025 = len([s for s in temporal_summary if 2025 in (s['缺失年份'] if isinstance(s['缺失年份'], list) else [])])
            missing_2024 = len([s for s in temporal_summary if 2024 in (s['缺失年份'] if isinstance(s['缺失年份'], list) else [])])
            print(f"  - 缺失2025年: {missing_2025}/{len(temporal_summary)} 个数据集")
            print(f"  - 缺失2024年: {missing_2024}/{len(temporal_summary)} 个数据集")
            
            if missing_2025 > 0:
                self.recommendations.append("💡 建议: 对缺失2025年数据的指标使用Holt-Winters或ARIMA进行短期外推")
    
    def analyze_country_coverage(self, csv_files):
        """分析国家覆盖度"""
        print("\n" + "=" * 80)
        print("🌍 2. 目标国家覆盖度分析")
        print("=" * 80)
        
        country_coverage = {}
        
        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path, nrows=5000)
                
                # 识别国家列
                country_col = None
                for col in df.columns:
                    col_lower = col.lower()
                    if 'country' in col_lower or 'territory' in col_lower or '国家' in col_lower:
                        country_col = col
                        break
                
                if country_col is None:
                    continue
                
                countries = df[country_col].dropna().unique()
                
                # 标准化国家名称
                standardized = set()
                for c in countries:
                    c_str = str(c).strip()
                    if c_str in TARGET_COUNTRIES:
                        standardized.add(c_str)
                    elif c_str in COUNTRY_ALIASES:
                        standardized.add(COUNTRY_ALIASES[c_str])
                
                target_found = [c for c in TARGET_COUNTRIES if c in standardized]
                target_missing = [c for c in TARGET_COUNTRIES if c not in standardized]
                
                country_coverage[file_path.name] = {
                    '找到': target_found,
                    '缺失': target_missing,
                    '覆盖率': f"{len(target_found)}/10"
                }
                
                if target_missing:
                    self.warnings.append(f"⚠️ {file_path.name}: 缺失国家 {target_missing}")
                    
            except Exception as e:
                continue
        
        # 打印汇总
        print(f"\n检查了 {len(country_coverage)} 个含国家信息的数据集")
        
        # 找出最常缺失的国家
        missing_counts = {}
        for info in country_coverage.values():
            for country in info['缺失']:
                missing_counts[country] = missing_counts.get(country, 0) + 1
        
        if missing_counts:
            print("\n📊 各国数据缺失频率:")
            for country, count in sorted(missing_counts.items(), key=lambda x: -x[1]):
                print(f"  - {country} ({TARGET_COUNTRIES[country]}): 在 {count} 个数据集中缺失")
    
    def analyze_2025_data_availability(self, csv_files):
        """详细分析2025年数据可用性"""
        print("\n" + "=" * 80)
        print("🎯 3. 2025年数据可用性详细分析（题目关键要求）")
        print("=" * 80)
        
        has_2025 = []
        missing_2025 = []
        
        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path)
                
                # 检查是否包含2025年数据
                has_2025_data = False
                
                # 方法1: 检查年份列
                for col in df.columns:
                    col_lower = str(col).lower()
                    if col_lower in ['year', 'date', 'quarter']:
                        values = df[col].astype(str)
                        if any('2025' in str(v) for v in values):
                            has_2025_data = True
                            break
                
                # 方法2: 检查列名
                if not has_2025_data:
                    if any('2025' in str(c) for c in df.columns):
                        has_2025_data = True
                
                # 方法3: 检查数据内容
                if not has_2025_data:
                    df_str = df.astype(str)
                    for col in df.columns:
                        if any('2025' in str(v) for v in df[col]):
                            has_2025_data = True
                            break
                
                if has_2025_data:
                    has_2025.append(file_path.name)
                else:
                    missing_2025.append(file_path.name)
                    
            except Exception as e:
                continue
        
        print(f"\n✅ 包含2025年数据的文件 ({len(has_2025)}):")
        for f in has_2025[:10]:  # 只显示前10个
            print(f"  - {f}")
        if len(has_2025) > 10:
            print(f"  ... 及其他 {len(has_2025) - 10} 个文件")
        
        print(f"\n❌ 缺失2025年数据的文件 ({len(missing_2025)}):")
        for f in missing_2025:
            print(f"  - {f}")
        
        if missing_2025:
            self.issues.append(f"🚨 关键问题: {len(missing_2025)} 个数据集缺失2025年数据，需要进行外推预测")
            self.recommendations.append("💡 对于缺失2025年的数据，建议使用Holt-Winters指数平滑进行短期外推")
    
    def analyze_emerging_countries_data(self):
        """分析阿联酋、印度等新兴国家数据质量"""
        print("\n" + "=" * 80)
        print("🌟 4. 新兴国家数据异质性分析（阿联酋ARE、印度IND）")
        print("=" * 80)
        
        emerging_countries = ['ARE', 'IND']
        
        # 读取主要数据集进行分析
        key_datasets = [
            ('各国历年人工智能出版物数量.csv', 'publications', 'Country/territory'),
            ('各国历年在人工智能领域所有行业的风险投资（百万美元）.csv', 'Sum_of_deals', 'Country'),
            ('各国历年在GitHub上的项目数.csv', 'AI_projects_fractional_count_based_on_contributions', 'Country'),
        ]
        
        for filename, value_col, country_col in key_datasets:
            file_path = DATA_DIR / filename
            if not file_path.exists():
                continue
            
            try:
                df = pd.read_csv(file_path)
                print(f"\n📊 {filename}")
                print("-" * 60)
                
                for country in emerging_countries:
                    country_data = df[df[country_col] == country]
                    
                    if len(country_data) == 0:
                        print(f"  ⚠️ {country}: 无数据")
                        self.issues.append(f"🚨 {filename} 中 {country} 无数据")
                        continue
                    
                    if value_col in country_data.columns:
                        values = country_data[value_col].dropna()
                        zero_count = (values == 0).sum()
                        missing_count = country_data[value_col].isna().sum()
                        
                        print(f"  {country} ({TARGET_COUNTRIES[country]}):")
                        print(f"    - 记录数: {len(country_data)}")
                        print(f"    - 缺失值: {missing_count} ({missing_count/len(country_data)*100:.1f}%)")
                        print(f"    - 零值: {zero_count} ({zero_count/len(values)*100:.1f}%)")
                        
                        if len(values) > 0:
                            print(f"    - 数值范围: [{values.min():.2f}, {values.max():.2f}]")
                            print(f"    - 均值: {values.mean():.2f}")
                        
                        if zero_count > len(values) * 0.5:
                            self.warnings.append(f"⚠️ {filename} 中 {country} 超过50%为零值，需特殊处理")
                            
            except Exception as e:
                print(f"  ❌ 读取失败: {e}")
        
        # 对比第一梯队国家
        print("\n📊 与第一梯队（美中）的数据对比:")
        print("-" * 60)
        
        try:
            pub_df = pd.read_csv(DATA_DIR / '各国历年人工智能出版物数量.csv')
            pub_2023 = pub_df[pub_df['year'] == 2023]
            
            comparison = []
            for country in ['USA', 'CHN', 'ARE', 'IND']:
                country_data = pub_2023[pub_2023['Country/territory'] == country]
                if len(country_data) > 0:
                    value = country_data['publications'].values[0]
                    comparison.append({'国家': f"{country} ({TARGET_COUNTRIES[country]})", '2023年AI出版物': value})
            
            if comparison:
                comp_df = pd.DataFrame(comparison)
                print(comp_df.to_string(index=False))
                
                # 计算差距倍数
                usa_val = [c['2023年AI出版物'] for c in comparison if c['国家'].startswith('USA')]
                are_val = [c['2023年AI出版物'] for c in comparison if c['国家'].startswith('ARE')]
                if usa_val and are_val and are_val[0] > 0:
                    ratio = usa_val[0] / are_val[0]
                    print(f"\n  📈 美国是阿联酋的 {ratio:.1f} 倍 → 强烈建议对数变换")
                    self.recommendations.append(f"💡 出版物数据: 美国是阿联酋的{ratio:.0f}倍，必须进行对数变换")
                    
        except Exception as e:
            print(f"  ❌ 对比分析失败: {e}")
    
    def analyze_value_distributions(self):
        """分析数值分布（检测长尾/幂律分布）"""
        print("\n" + "=" * 80)
        print("📈 5. 数值分布分析（长尾/偏度检测）")
        print("=" * 80)
        
        key_datasets = [
            ('各国历年人工智能出版物数量.csv', 'publications'),
            ('各国历年人工智能高影响力出版物数量.csv', 'publications'),
            ('各国历年在人工智能领域所有行业的风险投资（百万美元）.csv', 'Sum_of_deals'),
            ('各国历年对生成式人工智能初创企业的风险投资（百万美元）.csv', 'Sum_of_deals'),
            ('各国历年在GitHub上的项目数.csv', 'AI_projects_fractional_count_based_on_contributions'),
        ]
        
        distribution_report = []
        
        for filename, value_col in key_datasets:
            file_path = DATA_DIR / filename
            if not file_path.exists():
                continue
            
            try:
                df = pd.read_csv(file_path)
                if value_col not in df.columns:
                    continue
                
                values = df[value_col].dropna()
                if len(values) < 10:
                    continue
                
                # 计算统计量
                skewness = values.skew()
                kurtosis = values.kurtosis()
                cv = values.std() / values.mean() if values.mean() != 0 else 0
                
                # 判断是否需要对数变换
                need_log = abs(skewness) > 2 or cv > 2
                
                distribution_report.append({
                    '数据集': filename[:40] + '...' if len(filename) > 40 else filename,
                    '偏度': f"{skewness:.2f}",
                    '峰度': f"{kurtosis:.2f}",
                    '变异系数': f"{cv:.2f}",
                    '建议对数变换': '✅ 是' if need_log else '❌ 否',
                    'Min-Max比': f"{values.max()/values.min():.0f}x" if values.min() > 0 else 'N/A'
                })
                
                if need_log:
                    self.recommendations.append(f"💡 {filename}: 偏度={skewness:.2f}，强烈建议对数变换")
                    
            except Exception as e:
                continue
        
        if distribution_report:
            report_df = pd.DataFrame(distribution_report)
            print(report_df.to_string(index=False))
            
        print("\n📝 对数变换说明:")
        print("  - 偏度 > 2: 数据严重右偏，对数变换可改善")
        print("  - 变异系数 > 2: 数据离散度过大")
        print("  - Min-Max比过大: 头部与尾部差距悬殊")
    
    def analyze_currency_data(self):
        """分析货币数据（检测是否需要通胀/PPP调整）"""
        print("\n" + "=" * 80)
        print("💰 6. 货币数据分析（通胀/PPP调整需求）")
        print("=" * 80)
        
        currency_files = [f for f in DATA_DIR.glob("*.csv") if '美元' in f.name or 'USD' in f.name.upper() or '投资' in f.name]
        
        print(f"发现 {len(currency_files)} 个货币相关数据集:\n")
        
        for file_path in currency_files:
            try:
                df = pd.read_csv(file_path)
                
                print(f"📄 {file_path.name}")
                
                # 检查年份范围
                year_col = None
                for col in df.columns:
                    if 'year' in col.lower() or col.lower() == 'quarter':
                        year_col = col
                        break
                
                if year_col:
                    years = df[year_col].dropna().unique()
                    years_numeric = []
                    for y in years:
                        try:
                            if isinstance(y, (int, float)):
                                years_numeric.append(int(y))
                            else:
                                for i in range(2010, 2030):
                                    if str(i) in str(y):
                                        years_numeric.append(i)
                                        break
                        except:
                            pass
                    
                    if years_numeric:
                        year_span = max(years_numeric) - min(years_numeric)
                        print(f"  - 时间跨度: {min(years_numeric)}-{max(years_numeric)} ({year_span}年)")
                        
                        if year_span >= 5:
                            self.recommendations.append(f"💡 {file_path.name}: 时间跨度{year_span}年，必须进行通胀调整（转换为2020年不变价美元）")
                            print(f"  - ⚠️ 建议: 需要通胀调整 (CPI deflator)")
                        
                # 检查金额数据列
                value_cols = [c for c in df.columns if 'deal' in c.lower() or 'amount' in c.lower() or 'value' in c.lower()]
                if value_cols:
                    print(f"  - 金额列: {value_cols}")
                    
            except Exception as e:
                print(f"  ❌ 分析失败: {e}")
            print()
        
        print("\n📝 货币调整建议:")
        print("  1. 将所有美元金额转换为2020年不变价美元")
        print("  2. 对于薪资、基础设施投入，考虑使用PPP调整")
        print("  3. 中国的'1万亿元'投资需按当年汇率转换")
    
    def analyze_granularity(self):
        """分析数据颗粒度"""
        print("\n" + "=" * 80)
        print("⏰ 7. 数据颗粒度分析（时间粒度统一）")
        print("=" * 80)
        
        granularity_report = []
        
        for file_path in DATA_DIR.glob("*.csv"):
            try:
                df = pd.read_csv(file_path, nrows=100)
                
                # 检测时间颗粒度
                granularity = '未知'
                
                if 'Quarter' in df.columns:
                    granularity = '季度'
                elif any('月' in str(c) or 'Month' in str(c) for c in df.columns):
                    granularity = '月度'
                elif 'Year' in df.columns or 'year' in df.columns:
                    granularity = '年度'
                elif 'Date' in df.columns:
                    # 检查日期格式
                    date_vals = df['Date'].dropna().astype(str)
                    if len(date_vals) > 0:
                        sample = str(date_vals.iloc[0])
                        if len(sample) == 4 and sample.isdigit():
                            granularity = '年度'
                        elif 'Q' in sample:
                            granularity = '季度'
                        else:
                            granularity = '日期'
                
                if granularity != '年度' and granularity != '未知':
                    granularity_report.append({
                        '文件': file_path.name[:50],
                        '颗粒度': granularity,
                        '需要降采样': '✅ 是'
                    })
                    self.recommendations.append(f"💡 {file_path.name}: {granularity}数据，需降采样到年度")
                    
            except Exception as e:
                continue
        
        if granularity_report:
            print("需要降采样的数据集:")
            report_df = pd.DataFrame(granularity_report)
            print(report_df.to_string(index=False))
        else:
            print("✅ 所有数据集均为年度颗粒度，无需降采样")
        
        print("\n📝 降采样规则:")
        print("  - 存量指标（人才数、超算数）: 取年末值")
        print("  - 流量指标（论文数、投资额）: 年度求和")
        print("  - 率值指标（增长率、占比）: 加权平均")
    
    def generate_comprehensive_report(self):
        """生成综合报告"""
        print("\n" + "=" * 100)
        print("📋 综合数据质量报告")
        print("=" * 100)
        
        # 问题汇总
        print("\n🚨 关键问题 (Issues):")
        if self.issues:
            for i, issue in enumerate(self.issues, 1):
                print(f"  {i}. {issue}")
        else:
            print("  ✅ 无关键问题")
        
        # 警告汇总
        print(f"\n⚠️ 警告 ({len(self.warnings)} 条):")
        unique_warnings = list(set(self.warnings))[:10]  # 去重并限制数量
        for warning in unique_warnings:
            print(f"  - {warning}")
        if len(self.warnings) > 10:
            print(f"  ... 及其他 {len(self.warnings) - 10} 条警告")
        
        # 建议汇总
        print(f"\n💡 O奖级处理建议 ({len(self.recommendations)} 条):")
        unique_recs = list(set(self.recommendations))
        for rec in unique_recs:
            print(f"  - {rec}")
        
        # 保存报告
        report_content = f"""# 华数杯 B题 - 数据质量深度分析报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🚨 关键问题 (Issues)

"""
        for issue in self.issues:
            report_content += f"- {issue}\n"
        
        report_content += f"""

## ⚠️ 警告 ({len(self.warnings)} 条)

"""
        for warning in list(set(self.warnings)):
            report_content += f"- {warning}\n"
        
        report_content += f"""

## 💡 O奖级处理建议 ({len(self.recommendations)} 条)

"""
        for rec in list(set(self.recommendations)):
            report_content += f"- {rec}\n"
        
        report_content += """

## 📝 预处理清单 (Checklist)

### 1. 时间维度对齐
- [ ] 检查2016-2025年数据完整性
- [ ] 对中间缺失使用三次样条插值
- [ ] 对尾部缺失(2024-2025)使用Holt-Winters外推

### 2. 国家实体处理
- [ ] 统一国家代码（USA, CHN, GBR...）
- [ ] 处理阿联酋(ARE)的结构性缺失
- [ ] 明确中国数据口径（是否含港澳台）

### 3. 货币标准化
- [ ] 所有金额转换为2020年不变价美元
- [ ] 考虑PPP调整（如适用）
- [ ] 人民币按当年汇率转换

### 4. 长尾分布处理
- [ ] 对偏度>2的指标进行log1p变换
- [ ] 归一化前先对数变换
- [ ] 验证第二梯队国家区分度

### 5. 滞后特征
- [ ] 创建1-3年滞后特征
- [ ] 计算年度增长率
- [ ] 创建3年移动平均

### 6. 颗粒度统一
- [ ] 季度/月度数据降采样到年度
- [ ] 存量指标取年末值
- [ ] 流量指标取年度求和
"""
        
        report_path = OUTPUT_DIR / 'data_quality_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"\n✅ 详细报告已保存至: {report_path}")


def main():
    """主函数"""
    analyzer = DataQualityAnalyzer()
    analyzer.analyze_all_datasets()


if __name__ == "__main__":
    main()
