#!/usr/bin/env python3
"""
数据集合并与宽表生成脚本
========================
将工作区下的所有R&D相关数据集合并为统一的宽表格式

输出格式：一行 = 国家 + 年份，列 = 各指标
"""

import pandas as pd
import os
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 目标国家映射
TARGET_COUNTRIES = {
    "USA": {"name_en": "United States", "name_cn": "美国"},
    "CHN": {"name_en": "China", "name_cn": "中国"},
    "GBR": {"name_en": "United Kingdom", "name_cn": "英国"},
    "DEU": {"name_en": "Germany", "name_cn": "德国"},
    "KOR": {"name_en": "South Korea", "name_cn": "韩国"},
    "JPN": {"name_en": "Japan", "name_cn": "日本"},
    "FRA": {"name_en": "France", "name_cn": "法国"},
    "CAN": {"name_en": "Canada", "name_cn": "加拿大"},
    "ARE": {"name_en": "United Arab Emirates", "name_cn": "阿联酋"},
    "IND": {"name_en": "India", "name_cn": "印度"}
}

# 国家名称到ISO代码映射
COUNTRY_NAME_TO_CODE = {
    "United States": "USA",
    "China": "CHN", 
    "United Kingdom": "GBR",
    "Germany": "DEU",
    "Korea, Rep.": "KOR",
    "South Korea": "KOR",
    "Japan": "JPN",
    "France": "FRA",
    "Canada": "CAN",
    "United Arab Emirates": "ARE",
    "India": "IND",
    # 中文名称
    "美国": "USA",
    "中国": "CHN",
    "英国": "GBR",
    "德国": "DEU",
    "韩国": "KOR",
    "日本": "JPN",
    "法国": "FRA",
    "加拿大": "CAN",
    "阿联酋": "ARE",
    "印度": "IND",
}


def find_latest_file(directory: str, pattern: str) -> str:
    """找到目录下匹配模式的最新文件"""
    path = Path(directory)
    if not path.exists():
        return None
    
    files = list(path.glob(pattern))
    if not files:
        return None
    
    # 按修改时间排序，返回最新的
    return str(max(files, key=lambda x: x.stat().st_mtime))


def load_uis_data() -> pd.DataFrame:
    """加载UIS/SDG数据（长格式转宽格式）"""
    print("\n[1] 加载 UIS/SDG 数据...")
    
    # 尝试多个可能的路径
    possible_paths = [
        "uis_rd_data/processed",
        "Research and development investment and innovation foundation/uis_rd_data/processed",
    ]
    
    df_list = []
    for base_path in possible_paths:
        file_path = find_latest_file(base_path, "combined_data_*.csv")
        if file_path:
            print(f"  - 找到文件: {file_path}")
            df = pd.read_csv(file_path)
            df_list.append(df)
    
    if not df_list:
        print("  ⚠️ 未找到UIS数据文件")
        return pd.DataFrame()
    
    # 合并所有找到的UIS数据
    combined = pd.concat(df_list, ignore_index=True)
    
    # 去重（基于 country_code, year, indicator）
    combined = combined.drop_duplicates(subset=['country_code', 'year', 'indicator'])
    
    print(f"  - 原始记录数: {len(combined)}")
    print(f"  - 指标列表: {combined['indicator'].unique().tolist()}")
    
    # 转换为宽格式
    # 创建简短的列名
    indicator_rename = {
        "SDG_9.5.1": "rd_expenditure_pct_gdp",
        "SDG_9.5.2": "researchers_per_million",
    }
    
    combined['indicator_short'] = combined['indicator'].map(indicator_rename).fillna(combined['indicator'])
    
    wide_df = combined.pivot_table(
        index=['country_code', 'year'],
        columns='indicator_short',
        values='value',
        aggfunc='first'
    ).reset_index()
    
    # 扁平化列名
    wide_df.columns = [col if isinstance(col, str) else col for col in wide_df.columns]
    
    print(f"  ✓ UIS数据转换为宽格式: {len(wide_df)} 行, {len(wide_df.columns)} 列")
    return wide_df


def load_worldbank_ai_data() -> pd.DataFrame:
    """加载World Bank AI发展指标数据"""
    print("\n[2] 加载 World Bank AI发展指标数据...")
    
    possible_paths = [
        "Research and development investment and innovation foundation/World Bank Data",
    ]
    
    for base_path in possible_paths:
        file_path = find_latest_file(base_path, "AI_development_indicators_*.csv")
        if file_path:
            print(f"  - 找到文件: {file_path}")
            df = pd.read_csv(file_path)
            
            # 标准化列名
            print(f"  - 原始列: {df.columns.tolist()[:5]}...")
            
            # 重命名中文列名为英文
            column_rename = {
                "国家": "country_name",
                "年份": "year",
                "R&D支出占GDP比例(%)": "rd_expenditure_pct_gdp_wb",
                "每百万人研究人员数量": "researchers_per_million_wb",
                "高等教育毛入学率(%)": "higher_edu_enrollment_rate",
                "拥有学士及以上学历人口比例(%)": "bachelor_degree_pct",
                "拥有硕士及以上学历人口比例(%)": "master_degree_pct",
                "拥有博士学历人口比例(%)": "phd_degree_pct",
                "使用互联网人口比例(%)": "internet_users_pct",
                "每100人固定宽带订阅数": "fixed_broadband_per_100",
                "每100人移动电话订阅数": "mobile_subscriptions_per_100",
                "专利申请数(居民)": "patent_applications_resident",
                "专利申请数(非居民)": "patent_applications_nonresident",
                "ICT服务出口占服务出口比例(%)": "ict_service_exports_pct",
                "高科技出口占制成品出口比例(%)": "high_tech_exports_pct",
                "高科技产品出口(现价美元)": "high_tech_exports_usd",
                "政府教育支出占GDP比例(%)": "govt_edu_expenditure_pct_gdp",
                "高等教育支出占政府教育支出比例(%)": "higher_edu_expenditure_pct",
                "每百万人安全互联网服务器数": "secure_internet_servers_per_million",
                "具有高等教育程度的劳动力比例(%)": "labor_force_higher_edu_pct",
                "国家中文": "country_cn",
            }
            
            df = df.rename(columns=column_rename)
            
            # 添加国家代码
            if 'country_name' in df.columns:
                df['country_code'] = df['country_name'].map(COUNTRY_NAME_TO_CODE)
            
            # 过滤目标国家
            df = df[df['country_code'].isin(TARGET_COUNTRIES.keys())]
            
            print(f"  ✓ World Bank数据: {len(df)} 行, {len(df.columns)} 列")
            return df
    
    print("  ⚠️ 未找到World Bank AI数据文件")
    return pd.DataFrame()


def merge_all_datasets(uis_df: pd.DataFrame, wb_df: pd.DataFrame) -> pd.DataFrame:
    """合并所有数据集"""
    print("\n[3] 合并数据集...")
    
    if uis_df.empty and wb_df.empty:
        print("  ⚠️ 没有数据可合并")
        return pd.DataFrame()
    
    # 确保year列是整数类型
    if not uis_df.empty and 'year' in uis_df.columns:
        uis_df['year'] = pd.to_numeric(uis_df['year'], errors='coerce').astype('Int64')
    if not wb_df.empty and 'year' in wb_df.columns:
        wb_df['year'] = pd.to_numeric(wb_df['year'], errors='coerce').astype('Int64')
    
    if uis_df.empty:
        merged = wb_df.copy()
    elif wb_df.empty:
        merged = uis_df.copy()
    else:
        # 合并两个数据源
        # UIS数据已有 rd_expenditure_pct_gdp 和 researchers_per_million
        # WB数据有更多指标，但也有这两个（后缀_wb）
        
        # 选择WB数据中需要的列（排除与UIS重复的）
        wb_cols_to_use = ['country_code', 'year']
        for col in wb_df.columns:
            if col not in ['country_code', 'year', 'country_name', 'country_cn']:
                # 如果WB有_wb后缀的列，保留；如果UIS已有同名列，跳过
                if col.endswith('_wb') or col not in uis_df.columns:
                    wb_cols_to_use.append(col)
        
        wb_subset = wb_df[wb_cols_to_use].copy()
        
        # 左连接合并
        merged = pd.merge(
            uis_df,
            wb_subset,
            on=['country_code', 'year'],
            how='outer'
        )
        
        # 优先使用UIS的数据，WB的作为补充
        if 'rd_expenditure_pct_gdp' in merged.columns and 'rd_expenditure_pct_gdp_wb' in merged.columns:
            merged['rd_expenditure_pct_gdp'] = merged['rd_expenditure_pct_gdp'].fillna(merged['rd_expenditure_pct_gdp_wb'])
        if 'researchers_per_million' in merged.columns and 'researchers_per_million_wb' in merged.columns:
            merged['researchers_per_million'] = merged['researchers_per_million'].fillna(merged['researchers_per_million_wb'])
    
    # 添加国家名称
    merged['country_cn'] = merged['country_code'].map(
        lambda x: TARGET_COUNTRIES.get(x, {}).get('name_cn', x)
    )
    merged['country_en'] = merged['country_code'].map(
        lambda x: TARGET_COUNTRIES.get(x, {}).get('name_en', x)
    )
    
    # 过滤目标国家和时间范围
    merged = merged[merged['country_code'].isin(TARGET_COUNTRIES.keys())]
    merged = merged[merged['year'].notna()]
    merged = merged[(merged['year'] >= 2010) & (merged['year'] <= 2024)]
    
    # 排序
    merged = merged.sort_values(['country_code', 'year']).reset_index(drop=True)
    
    print(f"  ✓ 合并完成: {len(merged)} 行")
    
    return merged


def generate_summary(df: pd.DataFrame) -> str:
    """生成数据摘要"""
    summary = []
    summary.append("=" * 70)
    summary.append("合并数据集摘要报告")
    summary.append("=" * 70)
    summary.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    summary.append("\n" + "-" * 50)
    summary.append("数据概览")
    summary.append("-" * 50)
    summary.append(f"总记录数: {len(df)}")
    summary.append(f"覆盖国家: {df['country_code'].nunique()}")
    summary.append(f"时间范围: {df['year'].min()} - {df['year'].max()}")
    summary.append(f"指标数量: {len([c for c in df.columns if c not in ['country_code', 'year', 'country_cn', 'country_en']])}")
    
    summary.append("\n" + "-" * 50)
    summary.append("各国数据覆盖年份")
    summary.append("-" * 50)
    for code in sorted(df['country_code'].unique()):
        country_data = df[df['country_code'] == code]
        years = sorted(country_data['year'].dropna().unique())
        cn_name = TARGET_COUNTRIES.get(code, {}).get('name_cn', code)
        summary.append(f"{cn_name:<8} ({code}): {min(years)}-{max(years)}, 共 {len(years)} 年")
    
    summary.append("\n" + "-" * 50)
    summary.append("指标列表及非空率")
    summary.append("-" * 50)
    indicator_cols = [c for c in df.columns if c not in ['country_code', 'year', 'country_cn', 'country_en', 'country_name']]
    for col in indicator_cols:
        non_null_pct = df[col].notna().sum() / len(df) * 100
        summary.append(f"• {col}: {non_null_pct:.1f}% 非空")
    
    return "\n".join(summary)


def save_results(df: pd.DataFrame, output_dir: str = "merged_data"):
    """保存结果"""
    print("\n[4] 保存结果...")
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 保存完整宽表CSV
    csv_path = f"{output_dir}/rd_innovation_wide_{timestamp}.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"  - {csv_path}")
    
    # 2. 保存Excel（带多个sheet）
    excel_path = f"{output_dir}/rd_innovation_wide_{timestamp}.xlsx"
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # 主数据表
        df.to_excel(writer, sheet_name='完整数据', index=False)
        
        # 按国家分sheet
        for code in sorted(df['country_code'].unique()):
            country_df = df[df['country_code'] == code].copy()
            cn_name = TARGET_COUNTRIES.get(code, {}).get('name_cn', code)
            sheet_name = f"{cn_name}_{code}"[:31]  # Excel sheet名最长31字符
            country_df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"  - {excel_path}")
    
    # 3. 保存摘要报告
    summary = generate_summary(df)
    summary_path = f"{output_dir}/summary_report.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"  - {summary_path}")
    
    print("\n" + summary)
    
    return csv_path, excel_path


def main():
    print("=" * 70)
    print("数据集合并与宽表生成")
    print("=" * 70)
    
    # 1. 加载UIS数据
    uis_df = load_uis_data()
    
    # 2. 加载World Bank AI数据
    wb_df = load_worldbank_ai_data()
    
    # 3. 合并数据
    merged_df = merge_all_datasets(uis_df, wb_df)
    
    if merged_df.empty:
        print("\n⚠️ 没有数据可处理")
        return
    
    # 4. 保存结果
    csv_path, excel_path = save_results(merged_df)
    
    print("\n" + "=" * 70)
    print("处理完成！")
    print("=" * 70)
    print(f"\n数据预览（前10行）:")
    # 选择关键列显示
    display_cols = ['country_code', 'country_cn', 'year', 'rd_expenditure_pct_gdp', 'researchers_per_million']
    display_cols = [c for c in display_cols if c in merged_df.columns]
    print(merged_df[display_cols].head(10).to_string())


if __name__ == "__main__":
    main()
