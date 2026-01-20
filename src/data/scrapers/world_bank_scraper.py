# -*- coding: utf-8 -*-
"""
世界银行数据爬取脚本
获取10个国家在人工智能领域发展相关的宏观指标数据
包括：R&D支出、科研人员、高等教育、互联网普及等
"""

import wbdata
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置区域 ====================

# 目标国家列表（ISO 3166-1 alpha-2 代码）
COUNTRIES = {
    'US': '美国',
    'CN': '中国',
    'GB': '英国',
    'DE': '德国',
    'KR': '韩国',
    'JP': '日本',
    'FR': '法国',
    'CA': '加拿大',
    'AE': '阿联酋',
    'IN': '印度'
}

# 世界银行指标代码 - 与AI发展和创新能力相关
INDICATORS = {
    # === R&D 研发投入 ===
    'GB.XPD.RSDV.GD.ZS': 'R&D支出占GDP比例(%)',
    
    # === 科研人员 ===
    'SP.POP.SCIE.RD.P6': '每百万人研究人员数量',
    
    # === 高等教育 ===
    'SE.TER.ENRR': '高等教育毛入学率(%)',
    'SE.TER.CUAT.BA.ZS': '拥有学士及以上学历人口比例(%)',
    'SE.TER.CUAT.MS.ZS': '拥有硕士及以上学历人口比例(%)',
    'SE.TER.CUAT.DO.ZS': '拥有博士学历人口比例(%)',
    
    # === ICT与互联网普及 ===
    'IT.NET.USER.ZS': '使用互联网人口比例(%)',
    'IT.NET.BBND.P2': '每100人固定宽带订阅数',
    'IT.CEL.SETS.P2': '每100人移动电话订阅数',
    
    # === 创新与知识产权 ===
    'IP.PAT.RESD': '专利申请数(居民)',
    'IP.PAT.NRES': '专利申请数(非居民)',
    'IP.TMK.TOTL': '商标申请总数',
    'BX.GSR.CCIS.ZS': 'ICT服务出口占服务出口比例(%)',
    
    # === 高科技产业 ===
    'TX.VAL.TECH.MF.ZS': '高科技出口占制成品出口比例(%)',
    'TX.VAL.TECH.CD': '高科技产品出口(现价美元)',
    
    # === 教育投入 ===
    'SE.XPD.TOTL.GD.ZS': '政府教育支出占GDP比例(%)',
    'SE.XPD.TERT.ZS': '高等教育支出占政府教育支出比例(%)',
    
    # === 基础设施与数字化 ===
    'IT.NET.SECR.P6': '每百万人安全互联网服务器数',
    
    # === 劳动力与人才 ===
    'SL.TLF.ADVN.ZS': '具有高等教育程度的劳动力比例(%)',
}

# 数据时间范围（获取最近15年的数据）
START_YEAR = 2010
END_YEAR = 2024

# ==================== 数据获取函数 ====================

def get_country_list():
    """获取所有可用的国家列表"""
    print("正在获取国家列表...")
    countries = wbdata.get_country()
    return countries

def search_indicators(keyword):
    """根据关键词搜索可用指标"""
    print(f"正在搜索包含 '{keyword}' 的指标...")
    indicators = wbdata.search_indicators(keyword)
    return indicators

def fetch_single_indicator(indicator_code, indicator_name, countries_list):
    """获取单个指标的数据"""
    try:
        print(f"  正在获取: {indicator_name} ({indicator_code})")
        
        # 获取数据
        data = wbdata.get_dataframe(
            {indicator_code: indicator_name},
            country=countries_list
        )
        
        if data is not None and not data.empty:
            return data
        else:
            print(f"    ⚠ 无数据返回")
            return None
            
    except Exception as e:
        print(f"    ✗ 获取失败: {str(e)}")
        return None

def fetch_all_indicators():
    """获取所有指标数据"""
    print("\n" + "="*60)
    print("开始获取世界银行数据")
    print("="*60)
    
    countries_list = list(COUNTRIES.keys())
    all_data = {}
    
    for indicator_code, indicator_name in INDICATORS.items():
        df = fetch_single_indicator(indicator_code, indicator_name, countries_list)
        if df is not None:
            all_data[indicator_code] = {
                'name': indicator_name,
                'data': df
            }
    
    return all_data

def create_combined_dataframe(all_data):
    """将所有数据合并为一个综合DataFrame"""
    print("\n正在整合数据...")
    
    combined_frames = []
    
    for indicator_code, info in all_data.items():
        df = info['data'].copy()
        if not df.empty:
            # 重置索引
            df_reset = df.reset_index()
            # 重命名列
            df_reset.columns = ['国家', '年份', info['name']]
            combined_frames.append(df_reset)
    
    if not combined_frames:
        return None
    
    # 合并所有数据
    result = combined_frames[0]
    for df in combined_frames[1:]:
        result = pd.merge(result, df, on=['国家', '年份'], how='outer')
    
    # 添加中文国家名
    result['国家中文'] = result['国家'].map(COUNTRIES)
    
    # 按国家和年份排序
    result = result.sort_values(['国家', '年份'], ascending=[True, False])
    
    return result

def save_data(all_data, combined_df):
    """保存数据到文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 保存综合数据
    if combined_df is not None:
        filename = f'd:/华数杯/AI_development_indicators_{timestamp}.csv'
        combined_df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"\n✓ 综合数据已保存: {filename}")
    
    # 2. 保存各指标的详细数据
    excel_filename = f'd:/华数杯/AI_development_indicators_detail_{timestamp}.xlsx'
    
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        # 写入综合表
        if combined_df is not None:
            combined_df.to_excel(writer, sheet_name='综合数据', index=False)
        
        # 写入各指标详细数据
        for indicator_code, info in all_data.items():
            df = info['data']
            if not df.empty:
                # 清理sheet名称（Excel sheet名称限制31字符）
                sheet_name = info['name'][:31].replace('/', '_').replace('\\', '_')
                sheet_name = sheet_name.replace('[', '').replace(']', '')
                sheet_name = sheet_name.replace('*', '').replace('?', '')
                sheet_name = sheet_name.replace(':', '_')
                
                df_save = df.reset_index()
                df_save.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"✓ 详细数据已保存: {excel_filename}")
    
    return filename, excel_filename

def generate_summary_report(all_data, combined_df):
    """生成数据摘要报告"""
    print("\n" + "="*60)
    print("数据摘要报告")
    print("="*60)
    
    print(f"\n目标国家: {', '.join(COUNTRIES.values())}")
    print(f"数据年份范围: {START_YEAR} - {END_YEAR}")
    
    print(f"\n成功获取的指标数量: {len(all_data)}/{len(INDICATORS)}")
    print("\n获取成功的指标列表:")
    for indicator_code, info in all_data.items():
        df = info['data']
        if not df.empty:
            # 获取数据覆盖信息
            non_null_count = df.notna().sum().sum()
            print(f"  ✓ {info['name']}: {non_null_count} 个有效数据点")
    
    # 打印未能获取的指标
    failed_indicators = set(INDICATORS.keys()) - set(all_data.keys())
    if failed_indicators:
        print("\n未能获取的指标:")
        for code in failed_indicators:
            print(f"  ✗ {INDICATORS[code]} ({code})")
    
    # 打印最新年份各国数据概览
    if combined_df is not None and not combined_df.empty:
        print("\n各国最新数据概览:")
        print("-" * 40)
        
        # 获取每个国家最新年份的数据
        latest_data = combined_df.groupby('国家').first().reset_index()
        
        for _, row in latest_data.iterrows():
            country_cn = COUNTRIES.get(row['国家'], row['国家'])
            print(f"\n{country_cn}:")
            for col in latest_data.columns:
                if col not in ['国家', '国家中文', '年份']:
                    value = row[col]
                    if pd.notna(value):
                        if isinstance(value, float):
                            print(f"  {col}: {value:.2f}")
                        else:
                            print(f"  {col}: {value}")

def explore_available_indicators():
    """探索可用的相关指标（用于发现新指标）"""
    print("\n" + "="*60)
    print("探索相关指标")
    print("="*60)
    
    keywords = ['research', 'R&D', 'education', 'internet', 'patent', 
                'technology', 'innovation', 'science', 'ICT', 'broadband']
    
    all_found = []
    for keyword in keywords:
        try:
            results = wbdata.search_indicators(keyword)
            if results:
                for r in results:
                    all_found.append({
                        'keyword': keyword,
                        'id': r.get('id', ''),
                        'name': r.get('name', '')
                    })
        except:
            pass
    
    if all_found:
        df_indicators = pd.DataFrame(all_found)
        df_indicators.drop_duplicates(subset=['id'], inplace=True)
        df_indicators.to_csv('d:/华数杯/available_indicators.csv', 
                            index=False, encoding='utf-8-sig')
        print(f"✓ 找到 {len(df_indicators)} 个相关指标，已保存到 available_indicators.csv")
    
    return all_found

def get_latest_available_data():
    """获取各指标最新可用年份的数据"""
    print("\n" + "="*60)
    print("获取最新可用数据")
    print("="*60)
    
    countries_list = list(COUNTRIES.keys())
    latest_data = {}
    
    for indicator_code, indicator_name in INDICATORS.items():
        try:
            print(f"正在获取: {indicator_name}")
            
            # 使用 get_data 获取数据（返回更详细的信息）
            data = wbdata.get_data(indicator_code, country=countries_list)
            
            if data:
                # 处理返回的数据
                for item in data:
                    country = item.get('country', {}).get('id', '')
                    year = item.get('date', '')
                    value = item.get('value')
                    
                    if value is not None and country in COUNTRIES:
                        key = (country, indicator_code)
                        if key not in latest_data:
                            latest_data[key] = {
                                '国家': country,
                                '国家中文': COUNTRIES[country],
                                '指标代码': indicator_code,
                                '指标名称': indicator_name,
                                '年份': year,
                                '数值': value
                            }
                        # 只保留最新年份
                        elif year > latest_data[key]['年份']:
                            latest_data[key]['年份'] = year
                            latest_data[key]['数值'] = value
                            
        except Exception as e:
            print(f"  ✗ 获取失败: {str(e)}")
    
    if latest_data:
        df = pd.DataFrame(list(latest_data.values()))
        return df
    return None

# ==================== 主程序 ====================

def main():
    """主程序入口"""
    print("="*60)
    print("世界银行数据爬取工具 - AI发展指标")
    print("="*60)
    print(f"目标国家: {', '.join(COUNTRIES.values())}")
    print(f"指标数量: {len(INDICATORS)}")
    print("="*60)
    
    # 1. 获取所有指标数据
    all_data = fetch_all_indicators()
    
    if not all_data:
        print("\n✗ 未能获取任何数据，请检查网络连接或指标代码")
        return
    
    # 2. 整合数据
    combined_df = create_combined_dataframe(all_data)
    
    # 3. 保存数据
    csv_file, excel_file = save_data(all_data, combined_df)
    
    # 4. 生成摘要报告
    generate_summary_report(all_data, combined_df)
    
    # 5. 尝试获取最新数据
    print("\n正在获取各指标最新可用数据...")
    latest_df = get_latest_available_data()
    if latest_df is not None and not latest_df.empty:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        latest_file = f'd:/华数杯/AI_latest_data_{timestamp}.csv'
        latest_df.to_csv(latest_file, index=False, encoding='utf-8-sig')
        print(f"✓ 最新数据已保存: {latest_file}")
    
    print("\n" + "="*60)
    print("数据获取完成!")
    print("="*60)
    
    return combined_df, all_data

if __name__ == "__main__":
    # 安装必要的库（如果尚未安装）
    # pip install wbdata pandas openpyxl
    
    try:
        df, data = main()
        
        # 可选：探索更多可用指标
        # explore_available_indicators()
        
    except ImportError as e:
        print(f"缺少必要的库: {e}")
        print("请运行以下命令安装:")
        print("pip install wbdata pandas openpyxl")
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()
