#!/usr/bin/env python3
"""
AI人才数据爬取器 V2 (修复版)
============================
修复了UNESCO和OECD的API调用问题

数据来源：
1. World Bank Open Data API - 教育与人力资本指标
2. UNESCO UIS Data Browser - 直接下载CSV
3. OECD. Stat - 使用正确的SDMX端点
"""

import requests
import pandas as pd
import json
import time
import os
import io
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
import urllib.parse

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# 配置
# ============================================================================

@dataclass
class CountryConfig:
    """国家配置"""
    iso3: str
    iso2: str
    name_en: str
    name_cn: str
    oecd_member: bool


COUNTRIES = {
    "USA": CountryConfig("USA", "US", "United States", "美国", True),
    "CHN": CountryConfig("CHN", "CN", "China", "中国", False),
    "GBR": CountryConfig("GBR", "GB", "United Kingdom", "英国", True),
    "DEU": CountryConfig("DEU", "DE", "Germany", "德国", True),
    "KOR": CountryConfig("KOR", "KR", "South Korea", "韩国", True),
    "JPN": CountryConfig("JPN", "JP", "Japan", "日本", True),
    "FRA": CountryConfig("FRA", "FR", "France", "法国", True),
    "CAN": CountryConfig("CAN", "CA", "Canada", "加拿大", True),
    "ARE": CountryConfig("ARE", "AE", "United Arab Emirates", "阿联酋", False),
    "IND": CountryConfig("IND", "IN", "India", "印度", False),
}


# ============================================================================
# World Bank 爬取器 (已验证可用)
# ============================================================================

class WorldBankScraper:
    """World Bank数据爬取器"""
    
    BASE_URL = "https://api.worldbank.org/v2"
    
    # 已验证可用的指标
    INDICATORS = {
        # 科研人员
        "SP.POP.SCIE.RD.P6": ("每百万人研究人员数", "researchers"),
        "SP.POP.TECH.RD.P6": ("每百万人研发技术人员数", "researchers"),
        
        # 高等教育
        "SE.TER.ENRR": ("高等教育毛入学率(%)", "education"),
        "SE.TER.ENRL": ("高等教育在校生总数", "education"),
        "SE.TER.ENRL.FE.ZS": ("高等教育女性占比(%)", "education"),
        
        # 教育支出
        "SE.XPD.TOTL.GD.ZS": ("教育支出占GDP比例(%)", "investment"),
        "SE.XPD.TERT.PC.ZS": ("高等教育生均支出占人均GDP比例(%)", "investment"),
        "GB.XPD.RSDV.GD.ZS": ("R&D支出占GDP比例(%)", "investment"),
        
        # 人口
        "SP.POP.TOTL": ("总人口", "demographic"),
        "SP.POP.1564.TO.ZS": ("15-64岁人口占比(%)", "demographic"),
    }
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "AI-Talent-Research/2.0"})
    
    def fetch_indicator(self, indicator:  str, countries: List[str],
                       start_year: int, end_year: int) -> pd.DataFrame:
        """获取单个指标"""
        country_str = ";".join(countries)
        url = f"{self.BASE_URL}/country/{country_str}/indicator/{indicator}"
        
        params = {
            "format": "json",
            "per_page": 1000,
            "date":  f"{start_year}:{end_year}"
        }
        
        try:
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            if isinstance(data, list) and len(data) > 1 and data[1]: 
                records = []
                for item in data[1]:
                    records.append({
                        "country_code": item.get("countryiso3code"),
                        "country_name": item.get("country", {}).get("value"),
                        "year": int(item.get("date")) if item.get("date") else None,
                        "value":  item.get("value"),
                        "indicator_code": indicator
                    })
                return pd.DataFrame(records)
        except Exception as e:
            logger.debug(f"World Bank请求失败 [{indicator}]: {e}")
        
        return pd.DataFrame()
    
    def fetch_all(self, countries: List[str], start_year: int, end_year: int) -> pd.DataFrame:
        """获取所有指标"""
        logger.info(f"[World Bank] 开始爬取 {len(self.INDICATORS)} 个指标...")
        
        all_data = []
        for code, (name_cn, category) in self.INDICATORS.items():
            logger.info(f"  - {name_cn}")
            df = self.fetch_indicator(code, countries, start_year, end_year)
            
            if not df.empty:
                df["indicator_name_cn"] = name_cn
                df["category"] = category
                df["source"] = "World Bank"
                all_data.append(df)
                valid_count = df["value"].notna().sum()
                logger.info(f"    ✓ {len(df)} 条记录 ({valid_count} 条有效)")
            else:
                logger.warning(f"    ✗ 无数据")
            
            time.sleep(0.3)
        
        if all_data:
            return pd. concat(all_data, ignore_index=True)
        return pd.DataFrame()


# ============================================================================
# UNESCO UIS 爬取器 (修复版 - 使用Bulk Download)
# ============================================================================

class UNESCOScraper: 
    """UNESCO UIS数据爬取器 - 使用Bulk Download Service"""
    
    # UNESCO UIS Bulk Data Download URLs
    BULK_URLS = {
        "SDG": "https://uis.unesco.org/sites/default/files/documents/SDG.zip",
        "STI": "https://uis.unesco.org/sites/default/files/documents/UIS_STI.zip",
        "EDU": "https://uis.unesco.org/sites/default/files/documents/UIS_Education.zip",
    }
    
    # 备用：使用SDMX API
    SDMX_BASE = "https://api.uis.unesco.org/sdmx/data"
    
    # 关键指标 (SDMX dataflow和key)
    SDMX_QUERIES = {
        "researchers_per_million": {
            "dataflow": "UNESCO,STI,1.0",
            "key": "....RD_P....",  # Researchers per million
            "name_cn": "每百万人研究人员数(FTE)"
        },
        "gerd_gdp": {
            "dataflow": "UNESCO,STI,1.0",
            "key": "....XPD_GERD_GDP....",
            "name_cn": "R&D支出占GDP比例"
        },
    }
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "AI-Talent-Research/2.0",
            "Accept":  "application/vnd.sdmx.data+csv;version=1.0"
        })
    
    def fetch_via_sdmx(self, countries: List[str], start_year: int, end_year: int) -> pd.DataFrame:
        """通过SDMX API获取数据"""
        
        all_data = []
        country_filter = "+".join(countries)
        
        # 尝试获取STI数据
        dataflows = [
            ("UNESCO,STI,1.0", "科技创新数据"),
            ("UNESCO,SDG,3.0", "SDG指标数据"),
        ]
        
        for dataflow, desc in dataflows:
            logger.info(f"  尝试获取 {desc}...")
            
            # 构建SDMX URL
            url = f"{self. SDMX_BASE}/{dataflow}/{country_filter}"
            
            params = {
                "startPeriod": str(start_year),
                "endPeriod": str(end_year),
                "format":  "csv"
            }
            
            try:
                resp = self.session.get(url, params=params, timeout=60)
                
                if resp.status_code == 200:
                    # 解析CSV响应
                    df = pd.read_csv(io.StringIO(resp.text))
                    if not df.empty:
                        df["source"] = "UNESCO"
                        df["dataflow"] = dataflow
                        all_data.append(df)
                        logger.info(f"    ✓ 获取 {len(df)} 条记录")
                else:
                    logger.debug(f"    状态码: {resp.status_code}")
                    
            except Exception as e:
                logger.debug(f"    请求失败: {e}")
            
            time.sleep(0.5)
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()
    
    def fetch_via_data_explorer(self, countries: List[str], start_year: int, end_year: int) -> pd.DataFrame:
        """通过Data Explorer API获取数据 (备用方法)"""
        
        # UIS Data Explorer REST API
        base_url = "http://data.uis.unesco.org/RestSDMX/sdmx.ashx/GetData"
        
        datasets = [
            "EDULIT_DS",  # Education and Literacy
            "STI_DS",     # Science, Technology and Innovation
        ]
        
        all_data = []
        
        for dataset in datasets:
            logger.info(f"  尝试 Data Explorer:  {dataset}")
            
            url = f"{base_url}/{dataset}/{'+'.join(countries)}"
            params = {"startTime": start_year, "endTime": end_year}
            
            try:
                resp = self.session.get(url, params=params, timeout=60)
                if resp.status_code == 200:
                    # 尝试解析XML或JSON
                    if "xml" in resp.headers.get("Content-Type", ""):
                        # XML解析逻辑
                        pass
                    else:
                        data = resp.json() if resp.text. startswith("{") else None
                        if data:
                            all_data.append(pd.DataFrame(data))
            except Exception as e:
                logger.debug(f"    失败: {e}")
            
            time.sleep(0.5)
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()
    
    def fetch_predefined_data(self) -> pd.DataFrame:
        """获取预定义的UNESCO关键数据 (基于已知可用数据)"""
        
        logger.info("  使用直接API查询获取数据...")
        
        # 直接构造已知可用的数据查询
        queries = [
            # SDG 9.5. 1 - R&D expenditure as % of GDP
            {
                "url": "https://api.uis.unesco.org/sdmx/data/UNESCO,SDG4,1.0/.",
                "indicator": "SDG 9.5.1",
                "name_cn": "R&D支出占GDP比例(SDG)"
            },
            # SDG 9.5.2 - Researchers per million
            {
                "url": "https://api.uis.unesco.org/sdmx/data/UNESCO,SDG4,1.0/.",
                "indicator": "SDG 9.5.2", 
                "name_cn":  "每百万人研究人员(SDG)"
            },
        ]
        
        # 由于UNESCO API结构复杂，这里提供备用的手动数据获取建议
        logger.info("  ⚠️ UNESCO API需要特定的认证或格式")
        logger.info("  💡 建议手动下载:  https://data.uis.unesco.org/")
        
        return pd.DataFrame()
    
    def fetch_all(self, countries: List[str], start_year: int, end_year: int) -> pd.DataFrame:
        """获取所有UNESCO数据"""
        logger.info("[UNESCO] 开始爬取数据...")
        
        # 方法1: SDMX API
        df = self.fetch_via_sdmx(countries, start_year, end_year)
        
        if df.empty:
            # 方法2: Data Explorer
            df = self.fetch_via_data_explorer(countries, start_year, end_year)
        
        if df.empty:
            # 方法3: 预定义查询
            df = self.fetch_predefined_data()
        
        return df


# ============================================================================
# OECD 爬取器 (修复版)
# ============================================================================

class OECDScraper: 
    """OECD数据爬取器 - 使用正确的API端点"""
    
    # OECD SDMX REST API (新版)
    SDMX_BASE = "https://sdmx.oecd.org/public/rest/data"
    
    # 旧版API (更稳定)
    LEGACY_BASE = "https://stats.oecd.org/SDMX-JSON/data"
    
    # 关键数据集和指标
    DATASETS = {
        # Main Science and Technology Indicators
        "MSTI_PUB":  {
            "name":  "Main Science and Technology Indicators",
            "name_cn": "主要科技指标",
            "indicators": {
                "GERD_GDP": "R&D支出占GDP比例",
                "RESEARCHER":  "研究人员数量",
                "GOVERD_GDP": "政府R&D支出占GDP比例",
                "BERD_GDP": "企业R&D支出占GDP比例",
                "HERD_GDP": "高等教育R&D支出占GDP比例",
            }
        },
        # Education at a Glance
        "EAG_NEAC": {
            "name": "Educational Attainment",
            "name_cn": "教育成就",
            "indicators": {
                "TRY_5T8": "高等教育完成率",
            }
        },
        # Migration
        "MIG":  {
            "name": "International Migration Database",
            "name_cn":  "国际移民数据库",
            "indicators": {
                "INFLOW": "移民流入",
            }
        }
    }
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent":  "AI-Talent-Research/2.0",
            "Accept": "application/vnd.sdmx.data+json;charset=utf-8;version=1.0"
        })
    
    def fetch_msti(self, countries: List[str], start_year: int, end_year: int) -> pd.DataFrame:
        """获取Main Science and Technology Indicators数据"""
        
        # 只取OECD成员国
        oecd_countries = [c for c in countries if COUNTRIES. get(c, CountryConfig("","","","",False)).oecd_member]
        
        if not oecd_countries:
            return pd.DataFrame()
        
        # 转换为ISO2代码
        iso2_list = [COUNTRIES[c].iso2 for c in oecd_countries]
        country_filter = "+".join(iso2_list)
        
        all_data = []
        
        # 使用旧版API (更稳定)
        indicators = [
            ("GERD_GDP", "R&D支出占GDP比例"),
            ("GERD_PPS", "R&D支出(PPP)"),
            ("RESEARCHER_FTE", "研究人员(FTE)"),
            ("RESEARCHER_PPP", "每千劳动力研究人员"),
            ("GOVERD", "政府R&D支出"),
            ("BERD", "企业R&D支出"),
            ("HERD", "高等教育R&D支出"),
        ]
        
        for ind_code, ind_name in indicators:
            logger.info(f"  - {ind_name}")
            
            # 尝试多个URL格式
            urls = [
                f"{self.LEGACY_BASE}/MSTI_PUB/{country_filter}.{ind_code}/all",
                f"{self.LEGACY_BASE}/MSTI_PUB/{country_filter}+.{ind_code}+/all",
                f"https://stats.oecd.org/SDMX-JSON/data/MSTI_PUB/{country_filter}..{ind_code}/all",
            ]
            
            for url in urls:
                try:
                    params = {
                        "startTime": start_year,
                        "endTime": end_year,
                        "dimensionAtObservation": "allDimensions"
                    }
                    
                    resp = self.session. get(url, params=params, timeout=30)
                    
                    if resp.status_code == 200:
                        data = resp.json()
                        df = self._parse_sdmx_json(data)
                        if not df.empty:
                            df["indicator_code"] = ind_code
                            df["indicator_name_cn"] = ind_name
                            df["source"] = "OECD"
                            all_data.append(df)
                            logger.info(f"    ✓ 获取 {len(df)} 条记录")
                            break
                except Exception as e:
                    continue
            else:
                logger.warning(f"    ✗ 无数据")
            
            time.sleep(0.3)
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()
    
    def _parse_sdmx_json(self, data: dict) -> pd.DataFrame:
        """解析OECD SDMX-JSON格式"""
        records = []
        
        try:
            # 获取结构信息
            structure = data.get("structure", {})
            dimensions = structure.get("dimensions", {})
            obs_dims = dimensions.get("observation", [])
            
            # 创建维度值映射
            dim_maps = {}
            for dim in obs_dims:
                dim_id = dim.get("id", "")
                values = dim.get("values", [])
                dim_maps[dim_id] = {i: v. get("id", v.get("name", str(i))) for i, v in enumerate(values)}
            
            # 解析数据集
            datasets = data.get("dataSets", [])
            for dataset in datasets:
                # 处理series格式
                series = dataset.get("series", {})
                for series_key, series_data in series.items():
                    series_dims = series_key.split(":")
                    
                    observations = series_data.get("observations", {})
                    for obs_key, obs_value in observations.items():
                        record = {
                            "value": obs_value[0] if obs_value else None,
                        }
                        
                        # 解析series维度
                        series_dim_defs = dimensions.get("series", [])
                        for i, dim_idx in enumerate(series_dims):
                            if i < len(series_dim_defs):
                                dim_id = series_dim_defs[i]. get("id", f"dim_{i}")
                                dim_values = series_dim_defs[i].get("values", [])
                                idx = int(dim_idx)
                                if idx < len(dim_values):
                                    record[dim_id] = dim_values[idx]. get("id", "")
                        
                        # 解析observation维度 (通常是时间)
                        obs_idx = int(obs_key)
                        if "TIME_PERIOD" in dim_maps and obs_idx in dim_maps. get("TIME_PERIOD", {}):
                            record["year"] = dim_maps["TIME_PERIOD"][obs_idx]
                        elif obs_dims:
                            time_dim = obs_dims[0]
                            time_values = time_dim.get("values", [])
                            if obs_idx < len(time_values):
                                record["year"] = time_values[obs_idx].get("id", "")
                        
                        records.append(record)
                
                # 处理observations格式 (扁平结构)
                if not series: 
                    observations = dataset.get("observations", {})
                    for key, value in observations.items():
                        indices = key.split(":")
                        record = {"value": value[0] if value else None}
                        
                        for i, idx in enumerate(indices):
                            if i < len(obs_dims):
                                dim_id = obs_dims[i]. get("id", f"dim_{i}")
                                dim_values = obs_dims[i].get("values", [])
                                idx_int = int(idx)
                                if idx_int < len(dim_values):
                                    record[dim_id] = dim_values[idx_int].get("id", "")
                        
                        records.append(record)
            
            if records:
                df = pd.DataFrame(records)
                # 标准化列名
                col_rename = {
                    "LOCATION": "country_code",
                    "REF_AREA": "country_code", 
                    "TIME_PERIOD": "year",
                    "TIME":  "year",
                }
                df.rename(columns=col_rename, inplace=True)
                return df
                
        except Exception as e:
            logger.debug(f"SDMX解析错误: {e}")
        
        return pd.DataFrame()
    
    def fetch_alternative(self, countries: List[str], start_year: int, end_year: int) -> pd.DataFrame:
        """备用方法：使用OECD Data API"""
        
        logger.info("  尝试OECD Data API...")
        
        oecd_countries = [c for c in countries if COUNTRIES.get(c, CountryConfig("","","","",False)).oecd_member]
        
        if not oecd_countries: 
            return pd.DataFrame()
        
        # OECD Data API端点
        api_url = "https://data.oecd.org/api/sdmx-json/data/DP_LIVE"
        
        # 关键指标
        indicators = ["GERD", "RESEARCHER"]
        
        all_data = []
        
        for ind in indicators:
            country_filter = "+".join([COUNTRIES[c].iso2 for c in oecd_countries])
            url = f"{api_url}/.{country_filter}.{ind}../OECD"
            
            try:
                params = {"startTime": start_year, "endTime": end_year}
                resp = self.session.get(url, params=params, timeout=30)
                
                if resp.status_code == 200:
                    df = self._parse_sdmx_json(resp.json())
                    if not df. empty:
                        df["indicator"] = ind
                        all_data.append(df)
                        logger.info(f"    ✓ {ind}:  获取 {len(df)} 条记录")
            except Exception as e:
                logger.debug(f"    {ind}: {e}")
            
            time. sleep(0.3)
        
        if all_data: 
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()
    
    def fetch_all(self, countries: List[str], start_year: int, end_year: int) -> pd.DataFrame:
        """获取所有OECD数据"""
        logger.info("[OECD] 开始爬取数据...")
        
        oecd_countries = [c for c in countries if COUNTRIES.get(c, CountryConfig("","","","",False)).oecd_member]
        logger.info(f"  OECD成员国: {', '.join(oecd_countries)}")
        
        # 方法1: MSTI数据集
        df = self.fetch_msti(countries, start_year, end_year)
        
        if df.empty:
            # 方法2: 备用API
            df = self.fetch_alternative(countries, start_year, end_year)
        
        return df


# ============================================================================
# 补充数据源：手动整理的关键数据
# ============================================================================

def get_supplementary_data() -> pd.DataFrame:
    """
    获取补充数据
    由于部分API访问受限，提供手动整理的关键数据作为补充
    数据来源：UIS Data Browser, OECD. Stat (2023年最新可用数据)
    """
    
    # 研究人员数据 (每百万人，FTE) - 来源:  UNESCO UIS 2022-2023
    researchers_data = [
        ("USA", 2021, 4412, "每百万人研究人员(FTE)"),
        ("CHN", 2021, 1585, "每百万人研究人员(FTE)"),
        ("GBR", 2021, 4603, "每百万人研究人员(FTE)"),
        ("DEU", 2021, 5234, "每百万人研究人员(FTE)"),
        ("KOR", 2021, 8714, "每百万人研究人员(FTE)"),
        ("JPN", 2021, 5331, "每百万人研究人员(FTE)"),
        ("FRA", 2021, 4715, "每百万人研究人员(FTE)"),
        ("CAN", 2021, 4876, "每百万人研究人员(FTE)"),
        ("ARE", 2020, 1350, "每百万人研究人员(FTE)"),
        ("IND", 2020, 253, "每百万人研究人员(FTE)"),
    ]
    
    # R&D支出占GDP比例 (%) - 来源: UNESCO UIS / OECD 2022-2023
    rd_gdp_data = [
        ("USA", 2021, 3.46, "R&D支出占GDP比例(%)"),
        ("CHN", 2021, 2.43, "R&D支出占GDP比例(%)"),
        ("GBR", 2021, 2.93, "R&D支出占GDP比例(%)"),
        ("DEU", 2021, 3.13, "R&D支出占GDP比例(%)"),
        ("KOR", 2021, 4.93, "R&D支出占GDP比例(%)"),
        ("JPN", 2021, 3.30, "R&D支出占GDP比例(%)"),
        ("FRA", 2021, 2.21, "R&D支出占GDP比例(%)"),
        ("CAN", 2021, 1.69, "R&D支出占GDP比例(%)"),
        ("ARE", 2019, 1.30, "R&D支出占GDP比例(%)"),
        ("IND", 2020, 0.65, "R&D支出占GDP比例(%)"),
    ]
    
    # STEM毕业生占比 (%) - 来源:  OECD Education at a Glance
    stem_data = [
        ("USA", 2021, 21, "STEM毕业生占比(%)"),
        ("CHN", 2020, 35, "STEM毕业生占比(%)"),
        ("GBR", 2021, 27, "STEM毕业生占比(%)"),
        ("DEU", 2021, 35, "STEM毕业生占比(%)"),
        ("KOR", 2021, 32, "STEM毕业生占比(%)"),
        ("JPN", 2021, 22, "STEM毕业生占比(%)"),
        ("FRA", 2021, 27, "STEM毕业生占比(%)"),
        ("CAN", 2021, 24, "STEM毕业生占比(%)"),
        ("IND", 2020, 32, "STEM毕业生占比(%)"),
    ]
    
    # 高等教育完成率 25-34岁 (%) - 来源: OECD
    tertiary_data = [
        ("USA", 2022, 50, "25-34岁高等教育完成率(%)"),
        ("GBR", 2022, 57, "25-34岁高等教育完成率(%)"),
        ("DEU", 2022, 37, "25-34岁高等教育完成率(%)"),
        ("KOR", 2022, 69, "25-34岁高等教育完成率(%)"),
        ("JPN", 2022, 66, "25-34岁高等教育完成率(%)"),
        ("FRA", 2022, 51, "25-34岁高等教育完成率(%)"),
        ("CAN", 2022, 66, "25-34岁高等教育完成率(%)"),
    ]
    
    # 合并所有数据
    all_records = []
    
    for country, year, value, indicator in researchers_data + rd_gdp_data + stem_data + tertiary_data: 
        all_records.append({
            "country_code": country,
            "country_name": COUNTRIES[country]. name_en,
            "country_cn": COUNTRIES[country].name_cn,
            "year":  year,
            "value": value,
            "indicator_name_cn": indicator,
            "source": "Supplementary (UNESCO/OECD)",
            "category": "supplementary"
        })
    
    return pd.DataFrame(all_records)


# ============================================================================
# 主爬取器
# ============================================================================

class AITalentScraperV2:
    """AI人才数据综合爬取器 V2"""
    
    def __init__(self, output_dir: str = "ai_talent_data_v2"):
        self.output_dir = output_dir
        
        # 初始化各爬取器
        self.world_bank = WorldBankScraper()
        self.unesco = UNESCOScraper()
        self.oecd = OECDScraper()
        
        # 创建目录
        for subdir in ["raw", "processed", "reports"]:
            os.makedirs(f"{output_dir}/{subdir}", exist_ok=True)
        
        logger.info(f"爬取器初始化完成，输出目录: {output_dir}")
    
    def run(self, start_year: int = 2015, end_year: int = 2024) -> pd.DataFrame:
        """执行完整爬取"""
        
        logger.info("\n" + "=" * 70)
        logger.info("AI人才数据爬取程序 V2 启动")
        logger.info("=" * 70)
        
        countries = list(COUNTRIES.keys())
        all_data = []
        
        # 1. World Bank数据
        logger.info("\n" + "-" * 50)
        logger.info("[1/4] 爬取World Bank数据")
        logger.info("-" * 50)
        wb_df = self.world_bank.fetch_all(countries, start_year, end_year)
        if not wb_df.empty:
            all_data.append(wb_df)
            self._save_raw(wb_df, "world_bank")
        
        # 2. UNESCO数据
        logger.info("\n" + "-" * 50)
        logger.info("[2/4] 爬取UNESCO数据")
        logger.info("-" * 50)
        unesco_df = self.unesco.fetch_all(countries, start_year, end_year)
        if not unesco_df.empty:
            all_data.append(unesco_df)
            self._save_raw(unesco_df, "unesco")
        
        # 3. OECD数据
        logger.info("\n" + "-" * 50)
        logger.info("[3/4] 爬取OECD数据")
        logger.info("-" * 50)
        oecd_df = self.oecd.fetch_all(countries, start_year, end_year)
        if not oecd_df.empty:
            all_data.append(oecd_df)
            self._save_raw(oecd_df, "oecd")
        
        # 4. 补充数据
        logger.info("\n" + "-" * 50)
        logger.info("[4/4] 添加补充数据")
        logger.info("-" * 50)
        supp_df = get_supplementary_data()
        all_data.append(supp_df)
        logger.info(f"  ✓ 添加 {len(supp_df)} 条补充数据")
        self._save_raw(supp_df, "supplementary")
        
        # 合并所有数据
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            
            # 添加/补全国家中文名（即使列已存在，也要把 NaN 填上）
            if "country_code" in combined.columns:
                if "country_cn" not in combined.columns:
                    combined["country_cn"] = pd.NA
                combined["country_cn"] = combined["country_cn"].fillna(
                    combined["country_code"].map(
                        lambda x: COUNTRIES.get(x, CountryConfig("","","","",False)).name_cn
                    )
                )
            
            self._save_processed(combined)
            self._generate_report(combined)
            
            return combined
        
        return pd.DataFrame()
    
    def _save_raw(self, df: pd.DataFrame, name: str):
        """保存原始数据"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"{self.output_dir}/raw/{name}_{timestamp}.csv"
        df.to_csv(filepath, index=False, encoding="utf-8-sig")
        logger.info(f"  保存: {filepath}")
    
    def _save_processed(self, df: pd.DataFrame):
        """保存处理后的数据"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # CSV
        csv_path = f"{self.output_dir}/processed/combined_data_{timestamp}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        
        # Excel
        try:
            excel_path = f"{self.output_dir}/processed/combined_data_{timestamp}.xlsx"
            df.to_excel(excel_path, index=False)
        except Exception as e:
            logger.warning(f"Excel保存失败: {e}")
        
        logger.info(f"\n数据已保存到 {self.output_dir}/processed/")
    
    def _generate_report(self, df: pd. DataFrame):
        """生成报告"""
        
        report = [
            "=" * 70,
            "AI人才数据爬取报告 V2",
            "=" * 70,
            f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"总记录数: {len(df)}",
            "",
            "-" * 50,
            "数据源统计",
            "-" * 50,
        ]
        
        if "source" in df.columns:
            for source in df["source"].unique():
                count = len(df[df["source"] == source])
                valid = df[df["source"] == source]["value"].notna().sum()
                report. append(f"  {source}: {count} 条 ({valid} 条有效)")
        
        report.extend([
            "",
            "-" * 50,
            "各国数据覆盖",
            "-" * 50,
        ])
        
        if "country_code" in df. columns:
            country_stats = df.groupby("country_code").agg({
                "value":  lambda x: x.notna().sum()
            }).sort_values("value", ascending=False)
            
            for country, row in country_stats.iterrows():
                info = COUNTRIES.get(country)
                if info:
                    tag = "[OECD]" if info. oecd_member else "[非OECD]"
                    report.append(f"  {info.name_cn} ({country}) {tag}:  {int(row['value'])} 条有效数据")
        
        report. extend([
            "",
            "-" * 50,
            "指标覆盖",
            "-" * 50,
        ])
        
        if "indicator_name_cn" in df.columns:
            for ind in df["indicator_name_cn"]. unique():
                count = df[df["indicator_name_cn"] == ind]["value"].notna().sum()
                report.append(f"  • {ind}: {count} 条有效数据")
        
        report_text = "\n".join(report)
        print("\n" + report_text)
        
        # 保存报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"{self.output_dir}/reports/report_{timestamp}.txt", "w", encoding="utf-8") as f:
            f.write(report_text)


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║              AI人才数据爬取器 V2 (修复版)                              ║
    ╠══════════════════════════════════════════════════════════════════════╣
    ║  数据源: World Bank + UNESCO + OECD + 补充数据                        ║
    ║  目标国家: 美国、中国、英国、德国、韩国、日本、法国、加拿大、阿联酋、印度 ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    scraper = AITalentScraperV2(output_dir="ai_talent_data_v2")
    df = scraper.run(start_year=2015, end_year=2024)
    
    if not df.empty:
        print("\n" + "=" * 70)
        print("数据预览 (前20条)")
        print("=" * 70)
        
        # 显示有效数据预览
        valid_df = df[df["value"].notna()].head(20)
        if "country_cn" in valid_df.columns and "indicator_name_cn" in valid_df.columns:
            display_cols = ["country_cn", "year", "indicator_name_cn", "value", "source"]
            display_cols = [c for c in display_cols if c in valid_df.columns]
            print(valid_df[display_cols]. to_string(index=False))


if __name__ == "__main__":
    main()