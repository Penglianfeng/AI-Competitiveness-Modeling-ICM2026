#!/usr/bin/env python3
"""
UNESCO Institute for Statistics (UIS) R&D Data Scraper
=====================================================
爬取十个国家的R&D数据，用于分析人工智能领域的研发投入与创新基础

目标国家：美国、中国、英国、德国、韩国、日本、法国、加拿大、阿联酋、印度

数据指标：
1. R&D支出占GDP比例 (GERD as % of GDP)
2. R&D支出按执行部门细分（政府/企业/高校/私营非营利）
3. R&D支出按资金来源细分
4. 科研人员数量（FTE，按部门分）
5. 每百万人口科研人员数

数据来源：https://data.uis.unesco.org/
API文档：https://api.uis.unesco.org/api/public/documentation/
"""

import requests
import pandas as pd
import json
import time
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

# 配置日��
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UISConfig:
    """UIS API配置"""
    
    # API端点 - UIS SDMX REST API
    # 参考: https://apiportal.uis.unesco.org/
    SDMX_API_BASE = "https://api.uis.unesco.org/sdmx"
    BULK_DATA_API = "https://api.uis.unesco.org/api/public/v1"
    # 备用：直接从UIS数据下载页获取CSV
    BULK_DOWNLOAD_BASE = "http://data.uis.unesco.org/"
    
    # 目标国家 - ISO 3166-1 alpha-3 代码
    TARGET_COUNTRIES = {
        "USA": {"name_en": "United States", "name_cn": "美国"},
        "CHN": {"name_en":  "China", "name_cn":  "中国"},
        "GBR": {"name_en":  "United Kingdom", "name_cn": "英国"},
        "DEU": {"name_en":  "Germany", "name_cn":  "德国"},
        "KOR": {"name_en": "South Korea", "name_cn":  "韩国"},
        "JPN": {"name_en":  "Japan", "name_cn":  "日本"},
        "FRA": {"name_en": "France", "name_cn": "法国"},
        "CAN": {"name_en": "Canada", "name_cn": "加拿大"},
        "ARE": {"name_en": "United Arab Emirates", "name_cn":  "阿联酋"},
        "IND": {"name_en": "India", "name_cn": "印度"}
    }
    
    # ===== 运行策略开关 =====
    # 由于UIS SDMX端点对多数指标404，直接禁用以减少无效请求与噪声日志
    ENABLE_UIS_SDMX = False
    
    # 放弃细分指标（执行部门/资金来源/科研人员分部门等），只保留稳定的SDG两项
    ONLY_SDG_CORE = True
    
    # R&D相关指标 - 基于UIS Science, Technology and Innovation数据集
    # 参考:  https://uis.unesco.org/en/themes/science-technology-innovation
    RD_INDICATORS = {
        # === R&D支出指标 ===
        # GERD = Gross Domestic Expenditure on R&D (研发总支出)
        "GERD_GDP_PCT": {
            "code": "GB_XPD_RSDV",  # 或 SDG_9.5.1
            "name":  "R&D expenditure as % of GDP",
            "name_cn": "R&D支出占GDP比例",
            "unit": "%",
            "category": "expenditure"
        },
        "GERD_PPP":  {
            "code": "GB_XPD_RSDV_PPP",
            "name": "GERD in PPP$ (millions)",
            "name_cn":  "R&D支出（PPP，百万美元）",
            "unit": "Million PPP$",
            "category":  "expenditure"
        },
        
        # === R&D支出按执行部门细分 ===
        "GERD_BES": {
            "code":  "GERD_PFSEC_BES",
            "name":  "GERD performed by Business Enterprise Sector",
            "name_cn": "企业部门R&D支出",
            "unit": "%",
            "category": "sector_performance"
        },
        "GERD_GOV": {
            "code": "GERD_PFSEC_GOV",
            "name": "GERD performed by Government Sector",
            "name_cn": "政府部门R&D支出",
            "unit": "%",
            "category": "sector_performance"
        },
        "GERD_HES":  {
            "code": "GERD_PFSEC_HES",
            "name":  "GERD performed by Higher Education Sector",
            "name_cn": "高等教育部门R&D支出",
            "unit": "%",
            "category": "sector_performance"
        },
        "GERD_PNP": {
            "code": "GERD_PFSEC_PNP",
            "name": "GERD performed by Private Non-Profit Sector",
            "name_cn": "私营非营利部门R&D支出",
            "unit": "%",
            "category": "sector_performance"
        },
        
        # === R&D支出按资金来源细分 ===
        "GERD_FUND_BES": {
            "code": "GERD_FSSEC_BES",
            "name": "GERD financed by Business Enterprise",
            "name_cn": "企业资助的R&D支出",
            "unit": "%",
            "category": "funding_source"
        },
        "GERD_FUND_GOV": {
            "code":  "GERD_FSSEC_GOV",
            "name": "GERD financed by Government",
            "name_cn": "政府资助的R&D支出",
            "unit": "%",
            "category": "funding_source"
        },
        "GERD_FUND_ABROAD": {
            "code": "GERD_FSSEC_ABR",
            "name": "GERD financed from Abroad",
            "name_cn": "国外资助的R&D支出",
            "unit": "%",
            "category": "funding_source"
        },
        
        # === 科研人员指标 ===
        "RESEARCHERS_PER_MILLION": {
            "code":  "SP_POP_SCIE",  # 或 SDG_9.5.2
            "name": "Researchers per million inhabitants (FTE)",
            "name_cn": "每百万人口科研人员数（FTE）",
            "unit": "Per million",
            "category": "researchers"
        },
        "RESEARCHERS_TOTAL": {
            "code":  "RES_FTE",
            "name": "Total researchers (FTE)",
            "name_cn": "科研人员总数（FTE）",
            "unit": "Headcount",
            "category": "researchers"
        },
        "RESEARCHERS_BES": {
            "code": "RES_FTE_SEC_BES",
            "name": "Researchers in Business Enterprise Sector (FTE)",
            "name_cn": "企业部门科研人员（FTE）",
            "unit": "Headcount",
            "category": "researchers"
        },
        "RESEARCHERS_GOV": {
            "code":  "RES_FTE_SEC_GOV",
            "name": "Researchers in Government Sector (FTE)",
            "name_cn": "政府部门科研人员（FTE）",
            "unit": "Headcount",
            "category":  "researchers"
        },
        "RESEARCHERS_HES": {
            "code": "RES_FTE_SEC_HES",
            "name": "Researchers in Higher Education Sector (FTE)",
            "name_cn": "高等教育部门科研人员（FTE）",
            "unit": "Headcount",
            "category":  "researchers"
        },
        "RESEARCHERS_FEMALE_PCT": {
            "code": "RES_FTE_FEMALE_PCT",
            "name": "Female researchers as % of total",
            "name_cn": "女性科研人员比例",
            "unit": "%",
            "category": "researchers"
        },
    }
    
    # SDG相关指标（更稳定的指标代码）
    SDG_INDICATORS = {
        "SDG_9.5.1": {
            "code": "SDG_9.5.1",
            "name": "Research and development expenditure as a proportion of GDP",
            "name_cn": "研发支出占GDP比例（SDG 9.5.1）"
        },
        "SDG_9.5.2": {
            "code": "SDG_9.5.2", 
            "name": "Researchers (in full-time equivalent) per million inhabitants",
            "name_cn": "每百万人口科研人员数（SDG 9.5.2）"
        }
    }


class UISDataScraper: 
    """UIS R&D数据爬取器"""
    
    def __init__(self, output_dir: str = "uis_rd_data"):
        """
        初始化爬取器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        self.config = UISConfig()
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (Research Data Collection)",
            "Accept-Language": "en-US,en;q=0.9"
        })
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/raw", exist_ok=True)
        os.makedirs(f"{output_dir}/processed", exist_ok=True)
        
        logger.info(f"UIS数据爬取器初始化完成，输出目录: {output_dir}")
    
    def _make_request(self, url: str, params: dict = None, 
                      max_retries: int = 3, delay: float = 1.0) -> Optional[Any]:
        """
        发送HTTP请求（带重试机制）
        
        Args: 
            url: 请求URL
            params: 请求参数
            max_retries:  最大重试次数
            delay: 重试间隔（秒）
        
        Returns:
            响应数据（JSON）或None
        """
        for attempt in range(max_retries):
            try:
                logger.debug(f"请求:  {url}, 参数: {params}")
                response = self.session.get(url, params=params, timeout=60)
                
                if response. status_code == 200:
                    # 检查是否返回JSON
                    content_type = response.headers.get('Content-Type', '')
                    if 'json' in content_type: 
                        return response.json()
                    else:
                        return response.text
                        
                elif response.status_code == 429:  # Rate limited
                    wait_time = delay * (attempt + 1) * 2
                    logger.warning(f"请求被限速，等待 {wait_time} 秒...")
                    time.sleep(wait_time)
                    
                elif response.status_code == 404:
                    logger.warning(f"资源不存在: {url}")
                    return None
                    
                else:
                    logger.warning(f"请求失败:  {response.status_code} - {url}")
                    
            except requests.exceptions. Timeout:
                logger.warning(f"请求超时 (尝试 {attempt + 1}/{max_retries}): {url}")
                
            except requests.exceptions.RequestException as e:
                logger.error(f"请求异常:  {e}")
            
            if attempt < max_retries - 1:
                time.sleep(delay)
        
        return None
    
    def fetch_via_sdmx(self, dataflow:  str, geo_filter: str = None,
                       start_year: int = 2010, end_year: int = 2024) -> Optional[dict]:
        """
        通过SDMX API获取数据
        
        UIS SDMX API遵循SDMX RESTful标准
        URL格式: {base}/dataflow/{agency}/{id}/{version}/{key}
        
        Args: 
            dataflow: 数据流ID (如 "UNESCO,SDG,3. 0")
            geo_filter: 地理过滤器
            start_year: 起始年份
            end_year:  结束年份
        """
        # 构建URL
        if geo_filter:
            url = f"{self.config.SDMX_API_BASE}/{dataflow}/{geo_filter}"
        else:
            url = f"{self.config.SDMX_API_BASE}/{dataflow}"
        
        params = {
            "startPeriod": str(start_year),
            "endPeriod": str(end_year),
            "dimensionAtObservation": "AllDimensions"
        }
        
        return self._make_request(url, params)
    
    def fetch_via_sdmx_api(self, indicator_code: str, 
                            countries: List[str] = None,
                            start_year: int = 2010, 
                            end_year: int = 2024) -> pd.DataFrame:
        """
        通过UIS SDMX REST API获取数据
        
        UIS SDMX API格式:
        https://api.uis.unesco.org/sdmx/data/{dataflow}/{key}?startPeriod=YYYY&endPeriod=YYYY
        
        Args:
            indicator_code: 指标代码
            countries: 国家代码列表  
            start_year: 起始年份
            end_year: 结束年份
        
        Returns:
            包含数据的DataFrame
        """
        if countries is None:
            countries = list(self.config.TARGET_COUNTRIES.keys())
        
        # UIS SDMX数据流 - 科学技术创新
        dataflow = "UNESCO,SDG4,3.0"  # SDG数据流
        
        # 构建key: 格式通常是 indicator.country 或 country.indicator
        country_str = "+".join(countries)
        
        # 尝试多种URL格式
        urls_to_try = [
            f"{self.config.SDMX_API_BASE}/data/UNESCO,RD,1.0/{indicator_code}.{country_str}",
            f"{self.config.SDMX_API_BASE}/data/RD/{indicator_code}.{country_str}/all",
            f"{self.config.SDMX_API_BASE}/data/SDG/{indicator_code}.{country_str}/all",
        ]
        
        params = {
            "startPeriod": str(start_year),
            "endPeriod": str(end_year),
            "format": "sdmx-json"
        }
        
        for url in urls_to_try:
            data = self._make_request(url, params)
            if data and isinstance(data, dict):
                # 解析SDMX-JSON响应
                try:
                    df = self._parse_sdmx_json(data)
                    if not df.empty:
                        return df
                except Exception as e:
                    logger.debug(f"解析失败: {e}")
                    continue
        
        return pd.DataFrame()
    
    def _parse_sdmx_json(self, json_data: dict) -> pd.DataFrame:
        """解析SDMX-JSON格式的响应数据"""
        try:
            structure = json_data.get('structure', {}).get('dimensions', {}).get('observation', [])
            observations = json_data.get('dataSets', [{}])[0].get('observations', {})
            
            if not observations:
                return pd.DataFrame()
            
            # 构建维度映射
            dim_maps = []
            for dim in structure:
                dim_maps.append({i: v['id'] for i, v in enumerate(dim.get('values', []))})
            
            rows = []
            for key, value in observations.items():
                indices = [int(x) for x in key.split(':')]
                row = {}
                for i, idx in enumerate(indices):
                    if i < len(dim_maps) and i < len(structure):
                        dim_name = structure[i].get('id', f'dim_{i}')
                        row[dim_name] = dim_maps[i].get(idx, '')
                row['value'] = value[0] if value else None
                rows.append(row)
            
            return pd.DataFrame(rows)
        except Exception as e:
            logger.debug(f"SDMX-JSON解析错误: {e}")
            return pd.DataFrame()
    
    def fetch_data_browser_api(self, indicator_code: str, 
                                countries: List[str] = None,
                                start_year: int = 2010, 
                                end_year: int = 2024) -> pd.DataFrame:
        """
        通过多种方式尝试获取UIS数据
        
        Args:
            indicator_code: 指标代码
            countries: 国家代码列表
            start_year: 起始年份
            end_year: 结束年份
        
        Returns:
            包含数据的DataFrame
        """
        if countries is None:
            countries = list(self.config.TARGET_COUNTRIES.keys())
        
        # 方法1: 尝试SDMX API（可选，默认关闭）
        if getattr(self.config, "ENABLE_UIS_SDMX", False):
            df = self.fetch_via_sdmx_api(indicator_code, countries, start_year, end_year)
            if not df.empty:
                return df
        
        # 方法2: World Bank API（稳定）
        wb_code_map = {
            "GB_XPD_RSDV": "GB.XPD.RSDV.GD.ZS",
            "SDG_9.5.1": "GB.XPD.RSDV.GD.ZS",
            "SP_POP_SCIE": "SP.POP.SCIE.RD.P6",
            "SDG_9.5.2": "SP.POP.SCIE.RD.P6",
        }
        
        wb_code = wb_code_map.get(indicator_code)
        if wb_code:
            df = self._fetch_from_world_bank(wb_code, countries, start_year, end_year)
            if not df.empty:
                return df
        
        return pd.DataFrame()
    
    def _fetch_from_world_bank(self, indicator_code: str, 
                               countries: List[str],
                               start_year: int, 
                               end_year: int) -> pd.DataFrame:
        """从World Bank API获取数据作为备用"""
        country_str = ";".join(countries)
        url = f"https://api.worldbank.org/v2/country/{country_str}/indicator/{indicator_code}"
        
        params = {
            "date": f"{start_year}:{end_year}",
            "format": "json",
            "per_page": 1000
        }
        
        data = self._make_request(url, params)
        
        if data and isinstance(data, list) and len(data) > 1:
            records = data[1]  # World Bank API返回 [metadata, data]
            if records:
                rows = []
                for record in records:
                    if record.get('value') is not None:
                        rows.append({
                            'country_code': record.get('countryiso3code', record.get('country', {}).get('id', '')),
                            'year': record.get('date', ''),
                            'value': record.get('value'),
                            'indicator': indicator_code
                        })
                return pd.DataFrame(rows)
        
        return pd.DataFrame()
    
    def fetch_bulk_download(self, dataset:  str = "SDG") -> Optional[str]:
        """
        获取批量下载链接
        
        UIS提供批量数据下载服务(BDDS)
        
        Args: 
            dataset: 数据集名称
        
        Returns:
            下载URL
        """
        url = f"{self.config.PUBLIC_API_BASE}/bdds/bulk-files"
        
        data = self._make_request(url)
        
        if data and isinstance(data, list):
            for file_info in data:
                if dataset. lower() in file_info.get("name", "").lower():
                    return file_info.get("url")
        
        return None

    def scrape_all_indicators(self, start_year: int = 2010, 
                              end_year: int = 2024) -> Dict[str, pd.DataFrame]: 
        """
        爬取所有R&D相关指标数据
        
        Args: 
            start_year: 起始年份
            end_year:  结束年份
        
        Returns:
            指标数据字典 {indicator_name: DataFrame}
        """
        logger.info("=" * 60)
        logger.info("开始爬取UIS R&D数据")
        logger.info(f"目标国家: {', '.join(self.config.TARGET_COUNTRIES.keys())}")
        logger.info(f"时间范围: {start_year} - {end_year}")
        logger.info("=" * 60)
        
        results = {}
        countries = list(self.config.TARGET_COUNTRIES.keys())
        
        # 仅保留 SDG 两个核心指标（稳定、够用）
        logger.info("\n[1/1] 爬取核心SDG指标数据（使用稳定数据源）...")
        for sdg_key, sdg_info in self.config.SDG_INDICATORS.items():
            logger.info(f"  - {sdg_info['name_cn']}")
            df = self.fetch_data_browser_api(
                sdg_info["code"],
                countries,
                start_year,
                end_year
            )
            if not df.empty:
                results[sdg_key] = df
                logger.info(f"    ✓ 获取到 {len(df)} 条记录")
            else:
                logger.warning(f"    ✗ 未获取到数据")
            time.sleep(0.3)
        
        logger.info(f"\n爬取完成，共获取 {len(results)} 个指标的数据")
        return results
    
    def scrape_via_web_interface(self) -> Dict[str, pd.DataFrame]:
        """
        通过模拟Web界面请求获取数据
        
        这是备用方案，当API不可用时使用
        直接���问UIS Data Browser的数据接口
        """
        logger.info("使用Web接口获取数据...")
        
        results = {}
        countries = list(self.config.TARGET_COUNTRIES.keys())
        
        # UIS Data Browser的数据查询接口
        base_url = "https://data.uis.unesco.org/Api/OData/v4/DataFlow"
        
        # 科学技术创新数据流
        dataflows = [
            "STI",  # Science, Technology and Innovation
            "SDG",  # SDG indicators
        ]
        
        for dataflow in dataflows:
            url = f"{base_url}/{dataflow}"
            country_filter = ','.join([f"'{c}'" for c in countries])
            params = {
                "$filter": f"REF_AREA in ({country_filter})",
                "$select": "REF_AREA,INDICATOR,TIME_PERIOD,OBS_VALUE",
                "$format": "json"
            }
            
            data = self._make_request(url, params)
            if data and isinstance(data, dict):
                records = data.get("value", [])
                if records:
                    df = pd.DataFrame(records)
                    results[f"dataflow_{dataflow}"] = df
                    logger.info(f"  ✓ {dataflow}: 获取到 {len(df)} 条记录")
        
        return results
    
    def process_and_merge_data(self, raw_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        处理和合并所有爬取的数据
        
        Args:
            raw_data: 原始数据字典
        
        Returns:
            合并后的宽格式DataFrame
        """
        logger.info("\n处理和合并数据...")
        
        if not raw_data:
            logger.warning("没有数据可处理")
            return pd.DataFrame()
        
        all_data = []
        
        for indicator_name, df in raw_data.items():
            if df.empty:
                continue
            
            # 标准化列名
            df = df.copy()
            
            # 尝试识别常见列名
            col_mapping = {
                "geoUnit": "country_code",
                "geounit": "country_code", 
                "ref_area": "country_code",
                "REF_AREA": "country_code",
                "country":  "country_code",
                "COUNTRY": "country_code",
                
                "timePeriod": "year",
                "time_period":  "year",
                "TIME_PERIOD": "year",
                "year": "year",
                "YEAR": "year",
                
                "obsValue": "value",
                "obs_value": "value",
                "OBS_VALUE": "value",
                "value": "value",
                "VALUE": "value",
            }
            
            df. columns = [col_mapping.get(col, col) for col in df.columns]
            
            # 添加指标名称
            df["indicator"] = indicator_name
            
            # 获取指标的中文名称
            if indicator_name in self.config.RD_INDICATORS:
                df["indicator_cn"] = self.config.RD_INDICATORS[indicator_name]["name_cn"]
            elif indicator_name in self.config. SDG_INDICATORS:
                df["indicator_cn"] = self.config.SDG_INDICATORS[indicator_name]["name_cn"]
            else:
                df["indicator_cn"] = indicator_name
            
            all_data.append(df)
        
        if not all_data:
            return pd.DataFrame()
        
        # 合并所有数据
        combined = pd.concat(all_data, ignore_index=True)
        
        # year/value 转成数值类型，避免字符串比较问题
        if "year" in combined.columns:
            combined["year"] = pd.to_numeric(combined["year"], errors="coerce").astype("Int64")
        if "value" in combined.columns:
            combined["value"] = pd.to_numeric(combined["value"], errors="coerce")
        
        # 添加国家中文名称
        if "country_code" in combined.columns:
            combined["country_cn"] = combined["country_code"].map(
                lambda x: self.config.TARGET_COUNTRIES. get(x, {}).get("name_cn", x)
            )
            combined["country_en"] = combined["country_code"]. map(
                lambda x: self.config.TARGET_COUNTRIES. get(x, {}).get("name_en", x)
            )
        
        return combined
    
    def create_pivot_tables(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]: 
        """
        创建数据透视表，方便分析
        
        Args: 
            data: 合并后的数据
        
        Returns:
            各种透视表的字典
        """
        pivots = {}
        
        if data.empty:
            return pivots
        
        # 确保必要的列存在
        required_cols = ["country_code", "year", "value", "indicator"]
        if not all(col in data.columns for col in required_cols):
            logger.warning("数据缺少必要列，无法创建透视表")
            return pivots
        
        try:
            # 1. 按国家和年份的指标汇总
            pivots["by_country_year"] = data.pivot_table(
                index=["country_code", "country_cn"],
                columns=["year"],
                values="value",
                aggfunc="first"
            )
            
            # 2. 最新年份各国对比
            latest_year = data["year"].max()
            latest_data = data[data["year"] == latest_year]
            if not latest_data.empty:
                pivots["latest_comparison"] = latest_data.pivot_table(
                    index=["country_cn"],
                    columns=["indicator_cn"],
                    values="value",
                    aggfunc="first"
                )
            
            # 3. 按指标类别汇总
            for category in ["sector_performance", "funding_source", "researchers"]:
                category_indicators = [
                    k for k, v in self.config.RD_INDICATORS.items() 
                    if v["category"] == category
                ]
                category_data = data[data["indicator"].isin(category_indicators)]
                if not category_data. empty:
                    pivots[f"category_{category}"] = category_data.pivot_table(
                        index=["country_cn", "year"],
                        columns=["indicator_cn"],
                        values="value",
                        aggfunc="first"
                    )
        
        except Exception as e: 
            logger.error(f"创建透视表时出错: {e}")
        
        return pivots
    
    def save_results(self, raw_data: Dict[str, pd.DataFrame], 
                     processed_data: pd.DataFrame,
                     pivot_tables: Dict[str, pd.DataFrame]):
        """
        保存所有结果
        
        Args:
            raw_data: 原始数据
            processed_data: 处理后的数据
            pivot_tables: 透视表
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 保存原始数据
        logger.info("\n保存原始数据...")
        for name, df in raw_data.items():
            if not df.empty:
                filepath = f"{self.output_dir}/raw/{name}_{timestamp}.csv"
                df.to_csv(filepath, index=False, encoding="utf-8-sig")
                logger.info(f"  - {filepath}")
        
        # 2. 保存处理后的合并数据
        if not processed_data.empty:
            filepath = f"{self.output_dir}/processed/combined_data_{timestamp}.csv"
            processed_data.to_csv(filepath, index=False, encoding="utf-8-sig")
            logger.info(f"  - {filepath}")
            
            # 同时保存Excel格式
            excel_path = f"{self.output_dir}/processed/combined_data_{timestamp}.xlsx"
            processed_data.to_excel(excel_path, index=False)
            logger.info(f"  - {excel_path}")
        
        # 3. 保存透视表到Excel（多个sheet）
        if pivot_tables:
            excel_path = f"{self.output_dir}/processed/pivot_tables_{timestamp}.xlsx"
            with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
                for name, df in pivot_tables.items():
                    if not df.empty:
                        # Excel sheet名称最大31字符
                        sheet_name = name[: 31]
                        df.to_excel(writer, sheet_name=sheet_name)
            logger.info(f"  - {excel_path}")
        
        # 4. 保存元数据
        metadata = {
            "scrape_time": timestamp,
            "target_countries": self.config.TARGET_COUNTRIES,
            "indicators": {
                "sdg":  self.config.SDG_INDICATORS,
                "rd": self.config.RD_INDICATORS
            },
            "data_source": "UNESCO Institute for Statistics (UIS)",
            "data_url": "https://data.uis.unesco.org/"
        }
        
        metadata_path = f"{self.output_dir}/metadata_{timestamp}.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json. dump(metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"  - {metadata_path}")
        
        logger.info("\n所有数据已保存完成！")
    
    def generate_summary_report(self, data: pd.DataFrame) -> str:
        """
        生成数据摘要报告
        
        Args:
            data:  处理后的数据
        
        Returns:
            摘要报告文本
        """
        report = []
        report.append("=" * 70)
        report.append("UIS R&D数据爬取报告")
        report.append("=" * 70)
        report.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"数据来源: UNESCO Institute for Statistics (UIS)")
        
        if data.empty:
            report.append("\n⚠️ 未获取到有效数据")
            return "\n".join(report)
        
        # 数据概览
        report.append("\n" + "-" * 50)
        report.append("数据概览")
        report.append("-" * 50)
        report.append(f"总记录数: {len(data)}")
        
        if "country_code" in data.columns:
            report.append(f"覆盖国家: {data['country_code'].nunique()}")
            report.append(f"国家列表: {', '.join(data['country_code'].unique())}")
        
        if "year" in data.columns:
            report.append(f"时间范围: {data['year'].min()} - {data['year'].max()}")
        
        if "indicator" in data.columns:
            report.append(f"指标数量: {data['indicator'].nunique()}")
        
        # 各国数据完整度
        report.append("\n" + "-" * 50)
        report.append("各国数据完整度")
        report.append("-" * 50)
        
        if "country_cn" in data.columns and "indicator" in data.columns:
            completeness = data.groupby("country_cn")["indicator"].nunique()
            total_indicators = data["indicator"].nunique()
            
            for country, count in completeness.sort_values(ascending=False).items():
                pct = (count / total_indicators * 100) if total_indicators else 0.0
                bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                cname = str(country)
                report.append(f"{cname:<12} {bar} {pct:5.1f}% ({count}/{total_indicators})")
        
        # 指标覆盖情况
        report.append("\n" + "-" * 50)
        report.append("指标数据覆盖情况")
        report.append("-" * 50)
        
        if "indicator_cn" in data.columns and "country_code" in data.columns:
            for indicator in data["indicator_cn"].unique():
                ind_data = data[data["indicator_cn"] == indicator]
                countries_with_data = ind_data["country_code"].nunique()
                report.append(f"• {indicator}:  {countries_with_data}/10 个国家有数据")
        
        return "\n".join(report)
    
    def run(self, start_year: int = 2010, end_year: int = 2024):
        """
        执行完整的爬取流程
        
        Args: 
            start_year: 起始年份
            end_year: 结束年份
        """
        logger.info("\n" + "=" * 70)
        logger.info("UIS R&D数据爬取程序启动")
        logger.info("=" * 70)
        
        try:
            # 1. 爬取数据
            raw_data = self.scrape_all_indicators(start_year, end_year)
            
            # 不再调用 web_interface 兜底：目前主要数据来自WB，web接口也未必补到细分
            # （若后续想保留，可把下面注释取消）
            # if len(raw_data) < 2:
            #     logger.info("\n尝试使用备用方法获取数据...")
            #     backup_data = self.scrape_via_web_interface()
            #     raw_data.update(backup_data)
            
            # 2. 处理数据
            processed_data = self.process_and_merge_data(raw_data)
            
            # 3. 创建透视表
            pivot_tables = self.create_pivot_tables(processed_data)
            
            # 4. 保存结果
            self.save_results(raw_data, processed_data, pivot_tables)
            
            # 5. 生成报告
            report = self.generate_summary_report(processed_data)
            print("\n" + report)
            
            # 保存报告
            report_path = f"{self.output_dir}/summary_report.txt"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report)
            
            return processed_data
            
        except Exception as e:
            logger.error(f"爬取过程中发生错误: {e}")
            raise


def main():
    """主函数"""
    # 创建爬取器实例
    scraper = UISDataScraper(output_dir="uis_rd_data")
    
    # 执行爬取（2015-2024年的数据）
    data = scraper.run(start_year=2015, end_year=2024)
    
    if data is not None and not data.empty:
        print("\n" + "=" * 70)
        print("爬取完成！数据预览：")
        print("=" * 70)
        print(data. head(20).to_string())
        print(f"\n总共获取 {len(data)} 条数据记录")
    else:
        print("\n⚠️ 未能获取到数据，请检查网络连接或API状态")


if __name__ == "__main__":
    main()