# -*- coding: utf-8 -*-
"""
数据加载模块
============
统一的数据加载接口
"""

import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict
import sys

# 添加项目根目录到路径 (src/data/loaders.py -> src/data -> src -> 项目根目录)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from configs.config import (
    MASTER_TABLE_PATH,
    AI_TALENT_PREPROCESSED_PATH,
    RD_DATA_PATH,
    WORLD_BANK_PATH,
    TARGET_COUNTRIES,
    YEAR_START,
    YEAR_END
)


def load_master_table(filter_countries: bool = True, 
                      filter_years: bool = True) -> pd.DataFrame:
    """
    加载主数据表 (b题数据源预处理后的数据)
    
    Args:
        filter_countries: 是否过滤目标国家
        filter_years: 是否过滤分析年份范围
    
    Returns:
        主数据表DataFrame
    """
    df = pd.read_csv(MASTER_TABLE_PATH)
    df['Country'] = df['Country'].str.upper()
    
    if filter_countries:
        df = df[df['Country'].isin(TARGET_COUNTRIES)]
    
    if filter_years:
        df = df[(df['Year'] >= YEAR_START) & (df['Year'] <= YEAR_END)]
    
    return df


def load_ai_talent_data(filter_countries: bool = True,
                        filter_years: bool = True) -> pd.DataFrame:
    """
    加载AI人才数据
    
    Args:
        filter_countries: 是否过滤目标国家
        filter_years: 是否过滤分析年份范围
    
    Returns:
        AI人才数据DataFrame
    """
    df = pd.read_csv(AI_TALENT_PREPROCESSED_PATH)
    df['country_code'] = df['country_code'].str.upper()
    df = df.rename(columns={'country_code': 'Country', 'year': 'Year'})
    
    if filter_countries:
        df = df[df['Country'].isin(TARGET_COUNTRIES)]
    
    if filter_years:
        df = df[(df['Year'] >= YEAR_START) & (df['Year'] <= YEAR_END)]
    
    return df


def load_rd_innovation_data(filter_countries: bool = True,
                            filter_years: bool = True) -> pd.DataFrame:
    """
    加载研发创新数据
    
    Args:
        filter_countries: 是否过滤目标国家
        filter_years: 是否过滤分析年份范围
    
    Returns:
        研发创新数据DataFrame
    """
    df = pd.read_csv(RD_DATA_PATH)
    df['country_code'] = df['country_code'].str.upper()
    df = df.rename(columns={'country_code': 'Country', 'year': 'Year'})
    
    if filter_countries:
        df = df[df['Country'].isin(TARGET_COUNTRIES)]
    
    if filter_years:
        df = df[(df['Year'] >= YEAR_START) & (df['Year'] <= YEAR_END)]
    
    return df


def load_world_bank_data(filter_countries: bool = True,
                         filter_years: bool = True) -> pd.DataFrame:
    """
    加载世界银行数据
    
    Args:
        filter_countries: 是否过滤目标国家
        filter_years: 是否过滤分析年份范围
    
    Returns:
        世界银行数据DataFrame
    """
    country_mapping = {
        'United States': 'USA', 'China': 'CHN', 'United Kingdom': 'GBR',
        'Germany': 'DEU', 'Korea, Rep.': 'KOR', 'Japan': 'JPN',
        'France': 'FRA', 'Canada': 'CAN', 'United Arab Emirates': 'ARE',
        'India': 'IND', 'Korea': 'KOR', 'South Korea': 'KOR'
    }
    
    df = pd.read_csv(WORLD_BANK_PATH, encoding='utf-8')
    df['Country'] = df['国家'].map(country_mapping)
    df = df.rename(columns={'年份': 'Year'})
    
    if filter_countries:
        df = df[df['Country'].isin(TARGET_COUNTRIES)]
    
    if filter_years:
        df = df[(df['Year'] >= YEAR_START) & (df['Year'] <= YEAR_END)]
    
    return df


def load_all_data() -> Dict[str, pd.DataFrame]:
    """
    加载所有数据源
    
    Returns:
        包含所有数据源的字典
    """
    return {
        'master': load_master_table(),
        'talent': load_ai_talent_data(),
        'rd': load_rd_innovation_data(),
        'world_bank': load_world_bank_data()
    }
