# -*- coding: utf-8 -*-
"""
数据预处理工具函数
==================
包含异常值处理、归一化、对数变换等
"""

import numpy as np
import pandas as pd
from scipy.stats import mstats
from typing import List, Tuple, Optional


def winsorize_data(data: pd.DataFrame, columns: List[str], 
                   limits: Tuple[float, float] = (0.01, 0.01)) -> pd.DataFrame:
    """
    异常值处理 (Winsorization)
    对极端值进行缩尾处理
    
    Args:
        data: 输入数据框
        columns: 需要处理的列名列表
        limits: 缩尾百分位 (下限, 上限)
    
    Returns:
        处理后的数据框
    """
    df = data.copy()
    for col in columns:
        if col in df.columns:
            valid_data = df[col].dropna()
            if len(valid_data) > 0:
                df[col] = mstats.winsorize(df[col].fillna(df[col].median()), limits=limits)
    return df


def log_transform(data: pd.DataFrame, columns: List[str], 
                  suffix: str = '_log') -> pd.DataFrame:
    """
    对数变换: X' = ln(X + 1)
    
    Args:
        data: 输入数据框
        columns: 需要变换的列名列表
        suffix: 新列名后缀
    
    Returns:
        包含变换后列的数据框
    """
    df = data.copy()
    for col in columns:
        if col in df.columns:
            df[f'{col}{suffix}'] = np.log1p(df[col].fillna(0).clip(lower=0))
    return df


def min_max_normalize(data: pd.DataFrame, columns: List[str], 
                      feature_range: Tuple[float, float] = (0.01, 1),
                      suffix: str = '_norm') -> pd.DataFrame:
    """
    Min-Max 归一化到指定范围
    
    Args:
        data: 输入数据框
        columns: 需要归一化的列名列表
        feature_range: 目标范围 (min, max)
        suffix: 新列名后缀
    
    Returns:
        包含归一化后列的数据框
    """
    df = data.copy()
    min_val, max_val = feature_range
    
    for col in columns:
        if col in df.columns:
            col_min = df[col].min()
            col_max = df[col].max()
            if col_max > col_min:
                df[f'{col}{suffix}'] = (
                    (df[col] - col_min) / (col_max - col_min) * (max_val - min_val) + min_val
                )
            else:
                df[f'{col}{suffix}'] = min_val
    return df


def minmax_normalize_series(series: pd.Series) -> pd.Series:
    """
    对单个Series进行Min-Max归一化到[0, 1]范围
    
    Args:
        series: 输入Series
    
    Returns:
        归一化后的Series
    """
    s_min = series.min()
    s_max = series.max()
    return (series - s_min) / (s_max - s_min + 1e-10)


def z_score_normalize(data: pd.DataFrame, columns: List[str],
                      suffix: str = '_zscore') -> pd.DataFrame:
    """
    Z-score 标准化: X' = (X - μ) / σ
    
    Args:
        data: 输入数据框
        columns: 需要标准化的列名列表
        suffix: 新列名后缀
    
    Returns:
        包含标准化后列的数据框
    """
    df = data.copy()
    for col in columns:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df[f'{col}{suffix}'] = (df[col] - mean) / std
            else:
                df[f'{col}{suffix}'] = 0
    return df


def fill_missing_values(data: pd.DataFrame, method: str = 'interpolate',
                        columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    缺失值填充
    
    Args:
        data: 输入数据框
        method: 填充方法 ('interpolate', 'ffill', 'bfill', 'mean', 'median')
        columns: 需要填充的列 (None表示所有数值列)
    
    Returns:
        填充后的数据框
    """
    df = data.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if col in df.columns:
            if method == 'interpolate':
                df[col] = df[col].interpolate(method='linear')
            elif method == 'ffill':
                df[col] = df[col].fillna(method='ffill')
            elif method == 'bfill':
                df[col] = df[col].fillna(method='bfill')
            elif method == 'mean':
                df[col] = df[col].fillna(df[col].mean())
            elif method == 'median':
                df[col] = df[col].fillna(df[col].median())
    
    return df
