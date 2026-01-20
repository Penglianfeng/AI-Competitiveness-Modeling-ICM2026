# -*- coding: utf-8 -*-
"""
全局配置文件
============
包含项目所有超参数、路径配置等
"""

import os
from pathlib import Path
import platform

# =============================================================================
# 项目路径配置
# =============================================================================

def get_project_root() -> Path:
    """获取项目根目录"""
    # 从当前文件位置推断项目根目录
    return Path(__file__).parent.parent.absolute()


PROJECT_ROOT = get_project_root()

# 数据路径
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# 输出路径
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PROBLEM1_2_OUTPUT_DIR = OUTPUTS_DIR / "problem1_2"
PROBLEM3_OUTPUT_DIR = OUTPUTS_DIR / "problem3"
PROBLEM4_OUTPUT_DIR = OUTPUTS_DIR / "problem4"

# 文档路径
DOCS_DIR = PROJECT_ROOT / "docs"

# =============================================================================
# 数据源路径 (原始数据)
# =============================================================================

# B题数据源 (位于项目根目录)
B_DATA_DIR = PROJECT_ROOT / "b题数据源"
MASTER_TABLE_PATH = B_DATA_DIR / "preprocessed" / "master_table_o_award.csv"

# AI人才数据 (位于项目根目录)
AI_TALENT_DIR = PROJECT_ROOT / "Supply, Mobility and Quality of AI Talents"
AI_TALENT_PREPROCESSED_PATH = AI_TALENT_DIR / "preprocessed" / "ai_talent_preprocessed.csv"

# 研发创新数据 (位于项目根目录)
RD_INNOVATION_DIR = PROJECT_ROOT / "Research and development investment and innovation foundation"
RD_DATA_PATH = RD_INNOVATION_DIR / "preprocessed" / "rd_innovation_preprocessed.csv"
WORLD_BANK_PATH = RD_INNOVATION_DIR / "World Bank Data" / "AI_development_indicators.csv"

# =============================================================================
# 分析参数配置
# =============================================================================

# 目标国家列表 (ISO 3166-1 alpha-3)
TARGET_COUNTRIES = ['USA', 'CHN', 'GBR', 'DEU', 'KOR', 'JPN', 'FRA', 'CAN', 'ARE', 'IND']

# 国家名称映射 (中文)
COUNTRY_NAMES_CN = {
    'USA': '美国', 'CHN': '中国', 'GBR': '英国', 'DEU': '德国',
    'KOR': '韩国', 'JPN': '日本', 'FRA': '法国', 'CAN': '加拿大',
    'ARE': '阿联酋', 'IND': '印度'
}

# 国家名称映射 (英文)
COUNTRY_NAMES_EN = {
    'USA': 'United States', 'CHN': 'China', 'GBR': 'United Kingdom',
    'DEU': 'Germany', 'KOR': 'South Korea', 'JPN': 'Japan',
    'FRA': 'France', 'CAN': 'Canada', 'ARE': 'United Arab Emirates',
    'IND': 'India'
}

# 分析年份范围
YEAR_START = 2016
YEAR_END = 2025

# 预测年份范围 (问题三)
PREDICTION_YEAR_START = 2026
PREDICTION_YEAR_END = 2035

# =============================================================================
# GLV 模型参数 (问题三)
# =============================================================================

class GLVConfig:
    """GLV-SD 模型配置类"""
    
    # 治理修正因子
    GOV_IMPACT_FACTOR = 0.1  # delta
    
    # 产出函数参数
    BETA1 = 0.6  # Cobb-Douglas 人才弹性
    BETA2 = 0.4  # Cobb-Douglas 算力弹性
    MU_C = 1.0   # 科研产出系数
    MU_D = 1.0   # 开源产出系数
    CAPITAL_ACCELERATOR = 0.05  # lambda (资本加速因子)
    
    # 环境容纳量参数
    ETA = 1.0  # 能源约束系数
    K_LIMIT_A_MULTIPLIER = 2.5  # 算力 K 上限倍率
    K_B_MULTIPLIER = 2.0  # 人才 K 倍率
    K_E_MULTIPLIER = 2.0  # 资本 K 倍率
    
    # RK4 参数
    RK4_STEP_SIZE = 0.1  # 年步长

# =============================================================================
# 模型超参数
# =============================================================================

class ModelConfig:
    """模型配置类"""
    
    # DEMATEL 参数
    DEMATEL_EXPERT_WEIGHT = 0.7      # 专家矩阵权重
    DEMATEL_DATA_WEIGHT = 0.3        # 数据相关矩阵权重
    DEMATEL_CORRELATION_THRESHOLD = 0.3  # 相关性阈值
    
    # TOPSIS 参数
    TOPSIS_NORMALIZATION_MIN = 0.01  # 归一化下限 (避免0值)
    TOPSIS_NORMALIZATION_MAX = 1.0   # 归一化上限
    
    # 稳健性检验参数
    MONTE_CARLO_ITERATIONS = 1000    # Monte Carlo 迭代次数
    WEIGHT_PERTURBATION_STD = 0.1    # 权重扰动标准差
    
    # Malmquist 参数
    DEA_EPSILON = 1e-6               # DEA 数值稳定性参数
    
    # Winsorization 参数
    WINSORIZE_LIMITS = (0.01, 0.01)  # 缩尾处理百分位


# 创建默认配置实例
config = ModelConfig()

# =============================================================================
# 指标体系定义
# =============================================================================

INDICATOR_SYSTEM = {
    'A': {
        'name': '算力与数字基础设施',
        'name_en': 'Compute & Infrastructure',
        'indicators': {
            'A1': '硬件算力规模',
            'A2': '能源与数据中心支撑',
            'A3': '数字连接基础'
        }
    },
    'B': {
        'name': '人才与教育',
        'name_en': 'Talent & Skills',
        'indicators': {
            'B1': 'AI人才存量',
            'B3': 'STEM教育供给实力'
        }
    },
    'C': {
        'name': '科研产出与前沿影响',
        'name_en': 'Research & Frontier Impact',
        'indicators': {
            'C1': 'AI学术产出总量',
            'C2': '高影响力科研产出'
        }
    },
    'D': {
        'name': '开源生态与工程化能力',
        'name_en': 'Open-source & Engineering',
        'indicators': {
            'D1': 'GitHub活跃度',
            'D3': '开源项目影响力'
        }
    },
    'E': {
        'name': '产业转化与资本活力',
        'name_en': 'Adoption, Startups & Capital',
        'indicators': {
            'E1': 'AI垂直领域融资',
            'E2': '资本流动与总量',
            'E3': '行业渗透与技术变现'
        }
    },
    'F': {
        'name': '治理与采用准备度',
        'name_en': 'Governance & Adoption Readiness',
        'indicators': {
            'F1': '研发投入意愿',
            'F2': '知识产权保护'
        }
    }
}

# 所有指标ID列表
ALL_INDICATORS = ['A1', 'A2', 'A3', 'B1', 'B3', 'C1', 'C2', 'D1', 'D3', 'E1', 'E2', 'E3', 'F1', 'F2']

# =============================================================================
# 问题四：资金分配优化参数
# =============================================================================

# 额外投资总额 (人民币亿元)
ADDITIONAL_INVESTMENT_CNY = 10000  # 1万亿元 = 10000亿元

# 投资方向权重约束
INVESTMENT_CONSTRAINTS = {
    'compute': (0.10, 0.30),     # 算力基础设施: 10%-30%
    'talent': (0.15, 0.35),       # 人才培养: 15%-35%
    'research': (0.15, 0.30),     # 科研投入: 15%-30%
    'industry': (0.10, 0.25),     # 产业应用: 10%-25%
    'governance': (0.05, 0.15)    # 治理能力: 5%-15%
}


if __name__ == "__main__":
    # 测试配置加载
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"数据目录: {DATA_DIR}")
    print(f"目标国家: {TARGET_COUNTRIES}")
    print(f"分析年份: {YEAR_START}-{YEAR_END}")
