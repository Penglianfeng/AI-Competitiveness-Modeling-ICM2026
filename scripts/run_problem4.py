# -*- coding: utf-8 -*-
"""
问题四运行脚本
==============

运行 OCBAM-II 资金优化分配模型
基于跳跃-扩散过程的随机最优控制模型

使用方法:
    python scripts/run_problem4.py
"""

import sys
import os

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.models.problem4 import main, OCBAM2Parameters


def run_default():
    """使用默认参数运行"""
    print("使用默认参数运行 OCBAM-II 优化...")
    U_optimal, result_info = main()
    return U_optimal, result_info


def run_custom():
    """使用自定义参数运行"""
    import numpy as np
    
    # 自定义参数示例
    custom_params = OCBAM2Parameters(
        # 自定义初始状态
        X_2025=np.array([0.5, 0.4, 0.3, 0.2, 0.4, 0.6]),
        
        # 预算约束
        total_budget=10000.0,  # 1万亿元
        max_annual_investment=2000.0,
        
        # 算力参数
        depreciation_rate_A=0.35,
        moore_law_base=1.1,
        
        # 人才参数
        talent_lag=4,
        
        # 科研CES参数
        ces_rho=0.33,  # 替代弹性 σ=1.5
        ces_alpha=0.4,
        
        # TOPSIS权重
        topsis_weights=np.array([0.15, 0.2, 0.25, 0.15, 0.15, 0.1])
    )
    
    print("使用自定义参数运行 OCBAM-II 优化...")
    U_optimal, result_info = main(custom_params=custom_params)
    return U_optimal, result_info


if __name__ == '__main__':
    print("=" * 60)
    print("问题四：OCBAM-II 资金优化分配模型")
    print("Enhanced Optimal Control Budget Allocation Model")
    print("=" * 60)
    
    # 运行默认配置
    U_optimal, result_info = run_default()
    
    print("\n完成!")
    print("=" * 60)