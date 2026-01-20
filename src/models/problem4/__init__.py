# -*- coding: utf-8 -*-
"""
问题四模块：资金优化分配模型 (OCBAM-II)
======================================

基于跳跃-扩散过程的随机最优控制模型 (OCBAM-II)
- Enhanced Optimal Control Budget Allocation Model
- 1万亿元投资分配优化方案 (2026-2035)

核心创新:
1. 代际资本 (Vintage Capital Theory): 算力"买入即贬值" + 动态折旧
2. CES生产函数 (DeepSeek范式): 软硬替代效应
3. 人才回流机制 (Brain Gain): sigmoid函数建模中美竞争
4. 开源网络效应: 梅特卡夫效应非线性增长
5. 内生化战略阶段: 协态变量和罚函数自动涌现最优路径
"""

from .ocbam_optimizer import (
    OCBAM2Parameters,
    run_simulation,
    solve_optimization,
    plot_stacked_area_chart,
    plot_state_evolution,
    save_results,
    main
)

from .data_loader import (
    load_yaml_config,
    load_topsis_weights_from_problem1,
    load_china_initial_state_from_problem3,
    load_historical_investment_data,
    create_parameters_from_yaml,
    get_dynamics_config_from_yaml,
)

__all__ = [
    # 核心优化器
    'OCBAM2Parameters',
    'run_simulation',
    'solve_optimization',
    'plot_stacked_area_chart',
    'plot_state_evolution',
    'save_results',
    'main',
    # 数据加载器
    'load_yaml_config',
    'load_topsis_weights_from_problem1',
    'load_china_initial_state_from_problem3',
    'load_historical_investment_data',
    'create_parameters_from_yaml',
    'get_dynamics_config_from_yaml',
]