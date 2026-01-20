# 敏感性分析报告 / Sensitivity Analysis Report

生成时间: 2026-01-19 14:34:33

## 概述 / Overview

本报告汇总了全球AI竞争力评估模型的敏感性分析结果，涵盖问题1-4的关键参数稳健性检验。

## 任务执行状态 / Task Execution Status

| 任务ID | 名称 | 状态 | 耗时(秒) |
|--------|------|------|----------|
| SA1 | 数据口径敏感性分析 | ✅ 成功 | 3.0 |
| SA2 | 权重龙卷风图分析 | ✅ 成功 | 1.2 |
| SA3 | GLV参数敏感性分析 | ✅ 成功 | 12.0 |
| SA4 | 预算/约束敏感性分析 | ✅ 成功 | 1.5 |

## 生成的图表 / Generated Figures

1. **fig_sa1_data_robustness.png**: 数据口径敏感性分析（2024 vs 2025）
2. **fig_sa2_weight_tornado.png**: 权重龙卷风图（中美对比）
3. **fig_sa3_glv_sobol_heatmap.png**: GLV Sobol敏感性指数热力图
4. **fig_sa4_trajectory_bands.png**: 预测轨迹不确定性带
5. **fig_sa5_budget_scenarios.png**: 多场景预算分配堆叠面积图
6. **fig_sa6_pareto_frontier.png**: 预算-绩效Pareto前沿

## 生成的表格 / Generated Tables

- `sa1_ranking_stability.csv`: 排名稳定性分析结果
- `sa2_weight_sensitivity_CHN.csv`: 中国权重敏感性
- `sa2_weight_sensitivity_USA.csv`: 美国权重敏感性
- `sa3_sobol_indices.csv`: Sobol敏感性指数
- `sa3_trajectory_statistics.csv`: 轨迹统计量
- `sa4_optimal_allocation_*.csv`: 各场景最优分配
- `sa4_budget_elasticity.csv`: 预算弹性分析
- `sa4_pareto_frontier.csv`: Pareto前沿数据

## 关键发现 / Key Findings

### SA1: 数据口径敏感性分析


### SA2: 权重龙卷风图分析

- 中国最敏感指标: A1 Hardware Compute, E3 Industry Adoption, F1 Gov R&D
- 美国最敏感指标: F1 Gov R&D, E3 Industry Adoption, A1 Hardware Compute

### SA3: GLV参数敏感性分析

- 最敏感参数: K (容量)
- 预测不确定性最高国家: ARE

### SA4: 预算/约束敏感性分析



---
*本报告由敏感性分析模块自动生成*
