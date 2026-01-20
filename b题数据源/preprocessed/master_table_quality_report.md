# 主表数据质量报告
生成时间: 2026-01-17 17:04:46
## 1. 数据规模
- 总行数: 100
- 总列数: 45
- 国家数: 10
- 年份范围: 2016-2025

## 2. 缺失值统计
| 列名 | 缺失数 | 缺失率 |
|------|--------|--------|
| GenAI_VC_Investment | 6 | 6.0% |
| GenAI_VC_Investment_Constant2020 | 6 | 6.0% |
| AI_Compute_VC_Investment | 8 | 8.0% |
| AI_Compute_VC_Investment_Constant2020 | 8 | 8.0% |
| GenAI_VC_Investment_log | 6 | 6.0% |
| GenAI_VC_Investment_Constant2020_log | 6 | 6.0% |
| AI_Compute_VC_Investment_log | 8 | 8.0% |
| AI_Compute_VC_Investment_Constant2020_log | 8 | 8.0% |
| AI_VC_Investment_Constant2020_lag1 | 10 | 10.0% |
| AI_VC_Investment_Constant2020_lag2 | 20 | 20.0% |
| AI_VC_Investment_Constant2020_lag3 | 30 | 30.0% |
| GenAI_VC_Investment_Constant2020_lag1 | 16 | 16.0% |
| GenAI_VC_Investment_Constant2020_lag2 | 26 | 26.0% |
| GenAI_VC_Investment_Constant2020_lag3 | 36 | 36.0% |
| Electricity_Production_TWh_lag1 | 10 | 10.0% |
| Electricity_Production_TWh_lag2 | 20 | 20.0% |
| Electricity_Production_TWh_lag3 | 30 | 30.0% |
| AI_Publications_YoY_Growth | 10 | 10.0% |
| AI_VC_Investment_Constant2020_YoY_Growth | 10 | 10.0% |
| GitHub_AI_Projects_YoY_Growth | 10 | 10.0% |
| GenAI_VC_Investment_Constant2020_PPP | 6 | 6.0% |

## 3. 处理日志
- 已添加AI出版物数量
- 已添加高影响力出版物数量
- 已计算高影响力论文占比
- 已添加AI风险投资（名义值+2020不变价）
- 已添加生成式AI风险投资
- 已添加AI计算风险投资
- 已添加GitHub项目数（含2024-2025外推）
- 已添加高影响力GitHub项目数
- 已添加电能生产数据（全部来源合计，含2025外推）
- AI_Publications: 偏度=2.29，已添加对数变换列
- AI_High_Impact_Publications: 偏度=2.23，已添加对数变换列
- AI_VC_Investment: 偏度=4.05，已添加对数变换列
- AI_VC_Investment_Constant2020: 偏度=3.66，已添加对数变换列
- GenAI_VC_Investment: 偏度=5.89，已添加对数变换列
- GenAI_VC_Investment_Constant2020: 偏度=5.74，已添加对数变换列
- AI_Compute_VC_Investment: 偏度=7.17，已添加对数变换列
- AI_Compute_VC_Investment_Constant2020: 偏度=6.81，已添加对数变换列
- GitHub_AI_Projects: 偏度=2.73，已添加对数变换列
- GitHub_High_Impact_Projects: 偏度=2.53，已添加对数变换列
- Electricity_Production_TWh: 偏度=2.07，已添加对数变换列
- 已为 AI_VC_Investment_Constant2020 添加1-3年滞后特征
- 已为 GenAI_VC_Investment_Constant2020 添加1-3年滞后特征
- 已为 Electricity_Production_TWh 添加1-3年滞后特征
- 已计算 AI_Publications 年同比增长率
- 已计算 AI_VC_Investment_Constant2020 年同比增长率
- 已计算 GitHub_AI_Projects 年同比增长率
- 已计算 AI_Publications 人均值
- 已计算 AI_VC_Investment_Constant2020 人均值
- 已计算 GitHub_AI_Projects 人均值
- 已计算 AI_VC_Investment_Constant2020 PPP调整值
- 已计算 GenAI_VC_Investment_Constant2020 PPP调整值
