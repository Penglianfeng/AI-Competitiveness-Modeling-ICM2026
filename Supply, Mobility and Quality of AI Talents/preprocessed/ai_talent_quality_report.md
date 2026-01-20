# AI人才数据质量分析报告

生成时间: 2026-01-17 18:04:10

## 数据概览

- 数据来源: World Bank, UNESCO UIS
- 时间范围: 2015-2024
- 目标国家: ['USA', 'CHN', 'GBR', 'DEU', 'FRA', 'CAN', 'JPN', 'KOR', 'ARE', 'IND']
- 重点年份: 2016-2025（与AI主数据对齐）
- 核心指标: 研究人员密度、高等教育入学率、教育支出、R&D支出等

## 关键问题

- 🚨 AI人才数据集无2025年数据，需要外推预测

## 警告

- ⚠️ ARE: 数据覆盖率仅39.2%
- ⚠️ 5个指标覆盖率低于30%，建模时考虑剔除或降权
- ⚠️ stem_graduates_pct: 最新数据仅到2021年
- ⚠️ researchers_per_million: 最新数据仅到2022年
- ⚠️ researchers_per_million_fte: 最新数据仅到2021年
- ⚠️ technicians_per_million: 最新数据仅到2018年
- ⚠️ tertiary_spend_per_student_pct_gdp_pc: 最新数据仅到2016年
- ⚠️ tertiary_enrollment_total: 最新数据仅到2019年
- ⚠️ tertiary_completion_25_34_pct: 最新数据仅到2022年

## 预处理建议

- 💡 需要外推12个指标的2025年数据
- 💡 9个指标缺少2023年后数据，建议Holt-Winters外推

## 指标覆盖率

| 指标 | 非空率 | 建议处理 |
|------|--------|---------|
| pop_15_64_pct | 100% | 直接使用 |
| population_total | 100% | 对数变换 |
| tertiary_gross_enrollment_pct | 89% | 插值补全 |
| rd_expenditure_pct_gdp | 76% | 插值+外推 |
| researchers_per_million | 66% | 插值+外推 |
| education_expenditure_pct_gdp | 63% | 插值+外推 |
| stem_graduates_pct | 9% | 考虑剔除或特殊处理 |
| tertiary_completion_25_34_pct | 7% | 考虑剔除或特殊处理 |

## 预处理清单

### 1. 时间维度处理
- [ ] 2023-2025年缺失数据外推（Holt-Winters指数平滑）
- [ ] 中间年份缺失插值（三次样条）
- [ ] 与AI主数据2016-2025对齐

### 2. 缺失值处理
- [ ] 尾部缺失：时间序列外推
- [ ] 中间缺失：三次样条插值
- [ ] 稀疏指标：标记并考虑降权或剔除

### 3. 特征工程
- [ ] 年增长率特征
- [ ] 3年移动平均（平滑波动）
- [ ] 人才综合指数（标准化加权）
- [ ] 与AI产出的滞后关联特征

### 4. 数据标准化
- [ ] 绝对数量指标对数变换（population_total, tertiary_enrollment_total）
- [ ] 国家代码标准化（ISO 3166-1 alpha-3）
- [ ] 输出格式与主表一致
