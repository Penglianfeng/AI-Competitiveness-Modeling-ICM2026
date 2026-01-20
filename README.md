# 🏆 谁将在全球人工智能竞赛中获胜？

> **2025年数学建模竞赛 Problem B 求解方案**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 目录

- [一、项目背景](#一项目背景)
- [二、问题概述](#二问题概述)
- [三、项目结构](#三项目结构)
- [四、快速开始](#四快速开始)
- [五、问题一&二：指标构建与评价模型](#五问题一二指标构建与评价模型)
- [六、问题三：未来预测模型](#六问题三未来预测模型)
- [七、问题四：资金分配优化](#七问题四资金分配优化)
- [八、核心发现与结论](#八核心发现与结论)

---

## 一、项目背景

在当今时代，人工智能（AI）已成为全球科技竞争的核心领域，深刻影响着经济发展、社会进步和国家安全。世界各国都大幅增加了在AI领域的投入，旨在这一技术革命中占据领先地位。

本项目针对以下**十个国家**进行AI竞争力分析：
- 🇺🇸 美国 (USA)
- 🇨🇳 中国 (CHN)
- 🇬🇧 英国 (GBR)
- 🇩🇪 德国 (DEU)
- 🇰🇷 韩国 (KOR)
- 🇯🇵 日本 (JPN)
- 🇫🇷 法国 (FRA)
- 🇨🇦 加拿大 (CAN)
- 🇦🇪 阿联酋 (ARE)
- 🇮🇳 印度 (IND)

## 二、问题概述

| 问题 | 任务描述 | 核心方法 |
|------|---------|---------|
| **问题1** | 指标构建与分析：识别AI发展能力因素，量化并分析相互作用 | DEMATEL因果分析 + 相关性分析 |
| **问题2** | 评价模型与2025排名：构建评价模型，给出十国AI竞争力排名 | Global Dynamic TOPSIS + DEA-Malmquist |
| **问题3** | 未来预测：预测2026-2035年AI竞争力排名变化 | GLV-SD分层生态演化模型 |
| **问题4** | 资金分配优化：1万亿元专项资金最优分配方案 | OCBAM-II随机最优控制模型 |

## 三、项目结构

```
Who-will-win-the-global-AI-race/
├── 📂 b题数据源/                    # 原始数据文件
│   ├── 各国历年*.csv               # AI出版物、投资、开源等数据
│   ├── *_AI领域大学计算机排名.csv   # CS Rankings数据(2000-2025)
│   └── TOP500_*.xlsx              # 超算算力数据
├── 📂 src/                         # 源代码
│   ├── models/
│   │   ├── problem1_2/            # 问题一&二求解模块
│   │   │   └── solution.py        # DEMATEL + TOPSIS + Malmquist
│   │   ├── problem3/              # 问题三求解模块
│   │   │   └── glv_forecast.py    # GLV-SD生态演化模型
│   │   └── problem4/              # 问题四求解模块
│   │       └── ocbam_optimizer.py # OCBAM-II最优控制模型
│   ├── data/                      # 数据加载模块
│   └── utils/                     # 工具函数
├── 📂 scripts/                     # 运行脚本
│   ├── run_problem1_2.py          # 运行问题一&二
│   ├── run_problem3.py            # 运行问题三
│   └── run_problem4.py            # 运行问题四
├── 📂 outputs/                     # 输出结果
│   ├── problem1_2/                # 问题一&二结果
│   ├── problem3/                  # 问题三结果
│   └── problem4/                  # 问题四结果
├── 📂 configs/                     # 配置文件
├── requirements.txt               # Python依赖
└── README.md                      # 项目说明
```

## 四、快速开始

### 环境配置

```bash
# 克隆项目
git clone https://github.com/your-repo/Who-will-win-the-global-AI-race.git
cd Who-will-win-the-global-AI-race

# 安装依赖
pip install -r requirements.txt
```

### 运行求解

```bash
# 运行问题一&二：指标构建 + 2025排名
python scripts/run_problem1_2.py

# 运行问题三：2026-2035预测
python scripts/run_problem3.py

# 运行问题四：资金分配优化
python scripts/run_problem4.py
```

---

## 五、问题一&二：指标构建与评价模型

### 5.1 方法论流程

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│    主表数据     │    │    AI人才数据   │    │   研发创新数据  │
│ (master_table)  │    │ (ai_talent)     │    │ (rd_innovation) │
└────────┬────────┘    └────────┬────────┘    └────────┬────────┘
         │                      │                      │
         └──────────────────────┼──────────────────────┘
                                ▼
                    ┌───────────────────────┐
                    │  Step 0: 数据预处理   │
                    │  Winsorization + Log  │
                    └───────────┬───────────┘
                                ▼
                    ┌───────────────────────┐
                    │ Step 1: DEMATEL分析   │
                    │   因果权重计算        │
                    └───────────┬───────────┘
                                ▼
                    ┌───────────────────────┐
                    │ Step 2: Global TOPSIS │
                    │    动态评价排名       │
                    └───────────┬───────────┘
                                ▼
                    ┌───────────────────────┐
                    │ Step 3: DEA-Malmquist │
                    │    效率分析           │
                    └───────────────────────┘
```

### 5.2 指标体系（14项二级指标）

| 一级指标 | 二级指标 | 变量名 | 预处理方式 | 数据来源 |
|---------|---------|--------|-----------|---------|
| **A 算力与基建** | A1 硬件算力基建 | Hardware_Compute | log(x+1) | TOP500 Rmax + 芯片贸易指数 |
| | A2 能源支撑 | Energy_IDC | log(x+1) | 电力生产量 + 安全服务器 |
| | A3 数字连接 | Connectivity | Min-Max | 固定宽带订阅率 |
| **B 人才与教育** | B1 AI人才存量 | Talent_Stock | log(x+1) | 每百万人研究人员数 |
| | B3 STEM教育 | STEM_Supply | Min-Max | CS Rankings + 高等教育入学率 |
| **C 科研产出** | C1 学术产出量 | Research_Qty | log(x+1) | AI出版物数量 |
| | C2 高影响力产出 | High_Impact_Res | log(x+1) | 高影响力AI出版物 |
| **D 开源生态** | D1 GitHub活跃度 | GitHub_Activity | log(x+1) | GitHub AI项目数 |
| | D3 开源影响力 | OpenSource_Impact | log(x+1) | 高影响力GitHub项目 |
| **E 产业资本** | E1 垂直领域VC | Vertical_VC | log(x+1) | GenAI风险投资 |
| | E2 资本流动 | Capital_Flow | log(x+1) | AI_VC总投资 |
| | E3 产业转化 | Ind_Adoption | Min-Max | 高科技出口额 |
| **F 治理准备** | F1 研发投入 | Gov_RD_Exp | Min-Max | R&D支出占GDP比例 |
| | F2 知识产权 | IP_Protection | log(x+1) | 专利申请数 |

### 5.3 DEMATEL因果权重计算

**核心公式：**
- 归一化直接影响矩阵：$N = Z / \max(\sum_j z_{ij})$
- 总关系矩阵：$T = N \times (I - N)^{-1}$
- 影响度：$R_j = \sum_i t_{ij}$（行和）
- 被影响度：$C_j = \sum_i t_{ji}$（列和）
- 中心度权重：$w_j = (R_j + C_j) / \sum_k(R_k + C_k)$

**因果分析结论：**
- **原因因子** (驱动AI发展): F1研发投入、B1人才存量、A1算力基建、E2资本流动
- **结果因子** (AI发展成果): E3产业转化、D1开源活跃、C1科研产出

### 5.4 2025年AI竞争力排名

| 排名 | 国家 | TOPSIS得分 | 主要优势领域 |
|:---:|:---:|:---:|:---|
| 🥇 | 中国 | 0.7335 | 科研产出、资本投入、算力规模 |
| 🥈 | 美国 | 0.6727 | 开源生态、高影响力研究、人才 |
| 🥉 | 德国 | 0.5234 | 人才存量、研发投入、工业基础 |
| 4 | 韩国 | 0.5200 | 芯片产业、数字基建 |
| 5 | 法国 | 0.4829 | 科研基础、教育体系 |
| 6 | 日本 | 0.4599 | 专利保护、制造业基础 |
| 7 | 加拿大 | 0.4539 | 人才培养、开源贡献 |
| 8 | 英国 | 0.4493 | 学术影响力、金融资本 |
| 9 | 印度 | 0.4088 | 人才规模、IT服务 |
| 10 | 阿联酋 | 0.3314 | 政策支持、基建投入 |

### 5.5 稳健性检验

**Monte Carlo权重扰动 (N=1000, ±10%)**

| 国家 | 平均排名 | 第1名频率 | 第2名频率 | 稳居前二频率 |
|------|:-------:|:--------:|:--------:|:-----------:|
| 中国 | 1.25 | 75.2% | 24.8% | **100%** |
| 美国 | 1.75 | 24.8% | 75.2% | **100%** |
| 德国 | 3.01 | 0% | 0% | 99.4% (第3) |

### 5.6 输出文件

| 文件名 | 说明 |
|-------|------|
| [dematel_weights.csv](outputs/problem1_2/dematel_weights.csv) | DEMATEL权重计算结果 |
| [dematel_analysis.png](outputs/problem1_2/dematel_analysis.png) | 因果关系图 + 权重柱状图 |
| [topsis_scores.csv](outputs/problem1_2/topsis_scores.csv) | 各国各年TOPSIS得分 |
| [topsis_trends.png](outputs/problem1_2/topsis_trends.png) | 得分趋势折线图 |
| [malmquist_index.csv](outputs/problem1_2/malmquist_index.csv) | Malmquist效率指数 |
| [correlation_heatmap.png](outputs/problem1_2/correlation_heatmap.png) | 相关性热力图 |
| [robustness_2025_weight_perturbation.csv](outputs/problem1_2/robustness_2025_weight_perturbation.csv) | Monte Carlo稳健性结果 |
| [contribution_breakdown_2025.csv](outputs/problem1_2/contribution_breakdown_2025.csv) | 2025年贡献分解 |

---

## 六、问题三：未来预测模型

### 6.1 GLV-SD分层生态演化模型

```
┌─────────────────────────────────────────────────────────────┐
│             GLV-SD 三层架构 (Generalized Lotka-Volterra)    │
├─────────────────────────────────────────────────────────────┤
│  第1层：核心驱动层 (State Layer - A/B/E)                    │
│  ├─ A(算力基建)、B(人才)、E(资本) → "存量 Stocks"          │
│  └─ 广义 Lotka-Volterra 微分方程组建模                      │
├─────────────────────────────────────────────────────────────┤
│  第2层：产出映射层 (Outcome Layer - C/D)                    │
│  ├─ C(科研) → Cobb-Douglas 生产函数                         │
│  └─ D(开源) → 指数加速模型                                  │
├─────────────────────────────────────────────────────────────┤
│  第3层：环境调节层 (Parameter Layer - F)                    │
│  └─ F(治理准备度) → 修正增长率参数                          │
└─────────────────────────────────────────────────────────────┘
```

**核心方程：**

$$\frac{dX_i^{(k)}}{dt} = r_i^{(k)} X_i^{(k)} \left(1 - \frac{X_i^{(k)}}{K_i^{(k)}}\right) + \sum_{j \neq i} \alpha_{ij}^{(k)} X_i^{(k)} X_j^{(k)}$$

其中：
- $X_i^{(k)}$: 国家 $i$ 在维度 $k$ 的状态值
- $r_i^{(k)}$: 内禀增长率（从历史数据拟合）
- $K_i^{(k)}$: 环境容纳量（考虑能源约束）
- $\alpha_{ij}^{(k)}$: 国家间博弈交互系数

### 6.2 博弈矩阵设计

国家间竞合关系通过交互矩阵 $\alpha$ 建模：

| 关系类型 | $\alpha$ 值 | 典型案例 |
|---------|:----------:|---------|
| 合作促进 | +0.05 ~ +0.10 | 中欧科研合作 |
| 中性竞争 | 0 | 日韩技术竞争 |
| 零和博弈 | -0.03 ~ -0.05 | 中美科技遏制 |

### 6.3 2026-2035年预测结果

| 年份 | 第1名 | 第2名 | 第3名 | 第4名 | 第5名 |
|:----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| 2026 | 🇺🇸 美国 | 🇩🇪 德国 | 🇰🇷 韩国 | 🇯🇵 日本 | 🇫🇷 法国 |
| 2030 | 🇺🇸 美国 | 🇨🇳 中国 | 🇩🇪 德国 | 🇰🇷 韩国 | 🇯🇵 日本 |
| 2035 | 🇺🇸 美国 | 🇨🇳 中国 | 🇰🇷 韩国 | 🇩🇪 德国 | 🇯🇵 日本 |

**2035年详细排名：**

| 排名 | 国家 | TOPSIS得分 |
|:---:|:---:|:---:|
| 1 | 美国 | 0.7641 |
| 2 | 中国 | 0.5343 |
| 3 | 韩国 | 0.5263 |
| 4 | 德国 | 0.5136 |
| 5 | 日本 | 0.4813 |
| 6 | 法国 | 0.4626 |
| 7 | 加拿大 | 0.4521 |
| 8 | 阿联酋 | 0.3216 |
| 9 | 印度 | 0.3175 |
| 10 | 英国 | 0.3115 |

### 6.4 输出文件

| 文件名 | 说明 |
|-------|------|
| [topsis_forecast_scores_2026_2035.csv](outputs/problem3/topsis_forecast_scores_2026_2035.csv) | 预测得分与排名 |
| [glv_params_rK.csv](outputs/problem3/glv_params_rK.csv) | GLV拟合参数 (r, K) |
| [alpha_interaction_matrix.csv](outputs/problem3/alpha_interaction_matrix.csv) | 国家间博弈矩阵 |
| [forecast_trends.png](outputs/problem3/forecast_trends.png) | 预测趋势可视化 |

---

## 七、问题四：资金分配优化

### 7.1 OCBAM-II模型架构

**目标函数：** 最大化2035年中国AI综合竞争力得分

$$\max_{u(t)} \quad S_{2035} = \sum_{k} w_k \cdot X_k(2035)$$

**约束条件：**
- 总预算约束：$\sum_{t,k} u_k(t) = 10000$ 亿元
- 年度投入上限：$\sum_k u_k(t) \leq 2000$ 亿元/年
- 非负约束：$u_k(t) \geq 0$

### 7.2 六维动力学方程

| 领域 | 动力学模型 | 核心机制 |
|-----|-----------|---------|
| **A-基建** | Vintage Capital | 代际资本折旧 + 摩尔定律效率 |
| **B-人才** | Time-Lag Pipeline | 4年滞后 + Hill函数 + 人才回流 |
| **C-科研** | CES生产函数 | DeepSeek范式：软硬替代效应 |
| **D-开源** | Antifragility | 网络效应 + 反脆弱性 |
| **E-产业** | Dynamic Leverage | 杠杆效应 + 拥挤效应 |
| **F-治理** | Linear Growth | 制度效率提升 |

**CES生产函数（DeepSeek范式）：**

$$X_C = A \cdot \left(\alpha \cdot X_A^\rho + (1-\alpha) \cdot X_B^\rho\right)^{1/\rho}$$

其中 $\alpha = 0.4$ 反映"算法换算力"的软硬件替代关系。

### 7.3 最优资金分配方案

**十年总投入分配（单位：亿元人民币）：**

| 领域 | 投入金额 | 占比 | 关键作用 |
|-----|-------:|----:|---------|
| 🖥️ 基建(A) | 2,454 | 24.5% | 算力基础设施建设 |
| 🔬 科研(C) | 2,421 | 24.2% | 前沿技术研发 |
| 👨‍🎓 人才(B) | 1,968 | 19.7% | AI人才培养与引进 |
| 🏭 产业(E) | 1,483 | 14.8% | 产业化应用推广 |
| 💻 开源(D) | 1,213 | 12.1% | 开源生态建设 |
| ⚖️ 治理(F) | 461 | 4.6% | 政策法规完善 |

### 7.4 分阶段投资策略

```
2026-2028 (基础建设期)          2029-2032 (能力提升期)          2033-2035 (冲刺超越期)
┌─────────────────┐             ┌─────────────────┐             ┌─────────────────┐
│ 基建: 39%       │             │ 科研: 35%       │             │ 基建: 35%       │
│ 人才: 47%       │     →       │ 开源: 26%       │     →       │ 科研: 40%       │
│ 其他: 14%       │             │ 产业: 7%        │             │ 产业: 21%       │
└─────────────────┘             └─────────────────┘             └─────────────────┘
  重点：夯实基础                  重点：技术突破                  重点：产业落地
```

### 7.5 优化结果

| 指标 | 数值 |
|-----|------|
| 总预算 | 10,000 亿元 |
| 2035年预测得分 | **0.7201** |
| 优化收敛状态 | ✅ 成功 |

### 7.6 输出文件

| 文件名 | 说明 |
|-------|------|
| [optimal_investment_allocation.csv](outputs/problem4/optimal_investment_allocation.csv) | 逐年分领域最优投入 |
| [investment_allocation_percentage.csv](outputs/problem4/investment_allocation_percentage.csv) | 投入占比 |
| [state_evolution.csv](outputs/problem4/state_evolution.csv) | 状态变量演化轨迹 |
| [state_evolution.png](outputs/problem4/state_evolution.png) | 状态演化可视化 |
| [investment_allocation_stacked_area.png](outputs/problem4/investment_allocation_stacked_area.png) | 投资分配堆叠面积图 |
| [optimization_summary.csv](outputs/problem4/optimization_summary.csv) | 优化结果汇总 |

---

## 八、核心发现与结论

### 8.1 AI发展的因果机制

```
                    ┌─────────────┐
                    │  F1 研发投入 │ ─────────────────┐
                    └──────┬──────┘                   │
                           │                          │
                           ▼                          ▼
┌─────────────┐      ┌─────────────┐           ┌─────────────┐
│ A1 算力基建 │ ───→ │  B1 AI人才  │ ←──────── │ E2 资本流动 │
└──────┬──────┘      └──────┬──────┘           └──────┬──────┘
       │                    │                          │
       │                    ▼                          │
       │             ┌─────────────┐                   │
       └───────────→ │  C1 科研产出 │ ←────────────────┘
                     └──────┬──────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
       ┌───────────┐ ┌───────────┐ ┌───────────┐
       │ D1 开源   │ │ C2 高影响 │ │ E3 产业化 │
       └───────────┘ └───────────┘ └───────────┘
```

### 8.2 关键结论

1. **中美双雄格局稳固**：2025年中美稳居前二（100%稳健性），但竞争态势在预测期内可能发生变化

2. **三阶段追赶路径**：
   - **2026-2028**：夯实基础（基建+人才）
   - **2029-2032**：技术突破（科研+开源）
   - **2033-2035**：产业冲刺（规模化应用）

3. **软硬替代效应**：DeepSeek范式表明，优秀算法可部分替代算力需求（CES替代弹性σ≈1.5）

4. **人才滞后效应**：人才投入需4年才能转化为竞争力，早期投入至关重要

### 8.3 政策建议

| 优先级 | 建议 | 投入占比 |
|:-----:|------|:------:|
| ⭐⭐⭐ | 加大基础算力设施投入 | 24.5% |
| ⭐⭐⭐ | 强化前沿科研攻关 | 24.2% |
| ⭐⭐ | 培养引进AI人才 | 19.7% |
| ⭐⭐ | 推动产业应用落地 | 14.8% |
| ⭐ | 建设开源生态 | 12.1% |
| ⭐ | 完善治理体系 | 4.6% |

---

## 📚 参考文献

1. Hwang, C.L. & Yoon, K. (1981). *Multiple Attribute Decision Making*. Springer-Verlag.
2. Fontela, E. & Gabus, A. (1976). *The DEMATEL Observer*. DEMATEL Report.
3. Färe, R. et al. (1994). Productivity growth, technical progress, and efficiency change. *American Economic Review*.
4. Lotka, A.J. (1925). *Elements of Physical Biology*. Williams & Wilkins.

---

## 📄 许可证

本项目仅供学术研究使用。

---

*文档更新时间: 2026年1月*
