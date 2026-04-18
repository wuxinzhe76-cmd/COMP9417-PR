# COMP9417 Project Execution Plan (The HD Blueprint)

## 1. 实验数据集矩阵 (The 5 Datasets)

| # | 数据集 | UCI ID | 任务类型 | 样本量 | 特征数 | 特征类型 | 选择理由 |
|---|--------|--------|----------|--------|--------|----------|----------|
| 1 | **Dry Bean** | [602](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) | 多分类(7类) | 13,611 | 16 | 纯数值型 | **主实验 + Task B 可解释性分析**。7 种干豆的计算机视觉几何特征(面积/周长/偏心率等)，是极其优美的连续流形数据，适合 xRFM 发挥核方法优势，且特征具有明确物理意义，便于对比 AGOP/PCA/MI/Permutation Importance 四种方法 |
| 2 | **AI4I Predictive Maintenance** | [601](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) | 二分类 | 10,000 | 14 | 混合(数值+类别) | **Mixed-type 要求**。工业传感器数据(温度/转速/扭矩/刀具磨损)，特征存在真实物理乘法效应(转速×扭矩=功率)，类别特征(Type: L/M/H)需要 OneHot 编码，测试模型对混合特征的处理能力 |
| 3 | **Online News Popularity** | [332](https://archive.ics.uci.edu/dataset/332/online+news+popularity) | 回归 | 39,644 | 60 | 纯数值型 | **d > 50 高维要求**。Mashable 文章统计指标，涵盖文章字数、媒体数量、情感得分、星期几独热编码等。剔除 url 和 timedelta 两个非预测性标识列后仍有 60 维，测试模型在高维+文本噪音环境下的预测能力 |
| 4 | **Bank Marketing** | [222](https://archive.ics.uci.edu/dataset/222/bank+marketing+dataset) | 二分类 | 45,211 | 16 | 混合(数值+类别) | **极度不平衡社科数据**。葡萄牙银行电话营销记录，包含年龄/工作类型/婚姻/教育/信用等异构特征。标签 yes/no 比例严重失衡(~12%:88%)，是留给 XGBoost 发威、让 xRFM 吃瘪的"陷阱题"。y 映射为 {yes→1, no→0} |
| 5 | **Superconductivity Data** | [464](https://archive.ics.uci.edu/dataset/464/superconductivty+data) | 回归 | 21,263 | 81 | 纯数值型 | **n > 10,000 Scaling 实验**。超导材料的纯静态物理/化学属性数据。完美规避时间序列导致的数据泄漏风险，且两万多的样本量是专门用于 Task C 的“大杀器”，能够极其震撼地展示传统核方法(SVM/KRR) $O(n^2)$ 时间崩溃而 xRFM 依然稳健的对比 |

### 覆盖的作业要求检查
- [x] ≥ 5 个数据集
- [x] ≥ 2 个回归任务 (Online News, Superconductivity)
- [x] ≥ 2 个分类任务 (Dry Bean, AI4I, Bank Marketing)
- [x] ≥ 1 个 n > 10,000 (Superconductivity: 21,263)
- [x] ≥ 1 个 d > 50 (Online News: d=60, Superconductivity: d=81)
- [x] ≥ 1 个 mixed feature types (AI4I, Bank Marketing)

### 数据清洗处理规则
| 数据集 | 处理操作 | 理由 |
|--------|----------|------|
| Online News | 剔除 `url`, `timedelta` | url 是文章链接标识符，timedelta 是数据获取时间差，两者均无预测价值，传入模型会导致数据泄漏或记忆 |
| AI4I | 剔除 `UID`, `Product ID` | 唯一标识符对机器学习无预测意义，保留会导致模型记忆样本而非学习泛化规律 |
| Bank Marketing | `y` 映射: `{'yes': 1, 'no': 0}` | 将字符串标签转为数值二分类标签，统一后续评估接口 |

**不在 data_loader 阶段做的处理**（留给 `preprocessor.py` 或者接下来的训练代码）：
- StandardScaler — 必须在 train 上 fit，在 val/test 上 transform，避免数据泄漏
- OneHotEncoder — 同理，必须仅在 train 上 fit
- 缺失值填充 — 上述 5 个数据集均无缺失值

## 2. 基线模型阵容 (The "All-In" Roster)
- **核心主角:** xRFM
- **树模型阵营:** XGBoost, Random Forest
- **深度学习阵营 (不调参):** MLP, TabNet
- **反面教材 (仅限 Task C):** SVM / KRR

## 3. 核心任务清单 (Tasks to Implement)

### Task A: 数据管道与解耦训练流
- [x] `src/data_loader.py`: 统一的数据下载、清洗与切分 (60/20/20)。剔除无预测价值列，映射标签。
- [x] `src/train_trees_and_xrfm.py`: 动态判断任务类型并进行 StandardScaler + OneHotEncoder 处理，构建 xRFM 所需的 categorical_info。训练 xRFM, XGBoost, RF，并保存为 `.pkl`。
- [x] `src/train_deep_learning.py`: 训练 MLP, TabNet，并分别保存为 `.pkl` 和 `.zip`。
- [x] `notebooks/01_Main_Results.ipynb`: 加载所有模型，在测试集上预测，汇总 25 组结果 (RMSE/Acc, AUC, Time) 生成核心 CSV 表格。

### Task B: 可解释性对比 (仅针对 Dry Bean)
- [ ] `notebooks/02_Interpretability.ipynb`: 提取 xRFM 的 AGOP 对角线。
- [ ] 在同一数据上计算 PCA Loadings, Mutual Information, 和 Permutation Importance。
- [ ] 绘制并排柱状图，直观对比这 4 种方法选出的 Top 5 特征。

### Task C: 扩展性崩溃实验 (仅针对 Superconductivity)
- [ ] `notebooks/03_Scaling_Experiment.ipynb`: 按照 10%, 20%, 40%, 60%, 80%, 100% 递增截取训练集。
- [ ] 对比 xRFM, XGBoost 与 **传统 SVM/KRR**。
- [ ] 绘制两条折线图：`Test Performance vs n` 和 `Training Time vs n`，重点展示传统核方法 $O(n^2)$ 的时间爆炸。

### Task D: (Bonus +10%) 扩展 AGOP 框架探索
- [ ] `src/bonus_residual_agop.py`: 实现文档要求的 Residual-weighted AGOP：
  $$AGOP_{res}(f)=\frac{\sum_{i=1}^{n}w_{i}\nabla f(x_{i})\nabla f(x_{i})^{\top}}{\sum_{i=1}^{n}w_{i}}$$
  其中权重 $w_i = r_i^2$ ($r_i$ 为残差)。
- [ ] 在一个小数据集上比较它与标准 AGOP 选择的 split direction（寻找分歧案例并解释）。