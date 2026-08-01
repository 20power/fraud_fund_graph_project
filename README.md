# 基于资金图谱的涉诈账户发现与可疑链路解释

> 　目标环境：Windows、CPU、Python 3.11+　｜　系统定位：本地演示与辅助研判原型

## 一、项目成果概览

本项目已经实现从原始 Excel 数据到风险账户、资金关系图谱、可疑路径、典型案例和交互式研判界面的完整闭环。全部计算在本地完成，底层保留原始交易时间和金额，并以统一观察截止时间构建历史特征，避免观察截止之后的交易进入模型。

交付成果包括：

- 嫌疑人账户识别主模型，以及“嫌疑人＋受害人”广义风险账户辅助模型；
- 11,087 个有效账户的全量风险分、风险等级、排名和主要风险因素；
- 资金图谱、月度图谱统计、1—3 跳关联账户、可疑路径和异常资金模式；
- 5 个典型账户的独立 HTML 研判报告、关系图、交易明细和证据表；
- 深色金融科技风格的 Streamlit 研判界面；
- 84 条匿名链路、双审核席位的人工盲审工作簿；
- 数据质量、泄露审计、模型评估、链路分析、项目总结和稳健性等报告；
- 自动化测试、交付检查、环境文件、Dockerfile 和文件哈希清单。

## 二、业务口径

| 事项 | 本项目执行口径 |
|---|---|
| 核心任务 | 以“嫌疑人账户识别”为主任务。 |
| 辅助任务 | 建立“嫌疑人＋受害人”广义风险账户辅助模型，仅用于筛查和特征验证，不替代主模型指标。 |
| 标签时间 | 标签表未提供确认时间；标签视为数据期末状态。所有特征严格按交易时间截断，并披露无法验证标签确认时间层面未来信息泄露的限制。 |
| 时间与金额 | 底层保留精确时间和原始金额，同时生成时间分桶、金额分箱及衍生特征。 |
| 账户数量 | 说明文件写 11,088，账户表和标签表实际均为 11,087 条且一一对应；统一采用 11,087。 |
| 界面定位 | 提供演示和辅助研判原型，不作为生产级系统。 |
| 指标提升 | 主要按相对提升率解释，同时提供绝对值和百分点变化。 |
| 运行环境 | Windows、无 GPU；允许安装 Python 依赖和使用 Docker，不要求完全离线。 |

## 三、数据与质量结论

| 项目 | 结果 |
|---|---:|
| 有效账户 | 11,087 |
| 交易记录 | 904,395 |
| 嫌疑人标签 | 59 |
| 受害人标签 | 1,153 |
| 其它账户 | 9,875 |
| 有交易账户 | 7,795 |
| 无交易账户 | 3,292 |
| 交易时间范围 | 2025-07-01 02:15:01 至 2025-12-31 20:28:55 |
| 账户与标签 ID 集合 | 完全一致 |
| 孤立交易端点 | 0 |
| 精确重复交易 | 0 |

特殊数据按以下方式处理：负金额保留原值并生成负金额标记，聚合流量同时使用绝对金额；自环交易保留为行为特征，但从关系图中心性计算中剔除；无交易账户保留在全量评分范围内。

完整证据见 `outputs/data_quality/data_quality_report.json` 和 `reports/01_data_quality_report.md`。

## 四、建模与评估方法

### 4.1 数据划分与防泄露

账户按照 70% / 15% / 15% 划分为训练集、验证集和测试集，三个集合账户互斥：

| 集合 | 账户数 | 嫌疑人数 | 受害人数 |
|---|---:|---:|---:|
| 训练集 | 7,760 | 41 | 798 |
| 验证集 | 1,663 | 9 | 186 |
| 测试集 | 1,664 | 9 | 169 |

模型选择只使用验证集 PR-AUC；分类阈值只在验证集按 F1 确定；测试集只用于最终一次评估。风险邻居等涉及标签的图特征只允许使用训练集已知状态，测试标签不参与训练、特征构建或调参。

### 4.2 特征体系

特征覆盖五类信息：

- 账户静态属性：开户时长、地区、客户类型等；
- 资金行为：收支笔数、金额、净流量、对手方、活跃时间等；
- 时间模式：小时、星期、月份、夜间、短周期集中度等；
- 金额模式：金额分箱、大额占比、金额集中度和异常金额标记等；
- 图谱特征：度数、加权度、PageRank、社区、风险邻居和局部结构等。

### 4.3 模型策略

主任务比较规则基线、逻辑回归、随机森林和 LightGBM；正式主模型按验证集 PR-AUC 选择为随机森林。辅助模型选择 LightGBM；条件模型用于广义风险账户内部的嫌疑人辅助排序。所有模型均可在 CPU 环境运行。

### 4.4 主模型结果

| 指标 | 规则基线 | 正式主模型 | 变化 |
|---|---:|---:|---:|
| ROC-AUC | 0.8241 | 0.9364 | +0.1123，+11.23 个百分点，相对 +13.63% |
| PR-AUC | 0.0173 | 0.0634 | +0.0461，+4.61 个百分点，相对 +267.00% |
| Top 5% 召回 | 11.11% | 33.33% | +22.22 个百分点，相对 +200.00% |

测试集只有 9 个嫌疑人，因此 Top-K 每增加或减少一个命中都会产生约 11.1 个百分点波动。正式结论同时提供 bootstrap 置信区间，不把单次最优值解释为稳定的生产表现。

### 4.5 稳健性实验

扩展实验覆盖 5 组账户划分、3 个观察截止日期、5 组类别权重、6 组随机森林超参数和 Top 1% / 3% / 5% / 10% 工作量口径。实验仅使用验证集，正式测试结果保持冻结。详细结果见 `reports/06_model_robustness_report.md` 和 `outputs/model_results/extended_stability_*.csv/json`。

## 五、资金图谱与链路成果

项目对交易关系进行方向、金额、笔数和时间聚合，生成全量及月度有向关系表。链路分析对 5 个锚点账户提供最多 20 个、1—3 跳可达的重点关联账户，并输出：

- 关联账户、跳数、匿名或真实路径；
- 路径交易笔数、累计绝对金额、最近交易时间；
- 是否同一图社区、双向资金关系数量；
- 可读的证据摘要和研判解释；
- 资金快进快出、大额集中、活跃时间异常等模式。

内部解释规则通过率只说明字段完整、路径可达和解释模板一致，不等同认可率。外部解释合理性必须以 `outputs/manual_review/链路盲审表.xlsx` 的独立人工盲审结果为准。

## 六、典型案例成果

`outputs/case_studies/CASE-001` 至 `CASE-005` 各自包含：

- `report.html`：可单独打开的深色金融科技风格研判报告，关系图已经内嵌；
- `report.md`：可继续编辑的 Markdown 版本；
- `network.png`：资金关系子图；
- `account_profile.csv`：账户画像；
- `key_transactions.csv`：关键交易；
- `related_accounts.csv`：重点关联账户；
- `suspicious_paths.csv`：可疑路径。

五个案例可整体交付为 `outputs/investigation_reports/五个典型案例研判材料.zip`。

## 七、交互式研判界面

### 7.1 启动方式

在项目根目录执行：

```powershell
conda activate jk_new
streamlit run app.py
```

也可以双击 `run_app.bat`。默认访问地址为：`http://127.0.0.1:8501`。

### 7.2 功能页面

| 页面 | 主要功能 |
|---|---|
| 风险总览 | 全量风险指标、重点账户关系图、证据检查器、研判报告下载。 |
| 风险账户 | 按账户、风险等级和风险分筛选全量排名。 |
| 资金图谱 | 按精确日期、金额、方向、跳数和风险等级动态构图。 |
| 可疑路径 | 对指定锚点实时排序并下载 1—3 跳路径。 |
| 典型案例 | 查看 5 个案例的关系图、结论、因素及独立 HTML 报告。 |
| 模型与口径 | 对比模型、查看提升口径和扩展稳健性实验。 |

界面采用深海军蓝、科技蓝和克制金色的金融科技视觉体系。该界面是本地原型，不含用户权限、审计日志、实时数据接入、任务流转和生产级高可用能力。

## 八、关联链路盲审

盲审文件位于 `outputs/manual_review/`：

| 文件 | 用途 |
|---|---|
| `链路盲审表.xlsx` | 84 条匿名链路、A/B 双审核席位、下拉评分和自动汇总。 |
| `盲审使用指南.md` | 审核流程、统计口径和文件管理要求。 |
| `blind_review_key.csv` | 匿名编号到真实账户的内部映射，应与评分表分开保管。 |
| `review_template.csv` | 便于程序处理的评分空表。 |

回填并保存工作簿后执行：

```powershell
python scripts/aggregate_manual_review.py
```

脚本会生成 `manual_review_summary.json`、`manual_review_responses.csv` 和 `reports/07_client_manual_review_report.md`。解释合理率按“合理＋基本合理”除以有效评价计算；空白和“无法判断”不进入分母。70% 是参考目标，不能用内部规则通过率代替。

## 九、安装与环境

首次部署或对项目不熟悉的使用人员，请先阅读 [`docs/08_部署与使用手册.md`](docs/08_部署与使用手册.md)。该文档提供从环境准备、启动、页面使用、数据更新到故障处理的完整操作说明。

最终交付范围及逐项核对见 [`docs/最终交付内容清单.md`](docs/最终交付内容清单.md)。

### 9.1 复用现有环境

```powershell
conda activate jk_new
python -m pip install -e .
```

### 9.2 创建独立 Conda 环境

```powershell
conda env create -f environment.yml
conda activate fraud_fund_graph
python -m pip install -e .
```

### 9.3 使用 pip

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

如 PowerShell 禁止激活脚本，可在当前用户范围执行 `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser`，或直接使用 `conda run -n jk_new python ...`。

## 十、原始数据放置

项目采用单文件夹自包含结构，甲方原始文件统一保存在项目内部：

```text
fraud_fund_graph_project/
└─ data/raw/
   ├─ 账户表.xlsx
   ├─ 交易边表.xlsx
   ├─ 风险标签表.xlsx
   └─ 说明.txt
```

配置使用相对于项目根目录的路径，因此整个项目移动到其他磁盘或电脑后不需要修改路径。具体配置见 `configs/data_config.yaml`。流水线只读使用原始文件，不会覆盖它们；数据质量报告记录相对路径、文件大小和 SHA-256，便于确认数据版本。

## 十一、运行完整流程

```powershell
conda activate jk_new
python scripts/run_all_pipeline.py
```

流水线依次完成：

```mermaid
flowchart LR
    A[原始数据校验] --> B[历史特征与账户划分]
    B --> C[主模型与辅助模型]
    C --> D[图谱与链路分析]
    D --> E[稳健性实验]
    E --> F[报告与典型案例]
    F --> G[交互界面数据]
    G --> H[交付自动检查]
```

运行日志写入 `logs/pipeline_YYYYMMDD_HHMMSS.log`。如某阶段失败，修复后可以单独执行对应脚本；脚本编号即执行顺序。

盲审模板属于交付准备，不默认进入每次模型重跑；需要刷新时执行：

```powershell
python scripts/10_prepare_manual_review.py
```

如需重新生成 `.xlsx`，再运行 `review_tools/build_manual_review_workbook.mjs`。通常交付包中已包含生成完成的工作簿，无需安装 Node.js。

## 十二、自动化验证

### 12.1 单元与集成测试

```powershell
pytest -q
```

测试覆盖数据校验、时间截断、账户划分、防泄露、特征、模型、链路以及 Streamlit 六个页面的可运行性。

### 12.2 交付检查

```powershell
python scripts/verify_delivery.py
```

预期输出包括：

```text
DATA_CHECK=PASS
GRAPH_CHECK=PASS
MODEL_CHECK=PASS
LINK_CHECK=PASS
PATH_CHECK=PASS
REPORT_CHECK=PASS
DELIVERY_CHECK=PASS
```

检查通过后会重建 `MANIFEST.json`，记录交付文件相对路径、大小和 SHA-256。

## 十三、Docker 运行

在安装 Docker Desktop 的 Windows 环境执行：

```powershell
docker build -t fraud-fund-graph:0.2.0 .
docker run --rm -p 8501:8501 fraud-fund-graph:0.2.0
```

随后访问 `http://127.0.0.1:8501`。镜像包含界面运行所需的处理后数据、模型和结果，但出于数据安全考虑不把 `data/raw` 写入镜像。若要在容器内完整重跑，应把原始数据目录挂载到 `/app/data/raw`。

## 十四、目录说明

```text
fraud_fund_graph_project/
├─ app.py                      Streamlit 入口
├─ assets/                     界面主题样式
├─ configs/                    数据、特征、模型、图谱、验收配置
├─ data/raw/                   甲方原始 Excel 与数据说明（只读）
├─ data/interim/               清洗后的中间数据
├─ data/processed/             特征、图关系和界面数据
├─ docs/                       操作、技术和验收说明
├─ models/                     规则基线、正式模型和元数据
├─ outputs/                    风险排名、链路、案例、图、盲审与实验结果
├─ reports/                    01—07 项目报告
├─ review_tools/               盲审工作簿构建工具
├─ scripts/                    流水线、校验和盲审汇总入口
├─ src/fraud_graph/            可复用核心代码
├─ tests/                      自动化测试
├─ Dockerfile                  容器运行定义
├─ environment.yml             Conda 环境
├─ requirements.txt            Python 依赖
└─ MANIFEST.json               交付文件哈希清单
```

## 十五、重点证据索引

| 需要核验的结论 | 证据文件 |
|---|---|
| 有效账户、标签、交易和质量问题 | `outputs/data_quality/data_quality_report.json` |
| 账户互斥与未来交易计数 | `reports/02_data_split_and_leakage_report.md`、`outputs/graph_statistics/graph_statistics.json` |
| 主模型、基线和提升口径 | `outputs/model_results/model_evaluation.json` |
| 稳健性和 Top-K 工作量 | `reports/06_model_robustness_report.md`、`outputs/model_results/extended_stability_summary.json` |
| 全量风险账户 | `outputs/risk_accounts/risk_account_ranking.csv` |
| 关联账户和可疑路径 | `outputs/link_analysis/` |
| 5 个案例材料 | `outputs/case_studies/`、`outputs/investigation_reports/` |
| 外部解释评价 | `outputs/manual_review/`、`reports/07_client_manual_review_report.md` |
| 完整交付文件一致性 | `MANIFEST.json` |

## 十六、已知限制与正确使用方式

1. 标签确认时间缺失。虽然交易特征严格按时间截断，但无法进一步验证标签确认发生时间造成的未来信息泄露。
2. 嫌疑人样本少。全量仅 59 个，测试集 9 个，PR-AUC 和 Top-K 召回置信区间较宽。
3. 辅助任务较容易。受害人标签与账户属性可能高度相关，其高指标不能替代嫌疑人主任务指标。
4. 图谱只覆盖所给交易数据。未包含设备、手机号、IP、开户地址、工商关系、交易渠道和案件事实。
5. 风险分是相对排序信号，不是涉诈概率承诺；阈值需要结合实际人工核查容量和误报成本重新确认。
6. Streamlit 为原型，不具备生产权限控制、审计、实时更新、并发扩展和容灾能力。
7. 模型输出、链路解释和案例报告只用于辅助研判，不构成执法结论，也不能替代人工核查。

## 十七、常见问题

**界面提示缺少文件怎么办？**

先运行 `python scripts/run_all_pipeline.py`；若只缺少界面关系数据，可单独运行 `python scripts/09_prepare_ui_data.py`。

**为什么全量账户是 11,087 而不是说明中的 11,088？**

账户表和标签表均只有 11,087 条有效记录且一一对应；说明文件比实际数据多 1，项目按真实有效数据统计并在质量报告中保留差异。

**为什么模型 ROC-AUC 较高，但 Top 5% 只命中 3 个嫌疑人？**

类别极不平衡且测试集只有 9 个嫌疑人。ROC-AUC反映整体排序，Top 5% 是固定人工工作量下的命中；两者不能互相替代。

**能否直接把高风险账户认定为涉诈？**

不能。高风险只表示模型认为其更值得核查，必须结合身份、业务背景、交易凭证和案件信息进行人工研判。

**如何更新外部盲审结果？**

回填并保存工作簿后运行 `python scripts/aggregate_manual_review.py`，不要用内部解释通过率代替外部人工结论。

## 十八、数据安全与交付声明

项目默认不向外部服务发送账户或交易明细。建议在受控 Windows 主机运行，限制 `data/`、`outputs/manual_review/blind_review_key.csv`、风险排名和案例材料的访问权限；对外演示优先使用匿名化页面或脱敏副本。

本项目是可复现的辅助研判原型。任何账户处置、案件判断或业务拦截决定，均应由有权限的人员依据完整证据作出。
