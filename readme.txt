基于资金图谱的涉诈账户发现项目 
项目目录与文件夹说明
============================================================

一、首先阅读
------------------------------------------------------------
1. README.md
   完整项目说明，介绍成果、业务口径、数据、模型指标、
   图谱、案例、界面、盲审、运行方式、验收、限制和常见问题。

2. docs\08_部署与使用手册.md
   面向不熟悉本项目的使用人员，详细说明 Windows 环境安装、启动、
   六个页面使用、完整重跑、盲审、Docker、验收和故障处理。

3. run_app.bat
   Windows 快速启动入口。脚本先使用项目内 .venv；如不存在，则依次尝试
   fraud_fund_graph 和 jk_new Conda 环境，并始终从脚本所在目录定位项目文件。


二、项目根目录中的正式文件夹
------------------------------------------------------------

【assets】界面资源
  保存 Streamlit 界面的视觉资源。
  - fintech_theme.css：深色金融科技风格主题，包括颜色、间距、指标卡、
    侧边栏、按钮、表格和响应式样式。

【configs】项目配置
  集中保存项目参数，修改数据路径、时间截止、模型和图谱口径时使用。
  - data_config.yaml：原始数据路径、字段映射、标签、账户划分和时间口径。
  - feature_config.yaml：行为、时间、金额等特征配置。
  - graph_config.yaml：图谱、社区、路径和关联分析配置。
  - model_config.yaml：候选模型、随机种子、类别权重和校准配置。
  - acceptance_config.yaml：模型、链路、报告和运行环境的验收标准。

【data】项目数据层
  保存从原始文件读取后形成的中间数据、处理后数据和示例数据。
  甲方原始 Excel 保存在本项目的 data\raw，整个项目移动后无需修改绝对路径。

  【data\raw】原始数据目录
    保存账户表.xlsx、交易边表.xlsx、风险标签表.xlsx 和说明.txt。
    配置使用相对于项目根目录的路径，流水线只读使用这些文件。

  【data\interim】标准化中间数据
    保存账户、标签、交易边等经过字段统一和质量校验后的 Parquet 数据。
    主要供后续特征、图谱和模型脚本读取，不建议人工修改。

  【data\processed】处理后数据
    保存账户特征、账户划分、图谱特征、关系边、月度关系和界面专用数据。
    Streamlit 和模型评估会直接读取其中多个文件。

  【data\samples】样例数据预留目录
    用于开发、测试或演示的小规模样例数据。当前正式结果不依赖该目录。

【docs】项目说明文档
  保存实施、技术、操作、验收和部署文档。
  - 00_DOCUMENT_INDEX.md：文档索引。
  - 01_项目实施口径与数据限制说明.md：已确认业务口径和数据限制。
  - 02_数据字典.md：账户级建模字段、类型和含义。
  - 03_技术方案说明书.md：系统架构、技术栈和关键控制。
  - 04_用户操作手册.md：简版安装、启动、盲审和验收说明。
  - 05_甲方演示与验收指南.md：演示顺序和验收建议。
  - 06_验收指标与证据索引.md：验收项到证据文件的映射。
  - 07_项目局限性与风险说明.md：样本、标签时间、字段和原型限制。
  - 08_部署与使用手册.md：面向非技术人员的完整部署与使用说明。
  - 最终交付内容清单.md：最终交付范围、成果和文件核对清单。

【logs】运行日志
  保存完整流水线日志，文件名通常为 pipeline_日期_时间.log。
  流程失败时，应先查看最新日志中最后一个 START 阶段及其错误信息。

【models】模型文件
  保存规则基线、正式模型、校准器和模型元数据。

  【models\baseline】基线模型
    保存规则基线状态，用于与正式主模型比较。

  【models\final】正式模型
    保存嫌疑人主模型、广义风险辅助模型、条件模型及其概率校准器。
    启动现有系统时必须保留，不建议人工替换或改名。

  【models\metadata】模型元数据
    保存模型版本、选择策略、阈值、特征数量和训练/验证/测试账户数。

【outputs】机器可核验成果
  保存项目运行产生的风险排名、图谱统计、模型结果、链路、案例、图和盲审材料。

  【outputs\data_quality】数据质量结果
    保存账户、标签、交易、缺失值、异常金额、数据范围和原始文件哈希。

  【outputs\graph_statistics】图谱统计
    保存节点数、关系边数、月度图谱统计和未来交易泄露检查结果。

  【outputs\model_results】模型与实验结果
    保存模型评估、模型对比、特征消融、随机种子、观察截止、类别权重、
    超参数和 Top-K 稳定性结果。

  【outputs\risk_accounts】风险账户成果
    保存 11,087 个账户的全量风险分、风险等级、排名和解释。
    risk_account_ranking.csv 可直接使用 Excel 查看。

  【outputs\link_analysis】关联与路径成果
    保存重点关联账户、1—3 跳可疑路径、异常模式和链路验收结果。

  【outputs\case_studies】典型案例
    保存五个案例及其账户画像、关键交易、关联账户、可疑路径、关系图、
    Markdown 报告和独立 HTML 报告。

    【CASE-001 至 CASE-005】单个案例目录
      每个目录对应一个典型账户，report.html 可以直接用浏览器打开。

  【outputs\investigation_reports】研判报告交付区
    保存五个独立 HTML 报告、案例索引、结构化研判结果和案例材料压缩包。

  【outputs\figures】项目图表
    保存标签分布、模型比较、特征重要性和模型稳健性等 PNG 图。

  【outputs\manual_review】甲方盲审材料
    保存甲方链路盲审表、使用指南、匿名映射、空模板和盲审汇总结果。
    blind_review_key.csv 含真实账户映射，必须与评分表分开保管。

【reports】项目报告
  保存可阅读的 Markdown 报告：
  - 01：数据质量报告。
  - 02：账户划分与未来信息泄露审计。
  - 03：主模型和辅助模型评估。
  - 04：关联账户和可疑路径分析。
  - 05：最终项目总结。
  - 06：扩展稳健性实验。
  - 07：甲方人工盲审状态或汇总结果。

【review_tools】盲审工作簿构建工具
  保存甲方链路盲审 Excel 的构建脚本。
  - build_manual_review_workbook.mjs：使用项目表格运行库生成格式化工作簿。
  交付包不附带开发机的 node_modules；甲方直接使用已生成的盲审工作簿时无需安装 Node.js 依赖。

【scripts】可执行脚本
  保存完整流水线、分阶段运行、盲审汇总和交付检查入口。
  - run_all_pipeline.py：按正确顺序运行完整项目并执行交付检查。
  - 01_prepare_data.py：原始数据校验、标准化和账户划分。
  - 02_build_features.py：行为、时间、金额和图谱特征。
  - 03_train_evaluate.py：模型训练、校准、评估和全量风险评分。
  - 04_link_analysis.py：关联账户、可疑路径和异常模式。
  - 05_robustness_experiments.py：基础稳健性和特征消融。
  - 06_generate_reports.py：核心报告、基础案例和文档。
  - 07_extended_stability.py：扩展模型稳健性实验。
  - 08_enhance_cases.py：生成五个增强案例材料。
  - 09_prepare_ui_data.py：生成界面所需的有向关系数据。
  - 10_prepare_manual_review.py：生成匿名盲审样本和映射。
  - aggregate_manual_review.py：汇总甲方回填的盲审工作簿。
  - verify_delivery.py：检查数据、图谱、模型、链路、报告和交付文件。

【src】项目核心源代码
  保存可以被脚本和界面复用的 Python 包。

  【src\fraud_graph】核心功能包
    包含配置读取、数据 IO、数据校验、账户划分、特征工程、模型、评估、
    图谱、链路分析、报告支持和界面绘图等模块。

【tests】自动化测试
  保存 pytest 测试，覆盖数据质量、时间截断、防泄露、特征、模型、图谱、
  链路、交付结果和 Streamlit 六个页面。
  运行方式：pytest -q


三、自动生成的临时文件夹
------------------------------------------------------------

【.pytest_cache】pytest 缓存
  运行自动测试后自动生成，可以删除，不属于业务成果。

【__pycache__】Python 字节码缓存
  运行 Python 后自动生成，可以删除，不属于业务成果。

各级目录中如果出现其他 __pycache__，含义相同。


四、根目录关键文件
------------------------------------------------------------

app.py
  Streamlit 研判界面主入口。

README.md
  面向甲方的完整成果说明。

readme.txt
  当前目录导航文件。

run_app.bat
  Windows 快速启动脚本，自动选择项目 .venv、fraud_fund_graph 或 jk_new 环境，使用 8501 端口。

Dockerfile
  Docker 镜像构建与启动定义。

environment.yml
  Conda 环境定义，默认新建环境名为 fraud_fund_graph。

requirements.txt
  Python 依赖及版本范围。

pyproject.toml
  Python 项目名称、版本、依赖和测试配置。

VERSION
  当前交付版本号。

MANIFEST.json
  交付文件清单、文件大小和 SHA-256，用于完整性核验。

design-qa.md
  界面设计核验记录，不影响系统运行。


五、最常用命令
------------------------------------------------------------

完整重跑：
  python scripts\run_all_pipeline.py

启动界面：
  streamlit run app.py

自动测试：
  pytest -q

交付检查：
  python scripts\verify_delivery.py

汇总甲方盲审：
  python scripts\aggregate_manual_review.py


六、重要提醒
------------------------------------------------------------

1. 甲方原始 Excel 位于 data\raw，完整重跑前不要改变文件名或相对位置。
2. 如果只查看现有结果，无须重新运行模型，启动 app.py 即可。
3. 模型风险分和路径解释只用于辅助研判，不代表账户已被认定涉诈。
4. blind_review_key.csv、全量风险排名和案例材料应限制访问。
5. 修改配置或更换数据前，应备份 models、outputs、reports 和 MANIFEST.json。
6. 更详细的部署步骤和问题处理请阅读 docs\08_部署与使用手册.md。
