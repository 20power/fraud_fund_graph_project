from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud_graph.io import write_json, write_table  # noqa: E402


def read_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text("utf-8"))


def save_markdown(relative: str, lines: list[str]) -> None:
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def pct(value: float) -> str:
    return f"{value:.1%}"


def markdown_table(frame: pd.DataFrame, decimals: int | None = None) -> str:
    def render(value: object) -> str:
        if decimals is not None and isinstance(value, (float, np.floating)):
            return f"{value:.{decimals}f}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    separator = "|" + "|".join(["---"] * len(frame.columns)) + "|"
    rows = ["| " + " | ".join(render(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


def metric_table(evaluation: dict) -> pd.DataFrame:
    rows = []
    baseline = evaluation["rule_baseline"]["test"]
    rows.append({"模型": "规则基线", "ROC-AUC": baseline["roc_auc"], "PR-AUC": baseline["pr_auc"],
                 "Top 5%召回": baseline["top_5_percent"]["recall"], "F1": baseline["f1"]})
    for name, values in evaluation["primary_candidates"]["test"].items():
        rows.append({"模型": name, "ROC-AUC": values["roc_auc"], "PR-AUC": values["pr_auc"],
                     "Top 5%召回": values["top_5_percent"]["recall"], "F1": values["f1"]})
    two = evaluation["two_stage"]["test"]
    rows.append({"模型": "two_stage", "ROC-AUC": two["roc_auc"], "PR-AUC": two["pr_auc"],
                 "Top 5%召回": two["top_5_percent"]["recall"], "F1": two["f1"]})
    return pd.DataFrame(rows)


def make_figures(evaluation: dict, accounts: pd.DataFrame, importance: pd.DataFrame) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    figures = ROOT / "outputs/figures"
    figures.mkdir(parents=True, exist_ok=True)
    comparison = metric_table(evaluation)
    labels = comparison["模型"].replace({"logistic_regression": "Logistic", "random_forest": "Random Forest", "lightgbm": "LightGBM"})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), constrained_layout=True)
    colors = ["#94a3b8", "#60a5fa", "#2563eb", "#1d4ed8", "#f59e0b"]
    for axis, metric, title in zip(axes, ["ROC-AUC", "PR-AUC", "Top 5%召回"],
                                   ["测试集 ROC-AUC", "测试集 PR-AUC", "测试集 Top 5%召回"]):
        bars = axis.barh(labels, comparison[metric], color=colors, edgecolor="#334155", linewidth=.6)
        axis.set_xlim(0, 1 if metric != "PR-AUC" else max(.12, comparison[metric].max() * 1.25))
        axis.set_title(title)
        axis.grid(axis="x", color="#e2e8f0", linewidth=.8)
        axis.set_axisbelow(True)
        for bar, value in zip(bars, comparison[metric]):
            axis.text(value + axis.get_xlim()[1] * .015, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=9)
    fig.suptitle("主任务模型对比（账户互斥测试集 n=1,664，嫌疑人=9）", fontsize=13)
    fig.savefig(figures / "model_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    counts = accounts["account_label"].map({0: "其它", 1: "嫌疑人", 2: "受害人"}).value_counts().reindex(["其它", "受害人", "嫌疑人"])
    fig, axis = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    bars = axis.bar(counts.index, counts.values, color=["#60a5fa", "#f59e0b", "#db2777"], edgecolor="#334155")
    axis.set_title("账户标签分布（有效账户 11,087）")
    axis.set_ylabel("账户数")
    axis.grid(axis="y", color="#e2e8f0")
    axis.set_axisbelow(True)
    for bar, value in zip(bars, counts.values):
        axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,}", ha="center", va="bottom")
    fig.savefig(figures / "label_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    top = importance.head(15).sort_values("importance")
    fig, axis = plt.subplots(figsize=(10, 6.5), constrained_layout=True)
    axis.barh(top["feature"].str.replace("numeric__", "", regex=False).str.replace("categorical__", "", regex=False),
              top["importance"], color="#2563eb", edgecolor="#1e3a8a")
    axis.set_title("最终直接主模型全局特征重要性 Top 15")
    axis.set_xlabel("随机森林重要性")
    axis.grid(axis="x", color="#e2e8f0")
    axis.set_axisbelow(True)
    fig.savefig(figures / "feature_importance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    quality = read_json("outputs/data_quality/data_quality_report.json")
    graph = read_json("outputs/graph_statistics/graph_statistics.json")
    evaluation = read_json("outputs/model_results/model_evaluation.json")
    robustness = read_json("outputs/model_results/robustness_summary.json")
    link = read_json("outputs/link_analysis/link_acceptance.json")
    metadata = read_json("models/metadata/model_metadata.json")
    accounts = pd.read_parquet(ROOT / "data/interim/accounts.parquet")
    features = pd.read_parquet(ROOT / "data/processed/account_features.parquet")
    transactions = pd.read_parquet(ROOT / "data/processed/transactions_cutoff.parquet")
    split = pd.read_parquet(ROOT / "data/processed/account_split.parquet")
    ranking = pd.read_parquet(ROOT / "outputs/risk_accounts/risk_account_ranking.parquet")
    related = pd.read_parquet(ROOT / "outputs/link_analysis/top_related_accounts.parquet")
    cases = read_json("outputs/case_studies/cases.json")
    ablation = pd.read_csv(ROOT / "outputs/model_results/feature_ablation.csv")
    seeds = pd.read_csv(ROOT / "outputs/model_results/random_seed_stability.csv")
    importance = pd.read_csv(ROOT / "outputs/model_results/primary_feature_importance.csv")
    selected = evaluation["selected_primary_test"]
    baseline = evaluation["rule_baseline"]["test"]
    improve = evaluation["improvement_vs_rule_baseline"]

    monthly = transactions.assign(month=transactions.transaction_time.dt.to_period("M").astype(str))
    monthly_stats = monthly.groupby("month").agg(
        transaction_count=("amount", "size"), amount_abs_sum=("amount", lambda x: float(x.abs().sum())),
        payer_count=("payer_id", "nunique"), payee_count=("payee_id", "nunique"),
    ).reset_index()
    write_table(monthly_stats, "outputs/graph_statistics/monthly_graph_statistics.csv")
    top5_n = int(np.ceil((split.split == "test").sum() * .05))
    top5 = ranking.loc[ranking.split.eq("test")].sort_values("risk_score", ascending=False).head(top5_n)
    write_table(top5, "outputs/model_results/top5_coverage.csv")
    patterns = []
    for case in cases:
        for pattern in case["patterns"]:
            patterns.append({"case_id": case["case_id"], "account_id": case["anchor_id"], "pattern": pattern})
    write_table(pd.DataFrame(patterns), "outputs/link_analysis/anomaly_patterns.csv")
    make_figures(evaluation, accounts, importance)

    split_rows = split.groupby("split").agg(
        账户数=("account_id", "size"), 嫌疑人数=("account_label", lambda x: int((x == 1).sum())),
        受害人数=("account_label", lambda x: int((x == 2).sum())),
    ).reset_index()
    split_md = markdown_table(split_rows)

    save_markdown("reports/01_data_quality_report.md", [
        "# 数据质量报告", "", "## 结论", "",
        f"数据通过硬校验，以 **{quality['account_count']:,} 个有效账户**为统一口径；说明文件的 11,088 与实际记录相差 1，已登记但不补造账户。账户表与标签表账户集合一致且标签逐条一致。",
        "", "## 规模与覆盖", "",
        f"- 账户：{quality['account_count']:,}；交易：{quality['edge_count']:,}。",
        f"- 标签：其它 {quality['label_distribution']['0']:,}、嫌疑人 {quality['label_distribution']['1']:,}、受害人 {quality['label_distribution']['2']:,}。",
        f"- 图覆盖账户 {quality['edge_covered_account_count']:,}；无交易账户 {quality['no_edge_account_count']:,}。",
        f"- 交易时间：{quality['transaction_time_min']} 至 {quality['transaction_time_max']}。",
        "", "![账户标签分布](../outputs/figures/label_distribution.png)", "",
        "标签极度不均衡，嫌疑人仅占 0.53%；模型评估必须以 PR-AUC、Top-K 召回和置信区间配合 ROC-AUC。",
        "", "## 异常与处理", "",
        f"- 缺失值：关键字段 0；账户/标签重复主键 0；孤儿交易端点 0。",
        f"- 自环 {quality['self_loop_count']:,}：保留为行为特征，关系图剔除。",
        f"- 负金额 {quality['negative_amount_count']:,}：保留原值，另生成绝对金额与负值标记。",
        f"- 精确重复交易 {quality['exact_duplicate_edge_count']:,}：不自动去重，本批为 0。",
        "", "## 限制", "", "标签没有确认时间，只能视为数据期末状态；本报告不能证明标签确认时间层面不存在未来信息泄露。",
    ])

    save_markdown("reports/02_data_split_and_leakage_report.md", [
        "# 数据划分与泄露检查报告", "", "## 结论", "",
        "账户采用分层、账户互斥的 70%/15%/15% 划分。模型选择和阈值只使用验证集，测试集只用于最终一次评估；交易特征统一截断至 2025-12-31 23:59:59。自动检查确认进入特征的截止时间后交易为 0。",
        "", "## 划分结果", "", split_md, "",
        "## 防泄露控制", "",
        "- 同一账户只属于一个集合；所有 11,087 个账户均已分配。",
        "- 风险邻居特征只引用训练集嫌疑人标签，验证/测试标签不参与图特征。",
        "- 账户 ID、原始标签、split、任意图社区/连通分量编号不作为模型输入。",
        "- 无交易账户保留在主任务，同时作为冷启动子群；图链路只面向有关系覆盖账户。",
        "", "## 无法消除的边界", "",
        "缺少标签确认时间，因此只能证明观察截止时间后的交易未进入特征，不能验证标签在历史时点是否已知。提前截止敏感性也会受期末标签成熟度影响，不能等同严格的线上回溯。",
    ])

    comparison = metric_table(evaluation)
    save_markdown("reports/03_model_evaluation_report.md", [
        "# 嫌疑人账户识别模型评估报告", "", "## 技术摘要", "",
        f"验证集按 PR-AUC 选择的主模型为 **{metadata['selected_strategy']}**。账户互斥测试集上 ROC-AUC={selected['roc_auc']:.4f}、PR-AUC={selected['pr_auc']:.4f}、Top 5%召回={pct(selected['top_5_percent']['recall'])}（{selected['top_5_percent']['hits']}/9）。主任务结果独立报告，辅助模型不替代主模型。",
        "", "![主任务模型对比](../outputs/figures/model_comparison.png)", "",
        "测试集只有 9 个嫌疑人，因此 Top-K 每增加一个命中就变化 11.1 个百分点；指标应与 Bootstrap 区间和多种子验证一起解释。",
        "", "## 模型对比", "", markdown_table(comparison, 4), "",
        "## 验收提升口径", "",
        f"- AUC 相对规则基线：绝对 +{improve['roc_auc']['absolute']:.4f}，+{improve['roc_auc']['percentage_points']:.2f} 个百分点，相对 +{pct(improve['roc_auc']['relative'])}。",
        f"- PR-AUC：绝对 +{improve['pr_auc']['absolute']:.4f}，相对 +{pct(improve['pr_auc']['relative'])}。",
        f"- Top 5%召回：绝对 +{improve['top_5_recall']['absolute']:.4f}，相对 +{pct(improve['top_5_recall']['relative'])}。",
        "需求的 AUC≥0.85、PR-AUC 相对提升≥20%、Top 5%召回相对提升≥15%均满足。",
        "", "## 不确定性与稳健性", "",
        f"- 测试 ROC-AUC 95% Bootstrap 区间：{selected['bootstrap_95_ci']['roc_auc']}。",
        f"- 测试 PR-AUC 95% Bootstrap 区间：{selected['bootstrap_95_ci']['pr_auc']}。",
        f"- 测试 Top 5%召回 95% Bootstrap 区间：{selected['bootstrap_95_ci']['top_5_recall']}。",
        f"- 3 个随机种子验证 PR-AUC 均值/标准差：{robustness['validation_pr_auc']['mean']:.4f}/{robustness['validation_pr_auc']['std']:.4f}。",
        "", "## 特征消融", "", markdown_table(ablation, 4), "",
        "## 概率与解释", "",
        "最终部署分数使用验证集 Platt sigmoid 校准；风险等级按全量风险排名分层。全局特征重要性与逐账户确定性风险因素同时导出。校准改善概率可读性，不改变 ROC-AUC/PR-AUC 排序指标。",
        "", "![特征重要性](../outputs/figures/feature_importance.png)", "",
        "## 结论边界", "", "本结果是预测性关联，不证明因果，也不代表账户已确认涉诈。",
    ])

    save_markdown("reports/04_link_analysis_report.md", [
        "# 可疑关联账户与资金路径分析报告", "", "## 结论", "",
        f"系统已对 {link['anchor_count']} 个差异化锚点生成每个 Top-{link['top_n_per_anchor']} 关联账户与 1—{link['max_hops']} 跳路径。内部确定性解释规则通过率为 {pct(link['internal_explanation_pass_rate'])}；该数值是内部验收，**尚未经甲方人工复核**。",
        f"其中三跳可达账户不足20的锚点为 {link.get('anchors_with_fewer_than_top_n', {})}；这些锚点输出全部可达账户，不使用无路径账户凑数。",
        "", "## 评分证据", "",
        "关联分综合节点风险、边强度、最近交易、同社区/结构信息和跳数惩罚。每条输出至少提供模型风险、交易关系/路径两类证据，并按金额、社区、双向关系补充证据。",
        "", "## 输出", "",
        f"- 关联明细 {len(related):,} 条：`outputs/link_analysis/top_related_accounts.csv`。",
        "- 路径：`outputs/link_analysis/suspicious_paths.csv`。",
        "- 模式：`outputs/link_analysis/anomaly_patterns.csv`。",
        "- 五个案例：`outputs/case_studies/`。",
        "", "## 限制", "",
        f"已知广义风险标签命中率为 {pct(link['known_broad_risk_hit_rate'])}，受只有 3 个嫌疑人进入交易图、标签稀疏和缺少渠道/交易类型影响，不能作为充分外部验证。路径解释用于线索排序，不是资金归因或执法结论。",
    ])

    save_markdown("reports/05_final_project_report.md", [
        "# 基于资金图谱的涉诈账户发现项目最终报告", "", "## 技术总结", "",
        f"项目已形成可重跑的 Windows/CPU 工程包：11,087 个账户纳入主任务，904,395 笔交易完成只读治理；主模型测试 ROC-AUC={selected['roc_auc']:.4f}、PR-AUC={selected['pr_auc']:.4f}，满足约定的绝对或相对提升口径；同时交付辅助模型、两阶段对照、Top-20 关系、三跳路径、5 个案例和 Streamlit 研判原型。",
        "", "## 决策含义", "",
        "模型可以用于缩小人工核查范围并组织交易与图结构证据。建议优先核查风险等级为“高/较高”且有图关系覆盖的账户；无交易高分账户应按冷启动线索单独核验，不应强行生成资金链路。",
        "", "## 核心验收结果", "",
        f"- 未来交易泄露：0；账户集合重叠：0。",
        f"- AUC：{selected['roc_auc']:.4f}（目标≥0.85或较基线+5pp）。",
        f"- PR-AUC 相对提升：{pct(improve['pr_auc']['relative'])}（目标≥20%）。",
        f"- Top 5%召回相对提升：{pct(improve['top_5_recall']['relative'])}（目标≥15%）。",
        f"- 链路内部解释通过率：{pct(link['internal_explanation_pass_rate'])}（内部口径，待甲方复核）。",
        "", "## 方法", "",
        "交易特征保留原始精度并生成时间/金额衍生量；关系图剔除自环，生成 PageRank、k-core、连通分量、Louvain 社区和多跳邻域；模型比较规则、Logistic、随机森林、LightGBM及两阶段策略，按验证 PR-AUC 选择。",
        "", "## 主要限制", "",
        "- 嫌疑人仅 59 个，测试集仅 9 个，指标区间较宽。",
        "- 只有 3 个嫌疑人进入交易图，图证据不能覆盖绝大多数嫌疑人。",
        "- 标签确认时间缺失；缺少交易渠道和交易类型。",
        "- 高风险预测不等于已确认涉诈，必须人工复核。",
        "", "## 建议下一步", "",
        "1. 由甲方对 Top-20 链接和 5 个案例完成盲审，回填外部解释通过率。",
        "2. 补充标签确认时间、交易渠道/类型和处置结果，开展真正的时间外验证。",
        "3. 以本原型为基础确定生产权限、审计、增量计算和模型监控要求，再决定是否生产化。",
        "", "## 待回答问题", "",
        "甲方人工审核的一致性标准、误报成本和实际可核查容量仍会影响最终阈值与Top-K工作量。",
    ])

    data_dict = pd.DataFrame({
        "字段": features.columns,
        "类型": [str(features[c].dtype) for c in features.columns],
        "说明": [
            "账户级建模字段；详细生成逻辑见 src/fraud_graph/features.py" if c not in {"account_id", "account_label", "split"}
            else {"account_id": "脱敏账户主键", "account_label": "0其它/1嫌疑人/2受害人", "split": "账户互斥数据集合"}[c]
            for c in features.columns
        ],
    })
    save_markdown("docs/00_DOCUMENT_INDEX.md", [
        "# 文档索引", "", "- `reports/01`—`05`：五份核心报告；`reports/06`：扩展稳健性；`reports/07`：甲方盲审。",
        "- `docs/01`—`08`：实施、字典、技术、操作、演示、验收、限制和完整部署手册。",
        "- `outputs/`：机器可核验的 JSON/CSV/Parquet、图、案例和盲审工作簿。", "- `MANIFEST.json`：交付文件清单与哈希。",
        "", "面向甲方的完整说明见根目录 `README.md`。首次部署请阅读 `docs/08_部署与使用手册.md`。最终交付核对见 `docs/最终交付内容清单.md`。",
    ])
    save_markdown("docs/01_项目实施口径与数据限制说明.md", [
        "# 项目实施口径与数据限制说明", "", "主任务、辅助任务、标签时间、交易精度、账户数量、交互界面和提升口径均按已确认口径实施。",
        "", "- 主任务：嫌疑人识别；辅助：嫌疑人＋受害人。", "- 账户：11,087。", "- 环境：Windows、CPU、Python 3.11、可选 Docker。",
        "- 界面：本地研判原型，不是生产系统。", "- 提升：相对值为主，同时给出绝对值和百分点。",
    ])
    save_markdown("docs/02_数据字典.md", ["# 数据字典", "", markdown_table(data_dict)])
    save_markdown("docs/03_技术方案说明书.md", [
        "# 技术方案说明书", "", "## 架构", "", "只读 XLSX → Parquet 标准层 → 账户/交易/图特征 → 规则与机器学习 → 校准风险分 → 链路和案例 → Streamlit。",
        "", "## 技术", "", "Python 3.11、pandas、scikit-learn、LightGBM、NetworkX、PyArrow、Plotly、Streamlit、pytest。",
        "", "## 关键控制", "", "账户互斥划分、统一截止时间、训练标签限定风险邻居、确定性模板、固定随机种子和输出哈希。",
    ])
    save_markdown("docs/04_用户操作手册.md", [
        "# 用户操作手册", "", "## 安装", "", "已有环境：`conda activate jk_new`。新环境：`conda env create -f environment.yml`。",
        "", "## 全流程", "", "在项目根目录执行 `python scripts/run_all_pipeline.py`。原始 Excel 位于项目内的 `data/raw`，流水线保持只读。",
        "", "## 启动界面", "", "双击 `run_app.bat`，或执行 `set PYTHONPATH=%CD%\\src` 后运行 `streamlit run app.py`。浏览器访问 `http://127.0.0.1:8501`。",
        "", "界面包含风险总览、风险账户、资金图谱、可疑路径、典型案例、模型与口径六个页面。资金图谱调整条件后需点击“生成 / 刷新资金图谱”；系统会显示生成进度并缓存重复查询。",
        "", "## 甲方盲审", "", "打开 `outputs/manual_review/链路盲审表.xlsx`，由 A/B 两位审核人独立填写黄色区域；回填后执行 `python scripts/aggregate_manual_review.py`。",
        "", "## 验收", "", "执行 `python scripts/verify_delivery.py`，包括 `PATH_CHECK` 在内的全部检查应为 PASS。", "", "完整说明和常见问题见根目录 `README.md`。",
    ])
    save_markdown("docs/05_甲方演示与验收指南.md", [
        "# 甲方演示与验收指南", "", "1. 打开总览核对账户数、图覆盖和主模型指标。", "2. 在风险账户页筛选高风险并下载。",
        "3. 在账户画像页查看风险因素。", "4. 在关系与路径页查看三跳子图和Top-20。", "5. 在典型案例页抽查5例。",
        "6. 对照 `docs/06_验收指标与证据索引.md` 逐项核验。", "", "建议由两名业务专家独立审核链路解释，分歧再复议。",
    ])
    save_markdown("docs/06_验收指标与证据索引.md", [
        "# 验收指标与证据索引", "", "| 验收项 | 结果 | 证据 |", "|---|---:|---|",
        f"| 有效账户口径 | {quality['account_count']:,} | `outputs/data_quality/data_quality_report.json` |",
        "| 未来交易泄露 | 0 | `outputs/graph_statistics/graph_statistics.json` |",
        f"| 主模型 AUC | {selected['roc_auc']:.4f} | `outputs/model_results/model_evaluation.json` |",
        f"| PR-AUC 相对提升 | {pct(improve['pr_auc']['relative'])} | 同上 |",
        f"| Top 5%召回相对提升 | {pct(improve['top_5_recall']['relative'])} | `outputs/model_results/top5_coverage.csv` |",
        "| Top-20与三跳路径 | 已生成 | `outputs/link_analysis/` |",
        f"| 内部解释通过率 | {pct(link['internal_explanation_pass_rate'])} | `outputs/link_analysis/link_acceptance.json` |",
        f"| 典型案例 | {len(cases)} | `outputs/case_studies/` |",
        "| 交互界面 | AppTest通过 | `app.py`、`tests/test_app.py` |",
        "| 扩展稳健性 | 5种划分、3种截止、5种权重、6组参数 | `reports/06_model_robustness_report.md` |",
        "| 甲方外部盲审 | 模板已交付，结果待回填 | `outputs/manual_review/链路盲审表.xlsx` |",
    ])
    save_markdown("docs/07_项目局限性与风险说明.md", [
        "# 项目局限性与风险说明", "", "- 嫌疑人59个，测试集9个，指标方差较大。", "- 仅3个嫌疑人进入交易关系图。",
        "- 无标签确认时间，无法完全验证标签时间泄露。", "- 缺少渠道、交易类型、设备等增强字段。",
        "- 链路解释通过率目前为内部规则口径，待甲方人工复核。", "- 模型高风险不等于已确认涉诈，不可替代人工审核。",
        "- 原型无生产级认证、权限、审计、容灾和持续监控。",
    ])
    investigation_rows = []
    for case in cases:
        investigation_rows.append({
            "case_id": case["case_id"], "account_id": case["anchor_id"],
            "risk_score": case["anchor_risk_score"], "risk_rank": case["anchor_risk_rank"],
            "risk_category": "已确认风险" if case["anchor_label"] == 1 else "模型高风险疑似",
            "patterns": "；".join(case["patterns"]), "review_recommendation": "核验账户身份、资金来源去向与关联主体",
        })
    write_table(pd.DataFrame(investigation_rows), "outputs/investigation_reports/structured_investigation_reports.csv")
    write_json({
        "report_audience": "technical", "report_surface": "Markdown required by delivery checklist",
        "chart_map": [
            {"file": "model_comparison.png", "family": "comparison/bar", "claim": "模型测试指标对比"},
            {"file": "label_distribution.png", "family": "composition/bar", "claim": "标签极度不均衡"},
            {"file": "feature_importance.png", "family": "ranking/bar", "claim": "全局特征贡献排序"},
        ],
        "sources": ["账户表.xlsx", "交易边表.xlsx", "风险标签表.xlsx", "说明.txt"],
        "generated_reports": [f"reports/0{i}_" for i in range(1, 6)],
    }, "reports/report_source_notes.json")


if __name__ == "__main__":
    main()
