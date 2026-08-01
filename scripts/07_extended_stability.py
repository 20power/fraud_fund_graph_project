from __future__ import annotations

import copy
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud_graph.config import load_configs  # noqa: E402
from fraud_graph.evaluation import classification_metrics, select_f1_threshold, top_fraction_metrics  # noqa: E402
from fraud_graph.features import aggregate_relationship_edges, build_behavior_features, build_graph_features  # noqa: E402
from fraud_graph.io import write_json, write_table  # noqa: E402
from fraud_graph.models import build_candidates, model_frame  # noqa: E402
from fraud_graph.splits import make_account_split, validate_split  # noqa: E402


def evaluate_validation(features: pd.DataFrame, model_cfg: dict, seed: int = 42) -> tuple[dict, np.ndarray, np.ndarray]:
    X = model_frame(features)
    y = features["primary_target"].astype(int)
    train = features["split"].eq("train")
    validation = features["split"].eq("validation")
    model = build_candidates(X, model_cfg, seed=seed)["random_forest"]
    model.fit(X.loc[train], y.loc[train])
    scores = model.predict_proba(X.loc[validation])[:, 1]
    target = y.loc[validation].to_numpy()
    threshold = select_f1_threshold(target, scores)
    return classification_metrics(target, scores, threshold), target, scores


def metric_row(experiment: str, value: str, metrics: dict) -> dict:
    return {
        "experiment": experiment, "setting": value,
        "validation_accounts": metrics["n"], "validation_suspects": metrics["positive_count"],
        "roc_auc": metrics["roc_auc"], "pr_auc": metrics["pr_auc"],
        "top_1_recall": metrics["top_1_percent"]["recall"],
        "top_3_recall": metrics["top_3_percent"]["recall"],
        "top_5_recall": metrics["top_5_percent"]["recall"], "f1": metrics["f1"],
    }


def assemble_features(behavior: pd.DataFrame, graph_features: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    return (behavior.merge(graph_features, on="account_id", how="left")
            .merge(split[["account_id", "split"]], on="account_id", how="left"))


def make_figure(combined: pd.DataFrame) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    groups = ["split_seed", "feature_cutoff", "class_weight", "hyperparameters"]
    labels = ["不同账户划分", "不同观察截止", "不同类别权重", "超参数扰动"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for axis, group, title in zip(axes.flat, groups, labels):
        frame = combined.loc[combined.experiment.eq(group)]
        x = np.arange(len(frame))
        axis.plot(x, frame.roc_auc, marker="o", color="#2563eb", label="ROC-AUC")
        axis.plot(x, frame.top_5_recall, marker="s", color="#d9a441", label="Top 5%召回")
        axis.plot(x, frame.pr_auc, marker="^", color="#7dd3fc", label="PR-AUC")
        axis.set_xticks(x, frame.setting, rotation=25, ha="right")
        axis.set_ylim(0, 1)
        axis.set_title(title)
        axis.grid(axis="y", color="#dbe4f0", linewidth=.8)
        axis.set_axisbelow(True)
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle("主模型验证集稳健性实验（测试集不参与调参）", fontsize=14, y=.985)
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(.5, .955), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .91), h_pad=2.1, w_pad=1.5)
    fig.savefig(ROOT / "outputs/figures/model_stability.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, decimals: int = 4) -> str:
    def fmt(value):
        return f"{value:.{decimals}f}" if isinstance(value, (float, np.floating)) else str(value)
    return "\n".join([
        "| " + " | ".join(frame.columns) + " |",
        "|" + "|".join(["---"] * len(frame.columns)) + "|",
        *["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)],
    ])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_configs()
    accounts = pd.read_parquet(ROOT / "data/interim/accounts.parquet")
    edges = pd.read_parquet(ROOT / "data/interim/edges.parquet")
    current_features = pd.read_parquet(ROOT / "data/processed/account_features.parquet")
    current_graph = pd.read_parquet(ROOT / "data/processed/graph_features.parquet")
    current_split = pd.read_parquet(ROOT / "data/processed/account_split.parquet")
    current_pairs = pd.read_parquet(ROOT / "data/processed/relationship_edges.parquet")
    graph_columns = [c for c in current_graph.columns if c != "account_id"]
    behavior = current_features.drop(columns=graph_columns + ["split"], errors="ignore")
    all_rows: list[dict] = []

    split_seeds = [42, 73, 1307, 2026, 3407]
    for seed in split_seeds:
        logging.info("账户划分稳定性 seed=%s", seed)
        data_cfg = copy.deepcopy(cfg["data"])
        data_cfg["split"]["random_seed"] = seed
        split = make_account_split(accounts, data_cfg)
        split_check = validate_split(split)
        if not split_check["account_disjoint"] or not split_check["all_accounts_assigned"]:
            raise AssertionError(f"split seed {seed} invalid")
        graph_features, _ = build_graph_features(accounts, current_pairs, split)
        frame = assemble_features(behavior, graph_features, split)
        metrics, _, _ = evaluate_validation(frame, cfg["model"], seed=42)
        row = metric_row("split_seed", str(seed), metrics)
        row["account_disjoint"] = True
        row["future_transaction_count"] = 0
        all_rows.append(row)

    for cutoff in cfg["data"]["time"]["sensitivity_cutoffs"]:
        logging.info("观察截止稳定性 cutoff=%s", cutoff)
        cutoff_behavior, history = build_behavior_features(accounts, edges, cutoff)
        pairs = aggregate_relationship_edges(history)
        graph_features, _ = build_graph_features(accounts, pairs, current_split)
        frame = assemble_features(cutoff_behavior, graph_features, current_split)
        metrics, _, _ = evaluate_validation(frame, cfg["model"], seed=42)
        row = metric_row("feature_cutoff", str(cutoff)[:10], metrics)
        row["account_disjoint"] = True
        row["future_transaction_count"] = int((history.transaction_time > pd.Timestamp(cutoff)).sum())
        row["transactions_used"] = len(history)
        all_rows.append(row)

    for label, weight in [
        ("none", None), ("balanced", "balanced"), ("balanced_subsample", "balanced_subsample"),
        ("positive_50x", {0: 1, 1: 50}), ("positive_100x", {0: 1, 1: 100}),
    ]:
        model_cfg = copy.deepcopy(cfg["model"])
        model_cfg["random_forest"]["class_weight"] = weight
        metrics, _, _ = evaluate_validation(current_features, model_cfg, seed=42)
        all_rows.append(metric_row("class_weight", label, metrics))

    parameter_sets = [
        ("depth8_leaf3_trees300", 8, 3, 300),
        ("depth12_leaf3_trees300", 12, 3, 300),
        ("depth16_leaf3_trees300", 16, 3, 300),
        ("depth12_leaf1_trees300", 12, 1, 300),
        ("depth12_leaf5_trees300", 12, 5, 300),
        ("depth12_leaf3_trees500", 12, 3, 500),
    ]
    for label, depth, leaf, trees in parameter_sets:
        model_cfg = copy.deepcopy(cfg["model"])
        model_cfg["random_forest"].update({"max_depth": depth, "min_samples_leaf": leaf, "n_estimators": trees})
        metrics, _, _ = evaluate_validation(current_features, model_cfg, seed=42)
        all_rows.append(metric_row("hyperparameters", label, metrics))

    base_metrics, y_val, validation_scores = evaluate_validation(current_features, cfg["model"], seed=42)
    top_rows = []
    for fraction in [.01, .03, .05, .10]:
        values = top_fraction_metrics(y_val, validation_scores, fraction)
        top_rows.append({"top_fraction": fraction, **values})
    top_frame = pd.DataFrame(top_rows)
    write_table(top_frame, "outputs/model_results/topk_sensitivity.csv")

    combined = pd.DataFrame(all_rows)
    for experiment, filename in [
        ("split_seed", "split_stability.csv"), ("feature_cutoff", "cutoff_sensitivity.csv"),
        ("class_weight", "class_weight_sensitivity.csv"), ("hyperparameters", "hyperparameter_stability.csv"),
    ]:
        write_table(combined.loc[combined.experiment.eq(experiment)], f"outputs/model_results/{filename}")
    write_table(combined, "outputs/model_results/extended_stability_all.csv")
    make_figure(combined)
    summary = {}
    for experiment in combined.experiment.unique():
        frame = combined.loc[combined.experiment.eq(experiment)]
        summary[experiment] = {
            metric: {"mean": float(frame[metric].mean()), "std": float(frame[metric].std(ddof=1)),
                     "min": float(frame[metric].min()), "max": float(frame[metric].max())}
            for metric in ["roc_auc", "pr_auc", "top_5_recall", "f1"]
        }
    summary["model_replacement_policy"] = "当前正式模型冻结；扩展实验只用于稳健性评估，不使用测试集选择参数。"
    summary["label_time_caveat"] = "提前截止仍使用期末标签，不能等同严格时间外验证。"
    write_json(summary, "outputs/model_results/extended_stability_summary.json")

    report_lines = [
        "# 主模型扩展稳健性报告", "", "## 技术摘要", "",
        "扩展实验覆盖5组账户划分、3个观察截止时间、5组类别权重、6组随机森林参数和4个Top-K口径。所有比较均使用验证集，现有测试集结果保持冻结，不因本轮实验重新选择模型。",
        "", "![主模型稳健性](../outputs/figures/model_stability.png)", "",
        "图中不同实验的离散波动主要来自验证集只有9个嫌疑人；Top-K每命中一个嫌疑人就会产生约11.1个百分点变化，因此不应把单次最优值视为稳定提升。",
    ]
    for experiment, title in [
        ("split_seed", "不同账户划分"), ("feature_cutoff", "不同观察截止时间"),
        ("class_weight", "不同类别权重"), ("hyperparameters", "超参数扰动"),
    ]:
        report_lines += ["", f"## {title}", "", markdown_table(combined.loc[combined.experiment.eq(experiment), [
            "setting", "validation_accounts", "validation_suspects", "roc_auc", "pr_auc", "top_5_recall", "f1"
        ]]), ""]
    report_lines += [
        "## Top-K工作量", "", markdown_table(top_frame), "",
        "Top-K越大，嫌疑人召回一般提高，但人工复核量同步增加。正式阈值应结合甲方单周期可核查容量和误报成本确定。",
        "", "## 结论边界", "",
        "观察截止敏感性仍使用数据期末标签。由于缺少标签确认时间，这些实验只能说明特征窗口变化下的模型波动，不能作为严格的历史时间外验证。",
        "", "## 模型冻结规则", "",
        "本轮结果不替换当前正式随机森林。若未来考虑替换，必须先定义验证集选择规则、在新获得的独立时间外样本上一次性验证，并保留原模型作为对照。",
    ]
    (ROOT / "reports/06_model_robustness_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    logging.info("扩展稳健性实验完成")


if __name__ == "__main__":
    main()
