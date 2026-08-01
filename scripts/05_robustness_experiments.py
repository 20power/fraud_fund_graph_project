from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud_graph.config import load_configs  # noqa: E402
from fraud_graph.evaluation import classification_metrics, select_f1_threshold  # noqa: E402
from fraud_graph.io import write_json, write_table  # noqa: E402
from fraud_graph.models import build_candidates, model_frame  # noqa: E402


def evaluate_rf(X: pd.DataFrame, y: pd.Series, split: pd.Series, cfg: dict, seed: int) -> dict:
    X = model_frame(X)
    model = build_candidates(X, cfg, seed=seed)["random_forest"]
    train = split.eq("train")
    validation = split.eq("validation")
    model.fit(X.loc[train], y.loc[train])
    scores = model.predict_proba(X.loc[validation])[:, 1]
    threshold = select_f1_threshold(y.loc[validation].to_numpy(), scores)
    return classification_metrics(y.loc[validation].to_numpy(), scores, threshold)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_configs()
    features = pd.read_parquet(ROOT / "data/processed/account_features.parquet")
    split = features["split"]
    y = features["primary_target"].astype(int)
    groups = {
        "static_only": ["opening_months", "region_code", "customer_type", "has_transaction"],
        "static_plus_transaction": [
            c for c in features if c.startswith(("out_", "in_")) or c in {
                "opening_months", "region_code", "customer_type", "has_transaction",
                "transaction_count_total", "amount_abs_total", "net_flow_signed", "counterparty_total",
                "night_ratio", "weekend_ratio", "high_amount_ratio", "rapid_turnover_day_ratio",
                "negative_amount_count", "self_loop_count", "self_loop_amount_abs_sum",
            }
        ],
        "static_plus_graph": [
            c for c in features if c.startswith("graph_") or c in {
                "opening_months", "region_code", "customer_type", "has_transaction",
                "has_relationship_edge", "train_suspect_neighbor_count",
            }
        ],
        "all_without_risk_neighbor": [
            c for c in model_frame(features).columns if c != "train_suspect_neighbor_count"
        ],
        "all_features": list(model_frame(features).columns),
    }
    ablation_rows = []
    for name, columns in groups.items():
        available = list(dict.fromkeys(c for c in columns if c in features.columns))
        metrics = evaluate_rf(features[available], y, split, cfg["model"], seed=42)
        ablation_rows.append({
            "experiment": name, "feature_count": len(available),
            "validation_roc_auc": metrics["roc_auc"], "validation_pr_auc": metrics["pr_auc"],
            "validation_top5_recall": metrics["top_5_percent"]["recall"], "validation_f1": metrics["f1"],
        })
    ablation = pd.DataFrame(ablation_rows)
    write_table(ablation, "outputs/model_results/feature_ablation.csv")

    X = model_frame(features)
    seed_rows = []
    for seed in cfg["model"]["random_seeds"]:
        metrics = evaluate_rf(X, y, split, cfg["model"], int(seed))
        seed_rows.append({
            "seed": int(seed), "validation_roc_auc": metrics["roc_auc"],
            "validation_pr_auc": metrics["pr_auc"],
            "validation_top5_recall": metrics["top_5_percent"]["recall"],
            "validation_f1": metrics["f1"],
        })
    seed_frame = pd.DataFrame(seed_rows)
    write_table(seed_frame, "outputs/model_results/random_seed_stability.csv")
    summary = {
        column: {"mean": float(seed_frame[column].mean()), "std": float(seed_frame[column].std(ddof=1))}
        for column in seed_frame.columns if column != "seed"
    }
    summary["notes"] = [
        "稳定性实验固定账户划分，仅改变模型随机种子。",
        "嫌疑人总量仅59、验证集仅9个，离散的Top-K召回波动应结合置信区间解释。",
        "不同观察截止窗口已保留配置入口；由于标签是期末状态，提前截止实验会混入标签成熟度变化，不能等同线上回溯验证。",
    ]
    write_json(summary, "outputs/model_results/robustness_summary.json")
    logging.info("消融与随机种子稳定性实验完成")


if __name__ == "__main__":
    main()
