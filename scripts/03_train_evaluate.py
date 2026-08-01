from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud_graph.config import load_configs  # noqa: E402
from fraud_graph.evaluation import (  # noqa: E402
    bootstrap_ci, classification_metrics, improvement, select_f1_threshold,
)
from fraud_graph.io import write_json, write_table  # noqa: E402
from fraud_graph.models import (  # noqa: E402
    build_candidates, feature_importance, fit_rule_baseline, model_frame, rule_scores,
)


def _train_task(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    splits: pd.Series,
    cfg: dict,
) -> tuple[dict, dict, dict]:
    train_mask = splits.eq("train")
    val_mask = splits.eq("validation")
    test_mask = splits.eq("test")
    candidates = build_candidates(X, cfg, seed=int(cfg["random_seeds"][0]))
    validation: dict = {}
    test: dict = {}
    fitted: dict = {}
    for model_name, model in candidates.items():
        logging.info("训练 %s/%s", name, model_name)
        model.fit(X.loc[train_mask], y.loc[train_mask])
        val_scores = model.predict_proba(X.loc[val_mask])[:, 1]
        threshold = select_f1_threshold(y.loc[val_mask].to_numpy(), val_scores)
        validation[model_name] = classification_metrics(y.loc[val_mask].to_numpy(), val_scores, threshold)
        test_scores = model.predict_proba(X.loc[test_mask])[:, 1]
        test[model_name] = classification_metrics(y.loc[test_mask].to_numpy(), test_scores, threshold)
        test[model_name]["bootstrap_95_ci"] = bootstrap_ci(
            y.loc[test_mask].to_numpy(), test_scores, int(cfg["bootstrap_iterations"])
        )
        fitted[model_name] = model
    selected = max(validation, key=lambda key: validation[key][cfg["selection_metric"]])
    return {"validation": validation, "test": test, "selected": selected}, fitted, {
        "train_mask": train_mask, "validation_mask": val_mask, "test_mask": test_mask,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_configs()
    features = pd.read_parquet(ROOT / "data/processed/account_features.parquet")
    X = model_frame(features)
    splits = features["split"]
    y_primary = features["primary_target"].astype(int)
    y_aux = features["auxiliary_target"].astype(int)

    rule_state = fit_rule_baseline(features.loc[splits.eq("train")])
    rule_val_scores = rule_scores(features.loc[splits.eq("validation")], rule_state)
    rule_threshold = select_f1_threshold(y_primary.loc[splits.eq("validation")].to_numpy(), rule_val_scores)
    rule_test_scores = rule_scores(features.loc[splits.eq("test")], rule_state)
    rule_metrics = {
        "validation": classification_metrics(y_primary.loc[splits.eq("validation")], rule_val_scores, rule_threshold),
        "test": classification_metrics(y_primary.loc[splits.eq("test")], rule_test_scores, rule_threshold),
    }
    rule_metrics["test"]["bootstrap_95_ci"] = bootstrap_ci(
        y_primary.loc[splits.eq("test")].to_numpy(), rule_test_scores, int(cfg["model"]["bootstrap_iterations"])
    )

    primary_results, primary_fitted, masks = _train_task("primary", X, y_primary, splits, cfg["model"])
    aux_results, aux_fitted, _ = _train_task("auxiliary", X, y_aux, splits, cfg["model"])

    broad_mask = y_aux.eq(1)
    conditional_target = y_primary
    conditional_splits = splits.where(broad_mask, "excluded")
    conditional_results, conditional_fitted, _ = _train_task(
        "conditional_suspect_given_risk", X.loc[broad_mask], conditional_target.loc[broad_mask],
        conditional_splits.loc[broad_mask], cfg["model"]
    )

    aux_name = aux_results["selected"]
    conditional_name = conditional_results["selected"]
    aux_model = aux_fitted[aux_name]
    conditional_model = conditional_fitted[conditional_name]
    two_stage_validation_scores = (
        aux_model.predict_proba(X.loc[masks["validation_mask"]])[:, 1]
        * conditional_model.predict_proba(X.loc[masks["validation_mask"]])[:, 1]
    )
    two_stage_threshold = select_f1_threshold(
        y_primary.loc[masks["validation_mask"]].to_numpy(), two_stage_validation_scores
    )
    two_stage_test_scores = (
        aux_model.predict_proba(X.loc[masks["test_mask"]])[:, 1]
        * conditional_model.predict_proba(X.loc[masks["test_mask"]])[:, 1]
    )
    two_stage = {
        "validation": classification_metrics(
            y_primary.loc[masks["validation_mask"]], two_stage_validation_scores, two_stage_threshold
        ),
        "test": classification_metrics(
            y_primary.loc[masks["test_mask"]], two_stage_test_scores, two_stage_threshold
        ),
        "auxiliary_model": aux_name,
        "conditional_model": conditional_name,
    }
    two_stage["test"]["bootstrap_95_ci"] = bootstrap_ci(
        y_primary.loc[masks["test_mask"]], two_stage_test_scores, int(cfg["model"]["bootstrap_iterations"])
    )

    direct_name = primary_results["selected"]
    direct_val_pr = primary_results["validation"][direct_name]["pr_auc"]
    if two_stage["validation"]["pr_auc"] > direct_val_pr:
        selected_strategy = "two_stage"
        selected_test = two_stage["test"]
    else:
        selected_strategy = f"direct:{direct_name}"
        selected_test = primary_results["test"][direct_name]

    # 概率校准仅使用验证集；最终可部署模型保持只在训练集拟合，避免校准集回流。
    primary_final = primary_fitted[direct_name]
    aux_final = aux_fitted[aux_name]
    conditional_final = conditional_fitted[conditional_name]

    def fit_platt(raw_scores: np.ndarray, target: pd.Series) -> LogisticRegression:
        calibrator = LogisticRegression(random_state=42)
        calibrator.fit(raw_scores.reshape(-1, 1), target.to_numpy())
        return calibrator

    val_mask = masks["validation_mask"]
    primary_calibrator = fit_platt(primary_final.predict_proba(X.loc[val_mask])[:, 1], y_primary.loc[val_mask])
    aux_calibrator = fit_platt(aux_final.predict_proba(X.loc[val_mask])[:, 1], y_aux.loc[val_mask])
    conditional_val_mask = val_mask & broad_mask
    conditional_calibrator = fit_platt(
        conditional_final.predict_proba(X.loc[conditional_val_mask])[:, 1], y_primary.loc[conditional_val_mask]
    )
    joblib.dump(primary_final, ROOT / "models/final/primary_model.joblib")
    joblib.dump(aux_final, ROOT / "models/final/auxiliary_model.joblib")
    joblib.dump(conditional_final, ROOT / "models/final/conditional_model.joblib")
    joblib.dump(primary_calibrator, ROOT / "models/final/primary_calibrator.joblib")
    joblib.dump(aux_calibrator, ROOT / "models/final/auxiliary_calibrator.joblib")
    joblib.dump(conditional_calibrator, ROOT / "models/final/conditional_calibrator.joblib")
    joblib.dump(rule_state, ROOT / "models/baseline/rule_state.joblib")

    primary_score = primary_calibrator.predict_proba(primary_final.predict_proba(X)[:, 1].reshape(-1, 1))[:, 1]
    auxiliary_score = aux_calibrator.predict_proba(aux_final.predict_proba(X)[:, 1].reshape(-1, 1))[:, 1]
    conditional_score = conditional_calibrator.predict_proba(
        conditional_final.predict_proba(X)[:, 1].reshape(-1, 1)
    )[:, 1]
    two_stage_score = auxiliary_score * conditional_score
    selected_score = two_stage_score if selected_strategy == "two_stage" else primary_score
    predictions = features[[
        "account_id", "account_label", "primary_target", "auxiliary_target", "split",
        "has_transaction", "has_relationship_edge", "graph_component_id", "graph_community_id",
    ]].copy()
    predictions["primary_score"] = primary_score
    predictions["auxiliary_score"] = auxiliary_score
    predictions["conditional_score"] = conditional_score
    predictions["two_stage_score"] = two_stage_score
    predictions["risk_score"] = selected_score
    predictions["risk_rank"] = predictions["risk_score"].rank(method="first", ascending=False).astype(int)
    predictions["selected_strategy"] = selected_strategy
    rank_fraction = predictions["risk_rank"] / len(predictions)
    predictions["risk_level"] = pd.cut(
        rank_fraction, bins=[0, .01, .05, .20, 1.0], labels=["高", "较高", "中", "低"], include_lowest=True
    ).astype("string")
    reason_source = features.set_index("account_id")
    def local_reasons(account_id: str) -> str:
        row = reason_source.loc[account_id]
        reasons = []
        if not row["has_transaction"]:
            reasons.append("缺乏交易轨迹")
        if row["opening_months"] < 36:
            reasons.append("开户时长较短")
        if row["high_amount_ratio"] >= .25:
            reasons.append("大额交易占比较高")
        if row["rapid_turnover_day_ratio"] >= .5:
            reasons.append("同日收付活跃")
        if row["graph_core_number"] >= 5:
            reasons.append("处于较高k-core层")
        if row["train_suspect_neighbor_count"] > 0:
            reasons.append("连接训练集嫌疑账户")
        return "；".join(reasons[:3] or ["多变量组合风险"])
    predictions["top_risk_factors"] = predictions["account_id"].map(local_reasons)
    write_table(predictions.sort_values("risk_rank"), "outputs/risk_accounts/risk_account_ranking.parquet")
    write_table(predictions.sort_values("risk_rank"), "outputs/risk_accounts/risk_account_ranking.csv")
    importance = feature_importance(primary_final)
    write_table(importance, "outputs/model_results/primary_feature_importance.csv")

    test_eval = {
        "rule_baseline": rule_metrics,
        "primary_candidates": primary_results,
        "auxiliary_candidates": aux_results,
        "conditional_candidates": conditional_results,
        "two_stage": two_stage,
        "selected_primary_strategy": selected_strategy,
        "selected_primary_test": selected_test,
        "improvement_vs_rule_baseline": improvement(selected_test, rule_metrics["test"]),
        "evaluation_policy": {
            "split": "account-disjoint stratified 70/15/15",
            "model_selection": "validation PR-AUC",
            "threshold": "validation F1; fixed before test",
            "test_usage": "single final evaluation",
            "top_metric_definition": "Top x% of the evaluated account split",
            "auxiliary_metrics_replace_primary": False,
        },
    }
    write_json(test_eval, "outputs/model_results/model_evaluation.json")
    write_json({
        "selected_strategy": selected_strategy,
        "direct_model": direct_name,
        "auxiliary_model": aux_name,
        "conditional_model": conditional_name,
        "probability_calibration": "Platt sigmoid fitted on validation split",
        "primary_threshold_validation": primary_results["validation"][direct_name]["threshold"],
        "two_stage_threshold_validation": two_stage_threshold,
        "feature_count_before_encoding": X.shape[1],
        "training_account_count": int(masks["train_mask"].sum()),
        "validation_account_count": int(masks["validation_mask"].sum()),
        "test_account_count": int(masks["test_mask"].sum()),
    }, "models/metadata/model_metadata.json")
    logging.info("模型完成，主策略=%s，测试 PR-AUC=%.4f", selected_strategy, selected_test["pr_auc"])


if __name__ == "__main__":
    main()
