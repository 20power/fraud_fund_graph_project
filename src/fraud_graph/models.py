from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


NON_FEATURE_COLUMNS = {
    "account_id", "account_label", "primary_target", "auxiliary_target", "split",
    "cutoff_time", "out_first_transaction", "out_last_transaction",
    "in_first_transaction", "in_last_transaction", "graph_component_id", "graph_community_id",
}


def model_frame(features: pd.DataFrame) -> pd.DataFrame:
    frame = features.drop(columns=[c for c in NON_FEATURE_COLUMNS if c in features], errors="ignore").copy()
    for column in frame.select_dtypes(include=["datetime", "datetimetz"]).columns:
        frame[column] = frame[column].astype("int64") / 1e9
    return frame


def make_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    categorical = list(frame.select_dtypes(include=["object", "string", "category"]).columns)
    numeric = [c for c in frame.columns if c not in categorical]
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", RobustScaler(with_centering=False)),
        ]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=3)),
        ]), categorical),
    ], sparse_threshold=.3)


def build_candidates(frame: pd.DataFrame, cfg: dict[str, Any], seed: int = 42) -> dict[str, Pipeline]:
    model_cfg = cfg["models"]
    candidates: dict[str, Pipeline] = {}
    if model_cfg.get("logistic_regression"):
        params = cfg["logistic_regression"]
        candidates["logistic_regression"] = Pipeline([
            ("preprocessor", make_preprocessor(frame)),
            ("model", LogisticRegression(
                C=float(params["C"]), max_iter=int(params["max_iter"]),
                class_weight=params["class_weight"], random_state=seed,
            )),
        ])
    if model_cfg.get("random_forest"):
        params = cfg["random_forest"]
        candidates["random_forest"] = Pipeline([
            ("preprocessor", make_preprocessor(frame)),
            ("model", RandomForestClassifier(
                n_estimators=int(params["n_estimators"]), max_depth=int(params["max_depth"]),
                min_samples_leaf=int(params["min_samples_leaf"]), class_weight=params["class_weight"],
                random_state=seed, n_jobs=-1,
            )),
        ])
    if model_cfg.get("lightgbm"):
        params = cfg["lightgbm"]
        candidates["lightgbm"] = Pipeline([
            ("preprocessor", make_preprocessor(frame)),
            ("model", LGBMClassifier(
                objective="binary", verbosity=-1, random_state=seed, n_jobs=-1,
                **params,
            )),
        ])
    return candidates


def fit_rule_baseline(train: pd.DataFrame) -> dict[str, tuple[float, float, float, float]]:
    # 可审计的专家规则：新开户、缺乏常规交易轨迹、高额/快进快出和风险邻居提高风险。
    specification = {
        "opening_months": (-1.0, 1.0),
        "has_transaction": (-1.0, 1.5),
        "transaction_count_total": (1.0, .20),
        "amount_abs_total": (1.0, .20),
        "high_amount_ratio": (1.0, .40),
        "rapid_turnover_day_ratio": (1.0, .40),
        "train_suspect_neighbor_count": (1.0, 1.0),
    }
    result = {}
    for column, (direction, weight) in specification.items():
        values = np.log1p(train[column].clip(lower=0)) if column in {
            "transaction_count_total", "amount_abs_total", "counterparty_total"
        } else train[column]
        median = float(values.median())
        scale = float((values - median).abs().median()) or float(values.std()) or 1.0
        result[column] = (median, scale, direction, weight)
    return result


def rule_scores(frame: pd.DataFrame, state: dict[str, tuple[float, float, float, float]]) -> np.ndarray:
    score = np.zeros(len(frame), dtype=float)
    total_weight = 0.0
    for column, (median, scale, direction, weight) in state.items():
        values = np.log1p(frame[column].clip(lower=0)) if column in {
            "transaction_count_total", "amount_abs_total", "counterparty_total"
        } else frame[column]
        score += direction * weight * np.clip((values.to_numpy(dtype=float) - median) / scale, -5, 5)
        total_weight += weight
    return 1.0 / (1.0 + np.exp(-score / total_weight))


def feature_importance(pipeline: Pipeline, top_n: int = 80) -> pd.DataFrame:
    pre = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    names = pre.get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_[0])
    else:
        return pd.DataFrame(columns=["feature", "importance"])
    return (pd.DataFrame({"feature": names, "importance": importance})
            .sort_values("importance", ascending=False).head(top_n).reset_index(drop=True))
