import numpy as np
import pandas as pd

from fraud_graph.evaluation import classification_metrics, select_f1_threshold
from fraud_graph.models import build_candidates


def test_model_fit_and_inference():
    rng = np.random.default_rng(42)
    y = pd.Series([0, 1] * 40)
    X = pd.DataFrame({"x": y + rng.normal(scale=.15, size=80), "category": pd.Series(["a", "b"] * 40, dtype="string")})
    cfg = {
        "models": {"logistic_regression": True, "random_forest": False, "lightgbm": False},
        "logistic_regression": {"C": 1.0, "max_iter": 200, "class_weight": "balanced"},
    }
    model = build_candidates(X, cfg)["logistic_regression"].fit(X.iloc[:60], y.iloc[:60])
    score = model.predict_proba(X.iloc[60:])[:, 1]
    threshold = select_f1_threshold(y.iloc[60:].to_numpy(), score)
    assert classification_metrics(y.iloc[60:].to_numpy(), score, threshold)["roc_auc"] > .9
