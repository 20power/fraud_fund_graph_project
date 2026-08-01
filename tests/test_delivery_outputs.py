from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_core_outputs_exist_and_reconcile():
    ranking = pd.read_parquet(ROOT / "outputs/risk_accounts/risk_account_ranking.parquet")
    features = pd.read_parquet(ROOT / "data/processed/account_features.parquet")
    assert len(ranking) == len(features) == 11087
    assert ranking.risk_rank.nunique() == 11087
    assert (ROOT / "reports/05_final_project_report.md").exists()


def test_enhanced_delivery_outputs_are_self_contained():
    assert (ROOT / "outputs/figures/model_stability.png").stat().st_size > 10_000
    assert (ROOT / "reports/06_model_robustness_report.md").stat().st_size > 1_000
    assert (ROOT / "outputs/manual_review/链路盲审表.xlsx").stat().st_size > 10_000
    manifest = json.loads((ROOT / "outputs/manual_review/review_manifest.json").read_text("utf-8"))
    assert manifest["unique_samples"] == 84
    assert manifest["review_rows"] == 168
    for index in range(1, 6):
        report = ROOT / f"outputs/case_studies/CASE-{index:03d}/report.html"
        content = report.read_text("utf-8")
        assert "data:image/png;base64," in content
        assert len(content) > 50_000
