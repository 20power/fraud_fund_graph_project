from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import networkx as nx
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text("utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest() -> None:
    excluded_parts = {"__pycache__", ".pytest_cache", ".git", "node_modules"}
    files = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.name == "MANIFEST.json" or any(p in excluded_parts for p in path.parts):
            continue
        files.append({
            "path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path),
        })
    manifest = {
        "project": "fraud_fund_graph_project", "version": (ROOT / "VERSION").read_text().strip(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "file_count": len(files), "files": files,
    }
    (ROOT / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    try:
        quality = load_json("outputs/data_quality/data_quality_report.json")
        require(quality["status"] == "pass", "data quality status")
        require(quality["account_count"] == 11087 and quality["account_label_mismatch_count"] == 0, "account scope")
        print("DATA_CHECK=PASS")

        graph = load_json("outputs/graph_statistics/graph_statistics.json")
        require(graph["transactions_after_cutoff_used"] == 0, "future transaction leakage")
        require(graph["node_count"] == 11087 and graph["relationship_edge_count"] > 0, "graph size")
        print("GRAPH_CHECK=PASS")

        evaluation = load_json("outputs/model_results/model_evaluation.json")
        selected = evaluation["selected_primary_test"]
        improve = evaluation["improvement_vs_rule_baseline"]
        require(selected["roc_auc"] >= .85 or improve["roc_auc"]["absolute"] >= .05, "AUC acceptance")
        require(improve["pr_auc"]["relative"] is not None and improve["pr_auc"]["relative"] >= .20, "PR-AUC acceptance")
        top_relative = improve["top_5_recall"]["relative"]
        require(selected["top_5_percent"]["recall"] >= .50 or (top_relative is not None and top_relative >= .15), "Top5 acceptance")
        ranking = pd.read_parquet(ROOT / "outputs/risk_accounts/risk_account_ranking.parquet")
        require(len(ranking) == 11087 and ranking.risk_score.notna().all(), "risk ranking")
        print("MODEL_CHECK=PASS")

        link = load_json("outputs/link_analysis/link_acceptance.json")
        related = pd.read_csv(ROOT / "outputs/link_analysis/top_related_accounts.csv")
        require(link["anchor_count"] >= 5, "case anchors")
        pairs = pd.read_parquet(ROOT / "data/processed/relationship_edges.parquet")
        relationship_graph = nx.from_pandas_edgelist(pairs, "source", "target")
        observed_counts = related.groupby("anchor_id").size().to_dict()
        for anchor, observed in observed_counts.items():
            reachable = len(nx.single_source_shortest_path_length(relationship_graph, str(anchor), cutoff=3)) - 1
            require(observed == min(20, reachable), f"Top20/reachable mismatch for {anchor}")
        require(link["internal_explanation_pass_rate"] >= .70, "internal explanation")
        print("LINK_CHECK=PASS")

        data_cfg = yaml.safe_load((ROOT / "configs/data_config.yaml").read_text("utf-8"))
        for key in ("accounts", "edges", "labels", "data_note"):
            configured = Path(data_cfg["paths"][key])
            require(not configured.is_absolute(), f"absolute data path: {key}")
            resolved = (ROOT / configured).resolve()
            require(resolved == ROOT or ROOT in resolved.parents, f"path escapes project root: {key}")
            require(resolved.exists(), f"missing self-contained source: {key}")
        windows_root = r"(?<![A-Za-z])[A-Z]:" + r"[\\/]"
        mac_user_root = "/" + "Users/"
        linux_home_root = "/" + "home/"
        absolute_pattern = re.compile(
            rf"(?i)(?:{windows_root}|{re.escape(mac_user_root)}|{re.escape(linux_home_root)})"
        )
        portable_files = [ROOT / "app.py", ROOT / "run_app.bat", ROOT / "Dockerfile"]
        for folder, patterns in (("scripts", ("*.py",)), ("src", ("*.py",)),
                                 ("configs", ("*.yaml", "*.yml")), ("review_tools", ("*.mjs",))):
            for pattern in patterns:
                portable_files.extend((ROOT / folder).rglob(pattern))
        hardcoded = []
        for path in portable_files:
            for line_number, line in enumerate(path.read_text("utf-8", errors="ignore").splitlines(), 1):
                if absolute_pattern.search(line):
                    hardcoded.append(f"{path.relative_to(ROOT).as_posix()}:{line_number}")
        require(not hardcoded, "hardcoded absolute paths: " + ", ".join(hardcoded))
        print("PATH_CHECK=PASS")

        reports = [ROOT / f"reports/0{i}_{name}.md" for i, name in [
            (1, "data_quality_report"), (2, "data_split_and_leakage_report"),
            (3, "model_evaluation_report"), (4, "link_analysis_report"), (5, "final_project_report"),
            (6, "model_robustness_report"),
        ]]
        require(all(path.exists() and path.stat().st_size > 500 for path in reports), "six generated reports")
        require((ROOT / "reports/07_client_manual_review_report.md").exists(), "manual review report status")
        print("REPORT_CHECK=PASS")

        required = [
            "README.md", "readme.txt", "requirements.txt", "environment.yml", "Dockerfile", "VERSION", "app.py",
            "models/final/primary_model.joblib", "models/final/primary_calibrator.joblib",
            "docs/00_DOCUMENT_INDEX.md", "docs/06_验收指标与证据索引.md",
            "outputs/case_studies/cases.json", "outputs/investigation_reports/structured_investigation_reports.csv",
            "outputs/investigation_reports/五个典型案例研判材料.zip",
            "outputs/manual_review/链路盲审表.xlsx", "outputs/manual_review/盲审使用指南.md",
            "outputs/model_results/extended_stability_summary.json", "outputs/figures/model_stability.png",
            "data/raw/账户表.xlsx", "data/raw/交易边表.xlsx", "data/raw/风险标签表.xlsx", "data/raw/说明.txt",
            "docs/08_部署与使用手册.md", "docs/最终交付内容清单.md",
            "tests/test_data_validation.py", "tests/test_app.py", "run_app.bat",
        ]
        require(all((ROOT / path).exists() for path in required), "delivery files")
        build_manifest()
        require((ROOT / "MANIFEST.json").exists(), "manifest")
        print("DELIVERY_CHECK=PASS")
    except Exception as exc:
        print(f"DELIVERY_CHECK=FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
