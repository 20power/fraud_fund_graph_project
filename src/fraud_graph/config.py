from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(name: str) -> dict[str, Any]:
    path = PROJECT_ROOT / "configs" / name
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_configs() -> dict[str, dict[str, Any]]:
    return {
        key: load_yaml(f"{key}_config.yaml")
        for key in ("data", "feature", "model", "graph", "acceptance")
    }


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def ensure_output_dirs() -> None:
    for relative in (
        "data/interim", "data/processed", "data/samples", "logs",
        "models/baseline", "models/final", "models/metadata",
        "outputs/data_quality", "outputs/graph_statistics",
        "outputs/model_results", "outputs/risk_accounts",
        "outputs/link_analysis", "outputs/case_studies", "outputs/figures",
        "reports", "docs",
    ):
        (PROJECT_ROOT / relative).mkdir(parents=True, exist_ok=True)
