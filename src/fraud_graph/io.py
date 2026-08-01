from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import PROJECT_ROOT, resolve_project_path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest(data_cfg: dict[str, Any]) -> dict[str, Any]:
    manifest: dict[str, Any] = {}
    for key in ("accounts", "edges", "labels", "data_note"):
        path = resolve_project_path(data_cfg["paths"][key])
        try:
            display_path = path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            display_path = Path(data_cfg["paths"][key]).as_posix()
        manifest[key] = {
            "path": display_path,
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else None,
            "sha256": sha256_file(path) if path.exists() else None,
        }
    return manifest


def read_sources(data_cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = data_cfg["paths"]
    columns = data_cfg["columns"]
    accounts = pd.read_excel(resolve_project_path(paths["accounts"]), engine="openpyxl")
    labels = pd.read_excel(resolve_project_path(paths["labels"]), engine="openpyxl")
    edges = pd.read_excel(resolve_project_path(paths["edges"]), engine="openpyxl")

    accounts = accounts.rename(columns={
        columns["account_id"]: "account_id",
        columns["account_label"]: "account_label",
        columns["opening_months"]: "opening_months",
        columns["region_code"]: "region_code",
        columns["customer_type"]: "customer_type",
    })
    labels = labels.rename(columns={
        columns["account_id"]: "account_id",
        columns["label_type"]: "label_type",
    })
    edges = edges.rename(columns={
        columns["payer_id"]: "payer_id",
        columns["payee_id"]: "payee_id",
        columns["transaction_time"]: "transaction_time",
        columns["amount"]: "amount",
    })

    accounts["account_id"] = accounts["account_id"].astype("string")
    labels["account_id"] = labels["account_id"].astype("string")
    edges["payer_id"] = edges["payer_id"].astype("string")
    edges["payee_id"] = edges["payee_id"].astype("string")
    accounts["region_code"] = accounts["region_code"].astype("string")
    accounts["customer_type"] = accounts["customer_type"].astype("string")
    accounts["account_label"] = pd.to_numeric(accounts["account_label"], errors="coerce").astype("Int64")
    inverse_text_mapping = {str(v): int(k) for k, v in data_cfg["labels"]["text_mapping"].items()}
    label_numeric = pd.to_numeric(labels["label_type"], errors="coerce")
    label_text = labels["label_type"].astype("string").str.strip().map(inverse_text_mapping)
    labels["label_type_raw"] = labels["label_type"].astype("string")
    labels["label_type"] = label_numeric.fillna(label_text).astype("Int64")
    edges["transaction_time"] = pd.to_datetime(edges["transaction_time"], errors="coerce")
    edges["amount"] = pd.to_numeric(edges["amount"], errors="coerce")
    return accounts, labels, edges


def write_json(value: Any, relative_path: str | Path) -> Path:
    path = PROJECT_ROOT / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def write_table(frame: pd.DataFrame, relative_path: str | Path) -> Path:
    path = PROJECT_ROOT / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    elif path.suffix == ".csv":
        frame.to_csv(path, index=False, encoding="utf-8-sig")
    else:
        raise ValueError(f"Unsupported table format: {path.suffix}")
    return path
