from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


REQUIRED_ACCOUNT_COLUMNS = {"account_id", "account_label", "opening_months", "region_code", "customer_type"}
REQUIRED_LABEL_COLUMNS = {"account_id", "label_type"}
REQUIRED_EDGE_COLUMNS = {"payer_id", "payee_id", "transaction_time", "amount"}


def _required_columns(actual: set[str], required: set[str], table: str) -> None:
    missing = required - actual
    if missing:
        raise ValueError(f"{table} missing required columns: {sorted(missing)}")


def validate_sources(accounts: pd.DataFrame, labels: pd.DataFrame, edges: pd.DataFrame) -> dict[str, Any]:
    _required_columns(set(accounts.columns), REQUIRED_ACCOUNT_COLUMNS, "accounts")
    _required_columns(set(labels.columns), REQUIRED_LABEL_COLUMNS, "labels")
    _required_columns(set(edges.columns), REQUIRED_EDGE_COLUMNS, "edges")

    account_ids = set(accounts["account_id"].dropna())
    label_ids = set(labels["account_id"].dropna())
    endpoint_ids = set(edges["payer_id"].dropna()) | set(edges["payee_id"].dropna())
    joined = accounts[["account_id", "account_label"]].merge(labels, on="account_id", how="outer", indicator=True)
    label_mismatch = joined.loc[
        (joined["_merge"] == "both") & (joined["account_label"] != joined["label_type"])
    ]
    exact_duplicates = int(edges.duplicated().sum())
    quantiles = edges["amount"].quantile([0, .01, .5, .95, .99, 1]).replace({np.nan: None}).to_dict()

    report: dict[str, Any] = {
        "status": "pass",
        "account_count": int(len(accounts)),
        "label_row_count": int(len(labels)),
        "edge_count": int(len(edges)),
        "unique_account_ids": int(accounts["account_id"].nunique(dropna=True)),
        "unique_label_ids": int(labels["account_id"].nunique(dropna=True)),
        "duplicate_account_ids": int(accounts["account_id"].duplicated().sum()),
        "duplicate_label_ids": int(labels["account_id"].duplicated().sum()),
        "null_counts": {
            "accounts": {k: int(v) for k, v in accounts.isna().sum().items()},
            "labels": {k: int(v) for k, v in labels.isna().sum().items()},
            "edges": {k: int(v) for k, v in edges.isna().sum().items()},
        },
        "label_distribution": {str(k): int(v) for k, v in accounts["account_label"].value_counts().sort_index().items()},
        "account_label_id_sets_equal": account_ids == label_ids,
        "account_label_mismatch_count": int(len(label_mismatch)),
        "orphan_endpoint_count": int(len(endpoint_ids - account_ids)),
        "edge_covered_account_count": int(len(endpoint_ids & account_ids)),
        "no_edge_account_count": int(len(account_ids - endpoint_ids)),
        "self_loop_count": int((edges["payer_id"] == edges["payee_id"]).sum()),
        "negative_amount_count": int((edges["amount"] < 0).sum()),
        "zero_amount_count": int((edges["amount"] == 0).sum()),
        "exact_duplicate_edge_count": exact_duplicates,
        "transaction_time_min": edges["transaction_time"].min(),
        "transaction_time_max": edges["transaction_time"].max(),
        "amount_quantiles": {str(k): v for k, v in quantiles.items()},
        "known_note_account_count": 11088,
        "effective_account_count": int(len(accounts)),
        "known_note_difference": int(11088 - len(accounts)),
        "policies": {
            "negative_amount": "保留原始金额；另生成绝对金额与负金额标记；聚合流量使用绝对金额，净流量保留符号。",
            "self_loop": "保留用于行为特征；关系图构建时剔除，避免自环扭曲中心性。",
            "duplicate_edges": "精确重复交易不自动删除，仅记录；本批数据无精确重复。",
            "label_time": "标签统一视为数据期末状态；缺少标签确认时间，无法验证标签确认时间层面的未来信息泄露。",
        },
    }
    hard_failures = [
        report["duplicate_account_ids"] > 0,
        report["duplicate_label_ids"] > 0,
        not report["account_label_id_sets_equal"],
        report["account_label_mismatch_count"] > 0,
        report["orphan_endpoint_count"] > 0,
        any(report["null_counts"]["accounts"].values()),
        any(report["null_counts"]["labels"].values()),
        any(report["null_counts"]["edges"].values()),
    ]
    if any(hard_failures):
        report["status"] = "fail"
    return report
