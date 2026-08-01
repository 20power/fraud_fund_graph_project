from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud_graph.config import load_configs  # noqa: E402
from fraud_graph.features import (  # noqa: E402
    aggregate_relationship_edges,
    build_behavior_features,
    build_graph_features,
    graph_statistics,
)
from fraud_graph.io import write_json, write_table  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_configs()
    accounts = pd.read_parquet(ROOT / "data/interim/accounts.parquet")
    edges = pd.read_parquet(ROOT / "data/interim/edges.parquet")
    split = pd.read_parquet(ROOT / "data/processed/account_split.parquet")
    cutoff = cfg["data"]["time"]["feature_cutoff"]
    logging.info("生成交易行为特征，截止时间=%s", cutoff)
    behavior, history = build_behavior_features(accounts, edges, cutoff)
    if (history["transaction_time"] > pd.Timestamp(cutoff)).any():
        raise AssertionError("检测到截止时间后的交易进入特征")
    relationship_edges = aggregate_relationship_edges(history)
    logging.info("生成关系图特征：%s 条聚合关系", f"{len(relationship_edges):,}")
    graph_features, graph = build_graph_features(
        accounts, relationship_edges, split, cfg["feature"]["graph"]["pagerank_alpha"]
    )
    features = behavior.merge(graph_features, on="account_id", how="left").merge(
        split[["account_id", "split"]], on="account_id", how="left"
    )
    graph_numeric = [c for c in graph_features.columns if c != "account_id"]
    features[graph_numeric] = features[graph_numeric].fillna(0)
    write_table(history, "data/processed/transactions_cutoff.parquet")
    write_table(relationship_edges, "data/processed/relationship_edges.parquet")
    write_table(graph_features, "data/processed/graph_features.parquet")
    write_table(features, "data/processed/account_features.parquet")
    write_table(features, "data/processed/account_features.csv")
    stats = graph_statistics(graph, relationship_edges)
    stats.update({
        "feature_cutoff": str(cutoff),
        "transactions_used": int(len(history)),
        "transactions_after_cutoff_used": int((history["transaction_time"] > pd.Timestamp(cutoff)).sum()),
        "suspects_in_relationship_graph": int(features.loc[(features.account_label == 1) & (features.has_relationship_edge == 1)].shape[0]),
        "victims_in_relationship_graph": int(features.loc[(features.account_label == 2) & (features.has_relationship_edge == 1)].shape[0]),
    })
    write_json(stats, "outputs/graph_statistics/graph_statistics.json")
    logging.info("特征与图谱构建完成：%s 个账户，%s 个特征列", f"{len(features):,}", len(features.columns))


if __name__ == "__main__":
    main()
