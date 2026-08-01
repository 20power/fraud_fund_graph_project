from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import pandas as pd


def _flatten_columns(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    frame.columns = [prefix + "_" + "_".join(c).strip("_") if isinstance(c, tuple) else prefix + "_" + c for c in frame.columns]
    return frame


def _directional_features(edges: pd.DataFrame, id_col: str, prefix: str) -> pd.DataFrame:
    grouped = edges.groupby(id_col, observed=True).agg(
        transaction_count=("amount", "size"),
        counterparty_count=("payee_id" if id_col == "payer_id" else "payer_id", "nunique"),
        amount_signed_sum=("amount", "sum"),
        amount_abs_sum=("amount_abs", "sum"),
        amount_abs_mean=("amount_abs", "mean"),
        amount_abs_std=("amount_abs", "std"),
        amount_abs_max=("amount_abs", "max"),
        active_days=("transaction_date", "nunique"),
        first_transaction=("transaction_time", "min"),
        last_transaction=("transaction_time", "max"),
        negative_count=("is_negative", "sum"),
        high_amount_count=("is_high_amount", "sum"),
        night_count=("is_night", "sum"),
        weekend_count=("is_weekend", "sum"),
    ).reset_index().rename(columns={id_col: "account_id"})
    numeric = [c for c in grouped.columns if c not in {"account_id", "first_transaction", "last_transaction"}]
    grouped = grouped.rename(columns={c: f"{prefix}_{c}" for c in numeric + ["first_transaction", "last_transaction"]})
    return grouped


def build_behavior_features(
    accounts: pd.DataFrame,
    edges: pd.DataFrame,
    cutoff: str | pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff_ts = pd.Timestamp(cutoff)
    history = edges.loc[edges["transaction_time"] <= cutoff_ts].copy()
    history["amount_abs"] = history["amount"].abs()
    history["is_negative"] = (history["amount"] < 0).astype("int8")
    high_threshold = float(history["amount_abs"].quantile(.95)) if len(history) else 0.0
    history["is_high_amount"] = (history["amount_abs"] >= high_threshold).astype("int8")
    history["transaction_date"] = history["transaction_time"].dt.floor("D")
    history["is_night"] = history["transaction_time"].dt.hour.between(0, 5).astype("int8")
    history["is_weekend"] = (history["transaction_time"].dt.dayofweek >= 5).astype("int8")
    history["is_self_loop"] = (history["payer_id"] == history["payee_id"]).astype("int8")

    base = accounts[["account_id", "account_label", "opening_months", "region_code", "customer_type"]].copy()
    base["primary_target"] = (base["account_label"] == 1).astype("int8")
    base["auxiliary_target"] = base["account_label"].isin([1, 2]).astype("int8")
    outgoing = _directional_features(history, "payer_id", "out")
    incoming = _directional_features(history, "payee_id", "in")
    features = base.merge(outgoing, on="account_id", how="left").merge(incoming, on="account_id", how="left")

    events_in = (history[["payee_id", "transaction_date"]]
                 .rename(columns={"payee_id": "account_id"}).drop_duplicates().assign(has_in=1))
    events_out = (history[["payer_id", "transaction_date"]]
                  .rename(columns={"payer_id": "account_id"}).drop_duplicates().assign(has_out=1))
    daily_flags = events_in.merge(events_out, on=["account_id", "transaction_date"], how="outer").fillna(0)
    turnover = daily_flags.assign(both=lambda d: (d.has_in * d.has_out)).groupby("account_id").agg(
        active_days_any=("both", "size"), rapid_turnover_days=("both", "sum")
    ).reset_index()
    turnover["rapid_turnover_day_ratio"] = turnover["rapid_turnover_days"] / turnover["active_days_any"].clip(lower=1)
    features = features.merge(turnover, on="account_id", how="left")

    self_loop = history.loc[history["is_self_loop"] == 1].groupby("payer_id").agg(
        self_loop_count=("amount", "size"), self_loop_amount_abs_sum=("amount_abs", "sum")
    ).reset_index().rename(columns={"payer_id": "account_id"})
    features = features.merge(self_loop, on="account_id", how="left")

    numeric_cols = features.select_dtypes(include=["number"]).columns.difference(["account_label", "primary_target", "auxiliary_target"])
    features[numeric_cols] = features[numeric_cols].fillna(0)
    features["has_transaction"] = ((features.get("out_transaction_count", 0) + features.get("in_transaction_count", 0)) > 0).astype("int8")
    features["transaction_count_total"] = features.get("out_transaction_count", 0) + features.get("in_transaction_count", 0)
    features["amount_abs_total"] = features.get("out_amount_abs_sum", 0) + features.get("in_amount_abs_sum", 0)
    features["net_flow_signed"] = features.get("in_amount_signed_sum", 0) - features.get("out_amount_signed_sum", 0)
    features["counterparty_total"] = features.get("out_counterparty_count", 0) + features.get("in_counterparty_count", 0)
    features["night_ratio"] = (
        features.get("out_night_count", 0) + features.get("in_night_count", 0)
    ) / features["transaction_count_total"].clip(lower=1)
    features["weekend_ratio"] = (
        features.get("out_weekend_count", 0) + features.get("in_weekend_count", 0)
    ) / features["transaction_count_total"].clip(lower=1)
    features["high_amount_ratio"] = (
        features.get("out_high_amount_count", 0) + features.get("in_high_amount_count", 0)
    ) / features["transaction_count_total"].clip(lower=1)
    features["negative_amount_count"] = features.get("out_negative_count", 0) + features.get("in_negative_count", 0)
    features["out_in_amount_ratio"] = features.get("out_amount_abs_sum", 0) / features.get("in_amount_abs_sum", pd.Series(0, index=features.index)).clip(lower=1)
    features["cutoff_time"] = cutoff_ts
    return features, history


def aggregate_relationship_edges(history: pd.DataFrame) -> pd.DataFrame:
    rel = history.loc[history["payer_id"] != history["payee_id"]].copy()
    payer = rel["payer_id"].astype(str).to_numpy()
    payee = rel["payee_id"].astype(str).to_numpy()
    rel["source"] = np.minimum(payer, payee)
    rel["target"] = np.maximum(payer, payee)
    pairs = rel.groupby(["source", "target"], observed=True).agg(
        transaction_count=("amount", "size"),
        amount_abs_sum=("amount_abs", "sum"),
        amount_signed_sum=("amount", "sum"),
        first_transaction=("transaction_time", "min"),
        last_transaction=("transaction_time", "max"),
        direction_count=("payer_id", "nunique"),
    ).reset_index()
    pairs["weight"] = np.log1p(pairs["amount_abs_sum"]) + np.log1p(pairs["transaction_count"])
    return pairs


def build_graph_features(
    accounts: pd.DataFrame,
    relationship_edges: pd.DataFrame,
    split: pd.DataFrame,
    pagerank_alpha: float = .85,
) -> tuple[pd.DataFrame, nx.Graph]:
    graph = nx.Graph()
    graph.add_nodes_from(accounts["account_id"].astype(str))
    graph.add_weighted_edges_from(
        relationship_edges[["source", "target", "weight"]].itertuples(index=False, name=None)
    )
    degree = dict(graph.degree())
    weighted_degree = dict(graph.degree(weight="weight"))
    pagerank = nx.pagerank(graph, alpha=pagerank_alpha, weight="weight", max_iter=200)
    core_number = nx.core_number(graph) if graph.number_of_edges() else {n: 0 for n in graph}
    component_id: dict[str, int] = {}
    component_size: dict[str, int] = {}
    for index, nodes in enumerate(nx.connected_components(graph)):
        size = len(nodes)
        for node in nodes:
            component_id[node] = index
            component_size[node] = size
    try:
        communities = nx.community.louvain_communities(graph, weight="weight", seed=42)
    except Exception:
        communities = list(nx.connected_components(graph))
    community_id: dict[str, int] = {}
    community_size: dict[str, int] = {}
    for index, nodes in enumerate(communities):
        size = len(nodes)
        for node in nodes:
            community_id[node] = index
            community_size[node] = size

    train_suspects = set(split.loc[(split["split"] == "train") & (split["account_label"] == 1), "account_id"].astype(str))
    risk_neighbor_count = {
        node: sum(neighbor in train_suspects for neighbor in graph.neighbors(node)) for node in graph
    }
    two_hop_count = {}
    for node in graph:
        one_hop = set(graph.neighbors(node))
        two_hop = set().union(*(set(graph.neighbors(n)) for n in one_hop)) if one_hop else set()
        two_hop.discard(node)
        two_hop_count[node] = len(two_hop - one_hop)

    rows = []
    for node in graph:
        rows.append({
            "account_id": node,
            "graph_degree": degree.get(node, 0),
            "graph_weighted_degree": weighted_degree.get(node, 0.0),
            "graph_pagerank": pagerank.get(node, 0.0),
            "graph_core_number": core_number.get(node, 0),
            "graph_component_id": component_id.get(node, -1),
            "graph_component_size": component_size.get(node, 1),
            "graph_community_id": community_id.get(node, -1),
            "graph_community_size": community_size.get(node, 1),
            "graph_two_hop_count": two_hop_count.get(node, 0),
            "train_suspect_neighbor_count": risk_neighbor_count.get(node, 0),
            "has_relationship_edge": int(degree.get(node, 0) > 0),
        })
    return pd.DataFrame(rows), graph


def graph_statistics(graph: nx.Graph, relationship_edges: pd.DataFrame) -> dict[str, Any]:
    degrees = np.array([degree for _, degree in graph.degree()], dtype=float)
    non_isolates = [n for n, d in graph.degree() if d > 0]
    edge_subgraph = graph.subgraph(non_isolates)
    return {
        "node_count": graph.number_of_nodes(),
        "relationship_edge_count": graph.number_of_edges(),
        "isolated_account_count": int(np.sum(degrees == 0)),
        "edge_covered_account_count": int(np.sum(degrees > 0)),
        "mean_degree": float(degrees.mean()) if len(degrees) else 0.0,
        "max_degree": int(degrees.max()) if len(degrees) else 0,
        "connected_component_count": nx.number_connected_components(graph),
        "largest_component_size": max((len(c) for c in nx.connected_components(graph)), default=0),
        "density_edge_covered": nx.density(edge_subgraph) if len(non_isolates) > 1 else 0.0,
        "aggregated_amount_abs_sum": float(relationship_edges["amount_abs_sum"].sum()),
        "aggregated_transaction_count": int(relationship_edges["transaction_count"].sum()),
    }
