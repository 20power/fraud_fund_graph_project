from __future__ import annotations

import json
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd


def build_relationship_graph(pairs: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    for row in pairs.itertuples(index=False):
        graph.add_edge(
            str(row.source), str(row.target), weight=float(row.weight),
            transaction_count=int(row.transaction_count), amount_abs_sum=float(row.amount_abs_sum),
            first_transaction=row.first_transaction, last_transaction=row.last_transaction,
            direction_count=int(row.direction_count),
        )
    return graph


def select_case_anchors(features: pd.DataFrame, predictions: pd.DataFrame, count: int = 5) -> list[str]:
    merged = features.merge(predictions[["account_id", "risk_score", "risk_rank"]], on="account_id")
    covered = merged.loc[merged["has_relationship_edge"].eq(1)].sort_values("risk_rank")
    selected: list[str] = []
    used_communities: set[int] = set()
    known = covered.loc[covered["account_label"].eq(1)]
    for row in known.itertuples():
        selected.append(str(row.account_id))
        used_communities.add(int(row.graph_community_id))
    for row in covered.itertuples():
        if len(selected) >= count:
            break
        account = str(row.account_id)
        community = int(row.graph_community_id)
        if account not in selected and community not in used_communities:
            selected.append(account)
            used_communities.add(community)
    if len(selected) < count:
        selected.extend([str(x) for x in covered.account_id if str(x) not in selected][:count - len(selected)])
    return selected[:count]


def related_accounts(
    anchor: str,
    graph: nx.Graph,
    account_info: pd.DataFrame,
    max_hops: int = 3,
    top_n: int = 20,
) -> pd.DataFrame:
    if anchor not in graph:
        return pd.DataFrame()
    info = account_info.set_index("account_id")
    lengths = nx.single_source_shortest_path_length(graph, anchor, cutoff=max_hops)
    max_weight = max((data["weight"] for _, _, data in graph.edges(data=True)), default=1.0)
    rows = []
    for candidate, hops in lengths.items():
        if candidate == anchor:
            continue
        path = nx.shortest_path(graph, anchor, candidate)
        edge_data = [graph.get_edge_data(a, b) for a, b in zip(path[:-1], path[1:])]
        edge_strength = float(np.mean([e["weight"] / max_weight for e in edge_data]))
        recency_days = max(0, (pd.Timestamp("2025-12-31") - max(e["last_transaction"] for e in edge_data)).days)
        recency = float(np.exp(-recency_days / 90))
        risk = float(info.at[candidate, "risk_score"]) if candidate in info.index else 0.0
        same_community = int(
            candidate in info.index and anchor in info.index
            and info.at[candidate, "graph_community_id"] == info.at[anchor, "graph_community_id"]
        )
        structure = .5 * same_community + .5 * min(1.0, float(info.at[candidate, "graph_core_number"]) / 10) if candidate in info.index else 0.0
        hop_score = (max_hops + 1 - hops) / max_hops
        score = .35 * risk + .25 * edge_strength + .15 * recency + .15 * structure + .10 * hop_score
        rows.append({
            "anchor_id": anchor, "related_account_id": candidate, "hops": int(hops),
            "relationship_score": score, "related_risk_score": risk,
            "edge_strength": edge_strength, "recency_score": recency,
            "same_community": same_community, "path": json.dumps(path, ensure_ascii=False),
            "path_transaction_count": int(sum(e["transaction_count"] for e in edge_data)),
            "path_amount_abs_sum": float(sum(e["amount_abs_sum"] for e in edge_data)),
            "path_last_transaction": max(e["last_transaction"] for e in edge_data),
            "bidirectional_edge_count": int(sum(e["direction_count"] > 1 for e in edge_data)),
        })
    return pd.DataFrame(rows).sort_values("relationship_score", ascending=False).head(top_n).reset_index(drop=True)


def explain_link(row: pd.Series, account_label: int) -> tuple[list[str], str, bool]:
    evidence = [
        f"模型风险分={row['related_risk_score']:.4f}",
        f"{int(row['hops'])}跳资金关系，路径累计{int(row['path_transaction_count'])}笔",
    ]
    if row["path_amount_abs_sum"] > 0:
        evidence.append(f"路径累计绝对金额={row['path_amount_abs_sum']:.2f}")
    if row["same_community"]:
        evidence.append("与锚点处于同一图社区")
    if row["bidirectional_edge_count"]:
        evidence.append(f"含{int(row['bidirectional_edge_count'])}条双向资金关系")
    label_note = {0: "其它", 1: "嫌疑人", 2: "受害人"}.get(int(account_label), "未知")
    explanation = f"关联账户标签={label_note}；" + "；".join(evidence)
    internal_pass = len(evidence) >= 2 and int(row["hops"]) <= 3 and row["path_transaction_count"] > 0
    return evidence, explanation, internal_pass


def detect_patterns(anchor: str, features: pd.DataFrame, pairs: pd.DataFrame, cfg: dict[str, Any]) -> list[str]:
    row = features.set_index("account_id").loc[anchor]
    patterns = []
    thresholds = cfg["patterns"]
    if row.get("in_counterparty_count", 0) >= thresholds["fan_in_min_counterparties"]:
        patterns.append("多账户向单账户归集（fan-in）")
    if row.get("out_counterparty_count", 0) >= thresholds["fan_out_min_counterparties"]:
        patterns.append("单账户向多账户分散（fan-out）")
    if row.get("rapid_turnover_day_ratio", 0) >= thresholds["rapid_transfer_ratio_threshold"]:
        patterns.append("同日收付快进快出")
    incident = pairs.loc[(pairs.source == anchor) | (pairs.target == anchor)]
    if len(incident) and (incident["direction_count"] > 1).mean() >= thresholds["bidirectional_ratio_threshold"]:
        patterns.append("双向往来比例较高")
    if row.get("high_amount_ratio", 0) >= .25:
        patterns.append("大额交易占比较高")
    if row.get("self_loop_count", 0) > 0:
        patterns.append("存在账户自环交易")
    return patterns or ["未触发预设强规则，主要依据模型风险与图关系研判"]
