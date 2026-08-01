from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def label_name(value: int) -> str:
    return {0: "其它", 1: "嫌疑人", 2: "受害人"}.get(int(value), "未知")


def build_local_network_figure(
    anchor: str,
    related: pd.DataFrame,
    predictions: pd.DataFrame,
) -> go.Figure:
    graph = nx.Graph()
    graph.add_node(anchor)
    for row in related.head(20).itertuples():
        path = json.loads(row.path)
        for source, target in zip(path[:-1], path[1:]):
            graph.add_edge(str(source), str(target))
    if graph.number_of_nodes() == 1:
        return go.Figure().add_annotation(text="该账户无可展示的关系路径", showarrow=False)
    position = nx.spring_layout(graph, seed=42, k=1.3 / max(graph.number_of_nodes() ** .5, 1))
    edge_x, edge_y = [], []
    for source, target in graph.edges():
        x0, y0 = position[source]
        x1, y1 = position[target]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    score_map = predictions.set_index("account_id")["risk_score"].to_dict()
    label_map = predictions.set_index("account_id")["account_label"].to_dict()
    nodes = list(graph.nodes())
    node_x = [position[node][0] for node in nodes]
    node_y = [position[node][1] for node in nodes]
    colors = [score_map.get(node, 0) for node in nodes]
    sizes = [24 if node == anchor else 10 + 12 * score_map.get(node, 0) for node in nodes]
    text = [
        f"账户 {node}<br>风险分 {score_map.get(node, 0):.4f}<br>标签 {label_name(label_map.get(node, -1))}"
        for node in nodes
    ]
    figure = go.Figure([
        go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1, color="#45647d"), hoverinfo="skip"),
        go.Scatter(
            x=node_x, y=node_y, mode="markers", text=text, hoverinfo="text",
            marker=dict(size=sizes, color=colors, colorscale=[[0, "#2475d0"], [.55, "#f3b84b"], [1, "#ef5b5b"]], showscale=True,
                        colorbar=dict(title=dict(text="风险分", font=dict(color="#dce9f7")), tickfont=dict(color="#9bb2c7")),
                        line=dict(width=1, color="#d6e9f7")),
        ),
    ])
    figure.update_layout(
        height=560, margin=dict(l=10, r=10, t=40, b=10), title=f"账户 {anchor} 的三跳关系子图",
        xaxis=dict(visible=False), yaxis=dict(visible=False), showlegend=False,
        paper_bgcolor="#07111f", plot_bgcolor="#07111f", font=dict(color="#dce9f7"),
    )
    return figure


def build_directed_network_graph(aggregated_edges: pd.DataFrame) -> nx.DiGraph:
    edge_frame = aggregated_edges[[
        "source", "target", "transaction_count", "amount_abs_sum", "last_transaction"
    ]].rename(columns={
        "transaction_count": "count", "amount_abs_sum": "amount", "last_transaction": "last"
    })
    return nx.from_pandas_edgelist(
        edge_frame, "source", "target", edge_attr=["count", "amount", "last"], create_using=nx.DiGraph()
    )


def build_filtered_network_figure(
    anchor: str,
    transactions: pd.DataFrame | None,
    predictions: pd.DataFrame,
    direction: str = "双向",
    max_hops: int = 3,
    allowed_risk_levels: list[str] | None = None,
    node_limit: int = 80,
    aggregated_edges: pd.DataFrame | None = None,
    prebuilt_graph: nx.DiGraph | None = None,
    progress: Callable[[int, str], None] | None = None,
) -> tuple[go.Figure, pd.DataFrame, dict[str, int]]:
    def report(value: int, message: str) -> None:
        if progress is not None:
            progress(value, message)

    score_map = predictions.set_index("account_id")["risk_score"].to_dict()
    level_map = predictions.set_index("account_id")["risk_level"].astype(str).to_dict()
    report(30, "正在载入账户资金关系")
    if prebuilt_graph is not None:
        graph = prebuilt_graph
    elif aggregated_edges is None:
        if transactions is None:
            raise ValueError("transactions 和 aggregated_edges 不能同时为空")
        source = transactions
        if "amount_abs" not in source.columns:
            source = source.assign(amount_abs=source["amount"].abs())
        aggregated = source.groupby(["payer_id", "payee_id"], observed=True).agg(
            transaction_count=("amount", "size"), amount_abs_sum=("amount_abs", "sum"),
            last_transaction=("transaction_time", "max"),
        ).reset_index().rename(columns={"payer_id": "source", "payee_id": "target"})
        graph = build_directed_network_graph(aggregated)
    else:
        graph = build_directed_network_graph(aggregated_edges)
    report(52, "正在搜索锚点账户的多跳关联")
    if anchor not in graph:
        empty = go.Figure().add_annotation(text="筛选条件下该账户无可达资金关系", showarrow=False,
                                            font=dict(color="#9bb2c7", size=16))
        empty.update_layout(height=610, paper_bgcolor="#07111f", plot_bgcolor="#07111f",
                            xaxis=dict(visible=False), yaxis=dict(visible=False))
        return empty, pd.DataFrame(), {"nodes": 1, "edges": 0, "transactions": 0}
    traversal = graph if direction == "转出" else graph.reverse(copy=False) if direction == "转入" else graph.to_undirected()
    lengths = nx.single_source_shortest_path_length(traversal, anchor, cutoff=max_hops)
    candidates = [node for node in lengths if node == anchor or not allowed_risk_levels or level_map.get(node, "低") in allowed_risk_levels]
    candidates = sorted(candidates, key=lambda node: (node != anchor, -score_map.get(node, 0), lengths[node]))[:node_limit]
    node_set = set(candidates)
    subgraph = graph.subgraph(node_set).copy()
    undirected = subgraph.to_undirected()
    report(70, "正在计算图谱布局")
    position = nx.spring_layout(undirected, seed=42, k=1.55 / max(len(node_set) ** .5, 1), iterations=40)
    edge_x, edge_y, annotations = [], [], []
    for index, (source, target, data) in enumerate(subgraph.edges(data=True)):
        x0, y0 = position[source]
        x1, y1 = position[target]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        if index < 60:
            annotations.append(dict(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
                                    showarrow=True, arrowhead=2, arrowsize=.8, arrowwidth=.8, arrowcolor="#5f829d",
                                    opacity=.7, standoff=7, startstandoff=7))
    nodes = list(subgraph.nodes())
    colors = [score_map.get(node, 0) for node in nodes]
    sizes = [26 if node == anchor else 10 + 15 * score_map.get(node, 0) for node in nodes]
    labels = [
        f"账户 {node}<br>风险分 {score_map.get(node, 0):.4f}<br>风险等级 {level_map.get(node, '未知')}<br>距锚点 {lengths.get(node, 0)} 跳"
        for node in nodes
    ]
    figure = go.Figure([
        go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1, color="#45647d"), hoverinfo="skip"),
        go.Scatter(
            x=[position[n][0] for n in nodes], y=[position[n][1] for n in nodes], mode="markers", text=labels,
            hoverinfo="text", marker=dict(size=sizes, color=colors,
                colorscale=[[0, "#2475d0"], [.55, "#f3b84b"], [1, "#ef5b5b"]], cmin=0, cmax=1,
                showscale=True, colorbar=dict(title=dict(text="风险分", font=dict(color="#dce9f7")), thickness=10, len=.55,
                tickfont=dict(color="#9bb2c7")),
                line=dict(width=1, color="#d6e9f7")),
        ),
    ])
    figure.update_layout(
        height=610, margin=dict(l=5, r=5, t=42, b=5), title=f"账户 {anchor} · {direction} · {max_hops}跳资金关系",
        xaxis=dict(visible=False), yaxis=dict(visible=False), paper_bgcolor="#07111f", plot_bgcolor="#07111f",
        font=dict(color="#dce9f7"), showlegend=False, annotations=annotations,
    )
    report(86, "正在生成可疑路径排序")
    path_rows = []
    maximum_amount = max((data["amount"] for _, _, data in subgraph.edges(data=True)), default=1.0)
    for candidate in nodes:
        if candidate == anchor:
            continue
        try:
            path = nx.shortest_path(traversal, anchor, candidate)
        except nx.NetworkXNoPath:
            continue
        directed_path = path if direction != "转入" else list(reversed(path))
        edge_values = []
        for source, target in zip(directed_path[:-1], directed_path[1:]):
            data = graph.get_edge_data(source, target) or graph.get_edge_data(target, source) or {}
            edge_values.append(data)
        amount = sum(item.get("amount", 0) for item in edge_values)
        count = sum(item.get("count", 0) for item in edge_values)
        hops = len(path) - 1
        relationship_score = .5 * score_map.get(candidate, 0) + .3 * np.log1p(amount) / np.log1p(maximum_amount * max_hops) + .2 / hops
        path_rows.append({
            "related_account_id": candidate, "hops": hops, "path": " → ".join(path),
            "relationship_score": float(min(1, relationship_score)), "related_risk_score": score_map.get(candidate, 0),
            "path_transaction_count": count, "path_amount_abs_sum": amount,
        })
    paths = pd.DataFrame(path_rows).sort_values("relationship_score", ascending=False) if path_rows else pd.DataFrame()
    stats = {"nodes": subgraph.number_of_nodes(), "edges": subgraph.number_of_edges(),
             "transactions": int(sum(data["count"] for _, _, data in subgraph.edges(data=True)))}
    report(100, "资金图谱生成完成")
    return figure, paths, stats


def case_markdown(case: dict) -> str:
    patterns = "\n".join(f"- {item}" for item in case["patterns"])
    return (
        f"### {case['case_id']} · 账户 {case['anchor_id']}\n\n"
        f"风险分 **{case['anchor_risk_score']:.4f}**，全量排名 **{case['anchor_risk_rank']}**，"
        f"已知标签 **{label_name(case['anchor_label'])}**。\n\n"
        f"触发模式：\n{patterns}\n\n"
        "说明：确定性本地模板生成，仅作辅助研判，不代表执法结论。"
    )
