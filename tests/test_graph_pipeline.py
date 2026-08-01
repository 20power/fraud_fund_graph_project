import pandas as pd

from fraud_graph.features import aggregate_relationship_edges, build_behavior_features, build_graph_features


def test_graph_excludes_self_loops(synthetic_data):
    accounts, _, edges = synthetic_data
    _, history = build_behavior_features(accounts, edges, "2025-12-31")
    pairs = aggregate_relationship_edges(history)
    assert not (pairs.source == pairs.target).any()
    split = accounts[["account_id", "account_label"]].assign(split="train")
    graph_features, graph = build_graph_features(accounts, pairs, split)
    assert graph.number_of_nodes() == len(accounts)
    assert len(graph_features) == len(accounts)
