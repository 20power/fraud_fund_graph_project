import pandas as pd

from fraud_graph.link_analysis import build_relationship_graph, related_accounts


def test_three_hop_query():
    pairs = pd.DataFrame({
        "source": ["1", "2", "3"], "target": ["2", "3", "4"], "weight": [1., 1., 1.],
        "transaction_count": [1, 1, 1], "amount_abs_sum": [10., 20., 30.], "amount_signed_sum": [10., 20., 30.],
        "first_transaction": pd.to_datetime(["2025-01-01"] * 3),
        "last_transaction": pd.to_datetime(["2025-12-01"] * 3), "direction_count": [1, 1, 1],
    })
    graph = build_relationship_graph(pairs)
    info = pd.DataFrame({
        "account_id": ["1", "2", "3", "4"], "risk_score": [.9, .2, .3, .8],
        "graph_community_id": [1, 1, 1, 1], "graph_core_number": [1, 1, 1, 1],
    })
    result = related_accounts("1", graph, info, max_hops=3, top_n=20)
    assert set(result.hops) == {1, 2, 3}
