import pandas as pd

from fraud_graph.dashboard import build_directed_network_graph, build_filtered_network_figure


def test_dynamic_graph_raw_and_preaggregated_inputs_match():
    transactions = pd.DataFrame({
        "payer_id": ["1", "1", "2"],
        "payee_id": ["2", "2", "3"],
        "transaction_time": pd.to_datetime([
            "2025-07-01 10:00:00", "2025-07-02 11:00:00", "2025-07-03 12:00:00"
        ]),
        "amount": [10.0, 20.0, 30.0],
        "amount_abs": [10.0, 20.0, 30.0],
    })
    aggregated = pd.DataFrame({
        "source": ["1", "2"], "target": ["2", "3"],
        "transaction_count": [2, 1], "amount_abs_sum": [30.0, 30.0],
        "last_transaction": pd.to_datetime(["2025-07-02 11:00:00", "2025-07-03 12:00:00"]),
    })
    ranking = pd.DataFrame({
        "account_id": ["1", "2", "3"], "risk_score": [.9, .4, .7],
        "risk_level": ["高", "低", "中"],
    })
    progress_events = []
    raw_result = build_filtered_network_figure(
        "1", transactions, ranking, max_hops=2,
        allowed_risk_levels=["高", "中", "低"], progress=lambda value, message: progress_events.append((value, message)),
    )
    aggregated_result = build_filtered_network_figure(
        "1", None, ranking, max_hops=2, allowed_risk_levels=["高", "中", "低"],
        aggregated_edges=aggregated,
    )
    prebuilt_result = build_filtered_network_figure(
        "1", None, ranking, max_hops=2, allowed_risk_levels=["高", "中", "低"],
        prebuilt_graph=build_directed_network_graph(aggregated),
    )
    expected_stats = {"nodes": 3, "edges": 2, "transactions": 3}
    assert raw_result[2] == aggregated_result[2] == prebuilt_result[2] == expected_stats
    assert set(raw_result[1]["related_account_id"]) == set(aggregated_result[1]["related_account_id"]) == {"2", "3"}
    assert progress_events[-1][0] == 100
