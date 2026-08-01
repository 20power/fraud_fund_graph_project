from fraud_graph.features import build_behavior_features


def test_cutoff_and_cold_start(synthetic_data):
    accounts, _, edges = synthetic_data
    features, history = build_behavior_features(accounts, edges, "2025-01-02 23:59:59")
    assert history.transaction_time.max().day == 2
    assert len(features) == len(accounts)
    assert features.set_index("account_id").loc["8", "has_transaction"] == 0
