from fraud_graph.validation import validate_sources


def test_validation_accepts_consistent_sources(synthetic_data):
    report = validate_sources(*synthetic_data)
    assert report["status"] == "pass"
    assert report["negative_amount_count"] == 1
    assert report["self_loop_count"] == 1


def test_validation_rejects_label_mismatch(synthetic_data):
    accounts, labels, edges = synthetic_data
    labels.loc[0, "label_type"] = 0
    assert validate_sources(accounts, labels, edges)["status"] == "fail"
