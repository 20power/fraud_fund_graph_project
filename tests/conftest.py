from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def synthetic_data():
    accounts = pd.DataFrame({
        "account_id": [str(i) for i in range(1, 9)],
        "account_label": pd.Series([1, 0, 0, 2, 0, 0, 0, 0], dtype="Int64"),
        "opening_months": [6, 50, 70, 20, 90, 100, 120, 80],
        "region_code": pd.Series(["A", "A", "B", "B", "A", "C", "C", "A"], dtype="string"),
        "customer_type": pd.Series(["对私"] * 8, dtype="string"),
    })
    labels = accounts[["account_id", "account_label"]].rename(columns={"account_label": "label_type"})
    labels["label_type_raw"] = labels["label_type"].astype("string")
    edges = pd.DataFrame({
        "payer_id": pd.Series(["1", "2", "2", "4"], dtype="string"),
        "payee_id": pd.Series(["2", "3", "2", "3"], dtype="string"),
        "transaction_time": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]),
        "amount": [100.0, -5.0, 10.0, 200.0],
    })
    return accounts, labels, edges
