from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud_graph.io import write_table  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    transactions = pd.read_parquet(ROOT / "data/processed/transactions_cutoff.parquet")
    directed = transactions.loc[transactions.payer_id != transactions.payee_id].copy()
    directed["month"] = directed.transaction_time.dt.to_period("M").astype(str)
    directed["amount_abs"] = directed.amount.abs()
    monthly = directed.groupby(["month", "payer_id", "payee_id"], observed=True).agg(
        transaction_count=("amount", "size"), amount_signed_sum=("amount", "sum"),
        amount_abs_sum=("amount_abs", "sum"), amount_abs_min=("amount_abs", "min"),
        amount_abs_max=("amount_abs", "max"), first_transaction=("transaction_time", "min"),
        last_transaction=("transaction_time", "max"),
    ).reset_index().rename(columns={"payer_id": "source", "payee_id": "target"})
    all_time = directed.groupby(["payer_id", "payee_id"], observed=True).agg(
        transaction_count=("amount", "size"), amount_signed_sum=("amount", "sum"),
        amount_abs_sum=("amount_abs", "sum"), amount_abs_min=("amount_abs", "min"),
        amount_abs_max=("amount_abs", "max"), first_transaction=("transaction_time", "min"),
        last_transaction=("transaction_time", "max"),
    ).reset_index().rename(columns={"payer_id": "source", "payee_id": "target"})
    write_table(monthly, "data/processed/monthly_directed_relationship_edges.parquet")
    write_table(all_time, "data/processed/directed_relationship_edges.parquet")
    write_table(monthly, "data/processed/monthly_directed_relationship_edges.csv")
    logging.info("UI定向关系数据完成：月度%s条，全期%s条", f"{len(monthly):,}", f"{len(all_time):,}")


if __name__ == "__main__":
    main()
