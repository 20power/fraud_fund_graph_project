from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def make_account_split(accounts: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    split_cfg = cfg["split"]
    seed = int(split_cfg["random_seed"])
    train_ratio = float(split_cfg["train_ratio"])
    val_ratio = float(split_cfg["validation_ratio"])
    ids = accounts["account_id"]
    y = (accounts["account_label"] == int(cfg["labels"]["suspect"])).astype(int)
    train_ids, remaining_ids, _, y_remaining = train_test_split(
        ids, y, train_size=train_ratio, random_state=seed, stratify=y
    )
    val_share = val_ratio / (1.0 - train_ratio)
    val_ids, test_ids = train_test_split(
        remaining_ids, train_size=val_share, random_state=seed, stratify=y_remaining
    )
    mapping = pd.concat([
        pd.DataFrame({"account_id": train_ids, "split": "train"}),
        pd.DataFrame({"account_id": val_ids, "split": "validation"}),
        pd.DataFrame({"account_id": test_ids, "split": "test"}),
    ], ignore_index=True)
    return accounts[["account_id", "account_label"]].merge(mapping, on="account_id", how="left")


def validate_split(split: pd.DataFrame) -> dict:
    by_split = split.groupby("split").agg(
        accounts=("account_id", "size"),
        suspect_count=("account_label", lambda x: int((x == 1).sum())),
        victim_count=("account_label", lambda x: int((x == 2).sum())),
    ).reset_index()
    return {
        "account_disjoint": split.groupby("account_id")["split"].nunique().max() == 1,
        "all_accounts_assigned": bool(split["split"].notna().all()),
        "counts": by_split.to_dict(orient="records"),
    }
