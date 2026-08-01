from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud_graph.io import write_json  # noqa: E402


def main() -> None:
    source = pd.read_parquet(ROOT / "outputs/link_analysis/top_related_accounts.parquet").copy()
    output_dir = ROOT / "outputs/manual_review"
    output_dir.mkdir(parents=True, exist_ok=True)

    account_ids: set[str] = set(source.anchor_id.astype(str)) | set(source.related_account_id.astype(str))
    parsed_paths: list[list[str]] = []
    for raw in source.path:
        path = [str(value) for value in json.loads(raw)]
        parsed_paths.append(path)
        account_ids.update(path)
    shuffled = list(sorted(account_ids))
    np.random.default_rng(20260720).shuffle(shuffled)
    anonymized = {account_id: f"A{index:05d}" for index, account_id in enumerate(shuffled, 1)}

    source = source.reset_index(drop=True)
    source["匿名锚点账户"] = source.anchor_id.astype(str).map(anonymized)
    source["匿名关联账户"] = source.related_account_id.astype(str).map(anonymized)
    source["匿名资金路径"] = [" → ".join(anonymized[node] for node in path) for path in parsed_paths]
    source = source.sample(frac=1, random_state=20260720).reset_index(drop=True)
    source["样本编号"] = [f"LINK-{index:03d}" for index in range(1, len(source) + 1)]
    source["关系证据摘要"] = source.apply(
        lambda row: (
            f"{int(row.hops)}跳关系；路径累计{int(row.path_transaction_count)}笔；"
            f"累计绝对金额{float(row.path_amount_abs_sum):,.2f}元；"
            f"最近交易{pd.Timestamp(row.path_last_transaction):%Y-%m-%d %H:%M:%S}；"
            f"同社区={'是' if int(row.same_community) else '否'}；"
            f"双向关系{int(row.bidirectional_edge_count)}条"
        ), axis=1,
    )

    review_base = source[[
        "样本编号", "匿名锚点账户", "匿名关联账户", "hops", "匿名资金路径",
        "path_transaction_count", "path_amount_abs_sum", "path_last_transaction",
        "same_community", "bidirectional_edge_count", "关系证据摘要",
    ]].rename(columns={
        "hops": "路径跳数", "path_transaction_count": "路径交易笔数",
        "path_amount_abs_sum": "路径累计绝对金额", "path_last_transaction": "路径最近交易时间",
        "same_community": "是否同一图社区", "bidirectional_edge_count": "双向关系数量",
    })
    review_base["是否同一图社区"] = review_base["是否同一图社区"].map({1: "是", 0: "否"})

    rows = []
    for row in review_base.to_dict("records"):
        for slot in ("A", "B"):
            rows.append({
                **row, "审核席位": slot, "解释合理性": "", "是否值得进一步核查": "",
                "证据充分性": "", "不合理/不足原因": "", "补充意见": "", "审核人": "", "审核时间": "",
            })
    blind = pd.DataFrame(rows)
    blind.to_csv(output_dir / "review_template.csv", index=False, encoding="utf-8-sig")
    (output_dir / "review_records.json").write_text(
        blind.to_json(orient="records", force_ascii=False, date_format="iso"), encoding="utf-8"
    )

    key = source[[
        "样本编号", "anchor_id", "related_account_id", "path", "relationship_score",
        "related_risk_score", "related_label", "known_broad_risk_hit", "internal_explanation_pass",
    ]].rename(columns={
        "anchor_id": "真实锚点账户", "related_account_id": "真实关联账户", "path": "真实路径",
        "relationship_score": "内部关联评分", "related_risk_score": "关联账户模型风险分",
        "related_label": "关联账户标签", "known_broad_risk_hit": "命中广义已知风险",
        "internal_explanation_pass": "内部解释校验通过",
    })
    key.to_csv(output_dir / "blind_review_key.csv", index=False, encoding="utf-8-sig")
    write_json({
        "unique_samples": len(review_base), "review_rows": len(blind), "reviewer_slots": 2,
        "random_seed": 20260720, "blind_fields_hidden": [
            "真实账户编号", "模型风险分", "账户标签", "内部关联评分", "内部解释校验结果",
        ],
    }, "outputs/manual_review/review_manifest.json")


if __name__ == "__main__":
    main()
