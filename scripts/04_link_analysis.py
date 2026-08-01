from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud_graph.config import load_configs  # noqa: E402
from fraud_graph.io import write_json, write_table  # noqa: E402
from fraud_graph.link_analysis import (  # noqa: E402
    build_relationship_graph, detect_patterns, explain_link, related_accounts, select_case_anchors,
)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_configs()
    features = pd.read_parquet(ROOT / "data/processed/account_features.parquet")
    predictions = pd.read_parquet(ROOT / "outputs/risk_accounts/risk_account_ranking.parquet")
    pairs = pd.read_parquet(ROOT / "data/processed/relationship_edges.parquet")
    graph = build_relationship_graph(pairs)
    account_info = features.merge(predictions[["account_id", "risk_score", "risk_rank"]], on="account_id")
    anchors = select_case_anchors(features, predictions, count=max(5, int(cfg["graph"]["max_anchors"])))
    all_related = []
    cases = []
    for case_index, anchor in enumerate(anchors, 1):
        related = related_accounts(
            anchor, graph, account_info,
            max_hops=int(cfg["graph"]["max_hops"]), top_n=int(cfg["graph"]["top_related_accounts"]),
        )
        if related.empty:
            continue
        label_map = features.set_index("account_id")["account_label"]
        explanations, evidence_types, passes = [], [], []
        for _, row in related.iterrows():
            evidence, explanation, passed = explain_link(row, int(label_map.get(row.related_account_id, -1)))
            evidence_types.append(" | ".join(evidence))
            explanations.append(explanation)
            passes.append(passed)
        related["evidence"] = evidence_types
        related["explanation"] = explanations
        related["internal_explanation_pass"] = passes
        related["related_label"] = related["related_account_id"].map(label_map).fillna(-1).astype(int)
        related["known_broad_risk_hit"] = related["related_label"].isin([1, 2]).astype(int)
        all_related.append(related)
        anchor_row = account_info.set_index("account_id").loc[anchor]
        patterns = detect_patterns(anchor, features, pairs, cfg["graph"])
        top_paths = related.head(5).to_dict(orient="records")
        case = {
            "case_id": f"CASE-{case_index:03d}", "anchor_id": anchor,
            "anchor_label": int(anchor_row.account_label), "anchor_risk_score": float(anchor_row.risk_score),
            "anchor_risk_rank": int(anchor_row.risk_rank), "community_id": int(anchor_row.graph_community_id),
            "patterns": patterns, "top_paths": top_paths,
            "evidence_types": ["模型风险", "交易强度", "多跳路径", "图社区/结构", "行为模式"],
        }
        cases.append(case)
        if len(cases) >= 5:
            break
    combined = pd.concat(all_related, ignore_index=True) if all_related else pd.DataFrame()
    write_table(combined, "outputs/link_analysis/top_related_accounts.csv")
    write_table(combined, "outputs/link_analysis/top_related_accounts.parquet")
    path_columns = [
        "anchor_id", "related_account_id", "hops", "path", "relationship_score",
        "path_transaction_count", "path_amount_abs_sum", "path_last_transaction", "explanation",
    ]
    write_table(combined[path_columns], "outputs/link_analysis/suspicious_paths.csv")
    write_json(cases, "outputs/case_studies/cases.json")
    for case in cases:
        lines = [
            f"# {case['case_id']} 可疑资金关系案例", "",
            f"- 锚点账户：{case['anchor_id']}",
            f"- 已知标签：{case['anchor_label']}",
            f"- 模型风险分/排名：{case['anchor_risk_score']:.4f} / {case['anchor_risk_rank']}",
            f"- 图社区：{case['community_id']}", "", "## 触发模式", "",
            *[f"- {item}" for item in case["patterns"]], "", "## 重点路径", "",
        ]
        for path in case["top_paths"]:
            lines.append(f"- {path['path']}：{path['explanation']}")
        lines += ["", "## 研判提示", "", "本案例由确定性本地模板生成，仅用于辅助研判，不代表执法结论。"]
        (ROOT / f"outputs/case_studies/{case['case_id']}.md").write_text("\n".join(lines), encoding="utf-8")
    summary = {
        "anchor_count": len(cases), "top_n_per_anchor": int(cfg["graph"]["top_related_accounts"]),
        "max_hops": int(cfg["graph"]["max_hops"]),
        "known_broad_risk_hit_count": int(combined["known_broad_risk_hit"].sum()),
        "known_broad_risk_hit_rate": float(combined["known_broad_risk_hit"].mean()),
        "internal_explanation_pass_count": int(combined["internal_explanation_pass"].sum()),
        "internal_explanation_pass_rate": float(combined["internal_explanation_pass"].mean()),
        "review_status": "内部规则验收，尚未经甲方人工复核",
        "evidence_requirement": "每条链接至少两类证据",
        "anchors_with_fewer_than_top_n": {
            str(anchor): int(count) for anchor, count in combined.groupby("anchor_id").size().items()
            if count < int(cfg["graph"]["top_related_accounts"])
        },
        "top_n_shortfall_policy": "若三跳实际可达账户不足20，则输出全部可达账户，禁止用无路径账户补足。",
    }
    write_json(summary, "outputs/link_analysis/link_acceptance.json")
    logging.info("链路分析完成：%s 个案例，内部解释通过率 %.1f%%", len(cases), summary["internal_explanation_pass_rate"] * 100)


if __name__ == "__main__":
    main()
