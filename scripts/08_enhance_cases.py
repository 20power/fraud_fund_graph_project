from __future__ import annotations

import json
import logging
import sys
import zipfile
import base64
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from jinja2 import Template

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud_graph.io import write_json, write_table  # noqa: E402


HTML_TEMPLATE = Template("""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{ case.case_id }} 账户研判报告</title>
<style>
body{margin:0;background:#07111f;color:#dce9f7;font:15px/1.7 "Microsoft YaHei",Arial,sans-serif}
.page{max-width:1120px;margin:32px auto;padding:0 28px 48px}.header{border-bottom:1px solid #1c3853;padding-bottom:22px}
.eyebrow{color:#50a7ff;letter-spacing:.12em;font-size:12px}.title{font-size:30px;margin:8px 0}.muted{color:#8098ad}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:22px 0}.metric{background:#0b1b2c;border:1px solid #18364f;padding:18px;border-radius:8px}
.metric b{display:block;color:#fff;font-size:24px}.metric span{color:#7f9bb2;font-size:12px}.risk{color:#f3b84b!important}
section{background:#0a1929;border:1px solid #17334b;border-radius:8px;padding:20px 24px;margin:14px 0}h2{font-size:18px;color:#fff;margin:0 0 12px}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;border-bottom:1px solid #17334b;padding:9px 7px}th{color:#7fb9e8}
.tag{display:inline-block;border:1px solid #2a5c84;background:#0d2740;padding:3px 9px;margin:3px;border-radius:4px}.warning{border-left:3px solid #f3b84b;padding-left:14px}
img{width:100%;background:#07111f;border-radius:6px}.footer{color:#718ba2;font-size:12px;margin-top:24px}
@media(max-width:800px){.metrics{grid-template-columns:1fr 1fr}.page{padding:0 14px}}
</style></head><body><main class="page">
<header class="header"><div class="eyebrow">FUND GRAPH INVESTIGATION · LOCAL PROTOTYPE</div><h1 class="title">{{ case.case_id }} 账户研判报告</h1>
<div class="muted">观察截止：2025-12-31 23:59:59　｜　确定性本地模板生成　｜　仅用于辅助研判</div></header>
<div class="metrics"><div class="metric"><span>账户编号</span><b>{{ case.anchor_id }}</b></div><div class="metric"><span>风险分</span><b class="risk">{{ '%.4f'|format(case.risk_score) }}</b></div><div class="metric"><span>全量排名</span><b>{{ case.risk_rank }}</b></div><div class="metric"><span>风险类别</span><b>{{ case.risk_category }}</b></div></div>
<section><h2>1. 账户基本信息</h2>{{ profile_html }}</section>
<section><h2>2. 风险状态与模型解释</h2><p class="warning">{{ case.status_note }}</p><p>主要风险因素：{{ case.top_risk_factors }}</p></section>
<section><h2>3. 关键交易</h2>{{ transactions_html }}</section>
<section><h2>4. 主要关联账户</h2>{{ related_html }}</section>
<section><h2>5. 一至三跳可疑路径</h2>{{ paths_html }}</section>
<section><h2>6. 命中资金模式</h2>{% for p in case.patterns %}<span class="tag">{{ p }}</span>{% endfor %}</section>
<section><h2>7. 资金关系子图</h2><img src="{{ network_src }}" alt="资金关系子图"></section>
<section><h2>8. 研判结论</h2><p>{{ case.conclusion }}</p></section>
<section><h2>9. 建议核查事项</h2><ol>{% for item in case.review_actions %}<li>{{ item }}</li>{% endfor %}</ol></section>
<section><h2>10. 证据边界</h2><p>模型输出为预测性风险线索，不代表已确认涉诈。图谱仅覆盖现有交易数据；缺少标签确认时间、交易渠道和交易类型，最终结论必须结合账户身份、业务背景和人工核查。</p></section>
<div class="footer">报告版本 v0.2.0 · 数据不发送至外部服务 · {{ case.case_id }}</div>
</main></body></html>""")


def table_html(frame: pd.DataFrame, columns: list[str], names: list[str], limit: int = 15) -> str:
    if frame.empty:
        return '<p class="muted">无可展示记录。</p>'
    view = frame[columns].head(limit).copy()
    view.columns = names
    for column in view.select_dtypes(include=["float"]).columns:
        view[column] = view[column].map(lambda x: f"{x:,.2f}")
    return view.to_html(index=False, escape=True, border=0)


def draw_network(anchor: str, related: pd.DataFrame, transactions: pd.DataFrame, output: Path) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    nodes = {anchor}
    for raw_path in related.head(12).path:
        nodes.update(json.loads(raw_path))
    tx = transactions.loc[transactions.payer_id.isin(nodes) & transactions.payee_id.isin(nodes)].copy()
    directed = tx.groupby(["payer_id", "payee_id"]).agg(
        amount_abs_sum=("amount", lambda x: float(x.abs().sum())), transaction_count=("amount", "size")
    ).reset_index()
    graph = nx.DiGraph()
    for row in directed.itertuples():
        graph.add_edge(str(row.payer_id), str(row.payee_id), amount=row.amount_abs_sum, count=row.transaction_count)
    graph.add_nodes_from(nodes)
    pos = nx.spring_layout(graph.to_undirected(), seed=42, k=1.15 / max(len(nodes) ** .5, 1))
    risk_map = {str(row.related_account_id): float(row.related_risk_score) for row in related.itertuples()}
    colors = ["#f3b84b" if node == anchor else plt.cm.Blues(.35 + .6 * risk_map.get(node, 0)) for node in graph.nodes]
    sizes = [1500 if node == anchor else 500 + 700 * risk_map.get(node, 0) for node in graph.nodes]
    fig, axis = plt.subplots(figsize=(12, 7), facecolor="#07111f")
    axis.set_facecolor("#07111f")
    nx.draw_networkx_edges(graph, pos, ax=axis, edge_color="#4d718c", width=1.1, arrows=True,
                           arrowsize=13, connectionstyle="arc3,rad=0.08", alpha=.8)
    nx.draw_networkx_nodes(graph, pos, ax=axis, node_color=colors, node_size=sizes,
                           edgecolors="#d6e9f7", linewidths=.8)
    nx.draw_networkx_labels(graph, pos, ax=axis, font_color="#dce9f7", font_size=8,
                           font_family="Microsoft YaHei", labels={n: str(n) for n in graph.nodes})
    axis.set_title(f"账户 {anchor} 资金关系子图（最多展示12条重点路径）", color="#dce9f7", fontsize=14, pad=18)
    axis.axis("off")
    fig.savefig(output, dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    features = pd.read_parquet(ROOT / "data/processed/account_features.parquet")
    transactions = pd.read_parquet(ROOT / "data/processed/transactions_cutoff.parquet")
    ranking = pd.read_parquet(ROOT / "outputs/risk_accounts/risk_account_ranking.parquet")
    related_all = pd.read_parquet(ROOT / "outputs/link_analysis/top_related_accounts.parquet")
    cases = json.loads((ROOT / "outputs/case_studies/cases.json").read_text("utf-8"))
    profile = features.merge(ranking[["account_id", "risk_score", "risk_rank", "risk_level", "top_risk_factors"]], on="account_id")
    profile_map = profile.set_index("account_id")
    enhanced = []
    report_dir = ROOT / "outputs/investigation_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        case_id = case["case_id"]
        anchor = str(case["anchor_id"])
        case_dir = ROOT / "outputs/case_studies" / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        row = profile_map.loc[anchor]
        related = related_all.loc[related_all.anchor_id.astype(str).eq(anchor)].sort_values("relationship_score", ascending=False)
        incident = transactions.loc[(transactions.payer_id == anchor) | (transactions.payee_id == anchor)].copy()
        incident["direction"] = np.where(incident.payer_id == anchor, "转出", "转入")
        incident["counterparty_id"] = np.where(incident.payer_id == anchor, incident.payee_id, incident.payer_id)
        incident["amount_abs"] = incident.amount.abs()
        key_transactions = incident.sort_values(["amount_abs", "transaction_time"], ascending=[False, False]).head(30)
        paths = related[[
            "related_account_id", "hops", "path", "relationship_score", "path_transaction_count",
            "path_amount_abs_sum", "path_last_transaction", "explanation",
        ]].head(20)
        write_table(pd.DataFrame([{
            "account_id": anchor, "account_label": int(row.account_label), "opening_months": row.opening_months,
            "region_code": row.region_code, "customer_type": row.customer_type, "risk_score": row.risk_score,
            "risk_level": row.risk_level, "risk_rank": row.risk_rank, "top_risk_factors": row.top_risk_factors,
            "has_transaction": row.has_transaction, "graph_degree": row.graph_degree,
            "graph_community_id": row.graph_community_id,
        }]), f"outputs/case_studies/{case_id}/account_profile.csv")
        write_table(key_transactions[[
            "transaction_time", "direction", "counterparty_id", "amount", "amount_abs"
        ]], f"outputs/case_studies/{case_id}/key_transactions.csv")
        write_table(related, f"outputs/case_studies/{case_id}/related_accounts.csv")
        write_table(paths, f"outputs/case_studies/{case_id}/suspicious_paths.csv")
        draw_network(anchor, related, transactions, case_dir / "network.png")
        risk_category = "已知嫌疑人" if int(row.account_label) == 1 else "模型高风险疑似"
        status_note = (
            "该账户在甲方标签中为嫌疑人；本报告用于组织模型和图谱证据，不替代案件事实核查。"
            if int(row.account_label) == 1 else
            "该账户由模型识别为高风险疑似，尚无已知嫌疑标签，不能表述为已确认涉诈。"
        )
        conclusion = (
            f"账户风险排名为全量第{int(row.risk_rank)}位，命中“{'、'.join(case['patterns'])}”。"
            f"共展示{len(related)}个三跳内可达关联账户；建议结合资金来源、交易背景和关联主体身份优先人工核查。"
        )
        enhanced_case = {
            **case, "risk_score": float(row.risk_score), "risk_rank": int(row.risk_rank),
            "risk_level": str(row.risk_level), "risk_category": risk_category,
            "top_risk_factors": str(row.top_risk_factors), "status_note": status_note, "conclusion": conclusion,
            "review_actions": [
                "核验账户开户资料、实际控制人与受益所有人。",
                "核对重点交易的业务背景、合同或资金来源证明。",
                "核查Top关联账户与锚点账户是否存在人员、设备或经营关系。",
                "复核一至三跳路径中的中转账户及交易时间顺序。",
                "记录人工结论并回填盲审表，供模型和解释规则复盘。",
            ],
        }
        profile_frame = pd.DataFrame([
            ["开户时长（月）", f"{row.opening_months:.1f}"], ["客户类型", row.customer_type],
            ["地区编码", row.region_code], ["是否有交易", "是" if row.has_transaction else "否"],
            ["图度数", int(row.graph_degree)], ["图社区", int(row.graph_community_id)],
        ], columns=["项目", "值"])
        network_src = "data:image/png;base64," + base64.b64encode(
            (case_dir / "network.png").read_bytes()
        ).decode("ascii")
        html = HTML_TEMPLATE.render(
            case=enhanced_case,
            network_src=network_src,
            profile_html=table_html(profile_frame, ["项目", "值"], ["项目", "值"]),
            transactions_html=table_html(key_transactions, ["transaction_time", "direction", "counterparty_id", "amount"],
                                         ["交易时间", "方向", "交易对手", "金额"], 15),
            related_html=table_html(related, ["related_account_id", "hops", "relationship_score", "related_risk_score", "path_amount_abs_sum"],
                                    ["关联账户", "跳数", "关联分", "风险分", "路径金额"], 15),
            paths_html=table_html(paths, ["hops", "path", "relationship_score", "path_transaction_count", "path_amount_abs_sum"],
                                  ["跳数", "路径", "路径分", "交易笔数", "累计金额"], 12),
        )
        (case_dir / "report.html").write_text(html, encoding="utf-8")
        (report_dir / f"{case_id}_研判报告.html").write_text(html, encoding="utf-8")
        markdown = [
            f"# {case_id} 账户研判报告", "", f"- 账户：{anchor}", f"- 类别：{risk_category}",
            f"- 风险分/等级/排名：{row.risk_score:.4f} / {row.risk_level} / {int(row.risk_rank)}",
            f"- 风险因素：{row.top_risk_factors}", "", "## 研判结论", "", conclusion,
            "", "## 命中模式", "", *[f"- {p}" for p in case["patterns"]],
            "", "## 人工核查建议", "", *[f"- {item}" for item in enhanced_case["review_actions"]],
            "", "![资金关系子图](network.png)", "", "模型输出仅作辅助研判，不代表执法结论。",
        ]
        (case_dir / "report.md").write_text("\n".join(markdown), encoding="utf-8")
        enhanced.append(enhanced_case)
        logging.info("案例增强完成 %s", case_id)
    write_json(enhanced, "outputs/case_studies/enhanced_cases.json")
    index = pd.DataFrame([{k: case[k] for k in [
        "case_id", "anchor_id", "risk_score", "risk_rank", "risk_level", "risk_category", "top_risk_factors"
    ]} for case in enhanced])
    write_table(index, "outputs/investigation_reports/case_report_index.csv")
    with zipfile.ZipFile(report_dir / "五个典型案例研判材料.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for case in enhanced:
            case_dir = ROOT / "outputs/case_studies" / case["case_id"]
            for path in case_dir.iterdir():
                archive.write(path, arcname=f"{case['case_id']}/{path.name}")
    logging.info("5个案例材料全部生成")


if __name__ == "__main__":
    main()
