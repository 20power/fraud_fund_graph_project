from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from fraud_graph.dashboard import (
    build_directed_network_graph, build_filtered_network_figure, build_local_network_figure, label_name,
)


ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="涉诈账户资金图谱研判", page_icon=":material/hub:", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(f"<style>{(ROOT / 'assets/fintech_theme.css').read_text('utf-8')}</style>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_data() -> dict:
    paths = {
        "features": ROOT / "data/processed/account_features.parquet",
        "transactions": ROOT / "data/processed/transactions_cutoff.parquet",
        "directed_edges": ROOT / "data/processed/directed_relationship_edges.parquet",
        "ranking": ROOT / "outputs/risk_accounts/risk_account_ranking.parquet",
        "related": ROOT / "outputs/link_analysis/top_related_accounts.parquet",
        "evaluation": ROOT / "outputs/model_results/model_evaluation.json",
        "quality": ROOT / "outputs/data_quality/data_quality_report.json",
        "graph_stats": ROOT / "outputs/graph_statistics/graph_statistics.json",
        "link_acceptance": ROOT / "outputs/link_analysis/link_acceptance.json",
        "cases": ROOT / "outputs/case_studies/enhanced_cases.json",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("请先运行完整流水线。缺少：" + ", ".join(missing))
    return {
        "features": pd.read_parquet(paths["features"]),
        "transactions": pd.read_parquet(paths["transactions"]),
        "directed_edges": pd.read_parquet(paths["directed_edges"]),
        "ranking": pd.read_parquet(paths["ranking"]),
        "related": pd.read_parquet(paths["related"]),
        **{key: json.loads(paths[key].read_text("utf-8")) for key in
           ["evaluation", "quality", "graph_stats", "link_acceptance", "cases"]},
    }


@st.cache_resource(show_spinner=False)
def load_full_directed_graph(data_version: int):
    del data_version  # 结果文件变化时使资源缓存失效。
    edges = pd.read_parquet(ROOT / "data/processed/directed_relationship_edges.parquet")
    return build_directed_network_graph(edges)


@st.cache_data(show_spinner=False, max_entries=64)
def query_dynamic_graph(
    anchor: str,
    direction: str,
    hops: int,
    start_date_iso: str,
    end_date_iso: str,
    min_amount: float,
    max_amount: float,
    risk_levels: tuple[str, ...],
    data_version: int,
    _progress=None,
) -> dict:
    dataset = load_data()
    transactions = dataset["transactions"]
    ranking = dataset["ranking"]
    start_ts = pd.Timestamp(start_date_iso)
    end_exclusive = pd.Timestamp(end_date_iso) + pd.Timedelta(days=1)
    data_start = transactions["transaction_time"].min().normalize()
    data_end = transactions["transaction_time"].max().normalize()
    data_max_amount = float(transactions["amount_abs"].max())
    full_range = (
        start_ts <= data_start and end_exclusive > data_end
        and min_amount <= 0 and max_amount >= data_max_amount
    )

    if _progress:
        _progress(12, "正在筛选交易数据")
    if full_range:
        filtered_count = int(dataset["directed_edges"]["transaction_count"].sum())
        figure, paths, counts = build_filtered_network_figure(
            anchor, None, ranking, direction=direction, max_hops=hops,
            allowed_risk_levels=list(risk_levels), node_limit=80,
            prebuilt_graph=load_full_directed_graph(data_version), progress=_progress,
        )
        source_mode = "预聚合关系图"
    else:
        mask = (
            transactions["transaction_time"].ge(start_ts)
            & transactions["transaction_time"].lt(end_exclusive)
            & transactions["amount_abs"].between(min_amount, max_amount)
            & transactions["payer_id"].ne(transactions["payee_id"])
        )
        filtered = transactions.loc[mask]
        filtered_count = int(len(filtered))
        figure, paths, counts = build_filtered_network_figure(
            anchor, filtered, ranking, direction=direction, max_hops=hops,
            allowed_risk_levels=list(risk_levels), node_limit=80, progress=_progress,
        )
        source_mode = "动态交易聚合"
    return {
        "figure": figure, "paths": paths, "counts": counts,
        "filtered_count": filtered_count, "source_mode": source_mode,
    }


def page_header(title: str, subtitle: str) -> None:
    st.markdown(f"""
    <div class="fin-header"><div><div class="fin-kicker">FUND GRAPH INTELLIGENCE</div>
    <div class="fin-title">{title}</div></div><div class="fin-meta">
    数据观察截止：2025-12-31 23:59:59<br><span class="status-dot"></span>本地原型环境 · 数据不外发</div></div>
    <div class="section-note">{subtitle}</div>""", unsafe_allow_html=True)


def account_report_bytes(account_id: str, cases: list[dict]) -> tuple[bytes | None, str | None]:
    match = next((case for case in cases if str(case["anchor_id"]) == str(account_id)), None)
    if not match:
        return None, None
    path = ROOT / "outputs/investigation_reports" / f"{match['case_id']}_研判报告.html"
    return (path.read_bytes(), path.name) if path.exists() else (None, None)


try:
    data = load_data()
except Exception as exc:
    st.error(str(exc))
    st.stop()

features = data["features"]
ranking = data["ranking"]
transactions = data["transactions"]
detail = features.drop(columns=[c for c in features.columns if c != "account_id" and c in ranking.columns], errors="ignore")
accounts = ranking.merge(detail, on="account_id", how="left")
accounts["标签"] = accounts.account_label.map(label_name)
accounts["risk_level"] = accounts.risk_level.astype(str)
selected_test = data["evaluation"]["selected_primary_test"]

with st.sidebar:
    st.markdown('<div class="sidebar-brand">涉诈账户资金图谱研判</div><div class="sidebar-sub">FINANCIAL CRIME INTELLIGENCE</div>', unsafe_allow_html=True)
    page = st.radio("研判导航", ["风险总览", "风险账户", "资金图谱", "可疑路径", "典型案例", "模型与口径"],
                    label_visibility="collapsed")
    st.markdown(f"""<div class="sidebar-status"><b>系统状态</b><br>版本 v0.2.0<br>
    有效账户 {len(accounts):,}<br>交易记录 {len(transactions):,}<br>
    模型策略 {data['evaluation']['selected_primary_strategy']}<br><br>
    <span class="status-dot"></span>结果文件已就绪</div>""", unsafe_allow_html=True)
    st.caption("模型高风险不等于已确认涉诈。")

if page == "风险总览":
    page_header("风险指挥中心", "先看全局风险状态，再下钻账户、资金关系和可疑路径。")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("有效账户数", f"{len(accounts):,}", "统一统计口径")
    k2.metric("交易笔数", f"{len(transactions):,}", "数据期内累计")
    k3.metric("主模型 ROC-AUC", f"{selected_test['roc_auc']:.4f}", "账户互斥测试集")
    k4.metric("Top 5% 召回", f"{selected_test['top_5_percent']['recall']:.1%}",
              f"命中 {selected_test['top_5_percent']['hits']}/9")
    st.markdown('<div class="section-label">重点资金关系</div><div class="section-note">风险分负责排序，资金图谱负责组织可复核证据。</div>', unsafe_allow_html=True)
    anchors = sorted(data["related"].anchor_id.astype(str).unique(),
                     key=lambda x: int(accounts.set_index("account_id").loc[x, "risk_rank"]))
    default_anchor = anchors[0]
    graph_col, evidence_col = st.columns([3.5, 1.25])
    with graph_col:
        with st.container(border=True):
            anchor = st.selectbox("研判锚点", anchors, index=0, key="overview_anchor")
            related = data["related"].loc[data["related"].anchor_id.astype(str).eq(anchor)]
            st.plotly_chart(build_local_network_figure(anchor, related, ranking), width="stretch", key="overview_graph")
    with evidence_col:
        row = accounts.set_index("account_id").loc[anchor]
        with st.container(border=True):
            badge = "risk-badge" if str(row.risk_level) in ["高", "较高"] else "gold-badge"
            st.markdown(f'<span class="{badge}">{row.risk_level}风险</span>', unsafe_allow_html=True)
            st.metric("账户风险分", f"{row.risk_score:.4f}")
            st.metric("全量风险排名", f"{int(row.risk_rank):,} / {len(accounts):,}")
            st.markdown(f"""<div class="evidence-list"><div class="evidence-item">已知标签　{label_name(row.account_label)}</div>
            <div class="evidence-item">客户类型　{row.customer_type}</div><div class="evidence-item">图度数　{int(row.graph_degree)}</div>
            <div class="evidence-item">主要因素　{row.top_risk_factors}</div></div>""", unsafe_allow_html=True)
            report_bytes, report_name = account_report_bytes(anchor, data["cases"])
            if report_bytes:
                st.download_button("下载研判报告", report_bytes, report_name, "text/html", icon=":material/download:", width="stretch")
            else:
                st.caption("该账户暂未生成独立案例报告。")
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-label">高风险账户 Top 10</div>', unsafe_allow_html=True)
        view = accounts.sort_values("risk_rank")[["risk_rank", "account_id", "risk_score", "risk_level", "标签", "top_risk_factors"]].head(10)
        st.dataframe(view, width="stretch", hide_index=True, height=385)
    with right:
        st.markdown('<div class="section-label">近期重点路径</div>', unsafe_allow_html=True)
        paths = data["related"].sort_values(["path_last_transaction", "relationship_score"], ascending=False)[[
            "anchor_id", "related_account_id", "hops", "relationship_score", "path_amount_abs_sum", "path_last_transaction"
        ]].head(10)
        st.dataframe(paths, width="stretch", hide_index=True, height=385)

elif page == "风险账户":
    page_header("风险账户工作台", "按风险等级、标签和图关系覆盖情况筛选账户，并导出待核查清单。")
    f1, f2, f3, f4 = st.columns([1.6, 1, 1, 1])
    with f1:
        search = st.text_input("账户搜索", placeholder="输入脱敏账户ID")
    with f2:
        levels = st.multiselect("风险等级", ["高", "较高", "中", "低"], default=["高", "较高", "中", "低"])
    with f3:
        labels = st.multiselect("已知标签", ["其它", "嫌疑人", "受害人"], default=["其它", "嫌疑人", "受害人"])
    with f4:
        coverage = st.selectbox("图关系覆盖", ["全部", "仅有关系", "仅无关系"])
    filtered = accounts.loc[accounts.risk_level.isin(levels) & accounts["标签"].isin(labels)].copy()
    if search:
        filtered = filtered.loc[filtered.account_id.astype(str).str.contains(search.strip(), regex=False)]
    if coverage == "仅有关系":
        filtered = filtered.loc[filtered.has_relationship_edge.eq(1)]
    elif coverage == "仅无关系":
        filtered = filtered.loc[filtered.has_relationship_edge.eq(0)]
    c1, c2, c3 = st.columns(3)
    c1.metric("筛选账户", f"{len(filtered):,}")
    c2.metric("有图关系账户", f"{int(filtered.has_relationship_edge.sum()):,}")
    c3.metric("已知嫌疑人", f"{int((filtered.account_label == 1).sum()):,}")
    columns = ["risk_rank", "account_id", "risk_score", "risk_level", "标签", "top_risk_factors",
               "has_transaction", "has_relationship_edge", "opening_months", "customer_type", "region_code"]
    st.dataframe(filtered.sort_values("risk_rank")[columns].head(2000), width="stretch", hide_index=True, height=600)
    st.download_button("导出当前账户清单", filtered[columns].to_csv(index=False).encode("utf-8-sig"),
                       "风险账户筛选结果.csv", "text/csv", icon=":material/download:")

elif page == "资金图谱":
    page_header("动态资金图谱", "在指定时间、金额、方向和跳数范围内查询资金关系；节点颜色表示模型风险。")
    graph_accounts = accounts.loc[accounts.has_relationship_edge.eq(1)].sort_values("risk_rank")
    with st.form("dynamic_graph_filters", border=False):
        top_a, top_b, top_c = st.columns([2, 1, 1])
        with top_a:
            anchor = st.selectbox("锚点账户", graph_accounts.account_id.astype(str).tolist(), key="dynamic_anchor")
        with top_b:
            direction = st.selectbox("资金方向", ["双向", "转出", "转入"])
        with top_c:
            hops = st.select_slider("最大跳数", options=[1, 2, 3], value=3)
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            start_date = st.date_input("开始日期", value=date(2025, 7, 1), min_value=date(2025, 7, 1), max_value=date(2025, 12, 31))
        with f2:
            end_date = st.date_input("结束日期", value=date(2025, 12, 31), min_value=date(2025, 7, 1), max_value=date(2025, 12, 31))
        with f3:
            min_amount = st.number_input("单笔最低绝对金额", min_value=0.0, value=0.0, step=1000.0)
        with f4:
            max_amount = st.number_input("单笔最高绝对金额", min_value=0.0, value=float(transactions.amount_abs.max()), step=100000.0)
        risk_levels = st.multiselect("展示风险等级", ["高", "较高", "中", "低"], default=["高", "较高", "中", "低"])
        submitted = st.form_submit_button("生成 / 刷新资金图谱", type="primary", icon=":material/hub:", width="stretch")
    if start_date > end_date or min_amount > max_amount:
        st.error("时间或金额范围无效。")
    else:
        graph_data_version = max(
            (ROOT / "data/processed/transactions_cutoff.parquet").stat().st_mtime_ns,
            (ROOT / "data/processed/directed_relationship_edges.parquet").stat().st_mtime_ns,
            (ROOT / "outputs/risk_accounts/risk_account_ranking.parquet").stat().st_mtime_ns,
        )
        state_key = "dynamic_graph_result"
        should_generate = (
            submitted or state_key not in st.session_state
            or st.session_state.get("dynamic_graph_data_version") != graph_data_version
        )
        if should_generate:
            status = st.status("正在生成资金图谱", expanded=True)
            progress_bar = st.progress(3, text="正在准备查询条件")

            def update_graph_progress(value: int, message: str) -> None:
                progress_bar.progress(value, text=message)
                status.update(label=f"正在生成资金图谱 · {message}", state="running", expanded=True)

            try:
                result = query_dynamic_graph(
                    str(anchor), direction, int(hops), start_date.isoformat(), end_date.isoformat(),
                    float(min_amount), float(max_amount), tuple(risk_levels), graph_data_version,
                    _progress=update_graph_progress,
                )
            except Exception as exc:
                status.update(label="资金图谱生成失败", state="error", expanded=True)
                st.error(str(exc))
                st.stop()
            st.session_state[state_key] = result
            st.session_state["dynamic_graph_data_version"] = graph_data_version
            progress_bar.progress(100, text="资金图谱生成完成")
            status.update(
                label=f"资金图谱已生成 · {result['source_mode']}", state="complete", expanded=False
            )
            progress_bar.empty()
        result = st.session_state[state_key]
        figure = result["figure"]
        dynamic_paths = result["paths"]
        graph_counts = result["counts"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("筛选交易", f"{result['filtered_count']:,}")
        m2.metric("展示节点", f"{graph_counts['nodes']:,}")
        m3.metric("展示有向边", f"{graph_counts['edges']:,}")
        m4.metric("子图交易笔数", f"{graph_counts['transactions']:,}")
        st.caption(f"查询方式：{result['source_mode']}。调整筛选条件后，请点击“生成 / 刷新资金图谱”。")
        with st.container(border=True):
            st.plotly_chart(figure, width="stretch", key="dynamic_graph")
        if graph_counts["nodes"] >= 80:
            st.info("为保证交互性能，当前子图最多展示80个节点；路径表仍按关系分排序。")
        st.markdown('<div class="section-label">动态路径排序</div>', unsafe_allow_html=True)
        if dynamic_paths.empty:
            st.info("筛选条件下没有可展示路径。")
        else:
            st.dataframe(dynamic_paths.head(100), width="stretch", hide_index=True)
            st.download_button("导出当前路径", dynamic_paths.to_csv(index=False).encode("utf-8-sig"),
                               "动态路径结果.csv", "text/csv", icon=":material/download:")

elif page == "可疑路径":
    page_header("可疑路径研判", "查看预计算Top关联账户与三跳路径，按关系分、金额和跳数进一步筛选。")
    anchors = sorted(data["related"].anchor_id.astype(str).unique())
    p1, p2, p3 = st.columns(3)
    with p1:
        anchor = st.selectbox("锚点账户", anchors, key="path_anchor")
    with p2:
        max_hops = st.select_slider("最大跳数", [1, 2, 3], value=3, key="path_hops")
    with p3:
        min_relationship = st.slider("最低关系分", 0.0, 1.0, 0.0, .01)
    paths = data["related"].loc[
        data["related"].anchor_id.astype(str).eq(anchor) & data["related"].hops.le(max_hops)
        & data["related"].relationship_score.ge(min_relationship)
    ].sort_values("relationship_score", ascending=False)
    if paths.empty:
        st.info("当前条件下没有可疑路径。")
    else:
        st.dataframe(paths[["related_account_id", "hops", "relationship_score", "related_risk_score",
                            "path_transaction_count", "path_amount_abs_sum", "path_last_transaction", "explanation"]],
                     width="stretch", hide_index=True, height=610)
        st.download_button("导出路径研判表", paths.to_csv(index=False).encode("utf-8-sig"),
                           "可疑路径研判表.csv", "text/csv", icon=":material/download:")

elif page == "典型案例":
    page_header("典型案例中心", "5个案例均包含账户画像、关键交易、关联账户、可疑路径、资金子图和人工核查建议。")
    for case in data["cases"]:
        case_dir = ROOT / "outputs/case_studies" / case["case_id"]
        with st.container(border=True):
            title_col, action_col = st.columns([4, 1])
            with title_col:
                st.subheader(f"{case['case_id']} · 账户 {case['anchor_id']}")
                badge = "risk-badge" if case["risk_category"] == "已知嫌疑人" else "gold-badge"
                st.markdown(f'<span class="{badge}">{case["risk_category"]}</span>', unsafe_allow_html=True)
                st.caption(f"风险分 {case['risk_score']:.4f} · 全量排名 {case['risk_rank']} · {case['top_risk_factors']}")
            with action_col:
                html_path = ROOT / "outputs/investigation_reports" / f"{case['case_id']}_研判报告.html"
                st.download_button("下载报告", html_path.read_bytes(), html_path.name, "text/html",
                                   icon=":material/download:", width="stretch")
            left, right = st.columns([1.45, 1])
            with left:
                st.image(str(case_dir / "network.png"), width="stretch")
            with right:
                st.markdown("**命中模式**")
                for pattern in case["patterns"]:
                    st.markdown(f"- {pattern}")
                st.markdown("**研判结论**")
                st.write(case["conclusion"])

else:
    page_header("模型证据与口径", "主模型指标、稳健性实验和限制均独立保存，辅助模型不替代主任务验收。")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("测试 ROC-AUC", f"{selected_test['roc_auc']:.4f}")
    m2.metric("测试 PR-AUC", f"{selected_test['pr_auc']:.4f}")
    m3.metric("测试嫌疑人数", f"{selected_test['positive_count']}")
    m4.metric("未来交易泄露", "0")
    stability_path = ROOT / "outputs/figures/model_stability.png"
    if stability_path.exists():
        st.image(str(stability_path), caption="验证集稳健性实验；测试集不参与调参。", width="stretch")
    st.markdown("""
    <div class="callout"><b>必须保留的解释边界</b><br>
    嫌疑人仅59个，测试集仅9个；只有3个嫌疑人进入资金图；标签缺少确认时间。
    因此模型用于缩小人工核查范围，不代表账户已确认涉诈，也不能替代人工审核。</div>
    """, unsafe_allow_html=True)
    st.markdown("### 核心口径")
    st.markdown("""
    - 主任务：嫌疑人账户与非嫌疑人账户分类。
    - 辅助任务：嫌疑人＋受害人与其它账户分类，仅用于筛查和特征验证。
    - Top 5%召回：测试集风险分最高5%的账户覆盖的嫌疑人比例。
    - 自环保留为行为特征但不进入关系图；负金额保留原值并生成绝对金额和负值标记。
    - 风险邻居特征只使用训练集嫌疑人标签。
    """)
