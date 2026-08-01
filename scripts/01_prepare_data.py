from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud_graph.config import ensure_output_dirs, load_configs  # noqa: E402
from fraud_graph.io import read_sources, source_manifest, write_json, write_table  # noqa: E402
from fraud_graph.splits import make_account_split, validate_split  # noqa: E402
from fraud_graph.validation import validate_sources  # noqa: E402


def main() -> None:
    ensure_output_dirs()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_configs()
    logging.info("读取只读原始工作簿")
    accounts, labels, edges = read_sources(cfg["data"])
    report = validate_sources(accounts, labels, edges)
    report["source_manifest"] = source_manifest(cfg["data"])
    split = make_account_split(accounts, cfg["data"])
    report["split_validation"] = validate_split(split)
    if report["status"] != "pass":
        write_json(report, "outputs/data_quality/data_quality_report.json")
        raise RuntimeError("数据质量硬校验失败，详见 data_quality_report.json")

    write_table(accounts, "data/interim/accounts.parquet")
    write_table(labels, "data/interim/labels.parquet")
    write_table(edges, "data/interim/edges.parquet")
    write_table(split, "data/processed/account_split.parquet")
    write_table(split, "data/processed/account_split.csv")
    write_json(report, "outputs/data_quality/data_quality_report.json")
    summary = [
        "# 数据质量报告",
        "",
        f"- 校验状态：{report['status']}",
        f"- 实际有效账户：{report['account_count']:,}（说明文件为 11,088，差异 {report['known_note_difference']}）",
        f"- 交易记录：{report['edge_count']:,}",
        f"- 有交易覆盖账户：{report['edge_covered_account_count']:,}",
        f"- 无交易账户：{report['no_edge_account_count']:,}",
        f"- 标签分布：{json.dumps(report['label_distribution'], ensure_ascii=False)}",
        f"- 自环：{report['self_loop_count']:,}；负金额：{report['negative_amount_count']:,}；精确重复：{report['exact_duplicate_edge_count']:,}",
        f"- 时间范围：{report['transaction_time_min']} 至 {report['transaction_time_max']}",
        "",
        "## 已固化口径",
        "",
        *[f"- {key}：{value}" for key, value in report["policies"].items()],
        "",
        "原始文件保持只读，报告 JSON 中保存其路径、大小与 SHA-256，以便追溯。",
    ]
    (ROOT / "outputs/data_quality/data_quality_report.md").write_text("\n".join(summary), encoding="utf-8")
    logging.info("数据准备完成")


if __name__ == "__main__":
    main()
