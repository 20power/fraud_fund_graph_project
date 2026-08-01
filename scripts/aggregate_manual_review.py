from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def format_rate(value: float | None) -> str:
    return "—" if value is None else f"{value:.2%}"


def summarize(frame: pd.DataFrame) -> dict:
    rational = frame[frame["解释合理性"].isin(["合理", "基本合理", "不合理"])]
    actionable = frame[frame["是否值得进一步核查"].isin(["是", "否"])]
    sufficient = frame[frame["证据充分性"].isin(["充分", "基本充分", "不足"])]
    return {
        "submitted_rows": int(frame["解释合理性"].fillna("").ne("").sum()),
        "valid_rationality_reviews": int(len(rational)),
        "rationality_pass_count": int(rational["解释合理性"].isin(["合理", "基本合理"]).sum()),
        "rationality_pass_rate": ratio(int(rational["解释合理性"].isin(["合理", "基本合理"]).sum()), len(rational)),
        "valid_action_reviews": int(len(actionable)),
        "further_review_rate": ratio(int(actionable["是否值得进一步核查"].eq("是").sum()), len(actionable)),
        "valid_evidence_reviews": int(len(sufficient)),
        "evidence_pass_rate": ratio(int(sufficient["证据充分性"].isin(["充分", "基本充分"]).sum()), len(sufficient)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总甲方回填的关联链路盲审结果")
    parser.add_argument("workbook", nargs="?", default="outputs/manual_review/链路盲审表.xlsx")
    args = parser.parse_args()
    workbook = Path(args.workbook)
    if not workbook.is_absolute():
        workbook = ROOT / workbook
    frame = pd.read_excel(workbook, sheet_name="Blind_Review", header=None, skiprows=5)
    frame = frame.iloc[:, :19]
    frame.columns = [
        "样本编号", "审核席位", "匿名锚点账户", "匿名关联账户", "路径跳数", "匿名资金路径",
        "路径交易笔数", "路径累计绝对金额", "路径最近交易时间", "是否同一图社区", "双向关系数量",
        "关系证据摘要", "解释合理性", "是否值得进一步核查", "证据充分性", "不合理/不足原因",
        "补充意见", "审核人", "审核时间",
    ]
    frame = frame.loc[frame["样本编号"].notna()].copy()
    output_dir = ROOT / "outputs/manual_review"
    overall = summarize(frame)
    by_reviewer = {
        slot: summarize(group) for slot, group in frame.groupby("审核席位", dropna=False)
    }
    try:
        source_workbook = workbook.relative_to(ROOT).as_posix()
    except ValueError:
        source_workbook = workbook.name
    summary = {"source_workbook": source_workbook, "overall": overall, "by_reviewer_slot": by_reviewer}
    (output_dir / "manual_review_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    frame.to_csv(output_dir / "manual_review_responses.csv", index=False, encoding="utf-8-sig")
    rate = overall["rationality_pass_rate"]
    status = "待回填" if rate is None else ("达到70%参考目标" if rate >= .70 else "未达到70%参考目标")
    text = [
        "# 甲方关联链路盲审结果", "", f"- 状态：{status}",
        f"- 有效合理性评价：{overall['valid_rationality_reviews']}",
        f"- 解释合理率：{format_rate(rate)}",
        f"- 值得进一步核查比例：{format_rate(overall['further_review_rate'])}",
        f"- 证据充分率：{format_rate(overall['evidence_pass_rate'])}",
        "", "说明：无法判断和空白项不进入对应比例的分母；本报告不以内部标签或模型分数代替甲方人工判断。",
    ]
    (ROOT / "reports/07_client_manual_review_report.md").write_text("\n".join(text), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
