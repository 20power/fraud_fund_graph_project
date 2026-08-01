import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = path.resolve(import.meta.dirname, "..");
const outputDir = path.join(root, "outputs", "manual_review");
const records = JSON.parse(await fs.readFile(path.join(outputDir, "review_records.json"), "utf8"));
const workbook = Workbook.create();
const instructions = workbook.worksheets.add("Instructions");
const review = workbook.worksheets.add("Blind_Review");
const dictionary = workbook.worksheets.add("Rating_Dictionary");
const summary = workbook.worksheets.add("Summary");
const lists = workbook.worksheets.add("Lists");

const navy = "#071B2E";
const blue = "#1468D4";
const cyan = "#6EC8FF";
const gold = "#D9A441";
const pale = "#EAF3FA";
const line = "#BDD0DF";
const white = "#FFFFFF";
const text = "#203447";

function unicodeFormula(value) {
  const parts = [];
  let ascii = "";
  const flushAscii = () => {
    if (ascii) {
      parts.push(`"${ascii.replaceAll('"', '""')}"`);
      ascii = "";
    }
  };
  for (const char of value) {
    const code = char.codePointAt(0);
    if (code >= 32 && code <= 126) ascii += char;
    else {
      flushAscii();
      parts.push(`UNICHAR(${code})`);
    }
  }
  flushAscii();
  return `=${parts.join("&") || '""'}`;
}

function encodeNonAsciiValues(sheet) {
  const range = sheet.getUsedRange();
  if (!range) return;
  const values = range.values;
  const formulas = range.formulas;
  values.forEach((row, rowIndex) => row.forEach((value, colIndex) => {
    const formula = formulas?.[rowIndex]?.[colIndex];
    if ((!formula || !String(formula).startsWith("=")) && typeof value === "string" && /[^\x00-\x7F]/.test(value)) {
      sheet.getCell(rowIndex, colIndex).formulas = [[unicodeFormula(value)]];
    }
  }));
}

function titleBand(sheet, range, title, subtitle) {
  sheet.getRange(range).merge();
  const cell = sheet.getRange(range.split(":")[0]);
  cell.values = [[`${title}\n${subtitle}`]];
  cell.format = {
    fill: navy, font: { color: white, bold: true, size: 18 }, wrapText: true,
    verticalAlignment: "center", horizontalAlignment: "left",
  };
  sheet.getRange(range).format.rowHeight = 58;
}

function sectionHeader(range) {
  range.format = {
    fill: blue, font: { color: white, bold: true },
    verticalAlignment: "center", wrapText: true,
    borders: { preset: "all", style: "thin", color: line },
  };
}

// 使用说明
instructions.showGridLines = false;
titleBand(instructions, "A1:H2", "甲方关联链路盲审表", "用于独立评价资金链路解释的合理性、核查价值与证据充分性");
instructions.getRange("A4:H4").merge();
instructions.getRange("A4").values = [["使用流程"]];
sectionHeader(instructions.getRange("A4:H4"));
const guide = [
  ["1", "独立审核", "审核人A、审核人B应分别完成对应席位，不讨论后再统一填写。"],
  ["2", "保持盲态", "评分页不展示真实账户、标签、模型风险分和内部关联评分。请勿打开 blind_review_key.csv。"],
  ["3", "逐条评价", "在黄色列中填写解释合理性、是否值得进一步核查、证据充分性及必要说明。"],
  ["4", "无法判断", "证据不足以判断时请选择“无法判断”，该项不会进入对应有效比例分母。"],
  ["5", "提交与汇总", "完成后保存本工作簿；汇总结果页会自动计算。项目方可运行汇总脚本生成正式报告。"],
];
instructions.getRange("A5:C9").values = guide;
instructions.getRange("A5:A9").format = { fill: pale, font: { bold: true, color: blue }, horizontalAlignment: "center" };
instructions.getRange("B5:B9").format.font = { bold: true, color: text };
instructions.getRange("A5:C9").format = { ...instructions.getRange("A5:C9").format, wrapText: true, borders: { preset: "all", style: "thin", color: line } };
instructions.getRange("A11:H11").merge();
instructions.getRange("A11").values = [["关键口径"]];
sectionHeader(instructions.getRange("A11:H11"));
instructions.getRange("A12:H17").merge(true);
instructions.getRange("A12:A17").values = [
  ["• 解释合理率 =（合理 + 基本合理）/ 有效合理性评价；参考目标为不低于70%。"],
  ["• 证据充分率 =（充分 + 基本充分）/ 有效证据评价。"],
  ["• “是否值得进一步核查”衡量研判价值，不代表账户已被认定涉诈。"],
  ["• 空白项和“无法判断”不进入对应比例分母，但需关注其数量。"],
  ["• 两位审核人的评价应保留原始差异，不应为了提高一致率而事后改写。"],
  ["• 本文件含匿名业务证据，仍应按项目资料管理要求保存和传递。"],
];
instructions.getRange("A12:H17").format = { wrapText: true, fill: "#F6F9FC", font: { color: text }, borders: { preset: "all", style: "thin", color: line } };
instructions.getRange("A:C").format.columnWidth = 18;
instructions.getRange("C:C").format.columnWidth = 62;
instructions.getRange("D:H").format.columnWidth = 12;
instructions.getRange("4:17").format.rowHeight = 30;

// 盲审评分
review.showGridLines = false;
titleBand(review, "A1:S2", "关联链路盲审评分", "黄色列为甲方填写区；一条样本设置两个独立审核席位");
review.getRange("A3:S3").merge();
review.getRange("A3").values = [["注意：模型输出仅作辅助研判，不代表账户已被认定涉诈。请仅依据本页匿名证据评分。"]];
review.getRange("A3").format = { fill: "#FFF4D6", font: { color: "#6C4D00", bold: true }, wrapText: true };
const headers = ["样本编号", "审核席位", "匿名锚点账户", "匿名关联账户", "路径跳数", "匿名资金路径", "路径交易笔数", "路径累计绝对金额", "路径最近交易时间", "是否同一图社区", "双向关系数量", "关系证据摘要", "解释合理性*", "是否值得进一步核查*", "证据充分性*", "不合理/不足原因", "补充意见", "审核人", "审核时间"];
review.getRange("A5:S5").values = [headers];
sectionHeader(review.getRange("A5:S5"));
const columns = ["样本编号", "审核席位", "匿名锚点账户", "匿名关联账户", "路径跳数", "匿名资金路径", "路径交易笔数", "路径累计绝对金额", "路径最近交易时间", "是否同一图社区", "双向关系数量", "关系证据摘要", "解释合理性", "是否值得进一步核查", "证据充分性", "不合理/不足原因", "补充意见", "审核人", "审核时间"];
const values = records.map((record) => columns.map((column) => record[column] ?? null));
const lastRow = values.length + 5;
review.getRange(`A6:S${lastRow}`).values = values;
review.getRange(`A6:L${lastRow}`).format = { fill: "#F7FAFC", font: { color: text }, borders: { preset: "all", style: "thin", color: "#D9E3EB" }, wrapText: true };
review.getRange(`M6:S${lastRow}`).format = { fill: "#FFF4D6", font: { color: text }, borders: { preset: "all", style: "thin", color: "#E4C979" }, wrapText: true };
review.getRange(`M6:M${lastRow}`).dataValidation = { rule: { type: "list", formula1: "Lists!$A$1:$A$4" } };
review.getRange(`N6:N${lastRow}`).dataValidation = { rule: { type: "list", formula1: "Lists!$B$1:$B$3" } };
review.getRange(`O6:O${lastRow}`).dataValidation = { rule: { type: "list", formula1: "Lists!$C$1:$C$4" } };
review.getRange(`P6:P${lastRow}`).dataValidation = { rule: { type: "list", formula1: "Lists!$D$1:$D$7" } };
review.getRange(`H6:H${lastRow}`).setNumberFormat("#,##0.00");
review.getRange(`I6:I${lastRow}`).setNumberFormat("yyyy-mm-dd hh:mm:ss");
review.getRange(`S6:S${lastRow}`).setNumberFormat("yyyy-mm-dd hh:mm");
review.freezePanes.freezeRows(5);
review.freezePanes.freezeColumns(4);
const widths = [15,12,17,17,10,40,12,20,20,14,12,48,15,20,15,24,28,14,20];
widths.forEach((width, index) => review.getRangeByIndexes(0, index, lastRow, 1).format.columnWidth = width);
review.getRange(`6:${lastRow}`).format.rowHeight = 38;
review.getRange(`A5:S${lastRow}`).conditionalFormats.add("expression", { formula: "=$B6=Lists!$E$2", format: { fill: "#EEF5FB" } });

// 评分字典
dictionary.showGridLines = false;
titleBand(dictionary, "A1:F2", "评分字典", "统一两位审核人的判断尺度");
dictionary.getRange("A4:C4").values = [["字段", "选项", "判定说明"]];
sectionHeader(dictionary.getRange("A4:C4"));
const dictionaryRows = [
  ["解释合理性", "合理", "证据链与解释一致，关系方向、路径和强度足以支撑该解释。"],
  ["解释合理性", "基本合理", "总体解释成立，但仍存在次要证据缺口或表述可改进。"],
  ["解释合理性", "不合理", "核心证据与解释冲突，或不足以支撑当前解释。"],
  ["解释合理性", "无法判断", "仅凭现有匿名证据无法形成判断。"],
  ["进一步核查", "是", "该关系可能产生有效研判线索，值得补充身份、业务或外部证据。"],
  ["进一步核查", "否", "当前关系的核查价值较低。"],
  ["进一步核查", "无法判断", "现有证据不足以评价核查价值。"],
  ["证据充分性", "充分", "现有路径、金额、笔数、时间及结构证据可直接支撑研判。"],
  ["证据充分性", "基本充分", "主要证据具备，但仍需少量背景材料补强。"],
  ["证据充分性", "不足", "关键证据缺失或强度明显不足。"],
  ["证据充分性", "无法判断", "证据口径无法理解或无法据此判断。"],
];
dictionary.getRange("A5:C15").values = dictionaryRows;
dictionary.getRange("A5:C15").format = { wrapText: true, borders: { preset: "all", style: "thin", color: line }, font: { color: text } };
dictionary.getRange("A:A").format.columnWidth = 18;
dictionary.getRange("B:B").format.columnWidth = 16;
dictionary.getRange("C:C").format.columnWidth = 66;
dictionary.getRange("5:15").format.rowHeight = 34;
dictionary.freezePanes.freezeRows(4);

// 汇总结果
summary.showGridLines = false;
titleBand(summary, "A1:H2", "盲审结果自动汇总", "仅统计已填写且可判断的评价；无需手工修改公式");
summary.getRange("A4:D4").values = [["口径", "合计", "审核人A", "审核人B"]];
sectionHeader(summary.getRange("A4:D4"));
const metrics = [
  "已提交合理性评价", "有效合理性评价", "合理/基本合理数量", "解释合理率", "有效核查价值评价", "值得进一步核查比例", "有效证据评价", "证据充分率",
];
summary.getRange("A5:A12").values = metrics.map((value) => [value]);
const r = `'Blind_Review'!$M$6:$M$${lastRow}`;
const a = `'Blind_Review'!$N$6:$N$${lastRow}`;
const e = `'Blind_Review'!$O$6:$O$${lastRow}`;
const slot = `'Blind_Review'!$B$6:$B$${lastRow}`;
summary.getRange("B5:B12").formulas = [
  [`=COUNTIF(${r},"<>")`],
  [`=COUNTIF(${r},Lists!$A$1)+COUNTIF(${r},Lists!$A$2)+COUNTIF(${r},Lists!$A$3)`],
  [`=COUNTIF(${r},Lists!$A$1)+COUNTIF(${r},Lists!$A$2)`],
  ["=IF(B6=0,\"\",B7/B6)"],
  [`=COUNTIF(${a},Lists!$B$1)+COUNTIF(${a},Lists!$B$2)`],
  [`=IF(B9=0,"",COUNTIF(${a},Lists!$B$1)/B9)`],
  [`=COUNTIF(${e},Lists!$C$1)+COUNTIF(${e},Lists!$C$2)+COUNTIF(${e},Lists!$C$3)`],
  [`=IF(B11=0,"",(COUNTIF(${e},Lists!$C$1)+COUNTIF(${e},Lists!$C$2))/B11)`],
];
for (const [col, reviewerCell] of [["C", "Lists!$E$1"], ["D", "Lists!$E$2"]]) {
  summary.getRange(`${col}5:${col}12`).formulas = [
    [`=COUNTIFS(${slot},${reviewerCell},${r},"<>")`],
    [`=COUNTIFS(${slot},${reviewerCell},${r},Lists!$A$1)+COUNTIFS(${slot},${reviewerCell},${r},Lists!$A$2)+COUNTIFS(${slot},${reviewerCell},${r},Lists!$A$3)`],
    [`=COUNTIFS(${slot},${reviewerCell},${r},Lists!$A$1)+COUNTIFS(${slot},${reviewerCell},${r},Lists!$A$2)`],
    [`=IF(${col}6=0,"",${col}7/${col}6)`],
    [`=COUNTIFS(${slot},${reviewerCell},${a},Lists!$B$1)+COUNTIFS(${slot},${reviewerCell},${a},Lists!$B$2)`],
    [`=IF(${col}9=0,"",COUNTIFS(${slot},${reviewerCell},${a},Lists!$B$1)/${col}9)`],
    [`=COUNTIFS(${slot},${reviewerCell},${e},Lists!$C$1)+COUNTIFS(${slot},${reviewerCell},${e},Lists!$C$2)+COUNTIFS(${slot},${reviewerCell},${e},Lists!$C$3)`],
    [`=IF(${col}11=0,"",(COUNTIFS(${slot},${reviewerCell},${e},Lists!$C$1)+COUNTIFS(${slot},${reviewerCell},${e},Lists!$C$2))/${col}11)`],
  ];
}
summary.getRange("A5:D12").format = { borders: { preset: "all", style: "thin", color: line }, font: { color: text } };
summary.getRange("A5:A12").format.fill = pale;
summary.getRange("A5:A12").format.font = { bold: true, color: text };
summary.getRange("B8:D8").setNumberFormat("0.00%");
summary.getRange("B10:D10").setNumberFormat("0.00%");
summary.getRange("B12:D12").setNumberFormat("0.00%");
summary.getRange("A14:D14").merge();
summary.getRange("A14").values = [["参考目标：解释合理率 ≥ 70%。未完成审核前，空白比例不代表未达标。"]];
summary.getRange("A14").format = { fill: "#FFF4D6", font: { color: "#6C4D00", bold: true }, wrapText: true };
summary.getRange("A:A").format.columnWidth = 28;
summary.getRange("B:D").format.columnWidth = 18;
summary.getRange("4:14").format.rowHeight = 28;

lists.getRange("A1:E7").values = [
  ["合理", "是", "充分", "证据与结论不匹配", "A"],
  ["基本合理", "否", "基本充分", "路径过长或关系过弱", "B"],
  ["不合理", "无法判断", "不足", "金额/笔数不足", null],
  ["无法判断", null, "无法判断", "缺少业务背景", null],
  [null, null, null, "时间证据不足", null],
  [null, null, null, "其他", null],
  [null, null, null, "不适用", null],
];
lists.getRange("A1:E7").format = { font: { color: text }, borders: { preset: "all", style: "thin", color: line } };
lists.getRange("A:E").format.columnWidth = 24;

for (const sheet of [instructions, review, dictionary, summary, lists]) encodeNonAsciiValues(sheet);

const inspection = await workbook.inspect({ kind: "sheet,formula", sheetId: "Summary", range: "A1:D14", maxChars: 5000, options: { maxResults: 60 } });
await fs.mkdir(outputDir, { recursive: true });
await fs.writeFile(path.join(outputDir, "workbook_inspection.txt"), inspection.ndjson ?? String(inspection), "utf8");
const outputPath = path.join(outputDir, "链路盲审表.xlsx");
const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(outputPath);
console.log(outputPath);
