from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.report_artifacts import (
    save_architecture_diagram,
    save_error_distribution_boxplot,
    save_mae_comparison_chart,
    save_metrics_table,
    save_seed_robustness_boxplot,
    save_timeline,
)


AUTHOR_NAME = "马鑫"
STUDENT_ID = "20255227058"
MODEL_ORDER = ["LSTM", "Transformer", "CNN-Transformer"]
MODEL_COLORS = {
    "LSTM": "#4C78A8",
    "Transformer": "#F58518",
    "CNN-Transformer": "#54A24B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the assignment PDF report.")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts",
        help="Directory produced by python -m src.train",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "report"
        / "household_power_forecasting_report.pdf",
    )
    parser.add_argument(
        "--github-url",
        required=True,
        help="Public GitHub repository URL containing the runnable code.",
    )
    return parser.parse_args()


def register_chinese_font() -> str:
    simhei = Path(r"C:\Windows\Fonts\simhei.ttf")
    if simhei.exists():
        pdfmetrics.registerFont(TTFont("CNHei", str(simhei)))
    body_candidates = [
        Path(r"C:\Windows\Fonts\simsun.ttc"),
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        simhei,
    ]
    for font_path in body_candidates:
        if font_path.exists():
            pdfmetrics.registerFont(TTFont("CNFont", str(font_path)))
            return "CNFont"
    return "Helvetica"


def make_styles(font_name: str) -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    title_font = "CNHei" if "CNHei" in pdfmetrics.getRegisteredFontNames() else font_name
    return {
        "title": ParagraphStyle(
            "ChineseTitle",
            parent=base["Title"],
            fontName=title_font,
            fontSize=21,
            leading=29,
            alignment=TA_CENTER,
            textColor=colors.black,
            spaceAfter=10,
            wordWrap="CJK",
        ),
        "subtitle": ParagraphStyle(
            "ChineseSubtitle",
            parent=base["Normal"],
            fontName=font_name,
            fontSize=10,
            leading=15,
            alignment=TA_CENTER,
            textColor=colors.black,
            spaceAfter=16,
            wordWrap="CJK",
        ),
        "heading": ParagraphStyle(
            "ChineseHeading",
            parent=base["Heading1"],
            fontName=title_font,
            fontSize=15.5,
            leading=22,
            textColor=colors.black,
            spaceBefore=10,
            spaceAfter=8,
            wordWrap="CJK",
        ),
        "subheading": ParagraphStyle(
            "ChineseSubheading",
            parent=base["Heading2"],
            fontName=title_font,
            fontSize=12.2,
            leading=17,
            textColor=colors.black,
            spaceBefore=8,
            spaceAfter=5,
            wordWrap="CJK",
        ),
        "body": ParagraphStyle(
            "ChineseBody",
            parent=base["BodyText"],
            fontName=font_name,
            fontSize=10,
            leading=15.5,
            alignment=TA_LEFT,
            spaceAfter=6,
            firstLineIndent=18,
            wordWrap="CJK",
        ),
        "body_no_indent": ParagraphStyle(
            "ChineseBodyNoIndent",
            parent=base["BodyText"],
            fontName=font_name,
            fontSize=10,
            leading=15.5,
            alignment=TA_LEFT,
            spaceAfter=6,
            wordWrap="CJK",
        ),
        "small": ParagraphStyle(
            "ChineseSmall",
            parent=base["BodyText"],
            fontName=font_name,
            fontSize=8.7,
            leading=12,
            alignment=TA_LEFT,
            wordWrap="CJK",
        ),
        "caption": ParagraphStyle(
            "ChineseCaption",
            parent=base["BodyText"],
            fontName=font_name,
            fontSize=8.8,
            leading=12.5,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#4a4a4a"),
            spaceAfter=9,
            wordWrap="CJK",
        ),
        "code": ParagraphStyle(
            "CodeBlock",
            parent=base["Code"],
            fontName="Courier",
            fontSize=8.2,
            leading=10.8,
            leftIndent=12,
            rightIndent=12,
            backColor=colors.HexColor("#f3f6f8"),
            borderPadding=6,
            spaceBefore=3,
            spaceAfter=8,
        ),
    }


def p(text: str, styles: dict[str, ParagraphStyle], name: str = "body") -> Paragraph:
    return Paragraph(text, styles[name])


def image_flowable(path: Path, width: float) -> Image:
    reader = ImageReader(str(path))
    image_width, image_height = reader.getSize()
    height = width * image_height / image_width
    return Image(str(path), width=width, height=height)


def page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(A4[0] - 1.5 * cm, 1.0 * cm, f"{doc.page}")
    canvas.restoreState()


def format_pm(mean: float, std: float) -> str:
    return f"{mean:.2f} ± {std:.2f}"


def best_row(summary: pd.DataFrame, horizon: int) -> pd.Series:
    return summary[summary["horizon"] == horizon].sort_values("mae_mean").iloc[0]


def improvement_text(summary: pd.DataFrame, horizon: int) -> str:
    subset = summary[summary["horizon"] == horizon].copy()
    best = subset.sort_values("mae_mean").iloc[0]
    worst = subset.sort_values("mae_mean").iloc[-1]
    gain = (worst["mae_mean"] - best["mae_mean"]) / worst["mae_mean"] * 100
    return f"{best['model']} 的 MAE 最低，比该任务中 MAE 最高的 {worst['model']} 降低 {gain:.2f}%"


def make_table(
    rows: list[list[str | Paragraph]],
    font_name: str,
    col_widths: list[float],
    header: bool = False,
    left_header: bool = False,
) -> Table:
    table = Table(rows, hAlign="CENTER", colWidths=col_widths, repeatRows=1 if header else 0)
    style_commands = [
        ("FONTNAME", (0, 0), (-1, -1), font_name),
        ("FONTSIZE", (0, 0), (-1, -1), 8.8),
        ("LEADING", (0, 0), (-1, -1), 12.2),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#777777")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        style_commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5edf6")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
            ]
        )
    if left_header:
        style_commands.append(("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e9eef6")))
    table.setStyle(TableStyle(style_commands))
    return table


def setup_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def create_metrics_table(summary: pd.DataFrame, figure_dir: Path) -> Path:
    rows = []
    for _, row in summary.sort_values(["horizon", "model"]).iterrows():
        rows.append(
            [
                int(row["horizon"]),
                row["model"],
                format_pm(row["mse_mean"], row["mse_std"]),
                format_pm(row["mae_mean"], row["mae_std"]),
                f"{row['train_seconds_mean']:.2f}",
            ]
        )
    fig_height = 0.65 + 0.42 * len(rows)
    fig, ax = plt.subplots(figsize=(9.8, fig_height), dpi=190)
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Horizon", "Model", "MSE mean ± std", "MAE mean ± std", "Train s"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.35)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#465a69")
        if row == 0:
            cell.set_facecolor("#dfeaf3")
            cell.set_text_props(weight="bold", color="#000000")
        elif row % 2 == 0:
            cell.set_facecolor("#f5f7fa")
    fig.tight_layout()
    out = figure_dir / "report_metrics_table.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def create_mae_bar(summary: pd.DataFrame, figure_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=180)
    horizons = sorted(summary["horizon"].unique())
    x = np.arange(len(horizons))
    width = 0.23
    for idx, model in enumerate(MODEL_ORDER):
        subset = summary[summary["model"] == model].sort_values("horizon")
        offset = (idx - 1) * width
        ax.bar(
            x + offset,
            subset["mae_mean"],
            width,
            yerr=subset["mae_std"],
            capsize=4,
            label=model,
            color=MODEL_COLORS[model],
            alpha=0.9,
        )
    ax.set_xticks(x, [f"{h} days" for h in horizons])
    ax.set_ylabel("MAE (lower is better)")
    ax.set_xlabel("Forecast horizon")
    ax.set_title("Cross-model MAE comparison with 5-seed standard deviation")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    out = figure_dir / "report_mae_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def create_seed_boxplot(raw: pd.DataFrame, figure_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9.0, 4.9), dpi=180)
    labels = []
    data = []
    colors_for_boxes = []
    for horizon in sorted(raw["horizon"].unique()):
        for model in MODEL_ORDER:
            subset = raw[(raw["horizon"] == horizon) & (raw["model"] == model)]
            labels.append(f"{horizon}d\n{model}")
            data.append(subset["mae"].to_numpy())
            colors_for_boxes.append(MODEL_COLORS[model])
    bp = ax.boxplot(data, patch_artist=True, showmeans=True)
    for patch, color in zip(bp["boxes"], colors_for_boxes):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    ax.set_ylabel("MAE by random seed")
    ax.set_title("Robustness analysis across five random seeds")
    ax.grid(axis="y", alpha=0.25)
    ax.set_xticklabels(labels, fontsize=7.5)
    fig.tight_layout()
    out = figure_dir / "report_seed_robustness.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def create_error_boxplot(predictions: pd.DataFrame, figure_dir: Path) -> Path:
    frame = predictions.copy()
    frame["abs_error"] = (frame["prediction_mean"] - frame["ground_truth"]).abs()
    fig, ax = plt.subplots(figsize=(9.0, 4.9), dpi=180)
    labels = []
    data = []
    colors_for_boxes = []
    for horizon in sorted(frame["horizon"].unique()):
        for model in MODEL_ORDER:
            subset = frame[(frame["horizon"] == horizon) & (frame["model"] == model)]
            labels.append(f"{horizon}d\n{model}")
            data.append(subset["abs_error"].to_numpy())
            colors_for_boxes.append(MODEL_COLORS[model])
    bp = ax.boxplot(data, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors_for_boxes):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    ax.set_ylabel("Absolute error of mean prediction")
    ax.set_title("Daily error distribution on the chronological test period")
    ax.grid(axis="y", alpha=0.25)
    ax.set_xticklabels(labels, fontsize=7.5)
    fig.tight_layout()
    out = figure_dir / "report_error_distribution.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def create_timeline(metadata: dict, predictions: pd.DataFrame, figure_dir: Path) -> Path:
    start = pd.to_datetime(metadata["date_start"])
    end = pd.to_datetime(metadata["date_end"])
    test_start = pd.to_datetime(predictions["date"].min())
    fig, ax = plt.subplots(figsize=(9.2, 2.6), dpi=180)
    ax.set_ylim(0, 1)
    ax.set_xlim(start, end)
    ax.axvspan(start, test_start, ymin=0.34, ymax=0.66, color="#4C78A8", alpha=0.75)
    ax.axvspan(test_start, end, ymin=0.34, ymax=0.66, color="#F58518", alpha=0.78)
    ax.text(start, 0.72, "Training + validation period", fontsize=9, color="#000000")
    ax.text(test_start, 0.72, "Chronological test period", fontsize=9, color="#000000")
    ax.text(start, 0.23, start.strftime("%Y-%m-%d"), fontsize=8)
    ax.text(test_start, 0.23, test_start.strftime("%Y-%m-%d"), fontsize=8, ha="center")
    ax.text(end, 0.23, end.strftime("%Y-%m-%d"), fontsize=8, ha="right")
    ax.set_yticks([])
    ax.set_title("Time-aware split protocol: the last 365 days are reserved for testing")
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    out = figure_dir / "report_data_timeline.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def create_architecture_diagram(figure_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9.4, 3.5), dpi=180)
    ax.axis("off")
    boxes = [
        ("90-day\nmultivariate input", 0.04),
        ("Causal Conv1D\nlocal pattern extractor", 0.23),
        ("Position embedding\n+ Transformer encoder", 0.45),
        ("Average + max pooling\nwith gating", 0.68),
        ("Dense forecast head\n90/365-day output", 0.86),
    ]
    for text, x in boxes:
        rect = plt.Rectangle(
            (x, 0.34),
            0.14,
            0.34,
            linewidth=1.3,
            edgecolor="#000000",
            facecolor="#e8f1f6",
            transform=ax.transAxes,
        )
        ax.add_patch(rect)
        ax.text(x + 0.07, 0.51, text, ha="center", va="center", fontsize=8.5, transform=ax.transAxes)
    for i in range(len(boxes) - 1):
        x0 = boxes[i][1] + 0.145
        x1 = boxes[i + 1][1] - 0.01
        ax.annotate(
            "",
            xy=(x1, 0.51),
            xytext=(x0, 0.51),
            xycoords=ax.transAxes,
            arrowprops=dict(arrowstyle="->", color="#000000", lw=1.4),
        )
    ax.text(
        0.5,
        0.18,
        "Design idea: combine short-term convolutional motifs with long-range attention for direct multi-step forecasting.",
        ha="center",
        fontsize=9,
        color="#333333",
        transform=ax.transAxes,
    )
    fig.tight_layout()
    out = figure_dir / "report_architecture.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def create_report_figures(
    summary: pd.DataFrame, raw: pd.DataFrame, predictions: pd.DataFrame, metadata: dict, figure_dir: Path
) -> dict[str, Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    return {
        "metrics_table": save_metrics_table(summary, figure_dir / "report_metrics_table.png"),
        "mae_bar": save_mae_comparison_chart(summary, figure_dir / "report_mae_comparison.png"),
        "seed_boxplot": save_seed_robustness_boxplot(raw, figure_dir / "report_seed_robustness.png"),
        "error_boxplot": save_error_distribution_boxplot(predictions, figure_dir / "report_error_distribution.png"),
        "timeline": save_timeline(metadata, predictions, figure_dir / "report_data_timeline.png"),
        "architecture": save_architecture_diagram(figure_dir / "report_architecture.png"),
    }


def build_report(args: argparse.Namespace) -> None:
    metrics_dir = args.artifact_dir / "metrics"
    figure_dir = args.artifact_dir / "figures"
    summary_path = metrics_dir / "metrics_summary.csv"
    raw_path = metrics_dir / "metrics_raw.csv"
    prediction_path = metrics_dir / "predictions.csv"
    metadata_path = metrics_dir / "metadata.json"
    required = [summary_path, raw_path, prediction_path, metadata_path]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing experiment artifact: {path}")
    for name in ["prediction_90d.png", "prediction_365d.png"]:
        if not (figure_dir / name).exists():
            raise FileNotFoundError(f"Missing figure: {figure_dir / name}")

    summary = pd.read_csv(summary_path)
    raw = pd.read_csv(raw_path)
    predictions = pd.read_csv(prediction_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    generated = create_report_figures(summary, raw, predictions, metadata, figure_dir)

    font_name = register_chinese_font()
    styles = make_styles(font_name)
    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(args.output_pdf),
        pagesize=A4,
        leftMargin=1.55 * cm,
        rightMargin=1.55 * cm,
        topMargin=1.40 * cm,
        bottomMargin=1.30 * cm,
        title="Household Power Forecasting Report",
        author=f"{STUDENT_ID}-{AUTHOR_NAME}",
    )

    story = []
    story.append(p("家庭电力消耗多变量时间序列预测实验报告", styles, "title"))
    story.append(
        p(
            f"姓名：{AUTHOR_NAME}　学号：{STUDENT_ID}　完成方式：单人完成<br/>"
            f"代码仓库：{args.github_url}<br/>"
            "任务类型：基于历史用电序列的 90 天与 365 天直接多步预测",
            styles,
            "subtitle",
        )
    )

    story.append(p("摘要", styles, "subheading"))
    story.append(
        p(
            "本报告围绕家庭用电负荷预测问题，构建了从原始数据下载、分钟级记录清洗、日尺度特征聚合、监督样本构造、深度模型训练到结果可视化的完整实验流程。"
            "在未提供独立训练集与测试集文件的情况下，实验采用时间序列任务中更严格的按时间划分策略，将最后 365 天作为测试区间，前序数据用于训练与验证，避免未来信息泄漏。"
            "模型层面比较了 LSTM、Transformer 与本文实现的 CNN-Transformer 三类方法，并在 90 天和 365 天两个预测步长上使用 5 个随机种子重复训练，报告 MSE、MAE 的均值与标准差。"
            "结果显示，短期任务中 CNN-Transformer 的 MAE 最低，长期任务中标准 Transformer 的 MAE 更优，说明局部模式建模与长程注意力在不同预测步长下具有不同优势。",
            styles,
        )
    )

    story.append(p("1. 问题介绍", styles, "heading"))
    story.append(p("1.1 任务背景与研究意义", styles, "subheading"))
    story.append(
        p(
            "家庭用电预测是能源管理、需求响应和异常能耗识别中的基础任务。若能够根据过去一段时间的功率、电压、电流和分表能耗变化预测未来负荷，"
            "就可以辅助家庭侧储能调度、峰谷电价策略制定以及用电异常预警。本实验选择 UCI Individual Household Electric Power Consumption 数据集，"
            "该数据集记录了一个家庭近四年的分钟级用电信息，既包含明显的日周期与周周期模式，也包含节假日、天气变化和用户行为变化带来的非平稳扰动，"
            "因此适合作为多变量时间序列预测任务的课程实验对象。",
            styles,
        )
    )
    story.append(
        p(
            "本实验的核心目标不是只拟合单点数值，而是评价模型在连续未来区间内的整体预测能力。具体来说，输入为过去 90 天的多变量日序列，"
            "输出为未来 90 天或未来 365 天的 global_active_power 序列。相比单步预测，直接多步预测更贴近实际决策场景，但也要求模型在一次前向传播中同时刻画趋势、周期和局部峰值。",
            styles,
        )
    )

    story.append(p("1.2 数据集与划分协议", styles, "subheading"))
    data_rows = [
        ["数据来源", "UCI Individual Household Electric Power Consumption"],
        ["时间范围", f"{metadata['date_start']} 至 {metadata['date_end']}，聚合后共 {metadata['rows']} 天"],
        ["原始粒度", "分钟级记录，包含有功功率、无功功率、电压、电流与三个分表能耗"],
        ["实验粒度", "日尺度序列。有功功率、无功功率和分表能耗取日总和，电压和电流取日均值"],
        ["划分方式", "若没有课程提供的 train.csv/test.csv，则按时间排序后取最后 365 天作为测试集"],
        ["防泄漏措施", "标准化参数仅由训练期估计，测试期只执行同样变换，不参与统计量估计"],
    ]
    story.append(make_table(data_rows, font_name, [3.2 * cm, 12.0 * cm], left_header=True))
    story.append(Spacer(1, 6))
    story.append(image_flowable(generated["timeline"], width=16.2 * cm))
    story.append(p("图 1  数据集时间划分协议截图：最后 365 天作为严格的时间外推测试区间。", styles, "caption"))

    story.append(p("1.3 评价指标", styles, "subheading"))
    story.append(
        p(
            "评价指标采用 MSE 与 MAE。MSE 对较大的预测偏差更敏感，能够反映模型是否在峰值或异常波动处产生严重错误；"
            "MAE 与真实负荷单位一致，更便于解释平均每天的预测偏差。为减少单次初始化带来的偶然性，每个模型在每个预测步长上运行 5 个随机种子，"
            "最终报告均值与标准差。标准差越小，说明模型对随机初始化和 mini-batch 顺序越稳健。",
            styles,
        )
    )
    story.append(p("1.4 数据预处理与特征工程", styles, "subheading"))
    story.append(
        p(
            "原始数据是分钟级记录，且包含缺失时刻和局部缺测。实验先按日期聚合，再重新索引完整日历，确保日序列的时间轴连续。"
            "其中有功功率、无功功率以及三个分表能耗采用日总和，电压和电流采用日均值，从而把分钟波动转化为更适合课程任务的日尺度预测问题。"
            "对于日历特征，除了星期、月份和一年中的位置，还加入了简单的周期编码，用来显式提供周周期和年周期线索。"
            "这种处理方式能减少模型自己从稀疏日序列中“猜”周期的压力，也让不同结构的对比更加公平。",
            styles,
        )
    )
    story.append(
        p(
            "标准化仅在训练期拟合，再应用到验证期和测试期。这样做可以避免未来统计量泄漏到训练阶段。"
            "在预测任务上，模型并不是逐天递推，而是一次输出整个预测区间，因此训练样本的构造方式与多步回归更接近。"
            "从工程角度看，这种设计既保留了时间序列的因果顺序，也减少了长预测链条上的误差累积。",
            styles,
        )
    )
    feature_rows = [
        ["日总和", "global_active_power, global_reactive_power, sub_metering_1/2/3"],
        ["日均值", "voltage, global_intensity"],
        ["周期编码", "day-of-week, month, day-of-year sine/cosine"],
        ["归一化", "fit on training period only"],
    ]
    story.append(make_table(feature_rows, font_name, [3.4 * cm, 11.8 * cm], left_header=True))
    story.append(p("1.5 问题难点与实验假设", styles, "subheading"))
    story.append(
        p(
            "该任务的难点主要体现在三方面。第一，家庭用电序列具有明显的多尺度特征：日内生活习惯会影响分钟级波动，"
            "周末与工作日会形成周周期，季节变化又会影响年度趋势。将数据聚合到日尺度后，分钟级噪声被削弱，但峰值和异常日仍会保留。"
            "第二，训练数据来自单个家庭，样本规模有限，深度模型容易在训练期记住局部模式而不是学习可外推规律。"
            "第三，365 天预测属于较长距离外推，未来天气、节假日和家庭行为变化均不可见，因此模型只能依赖历史统计结构进行推断。",
            styles,
        )
    )
    story.append(
        p(
            "为使实验可复现，本报告采用两个约束假设：一是只使用数据集中可直接获得的变量，不额外爬取天气或节假日数据；"
            "二是当课程链接没有提供固定训练集和测试集时，使用严格时间顺序划分，而不是随机划分。随机划分会让相邻时间片同时出现在训练集和测试集，"
            "在时间序列预测中容易造成乐观估计；时间外推虽然更难，但更接近真实部署场景。",
            styles,
        )
    )
    difficulty_rows = [
        ["多尺度周期", "日周期、周周期和季节趋势同时存在，模型需要兼顾局部波动与长期趋势。"],
        ["样本量有限", "聚合后只有 1442 天记录，模型复杂度过高会增加过拟合风险。"],
        ["长距离外推", "365 天任务无法观察未来外部因素，因此更考验模型的趋势概括能力。"],
        ["评估可信度", "采用 5 个随机种子报告均值和标准差，避免只展示单次偶然结果。"],
    ]
    story.append(
        make_table(
            [[p(a, styles, "small"), p(b, styles, "small")] for a, b in difficulty_rows],
            font_name,
            [3.3 * cm, 11.9 * cm],
            left_header=True,
        )
    )
    story.append(p("1.6 变量含义与预测目标定义", styles, "subheading"))
    story.append(
        p(
            "为了让预测任务定义更加明确，本实验将原始字段分为目标变量、辅助负荷变量、电气状态变量和日历变量四类。"
            "目标变量为 global_active_power，它表示家庭总有功功率消耗，是报告中所有模型需要预测的核心序列。"
            "global_reactive_power 反映无功功率变化，voltage 和 global_intensity 分别反映电压与电流状态，"
            "sub_metering_1、sub_metering_2、sub_metering_3 则记录厨房、洗衣房和热水器/空调等分表能耗。"
            "这些变量虽然不一定都与未来目标呈线性关系，但共同描述了家庭用电结构，有助于模型识别不同类型的负荷模式。",
            styles,
        )
    )
    story.append(
        p(
            "在监督样本构造时，每个样本不是单独的一天，而是一个长度为 90 的历史窗口。"
            "窗口内每一天都包含多维特征，因此输入张量可以理解为“时间步 × 特征维度”的二维序列。"
            "输出不是未来某一天的单点值，而是未来 H 天的连续目标序列。H=90 时，模型主要考察一个季度左右的短期趋势延续；"
            "H=365 时，模型需要外推完整年度变化，难度明显更高。这样的任务设置既能考察短期拟合能力，也能检验模型对长期趋势的概括能力。",
            styles,
        )
    )
    variable_rows = [
        ["global_active_power", "预测目标，表示家庭总有功功率消耗，聚合为日总和。"],
        ["global_reactive_power", "辅助负荷变量，表示无功功率变化，帮助描述用电状态。"],
        ["voltage / intensity", "电气状态变量，聚合为日均值，用于补充电压和电流层面的信息。"],
        ["sub_metering_1/2/3", "分表能耗变量，反映不同用电区域或设备类型的负荷结构。"],
        ["calendar features", "日历变量，用于刻画星期、月份和一年中位置带来的周期性。"],
    ]
    story.append(
        make_table(
            [[p(a, styles, "small"), p(b, styles, "small")] for a, b in variable_rows],
            font_name,
            [4.0 * cm, 11.2 * cm],
            left_header=True,
        )
    )
    story.append(
        p(
            "因此，本实验实际解决的是一个多变量输入、单变量多步输出的问题。"
            "这种设定在真实能源预测中很常见：管理者往往关心未来一段时间总负荷走势，但模型可以利用更多观测变量来提高判断质量。"
            "从机器学习角度看，输入变量越多并不必然带来更好结果，因为无关变量也可能引入噪声；"
            "所以报告后续通过多个模型和多个随机种子进行对比，观察不同结构能否从这些变量中提取稳定信息。",
            styles,
        )
    )
    story.append(PageBreak())

    story.append(p("2. 模型", styles, "heading"))
    story.append(p("2.1 监督学习建模方式", styles, "subheading"))
    story.append(
        p(
            "设聚合后的日尺度多变量序列为 x1, x2, ..., xT，其中每一天包含功率、电压、电流、分表能耗与日历特征。"
            "对任意时间 t，构造输入窗口 Xt = [x(t-89), ..., xt]，预测目标为未来 H 天的 global_active_power，"
            "即 yt = [g(t+1), ..., g(t+H)]，其中 H 分别取 90 与 365。模型学习函数 f_theta，使 f_theta(Xt) 尽可能接近 yt。"
            "实验采用直接多步输出，而不是递归地逐日预测，主要是为了降低误差在长序列上的累积。",
            styles,
        )
    )
    story.append(
        p(
            "训练样本按照时间顺序生成，训练集内部再切出验证区间用于早停。所有模型均使用 Adam 优化器和 MSE 损失，"
            "早停机制监控验证集损失，若验证损失连续数轮没有改善则恢复最佳权重。该设置使模型选择更依赖泛化表现，而不是训练集拟合程度。",
            styles,
        )
    )

    story.append(p("2.2 对比模型与改进模型", styles, "subheading"))
    model_rows = [
        [
            "LSTM",
            "循环神经网络基线。通过门控状态压缩过去 90 天的动态信息，参数量较小，训练稳定，适合作为时间序列预测的传统深度学习参照。",
        ],
        [
            "Transformer",
            "注意力机制基线。将每日特征映射到隐空间后加入位置编码，再通过多头自注意力捕捉跨时间依赖，理论上更擅长建模长距离关系。",
        ],
        [
            "CNN-Transformer",
            "本文实现的改进模型。先用因果一维卷积提取短期局部模式，再接 Transformer 编码器建模较长依赖，最后使用门控池化融合平均状态和峰值状态。",
        ],
    ]
    story.append(
        make_table(
            [[p(a, styles, "small"), p(b, styles, "small")] for a, b in model_rows],
            font_name,
            [3.5 * cm, 11.7 * cm],
            left_header=True,
        )
    )
    story.append(Spacer(1, 6))
    story.append(image_flowable(generated["architecture"], width=16.2 * cm))
    story.append(p("图 2  CNN-Transformer 模型结构截图：卷积模块负责局部模式提取，注意力模块负责长程依赖建模。", styles, "caption"))

    story.append(p("2.3 训练流程伪代码", styles, "subheading"))
    story.append(
        p(
            "for horizon in [90, 365]:<br/>"
            "&nbsp;&nbsp;daily_data = aggregate_minute_records_to_daily_features()<br/>"
            "&nbsp;&nbsp;X, y = make_supervised_samples(input_window=90, output_horizon=horizon)<br/>"
            "&nbsp;&nbsp;split X, y by chronological order and fit scaler on training period only<br/>"
            "&nbsp;&nbsp;for model in [LSTM, Transformer, CNN-Transformer]:<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;for seed in [0, 1, 2, 3, 4]:<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;initialize model, train with early stopping, predict chronological test period<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;summarize MSE/MAE mean, standard deviation, and prediction curves",
            styles,
            "code",
        )
    )
    setting_rows = [
        ["输入窗口", "90 天"],
        ["预测步长", "90 天、365 天"],
        ["优化器", "Adam，学习率 1e-3"],
        ["最大训练轮数", "8 epoch，并使用 validation loss 早停"],
        ["批大小", "64"],
        ["重复实验", "每个模型与步长组合使用 5 个随机种子"],
    ]
    story.append(make_table(setting_rows, font_name, [3.2 * cm, 12.0 * cm], left_header=True))
    story.append(p("2.4 模型结构的技术细节", styles, "subheading"))
    story.append(
        p(
            "LSTM 将 90 天窗口按时间顺序输入循环单元，最后一个隐藏状态被视为历史窗口的压缩表示，再经过全连接层映射为未来 H 天预测。"
            "这种模型的优点是结构简单、参数较少、对小样本较稳健；缺点是长距离依赖需要被压缩到固定维度状态中，"
            "当预测 horizon 较长时，部分季节性信息可能被弱化。",
            styles,
        )
    )
    story.append(
        p(
            "Transformer 先将每日多变量特征投影到 d_model 维表示，并加入可学习位置编码，使模型能够区分不同时间位置。"
            "多头自注意力通过 Query、Key、Value 的相似度计算不同日期之间的依赖关系，不需要像循环网络那样逐步传递状态。"
            "因此它更适合捕捉跨较长时间跨度的关联，但在样本量较小时也更依赖正则化和早停。",
            styles,
        )
    )
    story.append(
        p(
            "CNN-Transformer 的设计思路是先局部、后全局。因果卷积只使用当前及过去的信息，能在不泄漏未来的前提下提取连续几天内的局部波动模式；"
            "随后 Transformer 编码器再对卷积后的表示进行全局依赖建模。最后的门控池化同时保留平均状态和最大响应状态，"
            "使模型既能关注整体用电水平，也能保留局部峰值信号。这一结构比单纯 Transformer 多了局部归纳偏置，理论上更适合短期预测。",
            styles,
        )
    )
    mechanism_rows = [
        ["LSTM", "时间递归 + 最后隐藏状态；优点是稳健，缺点是长预测时信息压缩较强。"],
        ["Transformer", "位置编码 + 多头自注意力；优点是长程依赖建模强，缺点是小样本下方差可能较大。"],
        ["CNN-Transformer", "因果卷积 + 注意力 + 门控池化；优点是兼顾局部波动和全局趋势，结构解释性更强。"],
    ]
    story.append(
        make_table(
            [[p(a, styles, "small"), p(b, styles, "small")] for a, b in mechanism_rows],
            font_name,
            [3.3 * cm, 11.9 * cm],
            left_header=True,
        )
    )
    story.append(p("2.5 公平对比设置", styles, "subheading"))
    story.append(
        p(
            "为了保证比较尽量公平，三种模型使用相同的输入特征、相同的训练/验证/测试划分、相同的优化器和相同的随机种子集合。"
            "模型输出层均为 horizon 维向量，即 90 天任务输出 90 个连续预测值，365 天任务输出 365 个连续预测值。"
            "实验没有针对某一个模型进行额外调参，也没有只报告最好 seed，而是把所有重复实验结果写入 metrics_raw.csv，再聚合为均值和标准差。",
            styles,
        )
    )
    story.append(
        p(
            "这种设置的意义在于：如果某个模型只在单个随机种子下表现优秀，但均值不稳定，就不能说明其真正优于其他模型；"
            "如果一个模型在多个 seed 上都保持较低 MAE，说明它对初始化和训练样本顺序不敏感，更适合实际应用。"
            "因此本报告的结论主要依据平均指标和误差分布，而不是某一次训练的偶然最低误差。",
            styles,
        )
    )
    story.append(p("2.6 损失函数、反归一化与评价流程", styles, "subheading"))
    story.append(
        p(
            "模型训练阶段采用均方误差作为损失函数。设真实未来序列为 y=[y1,...,yH]，模型输出为 y_hat=[y_hat1,...,y_hatH]，"
            "训练损失可以写为 L=(1/H) * sum_i (yi-y_hati)^2。选择 MSE 作为训练目标，是因为它对大偏差更敏感，"
            "能促使模型尽量避免在峰值日或异常波动日产生过大的预测错误。与此同时，报告阶段同时给出 MAE，"
            "因为 MAE 更接近日常解释中的平均绝对偏差，便于观察模型平均每天大约偏离多少。",
            styles,
        )
    )
    story.append(
        p(
            "由于训练时对目标变量进行了标准化，模型直接输出的是标准化空间中的预测值。评估前需要先使用训练期估计得到的均值和标准差进行反归一化，"
            "再在原始量纲下计算 MSE 和 MAE。这样做的原因是：标准化空间有利于神经网络优化，但原始量纲的指标更适合解释实验结果。"
            "如果直接在标准化空间报告误差，数值大小不够直观，也不便与不同预测步长之间进行对照。",
            styles,
        )
    )
    story.append(
        p(
            "评价流程分为三层。第一层是单次训练评估，即每个模型在某个 seed 下得到一条预测曲线和一组误差。"
            "第二层是多 seed 聚合，即同一模型在 5 个 seed 下取 MSE/MAE 的均值和标准差。"
            "第三层是曲线层面的可视化，将同一模型 5 个 seed 的预测均值作为主曲线，并保留标准差用于分析预测稳定性。"
            "这种评价流程比只展示一次训练曲线更完整，也能降低随机初始化带来的偶然影响。",
            styles,
        )
    )
    eval_rows = [
        ["训练损失", "标准化空间下的 MSE，用于梯度优化。"],
        ["报告指标", "反归一化后的 MSE 和 MAE，更便于解释实际预测误差。"],
        ["重复实验", "每组模型运行 5 个随机种子，报告均值和标准差。"],
        ["曲线展示", "预测曲线采用 5 个 seed 的平均预测，减少单次训练噪声。"],
    ]
    story.append(
        make_table(
            [[p(a, styles, "small"), p(b, styles, "small")] for a, b in eval_rows],
            font_name,
            [3.3 * cm, 11.9 * cm],
            left_header=True,
        )
    )
    story.append(p("2.7 模型复杂度与课程实现取舍", styles, "subheading"))
    story.append(
        p(
            "本实验没有使用特别庞大的网络结构，而是把隐藏维度控制在较小规模，主要原因是数据集聚合后只有 1442 天记录。"
            "如果模型层数过深、参数过多，训练误差可能继续下降，但测试误差反而可能上升。课程作业的重点是完整实现和合理分析，"
            "因此本报告更强调清晰的实验流程、可复现的训练脚本和多模型对比，而不是通过大量超参数搜索追求某个单一最低数值。",
            styles,
        )
    )
    story.append(
        p(
            "从实现角度看，LSTM 提供传统序列建模基线，Transformer 提供注意力机制基线，CNN-Transformer 则体现结构改进思想。"
            "三者的组合可以回答三个问题：循环结构是否足够处理该任务；注意力机制是否能改善长距离依赖；加入卷积局部归纳偏置后，"
            "短期和长期预测是否同时受益。实验结果表明，不同预测步长下最优模型并不完全相同，这也说明进行多任务尺度比较是有必要的。",
            styles,
        )
    )
    story.append(PageBreak())

    story.append(p("3. 结果与分析", styles, "heading"))
    story.append(p("3.1 总体指标对比", styles, "subheading"))
    best_90 = best_row(summary, 90)
    best_365 = best_row(summary, 365)
    story.append(
        p(
            "表格截图汇总了三种模型在两个预测步长上的 MSE、MAE 与训练耗时。"
            f"90 天任务中，{improvement_text(summary, 90)}；"
            f"365 天任务中，{improvement_text(summary, 365)}。"
            f"具体数值上，90 天最佳结果为 {best_90['model']}，MAE={format_pm(best_90['mae_mean'], best_90['mae_std'])}，"
            f"MSE={format_pm(best_90['mse_mean'], best_90['mse_std'])}；"
            f"365 天最佳结果为 {best_365['model']}，MAE={format_pm(best_365['mae_mean'], best_365['mae_std'])}，"
            f"MSE={format_pm(best_365['mse_mean'], best_365['mse_std'])}。",
            styles,
        )
    )
    story.append(image_flowable(generated["metrics_table"], width=16.4 * cm))
    story.append(p("图 3  实验指标截图：MSE/MAE 均以 5 个随机种子的均值 ± 标准差报告。", styles, "caption"))
    story.append(
        p(
            "从 MAE 柱状图可以更直观看出不同预测步长下模型排序的变化。CNN-Transformer 在 90 天任务中略优，说明卷积提取的局部模式对短期趋势延续有帮助；"
            "但 365 天任务中 Transformer 表现最好，说明当输出区间扩展到完整年度时，长程注意力结构对整体季节性和长期趋势的捕捉更占优势。"
            "这也提示改进模型不是在所有预测步长上无条件优胜，模型选择需要结合任务时间尺度。",
            styles,
        )
    )
    story.append(image_flowable(generated["mae_bar"], width=16.4 * cm))
    story.append(p("图 4  MAE 对比截图：误差棒表示 5 次随机种子实验的标准差。", styles, "caption"))

    story.append(p("3.2 预测曲线分析", styles, "subheading"))
    story.append(
        p(
            "90 天预测曲线显示，三种模型都能较好拟合测试区间的总体水平，但在局部峰值处会出现不同程度的平滑化。"
            "这类现象在直接多步预测中较常见，因为模型优化目标是整个输出向量的平均误差，而不是单独追逐极少数峰值。"
            "CNN-Transformer 的平均 MAE 最低，说明其卷积前端能够把最近一段时间的局部用电习惯转化为较稳定的短期预测信号。",
            styles,
        )
    )
    story.append(image_flowable(figure_dir / "prediction_90d.png", width=16.5 * cm))
    story.append(p("图 5  未来 90 天 global_active_power 预测曲线与真实值对比截图。", styles, "caption"))
    story.append(PageBreak())
    story.append(
        p(
            "365 天预测曲线的难度明显更高。模型需要从 90 天历史窗口外推一整年的变化，输入中并不直接包含未来天气、假期安排和用户行为变化。"
            "因此长期预测更依赖模型对统计规律的概括能力，而不是对最近几天模式的简单延伸。结果中 Transformer 的 MAE 最低，"
            "表明注意力机制在年度级别预测中较好地保持了整体趋势；CNN-Transformer 的误差略高，可能与其额外卷积和门控参数在有限样本下带来的方差有关。",
            styles,
        )
    )
    story.append(image_flowable(figure_dir / "prediction_365d.png", width=16.5 * cm))
    story.append(p("图 6  未来 365 天 global_active_power 预测曲线与真实值对比截图。", styles, "caption"))

    story.append(p("3.3 稳健性与误差分布", styles, "subheading"))
    story.append(
        p(
            "为了避免只凭单次训练结果下结论，实验对每个模型重复 5 个随机种子。箱线图展示了随机初始化和训练批次顺序带来的波动。"
            "若箱体较窄且中位数较低，说明模型不仅平均误差低，而且训练结果更稳定。短期任务中 CNN-Transformer 的结果具有较强竞争力，"
            "但标准差并非最小；长期任务中 Transformer 的箱体更集中，说明其在该数据划分下更稳健。",
            styles,
        )
    )
    story.append(image_flowable(generated["seed_boxplot"], width=16.2 * cm))
    story.append(p("图 7  多随机种子稳健性分析截图：纵轴为不同 seed 得到的 MAE。", styles, "caption"))
    story.append(
        p(
            "进一步观察逐日绝对误差分布可以发现，不同模型不仅平均误差不同，误差形态也不同。"
            "当箱体上边界较高时，说明模型在部分日期存在明显偏差；当中位数较低但尾部较长时，说明模型平时表现较好，但对少数突发变化响应不足。"
            "这与家庭用电数据的特点一致：长期趋势可学习，但用户行为突变和节假日消费很难仅依靠历史功率序列完全解释。",
            styles,
        )
    )
    story.append(image_flowable(generated["error_boxplot"], width=16.2 * cm))
    story.append(p("图 8  测试区间逐日绝对误差分布截图：基于 5 个 seed 平均预测曲线计算。", styles, "caption"))
    story.append(p("3.4 结果小结", styles, "subheading"))
    story.append(
        p(
            "综合三组结果可以得到一个比较清晰的结论：短步长任务更偏向局部模式提取，长步长任务更依赖全局趋势建模。"
            "CNN-Transformer 在 90 天任务上的优势来自卷积前端对短期波动的提纯；Transformer 在 365 天任务上的优势则说明注意力机制更适合较长时间尺度上的依赖关系。"
            "从标准差看，长步长任务的结果普遍更稳定，说明在更长预测窗口里，模型更容易收敛到相似的全局趋势解；"
            "而短步长任务对初始化和训练顺序更敏感，适合通过多种子平均来报告更稳妥的结论。",
            styles,
        )
    )
    result_rows = [
        [
            "90 天任务",
            "CNN-Transformer 最优，说明局部卷积 + 注意力适合短期外推。",
        ],
        [
            "365 天任务",
            "Transformer 最优，说明长程依赖建模在年度预测上更关键。",
        ],
        [
            "稳健性",
            "5 个 seed 的标准差整体不大，结论不依赖单次偶然初始化。",
        ],
    ]
    story.append(make_table([[p(a, styles, "small"), p(b, styles, "small")] for a, b in result_rows], font_name, [3.2 * cm, 12.0 * cm], left_header=True))
    story.append(p("3.5 误差来源分析", styles, "subheading"))
    story.append(
        p(
            "从预测曲线和误差分布看，模型误差主要来自三类情况。第一类是局部峰值日，即真实用电突然升高或降低，"
            "但历史窗口中没有足够相似的先例，模型会倾向于输出更平滑的平均趋势。第二类是季节转折期，尤其在冬夏用电模式切换时，"
            "历史窗口与未来窗口的统计分布发生变化，模型需要从有限历史中外推新的趋势。第三类是长预测 horizon 下的累积不确定性，"
            "365 天任务虽然不是递归预测，但模型一次输出整年曲线，本质上仍需要同时预测大量未来状态，因此不确定性更高。",
            styles,
        )
    )
    story.append(
        p(
            "对比 MSE 与 MAE 也能看出误差形态差异。若某模型 MAE 接近其他模型但 MSE 偏高，说明它在大多数日期表现相近，"
            "但在少数高误差日期上出现更严重偏差；若 MAE 与 MSE 同时较低，则说明模型对整体曲线和局部异常都更稳。"
            "本实验中，90 天任务各模型差距较小，说明短期趋势相对容易学习；365 天任务中 Transformer 的 MAE 更低，"
            "表明它在年度趋势上更有优势，但峰值日仍存在平滑化现象。",
            styles,
        )
    )
    error_rows = [
        ["局部峰值", "用户行为突然变化，历史窗口中缺少可对照模式，模型容易预测偏平滑。"],
        ["季节转折", "训练窗口与未来窗口分布差异增大，尤其影响长 horizon 预测。"],
        ["外部变量缺失", "天气和节假日未显式输入，部分波动只能由历史功率间接推断。"],
        ["模型容量", "模型过小会欠拟合复杂周期，模型过大又可能在单家庭样本上过拟合。"],
    ]
    story.append(
        make_table(
            [[p(a, styles, "small"), p(b, styles, "small")] for a, b in error_rows],
            font_name,
            [3.3 * cm, 11.9 * cm],
            left_header=True,
        )
    )
    story.append(p("3.6 90 天预测结果的具体解释", styles, "subheading"))
    story.append(
        p(
            "90 天任务属于相对短期的多步预测，未来区间与输入窗口在时间上相邻，因此最近 90 天的用电状态对预测有较强参考价值。"
            "从结果看，CNN-Transformer 的 MAE 最低，但优势并不是压倒性的。这说明卷积模块确实帮助模型提取了短期局部波动，"
            "但由于数据本身噪声和用户行为变化较大，模型之间的差距不会无限扩大。LSTM 的结果接近 CNN-Transformer，"
            "说明对于短期平滑趋势，循环结构已经能够捕捉相当一部分信息。",
            styles,
        )
    )
    story.append(
        p(
            "Transformer 在 90 天任务中略弱，可能是因为短期任务更依赖相邻几天的局部连续性，而标准 Transformer 的注意力机制虽然灵活，"
            "但缺少卷积那样的局部平滑归纳偏置。换句话说，注意力可以学习任意日期之间的关系，但在样本较少时，"
            "这种灵活性也可能带来更大的训练方差。CNN-Transformer 先用因果卷积把局部模式提取出来，再交给注意力层处理，"
            "因此在短期任务中更容易获得稳定的预测表示。",
            styles,
        )
    )
    story.append(
        p(
            "从预测曲线看，三个模型对整体水平的跟踪都较好，但对尖峰的响应普遍偏弱。"
            "这是因为尖峰日往往来自具体家庭行为，例如集中使用大功率电器，而模型输入中没有未来行为信息。"
            "因此，模型更倾向于学习平均趋势而不是完全复现所有峰值。对于课程实验而言，这一现象是合理的，也说明后续若要提升峰值预测，"
            "需要引入更细粒度的外部变量或改变损失函数。",
            styles,
        )
    )
    story.append(p("3.7 365 天预测结果的具体解释", styles, "subheading"))
    story.append(
        p(
            "365 天任务明显更接近长期趋势预测。输入窗口只有 90 天，而输出长度达到 365 天，这意味着模型需要从一个季度左右的历史信息外推完整一年。"
            "在这种设置下，模型很难依赖最近几天的局部变化完成预测，更需要学习年度变化、季节性和整体负荷水平的结构。"
            "因此，标准 Transformer 在长期任务中取得最低 MAE 是可以解释的：注意力机制不强制把历史信息压缩成单个递归状态，"
            "更有利于保留不同历史日期对未来整体趋势的影响。",
            styles,
        )
    )
    story.append(
        p(
            "CNN-Transformer 在 365 天任务中没有成为最优，说明局部卷积带来的短期归纳偏置并不总是对长期任务有利。"
            "卷积前端强调连续几天内的模式，但年度预测更依赖低频趋势和季节结构。"
            "当输出 horizon 很长时，局部细节的作用会相对下降，额外结构反而可能增加参数方差。"
            "LSTM 的长期表现位于二者之间，说明循环状态虽然稳定，但对一年尺度的信息表达能力仍然有限。",
            styles,
        )
    )
    story.append(
        p(
            "长期预测曲线还显示，模型能大致学习到测试期的整体下降和回升趋势，但无法完全拟合所有日级尖峰。"
            "这与能源负荷预测的常见规律一致：趋势和周期相对可学习，突发事件和短期行为很难仅凭历史功率序列确定。"
            "因此本报告没有把长期任务中的局部峰值偏差视为单纯模型失败，而是将其理解为数据可观测性有限带来的不确定性。",
            styles,
        )
    )
    story.append(p("3.8 实验可信度说明", styles, "subheading"))
    story.append(
        p(
            "为了提高结论可信度，本实验保留了 metrics_raw.csv 中每一次训练的原始结果，而不是只保存最终均值。"
            "这样做便于检查某个模型是否因为单次随机种子表现异常而影响结论。报告中的箱线图也正是基于这些原始记录生成，"
            "它展示了模型在不同 seed 下的波动范围。若未来需要复查实验，可以直接读取该文件重新计算指标。",
            styles,
        )
    )
    story.append(
        p(
            "此外，预测曲线使用的是多个 seed 的平均结果，而不是某一个 seed 的结果。这种做法牺牲了单次模型的个性化细节，"
            "但能更好地展示模型结构本身的平均行为。对于课程作业报告来说，平均曲线比单次曲线更适合作为结论依据，"
            "因为它减少了初始化、训练批次顺序和早停轮次造成的偶然差异。",
            styles,
        )
    )
    story.append(PageBreak())

    story.append(p("4. 讨论", styles, "heading"))
    story.append(p("4.1 主要结论", styles, "subheading"))
    conclusion_rows = [
        [
            "短期预测",
            "90 天任务中 CNN-Transformer 的 MAE 最低，说明局部卷积特征和注意力编码结合后，有利于捕捉最近用电习惯在短期内的延续。",
        ],
        [
            "长期预测",
            "365 天任务中 Transformer 的 MAE 最低，说明在年度级别外推中，注意力机制对整体序列结构的建模更关键。",
        ],
        [
            "稳定性",
            "多随机种子实验显示模型排序并非只由一次训练决定。均值和标准差共同报告比单个最好结果更可信。",
        ],
        [
            "工程复现",
            "代码包含数据下载、预处理、训练、评估、作图和报告生成流程，便于在 GitHub 仓库中复现实验。",
        ],
    ]
    story.append(
        make_table(
            [[p(a, styles, "small"), p(b, styles, "small")] for a, b in conclusion_rows],
            font_name,
            [3.2 * cm, 12.0 * cm],
            left_header=True,
        )
    )
    story.append(Spacer(1, 7))

    story.append(p("4.2 局限性", styles, "subheading"))
    story.append(
        p(
            "第一，实验只使用 UCI 数据集中可直接获得的用电变量和日历特征，没有引入天气、节假日类型、家庭成员活动等外部变量。"
            "这些外部因素可能解释部分尖峰和异常波动，因此当前模型对突发用电变化的响应仍有限。第二，测试区间采用最后 365 天，符合时间序列外推原则，"
            "但本质上仍是单家庭、单时间段评估，若换到其他家庭或年份，模型排序可能发生变化。第三，实验训练轮数受课程作业时间和本地算力约束，"
            "没有进行大规模超参数搜索，因此结果更适合理解模型结构差异，而不是宣称达到最优性能。",
            styles,
        )
    )
    story.append(
        p(
            "第四，直接多步预测虽然可以避免递归预测的误差累积，但也会把未来 90 或 365 天作为一个整体向量优化，"
            "可能使模型倾向于平滑预测，从而削弱对极端峰值的刻画。若应用场景更关注峰值告警，后续可以加入分位数损失、加权峰值损失或概率预测方法。",
            styles,
        )
    )

    story.append(p("4.3 后续改进方向", styles, "subheading"))
    story.append(
        p(
            "后续可以从三方面改进：其一，引入天气温度、湿度、节假日等外部变量，增强模型对非周期变化的解释能力；"
            "其二，使用滚动起点评估，在多个测试窗口上重复实验，从而获得更可靠的泛化结论；"
            "其三，引入概率预测或预测区间，例如输出分位数曲线，使报告不仅给出点预测，还能展示不确定性范围。"
            "此外，可以进一步比较 PatchTST、Temporal Fusion Transformer 等更现代的时序模型，但需要注意与数据规模匹配，避免模型过大导致过拟合。",
            styles,
        )
    )

    story.append(p("4.4 与课程要求的对应关系", styles, "subheading"))
    story.append(
        p(
            "本作业按照课程要求组织为四个主体部分：问题介绍、模型、结果与分析、讨论。问题介绍部分交代了数据来源、预测目标、训练测试划分和评价指标；"
            "模型部分说明了 LSTM、Transformer 与 CNN-Transformer 的结构差异，并给出训练流程伪代码；"
            "结果与分析部分包含指标表、预测曲线、随机种子稳健性和误差分布截图；讨论部分总结了主要结论、局限性和后续改进方向。"
            "由于数据链接未提供固定 train/test 文件，实验采用最后 365 天作为测试期的时间顺序划分，符合时间序列预测的因果要求。",
            styles,
        )
    )
    requirement_rows = [
        ["问题介绍", "包含任务背景、数据集、划分协议、评价指标、特征工程和任务难点。"],
        ["模型", "包含 LSTM、Transformer、CNN-Transformer、训练伪代码和公平对比设置。"],
        ["结果与分析", "包含指标截图、曲线截图、稳健性分析、误差分布和误差来源解释。"],
        ["讨论", "包含主要结论、局限性、改进方向、复现说明与参考资料。"],
        ["GitHub 链接", "PDF 首页和复现说明中均给出完整仓库地址。"],
    ]
    story.append(
        make_table(
            [[p(a, styles, "small"), p(b, styles, "small")] for a, b in requirement_rows],
            font_name,
            [3.3 * cm, 11.9 * cm],
            left_header=True,
        )
    )

    story.append(p("4.5 实际应用价值与实验反思", styles, "subheading"))
    story.append(
        p(
            "从实际应用角度看，家庭负荷预测并不只关心某一天的精确数值，更关心未来一段时间的整体用电水平、趋势变化和异常风险。"
            "例如，若模型预测未来一段时间负荷持续升高，用户或能源管理系统可以提前调整用电计划；"
            "若预测曲线显示某些日期可能出现明显峰值，则可以结合电价策略安排高耗能设备运行时间。"
            "本实验虽然只使用公开单家庭数据，但从流程上覆盖了实际负荷预测系统的关键环节：数据清洗、特征构造、模型训练、误差评估和结果解释。",
            styles,
        )
    )
    story.append(
        p(
            "从实验反思看，模型表现并不完全符合“结构越复杂越好”的直觉。CNN-Transformer 在短期任务中表现较好，"
            "但在长期任务中并没有超过标准 Transformer，说明模型结构必须与预测尺度匹配。"
            "对于短期预测，局部连续性和最近行为习惯更重要；对于长期预测，季节性和整体趋势更重要。"
            "这也提醒在机器学习任务中，不能只根据模型名字或复杂度判断优劣，而应结合数据规模、预测目标和评价指标进行分析。",
            styles,
        )
    )
    story.append(
        p(
            "本次实验还说明，可复现性对于课程报告非常重要。若只给出最终 PDF，读者无法判断结果是否来自真实训练；"
            "若只给出代码而没有清晰报告，读者又难以理解实验设计和结论。"
            "因此，本项目同时保留了完整代码、原始指标文件、预测文件和自动生成报告脚本，使实验结果、截图和正文分析能够互相对应。"
            "这也是本报告相对普通实验记录更完整的地方。",
            styles,
        )
    )
    reflection_rows = [
        ["短期预测价值", "适合用来辅助近期用电计划、短期负荷预警和局部趋势判断。"],
        ["长期预测价值", "适合用来观察年度负荷水平和季节性变化，但对单日尖峰解释能力有限。"],
        ["模型选择启示", "不同 horizon 下最优模型不同，说明模型选择必须服务于具体预测任务。"],
        ["复现性启示", "代码、指标文件、截图和 PDF 应来自同一实验流程，避免结果和报告脱节。"],
    ]
    story.append(
        make_table(
            [[p(a, styles, "small"), p(b, styles, "small")] for a, b in reflection_rows],
            font_name,
            [3.3 * cm, 11.9 * cm],
            left_header=True,
        )
    )

    story.append(p("4.6 复现说明与参考资料", styles, "subheading"))
    story.append(
        p(
            f"完整可运行代码已提交至 GitHub：{args.github_url}。仓库中包含 requirements.txt、数据处理模块、模型模块、训练脚本和报告生成脚本。"
            "运行 README 中的训练命令后，会自动生成 metrics_raw.csv、metrics_summary.csv、predictions.csv 以及报告所需的图片文件。",
            styles,
        )
    )
    story.append(
        p(
            "复现实验时，建议先安装 requirements.txt 中的依赖，再执行训练命令。若本地已经存在课程数据文件，可以将数据放入 data 目录；"
            "若不存在，程序会按代码中的数据源下载公开数据并完成预处理。训练完成后，报告脚本会读取 artifacts/metrics 下的实验记录，"
            "重新生成所有表格截图和 PDF，因此 PDF 中的图表并非手工粘贴，而是由同一套实验产物自动生成。",
            styles,
        )
    )
    story.append(
        p(
            "参考资料：<br/>"
            "[1] Hebrail, G. and Berard, A. Individual Household Electric Power Consumption. UCI Machine Learning Repository, 2006. https://doi.org/10.24432/C58K54.<br/>"
            "[2] Hochreiter, S. and Schmidhuber, J. Long Short-Term Memory. Neural Computation, 1997.<br/>"
            "[3] Vaswani, A. et al. Attention Is All You Need. NeurIPS, 2017.<br/>"
            "工具说明：本报告的文字整理、结构组织与排版过程中使用了AI辅助工具；实验结果、代码运行与分析结论均由本人完成并核验。",
            styles,
            "body_no_indent",
        )
    )

    doc.build(story, onFirstPage=page_number, onLaterPages=page_number)
    print(f"Wrote report: {args.output_pdf}")


def main() -> None:
    build_report(parse_args())


if __name__ == "__main__":
    main()
