from __future__ import annotations

import argparse
from pathlib import Path

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
AUTHOR_NAME = "马鑫"
STUDENT_ID = "20255227058"


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
    candidates = [
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
    ]
    for font_path in candidates:
        if font_path.exists():
            pdfmetrics.registerFont(TTFont("CNFont", str(font_path)))
            return "CNFont"
    return "Helvetica"


def make_styles(font_name: str) -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ChineseTitle",
            parent=base["Title"],
            fontName=font_name,
            fontSize=20,
            leading=28,
            alignment=TA_CENTER,
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
            textColor=colors.HexColor("#3a3a3a"),
            spaceAfter=18,
            wordWrap="CJK",
        ),
        "heading": ParagraphStyle(
            "ChineseHeading",
            parent=base["Heading1"],
            fontName=font_name,
            fontSize=15,
            leading=21,
            spaceBefore=10,
            spaceAfter=8,
            wordWrap="CJK",
        ),
        "body": ParagraphStyle(
            "ChineseBody",
            parent=base["BodyText"],
            fontName=font_name,
            fontSize=10.2,
            leading=16,
            alignment=TA_LEFT,
            spaceAfter=7,
            firstLineIndent=18,
            wordWrap="CJK",
        ),
        "body_no_indent": ParagraphStyle(
            "ChineseBodyNoIndent",
            parent=base["BodyText"],
            fontName=font_name,
            fontSize=10.2,
            leading=16,
            alignment=TA_LEFT,
            spaceAfter=7,
            wordWrap="CJK",
        ),
        "caption": ParagraphStyle(
            "ChineseCaption",
            parent=base["BodyText"],
            fontName=font_name,
            fontSize=9,
            leading=13,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#4a4a4a"),
            spaceAfter=10,
            wordWrap="CJK",
        ),
        "code": ParagraphStyle(
            "CodeBlock",
            parent=base["Code"],
            fontName="Courier",
            fontSize=8.5,
            leading=11,
            leftIndent=12,
            rightIndent=12,
            backColor=colors.HexColor("#f4f6f8"),
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


def format_best(summary: pd.DataFrame, horizon: int) -> str:
    row = summary[summary["horizon"] == horizon].sort_values("mae_mean").iloc[0]
    return (
        f"{row['model']}（MAE={row['mae_mean']:.2f}±{row['mae_std']:.2f}，"
        f"MSE={row['mse_mean']:.2f}±{row['mse_std']:.2f}）"
    )


def make_small_table(rows: list[list[str]], font_name: str) -> Table:
    table = Table(rows, hAlign="CENTER", colWidths=[3.2 * cm, 11.6 * cm])
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), font_name),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("LEADING", (0, 0), (-1, -1), 13),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#777777")),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e9eef6")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def build_report(args: argparse.Namespace) -> None:
    metrics_dir = args.artifact_dir / "metrics"
    figure_dir = args.artifact_dir / "figures"
    summary_path = metrics_dir / "metrics_summary.csv"
    metadata_path = metrics_dir / "metadata.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing metrics summary: {summary_path}")
    if not (figure_dir / "metrics_table.png").exists():
        raise FileNotFoundError("Missing figures. Run python -m src.train first.")

    summary = pd.read_csv(summary_path)
    metadata_text = metadata_path.read_text(encoding="utf-8") if metadata_path.exists() else "{}"
    font_name = register_chinese_font()
    styles = make_styles(font_name)
    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(args.output_pdf),
        pagesize=A4,
        leftMargin=1.65 * cm,
        rightMargin=1.65 * cm,
        topMargin=1.45 * cm,
        bottomMargin=1.35 * cm,
        title="Household Power Forecasting Report",
        author=f"{STUDENT_ID}-{AUTHOR_NAME}",
    )
    story = []
    story.append(p("家庭电力消耗多变量时间序列预测", styles, "title"))
    story.append(
        p(
            f"代码仓库：{args.github_url}<br/>"
            f"作者信息：{AUTHOR_NAME}（学号：{STUDENT_ID}）。完成方式：单人完成。"
            "贡献包括数据处理、模型实现、实验运行、结果分析与报告撰写。",
            styles,
            "subtitle",
        )
    )

    story.append(p("1. 问题介绍", styles, "heading"))
    story.append(
        p(
            "本项目面向家庭用电负荷预测问题，目标是根据最近90天的历史用电曲线，"
            "预测未来90天（短期）和365天（长期）的每日总有功功率变化。该任务具有典型的多变量时间序列特征："
            "全局有功功率、无功功率、电压、电流强度和三个分表能耗共同反映家庭用电行为，"
            "而季节、周内周期和缺失观测会进一步增加预测难度。",
            styles,
        )
    )
    story.append(
        p(
            "课程文件中若未提供独立的 train.csv 与 test.csv，本实验采用严格的时间顺序划分："
            "最后365天作为测试期，之前的数据用于训练和验证；所有标准化参数仅由训练期估计，避免测试信息泄漏。"
            "分钟级原始数据按天聚合：有功功率、无功功率和分表能耗取日总和，电压和电流取日均值，"
            "并按照 UCI 数据说明计算未被三个分表覆盖的剩余能耗。",
            styles,
        )
    )
    story.append(
        make_small_table(
            [
                ["输入窗口", "过去90天的多变量日序列"],
                ["预测目标", "未来90天与365天的 global_active_power 日序列"],
                ["评价指标", "MSE 与 MAE，5个随机种子取平均值和标准差"],
                ["测试区间", "按时间排序后的最后365天"],
            ],
            font_name,
        )
    )
    story.append(Spacer(1, 8))

    story.append(p("2. 模型", styles, "heading"))
    story.append(
        p(
            "三种模型均采用直接多步预测策略，即一次性输出完整预测区间，而不是逐日递推。"
            "这样可以减少长预测中误差逐步累积的问题，也便于分别训练90天和365天两个任务的独立参数。",
            styles,
        )
    )
    story.append(
        p(
            "LSTM 模型使用一个循环编码器压缩过去90天的时间动态，再经全连接层输出未来曲线。"
            "Transformer 模型先将每日特征映射到隐藏空间，加入可学习位置编码，再使用多头注意力捕获远距离依赖。"
            "改进模型为 CNN-Transformer：先用因果一维卷积提取局部用电模式，再进入 Transformer 编码器，"
            "最后用门控池化融合平均状态和峰值状态。该结构试图同时利用局部短周期波动和长期依赖，"
            "尤其适合365天预测中季节性与突发变化并存的场景。",
            styles,
        )
    )
    story.append(
        p(
            "伪代码：",
            styles,
            "body_no_indent",
        )
    )
    story.append(
        p(
            "for horizon in [90, 365]:<br/>"
            "&nbsp;&nbsp;build supervised samples: X = past 90 days, y = next horizon days<br/>"
            "&nbsp;&nbsp;for model in [LSTM, Transformer, CNN-Transformer]:<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;for seed in [0, 1, 2, 3, 4]: train model and predict test curve<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;compute MSE, MAE, mean prediction and prediction std",
            styles,
            "code",
        )
    )

    story.append(p("3. 结果与分析", styles, "heading"))
    story.append(
        p(
            f"实验元数据如下：<font name='Courier'>{metadata_text}</font>",
            styles,
        )
    )
    story.append(
        p(
            "图1为五轮实验后得到的指标截图。90天任务的最优模型为"
            f"{format_best(summary, 90)}；365天任务的最优模型为{format_best(summary, 365)}。"
            "两个任务使用独立模型训练，因此长期任务的参数不会复用短期任务。",
            styles,
        )
    )
    story.append(image_flowable(figure_dir / "metrics_table.png", width=16.4 * cm))
    story.append(p("图1  三种模型在90天与365天预测任务上的 MSE/MAE 均值与标准差截图。", styles, "caption"))

    story.append(
        p(
            "短期预测曲线如图2所示。短期任务中，模型更容易学习最近90天与未来90天之间的平滑延续关系，"
            "因此整体趋势通常能贴近 Ground Truth。若某个模型在局部峰值处偏差较大，主要原因是直接多步输出会把峰值视为"
            "整体序列的一部分优化，而不是单独针对尖峰事件建模。",
            styles,
        )
    )
    story.append(image_flowable(figure_dir / "prediction_90d.png", width=16.5 * cm))
    story.append(p("图2  未来90天 power 预测曲线与 Ground Truth 对比截图。", styles, "caption"))
    story.append(
        p(
            "长期预测曲线如图3所示。365天任务的误差明显更大，这是因为模型必须从90天输入中外推完整年度变化，"
            "而测试期中的季节变化、节假日行为和异常用电无法完全由短窗口解释。CNN-Transformer 使用卷积先提取局部模式，"
            "再由注意力建模较长依赖，因此若其长期误差低于基础 Transformer，说明局部模式提取对长期预测有帮助；"
            "若性能不稳定，则说明模型新颖性带来的额外参数也可能增加小样本训练方差。",
            styles,
        )
    )
    story.append(image_flowable(figure_dir / "prediction_365d.png", width=16.5 * cm))
    story.append(p("图3  未来365天 power 预测曲线与 Ground Truth 对比截图。", styles, "caption"))

    story.append(p("4. 讨论", styles, "heading"))
    story.append(
        p(
            "实验表明，在只有约四年日级样本的情况下，深度模型的表达能力与样本规模之间存在明显张力。"
            "LSTM 参数较少，训练稳定；Transformer 更擅长捕获跨时间依赖，但在数据量有限时容易出现方差；"
            "CNN-Transformer 的设计动机是先利用卷积降低局部噪声，再由注意力聚合长距离信息。"
            "该改进是否持续有效，取决于训练样本数量、正则化强度和测试期分布是否与训练期一致。",
            styles,
        )
    )
    story.append(
        p(
            "本实验的局限性包括：第一，未接入外部天气数据，因此无法直接利用降雨、雾天等变量解释部分用电变化；"
            "第二，测试仅使用最后365天，虽然符合时间序列外推原则，但仍可能受到单一家庭行为变化的影响；"
            "第三，直接多步预测降低了递推误差，却可能削弱对少数极端峰值的响应。后续可引入天气变量、节假日特征、"
            "分位数损失或概率预测，并用滚动起点评估进一步检验稳定性。",
            styles,
        )
    )
    story.append(
        p(
            "参考文献与工具说明：<br/>"
            "[1] Hebrail, G. & Berard, A. (2006). Individual Household Electric Power Consumption [Dataset]. "
            "UCI Machine Learning Repository. https://doi.org/10.24432/C58K54.<br/>"
            "[2] Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS.<br/>"
            "[3] Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation.<br/>"
            "本报告的文字整理、结构组织与排版过程中使用了AI辅助工具；实验结果与分析均由本人完成并核验。",
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
