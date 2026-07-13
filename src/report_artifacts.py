from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


MODEL_ORDER = ["LSTM", "Transformer", "CNN-Transformer"]
MODEL_COLORS = {
    "LSTM": "#4C78A8",
    "Transformer": "#F58518",
    "CNN-Transformer": "#54A24B",
}
MODEL_FILL = {
    "LSTM": "#DCE8F7",
    "Transformer": "#FBE3C4",
    "CNN-Transformer": "#DCEFD9",
}


@dataclass
class TextBox:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


def _font_candidates(bold: bool = False) -> list[Path]:
    if bold:
        return [
            Path(r"C:\Windows\Fonts\arialbd.ttf"),
            Path(r"C:\Windows\Fonts\segoeuib.ttf"),
            Path(r"C:\Windows\Fonts\msyhbd.ttc"),
            Path(r"C:\Windows\Fonts\simhei.ttf"),
            Path(r"C:\Windows\Fonts\arial.ttf"),
        ]
    return [
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
    ]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _font_candidates(bold=bold):
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt) -> tuple[int, int]:
    if text == "":
        return (0, int(getattr(fnt, "size", 12)))
    bbox = draw.textbbox((0, 0), text, font=fnt)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def wrap_text(draw: ImageDraw.ImageDraw, text: str, fnt, max_width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph:
            lines.append("")
            continue
        words = paragraph.split(" ")
        if len(words) == 1:
            current = ""
            for char in paragraph:
                candidate = current + char
                if current and text_size(draw, candidate, fnt)[0] > max_width:
                    lines.append(current)
                    current = char
                else:
                    current = candidate
            if current:
                lines.append(current)
            continue
        current = ""
        for word in words:
            candidate = word if not current else current + " " + word
            if current and text_size(draw, candidate, fnt)[0] > max_width:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
    return lines


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    box: TextBox,
    text: str,
    fnt,
    fill: str = "#111111",
    line_gap: int = 6,
    align: str = "left",
    anchor: str = "la",
) -> int:
    lines = wrap_text(draw, text, fnt, box.width)
    ascent, descent = fnt.getmetrics() if hasattr(fnt, "getmetrics") else (12, 4)
    line_height = ascent + descent + line_gap
    y = box.top
    for line in lines:
        if not line:
            y += line_height // 2
            continue
        w, h = text_size(draw, line, fnt)
        if align == "center":
            x = box.left + (box.width - w) // 2
        elif align == "right":
            x = box.right - w
        else:
            x = box.left
        draw.text((x, y), line, font=fnt, fill=fill, anchor=anchor)
        y += line_height
    return y


def new_image(width: int, height: int, bg: str = "white") -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), bg)
    return image, ImageDraw.Draw(image)


def save_image(image: Image.Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def _draw_rounded_rect(draw, box, fill, outline, radius=18, width=2):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def _draw_title(draw, width: int, title: str, subtitle: str | None = None) -> int:
    title_font = font(34, bold=True)
    subtitle_font = font(22)
    y = 34
    tw, th = text_size(draw, title, title_font)
    draw.text(((width - tw) // 2, y), title, font=title_font, fill="#18334c")
    y += th + 12
    if subtitle:
        sw, sh = text_size(draw, subtitle, subtitle_font)
        draw.text(((width - sw) // 2, y), subtitle, font=subtitle_font, fill="#4c4c4c")
        y += sh + 12
    return y


def save_metrics_table(summary: pd.DataFrame, output_path: Path) -> Path:
    rows = []
    for _, row in summary.sort_values(["horizon", "model"]).iterrows():
        rows.append(
            [
                str(int(row["horizon"])),
                str(row["model"]),
                f"{row['mse_mean']:.2f} ± {row['mse_std']:.2f}",
                f"{row['mae_mean']:.2f} ± {row['mae_std']:.2f}",
                f"{row['train_seconds_mean']:.2f}",
            ]
        )
    image, draw = new_image(1750, 250 + 82 * (len(rows) + 1), "white")
    top = _draw_title(draw, image.width, "Metrics Summary", "Mean ± standard deviation over 5 random seeds")
    left = 80
    top += 20
    col_widths = [150, 340, 360, 360, 220]
    header = ["Horizon", "Model", "MSE mean ± std", "MAE mean ± std", "Train s"]
    f_head = font(26, bold=True)
    f_body = font(24)
    row_h = 78
    x = left
    for idx, h in enumerate(header):
        fill = "#E3EDF6"
        _draw_rounded_rect(draw, (x, top, x + col_widths[idx], top + row_h), fill, "#6C7A89", radius=12, width=2)
        bbox = TextBox(x + 12, top + 18, x + col_widths[idx] - 12, top + row_h - 12)
        draw_wrapped_text(draw, bbox, h, f_head, fill="#18334c", align="center")
        x += col_widths[idx] + 10
    y = top + row_h + 10
    for ridx, row in enumerate(rows):
        x = left
        fill = "#F7FAFC" if ridx % 2 else "#FFFFFF"
        for cidx, value in enumerate(row):
            _draw_rounded_rect(draw, (x, y, x + col_widths[cidx], y + row_h), fill, "#B8C2CC", radius=10, width=1)
            bbox = TextBox(x + 12, y + 14, x + col_widths[cidx] - 12, y + row_h - 12)
            align = "center" if cidx == 0 else "left"
            draw_wrapped_text(draw, bbox, value, f_body, fill="#111111", align=align)
            x += col_widths[cidx] + 10
        y += row_h + 10
    return save_image(image, output_path)


def _value_range(values: np.ndarray, pad_ratio: float = 0.08) -> tuple[float, float]:
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    pad = (hi - lo) * pad_ratio if hi > lo else max(abs(hi), 1.0) * pad_ratio + 1.0
    return lo - pad, hi + pad


def save_prediction_chart(predictions: pd.DataFrame, horizon: int, output_path: Path) -> Path:
    subset = predictions[predictions["horizon"] == horizon].copy()
    subset["date"] = pd.to_datetime(subset["date"])
    truth = subset.drop_duplicates("date").sort_values("date")
    dates = truth["date"].to_list()
    n = len(dates)
    width, height = 1760, 880
    image, draw = new_image(width, height, "white")
    title_y = _draw_title(draw, width, f"Forecast vs Ground Truth ({horizon} days)", "Daily global active power")
    legend_font = font(20)
    axis_font = font(18)
    label_font = font(20, bold=True)
    left, right, top, bottom = 120, 1600, title_y + 40, 760
    draw.rectangle((left, top, right, bottom), outline="#4D5E6E", width=2)
    y_values = [truth["ground_truth"].to_numpy()]
    for model in MODEL_ORDER:
        y_values.append(subset[subset["model"] == model]["prediction_mean"].sort_values().to_numpy())
    y_min, y_max = _value_range(np.concatenate(y_values))

    def x_for(idx: int) -> int:
        if n <= 1:
            return left
        return int(left + idx * (right - left) / (n - 1))

    def y_for(value: float) -> int:
        return int(bottom - (value - y_min) * (bottom - top) / (y_max - y_min))

    for frac in np.linspace(0, 1, 6):
        y = int(bottom - frac * (bottom - top))
        draw.line((left, y, right, y), fill="#E2E8F0", width=1)
        v = y_min + frac * (y_max - y_min)
        txt = f"{v:.0f}"
        tw, th = text_size(draw, txt, axis_font)
        draw.text((left - 18 - tw, y - th // 2), txt, font=axis_font, fill="#475569")

    xticks = max(3, min(7, n // 50 + 1))
    tick_indices = np.unique(np.linspace(0, n - 1, xticks).astype(int))
    for idx in tick_indices:
        x = x_for(int(idx))
        draw.line((x, bottom, x, bottom + 8), fill="#475569", width=2)
        label = dates[int(idx)].strftime("%Y-%m-%d")
        tw, th = text_size(draw, label, axis_font)
        draw.text((x - tw // 2, bottom + 12), label, font=axis_font, fill="#475569")

    draw.text((left, bottom + 52), "Date", font=label_font, fill="#1f2937")
    draw.text((10, top + (bottom - top) // 2), "Daily global active power", font=label_font, fill="#1f2937")

    # Ground truth
    truth_points = [(x_for(i), y_for(v)) for i, v in enumerate(truth["ground_truth"].to_numpy())]
    draw.line(truth_points, fill="#111111", width=5)

    for model in MODEL_ORDER:
        model_df = subset[subset["model"] == model].sort_values("date")
        points = [(x_for(i), y_for(v)) for i, v in enumerate(model_df["prediction_mean"].to_numpy())]
        draw.line(points, fill=MODEL_COLORS[model], width=4)

    legend_x = right - 380
    legend_y = top + 24
    items = [("Ground Truth", "#111111")] + [(model, MODEL_COLORS[model]) for model in MODEL_ORDER]
    for name, color in items:
        draw.rectangle((legend_x, legend_y + 7, legend_x + 26, legend_y + 23), fill=color, outline=color)
        draw.text((legend_x + 36, legend_y), name, font=legend_font, fill="#1f2937")
        legend_y += 34

    return save_image(image, output_path)


def save_mae_comparison_chart(summary: pd.DataFrame, output_path: Path) -> Path:
    image, draw = new_image(1760, 920, "white")
    title_y = _draw_title(draw, image.width, "Cross-model MAE Comparison", "Error bars indicate 5-seed standard deviation")
    left, right, top, bottom = 130, 1600, title_y + 45, 760
    draw.rectangle((left, top, right, bottom), outline="#4D5E6E", width=2)
    rows = sorted(summary["horizon"].unique())
    max_val = float((summary["mae_mean"] + summary["mae_std"]).max()) * 1.12

    def x_for(group_idx: float) -> int:
        return int(left + group_idx * (right - left) / max(len(rows), 1))

    def y_for(value: float) -> int:
        return int(bottom - value * (bottom - top) / max_val)

    # grid
    for frac in np.linspace(0, 1, 6):
        y = int(bottom - frac * (bottom - top))
        draw.line((left, y, right, y), fill="#E2E8F0", width=1)
        v = max_val * frac
        txt = f"{v:.0f}"
        tw, th = text_size(draw, txt, font(18))
        draw.text((left - 16 - tw, y - th // 2), txt, font=font(18), fill="#475569")

    group_w = (right - left) / max(len(rows), 1)
    bar_w = group_w * 0.18
    label_font = font(18, bold=True)
    legend_font = font(20)
    for g_idx, horizon in enumerate(rows):
        center = x_for(g_idx + 0.5)
        for m_idx, model in enumerate(MODEL_ORDER):
            row = summary[(summary["horizon"] == horizon) & (summary["model"] == model)].iloc[0]
            x = center + (m_idx - 1) * (bar_w * 1.7)
            y = y_for(float(row["mae_mean"]))
            y0 = y_for(0)
            draw.rectangle((x, y, x + bar_w, y0), fill=MODEL_COLORS[model], outline=MODEL_COLORS[model])
            err_top = y_for(float(row["mae_mean"] + row["mae_std"]))
            err_bottom = y_for(float(row["mae_mean"] - row["mae_std"]))
            cx = x + bar_w / 2
            draw.line((cx, err_top, cx, err_bottom), fill="#1F2937", width=3)
            draw.line((cx - 10, err_top, cx + 10, err_top), fill="#1F2937", width=3)
            draw.line((cx - 10, err_bottom, cx + 10, err_bottom), fill="#1F2937", width=3)
        label = f"{horizon} days"
        tw, th = text_size(draw, label, label_font)
        draw.text((center - tw / 2, bottom + 16), label, font=label_font, fill="#1f2937")

    draw.text((left, bottom + 52), "Forecast horizon", font=label_font, fill="#1f2937")
    draw.text((10, top + (bottom - top) // 2), "MAE (lower is better)", font=label_font, fill="#1f2937")
    legend_x = right - 420
    legend_y = top + 20
    for name, color in [(m, MODEL_COLORS[m]) for m in MODEL_ORDER]:
        draw.rectangle((legend_x, legend_y + 7, legend_x + 26, legend_y + 23), fill=color, outline=color)
        draw.text((legend_x + 36, legend_y), name, font=legend_font, fill="#1f2937")
        legend_y += 34
    return save_image(image, output_path)


def _boxplot_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    q1 = float(np.percentile(values, 25))
    med = float(np.percentile(values, 50))
    q3 = float(np.percentile(values, 75))
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    within = values[(values >= lower_fence) & (values <= upper_fence)]
    low = float(within.min()) if len(within) else float(values.min())
    high = float(within.max()) if len(within) else float(values.max())
    return {"q1": q1, "med": med, "q3": q3, "low": low, "high": high}


def _save_boxplot(data_frame: pd.DataFrame, value_column: str, title: str, y_label: str, output_path: Path) -> Path:
    image, draw = new_image(1760, 930, "white")
    title_y = _draw_title(draw, image.width, title, "Grouped by horizon and model")
    left, right, top, bottom = 140, 1600, title_y + 50, 760
    draw.rectangle((left, top, right, bottom), outline="#4D5E6E", width=2)
    groups = []
    for horizon in sorted(data_frame["horizon"].unique()):
        for model in MODEL_ORDER:
            subset = data_frame[(data_frame["horizon"] == horizon) & (data_frame["model"] == model)]
            groups.append((f"{horizon}d\n{model}", subset[value_column].to_numpy(), MODEL_COLORS[model]))

    all_values = np.concatenate([g[1] for g in groups if len(g[1])])
    y_min, y_max = _value_range(all_values)

    def y_for(value: float) -> int:
        return int(bottom - (value - y_min) * (bottom - top) / (y_max - y_min))

    # grid and labels
    axis_font = font(18)
    label_font = font(18, bold=True)
    for frac in np.linspace(0, 1, 6):
        y = int(bottom - frac * (bottom - top))
        draw.line((left, y, right, y), fill="#E2E8F0", width=1)
        v = y_min + frac * (y_max - y_min)
        txt = f"{v:.0f}"
        tw, th = text_size(draw, txt, axis_font)
        draw.text((left - 18 - tw, y - th // 2), txt, font=axis_font, fill="#475569")

    group_xs = np.linspace(left + 90, right - 90, len(groups))
    for (label, values, color), x_center in zip(groups, group_xs):
        stats = _boxplot_stats(values)
        box_w = 34
        x0, x1 = int(x_center - box_w), int(x_center + box_w)
        draw.line((x_center, y_for(stats["low"]), x_center, y_for(stats["q1"])), fill="#1F2937", width=3)
        draw.line((x_center, y_for(stats["q3"]), x_center, y_for(stats["high"])), fill="#1F2937", width=3)
        draw.line((x0 - 10, y_for(stats["low"]), x1 + 10, y_for(stats["low"])), fill="#1F2937", width=3)
        draw.line((x0 - 10, y_for(stats["high"]), x1 + 10, y_for(stats["high"])), fill="#1F2937", width=3)
        draw.rectangle((x0, y_for(stats["q3"]), x1, y_for(stats["q1"])), fill=color, outline="#1F2937", width=2)
        draw.line((x0, y_for(stats["med"]), x1, y_for(stats["med"])), fill="#111111", width=3)
        for v in values:
            jitter = int((hash((float(v), label)) % 21) - 10)
            draw.ellipse((x_center + jitter - 3, y_for(float(v)) - 3, x_center + jitter + 3, y_for(float(v)) + 3), fill="#1F2937")
        tw, th = text_size(draw, label, label_font)
        draw.text((x_center - tw / 2, bottom + 14), label, font=label_font, fill="#1f2937")

    draw.text((left, bottom + 52), "Horizon and model", font=label_font, fill="#1f2937")
    draw.text((12, top + (bottom - top) // 2), y_label, font=label_font, fill="#1f2937")
    return save_image(image, output_path)


def save_seed_robustness_boxplot(raw: pd.DataFrame, output_path: Path) -> Path:
    return _save_boxplot(
        raw,
        "mae",
        "Robustness Across Five Random Seeds",
        "MAE by random seed",
        output_path,
    )


def save_error_distribution_boxplot(predictions: pd.DataFrame, output_path: Path) -> Path:
    frame = predictions.copy()
    frame["abs_error"] = (frame["prediction_mean"] - frame["ground_truth"]).abs()
    return _save_boxplot(
        frame,
        "abs_error",
        "Daily Absolute Error Distribution",
        "Absolute error of mean prediction",
        output_path,
    )


def save_timeline(metadata: dict, predictions: pd.DataFrame, output_path: Path) -> Path:
    start = pd.to_datetime(metadata["date_start"])
    end = pd.to_datetime(metadata["date_end"])
    test_start = pd.to_datetime(predictions["date"].min())
    width, height = 1760, 420
    image, draw = new_image(width, height, "white")
    _draw_title(draw, width, "Chronological Split Protocol", "Last 365 days reserved for testing")
    left, right, y = 140, 1600, 210
    draw.line((left, y, right, y), fill="#1F2937", width=4)

    def x_for(date):
        return int(left + (pd.to_datetime(date) - start) / (end - start) * (right - left))

    x_test = x_for(test_start)
    draw.rectangle((left, y - 30, x_test, y + 30), fill="#4C78A8")
    draw.rectangle((x_test, y - 30, right, y + 30), fill="#F58518")
    draw.line((x_test, y - 44, x_test, y + 44), fill="#111111", width=4)
    axis_font = font(20)
    label_font = font(22, bold=True)
    for date in [start, test_start, end]:
        x = x_for(date)
        draw.line((x, y + 38, x, y + 52), fill="#1F2937", width=3)
        label = pd.to_datetime(date).strftime("%Y-%m-%d")
        tw, th = text_size(draw, label, axis_font)
        draw.text((x - tw / 2, y + 58), label, font=axis_font, fill="#475569")
    draw.text((left, y - 78), "Training + validation period", font=label_font, fill="#18334c")
    draw.text((x_test + 20, y - 78), "Test period", font=label_font, fill="#18334c")
    return save_image(image, output_path)


def save_architecture_diagram(output_path: Path) -> Path:
    image, draw = new_image(1760, 520, "white")
    _draw_title(draw, image.width, "CNN-Transformer Architecture", "Short-term convolution + long-range attention")
    boxes = [
        ("90-day multivariate input", 80, 200, 280, 100, "#E8F1F6"),
        ("Causal Conv1D\nlocal pattern extractor", 410, 200, 280, 100, "#EEF6EA"),
        ("Position embedding\n+ Transformer encoder", 740, 200, 300, 100, "#FBECD7"),
        ("Gated pooling\nmean + max fusion", 1090, 200, 260, 100, "#E8F1F6"),
        ("Dense head\n90 / 365-day output", 1410, 200, 270, 100, "#EEF6EA"),
    ]
    for text, x, y, w, h, fill in boxes:
        _draw_rounded_rect(draw, (x, y, x + w, y + h), fill, "#6C7A89", radius=18, width=2)
        bbox = TextBox(x + 12, y + 16, x + w - 12, y + h - 12)
        draw_wrapped_text(draw, bbox, text, font(24, bold=True), fill="#18334c", align="center")
    for i in range(len(boxes) - 1):
        x1 = boxes[i][1] + boxes[i][3]
        x2 = boxes[i + 1][1]
        y = boxes[i][2] + boxes[i][4] // 2
        draw.line((x1 + 8, y, x2 - 8, y), fill="#18334c", width=4)
        draw.polygon([(x2 - 8, y), (x2 - 26, y - 10), (x2 - 26, y + 10)], fill="#18334c")
    note = (
        "The model first extracts short-term local patterns with causal convolutions, "
        "then models long-range dependence with self-attention, and finally fuses pooled representations for direct multi-step forecasting."
    )
    draw_wrapped_text(draw, TextBox(180, 350, 1580, 470), note, font(21), fill="#3B3B3B", align="center")
    return save_image(image, output_path)

