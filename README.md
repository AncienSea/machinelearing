# Household Power Forecasting Assignment

This repository contains a reproducible experiment pipeline for the 2026 machine learning course project on household electric power forecasting.

## Task

- Dataset: UCI Individual Household Electric Power Consumption.
- Preprocessing: aggregate minute-level records into daily records.
- Input window: previous 90 days.
- Forecast horizons: next 90 days and next 365 days.
- Models:
  - LSTM
  - Transformer
  - Proposed CNN-Transformer model
- Metrics: MSE and MAE.
- Stability check: 5 random seeds per model and horizon, then report mean and standard deviation.

## Quick Start

```bash
python -m src.train --epochs 8 --seeds 0 1 2 3 4
```

The command downloads the UCI zip file if needed, builds daily features, trains all models, and writes figures and metrics into `artifacts/`.

To generate the PDF report after training:

```bash
python scripts/build_report.py --github-url https://github.com/YOUR_NAME/household-power-forecasting
```

The generated PDF is written to `artifacts/report/household_power_forecasting_report.pdf`.

## Notes

- If the course-provided `train.csv` and `test.csv` files are unavailable, this code follows a chronological split: the last 365 days are used as the test period and all earlier days are used for training/validation.
- Missing minute-level values are normal in this dataset. The pipeline aggregates available records by day, reindexes the daily calendar, and fills short daily gaps by interpolation.
- The weather columns mentioned in the assignment are optional external features. This implementation uses power-consumption variables and calendar features so that it is fully reproducible from the UCI dataset alone.

## Current Results

Formal run command:

```bash
python -m src.train --epochs 8 --seeds 0 1 2 3 4 --horizons 90 365 --batch-size 64
```

| Horizon | Model | MSE mean ± std | MAE mean ± std |
|---:|---|---:|---:|
| 90 | CNN-Transformer | 190840.17 ± 11820.57 | 334.12 ± 15.39 |
| 90 | LSTM | 191204.48 ± 8489.34 | 336.64 ± 11.28 |
| 90 | Transformer | 197643.50 ± 9385.71 | 341.39 ± 10.67 |
| 365 | CNN-Transformer | 193798.21 ± 7116.17 | 333.30 ± 9.82 |
| 365 | LSTM | 187780.92 ± 3479.17 | 326.09 ± 4.98 |
| 365 | Transformer | 179280.26 ± 4341.41 | 319.12 ± 2.92 |
