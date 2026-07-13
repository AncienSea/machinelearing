from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests


UCI_ZIP_URLS = [
    (
        "https://archive.ics.uci.edu/static/public/235/"
        "individual+household+electric+power+consumption.zip"
    ),
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip",
    "http://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip",
]
RAW_ZIP_NAME = "individual_household_power_consumption.zip"
DAILY_FILE_NAME = "daily_power_features.csv"

RAW_RENAME = {
    "Global_active_power": "global_active_power",
    "Global_reactive_power": "global_reactive_power",
    "Voltage": "voltage",
    "Global_intensity": "global_intensity",
    "Sub_metering_1": "sub_metering_1",
    "Sub_metering_2": "sub_metering_2",
    "Sub_metering_3": "sub_metering_3",
}

SUM_COLUMNS = [
    "global_active_power",
    "global_reactive_power",
    "sub_metering_1",
    "sub_metering_2",
    "sub_metering_3",
    "sub_metering_remainder",
]
MEAN_COLUMNS = ["voltage", "global_intensity"]
TARGET_COLUMN = "global_active_power"
FEATURE_COLUMNS = [
    "global_active_power",
    "global_reactive_power",
    "voltage",
    "global_intensity",
    "sub_metering_1",
    "sub_metering_2",
    "sub_metering_3",
    "sub_metering_remainder",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
]


@dataclass
class SupervisedData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    test_dates: pd.Series
    feature_columns: list[str]
    target_mean: float
    target_std: float
    horizon: int
    input_len: int
    test_start_date: pd.Timestamp

    def inverse_target(self, values: np.ndarray) -> np.ndarray:
        return values * self.target_std + self.target_mean


def download_uci_zip(data_dir: Path) -> Path:
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / RAW_ZIP_NAME
    if zip_path.exists() and zip_path.stat().st_size > 0 and zipfile.is_zipfile(zip_path):
        return zip_path

    part_path = zip_path.with_suffix(zip_path.suffix + ".part")
    last_error: Exception | None = None
    for url in UCI_ZIP_URLS:
        for attempt in range(1, 4):
            print(f"Downloading UCI data to {zip_path} from {url} (attempt {attempt}/3) ...")
            if part_path.exists():
                part_path.unlink()
            try:
                with requests.get(url, stream=True, timeout=(30, None)) as response:
                    response.raise_for_status()
                    with part_path.open("wb") as fh:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                fh.write(chunk)
                if not zipfile.is_zipfile(part_path):
                    raise zipfile.BadZipFile(
                        f"Downloaded file is not a valid zip: {part_path}"
                    )
                part_path.replace(zip_path)
                return zip_path
            except Exception as exc:  # noqa: BLE001 - retry all transport/file errors.
                last_error = exc
                print(f"Download attempt failed: {exc}")
    raise RuntimeError("Could not download the UCI data zip.") from last_error


def _find_raw_text_file(zip_path: Path) -> str:
    with zipfile.ZipFile(zip_path) as zf:
        txt_members = [name for name in zf.namelist() if name.lower().endswith(".txt")]
    if not txt_members:
        raise FileNotFoundError(f"No raw .txt file found inside {zip_path}")
    return txt_members[0]


def _combine_parts(parts: Iterable[pd.DataFrame], min_count: int = 1) -> pd.DataFrame:
    frame = pd.concat(list(parts), axis=0)
    return frame.groupby(level=0).sum(min_count=min_count).sort_index()


def aggregate_daily(data_dir: Path, force: bool = False) -> pd.DataFrame:
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    daily_path = processed_dir / DAILY_FILE_NAME
    if daily_path.exists() and not force:
        return pd.read_csv(daily_path, parse_dates=["date"])

    zip_path = download_uci_zip(data_dir)
    raw_member = _find_raw_text_file(zip_path)
    sum_parts: list[pd.DataFrame] = []
    mean_sum_parts: list[pd.DataFrame] = []
    mean_count_parts: list[pd.DataFrame] = []

    with zipfile.ZipFile(zip_path) as zf, zf.open(raw_member) as raw_file:
        reader = pd.read_csv(
            raw_file,
            sep=";",
            na_values=["?"],
            chunksize=250_000,
            low_memory=False,
        )
        for chunk_id, chunk in enumerate(reader, start=1):
            chunk = chunk.rename(columns=RAW_RENAME)
            chunk["date"] = pd.to_datetime(
                chunk["Date"], format="%d/%m/%Y", errors="coerce"
            )
            for col in RAW_RENAME.values():
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
            chunk["sub_metering_remainder"] = (
                chunk["global_active_power"] * 1000.0 / 60.0
                - (
                    chunk["sub_metering_1"]
                    + chunk["sub_metering_2"]
                    + chunk["sub_metering_3"]
                )
            )
            chunk = chunk.dropna(subset=["date"])
            grouped = chunk.groupby("date")
            sum_parts.append(grouped[SUM_COLUMNS].sum(min_count=1))
            mean_sum_parts.append(grouped[MEAN_COLUMNS].sum(min_count=1))
            mean_count_parts.append(grouped[MEAN_COLUMNS].count())
            print(f"Processed raw chunk {chunk_id}")

    daily_sums = _combine_parts(sum_parts)
    mean_sums = _combine_parts(mean_sum_parts)
    mean_counts = _combine_parts(mean_count_parts, min_count=0)
    daily_means = mean_sums / mean_counts.replace(0, np.nan)
    daily = pd.concat([daily_sums, daily_means], axis=1).sort_index()

    full_index = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_index)
    daily.index.name = "date"
    daily[SUM_COLUMNS + MEAN_COLUMNS] = (
        daily[SUM_COLUMNS + MEAN_COLUMNS]
        .interpolate(method="time")
        .ffill()
        .bfill()
    )

    daily = daily.reset_index()
    daily = add_calendar_features(daily)
    daily.to_csv(daily_path, index=False)
    print(f"Wrote daily dataset: {daily_path} ({len(daily)} rows)")
    return daily


def add_calendar_features(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    date = pd.to_datetime(daily["date"])
    day_of_week = date.dt.dayofweek.to_numpy()
    month = date.dt.month.to_numpy()
    day_of_year = date.dt.dayofyear.to_numpy()
    daily["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    daily["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    daily["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    daily["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    daily["doy_sin"] = np.sin(2 * np.pi * day_of_year / 366)
    daily["doy_cos"] = np.cos(2 * np.pi * day_of_year / 366)
    return daily


def make_supervised(
    daily: pd.DataFrame,
    horizon: int,
    input_len: int = 90,
    test_days: int = 365,
    val_fraction: float = 0.2,
) -> SupervisedData:
    daily = daily.sort_values("date").reset_index(drop=True).copy()
    if len(daily) < input_len + horizon + test_days:
        raise ValueError(
            f"Dataset has only {len(daily)} days, too short for input={input_len}, "
            f"horizon={horizon}, test_days={test_days}."
        )

    test_start = len(daily) - test_days
    train_df = daily.iloc[:test_start].copy()

    feature_mean = train_df[FEATURE_COLUMNS].mean()
    feature_std = train_df[FEATURE_COLUMNS].std(ddof=0).replace(0, 1)
    target_mean = float(train_df[TARGET_COLUMN].mean())
    target_std = float(train_df[TARGET_COLUMN].std(ddof=0) or 1.0)

    scaled_features = (daily[FEATURE_COLUMNS] - feature_mean) / feature_std
    scaled_target = (daily[TARGET_COLUMN] - target_mean) / target_std

    x_windows: list[np.ndarray] = []
    y_windows: list[np.ndarray] = []
    max_start = test_start - input_len - horizon + 1
    for start in range(max_start):
        x_slice = scaled_features.iloc[start : start + input_len].to_numpy(dtype="float32")
        y_slice = scaled_target.iloc[
            start + input_len : start + input_len + horizon
        ].to_numpy(dtype="float32")
        x_windows.append(x_slice)
        y_windows.append(y_slice)

    x_all = np.stack(x_windows)
    y_all = np.stack(y_windows)
    val_size = max(1, int(round(len(x_all) * val_fraction)))
    split = len(x_all) - val_size

    x_test = scaled_features.iloc[test_start - input_len : test_start].to_numpy(
        dtype="float32"
    )[None, :, :]
    y_test = daily[TARGET_COLUMN].iloc[test_start : test_start + horizon].to_numpy(
        dtype="float32"
    )
    test_dates = daily["date"].iloc[test_start : test_start + horizon].reset_index(
        drop=True
    )

    return SupervisedData(
        x_train=x_all[:split],
        y_train=y_all[:split],
        x_val=x_all[split:],
        y_val=y_all[split:],
        x_test=x_test,
        y_test=y_test,
        test_dates=test_dates,
        feature_columns=list(FEATURE_COLUMNS),
        target_mean=target_mean,
        target_std=target_std,
        horizon=horizon,
        input_len=input_len,
        test_start_date=pd.Timestamp(daily["date"].iloc[test_start]),
    )
