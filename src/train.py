from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data import aggregate_daily, make_supervised
from src.models import MODEL_BUILDERS


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run household power experiments.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--artifact-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument("--input-len", type=int, default=90)
    parser.add_argument("--horizons", type=int, nargs="+", default=[90, 365])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--force-preprocess", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def train_one_model(
    model_name: str,
    data,
    seed: int,
    epochs: int,
    batch_size: int,
) -> dict:
    tf.keras.backend.clear_session()
    set_seed(seed)
    builder = MODEL_BUILDERS[model_name]
    input_shape = data.x_train.shape[1:]
    model = builder(input_shape, data.horizon)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
            min_delta=1e-4,
        )
    ]

    start = time.time()
    history = model.fit(
        data.x_train,
        data.y_train,
        validation_data=(data.x_val, data.y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks,
    )
    train_seconds = time.time() - start
    pred_scaled = model.predict(data.x_test, verbose=0)[0]
    pred = data.inverse_target(pred_scaled)
    mse = mean_squared_error(data.y_test, pred)
    mae = mean_absolute_error(data.y_test, pred)
    return {
        "model": model_name,
        "horizon": data.horizon,
        "seed": seed,
        "mse": float(mse),
        "mae": float(mae),
        "epochs_run": len(history.history["loss"]),
        "train_seconds": float(train_seconds),
        "prediction": pred.astype("float32"),
    }


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    summary = (
        metrics.groupby(["horizon", "model"], as_index=False)
        .agg(
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            train_seconds_mean=("train_seconds", "mean"),
            epochs_mean=("epochs_run", "mean"),
        )
        .sort_values(["horizon", "model"])
    )
    return summary


def plot_predictions(predictions: pd.DataFrame, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    for horizon in sorted(predictions["horizon"].unique()):
        subset = predictions[predictions["horizon"] == horizon].copy()
        subset["date"] = pd.to_datetime(subset["date"])
        fig, ax = plt.subplots(figsize=(10, 4.6), dpi=160)
        truth = subset.drop_duplicates("date").sort_values("date")
        ax.plot(
            truth["date"],
            truth["ground_truth"],
            color="black",
            linewidth=2.0,
            label="Ground Truth",
        )
        for model_name, model_df in subset.groupby("model"):
            model_df = model_df.sort_values("date")
            ax.plot(
                model_df["date"],
                model_df["prediction_mean"],
                linewidth=1.5,
                label=model_name,
            )
        ax.set_title(f"Power Forecast vs Ground Truth ({horizon} days)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Daily global active power")
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=2, fontsize=8)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(figure_dir / f"prediction_{horizon}d.png", bbox_inches="tight")
        plt.close(fig)


def plot_metrics_table(summary: pd.DataFrame, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    table = summary.copy()
    table["MSE mean±std"] = table.apply(
        lambda r: f"{r['mse_mean']:.2f} ± {r['mse_std']:.2f}", axis=1
    )
    table["MAE mean±std"] = table.apply(
        lambda r: f"{r['mae_mean']:.2f} ± {r['mae_std']:.2f}", axis=1
    )
    table = table[["horizon", "model", "MSE mean±std", "MAE mean±std"]]

    fig_height = 0.65 + 0.42 * len(table)
    fig, ax = plt.subplots(figsize=(9.2, fig_height), dpi=180)
    ax.axis("off")
    mpl_table = ax.table(
        cellText=table.values,
        colLabels=["Horizon", "Model", "MSE mean±std", "MAE mean±std"],
        loc="center",
        cellLoc="center",
    )
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(8.5)
    mpl_table.scale(1.0, 1.35)
    for (row, col), cell in mpl_table.get_celld().items():
        cell.set_edgecolor("#4a4a4a")
        if row == 0:
            cell.set_facecolor("#e9eef6")
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f6f8fb")
    fig.tight_layout()
    fig.savefig(figure_dir / "metrics_table.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.artifact_dir / "figures"
    metrics_dir = args.artifact_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    try:
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except RuntimeError:
        pass

    daily = aggregate_daily(args.data_dir, force=args.force_preprocess)
    metadata = {
        "rows": int(len(daily)),
        "date_start": str(pd.to_datetime(daily["date"]).min().date()),
        "date_end": str(pd.to_datetime(daily["date"]).max().date()),
        "input_len": args.input_len,
        "horizons": args.horizons,
        "seeds": args.seeds,
    }

    metrics_rows: list[dict] = []
    prediction_rows: list[dict] = []
    for horizon in args.horizons:
        supervised = make_supervised(daily, horizon=horizon, input_len=args.input_len)
        print(
            f"\nHorizon {horizon}: train={len(supervised.x_train)}, "
            f"val={len(supervised.x_val)}, test_start={supervised.test_start_date.date()}"
        )
        for model_name in MODEL_BUILDERS:
            print(f"  Training {model_name} ...")
            model_predictions = []
            for seed in args.seeds:
                result = train_one_model(
                    model_name=model_name,
                    data=supervised,
                    seed=seed,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                )
                pred = result.pop("prediction")
                model_predictions.append(pred)
                metrics_rows.append(result)
                print(
                    f"    seed={seed} mse={result['mse']:.2f} "
                    f"mae={result['mae']:.2f} epochs={result['epochs_run']}"
                )

            pred_stack = np.stack(model_predictions)
            pred_mean = pred_stack.mean(axis=0)
            pred_std = pred_stack.std(axis=0)
            for idx, date in enumerate(supervised.test_dates):
                prediction_rows.append(
                    {
                        "horizon": horizon,
                        "model": model_name,
                        "date": str(pd.to_datetime(date).date()),
                        "ground_truth": float(supervised.y_test[idx]),
                        "prediction_mean": float(pred_mean[idx]),
                        "prediction_std": float(pred_std[idx]),
                    }
                )

    metrics = pd.DataFrame(metrics_rows)
    predictions = pd.DataFrame(prediction_rows)
    summary = summarize_metrics(metrics)

    metrics.to_csv(metrics_dir / "metrics_raw.csv", index=False)
    summary.to_csv(metrics_dir / "metrics_summary.csv", index=False)
    predictions.to_csv(metrics_dir / "predictions.csv", index=False)
    (metrics_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    plot_predictions(predictions, figure_dir)
    plot_metrics_table(summary, figure_dir)

    print("\nSummary:")
    print(summary.to_string(index=False))
    print(f"\nArtifacts written to {args.artifact_dir}")


if __name__ == "__main__":
    main()
