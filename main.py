"""Run reproducible SPY Garman-Klass variance benchmark."""

from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.data_loader import load_spy_ohlc
from src.experiment_runner import evaluate_log_variance_models
from src.garch_model import blockwise_garch_variance_forecast
from src.pipeline import prepare_spy_log_variance_frame
from src.reporting import dm_against_qrc, summarize_forecasts


def load_or_download_ohlc(config):
    financial = config["data"]["financial"]
    cache_path = Path(financial["cache_path"])
    if cache_path.exists():
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)
    ohlc = load_spy_ohlc(financial["start"], financial["end"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    ohlc.to_csv(cache_path)
    return ohlc


def save_plot(forecasts, output_path):
    ax = forecasts[["actual", "naive", "ridge", "cosine", "garch", "qrc"]].tail(252).plot(
        figsize=(11, 5), linewidth=1
    )
    ax.set_title("SPY next-day Garman-Klass variance forecasts (last 252 observations)")
    ax.set_xlabel("Feature date t; target variance realized at t+1")
    ax.set_ylabel("Daily variance")
    ax.figure.tight_layout()
    ax.figure.savefig(output_path, dpi=180)
    plt.close(ax.figure)


def main():
    with open("config.yaml", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    np.random.seed(config["random_seed"])
    random.seed(config["random_seed"])

    output_dir = Path(config["output"]["run_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    ohlc = load_or_download_ohlc(config)
    frame = prepare_spy_log_variance_frame(
        ohlc,
        lookback=config["features"]["lookback"],
        variance_floor=config["target"]["variance_floor"],
    )
    forecasts = evaluate_log_variance_models(
        frame,
        initial_train_size=config["evaluation"]["initial_train_size"],
        refit_every=config["evaluation"]["refit_every"],
        ridge_alpha=config["models"]["ridge_alpha"],
        input_scale=config["models"]["input_scale"],
        qrc_layers=config["models"]["qrc_layers"],
    )
    close_returns = np.log(ohlc["Close"] / ohlc["Close"].shift(1)).dropna()
    forecasts["garch"] = blockwise_garch_variance_forecast(
        close_returns,
        forecasts.index,
        refit_every=config["evaluation"]["refit_every"],
    )
    forecasts = forecasts[["actual", "naive", "ridge", "cosine", "garch", "qrc"]]
    summary = summarize_forecasts(forecasts)
    dm_results = dm_against_qrc(forecasts)

    forecasts.to_csv(output_dir / "forecasts.csv")
    summary.to_csv(output_dir / "metrics.csv")
    dm_results.to_csv(output_dir / "diebold_mariano.csv")
    with open(output_dir / "config_used.yaml", "w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    save_plot(forecasts, output_dir / "forecast_plot.png")

    print("Saved results to", output_dir)
    print("\nMetrics\n", summary.round(8))
    print("\nDM tests: positive statistic favors QRC\n", dm_results.round(8))


if __name__ == "__main__":
    main()
