"""Run the stronger, nested-validation Q2 submission study."""

from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_gdp, load_indpro, load_spy_ohlc
from src.experiment_runner import evaluate_nested_log_variance_models
from src.garch_model import blockwise_garch_variance_forecast
from src.metrics import qlike
from src.models import QRCRegressor
from src.pipeline import (
    prepare_close_return_log_variance_frame,
    prepare_growth_proxy_log_frame,
    prepare_spy_log_variance_frame,
)
from src.quantum_reservoir import QuantumReservoir
from src.reporting import dm_against_qrc, summarize_forecasts, summarize_magnitude_forecasts
from src.robustness import (
    moving_block_bootstrap_loss_difference,
    nested_expanding_forecast,
    select_by_inner_qlike,
    summarize_crisis_periods,
)


def load_or_download_ohlc(ticker: str, config: dict) -> pd.DataFrame:
    cache_path = Path(config["data"]["cache_dir"]) / f"{ticker}.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)
    ohlc = load_spy_ohlc(
        config["data"]["start"], config["data"]["end"], ticker=ticker
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    ohlc.to_csv(cache_path)
    return ohlc


def qlike_losses(actual: pd.Series, prediction: pd.Series) -> np.ndarray:
    ratio = np.asarray(actual, dtype=float) / np.asarray(prediction, dtype=float)
    return ratio - np.log(ratio) - 1.0


def bootstrap_table(forecasts: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows = []
    for model in forecasts.columns.drop(["actual", "qrc"]):
        result = moving_block_bootstrap_loss_difference(
            qlike_losses(forecasts["actual"], forecasts[model]),
            qlike_losses(forecasts["actual"], forecasts["qrc"]),
            block_length=config["evaluation"]["bootstrap_block_length"],
            n_bootstrap=config["evaluation"]["bootstrap_repetitions"],
            seed=config["random_seed"],
        )
        rows.append({"comparator": model, **result})
    return pd.DataFrame(rows).set_index("comparator")


def selected_qrc_forecast(
    frame: pd.DataFrame,
    *,
    selections: pd.DataFrame,
    initial_train_size: int,
    refit_every: int,
    noise_probability: float = 0.0,
) -> pd.Series:
    """Re-estimate selected QRC configurations under a stated global noise level."""
    X, y = frame.drop(columns="target"), frame["target"]
    configurations = selections.reset_index(drop=True).to_dict("records")
    position = {"value": 0}

    def select_parameters(_train_x, _train_y):
        parameters = configurations[position["value"]]
        position["value"] += 1
        return {key: value for key, value in parameters.items() if key in {"alpha", "input_scale", "layers", "entangle", "include_pair_observables"}}

    def factory(parameters):
        return QRCRegressor(
            QuantumReservoir(
                X.shape[1],
                input_scale=parameters["input_scale"],
                layers=int(parameters["layers"]),
                entangle=parameters.get("entangle", True),
                include_pair_observables=parameters.get("include_pair_observables", True),
                depolarizing_probability=noise_probability,
            ),
            alpha=parameters["alpha"],
        )

    forecast, _ = nested_expanding_forecast(
        X, y,
        candidate_selector=select_parameters,
        estimator_factory=factory,
        initial_train_size=initial_train_size,
        refit_every=refit_every,
    )
    return pd.Series(np.exp(np.clip(forecast, -50, 50)), index=forecast.index)


def qrc_variant_forecast(frame: pd.DataFrame, config: dict, candidates: list[dict]) -> pd.Series:
    X, y = frame.drop(columns="target"), frame["target"]
    evaluation = config["evaluation"]

    def factory(parameters):
        return QRCRegressor(
            QuantumReservoir(
                X.shape[1],
                input_scale=parameters["input_scale"],
                layers=int(parameters["layers"]),
                entangle=parameters.get("entangle", True),
                include_pair_observables=parameters.get("include_pair_observables", True),
            ),
            alpha=parameters["alpha"],
        )

    def selector(train_x, train_y):
        return select_by_inner_qlike(
            train_x, train_y, candidates=candidates, estimator_factory=factory,
            n_splits=evaluation["inner_splits"],
            validation_size=evaluation["inner_validation_size"],
            min_train_size=evaluation["inner_validation_size"],
        )[0]

    log_forecast, _ = nested_expanding_forecast(
        X, y, candidate_selector=selector, estimator_factory=factory,
        initial_train_size=evaluation["initial_train_size"], refit_every=evaluation["refit_every"],
    )
    return pd.Series(np.exp(np.clip(log_forecast, -50, 50)), index=log_forecast.index)


def run_market_panel(config: dict, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_metrics, all_bootstrap, all_crisis = [], [], []
    primary_frame = primary_forecasts = primary_selections = None
    evaluation = config["evaluation"]
    for ticker, label in config["data"]["assets"].items():
        print(f"Running {ticker}: {label}", flush=True)
        ohlc = load_or_download_ohlc(ticker, config)
        frame = prepare_spy_log_variance_frame(
            ohlc, lookback=config["features"]["lookback"], variance_floor=config["target"]["variance_floor"]
        )
        forecasts, selections = evaluate_nested_log_variance_models(
            frame,
            initial_train_size=evaluation["initial_train_size"], refit_every=evaluation["refit_every"],
            inner_splits=evaluation["inner_splits"], inner_validation_size=evaluation["inner_validation_size"],
            ridge_candidates=config["models"]["ridge_candidates"],
            cosine_candidates=config["models"]["cosine_candidates"],
            qrc_candidates=config["models"]["qrc_candidates"],
        )
        returns = np.log(ohlc["Close"] / ohlc["Close"].shift(1)).dropna()
        forecasts["garch"] = blockwise_garch_variance_forecast(
            returns, forecasts.index, refit_every=evaluation["refit_every"]
        )
        forecasts = forecasts[["actual", "naive", "ridge", "cosine", "garch", "qrc"]]
        asset_dir = output_dir / "assets" / ticker
        asset_dir.mkdir(parents=True, exist_ok=True)
        forecasts.to_csv(asset_dir / "forecasts.csv")
        summarize_forecasts(forecasts).to_csv(asset_dir / "metrics.csv")
        dm_against_qrc(forecasts).to_csv(asset_dir / "diebold_mariano.csv")
        bootstrap_table(forecasts, config).to_csv(asset_dir / "bootstrap_qlike.csv")
        for model, selection in selections.items():
            selection.to_csv(asset_dir / f"selected_{model}_parameters.csv", index=False)
        metrics = summarize_forecasts(forecasts).reset_index(names="model")
        metrics.insert(0, "asset", ticker)
        all_metrics.append(metrics)
        bootstrap = bootstrap_table(forecasts, config).reset_index()
        bootstrap.insert(0, "asset", ticker)
        all_bootstrap.append(bootstrap)
        crisis = summarize_crisis_periods(forecasts, config["crisis_periods"]).reset_index()
        crisis.insert(0, "asset", ticker)
        all_crisis.append(crisis)
        if ticker == "SPY":
            primary_frame, primary_forecasts, primary_selections = frame, forecasts, selections
    pd.concat(all_metrics, ignore_index=True).to_csv(output_dir / "market_panel_metrics.csv", index=False)
    pd.concat(all_bootstrap, ignore_index=True).to_csv(output_dir / "market_panel_bootstrap_qlike.csv", index=False)
    pd.concat(all_crisis, ignore_index=True).to_csv(output_dir / "crisis_period_metrics.csv", index=False)
    return primary_frame, primary_forecasts, primary_selections


def run_spy_ablations_and_noise(frame: pd.DataFrame, forecasts: pd.DataFrame, selections: dict, config: dict, output_dir: Path) -> None:
    full = config["models"]["qrc_candidates"]
    variants = {
        "full_qrc": full,
        "no_entanglement": [{**candidate, "entangle": False} for candidate in full],
        "z_observables_only": [{**candidate, "include_pair_observables": False} for candidate in full],
        "one_layer": [{**candidate, "layers": 1} for candidate in full],
    }
    ablation = pd.DataFrame({"actual": forecasts["actual"]})
    for name, candidates in variants.items():
        ablation[name] = qrc_variant_forecast(frame, config, candidates)
    summarize_forecasts(ablation).to_csv(output_dir / "spy_ablation_metrics.csv")
    ablation.to_csv(output_dir / "spy_ablation_forecasts.csv")
    noise = pd.DataFrame({"actual": forecasts["actual"]})
    for probability in [0.0, 0.001, 0.005, 0.01, 0.02]:
        noise[f"p_{probability:g}"] = selected_qrc_forecast(
            frame, selections=selections["qrc"], initial_train_size=config["evaluation"]["initial_train_size"],
            refit_every=config["evaluation"]["refit_every"], noise_probability=probability,
        )
    summarize_forecasts(noise).to_csv(output_dir / "spy_global_depolarizing_noise_metrics.csv")
    noise.to_csv(output_dir / "spy_global_depolarizing_noise_forecasts.csv")


def run_matched_return_target(config: dict, output_dir: Path) -> None:
    ohlc = load_or_download_ohlc("SPY", config)
    frame = prepare_close_return_log_variance_frame(
        ohlc, lookback=config["features"]["lookback"], variance_floor=config["target"]["variance_floor"]
    )
    evaluation = config["evaluation"]
    forecasts, selections = evaluate_nested_log_variance_models(
        frame, initial_train_size=evaluation["initial_train_size"], refit_every=evaluation["refit_every"],
        inner_splits=evaluation["inner_splits"], inner_validation_size=evaluation["inner_validation_size"],
        ridge_candidates=config["models"]["ridge_candidates"], cosine_candidates=config["models"]["cosine_candidates"],
        qrc_candidates=config["models"]["qrc_candidates"],
    )
    returns = np.log(ohlc["Close"] / ohlc["Close"].shift(1)).dropna()
    forecasts["garch"] = blockwise_garch_variance_forecast(returns, forecasts.index, refit_every=evaluation["refit_every"])
    forecasts = forecasts[["actual", "naive", "ridge", "cosine", "garch", "qrc"]]
    path = output_dir / "spy_close_return_variance"
    path.mkdir(exist_ok=True)
    forecasts.to_csv(path / "forecasts.csv")
    summarize_forecasts(forecasts).to_csv(path / "metrics.csv")
    dm_against_qrc(forecasts).to_csv(path / "diebold_mariano.csv")
    bootstrap_table(forecasts, config).to_csv(path / "bootstrap_qlike.csv")
    for model, selection in selections.items():
        selection.to_csv(path / f"selected_{model}_parameters.csv", index=False)


def run_macro_robustness(config: dict, output_dir: Path) -> None:
    macro = {
        "gdp_growth": (load_gdp("2000-01-01", "2026-01-01"), 40, 4, 4),
        "indpro_growth": (load_indpro("2000-01-01", "2026-01-01"), 100, 12, 12),
    }
    records = []
    for name, (series, initial, refit, validation) in macro.items():
        frame = prepare_growth_proxy_log_frame(series, lookback=5, floor=1e-12)
        forecasts, selections = evaluate_nested_log_variance_models(
            frame, initial_train_size=initial, refit_every=refit, inner_splits=3,
            inner_validation_size=validation, ridge_candidates=config["models"]["ridge_candidates"],
            cosine_candidates=config["models"]["cosine_candidates"], qrc_candidates=config["models"]["qrc_candidates"],
        )
        metrics = summarize_magnitude_forecasts(forecasts)
        metrics.to_csv(output_dir / f"{name}_metrics.csv")
        forecasts.to_csv(output_dir / f"{name}_forecasts.csv")
        for model, selection in selections.items():
            selection.to_csv(output_dir / f"{name}_{model}_selected_parameters.csv", index=False)
        result = metrics.reset_index(names="model")
        result.insert(0, "series", name)
        records.append(result)
    pd.concat(records, ignore_index=True).to_csv(output_dir / "macro_robustness_metrics.csv", index=False)


def main() -> None:
    with open("config_q2.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    np.random.seed(config["random_seed"])
    random.seed(config["random_seed"])
    output_dir = Path(config["output"]["run_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config_used.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    frame, forecasts, selections = run_market_panel(config, output_dir)
    run_spy_ablations_and_noise(frame, forecasts, selections, config, output_dir)
    run_matched_return_target(config, output_dir)
    run_macro_robustness(config, output_dir)
    print(f"Saved Q2 study to {output_dir}")


if __name__ == "__main__":
    main()
