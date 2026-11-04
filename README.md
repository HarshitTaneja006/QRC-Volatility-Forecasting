# Quantum Reservoir-Style Volatility Forecasting

[![Tests](https://github.com/HarshitTaneja006/QRC-Volatility-Forecasting/actions/workflows/tests.yml/badge.svg)](https://github.com/HarshitTaneja006/QRC-Volatility-Forecasting/actions/workflows/tests.yml)

Repository: <https://github.com/HarshitTaneja006/QRC-Volatility-Forecasting>

Reproducible benchmark for next-day SPY Garman-Klass variance forecasting. Project compares a non-commuting quantum feature map with naive persistence, ridge regression, a matched cosine-map control, and Student-t GARCH(1,1).

## Q2 submission study

`main_q2.py` runs strengthened reproducibility study used by anonymous Q2 submission manuscript. It adds:

- Nested rolling-origin hyperparameter selection inside each outer expanding forecast window.
- Seven-asset liquid ETF panel: SPY, QQQ, EFA, EEM, TLT, GLD, and FXE.
- Crisis-period QLIKE tables, Diebold-Mariano tests, and 2,000-replication moving-block bootstrap intervals.
- SPY QRC ablations (entanglement, pair observables, and depth), plus stated global-depolarizing stress results.
- Matched close-return-variance target alongside primary next-day Garman-Klass variance; this makes GARCH proxy distinction explicit.
- GDP and industrial-production magnitude robustness tests.

Run:

```powershell
python main_q2.py
```

Outputs are written to `results/q2_study/`. Submission files are in `output/q2_submission/`. Anonymous manuscript excludes author details. See `submission/SUBMISSION_CHECKLIST.md` before upload.

## What changed

Original circuit measured local Pauli-Z expectations after a CZ ring. Those measurements reduce to coordinate-wise cosine features because CZ does not change local Z expectations. New `QuantumReservoir` uses repeated angle encoding, Rx mixing, and a CNOT ring, then measures local Z and pairwise ZZ observables. Tests confirm entangling dynamics alter feature vector.

Pipeline now:

- downloads or caches SPY OHLC data for 2005-2025;
- targets next-day Garman-Klass variance;
- models log variance and exponentiates forecasts before scoring;
- evaluates all primary models on one expanding-window forecast index;
- saves forecasts, metrics, Diebold-Mariano comparisons, configuration, and a plot;
- reports GDP and industrial-production growth only as finalized-vintage robustness tests.

## Run

```powershell
python -m pip install -r requirements.txt pytest
python -m pytest -q
python main.py
```

Outputs in `results/spy_gk_2005_2025/`:

- `forecasts.csv` -- aligned realized and forecast variance series.
- `metrics.csv` -- RMSE, MAE, and QLIKE.
- `diebold_mariano.csv` -- QRC comparisons; positive statistics favor QRC.
- `forecast_plot.png` -- final 252 aligned forecasts.
- `config_used.yaml` -- experiment configuration.

## Interpretation limits

- Exact statevector simulation is not hardware evidence or quantum advantage.
- Range-based daily variance is an imperfect volatility proxy.
- Fixed hyperparameters are a reproducible benchmark, not nested time-series tuning.
- Primary results show ridge outperforming QRC; QRC improves on GARCH and matched cosine control in this configured experiment.
- GDP and industrial-production results are low-frequency robustness checks, not real-time macroeconomic forecasts.

## Layout

```text
src/
  targets.py              # Garman-Klass target timing
  preprocessing.py        # causal lag-frame construction
  quantum_reservoir.py    # non-commuting quantum feature map
  models.py               # QRC and cosine ridge adapters
  evaluation.py           # expanding-window evaluation
  reporting.py            # metrics and DM tables
  garch_model.py          # blockwise Student-t GARCH baseline
tests/                    # target, leakage, QRC, metric, and evaluation tests
```

## Manuscript

`output/manuscript/qrc_volatility_manuscript.docx` is generated from current result files by `helper_scripts/create_manuscript.py`.

## Reproducibility and release

Primary Q2 study code is versioned; generated OHLC caches, large forecast panels, and figures are intentionally ignored. Reproduce them with `python main_q2.py` after installation. Before journal submission, create a versioned GitHub release and archive its source/data package in a DOI-minting repository. Exact archive procedure and author-side fields are in [`submission/SUBMISSION_CHECKLIST.md`](submission/SUBMISSION_CHECKLIST.md).
