# Quantum Reservoir-Style Volatility Forecasting

[![Tests](https://github.com/HarshitTaneja006/QRC-Volatility-Forecasting/actions/workflows/tests.yml/badge.svg)](https://github.com/HarshitTaneja006/QRC-Volatility-Forecasting/actions/workflows/tests.yml)

Reproducible benchmark for next-day Garman-Klass (GK) variance forecasting. A non-commuting quantum feature map is compared with naive persistence, ridge regression, a matched cosine-map control, and Student-t GARCH(1,1).

## Method

The `QuantumReservoir` uses repeated angle encoding, Rx mixing, and a CNOT ring, then measures local Z and pairwise ZZ observables. A CZ-ring circuit with local Z readout collapses to coordinate-wise cosine features, so it is not used.

- Targets next-day GK variance; models log variance and exponentiates forecasts before scoring.
- All models are scored on one expanding-window forecast index.
- Metrics: RMSE, MAE, QLIKE, Diebold-Mariano tests (positive statistics favour QRC).
- GDP and industrial-production growth are used only as finalized-vintage robustness checks.

## Installation

```powershell
python -m pip install -r requirements.txt pytest
python -m pytest -q
```

## Experiments

| Entry point | Config | Purpose | Output |
|---|---|---|---|
| `python main.py` | `config.yaml` | Baseline SPY study with fixed hyperparameters | `results/spy_gk_2005_2025/` |
| `python main_q2.py` | `config_q2.yaml` | Full study for the journal submission | `results/q2_study/` |

`main_q2.py` adds:

- nested rolling-origin hyperparameter selection inside each outer expanding window;
- a seven-asset ETF panel: SPY, QQQ, EFA, EEM, TLT, GLD, FXE;
- crisis-period QLIKE tables, Diebold-Mariano tests, and 2,000-replication moving-block bootstrap intervals;
- SPY QRC ablations (entanglement, pair observables, depth) and global-depolarizing stress results;
- a close-return-variance target alongside GK variance;
- GDP and industrial-production magnitude robustness tests.

Market data are downloaded with `yfinance` and cached under `data/`; caches and `results/` are git-ignored.

## Paper figures and tables

Scripts in `helper_scripts/` regenerate the paper assets from result files (images go to `figures/`, tables to `results/`):

- `generate_figure1_circuit.py`, `generate_figure2_gdp_volatility.py`
- `generate_table1_hyperparameters.py`, `generate_table2_diebold_mariano.py`, `generate_table3_spy_summary.py`
- `create_q2_submission.py` builds the anonymous manuscript and the author-side metadata template in `output/q2_submission/`.

## Repository layout

```text
main.py, main_q2.py       # experiment entry points
config.yaml, config_q2.yaml
src/
  targets.py              # Garman-Klass target timing
  preprocessing.py        # causal lag-frame construction
  quantum_reservoir.py    # non-commuting quantum feature map
  models.py               # QRC and cosine ridge adapters
  evaluation.py           # expanding-window evaluation
  reporting.py            # metrics and DM tables
  robustness.py           # bootstrap, ablation, and macro robustness
  garch_model.py          # blockwise Student-t GARCH baseline
helper_scripts/           # figure, table, and manuscript builders
submission/               # journal submission checklist
tests/                    # target, leakage, QRC, metric, and evaluation tests
```

## Interpretation limits

- Exact statevector simulation is not hardware evidence or quantum advantage.
- Range-based daily variance is an imperfect volatility proxy.
- Ridge outperforms QRC in the primary results; QRC improves on GARCH and the matched cosine control in the configured experiment.
- GDP and industrial-production results are low-frequency robustness checks, not real-time macroeconomic forecasts.

## Submission

See [`submission/SUBMISSION_CHECKLIST.md`](submission/SUBMISSION_CHECKLIST.md). Before submission, tag a release and archive it in a DOI-minting repository (Zenodo, OSF, or Figshare).

## License

MIT. See `LICENSE`.
