"""
helper_scripts/generate_table2_diebold_mariano.py

Placeholder 3 — Table 2: Diebold-Mariano Statistics and p-values
Paper section: 3.9 (Diebold-Mariano Predictive Accuracy Test)

Uses REAL data via repo's own src/ loaders:
  - SPY daily returns        → yfinance
  - Real GDP growth          → FRED (GDPC1)
  - Industrial Production    → FRED (INDPRO)

Requires internet access for first run. Data is cached to
data/spy_returns.csv, data/gdp_growth.csv, data/indpro_growth.csv
so subsequent runs work offline.

Outputs:
    figures/table2_diebold_mariano.png
    results/table2_diebold_mariano.csv
"""

import os
import sys
import pathlib
import csv
import yaml
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── repo modules ──────────────────────────────────────────────────────────────
from src.data_loader       import load_spy, load_gdp, load_indpro
from src.preprocessing     import normalize_series, create_windows
from src.quantum_reservoir import QuantumReservoir
from src.experiment_runner import run_qrc
from src.garch_model       import run_garch
from src.metrics           import rmse, directional_accuracy, diebold_mariano

# ── load config ───────────────────────────────────────────────────────────────
with open(ROOT / "config.yaml") as f:
    cfg = yaml.safe_load(f)

np.random.seed(cfg["random_seed"])

DATA_DIR = ROOT / "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ── real data loader with local cache ─────────────────────────────────────────
def load_or_cache(name, loader_fn, cache_path):
    """Load from cache CSV if present, otherwise call loader and save cache."""
    if pathlib.Path(cache_path).exists():
        print(f"  [{name}] loading from cache: {cache_path}")
        return pd.read_csv(cache_path, index_col=0, parse_dates=True).squeeze()
    print(f"  [{name}] downloading via API ...")
    series = loader_fn()
    series.to_csv(cache_path)
    print(f"  [{name}] cached → {cache_path}")
    return series

fin_start  = cfg["data"]["financial"]["start"]
fin_end    = cfg["data"]["financial"]["end"]
mac_start  = cfg["data"]["macro"]["start"]
mac_end    = cfg["data"]["macro"]["end"]

print("\n── Loading datasets ──────────────────────────────────────────────────")
spy    = load_or_cache("SPY",    lambda: load_spy(fin_start, fin_end),
                       DATA_DIR / "spy_returns.csv")
gdp    = load_or_cache("GDP",    lambda: load_gdp(mac_start, mac_end),
                       DATA_DIR / "gdp_growth.csv")
indpro = load_or_cache("INDPRO", lambda: load_indpro(mac_start, mac_end),
                       DATA_DIR / "indpro_growth.csv")

datasets = {
    "SPY (Daily Returns)":           spy,
    "Real GDP Growth (Quarterly)":   gdp,
    "Indust. Production (Monthly)":  indpro,
}

for name, s in datasets.items():
    print(f"  {name}: n={len(s)}  [{s.index[0].date()} → {s.index[-1].date()}]")

# ── reservoir (shared across all datasets, matches config) ────────────────────
reservoir = QuantumReservoir(
    n_qubits=cfg["window_size"],
    scale_factor=cfg["scale_factor"],
)

# ── run pipeline per dataset ──────────────────────────────────────────────────
results = []

print("\n── Running QRC + GARCH pipeline ─────────────────────────────────────")
for dataset_name, series in datasets.items():
    print(f"\n  Processing: {dataset_name}  (n={len(series)})")

    split = int(len(series) * cfg["train_split"])

    # Normalize using repo's StandardScaler pipeline
    train_scaled, test_scaled = normalize_series(series, split)

    # QRC predictions
    y_test, qrc_preds = run_qrc(train_scaled, test_scaled, cfg, reservoir)

    # GARCH predictions
    garch_preds = run_garch(train_scaled, test_scaled)

    # Align lengths (GARCH rolling forecast may differ by 1)
    min_len     = min(len(y_test), len(qrc_preds), len(garch_preds))
    y_test      = y_test[:min_len]
    qrc_preds   = qrc_preds[:min_len]
    garch_preds = garch_preds[:min_len]

    # ── metrics ───────────────────────────────────────────────────────────────
    qrc_rmse   = rmse(y_test, qrc_preds)
    garch_rmse = rmse(y_test, garch_preds)
    qrc_da     = directional_accuracy(y_test, qrc_preds)
    garch_da   = directional_accuracy(y_test, garch_preds)

    # DM test: pred1=GARCH (reference), pred2=QRC (challenger)
    dm_stat, p_val = diebold_mariano(y_test, garch_preds, qrc_preds)
    sig = "Yes *" if p_val < 0.05 else "No"

    results.append({
        "Dataset":       dataset_name,
        "GARCH RMSE":    f"{garch_rmse:.5f}",
        "QRC RMSE":      f"{qrc_rmse:.5f}",
        "GARCH DA":      f"{garch_da:.3f}",
        "QRC DA":        f"{qrc_da:.3f}",
        "DM Statistic":  f"{dm_stat:.4f}",
        "p-value":       f"{p_val:.4f}",
        "Sig. at 5%?":   sig,
    })

    print(f"    GARCH RMSE={garch_rmse:.5f}  QRC RMSE={qrc_rmse:.5f}  "
          f"DM={dm_stat:.4f}  p={p_val:.4f}  Sig={sig}")

# ── save CSV ──────────────────────────────────────────────────────────────────
results_dir = ROOT / "results"
os.makedirs(results_dir, exist_ok=True)
csv_path = results_dir / "table2_diebold_mariano.csv"

fieldnames = list(results[0].keys())
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"\n✅  CSV saved  →  {csv_path}")

# ── render PNG ────────────────────────────────────────────────────────────────
figures_dir = ROOT / "figures"
os.makedirs(figures_dir, exist_ok=True)
png_path = figures_dir / "table2_diebold_mariano.png"

headers = fieldnames
rows    = [[r[h] for h in headers] for r in results]

fig, ax = plt.subplots(figsize=(15, 2.8))
ax.axis("off")

tbl = ax.table(
    cellText=rows,
    colLabels=headers,
    loc="center",
    cellLoc="center",
)

tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.auto_set_column_width(list(range(len(headers))))

HEADER_COLOR = "#2C3E6B"
ROW_EVEN     = "#EEF1F8"
ROW_ODD      = "#FFFFFF"
SIG_COLOR    = "#D4EDDA"    # green highlight = DM significant

for (row_idx, col_idx), cell in tbl.get_celld().items():
    cell.set_edgecolor("#CCCCCC")
    cell.set_height(0.22)
    if row_idx == 0:
        cell.set_facecolor(HEADER_COLOR)
        cell.set_text_props(color="white", fontweight="bold")
    else:
        row_data = rows[row_idx - 1]
        if "Yes" in row_data[-1]:
            cell.set_facecolor(SIG_COLOR)
        else:
            cell.set_facecolor(ROW_EVEN if row_idx % 2 == 0 else ROW_ODD)

fig.suptitle(
    "Table 2 — Diebold–Mariano Predictive Accuracy Test Results\n"
    "H₀: Equal predictive accuracy (GARCH vs QRC)  |  * significant at 5% level",
    fontsize=10,
    fontweight="bold",
    y=1.06,
)

fig.tight_layout()
fig.savefig(png_path, dpi=180, bbox_inches="tight")
plt.close(fig)

# ── verify ────────────────────────────────────────────────────────────────────
print()
for p in [png_path, csv_path]:
    pp = pathlib.Path(p)
    assert pp.exists(), f"ERROR: missing {pp}"
    print(f"✅  Saved  →  {pp}  ({pp.stat().st_size:,} bytes)")