"""
helper_scripts/generate_table1_hyperparameters.py

Placeholder 2 — Table 1: Optimal Hyperparameters
Paper section: 3.6 (Ridge Regression Readout Training)

Reads all values directly from config.yaml and source files.
Renders a publication-quality table as a PNG figure.

Outputs:
    figures/table1_hyperparameters.png   ← paper-insertion image
    results/table1_hyperparameters.csv   ← raw data backup
"""

import os
import sys
import pathlib
import yaml
import csv

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── load config ───────────────────────────────────────────────────────────────
with open(ROOT / "config.yaml") as f:
    config = yaml.safe_load(f)

# ── build table rows from config + hardcoded architectural choices ─────────────
rows = [
    # (Hyperparameter,              Value,                         Source)
    ("Number of qubits",            str(config["window_size"]),    "config.yaml → window_size"),
    ("Sliding-window length",       str(config["window_size"]),    "config.yaml → window_size"),
    ("Input scaling factor  S",     str(config["scale_factor"]),   "config.yaml → scale_factor"),
    ("Ridge regularization  α",     str(config["ridge_alpha"]),    "config.yaml → ridge_alpha"),
    ("Train / test split",          f"{int(config['train_split']*100)}% / "
                                    f"{100 - int(config['train_split']*100)}%",
                                                                   "config.yaml → train_split"),
    ("Random seed",                 str(config["random_seed"]),    "config.yaml → random_seed"),
    ("Entanglement topology",       "Ring CZ (periodic)",          "quantum_reservoir.py"),
    ("Readout observable",          "Pauli-Z expectation ⟨Zᵢ⟩",   "quantum_reservoir.py"),
    ("Regression method",           "Ridge (scikit-learn)",        "experiment_runner.py"),
    ("SPY data ticker",             config["data"]["financial"]["ticker"],
                                                                   "config.yaml → data.financial"),
    ("Data start date",             config["data"]["financial"]["start"],
                                                                   "config.yaml → data.financial"),
    ("Data end date",               config["data"]["financial"]["end"],
                                                                   "config.yaml → data.financial"),
]

headers = ["Hyperparameter", "Value", "Source"]

# ── save CSV ──────────────────────────────────────────────────────────────────
results_dir = ROOT / "results"
os.makedirs(results_dir, exist_ok=True)
csv_path = results_dir / "table1_hyperparameters.csv"

with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(rows)

print(f"✅  CSV  saved  →  {csv_path}")

# ── render PNG ────────────────────────────────────────────────────────────────
figures_dir = ROOT / "figures"
os.makedirs(figures_dir, exist_ok=True)
png_path = figures_dir / "table1_hyperparameters.png"

fig, ax = plt.subplots(figsize=(11, 4.2))
ax.axis("off")

tbl = ax.table(
    cellText=rows,
    colLabels=headers,
    loc="center",
    cellLoc="left",
)

tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)
tbl.auto_set_column_width([0, 1, 2])

# ── style header row ──────────────────────────────────────────────────────────
HEADER_COLOR = "#2C3E6B"   # dark navy, matches quantum/academic aesthetic
ROW_EVEN     = "#EEF1F8"
ROW_ODD      = "#FFFFFF"

for (row_idx, col_idx), cell in tbl.get_celld().items():
    cell.set_edgecolor("#CCCCCC")
    if row_idx == 0:
        cell.set_facecolor(HEADER_COLOR)
        cell.set_text_props(color="white", fontweight="bold")
    else:
        cell.set_facecolor(ROW_EVEN if row_idx % 2 == 0 else ROW_ODD)
    cell.set_height(0.075)

fig.suptitle(
    "Table 1 — QRC Framework: Optimal Hyperparameters",
    fontsize=12,
    fontweight="bold",
    y=0.97,
)

fig.tight_layout()
fig.savefig(png_path, dpi=180, bbox_inches="tight")
plt.close(fig)

# ── verify ────────────────────────────────────────────────────────────────────
for p in [png_path, csv_path]:
    assert pathlib.Path(p).exists(), f"ERROR: missing {p}"
    print(f"✅  PNG  saved  →  {p}")
    print(f"    Size : {pathlib.Path(p).stat().st_size:,} bytes")
    print(f"    Rows : {len(rows)} hyperparameters")