"""
helper_scripts/generate_table3_spy_summary.py

Placeholder 4 — Table 3: Summary Statistics of S&P 500 Returns Dataset
Paper section: 4.1 (Financial Dataset: S&P 500 Returns)

Loads REAL SPY data via src/data_loader.load_spy().
Reuses data/spy_returns.csv cache if already downloaded (e.g. from Table 2).

Statistics computed on RAW log-returns (before StandardScaler normalization)
to match standard academic reporting conventions.

Outputs:
    figures/table3_spy_summary_stats.png
    results/table3_spy_summary_stats.csv
"""

import os
import sys
import pathlib
import csv
import yaml
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_loader import load_spy

# ── load config ───────────────────────────────────────────────────────────────
with open(ROOT / "config.yaml") as f:
    cfg = yaml.safe_load(f)

np.random.seed(cfg["random_seed"])

DATA_DIR = ROOT / "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ── load SPY with cache ───────────────────────────────────────────────────────
cache_path = DATA_DIR / "spy_returns.csv"
fin_start  = cfg["data"]["financial"]["start"]
fin_end    = cfg["data"]["financial"]["end"]

if cache_path.exists():
    print(f"Loading SPY from cache: {cache_path}")
    spy = pd.read_csv(cache_path, index_col=0, parse_dates=True).squeeze()
else:
    print("Downloading SPY via yfinance ...")
    spy = load_spy(fin_start, fin_end)
    spy.to_csv(cache_path)
    print(f"Cached → {cache_path}")

print(f"SPY series: n={len(spy)}  [{spy.index[0].date()} → {spy.index[-1].date()}]")

# ── compute summary statistics ────────────────────────────────────────────────
n          = len(spy)
split      = int(n * cfg["train_split"])
train_size = split
test_size  = n - split

mean_r     = spy.mean()
std_r      = spy.std(ddof=1)
skew_r     = sp_stats.skew(spy)
kurt_r     = sp_stats.kurtosis(spy)          # excess kurtosis (Fisher, normal=0)
min_r      = spy.min()
max_r      = spy.max()
median_r   = spy.median()

# Jarque-Bera normality test
jb_stat, jb_p = sp_stats.jarque_bera(spy)

# Augmented Dickey-Fuller stationarity test
try:
    from statsmodels.tsa.stattools import adfuller
    adf_result = adfuller(spy, autolag="AIC")
    adf_stat   = adf_result[0]
    adf_p      = adf_result[1]
    adf_str    = f"{adf_p:.4f}"
except ImportError:
    adf_str    = "statsmodels not installed"

# ── build rows ────────────────────────────────────────────────────────────────
rows = [
    ("Ticker",                 "SPY (SPDR S&P 500 ETF Trust)"),
    ("Date Range",             f"{spy.index[0].date()}  →  {spy.index[-1].date()}"),
    ("Total Observations (N)", f"{n:,}"),
    ("Training Samples",       f"{train_size:,}  ({int(cfg['train_split']*100)}%)"),
    ("Test Samples",           f"{test_size:,}  ({100 - int(cfg['train_split']*100)}%)"),
    ("Mean Daily Return",      f"{mean_r:.6f}"),
    ("Std Deviation",          f"{std_r:.6f}"),
    ("Median",                 f"{median_r:.6f}"),
    ("Minimum",                f"{min_r:.6f}"),
    ("Maximum",                f"{max_r:.6f}"),
    ("Skewness",               f"{skew_r:.4f}"),
    ("Excess Kurtosis",        f"{kurt_r:.4f}"),
    ("Jarque-Bera Statistic",  f"{jb_stat:.4f}"),
    ("Jarque-Bera p-value",    f"{jb_p:.6f}"),
    ("ADF p-value (returns)",  adf_str),
    ("Return Type",            "Log-returns: ln(Pₜ / Pₜ₋₁)"),
    ("Normalization",          "StandardScaler (training set statistics)"),
    ("Window Size",            f"{cfg['window_size']} observations"),
]

headers = ["Statistic", "Value"]

# ── save CSV ──────────────────────────────────────────────────────────────────
results_dir = ROOT / "results"
os.makedirs(results_dir, exist_ok=True)
csv_path = results_dir / "table3_spy_summary_stats.csv"

with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(rows)

print(f"\n✅  CSV saved  →  {csv_path}")

# ── render PNG ────────────────────────────────────────────────────────────────
figures_dir = ROOT / "figures"
os.makedirs(figures_dir, exist_ok=True)
png_path = figures_dir / "table3_spy_summary_stats.png"

# Split into two columns for compact layout
mid        = len(rows) // 2 + len(rows) % 2
left_rows  = rows[:mid]
right_rows = rows[mid:]

# Pad right column if uneven
while len(right_rows) < len(left_rows):
    right_rows.append(("", ""))

combined_rows = [
    (l[0], l[1], r[0], r[1])
    for l, r in zip(left_rows, right_rows)
]
combined_headers = ["Statistic", "Value", "Statistic", "Value"]

fig, ax = plt.subplots(figsize=(16, 5.5))
ax.axis("off")

tbl = ax.table(
    cellText=combined_rows,
    colLabels=combined_headers,
    loc="center",
    cellLoc="left",
)

tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.auto_set_column_width([0, 1, 2, 3])

HEADER_COLOR = "#2C3E6B"
ROW_EVEN     = "#EEF1F8"
ROW_ODD      = "#FFFFFF"

# Highlight key statistical rows
HIGHLIGHT_STATS = {"Skewness", "Excess Kurtosis", "ADF p-value (returns)",
                   "Jarque-Bera p-value"}

for (row_idx, col_idx), cell in tbl.get_celld().items():
    cell.set_edgecolor("#CCCCCC")
    cell.set_height(0.068)
    if row_idx == 0:
        cell.set_facecolor(HEADER_COLOR)
        cell.set_text_props(color="white", fontweight="bold")
    else:
        stat_name = combined_rows[row_idx - 1][col_idx - (col_idx > 1) * 2
                                                if col_idx in (2, 3) else col_idx][0] \
                    if col_idx % 2 == 0 else ""
        row_label_left  = combined_rows[row_idx - 1][0]
        row_label_right = combined_rows[row_idx - 1][2]
        if row_label_left in HIGHLIGHT_STATS or row_label_right in HIGHLIGHT_STATS:
            cell.set_facecolor("#FFF8E1")   # soft amber for key stats
        else:
            cell.set_facecolor(ROW_EVEN if row_idx % 2 == 0 else ROW_ODD)

# Divider between left and right panels
for row_idx in range(len(combined_rows) + 1):
    cell = tbl[row_idx, 1]
    cell.set_edgecolor("#888888")

fig.suptitle(
    "Table 3 — Summary Statistics: SPY Daily Log-Returns\n"
    f"Source: Yahoo Finance via yfinance  |  Period: {fin_start} to {fin_end}",
    fontsize=11,
    fontweight="bold",
    y=1.02,
)

fig.tight_layout()
fig.savefig(png_path, dpi=180, bbox_inches="tight")
plt.close(fig)

# ── print summary to console ──────────────────────────────────────────────────
print("\n── Key Statistics ───────────────────────────────────────────────────")
for stat, val in rows:
    print(f"  {stat:<30} {val}")

# ── verify ────────────────────────────────────────────────────────────────────
print()
for p in [png_path, csv_path]:
    pp = pathlib.Path(p)
    assert pp.exists(), f"ERROR: missing {pp}"
    print(f"✅  Saved  →  {pp}  ({pp.stat().st_size:,} bytes)")