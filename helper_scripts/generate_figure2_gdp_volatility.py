"""
helper_scripts/generate_figure2_gdp_volatility.py

Placeholder 4 — Figure 2: GDP Growth Volatility Proxy Time Series
Paper section: 4.2 (Macroeconomic Dataset: U.S. Real GDP Growth)

Loads REAL GDP data via src/data_loader.load_gdp() from FRED (GDPC1).
Computes volatility proxy as |g_t^GDP| per Section 3.1 of the paper.

Top panel    : Raw quarterly GDP growth rate g_t^GDP
Bottom panel : Volatility proxy |g_t^GDP| used as forecasting target
Shading      : NBER recession periods (GFC 2008-09, COVID 2020)
Dashed line  : Train/test split boundary (80%)

Outputs:
    figures/figure2_gdp_volatility.png
    data/gdp_growth.csv   (cache for reuse in later scripts)
"""

import os
import sys
import pathlib
import yaml
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from src.data_loader import load_gdp

# ── load config ───────────────────────────────────────────────────────────────
with open(ROOT / "config.yaml") as f:
    cfg = yaml.safe_load(f)

np.random.seed(cfg["random_seed"])

DATA_DIR = ROOT / "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ── load GDP with cache ───────────────────────────────────────────────────────
cache_path = DATA_DIR / "gdp_growth.csv"
mac_start  = cfg["data"]["macro"]["start"]
mac_end    = cfg["data"]["macro"]["end"]

if cache_path.exists():
    print(f"Loading GDP from cache: {cache_path}")
    gdp = pd.read_csv(cache_path, index_col=0, parse_dates=True).squeeze()
else:
    print("Downloading GDPC1 from FRED via pandas_datareader ...")
    gdp = load_gdp(mac_start, mac_end)
    gdp.to_csv(cache_path)
    print(f"Cached → {cache_path}")

print(f"GDP series: n={len(gdp)}  [{gdp.index[0].date()} → {gdp.index[-1].date()}]")

# ── compute volatility proxy (Section 3.1) ────────────────────────────────────
vol_proxy = gdp.abs()   # σ_{t+1} ≈ |g_{t+1}|

# ── train/test split boundary ─────────────────────────────────────────────────
split_idx  = int(len(gdp) * cfg["train_split"])
split_date = gdp.index[split_idx]

# ── NBER recession periods relevant to the date range ────────────────────────
# Source: NBER Business Cycle Dating Committee
RECESSIONS = [
    ("2001-03-01", "2001-11-01"),   # Dot-com recession
    ("2007-12-01", "2009-06-01"),   # Great Financial Crisis
    ("2020-02-01", "2020-04-01"),   # COVID-19 recession
]

# ── plot ──────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(13, 7),
    sharex=True,
    gridspec_kw={"height_ratios": [1.4, 1]},
)
fig.subplots_adjust(hspace=0.08)

TRAIN_COLOR     = "#2C3E6B"
TEST_COLOR      = "#E07B39"
RECESSION_COLOR = "#F0E6E6"
PROXY_COLOR     = "#C0392B"

# ── Panel 1: Raw GDP growth rate ──────────────────────────────────────────────
train_gdp = gdp.iloc[:split_idx]
test_gdp  = gdp.iloc[split_idx:]

ax1.plot(train_gdp.index, train_gdp.values,
         color=TRAIN_COLOR, linewidth=1.4, label="GDP Growth (train)")
ax1.plot(test_gdp.index,  test_gdp.values,
         color=TEST_COLOR,  linewidth=1.4, label="GDP Growth (test)", linestyle="--")
ax1.axhline(0, color="grey", linewidth=0.6, linestyle=":")

for rec_start, rec_end in RECESSIONS:
    rs = pd.Timestamp(rec_start)
    re = pd.Timestamp(rec_end)
    if rs >= gdp.index[0] and re <= gdp.index[-1]:
        ax1.axvspan(rs, re, color=RECESSION_COLOR, alpha=0.7, zorder=0)

ax1.axvline(split_date, color="black", linewidth=1.2,
            linestyle="--", alpha=0.7, label=f"Train/Test Split ({split_date.date()})")

ax1.set_ylabel("GDP Growth Rate\n$g_t^{GDP}$ (%)", fontsize=10)
ax1.set_title(
    "Figure 2 — U.S. Real GDP Growth: Series and Volatility Proxy\n"
    f"FRED: GDPC1  |  Period: {mac_start} to {mac_end}  |  "
    f"$g_t^{{GDP}} = 100 \\times \\ln(Y_t / Y_{{t-4}})$",
    fontsize=11, fontweight="bold", pad=10,
)
ax1.legend(loc="lower left", fontsize=8.5, framealpha=0.9)
ax1.grid(axis="y", linewidth=0.4, alpha=0.5)
ax1.set_ylim(gdp.min() * 1.15, gdp.max() * 1.15)

# ── Panel 2: Volatility proxy |g_t^GDP| ──────────────────────────────────────
train_vol = vol_proxy.iloc[:split_idx]
test_vol  = vol_proxy.iloc[split_idx:]

ax2.fill_between(train_vol.index, train_vol.values,
                 alpha=0.35, color=TRAIN_COLOR, label="|GDP Growth| — train")
ax2.plot(train_vol.index, train_vol.values,
         color=TRAIN_COLOR, linewidth=1.2)

ax2.fill_between(test_vol.index, test_vol.values,
                 alpha=0.35, color=TEST_COLOR, label="|GDP Growth| — test")
ax2.plot(test_vol.index, test_vol.values,
         color=TEST_COLOR, linewidth=1.2, linestyle="--")

for rec_start, rec_end in RECESSIONS:
    rs = pd.Timestamp(rec_start)
    re = pd.Timestamp(rec_end)
    if rs >= gdp.index[0] and re <= gdp.index[-1]:
        ax2.axvspan(rs, re, color=RECESSION_COLOR, alpha=0.7, zorder=0)

ax2.axvline(split_date, color="black", linewidth=1.2, linestyle="--", alpha=0.7)

ax2.set_ylabel("Volatility Proxy\n$|g_t^{GDP}|$ (%)", fontsize=10)
ax2.set_xlabel("Date", fontsize=10)
ax2.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
ax2.grid(axis="y", linewidth=0.4, alpha=0.5)
ax2.set_ylim(0, vol_proxy.max() * 1.15)

# ── shared recession legend patch ─────────────────────────────────────────────
rec_patch = mpatches.Patch(color=RECESSION_COLOR, alpha=0.7, label="NBER Recession")
fig.legend(
    handles=[rec_patch],
    loc="lower center",
    ncol=1,
    fontsize=8.5,
    framealpha=0.9,
    bbox_to_anchor=(0.5, -0.02),
)

# ── save ──────────────────────────────────────────────────────────────────────
figures_dir = ROOT / "figures"
os.makedirs(figures_dir, exist_ok=True)
png_path = figures_dir / "figure2_gdp_volatility.png"

fig.savefig(png_path, dpi=180, bbox_inches="tight")
plt.close(fig)

# ── verify ────────────────────────────────────────────────────────────────────
pp = pathlib.Path(png_path)
assert pp.exists(),              f"ERROR: missing {pp}"
assert pp.stat().st_size > 5000, "ERROR: file suspiciously small"

print(f"\n── Key statistics ──────────────────────────────────────────────────")
print(f"  Total observations : {len(gdp)}")
print(f"  Train size         : {split_idx} obs up to {split_date.date()}")
print(f"  Test size          : {len(gdp) - split_idx} obs")
print(f"  Volatility proxy   : mean={vol_proxy.mean():.4f}  max={vol_proxy.max():.4f}")
print(f"\n✅  Figure 2 saved  →  {pp}  ({pp.stat().st_size:,} bytes)")