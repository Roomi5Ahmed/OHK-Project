"""Generate validation chart: flood-water% with rainfall overlay."""
import sys
import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.aoi_config import DATA_DIR, OUTPUT_DIR
from src.preprocessing.sar import preprocess_sar
from src.preprocessing.alignment import resample_array
from src.scoring.threshold import sar_water_index

# --- Data collection ---
s1_dir = DATA_DIR / "s1"
rain_dir = DATA_DIR / "rainfall"

with rasterio.open(DATA_DIR / "jrc_permanent_water.tif") as src:
    jrc_raw = src.read(1).astype(np.float32)
    jrc_transform = src.transform
    jrc_crs = src.crs


def compute_rolling(target_date, days):
    dt = datetime.strptime(target_date, "%Y-%m-%d")
    acc = None
    for i in range(days):
        d = (dt - timedelta(days=i)).strftime("%Y-%m-%d")
        f = rain_dir / ("rainfall_%s.tif" % d)
        if f.exists():
            with rasterio.open(f) as src:
                daily = src.read(1).astype(np.float32)
                acc = daily if acc is None else acc + daily
    return float(acc.mean()) if acc is not None else 0.0


dates = []
flood_water_pcts = []
cnn_water_pcts = []
rain_7d = []

for f in sorted(s1_dir.glob("s1_vh_*.tif")):
    date_str = f.stem.replace("s1_vh_", "")
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    dates.append(dt)

    sar_db, sar_p = preprocess_sar(f)

    jrc = resample_array(jrc_raw, jrc_transform, jrc_crs,
                         sar_db.shape, sar_p["transform"], sar_p["crs"])
    jrc_binary = (jrc > 0.5).astype(np.float32)

    raw_water = sar_water_index(sar_db)
    flood_water = raw_water * (1 - jrc_binary)
    flood_pct = float((flood_water > 0.5).sum() / flood_water.size * 100)
    flood_water_pcts.append(flood_pct)

    cnn_path = OUTPUT_DIR / "risk_maps" / f"cnn_water_{date_str}.tif"
    if cnn_path.exists():
        with rasterio.open(cnn_path) as src:
            cnn = src.read(1)
            cnn_valid = cnn[cnn >= 0]
            cnn_pct = float((cnn_valid > 0.5).sum() / cnn_valid.size * 100) if len(cnn_valid) > 0 else 0
    else:
        cnn_pct = 0
    cnn_water_pcts.append(cnn_pct)

    r7 = compute_rolling(date_str, 7)
    rain_7d.append(r7)

    print("  %s  flood=%.2f%%  cnn=%.2f%%  rain7d=%.1fmm" % (date_str, flood_pct, cnn_pct, r7))

# --- Chart ---
baseline = flood_water_pcts[dates.index(datetime(2018, 7, 28))]
baseline_date = datetime(2018, 7, 28)

fig, ax1 = plt.subplots(figsize=(10, 5.5))

# Rainfall bars (secondary axis)
ax2 = ax1.twinx()
bar_width = 3
ax2.bar(dates, rain_7d, width=bar_width, color="#a6cee3", alpha=0.5, label="7-day rainfall (mm)", zorder=1)
ax2.set_ylabel("7-day rainfall accumulation (mm)", color="#666", fontsize=10)
ax2.tick_params(axis="y", labelcolor="#666")
ax2.set_ylim(0, max(rain_7d) * 1.3)

# Flood water line (primary axis)
ax1.plot(dates, flood_water_pcts, "o-", color="#1f78b4", linewidth=2.5, markersize=7, label="Threshold flood %", zorder=3)

# CNN water line
if any(c > 0 for c in cnn_water_pcts):
    # Normalize CNN to relative change from baseline for comparison
    cnn_baseline = cnn_water_pcts[dates.index(datetime(2018, 7, 28))]
    if cnn_baseline > 0:
        cnn_relative = [(c - cnn_baseline) / cnn_baseline * 100 for c in cnn_water_pcts]
        ax1.plot(dates, cnn_relative, "s--", color="#e31a1c", linewidth=2, markersize=6, label="CNN flood % (relative)", zorder=3)

# Baseline reference line
ax1.axhline(y=baseline, color="#999", linestyle="--", linewidth=1, zorder=2)
ax1.annotate("Baseline: %.2f%%" % baseline,
             xy=(baseline_date, baseline), xytext=(baseline_date + timedelta(days=5), baseline + 0.3),
             fontsize=8.5, color="#666",
             arrowprops=dict(arrowstyle="-", color="#ccc", lw=0.8))

# Peak flood annotation (Aug 21)
peak_date = datetime(2018, 8, 21)
peak_val = flood_water_pcts[dates.index(peak_date)]
ax1.annotate("Peak flood: %.2f%%\n(+%.0f%% vs baseline)" % (peak_val, (peak_val - baseline) / baseline * 100),
             xy=(peak_date, peak_val), xytext=(peak_date - timedelta(days=10), peak_val + 0.6),
             fontsize=8.5, color="#1f78b4", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="#1f78b4", lw=1.2))

# Jul 15-16 rain event annotation
jul16_date = datetime(2018, 7, 16)
jul16_val = flood_water_pcts[0]
ax1.annotate("Jul 15: 150mm rain\n(separate pre-monsoon event)",
             xy=(jul16_date, jul16_val), xytext=(jul16_date - timedelta(days=7), jul16_val + 0.55),
             fontsize=8, color="#e31a1c", fontstyle="italic",
             arrowprops=dict(arrowstyle="->", color="#e31a1c", lw=1))

# Flood window shading
ax1.axvspan(datetime(2018, 8, 7), datetime(2018, 8, 23), alpha=0.08, color="blue", zorder=0)
ax1.text(datetime(2018, 8, 15), ax1.get_ylim()[0] + 0.05, "Flood window",
         ha="center", fontsize=8, color="#444", fontstyle="italic")

# Formatting
ax1.set_xlabel("Date", fontsize=10)
ax1.set_ylabel("Flood water % (permanent water subtracted)", fontsize=10, color="#1f78b4")
ax1.tick_params(axis="y", labelcolor="#1f78b4")
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=7))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")

# Title and legend
fig.suptitle("Flood Risk Validation — Periyar Basin, Kerala 2018", fontsize=13, fontweight="bold", y=0.98)
ax1.set_title("Flood water fraction rises during Aug 8–21 flood window, recedes after rain stops",
              fontsize=9, color="#666", pad=8)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

# Caption
fig.text(0.5, 0.01,
         "Baseline (Jul 28): %.2f%%  |  Flood peak (Aug 21): %.2f%% (+%.0f%%)  |  "
         "CNN peak: +%.0f%% vs baseline  |  "
         "Data: Sentinel-1 VH, CHIRPS, JRC GSW" %
         (baseline, peak_val, (peak_val - baseline) / baseline * 100,
          cnn_water_pcts[dates.index(peak_date)] / cnn_water_pcts[dates.index(datetime(2018, 7, 28))] * 100 - 100 if any(c > 0 for c in cnn_water_pcts) else 0),
         ha="center", fontsize=7.5, color="#888")

plt.tight_layout(rect=[0, 0.04, 1, 0.96])

out_path = OUTPUT_DIR / "validation_charts" / "validation_flood_water.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("\nSaved: %s" % out_path)
