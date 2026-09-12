import numpy as np
import rasterio
import streamlit as st
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
MODEL_DIR = ROOT / "models"

DATES = ['2018-07-16', '2018-07-28', '2018-08-09', '2018-08-21', '2018-08-27', '2018-09-02']
DATE_LABELS = ['Jul 16', 'Jul 28', 'Aug 9', 'Aug 21', 'Aug 27', 'Sep 2']

def get_flood_stats():
    results = []
    for d in DATES:
        with rasterio.open(OUTPUT_DIR / "risk_maps" / f"risk_{d}.tif") as src:
            risk = src.read(1)
        t_pct = float((risk > 0.5).sum() / risk.size * 100)

        cnn_pct = 0
        cnn_path = OUTPUT_DIR / "risk_maps" / f"cnn_water_{d}.tif"
        if cnn_path.exists():
            with rasterio.open(cnn_path) as src:
                cnn = src.read(1)
                cnn_valid = cnn[cnn >= 0]
                if len(cnn_valid) > 0:
                    cnn_pct = float((cnn_valid > 0.5).sum() / cnn_valid.size * 100)

        rain7d = compute_rolling(d, 7)
        results.append({
            'date': d,
            'threshold': t_pct,
            'cnn': cnn_pct,
            'rain7d': rain7d,
        })
    return results

def compute_rolling(date_str, window):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    total = 0
    for i in range(window):
        from datetime import timedelta
        d = dt - timedelta(days=i)
        rf = DATA_DIR / "rainfall" / f"rainfall_{d.strftime('%Y-%m-%d')}.tif"
        if rf.exists():
            with rasterio.open(rf) as src:
                arr = src.read(1).astype(np.float32)
                total += float(np.nanmean(arr))
    return total

@st.cache_data
def load_raster(path):
    with rasterio.open(path) as src:
        data = src.read(1)
        bounds = src.bounds
        transform = src.transform
    return data, bounds, transform

def get_sar_image(date_str):
    path = DATA_DIR / "s1" / f"s1_vh_{date_str}.tif"
    if path.exists():
        with rasterio.open(path) as src:
            return src.read(1)
    return None
