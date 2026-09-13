import numpy as np
import rasterio
import streamlit as st
import sys
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
MODEL_DIR = ROOT / "models"

DATES = ['2018-07-16', '2018-07-28', '2018-08-09', '2018-08-21', '2018-08-27', '2018-09-02']
DATE_LABELS = ['Jul 16', 'Jul 28', 'Aug 9', 'Aug 21', 'Aug 27', 'Sep 2']


@st.cache_data
def get_flood_stats():
    """Compute flood water % using the same method as generate_validation_chart.py.

    Threshold: preprocess_sar → sar_water_index → JRC subtract → pixels > 0.5
    CNN: read cnn_water_*.tif → pixels > 0.5

    Returns empty results if data files are missing (cloud deployment).
    """
    s1_dir = DATA_DIR / "s1"
    rain_dir = DATA_DIR / "rainfall"
    jrc_path = DATA_DIR / "jrc_permanent_water.tif"

    # Check if required data exists
    if not s1_dir.exists() or not jrc_path.exists():
        # Return placeholder data for cloud deployment
        return [
            {'date': d, 'threshold': 0.0, 'cnn': 0.0, 'rain7d': 0.0}
            for d in DATES
        ]

    try:
        from src.preprocessing.sar import preprocess_sar
        from src.preprocessing.alignment import resample_array
        from src.scoring.threshold import sar_water_index
    except ImportError:
        # Dependencies not available (cloud deployment)
        return [
            {'date': d, 'threshold': 0.0, 'cnn': 0.0, 'rain7d': 0.0}
            for d in DATES
        ]

    with rasterio.open(jrc_path) as src:
        jrc_raw = src.read(1).astype(np.float32)
        jrc_transform = src.transform
        jrc_crs = src.crs

    results = []
    for d in DATES:
        # --- Threshold flood water % ---
        sar_path = s1_dir / f"s1_vh_{d}.tif"
        if sar_path.exists():
            sar_db, sar_p = preprocess_sar(sar_path)
            jrc = resample_array(jrc_raw, jrc_transform, jrc_crs,
                                 sar_db.shape, sar_p["transform"], sar_p["crs"])
            jrc_binary = (jrc > 0.5).astype(np.float32)
            raw_water = sar_water_index(sar_db)
            flood_water = raw_water * (1 - jrc_binary)
            t_pct = float((flood_water > 0.5).sum() / flood_water.size * 100)
        else:
            t_pct = 0.0

        # --- CNN flood water % ---
        cnn_pct = 0.0
        cnn_path = OUTPUT_DIR / "risk_maps" / f"cnn_water_{d}.tif"
        if cnn_path.exists():
            with rasterio.open(cnn_path) as src:
                cnn = src.read(1)
                cnn_valid = cnn[cnn >= 0]
                if len(cnn_valid) > 0:
                    cnn_pct = float((cnn_valid > 0.5).sum() / cnn_valid.size * 100)

        # --- 7-day rainfall ---
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
    total = 0.0
    for i in range(window):
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
