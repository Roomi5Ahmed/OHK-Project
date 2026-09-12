"""
Single source of truth for AOI geometry, date windows, thresholds, and weights.
All modules import from here. Adding Pamba = one new dict entry in AOI_REGIONS.
"""

import os
from pathlib import Path

# ── Project paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
RISK_MAPS_DIR = OUTPUT_DIR / "risk_maps"
CHARTS_DIR = OUTPUT_DIR / "validation_charts"
ALERTS_DIR = OUTPUT_DIR / "alerts"

for d in [DATA_DIR, RISK_MAPS_DIR, CHARTS_DIR, ALERTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── AOI regions ───────────────────────────────────────────────────────────────
AOI_REGIONS = {
    "periyar_aluva": {
        "name": "Periyar River Basin (Aluva/Ernakulam)",
        "bbox": {"west": 76.25, "east": 76.45, "south": 9.95, "north": 10.15},
    },
    # Day 2 bonus — uncomment to extend:
    # "pamba_kuttanad": {
    #     "name": "Pamba River Basin (Alappuzha/Kuttanad)",
    #     "bbox": {"west": 76.35, "east": 76.55, "south": 9.35, "north": 9.55},
    # },
}

ACTIVE_REGION = os.environ.get("FLOOD_AOI", "periyar_aluva")

# ── Date windows ──────────────────────────────────────────────────────────────
VALIDATION_DATES = {
    "flood_event_1": "2018-08-09",
    "flood_event_2": "2018-08-21",
}

DATA_WINDOW = {
    "start": "2018-07-15",
    "end": "2018-09-10",
}

# ── Thresholds (MVP) ─────────────────────────────────────────────────────────
THRESHOLDS = {
    "sar_vh_db": -19.5,        # Below this = water (published Kerala benchmark)
    "ndwi": 0.3,               # Above this = water
    "rainfall_3day_mm": 50,    # 3-day accumulation trigger
    "rainfall_7day_mm": 100,   # 7-day accumulation trigger
    "risk_alert": 0.7,         # High-risk flag threshold for alerts
}

# ── Risk scoring weights ──────────────────────────────────────────────────────
WEIGHTS = {
    "water_signal": 0.5,       # SAR + NDWI combined water detection
    "terrain": 0.25,           # Low elevation + low slope = higher risk
    "rainfall": 0.25,          # Recent rainfall accumulation
}

# ── Satellite metadata ────────────────────────────────────────────────────────
SATELLITE_INFO = {
    "sentinel1_revisit_days": 6,   # ~6-12 days, conservative estimate
    "sentinel2_revisit_days": 5,
}

# ── GEE collection IDs ────────────────────────────────────────────────────────
GEE_COLLECTIONS = {
    "sentinel1": "COPERNICUS/S1_GRD",
    "sentinel2": "COPERNICUS/S2_SR_HARMONIZED",
    "dem": "USGS/SRTMGL1_003",
    "rainfall": "UCSB-CHG/CHIRPS/DAILY",
}


def get_aoi_geometry():
    """Return AOI as ee.Geometry.Rectangle (requires earthengine-api)."""
    import ee
    bbox = AOI_REGIONS[ACTIVE_REGION]["bbox"]
    return ee.Geometry.Rectangle(
        [bbox["west"], bbox["south"], bbox["east"], bbox["north"]]
    )


def get_aoi_bbox():
    """Return AOI bounding box dict (no GEE dependency)."""
    return AOI_REGIONS[ACTIVE_REGION]["bbox"]


def get_region_name():
    """Return human-readable region name."""
    return AOI_REGIONS[ACTIVE_REGION]["name"]
