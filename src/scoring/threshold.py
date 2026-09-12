"""MVP threshold-based risk scoring module — with permanent water subtraction."""

import numpy as np
import rasterio
from pathlib import Path

from config.aoi_config import THRESHOLDS, DATA_DIR


# Load JRC permanent water mask once (module-level cache)
_JRC_WATER = None

def _load_jrc_permanent_water(target_shape=None, target_transform=None, target_crs=None):
    """Load and optionally resample JRC permanent water mask."""
    global _JRC_WATER
    jrc_path = DATA_DIR / "jrc_permanent_water.tif"

    if not jrc_path.exists():
        return None

    with rasterio.open(jrc_path) as src:
        jrc = src.read(1).astype(np.float32)
        jrc_transform = src.transform
        jrc_crs = src.crs

    if target_shape is not None and jrc.shape != target_shape:
        from src.preprocessing.alignment import resample_array
        jrc = resample_array(jrc, jrc_transform, jrc_crs,
                             target_shape, target_transform, target_crs)
        # Binarize after resampling
        jrc = (jrc > 0.5).astype(np.float32)

    return jrc


def sar_water_index(sar_db):
    """
    Derive water probability from SAR VH backscatter using fixed threshold
    from published Kerala 2018 study (validated at ~94% accuracy).

    Maps to 0-1 probability: -14dB = 0.0, -19.5dB = 0.5, -25dB = 1.0.
    """
    water_prob = np.clip((-sar_db - 14) / 11.0, 0, 1).astype(np.float32)
    return water_prob


def water_detection(sar_db, ndwi, sar_shape=None, sar_transform=None, sar_crs=None):
    """
    Combined water detection from SAR and NDWI, with permanent water subtraction.

    Returns (flood_water_signal, has_ndwi_flag).
    flood_water_signal = SAR water pixels that are NOT permanent water.
    """
    # SAR water index
    sar_water = sar_water_index(sar_db)

    # Subtract permanent water (JRC GSW)
    jrc = _load_jrc_permanent_water(sar_shape, sar_transform, sar_crs)
    if jrc is not None:
        # Permanent water pixels get zero flood signal
        flood_water = sar_water * (1 - jrc)
    else:
        flood_water = sar_water

    # Check if NDWI has actual data
    ndwi_has_data = np.any(ndwi != 0)

    if ndwi_has_data:
        ndwi_water = np.clip(ndwi / 0.5, 0, 1)
        # Also subtract permanent water from NDWI
        if jrc is not None:
            ndwi_water = ndwi_water * (1 - jrc)
        water_signal = 0.5 * flood_water + 0.5 * ndwi_water
    else:
        water_signal = flood_water

    return water_signal, ndwi_has_data


def terrain_susceptibility(slope_risk):
    """
    Terrain risk factor: low slope = high susceptibility.
    Input is already normalized 0-1 (from terrain.normalize_slope).
    """
    return slope_risk


def rainfall_regional_bias(rainfall_3day, rainfall_7day):
    """
    Rainfall as a REGIONAL bias scalar (0-1), not per-pixel.
    Uses basin-wide mean accumulation to modulate overall risk.
    """
    mean_3d = np.mean(rainfall_3day)
    mean_7d = np.mean(rainfall_7day)

    trigger_3d = np.clip(mean_3d / (THRESHOLDS["rainfall_3day_mm"] * 2), 0, 1)
    trigger_7d = np.clip(mean_7d / (THRESHOLDS["rainfall_7day_mm"] * 2), 0, 1)

    return float(0.6 * trigger_3d + 0.4 * trigger_7d)


def compute_risk(sar_db, ndwi, slope_risk, rainfall_3day, rainfall_7day,
                 sar_shape=None, sar_transform=None, sar_crs=None):
    """
    Compute per-pixel risk score (0-1) as weighted composite.

    Components:
        - Flood water signal (SAR - permanent water, + NDWI if available): 70%
        - Terrain susceptibility (slope): 15%
        - Regional rainfall bias: 15%

    Returns risk_raster (0-1 float32).
    """
    if sar_shape is None:
        sar_shape = sar_db.shape

    # Component 1: Flood water detection (permanent water subtracted)
    water_signal, has_ndwi = water_detection(
        sar_db, ndwi,
        sar_shape=sar_shape,
        sar_transform=sar_transform,
        sar_crs=sar_crs,
    )

    # Component 2: Terrain susceptibility
    terrain = terrain_susceptibility(slope_risk)

    # Component 3: Regional rainfall bias
    rain_bias = rainfall_regional_bias(rainfall_3day, rainfall_7day)

    # Adaptive weights
    if has_ndwi:
        w_water, w_terrain, w_rain = 0.50, 0.25, 0.25
    else:
        w_water, w_terrain, w_rain = 0.70, 0.15, 0.15

    # Weighted composite
    risk = (
        w_water * water_signal
        + w_terrain * terrain
        + w_rain * rain_bias
    )

    risk = np.clip(risk, 0, 1).astype(np.float32)
    return risk


def compute_risk_from_aligned(aligned_data):
    """
    Compute risk from aligned layer dict (output of alignment.align_layers).
    Expects aligned_data with keys: sar, ndwi, slope, rainfall.
    Rainfall is split into 3-day and 7-day accumulations externally.
    """
    return compute_risk(
        sar_db=aligned_data["sar"],
        ndwi=aligned_data["ndwi"],
        slope_risk=aligned_data["slope"],
        rainfall_3day=aligned_data.get("rainfall_3d", aligned_data["rainfall"]),
        rainfall_7day=aligned_data.get("rainfall_7d", aligned_data["rainfall"]),
    )


def save_risk_raster(risk, profile, output_path, satellite_date=None, next_pass_date=None):
    """Save risk raster as GeoTIFF with metadata disclosure."""
    out_profile = profile.copy()
    out_profile.update(dtype="float32", count=1)

    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.write(risk, 1)

        tags = {}
        if satellite_date:
            tags["satellite_pass_date"] = satellite_date
        if next_pass_date:
            tags["next_pass_expected"] = next_pass_date
        tags["data_disclosure"] = (
            f"Data as of satellite pass {satellite_date}. "
            f"Next pass expected {next_pass_date}. "
            f"Actual conditions may have changed."
        )
        dst.update_tags(**tags)
