"""Terrain preprocessing: slope computation from SRTM DEM."""

import numpy as np
import rasterio


def compute_slope(dem_data, pixel_size=30.0):
    """
    Compute slope in degrees from DEM array.
    pixel_size: ground resolution in meters (30m for SRTM).
    Returns slope array in degrees (0-90).
    """
    # Gradient in pixel units, then convert using pixel size
    dy, dx = np.gradient(dem_data, pixel_size)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_rad).astype(np.float32)

    return slope_deg


def normalize_slope(slope_deg):
    """
    Normalize slope to 0-1 where 0 = flat (high risk), 1 = steep (low risk).
    Inverts so low slope = high susceptibility.
    """
    max_slope = 45.0  # slopes > 45° are extremely steep
    normalized = np.clip(slope_deg / max_slope, 0, 1)
    return (1 - normalized).astype(np.float32)


def preprocess_dem(filepath, output_dir=None):
    """
    Full DEM preprocessing: compute slope + normalize.
    Returns (slope_risk, elevation, profile).
    """
    from pathlib import Path

    with rasterio.open(filepath) as src:
        elevation = src.read(1).astype(np.float32)
        profile = src.profile.copy()

    slope_deg = compute_slope(elevation)
    slope_risk = normalize_slope(slope_deg)

    if output_dir:
        out_path = Path(output_dir) / "slope_risk.tif"
        out_profile = profile.copy()
        out_profile.update(dtype="float32", count=1)
        with rasterio.open(str(out_path), "w", **out_profile) as dst:
            dst.write(slope_risk, 1)

    return slope_risk, elevation, profile
