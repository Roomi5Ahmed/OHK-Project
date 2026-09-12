"""SAR preprocessing: dB conversion and optional Lee speckle filtering."""

import numpy as np
import rasterio


def to_db(filepath, output_path=None):
    """Convert Sentinel-1 linear power image to dB scale.
    Skips conversion if data is already in dB (has negative values)."""
    with rasterio.open(filepath) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile.copy()

    # Check if already in dB (S1_GRD from GEE exports in dB)
    if data.min() < 0:
        data_db = data  # Already in dB
    else:
        data = np.where(data <= 0, 1e-10, data)
        data_db = 10.0 * np.log10(data)

    if output_path:
        profile.update(dtype="float32")
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data_db, 1)

    return data_db, profile


def lee_filter(data, size=5):
    """
    Apply Lee speckle filter to SAR image.
    Uses local statistics to smooth noise while preserving edges.
    """
    from scipy.ndimage import uniform_filter

    data = data.astype(np.float64)
    window = uniform_filter(data, size=size)
    window_sq = uniform_filter(data**2, size=size)

    variance = window_sq - window**2
    overall_var = np.var(data)

    # Weight: high variance -> keep original, low variance -> use smoothed
    weight = np.where(variance > 0, variance / (variance + overall_var), 0)
    filtered = weight * data + (1 - weight) * window

    return filtered.astype(np.float32)


def preprocess_sar(filepath, output_dir=None, apply_filter=True, filter_size=5):
    """
    Full SAR preprocessing pipeline: dB conversion + optional Lee filter.
    Returns (processed_array, profile).
    """
    from pathlib import Path

    db_path = str(Path(output_dir or Path(filepath).parent) / "processed_db.tif") if output_dir else None
    data_db, profile = to_db(filepath, output_path=db_path)

    if apply_filter:
        data_db = lee_filter(data_db, size=filter_size)
        if db_path:
            profile.update(dtype="float32")
            with rasterio.open(db_path, "w", **profile) as dst:
                dst.write(data_db, 1)

    return data_db, profile
