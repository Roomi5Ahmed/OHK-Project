"""Optical preprocessing: NDWI computation from Sentinel-2 bands."""

import numpy as np
import rasterio


def compute_ndwi(green_band, nir_band, output_path=None, profile=None):
    """
    Compute Normalized Difference Water Index:
        NDWI = (Green - NIR) / (Green + NIR)

    Green = Sentinel-2 B3, NIR = Sentinel-2 B8.
    Returns NDWI array (values -1 to 1).
    """
    green = green_band.astype(np.float64)
    nir = nir_band.astype(np.float64)

    denominator = green + nir
    ndwi = np.where(denominator != 0, (green - nir) / denominator, 0)
    ndwi = ndwi.astype(np.float32)

    if output_path and profile:
        out_profile = profile.copy()
        out_profile.update(dtype="float32", count=1)
        with rasterio.open(output_path, "w", **out_profile) as dst:
            dst.write(ndwi, 1)

    return ndwi


def extract_bands_from_s2(filepath):
    """
    Read multi-band Sentinel-2 file and extract Green (B3) and NIR (B8).
    Returns (green_array, nir_array, profile).
    """
    with rasterio.open(filepath) as src:
        # B3 = band 1, B8 = band 2 in the exported 2-band file
        if src.count >= 2:
            green = src.read(1).astype(np.float32)
            nir = src.read(2).astype(np.float32)
        else:
            raise ValueError(f"Expected >=2 bands in {filepath}, got {src.count}")
        profile = src.profile.copy()

    return green, nir, profile


def preprocess_optical(filepath, output_dir=None):
    """
    Full optical preprocessing: extract bands + compute NDWI.
    Returns (ndwi_array, profile).
    """
    from pathlib import Path

    green, nir, profile = extract_bands_from_s2(filepath)

    ndwi_path = None
    if output_dir:
        ndwi_path = str(Path(output_dir) / "ndwi.tif")

    ndwi = compute_ndwi(green, nir, output_path=ndwi_path, profile=profile)
    return ndwi, profile
