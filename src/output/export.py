"""Output module: export risk rasters as GeoTIFF with metadata."""

import numpy as np
import rasterio
from pathlib import Path

from config.aoi_config import RISK_MAPS_DIR, SATELLITE_INFO


def save_geotiff(array, profile, output_path, tags=None):
    """
    Write a single-band array as GeoTIFF.
    tags: optional dict of metadata tags to embed.
    """
    out_profile = profile.copy()
    out_profile.update(dtype="float32", count=1)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(str(output_path), "w", **out_profile) as dst:
        dst.write(array, 1)
        if tags:
            dst.update_tags(**tags)

    return str(output_path)


def save_risk_map(risk_raster, profile, date_str, satellite_date=None):
    """
    Save a risk raster with full metadata disclosure.
    """
    if satellite_date is None:
        satellite_date = date_str

    # Compute next pass estimate
    from datetime import datetime, timedelta
    pass_date = datetime.strptime(satellite_date, "%Y-%m-%d")
    next_pass = pass_date + timedelta(days=SATELLITE_INFO["sentinel1_revisit_days"])

    tags = {
        "satellite_pass_date": satellite_date,
        "next_pass_expected": next_pass.strftime("%Y-%m-%d"),
        "region": "Periyar/Aluva, Kerala, India",
        "data_source": "Sentinel-1 SAR (VH backscatter)",
        "data_disclosure": (
            f"Data as of satellite pass {satellite_date}. "
            f"Next pass expected {next_pass.strftime('%Y-%m-%d')}. "
            f"Actual conditions may have changed."
        ),
        "risk_method": "Threshold-based composite (SAR + NDWI + terrain + rainfall)",
        "resolution_meters": 10,
    }

    output_path = RISK_MAPS_DIR / f"risk_{date_str}.tif"
    return save_geotiff(risk_raster, profile, output_path, tags=tags)


def save_risk_map_with_metadata(risk_raster, profile, date_str, output_dir=None):
    """Convenience wrapper for saving risk maps."""
    if output_dir is None:
        output_dir = RISK_MAPS_DIR
    output_path = Path(output_dir) / f"risk_{date_str}.tif"

    tags = {
        "satellite_pass_date": date_str,
        "region": "Periyar/Aluva, Kerala, India",
        "data_source": "Sentinel-1 SAR (VH backscatter)",
        "resolution_meters": 10,
    }

    return save_geotiff(risk_raster, profile, output_path, tags=tags)
