"""Align all layers to a common pixel grid (resample to SAR 10m grid)."""

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling


def get_raster_meta(filepath):
    """Read raster metadata without loading full data."""
    with rasterio.open(filepath) as src:
        return {
            "crs": src.crs,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
            "bounds": src.bounds,
        }


def reproject_to_match(source_path, target_meta, resampling=Resampling.bilinear):
    """
    Reproject a source raster to match target grid (crs, transform, shape).
    Returns reprojected array.
    """
    with rasterio.open(source_path) as src:
        source = src.read(1).astype(np.float32)
        src_crs = src.crs

    target_shape = (target_meta["height"], target_meta["width"])
    target_array = np.empty(target_shape, dtype=np.float32)

    reproject(
        source=source,
        destination=target_array,
        src_transform=src.transform if 'transform' in dir(src) else None,
        src_crs=src_crs,
        dst_transform=target_meta["transform"],
        dst_crs=target_meta["crs"],
        resampling=resampling,
    )

    return target_array


def resample_array(data, src_transform, src_crs, target_shape, target_transform, target_crs):
    """
    Resample an in-memory array to a target grid shape.
    """
    target_array = np.empty(target_shape, dtype=np.float32)

    reproject(
        source=data,
        destination=target_array,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=target_transform,
        dst_crs=target_crs,
        resampling=Resampling.bilinear,
    )

    return target_array


def align_layers(sar_data, ndwi_data, slope_data, rainfall_data,
                 sar_profile, ndwi_profile, slope_profile, rain_profile):
    """
    Align all layers to SAR grid (10m resolution, same CRS/transform).
    Returns dict of aligned arrays + the target profile.
    """
    target_profile = sar_profile.copy()
    target_shape = (target_profile["height"], target_profile["width"])
    target_transform = target_profile["transform"]
    target_crs = target_profile["crs"]

    aligned = {}

    # SAR is already on target grid
    aligned["sar"] = sar_data

    # Align NDWI
    aligned["ndwi"] = resample_array(
        ndwi_data,
        ndwi_profile["transform"], ndwi_profile["crs"],
        target_shape, target_transform, target_crs,
    )

    # Align slope
    aligned["slope"] = resample_array(
        slope_data,
        slope_profile["transform"], slope_profile["crs"],
        target_shape, target_transform, target_crs,
    )

    # Align rainfall (from 5km to 10m — significant upscale, but acceptable for MVP)
    aligned["rainfall"] = resample_array(
        rainfall_data,
        rain_profile["transform"], rain_profile["crs"],
        target_shape, target_transform, target_crs,
    )

    aligned["profile"] = target_profile
    return aligned
