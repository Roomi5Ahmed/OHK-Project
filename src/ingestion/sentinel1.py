"""Fetch Sentinel-1 SAR (GRD) imagery via Google Earth Engine."""

import ee
import os
from pathlib import Path
from config.aoi_config import (
    GEE_COLLECTIONS, DATA_WINDOW, get_aoi_geometry, DATA_DIR,
)

GEE_PROJECT_ID = os.environ.get("GEE_PROJECT", "midyear-byway-508408-d6")


def init_gee():
    """Initialize Earth Engine session. Call once at pipeline start."""
    try:
        ee.Initialize(project=GEE_PROJECT_ID, opt_url="https://earthengine-highvolume.googleapis.com")
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=GEE_PROJECT_ID, opt_url="https://earthengine-highvolume.googleapis.com")


def fetch_sar(output_dir=None):
    """
    Fetch Sentinel-1 GRD VH imagery for the AOI over the data window.
    Returns dict: {date_str: filepath_to_geotiff}
    """
    output_dir = output_dir or DATA_DIR / "s1"
    output_dir.mkdir(parents=True, exist_ok=True)

    aoi = get_aoi_geometry()
    collection = (
        ee.ImageCollection(GEE_COLLECTIONS["sentinel1"])
        .filterBounds(aoi)
        .filterDate(DATA_WINDOW["start"], DATA_WINDOW["end"])
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
        .select("VH")
    )

    dates = collection.aggregate_array("system:time_start").getInfo()
    if not dates:
        print("WARNING: No Sentinel-1 scenes found for AOI/date range.")
        return {}

    saved = {}
    for idx, date_ms in enumerate(dates):
        date_obj = ee.Date(date_ms)
        date_str = date_obj.format("YYYY-MM-dd").getInfo()
        image = collection.filterDate(date_obj, date_obj.advance(1, "day")).first()

        out_path = output_dir / f"s1_vh_{date_str}.tif"

        # Skip if already downloaded
        if out_path.exists() and out_path.stat().st_size > 0:
            saved[date_str] = str(out_path)
            print(f"  S1 [{idx+1}/{len(dates)}]: {date_str} -> cached")
            continue

        # Export to Google Drive then download, or use geemap direct export
        try:
            import geemap
            geemap.ee_export_image(
                image.clip(aoi),
                filename=str(out_path),
                scale=10,
                region=aoi,
                file_per_band=False,
            )
            saved[date_str] = str(out_path)
            print(f"  S1 [{idx+1}/{len(dates)}]: {date_str} -> {out_path.name}")
        except Exception as e:
            print(f"  S1 [{idx+1}/{len(dates)}]: FAILED {date_str} -- {e}")

    return saved


if __name__ == "__main__":
    init_gee()
    results = fetch_sar()
    print(f"\nFetched {len(results)} Sentinel-1 scenes.")
