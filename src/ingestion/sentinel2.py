"""Fetch Sentinel-2 optical imagery via Google Earth Engine."""

import ee
from config.aoi_config import (
    GEE_COLLECTIONS, DATA_WINDOW, get_aoi_geometry, DATA_DIR,
)


def fetch_optical(max_cloud_pct=20, output_dir=None):
    """
    Fetch Sentinel-2 SR harmonized imagery for the AOI over the data window.
    Returns dict: {date_str: {"green": path, "nir": path, "all_bands": path}}
    """
    output_dir = output_dir or DATA_DIR / "s2"
    output_dir.mkdir(parents=True, exist_ok=True)

    aoi = get_aoi_geometry()
    collection = (
        ee.ImageCollection(GEE_COLLECTIONS["sentinel2"])
        .filterBounds(aoi)
        .filterDate(DATA_WINDOW["start"], DATA_WINDOW["end"])
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_pct))
        .select(["B3", "B8"])  # Green, NIR for NDWI
    )

    dates = collection.aggregate_array("system:time_start").getInfo()
    if not dates:
        print("WARNING: No Sentinel-2 scenes found for AOI/date range.")
        return {}

    saved = {}
    for idx, date_ms in enumerate(dates):
        date_obj = ee.Date(date_ms)
        date_str = date_obj.format("YYYY-MM-dd").getInfo()
        image = collection.filterDate(date_obj, date_obj.advance(1, "day")).first()

        out_path = output_dir / f"s2_bands_{date_str}.tif"

        try:
            import geemap
            geemap.ee_export_image(
                image.clip(aoi),
                filename=str(out_path),
                scale=10,
                region=aoi,
                file_per_band=False,
            )
            saved[date_str] = {
                "green": str(out_path),   # B3
                "nir": str(out_path),     # B8 (same file, multi-band)
                "all_bands": str(out_path),
            }
            print(f"  S2 [{idx+1}/{len(dates)}]: {date_str} -> {out_path.name}")
        except Exception as e:
            print(f"  S2 [{idx+1}/{len(dates)}]: FAILED {date_str} — {e}")

    return saved


if __name__ == "__main__":
    from ingestion.sentinel1 import init_gee
    init_gee()
    results = fetch_optical()
    print(f"\nFetched {len(results)} Sentinel-2 scenes.")
