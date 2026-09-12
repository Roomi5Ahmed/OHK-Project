"""Fetch CHIRPS daily rainfall data via Google Earth Engine."""

import ee
from config.aoi_config import GEE_COLLECTIONS, DATA_WINDOW, get_aoi_geometry, DATA_DIR


def fetch_rainfall(output_dir=None):
    """
    Fetch CHIRPS daily rainfall for the AOI over the data window.
    Returns dict: {date_str: filepath_to_geotiff}
    """
    output_dir = output_dir or DATA_DIR / "rainfall"
    output_dir.mkdir(parents=True, exist_ok=True)

    aoi = get_aoi_geometry()
    collection = (
        ee.ImageCollection(GEE_COLLECTIONS["rainfall"])
        .filterBounds(aoi)
        .filterDate(DATA_WINDOW["start"], DATA_WINDOW["end"])
        .select("precipitation")
    )

    dates = collection.aggregate_array("system:time_start").getInfo()
    if not dates:
        print("WARNING: No CHIRPS rainfall data found for AOI/date range.")
        return {}

    saved = {}
    for idx, date_ms in enumerate(dates):
        date_obj = ee.Date(date_ms)
        date_str = date_obj.format("YYYY-MM-dd").getInfo()
        image = collection.filterDate(date_obj, date_obj.advance(1, "day")).first()

        out_path = output_dir / f"rainfall_{date_str}.tif"

        # Skip if already downloaded
        if out_path.exists() and out_path.stat().st_size > 0:
            saved[date_str] = str(out_path)
            print(f"  Rain [{idx+1}/{len(dates)}]: {date_str} -> cached")
            continue

        try:
            import geemap
            geemap.ee_export_image(
                image.clip(aoi),
                filename=str(out_path),
                scale=5000,  # CHIRPS native ~5km
                region=aoi,
                file_per_band=False,
            )
            saved[date_str] = str(out_path)
            print(f"  Rain [{idx+1}/{len(dates)}]: {date_str} -> {out_path.name}")
        except Exception as e:
            print(f"  Rain [{idx+1}/{len(dates)}]: FAILED {date_str} -- {e}")

    return saved


if __name__ == "__main__":
    from ingestion.sentinel1 import init_gee
    init_gee()
    results = fetch_rainfall()
    print(f"\nFetched {len(results)} rainfall days.")
