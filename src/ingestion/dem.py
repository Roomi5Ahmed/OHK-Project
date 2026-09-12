"""Fetch SRTM DEM (30m) via Google Earth Engine — static layer, one-time fetch."""

import ee
from config.aoi_config import GEE_COLLECTIONS, get_aoi_geometry, DATA_DIR


def fetch_dem(output_dir=None):
    """
    Fetch SRTM 30m DEM clipped to AOI. Returns filepath to GeoTIFF.
    """
    output_dir = output_dir or DATA_DIR / "dem"
    output_dir.mkdir(parents=True, exist_ok=True)

    aoi = get_aoi_geometry()
    dem = ee.Image(GEE_COLLECTIONS["dem"]).select("elevation").clip(aoi)

    out_path = output_dir / "srtm_dem.tif"

    # Skip if already downloaded
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"  DEM: cached")
        return str(out_path)

    try:
        import geemap
        geemap.ee_export_image(
            dem,
            filename=str(out_path),
            scale=30,
            region=aoi,
            file_per_band=False,
        )
        print(f"  DEM: {out_path.name}")
        return str(out_path)
    except Exception as e:
        print(f"  DEM: FAILED -- {e}")
        return None


if __name__ == "__main__":
    from ingestion.sentinel1 import init_gee
    init_gee()
    result = fetch_dem()
    print(f"\nDEM: {result}")
