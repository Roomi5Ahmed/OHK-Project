"""Quick processing script using already-downloaded data. No GEE calls."""
import sys
import numpy as np
import rasterio
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.aoi_config import DATA_DIR, OUTPUT_DIR, VALIDATION_DATES, get_region_name
from src.preprocessing.sar import preprocess_sar
from src.preprocessing.optical import preprocess_optical
from src.preprocessing.terrain import preprocess_dem
from src.preprocessing.alignment import align_layers, resample_array
from src.scoring.threshold import compute_risk, save_risk_raster, _load_jrc_permanent_water
from src.scoring.cnn_water import predict_water_mask
from src.validation.validate import plot_validation, compute_validation_stats, print_validation_report
from src.alerts.alert import flag_high_risk, save_alerts, compute_alert_summary, print_alert_report

def compute_rolling_accumulation(target_date, rainfall_dir, days=3):
    from datetime import timedelta
    target_dt = datetime.strptime(target_date, "%Y-%m-%d")
    accumulation = None
    for i in range(days):
        check_date = (target_dt - timedelta(days=i)).strftime("%Y-%m-%d")
        rain_file = rainfall_dir / f"rainfall_{check_date}.tif"
        if rain_file.exists():
            with rasterio.open(rain_file) as src:
                daily = src.read(1).astype(np.float32)
                accumulation = daily if accumulation is None else accumulation + daily
    return accumulation if accumulation is not None else None

def main():
    print(f"Flood Risk Prediction Pipeline (cached data)")
    print(f"Region: {get_region_name()}")

    s1_dir = DATA_DIR / "s1"
    s2_dir = DATA_DIR / "s2"
    dem_dir = DATA_DIR / "dem"
    rain_dir = DATA_DIR / "rainfall"

    # Find available S1 dates
    s1_files = sorted(s1_dir.glob("s1_vh_*.tif"))
    dates = [f.stem.replace("s1_vh_", "") for f in s1_files]
    print(f"Available dates: {dates}")

    # Precompute static DEM slope
    print("\nPreprocessing DEM...")
    slope_risk, elevation, slope_profile = preprocess_dem(dem_dir / "srtm_dem.tif")

    # Preprocess all dates
    print("\nPreprocessing...")
    risk_timeseries = {}
    for date_str in dates:
        print(f"  {date_str}...")
        sar_db, sar_profile = preprocess_sar(s1_files[dates.index(date_str)])

        # NDWI (zeros if no S2)
        ndwi = np.zeros_like(sar_db)

        # Resample slope to SAR grid (DEM is 30m, SAR is 10m)
        slope_aligned = resample_array(
            slope_risk,
            slope_profile["transform"], slope_profile["crs"],
            sar_db.shape, sar_profile["transform"], sar_profile["crs"],
        )

        # Rainfall accumulation
        rain_3d = compute_rolling_accumulation(date_str, rain_dir, days=3)
        rain_7d = compute_rolling_accumulation(date_str, rain_dir, days=7)

        if rain_3d is None:
            rain_3d = np.zeros_like(sar_db)
        if rain_7d is None:
            rain_7d = np.zeros_like(sar_db)

        # Get rainfall file metadata for proper resampling
        rain_file_path = rain_dir / f"rainfall_{date_str}.tif"
        with rasterio.open(rain_file_path) as rain_src:
            rain_transform = rain_src.transform
            rain_crs = rain_src.crs

        # Resample rainfall to SAR grid (using rainfall's own transform as source)
        rain_3d_aligned = resample_array(
            rain_3d, rain_transform, rain_crs,
            sar_db.shape, sar_profile["transform"], sar_profile["crs"],
        )
        rain_7d_aligned = resample_array(
            rain_7d, rain_transform, rain_crs,
            sar_db.shape, sar_profile["transform"], sar_profile["crs"],
        )

        # Score
        risk = compute_risk(
            sar_db=sar_db, ndwi=ndwi,
            slope_risk=slope_aligned,
            rainfall_3day=rain_3d_aligned,
            rainfall_7day=rain_7d_aligned,
            sar_shape=sar_db.shape,
            sar_transform=sar_profile["transform"],
            sar_crs=sar_profile["crs"],
        )

        risk_timeseries[date_str] = float(risk.mean())

        # CNN water detection
        cnn_water = predict_water_mask(sar_db, sar_db)
        if cnn_water is not None:
            from src.scoring.threshold import _load_jrc_permanent_water
            jrc = _load_jrc_permanent_water(sar_db.shape, sar_profile["transform"], sar_profile["crs"])
            if jrc is not None:
                cnn_flood = cnn_water * (1 - jrc)
            else:
                cnn_flood = cnn_water

            cnn_pct = float(cnn_flood.mean()) * 100
            print(f"    CNN flood water: {cnn_pct:.2f}%")

            out_profile = sar_profile.copy()
            out_profile.update(dtype="float32", count=1)
            cnn_path = OUTPUT_DIR / "risk_maps" / f"cnn_water_{date_str}.tif"
            cnn_path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(cnn_path, "w", **out_profile) as dst:
                dst.write(cnn_flood, 1)

        # Save
        save_risk_raster(risk, sar_profile,
                         OUTPUT_DIR / "risk_maps" / f"risk_{date_str}.tif",
                         satellite_date=date_str)
        print(f"    mean={risk.mean():.3f} max={risk.max():.3f}")

    # Validation
    print("\nValidation...")
    chart_path = plot_validation(risk_timeseries,
                                 OUTPUT_DIR / "validation_charts" / "validation_risk_timeseries.png")
    stats = compute_validation_stats(risk_timeseries)
    print_validation_report(stats)

    # Alerts on peak date
    peak_date = stats.get("peak_date")
    if peak_date:
        with rasterio.open(OUTPUT_DIR / "risk_maps" / f"risk_{peak_date}.tif") as src:
            risk_peak = src.read(1)
            peak_profile = src.profile.copy()
        alerts = flag_high_risk(risk_peak, peak_profile)
        save_alerts(alerts, satellite_date=peak_date)
        summary = compute_alert_summary(alerts, risk_peak.size)
        print_alert_report(summary)

    print("\nDone!")

if __name__ == "__main__":
    main()
