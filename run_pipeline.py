"""
Flood Risk Prediction Pipeline — Single Entry Point
====================================================
Run: python run_pipeline.py

Steps:
  1. Ingest Sentinel-1 SAR, Sentinel-2 optical, SRTM DEM, CHIRPS rainfall
  2. Preprocess: dB conversion, NDWI, slope, alignment
  3. Score: threshold-based composite risk (MVP)
  4. Validate: risk timeseries chart against Aug 2018 Kerala floods
  5. Alert: flag high-risk regions
  6. Export: GeoTIFF risk maps + validation chart + alert CSV
"""

import sys
import os
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.aoi_config import (
    DATA_WINDOW, VALIDATION_DATES, DATA_DIR, OUTPUT_DIR,
    get_region_name, get_aoi_bbox,
)


def step_header(step_num, title):
    print(f"\n{'='*60}")
    print(f"  STEP {step_num}: {title}")
    print(f"{'='*60}")


def main():
    start_time = datetime.now()
    print("Flood Risk Prediction Pipeline")
    print(f"Region: {get_region_name()}")
    print(f"Data window: {DATA_WINDOW['start']} to {DATA_WINDOW['end']}")
    print(f"Validation dates: {list(VALIDATION_DATES.values())}")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # ── Step 1: Data Ingestion ────────────────────────────────────────────
    step_header(1, "DATA INGESTION")

    from src.ingestion.sentinel1 import init_gee, fetch_sar
    from src.ingestion.sentinel2 import fetch_optical
    from src.ingestion.dem import fetch_dem
    from src.ingestion.rainfall import fetch_rainfall

    print("Initializing Google Earth Engine...")
    init_gee()

    print("\nFetching Sentinel-1 SAR (VH)...")
    sar_files = fetch_sar()

    print("\nFetching Sentinel-2 optical (B3/Green, B8/NIR)...")
    optical_files = fetch_optical()

    print("\nFetching SRTM DEM (30m)...")
    dem_file = fetch_dem()

    print("\nFetching CHIRPS rainfall...")
    rainfall_files = fetch_rainfall()

    print(f"\nIngestion complete: {len(sar_files)} S1, {len(optical_files)} S2, "
          f"1 DEM, {len(rainfall_files)} rainfall days")

    # ── Step 2: Preprocessing ─────────────────────────────────────────────
    step_header(2, "PREPROCESSING")

    from src.preprocessing.sar import preprocess_sar
    from src.preprocessing.optical import preprocess_optical
    from src.preprocessing.terrain import preprocess_dem
    from src.preprocessing.alignment import align_layers

    processed = {}
    for date_str in sorted(sar_files.keys()):
        print(f"\n  Processing {date_str}...")

        # SAR: dB conversion + Lee filter
        sar_db, sar_profile = preprocess_sar(
            sar_files[date_str],
            output_dir=DATA_DIR / "s1",
            apply_filter=True,
        )

        # Optical: NDWI
        if date_str in optical_files:
            ndwi, ndwi_profile = preprocess_optical(
                optical_files[date_str]["all_bands"],
                output_dir=DATA_DIR / "s2",
            )
        else:
            print(f"    No S2 data for {date_str}, using zeros for NDWI")
            ndwi = np.zeros_like(sar_db)
            ndwi_profile = sar_profile

        # DEM: slope (static, compute once)
        if "slope" not in processed:
            slope_risk, elevation, slope_profile = preprocess_dem(
                dem_file, output_dir=DATA_DIR / "dem"
            )
            processed["slope"] = slope_risk
            processed["elevation"] = elevation
            processed["slope_profile"] = slope_profile

        # Rainfall
        if date_str in rainfall_files:
            import rasterio
            with rasterio.open(rainfall_files[date_str]) as src:
                rainfall = src.read(1).astype(np.float32)
                rain_profile = src.profile.copy()
        else:
            rainfall = np.zeros_like(sar_db)
            rain_profile = sar_profile

        # Compute 3-day and 7-day rolling accumulation (at CHIRPS resolution)
        rainfall_3day = compute_rolling_accumulation(date_str, rainfall_files, days=3)
        rainfall_7day = compute_rolling_accumulation(date_str, rainfall_files, days=7)

        # Align all layers to SAR grid — including rainfall accumulations
        aligned = align_layers(
            sar_db, ndwi, processed["slope"], rainfall_3day,
            sar_profile, ndwi_profile, processed["slope_profile"], rain_profile,
        )
        # Also align 7-day accumulation to SAR grid
        from src.preprocessing.alignment import resample_array
        aligned["rainfall_3d"] = aligned["rainfall"]  # already aligned by align_layers
        aligned["rainfall_7d"] = resample_array(
            rainfall_7day,
            rain_profile["transform"], rain_profile["crs"],
            (sar_profile["height"], sar_profile["width"]),
            sar_profile["transform"], sar_profile["crs"],
        )

        processed[date_str] = aligned
        print(f"    Aligned: SAR {sar_db.shape}, NDWI {ndwi.shape}")

    print(f"\nPreprocessed {len(processed) - 3} dates (plus static DEM/slope)")

    # ── Step 3: Risk Scoring ──────────────────────────────────────────────
    step_header(3, "RISK SCORING (MVP Threshold)")

    from src.scoring.threshold import compute_risk, save_risk_raster

    risk_timeseries = {}
    risk_rasters = {}

    for date_str in sorted(processed.keys()):
        if date_str in ("slope", "elevation", "slope_profile"):
            continue

        data = processed[date_str]
        risk = compute_risk(
            sar_db=data["sar"],
            ndwi=data["ndwi"],
            slope_risk=data["slope"],
            rainfall_3day=data.get("rainfall_3d", data["rainfall"]),
            rainfall_7day=data.get("rainfall_7d", data["rainfall"]),
        )

        risk_rasters[date_str] = risk
        risk_timeseries[date_str] = float(risk.mean())

        # Save GeoTIFF
        save_risk_raster(risk, data["profile"],
                         output_path=OUTPUT_DIR / "risk_maps" / f"risk_{date_str}.tif",
                         satellite_date=date_str)

        print(f"  {date_str}: mean risk = {risk.mean():.3f}, max = {risk.max():.3f}")

    print(f"\nScored {len(risk_rasters)} time steps")

    # ── Step 4: Validation ────────────────────────────────────────────────
    step_header(4, "VALIDATION")

    from src.validation.validate import (
        plot_validation, compute_validation_stats, print_validation_report,
    )

    chart_path = plot_validation(
        risk_timeseries,
        output_path=OUTPUT_DIR / "validation_charts" / "validation_risk_timeseries.png",
    )

    stats = compute_validation_stats(risk_timeseries)
    print_validation_report(stats)

    # ── Step 5: Alerts ────────────────────────────────────────────────────
    step_header(5, "ALERTS")

    from src.alerts.alert import (
        flag_high_risk, save_alerts, compute_alert_summary, print_alert_report,
    )

    # Use the peak risk date for alerts
    peak_date = stats.get("peak_date")
    if peak_date and peak_date in risk_rasters:
        alerts = flag_high_risk(risk_rasters[peak_date], processed[peak_date]["profile"])
        save_alerts(alerts, satellite_date=peak_date)
        summary = compute_alert_summary(alerts, risk_rasters[peak_date].size)
        print_alert_report(summary)
    else:
        print("  No peak date found for alert generation.")

    # ── Step 6: Summary ───────────────────────────────────────────────────
    step_header(6, "PIPELINE COMPLETE")

    elapsed = datetime.now() - start_time
    print(f"  Total time: {elapsed}")
    print(f"  Region: {get_region_name()}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"\n  Generated files:")
    print(f"    Risk maps:     {OUTPUT_DIR / 'risk_maps'}")
    print(f"    Validation:    {chart_path}")
    print(f"    Alerts:        {OUTPUT_DIR / 'alerts'}")
    print(f"\n  Data disclosure: Data as of satellite pass dates.")
    print(f"  Next pass expected per S1 revisit schedule (~{6} days).")


def compute_rolling_accumulation(target_date, rainfall_files, days=3):
    """
    Compute rolling rainfall accumulation for `days` days ending at target_date.
    Falls back to available data if not enough days.
    """
    import rasterio

    target_dt = datetime.strptime(target_date, "%Y-%m-%d")
    accumulation = None

    for i in range(days):
        check_date = (target_dt - timedelta(days=i)).strftime("%Y-%m-%d")
        if check_date in rainfall_files:
            with rasterio.open(rainfall_files[check_date]) as src:
                daily = src.read(1).astype(np.float32)
                if accumulation is None:
                    accumulation = daily
                else:
                    accumulation = accumulation + daily

    if accumulation is None:
        # No rainfall data available, return zeros with a default shape
        return np.zeros((100, 100), dtype=np.float32)

    return accumulation


if __name__ == "__main__":
    main()
