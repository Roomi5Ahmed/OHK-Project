"""
Tier 2: T+3 day forecasted risk map.

Hold SAR water signal and terrain fixed at last-observed values,
replace rainfall with projected value 3 days ahead,
then run the composite risk formula to produce a forecasted risk map.
"""
import sys
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.aoi_config import DATA_DIR, OUTPUT_DIR, THRESHOLDS, WEIGHTS
from src.preprocessing.sar import preprocess_sar
from src.preprocessing.alignment import resample_array
from src.scoring.threshold import sar_water_index, rainfall_regional_bias


def project_rainfall_trend(daily_rain_series, days_ahead=3):
    """Simple linear extrapolation of recent rainfall trend."""
    recent = np.array(daily_rain_series[-7:])
    x = np.arange(len(recent))
    slope, intercept = np.polyfit(x, recent, 1)
    projected = [max(0, slope * (len(recent) + i) + intercept) for i in range(days_ahead)]
    return projected


def main():
    # --- Load rainfall series ---
    rain_dir = DATA_DIR / "rainfall"
    rain_files = sorted(rain_dir.glob("rainfall_*.tif"))

    daily_rain = {}
    for f in rain_files:
        date_str = f.stem.replace("rainfall_", "")
        with rasterio.open(f) as src:
            arr = src.read(1).astype(np.float32)
            daily_rain[date_str] = float(np.nanmean(arr))

    dates_sorted = sorted(daily_rain.keys())

    # --- Scenario: as of Aug 9 (last SAR pass), predict Aug 12 ---
    scenario_date = '2018-08-09'
    forecast_date = '2018-08-12'

    # Get rainfall data available as of scenario_date
    rain_up_to = [daily_rain[d] for d in dates_sorted if d <= scenario_date]

    print(f"Scenario: as of {scenario_date}, forecast to {forecast_date}")
    print(f"Rainfall data available: {len(rain_up_to)} days")

    # Project rainfall 3 days ahead
    projected = project_rainfall_trend(rain_up_to, days_ahead=3)
    print(f"Projected rainfall (next 3 days): {[f'{r:.1f}' for r in projected]}")
    print(f"Projected rainfall on {forecast_date}: {projected[-1]:.1f} mm")

    # --- Compute rainfall risk using projected data ---
    # Build extended series: actual up to Aug 10 + projected
    extended_rain = rain_up_to + projected

    # Compute 3-day and 7-day windows ending on forecast date
    forecast_idx = len(extended_rain) - 1
    window3 = extended_rain[max(0, forecast_idx - 2):forecast_idx + 1]
    window7 = extended_rain[max(0, forecast_idx - 6):forecast_idx + 1]

    rain_bias = rainfall_regional_bias(np.array(window3), np.array(window7))
    print(f"Rainfall risk bias (projected): {rain_bias:.3f}")

    # --- Load SAR water signal from Aug 9 (last observed) ---
    sar_path = DATA_DIR / "s1" / f"s1_vh_{scenario_date}.tif"
    if not sar_path.exists():
        print(f"SAR file not found: {sar_path}")
        return

    sar_db, sar_p = preprocess_sar(sar_path)
    raw_water = sar_water_index(sar_db)

    # Get bounds and transform from the SAR file directly
    with rasterio.open(sar_path) as src:
        sar_bounds = src.bounds
        sar_transform = src.transform
        sar_crs = src.crs
        sar_shape = (src.height, src.width)

    # --- Load terrain slope risk ---
    slope_path = DATA_DIR / "dem" / "slope_risk.tif"
    with rasterio.open(slope_path) as src:
        slope_raw = src.read(1).astype(np.float32)
    slope_risk = resample_array(slope_raw, src.transform, src.crs,
                                sar_shape, sar_transform, sar_crs)

    # --- Compute forecasted risk map ---
    water_weight = WEIGHTS["water_signal"]
    terrain_weight = WEIGHTS["terrain"]
    rainfall_weight = WEIGHTS["rainfall"]

    risk = (water_weight * raw_water +
            terrain_weight * slope_risk +
            rainfall_weight * rain_bias)

    risk = np.clip(risk, 0, 1)

    print(f"\nRisk map stats:")
    print(f"  Min: {risk.min():.3f}  Max: {risk.max():.3f}  Mean: {risk.mean():.3f}")
    print(f"  Pixels > 0.5: {(risk > 0.5).sum() / risk.size * 100:.1f}%")
    print(f"  Pixels > 0.7: {(risk > 0.7).sum() / risk.size * 100:.1f}%")

    # --- Save as GeoTIFF ---
    out_dir = OUTPUT_DIR / "risk_maps"
    out_path = out_dir / f"forecast_t3_{forecast_date}.tif"

    transform = from_bounds(sar_bounds.left, sar_bounds.bottom,
                            sar_bounds.right, sar_bounds.top,
                            risk.shape[1], risk.shape[0])

    profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'width': risk.shape[1],
        'height': risk.shape[0],
        'count': 1,
        'crs': sar_crs,
        'transform': transform,
    }

    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(risk.astype(np.float32), 1)
        dst.update_tags(
            description=f"PROJECTED T+3 risk map ({forecast_date})",
            scenario_date=scenario_date,
            forecast_date=forecast_date,
            rainfall_projected=f"{projected[-1]:.1f}mm",
            rainfall_bias=f"{rain_bias:.3f}",
            methodology="SAR water (Aug 10) + terrain + projected rainfall",
        )

    print(f"\nSaved: {out_path}")

    # --- Also save as PNG for visualization ---
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = plt.get_cmap('YlOrRd')
    norm = mcolors.Normalize(vmin=0, vmax=1)
    im = ax.imshow(risk, cmap=cmap, norm=norm)
    ax.set_title(f'Projected T+3 Risk Map\n(as of {scenario_date}, forecast to {forecast_date})',
                 fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Risk Score')

    # Add text annotation
    ax.text(0.02, 0.98,
            f'Projected rainfall: {projected[-1]:.1f}mm\n'
            f'Rainfall risk bias: {rain_bias:.3f}\n'
            f'Pixels > 0.5: {(risk > 0.5).sum() / risk.size * 100:.1f}%',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    png_path = OUTPUT_DIR / "validation_charts" / f"forecast_t3_{forecast_date}.png"
    fig.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
