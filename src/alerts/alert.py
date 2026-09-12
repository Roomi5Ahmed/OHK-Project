"""Alert module: flag high-risk regions from risk raster."""

import numpy as np
import rasterio
import csv
from pathlib import Path
from config.aoi_config import THRESHOLDS, ALERTS_DIR


def flag_high_risk(risk_raster, profile, threshold=None):
    """
    Identify pixels exceeding the risk alert threshold.
    Returns list of dicts: [{"row": int, "col": int, "lat": float, "lon": float, "risk": float}]
    """
    if threshold is None:
        threshold = THRESHOLDS["risk_alert"]

    risk = risk_raster if isinstance(risk_raster, np.ndarray) else risk_raster.read(1)
    rows, cols = np.where(risk > threshold)

    if len(rows) == 0:
        return []

    # Convert pixel indices to geographic coordinates
    transform = profile["transform"]
    alerts = []
    for r, c in zip(rows, cols):
        lon, lat = rasterio.transform.xy(transform, r, c)
        alerts.append({
            "row": int(r),
            "col": int(c),
            "lat": float(lat),
            "lon": float(lon),
            "risk": float(risk[r, c]),
        })

    # Sort by risk descending
    alerts.sort(key=lambda x: x["risk"], reverse=True)
    return alerts


def save_alerts(alerts, output_path=None, satellite_date=None):
    """Save flagged regions to CSV."""
    if output_path is None:
        output_path = ALERTS_DIR / "high_risk_regions.csv"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["lat", "lon", "risk", "row", "col"])
        writer.writeheader()
        writer.writerows(alerts)

    print(f"  Alerts saved: {output_path} ({len(alerts)} regions)")
    return str(output_path)


def compute_alert_summary(alerts, total_pixels):
    """Compute summary stats for alert regions."""
    if not alerts:
        return {
            "total_flagged": 0,
            "pct_flagged": 0.0,
            "mean_risk_flagged": 0.0,
            "max_risk_flagged": 0.0,
        }

    risks = [a["risk"] for a in alerts]
    return {
        "total_flagged": len(alerts),
        "pct_flagged": len(alerts) / total_pixels * 100,
        "mean_risk_flagged": float(np.mean(risks)),
        "max_risk_flagged": float(np.max(risks)),
    }


def print_alert_report(summary):
    """Print human-readable alert summary."""
    print("\n--- Alert Summary ---")
    print(f"  High-risk pixels:  {summary['total_flagged']}")
    print(f"  % of AOI:          {summary['pct_flagged']:.2f}%")
    print(f"  Mean risk (flagged): {summary['mean_risk_flagged']:.3f}")
    print(f"  Max risk (flagged):  {summary['max_risk_flagged']:.3f}")
