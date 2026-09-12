"""Validation module: compare risk scores against known historical flood dates."""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from pathlib import Path

from config.aoi_config import VALIDATION_DATES, CHARTS_DIR, get_region_name


def plot_validation(risk_timeseries, output_path=None):
    """
    Plot aggregate risk score over time with flood date markers.

    risk_timeseries: dict of {date_str: mean_risk_value}
    output_path: where to save the chart PNG.
    """
    if not risk_timeseries:
        print("WARNING: No risk timeseries data to plot.")
        return None

    # Sort by date
    sorted_dates = sorted(risk_timeseries.keys())
    dates = [datetime.strptime(d, "%Y-%m-%d") for d in sorted_dates]
    values = [risk_timeseries[d] for d in sorted_dates]

    fig, ax = plt.subplots(figsize=(14, 6))

    # Risk line
    ax.plot(dates, values, "b-o", linewidth=2, markersize=4, label="Mean Risk Score", zorder=3)

    # Fill under curve
    ax.fill_between(dates, values, alpha=0.15, color="blue")

    # Flood event markers
    flood_colors = ["red", "darkred"]
    flood_labels = ["Flood Event 1 (Aug 9)", "Flood Event 2 (Aug 21)"]
    for i, (key, date_str) in enumerate(VALIDATION_DATES.items()):
        flood_date = datetime.strptime(date_str, "%Y-%m-%d")
        if dates[0] <= flood_date <= dates[-1]:
            ax.axvline(x=flood_date, color=flood_colors[i], linestyle="--",
                       linewidth=2, label=flood_labels[i], zorder=2)

    # Formatting
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Mean Risk Score (0-1)", fontsize=12)
    ax.set_title(
        f"Flood Risk Score Over Time — {get_region_name()}\n"
        f"Validation against Kerala August 2018 Floods",
        fontsize=14, fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3)

    # Latency disclosure text
    ax.text(
        0.99, 0.01,
        "Data as of satellite pass dates shown.\nActual conditions may have changed.",
        transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
        style="italic", alpha=0.6,
    )

    plt.tight_layout()

    if output_path is None:
        output_path = CHARTS_DIR / "validation_risk_timeseries.png"

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Validation chart: {output_path}")
    return str(output_path)


def compute_validation_stats(risk_timeseries):
    """
    Compute summary statistics for validation.
    Returns dict with key metrics.
    """
    if not risk_timeseries:
        return {}

    sorted_dates = sorted(risk_timeseries.keys())
    values = [risk_timeseries[d] for d in sorted_dates]

    flood_dates = list(VALIDATION_DATES.values())
    pre_flood = [risk_timeseries[d] for d in sorted_dates
                 if d < flood_dates[0]]
    during_flood = [risk_timeseries[d] for d in sorted_dates
                    if flood_dates[0] <= d <= flood_dates[1]]
    post_flood = [risk_timeseries[d] for d in sorted_dates
                  if d > flood_dates[1]]

    stats = {
        "overall_mean": float(np.mean(values)),
        "overall_max": float(np.max(values)),
        "pre_flood_mean": float(np.mean(pre_flood)) if pre_flood else None,
        "during_flood_mean": float(np.mean(during_flood)) if during_flood else None,
        "post_flood_mean": float(np.mean(post_flood)) if post_flood else None,
        "peak_date": sorted_dates[int(np.argmax(values))],
        "peak_value": float(np.max(values)),
    }

    # Compute risk increase ratio
    if stats["pre_flood_mean"] and stats["during_flood_mean"] and stats["pre_flood_mean"] > 0:
        stats["risk_increase_ratio"] = stats["during_flood_mean"] / stats["pre_flood_mean"]
    else:
        stats["risk_increase_ratio"] = None

    return stats


def print_validation_report(stats):
    """Print human-readable validation summary."""
    print("\n" + "=" * 60)
    print("VALIDATION REPORT — Kerala August 2018 Floods")
    print("=" * 60)
    print(f"  Overall mean risk:     {stats.get('overall_mean', 'N/A'):.3f}")
    print(f"  Overall max risk:      {stats.get('overall_max', 'N/A'):.3f}")
    print(f"  Peak risk date:        {stats.get('peak_date', 'N/A')}")
    print(f"  Peak risk value:       {stats.get('peak_value', 'N/A'):.3f}")
    print("-" * 60)
    print(f"  Pre-flood mean:        {stats.get('pre_flood_mean', 'N/A'):.3f}")
    print(f"  During-flood mean:     {stats.get('during_flood_mean', 'N/A'):.3f}")
    print(f"  Post-flood mean:       {stats.get('post_flood_mean', 'N/A'):.3f}")
    print("-" * 60)
    ratio = stats.get("risk_increase_ratio")
    if ratio:
        print(f"  Risk increase ratio:   {ratio:.2f}x (during / pre-flood)")
        if ratio > 1.5:
            print("  [OK] Risk demonstrably rose during the flood event.")
        else:
            print("  [!] Risk increase is modest - review thresholds.")
    print("=" * 60)
