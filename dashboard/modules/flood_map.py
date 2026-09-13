import streamlit as st
import folium
from folium import plugins
from streamlit_folium import st_folium
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import functools
from utils.load_data import DATES, DATE_LABELS, DATA_DIR, OUTPUT_DIR

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


@functools.lru_cache(maxsize=None)
def _global_vrange(layer_choice):
    """Compute shared vmin/vmax across all 6 dates so severity is visually comparable."""
    if not HAS_RASTERIO:
        return 0.0, 1.0
    prefix = "risk_" if layer_choice == "Threshold Risk" else "cnn_water_"
    all_vals = []
    for d in DATES:
        path = OUTPUT_DIR / "risk_maps" / f"{prefix}{d}.tif"
        if path.exists():
            with rasterio.open(path) as src:
                arr = src.read(1)
                all_vals.append(arr[~np.isnan(arr)])
    if not all_vals:
        return 0.0, 1.0
    combined = np.concatenate(all_vals)
    return float(np.percentile(combined, 5)), float(np.percentile(combined, 95))


def render():
    st.title("Interactive Flood Map")
    st.caption("Explore flood risk across the Periyar Basin for each analysis date")

    st.divider()

    if not HAS_RASTERIO:
        st.warning("**Map requires rasterio** — not available in this deployment. "
                   "Run the dashboard locally with `pip install rasterio` to view interactive maps.")
        st.info("The flood risk data is pre-generated in `demo_output/` — see the Validation Summary page for charts.")
        return

    col1, col2 = st.columns([3, 1])

    with col2:
        st.markdown("**Controls**")
        date_idx = st.slider("Select Date", min_value=0, max_value=len(DATES)-1, value=3)
        layer_choice = st.radio("Data Layer", ["Threshold Risk", "CNN Water Detection"], horizontal=True)

    selected_date = DATES[date_idx]
    selected_label = DATE_LABELS[date_idx]

    with col1:
        st.markdown(f"**{selected_label} ({selected_date})**")

        risk_path = OUTPUT_DIR / "risk_maps" / f"risk_{selected_date}.tif"
        cnn_path = OUTPUT_DIR / "risk_maps" / f"cnn_water_{selected_date}.tif"

        if not risk_path.exists() and not cnn_path.exists():
            st.warning(f"Risk map files not found — data not available in cloud deployment.")
            st.info("Run `python process_cached.py` locally to generate risk maps.")
            return

        try:
            if layer_choice == "Threshold Risk":
                tif_path = risk_path
                label = "Threshold Risk Score"
            else:
                tif_path = cnn_path
                label = "CNN Water Change vs Baseline"

            with rasterio.open(tif_path) as src:
                data = src.read(1)
                bounds = src.bounds
                crs = src.crs

            # CNN: show change relative to baseline (Jul 28)
            if layer_choice == "CNN Water Detection":
                baseline_path = OUTPUT_DIR / "risk_maps" / "cnn_water_2018-07-28.tif"
                with rasterio.open(baseline_path) as src:
                    baseline_data = src.read(1)
                # Handle shape mismatch by cropping to common dimensions
                min_rows = min(data.shape[0], baseline_data.shape[0])
                min_cols = min(data.shape[1], baseline_data.shape[1])
                data = data[:min_rows, :min_cols] - baseline_data[:min_rows, :min_cols]

            from pyproj import Transformer
            transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            lon_min, lat_min = transformer.transform(bounds.left, bounds.bottom)
            lon_max, lat_max = transformer.transform(bounds.right, bounds.top)

            center_lat = (lat_min + lat_max) / 2
            center_lon = (lon_min + lon_max) / 2

            m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

            valid_data = data[~np.isnan(data)]
            if len(valid_data) == 0:
                st.warning("No valid data in raster")
                return

            # Global color scale for threshold; per-range for CNN diff
            if layer_choice == "Threshold Risk":
                vmin, vmax = _global_vrange("Threshold Risk")
            else:
                abs_max = max(abs(float(np.percentile(valid_data, 5))),
                              abs(float(np.percentile(valid_data, 95))))
                vmin, vmax = -abs_max, abs_max

            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
            import tempfile
            from PIL import Image

            if layer_choice == "CNN Water Detection":
                # Diverging colormap: blue = drier than baseline, white = normal, red = wetter
                cmap = plt.get_cmap('RdBu_r')
                norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
            else:
                cmap = plt.get_cmap('Blues')
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

            rgba_data = cmap(norm(data))
            rgba_data[np.isnan(data)] = [0, 0, 0, 0]
            rgba_img = (rgba_data * 255).astype(np.uint8)

            img = Image.fromarray(rgba_img)
            tmp_png = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            img.save(tmp_png.name, format="PNG")

            folium.raster_layers.ImageOverlay(
                image=tmp_png.name,
                bounds=[[lat_min, lon_min], [lat_max, lon_max]],
                opacity=0.6,
                name=label
            ).add_to(m)

            folium.LayerControl().add_to(m)
            plugins.MiniMap(toggle_display=True).add_to(m)

            st_folium(m, width=700, height=500)

        except Exception as e:
            st.error(f"Error rendering map: {e}")
            import traceback
            st.code(traceback.format_exc())

    # Stats below map
    st.divider()

    # Satellite pass lag disclosure
    pass_dt = datetime.strptime(selected_date, "%Y-%m-%d")
    next_pass_dt = pass_dt + timedelta(days=9)
    st.caption(
        f"📡 **Satellite pass:** {selected_date} · "
        f"**Next Sentinel-1 pass expected:** ~{next_pass_dt.strftime('%Y-%m-%d')} (~9 days) · "
        f"Conditions may have changed since this pass."
    )

    st.subheader(f"Flood Extent — {selected_label}")

    from utils.load_data import get_flood_stats
    stats = get_flood_stats()
    s = stats[date_idx]

    c1, c2, c3 = st.columns(3)
    c1.metric("Threshold Flood %", f"{s['threshold']:.2f}%")
    c2.metric("CNN Flood %", f"{s['cnn']:.2f}%")
    c3.metric("7-day Rainfall", f"{s['rain7d']:.1f}mm")
