import streamlit as st
import folium
from folium import plugins
from streamlit_folium import st_folium
import rasterio
import numpy as np
from pathlib import Path
from utils.load_data import DATES, DATE_LABELS, DATA_DIR, OUTPUT_DIR

def render():
    st.title("Interactive Flood Map")
    st.caption("Explore flood risk across the Periyar Basin for each analysis date")

    st.divider()

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

        tif_path = risk_path if layer_choice == "Threshold Risk" else cnn_path
        label = "Threshold Risk Score" if layer_choice == "Threshold Risk" else "CNN Water Probability"

        if not tif_path.exists():
            st.error(f"File not found: {tif_path}")
            return

        try:
            with rasterio.open(tif_path) as src:
                data = src.read(1)
                bounds = src.bounds
                crs = src.crs

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

            vmin = float(np.percentile(valid_data, 5))
            vmax = float(np.percentile(valid_data, 95))

            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import tempfile
            from PIL import Image

            cmap_name = 'Blues' if layer_choice == "Threshold Risk" else 'Reds'
            cmap = plt.get_cmap(cmap_name)

            norm_data = np.clip((data - vmin) / (vmax - vmin + 1e-10), 0, 1)
            rgba_data = cmap(norm_data)
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
    st.subheader(f"Flood Extent — {selected_label}")

    from utils.load_data import get_flood_stats
    stats = get_flood_stats()
    s = stats[date_idx]

    c1, c2, c3 = st.columns(3)
    c1.metric("Threshold Flood %", f"{s['threshold']:.2f}%")
    c2.metric("CNN Flood %", f"{s['cnn']:.2f}%")
    c3.metric("7-day Rainfall", f"{s['rain7d']:.1f}mm")
