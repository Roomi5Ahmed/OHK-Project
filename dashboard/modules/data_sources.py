import streamlit as st
import folium
from streamlit_folium import st_folium

def render():
    st.title("Data Sources")
    st.caption("Satellite and ground data for the Periyar River Basin study area")

    st.divider()

    # AOI Map
    st.subheader("Study Area")

    m = folium.Map(location=[10.05, 76.35], zoom_start=12, tiles="OpenStreetMap")
    folium.Rectangle(
        bounds=[(9.95, 76.25), (10.15, 76.45)],
        color="#B6DC76", fill=True, fill_opacity=0.15, weight=2,
    ).add_to(m)
    folium.Marker(
        [10.05, 76.35], popup="Aluva Town",
        icon=folium.Icon(color="green", icon="info-sign")
    ).add_to(m)
    st_folium(m, width=700, height=400)

    st.divider()

    # Data sources
    st.subheader("Satellite & Ground Data")

    import pandas as pd
    sources = pd.DataFrame([
        ["Sentinel-1 SAR", "C-band SAR (5.405 GHz)", "VH, 10m", "COPERNICUS/S1_GRD"],
        ["Sentinel-2 Optical", "Multispectral (13 bands)", "10m, cloud-masked", "COPERNICUS/S2_SR"],
        ["SRTM DEM", "Shuttle Radar Topography", "30m", "USGS/SRTMGL1_003"],
        ["CHIRPS Rainfall", "Climate Hazards Group", "Daily, 5km", "UCSB-CHG/CHIRPS/DAILY"],
        ["JRC Global Surface Water", "Joint Research Centre", "Monthly occurrence", "JRC/GSW1_4/GlobalSurfaceWater"],
        ["Sen1Floods11", "Hand-labeled benchmark", "431 chips, 10 countries", "Google Cloud Storage"],
    ], columns=["Source", "Description", "Resolution", "Collection ID"])

    st.dataframe(sources, use_container_width=True, hide_index=True)

    st.divider()

    # Temporal coverage
    st.subheader("Temporal Coverage")

    c1, c2, c3 = st.columns(3)
    c1.metric("Data Window", "Jul 15 - Sep 10")
    c2.metric("Analysis Dates", "6")
    c3.metric("Validation Event", "Aug 2018")

    st.markdown("""
    | Parameter | Value |
    |-----------|-------|
    | **GEE Project ID** | Set via `GEE_PROJECT` env var |
    | **Peak Flood Date** | August 21, 2018 |
    | **Analysis Dates** | Jul 16, Jul 28, Aug 9, Aug 21, Aug 27, Sep 2 |
    """)

    st.divider()

    # Known limitations
    st.subheader("Known Limitations")

    st.warning("**No Sentinel-2 During Monsoon** — 100% cloud cover, NDWI unavailable")
    st.warning("**VH-Only SAR** — No VV polarization, CNN duplicates VH as pseudo-dual-pol")
    st.warning("**Rainfall Not Bias-Corrected** — CHIRPS may overestimate tropical precipitation")
    st.info("**Domain Gap** — Sen1Floods11 covers 10 countries, not Kerala-specific")
