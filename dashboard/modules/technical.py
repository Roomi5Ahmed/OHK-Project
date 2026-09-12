import streamlit as st

def render():
    st.title("Technical Details")
    st.caption("Project structure, configuration, dependencies, and data disclosure")

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(["Project Structure", "Configuration", "Dependencies", "Data Disclosure"])

    with tab1:
        render_project_structure()
    with tab2:
        render_configuration()
    with tab3:
        render_dependencies()
    with tab4:
        render_disclosure()


def render_project_structure():
    st.subheader("Project Structure")

    st.code("""
OHK Project/
├── config/
│   └── aoi_config.py          # AOI, thresholds, weights, GEE IDs
├── src/
│   ├── ingestion/             # GEE data download (S1, S2, DEM, rainfall)
│   ├── preprocessing/         # SAR dB, Lee filter, NDWI, slope, alignment
│   ├── scoring/
│   │   ├── threshold.py       # Threshold-based risk scoring
│   │   └── cnn_water.py       # U-Net water detection
│   ├── validation/            # Flood validation
│   ├── alerts/                # High-risk region alerting
│   └── output/                # GeoTIFF export
├── data/                      # Cached satellite data (864 MB)
├── models/                    # Trained U-Net weights (112 MB)
├── output/                    # Risk maps, charts, alerts (192 MB)
├── dashboard/                 # Streamlit dashboard
├── train_flood_model.py       # U-Net training script
├── process_cached.py          # Offline pipeline
├── run_pipeline.py            # Full GEE pipeline
├── README.md                  # Setup & usage docs
└── REPORT.md                  # Technical report
    """, language=None)


def render_configuration():
    st.subheader("Configuration")

    st.markdown("""
    | Parameter | Value |
    |-----------|-------|
    | **GEE Project ID** | Set via `GEE_PROJECT` env var |
    | **AOI Bounds** | 76.25–76.45°E, 9.95–10.15°N |
    | **Data Window** | 2018-07-15 to 2018-09-10 |
    | **Validation Dates** | 2018-08-09, 2018-08-21 |
    | **SAR Water Index** | `3 * dB_VH + 12` |
    | **JRC Permanent Water** | occurrence > 0.5 |
    | **Risk Threshold** | > 0.7 for high-risk alerts |
    | **Weights** | Water: 50%, Terrain: 25%, Rainfall: 25% |
    """)


def render_dependencies():
    st.subheader("Python Dependencies")

    st.markdown("""
    | Category | Packages |
    |----------|----------|
    | **Core** | numpy, rasterio, scipy, pyproj |
    | **ML** | torch, segmentation-models-pytorch |
    | **Visualization** | matplotlib, plotly, streamlit, folium |
    | **Data** | pandas, geemap, earthengine-api |
    """)


def render_disclosure():
    st.subheader("Data & Methodology Disclosure")

    st.markdown("**Data Sources:**")
    st.markdown("- All satellite imagery accessed via Google Earth Engine (GEE)")
    st.markdown("- Sen1Floods11 benchmark dataset from Google Cloud Storage")
    st.markdown("- No proprietary or restricted data used")

    st.markdown("**Methodology:**")
    st.markdown("- Threshold method based on peer-reviewed SAR water detection literature")
    st.markdown("- CNN architecture: standard U-Net with ResNet18 encoder")
    st.markdown("- Training on publicly available Sen1Floods11 benchmark")
    st.markdown("- No custom labeled training data created")

    st.markdown("**Reproducibility:**")
    st.markdown("- All code is in the project repository")
    st.markdown("- GEE project ID is public")
    st.markdown("- Sen1Floods11 dataset is publicly available")
    st.markdown("- Model weights saved and can be retrained")
