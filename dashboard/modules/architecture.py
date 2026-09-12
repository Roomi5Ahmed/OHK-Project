import streamlit as st

def render():
    st.title("Pipeline Architecture")
    st.caption("From raw satellite data to validated flood risk results")

    st.divider()

    stages = [
        ("1. Data Ingestion — GEE-based collection", [
            "Sentinel-1 SAR (C-band, VH polarization, 5.4 GHz)",
            "Sentinel-2 Optical (cloud-masked NDWI)",
            "SRTM DEM (30m resolution, terrain slope)",
            "CHIRPS Rainfall (daily precipitation, 5km resolution)",
            "JRC Global Surface Water (permanent water mask)",
            "Sen1Floods11 (hand-labeled benchmark, 446 chips)",
        ]),
        ("2. Preprocessing — Noise reduction & standardization", [
            "SAR: Auto-detect dB vs linear (GEE data is dB)",
            "Lee speckle filter (3x3 kernel, sigma=1.0)",
            "DEM → slope risk layer (tan, 0-1 range)",
            "Rainfall → 7-day rolling accumulation",
            "All layers resampled to 10m common grid (bilinear)",
        ]),
        ("3. Water Detection — Two independent methods", [
            "Threshold: SAR water index (3×dB+12) minus JRC permanent water",
            "CNN: ResNet18 U-Net trained on Sen1Floods11 (India 3× oversample)",
            "Both produce binary water masks (> 0.5 threshold)",
        ]),
        ("4. Risk Scoring — Composite risk map", [
            "Water detection: 50% weight (peer-reviewed benchmark)",
            "Terrain risk: 25% weight (low-lying areas)",
            "Rainfall risk: 25% weight (recent precipitation)",
        ]),
        ("5. Validation — Ground-truth comparison", [
            "Event: August 2018 Kerala floods",
            "Result: Both methods identify Aug 21 as peak",
            "Threshold: +73%, CNN: +17% vs baseline",
        ]),
        ("6. Alerts & Export — Actionable output", [
            "High-risk regions (risk > 0.7) → CSV",
            "GeoTIFF risk maps (10m resolution)",
            "Validation charts (Plotly + Matplotlib)",
        ]),
    ]

    for title, items in stages:
        with st.expander(f"**{title}**", expanded=False):
            for item in items:
                st.markdown(f"- {item}")

    # Data flow diagram
    st.divider()
    st.subheader("Data Flow Diagram")

    st.code("""
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │ Sentinel-1  │ │ Sentinel-2  │ │  SRTM DEM   │ │  CHIRPS     │ │   JRC GSW   │
    │   SAR VH    │ │  Optical    │ │    30m      │ │  Rainfall   │ │ Perm. Water │
    └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
           │               │               │               │               │
           ▼               ▼               ▼               ▼               │
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐      │
    │ dB Convert  │ │    NDWI     │ │ Slope Risk  │ │  7-day      │      │
    │ Lee Filter  │ │ Cloud Mask  │ │  tan(theta) │ │  Accum      │      │
    └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘      │
           │               │               │               │              │
           ▼               ▼               ▼               ▼              ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                   COMPOSITE RISK SCORING                               │
    │        Water (50%) + Terrain (25%) + Rainfall (25%) = Risk [0,1]      │
    └───────────────────────────────────┬─────────────────────────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                                       ▼
             ┌─────────────┐                        ┌─────────────┐
             │  THRESHOLD  │                        │     CNN     │
             │  (Primary)  │                        │ (Supporting)│
             └──────┬──────┘                        └──────┬──────┘
                    ▼                                      ▼
             ┌─────────────┐                        ┌─────────────┐
             │  GeoTIFF    │                        │  GeoTIFF    │
             │  Risk Maps  │                        │  Water Mask │
             └─────────────┘                        └─────────────┘
    """, language=None)
