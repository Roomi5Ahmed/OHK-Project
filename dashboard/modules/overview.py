import streamlit as st
import plotly.graph_objects as go
from utils.load_data import get_flood_stats, DATES, DATE_LABELS

def render():
    st.title("Kerala Flood Risk Prediction 2018")
    st.caption("Satellite-based flood monitoring for the Periyar River Basin, Kerala, India")

    st.divider()

    # Key metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Validation Event", "Aug 2018")
    c2.metric("Data Sources", "5")
    c3.metric("Analysis Dates", "6")
    c4.metric("Output Files", "192 MB")

    st.divider()

    # Summary chart
    st.subheader("Flood Water Detection Summary")

    stats = get_flood_stats()
    dates_plot = [s['date'] for s in stats]
    thresh = [s['threshold'] for s in stats]
    cnn = [s['cnn'] for s in stats]
    rain = [s['rain7d'] for s in stats]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=dates_plot, y=rain, name="7-day Rainfall (mm)",
        marker_color="rgba(152,176,111,0.25)", yaxis="y2",
        hovertemplate="%{x}<br>Rainfall: %{y:.1f}mm<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=dates_plot, y=thresh, name="Threshold Model",
        line=dict(color="#98B06F", width=3), mode="lines+markers",
        marker=dict(size=8, color="#98B06F"),
        hovertemplate="%{x}<br>Threshold: %{y:.2f}%<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=dates_plot, y=cnn, name="CNN Model",
        line=dict(color="#B6DC76", width=3, dash="dash"), mode="lines+markers",
        marker=dict(size=8, symbol="diamond", color="#B6DC76"),
        hovertemplate="%{x}<br>CNN: %{y:.2f}%<extra></extra>"
    ))

    fig.update_layout(
        xaxis_title="Date",
        yaxis=dict(title="Flood Water %", gridcolor="#30363D", zeroline=False),
        yaxis2=dict(title="Rainfall (mm)", side="right", overlaying="y", gridcolor="#30363D"),
        template="plotly_dark",
        height=420,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(30,33,39,0.5)",
        margin=dict(l=40, r=40, t=20, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Results table
    st.subheader("Detailed Results")

    import pandas as pd
    df = pd.DataFrame(stats)
    df.columns = ["Date", "Threshold %", "CNN %", "7-day Rainfall (mm)"]
    df["Date"] = DATE_LABELS

    baseline_t = df.loc[df["Date"] == "Jul 28", "Threshold %"].values[0]
    baseline_c = df.loc[df["Date"] == "Jul 28", "CNN %"].values[0]

    def fmt_change(val, base):
        pct = (val / base - 1) * 100
        return f"+{pct:.0f}%" if pct > 0 else f"{pct:.0f}%"

    df["Threshold vs Baseline"] = [fmt_change(v, baseline_t) for v in df["Threshold %"]]
    df["CNN vs Baseline"] = [fmt_change(v, baseline_c) for v in df["CNN %"]]

    st.dataframe(df, use_container_width=True, hide_index=True)

    # Key findings
    st.divider()
    st.subheader("Key Findings")

    st.success("**Both methods identify Aug 21 as peak flood** — Threshold +73%, CNN +17% above baseline")
    st.info("**Jul 16 spike is legitimate** — Pre-monsoon heavy rain (150mm on Jul 15), not the August flood")
    st.success("**Flood recedes after Aug 21** — Both methods return to near-baseline by Aug 27")
    st.warning("**CNN as supporting evidence** — Independently reproduces peak-date trend despite domain gap")

    # Pipeline preview
    st.divider()
    st.subheader("How It Works")

    st.code("""Sentinel-1 SAR → dB Conversion → Lee Filter → Water Index → JRC Subtraction → Risk Score
SRTM DEM     → Slope        →           →           →           → Risk Score
CHIRPS Rain  → 7-day Accum  →           →           →           → Risk Score
Sen1Floods11 → U-Net Train  → CNN Infer → Water Mask→ JRC Sub   → CNN Output""", language=None)
