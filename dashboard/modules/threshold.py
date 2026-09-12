import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.load_data import get_flood_stats

def render():
    st.title("Threshold Model Results")
    st.caption("PRIMARY METHOD — Peer-reviewed SAR thresholding with JRC permanent water subtraction")

    st.divider()

    stats = get_flood_stats()
    dates = [s['date'] for s in stats]
    thresh = [s['threshold'] for s in stats]
    rain = [s['rain7d'] for s in stats]

    # Main chart
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(x=dates, y=rain, name="7-day Rainfall (mm)",
               marker_color="rgba(152,176,111,0.2)", opacity=0.6),
        secondary_y=True,
    )

    fig.add_trace(
        go.Scatter(x=dates, y=thresh, name="Threshold Flood Water %",
                   line=dict(color="#98B06F", width=3), mode="lines+markers",
                   marker=dict(size=10, color="#98B06F"),
                   hovertemplate="%{x}<br>Flood Water: %{y:.2f}%<extra></extra>"),
        secondary_y=False,
    )

    baseline_idx = dates.index('2018-07-28')
    peak_idx = dates.index('2018-08-21')

    fig.add_annotation(x='2018-07-28', y=thresh[baseline_idx],
                      text=f"Baseline: {thresh[baseline_idx]:.2f}%",
                      showarrow=True, arrowhead=2, ax=0, ay=-40,
                      font=dict(color="#98B06F"))

    fig.add_annotation(x='2018-08-21', y=thresh[peak_idx],
                      text=f"Peak: {thresh[peak_idx]:.2f}% (+73%)",
                      showarrow=True, arrowhead=2, ax=0, ay=-40,
                      font=dict(color="#B6DC76"))

    fig.update_layout(
        yaxis=dict(title="Flood Water %", range=[0, max(thresh)*1.5], gridcolor="#30363D"),
        template="plotly_dark", height=420, hovermode="x unified",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(30,33,39,0.5)",
        margin=dict(l=40, r=40, t=20, b=40),
    )
    fig.update_yaxes(title_text="Rainfall (mm)", secondary_y=True, gridcolor="#30363D")

    st.plotly_chart(fig, use_container_width=True)

    # Method
    st.divider()
    st.subheader("Method")

    st.code("""
# 1. SAR water index (empirical threshold)
raw_water = 3 * dB_VH + 12

# 2. Subtract permanent water (JRC GSW)
jrc_binary = (jrc_occurrence > 0.5).astype(float)
flood_water = raw_water * (1 - jrc_binary)

# 3. Binary water detection
water_mask = (flood_water > 0.5).astype(float)
flood_pct = water_mask.sum() / water_mask.size * 100
    """, language="python")

    st.markdown("""
    | Parameter | Value |
    |-----------|-------|
    | SAR Water Index | `3 × dB_VH + 12` |
    | Permanent Water | JRC occurrence > 0.5 |
    | Binary Threshold | > 0.5 |
    | JRC Coverage | 5.5% of AOI |
    """)

    st.divider()

    # Per-date breakdown
    st.subheader("Per-Date Analysis")

    notes = {
        '2018-07-16': ("Pre-monsoon spike", "150mm rainfall on Jul 15, before the August flood event"),
        '2018-07-28': ("Pre-flood baseline", "Lowest flood water extent"),
        '2018-08-09': ("Early flood onset", "Water extent increasing with rainfall"),
        '2018-08-21': ("Peak flood", "Maximum water extent, +73% above baseline"),
        '2018-08-27': ("Flood receding", "Returning toward baseline"),
        '2018-09-02': ("Post-flood", "Near baseline levels"),
    }

    for s in stats:
        title, desc = notes.get(s['date'], ("", ""))
        with st.expander(f"{s['date']} — {title}"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Flood Water", f"{s['threshold']:.2f}%")
            c2.metric("7-day Rain", f"{s['rain7d']:.1f}mm")
            baseline = stats[1]['threshold']
            change = (s['threshold'] / baseline - 1) * 100
            c3.metric("vs Baseline", f"{change:+.1f}%")
            st.caption(desc)
