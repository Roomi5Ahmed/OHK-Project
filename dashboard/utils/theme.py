COLORS = {
    "primary": "#1B2D2A",
    "secondary": "#98B06F",
    "tertiary": "#B6DC76",
}

CUSTOM_CSS = """
<style>
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1B2D2A;
    }

    /* Headings */
    h1 { color: #FAFAFA !important; }
    h2 { color: #B6DC76 !important; }
    h3 { color: #98B06F !important; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #1E2127;
        border: 1px solid #30363D;
        border-radius: 10px;
        padding: 16px 20px;
    }
    [data-testid="stMetricValue"] {
        color: #B6DC76 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #98B06F !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #1E2127;
        border-radius: 8px;
        padding: 4px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1B2D2A !important;
        color: #B6DC76 !important;
    }

    /* Dataframe */
    .stDataFrame {
        border: 1px solid #30363D;
        border-radius: 8px;
    }

    /* Dividers */
    hr {
        border: none;
        border-top: 1px solid #30363D;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1E2127;
        border: 1px solid #30363D;
        border-radius: 8px;
    }
</style>
"""


def inject_theme():
    import streamlit as st
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
