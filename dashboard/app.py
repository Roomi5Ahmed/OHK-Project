import streamlit as st
from utils.theme import inject_theme

st.set_page_config(
    page_title="Kerala Flood Risk Prediction",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_theme()

st.sidebar.title("🌊 Kerala Flood 2018")
st.sidebar.markdown("Flood Risk Prediction Dashboard")
st.sidebar.divider()

PAGES = {
    "🏠 Overview": "overview",
    "🔄 Pipeline Architecture": "architecture",
    "📡 Data Sources": "data_sources",
    "📊 Threshold Model": "threshold",
    "🧠 CNN Model": "cnn",
    "🗺️ Interactive Flood Map": "flood_map",
    "✅ Validation Summary": "validation",
    "📋 Technical Details": "technical",
}

selection = st.sidebar.radio("Navigate to", list(PAGES.keys()))

page = PAGES[selection]

# Load page module
import importlib
try:
    mod = importlib.import_module(f"modules.{page}")
    mod.render()
except Exception as e:
    st.error(f"Error loading page '{page}': {e}")
    import traceback
    st.code(traceback.format_exc())
