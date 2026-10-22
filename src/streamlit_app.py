import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="Effecom",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

GREEN_PRIMARY = "#2E7D32"
GREEN_LIGHT = "#4CAF50"
GREEN_BG = "#E8F5E9"
GREEN_DARK = "#1B5E20"

st.markdown(
    f"""
<style>
    .stApp {{
        background-color: #FAFAFA;
    }}
    .stSidebar {{
        background-color: {GREEN_BG};
    }}
    .stSidebar .stMarkdown, .stSidebar .stText, .stSidebar label, .stSidebar .stSelectbox label,
    .stSidebar .stNumberInput label, .stSidebar .stSlider label, .stSidebar .stRadio label,
    .stSidebar .stCheckbox label, .stSidebar .stMultiselect label {{
        color: {GREEN_DARK} !important;
        font-weight: 500;
    }}
    .stButton > button[kind="primary"] {{
        background-color: {GREEN_PRIMARY} !important;
        border-color: {GREEN_PRIMARY} !important;
        color: white !important;
    }}
    .stButton > button[kind="primary"]:hover {{
        background-color: {GREEN_DARK} !important;
        border-color: {GREEN_DARK} !important;
    }}
    .stButton > button[kind="secondary"] {{
        background-color: white !important;
        border-color: {GREEN_PRIMARY} !important;
        color: {GREEN_PRIMARY} !important;
    }}
    .stButton > button[kind="secondary"]:hover {{
        background-color: {GREEN_BG} !important;
        border-color: {GREEN_DARK} !important;
        color: {GREEN_DARK} !important;
    }}
    .stProgress > div > div > div > div {{
        background-color: {GREEN_PRIMARY} !important;
    }}
    .stMetric {{
        background-color: {GREEN_BG};
        border: 1px solid {GREEN_PRIMARY};
        border-radius: 8px;
        padding: 12px;
    }}
    .stMetric label {{
        color: {GREEN_DARK} !important;
    }}
    .stMetric div[data-testid="stMetricValue"] {{
        color: {GREEN_PRIMARY} !important;
        font-weight: 700;
    }}
    .health-card {{
        background-color: {GREEN_BG};
        border: 1px solid {GREEN_PRIMARY};
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
    }}
    .health-header {{
        color: {GREEN_DARK};
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 8px;
    }}
    .health-text {{
        color: {GREEN_DARK};
    }}
    .stAlert > div {{
        border-radius: 8px;
    }}
    .stSuccess {{
        background-color: {GREEN_BG} !important;
        border-color: {GREEN_PRIMARY} !important;
        color: {GREEN_DARK} !important;
    }}
    .stError {{
        background-color: #FDECEA !important;
        border-color: #C62828 !important;
        color: #C62828 !important;
    }}
    .stWarning {{
        background-color: #FFF8E1 !important;
        border-color: #F57F17 !important;
        color: #F57F17 !important;
    }}
    .stInfo {{
        background-color: #E3F2FD !important;
        border-color: #1565C0 !important;
        color: #1565C0 !important;
    }}
    h1, h2, h3 {{
        color: {GREEN_DARK} !important;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {GREEN_BG};
        border-radius: 8px 8px 0 0;
        color: {GREEN_DARK};
        font-weight: 500;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {GREEN_PRIMARY} !important;
        color: white !important;
    }}
    div[data-testid="stExpander"] {{
        background-color: {GREEN_BG};
        border: 1px solid {GREEN_PRIMARY};
        border-radius: 8px;
    }}
    div[data-testid="stExpander"] summary {{
        color: {GREEN_DARK} !important;
        font-weight: 600;
    }}
    .stCodeBlock {{
        background-color: #1B1B1B;
        border-radius: 8px;
    }}
    .main-title {{
        color: {GREEN_PRIMARY} !important;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }}
    .subtitle {{
        color: {GREEN_DARK} !important;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }}
</style>
""",
    unsafe_allow_html=True,
)

from streamlit_pages.train import render_train_page
def main():
    st.markdown('<h1 class="main-title">Effecom</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">USL Translator - Training Dashboard</p>', unsafe_allow_html=True)

    render_train_page()

if __name__ == "__main__":
    main()