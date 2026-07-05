"""Community Health Intelligence Assistant Streamlit application."""

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st

from ui.components import render_disclaimer, render_mode_badge
from ui.styles import get_custom_css


st.set_page_config(
    page_title="Community Health Intelligence Assistant",
    page_icon="health",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "Community Health Intelligence Assistant is an AI-assisted workspace "
            "for patient report understanding and community health trend analysis."
        ),
    },
)


with st.sidebar:
    st.markdown(
        """<div class="sidebar-title">Health Intelligence</div>
        <div class="sidebar-subtitle">Patient reports and population insights</div>""",
        unsafe_allow_html=True,
    )

    mode = st.radio(
        "Workspace",
        ["Patient Mode", "Community Mode"],
        index=0,
        key="app_mode",
        help="Patient Mode explains personal reports. Community Mode analyzes population-level trends.",
    )

    theme = st.segmented_control(
        "Theme",
        ["Dark", "Light"],
        default="Dark",
        key="app_theme",
    )

    st.markdown("---")
    st.markdown("### System")
    st.caption("LLM: Gemini or configured fallback")
    st.caption("Retrieval: ChromaDB vector search")
    st.caption("Analytics: SQLite, pandas, Plotly")

    st.markdown("---")
    st.caption("v2.1 - redesigned Streamlit interface")


theme = theme if isinstance(theme, str) else "Dark"

try:
    custom_css = get_custom_css(theme)
except TypeError:
    custom_css = get_custom_css()

st.markdown(custom_css, unsafe_allow_html=True)

current_mode = "patient" if mode == "Patient Mode" else "community"

st.markdown(
    """<section class="app-hero">
        <div class="app-kicker">AI-assisted health workspace</div>
        <h1>Community Health Intelligence Assistant</h1>
        <p>
            Review patient reports, surface abnormal values, ask grounded questions,
            and monitor community-level health trends from one focused dashboard.
        </p>
    </section>""",
    unsafe_allow_html=True,
)

render_mode_badge(current_mode)
render_disclaimer()

if current_mode == "patient":
    from ui.patient_mode import render_patient_mode

    render_patient_mode()
else:
    from ui.community_mode import render_community_mode

    render_community_mode()
