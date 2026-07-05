"""
Community Health Intelligence Assistant
Main Streamlit Application — Dual-Mode Interface

A platform that helps both individual patients AND community health stakeholders
understand health data and make better decisions.

Modes:
- 🧑‍⚕️ Patient Mode: Upload your report, get explanations, ask questions
- 🏥 Community Mode: Aggregate dashboard with trends, alerts, and NL queries
"""

import streamlit as st
from ui.styles import get_custom_css
from ui.components import render_disclaimer, render_mode_badge
from ui.patient_mode import render_patient_mode
from ui.community_mode import render_community_mode

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Community Health Intelligence Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "Community Health Intelligence Assistant — "
            "An AI-powered platform for patient report understanding "
            "and community health trend analysis."
        ),
    },
)

# ---------- INJECT CUSTOM CSS ----------
st.markdown(get_custom_css(), unsafe_allow_html=True)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("## 🏥 Health Intelligence")
    st.markdown("---")

    mode = st.radio(
        "Select Mode",
        ["🧑‍⚕️ Patient Mode", "🏥 Community Mode"],
        index=0,
        key="app_mode",
        help="Patient Mode: Understand your personal report.\n"
             "Community Mode: Analyze population health trends.",
    )

    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        """
        **Community Health Intelligence Assistant** helps both individual patients
        and community health stakeholders (clinics, ASHA workers, local health
        departments) understand health data and make better decisions.

        **Built with:**
        - 🤖 Groq / Llama-3.1 (LLM)
        - 📊 ChromaDB (Vector Search)
        - 🧬 SentenceTransformers (Embeddings)
        - 📈 Plotly (Visualizations)
        - 🗃️ SQLite (Aggregate Data)
        """
    )

    st.markdown("---")
    st.markdown(
        '<p style="font-size:0.75rem; opacity:0.5;">v2.0 — AI for Better Living & Smarter Communities</p>',
        unsafe_allow_html=True,
    )

# ---------- HEADER ----------
st.markdown(
    """<h1 style="margin-bottom: 4px;">
    🏥 Community Health Intelligence Assistant
    </h1>""",
    unsafe_allow_html=True,
)
st.caption("AI for Better Living and Smarter Communities — Powered by RAG + Anomaly Detection + Aggregate Analytics")

# Mode badge
current_mode = "patient" if "Patient" in mode else "community"
render_mode_badge(current_mode)
st.markdown("")  # spacer

# Disclaimer
render_disclaimer()

# ---------- ROUTE TO MODE ----------
if current_mode == "patient":
    render_patient_mode()
else:
    render_community_mode()