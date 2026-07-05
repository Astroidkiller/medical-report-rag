"""Shared UI components for the Community Health Intelligence Assistant."""

import html as html_mod

import streamlit as st

from core.anomaly_detector import FlaggedValue


def _esc(text) -> str:
    """Escape text for safe insertion into HTML markup."""
    if text is None:
        return ""
    return html_mod.escape(str(text))


def render_disclaimer():
    """Render the medical disclaimer bar."""
    st.markdown(
        """<div class="disclaimer-bar">
        <strong>Medical disclaimer:</strong> This tool is for
        <strong>educational and informational purposes only</strong>. It is not a
        diagnostic tool and should not be used for real medical decisions. Always
        consult a qualified healthcare professional for advice, diagnosis, or treatment.
        </div>""",
        unsafe_allow_html=True,
    )


def render_mode_badge(mode: str):
    """Render a mode indicator badge."""
    badge = "Patient Mode" if mode == "patient" else "Community Intelligence Mode"
    css = "mode-badge-patient" if mode == "patient" else "mode-badge-community"
    st.markdown(
        f'<span class="mode-badge {css}">{badge}</span>',
        unsafe_allow_html=True,
    )


def render_risk_card(risk_card: dict):
    """Render a visual risk assessment card."""
    level = _esc(risk_card["risk_level"]).lower()
    css_class = f"risk-card risk-card-{level}"

    st.markdown(
        f"""<div class="{css_class} animate-in">
            <h3>{_esc(risk_card['icon'])} Risk Level: {_esc(risk_card['risk_level'])}</h3>
            <p>{_esc(risk_card['headline'])}</p>
            <p style="margin-top: 8px; font-size: 0.85rem;">
                {_esc(risk_card['total_tests'])} tests analyzed -
                {_esc(risk_card['normal_count'])} normal,
                {_esc(risk_card['abnormal_count'])} abnormal,
                {_esc(risk_card['critical_count'])} critical
            </p>
        </div>""",
        unsafe_allow_html=True,
    )


def render_lab_value_card(fv: FlaggedValue):
    """Render a single lab value as a styled card."""
    flag_lower = fv.flag.lower()
    flag_class = f"lab-card lab-card-{_esc(flag_lower)}"

    if fv.severity == 2:
        value_color = "#dc2626"
        badge_class = "flag-badge flag-critical"
    elif fv.severity == 1:
        value_color = "#d97706" if "HIGH" in fv.flag else "#2563eb"
        badge_class = "flag-badge flag-high" if "HIGH" in fv.flag else "flag-badge flag-low"
    else:
        value_color = "#16a34a"
        badge_class = "flag-badge flag-normal"

    if fv.reference_low is not None:
        ref_text = f"{_esc(fv.reference_low)} - {_esc(fv.reference_high)} {_esc(fv.unit)}"
    else:
        ref_text = "N/A"

    st.markdown(
        f"""<div class="{flag_class}">
            <div style="display: flex; justify-content: space-between; align-items: center; gap: 10px;">
                <div class="lab-name">{_esc(fv.test_name)}</div>
                <span class="{badge_class}">{_esc(fv.flag.replace('_', ' '))}</span>
            </div>
            <div class="lab-value" style="color: {value_color};">{_esc(fv.value)} {_esc(fv.unit)}</div>
            <div class="lab-ref">Reference: {ref_text}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_lab_values_grid(flagged_values: list[FlaggedValue]):
    """Render lab values in a responsive grid layout."""
    displayable = [fv for fv in flagged_values if fv.flag != "UNKNOWN"]

    if not displayable:
        st.info("No lab values could be extracted and assessed from this report.")
        return

    abnormal = [fv for fv in displayable if fv.severity > 0]
    normal = [fv for fv in displayable if fv.severity == 0]

    if abnormal:
        st.markdown("#### Values Requiring Attention")
        cols = st.columns(min(len(abnormal), 3))
        for i, fv in enumerate(abnormal):
            with cols[i % 3]:
                render_lab_value_card(fv)

    if normal:
        with st.expander(f"Normal Values ({len(normal)})"):
            cols = st.columns(min(len(normal), 3))
            for i, fv in enumerate(normal):
                with cols[i % 3]:
                    render_lab_value_card(fv)


def render_source_evidence(source_chunks: list[str], source_metadata: list[dict] = None):
    """Render source evidence chunks."""
    if not source_chunks:
        return

    with st.expander("Source Evidence (Retrieved Report Sections)", expanded=False):
        st.markdown(
            '<div class="source-label">Report sections used to generate this answer</div>',
            unsafe_allow_html=True,
        )
        for i, chunk in enumerate(source_chunks):
            meta_info = ""
            if source_metadata and i < len(source_metadata):
                meta = source_metadata[i]
                if meta.get("filename"):
                    meta_info = f" - from {_esc(meta['filename'])}"

            st.markdown(
                f'<div class="source-chunk"><strong>Source {i + 1}{meta_info}:</strong><br>{_esc(chunk)}</div>',
                unsafe_allow_html=True,
            )


def render_metric_card(value: str, label: str):
    """Render a styled metric card."""
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-value">{_esc(value)}</div>
            <div class="metric-label">{_esc(label)}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_community_alert(alert: dict):
    """Render a community health alert."""
    css_class = "alert-critical" if alert["severity"] == "critical" else "alert-warning"
    icon = "HIGH" if alert["severity"] == "critical" else "WATCH"

    st.markdown(
        f"""<div class="alert-card {css_class}">
            <div class="alert-icon">{icon}</div>
            <div class="alert-text">{_esc(alert['message'])}</div>
        </div>""",
        unsafe_allow_html=True,
    )
