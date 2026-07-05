"""Community Mode UI: aggregate dashboard, trends, forecasts, and Q&A."""

import os
import re
import uuid
import hashlib

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from agents.community_agent import answer_community_question, get_dashboard_data
from agents.extraction_agent import ingest_report
from data_store.sqlite_store import forecast_abnormal_trend, get_risk_forecast_by_region, get_test_trend
from ui.components import render_community_alert, render_metric_card


def _sanitize_filename(name: str) -> str:
    """Sanitize an uploaded filename to prevent path traversal."""
    name = os.path.basename(name)
    name = re.sub(r"[^\w\-.]", "_", name)
    if not name or name.startswith("."):
        name = f"upload_{uuid.uuid4().hex[:8]}.pdf"
    return name


def _batch_fingerprint(uploaded_files, region: str, age_group: str) -> str:
    """Return a stable fingerprint for a community upload batch."""
    h = hashlib.sha256()
    h.update(region.encode("utf-8"))
    h.update(age_group.encode("utf-8"))
    for uploaded_file in uploaded_files:
        h.update(_sanitize_filename(uploaded_file.name).encode("utf-8"))
        h.update(uploaded_file.getvalue())
    return h.hexdigest()


def _chart_layout(height: int = 380) -> dict:
    """Shared transparent Plotly layout."""
    return {
        "height": height,
        "margin": dict(l=0, r=0, t=12, b=0),
        "plot_bgcolor": "rgba(0,0,0,0)",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "font": dict(family="Inter"),
        "legend": dict(orientation="h", yanchor="bottom", y=1.02),
    }


def render_community_mode():
    """Render the full community intelligence dashboard."""
    st.markdown(
        """<div class="section-band">
            <h3>Community health operations</h3>
            <p>Upload batches of reports to monitor abnormal rates, regional patterns,
            age-group risk, and short-term forecast signals.</p>
        </div>""",
        unsafe_allow_html=True,
    )

    _render_bulk_upload()

    if "community_dashboard_data" not in st.session_state or st.session_state.community_dashboard_data is None:
        st.session_state.community_dashboard_data = get_dashboard_data()

    data = st.session_state.community_dashboard_data

    if data["metrics"]["total_reports"] == 0:
        _render_empty_dashboard()
        return

    top_col, refresh_col = st.columns([4, 1])
    with top_col:
        st.markdown("### Population Snapshot")
    with refresh_col:
        if st.button("Refresh", key="refresh_dash", use_container_width=True):
            st.session_state.community_dashboard_data = get_dashboard_data()
            data = st.session_state.community_dashboard_data

    m1, m2, m3 = st.columns(3)
    with m1:
        render_metric_card(str(data["metrics"]["total_reports"]), "Reports Analyzed")
    with m2:
        render_metric_card(str(data["metrics"]["total_lab_values"]), "Lab Values Recorded")
    with m3:
        render_metric_card(f"{data['metrics']['abnormal_rate']}%", "Abnormal Rate")

    if data["alerts"]:
        st.markdown("### Active Alerts")
        for alert in data["alerts"]:
            render_community_alert(alert)

    _render_dashboard_charts(data)
    _render_trends_and_forecasts(data)
    _render_community_chat()


def _render_bulk_upload():
    """Render bulk upload controls."""
    with st.expander("Upload reports for community analysis", expanded=True):
        uploaded_files = st.file_uploader(
            "Upload multiple medical reports",
            type=["pdf"],
            accept_multiple_files=True,
            key="community_uploader",
            help="Use aggregate, de-identified reports for community analysis.",
        )

        col_region, col_age, col_button = st.columns([2, 2, 1.2])
        with col_region:
            region = st.selectbox(
                "Region or locality",
                [
                    "Auto-assign (random)",
                    "Urban-Central",
                    "Urban-East",
                    "Suburban-North",
                    "Rural-West",
                    "Rural-South",
                ],
                key="community_region",
            )
        with col_age:
            age_group = st.selectbox(
                "Age group",
                ["Auto-assign (random)", "0-18", "19-30", "31-45", "46-60", "60+"],
                key="community_age",
            )
        with col_button:
            st.write("")
            process = st.button(
                "Process reports",
                key="process_community",
                use_container_width=True,
                disabled=not uploaded_files,
            )

        if uploaded_files and process:
            batch_key = _batch_fingerprint(uploaded_files, region, age_group)
            processed_batches = st.session_state.setdefault("community_processed_batches", set())
            if batch_key in processed_batches:
                st.info("This exact batch has already been processed in this session.")
                return

            with st.status(f"Processing {len(uploaded_files)} reports...", expanded=True):
                success_count = 0
                fail_count = 0
                data_dir = os.path.join("data", "community_uploads")
                os.makedirs(data_dir, exist_ok=True)

                for i, uploaded_file in enumerate(uploaded_files):
                    try:
                        safe_name = _sanitize_filename(uploaded_file.name)
                        st.write(f"[{i + 1}/{len(uploaded_files)}] {safe_name}")
                        file_path = os.path.join(data_dir, safe_name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        selected_region = region if region != "Auto-assign (random)" else None
                        selected_age = age_group if age_group != "Auto-assign (random)" else None

                        result = ingest_report(
                            file_path=file_path,
                            filename=safe_name,
                            mode="community",
                            collection_name="community_reports",
                            anonymized_region=selected_region,
                            age_group=selected_age,
                        )
                        st.write(
                            f"Done: {result['risk_summary']['total']} tests, "
                            f"{result['risk_summary']['abnormal']} abnormal"
                        )
                        success_count += 1
                    except Exception:
                        fail_count += 1
                        st.write(f"Failed: {uploaded_file.name}")

                if fail_count:
                    st.warning(f"{success_count} reports processed, {fail_count} failed.")
                else:
                    st.success(f"All {success_count} reports processed successfully.")

            if success_count:
                processed_batches.add(batch_key)
            st.session_state.community_dashboard_data = None


def _render_dashboard_charts(data: dict):
    """Render core community charts."""
    st.markdown("### Analytics")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("#### Most Common Abnormal Findings")
        if data["top_abnormal"]:
            df_abnormal = pd.DataFrame(data["top_abnormal"])
            fig = px.bar(
                df_abnormal,
                x="flag_count",
                y="test_name",
                orientation="h",
                color="percentage",
                color_continuous_scale=["#0f766e", "#d97706", "#dc2626"],
                labels={"flag_count": "Abnormal Count", "test_name": "Test", "percentage": "% Abnormal"},
            )
            fig.update_layout(**_chart_layout(390), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No abnormal values recorded yet.")

    with chart_col2:
        st.markdown("#### Flag Distribution")
        if data["flag_distribution"]:
            flag_colors = {
                "NORMAL": "#16a34a",
                "HIGH": "#d97706",
                "LOW": "#2563eb",
                "CRITICAL_HIGH": "#dc2626",
                "CRITICAL_LOW": "#b91c1c",
                "UNKNOWN": "#64748b",
            }
            labels = list(data["flag_distribution"].keys())
            values = list(data["flag_distribution"].values())
            colors = [flag_colors.get(label, "#94a3b8") for label in labels]

            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=[label.replace("_", " ") for label in labels],
                        values=values,
                        marker=dict(colors=colors),
                        hole=0.45,
                        textinfo="label+percent",
                    )
                ]
            )
            fig.update_layout(**_chart_layout(390), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for chart.")

    breakdown_col1, breakdown_col2 = st.columns(2)

    with breakdown_col1:
        st.markdown("#### Anomaly Rate by Region")
        if data["region_summary"]:
            df_region = pd.DataFrame(data["region_summary"])
            fig = px.bar(
                df_region,
                x="region",
                y="percentage",
                color="percentage",
                color_continuous_scale=["#16a34a", "#d97706", "#dc2626"],
                labels={"percentage": "Abnormal %", "region": "Region"},
            )
            fig.update_layout(**_chart_layout(340))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No regional data available.")

    with breakdown_col2:
        st.markdown("#### Anomaly Rate by Age Group")
        if data["age_group_summary"]:
            df_age = pd.DataFrame(data["age_group_summary"])
            fig = px.bar(
                df_age,
                x="age_group",
                y="percentage",
                color="percentage",
                color_continuous_scale=["#16a34a", "#d97706", "#dc2626"],
                labels={"percentage": "Abnormal %", "age_group": "Age Group"},
            )
            fig.update_layout(**_chart_layout(340))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No age-group data available.")


def _render_trends_and_forecasts(data: dict):
    """Render trend analysis and forecasts."""
    available_tests = data.get("available_tests", [])

    st.markdown("### Trends and Forecasts")
    trend_tab, forecast_tab = st.tabs(["Test Trend", "Forecast"])

    with trend_tab:
        if not available_tests:
            st.info("Upload reports first to enable trend analysis.")
        else:
            selected_test = st.selectbox("Select a test", available_tests)
            trend_data = get_test_trend(selected_test) if selected_test else []
            if trend_data:
                df_trend = pd.DataFrame(trend_data)
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=df_trend["date"],
                        y=df_trend["avg_value"],
                        mode="lines+markers",
                        name="Average Value",
                        line=dict(color="#2563eb", width=3),
                        marker=dict(size=7),
                    )
                )
                if "abnormal_count" in df_trend.columns:
                    fig.add_trace(
                        go.Bar(
                            x=df_trend["date"],
                            y=df_trend["abnormal_count"],
                            name="Abnormal Count",
                            marker=dict(color="rgba(220, 38, 38, 0.28)"),
                            yaxis="y2",
                        )
                    )
                fig.update_layout(
                    **_chart_layout(400),
                    xaxis_title="Date",
                    yaxis_title="Average Value",
                    yaxis2=dict(title="Abnormal Count", overlaying="y", side="right"),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"No trend data available for {selected_test}.")

    with forecast_tab:
        forecast_col1, forecast_col2 = st.columns([2, 1])

        with forecast_col1:
            forecast_test = st.selectbox(
                "Forecast abnormal rate for",
                ["All Tests"] + (available_tests if available_tests else []),
                key="forecast_test",
            )
            days_ahead = st.slider("Forecast horizon (days)", 7, 90, 30, key="forecast_days")

            test_filter = None if forecast_test == "All Tests" else forecast_test
            forecast = forecast_abnormal_trend(test_filter, days_ahead)

            if forecast.get("forecast_data"):
                st.markdown(f"**{forecast['message']}**")
                hist_dates = [h["date"] for h in forecast["historical_data"]]
                hist_rates = [h["rate"] for h in forecast["historical_data"]]
                fc_dates = [f["date"] for f in forecast["forecast_data"]]
                fc_rates = [f["projected_rate"] for f in forecast["forecast_data"]]

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=hist_dates,
                        y=hist_rates,
                        mode="lines+markers",
                        name="Historical",
                        line=dict(color="#2563eb", width=3),
                        marker=dict(size=6),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=fc_dates,
                        y=fc_rates,
                        mode="lines",
                        name="Forecast",
                        line=dict(color="#d97706", width=3, dash="dash"),
                    )
                )
                if hist_dates:
                    fig.add_vline(
                        x=hist_dates[-1],
                        line_dash="dot",
                        line_color="rgba(148,163,184,0.8)",
                        annotation_text="Forecast start",
                    )
                fig.update_layout(
                    **_chart_layout(360),
                    xaxis_title="Date",
                    yaxis_title="Abnormal Rate (%)",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(forecast.get("message", "Upload more reports across different days to enable forecasting."))

        with forecast_col2:
            st.markdown("#### Regional Risk")
            regional_forecasts = get_risk_forecast_by_region(days_ahead)
            if regional_forecasts:
                for rf in regional_forecasts:
                    risk_label = rf["risk_level"].upper()
                    arrow = "up" if rf["trend_direction"] == "increasing" else "down" if rf["trend_direction"] == "decreasing" else "flat"
                    st.markdown(
                        f"**{rf['region']}**  \n"
                        f"{risk_label}: {rf['current_rate']}% to {rf['projected_rate']}% ({arrow})"
                    )
            else:
                st.info("No regional data for forecasting yet.")


def _render_community_chat():
    """Render natural-language community Q&A."""
    st.markdown("### Ask About Community Health")
    st.caption("Try: Which region has the most abnormal blood sugar readings?")

    if "community_chat" not in st.session_state:
        st.session_state.community_chat = []

    for msg in st.session_state.community_chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    st.markdown("**Suggested questions**")
    sq_cols = st.columns(4)
    sq_query = None
    if sq_cols[0].button("Regional trends", key="sq_region"):
        sq_query = "Which locality or region has the highest rate of abnormal lab values?"
    if sq_cols[1].button("Top concerns", key="sq_top"):
        sq_query = "What are the top 3 health concerns based on the lab data from all reports?"
    if sq_cols[2].button("Age risks", key="sq_age"):
        sq_query = "Which age group shows the most concerning health indicators?"
    if sq_cols[3].button("Predict outcomes", key="sq_predict"):
        sq_query = "Based on current trends, which health indicators are predicted to worsen in the next 30 days?"

    community_input = st.chat_input("Ask a community health question...", key="community_chat_input")
    cq = sq_query or community_input

    if not cq:
        return

    st.session_state.community_chat.append({"role": "user", "content": cq})
    with st.chat_message("user"):
        st.markdown(cq)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing community data..."):
            try:
                result = answer_community_question(cq)
                answer_text = result["answer"]
            except Exception:
                answer_text = "Unable to analyze community data right now. Please try again in a moment."
            st.markdown(answer_text)

    st.session_state.community_chat.append({"role": "assistant", "content": answer_text})


def _render_empty_dashboard():
    """Render empty state for the community dashboard."""
    st.markdown(
        """<div class="empty-state">
            <h2>No community data yet</h2>
            <p>Upload medical reports above to populate aggregate trends, alerts, and forecasts.</p>
        </div>
        <div class="feature-grid">
            <div class="feature-card">
                <div class="label">Monitor</div>
                <h4>Population snapshot</h4>
                <p>Track reports, lab values, abnormal rates, and active alert signals.</p>
            </div>
            <div class="feature-card">
                <div class="label">Compare</div>
                <h4>Region and age views</h4>
                <p>Find where abnormal patterns are concentrated across local groups.</p>
            </div>
            <div class="feature-card">
                <div class="label">Plan</div>
                <h4>Forecast risk</h4>
                <p>Project abnormal-rate trends to support resource planning.</p>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )
