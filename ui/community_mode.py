"""
Community Mode UI — bulk upload, aggregate dashboard, trend analysis,
and natural language queries for health workers.
"""

import re
import uuid
import streamlit as st
import os
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from agents.extraction_agent import ingest_report
from agents.community_agent import get_dashboard_data, answer_community_question
from ui.components import render_community_alert, render_metric_card
from core.embeddings import clear_collection
from data_store.sqlite_store import get_test_trend, get_all_test_names, forecast_abnormal_trend, get_risk_forecast_by_region


def _sanitize_filename(name: str) -> str:
    """Sanitize an uploaded filename to prevent path traversal."""
    name = os.path.basename(name)
    name = re.sub(r"[^\w\-.]", "_", name)
    if not name or name.startswith("."):
        name = f"upload_{uuid.uuid4().hex[:8]}.pdf"
    return name


def render_community_mode():
    """Render the full community intelligence dashboard."""

    st.markdown("### 🏥 Community Health Intelligence Dashboard")
    st.caption(
        "Upload population-level medical reports to surface health trends, "
        "anomaly patterns, and actionable insights for your community."
    )

    # --- Bulk Upload Section ---
    with st.expander("📤 Upload Reports for Community Analysis", expanded=True):
        uploaded_files = st.file_uploader(
            "Upload multiple medical reports (PDF)",
            type=["pdf"],
            accept_multiple_files=True,
            key="community_uploader",
            help="Upload reports from your clinic or health center for aggregate analysis.",
        )

        col_region, col_age = st.columns(2)
        with col_region:
            region = st.selectbox(
                "Region/Locality (simulated)",
                ["Auto-assign (random)", "Urban-Central", "Urban-East",
                 "Suburban-North", "Rural-West", "Rural-South"],
                key="community_region",
            )
        with col_age:
            age_group = st.selectbox(
                "Age Group (simulated)",
                ["Auto-assign (random)", "0-18", "19-30", "31-45", "46-60", "60+"],
                key="community_age",
            )

        if uploaded_files and st.button("🚀 Process & Analyze Reports", key="process_community"):
            with st.status(f"Processing {len(uploaded_files)} reports...", expanded=True):
                success_count = 0
                fail_count = 0
                data_dir = os.path.join("data", "community_uploads")
                os.makedirs(data_dir, exist_ok=True)

                for i, uploaded_file in enumerate(uploaded_files):
                    st.write(f"[{i+1}/{len(uploaded_files)}] {uploaded_file.name}...")

                    # FIX #6: Isolate per-file failures
                    try:
                        # FIX #8: Sanitize uploaded filename
                        safe_name = _sanitize_filename(uploaded_file.name)
                        file_path = os.path.join(data_dir, safe_name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        r = region if region != "Auto-assign (random)" else None
                        a = age_group if age_group != "Auto-assign (random)" else None

                        result = ingest_report(
                            file_path=file_path,
                            filename=safe_name,
                            mode="community",
                            collection_name="community_reports",
                            anonymized_region=r,
                            age_group=a,
                        )
                        st.write(
                            f"  ✅ {result['risk_summary']['total']} tests, "
                            f"{result['risk_summary']['abnormal']} abnormal"
                        )
                        success_count += 1
                    except Exception as exc:
                        fail_count += 1
                        st.write(f"  ❌ Failed to process {uploaded_file.name}")

                if fail_count:
                    st.warning(f"⚠️ {success_count} reports processed, {fail_count} failed.")
                else:
                    st.success(f"✅ All {success_count} reports processed successfully!")
            # Force dashboard refresh
            st.session_state.community_dashboard_data = None

    # --- Dashboard ---
    st.markdown("---")

    # Load dashboard data
    if "community_dashboard_data" not in st.session_state or st.session_state.community_dashboard_data is None:
        st.session_state.community_dashboard_data = get_dashboard_data()

    data = st.session_state.community_dashboard_data

    if data["metrics"]["total_reports"] == 0:
        _render_empty_dashboard()
        return

    # Refresh button
    if st.button("🔄 Refresh Dashboard", key="refresh_dash"):
        st.session_state.community_dashboard_data = get_dashboard_data()
        data = st.session_state.community_dashboard_data

    # --- Metrics Row ---
    m1, m2, m3 = st.columns(3)
    with m1:
        render_metric_card(str(data["metrics"]["total_reports"]), "Reports Analyzed")
    with m2:
        render_metric_card(str(data["metrics"]["total_lab_values"]), "Lab Values Recorded")
    with m3:
        render_metric_card(f"{data['metrics']['abnormal_rate']}%", "Abnormal Rate")

    # --- Alerts ---
    if data["alerts"]:
        st.markdown("### 🚨 Community Health Alerts")
        for alert in data["alerts"]:
            render_community_alert(alert)

    st.markdown("---")

    # --- Charts Section ---
    chart_col1, chart_col2 = st.columns(2)

    # Top Abnormal Tests (Bar Chart)
    with chart_col1:
        st.markdown("#### 📊 Most Common Abnormal Findings")
        if data["top_abnormal"]:
            df_abnormal = pd.DataFrame(data["top_abnormal"])
            fig = px.bar(
                df_abnormal,
                x="flag_count",
                y="test_name",
                orientation="h",
                color="percentage",
                color_continuous_scale=["#3b82f6", "#f59e0b", "#ef4444"],
                labels={"flag_count": "Abnormal Count", "test_name": "Test", "percentage": "% Abnormal"},
            )
            fig.update_layout(
                height=400,
                margin=dict(l=0, r=0, t=10, b=0),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No abnormal values recorded yet.")

    # Flag Distribution (Pie Chart)
    with chart_col2:
        st.markdown("#### 🎯 Flag Distribution")
        if data["flag_distribution"]:
            flag_colors = {
                "NORMAL": "#22c55e",
                "HIGH": "#f59e0b",
                "LOW": "#3b82f6",
                "CRITICAL_HIGH": "#ef4444",
                "CRITICAL_LOW": "#dc2626",
                "UNKNOWN": "#64748b",
            }
            labels = list(data["flag_distribution"].keys())
            values = list(data["flag_distribution"].values())
            colors = [flag_colors.get(l, "#94a3b8") for l in labels]

            fig = go.Figure(data=[go.Pie(
                labels=[l.replace("_", " ") for l in labels],
                values=values,
                marker=dict(colors=colors),
                hole=0.4,
                textinfo="label+percent",
            )])
            fig.update_layout(
                height=400,
                margin=dict(l=0, r=0, t=10, b=0),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for chart.")

    st.markdown("---")

    # --- Region & Age Breakdowns ---
    breakdown_col1, breakdown_col2 = st.columns(2)

    with breakdown_col1:
        st.markdown("#### 📍 Anomaly Rate by Region")
        if data["region_summary"]:
            df_region = pd.DataFrame(data["region_summary"])
            fig = px.bar(
                df_region,
                x="region",
                y="percentage",
                color="percentage",
                color_continuous_scale=["#22c55e", "#f59e0b", "#ef4444"],
                labels={"percentage": "Abnormal %", "region": "Region"},
            )
            fig.update_layout(
                height=350,
                margin=dict(l=0, r=0, t=10, b=0),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No regional data available.")

    with breakdown_col2:
        st.markdown("#### 👥 Anomaly Rate by Age Group")
        if data["age_group_summary"]:
            df_age = pd.DataFrame(data["age_group_summary"])
            fig = px.bar(
                df_age,
                x="age_group",
                y="percentage",
                color="percentage",
                color_continuous_scale=["#22c55e", "#f59e0b", "#ef4444"],
                labels={"percentage": "Abnormal %", "age_group": "Age Group"},
            )
            fig.update_layout(
                height=350,
                margin=dict(l=0, r=0, t=10, b=0),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No age group data available.")

    # --- Trend Analysis ---
    st.markdown("---")
    st.markdown("### 📈 Test Trend Analysis")
    available_tests = data.get("available_tests", [])
    if available_tests:
        selected_test = st.selectbox("Select a test to view trends:", available_tests)
        if selected_test:
            trend_data = get_test_trend(selected_test)
            if trend_data:
                df_trend = pd.DataFrame(trend_data)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_trend["date"],
                    y=df_trend["avg_value"],
                    mode="lines+markers",
                    name="Average Value",
                    line=dict(color="#6366f1", width=3),
                    marker=dict(size=8),
                ))
                if "abnormal_count" in df_trend.columns:
                    fig.add_trace(go.Bar(
                        x=df_trend["date"],
                        y=df_trend["abnormal_count"],
                        name="Abnormal Count",
                        marker=dict(color="rgba(239, 68, 68, 0.3)"),
                        yaxis="y2",
                    ))
                fig.update_layout(
                    height=400,
                    margin=dict(l=0, r=0, t=10, b=0),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter"),
                    xaxis_title="Date",
                    yaxis_title="Average Value",
                    yaxis2=dict(
                        title="Abnormal Count",
                        overlaying="y",
                        side="right",
                    ),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"No trend data available for {selected_test}.")
    else:
        st.info("Upload reports first to enable trend analysis.")

    # --- Predictive Forecasting ---
    st.markdown("---")
    st.markdown("### 🔮 Predictive Health Forecasting")
    st.caption(
        "AI-powered trend projection to help predict outcomes and allocate resources proactively. "
        "Based on linear regression over historical abnormal rates."
    )

    forecast_col1, forecast_col2 = st.columns([2, 1])

    with forecast_col1:
        # Test-specific forecast
        forecast_test = st.selectbox(
            "Forecast abnormal rate for:",
            ["All Tests"] + (available_tests if available_tests else []),
            key="forecast_test",
        )
        days_ahead = st.slider("Forecast horizon (days):", 7, 90, 30, key="forecast_days")

        test_filter = None if forecast_test == "All Tests" else forecast_test
        forecast = forecast_abnormal_trend(test_filter, days_ahead)

        if forecast.get("forecast_data"):
            st.markdown(f"**{forecast['message']}**")

            # Build combined historical + forecast chart
            hist_dates = [h["date"] for h in forecast["historical_data"]]
            hist_rates = [h["rate"] for h in forecast["historical_data"]]
            fc_dates = [f["date"] for f in forecast["forecast_data"]]
            fc_rates = [f["projected_rate"] for f in forecast["forecast_data"]]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist_dates, y=hist_rates,
                mode="lines+markers", name="Historical",
                line=dict(color="#6366f1", width=3),
                marker=dict(size=6),
            ))
            fig.add_trace(go.Scatter(
                x=fc_dates, y=fc_rates,
                mode="lines", name="Forecast",
                line=dict(color="#f59e0b", width=3, dash="dash"),
            ))
            # Vertical line at forecast start
            if hist_dates:
                fig.add_vline(
                    x=hist_dates[-1], line_dash="dot",
                    line_color="rgba(255,255,255,0.3)",
                    annotation_text="Forecast Start",
                )
            fig.update_layout(
                height=350,
                margin=dict(l=0, r=0, t=10, b=0),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
                xaxis_title="Date",
                yaxis_title="Abnormal Rate (%)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(forecast.get("message", "Upload more reports across different days to enable forecasting."))

    with forecast_col2:
        # Regional risk forecast
        st.markdown("#### 📍 Regional Risk Forecast")
        regional_forecasts = get_risk_forecast_by_region(days_ahead)
        if regional_forecasts:
            for rf in regional_forecasts:
                risk_emoji = "🔴" if rf["risk_level"] == "high" else "🟡" if rf["risk_level"] == "medium" else "🟢"
                trend_arrow = "↑" if rf["trend_direction"] == "increasing" else "↓" if rf["trend_direction"] == "decreasing" else "→"
                st.markdown(
                    f"{risk_emoji} **{rf['region']}**: {rf['current_rate']}% → {rf['projected_rate']}% {trend_arrow}"
                )
        else:
            st.info("No regional data for forecasting yet.")

    # --- Natural Language Query ---
    st.markdown("---")
    st.markdown("### 🗣️ Ask About Community Health")
    st.caption("Ask questions like: *'Which region has the most abnormal blood sugar readings?'*")

    # Initialize community chat
    if "community_chat" not in st.session_state:
        st.session_state.community_chat = []

    for msg in st.session_state.community_chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Suggested queries
    st.markdown("**Suggested Questions:**")
    sq_cols = st.columns(4)
    sq_query = None
    if sq_cols[0].button("🏘️ Regional Trends", key="sq_region"):
        sq_query = "Which locality or region has the highest rate of abnormal lab values?"
    if sq_cols[1].button("📊 Top Health Concerns", key="sq_top"):
        sq_query = "What are the top 3 health concerns based on the lab data from all reports?"
    if sq_cols[2].button("👶 Age Group Risks", key="sq_age"):
        sq_query = "Which age group shows the most concerning health indicators?"
    if sq_cols[3].button("🔮 Predict Outcomes", key="sq_predict"):
        sq_query = "Based on current trends, which health indicators are predicted to worsen in the next 30 days and which regions need urgent intervention?"

    community_input = st.chat_input("Ask a community health question...", key="community_chat_input")
    cq = sq_query or community_input

    if cq:
        st.session_state.community_chat.append({"role": "user", "content": cq})
        with st.chat_message("user"):
            st.markdown(cq)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing community data..."):
                # FIX #7: Guard community question call against exceptions
                try:
                    result = answer_community_question(cq)
                    answer_text = result["answer"]
                except Exception:
                    answer_text = (
                        "⚠️ Unable to analyze community data at this time. "
                        "Please try again in a moment."
                    )
                st.markdown(answer_text)

        st.session_state.community_chat.append({"role": "assistant", "content": answer_text})


def _render_empty_dashboard():
    """Render empty state for the community dashboard."""
    st.markdown("---")
    st.markdown(
        """<div style="text-align: center; padding: 60px 20px; opacity: 0.7;">
            <h2>📊 No Community Data Yet</h2>
            <p>Upload medical reports above to populate the community health dashboard.</p>
            <p style="font-size: 0.85rem;">
                The dashboard will show aggregate trends, anomaly patterns,<br/>
                and actionable insights once reports are processed.
            </p>
        </div>""",
        unsafe_allow_html=True,
    )
