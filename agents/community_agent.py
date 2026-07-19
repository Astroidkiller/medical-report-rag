"""
Community Agent — handles aggregate/community-level queries.

Translates natural language questions from health workers into
SQL-backed aggregate analysis. Returns data for dashboard visualization
and narrative summaries of community health trends.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.llm_client import generate

from data_store.sqlite_store import (
    get_top_abnormal_tests,
    get_flag_distribution,
    get_test_trend,
    get_region_summary,
    get_age_group_summary,
    get_total_reports,
    get_total_lab_values,
    get_abnormal_rate,
    generate_community_alerts,
    get_all_test_names,
    forecast_abnormal_trend,
    get_risk_forecast_by_region,
)


def get_dashboard_data(time_period: str = None, use_dp: bool = True) -> dict:
    """
    Get all data needed to render the community dashboard.

    Args:
        time_period: Optional ISO date string to filter.
        use_dp: Whether to apply Differential Privacy.

    Returns:
        Dict with all dashboard metrics, charts data, and alerts.
    """
    return {
        "metrics": {
            "total_reports": get_total_reports(use_dp=use_dp),
            "total_lab_values": get_total_lab_values(use_dp=use_dp),
            "abnormal_rate": round(get_abnormal_rate(use_dp=use_dp), 1),
        },
        "top_abnormal": get_top_abnormal_tests(n=10, time_period=time_period),
        "flag_distribution": get_flag_distribution(time_period=time_period),
        "region_summary": get_region_summary(time_period=time_period, use_dp=use_dp),
        "age_group_summary": get_age_group_summary(time_period=time_period, use_dp=use_dp),
        "alerts": [
            {
                "severity": a.severity,
                "message": a.message,
                "test_name": a.test_name,
                "percentage": a.percentage,
                "region": a.region,
                "age_group": a.age_group,
                "action_recommendation": (
                    f"Deploy a mobile diabetes screening van to {a.region} and prioritize HbA1c testing for the {a.age_group} cohort."
                    if a.test_name == "HbA1c" and a.region
                    else f"Initiate cardiovascular screening programs and healthy lifestyle workshops in {a.region}."
                    if a.test_name == "Cholesterol" and a.region
                    else f"Launch an iron-deficiency anemia screening and nutritional support drive in {a.region}."
                    if a.test_name == "Hemoglobin" and a.region
                    else f"Organize thyroid health checkups and diagnostic counseling services in {a.region}."
                    if a.test_name == "TSH" and a.region
                    else f"Consider organizing a targeted diabetes screening camp in the community."
                    if not a.region and a.test_name == "HbA1c"
                    else f"Organize a free lipid profiling and cardiovascular health camp."
                    if not a.region and a.test_name == "Cholesterol"
                    else f"Distribute iron supplements and organize nutritional counseling."
                    if not a.region and a.test_name == "Hemoglobin"
                    else f"Arrange a thyroid disorder awareness and diagnostic screening camp."
                    if not a.region and a.test_name == "TSH"
                    else "Monitor local intake logs and screen the target population for active clinical symptoms."
                )
            }
            for a in generate_community_alerts(time_period=time_period, use_dp=use_dp)
        ],
        "available_tests": get_all_test_names(),
        "forecast": forecast_abnormal_trend(days_ahead=30),
        "regional_forecast": get_risk_forecast_by_region(days_ahead=30),
    }


def answer_community_question(query: str, time_period: str = None) -> dict:
    """
    Answer a community health worker's aggregate question.

    Uses the LLM to interpret the question and generate a narrative
    response based on available aggregate data.

    Args:
        query: Natural language question from a health worker.
        time_period: Optional time filter.

    Returns:
        Dict with 'answer' (narrative text), 'data' (supporting data),
        and 'data_summary' (formatted data context).
    """
    # Gather relevant data
    data = get_dashboard_data(time_period)

    # Build data context for the LLM
    data_context = f"""COMMUNITY HEALTH DATA SUMMARY:

📊 Overview:
- Total reports analyzed: {data['metrics']['total_reports']}
- Total lab values recorded: {data['metrics']['total_lab_values']}
- Overall abnormal rate: {data['metrics']['abnormal_rate']}%

🔍 Top Abnormal Tests:
"""
    for item in data["top_abnormal"][:10]:
        data_context += f"- {item['test_name']}: {item['flag_count']} abnormal ({item['percentage']}%)\n"

    data_context += "\n📍 By Region:\n"
    for item in data["region_summary"]:
        data_context += f"- {item['region']}: {item['abnormal']}/{item['total']} abnormal ({item['percentage']}%)\n"

    data_context += "\n👥 By Age Group:\n"
    for item in data["age_group_summary"]:
        data_context += f"- {item['age_group']}: {item['abnormal']}/{item['total']} abnormal ({item['percentage']}%)\n"

    if data["alerts"]:
        data_context += "\n🚨 Active Alerts:\n"
        for alert in data["alerts"]:
            data_context += f"- [{alert['severity'].upper()}] {alert['message']}\n"

    # Predictive forecast context
    forecast = data.get("forecast", {})
    if forecast.get("forecast_data"):
        data_context += f"\n🔮 Predictive Forecast (30-day):\n"
        data_context += f"- {forecast['message']}\n"

    regional_fc = data.get("regional_forecast", [])
    if regional_fc:
        data_context += "\n📍 Regional Risk Forecast (30-day projection):\n"
        for rf in regional_fc:
            data_context += (f"- {rf['region']}: Current {rf['current_rate']}% → "
                           f"Projected {rf['projected_rate']}% ({rf['trend_direction']}, "
                           f"risk: {rf['risk_level']})\n")

    system_prompt = "You are a Community Health Intelligence Assistant helping health workers and clinic administrators understand population-level health trends."

    prompt = f"""{data_context}

Health Worker's Question:
{query}

Instructions:
1. Answer the question based on the community health data above.
2. Reference specific numbers and percentages from the data.
3. Highlight any concerning trends or patterns.
4. Suggest actionable next steps for the health worker (e.g., "Consider organizing a diabetes screening camp in Region X").
5. Keep the tone professional but accessible.
6. If the data doesn't contain enough information to fully answer the question, say so.
7. End with: "**📊 This analysis is based on aggregated, anonymized lab data and is intended for public health planning purposes.**"
"""

    try:
        answer = generate(prompt, system_prompt=system_prompt)
    except Exception:
        answer = (
            "⚠️ Unable to generate community analysis at this time. "
            "Please try again in a moment. The dashboard data above "
            "is still available for manual review."
        )

    return {
        "answer": answer,
        "data": data,
        "data_summary": data_context,
    }
