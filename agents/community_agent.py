"""
Community Agent — handles aggregate/community-level queries.

Translates natural language questions from health workers into
SQL-backed aggregate analysis. Returns data for dashboard visualization
and narrative summaries of community health trends.
"""

import os
from groq import Groq
from dotenv import load_dotenv

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

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LLM_MODEL, GROQ_API_KEY

load_dotenv()


def _get_groq_client() -> Groq:
    """Get Groq client with API key and timeout."""
    api_key = GROQ_API_KEY or os.getenv("GROQ_API_KEY")
    return Groq(api_key=api_key, timeout=30)


def get_dashboard_data(time_period: str = None) -> dict:
    """
    Get all data needed to render the community dashboard.

    Args:
        time_period: Optional ISO date string to filter.

    Returns:
        Dict with all dashboard metrics, charts data, and alerts.
    """
    return {
        "metrics": {
            "total_reports": get_total_reports(),
            "total_lab_values": get_total_lab_values(),
            "abnormal_rate": round(get_abnormal_rate(), 1),
        },
        "top_abnormal": get_top_abnormal_tests(n=10, time_period=time_period),
        "flag_distribution": get_flag_distribution(time_period=time_period),
        "region_summary": get_region_summary(time_period=time_period),
        "age_group_summary": get_age_group_summary(time_period=time_period),
        "alerts": [
            {
                "severity": a.severity,
                "message": a.message,
                "test_name": a.test_name,
                "percentage": a.percentage,
            }
            for a in generate_community_alerts(time_period=time_period)
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
    client = _get_groq_client()

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

    prompt = f"""You are a Community Health Intelligence Assistant helping health workers and clinic administrators understand population-level health trends.

{data_context}

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
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response.choices[0].message.content
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
