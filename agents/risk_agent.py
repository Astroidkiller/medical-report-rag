"""
Risk Agent — automatic risk analysis and risk card generation.

Summarizes flagged values, generates severity-based risk cards,
and provides plain-language explanations of findings.
"""

import os
from groq import Groq
from dotenv import load_dotenv

from core.anomaly_detector import FlaggedValue, FLAG_NORMAL, FLAG_UNKNOWN

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LLM_MODEL, GROQ_API_KEY

load_dotenv()

_LLM_TIMEOUT = 30


def _get_groq_client() -> Groq:
    """Get Groq client with API key and timeout."""
    api_key = GROQ_API_KEY or os.getenv("GROQ_API_KEY")
    return Groq(api_key=api_key, timeout=_LLM_TIMEOUT)


def generate_risk_card(flagged_values: list[FlaggedValue], risk_summary: dict) -> dict:
    """
    Generate a structured risk card from flagged values.

    FIX #14 (partial): Uses `is not None` instead of falsy check on
    reference_low so a legitimate 0 is displayed correctly.

    Args:
        flagged_values: List of FlaggedValue objects from anomaly detector.
        risk_summary: Risk summary dict from anomaly detector.

    Returns:
        Dict with risk_level, color, icon, headline, findings (grouped by severity).
    """
    score = risk_summary["risk_score"]

    if risk_summary["critical"] > 0:
        risk_level = "Critical"
        color = "#dc2626"
        icon = "🔴"
        headline = f"{risk_summary['critical']} critical finding(s) require immediate attention"
    elif score > 0.3:
        risk_level = "Elevated"
        color = "#f59e0b"
        icon = "🟡"
        headline = f"{risk_summary['abnormal']} abnormal value(s) detected"
    elif score > 0:
        risk_level = "Mild"
        color = "#3b82f6"
        icon = "🔵"
        headline = f"{risk_summary['abnormal']} value(s) slightly outside range"
    else:
        risk_level = "Normal"
        color = "#22c55e"
        icon = "🟢"
        headline = "All tested values are within normal range"

    # Group findings by severity
    critical_findings = []
    abnormal_findings = []
    normal_findings = []

    for fv in flagged_values:
        if fv.flag == FLAG_UNKNOWN:
            continue
        # FIX #14: use `is not None` to preserve a legitimate zero ref_low
        ref_str = (
            f"{fv.reference_low}-{fv.reference_high}"
            if fv.reference_low is not None
            else "N/A"
        )
        entry = {
            "test_name": fv.test_name,
            "value": fv.value,
            "unit": fv.unit,
            "flag": fv.flag,
            "reference": ref_str,
            "explanation": fv.explanation,
        }
        if fv.severity == 2:
            critical_findings.append(entry)
        elif fv.severity == 1:
            abnormal_findings.append(entry)
        else:
            normal_findings.append(entry)

    return {
        "risk_level": risk_level,
        "risk_score": round(score, 2),
        "color": color,
        "icon": icon,
        "headline": headline,
        "total_tests": risk_summary["total"],
        "normal_count": risk_summary["normal"],
        "abnormal_count": risk_summary["abnormal"],
        "critical_count": risk_summary["critical"],
        "critical_findings": critical_findings,
        "abnormal_findings": abnormal_findings,
        "normal_findings": normal_findings,
    }


def generate_risk_explanation(risk_card: dict) -> str:
    """
    Use LLM to generate a patient-friendly explanation of the risk card.

    FIX #17: Added error handling — returns user-friendly message
    on failure instead of raw exception text.

    Args:
        risk_card: Risk card dict from generate_risk_card.

    Returns:
        Plain-language risk explanation string.
    """
    client = _get_groq_client()

    # Build a concise context from the risk card
    findings_text = ""
    if risk_card["critical_findings"]:
        findings_text += "CRITICAL:\n"
        for f in risk_card["critical_findings"]:
            findings_text += f"- {f['test_name']}: {f['value']} {f['unit']} (ref: {f['reference']}) — {f['explanation']}\n"

    if risk_card["abnormal_findings"]:
        findings_text += "\nABNORMAL:\n"
        for f in risk_card["abnormal_findings"]:
            findings_text += f"- {f['test_name']}: {f['value']} {f['unit']} (ref: {f['reference']}) — {f['explanation']}\n"

    if not findings_text:
        return "✅ All your test results are within the expected normal ranges. No immediate concerns were identified."

    prompt = f"""Based on these medical report findings, provide a brief, reassuring but honest explanation for the patient.
Keep it to 3-4 sentences. Don't use overly alarming language but be clear about what needs attention.

Risk Level: {risk_card['risk_level']}
{findings_text}

End with a recommendation about which type of specialist to consult, if applicable.
Always include the disclaimer that this is AI-generated and not medical advice."""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception:
        return (
            "⚠️ Unable to generate a detailed risk explanation at this time. "
            "Please review the risk card above for a summary of findings, "
            "and consult a healthcare professional for interpretation.\n\n"
            "**⚕️ DISCLAIMER: This is for informational purposes only.**"
        )
