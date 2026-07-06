"""
Risk Agent — automatic risk analysis and risk card generation.

Summarizes flagged values, generates severity-based risk cards,
and provides plain-language explanations of findings.
"""

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.anomaly_detector import FlaggedValue, FLAG_NORMAL, FLAG_UNKNOWN
from core.llm_client import generate

logger = logging.getLogger(__name__)


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

    prompt = f"""You are explaining medical test results to a patient who has NO medical background.

RULES — follow these strictly:
1. Use very simple, everyday language — imagine you are explaining to a friend over coffee.
2. NEVER use medical jargon. If you must mention a test name, immediately explain what it checks in plain words (e.g., "Your HbA1c — this checks your average blood sugar over the past few months — is a bit high").
3. Use short sentences. Keep paragraphs to 2-3 sentences max.
4. Be honest but kind — do not scare the patient, but do not hide important information either.
5. Use a warm, supportive tone like a caring family doctor would.
6. Write at a 5th-grade reading level.
7. Do NOT use any emojis anywhere in your response.

FORMAT your response EXACTLY like this (use markdown headings and horizontal rules for clear separation):

## Overview

A brief 1-2 sentence summary of what tests were done and the overall result.

---

## What Needs Your Attention

For each abnormal or critical result, explain in bullet points:
- What the test checks (in plain words)
- What your result means
- Why it matters for your health

---

## What Looks Good

Briefly mention the normal results to reassure the patient.

---

## Suggested Next Steps

Simple, actionable advice — which type of doctor to see, written in plain words like "a heart doctor" instead of "cardiologist".

---

## Disclaimer

This is an AI-generated summary for informational purposes only. It is not a medical diagnosis. Please consult your doctor for proper medical advice.

---
Here are the findings:

Risk Level: {risk_card['risk_level']}
{findings_text}"""

    try:
        return generate(prompt)
    except Exception:
        logger.exception("Risk explanation generation failed")
        return (
            "⚠️ Unable to generate a detailed risk explanation at this time. "
            "Please review the risk card above for a summary of findings, "
            "and consult a healthcare professional for interpretation.\n\n"
            "**⚕️ DISCLAIMER: This is for informational purposes only.**"
        )
