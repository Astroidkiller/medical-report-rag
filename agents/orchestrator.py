"""
Multi-Agent Orchestrator — ADK-style pipeline coordinator.

Coordinates four specialized agents in a structured pipeline:
  1. Extraction Agent  → PDF parsing, lab value extraction
  2. Risk Agent        → anomaly flagging, risk card generation
  3. QA Agent          → context retrieval, LLM Q&A
  4. Community Agent   → aggregate analytics, population trends

Designed to mirror the Google Agent Development Kit (ADK) pattern
for clean production migration.
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.extraction_agent import ingest_report
from agents.risk_agent import generate_risk_card, generate_risk_explanation
from agents.qa_agent import answer_patient_question
from agents.community_agent import get_dashboard_data, answer_community_question
from core.anomaly_detector import (
    detect_population_anomalies,
    detect_seasonal_spikes,
    detect_demographic_clusters,
    PopulationAnomaly,
)
from data_store.sqlite_store import (
    get_all_lab_records,
    get_lab_records_by_period,
)

logger = logging.getLogger(__name__)


# ============================================================
# Session — shared state across agent invocations
# ============================================================


@dataclass
class Session:
    """Shared state for a multi-agent pipeline run."""
    session_id: str = ""
    mode: str = "community"                # "patient" or "community"
    report_ids: list[str] = field(default_factory=list)
    ingestion_results: list[dict] = field(default_factory=list)
    risk_cards: list[dict] = field(default_factory=list)
    population_anomalies: list = field(default_factory=list)
    collection_name: str = "medical_report"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


# ============================================================
# Pipeline Steps
# ============================================================


def step_ingest(
    session: Session,
    file_path: str,
    filename: str,
    region: str = None,
    age_group: str = None,
) -> dict:
    """
    Step 1: Extraction Agent — parse PDF and ingest into data stores.

    Args:
        session: Current session state.
        file_path: Path to the uploaded PDF.
        filename: Original filename.
        region: Anonymized region (simulated if None).
        age_group: Age group label (simulated if None).

    Returns:
        Ingestion result dict from extraction_agent.
    """
    result = ingest_report(
        file_path=file_path,
        filename=filename,
        mode=session.mode,
        collection_name=session.collection_name,
        anonymized_region=region,
        age_group=age_group,
    )

    session.report_ids.append(result["report_id"])
    session.ingestion_results.append(result)
    return result


def step_risk_analysis(session: Session, ingestion_result: dict) -> dict:
    """
    Step 2: Risk Agent — generate risk card and explanation.

    Args:
        session: Current session state.
        ingestion_result: Output from step_ingest.

    Returns:
        Risk card dict with 'card' and 'explanation' keys.
    """
    flagged = ingestion_result["flagged_values"]
    risk_summary = ingestion_result["risk_summary"]

    card = generate_risk_card(flagged, risk_summary)

    # Only call LLM for explanation if there are findings
    if card["abnormal_count"] > 0 or card["critical_count"] > 0:
        try:
            explanation = generate_risk_explanation(card)
        except Exception:
            logger.exception("Risk explanation generation failed")
            explanation = "⚠️ Unable to generate risk explanation."
    else:
        explanation = "✅ All tested values are within normal ranges."

    result = {
        "card": card,
        "explanation": explanation,
    }
    session.risk_cards.append(result)
    return result


def step_qa(
    session: Session,
    query: str,
    full_text: str = None,
    stream: bool = False,
) -> dict:
    """
    Step 3: QA Agent — answer a patient's question using RAG.

    Args:
        session: Current session state.
        query: The patient's question.
        full_text: Optional full report text for context override.
        stream: If True, returns streaming response.

    Returns:
        QA result dict with 'answer', 'source_chunks', 'source_metadata'.
    """
    return answer_patient_question(
        query=query,
        collection_name=session.collection_name,
        full_text_override=full_text,
        stream=stream,
    )


def step_community_analysis(
    session: Session,
    query: str = None,
    time_period: str = None,
) -> dict:
    """
    Step 4: Community Agent — aggregate analytics and population insights.

    If a query is provided, answers it using aggregate data.
    Otherwise returns the full dashboard data with population anomalies.

    Args:
        session: Current session state.
        query: Optional natural-language question from health worker.
        time_period: Optional ISO date filter.

    Returns:
        Dict with dashboard data, population anomalies, and optionally
        the community agent's narrative answer.
    """
    dashboard = get_dashboard_data(time_period)

    # Run population-level anomaly detection
    all_lab_data = get_all_lab_records(time_period)

    pop_anomalies = detect_population_anomalies(all_lab_data)
    demographic_clusters = detect_demographic_clusters(all_lab_data)

    # Seasonal spike detection: compare recent 30 days to previous 30 days
    seasonal_spikes = []
    try:
        from datetime import datetime as _dt, timedelta as _td
        now = _dt.now()
        current_start = (now - _td(days=30)).isoformat()
        historical_end = current_start
        historical_start = (now - _td(days=60)).isoformat()
        current_data = get_lab_records_by_period(current_start, now.isoformat())
        historical_data = get_lab_records_by_period(historical_start, historical_end)
        if current_data and historical_data:
            seasonal_spikes = detect_seasonal_spikes(current_data, historical_data)
    except Exception:
        logger.exception("Seasonal spike detection failed")

    session.population_anomalies = pop_anomalies + demographic_clusters + seasonal_spikes

    result = {
        "dashboard": dashboard,
        "population_anomalies": [
            {
                "type": a.anomaly_type,
                "test_name": a.test_name,
                "metric": a.metric,
                "severity": a.severity,
                "message": a.message,
                "region": a.region,
                "age_group": a.age_group,
            }
            for a in session.population_anomalies
        ],
    }

    if query:
        community_response = answer_community_question(query, time_period)
        result["answer"] = community_response["answer"]
        result["data_summary"] = community_response["data_summary"]

    return result


# ============================================================
# Full Pipeline Runners
# ============================================================


def run_patient_pipeline(
    file_path: str,
    filename: str,
    query: str = None,
    region: str = None,
    age_group: str = None,
) -> dict:
    """
    Run the full patient pipeline: Ingest → Risk → (optional) QA.

    Args:
        file_path: Path to PDF.
        filename: Original filename.
        query: Optional patient question.
        region: Anonymized region.
        age_group: Age group.

    Returns:
        Dict with ingestion, risk_card, and optionally qa_response.
    """
    session = Session(mode="patient")

    # Step 1: Ingest
    ingestion = step_ingest(session, file_path, filename, region, age_group)

    # Step 2: Risk analysis
    risk = step_risk_analysis(session, ingestion)

    result = {
        "session": session,
        "ingestion": ingestion,
        "risk": risk,
    }

    # Step 3: Optional QA
    if query:
        qa = step_qa(session, query, full_text=ingestion.get("raw_text"))
        result["qa"] = qa

    return result


def run_community_pipeline(
    file_paths: list[tuple[str, str]] = None,
    query: str = None,
    time_period: str = None,
) -> dict:
    """
    Run the community pipeline: Batch Ingest → Population Analysis → Query.

    Args:
        file_paths: Optional list of (file_path, filename) tuples to ingest.
        query: Optional community health worker question.
        time_period: Optional ISO date filter.

    Returns:
        Dict with ingestions, population analysis, and optionally answer.
    """
    session = Session(mode="community")

    result = {
        "session": session,
        "ingestions": [],
        "risk_cards": [],
    }

    # Step 1 & 2: Batch ingest + risk analysis
    if file_paths:
        for fp, fn in file_paths:
            # Separate try blocks: ingestion success is recorded even if risk fails
            ingestion = None
            try:
                ingestion = step_ingest(session, fp, fn)
            except Exception as e:
                logger.exception("Ingestion failed for %s", fn)
                result["ingestions"].append({"error": str(e), "filename": fn})
                result["risk_cards"].append(None)
                continue

            result["ingestions"].append(ingestion)

            try:
                risk = step_risk_analysis(session, ingestion)
                result["risk_cards"].append(risk)
            except Exception as e:
                logger.exception("Risk analysis failed for %s", fn)
                result["risk_cards"].append({"error": str(e), "filename": fn})

    # Step 4: Community analysis
    community = step_community_analysis(session, query, time_period)
    result["community"] = community

    return result
