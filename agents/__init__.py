"""
Agents package for Community Health Intelligence Assistant.

Multi-agent architecture (ADK-style):
  - orchestrator: Pipeline coordinator
  - extraction_agent: PDF parsing, lab extraction
  - risk_agent: Anomaly flagging, risk cards
  - qa_agent: RAG-based Q&A
  - community_agent: Population analytics
"""

from agents.orchestrator import (
    Session,
    run_patient_pipeline,
    run_community_pipeline,
    step_ingest,
    step_risk_analysis,
    step_qa,
    step_community_analysis,
)

__all__ = [
    "Session",
    "run_patient_pipeline",
    "run_community_pipeline",
    "step_ingest",
    "step_risk_analysis",
    "step_qa",
    "step_community_analysis",
]
