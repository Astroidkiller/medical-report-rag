"""
Data models for the Community Health Intelligence Assistant.
Dataclasses for structured lab values, report metadata, and community records.
"""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class LabValueRecord:
    """A structured lab value stored in the database for aggregate analysis."""
    id: str
    report_id: str
    test_name: str          # e.g., "Hemoglobin"
    value: float
    unit: str               # e.g., "g/dL"
    reference_low: float | None
    reference_high: float | None
    flag: str               # NORMAL / HIGH / LOW / CRITICAL_HIGH / CRITICAL_LOW / UNKNOWN
    severity: int            # 0=normal, 1=mild, 2=critical
    timestamp: str          # Report upload date (ISO format)
    anonymized_region: str  # e.g., "Urban-Central" (simulated)
    age_group: str          # e.g., "31-45" (simulated)


@dataclass
class ReportRecord:
    """Metadata for an uploaded medical report."""
    id: str
    filename: str
    upload_timestamp: str   # ISO format
    total_tests: int
    normal_count: int
    abnormal_count: int
    critical_count: int
    risk_score: float
    anonymized_region: str
    age_group: str
    mode: str               # "patient" or "community"


@dataclass
class CommunityAlert:
    """A flagged community-level health pattern."""
    id: str
    alert_type: str         # e.g., "elevated_hba1c", "low_hemoglobin"
    test_name: str
    percentage: float       # % of reports showing this anomaly
    total_reports: int
    affected_reports: int
    time_period: str        # e.g., "2026-07", "last_30_days"
    region: str | None
    age_group: str | None
    severity: str           # "warning" or "critical"
    message: str            # Human-readable alert message
    generated_at: str       # ISO format
