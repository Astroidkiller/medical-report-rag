"""
Anomaly detection module.
Flags lab values against reference ranges at ingestion time.
Supports both report-extracted ranges and built-in reference ranges.
"""

from dataclasses import dataclass
from core.lab_value_parser import ExtractedLabValue

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import REFERENCE_RANGES


# Severity levels
FLAG_NORMAL = "NORMAL"
FLAG_HIGH = "HIGH"
FLAG_LOW = "LOW"
FLAG_CRITICAL_HIGH = "CRITICAL_HIGH"
FLAG_CRITICAL_LOW = "CRITICAL_LOW"
FLAG_UNKNOWN = "UNKNOWN"  # Cannot determine (no reference range available)


@dataclass
class FlaggedValue:
    """A lab value with anomaly flag attached."""
    test_name: str
    value: float
    unit: str
    reference_low: float | None
    reference_high: float | None
    flag: str          # One of FLAG_* constants
    severity: int      # 0=normal, 1=mild, 2=critical
    raw_line: str
    explanation: str   # Brief auto-generated explanation


def _lookup_reference(test_name: str) -> tuple | None:
    """
    Look up a test name in the built-in reference ranges.
    Tries exact match first, then word-boundary-aware partial matching.

    FIX #18: Use word-boundary matching instead of substring `in` check
    to prevent e.g. "iron" matching "ferritin" or "t3" matching "ft3".

    Returns:
        Tuple of (low, high, unit, critical_low, critical_high) or None.
    """
    normalized = test_name.lower().strip()

    # Exact match (preferred)
    if normalized in REFERENCE_RANGES:
        return REFERENCE_RANGES[normalized]

    # Word-boundary partial match: the query must contain the key as
    # a complete word sequence, or the key must contain the query as one.
    # We only accept matches where the key equals a whitespace-delimited
    # token sequence within normalized (or vice-versa).
    best_match = None
    best_len = 0
    for key in REFERENCE_RANGES:
        # Check if key appears as whole-word(s) inside normalized
        # e.g. key="hemoglobin" in normalized="glycated hemoglobin" → ok
        # e.g. key="iron" in normalized="ferritin" → NOT ok
        if _is_word_match(key, normalized):
            if len(key) > best_len:
                best_match = key
                best_len = len(key)

    if best_match:
        return REFERENCE_RANGES[best_match]

    return None


def _is_word_match(key: str, text: str) -> bool:
    """Check if key appears as a whole-word subsequence in text or vice versa."""
    import re
    # Escape the key for regex, then check word boundaries
    pattern = r"\b" + re.escape(key) + r"\b"
    if re.search(pattern, text):
        return True
    # Also check reverse: text as whole word in key
    pattern_rev = r"\b" + re.escape(text) + r"\b"
    if re.search(pattern_rev, key):
        return True
    return False


def flag_lab_value(lab: ExtractedLabValue) -> FlaggedValue:
    """
    Flag a single lab value against reference ranges.

    Priority:
    1. Use report-extracted reference ranges if available
    2. Fall back to built-in reference ranges from config
    3. Mark as UNKNOWN if no ranges found

    FIX #11: Use `is None` instead of falsy checks so that
    a legitimate reference_low of 0 is not treated as missing.

    Args:
        lab: An extracted lab value.

    Returns:
        FlaggedValue with anomaly flag and severity.
    """
    ref_low = lab.reference_low
    ref_high = lab.reference_high
    critical_low = None
    critical_high = None

    # If no ranges from the report, look up built-in
    if ref_low is None or ref_high is None:
        lookup = _lookup_reference(lab.test_name)
        if lookup:
            built_low, built_high, _, built_crit_low, built_crit_high = lookup
            if ref_low is None:
                ref_low = built_low
            if ref_high is None:
                ref_high = built_high
            critical_low = built_crit_low
            critical_high = built_crit_high
    else:
        # We have report ranges, still check built-in for critical thresholds
        lookup = _lookup_reference(lab.test_name)
        if lookup:
            _, _, _, critical_low, critical_high = lookup

    # Determine flag — use `is None` not falsy check
    if ref_low is None or ref_high is None:
        return FlaggedValue(
            test_name=lab.test_name,
            value=lab.value,
            unit=lab.unit,
            reference_low=ref_low,
            reference_high=ref_high,
            flag=FLAG_UNKNOWN,
            severity=0,
            raw_line=lab.raw_line,
            explanation=f"No reference range available for {lab.test_name}.",
        )

    flag = FLAG_NORMAL
    severity = 0
    explanation = f"{lab.test_name}: {lab.value} is within normal range ({ref_low}-{ref_high})."

    if lab.value < ref_low:
        if critical_low is not None and lab.value <= critical_low:
            flag = FLAG_CRITICAL_LOW
            severity = 2
            explanation = (
                f"⚠️ CRITICAL: {lab.test_name} is critically low at {lab.value} "
                f"(normal: {ref_low}-{ref_high}, critical threshold: {critical_low})."
            )
        else:
            flag = FLAG_LOW
            severity = 1
            explanation = (
                f"{lab.test_name} is below normal at {lab.value} "
                f"(normal range: {ref_low}-{ref_high})."
            )
    elif lab.value > ref_high:
        if critical_high is not None and lab.value >= critical_high:
            flag = FLAG_CRITICAL_HIGH
            severity = 2
            explanation = (
                f"⚠️ CRITICAL: {lab.test_name} is critically high at {lab.value} "
                f"(normal: {ref_low}-{ref_high}, critical threshold: {critical_high})."
            )
        else:
            flag = FLAG_HIGH
            severity = 1
            explanation = (
                f"{lab.test_name} is above normal at {lab.value} "
                f"(normal range: {ref_low}-{ref_high})."
            )

    return FlaggedValue(
        test_name=lab.test_name,
        value=lab.value,
        unit=lab.unit,
        reference_low=ref_low,
        reference_high=ref_high,
        flag=flag,
        severity=severity,
        raw_line=lab.raw_line,
        explanation=explanation,
    )


def flag_all_values(lab_values: list[ExtractedLabValue]) -> list[FlaggedValue]:
    """
    Flag all lab values from a report.

    Args:
        lab_values: List of extracted lab values.

    Returns:
        List of FlaggedValue objects, sorted by severity (critical first).
    """
    flagged = [flag_lab_value(lab) for lab in lab_values]
    # Sort: critical first, then abnormal, then normal
    flagged.sort(key=lambda f: (-f.severity, f.test_name))
    return flagged


def compute_risk_score(flagged_values: list[FlaggedValue]) -> float:
    """
    Compute a simple risk score for a report.

    Score = (2 * critical_count + 1 * abnormal_count) / total_known_count
    Returns 0.0 if no values could be assessed.

    Args:
        flagged_values: List of flagged values from a report.

    Returns:
        Risk score between 0.0 and 1.0+ (can exceed 1.0 if many criticals).
    """
    known = [f for f in flagged_values if f.flag != FLAG_UNKNOWN]
    if not known:
        return 0.0

    critical = sum(1 for f in known if f.severity == 2)
    abnormal = sum(1 for f in known if f.severity == 1)

    return (2 * critical + abnormal) / len(known)


def generate_risk_summary(flagged_values: list[FlaggedValue]) -> dict:
    """
    Generate a summary of risk findings for a report.

    Returns:
        Dict with keys: total, normal, abnormal, critical, risk_score,
        top_concerns (list of explanations for flagged values).
    """
    known = [f for f in flagged_values if f.flag != FLAG_UNKNOWN]
    normal = sum(1 for f in known if f.flag == FLAG_NORMAL)
    abnormal = sum(1 for f in known if f.severity == 1)
    critical = sum(1 for f in known if f.severity == 2)

    top_concerns = [
        f.explanation for f in flagged_values
        if f.severity > 0
    ]

    return {
        "total": len(known),
        "normal": normal,
        "abnormal": abnormal,
        "critical": critical,
        "risk_score": compute_risk_score(flagged_values),
        "top_concerns": top_concerns,
    }


# ============================================================
# Population-Level Anomaly Detection
# ============================================================
# These functions operate on aggregate data from SQLite,
# surfacing community-wide health trends and demographic patterns.


@dataclass
class PopulationAnomaly:
    """A population-level anomaly detected across aggregated reports."""
    anomaly_type: str       # "elevated_rate", "seasonal_spike", "demographic_cluster"
    test_name: str
    metric: float           # The abnormal rate or spike magnitude
    threshold: float        # The threshold that was exceeded
    region: str | None
    age_group: str | None
    severity: str           # "warning" or "critical"
    message: str
    details: dict           # Supporting data for the anomaly


def _text_or_default(row: dict, key: str, default: str) -> str:
    """Return default for missing, None, or empty-string schema values."""
    value = row.get(key)
    if value is None:
        return default
    value = str(value).strip()
    return value if value else default


def detect_population_anomalies(
    lab_data: list[dict],
    threshold_pct: float = 25.0,
    critical_pct: float = 45.0,
) -> list[PopulationAnomaly]:
    """
    Detect population-level anomalies from aggregate lab value data.

    Flags tests where the abnormal rate across the population exceeds
    the given threshold, broken down by region and age group.

    Args:
        lab_data: List of dicts with keys: test_name, flag, anonymized_region, age_group.
        threshold_pct: Minimum abnormal % to trigger a warning.
        critical_pct: Minimum abnormal % to trigger a critical alert.

    Returns:
        List of PopulationAnomaly objects sorted by severity then metric.
    """
    from collections import defaultdict

    # Aggregate by test_name
    test_counts = defaultdict(lambda: {"total": 0, "abnormal": 0})
    for row in lab_data:
        key = _text_or_default(row, "test_name", "").lower()
        flag = _text_or_default(row, "flag", "UNKNOWN")
        if not key:
            continue
        if flag == "UNKNOWN":
            continue
        test_counts[key]["total"] += 1
        if flag not in ("NORMAL", "UNKNOWN"):
            test_counts[key]["abnormal"] += 1

    anomalies = []
    for test_name, counts in test_counts.items():
        if counts["total"] == 0:
            continue
        rate = (counts["abnormal"] / counts["total"]) * 100

        if rate >= threshold_pct:
            severity = "critical" if rate >= critical_pct else "warning"
            anomalies.append(PopulationAnomaly(
                anomaly_type="elevated_rate",
                test_name=test_name,
                metric=round(rate, 1),
                threshold=critical_pct if severity == "critical" else threshold_pct,
                region=None,
                age_group=None,
                severity=severity,
                message=(
                    f"{'🔴' if severity == 'critical' else '🟡'} {test_name}: "
                    f"{rate:.1f}% abnormal rate across population "
                    f"({counts['abnormal']}/{counts['total']} values)"
                ),
                details=counts,
            ))

    anomalies.sort(key=lambda a: (-{"critical": 2, "warning": 1}.get(a.severity, 0), -a.metric))
    return anomalies


def detect_seasonal_spikes(
    current_period_data: list[dict],
    historical_data: list[dict],
    spike_factor: float = 1.5,
) -> list[PopulationAnomaly]:
    """
    Detect seasonal illness spikes by comparing current period abnormal rates
    to historical averages.

    A spike is flagged when current_rate > historical_avg * spike_factor.

    Args:
        current_period_data: Lab value records from the current period.
        historical_data: Lab value records from the comparison period.
        spike_factor: Multiplier above which a spike is flagged (default 1.5x).

    Returns:
        List of PopulationAnomaly objects for detected seasonal spikes.
    """
    from collections import defaultdict

    def _compute_rates(data):
        counts = defaultdict(lambda: {"total": 0, "abnormal": 0})
        for row in data:
            key = _text_or_default(row, "test_name", "").lower()
            flag = _text_or_default(row, "flag", "UNKNOWN")
            if not key:
                continue
            if flag == "UNKNOWN":
                continue
            counts[key]["total"] += 1
            if flag not in ("NORMAL", "UNKNOWN"):
                counts[key]["abnormal"] += 1
        rates = {}
        for k, v in counts.items():
            if v["total"] > 0:
                rates[k] = (v["abnormal"] / v["total"]) * 100
        return rates

    current_rates = _compute_rates(current_period_data)
    historical_rates = _compute_rates(historical_data)

    anomalies = []
    for test_name, current_rate in current_rates.items():
        hist_rate = historical_rates.get(test_name)
        if hist_rate is None:
            continue

        # Zero baseline: any current abnormals represent a new emergence
        if hist_rate == 0:
            if current_rate > 0:
                anomalies.append(PopulationAnomaly(
                    anomaly_type="seasonal_spike",
                    test_name=test_name,
                    metric=round(current_rate, 1),
                    threshold=0.0,
                    region=None,
                    age_group=None,
                    severity="critical",
                    message=(
                        f"🆕 New emergence: {test_name} abnormal rate rose from "
                        f"0% to {current_rate:.1f}% (previously unseen)"
                    ),
                    details={
                        "current_rate": round(current_rate, 1),
                        "historical_rate": 0.0,
                        "spike_magnitude": float("inf"),
                    },
                ))
            continue

        if current_rate > hist_rate * spike_factor:
            spike_magnitude = current_rate / hist_rate
            severity = "critical" if spike_magnitude >= 2.0 else "warning"
            anomalies.append(PopulationAnomaly(
                anomaly_type="seasonal_spike",
                test_name=test_name,
                metric=round(current_rate, 1),
                threshold=round(hist_rate * spike_factor, 1),
                region=None,
                age_group=None,
                severity=severity,
                message=(
                    f"📈 Seasonal spike: {test_name} abnormal rate is "
                    f"{current_rate:.1f}% vs historical {hist_rate:.1f}% "
                    f"({spike_magnitude:.1f}x increase)"
                ),
                details={
                    "current_rate": round(current_rate, 1),
                    "historical_rate": round(hist_rate, 1),
                    "spike_magnitude": round(spike_magnitude, 2),
                },
            ))

    anomalies.sort(key=lambda a: -a.metric)
    return anomalies


def detect_demographic_clusters(
    lab_data: list[dict],
    cluster_threshold_pct: float = 35.0,
    min_samples: int = 3,
) -> list[PopulationAnomaly]:
    """
    Detect demographic clusters — specific age_group × region combinations
    with disproportionately high abnormal rates.

    Args:
        lab_data: Lab value records with region and age_group fields.
        cluster_threshold_pct: Minimum abnormal % for a cluster to be flagged.
        min_samples: Minimum number of samples in a group to be considered.

    Returns:
        List of PopulationAnomaly objects for demographic clusters.
    """
    from collections import defaultdict

    # Aggregate by (test_name, region, age_group)
    groups = defaultdict(lambda: {"total": 0, "abnormal": 0})
    for row in lab_data:
        test = _text_or_default(row, "test_name", "").lower()
        region = _text_or_default(row, "anonymized_region", "Unknown")
        age = _text_or_default(row, "age_group", "Unknown")
        flag = _text_or_default(row, "flag", "UNKNOWN")
        if not test:
            continue
        if flag == "UNKNOWN":
            continue

        key = (test, region, age)
        groups[key]["total"] += 1
        if flag not in ("NORMAL", "UNKNOWN"):
            groups[key]["abnormal"] += 1

    anomalies = []
    for (test_name, region, age_group), counts in groups.items():
        if counts["total"] < min_samples:
            continue
        rate = (counts["abnormal"] / counts["total"]) * 100

        if rate >= cluster_threshold_pct:
            severity = "critical" if rate >= 60.0 else "warning"
            anomalies.append(PopulationAnomaly(
                anomaly_type="demographic_cluster",
                test_name=test_name,
                metric=round(rate, 1),
                threshold=cluster_threshold_pct,
                region=region,
                age_group=age_group,
                severity=severity,
                message=(
                    f"👥 Demographic cluster: {test_name} — {rate:.1f}% abnormal "
                    f"in {age_group} / {region} "
                    f"({counts['abnormal']}/{counts['total']} values)"
                ),
                details={
                    "region": region,
                    "age_group": age_group,
                    **counts,
                },
            ))

    anomalies.sort(key=lambda a: (-{"critical": 2, "warning": 1}.get(a.severity, 0), -a.metric))
    return anomalies
