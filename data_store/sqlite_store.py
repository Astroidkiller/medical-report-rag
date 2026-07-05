"""
SQLite persistence layer for structured lab values and report metadata.
Serves as the BigQuery analog for aggregate queries and trend analysis.
"""

import sqlite3
import os
from contextlib import contextmanager
from datetime import datetime, timedelta

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SQLITE_DB_PATH
from data_store.models import LabValueRecord, ReportRecord, CommunityAlert
import uuid


@contextmanager
def get_connection():
    """Context manager for SQLite connections."""
    os.makedirs(os.path.dirname(SQLITE_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Create database tables if they don't exist."""
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS reports (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                upload_timestamp TEXT NOT NULL,
                total_tests INTEGER DEFAULT 0,
                normal_count INTEGER DEFAULT 0,
                abnormal_count INTEGER DEFAULT 0,
                critical_count INTEGER DEFAULT 0,
                risk_score REAL DEFAULT 0.0,
                anonymized_region TEXT DEFAULT '',
                age_group TEXT DEFAULT '',
                mode TEXT DEFAULT 'patient'
            );

            CREATE TABLE IF NOT EXISTS lab_values (
                id TEXT PRIMARY KEY,
                report_id TEXT NOT NULL,
                test_name TEXT NOT NULL,
                value REAL NOT NULL,
                unit TEXT DEFAULT '',
                reference_low REAL,
                reference_high REAL,
                flag TEXT DEFAULT 'UNKNOWN',
                severity INTEGER DEFAULT 0,
                timestamp TEXT NOT NULL,
                anonymized_region TEXT DEFAULT '',
                age_group TEXT DEFAULT '',
                FOREIGN KEY (report_id) REFERENCES reports(id)
            );

            CREATE INDEX IF NOT EXISTS idx_lab_test_name ON lab_values(test_name);
            CREATE INDEX IF NOT EXISTS idx_lab_flag ON lab_values(flag);
            CREATE INDEX IF NOT EXISTS idx_lab_timestamp ON lab_values(timestamp);
            CREATE INDEX IF NOT EXISTS idx_lab_region ON lab_values(anonymized_region);
            CREATE INDEX IF NOT EXISTS idx_lab_age ON lab_values(age_group);
            CREATE INDEX IF NOT EXISTS idx_report_timestamp ON reports(upload_timestamp);
        """)


# ---------- INSERT OPERATIONS ----------

def insert_report(report: ReportRecord) -> None:
    """Insert a report record."""
    with get_connection() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO reports
               (id, filename, upload_timestamp, total_tests, normal_count,
                abnormal_count, critical_count, risk_score, anonymized_region, age_group, mode)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (report.id, report.filename, report.upload_timestamp,
             report.total_tests, report.normal_count, report.abnormal_count,
             report.critical_count, report.risk_score, report.anonymized_region,
             report.age_group, report.mode),
        )


def insert_lab_values(values: list[LabValueRecord]) -> None:
    """Insert multiple lab value records."""
    with get_connection() as conn:
        conn.executemany(
            """INSERT OR REPLACE INTO lab_values
               (id, report_id, test_name, value, unit, reference_low, reference_high,
                flag, severity, timestamp, anonymized_region, age_group)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (v.id, v.report_id, v.test_name, v.value, v.unit,
                 v.reference_low, v.reference_high, v.flag, v.severity,
                 v.timestamp, v.anonymized_region, v.age_group)
                for v in values
            ],
        )


# ---------- AGGREGATE QUERIES ----------

def get_total_reports() -> int:
    """Get total number of reports in the database."""
    with get_connection() as conn:
        row = conn.execute("SELECT COUNT(*) as cnt FROM reports").fetchone()
        return row["cnt"] if row else 0


def get_total_lab_values() -> int:
    """Get total number of lab value records."""
    with get_connection() as conn:
        row = conn.execute("SELECT COUNT(*) as cnt FROM lab_values").fetchone()
        return row["cnt"] if row else 0


def get_abnormal_rate() -> float:
    """Get the percentage of lab values flagged as abnormal or critical."""
    with get_connection() as conn:
        row = conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN flag NOT IN ('NORMAL', 'UNKNOWN') THEN 1 ELSE 0 END) as abnormal
            FROM lab_values
        """).fetchone()
        if row and row["total"] > 0:
            return (row["abnormal"] / row["total"]) * 100
        return 0.0


def get_top_abnormal_tests(n: int = 10, time_period: str = None) -> list[dict]:
    """
    Get the most commonly flagged abnormal tests.

    Args:
        n: Number of top tests to return.
        time_period: Optional ISO date string to filter (>= this date).

    Returns:
        List of dicts with test_name, count, percentage.
    """
    with get_connection() as conn:
        where_clause = ""
        params = []
        if time_period:
            where_clause = "AND timestamp >= ?"
            params.append(time_period)

        rows = conn.execute(f"""
            SELECT
                test_name,
                SUM(CASE WHEN flag NOT IN ('NORMAL', 'UNKNOWN') THEN 1 ELSE 0 END) as flag_count,
                ROUND(
                    SUM(CASE WHEN flag NOT IN ('NORMAL', 'UNKNOWN') THEN 1 ELSE 0 END) * 100.0
                    / COUNT(*),
                1) as percentage
            FROM lab_values
            WHERE flag != 'UNKNOWN' {where_clause}
            GROUP BY test_name
            HAVING flag_count > 0
            ORDER BY flag_count DESC
            LIMIT ?
        """, params + [n]).fetchall()

        return [dict(row) for row in rows]


def get_flag_distribution(time_period: str = None) -> dict:
    """
    Get distribution of flags (NORMAL, HIGH, LOW, CRITICAL_HIGH, CRITICAL_LOW).

    Returns:
        Dict mapping flag -> count.
    """
    with get_connection() as conn:
        where_clause = ""
        params = []
        if time_period:
            where_clause = "WHERE timestamp >= ?"
            params.append(time_period)

        rows = conn.execute(f"""
            SELECT flag, COUNT(*) as cnt
            FROM lab_values
            {where_clause}
            GROUP BY flag
        """, params).fetchall()

        return {row["flag"]: row["cnt"] for row in rows}


def get_test_trend(test_name: str, time_period: str = None) -> list[dict]:
    """
    Get trend data for a specific test over time.

    Returns:
        List of dicts with date, avg_value, count, abnormal_count.
    """
    with get_connection() as conn:
        where_clause = "WHERE LOWER(test_name) = LOWER(?)"
        params = [test_name]
        if time_period:
            where_clause += " AND timestamp >= ?"
            params.append(time_period)

        rows = conn.execute(f"""
            SELECT
                DATE(timestamp) as date,
                AVG(value) as avg_value,
                COUNT(*) as count,
                SUM(CASE WHEN flag NOT IN ('NORMAL', 'UNKNOWN') THEN 1 ELSE 0 END) as abnormal_count
            FROM lab_values
            {where_clause}
            GROUP BY DATE(timestamp)
            ORDER BY date
        """, params).fetchall()

        return [dict(row) for row in rows]


def get_region_summary(time_period: str = None) -> list[dict]:
    """
    Get anomaly summary by region.

    Returns:
        List of dicts with region, total, abnormal, percentage.
    """
    with get_connection() as conn:
        where_clause = ""
        params = []
        if time_period:
            where_clause = "WHERE timestamp >= ?"
            params.append(time_period)

        rows = conn.execute(f"""
            SELECT
                anonymized_region as region,
                COUNT(*) as total,
                SUM(CASE WHEN flag NOT IN ('NORMAL', 'UNKNOWN') THEN 1 ELSE 0 END) as abnormal,
                ROUND(SUM(CASE WHEN flag NOT IN ('NORMAL', 'UNKNOWN') THEN 1 ELSE 0 END)
                    * 100.0 / COUNT(*), 1) as percentage
            FROM lab_values
            {where_clause}
            GROUP BY anonymized_region
            ORDER BY percentage DESC
        """, params).fetchall()

        return [dict(row) for row in rows]


def get_age_group_summary(time_period: str = None) -> list[dict]:
    """
    Get anomaly summary by age group.

    Returns:
        List of dicts with age_group, total, abnormal, percentage.
    """
    with get_connection() as conn:
        where_clause = ""
        params = []
        if time_period:
            where_clause = "WHERE timestamp >= ?"
            params.append(time_period)

        rows = conn.execute(f"""
            SELECT
                age_group,
                COUNT(*) as total,
                SUM(CASE WHEN flag NOT IN ('NORMAL', 'UNKNOWN') THEN 1 ELSE 0 END) as abnormal,
                ROUND(SUM(CASE WHEN flag NOT IN ('NORMAL', 'UNKNOWN') THEN 1 ELSE 0 END)
                    * 100.0 / COUNT(*), 1) as percentage
            FROM lab_values
            {where_clause}
            GROUP BY age_group
            ORDER BY percentage DESC
        """, params).fetchall()

        return [dict(row) for row in rows]


def generate_community_alerts(time_period: str = None, threshold: float = 20.0) -> list[CommunityAlert]:
    """
    Generate community-level health alerts.

    Flags tests where the abnormal percentage exceeds the threshold.

    Args:
        time_period: Optional ISO date string to filter.
        threshold: Minimum abnormal percentage to trigger an alert.

    Returns:
        List of CommunityAlert objects.
    """
    alerts = []
    top_abnormal = get_top_abnormal_tests(n=20, time_period=time_period)

    for item in top_abnormal:
        if item["percentage"] >= threshold:
            severity = "critical" if item["percentage"] >= 40 else "warning"
            alert = CommunityAlert(
                id=str(uuid.uuid4()),
                alert_type=f"elevated_{item['test_name'].lower().replace(' ', '_')}",
                test_name=item["test_name"],
                percentage=item["percentage"],
                total_reports=get_total_reports(),
                affected_reports=item["flag_count"],
                time_period=time_period or "all_time",
                region=None,
                age_group=None,
                severity=severity,
                message=(
                    f"🚨 {item['percentage']}% of lab values for {item['test_name']} "
                    f"are outside normal range ({item['flag_count']} flagged values)."
                ),
                generated_at=datetime.now().isoformat(),
            )
            alerts.append(alert)

    return alerts


def get_all_test_names() -> list[str]:
    """Get all distinct test names in the database."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT DISTINCT test_name FROM lab_values ORDER BY test_name"
        ).fetchall()
        return [row["test_name"] for row in rows]


def get_recent_reports(n: int = 10) -> list[dict]:
    """Get the most recent report records."""
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM reports
            ORDER BY upload_timestamp DESC
            LIMIT ?
        """, [n]).fetchall()
        return [dict(row) for row in rows]


def get_abnormal_rate_over_time(test_name: str = None) -> list[dict]:
    """
    Get abnormal rate over time (by day) for forecasting.

    Args:
        test_name: Optional test name to filter.

    Returns:
        List of dicts with date, total, abnormal, rate.
    """
    with get_connection() as conn:
        where_clause = "WHERE flag != 'UNKNOWN'"
        params = []
        if test_name:
            where_clause += " AND LOWER(test_name) = LOWER(?)"
            params.append(test_name)

        rows = conn.execute(f"""
            SELECT
                DATE(timestamp) as date,
                COUNT(*) as total,
                SUM(CASE WHEN flag NOT IN ('NORMAL', 'UNKNOWN') THEN 1 ELSE 0 END) as abnormal,
                ROUND(SUM(CASE WHEN flag NOT IN ('NORMAL', 'UNKNOWN') THEN 1 ELSE 0 END)
                    * 100.0 / COUNT(*), 1) as rate
            FROM lab_values
            {where_clause}
            GROUP BY DATE(timestamp)
            ORDER BY date
        """, params).fetchall()

        return [dict(row) for row in rows]


def forecast_abnormal_trend(test_name: str = None, days_ahead: int = 30) -> dict:
    """
    Predict future abnormal rate using simple linear trend projection.

    Uses least-squares linear regression on historical daily abnormal rates
    to forecast trends. This is the 'predictive analytics' component.

    Args:
        test_name: Optional test name to forecast for.
        days_ahead: Number of days to project forward.

    Returns:
        Dict with current_rate, projected_rate, trend_direction,
        trend_slope, confidence, historical_data, forecast_data.
    """
    history = get_abnormal_rate_over_time(test_name)

    if len(history) < 2:
        return {
            "current_rate": history[0]["rate"] if history else 0,
            "projected_rate": None,
            "trend_direction": "insufficient_data",
            "trend_slope": 0,
            "confidence": "low",
            "historical_data": history,
            "forecast_data": [],
            "message": "Not enough historical data points for forecasting. Need at least 2 days of data.",
        }

    # Simple linear regression: y = mx + b
    n = len(history)
    x = list(range(n))
    y = [h["rate"] for h in history]

    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))
    sum_x2 = sum(xi ** 2 for xi in x)

    denominator = n * sum_x2 - sum_x ** 2
    if denominator == 0:
        slope = 0
        intercept = sum_y / n
    else:
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n

    # Current rate (last data point)
    current_rate = y[-1]

    # Project forward
    forecast_points = []
    last_date = history[-1]["date"]
    for d in range(1, days_ahead + 1):
        projected_idx = n - 1 + d
        projected_rate = max(0, min(100, slope * projected_idx + intercept))
        # Simple date projection
        from datetime import datetime as dt, timedelta as td
        try:
            future_date = (dt.strptime(last_date, "%Y-%m-%d") + td(days=d)).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            future_date = f"day_{d}"

        forecast_points.append({
            "date": future_date,
            "projected_rate": round(projected_rate, 1),
        })

    projected_rate_end = forecast_points[-1]["projected_rate"] if forecast_points else current_rate

    # Determine trend direction
    if slope > 0.5:
        trend_direction = "increasing"
        trend_emoji = "📈"
    elif slope < -0.5:
        trend_direction = "decreasing"
        trend_emoji = "📉"
    else:
        trend_direction = "stable"
        trend_emoji = "➡️"

    # Confidence based on data points
    if n >= 10:
        confidence = "high"
    elif n >= 5:
        confidence = "medium"
    else:
        confidence = "low"

    test_label = test_name or "all tests"
    message = (
        f"{trend_emoji} Abnormal rate for {test_label} is **{trend_direction}**. "
        f"Current rate: {current_rate}%. "
        f"Projected rate in {days_ahead} days: {projected_rate_end}%. "
        f"(Confidence: {confidence}, based on {n} data points)"
    )

    return {
        "current_rate": current_rate,
        "projected_rate": projected_rate_end,
        "trend_direction": trend_direction,
        "trend_slope": round(slope, 3),
        "confidence": confidence,
        "historical_data": history,
        "forecast_data": forecast_points,
        "message": message,
    }


def get_risk_forecast_by_region(days_ahead: int = 30) -> list[dict]:
    """
    Generate risk forecasts for each region.

    Returns:
        List of dicts with region, current_rate, projected_rate,
        trend_direction, risk_level.
    """
    with get_connection() as conn:
        regions = conn.execute(
            "SELECT DISTINCT anonymized_region FROM lab_values WHERE anonymized_region != ''"
        ).fetchall()

    forecasts = []
    for row in regions:
        region = row["anonymized_region"]
        # Get region-specific data
        with get_connection() as conn:
            history = conn.execute("""
                SELECT
                    DATE(timestamp) as date,
                    COUNT(*) as total,
                    SUM(CASE WHEN flag NOT IN ('NORMAL', 'UNKNOWN') THEN 1 ELSE 0 END) as abnormal,
                    ROUND(SUM(CASE WHEN flag NOT IN ('NORMAL', 'UNKNOWN') THEN 1 ELSE 0 END)
                        * 100.0 / COUNT(*), 1) as rate
                FROM lab_values
                WHERE flag != 'UNKNOWN' AND anonymized_region = ?
                GROUP BY DATE(timestamp)
                ORDER BY date
            """, [region]).fetchall()

        rates = [dict(h)["rate"] for h in history]
        current = rates[-1] if rates else 0

        # Simple projection
        if len(rates) >= 2:
            avg_change = (rates[-1] - rates[0]) / len(rates)
            projected = max(0, min(100, current + avg_change * days_ahead))
        else:
            projected = current

        if projected > 40:
            risk_level = "high"
        elif projected > 25:
            risk_level = "medium"
        else:
            risk_level = "low"

        forecasts.append({
            "region": region,
            "current_rate": round(current, 1),
            "projected_rate": round(projected, 1),
            "trend_direction": "increasing" if projected > current else "decreasing" if projected < current else "stable",
            "risk_level": risk_level,
            "data_points": len(rates),
        })

    forecasts.sort(key=lambda f: f["projected_rate"], reverse=True)
    return forecasts


# Initialize DB on module import
init_db()
