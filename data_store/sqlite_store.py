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
import random
import math

def _apply_laplace_noise(val: float, sensitivity: float = 1.0, epsilon: float = 0.5) -> float:
    """Applies calibrated Laplace noise for Differential Privacy (DP) guarantees."""
    scale = sensitivity / epsilon
    u = random.random() - 0.5
    if u == 0:
        noise = 0.0
    else:
        noise = -scale * math.copysign(math.log(1.0 - 2.0 * abs(u)), u)
    return val + noise


def seed_mock_data():
    """Seed the database with 30 days of historical reports to enable forecasting and EARS alerts."""
    with get_connection() as conn:
        # Check if already seeded
        try:
            row = conn.execute("SELECT COUNT(*) as cnt FROM reports").fetchone()
            if row and row["cnt"] > 0:
                return
        except sqlite3.OperationalError:
            # Table doesn't exist yet, we will init_db first which calls this
            return

        print("[DATABASE] Seeding mock historical data for forecasting and ESSENCE alerts...")
        
        # We will insert reports and lab values for the last 30 days
        base_date = datetime.now() - timedelta(days=30)
        
        tests = [
            {"name": "Hemoglobin", "low": 12.0, "high": 16.0, "unit": "g/dL", "normal_val": 14.0, "abnormal_val": 10.5},
            {"name": "HbA1c", "low": 4.0, "high": 5.6, "unit": "%", "normal_val": 5.2, "abnormal_val": 7.8},
            {"name": "Cholesterol", "low": 100.0, "high": 199.0, "unit": "mg/dL", "normal_val": 160.0, "abnormal_val": 240.0},
            {"name": "TSH", "low": 0.45, "high": 4.5, "unit": "uIU/mL", "normal_val": 1.8, "abnormal_val": 8.5}
        ]
        
        regions = ["Urban-Central", "Rural-East", "Suburban-West", "Coastal-South"]
        age_groups = ["0-18", "19-30", "31-45", "46-60", "60+"]
        
        # Seed 30 days of baseline data
        for d in range(30):
            current_day = base_date + timedelta(days=d)
            timestamp_str = current_day.isoformat()
            
            # 2-3 reports per day
            for r_idx in range(random.randint(2, 3)):
                report_id = str(uuid.uuid4())
                region = random.choice(regions)
                age = random.choice(age_groups)
                
                # Create lab values
                lab_values = []
                abnormal_cnt = 0
                normal_cnt = 0
                
                for t in tests:
                    # Normally 15% abnormal rate in baseline
                    is_abnormal = random.random() < 0.15
                    # Exclude HbA1c spike in Urban-Central/46-60 during baseline
                    if t["name"] == "HbA1c" and region == "Urban-Central" and age == "46-60":
                        is_abnormal = False
                        
                    val = t["abnormal_val"] if is_abnormal else t["normal_val"]
                    flag = "HIGH" if val > t["high"] else "LOW" if val < t["low"] else "NORMAL"
                    severity = 1 if flag != "NORMAL" else 0
                    
                    if severity > 0:
                        abnormal_cnt += 1
                    else:
                        normal_cnt += 1
                        
                    lab_values.append((
                        str(uuid.uuid4()), report_id, t["name"], val, t["unit"],
                        t["low"], t["high"], flag, severity, timestamp_str, region, age
                    ))
                
                # Insert report
                conn.execute(
                    """INSERT INTO reports (id, filename, upload_timestamp, total_tests,
                       normal_count, abnormal_count, critical_count, risk_score,
                       anonymized_region, age_group, mode)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (report_id, f"simulated_report_{d}_{r_idx}.pdf", timestamp_str, len(tests),
                     normal_cnt, abnormal_cnt, 0, abnormal_cnt * 1.5, region, age, "community")
                )
                
                # Insert lab values
                conn.executemany(
                    """INSERT INTO lab_values (id, report_id, test_name, value, unit,
                       reference_low, reference_high, flag, severity, timestamp,
                       anonymized_region, age_group)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    lab_values
                )
        
        # Now, create a spike (aberration) in the last 2 days:
        # HbA1c anomalies in Urban-Central region for 46-60 age group!
        for d in [28, 29]:
            current_day = base_date + timedelta(days=d)
            timestamp_str = current_day.isoformat()
            
            # Insert 5 additional reports for this specific cluster with abnormal HbA1c
            for r_idx in range(5):
                report_id = str(uuid.uuid4())
                region = "Urban-Central"
                age = "46-60"
                
                # Create abnormal HbA1c record
                hba1c_test = tests[1] # HbA1c
                
                conn.execute(
                    """INSERT INTO reports (id, filename, upload_timestamp, total_tests,
                       normal_count, abnormal_count, critical_count, risk_score,
                       anonymized_region, age_group, mode)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (report_id, f"spike_report_{d}_{r_idx}.pdf", timestamp_str, 1,
                     0, 1, 0, 3.0, region, age, "community")
                )
                
                conn.execute(
                    """INSERT INTO lab_values (id, report_id, test_name, value, unit,
                       reference_low, reference_high, flag, severity, timestamp,
                       anonymized_region, age_group)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (str(uuid.uuid4()), report_id, hba1c_test["name"], hba1c_test["abnormal_val"],
                     hba1c_test["unit"], hba1c_test["low"], hba1c_test["high"], "HIGH", 1,
                     timestamp_str, region, age)
                )
        
        print("[DATABASE] Seeding complete! Outbreak spike seeded for HbA1c in Urban-Central (46-60).")


def detect_epidemiological_aberrations(use_dp: bool = True) -> list[dict]:
    """
    ESSENCE-inspired Spatiotemporal Aberration Detection (Z-score C2 algorithm).
    Detects sudden increases in abnormal rate velocity for specific test x region x age clusters.
    """
    alerts = []
    
    with get_connection() as conn:
        # Get all distinct test x region x age combinations
        clusters = conn.execute("""
            SELECT DISTINCT test_name, anonymized_region as region, age_group
            FROM lab_values
            WHERE anonymized_region != '' AND age_group != '' AND test_name != ''
        """).fetchall()
        
    for cluster in clusters:
        test_name = cluster["test_name"]
        region = cluster["region"]
        age_group = cluster["age_group"]
        
        # Fetch daily count of abnormalities for this cluster over last 30 days
        with get_connection() as conn:
            daily_data = conn.execute("""
                SELECT 
                    DATE(timestamp) as date,
                    COUNT(*) as total,
                    SUM(CASE WHEN flag NOT IN ('NORMAL', 'UNKNOWN') THEN 1 ELSE 0 END) as abnormal
                FROM lab_values
                WHERE test_name = ? AND anonymized_region = ? AND age_group = ?
                GROUP BY DATE(timestamp)
                ORDER BY date ASC
            """, [test_name, region, age_group]).fetchall()
            
        if len(daily_data) < 10:
            continue # Need sufficient baseline data
            
        # Convert to daily list of abnormal counts
        history = [row["abnormal"] if row["abnormal"] is not None else 0 for row in daily_data]
        
        # We split the history into:
        # - Evaluation period: last 2 days
        # - Guard band: 2 days (index -4 to -3) to prevent the spike from inflating baseline std dev
        # - Baseline period: everything preceding (index 0 to -5)
        evaluation_vals = history[-2:]
        baseline_vals = history[:-4]
        
        if not baseline_vals:
            continue
            
        # Calculate mean and standard deviation of baseline
        n = len(baseline_vals)
        mean_val = sum(baseline_vals) / n
        variance = sum((x - mean_val) ** 2 for x in baseline_vals) / n
        std_dev = math.sqrt(variance)
        
        # Ensure standard deviation is not zero to avoid division by zero
        if std_dev < 0.5:
            std_dev = 0.5
            
        # Evaluate the average count in the evaluation period
        current_val = sum(evaluation_vals) / len(evaluation_vals)
        
        # Calculate Z-score
        z_score = (current_val - mean_val) / std_dev
        
        # Apply Differential Privacy noise to the Z-score calculation if requested
        if use_dp:
            z_score = _apply_laplace_noise(z_score, sensitivity=0.5, epsilon=0.5)
            
        # Trigger levels:
        # Z > 3.0: High-alert (Critical Outbreak Spike)
        # Z > 2.0: Warning (Mild Deviation)
        if z_score >= 2.0:
            severity = "critical" if z_score >= 3.0 else "warning"
            alert_msg = (
                f"🚨 ESSENCE Alert: Statistically significant spike in abnormal {test_name} "
                f"detected in {region} among age group {age_group}. "
                f"Aberration velocity: Z-score = {z_score:.2f}."
            )
            alerts.append({
                "test_name": test_name,
                "region": region,
                "age_group": age_group,
                "z_score": round(z_score, 2),
                "severity": severity,
                "message": alert_msg,
                "current_average": round(current_val, 1),
                "baseline_mean": round(mean_val, 1)
            })
            
    # Sort by Z-score descending
    alerts.sort(key=lambda x: x["z_score"], reverse=True)
    return alerts


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
    seed_mock_data()


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

def get_total_reports(use_dp: bool = True) -> int:
    """Get total number of reports in the database, with optional DP."""
    with get_connection() as conn:
        row = conn.execute("SELECT COUNT(*) as cnt FROM reports").fetchone()
        count = row["cnt"] if row else 0
        if use_dp and count > 0:
            count = max(0, int(round(_apply_laplace_noise(count, sensitivity=1.0, epsilon=0.5))))
        return count


def get_total_lab_values(use_dp: bool = True) -> int:
    """Get total number of lab value records, with optional DP."""
    with get_connection() as conn:
        row = conn.execute("SELECT COUNT(*) as cnt FROM lab_values").fetchone()
        count = row["cnt"] if row else 0
        if use_dp and count > 0:
            count = max(0, int(round(_apply_laplace_noise(count, sensitivity=1.0, epsilon=0.5))))
        return count


def get_abnormal_rate(use_dp: bool = True) -> float:
    """Get the percentage of lab values flagged as abnormal or critical, with optional DP."""
    with get_connection() as conn:
        row = conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN flag NOT IN ('NORMAL', 'UNKNOWN') THEN 1 ELSE 0 END) as abnormal
            FROM lab_values
        """).fetchone()
        if row and row["total"] > 0:
            total = row["total"]
            abnormal = row["abnormal"] if row["abnormal"] is not None else 0
            if use_dp:
                total_dp = max(1, _apply_laplace_noise(total, sensitivity=1.0, epsilon=0.5))
                abnormal_dp = max(0, _apply_laplace_noise(abnormal, sensitivity=1.0, epsilon=0.5))
                abnormal_dp = min(total_dp, abnormal_dp)
                return round((abnormal_dp / total_dp) * 100, 1)
            return round((abnormal / total) * 100, 1)
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
                SUM(CASE WHEN flag NOT IN ('NORMAL', 'UNKNOWN') THEN 1 ELSE 0 END) as abnormal_count,
                ROUND(SUM(CASE WHEN flag NOT IN ('NORMAL', 'UNKNOWN') THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as abnormal_rate
            FROM lab_values
            {where_clause}
            GROUP BY DATE(timestamp)
            ORDER BY date
        """, params).fetchall()

        return [dict(row) for row in rows]


def get_region_summary(time_period: str = None, use_dp: bool = True) -> list[dict]:
    """Get anomaly summary by region, with optional Differential Privacy."""
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
                SUM(CASE WHEN flag NOT IN ('NORMAL', 'UNKNOWN') THEN 1 ELSE 0 END) as abnormal
            FROM lab_values
            {where_clause}
            GROUP BY anonymized_region
        """, params).fetchall()

        results = []
        for row in rows:
            region = row["region"]
            total = row["total"]
            abnormal = row["abnormal"] if row["abnormal"] is not None else 0
            if use_dp:
                total_dp = max(1, _apply_laplace_noise(total, sensitivity=1.0, epsilon=0.5))
                abnormal_dp = max(0, _apply_laplace_noise(abnormal, sensitivity=1.0, epsilon=0.5))
                abnormal_dp = min(total_dp, abnormal_dp)
                rate = round((abnormal_dp / total_dp) * 100, 1)
            else:
                rate = round((abnormal / total) * 100, 1) if total > 0 else 0.0
            
            results.append({
                "region": region,
                "total": int(round(total_dp)) if use_dp else total,
                "abnormal": int(round(abnormal_dp)) if use_dp else abnormal,
                "percentage": rate
            })
        
        results.sort(key=lambda r: r["percentage"], reverse=True)
        return results


def get_age_group_summary(time_period: str = None, use_dp: bool = True) -> list[dict]:
    """Get anomaly summary by age group, with optional Differential Privacy."""
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
                SUM(CASE WHEN flag NOT IN ('NORMAL', 'UNKNOWN') THEN 1 ELSE 0 END) as abnormal
            FROM lab_values
            {where_clause}
            GROUP BY age_group
        """, params).fetchall()

        results = []
        for row in rows:
            age = row["age_group"]
            total = row["total"]
            abnormal = row["abnormal"] if row["abnormal"] is not None else 0
            if use_dp:
                total_dp = max(1, _apply_laplace_noise(total, sensitivity=1.0, epsilon=0.5))
                abnormal_dp = max(0, _apply_laplace_noise(abnormal, sensitivity=1.0, epsilon=0.5))
                abnormal_dp = min(total_dp, abnormal_dp)
                rate = round((abnormal_dp / total_dp) * 100, 1)
            else:
                rate = round((abnormal / total) * 100, 1) if total > 0 else 0.0
            
            results.append({
                "age_group": age,
                "total": int(round(total_dp)) if use_dp else total,
                "abnormal": int(round(abnormal_dp)) if use_dp else abnormal,
                "percentage": rate
            })
        
        results.sort(key=lambda r: r["percentage"], reverse=True)
        return results


def generate_community_alerts(time_period: str = None, threshold: float = 20.0, use_dp: bool = True) -> list[CommunityAlert]:
    """
    Generate community-level health alerts.

    Flags tests where the abnormal percentage exceeds the threshold,
    and calculates spatiotemporal aberrations (CDC EARS C2).
    """
    alerts = []
    top_abnormal = get_top_abnormal_tests(n=20, time_period=time_period)

    # 1. Volume-based static alerts
    for item in top_abnormal:
        if item["percentage"] >= threshold:
            severity = "critical" if item["percentage"] >= 40 else "warning"
            alert = CommunityAlert(
                id=str(uuid.uuid4()),
                alert_type=f"elevated_{item['test_name'].lower().replace(' ', '_')}",
                test_name=item["test_name"],
                percentage=item["percentage"],
                total_reports=get_total_reports(use_dp=use_dp),
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

    # 2. Spatiotemporal EARS Aberration alerts (ESSENCE Framework)
    try:
        aberrations = detect_epidemiological_aberrations(use_dp=use_dp)
        for aa in aberrations:
            alert = CommunityAlert(
                id=str(uuid.uuid4()),
                alert_type=f"aberration_{aa['test_name'].lower().replace(' ', '_')}",
                test_name=aa["test_name"],
                percentage=aa["z_score"],  # Store Z-score as percentage for dashboard mapping
                total_reports=get_total_reports(use_dp=use_dp),
                affected_reports=int(aa["current_average"]),
                time_period="last_30_days",
                region=aa["region"],
                age_group=aa["age_group"],
                severity=aa["severity"],
                message=aa["message"],
                generated_at=datetime.now().isoformat(),
            )
            alerts.append(alert)
    except Exception as e:
        print(f"[ALERTS] Failed to compute spatiotemporal aberrations: {e}")

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


# ---------- POPULATION ANOMALY QUERIES ----------

def get_all_lab_records(time_period: str = None) -> list[dict]:
    """
    Get all lab value records as dicts for population-level anomaly detection.

    Args:
        time_period: Optional ISO date string filter (>= this date).

    Returns:
        List of dicts with test_name, value, unit, flag, severity,
        anonymized_region, age_group, timestamp.
    """
    with get_connection() as conn:
        where_clause = ""
        params = []
        if time_period:
            where_clause = "WHERE timestamp >= ?"
            params.append(time_period)

        rows = conn.execute(f"""
            SELECT test_name, value, unit, flag, severity,
                   anonymized_region, age_group, timestamp
            FROM lab_values
            {where_clause}
            ORDER BY timestamp
        """, params).fetchall()

        return [dict(row) for row in rows]


def get_lab_records_by_period(start_date: str, end_date: str) -> list[dict]:
    """
    Get lab value records within a specific date range.

    Args:
        start_date: ISO date string for range start.
        end_date: ISO date string for range end.

    Returns:
        List of lab value dicts.
    """
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT test_name, value, unit, flag, severity,
                   anonymized_region, age_group, timestamp
            FROM lab_values
            WHERE timestamp >= ? AND timestamp < ?
            ORDER BY timestamp
        """, [start_date, end_date]).fetchall()

        return [dict(row) for row in rows]


def get_demographic_cross_tab(time_period: str = None) -> list[dict]:
    """
    Get cross-tabulated abnormal rates by test × region × age_group.

    Returns:
        List of dicts with test_name, region, age_group, total, abnormal, rate.
    """
    with get_connection() as conn:
        where_clause = "WHERE flag != 'UNKNOWN'"
        params = []
        if time_period:
            where_clause += " AND timestamp >= ?"
            params.append(time_period)

        rows = conn.execute(f"""
            SELECT
                test_name,
                anonymized_region as region,
                age_group,
                COUNT(*) as total,
                SUM(CASE WHEN flag NOT IN ('NORMAL', 'UNKNOWN') THEN 1 ELSE 0 END) as abnormal,
                ROUND(SUM(CASE WHEN flag NOT IN ('NORMAL', 'UNKNOWN') THEN 1 ELSE 0 END)
                    * 100.0 / COUNT(*), 1) as rate
            FROM lab_values
            {where_clause}
            GROUP BY test_name, anonymized_region, age_group
            HAVING total >= 3
            ORDER BY rate DESC
        """, params).fetchall()

        return [dict(row) for row in rows]


# Initialize DB on module import
init_db()
