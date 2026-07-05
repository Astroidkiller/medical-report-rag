"""
Community Health Intelligence Assistant — CLI Entry Point

Interactive command-line interface for the multi-agent pipeline.
Supports both patient and community modes without Streamlit.

Usage:
    py app.py
"""

import os
import sys
import glob

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from agents.orchestrator import (
    Session,
    run_patient_pipeline,
    run_community_pipeline,
    step_community_analysis,
)
from data_store.sqlite_store import get_total_reports, get_total_lab_values


# ============================================================
# Display Helpers
# ============================================================

def _print_header():
    """Print application header."""
    print("\n" + "=" * 65)
    print("  🏥 Community Health Intelligence Assistant")
    print("  AI for Better Living & Smarter Communities")
    print("  Powered by Vertex AI Gemini + RAG + Multi-Agent Pipeline")
    print("=" * 65)


def _print_section(title: str):
    """Print a section header."""
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")


def _print_risk_card(risk: dict):
    """Pretty-print a risk card to the terminal."""
    card = risk["card"]
    print(f"\n  {card['icon']} Risk Level: {card['risk_level']} (score: {card['risk_score']})")
    print(f"  {card['headline']}")
    print(f"  Tests: {card['total_tests']} total | "
          f"{card['normal_count']} normal | "
          f"{card['abnormal_count']} abnormal | "
          f"{card['critical_count']} critical")

    if card["critical_findings"]:
        print("\n  🔴 Critical Findings:")
        for f in card["critical_findings"]:
            print(f"     • {f['test_name']}: {f['value']} {f['unit']} "
                  f"(ref: {f['reference']}) — {f['explanation']}")

    if card["abnormal_findings"]:
        print("\n  🟡 Abnormal Findings:")
        for f in card["abnormal_findings"]:
            print(f"     • {f['test_name']}: {f['value']} {f['unit']} "
                  f"(ref: {f['reference']}) — {f['explanation']}")

    print(f"\n  📝 AI Explanation:\n  {risk['explanation']}")


def _print_community_dashboard(community: dict):
    """Pretty-print community dashboard data."""
    dashboard = community["dashboard"]
    metrics = dashboard["metrics"]

    print(f"\n  📊 Reports Analyzed: {metrics['total_reports']}")
    print(f"  🔬 Lab Values Recorded: {metrics['total_lab_values']}")
    print(f"  ⚠️  Overall Abnormal Rate: {metrics['abnormal_rate']}%")

    if dashboard.get("top_abnormal"):
        print("\n  🔍 Top Abnormal Tests:")
        for item in dashboard["top_abnormal"][:5]:
            print(f"     • {item['test_name']}: {item['flag_count']} flagged "
                  f"({item['percentage']}% abnormal)")

    if dashboard.get("region_summary"):
        print("\n  📍 By Region:")
        for item in dashboard["region_summary"]:
            print(f"     • {item['region']}: {item['abnormal']}/{item['total']} "
                  f"abnormal ({item['percentage']}%)")

    if dashboard.get("age_group_summary"):
        print("\n  👥 By Age Group:")
        for item in dashboard["age_group_summary"]:
            print(f"     • {item['age_group']}: {item['abnormal']}/{item['total']} "
                  f"abnormal ({item['percentage']}%)")

    # Population anomalies (new)
    pop_anomalies = community.get("population_anomalies", [])
    if pop_anomalies:
        print("\n  🚨 Population-Level Anomalies:")
        for a in pop_anomalies[:10]:
            print(f"     • [{a['severity'].upper()}] {a['message']}")

    # Community alerts
    alerts = dashboard.get("alerts", [])
    if alerts:
        print("\n  🔔 Active Alerts:")
        for alert in alerts[:5]:
            print(f"     • [{alert['severity'].upper()}] {alert['message']}")


def _print_qa_response(qa: dict):
    """Pretty-print a QA response."""
    print(f"\n  💬 Answer:\n")
    # Word-wrap the answer for terminal display
    answer = qa["answer"]
    for line in answer.split("\n"):
        print(f"  {line}")

    if qa.get("source_chunks"):
        print(f"\n  📎 Sources: {len(qa['source_chunks'])} relevant chunks retrieved")


# ============================================================
# Interactive Menu
# ============================================================


def _get_pdf_files() -> list[str]:
    """Find PDF files in the data/ directory."""
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    return glob.glob(os.path.join(data_dir, "*.pdf"))


def _ingest_reports_interactive() -> list[dict]:
    """Interactive report ingestion."""
    _print_section("📄 Report Ingestion")

    # Check for PDFs in data/
    pdf_files = _get_pdf_files()
    if pdf_files:
        print(f"\n  Found {len(pdf_files)} PDF(s) in data/:")
        for i, f in enumerate(pdf_files, 1):
            print(f"    {i}. {os.path.basename(f)}")
    else:
        print("\n  No PDFs found in data/ directory.")
        custom = input("\n  Enter PDF path (or press Enter to skip): ").strip()
        if custom and os.path.exists(custom):
            pdf_files = [custom]
        else:
            print("  Skipping ingestion.")
            return []

    choice = input(f"\n  Ingest all {len(pdf_files)} file(s)? [Y/n]: ").strip().lower()
    if choice == "n":
        return []

    file_tuples = [(f, os.path.basename(f)) for f in pdf_files]

    print(f"\n  ⏳ Ingesting {len(file_tuples)} report(s)...")
    result = run_community_pipeline(file_paths=file_tuples)

    for i, ingestion in enumerate(result["ingestions"]):
        if "error" in ingestion:
            print(f"  ❌ {ingestion['filename']}: {ingestion['error']}")
        else:
            print(f"  ✅ {ingestion['filename']}: "
                  f"{ingestion['risk_summary']['total']} tests, "
                  f"{ingestion['risk_summary']['abnormal']} abnormal, "
                  f"{ingestion['risk_summary']['critical']} critical")
            risk = result["risk_cards"][i] if i < len(result["risk_cards"]) else None
            if risk and "card" in risk:
                _print_risk_card(risk)
            elif risk and "error" in risk:
                print(f"  Risk analysis unavailable: {risk['error']}")

    return result.get("ingestions", [])


def _community_analysis_interactive():
    """Interactive community analysis."""
    _print_section("🏥 Community Health Analysis")

    total = get_total_reports()
    if total == 0:
        print("\n  ⚠️  No reports in database. Ingest reports first (option 1).")
        return

    session = Session(mode="community")
    result = step_community_analysis(session)
    _print_community_dashboard(result)

    # Interactive Q&A loop
    print("\n  💬 Ask questions about community health trends (type 'back' to return):\n")
    while True:
        query = input("  You: ").strip()
        if not query or query.lower() in ("back", "exit", "quit", "q"):
            break

        print("\n  ⏳ Analyzing...")
        result = step_community_analysis(session, query=query)
        if result.get("answer"):
            print(f"\n  🤖 Assistant:\n")
            for line in result["answer"].split("\n"):
                print(f"  {line}")
        print()


def _patient_qa_interactive():
    """Interactive patient Q&A."""
    _print_section("🧑‍⚕️ Patient Report Q&A")

    pdf_files = _get_pdf_files()
    if not pdf_files:
        print("\n  ⚠️  No PDFs in data/ directory.")
        return

    print(f"\n  Select a report:")
    for i, f in enumerate(pdf_files, 1):
        print(f"    {i}. {os.path.basename(f)}")

    try:
        idx = int(input(f"\n  Choice [1-{len(pdf_files)}]: ").strip()) - 1
        if idx < 0 or idx >= len(pdf_files):
            raise IndexError
        selected = pdf_files[idx]
    except (ValueError, IndexError):
        print("  Invalid choice.")
        return

    print(f"\n  ⏳ Processing {os.path.basename(selected)}...")
    result = run_patient_pipeline(selected, os.path.basename(selected))

    _print_risk_card(result["risk"])

    # Interactive Q&A loop
    print("\n  💬 Ask questions about this report (type 'back' to return):\n")
    session = result["session"]
    raw_text = result["ingestion"].get("raw_text", "")

    while True:
        query = input("  You: ").strip()
        if not query or query.lower() in ("back", "exit", "quit", "q"):
            break

        print("\n  ⏳ Searching report...")
        from agents.orchestrator import step_qa
        qa = step_qa(session, query, full_text=raw_text)
        _print_qa_response(qa)
        print()


def main():
    """Main interactive CLI loop."""
    _print_header()

    while True:
        total = get_total_reports()
        total_vals = get_total_lab_values()

        print(f"\n  📊 Database: {total} reports, {total_vals} lab values\n")
        print("  Choose an action:")
        print("    1. 📄 Ingest Reports (PDF → Extract → Flag → Store)")
        print("    2. 🏥 Community Health Analysis (Population Trends)")
        print("    3. 🧑‍⚕️ Patient Report Q&A (Individual Report)")
        print("    4. 🚪 Exit")

        choice = input("\n  Choice [1-4]: ").strip()

        if choice == "1":
            _ingest_reports_interactive()
        elif choice == "2":
            _community_analysis_interactive()
        elif choice == "3":
            _patient_qa_interactive()
        elif choice == "4":
            print("\n  👋 Goodbye!\n")
            break
        else:
            print("  Invalid choice. Try again.")


if __name__ == "__main__":
    main()
