"""
Patient Mode UI — upload a single report, get risk analysis,
ask questions with source attribution.
"""

import hashlib
import re
import uuid
import streamlit as st
import os
import time

from agents.extraction_agent import ingest_report
from agents.qa_agent import answer_patient_question
from agents.risk_agent import generate_risk_card, generate_risk_explanation
from ui.components import (
    render_risk_card,
    render_lab_values_grid,
    render_source_evidence,
)
from core.embeddings import clear_collection


def _sanitize_filename(name: str) -> str:
    """Sanitize an uploaded filename to prevent path traversal.

    FIX #2 / #8: Strips directory components and replaces unsafe chars.
    """
    # Take only the basename (strip any directory traversal)
    name = os.path.basename(name)
    # Replace anything that isn't alphanumeric, dash, underscore, or dot
    name = re.sub(r"[^\w\-.]", "_", name)
    # Prevent empty filenames
    if not name or name.startswith("."):
        name = f"upload_{uuid.uuid4().hex[:8]}.pdf"
    return name


def _content_hash(uploaded_files) -> str:
    """Generate a hash from file content, not just name+size.

    FIX #4: File-change detection now uses content hashing.
    """
    h = hashlib.sha256()
    for f in uploaded_files:
        h.update(f.getvalue())
    return h.hexdigest()


def _get_session_collection() -> str:
    """Return a per-session ChromaDB collection name.

    FIX #1 / #16: Prevents cross-session PHI leakage by
    scoping data to the current Streamlit session.
    """
    if "patient_collection_id" not in st.session_state:
        st.session_state.patient_collection_id = f"patient_{uuid.uuid4().hex[:12]}"
    return st.session_state.patient_collection_id


def render_patient_mode():
    """Render the full patient mode interface."""

    st.markdown("### 📄 Upload Your Medical Report")
    st.caption("Upload a medical report PDF and get AI-powered explanations in simple language.")

    uploaded_files = st.file_uploader(
        "Upload medical reports (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        key="patient_uploader",
        help="Upload one or more diagnostic reports to analyze.",
    )

    if not uploaded_files:
        _render_empty_state()
        return

    # FIX #4: content-based change detection instead of name+size
    current_hash = _content_hash(uploaded_files)
    if current_hash != st.session_state.get("patient_files_hash"):
        st.session_state.patient_files_hash = current_hash
        st.session_state.patient_chat_history = []
        st.session_state.patient_ingestion_results = []
        # FIX #3: clear per-report explanations, not a single shared one
        st.session_state.patient_risk_explanations = {}

    # Session-scoped collection name
    collection_name = _get_session_collection()

    # Ingest reports (only if not already done for these files)
    if not st.session_state.get("patient_ingestion_results"):
        with st.status("🔬 Analyzing your report(s)...", expanded=True):
            clear_collection(collection_name)
            results = []

            # FIX #1: Session-scoped upload directory
            session_dir = os.path.join(
                "data", "patient_uploads",
                st.session_state.patient_collection_id,
            )
            os.makedirs(session_dir, exist_ok=True)

            for uploaded_file in uploaded_files:
                st.write(f"Processing: {uploaded_file.name}...")

                # FIX #2/#8: Sanitize filename before writing
                safe_name = _sanitize_filename(uploaded_file.name)
                file_path = os.path.join(session_dir, safe_name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                result = ingest_report(
                    file_path=file_path,
                    filename=safe_name,
                    mode="patient",
                    collection_name=collection_name,
                )
                results.append(result)
                st.write(f"✅ {safe_name}: {result['risk_summary']['total']} tests found, "
                         f"{result['risk_summary']['abnormal']} abnormal, "
                         f"{result['risk_summary']['critical']} critical")

            st.session_state.patient_ingestion_results = results

    results = st.session_state.patient_ingestion_results

    # --- Overview Metrics ---
    total_tests = sum(r["risk_summary"]["total"] for r in results)
    total_abnormal = sum(r["risk_summary"]["abnormal"] for r in results)
    total_critical = sum(r["risk_summary"]["critical"] for r in results)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📄 Reports", len(results))
    col2.metric("🧪 Tests Found", total_tests)
    col3.metric("⚠️ Abnormal", total_abnormal)
    col4.metric("🔴 Critical", total_critical)

    # --- Risk Card ---
    st.markdown("---")

    # FIX #3: Initialize per-report explanation dict
    if "patient_risk_explanations" not in st.session_state:
        st.session_state.patient_risk_explanations = {}

    for idx, result in enumerate(results):
        if len(results) > 1:
            st.markdown(f"#### 📋 {result['filename']}")

        risk_card = generate_risk_card(result["flagged_values"], result["risk_summary"])
        render_risk_card(risk_card)

        # FIX #3: Risk explanation generated per-report, keyed by index
        report_key = f"report_{idx}"
        if report_key not in st.session_state.patient_risk_explanations:
            with st.spinner("Generating risk assessment..."):
                explanation = generate_risk_explanation(risk_card)
                st.session_state.patient_risk_explanations[report_key] = explanation

        st.markdown(st.session_state.patient_risk_explanations[report_key])

        # Lab value cards
        st.markdown("---")
        st.markdown("### 🧪 Detailed Lab Values")
        render_lab_values_grid(result["flagged_values"])

    # --- Chat Interface ---
    st.markdown("---")
    st.markdown("### 💬 Ask Questions About Your Report")

    # Initialize chat history
    if "patient_chat_history" not in st.session_state:
        st.session_state.patient_chat_history = []

    # Display chat history
    for msg in st.session_state.patient_chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                render_source_evidence(msg["sources"], msg.get("source_meta"))

    # Quick actions
    st.markdown("**Quick Actions:**")
    quick_cols = st.columns(4)
    quick_query = None

    if quick_cols[0].button("📋 Summarize Report", key="q_summary"):
        quick_query = "Summarize this medical report in simple, patient-friendly language."
    if quick_cols[1].button("🔍 Key Findings", key="q_findings"):
        quick_query = "What are the most important findings from this report?"
    if quick_cols[2].button("⚠️ Abnormal Values", key="q_abnormal"):
        quick_query = "Explain any values that are outside the normal range."
    if quick_cols[3].button("💊 Next Steps", key="q_nextsteps"):
        quick_query = "Based on these results, what type of specialist should I see?"

    # Text input
    user_input = st.chat_input("Ask a question about your report...")

    query = quick_query or user_input

    if query:
        # Rate limiting
        current_time = time.time()
        if current_time - st.session_state.get("last_qa_time", 0) < 1.5:
            st.warning("⏳ Please wait a moment before asking another question.")
            return

        st.session_state.last_qa_time = current_time

        # Add user message
        st.session_state.patient_chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Get answer with streaming
        with st.chat_message("assistant"):
            # Combine all report texts for context
            combined_text = "\n\n---\n\n".join(
                f"Report: {r['filename']}\n{r['raw_text']}" for r in results
            )

            # FIX #5: Wrap streaming LLM call in try/except to prevent
            # chat history desync on error
            full_response = ""
            source_chunks = []
            source_metadata = []

            try:
                response = answer_patient_question(
                    query=query,
                    collection_name=collection_name,
                    full_text_override=combined_text[:12000],
                    stream=True,
                )

                source_chunks = response["source_chunks"]
                source_metadata = response["source_metadata"]

                # Stream the response
                placeholder = st.empty()
                for chunk in response["answer"]:
                    delta = getattr(chunk.choices[0].delta, "content", None) if chunk.choices else None
                    if delta:
                        full_response += delta
                        placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)

            except Exception:
                full_response = (
                    "⚠️ An error occurred while generating the response. "
                    "Please try asking your question again."
                )
                st.markdown(full_response)

            # Show source evidence
            render_source_evidence(source_chunks, source_metadata)

            # Save to history (always, even on error — keeps chat in sync)
            st.session_state.patient_chat_history.append({
                "role": "assistant",
                "content": full_response,
                "sources": source_chunks,
                "source_meta": source_metadata,
            })

    # --- Export ---
    if st.session_state.patient_chat_history:
        st.markdown("---")
        export_text = "# Medical Analysis Transcript\n\n"
        export_text += "## Risk Assessment\n"
        for key, expl in st.session_state.get("patient_risk_explanations", {}).items():
            export_text += f"{expl}\n\n"
        export_text += "## Q&A History\n\n"
        for msg in st.session_state.patient_chat_history:
            role = "Patient" if msg["role"] == "user" else "AI Assistant"
            export_text += f"### {role}\n{msg['content']}\n\n"

        st.download_button(
            label="📥 Download Analysis Transcript",
            data=export_text,
            file_name="medical_analysis.txt",
            mime="text/plain",
        )


def _render_empty_state():
    """Render the empty state before any files are uploaded."""
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        #### 🔬 Smart Analysis
        Automatically detects abnormal values and flags them with severity levels.
        """)
    with col2:
        st.markdown("""
        #### 💬 Ask Anything
        Ask questions in plain language and get clear, referenced explanations.
        """)
    with col3:
        st.markdown("""
        #### 🛡️ Responsible AI
        Every answer shows the source evidence from your report. Nothing is fabricated.
        """)
