"""Patient Mode UI: upload reports, review risk, and ask grounded questions."""

import hashlib
import os
import re
import time
import uuid

import streamlit as st

from agents.extraction_agent import ingest_report
from agents.qa_agent import answer_patient_question
from agents.risk_agent import generate_risk_card, generate_risk_explanation
from core.embeddings import clear_collection
from ui.components import render_lab_values_grid, render_risk_card, render_source_evidence


def _sanitize_filename(name: str) -> str:
    """Sanitize an uploaded filename to prevent path traversal."""
    name = os.path.basename(name)
    name = re.sub(r"[^\w\-.]", "_", name)
    if not name or name.startswith("."):
        name = f"upload_{uuid.uuid4().hex[:8]}.pdf"
    return name


def _content_hash(uploaded_files) -> str:
    """Generate a hash from file content."""
    h = hashlib.sha256()
    for f in uploaded_files:
        h.update(f.getvalue())
    return h.hexdigest()


def _get_session_collection() -> str:
    """Return a per-session ChromaDB collection name."""
    if "patient_collection_id" not in st.session_state:
        st.session_state.patient_collection_id = f"patient_{uuid.uuid4().hex[:12]}"
    return st.session_state.patient_collection_id


def render_patient_mode():
    """Render the full patient mode interface."""
    st.markdown(
        """<div class="section-band">
            <h3>Patient report review</h3>
            <p>Upload one or more PDF reports to extract lab values, flag abnormal results,
            and ask source-grounded questions in plain language.</p>
        </div>""",
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Upload medical reports",
        type=["pdf"],
        accept_multiple_files=True,
        key="patient_uploader",
        help="PDF diagnostic reports are analyzed in this browser session.",
    )

    if not uploaded_files:
        _render_empty_state()
        return

    current_hash = _content_hash(uploaded_files)
    if current_hash != st.session_state.get("patient_files_hash"):
        st.session_state.patient_files_hash = current_hash
        st.session_state.patient_chat_history = []
        st.session_state.patient_ingestion_results = []
        st.session_state.patient_risk_explanations = {}

    collection_name = _get_session_collection()

    if not st.session_state.get("patient_ingestion_results"):
        with st.status("Analyzing report files...", expanded=True):
            clear_collection(collection_name)
            results = []

            session_dir = os.path.join(
                "data",
                "patient_uploads",
                st.session_state.patient_collection_id,
            )
            os.makedirs(session_dir, exist_ok=True)

            for uploaded_file in uploaded_files:
                safe_name = _sanitize_filename(uploaded_file.name)
                st.write(f"Processing {safe_name}")
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
                st.write(
                    f"Done: {result['risk_summary']['total']} tests, "
                    f"{result['risk_summary']['abnormal']} abnormal, "
                    f"{result['risk_summary']['critical']} critical"
                )

            st.session_state.patient_ingestion_results = results

    results = st.session_state.patient_ingestion_results

    total_tests = sum(r["risk_summary"]["total"] for r in results)
    total_abnormal = sum(r["risk_summary"]["abnormal"] for r in results)
    total_critical = sum(r["risk_summary"]["critical"] for r in results)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Reports", len(results))
    col2.metric("Tests Found", total_tests)
    col3.metric("Abnormal", total_abnormal)
    col4.metric("Critical", total_critical)

    if "patient_risk_explanations" not in st.session_state:
        st.session_state.patient_risk_explanations = {}

    st.markdown("### Clinical Snapshot")
    tabs = st.tabs([r["filename"] for r in results])

    for idx, result in enumerate(results):
        with tabs[idx]:
            risk_card = generate_risk_card(result["flagged_values"], result["risk_summary"])
            render_risk_card(risk_card)

            report_key = f"report_{idx}"
            if report_key not in st.session_state.patient_risk_explanations:
                with st.spinner("Generating patient-friendly risk assessment..."):
                    explanation = generate_risk_explanation(risk_card)
                    st.session_state.patient_risk_explanations[report_key] = explanation

            st.markdown(st.session_state.patient_risk_explanations[report_key])
            st.markdown("### Lab Values")
            render_lab_values_grid(result["flagged_values"])

    _render_patient_chat(results, collection_name)
    _render_export()


def _render_patient_chat(results: list[dict], collection_name: str):
    """Render patient question answering."""
    st.markdown("### Ask About This Report")

    if "patient_chat_history" not in st.session_state:
        st.session_state.patient_chat_history = []

    for msg in st.session_state.patient_chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                render_source_evidence(msg["sources"], msg.get("source_meta"))

    st.markdown("**Quick questions**")
    quick_cols = st.columns(4)
    quick_query = None

    if quick_cols[0].button("Summarize", key="q_summary"):
        quick_query = "Summarize this medical report in simple, patient-friendly language."
    if quick_cols[1].button("Key findings", key="q_findings"):
        quick_query = "What are the most important findings from this report?"
    if quick_cols[2].button("Abnormal values", key="q_abnormal"):
        quick_query = "Explain any values that are outside the normal range."
    if quick_cols[3].button("Next steps", key="q_nextsteps"):
        quick_query = "Based on these results, what type of clinician or specialist should I speak with?"

    user_input = st.chat_input("Ask a question about your report...")
    query = quick_query or user_input

    if not query:
        return

    current_time = time.time()
    if current_time - st.session_state.get("last_qa_time", 0) < 1.5:
        st.warning("Please wait a moment before asking another question.")
        return

    st.session_state.last_qa_time = current_time
    st.session_state.patient_chat_history.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        combined_text = "\n\n---\n\n".join(
            f"Report: {r['filename']}\n{r['raw_text']}" for r in results
        )
        full_response = ""
        source_chunks = []
        source_metadata = []
        placeholder = st.empty()

        try:
            response = answer_patient_question(
                query=query,
                collection_name=collection_name,
                full_text_override=combined_text[:12000],
                stream=True,
            )

            source_chunks = response["source_chunks"]
            source_metadata = response["source_metadata"]

            for chunk in response["answer"]:
                if isinstance(chunk, str):
                    delta = chunk
                else:
                    choices = getattr(chunk, "choices", None)
                    delta = getattr(choices[0].delta, "content", None) if choices else None
                if delta:
                    full_response += delta
                    placeholder.markdown(full_response + "...")
            placeholder.markdown(full_response)

        except Exception:
            full_response = (
                "Unable to generate a response right now. Please check the configured "
                "LLM API key and try again."
            )
            source_chunks = []
            source_metadata = []
            placeholder.markdown(full_response)

        render_source_evidence(source_chunks, source_metadata)
        st.session_state.patient_chat_history.append(
            {
                "role": "assistant",
                "content": full_response,
                "sources": source_chunks,
                "source_meta": source_metadata,
            }
        )


def _render_export():
    """Render transcript export button."""
    if not st.session_state.get("patient_chat_history"):
        return

    st.markdown("### Export")
    export_text = "# Medical Analysis Transcript\n\n"
    export_text += "## Risk Assessment\n"
    for _, expl in st.session_state.get("patient_risk_explanations", {}).items():
        export_text += f"{expl}\n\n"
    export_text += "## Q&A History\n\n"
    for msg in st.session_state.patient_chat_history:
        role = "Patient" if msg["role"] == "user" else "AI Assistant"
        export_text += f"### {role}\n{msg['content']}\n\n"

    st.download_button(
        label="Download Analysis Transcript",
        data=export_text,
        file_name="medical_analysis.txt",
        mime="text/plain",
    )


def _render_empty_state():
    """Render the empty state before files are uploaded."""
    st.markdown(
        """<div class="empty-state">
            <h2>Upload reports to begin</h2>
            <p>PDF reports are processed into lab values, risk summaries, and grounded Q&A.</p>
        </div>
        <div class="feature-grid">
            <div class="feature-card">
                <div class="label">Triage</div>
                <h4>Smart extraction</h4>
                <p>Detects lab values, reference ranges, abnormal flags, and critical findings.</p>
            </div>
            <div class="feature-card">
                <div class="label">Explain</div>
                <h4>Plain-language answers</h4>
                <p>Turns technical report details into patient-friendly explanations.</p>
            </div>
            <div class="feature-card">
                <div class="label">Ground</div>
                <h4>Source evidence</h4>
                <p>Shows retrieved report sections beside AI answers for easier review.</p>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )
