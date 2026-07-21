"""
Extraction Agent — orchestrates the full ingestion pipeline.

1. PDF → raw text (+ tables)
2. Text → structured lab values
3. Lab values → anomaly flags
4. Store in vector DB (for RAG) + SQLite (for aggregates)
"""

import uuid
import random
from datetime import datetime

from core.pdf_extractor import extract_text_from_pdf, extract_tables_from_pdf
from core.chunker import chunk_text
from core.lab_value_parser import parse_lab_values, parse_lab_values_from_tables, parse_all_lab_values_llm_fallback
from core.anomaly_detector import flag_all_values, generate_risk_summary
from core.embeddings import embed_texts, store_chunks, clear_collection
from core.anonymizer import anonymize_text
from data_store.sqlite_store import insert_report, insert_lab_values
from data_store.models import LabValueRecord, ReportRecord

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEMO_REGIONS, DEMO_AGE_GROUPS


def ingest_report(
    file_path: str,
    filename: str,
    mode: str = "patient",
    collection_name: str = "medical_report",
    anonymized_region: str = None,
    age_group: str = None,
    store_vectors: bool = True,
) -> dict:
    """
    Full ingestion pipeline for a medical report PDF.

    Args:
        file_path: Path to the uploaded PDF.
        filename: Original filename.
        mode: "patient" or "community".
        collection_name: ChromaDB collection name.
        anonymized_region: Region label (simulated if None).
        age_group: Age group label (simulated if None).

    Returns:
        Dict with report_id, chunks, flagged_values, risk_summary, raw_text.
    """
    import hashlib
    report_id = str(uuid.uuid4())
    patient_id = hashlib.sha256(filename.encode()).hexdigest()[:12]
    timestamp = datetime.now().isoformat()

    # Simulate demographics if not provided (hackathon demo)
    if anonymized_region is None:
        anonymized_region = random.choice(DEMO_REGIONS)
    if age_group is None:
        age_group = random.choice(DEMO_AGE_GROUPS)

    # 1. Extract text
    raw_text = extract_text_from_pdf(file_path)
    tables = extract_tables_from_pdf(file_path)

    # DEBUG: Log extraction results
    print(f"[EXTRACTION DEBUG] PDF text length: {len(raw_text)} chars, {len(raw_text.splitlines())} lines")
    print(f"[EXTRACTION DEBUG] Tables found: {len(tables)}")
    if tables:
        for ti, tbl in enumerate(tables):
            print(f"[EXTRACTION DEBUG] Table {ti}: {len(tbl)} rows, headers: {tbl[0] if tbl else 'EMPTY'}")
    if raw_text:
        # Print first 500 chars to see the format
        print(f"[EXTRACTION DEBUG] Text preview: {raw_text[:500]}")

    # 2. Parse values
    # Try LLM extraction first (highly robust, clinically focused, filters out address lines & noise)
    print("[EXTRACTION] Running primary LLM extraction...")
    all_lab_values = parse_all_lab_values_llm_fallback(raw_text, tables)
    print(f"[EXTRACTION] LLM extraction returned: {len(all_lab_values)} values")

    # If LLM fails or returns empty, run backup regex parsers
    if not all_lab_values:
        print("[EXTRACTION] LLM returned 0 values - running backup regex parser...")
        text_lab_values = parse_lab_values(raw_text)
        table_lab_values = parse_lab_values_from_tables(tables)
        
        # Merge, deduplicate by test name (prefer table-extracted if both exist)
        seen = {}
        for lv in table_lab_values:
            seen[lv.test_name.lower()] = lv
        for lv in text_lab_values:
            key = lv.test_name.lower()
            if key not in seen:
                seen[key] = lv
        all_lab_values = list(seen.values())
        print(f"[EXTRACTION] Backup regex parser returned: {len(all_lab_values)} values")

    # 3. Flag anomalies
    flagged_values = flag_all_values(all_lab_values)
    risk_summary = generate_risk_summary(flagged_values)

    # 4. Anonymize text to prevent PHI leakage, then chunk for vector store
    anonymized_text = anonymize_text(raw_text)
    chunks = chunk_text(anonymized_text)

    # 5. Store in vector DB
    if chunks and store_vectors:
        embeddings = embed_texts(chunks)
        metadata_list = [
            {"report_id": report_id, "filename": filename, "region": anonymized_region}
            for _ in chunks
        ]
        store_chunks(collection_name, chunks, embeddings, metadata_list, id_prefix=report_id)

    # 6. Store in SQLite for aggregate analysis
    report_record = ReportRecord(
        id=report_id,
        filename=filename,
        upload_timestamp=timestamp,
        total_tests=risk_summary["total"],
        normal_count=risk_summary["normal"],
        abnormal_count=risk_summary["abnormal"],
        critical_count=risk_summary["critical"],
        risk_score=risk_summary["risk_score"],
        anonymized_region=anonymized_region,
        age_group=age_group,
        mode=mode,
    )
    insert_report(report_record)

    lab_records = []
    for fv in flagged_values:
        lab_records.append(LabValueRecord(
            id=str(uuid.uuid4()),
            report_id=report_id,
            test_name=fv.test_name,
            value=fv.value,
            unit=fv.unit,
            reference_low=fv.reference_low,
            reference_high=fv.reference_high,
            flag=fv.flag,
            severity=fv.severity,
            timestamp=timestamp,
            anonymized_region=anonymized_region,
            age_group=age_group,
        ))

    if lab_records:
        insert_lab_values(lab_records)

    from core.fhir_models import create_fhir_observation_from_flagged_value
    from core.fhir_converter import build_fhir_bundle
    fhir_observations = [
        create_fhir_observation_from_flagged_value(fv, report_id, patient_id)
        for fv in flagged_values
    ]
    fhir_bundle = build_fhir_bundle(fhir_observations)

    return {
        "report_id": report_id,
        "patient_id": patient_id,
        "filename": filename,
        "raw_text": raw_text,
        "chunks": chunks,
        "lab_values": all_lab_values,
        "flagged_values": flagged_values,
        "fhir_observations": fhir_observations,
        "fhir_bundle": fhir_bundle,
        "risk_summary": risk_summary,
        "region": anonymized_region,
        "age_group": age_group,
    }
