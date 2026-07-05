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
from core.lab_value_parser import parse_lab_values, parse_lab_values_from_tables
from core.anomaly_detector import flag_all_values, generate_risk_summary
from core.embeddings import embed_texts, store_chunks, clear_collection
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
    report_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()

    # Simulate demographics if not provided (hackathon demo)
    if anonymized_region is None:
        anonymized_region = random.choice(DEMO_REGIONS)
    if age_group is None:
        age_group = random.choice(DEMO_AGE_GROUPS)

    # 1. Extract text
    raw_text = extract_text_from_pdf(file_path)
    tables = extract_tables_from_pdf(file_path)

    # 2. Extract structured lab values
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

    # 3. Flag anomalies
    flagged_values = flag_all_values(all_lab_values)
    risk_summary = generate_risk_summary(flagged_values)

    # 4. Chunk text for vector store
    chunks = chunk_text(raw_text)

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

    return {
        "report_id": report_id,
        "filename": filename,
        "raw_text": raw_text,
        "chunks": chunks,
        "lab_values": all_lab_values,
        "flagged_values": flagged_values,
        "risk_summary": risk_summary,
        "region": anonymized_region,
        "age_group": age_group,
    }
