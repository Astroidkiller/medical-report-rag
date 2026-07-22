"""FastAPI Server for Community Health Intelligence Assistant."""

import os
import json
import uuid
import shutil
import logging
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Ensure path imports work
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.extraction_agent import ingest_report
from agents.community_agent import get_dashboard_data, answer_community_question
from agents.qa_agent import answer_patient_question
from data_store.sqlite_store import (
    get_total_reports,
    get_total_lab_values,
    get_abnormal_rate,
    get_region_summary,
    get_age_group_summary,
    forecast_abnormal_trend,
    get_risk_forecast_by_region,
    get_test_trend,
    get_all_test_names,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Community Health Intelligence Assistant API",
    description="FHIR R4 & DP compliant backend for medical intelligence.",
    version="2.0.0"
)

# CORS configurations for seamless React integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adapt to specific domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temp storage for uploads
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "patient_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Mount React frontend build if present (for single-port Cloud Run deployment)
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "dist")
if os.path.exists(STATIC_DIR):
    from fastapi.staticfiles import StaticFiles
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ChatRequest(BaseModel):
    query: str
    collection_name: str
    combined_text: Optional[str] = None
    language: Optional[str] = "English"


class CommunityChatRequest(BaseModel):
    query: str


@app.get("/")
async def root():
    """Root health-check endpoint returning API info and available routes."""
    return {
        "status": "active",
        "service": "Community Health Intelligence Assistant API",
        "version": "2.0.0",
        "endpoints": [
            "/api/upload",
            "/api/chat/stream",
            "/api/community/dashboard",
            "/api/community/trends",
            "/api/community/tests",
            "/api/community/chat"
        ]
    }


@app.post("/api/upload")
async def upload_reports(
    files: List[UploadFile] = File(...),
    region: Optional[str] = Form(None),
    age_group: Optional[str] = Form(None),
    mode: Optional[str] = Form("patient"),
):
    """
    Ingests and parses medical report PDFs.
    Returns parsed lab results, risk summary, and FHIR Observation resources.
    """
    results = []
    session_id = f"session_{uuid.uuid4().hex[:12]}"
    session_dir = os.path.join(UPLOAD_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    for upload_file in files:
        # Sanitize filename
        filename = os.path.basename(upload_file.filename)
        file_path = os.path.join(session_dir, filename)

        with open(file_path, "wb") as f:
            shutil.copyfileobj(upload_file.file, f)

        try:
            # Run ingestion pipeline
            res = ingest_report(
                file_path=file_path,
                filename=filename,
                mode=mode,
                collection_name=session_id,
                anonymized_region=None if region == "Auto-assign (random)" else region,
                age_group=None if age_group == "Auto-assign (random)" else age_group,
                store_vectors=True,
            )
            # Remove raw local values before JSON output to keep payload lean
            res.pop("chunks", None)
            res.pop("lab_values", None)
            results.append(res)
        except Exception:
            # Clean up files on error
            shutil.rmtree(session_dir, ignore_errors=True)
            logger.exception("Failed to ingest uploaded file: %s", filename)
            raise HTTPException(status_code=500, detail=f"Failed to ingest {filename}. Please try again.")

    return {"session_id": session_id, "results": results}


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Server-Sent Events (SSE) streaming endpoint for Patient RAG Q&A.
    """
    async def sse_generator():
        try:
            # Call streaming RAG QA
            response = answer_patient_question(
                query=request.query,
                collection_name=request.collection_name,
                full_text_override=request.combined_text[:12000] if request.combined_text else None,
                stream=True,
                language=request.language
            )

            # Yield chunks as they generate
            for chunk in response["answer"]:
                yield f"data: {json.dumps({'type': 'chunk', 'text': chunk})}\n\n"

            # Yield source attributions at the end (Responsible AI grounding)
            yield f"data: {json.dumps({'type': 'sources', 'sources': response['source_chunks'], 'metadata': response['source_metadata']})}\n\n"
            yield "data: [DONE]\n\n"

        except Exception:
            logger.exception("Streaming chat request failed")
            yield f"data: {json.dumps({'type': 'error', 'detail': 'An internal error occurred. Please try again.'})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(sse_generator(), media_type="text/event-stream; charset=utf-8")


@app.get("/api/community/dashboard")
async def get_community_dashboard(use_dp: bool = Query(True)):
    """
    Retrieves aggregated population statistics, alert signals, and demographic
    summaries with Differential Privacy Laplace noise.
    """
    dash_data = get_dashboard_data(use_dp=use_dp)
    return dash_data


@app.get("/api/community/trends")
async def get_community_trends(
    test_name: str,
    days_ahead: Optional[int] = 30,
):
    """
    Returns time-series values and linear regression forecasting for a test.
    """
    available_tests = get_all_test_names()
    if not test_name or test_name not in available_tests:
         raise HTTPException(status_code=400, detail="Test parameter name invalid or empty")

    trend_data = get_test_trend(test_name)
    forecast = forecast_abnormal_trend(test_name, days_ahead)

    return {
        "historical": trend_data,
        "forecast": forecast
    }


@app.get("/api/community/tests")
async def get_tests():
    """Returns list of distinct tests recorded in the database."""
    return {"tests": get_all_test_names()}


@app.post("/api/community/chat")
async def community_chat(request: CommunityChatRequest):
    """
    Answers natural language queries about population health aggregates.
    """
    try:
        res = answer_community_question(request.query)
        return {"answer": res["answer"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# Serve static frontend files if compiled (for Cloud Run / Docker deployment)
frontend_dist = os.path.abspath(os.path.join(os.path.dirname(__file__), "frontend", "dist"))
if os.path.exists(frontend_dist):
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse

    assets_dir = os.path.abspath(os.path.join(frontend_dist, "assets"))
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API endpoint not found")
        # Secure path normalization & boundary validation against Path Injection (CodeQL py/path-injection)
        safe_root = os.path.abspath(frontend_dist)
        requested_path = os.path.abspath(os.path.normpath(os.path.join(safe_root, full_path)))

        if not requested_path.startswith(safe_root):
            raise HTTPException(status_code=403, detail="Access denied")

        if full_path and os.path.exists(requested_path) and os.path.isfile(requested_path):
            return FileResponse(requested_path)

        return FileResponse(os.path.join(safe_root, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
