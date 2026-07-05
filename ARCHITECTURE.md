# Architecture — Community Health Intelligence Assistant

## Overview

The Community Health Intelligence Assistant is a **multi-agent RAG platform** that transforms individual medical report PDFs into **population-level health intelligence**. It serves both individual patients and community health stakeholders (clinics, ASHA workers, local health departments).

### Key Capabilities
- **Report Ingestion**: PDF → structured lab values → anomaly flags → vector + SQL storage
- **Anomaly Detection**: Individual-level flagging + population-level trend analysis
- **RAG-based Q&A**: Semantic search + LLM explanation for patient reports
- **Community Intelligence**: Aggregate analytics, demographic clustering, seasonal spike detection
- **Multi-Agent Pipeline**: ADK-style orchestration with 4 specialized agents

---

## Technology Stack

```
┌─────────────────── PRIMARY (GCP) ──────────────────┐
│                                                     │
│  CLI (app.py)                                       │
│     → pdfplumber (PDF extraction)                   │
│     → Vertex AI Gemini (LLM — gemini-2.0-flash)    │
│     → SentenceTransformers / Vertex AI Embeddings   │
│     → ChromaDB / AlloyDB pgvector (vector search)   │
│     → SQLite (structured lab values & aggregates)   │
│     → Multi-Agent Orchestrator (ADK-style)          │
│                                                     │
└─────────────────────────────────────────────────────┘
                    │
             GCP Migration Path
                    │
                    ▼
┌─────────────────── TARGET (GCP Full) ───────────────┐
│                                                      │
│  Cloud Run (API)                                     │
│     → Document AI (PDF processing)                   │
│     → Vertex AI Embeddings (text-embedding-005)      │
│     → AlloyDB pgvector (vector search)               │
│     → BigQuery (structured lab values)               │
│     → Vertex AI Gemini (LLM)                         │
│     → Looker Studio (dashboards)                     │
│     → Agent Development Kit (ADK)                    │
│                                                      │
│  Cloud Pub/Sub → Cloud Functions (alert triggers)    │
│  Cloud IAM → per-clinic access control               │
│  Cloud Healthcare API → FHIR integration             │
│                                                      │
└──────────────────────────────────────────────────────┘
```

### Backend Switching

The system supports dual-mode operation via environment variables:

| Component | Local (Dev) | GCP (Production) |
|---|---|---|
| **LLM** | Google Gemini 2.0 Flash | Vertex AI Gemini 2.0 Flash |
| **Embeddings** | SentenceTransformers (all-MiniLM-L6-v2) | Vertex AI Embeddings (text-embedding-005) |
| **Vector Store** | ChromaDB (ephemeral) | AlloyDB pgvector |
| **Structured Data** | SQLite | BigQuery |

---

## Agent Architecture (ADK-Style)

The system uses four specialized agents coordinated by a central orchestrator. Each agent has a clear input/output contract and can be independently tested.

```
┌─────────────────────────────────────────────────────────────┐
│                   ORCHESTRATOR (orchestrator.py)              │
│                                                              │
│   Session state: report_ids, risk_cards, anomalies           │
│                                                              │
│  ┌──────────────────┐    ┌──────────────────┐              │
│  │ EXTRACTION AGENT │    │    QA AGENT       │              │
│  │                  │    │                   │              │
│  │ • PDF → Text     │    │ • Semantic Search │              │
│  │ • Table Extract  │    │ • Context Build   │              │
│  │ • Lab Parsing    │    │ • LLM Q&A         │              │
│  │ • Anomaly Flag   │    │ • Source Attrib.  │              │
│  │ • Store Data     │    │                   │              │
│  └──────────────────┘    └──────────────────┘              │
│                                                              │
│  ┌──────────────────┐    ┌──────────────────┐              │
│  │   RISK AGENT     │    │ COMMUNITY AGENT  │              │
│  │                  │    │                   │              │
│  │ • Risk Scoring   │    │ • Aggregate Query │              │
│  │ • Risk Cards     │    │ • NL → Insights   │              │
│  │ • LLM Explain    │    │ • Trend Analysis  │              │
│  │ • Severity Map   │    │ • Alert Generate  │              │
│  └──────────────────┘    └──────────────────┘              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Pipeline Flows

**Patient Pipeline**: `Ingest → Risk Analysis → QA`
```
PDF Upload → Extraction Agent → Risk Agent → QA Agent (interactive)
```

**Community Pipeline**: `Batch Ingest → Population Analysis → Community Q&A`
```
Multiple PDFs → Extraction Agent (batch) → Risk Agent (batch)
    → Population Anomaly Detection → Community Agent (interactive)
```

---

## Data Architecture

### Dual Storage Strategy

```
                    PDF Upload
                        │
                        ▼
               ┌─────────────────┐
               │ Extraction Agent│
               └────┬───────┬────┘
                    │       │
            ┌───────┘       └────────┐
            ▼                        ▼
    ┌───────────────┐     ┌──────────────────┐
    │   ChromaDB    │     │    SQLite DB      │
    │  (Vectors)    │     │  (Structured)     │
    │               │     │                   │
    │ • Text chunks │     │ • Lab values      │
    │ • Embeddings  │     │ • Report metadata │
    │ • Metadata    │     │ • Flags/severity  │
    │               │     │ • Region/age      │
    │ Used by:      │     │ • Timestamps      │
    │ → QA Agent    │     │                   │
    │ (similarity   │     │ Used by:          │
    │  search)      │     │ → Community Agent │
    │               │     │ (aggregate SQL)   │
    └───────────────┘     └──────────────────┘
```

---

## Anomaly Detection Pipeline

### Individual-Level (per report)

```
Lab Value Extracted
        │
        ▼
┌─────────────────────┐
│ Reference Range      │
│ Lookup               │
│                      │
│ 1. Report-provided   │
│    ranges (priority) │
│ 2. Built-in ranges   │
│    (100+ tests)      │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Flag Assignment      │
│                      │
│ NORMAL:  in range    │
│ HIGH:    > ref_high  │
│ LOW:     < ref_low   │
│ CRIT_HI: > critical  │
│ CRIT_LO: < critical  │
│ UNKNOWN: no range    │
└────────┬────────────┘
         │
         ▼
    Risk Score & Card
```

### Population-Level (across reports) — NEW

```
All Lab Records (SQLite)
         │
    ┌────┼────────────────┐
    ▼    ▼                ▼
┌────────┐  ┌──────────┐  ┌────────────────┐
│Elevated│  │ Seasonal │  │ Demographic    │
│Rate    │  │ Spikes   │  │ Clusters       │
│        │  │          │  │                │
│ % abn. │  │ Current  │  │ age × region   │
│ > 25%  │  │ vs hist  │  │ cross-tab      │
│ by test│  │ > 1.5x   │  │ > 35% abnormal │
└────────┘  └──────────┘  └────────────────┘
    │            │                │
    └────────────┼────────────────┘
                 ▼
    Population Anomaly Alerts
    → Community Dashboard
    → Health Worker Recommendations
```

---

## Responsible AI Architecture

```
┌─────────────────────────────────────────────────────┐
│               RESPONSIBLE AI LAYER                   │
│                                                      │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ Source       │  │ Privacy      │  │ Safety     │ │
│  │ Attribution  │  │ Protection   │  │ Guardrails │ │
│  │             │  │              │  │            │ │
│  │ • Retrieved │  │ • No PII in  │  │ • Medical  │ │
│  │   chunks    │  │   responses  │  │   disclaim │ │
│  │   shown     │  │ • Anonymized │  │ • No diag- │ │
│  │ • Source    │  │   community  │  │   nosis    │ │
│  │   labels    │  │   data       │  │ • Prompt   │ │
│  │ • Evidence  │  │ • System     │  │   injection│ │
│  │   expander  │  │   prompt     │  │   defense  │ │
│  │             │  │   blocks PII │  │            │ │
│  └─────────────┘  └──────────────┘  └────────────┘ │
│                                                      │
└─────────────────────────────────────────────────────┘
```

Every answer passes through all three layers before reaching the user.

---

## LLM Client Architecture

The unified LLM client (`core/llm_client.py`) abstracts provider-specific SDKs:

```
┌─────────────────────────────────┐
│      core/llm_client.py         │
│                                 │
│  generate(prompt, system)       │
│  generate_stream(prompt, sys)   │
│                                 │
│  ┌─────────┐    ┌──────────┐  │
│  │ Gemini  │    │Vertex AI │  │
│  │ Gemini  │    │ (fallback│  │
│  │         │    │  for dev)│  │
│  └─────────┘    └──────────┘  │
└─────────────────────────────────┘
         ▲
         │
    All 4 agents import from here
    (no direct SDK coupling)
```
