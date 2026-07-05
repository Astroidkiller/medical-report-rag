# Architecture — Community Health Intelligence Assistant

## Overview

The Community Health Intelligence Assistant uses a **dual-mode architecture** with a **multi-agent pipeline** to serve both individual patients and community health stakeholders from the same platform.

---

## Agent Architecture (ADK-Style)

The system is decomposed into four specialized agents, each with a clear responsibility boundary. This mirrors the Agent Development Kit (ADK) pattern for production migration.

```
┌─────────────────────────────────────────────────────────────┐
│                     AGENT ORCHESTRATION                      │
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
│  │ • Risk Cards     │    │ • NL → SQL        │              │
│  │ • LLM Explain    │    │ • Trend Analysis  │              │
│  │ • Severity Map   │    │ • Alert Generate  │              │
│  └──────────────────┘    └──────────────────┘              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
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

### Why Two Stores?

| Store | Purpose | Query Pattern |
|---|---|---|
| **ChromaDB** | Semantic similarity search for RAG | "Find report sections about hemoglobin" → vector similarity |
| **SQLite** | Structured aggregate analysis | "What % of reports show elevated HbA1c?" → SQL GROUP BY |

Both are populated by the **Extraction Agent** at ingestion time. ChromaDB enables the patient-facing Q&A (unstructured), while SQLite enables the community-facing analytics (structured).

---

## GCP Migration Architecture

The current demo stack maps cleanly to GCP services for production scale:

```
┌────────────────── CURRENT (Demo) ──────────────────┐
│                                                     │
│  Streamlit → pdfplumber → SentenceTransformer      │
│     → ChromaDB → SQLite → Groq/Llama-3.1          │
│     → Plotly (in-app charts)                       │
│                                                     │
└─────────────────────────────────────────────────────┘
                        │
                   Migration
                        │
                        ▼
┌────────────────── TARGET (GCP) ─────────────────────┐
│                                                      │
│  Cloud Run (Streamlit)                               │
│     → Document AI (PDF processing)                   │
│     → Vertex AI Embeddings                           │
│     → Vertex AI Vector Search / AlloyDB              │
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

---

## Anomaly Detection Pipeline

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
┌─────────────────────┐
│ Risk Score           │
│                      │
│ (2×critical + 1×abn) │
│ ────────────────────  │
│     total_known       │
└────────┬────────────┘
         │
    ┌────┴────┐
    ▼         ▼
Individual   Aggregate
Risk Card    Community
(Patient)    Alerts
             (Dashboard)
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
