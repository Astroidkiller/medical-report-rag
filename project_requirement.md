# Project Requirements

This document outlines the detailed requirements of the **Community Health Intelligence Assistant** platform.

---

## 1. Scope & Objectives
The platform is a decision-intelligence and RAG-based workspace designed to solve two main challenges:
*   **Patient Understanding (Individual):** Translate complex, jargon-heavy PDF lab reports into friendly, simple, source-grounded summaries with visual risk cards.
*   **Community Health Operations (Population):** Aggregate and anonymize test results from individual PDFs to monitor local trends, detect anomaly spikes, and forecast disease patterns.

---

## 2. Functional Requirements

### A. Patient Mode (Individual Insight)
*   **PDF Report Ingestion:** Extract raw text and tables from standard diagnostic report PDFs (supporting multi-page reports).
*   **Lab Value Parser:** Identify test names, numerical values, units of measurement, and biological reference ranges.
*   **Anomaly Flagging:** Compare extracted lab values against both report-specific ranges and a built-in dictionary of 100+ common tests (e.g., CBC, lipid profile, LFT, KFT, thyroid, electrolytes). Categorize as:
    *   `NORMAL`
    *   `LOW` / `HIGH`
    *   `CRITICAL_LOW` / `CRITICAL_HIGH`
    *   `UNKNOWN` (when reference ranges are absent)
*   **Risk Scoring:** Generate a unified risk card outlining the overall status (Normal, Mild, Elevated, Critical) and severity distribution.
*   **Patient-Friendly Explanation:** Translate technical clinical flags into 5th-grade reading level text without medical jargon (e.g., explaining creatinine in terms of kidney filter efficiency). No emojis in descriptions.
*   **Source-Grounded Chat (RAG):** Patients must be able to ask natural language questions about their reports. The AI must retrieve exact report chunks, formulate responses restricted strictly to the document content, and display verbatim source evidence blocks.
*   **Disclaimers:** Mandatory display of a clear medical disclaimer on the dashboard header and within every generated summary.

### B. Community Mode (Population Operations)
*   **Bulk Uploading:** Allow health workers to select a Region/Locality and Age Group, then upload batches of anonymized report PDFs.
*   **Data Scrubber:** Anonymize and strip Patient Identifiable Information (PII) before storage.
*   **Analytics Dashboard:** Present population metrics:
    *   *Total Reports Analyzed*
    *   *Total Lab Values Logged*
    *   *Aggregate Anomaly Rate*
*   **Interactive Visualizations:**
    *   Horizontal bar chart of *Most Common Abnormal Findings*.
    *   Donut chart of *Flag Distribution*.
    *   Bar charts of *Anomaly Rate by Region* and *Anomaly Rate by Age Group*.
*   **Predictive Forecasting:** Implement linear regression forecasting to project abnormal rates for specific tests 7 to 90 days into the future. Plot historical rates against projected timelines.
*   **Active Community Alerts:** Automatically flag high-risk public health concerns (e.g., when abnormal HbA1c rates cross a 25% warning or 40% critical threshold).
*   **Natural Language Database Queries:** Health workers must be able to ask aggregate questions (e.g., "Which region has the highest rate of abnormal cholesterol this month?"). The agent translates queries into SQL, aggregates database counts, and returns a narrative summary with recommendations.

---

## 3. Non-Functional & System Requirements

*   **Low RAM footprint (Render Deploy):**
    *   Prevent memory-related crashes on resource-constrained cloud hosting (like free-tier Render).
    *   Configure lazy loading for database modules (ChromaDB) and embedding clients so dependencies are initialized only when files are uploaded.
*   **Direct REST API Calls:** Avoid SDK dependency version locks. Call Google Gemini via direct REST requests using the standard request library.
*   **Data Persistence & Isolation:**
    *   Store vector embeddings in a per-session directory for patient mode to isolate user data.
    *   Maintain an indexed SQLite database under `community_db/community.db` for aggregate population tracking.
*   **SQLite Optimization:** Enable Write-Ahead Logging (`WAL`) mode to support multi-thread concurrency during bulk uploads.
