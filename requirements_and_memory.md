# Requirements and Agent Memory

This document is a persistent context file for AI coding agents. It provides directory mapping, critical design decisions, coding rules, and system limitations to prevent hallucinations and ensure high performance.

---

## 1. Project Directory Mapping

```
├── streamlit_app.py          # App entry point (sidebar, modes, theme config)
├── config.py                 # Central configurations, reference ranges, directories
├── run.bat                   # Local run batch file
│
├── core/                     # Core processing scripts
│   ├── pdf_extractor.py      # Extract text/tables using pdfplumber
│   ├── chunker.py            # Splits text into overlapping parts for embeddings
│   ├── embeddings.py         # Handles vector embedding generation & ChromaDB storage
│   ├── lab_value_parser.py   # Regex parsers & LLM parser fallbacks
│   └── anomaly_detector.py   # Anomaly checks against reference ranges
│
├── agents/                   # Agent logic
│   ├── extraction_agent.py   # Ingestion pipeline orchestrator (combines parser + detector)
│   ├── qa_agent.py           # Patient RAG Q&A (responsible AI, source attribution)
│   ├── risk_agent.py         # Visual risk card values and patient-friendly reports
│   └── community_agent.py    # Public health dashboard queries & SQL translation
│
├── data_store/               # Database handlers
│   ├── models.py             # SQLite data schemas (ReportRecord, LabValueRecord, etc.)
│   ├── sqlite_store.py       # SQL aggregate operations and linear regression forecasts
│   └── vector_store.py       # Local ChromaDB connection wrapper
│
└── ui/                       # Interface rendering
    ├── styles.py             # Custom HSL palette CSS definitions
    ├── components.py         # Card layout templates (Risk Card, Lab Card, Metric Card)
    ├── patient_mode.py       # Patient-mode view (Q&A, single file upload)
    └── community_mode.py     # Community dashboard (multi-upload, regional heatmaps)
```

---

## 2. Critical Architecture Rules (Do Not Modify)

### A. SQLite Module Override (Render Patch)
*   **Location:** `streamlit_app.py` (lines 3-5)
*   **Rule:** The lines overriding `sqlite3` with `pysqlite3` must remain at the very top of `streamlit_app.py` before any other module imports:
    ```python
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    ```
*   **Reason:** Cloud hosts (like Render) run old versions of `sqlite3` in their system environments, which crashes modern `chromadb`. This override forces Streamlit to compile using the binary version installed via `requirements.txt`.

### B. Direct REST API calls (No SDK dependency)
*   **Location:** `core/llm_client.py`
*   **Rule:** Do not import or use `google-genai` or `google-generativeai` libraries. All Gemini requests (generate, stream, embeddings) must be executed using Python's standard `requests` library pointing to the Google REST API endpoint.
*   **Reason:** Standard SDKs create strict dependency version conflicts with Pillow, streamlit, and numpy, preventing deployment on web servers.

### C. Lazy Loading of High-Memory Libraries
*   **Rule:** Libraries like `chromadb` and `sentence-transformers` consume hundreds of megabytes of RAM. Never import them globally at the top of files that are executed during Streamlit startup. Import them inside the local functions that require them (e.g., inside embedding storage or query functions).
*   **Reason:** Prevents immediate OOM (Out Of Memory) crashes on free-tier web hosting.

### D. SQLite Concurrency (WAL Mode)
*   **Rule:** Keep `conn.execute("PRAGMA journal_mode=WAL")` in the SQLite connection helper (`data_store/sqlite_store.py`).
*   **Reason:** Enables fast, simultaneous read/write cycles during bulk PDF uploads in Community Mode without locking the database file.

---

## 3. Data Processing Constants & Thresholds

*   **RAG Chunking:** Max 300 characters, overlap 50 characters (`config.py`).
*   **Vector Search:** Top K results = 5 (`config.py`).
*   **Forecasting Horizon:** 7 to 90 days. Uses linear least-squares regression (`y = mx + b`) on historical daily aggregate rates. Requires at least 2 days of reports.
*   **Community Alert Thresholds:**
    *   *Warning Alert:* Triggered when abnormal rate for a test is `>= 20.0%` of logged readings.
    *   *Critical Alert:* Triggered when abnormal rate is `>= 40.0%`.
*   **Gemini Models:**
    *   *LLM:* `gemini-2.5-flash`
    *   *Embedding:* `gemini-embedding-2`

---

## 4. UI Refinements & Translation Memory

### A. Non-Markdown Formatting Policy
*   **Rule:** The LLM generator in `agents/qa_agent.py` must NOT output markdown heading tags (`##` or `###`) or bold brackets (`**`). Section breaks must be denoted by plain CAPITALIZED words.
*   **UI Parsing:** The frontend `App.jsx` handles basic bolding and horizontal lines using `renderTextFormat()` to guarantee safe rendering inside standard text containers.

### B. High-Fidelity Local Translation Guidelines (Tamil, Hindi, etc.)
*   **Rule:** When translating to local Indian languages (Tamil, Hindi, Bengali, Telugu, etc.), the translation must:
    1.  Use the native script characters (no English-transliterated text).
    2.  Use standard conversational, spoken phrasing suitable for patient comprehension.
    3.  Avoid dry literal machine translations of clinical jargon.
    4.  Preserve common abbreviations (like HbA1c or LOINC) in English.
*   **Encoding & Streaming Safeguards:** 
    1.  All streaming endpoints in `main.py` must return `media_type="text/event-stream; charset=utf-8"` to prevent browser encoding fallbacks.
    2.  All frontend stream readers in `App.jsx` must instantiate `decoder.decode(value, { stream: true })` to prevent multibyte characters from breaking when split across streaming chunks (Mojibake prevention).
    3.  All backend HTTP stream readers (e.g. `_stream_gemini_rest` in `core/llm_client.py`) must set `resp.encoding = 'utf-8'` immediately after `resp.raise_for_status()`. This prevents the Python `requests` library from guessing `ISO-8859-1` and corrupting multibyte UTF-8 characters prior to parsing.

### C. Performance Optimizations
*   **Vector Batching:** Text embeddings must be generated in a single HTTP request using the Gemini `batchEmbedContents` endpoint.
*   **Blinking Typing Indicators:** To ensure the user knows the AI is generating, the React chat bubble displays a CSS-styled `.typing-indicator` when `isChatSending` is true and text is empty.
*   **Ingestion Spinner:** CSS `.animate-spin` rules must be explicitly declared in Vanilla CSS to keep the PDF processor spinner animating smoothly.
*   **TRIP-RAG Selective Entity Anonymizer:** Raw text from PDF is de-identified using `anonymize_text()` in `agents/extraction_agent.py` before chunking and vector store storage. This prevents PHI (names, addresses, phone numbers, and dates) from leaking into the semantic embeddings.
*   **Collapsible Workspace Sidebar & Mobile Responsiveness:** The sidebar aside container toggles classes (`.collapsed`) linked to state, hiding the sidebar to 0px to reclaim visual canvas. The toggle button is a sibling of the sidebar, positioned using `position: fixed` centered vertically (`top: 50%`, `transform: translateY(-50%)`, with `left: 264px` when expanded, and `left: 16px` when collapsed) to avoid pointer-events or opacity inheritance issues. To ensure titles and subtitles are never cut off by the button, `.sidebar-title` and `.sidebar-subtitle` have a `padding-right: 20px` clearance offset. Responsive media queries in `index.css` handle tablet and mobile views (max-width 768px/480px) by stacking metrics/grids and converting the sidebar into a fixed sliding drawer overlay to preserve layout ratios on iPhone and Android screens.
*   **ESSENCE-inspired Aberration Detection & Seeding:** Public health monitoring evaluates spatiotemporal velocity aberrations using a CDC EARS C2-inspired Z-score thresholding on demographic clusters. Seeding mock reports during DB initialization ensures full analytical charts and forecasting inputs are available.
*   **Free Maps & Emergency First Aid Widgets:** Embedded Leaflet.js with browser geolocation APIs to find local clinics and medical stores. Placed a side-by-side First Aid Guide for high-acuity incident support below the upload panel.
*   **Sticky Split Workstation & Nav Separation:** Locked the sidebar to `100vh` sticky scrolling and styled buttons with border lines to prevent blending. Refactored the dashboard layout to a sticky split column (parameters on the left, RAG chat locked to the viewport on the right) to eliminate scroll strain.
*   **Fixed Sidebar & Navigation Redirects:** Desktop sidebar is locked with `position: fixed` to prevent it from scrolling away. The main dashboard content container shifts with a matching `margin-left` that transitions dynamically in sync with sidebar collapses. Medical locator places list contains clickable link cards that open search-focused Google Maps query URLs for directions.

