# Repository Activity Log

This log tracks all major development milestones, features, fixes, and code migrations in the **Community Health Intelligence Assistant** repository.

## Development Phases & History

### Phase 1: Core Modular Ingestion (Initial Commit to Feat/Modular RAG)
*   **Modular Medical Report RAG:** Set up the modular PDF extraction pipeline using `pdfplumber` and agentic modules.
*   **Custom SQLite Data Store:** Created a local SQL representation of parsed laboratory test values to serve as an analytics backend.
*   **Git Config:** Configured `.gitignore` to exclude raw SQLite databases (`.db`), vector stores (`chroma_db/`), and temporary PDF uploads (`data/`).

### Phase 2: User Interface & Deployment (Streamlit Transition)
*   **Modular Streamlit App:** Migrated from command-line interface (`app.py`) to a dual-mode Streamlit dashboard (`streamlit_app.py`).
*   **ASHA/Community Health workers mode:** Added bulk report analysis, regional trend visualizations, and short-term forecasting charts using Plotly.
*   **Render Compatibility patches:**
    *   Added `pysqlite3-binary` configuration override at the top of `streamlit_app.py` to fix ChromaDB SQLite dependency conflicts on cloud hosts.
    *   Set up lazy loading of major modules (like ChromaDB and sentence-transformers) to reduce startup RAM usage and prevent Render OOM (Out Of Memory) crashes.
    *   Switched default embedding model to Gemini API REST to avoid heavy local models.

### Phase 3: Robust Parser & Client Stability (SDK Replaced)
*   **Replacing Google-GenAI SDK:** Removed the standard Google GenAI SDK dependency and replaced all calls with direct REST API requests in `core/llm_client.py`. This fixed persistent conflicts between Pillow, pdfplumber, and Google libraries.
*   **Robust LLM Fallback Parser:** Added `parse_all_lab_values_llm_fallback` to recover parameters from complex tables and text rows when regex parser misses them.
*   **Diagnostic Tools:** Added visible extraction expanders and debugger tools in the UI when a report returns 0 extracted tests.
*   **Tacrolimus range:** Added reference ranges for immunosuppressants and handled qualitative results (e.g. Negative = 0.0).

### Phase 4: Decoupled Rebuild & Interface Refinements
*   **React + FastAPI Rebuild:** Replaced the monolithic Streamlit dashboard with a decoupled React frontend (deployed via static CDN/Vite) and FastAPI backend (deployed via serverless Cloud Run).
*   **FHIR R4 observations & LOINC mappings:** Added standard FHIR Observations pydantic models and converted parsed lab results into FHIR objects, mapping tests to global LOINC codes.
*   **Differential Privacy:** Implemented a Laplace noise aggregate algorithm ($\epsilon = 0.5$) in SQLite aggregate metrics to enforce database privacy guarantees.
*   **SSE Chat Streaming:** Engineered real-time chat tokens streaming using Server-Sent Events (SSE).
*   **20x Ingestion Speedup:** Optimized embedding requests to batch all chunk embeddings into a single HTTP request using Gemini's `batchEmbedContents` endpoint.
*   **Blinking Indicators & Spinner Animation:** Declared the CSS spin animations and blinking dot indicators to give a premium clinical loading state.
*   **Markdown-to-HTML formatter:** Added a client-side parser to cleanly render bold and headings without leaving raw `**` or `##` markdown symbols in text.
*   **Encoding Fixes for Local Languages:** Configured explicit UTF-8 charsets on StreamingResponses in `main.py`, forced `resp.encoding = 'utf-8'` on the Google API stream requests in `core/llm_client.py` to prevent Python requests from defaulting to ISO-8859-1, and enabled `{ stream: true }` decoding state in frontend `TextDecoder` calls to prevent split multibyte character corruption (Mojibake) in Hindi, Tamil, etc.
*   **TRIP-RAG Selective Entity Anonymization:** Added `core/anonymizer.py` and integrated it in `agents/extraction_agent.py` to scrub patient names, referring doctors, email addresses, phone numbers, and physical addresses from raw text before chunking and vector storage, preventing PHI leakage in downstream semantic search.
*   **Collapsible Sidebar & Mobile Responsiveness:** Built a slide-to-hide sidebar collapse feature in React (`App.jsx` & `index.css`) with a floating toggle control button (anchored to the right border line), and added responsive media queries for Apple iPhone & Android Samsung screen ratios (stacking metrics in 1/2 columns and translating the sidebar into a sliding slide-out drawer on narrow screens).
*   **ESSENCE-inspired Aberration Detection & Historical Seeding:** Implemented a CDC EARS C2-inspired Z-score spatiotemporal aberration algorithm in `data_store/sqlite_store.py` to identify rapid spikes/velocities of anomalies across region x age demographics, and built a database initialization seeder that populates 30 days of mock historical reports and a target HbA1c spike in Urban-Central, rendering full charts and alerts on first load.
*   **Free Hospital/Pharmacy Map & First Aid Widgets:** Added a keyless Leaflet.js locator map (fetching geolocation coordinates and displaying nearest medical points) and an interactive First Aid Guide (addressing cardiac arrest, seizures, snake bites, bleeding, choking) below the upload panel in Patient Mode.
*   **Sticky Split Clinical Layout:** Redesigned active/inactive sidebar pills to avoid background blending and lock the menu to `100vh`, and refactored the ingested Patient Dashboard into a 2-column workstation split (report parameters on the left, sticky RAG chat floating on the right).
*   **Fixed Desktop Sidebar & Map Redirects:** Reconfigured the sidebar layout to `position: fixed` to completely prevent the menu from scrolling away on long dashboards, and made the hospital and pharmacy listing cards clickable, redirecting to active directions via Google Maps query URLs.
*   **Centered Dashboard Panel Alignment:** Reverted `.dashboard-main` margins to `0 auto` to keep it perfectly centered on the screen, moving the fixed sidebar offset shift to dynamic `padding-left` toggling on the outer `.stApp-container` wrapper.
*   **OSM Overpass API Integration:** Replaced the local mock clinic locations with a dynamic OpenStreetMap Overpass API POST interpreter. The app now fetches real nearby hospital, clinic, and pharmacy coordinates relative to the user's active browser geolocation, sorts them by distance, and updates Leaflet map markers.
*   **Multi-Tier Geolocation Resolver:** Configured a three-stage fallback system: browser geolocation (GPS), IP-based geolocating (`ipapi.co`) to locate users when coordinates are inaccurate or blocked, and a manual Search Box integrated with Nominatim geocoding to search any city, zip code, or address and re-center the map.
*   **Leaflet Map Lifecycle Fix:** Decoupled map initialization, user tracking, and marker loading from the API places response lifecycle. The map now instantiates once on boot and smoothly pans to coordinates via `.setView()`, preventing visual lockups or race conditions when resolving geographic locations. Integrated `ipinfo.io` as a high-reliability fallback for CORS/SSL constraints.
*   **User-Initiated Geocode & Disambiguation UI:** Disabled all automatic mount GPS/IP geolocation. The map now prompts the user to type their location. Nominatim is used to handle spelling errors (fuzzy lookup) and if duplicate matching cities exist (e.g. Hyderabad in India vs Pakistan), the app displays a confirmation list to let the user select the correct coordinates.

---

## Detailed Commit Log (Latest First)

| Commit Hash | Commit Message Summary | Key Changes / Impacts |
| :--- | :--- | :--- |
| `94d70f3` | feat: switch to user-initiated search with fuzzy spelling resolver and disambiguation confirmation list | Added fuzzy search spelling resolver and duplicate city confirmations |
| `2f38145` | fix: resolve leaflet rendering lifecycle race conditions and add ipinfo.io resolver fallback | Resolved map lockups and improved IP geolocation resolution |
| `1fc3a80` | feat: implement multi-tier geolocator (browser GPS, IP-based lookup, and Nominatim address search) | Added geolocator fallbacks and search bar |
| `3dd4137` | feat: query OpenStreetMap Overpass API for real nearby clinics and pharmacies | Integrated dynamic OSM Overpass API node queries on map |
| `7124ca6` | fix: resolve dashboard-main shifting by offset padding-left on outer stApp-container | Fixed container centering issues during sidebar toggle events |
| `c0f837c` | feat: make sidebar layout fixed desktop-side and add map navigation redirect links | Locked sidebar position and added LeafletMap anchor navigation links |
| `4bd8992` | feat: implement medical locator, first aid guide, sidebar style fix, and sticky split chat layout | Pushed locator map, first aid guide, sidebar pill fixes, and split sticky chat |
| `8b5b7e4` | fix: resolve React TypeError on trendData.forecast.map by mapping forecast_data | Fixed white screen crash on community dashboard render |
| `f65e4bb` | feat: implement ESSENCE-inspired spatiotemporal aberration detection and historical db seeding | Added EARS Z-score calculations and 30-day mock seeding |
| `9a1646c` | Simplify AI summaries: plain language, clean headings, no emojis | Updated patient summary prompts in `risk_agent.py` |
| `21bc392` | fix: Update default model to gemini-2.5-flash & embeddings | Upgraded model defaults and masked API keys in errors |
| `b40a12b` | fix: mask API key in all error messages to prevent leakage | Hardened security in REST call exceptions |
| `0c72fcb` | fix: change default model to gemini-1.5-flash | Resolved temporary flash-2.0 rate limits |
| `548577f` | fix: add Tacrolimus range & render unassessed values | Expanded reference range dict in `config.py` |
| `5cb2eb4` | debug: add diagnostic expander to patient mode | Surface text extraction failures in patient UI |
| `b4a6f75` | debug: add visible extraction diagnostics to UI | Added verbose print logging during parsing |
| `0bccb63` | fix: replace google-genai SDK with direct REST API | **Major stability change** - eliminated SDK dependency conflicts |
| `54e186e` | fix: unpin pdfplumber to resolve Pillow conflict | Resolved dependency locks |
| `bf0256b` | fix: correct SDK version and surface LLM errors | Improved traceback handling |
| `18ebed0` | feat: improve LLM fallback to parse entire report | Enhanced prompts for raw table extraction |
| `33067b9` | fix: improve regex patterns and add LLM fallback | Merged table parser and text parser |
| `0104af4` | fix: lazy load chromadb to prevent memory usage | Boosted startup performance |
| `dc9183b` | perf: use gemini api for embeddings | Avoided rendering OOM crash |
| `1860dcc` | fix: update pysqlite3-binary version | Reconfigured Render compilation packages |
| `fe9595d` | fix: add store_vectors parameter to ingest_report | Allows switching vector ingestion off |
| `a20919e` | fix: add pysqlite3 patch to support chromadb on render | Fixed SQLite version issues |
| `750e724` | Redesign Streamlit UI and harden deployment config | Styled metrics, headers, and dashboard sheets |
| `f304805` | feat: implement community mode dashboard | Added bulk upload, Plotly charts, alerts, and forecasts |
| `05fbf78` | feat: implement modular medical report RAG system | Created multi-agent architecture and PDF pipeline |
| `7c6078f` | chore: update gitignore | Excluded logs, databases, and uploaded PDFs |
