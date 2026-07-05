"""Custom CSS styles for the Community Health Intelligence Assistant."""


def get_custom_css(theme: str | None = "Dark") -> str:
    """Return custom CSS for the Streamlit app."""
    theme_name = theme if isinstance(theme, str) else "Dark"
    is_light = theme_name.lower() == "light"
    palette = {
        "app_bg": "#f6f8fb" if is_light else "#0b1120",
        "surface": "#ffffff" if is_light else "#111827",
        "surface_2": "#eef2f7" if is_light else "#172033",
        "text": "#0f172a" if is_light else "#e5edf7",
        "muted": "#64748b" if is_light else "#97a3b6",
        "border": "#d9e2ef" if is_light else "#263247",
        "shadow": "0 16px 40px rgba(15, 23, 42, 0.08)" if is_light else "0 18px 48px rgba(0, 0, 0, 0.28)",
        "input": "#ffffff" if is_light else "#0f172a",
    }

    css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --app-bg: __APP_BG__;
        --surface: __SURFACE__;
        --surface-2: __SURFACE_2__;
        --text: __TEXT__;
        --muted: __MUTED__;
        --border: __BORDER__;
        --input-bg: __INPUT__;
        --shadow: __SHADOW__;
        --primary: #2563eb;
        --primary-soft: rgba(37, 99, 235, 0.12);
        --teal: #0f766e;
        --teal-soft: rgba(15, 118, 110, 0.12);
        --success: #16a34a;
        --success-soft: rgba(22, 163, 74, 0.12);
        --warning: #d97706;
        --warning-soft: rgba(217, 119, 6, 0.14);
        --danger: #dc2626;
        --danger-soft: rgba(220, 38, 38, 0.13);
        --critical: #b91c1c;
        --critical-soft: rgba(185, 28, 28, 0.16);
    }

    .stApp {
        font-family: 'Inter', sans-serif !important;
        background:
            radial-gradient(circle at top left, rgba(37, 99, 235, 0.10), transparent 32rem),
            linear-gradient(180deg, var(--app-bg), var(--app-bg));
        color: var(--text);
    }

    .block-container {
        max-width: 1280px;
        padding-top: 1.4rem;
        padding-bottom: 4rem;
    }

    section[data-testid="stSidebar"] {
        background: var(--surface);
        border-right: 1px solid var(--border);
    }

    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] label {
        color: var(--muted);
    }

    h1, h2, h3, h4, h5, h6 {
        color: var(--text);
        letter-spacing: 0;
    }

    p, li, span, label, div {
        letter-spacing: 0;
    }

    div[data-testid="stMetric"],
    div[data-testid="stExpander"],
    div[data-testid="stFileUploaderDropzone"],
    div[data-testid="stAlert"] {
        border-radius: 8px;
    }

    div[data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--border);
        padding: 14px 16px;
        box-shadow: var(--shadow);
    }

    div[data-testid="stFileUploaderDropzone"] {
        background: var(--surface);
        border: 1px dashed rgba(37, 99, 235, 0.45);
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 8px;
        border: 1px solid var(--border);
        background: var(--surface);
        color: var(--text);
        font-weight: 600;
        min-height: 42px;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        border-color: var(--primary);
        color: var(--primary);
    }

    .stChatInput textarea,
    textarea,
    input {
        background: var(--input-bg) !important;
        border-color: var(--border) !important;
        color: var(--text) !important;
    }

    .app-hero {
        border: 1px solid var(--border);
        border-radius: 8px;
        background:
            linear-gradient(135deg, rgba(37, 99, 235, 0.16), rgba(15, 118, 110, 0.10)),
            var(--surface);
        padding: 24px 28px;
        margin: 0 0 18px 0;
        box-shadow: var(--shadow);
    }

    .app-kicker {
        color: var(--teal);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .app-hero h1 {
        margin: 0 0 8px 0;
        font-size: clamp(1.8rem, 3vw, 3rem);
        line-height: 1.05;
    }

    .app-hero p {
        max-width: 820px;
        margin: 0;
        color: var(--muted);
        font-size: 1rem;
        line-height: 1.6;
    }

    .sidebar-title {
        font-size: 1rem;
        font-weight: 800;
        color: var(--text);
        margin: 0;
    }

    .sidebar-subtitle {
        color: var(--muted);
        font-size: 0.82rem;
        margin: 2px 0 14px;
    }

    .mode-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 12px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0;
        margin-bottom: 8px;
    }

    .mode-badge-patient {
        background: var(--primary-soft);
        color: var(--primary);
        border: 1px solid rgba(37, 99, 235, 0.25);
    }

    .mode-badge-community {
        background: var(--teal-soft);
        color: var(--teal);
        border: 1px solid rgba(15, 118, 110, 0.25);
    }

    .section-band {
        border: 1px solid var(--border);
        border-radius: 8px;
        background: var(--surface);
        padding: 18px 20px;
        margin: 14px 0;
        box-shadow: var(--shadow);
    }

    .section-band h3 {
        margin: 0 0 6px 0;
        font-size: 1.1rem;
    }

    .section-band p {
        margin: 0;
        color: var(--muted);
        line-height: 1.55;
    }

    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 14px;
        margin-top: 18px;
    }

    .feature-card {
        border: 1px solid var(--border);
        border-radius: 8px;
        background: var(--surface);
        padding: 18px;
        min-height: 132px;
    }

    .feature-card .label {
        color: var(--primary);
        font-weight: 800;
        font-size: 0.78rem;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .feature-card h4 {
        margin: 0 0 8px 0;
        font-size: 1rem;
    }

    .feature-card p {
        margin: 0;
        color: var(--muted);
        font-size: 0.9rem;
        line-height: 1.5;
    }

    .risk-card {
        border-radius: 8px;
        padding: 20px;
        margin: 14px 0;
        border: 1px solid var(--border);
        background: var(--surface);
        position: relative;
        overflow: hidden;
        box-shadow: var(--shadow);
    }

    .risk-card::before {
        content: '';
        position: absolute;
        inset: 0 auto 0 0;
        width: 5px;
    }

    .risk-card-critical { background: linear-gradient(90deg, var(--critical-soft), transparent 55%), var(--surface); }
    .risk-card-critical::before { background: var(--critical); }
    .risk-card-elevated { background: linear-gradient(90deg, var(--warning-soft), transparent 55%), var(--surface); }
    .risk-card-elevated::before { background: var(--warning); }
    .risk-card-mild { background: linear-gradient(90deg, var(--primary-soft), transparent 55%), var(--surface); }
    .risk-card-mild::before { background: var(--primary); }
    .risk-card-normal { background: linear-gradient(90deg, var(--success-soft), transparent 55%), var(--surface); }
    .risk-card-normal::before { background: var(--success); }

    .risk-card h3 {
        margin: 0 0 8px 0;
        font-size: 1.15rem;
        font-weight: 800;
    }

    .risk-card p {
        margin: 0;
        color: var(--muted);
        font-size: 0.94rem;
        line-height: 1.55;
    }

    .lab-card {
        border-radius: 8px;
        padding: 15px;
        margin: 8px 0;
        border: 1px solid var(--border);
        background: var(--surface);
        min-height: 124px;
    }

    .lab-card-normal { border-left: 4px solid var(--success); }
    .lab-card-high, .lab-card-low { border-left: 4px solid var(--warning); }
    .lab-card-critical_high, .lab-card-critical_low { border-left: 4px solid var(--danger); }

    .lab-card .lab-name {
        font-weight: 700;
        font-size: 0.94rem;
        margin-bottom: 4px;
    }

    .lab-card .lab-value {
        font-size: 1.45rem;
        font-weight: 800;
        margin: 8px 0 4px;
    }

    .lab-card .lab-ref {
        color: var(--muted);
        font-size: 0.8rem;
    }

    .flag-badge {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 999px;
        font-size: 0.70rem;
        font-weight: 800;
        text-transform: uppercase;
    }

    .flag-normal { background: var(--success-soft); color: var(--success); }
    .flag-high { background: var(--warning-soft); color: var(--warning); }
    .flag-low { background: var(--primary-soft); color: var(--primary); }
    .flag-critical { background: var(--danger-soft); color: var(--danger); }

    .metric-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 18px;
        min-height: 112px;
        box-shadow: var(--shadow);
    }

    .metric-card .metric-value {
        color: var(--primary);
        font-size: 2rem;
        font-weight: 800;
        line-height: 1;
    }

    .metric-card .metric-label {
        color: var(--muted);
        font-size: 0.78rem;
        margin-top: 8px;
        text-transform: uppercase;
        font-weight: 700;
    }

    .alert-card {
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
        display: flex;
        align-items: flex-start;
        gap: 12px;
        border: 1px solid var(--border);
        background: var(--surface);
    }

    .alert-warning {
        border-color: rgba(217, 119, 6, 0.36);
        background: var(--warning-soft);
    }

    .alert-critical {
        border-color: rgba(220, 38, 38, 0.36);
        background: var(--danger-soft);
    }

    .alert-icon {
        font-size: 0.75rem;
        line-height: 1;
        padding: 5px 8px;
        border-radius: 999px;
        background: var(--surface);
        font-weight: 800;
        color: var(--danger);
    }

    .alert-text {
        color: var(--text);
        font-size: 0.9rem;
        line-height: 1.5;
    }

    .source-chunk {
        background: var(--surface-2);
        border: 1px solid var(--border);
        border-left: 3px solid var(--primary);
        border-radius: 8px;
        padding: 12px 14px;
        margin: 8px 0;
        color: var(--text);
        font-size: 0.86rem;
        line-height: 1.6;
    }

    .source-label {
        color: var(--muted);
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
    }

    .disclaimer-bar {
        background: var(--warning-soft);
        border: 1px solid rgba(217, 119, 6, 0.28);
        border-radius: 8px;
        padding: 12px 14px;
        font-size: 0.85rem;
        line-height: 1.5;
        margin-bottom: 16px;
    }

    .empty-state {
        text-align: center;
        border: 1px dashed var(--border);
        border-radius: 8px;
        padding: 48px 22px;
        background: var(--surface);
        margin: 16px 0;
    }

    .empty-state h2 {
        margin: 0 0 8px;
        font-size: 1.45rem;
    }

    .empty-state p {
        margin: 0;
        color: var(--muted);
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .animate-in {
        animation: fadeInUp 0.4s ease-out;
    }

    .stChatMessage {
        border-radius: 8px !important;
        border: 1px solid var(--border);
        background: var(--surface);
    }

    @media (max-width: 820px) {
        .feature-grid {
            grid-template-columns: 1fr;
        }

        .app-hero {
            padding: 20px;
        }
    }
</style>
"""
    return (
        css.replace("__APP_BG__", palette["app_bg"])
        .replace("__SURFACE__", palette["surface"])
        .replace("__SURFACE_2__", palette["surface_2"])
        .replace("__TEXT__", palette["text"])
        .replace("__MUTED__", palette["muted"])
        .replace("__BORDER__", palette["border"])
        .replace("__INPUT__", palette["input"])
        .replace("__SHADOW__", palette["shadow"])
    )
