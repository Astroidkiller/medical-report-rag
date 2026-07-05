"""
Custom CSS styles for the Community Health Intelligence Assistant.
Modern, premium design with dark mode support and color-coded severity.
"""


def get_custom_css() -> str:
    """Return custom CSS for the Streamlit app."""
    return """
<style>
    /* ---------- IMPORTS ---------- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ---------- ROOT VARIABLES ---------- */
    :root {
        --primary: #6366f1;
        --primary-light: #818cf8;
        --primary-dark: #4f46e5;
        --success: #22c55e;
        --success-bg: rgba(34, 197, 94, 0.1);
        --warning: #f59e0b;
        --warning-bg: rgba(245, 158, 11, 0.1);
        --danger: #ef4444;
        --danger-bg: rgba(239, 68, 68, 0.1);
        --critical: #dc2626;
        --critical-bg: rgba(220, 38, 38, 0.15);
        --info: #3b82f6;
        --info-bg: rgba(59, 130, 246, 0.1);
        --bg-dark: #0f172a;
        --bg-card: #1e293b;
        --bg-card-hover: #334155;
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --border: #334155;
        --gradient-1: linear-gradient(135deg, #6366f1, #8b5cf6);
        --gradient-2: linear-gradient(135deg, #3b82f6, #06b6d4);
        --gradient-danger: linear-gradient(135deg, #ef4444, #f97316);
        --gradient-success: linear-gradient(135deg, #22c55e, #14b8a6);
    }

    /* ---------- GLOBAL OVERRIDES ---------- */
    .stApp {
        font-family: 'Inter', sans-serif !important;
    }

    /* ---------- HEADER BADGE ---------- */
    .mode-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    .mode-badge-patient {
        background: var(--info-bg);
        color: var(--info);
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    .mode-badge-community {
        background: rgba(139, 92, 246, 0.1);
        color: #a78bfa;
        border: 1px solid rgba(139, 92, 246, 0.3);
    }

    /* ---------- RISK CARD ---------- */
    .risk-card {
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
        border: 1px solid var(--border);
        position: relative;
        overflow: hidden;
    }
    .risk-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
    }
    .risk-card-critical {
        background: var(--critical-bg);
        border-color: rgba(220, 38, 38, 0.4);
    }
    .risk-card-critical::before { background: var(--gradient-danger); }

    .risk-card-elevated {
        background: var(--warning-bg);
        border-color: rgba(245, 158, 11, 0.4);
    }
    .risk-card-elevated::before { background: linear-gradient(135deg, #f59e0b, #fbbf24); }

    .risk-card-mild {
        background: var(--info-bg);
        border-color: rgba(59, 130, 246, 0.4);
    }
    .risk-card-mild::before { background: var(--gradient-2); }

    .risk-card-normal {
        background: var(--success-bg);
        border-color: rgba(34, 197, 94, 0.4);
    }
    .risk-card-normal::before { background: var(--gradient-success); }

    .risk-card h3 {
        margin: 0 0 8px 0;
        font-size: 1.3rem;
        font-weight: 700;
    }
    .risk-card p {
        margin: 0;
        font-size: 0.95rem;
        opacity: 0.9;
    }

    /* ---------- LAB VALUE CARD ---------- */
    .lab-card {
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
        border: 1px solid var(--border);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .lab-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    .lab-card-normal { border-left: 4px solid var(--success); }
    .lab-card-high, .lab-card-low { border-left: 4px solid var(--warning); }
    .lab-card-critical_high, .lab-card-critical_low { border-left: 4px solid var(--danger); }

    .lab-card .lab-name {
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 4px;
    }
    .lab-card .lab-value {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .lab-card .lab-ref {
        font-size: 0.8rem;
        opacity: 0.7;
    }

    /* ---------- FLAG BADGES ---------- */
    .flag-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .flag-normal { background: var(--success-bg); color: var(--success); }
    .flag-high { background: var(--warning-bg); color: var(--warning); }
    .flag-low { background: var(--info-bg); color: var(--info); }
    .flag-critical { background: var(--danger-bg); color: var(--danger); }

    /* ---------- METRIC CARD ---------- */
    .metric-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(139, 92, 246, 0.05));
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
    }
    .metric-card .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        background: var(--gradient-1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .metric-card .metric-label {
        font-size: 0.85rem;
        opacity: 0.7;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* ---------- ALERT CARD ---------- */
    .alert-card {
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
        display: flex;
        align-items: flex-start;
        gap: 12px;
    }
    .alert-warning {
        background: var(--warning-bg);
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .alert-critical {
        background: var(--danger-bg);
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .alert-card .alert-icon {
        font-size: 1.5rem;
        flex-shrink: 0;
    }
    .alert-card .alert-text {
        font-size: 0.9rem;
        line-height: 1.5;
    }

    /* ---------- SOURCE EVIDENCE ---------- */
    .source-chunk {
        background: rgba(99, 102, 241, 0.05);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-left: 3px solid var(--primary);
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 0.85rem;
        line-height: 1.6;
        font-family: 'Inter', sans-serif;
    }
    .source-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--primary-light);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }

    /* ---------- SECTION DIVIDER ---------- */
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 24px 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid var(--border);
    }
    .section-header h2 {
        font-size: 1.3rem;
        font-weight: 600;
        margin: 0;
    }

    /* ---------- DISCLAIMER BAR ---------- */
    .disclaimer-bar {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(239, 68, 68, 0.05));
        border: 1px solid rgba(245, 158, 11, 0.2);
        border-radius: 12px;
        padding: 12px 20px;
        font-size: 0.85rem;
        line-height: 1.5;
        margin-bottom: 20px;
    }

    /* ---------- ANIMATIONS ---------- */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .animate-in {
        animation: fadeInUp 0.4s ease-out;
    }

    /* ---------- CHAT STYLING ---------- */
    .stChatMessage {
        border-radius: 12px !important;
    }
</style>
"""
