"""
Centralized configuration for Community Health Intelligence Assistant.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ---------- LLM CONFIGURATION ----------
LLM_MODEL = "llama-3.1-8b-instant"
LLM_PROVIDER = "groq"  # Future: "vertex_ai" for Gemini migration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ---------- EMBEDDING CONFIGURATION ----------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# ---------- CHUNKING CONFIGURATION ----------
CHUNK_MAX_CHARS = 300
CHUNK_OVERLAP_CHARS = 50

# ---------- RETRIEVAL CONFIGURATION ----------
TOP_K_RESULTS = 5

# ---------- DATABASE PATHS ----------
CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
SQLITE_DB_PATH = os.path.join(os.path.dirname(__file__), "community_db", "community.db")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Ensure directories exist
os.makedirs(os.path.dirname(SQLITE_DB_PATH), exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ---------- REFERENCE RANGES ----------
# Common lab test reference ranges for anomaly detection.
# Format: test_name_lower -> (low, high, unit, critical_low, critical_high)
# Values are typical adult ranges — not a clinical reference.
REFERENCE_RANGES = {
    # Complete Blood Count (CBC)
    "hemoglobin": (12.0, 17.5, "g/dL", 7.0, 20.0),
    "hb": (12.0, 17.5, "g/dL", 7.0, 20.0),
    "haemoglobin": (12.0, 17.5, "g/dL", 7.0, 20.0),
    "rbc": (4.0, 6.0, "million/uL", 2.5, 8.0),
    "rbc count": (4.0, 6.0, "million/uL", 2.5, 8.0),
    "red blood cell count": (4.0, 6.0, "million/uL", 2.5, 8.0),
    "wbc": (4000, 11000, "/uL", 2000, 30000),
    "wbc count": (4000, 11000, "/uL", 2000, 30000),
    "white blood cell count": (4000, 11000, "/uL", 2000, 30000),
    "total wbc count": (4000, 11000, "/uL", 2000, 30000),
    "platelet count": (150000, 400000, "/uL", 50000, 800000),
    "platelets": (150000, 400000, "/uL", 50000, 800000),
    "hematocrit": (36.0, 54.0, "%", 20.0, 65.0),
    "hct": (36.0, 54.0, "%", 20.0, 65.0),
    "pcv": (36.0, 54.0, "%", 20.0, 65.0),
    "packed cell volume": (36.0, 54.0, "%", 20.0, 65.0),
    "mcv": (80.0, 100.0, "fL", 60.0, 120.0),
    "mch": (27.0, 33.0, "pg", 20.0, 40.0),
    "mchc": (32.0, 36.0, "g/dL", 28.0, 40.0),
    "rdw": (11.5, 14.5, "%", 10.0, 20.0),
    "esr": (0, 20, "mm/hr", 0, 100),

    # Diabetes Markers
    "hba1c": (4.0, 5.6, "%", 3.0, 14.0),
    "glycated hemoglobin": (4.0, 5.6, "%", 3.0, 14.0),
    "glycosylated hemoglobin": (4.0, 5.6, "%", 3.0, 14.0),
    "fasting blood sugar": (70, 100, "mg/dL", 40, 400),
    "fasting glucose": (70, 100, "mg/dL", 40, 400),
    "fbs": (70, 100, "mg/dL", 40, 400),
    "blood sugar fasting": (70, 100, "mg/dL", 40, 400),
    "postprandial blood sugar": (70, 140, "mg/dL", 40, 400),
    "ppbs": (70, 140, "mg/dL", 40, 400),
    "random blood sugar": (70, 140, "mg/dL", 40, 400),
    "rbs": (70, 140, "mg/dL", 40, 400),

    # Lipid Profile
    "total cholesterol": (0, 200, "mg/dL", 0, 400),
    "cholesterol": (0, 200, "mg/dL", 0, 400),
    "hdl cholesterol": (40, 60, "mg/dL", 20, 100),
    "hdl": (40, 60, "mg/dL", 20, 100),
    "ldl cholesterol": (0, 100, "mg/dL", 0, 300),
    "ldl": (0, 100, "mg/dL", 0, 300),
    "triglycerides": (0, 150, "mg/dL", 0, 500),
    "vldl": (0, 30, "mg/dL", 0, 80),

    # Liver Function Tests (LFT)
    "sgpt": (7, 56, "U/L", 0, 500),
    "alt": (7, 56, "U/L", 0, 500),
    "sgot": (10, 40, "U/L", 0, 500),
    "ast": (10, 40, "U/L", 0, 500),
    "alkaline phosphatase": (44, 147, "U/L", 0, 500),
    "alp": (44, 147, "U/L", 0, 500),
    "total bilirubin": (0.1, 1.2, "mg/dL", 0, 15.0),
    "bilirubin total": (0.1, 1.2, "mg/dL", 0, 15.0),
    "direct bilirubin": (0.0, 0.3, "mg/dL", 0, 10.0),
    "indirect bilirubin": (0.1, 0.9, "mg/dL", 0, 10.0),
    "total protein": (6.0, 8.3, "g/dL", 3.0, 12.0),
    "albumin": (3.5, 5.5, "g/dL", 1.5, 7.0),
    "globulin": (2.0, 3.5, "g/dL", 1.0, 6.0),
    "a/g ratio": (1.0, 2.5, "", 0.5, 4.0),
    "ggt": (9, 48, "U/L", 0, 300),
    "gamma gt": (9, 48, "U/L", 0, 300),

    # Kidney Function Tests (KFT / RFT)
    "creatinine": (0.6, 1.2, "mg/dL", 0.3, 10.0),
    "serum creatinine": (0.6, 1.2, "mg/dL", 0.3, 10.0),
    "blood urea": (15, 40, "mg/dL", 5, 150),
    "urea": (15, 40, "mg/dL", 5, 150),
    "bun": (7, 20, "mg/dL", 3, 100),
    "blood urea nitrogen": (7, 20, "mg/dL", 3, 100),
    "uric acid": (3.5, 7.2, "mg/dL", 1.0, 15.0),
    "serum uric acid": (3.5, 7.2, "mg/dL", 1.0, 15.0),
    "gfr": (90, 120, "mL/min", 15, 150),
    "egfr": (90, 120, "mL/min", 15, 150),

    # Thyroid
    "tsh": (0.4, 4.0, "mIU/L", 0.01, 50.0),
    "t3": (80, 200, "ng/dL", 40, 400),
    "t4": (5.0, 12.0, "ug/dL", 2.0, 25.0),
    "free t3": (2.3, 4.2, "pg/mL", 1.0, 8.0),
    "free t4": (0.8, 1.8, "ng/dL", 0.3, 5.0),
    "ft3": (2.3, 4.2, "pg/mL", 1.0, 8.0),
    "ft4": (0.8, 1.8, "ng/dL", 0.3, 5.0),

    # Electrolytes
    "sodium": (136, 145, "mEq/L", 120, 160),
    "potassium": (3.5, 5.0, "mEq/L", 2.5, 7.0),
    "chloride": (98, 106, "mEq/L", 80, 120),
    "calcium": (8.5, 10.5, "mg/dL", 6.0, 14.0),
    "magnesium": (1.7, 2.2, "mg/dL", 1.0, 4.0),
    "phosphorus": (2.5, 4.5, "mg/dL", 1.0, 8.0),

    # Iron Studies
    "serum iron": (60, 170, "ug/dL", 20, 300),
    "iron": (60, 170, "ug/dL", 20, 300),
    "tibc": (250, 400, "ug/dL", 100, 600),
    "ferritin": (12, 300, "ng/mL", 5, 1000),
    "transferrin saturation": (20, 50, "%", 5, 90),

    # Vitamins
    "vitamin d": (30, 100, "ng/mL", 10, 150),
    "vitamin d3": (30, 100, "ng/mL", 10, 150),
    "25-hydroxy vitamin d": (30, 100, "ng/mL", 10, 150),
    "vitamin b12": (200, 900, "pg/mL", 100, 2000),
    "folate": (2.7, 17.0, "ng/mL", 1.0, 30.0),
    "folic acid": (2.7, 17.0, "ng/mL", 1.0, 30.0),

    # Cardiac Markers
    "troponin": (0, 0.04, "ng/mL", 0, 2.0),
    "troponin i": (0, 0.04, "ng/mL", 0, 2.0),
    "ck-mb": (0, 25, "U/L", 0, 200),
    "bnp": (0, 100, "pg/mL", 0, 5000),
    "crp": (0, 3.0, "mg/L", 0, 100),
    "hs-crp": (0, 3.0, "mg/L", 0, 100),
}

# ---------- SIMULATED DEMOGRAPHICS ----------
# For demo/hackathon: simulated regions and age groups
DEMO_REGIONS = ["Urban-Central", "Urban-East", "Suburban-North", "Rural-West", "Rural-South"]
DEMO_AGE_GROUPS = ["0-18", "19-30", "31-45", "46-60", "60+"]
