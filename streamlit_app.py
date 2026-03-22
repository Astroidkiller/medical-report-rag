import streamlit as st
import os
import hashlib
import pdfplumber
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

import time

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Medical Report Assistant",
    page_icon="🩺",
    layout="wide"
)

# ---------- API / CLIENT SETUP ----------
try:
    api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    api_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=api_key)


# ---------- CACHED HELPERS ----------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def get_chroma_client():
    return chromadb.Client(Settings())


@st.cache_data
def process_pdf(file_path):
    full_text = ""

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"

    lines = full_text.split("\n")
    chunks = []
    current_chunk = ""

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if len(current_chunk) + len(line) < 300:
            current_chunk += line + "\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = line + "\n"

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks, full_text


def get_file_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()


# ---------- UI ----------
st.title("🩺 Patient-Centric Medical Report Understanding Assistant")
st.markdown(
    """
    Upload a medical report, ask questions in simple language, and get an AI-generated explanation based on the report content.
    """
)
st.info("This tool is for educational use only and does not replace professional medical advice.")

st.subheader("📄 Upload Medical Report")
uploaded_file = st.file_uploader(
    "Upload a medical report (PDF format)",
    type=["pdf"],
    help="Upload a diagnostic report to analyze and ask questions about it."
)

if uploaded_file is not None:
    st.success("✅ File uploaded successfully.")

    os.makedirs("data", exist_ok=True)
    file_path = os.path.join("data", "uploaded_report.pdf")

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Process PDF
    chunks, full_report_text = process_pdf(file_path)

    st.subheader("📊 Report Overview")
    col1, col2 = st.columns(2)
    col1.metric("Chunks Created", len(chunks))
    col2.metric("Upload Status", "Ready")

    # Embeddings + DB
    model = load_embedding_model()
    embeddings = model.encode(chunks)

    chroma_client = get_chroma_client()
    collection = chroma_client.get_or_create_collection(name="medical_report")

    # Clear old data before adding new chunks
    existing_data = collection.get()
    if existing_data["ids"]:
        collection.delete(ids=existing_data["ids"])

    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            embeddings=[embeddings[i].tolist()],
            ids=[str(i)]
        )
    
    st.subheader("❓ Ask a Question")

    default_question = ""

    user_question = st.text_input(
            "Type your question about the report",
            value=default_question,
            placeholder="Example: What does hemoglobin mean?"
    )

    # Quick question suggestions
    st.markdown("**Report Quick Actions:**")
    suggestion_cols = st.columns(3)
    if suggestion_cols[0].button("📋 Summarize Report"):
        user_question = "Summarize this medical report in simple, patient-friendly language."
    if suggestion_cols[1].button("🔍 Important Findings"):
        user_question = "List the most important or unusual findings from this report."
    if suggestion_cols[2].button("⚠️ Abnormal Values"):
        user_question = "Identify any values that are outside the normal range and explain what they mean briefly."

    if user_question:
        # Bypassing retrieval to ensure ALL details are captured
        # We use the full_report_text collected during processing
        retrieved_text = full_report_text

        if not retrieved_text.strip():
            st.error("No relevant context was found in the report for this question.")
        else:
            # Simple Rate Limiting (Session Based)
            if "last_request_time" not in st.session_state:
                st.session_state.last_request_time = 0
            
            current_time = time.time()
            if current_time - st.session_state.last_request_time < 2:  # 2 second cooldown
                st.warning("⚠️ Slow down! Please wait a moment between requests.")
            else:
                st.session_state.last_request_time = current_time
                
                prompt = f"""
You are a Highly Expert Medical AI Analysis Systems. 
Your objective is to provide a comprehensive, clear, and professional analysis of the provided medical report.

### PRIVACY & SAFETY RULES:
1. **STRICT PRIVACY**: NEVER mention the patient's name, age, gender, or any personal IDs.
2. **STRICT CONTEXT**: Use ONLY the provided report data. 
3. **NO INDEPENDENT DIAGNOSIS**: Do not invent new diagnoses. Summarize the findings *already present* in the report.

### Context (Full Medical Report):
{retrieved_text}

### User's Question/Action:
{user_question}

### Response Structure:
1. **Executive Summary**: A high-level, professional summary of what this report indicates.
2. **Detailed Clinical Findings**: A strictly organized list of test results, focusing on anything abnormal or out of range. Explain each finding in 1 concise sentence.
3. **Next Steps Recommendation**: Suggest what kind of specialist the patient might need to see based on the report (e.g., "Consult a Cardiologist").
4. **MANDATORY DISCLAIMER**: You MUST end every response with this exact text in bold:
   "**DISCLAIMER: This AI analysis is for informational purposes only. It is NOT a medical diagnosis. You MUST consult a qualified doctor or healthcare professional to discuss these results and receive proper medical advice or treatment.**"

### Formatting:
- Use bold headings (`### Heading`).
- Use clear bullet points.
- Keep the language professional yet accessible.
"""

                with st.spinner("Analyzing report..."):
                    stream = groq_client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[{"role": "user", "content": prompt}],
                        stream=True
                    )
                    
                    st.subheader("🧠 AI Explanation")
                    response_placeholder = st.empty()
                    full_response = ""
                    
                    for chunk in stream:
                        if chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                            response_placeholder.markdown(full_response + "▌")
                    
                    response_placeholder.markdown(full_response)

            with st.expander("Show retrieved context"):
                st.write(retrieved_text)