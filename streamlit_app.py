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

    return chunks


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
    chunks = process_pdf(file_path)

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

    if st.button("📋 Generate Report Summary"):
            default_question = "Give a simple summary of this medical report"

    user_question = st.text_input(
            "Type your question about the report",
            value=default_question,
            placeholder="Example: What does hemoglobin mean?"
    )
    "Type your question about the report",
    placeholder="Example: What does hemoglobin mean?"


    # Quick question suggestions
    st.markdown("**Suggested questions:**")
    suggestion_cols = st.columns(3)
    if suggestion_cols[0].button("What does hemoglobin mean?"):
        user_question = "What does hemoglobin mean?"
    if suggestion_cols[1].button("Explain HbA1c"):
        user_question = "Explain HbA1c"
    if suggestion_cols[2].button("Summarize my report"):
        user_question = "Summarize my report"

    if user_question:
        query_embedding = model.encode([user_question])[0]

        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=2
        )

        retrieved_docs = results.get("documents", [[]])
        retrieved_text = "\n".join(retrieved_docs[0]) if retrieved_docs and retrieved_docs[0] else ""

        if not retrieved_text.strip():
            st.error("No relevant context was found in the report for this question.")
        else:
            prompt = f"""
You are a medical report assistant.

Your job is to explain the uploaded report in simple, patient-friendly language.

Rules:
- Use only the report context provided below.
- If the answer is not clearly present in the context, say that the report does not provide enough information.
- Do not give a diagnosis.
- Do not suggest medicines.
- Keep the explanation clear, short, and easy to understand.

Report Context:
{retrieved_text}

Question:
{user_question}
"""

            with st.spinner("Generating explanation..."):
                response = groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )

            ai_answer = response.choices[0].message.content

            st.subheader("🧠 AI Explanation")
            st.success(ai_answer)

            with st.expander("Show retrieved context"):
                st.write(retrieved_text)