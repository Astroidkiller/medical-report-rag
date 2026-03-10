import streamlit as st
import os
import pdfplumber
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
groq_client = Groq()

st.title("Patient-Centric Medical Report Understanding Assistant")
st.write("Upload a medical report and ask questions in simple language.")

uploaded_file = st.file_uploader("Upload your medical report (PDF)", type=["pdf"])

if uploaded_file is not None:
    st.success("File uploaded successfully.")
    os.makedirs("data", exist_ok=True)
    file_path = os.path.join("data", "uploaded_report.pdf")

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

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

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)

    chroma_client = chromadb.Client(Settings())
    collection = chroma_client.get_or_create_collection(name="medical_report")

    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
        
            embeddings=[embeddings[i].tolist()],
            ids=[str(i)]
        )

    user_question = st.text_input("Ask a question about the report")

    if user_question:
        st.write("Your question:", user_question)
        query_embedding = model.encode([user_question])[0]
        
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=2
        )

        retrieved_text = "\n".join(results["documents"][0])
        st.write("Retrieved Context:")
        st.write(retrieved_text)
        
        prompt = f"""
        You are a medical report assistant.
        Using the report context below, answer the question in simple language so a patient can understand.

        Report Context:
        {retrieved_text}

        Question:
        {user_question}
        """
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        ai_answer = response.choices[0].message.content
        st.write("AI Explanation:")
        st.write(ai_answer)