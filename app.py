from groq import Groq
from dotenv import load_dotenv
import os
import pdfplumber
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

load_dotenv()
client = Groq()

pdf_path = "data/sample_report.pdf"

# Step 1: Read PDF text
with pdfplumber.open(pdf_path) as pdf:
    full_text = ""
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

print("PDF loaded successfully.\n")

# Step 2: Split text into chunks
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

print(f"Total chunks created: {len(chunks)}\n")

# Step 3: Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 4: Convert chunks to embeddings
embeddings = model.encode(chunks)

print("Embeddings created successfully.\n")

# Step 5: Create ChromaDB database
chroma_client = chromadb.Client(Settings())

collection = chroma_client.create_collection(name="medical_report")

# Step 6: Store embeddings in DB
for i, chunk in enumerate(chunks):
    collection.add(
        documents=[chunk],
        embeddings=[embeddings[i].tolist()],
        ids=[str(i)]
    )

print("Chunks stored in ChromaDB successfully.")
# Step 7: Ask a query
query = "What is HbA1c? Explain the lipid profiles is my hemoglobin normal ?"

# Convert query to embedding
query_embedding = model.encode([query])[0]

# Search in vector database
results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=2
)
retrieved_text = "\n".join(results["documents"][0])
prompt = f"""
You are a helpful medical report assistant.

Using the report context below, answer the user's question in simple language.

Report Context:
{retrieved_text}

Question:
{query}

Answer clearly and simply.
"""
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "user", "content": prompt}
    ]
)
print("\nAI Explanation:\n")
print(response.choices[0].message.content)
print("\nQuery:", query)
print("\nMost relevant chunks:\n")

for doc in results["documents"][0]:
    print(doc)
    print("------")