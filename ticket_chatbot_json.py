import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import ollama
import uvicorn
import io

app = FastAPI()

# Globals to store chatbot state
df = None
index = None
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Reusable function to prepare FAISS and DataFrame
def initialize_chatbot(file_bytes):
    global df, index

    # Load and clean CSV
    df = pd.read_csv(io.BytesIO(file_bytes))
    df.fillna("", inplace=True)

    def combine_text(row):
        return f"""Ticket ID: {row['ticket_id']} | Complaint Type: {row['complaint_type']} | Subtype: {row['complaint_subtype']} | 
Complaint Text: {row['complaint_text']} | Status: {row['status']} | 
Response: {row['response_text']} | Email: {row['email_id']}"""

    df['combined'] = df.apply(combine_text, axis=1)

    # Embed and index
    print("[INFO] Generating embeddings...")
    embeddings = embedding_model.encode(df['combined'].tolist(), show_progress_bar=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    print("[INFO] Chatbot initialized.")

# Request schema for questions
class QueryRequest(BaseModel):
    question: str

@app.post("/load_csv")
async def load_csv(file: UploadFile = File(...)):
    file_bytes = await file.read()
    initialize_chatbot(file_bytes)
    return {"status": "CSV loaded and chatbot initialized."}

@app.post("/ask")
async def ask_question(request: QueryRequest):
    if df is None or index is None:
        return {"error": "No data loaded. Please POST a CSV to /load_csv first."}

    query = request.question
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k=3)
    results = df.iloc[indices[0]]
    context = "\n---\n".join(results['combined'].tolist())

    prompt = f"""You are a helpful support assistant.
Based on the following support ticket entries:
{context}

Answer this user question: {query}
"""

    response = ollama.chat(
        model='mistral',
        messages=[{"role": "user", "content": prompt}]
    )

    return {"answer": response['message']['content']}
