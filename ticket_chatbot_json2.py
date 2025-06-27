import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import ollama
import uvicorn
import io
import json

app = FastAPI()

# Globals
df = None
index = None
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# ✅ Validate that the uploaded JSONL contains dicts with keys
def validate_json_structure(data):
    if not isinstance(data, list):
        raise ValueError("JSON must be a list of objects.")
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"Row {i+1} is not a JSON object.")
        if len(row) == 0 or all(str(k).strip() == "" for k in row.keys()):
            raise ValueError(f"Row {i+1} has no headers.")

# ✅ Initialize chatbot with JSONL input
def initialize_chatbot(file_bytes):
    global df, index

    try:
        # Parse JSONL file: one JSON object per line
        lines = file_bytes.decode('utf-8').splitlines()
        data = [json.loads(line) for line in lines if line.strip()]
        validate_json_structure(data)
        df_local = pd.DataFrame(data)
    except Exception as e:
        raise ValueError(f"Invalid JSONL file: {e}")

    df_local.fillna("", inplace=True)

    # Combine all fields into a text string per row
    def combine_text(row):
        return " | ".join(f"{k}: {v}" for k, v in row.items())

    df_local['combined'] = df_local.apply(combine_text, axis=1)

    # Create embeddings and build FAISS index
    print("[INFO] Generating embeddings...")
    embeddings = embedding_model.encode(df_local['combined'].tolist(), show_progress_bar=True)
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(np.array(embeddings))

    df = df_local
    index = faiss_index
    print("[INFO] Chatbot initialized.")

# ✅ Pydantic model for user query
class QueryRequest(BaseModel):
    question: str

@app.post("/load_jsonl")
async def load_jsonl(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        initialize_chatbot(file_bytes)
        return {"status": "✅ JSONL loaded and chatbot initialized."}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

@app.post("/ask")
async def ask_question(request: QueryRequest):
    print(request.question)
    if df is None or index is None:
        raise HTTPException(status_code=400, detail="❌ No data loaded. Please POST a JSONL to /load_jsonl first.")

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
    print(prompt)
    response = ollama.chat(
        model='mistral',
        messages=[{"role": "user", "content": prompt}]
    )

    return {"answer": response['message']['content']}

# ✅ Run the app: uvicorn main:app --reload
