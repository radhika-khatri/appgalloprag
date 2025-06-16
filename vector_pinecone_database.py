import os
import uuid
import fitz  # PyMuPDF
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import ollama

# STEP 1: Initialize Pinecone
print("[DEBUG] Initializing Pinecone...")
api_key = "pcsk_3idWwW_4yuYEjXjmfSR3GGuLhmw9M6jzntNwQ3Bf8Cxf8zD9zgkMYADBM5kWqgPsdDNfoU"  # REPLACE WITH YOUR API KEY
pc = Pinecone(api_key=api_key)
index_name = "pdf-rag-index"

# Check if index exists and create if not
if index_name not in pc.list_indexes().names():
    print(f"[DEBUG] Creating Pinecone index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
else:
    print(f"[DEBUG] Index '{index_name}' already exists.")

index = pc.Index(index_name)
print(f"[DEBUG] Connected to Pinecone index: {index_name}")

# STEP 2: Load Embedding Model
print("[DEBUG] Loading sentence-transformer model 'all-MiniLM-L6-v2'...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("[DEBUG] Model loaded successfully.")

# STEP 3: Read and Chunk PDFs
def extract_chunks_from_pdf(pdf_path, chunk_size=200, overlap=50):
    print(f"[DEBUG] Reading and chunking PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    text = ""
    for page_num, page in enumerate(doc):
        page_text = page.get_text()
        print(f"[DEBUG] Extracted {len(page_text)} characters from page {page_num}")
        text += page_text

    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    print(f"[DEBUG] Total chunks created from '{os.path.basename(pdf_path)}': {len(chunks)}")
    return chunks

# STEP 4: Process and Store PDFs into Pinecone
pdf_folder = "pdfs"
if not os.path.exists(pdf_folder):
    os.mkdir(pdf_folder)
    print(f"[DEBUG] Created directory '{pdf_folder}'")

pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")][:2]
print(f"[DEBUG] Scanning folder for PDFs")

for filename in tqdm(pdf_files, desc="Processing first 2 PDFs only"):
    file_path = os.path.join(pdf_folder, filename)
    chunks = extract_chunks_from_pdf(file_path)
    embeddings = model.encode(chunks).tolist()
    print(f"[DEBUG] Generated embeddings for {len(chunks)} chunks from '{filename}'")

    vectors = []
    for i, embedding in enumerate(embeddings):
        vector_id = str(uuid.uuid4())
        metadata = {
            "source": filename,
            "chunk": chunks[i]
        }
        vectors.append((vector_id, embedding, metadata))
    
    print(f"[DEBUG] Upserting {len(vectors)} vectors into Pinecone...")
    for i in range(0, len(vectors), 100):
        batch = vectors[i:i+100]
        index.upsert(vectors=batch)
        print(f"[DEBUG] Upserted batch {i+1}–{i+len(batch)}")

print("[DEBUG] All PDFs processed and indexed.\n")

# STEP 5: Query Handler with LLaMA and Basic NLP
def is_greeting(query):
    greetings = ["hi", "hello", "hey", "good morning", "good evening"]
    result = query.lower().strip() in greetings
    print(f"[DEBUG] is_greeting('{query}') = {result}")
    return result

def search_pinecone(query, top_k=3):
    print(f"[DEBUG] Encoding query: '{query}'")
    query_embedding = model.encode(query).tolist()
    print(f"[DEBUG] Querying Pinecone with top_k={top_k}...")
    result = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    print(f"[DEBUG] Pinecone returned {len(result['matches'])} matches.")
    return [match['metadata']['chunk'] for match in result['matches']]

def ask_llama(prompt):
    print(f"[DEBUG] Sending prompt to LLaMA (length={len(prompt)} chars)...")
    try:
        response = ollama.generate(model="llama3", prompt=prompt)
        print(f"[DEBUG] LLaMA responded with {len(response['response'])} characters.")
        return response["response"].strip()
    except Exception as e:
        print(f"[ERROR] LLaMA generation failed: {e}")
        return "Sorry, I couldn't generate a response at this time."

def answer_query(query):
    print(f"\n[DEBUG] User query received: '{query}'")

    if is_greeting(query):
        return "Hello! How can I assist you today with your documents?"
    
    retrieved_chunks = search_pinecone(query)
    if not retrieved_chunks:
        print("[DEBUG] No relevant chunks found for query.")
        return "Sorry, I couldn't find relevant information in the documents."

    context = "\n\n".join(retrieved_chunks)
    prompt = f"""You are a helpful assistant answering questions based on PDF content.

Context:
{context}

Question: {query}

Answer:"""
    return ask_llama(prompt)

# STEP 6: Interactive Chat Loop
print("\n📄 RAG Chatbot with LLaMA + Pinecone (type 'exit' to quit)")
while True:
    user_query = input("You: ")
    if user_query.lower() == "exit":
        print("[DEBUG] Exiting chatbot. Goodbye!")
        break
    try:
        response = answer_query(user_query)
        print("Bot:", response)
    except Exception as e:
        print(f"[ERROR] Something went wrong: {e}")
