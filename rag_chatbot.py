import ollama
import numpy as np
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF

# 1. Load your PDF and extract text
def extract_text_from_pdf(pdf_path):
    print(f"[DEBUG] Opening PDF: {pdf_path}")

    doc = fitz.open(pdf_path)
    text = ""
    for i, page in enumerate(doc):
        page_text = page.get_text()
        print(f"[DEBUG] Extracted text from page {i}: {len(page_text)} characters")
        text += page_text
    print(f"[DEBUG] Total extracted text length: {len(text)} characters")
    return text

pdf_path = "output.pdf"  # Replace with your PDF file
print(f"[DEBUG] Starting PDF extraction from: {pdf_path}")
pdf_text = extract_text_from_pdf(pdf_path)

# 2. Split text into chunks
def split_text(text, chunk_size=1000, overlap=200):
    print(f"[DEBUG] Splitting text into chunks (chunk_size={chunk_size}, overlap={overlap})")
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        print(f"[DEBUG] Created chunk from {start} to {end} ({len(chunk)} characters)")
        chunks.append(chunk)
        start = end - overlap
    print(f"[DEBUG] Total chunks created: {len(chunks)}")
    return chunks

chunks = split_text(pdf_text)

# 3. Generate embeddings for each chunk
model_name = "CompendiumLabs/bge-base-en-v1.5-gguf"  # Or use sentence-transformers locally
try:
    print(f"[DEBUG] Loading SentenceTransformer model: all-MiniLM-L6-v2")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight and local
    print(f"[DEBUG] Generating embeddings for {len(chunks)} chunks...")
    chunk_embeddings = np.array([embedder.encode(chunk) for chunk in chunks])
    print(f"[DEBUG] Embedding shape: {chunk_embeddings.shape}")
except Exception as e:
    print(f"[ERROR] Could not use ollama for embeddings. Using sentence-transformers as fallback. Exception: {e}")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    chunk_embeddings = np.array([embedder.encode(chunk) for chunk in chunks])

# 4. Simple in-memory vector database
class VectorDB:
    def __init__(self, chunks, embeddings):
        print(f"[DEBUG] Initializing VectorDB with {len(chunks)} chunks and embeddings of shape {embeddings.shape}")
        self.chunks = chunks
        self.embeddings = embeddings

    def search(self, query_embedding, top_k=3):
        print(f"[DEBUG] Searching for top {top_k} similar chunks...")
        scores = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        top_indices = np.argsort(-scores)[:top_k]
        print(f"[DEBUG] Top indices: {top_indices}")
        return [self.chunks[i] for i in top_indices]

vector_db = VectorDB(chunks, chunk_embeddings)

# 5. RAG pipeline
def answer_query(query):
    print(f"[DEBUG] Received query: {query}")
    # Embed the query
    query_embedding = embedder.encode(query)
    print(f"[DEBUG] Query embedding shape: {query_embedding.shape}")
    # Retrieve relevant chunks
    relevant_chunks = vector_db.search(query_embedding)
    print(f"[DEBUG] Retrieved {len(relevant_chunks)} relevant chunks.")
    context = "\n\n".join(relevant_chunks)
    # Generate answer using local LLM (Ollama)
    prompt = f"Answer the following question based on the provided context:\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    print(f"[DEBUG] Sending prompt to Ollama model 'llama3'. Prompt length: {len(prompt)} characters")
    response = ollama.generate(model="llama3", prompt=prompt)
    print(f"[DEBUG] Ollama response received.")
    return response["response"]

# 6. Chatbot interface
print("RAG Chatbot - Ask questions about your PDF (type 'exit' to quit)")
while True:
    query = input("Question: ")
    if query.lower() == "exit":
        print("[DEBUG] Exiting chatbot loop.")
        break
    print("Answer:", answer_query(query))
