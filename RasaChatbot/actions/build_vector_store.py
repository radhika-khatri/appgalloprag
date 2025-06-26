import json
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv

load_dotenv()

def build_vector_store():
    print("[DEBUG] Loading PDF knowledge from JSON...")
    with open("../data/structured_output.json", "r", encoding="utf-8") as f:
        knowledge = json.load(f)
    print(f"[DEBUG] Loaded {len(knowledge)} PDF chunks.")

    print("[DEBUG] Loading SentenceTransformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("[DEBUG] Initializing Pinecone client...")
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    index_name = "pdf-knowledge-index"
    print(f"[DEBUG] Checking if index '{index_name}' exists...")

    if index_name not in pc.list_indexes().names():
        print(f"[DEBUG] Index '{index_name}' not found. Creating new index...")
        pc.create_index(
            name=index_name,
            dimension=384,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        print(f"[DEBUG] Index '{index_name}' created.")
    else:
        print(f"[DEBUG] Index '{index_name}' already exists.")

    index = pc.Index(index_name)

    print("[DEBUG] Generating and upserting embeddings...")
    for i, entry in enumerate(knowledge):
        text = entry["text"]
        page = entry["page"]
        embedding = model.encode([text])[0].tolist()
        vector_id = f"page-{page}-{i}"  # Ensure unique IDs even if pages repeat
        index.upsert([(vector_id, embedding, entry)])
        print(f"[DEBUG] Upserted chunk {i+1}/{len(knowledge)} → ID: {vector_id}")

    print("[DEBUG] Vector store build complete.")

if __name__ == "__main__":
    build_vector_store()
