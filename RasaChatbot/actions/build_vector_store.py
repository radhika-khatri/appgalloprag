import json
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv

load_dotenv()

def build_vector_store():
    with open("../data/pdf_knowledge.json", "r", encoding="utf-8") as f:
        knowledge = json.load(f)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    index_name = "pdf-knowledge-index"

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )

    index = pc.Index(index_name)

    for entry in knowledge:
        text = entry["text"]
        embedding = model.encode([text])[0].tolist()
        index.upsert([(f"page-{entry['page']}", embedding, entry)])

if __name__ == "__main__":
    build_vector_store()
