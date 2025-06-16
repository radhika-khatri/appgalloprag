import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import ollama
from symspellpy import SymSpell, Verbosity
import spacy
from transformers import pipeline
from fuzzywuzzy import fuzz

# === Setup ===
print("[DEBUG] Downloading NLTK stopwords...")
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
print("[DEBUG] Stopwords loaded.")

print("[DEBUG] Loading SentenceTransformer model...")
embedder = SentenceTransformer("all-mpnet-base-v2")  # Higher-quality embeddings
print("[DEBUG] Model loaded.")

print("[DEBUG] Initializing spell corrector...")
sym_spell = SymSpell(max_dictionary_edit_distance=2)
sym_spell.load_dictionary("frequency_dictionary_en_82_765.txt", 0, 1)

print("[DEBUG] Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")

print("[DEBUG] Loading sentiment analysis pipeline...")
sentiment_analyzer = pipeline("sentiment-analysis")

# === Utility Functions ===
def correct_spelling(query):
    suggestions = sym_spell.lookup_compound(query, max_edit_distance=2)
    return suggestions[0].term if suggestions else query

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = text.split()
    filtered = [word for word in tokens if word not in stop_words]
    return " ".join(filtered)

def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def detect_sentiment(text):
    result = sentiment_analyzer(text)[0]
    return result['label'], result['score']

def fuzzy_fallback(query, df):
    best_score = 0
    best_row = None
    for _, row in df.iterrows():
        combined = f"{row['title']} {row['article_text']}"
        score = fuzz.partial_ratio(query.lower(), combined.lower())
        if score > best_score:
            best_score = score
            best_row = row
    return best_row if best_score > 80 else None

# === Load Knowledge Base ===
def load_knowledge_base(file_path):
    print(f"[DEBUG] Loading knowledge base from: {file_path}")
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path, encoding='ISO-8859-1')
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Use CSV or XLSX.")
    df.fillna("", inplace=True)
    print(f"[DEBUG] Loaded {len(df)} rows.")
    return df

# === Generate Embeddings for Knowledge Base ===
def generate_kb_embeddings(df):
    print("[DEBUG] Generating embeddings for knowledge base...")
    embeddings = []
    for _, row in df.iterrows():
        combined = f"{row['category']} {row['title']} {row['article_text']}"
        cleaned = clean_text(combined)
        embedding = embedder.encode(cleaned)
        embeddings.append(embedding)
    print(f"[DEBUG] Generated {len(embeddings)} embeddings.")
    return np.array(embeddings)

# === Find Most Similar Row ===
def find_best_match(query, kb_embeddings, df):
    cleaned_query = clean_text(query)
    query_embedding = embedder.encode(cleaned_query).reshape(1, -1)
    similarities = cosine_similarity(query_embedding, kb_embeddings)[0]
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]
    print(f"[DEBUG] Best match index: {best_idx}, score: {best_score:.4f}")
    if best_score < 0.4:
        return None
    return df.iloc[best_idx]

# === Generate Response ===
def generate_response_with_mistral(context, query):
    prompt = f"""
Use the following knowledge base entry to answer the user's query.

Knowledge:
{context}

Question:
{query}

Answer:"""
    print("[DEBUG] Generating response from Mistral...")
    response = ollama.generate(model="mistral", prompt=prompt)
    return response["response"]

# === Chatbot Logic ===
def chatbot_response(user_query, df, kb_embeddings):
    print(f"[DEBUG] Received query: {user_query}")
    user_query = correct_spelling(user_query)

    sentiment, confidence = detect_sentiment(user_query)
    entities = extract_entities(user_query)
    print(f"[DEBUG] Sentiment: {sentiment}, Entities: {entities}")

    best_row = find_best_match(user_query, kb_embeddings, df)
    if best_row is None:
        best_row = fuzzy_fallback(user_query, df)

    if best_row is not None:
        context = f"{best_row['title']}\n\n{best_row['article_text']}\n\nSteps: {best_row['steps']}"
        return generate_response_with_mistral(context, user_query)
    else:
        return "Sorry, I couldn't find a solution for your query."

# === Main Usage ===
if __name__ == "__main__":
    file_path = "KnowledgeBase.xlsx"
    df = load_knowledge_base(file_path)
    kb_embeddings = generate_kb_embeddings(df)

    print("Chatbot is ready! (Type 'exit' to quit)\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("[DEBUG] User exited the chat.")
            break
        response = chatbot_response(user_input, df, kb_embeddings)
        print("Bot:", response)