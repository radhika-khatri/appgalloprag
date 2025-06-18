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
from flask import Flask, request, jsonify
from urllib.parse import urlparse

app = Flask(__name__)

# Initialize global variables
df = None
kb_embeddings = None

# Download NLTK data
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Compile URL regex pattern once
URL_REGEX = re.compile(r'''
    (?:https?://|www\.)                # Protocol or www
    (?:[a-zA-Z0-9-]+\.)+               # Domain parts
    [a-zA-Z]{2,}                       # TLD
    (?::\d+)?                          # Optional port
    (?:/[^\s\]\[{}()'"<>]*)?           # Path
    (?:\?[^\s\]\[{}()'"<>]*)?          # Query string
    (?:\#[^\s\]\[{}()'"<>]*)?          # Fragment
''', re.VERBOSE)

# Initialize models and components
try:
    embedder = SentenceTransformer("all-mpnet-base-v2")
except Exception as e:
    print(f"Error loading SentenceTransformer: {e}")
    embedder = None

# Initialize SymSpell
sym_spell = SymSpell(max_dictionary_edit_distance=2)
try:
    sym_spell.load_dictionary("frequency_dictionary_en_82_765.txt", 0, 1)
except Exception as e:
    print(f"Error loading SymSpell dictionary: {e}")

# Initialize spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    print(f"Error loading spaCy model: {e}")
    nlp = None

# Initialize sentiment analyzer
try:
    sentiment_analyzer = pipeline("sentiment-analysis")
except Exception as e:
    print(f"Error loading sentiment analyzer: {e}")
    sentiment_analyzer = None

def is_valid_url(url):
    """Validate URL structure"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def extract_urls(text):
    """Accurately extract URLs and identify image URLs from text."""
    if not text or not isinstance(text, str):
        return [], []

    # Comprehensive URL pattern
    url_pattern = re.compile(
        r'(https?://(?:www\.)?[^\s\]\[{}()<>,"\']+|www\.[^\s\]\[{}()<>,"\']+)', 
        re.IGNORECASE
    )

    # Find all matches
    potential_urls = url_pattern.findall(text)

    cleaned_urls = []
    seen_urls = set()
    for url in potential_urls:
        # Strip unwanted trailing/surrounding punctuation
        clean_url = url.strip("()[]{}<>'\".,; \n\r\t")
        if clean_url.startswith("www."):
            clean_url = "http://" + clean_url
        if clean_url not in seen_urls:
            seen_urls.add(clean_url)
            cleaned_urls.append(clean_url)

    # Separate image URLs
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp')
    image_urls = [u for u in cleaned_urls if u.lower().endswith(image_extensions) or 
                  any(tag in u.lower() for tag in ['/images/', '/img/', 'image=true'])]
    return cleaned_urls, image_urls


def correct_spelling(query):
    if not query:
        return query
    try:
        suggestions = sym_spell.lookup_compound(query, max_edit_distance=2)
        return suggestions[0].term if suggestions else query
    except Exception as e:
        print(f"Error in spell correction: {e}")
        return query

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = text.split()
    filtered = [word for word in tokens if word not in stop_words]
    return " ".join(filtered)

def extract_entities(text):
    if not nlp or not text:
        return []
    try:
        doc = nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]
    except Exception as e:
        print(f"Error in entity extraction: {e}")
        return []

def detect_sentiment(text):
    if not sentiment_analyzer or not text:
        return "NEUTRAL", 0.0
    try:
        result = sentiment_analyzer(text)[0]
        return result['label'], result['score']
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return "NEUTRAL", 0.0

def fuzzy_fallback(query, df):
    if not query or df.empty:
        return None
    best_score = 0
    best_row = None
    for _, row in df.iterrows():
        combined = f"{row.get('title', '')} {row.get('article_text', '')}"
        score = fuzz.partial_ratio(query.lower(), combined.lower())
        if score > best_score:
            best_score = score
            best_row = row
    return best_row if best_score > 80 else None

def load_knowledge_base(file_path):
    print(f"[DEBUG] Loading knowledge base from: {file_path}")
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, encoding='ISO-8859-1')
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or XLSX.")
        df.fillna("", inplace=True)
        print(f"[DEBUG] Loaded {len(df)} rows.")
        return df
    except Exception as e:
        print(f"Error loading knowledge base: {e}")
        return pd.DataFrame()

def generate_kb_embeddings(df):
    if df.empty or not embedder:
        return np.array([])
    print("[DEBUG] Generating embeddings for knowledge base...")
    embeddings = []
    for _, row in df.iterrows():
        combined = f"{row.get('category', '')} {row.get('title', '')} {row.get('article_text', '')}"
        cleaned = clean_text(combined)
        try:
            embedding = embedder.encode(cleaned)
            embeddings.append(embedding)
        except Exception as e:
            print(f"Error generating embedding: {e}")
            embeddings.append(np.zeros(embedder.get_sentence_embedding_dimension()))
    print(f"[DEBUG] Generated {len(embeddings)} embeddings.")
    return np.array(embeddings)

def find_best_match(query, kb_embeddings, df):
    if not query or kb_embeddings.size == 0 or df.empty:
        return None
    cleaned_query = clean_text(query)
    try:
        query_embedding = embedder.encode(cleaned_query).reshape(1, -1)
        similarities = cosine_similarity(query_embedding, kb_embeddings)[0]
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        print(f"[DEBUG] Best match index: {best_idx}, score: {best_score:.4f}")
        if best_score < 0.4:
            return None
        return df.iloc[best_idx]
    except Exception as e:
        print(f"Error in finding best match: {e}")
        return None

def generate_response_with_mistral(context, query):
    if not context or not query:
        return ""
    prompt = f"""
Use the following knowledge base entry to answer the user's query.

Knowledge:
{context}

Question:
{query}

Answer:"""
    print("[DEBUG] Generating response from Mistral...")
    try:
        response = ollama.generate(model="mistral", prompt=prompt)
        return response["response"]
    except Exception as e:
        print(f"Error generating Mistral response: {e}")
        return "I couldn't generate a response for your query."

def chatbot_response(user_query, df, kb_embeddings):
    if not user_query or df.empty or kb_embeddings.size == 0:
        return {
            "response": "Sorry, I'm not properly initialized to handle your query.",
            "links": []
        }
    print(f"[DEBUG] Received query: {user_query}")
    user_query = correct_spelling(user_query)
    print(f"[DEBUG] Corrected query: {user_query}")

    sentiment, confidence = detect_sentiment(user_query)
    entities = extract_entities(user_query)
    print(f"[DEBUG] Sentiment: {sentiment} (confidence: {confidence:.2f})")
    print(f"[DEBUG] Named Entities: {entities}")

    best_row = find_best_match(user_query, kb_embeddings, df)
    if best_row is None:
        print("[DEBUG] No strong semantic match found, using fuzzy fallback...")
        best_row = fuzzy_fallback(user_query, df)

    if best_row is not None:
        article_text = best_row.get('article_text', '')
        steps_text = best_row.get('steps', '')

        context = f"{best_row.get('title', '')}\n\n{article_text}\n\nSteps: {steps_text}"
        bot_response = generate_response_with_mistral(context, user_query)

        step_lines = steps_text.strip().split('\n') if steps_text else []
        step_map = {}
        all_step_images = {}

        # Process each step line
        for line in step_lines:
            line = line.strip()
            if not line:
                continue
                
            # Handle different step number formats (1., 1), 1:, etc.)
            step_match = re.match(r'^(\d+)[\.:)]?\s*(.*)', line)
            if step_match:
                step_num, content = step_match.groups()
                step_map[step_num] = content
                print(f"[DEBUG] Processing step {step_num}: {content}")
                
                # Extract URLs with more aggressive pattern
                _, images = extract_urls(content)
                print(f"[DEBUG] Found {len(images)} image URLs in step {step_num}")
                
                if images:
                    all_step_images[step_num] = images
                    print(f"[DEBUG] Added images for step {step_num}: {images}")

        step_embeddings = {k: embedder.encode(clean_text(v)) for k, v in step_map.items()} if embedder else {}

        bot_lines = bot_response.strip().split('\n') if bot_response else []
        final_lines = []
        posted_images = set()

        for line in bot_lines:
            final_lines.append(line)
            cleaned_line = clean_text(line)
            if embedder:
                try:
                    line_embedding = embedder.encode(cleaned_line)
                    best_step = None
                    best_score = -1
                    for step_num, step_emb in step_embeddings.items():
                        sim = cosine_similarity([line_embedding], [step_emb])[0][0]
                        if sim > best_score:
                            best_score = sim
                            best_step = step_num

                    print(f"[DEBUG] Matched line: '{line.strip()}' with step {best_step} (similarity: {best_score:.4f})")

                    if best_step and best_step in all_step_images:
                        for img_url in all_step_images[best_step]:
                            if img_url not in posted_images:
                                final_lines.append(f"[Image] {img_url}")
                                posted_images.add(img_url)
                except Exception as e:
                    print(f"Error in embedding matching: {e}")

        # Add any remaining images not matched to specific lines
        for step_num, images in all_step_images.items():
            for img_url in images:
                if img_url not in posted_images:
                    final_lines.append(f"[Image] {img_url}")
                    posted_images.add(img_url)

        return {
            "response": "\n".join(final_lines),
            "links": list(posted_images)
        }

    print("[DEBUG] No match found even after fuzzy search.")
    return {
        "response": "Sorry, I couldn't find a solution for your query.",
        "links": []
    }

@app.route("/chat", methods=["POST"])
def chat():
    global df, kb_embeddings
    if df is None or kb_embeddings is None:
        return jsonify({"response": "Knowledge base not loaded", "links": []}), 500
    
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"response": "Invalid request", "links": []}), 400
        
    user_query = data.get("query", "")
    result = chatbot_response(user_query, df, kb_embeddings)
    return jsonify(result)

def initialize_system():
    global df, kb_embeddings
    file_path = "KnowledgeBase.csv"
    df = load_knowledge_base(file_path)
    if not df.empty:
        kb_embeddings = generate_kb_embeddings(df)
    else:
        print("Failed to load knowledge base")

def run_terminal_chat():
    global df, kb_embeddings
    if df is None or kb_embeddings is None:
        print("System not initialized properly. Exiting terminal chat.")
        return
        
    print("Terminal Chatbot is ready! (Type 'exit' to quit)\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("[DEBUG] User exited the terminal chat.")
            break
        result = chatbot_response(user_input, df, kb_embeddings)
        print("Bot:", result["response"])
        for link in result["links"]:
            print(f"[Image] {link}")

if __name__ == "__main__":
    initialize_system()
    
    if not df.empty:
        import threading
        threading.Thread(target=run_terminal_chat, daemon=True).start()
        app.run(debug=False)
    else:
        print("Failed to initialize system. Exiting.")