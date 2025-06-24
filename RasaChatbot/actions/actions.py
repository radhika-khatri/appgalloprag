from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer, util
import os
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load the embedding model once
model = SentenceTransformer('all-MiniLM-L6-v2')

# Pinecone client (unchanged)
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index("pdf-knowledge-index")

# Base URL to fix relative image paths
BASE_URL = "https://xtributor.com/US"

# Similarity threshold
SIMILARITY_THRESHOLD = 0.4


def extract_all_steps_if_any_match(text, query, model, threshold=SIMILARITY_THRESHOLD):
    """If any 'Step X' matches, return all contiguous steps from that chunk."""
    lines = [s.strip() for s in text.strip().split('\n') if s.strip()]
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Identify lines with 'Step X:'
    step_indexes = []
    step_lines = []
    for i, line in enumerate(lines):
        if re.match(r"(?i)^step\s*\d+\s*[:.]", line):
            step_indexes.append(i)
            step_lines.append(line)

    if not step_indexes:
        return []

    for i, idx in enumerate(step_indexes):
        step_embedding = model.encode(step_lines[i], convert_to_tensor=True)
        score = util.pytorch_cos_sim(query_embedding, step_embedding).item()
        if score > threshold:
            start = step_indexes[0]
            end = step_indexes[-1] + 1
            return lines[start:end]

    return []


def extract_relevant_lines(text, query, model, threshold=SIMILARITY_THRESHOLD):
    """Fallback: extract lines with general semantic similarity."""
    lines = [s.strip() for s in text.strip().split('\n') if s.strip()]
    query_embedding = model.encode(query, convert_to_tensor=True)
    relevant_lines = []

    for line in lines:
        line_embedding = model.encode(line, convert_to_tensor=True)
        score = util.pytorch_cos_sim(query_embedding, line_embedding).item()
        if score > threshold:
            relevant_lines.append(line)

    return relevant_lines


class ActionRespondPdfContent(Action):
    def name(self):
        return "action_respond_pdf_content"

    async def run(self, dispatcher: CollectingDispatcher,
                  tracker: Tracker,
                  domain: dict):

        query = tracker.latest_message.get("text")

        if not query:
            dispatcher.utter_message(text="I didn't understand your question.")
            return []

        # Encode the query
        embedding = model.encode(query).tolist()

        # Query Pinecone
        result = index.query(vector=embedding, top_k=3, include_metadata=True)

        if result and result.get("matches"):
            seen_texts = set()
            seen_images = set()
            combined_sentences = []
            images = []

            for match in result["matches"]:
                metadata = match.get("metadata", {})
                text = metadata.get("text", "").strip()

                if text and text not in seen_texts:
                    # Step-based matching
                    relevant_sentences = extract_all_steps_if_any_match(text, query, model)

                    # Fallback to general sentence match
                    if not relevant_sentences:
                        relevant_sentences = extract_relevant_lines(text, query, model)

                    if relevant_sentences:
                        combined_sentences.extend(relevant_sentences)
                        seen_texts.add(text)

                # Handle images
                for img_url in metadata.get("images", []):
                    full_url = img_url if not img_url.startswith("/") else BASE_URL + img_url
                    if full_url not in seen_images:
                        images.append(full_url)
                        seen_images.add(full_url)

            # Format and send response
            if combined_sentences:
                formatted = ""
                for line in combined_sentences:
                    if re.match(r"(?i)^step\s*\d+\s*[:.]", line):
                        formatted += f"✅ {line}\n\n"
                    elif line.startswith("•") or line.startswith("1.") or line.lower().startswith("note"):
                        formatted += f"🔸 {line}\n\n"
                    else:
                        formatted += f"{line}\n\n"

                dispatcher.utter_message(text=formatted.strip())
            else:
                dispatcher.utter_message(text="I found a match, but couldn’t extract clear steps.")

            # Send image(s)
            for url in images:
                dispatcher.utter_message(image=url)

        else:
            dispatcher.utter_message(text="Sorry, I couldn’t find any information related to that.")

        return []
