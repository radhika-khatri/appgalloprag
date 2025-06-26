from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from sentence_transformers import SentenceTransformer, util
import os
import re
import json
from dotenv import load_dotenv
from urllib.parse import urljoin

# Load environment variables
load_dotenv()

# Constants
BASE_URL = "https://xtributor.com/US"
SIMILARITY_THRESHOLD = 0.4
STRUCTURED_JSON_PATH = "data/structured_output.json"  # Adjust if needed

# Load model and structured data once
model = SentenceTransformer('all-MiniLM-L6-v2')

with open(STRUCTURED_JSON_PATH, "r", encoding="utf-8") as f:
    structured_data = json.load(f)


def extract_all_steps_if_any_match(text, query, model, threshold=SIMILARITY_THRESHOLD):
    lines = [s.strip() for s in text.strip().split('\n') if s.strip()]
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Match: Step 1:, • Step 1:, • 1.
    step_pattern = re.compile(r"(?i)^(•\s*)?step\s*\d+[:.]|•\s*\d+[.:]")

    matching_steps = []
    for i, line in enumerate(lines):
        if step_pattern.match(line):
            line_embedding = model.encode(line, convert_to_tensor=True)
            score = util.pytorch_cos_sim(query_embedding, line_embedding).item()
            if score > threshold:
                matching_steps.append(i)

    if not matching_steps:
        return []

    # Return all lines that are steps
    return [line for line in lines if step_pattern.match(line)]


def extract_relevant_lines(text, query, model, threshold=SIMILARITY_THRESHOLD):
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

        query_embedding = model.encode(query, convert_to_tensor=True)

        seen_texts = set()
        seen_images = set()
        combined_sentences = []
        images = []

        for chunk in structured_data:
            text = chunk.get("text", "").strip()
            if not text or text in seen_texts:
                continue

            text_embedding = model.encode(text, convert_to_tensor=True)
            score = util.pytorch_cos_sim(query_embedding, text_embedding).item()

            if score > SIMILARITY_THRESHOLD:
                # Try to extract steps
                relevant_sentences = extract_all_steps_if_any_match(text, query, model)

                # Fallback to general matching
                if not relevant_sentences:
                    relevant_sentences = extract_relevant_lines(text, query, model)

                if relevant_sentences:
                    combined_sentences.extend(relevant_sentences)
                    seen_texts.add(text)

                # Collect image URLs
                for img_url in chunk.get("images", []):
                    full_url = urljoin(BASE_URL, img_url)
                    if full_url not in seen_images:
                        images.append(full_url)
                        seen_images.add(full_url)

        # Format and send response
        if combined_sentences:
            formatted = ""
            for line in combined_sentences:
                if re.match(r"(?i)^step\s*\d+[:.]", line):
                    formatted += f"✅ {line}\n\n"
                elif re.match(r"•\s*\d+[.:]", line) or re.match(r"(?i)^(•\s*)?step\s*\d+[:.]", line):
                    formatted += f"🔸 {line}\n\n"
                elif line.lower().startswith("note"):
                    formatted += f"📝 {line}\n\n"
                else:
                    formatted += f"{line}\n\n"

            dispatcher.utter_message(text=formatted.strip())
        else:
            dispatcher.utter_message(text="I couldn’t find anything relevant.")

        for url in images:
            dispatcher.utter_message(image=url)

        return []
