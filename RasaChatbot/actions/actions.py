from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load the embedding model once (recommended for efficiency)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Pinecone client and connect to index
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index("pdf-knowledge-index")

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

        # Generate embedding
        embedding = model.encode(query).tolist()

        # Query the vector index
        result = index.query(vector=embedding, top_k=1, include_metadata=True)

        # Handle response
        if result and result.get("matches"):
            best_match = result["matches"][0]
            metadata = best_match.get("metadata", {})

            text = metadata.get("text", "")
            images = metadata.get("images", [])

            # Send text response
            if text:
                dispatcher.utter_message(text=text)
            else:
                dispatcher.utter_message(text="I found a match, but it has no text content.")

            # Send image(s) if available
            if images:
                for img_url in images:
                    dispatcher.utter_message(image=img_url)
        else:
            dispatcher.utter_message(text="Sorry, I couldn’t find any information related to that.")

        return []
