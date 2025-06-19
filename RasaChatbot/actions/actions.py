from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
import pytesseract
from PIL import Image
import base64
import io
import fitz  # PyMuPDF

class ActionProcessImage(Action):
    def name(self):
        return "action_process_image"

    def run(self, dispatcher, tracker, domain):
        image_data = tracker.get_slot("image_data")
        if not image_data:
            dispatcher.utter_message(text="No image provided.")
            return []

        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image)

        dispatcher.utter_message(text=f"Text from image: {text.strip()}")
        return []

class ActionAnswerFromPDF(Action):
    def name(self):
        return "action_answer_from_pdf"

    def run(self, dispatcher, tracker, domain):
        question = tracker.latest_message.get("text")
        with fitz.open("pdfs/sample.pdf") as doc:
            full_text = "".join(page.get_text() for page in doc)

        if question.lower() in full_text.lower():
            dispatcher.utter_message(text="Yes, that's mentioned in the PDF.")
        else:
            dispatcher.utter_message(text="Sorry, I couldn't find that in the PDF.")
        return []