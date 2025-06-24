import fitz  # PyMuPDF
import json
import os

PDF_PATH = "../docs/sample.pdf"
OUTPUT_JSON = "../data/pdf_knowledge.json"

def extract_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    knowledge = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        knowledge.append({
            "page": page_num + 1,
            "text": text,
            "images": []
        })
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(knowledge, f, indent=2)

if __name__ == "__main__":
    extract_pdf(PDF_PATH)
