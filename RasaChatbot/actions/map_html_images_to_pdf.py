import json
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util

PDF_PATH = "../docs/sample.pdf"
HTML_PATH = "../docs/fullcode.html"
OUTPUT_JSON = "../data/pdf_knowledge.json"

model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 1: Extract PDF chunks
def extract_pdf_chunks(pdf_path, chunk_size=3):
    print("[DEBUG] Extracting chunks from PDF...")
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        text = doc[page_num].get_text()
        pages.append({"page": page_num + 1, "text": text, "images": []})

    print(f"[DEBUG] Total chunks extracted: {len(pages)}")
    return pages


# Step 2: Extract image-text pairs using full <li> context
def extract_image_text_pairs_from_html(html_path):
    print("[DEBUG] Extracting image-context pairs from HTML...")
    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    img_pairs = []

    for img in soup.find_all("img"):
        img_url = img.get("src")
        context = ""

        # Step 1: Try finding text from any parent tag (regardless of tag name)
        parent = img.find_parent()
        while parent:
            context = parent.get_text(separator=" ", strip=True)
            if context:
                break
            parent = parent.find_parent()

        # Step 2: If no context from parent, try previous or next string
        if not context:
            prev_text = img.find_previous(string=True)
            next_text = img.find_next(string=True)
            context = (prev_text or next_text or "").strip()

        # Step 3: Fallback to alt attribute
        if not context and img.get("alt"):
            context = img.get("alt").strip()

        # Add to list if both URL and context exist
        if context and img_url:
            img_pairs.append({"url": img_url, "context": context})
            print(f"[DEBUG] Found image: {img_url}")
            print(f"[DEBUG] → Context: {context}")

    print(f"[DEBUG] Total images found: {len(img_pairs)}")
    return img_pairs


# Step 3: Map images to best PDF chunk regardless of similarity
def map_images_one_by_one(pdf_chunks, img_pairs):
    print("[DEBUG] Mapping images to PDF chunks (no threshold)...")
    chunk_embeddings = [model.encode(chunk["text"], convert_to_tensor=True) for chunk in pdf_chunks]

    for i, pair in enumerate(img_pairs):
        print(f"\n[DEBUG] Mapping image {i+1}/{len(img_pairs)}: {pair['url']}")
        print(f"[DEBUG] Context: {pair['context']}")

        img_embed = model.encode(pair["context"], convert_to_tensor=True)
        scores = [util.pytorch_cos_sim(img_embed, chunk_emb).item() for chunk_emb in chunk_embeddings]
        best_idx = scores.index(max(scores))

        pdf_chunks[best_idx]["images"].append(pair["url"])
        print(f"[DEBUG] ✔ Image mapped to chunk index {best_idx} (Page {pdf_chunks[best_idx]['page']})")

    return pdf_chunks


# Main
if __name__ == "__main__":
    pdf_chunks = extract_pdf_chunks(PDF_PATH)
    html_img_text = extract_image_text_pairs_from_html(HTML_PATH)
    enriched_chunks = map_images_one_by_one(pdf_chunks, html_img_text)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(enriched_chunks, f, indent=2)
        print(f"\n[DEBUG] ✅ Output written to: {OUTPUT_JSON}")

    total_mapped = sum(len(chunk['images']) for chunk in enriched_chunks)
    print(f"[DEBUG] ✅ Total images mapped: {total_mapped}/{len(html_img_text)}")
