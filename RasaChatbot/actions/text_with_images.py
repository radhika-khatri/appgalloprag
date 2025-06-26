import json
from bs4 import BeautifulSoup

HTML_PATH = "../docs/fullcode.html"
OUTPUT_JSON = "../data/structured_output.json"
BLOCKS_PER_PAGE = 8  # you can tweak this to split more or less aggressively

def html_to_structured_json_by_blocks(html_path, blocks_per_page=BLOCKS_PER_PAGE):
    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    body = soup.body or soup
    sections = []

    current_section = {
        "page": 1,
        "text": "",
        "images": []
    }

    block_count = 0

    for elem in body.descendants:
        if not hasattr(elem, "name"):
            continue

        # Add text content
        if elem.name == "p":
            current_section["text"] += elem.get_text(strip=True) + "\n"
            block_count += 1

        elif elem.name == "ul":
            for li in elem.find_all("li"):
                current_section["text"] += "• " + li.get_text(strip=True) + "\n"
            block_count += 1

        elif elem.name == "img":
            src = elem.get("src")
            if src:
                current_section["images"].append(src)
                block_count += 1

        # If enough blocks gathered, start a new section
        if block_count >= blocks_per_page:
            sections.append(current_section)
            current_section = {
                "page": len(sections) + 1,
                "text": "",
                "images": []
            }
            block_count = 0

    # Add the final section if it has content
    if current_section["text"].strip() or current_section["images"]:
        sections.append(current_section)

    return sections

# Main
if __name__ == "__main__":
    result = html_to_structured_json_by_blocks(HTML_PATH)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[DEBUG] ✅ Structured JSON written to {OUTPUT_JSON}")
        print(f"[DEBUG] Total sections: {len(result)}")
