import fitz  # PyMuPDF
import difflib

def extract_text_from_pdf(pdf_path):
    print(f"[DEBUG] Opening PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    text = ""
    for i, page in enumerate(doc):
        print(f"[DEBUG] Extracting text from page {i+1}")
        text += page.get_text()
    print(f"[DEBUG] Finished extracting text from: {pdf_path}")
    return text

def clean_lines(text):
    # Remove extra spaces and blank lines
    lines = text.splitlines()
    cleaned = [line.strip() for line in lines if line.strip()]
    return cleaned

def compare_texts_ignore_spaces(text1, text2):
    print("[DEBUG] Cleaning and splitting text for comparison")
    lines1 = clean_lines(text1)
    lines2 = clean_lines(text2)

    print("[DEBUG] Performing unified diff comparison (ignoring spaces)")
    diff = difflib.unified_diff(
        lines1, lines2,
        fromfile='file1',
        tofile='file2',
        lineterm=''
    )
    return '\n'.join(diff)

# --- File paths ---
pdf_file1 = "Name.pdf"
pdf_file2 = "extracted_info.pdf"

print(f"[DEBUG] Starting comparison between {pdf_file1} and {pdf_file2}")

# --- Extract text ---
text1 = extract_text_from_pdf(pdf_file1)
text2 = extract_text_from_pdf(pdf_file2)

# --- Compare and save differences ---
differences = compare_texts_ignore_spaces(text1, text2)

output_file = "differences.txt"
print(f"[DEBUG] Saving differences to {output_file}")
with open(output_file, "w", encoding="utf-8") as f:
    f.write(differences)

print("[DEBUG] Comparison complete. Differences saved.")
