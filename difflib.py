import fitz  # PyMuPDF
import difflib

def extract_text_from_pdf(pdf_path):
    print(f"[DEBUG] Opening PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    text = ""
    for i, page in enumerate(doc):
        print(f"[DEBUG] Extracting text from page {i+1}")
        page_text = page.get_text()
        text += page_text
    print(f"[DEBUG] Finished extracting text from: {pdf_path}")
    return text

def compare_texts(text1, text2):
    print("[DEBUG] Splitting text into lines for comparison")
    lines1 = text1.splitlines()
    lines2 = text2.splitlines()

    print("[DEBUG] Performing unified diff comparison")
    diff = difflib.unified_diff(
        lines1, lines2,
        fromfile='file1',
        tofile='file2',
        lineterm=''
    )
    diff_output = '\n'.join(diff)
    print("[DEBUG] Diff comparison complete")
    return diff_output

# Paths to your two PDF files
pdf_file1 = "deal_report.pdf"
pdf_file2 = "extracted_info.pdf"

print(f"[DEBUG] Starting comparison between {pdf_file1} and {pdf_file2}")

# Extract text
text1 = extract_text_from_pdf(pdf_file1)
text2 = extract_text_from_pdf(pdf_file2)

# Compare and get differences
print("[DEBUG] Comparing extracted texts")
differences = compare_texts(text1, text2)

# Save differences to a file
output_file = "differences.txt"
print(f"[DEBUG] Saving differences to {output_file}")
with open(output_file, "w", encoding="utf-8") as f:
    f.write(differences)

print("[DEBUG] Comparison complete. Differences saved.")
