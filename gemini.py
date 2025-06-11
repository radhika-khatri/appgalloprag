import google.generativeai as genai

# === CONFIGURATION ===
API_KEY = ""  # 🔐 Replace with your Gemini API key
INPUT_FILE = "cleaned_output.txt"
INTERMEDIATE_FILE = "filtered_output.txt"
OUTPUT_FILE = "gemini_summary.txt"

# === STEP 1: Load Raw Text ===
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    raw_lines = f.readlines()

# === STEP 2: Define Filtering Rules ===
junk_keywords = [
    'subscribe', 'terms', 'privacy', 'login', 'sign up',
    'back to top', 'cookie', 'advertisement', 'faq', 'contact us'
]

important_keywords = [
    'Organization', 'Contact', 'Email', 'Phone',
    'Amount', 'Credited', 'Expected', 'Assigned', 'Status'
]

# === STEP 3: Filter the Text ===
filtered_lines = []
for line in raw_lines:
    stripped = line.strip()
    if not stripped:
        continue
    if any(junk in stripped.lower() for junk in junk_keywords):
        continue
    if any(key in stripped for key in important_keywords):
        filtered_lines.append(stripped)

filtered_text = "\n".join(filtered_lines)

# Save intermediate filtered text (optional)
with open(INTERMEDIATE_FILE, "w", encoding="utf-8") as f:
    f.write(filtered_text)

# === STEP 4: Setup Gemini ===
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

# === STEP 5: Send Prompt to Gemini ===
prompt = f"""
The following text was extracted from a website and cleaned to remove junk content.
Now, generate a detailed explaination of the data and remove all the redundant data from the given textual file. Group related fields and present the information in paragraph form.

Cleaned Content:
{filtered_text}
"""

response = model.generate_content(prompt)

# === STEP 6: Save Gemini Response ===
# Gemini returns plain text. Let's keep it exactly as-is.
output_text = response.text

# Optional: Fix any escaped `\n` that may appear as strings
output_text = output_text.replace("\\n", "\n").strip()

# Print to console (preserves line breaks)
print("\n=== Gemini Output ===\n")
print(output_text)

# Write to file with line breaks preserved
with open(OUTPUT_FILE, "w", encoding="utf-8") as out_file:
    out_file.write(output_text)

print(f"\n[✔] Properly formatted summary saved to {OUTPUT_FILE}")
