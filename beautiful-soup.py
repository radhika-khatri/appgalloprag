import requests
from bs4 import BeautifulSoup

# Create a session
session = requests.Session()

# Define login details and login URL
login_url = 'https://demo1.sellarge.com/config/product_login'  # Replace with actual login URL
payload = {
    'username': '',
    'password': ''
}

# Log in
session.post(login_url, data=payload)

# Scrape a protected page
target_url = 'https://demo1.sellarge.com/knowledge-base'  # Replace with actual page
response = session.get(target_url)

# Parse HTMLc:\Users\Radhika Khatri\Desktop\AppGallop\RasaChatbot
soup = BeautifulSoup(response.text, 'html.parser')

# Remove unwanted elements
for tag in soup(['script', 'style', 'meta', 'noscript']):
    tag.extract()

# Extract visible text
text = soup.get_text(separator='\n')

# Clean whitespace and blank lines
cleaned_text = "\n".join(
    line.strip() for line in text.splitlines() if line.strip()
)

# Print the cleaned, readable text
print("\n=== Extracted Text ===\n")
print(cleaned_text)

# Save cleaned text
with open("cleaned_output.txt", "w", encoding="utf-8") as output_file:
    output_file.write(cleaned_text)

print("[✔] Cleaned text saved to cleaned_output.txt")