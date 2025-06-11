from xhtml2pdf import pisa

# Input and output file paths
input_html = "fullcode.html"
output_pdf = "output.pdf"

# Read the HTML file
with open(input_html, "r", encoding="utf-8") as html_file:
    html_content = html_file.read()

# Convert HTML to PDF
with open(output_pdf, "w+b") as pdf_file:
    pisa_status = pisa.CreatePDF(html_content, dest=pdf_file)
    if pisa_status.err:
        print("An error occurred during PDF creation!")
    else:
        print("PDF created successfully!")
