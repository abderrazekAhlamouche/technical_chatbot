import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def save_text_to_file(text, output_path):
    """Save extracted text to a file."""
    with open(output_path, "w") as f:
        f.write(text)

if __name__ == "__main__":
    # Example usage
    pdf_path = "../data/product_docs.pdf"
    output_path = "../data/product_docs.txt"
    text = extract_text_from_pdf(pdf_path)
    save_text_to_file(text, output_path)
    print(f"Text extracted and saved to {output_path}")
    