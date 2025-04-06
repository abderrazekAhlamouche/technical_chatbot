import fitz
import re
from pathlib import Path

def enhance_cxx_doc(text):
    """Improved C++ documentation structuring"""
    # Fix common OCR errors
    corrections = {
        r'std;;': 'std::',
        r'Outf': '0xff',
        r'#hort': 'short',
        r'eeplicit': 'explicit'
    }
    for wrong, right in corrections.items():
        text = re.sub(wrong, right, text)
    
    # Add explicit section markers
    text = re.sub(r'^([A-Z]{2,}[A-Z\s]+)$', r'[SECTION] \1', text, flags=re.MULTILINE)
    
    # Format code blocks
    text = re.sub(r'^(\s*[A-Za-z_].*?[;{}]\s*)$', r'[CODE]\n\1\n[/CODE]', text, flags=re.MULTILINE)
    
    return text

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = []
    
    for page in doc:
        text = page.get_text()
        if text.strip():
            enhanced = enhance_cxx_doc(text)
            full_text.append(enhanced)
    
    return "\n\n".join(full_text)

def save_text_to_file(text, output_path):
    with open(output_path, "w", encoding='utf-8') as f:
        f.write(text)

if __name__ == "__main__":
    pdf_path = "data/product_docs.pdf"
    output_path = "data/processed_docs.txt"
    
    print(f"Processing {pdf_path}...")
    text = extract_text_from_pdf(pdf_path)
    save_text_to_file(text, output_path)
    print(f"Saved enhanced documentation to {output_path}")