import fitz
import re
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def enhance_cxx_doc(text: str) -> str:
    """Improved C++ documentation structuring"""
    try:
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
    except Exception as e:
        logger.error(f"Error enhancing C++ documentation: {e}")
        raise

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract and enhance text from a PDF file"""
    try:
        doc = fitz.open(pdf_path)
        full_text = []
        
        for page in doc:
            text = page.get_text()
            if text.strip():
                enhanced = enhance_cxx_doc(text)
                full_text.append(enhanced)
        
        logger.info(f"Extracted and enhanced text from {pdf_path}")
        return "\n\n".join(full_text)
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise