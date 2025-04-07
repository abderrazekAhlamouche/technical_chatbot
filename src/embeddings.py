import logging
from typing import Optional
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_embeddings(text_path: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = 'cpu') -> FAISS:
    """Create optimized embeddings for C++ docs"""
    try:
        loader = TextLoader(text_path)
        docs = loader.load()
        
        # Custom splitting for C++ documentation
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=150,
            separators=["\n[SECTION]", "\n[CODE]", "\n\n", "\n"],
            keep_separator=True
        )
        
        chunks = text_splitter.split_documents(docs)
        logger.info(f"Split into {len(chunks)} meaningful chunks")
        
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': False}
        )
        
        return FAISS.from_documents(chunks, embeddings)
    
    except Exception as e:
        logger.error(f"Error creating embeddings: {e}")
        raise

# Example usage
if __name__ == "__main__":
    text_path = "path/to/your/textfile.txt"
    faiss_index = create_embeddings(text_path)
    logger.info("Embeddings created successfully")