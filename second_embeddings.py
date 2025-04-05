import os
from pathlib import Path
from langchain_community.document_loaders import TextLoader  # Updated import
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configuration
MODEL_PATH = Path("/home/abderrazek/technical_chatbot/models/all-MiniLM-L6-v2")

def create_embeddings(text_path):
    """Create embeddings from documents using local model"""
    
    # 1. Verify model files (critical check)
    required_files = {
        "config.json": "Model configuration",
        "pytorch_model.bin": "Model weights",
        "tokenizer_config.json": "Tokenizer config",
        "vocab.txt": "Vocabulary",
        "special_tokens_map.json": "Special tokens"
    }
    
    missing_files = [f for f in required_files if not (MODEL_PATH / f).exists()]
    if missing_files:
        raise FileNotFoundError(
            f"Missing files in {MODEL_PATH}:\n"
            f"{', '.join(missing_files)}\n"
            f"Required files: {list(required_files.keys())}"
        )

    # 2. Initialize embeddings with proper local path handling
    embeddings = HuggingFaceEmbeddings(
        model_name=str(MODEL_PATH.absolute()),
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,
            "show_progress_bar": False
        }
    )
    
    # 3. Process documents with fixed separator
    loader = TextLoader(text_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", "(?<=. )", " ", ""]  # Fixed regex escape
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Processed {len(chunks)} document chunks")
    
    # 4. Create and return vectorstore
    return FAISS.from_documents(chunks, embeddings)