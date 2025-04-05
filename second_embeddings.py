import os
from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader

# Configuration
MODEL_PATH = Path("/home/abderrazek/technical_chatbot/models/all-MiniLM-L6-v2")

def create_embeddings(text_path):
    """Create embeddings from documents using local model"""
    
    # 1. Verify model files
    required_files = {
        "config.json": "Model configuration",
        "pytorch_model.bin": "Model weights",
        "tokenizer_config.json": "Tokenizer config"
    }
    
    for file, desc in required_files.items():
        if not (MODEL_PATH / file).exists():
            raise FileNotFoundError(
                f"Missing {desc} at {MODEL_PATH/file}\n"
                f"Required files: {list(required_files.keys())}"
            )

    # 2. Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=str(MODEL_PATH.absolute()),  # Convert to absolute path string
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,
            "show_progress_bar": False
        }
    )
    
    # 3. Process documents
    loader = TextLoader(text_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", "(?<=\. )", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Processed {len(chunks)} document chunks")
    
    # 4. Create vectorstore
    return FAISS.from_documents(chunks, embeddings)