import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import TextLoader
import os

def clean_technical_text(text):
    """Pre-process technical manual text"""
    text = re.sub(r"^\d+\s", "", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def create_embeddings(text_path, embedding_model_path="./models", vectorstore_path="./vectorstore"):
    """Create and save embeddings locally"""
    
    # Load and clean document
    loader = TextLoader(text_path)
    raw_docs = loader.load()
    
    for doc in raw_docs:
        doc.page_content = clean_technical_text(doc.page_content)
    
    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
        separators=["\n• ", "\n", ". ", "; ", " ", ""],
    )
    chunks = text_splitter.split_documents(raw_docs)
    
    # Load local embeddings (download model first if needed)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder=embedding_model_path,  # Store model locally
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Create and save vectorstore locally
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Save embeddings to disk
    os.makedirs(vectorstore_path, exist_ok=True)
    vectorstore.save_local(vectorstore_path)
    print(f"Vectorstore saved to {vectorstore_path}")
    return vectorstore

def load_embeddings(vectorstore_path="./vectorstore", embedding_model_path="./models"):
    """Load pre-computed embeddings from disk"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder=embedding_model_path,
        model_kwargs={'device': 'cpu'},
    )
    vectorstore = FAISS.load_local(vectorstore_path, embeddings)
    return vectorstore