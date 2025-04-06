from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import re

def create_embeddings(text_path):
    """Create optimized embeddings for C++ docs"""
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
    print(f"Split into {len(chunks)} meaningful chunks")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )
    
    return FAISS.from_documents(chunks, embeddings)