import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import TextLoader
import os

# Absolute paths (modify if needed)
MODELS_DIR = "/home/rayen/chatbot_env/models"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def clean_technical_text(text):
    """Pre-process technical manual text"""
    text = re.sub(r"^\d+\s", "", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def get_model_path():
    """Get the exact path to the downloaded model snapshot"""
    # Find the latest snapshot in the Hugging Face cache structure
    snapshots_dir = os.path.join(
        MODELS_DIR,
        f"models--{MODEL_NAME.replace('/', '--')}",
        "snapshots"
    )
    
    if not os.path.exists(snapshots_dir):
        raise FileNotFoundError(f"Model snapshots not found in {snapshots_dir}")
    
    # Get the most recent snapshot
    snapshots = os.listdir(snapshots_dir)
    if not snapshots:
        raise FileNotFoundError(f"No snapshots found in {snapshots_dir}")
    
    return os.path.join(snapshots_dir, snapshots[0])

def create_embeddings(text_path):
    """Create embeddings using the local model only"""
    model_path = get_model_path()
    print(f"Using local model from: {model_path}")
    
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
    
    # Initialize embeddings with local model
    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Create vectorstore
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("Embeddings created successfully (offline mode)")
    return vectorstore