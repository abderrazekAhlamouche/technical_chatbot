import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import TextLoader

def clean_technical_text(text):
    """Pre-process technical manual text"""
    # Remove section numbers (e.g., "16 Configuring NTP...")
    text = re.sub(r"^\d+\s", "", text)
    # Collapse multiple newlines
    text = re.sub(r"\n+", " ", text)
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def create_embeddings(text_path):
    """Process technical manual into optimized embeddings"""
    print(f"Loading technical manual from: {text_path}")
    
    # Load and clean document
    loader = TextLoader(text_path)
    raw_docs = loader.load()
    
    # Pre-process text
    for doc in raw_docs:
        doc.page_content = clean_technical_text(doc.page_content)
    
    # Technical manual optimized splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,  # Ideal for technical paragraphs
        chunk_overlap=80,  # Maintains context
        separators=["\n• ", "\n", ". ", "; ", " ", ""],
        length_function=len
    )
    
    chunks = text_splitter.split_documents(raw_docs)
    print(f"Created {len(chunks)} technical document chunks")
    
    # Verify chunk quality
    print("\nSample processed chunk:")
    print(chunks[0].page_content[:200] + "...")
    
    # Create embeddings optimized for technical terms
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}  # Better for technical similarity
    )
    
    # Create vectorstore with MMR indexing
    vectorstore = FAISS.from_documents(
        chunks, 
        embeddings,
        distance_strategy="COSINE"  # Best for technical similarity
    )
    
    print(f"\nVectorstore created with {vectorstore.index.ntotal} entries")
    return vectorstore