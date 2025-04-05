from src.embeddings import create_embeddings
from src.chatbot import setup_chatbot, run_chatbot

def main():
    # Path to your text file
    text_path = "data/product_docs.txt"
    
    # Create embeddings
    print("Creating vectorstore...")
    vectorstore = create_embeddings(text_path)
    print(f"Loaded {vectorstore.index.ntotal} document chunks")
    
    # Setup chatbot
    qa_system = setup_chatbot(vectorstore)
    
    # Run chatbot
    run_chatbot(qa_system)

if __name__ == "__main__":
    main()