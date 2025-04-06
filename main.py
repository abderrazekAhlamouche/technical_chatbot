from src.embeddings import create_embeddings
from src.chatbot import setup_chatbot, run_chatbot
from pathlib import Path

def main():
    try:
        base_dir = Path(__file__).parent
        text_path = base_dir/"data"/"processed_docs.txt"
        
        print("Creating embeddings...")
        vectorstore = create_embeddings(text_path)
        print(f"Vectorstore ready with {vectorstore.index.ntotal} chunks")
        
        qa_system = setup_chatbot(vectorstore)
        run_chatbot(qa_system)
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Ensure you've:")
        print("1. Run data_loader.py first")
        print("2. Installed all requirements")

if __name__ == "__main__":
    main()