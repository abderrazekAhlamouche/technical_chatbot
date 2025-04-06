from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from pathlib import Path
               
TECHNICAL_PROMPT = """Extract the exact technical specification from this documentation:

{context}

Question: {question}

Answer ONLY with the specific technical details requested. If unavailable, respond "Not specified in documentation"."""

def setup_chatbot(vectorstore):
    # Get the absolute path to the model
    model_dir = Path("./models/models--google--flan-t5-base/snapshots/")
    snapshots = list(model_dir.glob("*"))
    
    if not snapshots:
        raise FileNotFoundError(
            f"No model snapshot found in {model_dir}\n"
            f"Expected folder structure: models/models--google--flan-t5-base/snapshots/<hash>"
        )
    
    model_path = str(snapshots[0])  # Use the first snapshot
    
    try:
        # Load tokenizer and model locally
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)
        
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=150,
            temperature=0.0
        )
        
        # ... rest of your existing code ...
        
    except Exception as e:
        print(f"\nERROR: Failed to load local model from {model_path}")
        print("Verify these files exist:")
        print(f"- {model_path}/config.json")
        print(f"- {model_path}/pytorch_model.bin")
        print(f"- {model_path}/tokenizer_config.json")
        raise e
    
    llm = HuggingFacePipeline(pipeline=pipe)
    
    PROMPT = PromptTemplate(
        template=TECHNICAL_PROMPT,
        input_variables=["context", "question"]
    )
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 2}  # Only top 2 most relevant chunks
        ),
        chain_type_kwargs={"prompt": PROMPT}
    )

def run_chatbot(qa_system):
    print("Technical Documentation Expert ready:")
    while True:
        query = input("\nYour technical question: ").strip()
        if query.lower() == 'exit':
            break
        
        try:
            result = qa_system.invoke({"query": query})
            answer = result["result"].strip()
            
            # Clean up bullet points and numbering
            if answer.startswith(("1.", "- ", "* ")):
                answer = answer.split("\n")[0][2:].strip()
                
            print(f"\nANSWER: {answer}")
            
        except Exception as e:
            print(f"\nSYSTEM ERROR: Please rephrase your question")