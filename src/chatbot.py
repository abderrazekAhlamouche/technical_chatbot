from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

TECHNICAL_PROMPT = """Extract the exact technical specification from this documentation:

{context}

Question: {question}

Answer ONLY with the specific technical details requested. If unavailable, respond "Not specified in documentation"."""

def setup_chatbot(vectorstore):
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=150,  # Shorter, focused answers
        do_sample=False,
        temperature=0.0,
        no_repeat_ngram_size=2  # Prevents repeated phrases
    )
    
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