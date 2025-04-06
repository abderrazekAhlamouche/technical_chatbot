from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

CXX_PROMPT = """You are a C++ documentation expert. Provide responses in this exact format:

For syntax: [SYNTAX] <exact syntax>
For examples: [EXAMPLE] <complete example>
For definitions: [DEFINITION] <explanation>

If the information cannot be found, respond: "Not found in documentation"

Context:
{context}

Question: {question}

Answer:"""

def setup_chatbot(vectorstore):
    try:
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
        
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=300,
            temperature=0.1,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.2
        )
        
        llm = HuggingFacePipeline(pipeline=pipe)
        
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(
                search_kwargs={
                    "k": 4,
                    "score_threshold": 0.75,
                    "search_type": "mmr"  # Maximal Marginal Relevance
                }
            ),
            chain_type_kwargs={
                "prompt": PromptTemplate(
                    template=CXX_PROMPT,
                    input_variables=["context", "question"]
                )
            },
            return_source_documents=True
        )
    except Exception as e:
        print(f"Failed to initialize chatbot: {str(e)}")
        raise

def format_response(response, query):
    """Strict response formatting"""
    answer = response["result"].split("Source Context:")[0].strip()
    
    # Force standard formats
    if "example" in query.lower() and not answer.startswith("[EXAMPLE]"):
        answer = f"[EXAMPLE]\n{answer}"
    elif ("syntax" in query.lower() or "declare" in query.lower()) and not answer.startswith("[SYNTAX]"):
        answer = f"[SYNTAX]\n{answer}"
    
    return answer if answer else "Not found in documentation"

def run_chatbot(qa_system):
    print("C++ Documentation Expert ready. Ask your questions:")
    while True:
        query = input("\n> ").strip()
        if query.lower() in ('exit', 'quit'):
            break
        
        try:
            result = qa_system.invoke({"query": query})
            print(f"\n{format_response(result, query)}")
            
            # Show context snippets if needed
            if result.get('source_documents'):
                print("\n[Relevant Documentation Context]")
                for i, doc in enumerate(result['source_documents'][:2], 1):
                    print(f"{i}. {doc.page_content[:200]}...")
                    
        except Exception as e:
            print(f"\nError processing query: {str(e)}")