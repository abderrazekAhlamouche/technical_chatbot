import logging
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CXX_PROMPT = """You are a C++ documentation expert. Provide responses in this exact format:

For syntax: [SYNTAX] <exact syntax>
For examples: [EXAMPLE] <complete example>
For definitions: [DEFINITION] <explanation>

If the information cannot be found, respond: "Not found in documentation"

Context:
{context}

Question: {question}

Answer:"""

def setup_chatbot(vectorstore, model_name: str = "google/flan-t5-base", device: str = 'cpu'):
    """Setup the chatbot with the given vectorstore, model, and device."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=300,
            temperature=0.1,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.2,
            device=0 if device == 'cuda' else -1  # Use GPU if available
        )
        
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=CXX_PROMPT
        )
        
        qa_chain = RetrievalQA(
            retriever=vectorstore.as_retriever(),
            llm=HuggingFacePipeline(pipeline=pipe),
            prompt_template=prompt_template
        )
        
        logger.info("Chatbot setup successfully")
        return qa_chain
    
    except Exception as e:
        logger.error(f"Error setting up chatbot: {e}")
        raise