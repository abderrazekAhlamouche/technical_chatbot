import logging
from datetime import datetime

def setup_logging(log_file="cxx_chatbot.log"):
    """Enhanced logging for C++ queries"""
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logging.info("C++ Reference Chatbot initialized")

def log_query(query, response):
    """Log C++ specific interactions"""
    logging.info(f"QUERY: {query}")
    logging.info(f"RESPONSE: {response[:200]}...")  # Truncate long responses

if __name__ == "__main__":
    setup_logging()