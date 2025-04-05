import logging

def setup_logging(log_file="chatbot.log"):
    """Set up logging for the chatbot."""
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

if __name__ == "__main__":
    # Example usage
    setup_logging()
    logging.info("Logging setup complete.")