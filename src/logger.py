# Standard Library
import os
import logging
from datetime import datetime


# Function Definition to log the errors and warnings in the logs directory of the project folder
def create_logger():
    """
    Create logger for the program.
    """
    # Create logs directory
    LOG_FILE = f"{datetime.now().strftime('%m-%d%Y-%H-%M-%S')}.log"
    logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)

    os.makedirs(logs_path, exist_ok=True)

    # Create log file path
    LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

    # Configure logging
    logging.basicConfig(
        filename=LOG_FILE_PATH,
        format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
