# Standard Library
import sys

# Local
from src.logger import logging


def get_error_message(error, error_detail: sys) -> str:
    """Get error message"""

    _, _, tb = error_detail.exc_info()

    # Get error details
    file_name = tb.tb_frame.f_code.co_filename
    line_number = tb.tb_lineno
    string_error = str(error)

    # Return error message
    return "Error in python script name [{0}] line number [{1}] with error message [{2}]".format(
        file_name, line_number, string_error
    )


class CustomExceptionHandling(Exception):

    """Custom Exception Handling"""

    # Constructor
    def __init__(self, error_message, error_detail: sys):
        """Initialize the exception"""
        super().__init__(error_message)

        self.error_message = get_error_message(error_message, error_detail=error_detail)

    def __str__(self):
        """String representation of the exception"""
        return self.error_message
