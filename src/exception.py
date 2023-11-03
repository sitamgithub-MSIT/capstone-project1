"""
This module defines a custom exception handling class and a function to get error message with details of the error.
"""

# Standard Library
import sys

# Local
from src.logger import logging


# Function Definition to get error message with details of the error (file name and line number) when an error occurs in the program
def get_error_message(error, error_detail: sys) -> str:
    """
    Get error message with details of the error.

    Parameters:
    error (Exception): The error that occurred.
    error_detail (sys): The details of the error.

    Returns:
    str: A string containing the error message along with the file name and line number where the error occurred.
    """

    _, _, tb = error_detail.exc_info()

    # Get error details
    file_name = tb.tb_frame.f_code.co_filename
    line_number = tb.tb_lineno
    string_error = str(error)

    # Return error message
    return "Error in python script name [{0}] line number [{1}] with error message [{2}]".format(
        file_name, line_number, string_error
    )


# Custom Exception Handling Class Definition
class CustomExceptionHandling(Exception):
    """
    Custom Exception Handling

    This class defines a custom exception that can be raised when an error occurs in the program.
    It takes an error message and an error detail as input and returns a formatted error message
    when the exception is raised.
    """

    # Constructor
    def __init__(self, error_message, error_detail: sys):
        """Initialize the exception"""
        super().__init__(error_message)

        self.error_message = get_error_message(error_message, error_detail=error_detail)

    def __str__(self):
        """String representation of the exception"""
        return self.error_message
