# Importing the required libraries
import os
import sys

import numpy as np
import pandas as pd

# Exception handling imports
from src.exception import CustomExceptionHandling


def save_object(obj, file_path):
    """
    Save the given object as a pickle file at the specified file path.

    Args:
        obj: The object to be saved.
        file_path: The path where the object will be saved.

    Raises:
        CustomExceptionHandling: If there is an error while saving the object.

    """
    try:
        # Creating the directory if it does not exist
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Saving the object as a pickle file
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        # Handling the exception
        raise CustomExceptionHandling(e, sys) from e
