# Importing the required libraries
import os
import sys
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

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


def evaluate_model(X_train, y_train, X_test, y_test, models):
    """
    Evaluate the performance of different models on the given training and testing data.

    Parameters:
    X_train (array-like): Training data features.
    y_train (array-like): Training data labels.
    X_test (array-like): Testing data features.
    y_test (array-like): Testing data labels.
    models (dict): Dictionary of models to evaluate, where the keys are model names and the values are model objects.

    Returns:
    dict: A dictionary containing the model names as keys and the corresponding R-squared scores on the testing data as values.
    """
    try:
        # Creating a dictionary to store the model names and their R-squared scores
        report = {}

        for i in range(len(list(models))):
            # Fitting the model on the training data
            model = list(models.values())[i]
            model.fit(X_train, y_train)

            # Predicting the labels for the training and testing data
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculating the R-squared scores for the training and testing data
            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            # Storing the scores in the report dictionary
            report[list(models.keys())[i]] = test_score

        return report

    except Exception as e:
        # Handling the exception
        raise CustomExceptionHandling(e, sys) from e
