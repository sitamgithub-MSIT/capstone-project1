# Required Imports
import sys
import os
import numpy as np
import pandas as pd

# Local Imports
from src.exception import CustomExceptionHandling
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        """
        Predict the target variable using the given features.
        Parameters:
        features (array-like): The features to use for predicting the target variable.
        Returns:
        array-like: The predicted target variable.
        """
        try:
            # Loading the model
            model_path = os.path.join("artifacts", "training_model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            # Predicting the target variable
            transformed_features = preprocessor.transform(features)
            return model.predict(transformed_features)

        except Exception as e:
            # Handling the exception
            raise CustomExceptionHandling(e, sys) from e


class CustomData:
    def __init__(
        self,
        age: int,
        sex: str,
        bmi: float,
        children: int,
        smoker: str,
        region: str,
    ):
        """
        CustomData class represents a custom data input for prediction.

        Args:
            age (int): The age of the individual.
            sex (str): The sex of the individual.
            bmi (int): The body mass index of the individual.
            children (int): The number of children the individual has.
            smoker (str): The smoking status of the individual.
            region (str): The region where the individual is from.
        """
        self.age = age
        self.sex = sex
        self.bmi = bmi
        self.children = children
        self.smoker = smoker
        self.region = region

    def get_data_as_data_frame(self):
        """
        Converts the custom data input into a Pandas DataFrame.

        Returns:
            pandas.DataFrame: The custom data input as a DataFrame.

        Raises:
            CustomExceptionHandling: If an exception occurs during the conversion process.
        """
        try:
            # Creating a dictionary to store the custom data input
            custom_data_input_dict = {
                "age": [self.age],
                "sex": [self.sex],
                "bmi": [self.bmi],
                "children": [self.children],
                "smoker": [self.smoker],
                "region": [self.region],
            }

            return pd.DataFrame(custom_data_input_dict).astype({"children": "object"})
        except Exception as e:
            raise CustomExceptionHandling(e, sys) from e
