# Required imports
import sys
import os

from dataclasses import dataclass
import numpy as np
import pandas as pd

# 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Local imports
from src.exception import CustomExceptionHandling
from src.logger import logging


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            numerical_features = ["Age", "Fare"]

            categorical_features = []

            # Creating the pipeline object

            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            )

            # Creating the column transformer object

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numerical_pipeline, numerical_features),
                    ("cat", categorical_pipeline, categorical_features),
                ]
            )

            return preprocessor

        except Exception as e:
            logging.error("Error occurred while loading the preprocessor object")
            CustomExceptionHandling(e)

    def initiate_data_transformation(self, train_data_path, test_data_path):
        """
        Initiates the data transformation process.

        Args:
            train_data_path (str): The file path of the training data.
            test_data_path (str): The file path of the test data.

        Returns:
            tuple: A tuple containing the transformed training and test data arrays, and the file path of the saved preprocessor object.
        """
        try:
            # Reading the train and test data
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            # Converting the children column to object type
            train_df["children"] = train_df["children"].astype(object)
            test_df["children"] = test_df["children"].astype(object)

            logging.info(
                "Train data and Test data read successfully from the source location"
            )
            logging.info(
                "Obtaining the preprocessor object for data transformation process"
            )

            # Obtaining the preprocessor object
            preprocessing_obj = self.get_data_transformation_object()
            target_col = "charges"

            # Splitting the train and test data into input features and target feature
            input_features_train_df = train_df.drop(target_col, axis=1)
            target_feature_train_df = train_df[target_col]

            input_features_test_df = test_df.drop(target_col, axis=1)
            target_feature_test_df = test_df[target_col]

            logging.info(
                "Fitting the preprocessor object on both train and test data sets"
            )

            # Fitting the preprocessor object on both train and test data sets
            input_features_train_arr = preprocessing_obj.fit_transform(
                input_features_train_df
            )
            input_features_test_arr = preprocessing_obj.transform(
                input_features_test_df
            )

            # Concatenating the input features and target feature
            train_arr = np.c_[
                input_features_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_features_test_arr, np.array(target_feature_test_df)]

            logging.info("Saving the preprocessor object")

            # Saving the preprocessor object
            saved_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )

            # Returning the transformed train and test data arrays along with the file path of the saved preprocessor object
            return (
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_obj_file_path,
            )

        # Handling the custom exceptions
        except Exception as e:
            raise CustomExceptionHandling(e, sys) from e
