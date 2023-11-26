# Required imports
import sys
import os

from dataclasses import dataclass
import numpy as np
import pandas as pd

# Feature engineering imports
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# Local imports
from src.exception import CustomExceptionHandling
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        """
        Returns a preprocessor object that performs data transformation on numerical and categorical features.

        Returns:
            preprocessor (ColumnTransformer): Preprocessor object for data transformation.
        Raises:
            CustomExceptionHandling: If an exception occurs during the data transformation process.
        """
        try:
            numerical_features = ["age", "bmi"]
            categorical_features = ["sex", "smoker", "region", "children"]

            # Creating the pipeline objects for numerical and categorical features
            numerical_pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ("ordinal_encoder", OrdinalEncoder()),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            # Logging the numerical and categorical columns
            logging.info(f"Numerical columns : {numerical_features}")
            logging.info(f"Categorical columns: {categorical_features}")

            # Creating the column transformer object
            preprocessor = ColumnTransformer(
                transformers=[
                    ("numerical_pipeline", numerical_pipeline, numerical_features),
                    (
                        "categorical_pipeline",
                        categorical_pipeline,
                        categorical_features,
                    ),
                ]
            )

            return preprocessor

        # Handling the custom exceptions
        except Exception as e:
            raise CustomExceptionHandling(e, sys) from e

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
            input_features_train_df = train_df.drop(columns=[target_col], axis=1)
            target_feature_train_df = train_df[target_col]

            input_features_test_df = test_df.drop(columns=[target_col], axis=1)
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
            save_object(
                obj=preprocessing_obj,
                file_path=self.transformation_config.preprocessor_obj_file_path,
            )

            # Returning the transformed train and test data arr with path of the preprocessor obj
            return (
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_obj_file_path,
            )

        # Handling the custom exceptions
        except Exception as e:
            raise CustomExceptionHandling(e, sys) from e
