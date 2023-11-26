# Required imports 
import sys
import os

from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Local imports
from src.exception import CustomExceptionHandling
from src.logger import logging


@dataclass
class DataTransformationConfig:

    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:

    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):

        try:
            numerical_features = ['Age', 'Fare']

            categorical_features = []

            # Creating the pipeline object

            numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            # Creating the column transformer object

            preprocessor = ColumnTransformer(transformers=[
                ('num', numerical_pipeline, numerical_features),
                ('cat', categorical_pipeline, categorical_features)
            ])

            return preprocessor
        
        
        except Exception as e:
            logging.error('Error occurred while loading the preprocessor object')
            CustomExceptionHandling(e)


    def initiate_data_transformation(self, train_data_path, test_data_path):
        pass

