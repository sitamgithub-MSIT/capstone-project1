# Required imports
import os
import sys
from dataclasses import dataclass

# Model training imports
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
)
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

# Local imports
from src.exception import CustomExceptionHandling
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    training_model_file_path = os.path.join("artifacts", "training_model.pkl")

class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array, preprocessor_path):
        try:
            pass

        # Handling the custom exceptions
        except Exception as e:
            raise CustomExceptionHandling(e, sys) from e
