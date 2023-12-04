# Required imports
import os
import sys
from dataclasses import dataclass

# Model training imports
from sklearn.linear_model import LinearRegression
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
from src.utils import save_object, evaluate_model


@dataclass
class ModelTrainerConfig:
    training_model_file_path = os.path.join("artifacts", "training_model.pkl")


class ModelTrainer:
    """
    Class for training machine learning models.

    Attributes:
        model_trainer_config (ModelTrainerConfig): Configuration object for model training.

    Methods:
        initiate_model_training(train_array, test_array): Initiates the model training process.
    """

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        """
        Initiates the model training process.

        Args:
            train_array (numpy.ndarray): Array containing the training data.
            test_array (numpy.ndarray): Array containing the test data.

        Returns:
            dict: Dictionary containing the model report, best model name, best model score, and r2 score.
        Raises:
            CustomExceptionHandling: If the best model score is less than 0.6.
        """
        try:
            logging.info("Splitting the data into train and test sets")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            # Creating a dictionary of models to evaluate
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # Creating a dictionary of model parameters
            params = {
                "Random Forest": {"n_estimators": [8, 16, 32, 64, 128, 256]},
                "Decision Tree": {
                    "criterion": [
                        "squared_error",
                        "friedman_mse",
                        "absolute_error",
                        "poisson",
                    ],
                },
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "CatBoosting Regressor": {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [30, 50, 100],
                },
                "AdaBoost Regressor": {
                    "learning_rate": [0.1, 0.01, 0.5, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
            }

            # Evaluating the models on the training and testing data
            model_report = dict(
                evaluate_model(X_train, y_train, X_test, y_test, models, params)
            )

            # Finding the best model
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            # Checking if the best model score is greater than 0.6 or not
            if best_model_score < 0.6:
                raise CustomExceptionHandling(
                    "Best model score is less than 0.6. Please try again with a different set of features."
                )
            logging.info(f"Best model on both train and test data: {best_model_name}")

            # Saving the best model as a pickle file and calculating the r2 score
            save_object(best_model, self.model_trainer_config.training_model_file_path)

            prediction = best_model.predict(X_test)
            r2 = r2_score(y_test, prediction)

            # Returning the model report, best model name, best model score, and r2 score
            return {
                "model_report": model_report,
                "best_model_name": best_model_name,
                "best_model_score": best_model_score,
                "r2_score": r2,
            }

        # Handling the custom exceptions
        except Exception as e:
            raise CustomExceptionHandling(e, sys) from e
