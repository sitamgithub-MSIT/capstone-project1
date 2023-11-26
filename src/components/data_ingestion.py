# Required imports for the class to work
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# Local imports
from src.exception import CustomExceptionHandling
from src.logger import logging

# Transformation imports
from src.components.data_transformation import (
    DataTransformationConfig,
    DataTransformation,
)


@dataclass
class DataIngestionConfig:
    """
    A configuration class for data ingestion.

    Attributes:
    -----------
    train_data_path : str
        The path to the training data file.
    test_data_path : str
        The path to the test data file.
    raw_data_path : str
        The path to the raw data file.
    """

    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


# Data ingestion class to load the data from the source location to the destination location
class DataIngestion:
    """
    A class used to ingest data from a source location, split it into train and test sets,
    and save the resulting data to a destination location.

    Attributes
    ----------
    ingestion_config : DataIngestionConfig
        The configuration object used to store data ingestion settings.

    Methods
    -------
    initiate_data_ingestion()
        Initiates the data ingestion process and returns the paths to the resulting train and test sets.
    """

    def __init__(self):
        """
        Constructor to initialize the class variables.
        """
        self.ingestion_config = (
            DataIngestionConfig()
        )  # ingestionConfig is where data is stored

    def initiate_data_ingestion(self):
        """
        Initiates the data ingestion process and returns the paths to the resulting train and test sets.

        Returns
        -------
        tuple
            A tuple containing the paths to the resulting train and test sets.
        """
        logging.info(
            "data ingestion is started from the source location to the destination location"
        )

        try:
            # Loading the data from the source location to the dataframe object
            df = pd.read_csv("notebook/data/insurance.csv").drop_duplicates()
            logging.info(
                "data is loaded successfully from the source location to the dataframe object"
            )

            # Creating the destination location if it does not exist and saving the data as csv file
            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True
            )
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(
                "train test split is getting ready to split the data into train and test sets"
            )

            # Splitting the data into train and test sets
            train_set, test_set = train_test_split(
                df, train_size=0.8, test_size=0.2, random_state=42
            )

            # Saving the train and test sets as csv files
            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )
            test_set.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )

            # Logging the completion of the data ingestion process
            logging.info(
                "ingestion is completed and train and test sets are created successfully"
            )
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            # Custom exception handling
            raise CustomExceptionHandling(e, sys) from e


if __name__ == "__main__":
    # Initiating the data ingestion process
    data_ingestion = DataIngestion()
    train_data, test_data = data_ingestion.initiate_data_ingestion()

    # Initiating the data transformation process
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
        train_data, test_data
    )
