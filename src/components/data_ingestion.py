# Required imports for the class to work
import os
import sys
from src.exception import CustomExceptionHandling
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


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

    # Constructor to initialize the class variables
    def __init__(self):
        self.ingestion_config = (
            DataIngestionConfig()
        )  # ingestionConfig is where data is stored

    # Function to initiate the data ingestion process
    def initiate_data_ingestion(self):
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


# Main function to initiate the data ingestion process
if __name__ == "__main__":
    # Creating the object for the class and calling the function
    data_ingestion = DataIngestion()
    data_ingestion.initiate_data_ingestion()
