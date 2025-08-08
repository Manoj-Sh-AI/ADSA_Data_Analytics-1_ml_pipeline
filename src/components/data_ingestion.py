import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# DATA TRANSFORMATION IMPORTS
# Importing the DataTransformation class and DataTransformationConfig from the data_transformation module
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

# MODEL TRAINING IMPORTS
# Importing the ModelTrainerConfig and ModelTrainer from the model_trainer module
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    # Creates the artifacts directory if it doesn't exist, artifacts contains the train, test, and raw data files in CSV format
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """_summary_"""
        logging.info("Entered the data ingestion method or component")
        try:
            # Read the Data from files/ API / Database / etc.
            df = pd.read_csv("src/notebooks/data/train.csv")
            logging.info("Read the dataset as dataframe")

            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True
            )

            # Converting the raw data into csv
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Conducting the train test split
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Saving the train and test data
            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )
            test_set.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )
            logging.info("Ingestion of the data is completed")

            # Returning the train and test data paths
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":

    # Data Ingestion
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    # Data Transformation
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
        train_data, test_data
    )

    # Model Trainer
    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))
