import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object


# @dataclass is a Python decorator from the dataclasses module that makes it easier to create classes meant to store data without writing a lot of boilerplate code.
@dataclass
class DataTransformationConfig:
    """
    > It just take the input data and transforms it into a format that can be used by the model.
    > Stores the file path where the preprocessor object will be saved for later use.
    """

    preprocessor_obj_file_path = os.path.join("artifacts", "proprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function is responsible for data trnasformation

        """
        try:
            # Specifying the numerical and categorical columns required for the transformation
            numerical_columns = [
                "Occupation",
                "Marital_Status",
                "Product_Category_1",
                "Product_Category_2",
                "Product_Category_3",
            ]

            categorical_columns = [
                "Gender",
                "Age",
                "City_Category",
                "Stay_In_Current_City_Years",
            ]

            # Creating a numerical pipeline that will handle missing values and scale the numerical features using StandardScaler and SimpleImputer
            num_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="median"),
                    ),  # Mean or Median anything can be used according to the data, for ourcase median is used
                    ("scaler", StandardScaler()),
                ]
            )

            # Creating a categorical pipeline that will handle missing values, one-hot encode the categorical features, and scale them using StandardScaler
            # StandardScaler is used with with_mean=False because OneHotEncoder will create a sparse matrix
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            # Logging the categorical and numerical columns for debugging purposes
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Combining the numerical and categorical pipelines into a single ColumnTransformer
            # ColumnTransformer allows us to apply different transformations to different columns
            # It applies the num_pipeline to numerical_columns and cat_pipeline to categorical_columns
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipelines", cat_pipeline, categorical_columns),
                ]
            )

            # Returns the preprocessor object which can be used to transform the data
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """

        This function is responsible for data transformation
        It reads the train and test data, applies preprocessing, and returns the transformed arrays.
        It also saves the preprocessing object for future use.

        """
        try:
            # Read the train and test data from the specified paths
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            # Specifying the target column name and numerical columns
            target_column_name = "Purchase"
            numerical_columns = [
                "Occupation",
                "Marital_Status",
                "Product_Category_1",
                "Product_Category_2",
                "Product_Category_3",
            ]

            # Separating the input features and target feature for both train and test data
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")
            # save_object function is inside utils.py
            # its used to save the preprocessor object to the specified file path
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )

            # Returning the transformed train and test arrays in the format (train_arr, test_arr, preprocessor_obj_file_path)
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
