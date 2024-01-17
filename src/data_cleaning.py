import logging 
from typing  import Union

from abc import ABC, abstractmethod
import pandas as pd
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    """Abstract base class for data cleaning strategies."""

    @abstractmethod
    def process_data(self, data: pd.DataFrame) -> Union[pd.DataFrame,pd.Series]:
        """Process the data.
        Args:
            data: Dataframe containing the data to process.
        Returns:
            Processed dataframe.
        """
        pass

class DataPreprocessing(DataStrategy):
    """Data cleaning strategy for preprocessing."""

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data.
        Args:
            df: Dataframe containing the data to preprocess.
        Returns:
            Preprocessed dataframe.
        """
        try:
            df = df.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                ],
                axis=1,
            )
            df["product_weight_g"].fillna(
                df["product_weight_g"].median(), inplace=True
            )
            df["product_length_cm"].fillna(
                df["product_length_cm"].median(), inplace=True
            )
            df["product_height_cm"].fillna(
                df["product_height_cm"].median(), inplace=True
            )
            df["product_width_cm"].fillna(
                df["product_width_cm"].median(), inplace=True
            )
            # write "No review" in review_comment_message column
            df["review_comment_message"].fillna("No review", inplace=True)

            df = df.select_dtypes(include=[np.number])
            cols_to_drop = [
                "customer_zip_code_prefix",
                "order_item_id",
            ]
            df = df.drop(cols_to_drop, axis=1)

            # Catchall fillna in case any where missed
            df.fillna(df.mean(), inplace=True)

            return df
        except Exception as e:
            logging.error(e)
            raise e
        
class DataSplitting(DataStrategy):
    """Data cleaning strategy for splitting."""

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Split the data.
        Args:
            df: Dataframe containing the data to split.
        Returns:
            Split dataframe.
        """
        try:
            # Split into train and test sets
            X = df.drop("order_status", axis=1)
            y = df["order_status"]
            x_train, x_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            return x_train, x_test, y_train, y_test
        except Exception as e:
            logging.error(e)
            raise e
        
class DataCleaning:
    def __init__(self, strategy: DataStrategy):
        self.strategy = strategy()

    def process_data(self, data: pd.DataFrame) -> Union[pd.DataFrame,pd.Series]:
        """Process the data.
        Args:
            data: Dataframe containing the data to process.
        Returns:
            Processed dataframe.
        """
        return self.strategy.process_data(data)