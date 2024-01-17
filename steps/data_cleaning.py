import logging
import pandas as pd
from typing import Annotated,Tuple
from zenml import step

from src.data_cleaning import DataCleaning,DataPreprocessing,DataSplitting

@step
def data_cleaning(data: pd.DataFrame
                  ) -> Tuple[Annotated[pd.DataFrame, "x_train"],
                            Annotated[pd.DataFrame,"x_test"],
                            Annotated[pd.DataFrame, "y_train"],
                            Annotated[pd.DataFrame, "y_test"]
    ]:

    try:
        preprocess_data = DataCleaning(DataPreprocessing)
        clean_data = preprocess_data.process_data(data)

        split_data = DataCleaning(DataSplitting)
        x_train, x_test, y_train, y_test = split_data.process_data(clean_data)
        return x_train, x_test, y_train, y_test
    except Exception as e:
        logging.error(f"Failed to clean data: {e}")
        raise e


