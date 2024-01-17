import logging 
import pandas as pd
from zenml import step

class DataIngest:
    def __init__(self, path):
        self.path = path

    def read_csv(self):
        return pd.read_csv(self.path)
    
@step
def data_ingest(data_path:str) -> pd.DataFrame:
    """Reads data from the path specified in the config file."""
    try:
        logging.info("Reading data...")
        data = DataIngest(path=data_path)
        return data.read_csv()
    except Exception as e:
        logging.error(f"Failed to read data: {e}")
        raise e

