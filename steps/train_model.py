import logging
from zenml import step
import pandas as pd

from src.model import LinearRegressionModel
from sklearn.base import RegressorMixin

@step
def train_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,

) -> RegressorMixin:
    """Trains the model.
    Args:
        x_train: Dataframe containing the training features.
        x_test: Dataframe containing the testing features.
        y_train: Dataframe containing the training labels.
        y_test: Dataframe containing the testing labels.
    Returns:
        model: Trained model.
    """
    try:
        logging.info("Training model...")
        model = LinearRegressionModel()
        model.train(x_train, y_train)
        return model
    except Exception as e:
        logging.error(f"Failed to train model: {e}")
        raise e

