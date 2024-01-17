import logging
from abc import ABC, abstractmethod
import pandas as pd

from sklearn.metrics import mean_squared_error, r2_score

class Evaluate(ABC):
    """Abstract base class for evaluation strategies."""

    @abstractmethod
    def score(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        """Score the model.
        Args:
            y_true: Dataframe containing the true labels.
            y_pred: Dataframe containing the predicted labels.
        Returns:
            Score.
        """
        pass

class MSE(Evaluate):
    """Evaluation strategy for mean squared error."""

    def score(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        """Score the model.
        Args:
            y_true: Dataframe containing the true labels.
            y_pred: Dataframe containing the predicted labels.
        Returns:
            Score.
        """
        try:
            return mean_squared_error(y_true, y_pred)
        except Exception as e:
            logging.error(f"Failed to score model: {e}")
            raise e
        
class R2(Evaluate):
    """Evaluation strategy for R2."""

    def score(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        """Score the model.
        Args:
            y_true: Dataframe containing the true labels.
            y_pred: Dataframe containing the predicted labels.
        Returns:
            Score.
        """
        try:
            return r2_score(y_true, y_pred)
        except Exception as e:
            logging.error(f"Failed to score model: {e}")
            raise e
        
class RMSE(Evaluate):
    """Evaluation strategy for root mean squared error."""

    def score(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        """Score the model.
        Args:
            y_true: Dataframe containing the true labels.
            y_pred: Dataframe containing the predicted labels.
        Returns:
            Score.
        """
        try:
            return mean_squared_error(y_true, y_pred, squared=False)
        except Exception as e:
            logging.error(f"Failed to score model: {e}")
            raise e