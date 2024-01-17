import logging
import pandas as pd
from zenml import step
from typing import Annotated,Tuple

from sklearn.base import RegressorMixin
from src.evaluation import MSE,R2,RMSE

@step
def evaluate_model(model:RegressorMixin,
                    x_test: pd.DataFrame,
                    y_test: pd.DataFrame,
) ->Tuple[Annotated[float,"mse_score"],
          Annotated[float,"r2_score"],
          Annotated[float,"rmse_score"]]:
    """Evaluates the model.
    Args:
        model: Trained model.
        x_test: Dataframe containing the testing features.
        y_test: Dataframe containing the testing labels.
    Returns:
        score: Score of the model.
    """
    try:
        logging.info("Evaluating model...")
        y_pred = model.predict(x_test)
        mse = MSE()
        r2 = R2()
        rmse = RMSE()
        mse_score = mse.score(y_test,y_pred)
        r2_score = r2.score(y_test,y_pred)
        rmse_score = rmse.score(y_test,y_pred)
        return mse_score,r2_score,rmse_score
    except Exception as e:
        logging.error(f"Failed to evaluate model: {e}")
        raise e