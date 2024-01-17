from zenml import pipeline

from steps.data_ingest import data_ingest
from steps.data_cleaning import data_cleaning
from steps.train_model import train_model
from steps.evaluation import evaluate_model


@pipeline(enable_cache=True)
def training_pipeline(data_path: str) -> None:
    """Training pipeline definition."""
    data = data_ingest(data_path)
    x_train, x_test, y_train, y_test = data_cleaning(data)
    model = train_model(x_train, x_test, y_train, y_test)
    mse_score,r2_score,rmse_score = evaluate_model(model, x_test, y_test)
    print(f"mse_score: {mse_score}")
    print(f"r2_score: {r2_score}")
    print(f"rmse_score: {rmse_score}")
