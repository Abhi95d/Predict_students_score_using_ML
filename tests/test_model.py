from pathlib import Path
import pandas as pd

from score_prediction.data import load_data
from score_prediction.preprocess import remove_outliers
from score_prediction.model import train_model, evaluate_model


def test_training_pipeline_runs():
    root = Path(__file__).resolve().parents[1]
    df = load_data(root / "StudentsPerformance.csv")
    df_clean = remove_outliers(df)
    target_column = "math score"
    X = df_clean.drop(columns=[target_column])
    y = df_clean[target_column]

    pipeline = train_model(X, y)
    metrics = evaluate_model(pipeline, X, y)

    assert metrics["r2_score"] >= 0.5
    assert metrics["rmse"] >= 0
