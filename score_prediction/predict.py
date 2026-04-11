import pandas as pd
from pathlib import Path

from .model import load_model


def prepare_input(
    gender: str,
    race_ethnicity: str,
    parental_education: str,
    lunch: str,
    test_prep: str,
    reading_score: int,
    writing_score: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gender": [gender],
            "race/ethnicity": [race_ethnicity],
            "parental level of education": [parental_education],
            "lunch": [lunch],
            "test preparation course": [test_prep],
            "reading score": [reading_score],
            "writing score": [writing_score],
        }
    )


def predict_score(pipeline, input_df: pd.DataFrame) -> float:
    return float(pipeline.predict(input_df)[0])


def load_trained_model(model_path: Path | str | None = None):
    if model_path is None:
        model_path = Path(__file__).resolve().parents[1] / "saved_models" / "student_score_model.pkl"
    return load_model(model_path)
