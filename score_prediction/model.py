from pathlib import Path
import joblib
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .preprocess import build_preprocessor

MODEL_FILENAME = "student_score_model.pkl"


def get_default_model_path(root: Path | None = None) -> Path:
    root_path = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    return root_path / "saved_models" / MODEL_FILENAME


def create_pipeline(df):
    preprocessor = build_preprocessor(df)
    regressor = TransformedTargetRegressor(
        regressor=GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
        transformer=StandardScaler(),
    )
    return Pipeline([("preprocessor", preprocessor), ("regressor", regressor)])


def train_model(X, y):
    pipeline = create_pipeline(X)
    pipeline.fit(X, y)
    return pipeline


def evaluate_model(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return {
        "r2_score": r2_score(y_test, y_pred),
        "rmse": np.sqrt(mse),
    }


def save_model(pipeline, path):
    joblib.dump(pipeline, path)


def load_model(path):
    return joblib.load(path)
