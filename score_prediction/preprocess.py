from pathlib import Path
from typing import List
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

TARGET_COLUMN = "math score"


def get_categorical_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=["object"]).columns.tolist()


def remove_outliers(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> pd.DataFrame:
    q1 = df[target_column].quantile(0.25)
    q3 = df[target_column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return df[(df[target_column] >= lower) & (df[target_column] <= upper)].copy()


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    feature_df = df.drop(columns=[TARGET_COLUMN], errors="ignore")
    cat_cols = get_categorical_columns(feature_df)
    numeric_cols = [col for col in ["reading score", "writing score"] if col in feature_df.columns]

    transformer_list = [
        (
            "onehot",
            OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
            cat_cols,
        )
    ]

    if numeric_cols:
        transformer_list.append(("minmax", MinMaxScaler(), numeric_cols))

    return ColumnTransformer(
        transformer_list,
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
