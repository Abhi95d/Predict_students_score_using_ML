from pathlib import Path
from sklearn.model_selection import train_test_split

from score_prediction.data import load_data
from score_prediction.model import evaluate_model, save_model, train_model, get_default_model_path
from score_prediction.preprocess import remove_outliers


def main() -> None:
    root = Path(__file__).resolve().parent
    model_path = get_default_model_path(root)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_data(root / "data" / "StudentsPerformance.csv")
    df_clean = remove_outliers(df)

    target_column = "math score"
    X = df_clean.drop(columns=[target_column])
    y = df_clean[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = train_model(X_train, y_train)
    save_model(pipeline, model_path)

    metrics = evaluate_model(pipeline, X_test, y_test)
    print("✅ Training complete")
    print(f"Model saved to: {model_path}")
    print("Evaluation:")
    print(f"  - R2: {metrics['r2_score']:.4f}")
    print(f"  - RMSE: {metrics['rmse']:.4f}")


if __name__ == "__main__":
    main()
