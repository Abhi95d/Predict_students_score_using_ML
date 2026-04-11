from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT / "data" / "StudentsPerformance.csv"


def load_data(csv_path: Path | str | None = None) -> pd.DataFrame:
    """Load the student performance dataset."""
    dataset_path = Path(csv_path) if csv_path is not None else DEFAULT_DATA_PATH
    return pd.read_csv(dataset_path)
