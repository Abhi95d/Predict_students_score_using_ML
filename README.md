# Student Score Prediction

This project trains a student math score prediction model and exposes it through a Streamlit app.

## Project Structure

- `data/StudentsPerformance.csv` - organized dataset folder
- `scr_prediction.ipynb` - exploratory notebook for data analysis and initial modeling
- `app.py` - Streamlit app entrypoint
- `pages/Model_Explainability.py` - Streamlit model explainability page
- `train.py` - training script for building and saving the model
- `saved_models/` - trained model artifacts
- `score_prediction/` - reusable Python package for data, preprocessing, modeling, and prediction
- `requirements.txt` - Python dependencies
- `Dockerfile` - container deployment definition
- `.dockerignore` - Docker build ignore patterns

## Setup

1. Create a virtual environment:

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Train the model

```bash
python train.py
```

This creates `saved_models/student_score_model.pkl`.

## Run the Streamlit app locally

```bash
streamlit run app.py
```

## Docker deployment

Build the image:

```bash
docker build -t student-score-predictor .
```

Run the container:

```bash
docker run -p 8501:8501 student-score-predictor
```

## Project notes

- The `score_prediction` package contains reusable training and prediction components.
- `train.py` loads the CSV, preprocesses data, trains a Gradient Boosting model, evaluates performance, and saves the pipeline.
- The app loads the serialized model and serves predictions via a web form.
