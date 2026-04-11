from pathlib import Path

import pandas as pd
import streamlit as st

from score_prediction.predict import load_trained_model, prepare_input, predict_score

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "saved_models" / "student_score_model.pkl"

st.set_page_config(
    page_title="Student Math Score Predictor",
    page_icon="📚",
    layout="wide",
)

st.title("📊 Student Math Score Predictor")
st.markdown(
    "This app predicts student math scores based on demographics, lunch status, and test preparation."
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found. Run `python train.py` to create {MODEL_PATH.name}."
        )
    return load_trained_model(MODEL_PATH)


try:
    model = load_model()
    st.success("✅ Model loaded successfully")
except Exception as error:
    st.error(f"Error loading model: {error}")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Student Demographics")
    gender = st.selectbox("Gender", ["male", "female"], help="Select student's gender")
    race_ethnicity = st.selectbox(
        "Race/Ethnicity",
        ["group A", "group B", "group C", "group D", "group E"],
        help="Select student's race/ethnicity",
    )
    parental_education = st.selectbox(
        "Parental Level of Education",
        [
            "some high school",
            "high school",
            "some college",
            "associate's degree",
            "bachelor's degree",
            "master's degree",
        ],
        help="Select parental education level",
    )

with col2:
    st.subheader("Test Scores & Preparation")
    lunch = st.selectbox(
        "Lunch", ["standard", "free/reduced"], help="Select lunch type"
    )
    test_prep = st.selectbox(
        "Test Preparation Course", ["none", "completed"], help="Select test prep status"
    )
    reading_score = st.slider("Reading Score", 0, 100, 50, help="Student reading score")
    writing_score = st.slider("Writing Score", 0, 100, 50, help="Student writing score")

if st.button("🎯 Predict Math Score", use_container_width=True):
    input_df = prepare_input(
        gender=gender,
        race_ethnicity=race_ethnicity,
        parental_education=parental_education,
        lunch=lunch,
        test_prep=test_prep,
        reading_score=reading_score,
        writing_score=writing_score,
    )

    try:
        prediction = predict_score(model, input_df)
        performance = (
            "Excellent"
            if prediction >= 80
            else "Good"
            if prediction >= 70
            else "Average"
            if prediction >= 60
            else "Needs Improvement"
        )
        confidence = "Low" if 60 <= prediction <= 90 else "Moderate"

        st.markdown("---")
        st.subheader("📈 Prediction Results")
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted Math Score", f"{prediction:.1f}")
        c2.metric("Performance Level", performance)
        c3.metric("Prediction Confidence", confidence)

        st.info(
            f"""
**Score Interpretation:**
- **90-100**: Excellent
- **80-89**: Good
- **70-79**: Average
- **Below 70**: Needs Improvement

**Predicted score:** {prediction:.1f}/100
"""
        )
    except Exception as error:
        st.error(f"Prediction failed: {error}")

st.markdown("---")
st.markdown(
    """
<div style='text-align:center; color:gray; font-size:12px;'>
<p>Model: Gradient Boosting Regressor | Trained with TransformedTargetRegressor</p>
<p>Run `python train.py` to retrain the model.</p>
</div>
""",
    unsafe_allow_html=True,
)
