import streamlit as st
import pandas as pd

from pathlib import Path
from score_prediction.data import load_data
from score_prediction.model import load_model, get_default_model_path

st.set_page_config(page_title="Model Explainability", page_icon="🧠")

st.title("🔍 Model Explainability")
st.markdown(
    "This page explains how the student score model makes predictions and what features are most important."
)

MODEL_PATH = get_default_model_path(Path(__file__).resolve().parents[1])

@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)

model = get_model()

if "regressor_" not in dir(model.named_steps["regressor"]):
    st.warning(
        "The loaded model does not expose feature importances in the expected format."
    )
else:
    preprocessor = model.named_steps["preprocessor"]
    transformer_names = preprocessor.get_feature_names_out()
    importances = model.named_steps["regressor"].regressor_.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": transformer_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)

    st.subheader("Feature Importances")
    st.write(
        "These features are ranked by importance for the Gradient Boosting model predictions."
    )
    st.bar_chart(importance_df.set_index("feature")[:10])

    st.markdown("---")
    st.subheader("Top Features")
    top_text = "\n".join(
        [f"- **{row.feature}**: importance {row.importance:.3f}" for row in importance_df.head(8).itertuples()]
    )
    st.markdown(top_text)

st.markdown("---")
st.subheader("Model Metadata")
st.write(
    {
        "model_type": type(model.named_steps["regressor"].regressor_).__name__,
        "n_estimators": model.named_steps["regressor"].regressor_.n_estimators,
        "learning_rate": model.named_steps["regressor"].regressor_.learning_rate,
        "max_depth": model.named_steps["regressor"].regressor_.max_depth,
    }
)

st.markdown("---")
st.write("If you want, retrain the model by running `python train.py` and refresh this page.")
