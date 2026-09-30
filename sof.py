import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_model_artifacts():
    model = joblib.load(BASE_DIR / "navy_job.pkl")
    scaler = joblib.load(BASE_DIR / "scaler.pkl")
    columns = list(joblib.load(BASE_DIR / "columns.pkl"))
    return model, scaler, columns


model, scaler, columns = load_model_artifacts()

if len(columns) != model.n_features_in_:
    st.error(
        f"Feature mismatch: the saved model expects {model.n_features_in_} inputs, "
        f"but the saved column list has {len(columns)} values."
    )
    st.stop()

if list(scaler.feature_names_in_) != columns:
    st.warning("The scaler feature order does not match the saved column order. Using the saved column order for prediction.")

int_features = {
    "backlogs",
    "projects_count",
    "internships_count",
    "certifications_count",
    "dsa_problems_solved",
}


def predict_job_probability(values):
    row = pd.DataFrame([values], columns=columns)
    scaled = scaler.transform(row)
    probabilities = model.predict_proba(scaled)[0]

    if 1 in model.classes_:
        positive_index = list(model.classes_).index(1)
    else:
        positive_index = max(0, len(probabilities) - 1)

    probability = float(probabilities[positive_index])
    predicted_label = int(model.predict(scaled)[0])
    return probability, predicted_label


st.title("Job Placement Probability Predictor By Anuj Kashyap")
st.write("Enter student details to estimate the probability of getting placed in a job.")

user_inputs = {}
for column in columns:
    if column in int_features:
        user_inputs[column] = st.number_input(
            label=column.replace("_", " ").title(),
            min_value=0,
            value=0,
            step=1,
        )
    else:
        user_inputs[column] = st.number_input(
            label=column.replace("_", " ").title(),
            min_value=0.0,
            value=0.0,
            step=0.1,
        )

if st.button("Predict Job Probability"):
    probability, prediction = predict_job_probability(user_inputs)
    probability_percent = probability * 100

    st.metric("Job Probability", f"{probability_percent:.2f}%")

    if prediction == 1:
        st.success("Predicted: Job lag sakti hai.")
    else:
        st.warning("Predicted: Aise to job nhi milegi thoda or padhna padegs .")

    st.caption(
        "This probability is based on the trained Gaussian Naive Bayes model and the saved scaler."
    )
