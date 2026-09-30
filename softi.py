import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

import streamlit as st

st.set_page_config(
    page_title="Job probablity by Anuj Kashyap",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0f0f0f, #1c1c1c, #252525);
}

/* Main heading */
h1 {
    color: #ffffff;
    text-align: center;
    font-size: 45px;
    font-weight: 700;
}

/* Normal text */
p {
    color: #d6d6d6;
}

/* Cards */
.card {
    background: rgba(255,255,255,0.06);
    padding: 25px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow: 0px 8px 25px rgba(0,0,0,0.35);
    margin-bottom: 20px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #6a5acd, #8a2be2);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 10px 25px;
    font-weight: bold;
}

.stButton > button:hover {
    transform: scale(1.03);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #151515;
}

</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="card">
    <h2>🤖 Only Education My ExcelR Friends</h2>
    <p>Predict, analyze and explore your dataset Anuj.</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Accuracy", "74%")

with col2:
    st.metric("F1 Score", "True : 79% And False : 66%")

with col3:
    st.metric("Models", "5")





# MAIN CODE
@st.cache_resource
def load_model_artifacts():
    model = joblib.load(BASE_DIR / "logistic_job.pkl")
    scaler = joblib.load(BASE_DIR / "scaler.pkl")
    expected_columns = list(joblib.load(BASE_DIR / "columns.pkl"))
    return model, scaler, expected_columns


model, scaler, expected_columns = load_model_artifacts()


def predict_probability(raw_input):
    input_df = pd.DataFrame([raw_input], columns=expected_columns)
    scaled_input = scaler.transform(input_df)
    probabilities = model.predict_proba(scaled_input)[0]
    positive_class_index = list(model.classes_).index(1)
    probability = float(probabilities[positive_class_index])
    prediction = int(model.predict(scaled_input)[0])
    return probability, prediction


st.title("Job Prediction By Anuj Kashyap")
st.markdown("Provide the following details to check your job probability:")

raw_input = {
    "cgpa": st.number_input("CGPA", 0.0, 10.0, 8.0, 0.1),
    "tenth_percentage": st.number_input("10th Percentage", 0, 100, 80, 1),
    "twelfth_percentage": st.number_input("12th Percentage", 0, 100, 85, 1),
    "backlogs": st.number_input("Backlogs", 0, 10, 0, 1),
    "projects_count": st.number_input("Projects Count", 0, 10, 2, 1),
    "internships_count": st.number_input("Internships Count", 0, 10, 1, 1),
    "certifications_count": st.number_input("Certifications Count", 0, 10, 1, 1),
    "coding_skill_1_10": st.slider("Coding Skill", 1, 10, 7),
    "communication_skill_1_10": st.slider("Communication Skill", 1, 10, 7),
    "aptitude_score": st.number_input("Aptitude Score", 0, 100, 60, 1),
    "dsa_problems_solved": st.number_input("DSA Problems Solved", 0, 500, 60, 1),
}

if st.button("Predict"):
    probability, prediction = predict_probability(raw_input)
    probability_percent = probability * 100

    if prediction == 1:
        st.success(f"✅ Job probability: {probability_percent:.2f}%")
    else:
        st.error(f"⚠️ Job probability (May be aapko padhna chaiye ): {probability_percent:.2f}%")