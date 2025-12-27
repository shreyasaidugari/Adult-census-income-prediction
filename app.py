import streamlit as st
import pickle
import pandas as pd
import numpy as np

# =================================================
# Load model
# =================================================
@st.cache_resource
def load_model():
    with open("dectree_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()


# =================================================
# Prediction function
# =================================================
def predict_income(input_df):
    prediction = model.predict(input_df)[0]
    return prediction

# =================================================
# Streamlit App
# =================================================

st.set_page_config(
    page_title="Adult Income Prediction",
    page_icon="💼",
    layout="centered"
)

# ---------- Styling ----------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSsgrgtdLTIkzL5IILx8cpt-3uXyJS7C9kXew&s");
    background-size: cover;
    background-position: center;
}

.header-box {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    padding: 30px;
    border-radius: 24px;
    text-align: center;
    color: white;
    margin-bottom: 30px;
}

.center-card {
    background: rgba(20, 20, 20, 0.92);
    padding: 35px;
    border-radius: 24px;
    max-width: 720px;
    margin: auto;
    color: white;
}

label { color: white !important; }
input, select {
    background-color: #1f2937 !important;
    color: white !important;
}

button {
    width: 100%;
    height: 50px;
    font-size: 18px;
    background-color: #2563eb !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("""
<div class="header-box">
    <h1>💼 Adult Income Prediction</h1>
    <p>Predict whether income exceeds ₹50,000 per month</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='center-card'>", unsafe_allow_html=True)

# ---------- Inputs ----------
age = st.number_input("Age", 17, 90, 30)

workclass = st.selectbox("Workclass", [
    "Private", "Self-emp-not-inc", "Self-emp-inc",
    "Federal-gov", "Local-gov", "State-gov",
    "Without-pay", "Never-worked"
])

education = st.selectbox("Education", [
    "Preschool", "1st-4th", "5th-6th", "7th-8th", "9th",
    "10th", "11th", "12th", "HS-grad", "Some-college",
    "Assoc-voc", "Assoc-acdm", "Bachelors",
    "Masters", "Prof-school", "Doctorate"
])

# Map education to education number
education_map = {
    "Preschool": 1, "1st-4th": 2, "5th-6th": 3,
    "7th-8th": 4, "9th": 5, "10th": 6,
    "11th": 7, "12th": 8, "HS-grad": 9,
    "Some-college": 10, "Assoc-voc": 11,
    "Assoc-acdm": 12, "Bachelors": 13,
    "Masters": 14, "Prof-school": 15,
    "Doctorate": 16
}
education_num = education_map[education]

marital_status = st.selectbox("Marital Status", [
    "Married-civ-spouse", "Divorced", "Never-married",
    "Separated", "Widowed", "Married-spouse-absent",
    "Married-AF-spouse"
])

occupation = st.selectbox("Occupation", [
    "Tech-support", "Craft-repair", "Other-service", "Sales",
    "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
    "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
    "Transport-moving", "Priv-house-serv",
    "Protective-serv", "Armed-Forces"
])

relationship = st.selectbox("Relationship", [
    "Wife", "Own-child", "Husband",
    "Not-in-family", "Other-relative", "Unmarried"
])

race = st.selectbox("Race", [
    "White", "Asian-Pac-Islander",
    "Amer-Indian-Eskimo", "Other", "Black"
])

sex = st.selectbox("Sex", ["Male", "Female"])
capital_gain = st.number_input("Capital Gain", 0, 99999, 0)
capital_loss = st.number_input("Capital Loss", 0, 4356, 0)
hours_per_week = st.number_input("Hours per Week", 1, 99, 40)

native_country = st.selectbox("Native Country", [
    "United-States", "India", "Mexico", "Philippines", "Germany",
    "Canada", "China", "Japan", "England", "Italy", "France"
])

# ---------- Prediction ----------
if st.button("Predict Income"):
    input_df = pd.DataFrame([[ 
        age, workclass, education, education_num, marital_status,
        occupation, relationship, race, sex,
        capital_gain, capital_loss, hours_per_week, native_country
    ]], columns=[
        "age", "workclass", "education", "education.num",
        "marital.status", "occupation", "relationship",
        "race", "sex", "capital.gain", "capital.loss",
        "hours.per.week", "native.country"
    ])
    
    result = predict_income(input_df)
    
    if result == 1:
        st.success("💰 Income > ₹50,000")
    else:
        st.warning("📉 Income ≤ ₹50,000")

st.markdown("</div>", unsafe_allow_html=True)
