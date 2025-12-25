
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import sklearn

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Adult Income Prediction",
    page_icon="💼",
    layout="centered"
)

# -------------------------------------------------
# Background + Dark Theme Styling
# -------------------------------------------------
st.markdown("""
<style>

/* Full background image */
[data-testid="stAppViewContainer"] {
    background-image: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSsgrgtdLTIkzL5IILx8cpt-3uXyJS7C9kXew&s");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}

/* Remove default background */
[data-testid="stMainContent"] {
    background: transparent;
}

/* Header black box */
.header-box {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    padding: 30px 25px;
    border-radius: 24px;
    box-shadow: 0px 20px 45px rgba(0,0,0,0.6);
    text-align: center;
    margin-bottom: 30px;
}

/* Header title */
.header-title {
    font-size: 40px;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 6px;
}

/* Header subtitle */
.header-subtitle {
    font-size: 16px;
    color: #d1d5db;
}

/* Center dark card */
.center-card {
    background: rgba(20, 20, 20, 0.92);
    padding: 35px;
    border-radius: 24px;
    box-shadow: 0px 25px 50px rgba(0,0,0,0.7);
    max-width: 720px;
    margin: auto;
}

/* Labels */
label {
    color: #e5e7eb !important;
}

/* Inputs */
input, select, textarea {
    background-color: #1f2937 !important;
    color: white !important;
    border-radius: 10px !important;
    border: 1px solid #374151 !important;
}

/* Button */
button {
    width: 100%;
    height: 50px;
    font-size: 18px;
    border-radius: 12px;
    background-color: #2563eb !important;
    color: white !important;
    border: none;
}

button:hover {
    background-color: #1d4ed8 !important;
}

/* Result box */
.result-box {
    margin-top: 25px;
    padding: 22px;
    border-radius: 14px;
    background-color: #111827;
    text-align: center;
    font-size: 18px;
    color: #f9fafb;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Load Model
# -------------------------------------------------
with open("dectree_model.pkl", "rb") as f:
    model = pickle.load(f)

# -------------------------------------------------
# Header Box (BLACK)
# -------------------------------------------------
st.markdown("""
<div class="header-box">
    <div class="header-title">💼 Adult Income Prediction</div>
    <div class="header-subtitle">
        Predict whether income exceeds 50,000 rupees per month
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Center Card Start
# -------------------------------------------------
st.markdown("<div class='center-card'>", unsafe_allow_html=True)

# -------------------------------------------------
# Inputs
# -------------------------------------------------
age = st.number_input("Age", 18, 90, 30)

workclass = st.selectbox("Workclass", [
    "Private", "Self-emp-not-inc", "Self-emp-inc",
    "Federal-gov", "Local-gov", "State-gov", "Without-pay"
])

education = st.selectbox("Education", [
    "Bachelors", "HS-grad", "Masters", "Assoc-acdm",
    "Assoc-voc", "Some-college", "Doctorate", "Prof-school"
])

education_num = st.number_input("Education Number", 1, 16, 10)

marital_status = st.selectbox("Marital Status", [
    "Never-married", "Married-civ-spouse", "Divorced",
    "Separated", "Widowed", "Married-spouse-absent"
])

occupation = st.selectbox("Occupation", [
    "Tech-support", "Craft-repair", "Other-service",
    "Sales", "Exec-managerial", "Prof-specialty",
    "Handlers-cleaners", "Machine-op-inspct",
    "Adm-clerical", "Farming-fishing", "Transport-moving"
])

relationship = st.selectbox("Relationship", [
    "Husband", "Not-in-family", "Own-child",
    "Unmarried", "Wife", "Other-relative"
])

race = st.selectbox("Race", [
    "White", "Black", "Asian-Pac-Islander",
    "Amer-Indian-Eskimo", "Other"
])

sex = st.selectbox("Sex", ["Male", "Female"])

capital_gain = st.number_input("Capital Gain", 0, 100000, 0)
capital_loss = st.number_input("Capital Loss", 0, 100000, 0)
hours_per_week = st.number_input("Hours Per Week", 1, 99, 40)

native_country = st.selectbox("Native Country", [
    "United-States", "India", "Mexico", "Philippines",
    "Germany", "Canada", "England", "China", "Japan"
])

# -------------------------------------------------
# Prediction
# -------------------------------------------------
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

    prediction = model.predict(input_df)[0]

    st.markdown("<div class='result-box'>", unsafe_allow_html=True)

    if prediction == 1:
        st.write("💰 **Income Level:** More than 50,000 rupees")
    else:
        st.write("📉 **Income Level:** 50,000 rupees or less")

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# Center Card End
# -------------------------------------------------
st.markdown("</div>", unsafe_allow_html=True)
