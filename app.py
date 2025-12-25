
import streamlit as st
import pickle
import numpy as np 
import pandas as pd 
import sklearn

# =================================================
# Load Model (USED BOTH INSIDE & OUTSIDE STREAMLIT)
# =================================================
def load_model(path="dectree_model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

model = load_model()

# =================================================
# Prediction Function (REUSABLE)
# =================================================
def predict_income(
    age,
    workclass,
    education,
    education_num,
    marital_status,
    occupation,
    relationship,
    race,
    sex,
    capital_gain,
    capital_loss,
    hours_per_week,
    native_country
):
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
    return prediction


# =================================================
# STREAMLIT APP
# =================================================
def run_streamlit_app():
    import streamlit as st

    st.set_page_config(
        page_title="Adult Income Prediction",
        page_icon="💼",
        layout="centered"
    )

    # ---------- STYLING ----------
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
        margin-bottom: 30px;
        color: white;
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

    st.markdown("""
    <div class="header-box">
        <h1>💼 Adult Income Prediction</h1>
        <p>Predict whether income exceeds ₹50,000 per month</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='center-card'>", unsafe_allow_html=True)

    # ---------- INPUTS ----------
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

    if st.button("Predict Income"):
        result = predict_income(
            age, workclass, education, education_num,
            marital_status, occupation, relationship,
            race, sex, capital_gain, capital_loss,
            hours_per_week, native_country
        )

        if result == 1:
            st.success("💰 Income > ₹50,000")
        else:
            st.warning("📉 Income ≤ ₹50,000")

    st.markdown("</div>", unsafe_allow_html=True)


# =================================================
# TERMINAL MODE (ALL INPUTS)
# =================================================
def run_terminal_mode():
    print("\n=== Adult Income Prediction (Terminal Mode) ===\n")

    age = int(input("Age: "))
    workclass = input("Workclass: ")
    education = input("Education: ")
    education_num = int(input("Education Number (1–16): "))
    marital_status = input("Marital Status: ")
    occupation = input("Occupation: ")
    relationship = input("Relationship: ")
    race = input("Race: ")
    sex = input("Sex (Male/Female): ")
    capital_gain = int(input("Capital Gain: "))
    capital_loss = int(input("Capital Loss: "))
    hours_per_week = int(input("Hours per Week: "))
    native_country = input("Native Country: ")

    result = predict_income(
        age, workclass, education, education_num,
        marital_status, occupation, relationship,
        race, sex, capital_gain, capital_loss,
        hours_per_week, native_country
    )

    print("\n=== Prediction Result ===")
    print("💰 Income > ₹50,000" if result == 1 else "📉 Income ≤ ₹50,000")


# =================================================
# MAIN
# =================================================
if __name__ == "__main__":
    try:
        import streamlit
        run_streamlit_app()
    except:
        run_terminal_mode()
