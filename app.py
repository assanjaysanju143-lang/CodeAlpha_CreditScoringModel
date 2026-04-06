import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
df = pd.read_csv("german_credit_data.csv")

if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)

df = df.dropna()

# Create target
df["Risk"] = df["Credit amount"].apply(lambda x: 0 if x > 5000 else 1)

# Encode categorical columns
le_dict = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le  # store encoder

# Split features
X = df.drop("Risk", axis=1)
y = df["Risk"]

# Scale + Train
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

# ---------------- UI ---------------- #

st.title("💳 Credit Risk Predictor")

st.write("Enter customer details:")

age = st.slider("Age", 18, 75, 30)

sex = st.selectbox("Sex", le_dict["Sex"].classes_)
job = st.selectbox("Job (0=unskilled, 3=highly skilled)", [0,1,2,3])
housing = st.selectbox("Housing", le_dict["Housing"].classes_)
saving = st.selectbox("Saving Account", le_dict["Saving accounts"].classes_)
checking = st.selectbox("Checking Account", le_dict["Checking account"].classes_)
purpose = st.selectbox("Purpose", le_dict["Purpose"].classes_)

credit_amount = st.number_input("Credit Amount", 0, 20000, 3000)
duration = st.number_input("Duration (months)", 1, 72, 12)

# Convert inputs
if st.button("Predict"):
    input_dict = {
        "Age": age,
        "Sex": le_dict["Sex"].transform([sex])[0],
        "Job": job,
        "Housing": le_dict["Housing"].transform([housing])[0],
        "Saving accounts": le_dict["Saving accounts"].transform([saving])[0],
        "Checking account": le_dict["Checking account"].transform([checking])[0],
        "Credit amount": credit_amount,
        "Duration": duration,
        "Purpose": le_dict["Purpose"].transform([purpose])[0]
    }

    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("✅ Safe Customer")
    else:
        st.error("❌ Risky Customer")