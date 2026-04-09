import streamlit as st
st.set_page_config(page_title="Credit AI", layout="wide")

# -------- CSS --------
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #0E1117, #1c1f26);
    font-family: 'Segoe UI', sans-serif;
}

h1 {
    font-size: 42px !important;
    font-weight: 800 !important;
    text-align: center;
    color: #00FFAA;
}

h2, h3 {
    color: #00FFAA;
    font-weight: 700;
}

.stButton>button {
    background-color: transparent;
    color: white;
    border: 2px solid rgba(255,255,255,0.3);
    border-radius: 14px;
    padding: 14px;
    font-size: 22px;
    font-weight: 900;
    width: 100%;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1c1f26, #0E1117);
}
</style>
""", unsafe_allow_html=True)

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt

# -------- LOAD DATA --------
df = pd.read_csv("german_credit_data.csv")

if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)

df = df.dropna()

df["Risk"] = df["Credit amount"].apply(lambda x: 0 if x > 5000 else 1)

# -------- ENCODE --------
le_dict = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

X = df.drop("Risk", axis=1)
y = df["Risk"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

# -------- SIDEBAR --------
st.sidebar.title("⚙️ Input Details")

age = st.sidebar.slider("Age", 18, 75, 30)
sex = st.sidebar.selectbox("Sex", le_dict["Sex"].classes_)
job = st.sidebar.selectbox("Job Level", [0,1,2,3])
housing = st.sidebar.selectbox("Housing", le_dict["Housing"].classes_)
saving = st.sidebar.selectbox("Saving Account", le_dict["Saving accounts"].classes_)
checking = st.sidebar.selectbox("Checking Account", le_dict["Checking account"].classes_)
purpose = st.sidebar.selectbox("Purpose", le_dict["Purpose"].classes_)

credit_amount = st.sidebar.number_input("Credit Amount", 0, 20000, 3000)
duration = st.sidebar.number_input("Duration", 1, 72, 12)

# -------- MAIN --------
st.title("🚀 AI Credit Risk Analyzer")

if st.button("🚀 PREDICT RISK"):

    # ✅ INPUT DICT (FIRST)
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
    prob = model.predict_proba(input_scaled)[0][1]

    # -------- RESULT --------
    if prediction[0] == 1:
        st.success(f"✅ Safe Customer ({prob*100:.2f}%)")
    else:
        st.error(f"❌ Risky Customer ({(1-prob)*100:.2f}%)")

    # -------- RISK LEVEL --------
    if prob > 0.8:
        st.info("Very Low Risk 🟢")
    elif prob > 0.5:
        st.warning("Moderate Risk 🟡")
    else:
        st.error("High Risk 🔴")

    # -------- SAVE HISTORY --------
    new_data = pd.DataFrame([input_dict])
    new_data["Result"] = prediction[0]
    new_data["Confidence"] = prob

    new_data.to_csv("history.csv", mode='a', header=False, index=False)

# -------- SHOW HISTORY --------
st.markdown("---")

if st.button("📜 Show History"):
    try:
        data = pd.read_csv("history.csv")
        st.dataframe(data)
    except:
        st.warning("No history yet!")

# -------- CHARTS --------
st.markdown("---")
st.header("📊 Data Insights")

col1, col2 = st.columns(2)

with col1:
    fig1 = plt.figure()
    plt.hist(df["Credit amount"], bins=20)
    st.pyplot(fig1)

with col2:
    fig2 = plt.figure()
    df["Risk"].value_counts().plot(kind="bar")
    st.pyplot(fig2)

fig3 = plt.figure()
plt.scatter(df["Age"], df["Credit amount"])
st.pyplot(fig3)

# -------- ABOUT --------
st.markdown("---")
st.markdown("""
### 📌 About Project
AI-based Credit Risk Prediction System  
Stores history + shows analytics  
""")
