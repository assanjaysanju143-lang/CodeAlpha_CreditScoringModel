# # import streamlit as st
# # st.set_page_config(page_title="Credit AI", layout="wide")
# # st.markdown("""
# #     <style>
# #     .main {
# #         background-color: #0E1117;
# #     }
# #     h1, h2, h3 {
# #         color: #00FFAA;
# #     }
# #     .stButton>button {
# #         background-color: #00FFAA;
# #         color: black;
# #         border-radius: 10px;
# #         height: 3em;
# #         width: 100%;
# #         font-size: 16px;
# #     }
# #     .stSidebar {
# #         background-color: #262730;
# #     }
# #     </style>
# # """, unsafe_allow_html=True)  
# # import numpy as np
# # import pandas as pd
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.preprocessing import LabelEncoder, StandardScaler

# # # Load dataset
# # df = pd.read_csv("german_credit_data.csv")

# # if "Unnamed: 0" in df.columns:
# #     df.drop("Unnamed: 0", axis=1, inplace=True)

# # df = df.dropna()

# # # Create target
# # df["Risk"] = df["Credit amount"].apply(lambda x: 0 if x > 5000 else 1)

# # # Encode categorical columns
# # le_dict = {}
# # for col in df.select_dtypes(include='object').columns:
# #     le = LabelEncoder()
# #     df[col] = le.fit_transform(df[col])
# #     le_dict[col] = le  # store encoder

# # # Split features
# # X = df.drop("Risk", axis=1)
# # y = df["Risk"]

# # # Scale + Train
# # scaler = StandardScaler()
# # X_scaled = scaler.fit_transform(X)

# # model = LogisticRegression()
# # model.fit(X_scaled, y)

# # # ---------------- UI ---------------- #

# # # st.title("💳 Credit Risk Predictor")

# # # st.write("Enter customer details:")

# # # age = st.slider("Age", 18, 75, 30)

# # # sex = st.selectbox("Sex", le_dict["Sex"].classes_)
# # # job = st.selectbox("Job (0=unskilled, 3=highly skilled)", [0,1,2,3])
# # # housing = st.selectbox("Housing", le_dict["Housing"].classes_)
# # # saving = st.selectbox("Saving Account", le_dict["Saving accounts"].classes_)
# # # checking = st.selectbox("Checking Account", le_dict["Checking account"].classes_)
# # # purpose = st.selectbox("Purpose", le_dict["Purpose"].classes_)

# # # credit_amount = st.number_input("Credit Amount", 0, 20000, 3000)
# # # duration = st.number_input("Duration (months)", 1, 72, 12)

# # # # Convert inputs
# # # if st.button("Predict"):
# # #     input_dict = {
# # #         "Age": age,
# # #         "Sex": le_dict["Sex"].transform([sex])[0],
# # #         "Job": job,
# # #         "Housing": le_dict["Housing"].transform([housing])[0],
# # #         "Saving accounts": le_dict["Saving accounts"].transform([saving])[0],
# # #         "Checking account": le_dict["Checking account"].transform([checking])[0],
# # #         "Credit amount": credit_amount,
# # #         "Duration": duration,
# # #         "Purpose": le_dict["Purpose"].transform([purpose])[0]
# # #     }

# # #     input_df = pd.DataFrame([input_dict])
# # #     input_scaled = scaler.transform(input_df)

# # #     prediction = model.predict(input_scaled)
# # #     prob = model.predict_proba(input_scaled)[0][1]

# # #     if prediction[0] == 1:
# # #         st.success(f"✅ Safe Customer (Confidence: {prob*100:.2f}%)")
# # #     else:
# # #         st.error(f"❌ Risky Customer (Confidence: {(1-prob)*100:.2f}%)")

# # # -------- SIDEBAR --------
# # st.sidebar.title("⚙️ Input Details")

# # age = st.sidebar.slider("Age", 18, 75, 30)

# # sex = st.sidebar.selectbox("Sex", le_dict["Sex"].classes_)
# # job = st.sidebar.selectbox("Job Level (0-3)", [0,1,2,3])
# # housing = st.sidebar.selectbox("Housing", le_dict["Housing"].classes_)
# # saving = st.sidebar.selectbox("Saving Account", le_dict["Saving accounts"].classes_)
# # checking = st.sidebar.selectbox("Checking Account", le_dict["Checking account"].classes_)
# # purpose = st.sidebar.selectbox("Purpose", le_dict["Purpose"].classes_)

# # credit_amount = st.sidebar.number_input("Credit Amount", 0, 20000, 3000)
# # duration = st.sidebar.number_input("Duration (months)", 1, 72, 12)

# # # -------- MAIN PAGE --------
# # st.title("💳 Credit Risk Prediction System")
# # st.markdown("### Enter details from sidebar and click predict")

# # if st.button("🚀 Predict Risk"):
# #     input_dict = {
# #         "Age": age,
# #         "Sex": le_dict["Sex"].transform([sex])[0],
# #         "Job": job,
# #         "Housing": le_dict["Housing"].transform([housing])[0],
# #         "Saving accounts": le_dict["Saving accounts"].transform([saving])[0],
# #         "Checking account": le_dict["Checking account"].transform([checking])[0],
# #         "Credit amount": credit_amount,
# #         "Duration": duration,
# #         "Purpose": le_dict["Purpose"].transform([purpose])[0]
# #     }

# #     input_df = pd.DataFrame([input_dict])
# #     input_scaled = scaler.transform(input_df)

# #     prediction = model.predict(input_scaled)
# #     prob = model.predict_proba(input_scaled)[0][1]

# #     st.subheader("📊 Prediction Result")

# #     if prediction[0] == 1:
# #         st.success(f"✅ Safe Customer (Confidence: {prob*100:.2f}%)")
# #     else:
# #         st.error(f"❌ Risky Customer (Confidence: {(1-prob)*100:.2f}%)")

    
# #     # ---------------------MATPLOTLIB------------

# # import matplotlib.pyplot as plt

# # st.markdown("---")
# # st.header("📊 Data Insights Dashboard")

# # # 🔹 Top Row (2 Charts)
# # col1, col2 = st.columns(2)

# # with col1:
# #     st.markdown("### 📈 Credit Distribution")
# #     fig1 = plt.figure(figsize=(6,4))
# #     plt.hist(df["Credit amount"], bins=20)
# #     plt.xlabel("Credit Amount")
# #     plt.ylabel("Count")
# #     st.pyplot(fig1)

# # with col2:
# #     st.markdown("### 📊 Risk Distribution")
# #     fig2 = plt.figure(figsize=(6,4))
# #     df["Risk"].value_counts().plot(kind="bar")
# #     plt.xlabel("Risk (1=Safe, 0=Risky)")
# #     plt.ylabel("Count")
# #     st.pyplot(fig2)

# # # 🔹 Bottom Full Width Chart
# # st.markdown("### 🔍 Age vs Credit Analysis")

# # fig3 = plt.figure(figsize=(10,4))
# # plt.scatter(df["Age"], df["Credit amount"])
# # plt.xlabel("Age")
# # plt.ylabel("Credit Amount")

# # st.pyplot(fig3)


# import streamlit as st
# st.set_page_config(page_title="Credit AI", layout="wide")

# # 🔥 CRAZY UI CSS
# # st.markdown("""
# # <style>

# # /* 🌑 Background */
# # .main {
# #     background: linear-gradient(135deg, #0E1117, #1c1f26);
# # }

# # /* 🟢 Headings */
# # h1, h2, h3 {
# #     color: #00FFAA;
# #     text-shadow: 0px 0px 10px rgba(0,255,170,0.5);
# # }

# # /* 🎯 Buttons */
# # .stButton>button {
# #     background: linear-gradient(90deg, #00FFAA, #00c896);
# #     color: black;
# #     border-radius: 12px;
# #     height: 3em;
# #     width: 100%;
# #     font-size: 16px;
# #     transition: 0.3s;
# # }
# # .stButton>button:hover {
# #     transform: scale(1.05);
# #     box-shadow: 0px 0px 15px rgba(0,255,170,0.7);
# # }

# # /* 📦 Card Style */
# # .card {
# #     padding: 20px;
# #     border-radius: 15px;
# #     background: rgba(255,255,255,0.05);
# #     box-shadow: 0 0 15px rgba(0,255,170,0.2);
# #     margin-bottom: 20px;
# # }

# # /* Sidebar */
# # section[data-testid="stSidebar"] {
# #     background: linear-gradient(180deg, #1c1f26, #0E1117);
# # }

# # </style>
# # """, unsafe_allow_html=True)
# st.markdown("""
# <style>

# /* 🌑 Background */
# .main {
#     background: linear-gradient(135deg, #0E1117, #1c1f26);
#     font-family: 'Segoe UI', sans-serif;
# }

# /* 🔥 BIG MAIN TITLE */

# h1 {
#     font-size: 50px !important;
#     font-weight: 900 !important;
#     text-transform: capitalize;
#     color: #00FFAA;
#     text-align: center;
#     # text-shadow: 0px 0px 20px rgba(0,255,170,0.9);
# }


# /* 🟢 Sub Headings */
# h2, h3 {
#     font-weight: 700 !important;
#     text-transform: capitalize;
#     color: #00FFAA;
# }

# /* 🎯 Buttons */

            
# /* 🔥 BUTTON STYLE */
# .stButton>button {
#     background-color: transparent;
#     color: white;
#     border: 2px solid rgba(255,255,255,0.3);
#     border-radius: 14px;
#     padding: 14px 28px;
#     font-size: 35px;
#     font-weight: 800;  /* 🔥 BOLD */
#     letter-spacing: 1px;
#     font-family: 'Segoe UI', sans-serif;
#     width: 100%;
#     transition: all 0.3s ease;
# }

# /* ✨ HOVER EFFECT */
# .stButton>button:hover {
#     border: 2px solid ;
#     # color: #4facfe;
#     # box-shadow: 0px 0px 12px rgba(79,172,254,0.6);
#     transform: scale(1.03);
# }

# /* 📦 Card Style */
# .card {
#     padding: 25px;
#     border-radius: 15px;
#     background: rgba(255,255,255,0.05);
#     box-shadow: 0 0 20px rgba(0,255,170,0.3);
#     margin-bottom: 20px;
#     font-weight: 500;
# }

# /* Sidebar */
# section[data-testid="stSidebar"] {
#     background: linear-gradient(180deg, #1c1f26, #0E1117);
#     font-weight: bold;
# }

# </style>
# """, unsafe_allow_html=True)

# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# import matplotlib.pyplot as plt

# # Load dataset
# df = pd.read_csv("german_credit_data.csv")

# if "Unnamed: 0" in df.columns:
#     df.drop("Unnamed: 0", axis=1, inplace=True)

# df = df.dropna()

# # Create target
# df["Risk"] = df["Credit amount"].apply(lambda x: 0 if x > 5000 else 1)

# # Encode
# le_dict = {}
# for col in df.select_dtypes(include='object').columns:
#     le = LabelEncoder()
#     df[col] = le.fit_transform(df[col])
#     le_dict[col] = le

# X = df.drop("Risk", axis=1)
# y = df["Risk"]

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# model = LogisticRegression()
# model.fit(X_scaled, y)

# # -------- SIDEBAR --------
# st.sidebar.title("⚙️ Input Details")

# age = st.sidebar.slider("Age", 18, 75, 30)
# sex = st.sidebar.selectbox("Sex", le_dict["Sex"].classes_)
# job = st.sidebar.selectbox("Job Level (0-3)", [0,1,2,3])
# housing = st.sidebar.selectbox("Housing", le_dict["Housing"].classes_)
# saving = st.sidebar.selectbox("Saving Account", le_dict["Saving accounts"].classes_)
# checking = st.sidebar.selectbox("Checking Account", le_dict["Checking account"].classes_)
# purpose = st.sidebar.selectbox("Purpose", le_dict["Purpose"].classes_)

# credit_amount = st.sidebar.number_input("Credit Amount", 0, 20000, 3000)
# duration = st.sidebar.number_input("Duration (months)", 1, 72, 12)

# # -------- MAIN --------
# st.title("🚀 AI Credit Risk Analyzer")

# st.markdown('<div class="card">', unsafe_allow_html=True)
# st.markdown("### Enter details from sidebar and click predict")

# if st.button("🚀 Predict Risk"):
#     new_data = pd.DataFrame([input_dict])
#     new_data["Result"] = prediction[0]

#     new_data.to_csv("history.csv", mode='a', header=False, index=False)
#     input_dict = {
#         "Age": age,
#         "Sex": le_dict["Sex"].transform([sex])[0],
#         "Job": job,
#         "Housing": le_dict["Housing"].transform([housing])[0],
#         "Saving accounts": le_dict["Saving accounts"].transform([saving])[0],
#         "Checking account": le_dict["Checking account"].transform([checking])[0],
#         "Credit amount": credit_amount,
#         "Duration": duration,
#         "Purpose": le_dict["Purpose"].transform([purpose])[0]
#     }

#     input_df = pd.DataFrame([input_dict])
#     input_scaled = scaler.transform(input_df)

#     prediction = model.predict(input_scaled)
#     prob = model.predict_proba(input_scaled)[0][1]

#     if prediction[0] == 1:
#         st.markdown(f"""
#         <div class="card">
#             <h2>✅ Safe Customer</h2>
#             <p>Confidence: {prob*100:.2f}%</p>
#         </div>
#         """, unsafe_allow_html=True)
#     else:
#         st.markdown(f"""
#         <div class="card">
#             <h2 style="color:#FF4B4B;">❌ Risky Customer</h2>
#             <p>Confidence: {(1-prob)*100:.2f}%</p>
#         </div>
#         """, unsafe_allow_html=True)

# st.markdown('</div>', unsafe_allow_html=True)

# # -------- CHARTS --------
# st.markdown("---")
# st.header("📊 Data Insights Dashboard")

# col1, col2 = st.columns(2)

# with col1:
#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown("### 📈 Credit Distribution")
#     fig1 = plt.figure(figsize=(6,4))
#     plt.hist(df["Credit amount"], bins=20)
#     st.pyplot(fig1)
#     st.markdown('</div>', unsafe_allow_html=True)

# with col2:
#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown("### 📊 Risk Distribution")
#     fig2 = plt.figure(figsize=(6,4))
#     df["Risk"].value_counts().plot(kind="bar")
#     st.pyplot(fig2)
#     st.markdown('</div>', unsafe_allow_html=True)

# st.markdown('<div class="card">', unsafe_allow_html=True)
# st.markdown("### 🔍 Age vs Credit Analysis")
# fig3 = plt.figure(figsize=(10,4))
# plt.scatter(df["Age"], df["Credit amount"])
# st.pyplot(fig3)
# st.markdown('</div>', unsafe_allow_html=True)

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