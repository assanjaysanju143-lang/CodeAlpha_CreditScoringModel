from __future__ import annotations

import random

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from credit_model import get_feature_options, train_credit_model
from database import (
    create_user,
    get_prediction_history,
    get_user_by_contact,
    init_db,
    save_prediction,
    update_user_password,
    verify_user,
)

st.set_page_config(page_title="Credit AI", layout="wide")


@st.cache_resource
def load_model_bundle():
    model, df, metrics = train_credit_model()
    options = get_feature_options(df)
    return model, df, metrics, options


def apply_custom_style() -> None:
    st.markdown(
        """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700;800&display=swap');

    .stApp {
        background:
            radial-gradient(circle at 12% 18%, rgba(0, 255, 170, 0.14), transparent 20%),
            radial-gradient(circle at 88% 12%, rgba(0, 153, 255, 0.16), transparent 22%),
            linear-gradient(135deg, #06101d 0%, #0f172a 46%, #111827 100%);
        color: #e5eefb;
        font-family: 'Sora', sans-serif;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1323 0%, #0f172a 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.15);
        font-family: 'Sora', sans-serif;
    }

    .hero-card {
        padding: 1.6rem 1.8rem;
        border-radius: 22px;
        background: rgba(15, 23, 42, 0.82);
        border: 1px solid rgba(148, 163, 184, 0.18);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.25);
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 0.35rem;
    }

    .hero-subtitle {
        font-size: 1.02rem;
        color: #cbd5e1;
        margin: 0;
    }

    .auth-shell {
        max-width: 540px;
        margin: 1.6rem auto 1rem auto;
    }

    .auth-badge {
        display: inline-block;
        padding: 0.4rem 0.85rem;
        border-radius: 999px;
        background: rgba(0, 195, 137, 0.12);
        border: 1px solid rgba(0, 195, 137, 0.28);
        color: #9ff6d7;
        font-size: 0.82rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }

    .auth-title {
        font-size: 2.35rem;
        font-weight: 900;
        letter-spacing: -0.03em;
        color: #ffffff;
        margin-bottom: 0.35rem;
    }

    .auth-copy {
        color: #dbe7ff;
        font-size: 1rem;
        line-height: 1.6;
        margin-bottom: 1.1rem;
        font-weight: 600;
    }

    .google-chip {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.6rem;
        width: 100%;
        padding: 0.9rem 1rem;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.95);
        color: #0f172a;
        font-weight: 700;
        border: 1px solid rgba(255,255,255,0.5);
        margin: 0.4rem 0 0.8rem 0;
    }

    .auth-meta {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.75rem;
        margin: 1rem 0 1.2rem 0;
    }

    .auth-stat {
        padding: 0.8rem;
        border-radius: 18px;
        background: rgba(148, 163, 184, 0.08);
        border: 1px solid rgba(148, 163, 184, 0.14);
        text-align: center;
    }

    .auth-stat strong {
        display: block;
        color: #ffffff;
        font-size: 1rem;
        margin-bottom: 0.15rem;
        font-weight: 800;
    }

    .auth-stat span {
        color: #cbd5e1;
        font-size: 0.82rem;
        font-weight: 600;
    }

    .google-logo {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: conic-gradient(#4285F4 0 25%, #34A853 25% 50%, #FBBC05 50% 75%, #EA4335 75% 100%);
        color: white;
        font-size: 0.95rem;
        font-weight: 800;
    }

    .or-line {
        text-align: center;
        color: #dbe7ff;
        font-size: 0.88rem;
        margin: 0.45rem 0;
        font-weight: 700;
    }

    div[data-testid="stTabs"] {
        margin-top: 0.3rem;
    }

    div[data-testid="stTabs"] button {
        font-family: 'Sora', sans-serif;
        font-weight: 800;
        font-size: 1rem;
        color: #f8fafc;
    }

    div[data-testid="stTextInput"] label,
    div[data-testid="stNumberInput"] label,
    div[data-testid="stRadio"] label,
    div[data-testid="stExpander"] label {
        font-family: 'Sora', sans-serif;
        font-weight: 800;
        font-size: 0.98rem;
        color: #f8fafc;
        letter-spacing: 0.01em;
    }

    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input {
        font-family: 'Sora', sans-serif;
        font-weight: 700;
        color: #ffffff;
        background: rgba(8, 15, 30, 0.72);
        border: 1px solid rgba(56, 189, 248, 0.22);
        border-radius: 14px;
    }

    div[data-testid="stTextInput"] input::placeholder,
    div[data-testid="stNumberInput"] input::placeholder {
        color: #93a7c7;
        font-weight: 500;
    }

    p, label, span, div {
        font-family: 'Sora', sans-serif;
    }

    .stButton > button {
        width: 100%;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        background: linear-gradient(90deg, #1d4ed8, #38bdf8);
        color: #ffffff;
        text-shadow: 0 1px 2px rgba(8, 17, 31, 0.35);
        font-weight: 900;
        letter-spacing: 0.04em;
        font-size: 1rem;
        padding: 0.8rem 1rem;
        box-shadow: 0 16px 32px rgba(56, 189, 248, 0.24);
    }

    div[data-testid="stFormSubmitButton"] > button {
        width: 100%;
        border-radius: 16px;
        border: 1px solid rgba(167, 243, 208, 0.4);
        background: linear-gradient(90deg, #a7f3d0, #67e8f9);
        color: #04121f;
        font-weight: 900;
        letter-spacing: 0.05em;
        font-size: 1.02rem;
        padding: 0.85rem 1rem;
        box-shadow: 0 18px 36px rgba(103, 232, 249, 0.22);
        text-transform: uppercase;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 30px;
        border: 1px solid rgba(148, 163, 184, 0.22);
        background:
            linear-gradient(180deg, rgba(15, 23, 42, 0.96), rgba(15, 23, 42, 0.86)),
            radial-gradient(circle at top right, rgba(56, 189, 248, 0.18), transparent 26%);
        box-shadow: 0 32px 90px rgba(0, 0, 0, 0.38);
    }
</style>
""",
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    st.session_state.setdefault("logged_in", False)
    st.session_state.setdefault("user_id", None)
    st.session_state.setdefault("username", "")
    st.session_state.setdefault("otp_code", None)
    st.session_state.setdefault("otp_user", None)
    st.session_state.setdefault("otp_purpose", "")


def send_demo_otp(identifier: str, contact_type: str, purpose: str):
    user = get_user_by_contact(identifier, contact_type)
    if not user:
        return None

    otp_code = f"{random.randint(100000, 999999)}"
    st.session_state.otp_code = otp_code
    st.session_state.otp_user = user
    st.session_state.otp_purpose = purpose
    return otp_code


def clear_otp_state() -> None:
    st.session_state.otp_code = None
    st.session_state.otp_user = None
    st.session_state.otp_purpose = ""


def complete_login(user: dict) -> None:
    st.session_state.logged_in = True
    st.session_state.user_id = user["id"]
    st.session_state.username = user["username"]
    clear_otp_state()
    st.success("Login Successful")
    st.rerun()


def render_login_page() -> None:
    left, center, right = st.columns([1, 1.4, 1])

    with center:
        with st.container(border=True):
            st.markdown(
                """
<div class="auth-shell">
    <div class="auth-badge">Secure Access</div>
    <div class="auth-title">Welcome Back</div>
    <div class="auth-copy">
        Login With Username Or Email, Reset Password Using Email Or Phone, Or Sign In Directly With OTP.
    </div>
    <div class="google-chip">
        <span class="google-logo">G</span>
        <span>Secure Account Recovery And OTP Access</span>
    </div>
    <div class="auth-meta">
        <div class="auth-stat">
            <strong>Email</strong>
            <span>Fast Login</span>
        </div>
        <div class="auth-stat">
            <strong>Phone</strong>
            <span>OTP Verify</span>
        </div>
        <div class="auth-stat">
            <strong>Secure</strong>
            <span>Reset Access</span>
        </div>
    </div>
    <div class="or-line">Or</div>
    <div class="or-line">Use Your Account Details</div>
</div>
""",
                unsafe_allow_html=True,
            )

            login_tab, register_tab = st.tabs(["Login", "Register"])

            with login_tab:
                mode_tab1, mode_tab2 = st.tabs(["Password Login", "Login With OTP"])

                with mode_tab1:
                    with st.form("login_form"):
                        identifier = st.text_input("Username Or Email")
                        phone = st.text_input("Phone Number (Optional)")
                        password = st.text_input("Password", type="password")
                        submitted = st.form_submit_button("Login")

                    if submitted:
                        user = verify_user(identifier, password, phone)
                        if user:
                            complete_login(user)
                        else:
                            st.error("Invalid Login Details")

                    with st.expander("Forgot Password?"):
                        reset_type = st.radio(
                            "Choose Reset Method",
                            ["Email", "Phone Number"],
                            horizontal=True,
                            key="reset_type",
                        )
                        reset_identifier = st.text_input(
                            "Enter Your Email Or Phone",
                            key="reset_identifier",
                        )

                        if st.button("Send Reset OTP"):
                            otp = send_demo_otp(
                                reset_identifier,
                                "email" if reset_type == "Email" else "phone",
                                "reset_password",
                            )
                            if otp:
                                st.success("OTP Generated Successfully")
                                st.info(
                                    f"Demo OTP For {reset_type}: {otp}. "
                                    "Real Email Or SMS Delivery Needs A Service Like Gmail SMTP Or Twilio."
                                )
                            else:
                                st.error("No Account Found With Those Details")

                        with st.form("reset_password_form"):
                            entered_otp = st.text_input("Enter OTP")
                            new_password = st.text_input("New Password", type="password")
                            reset_submit = st.form_submit_button("Reset Password")

                        if reset_submit:
                            if (
                                entered_otp
                                and entered_otp == st.session_state.otp_code
                                and st.session_state.otp_purpose == "reset_password"
                                and st.session_state.otp_user
                                and new_password
                            ):
                                update_user_password(st.session_state.otp_user["id"], new_password)
                                st.success("Password Reset Successful")
                                clear_otp_state()
                            else:
                                st.error("Invalid OTP Or Reset Session")

                with mode_tab2:
                    otp_login_type = st.radio(
                        "Verify Using",
                        ["Email", "Phone Number"],
                        horizontal=True,
                        key="otp_login_type",
                    )
                    otp_login_identifier = st.text_input(
                        "Enter Email Or Phone Number",
                        key="otp_login_identifier",
                    )

                    if st.button("Send Login OTP"):
                        otp = send_demo_otp(
                            otp_login_identifier,
                            "email" if otp_login_type == "Email" else "phone",
                            "login_otp",
                        )
                        if otp:
                            st.success("OTP Generated Successfully")
                            st.info(
                                f"Demo OTP For {otp_login_type}: {otp}. "
                                "Real Email Or SMS Delivery Needs A Service Like Gmail SMTP Or Twilio."
                            )
                        else:
                            st.error("No Account Found With Those Details")

                    with st.form("login_otp_form"):
                        entered_login_otp = st.text_input("Enter OTP")
                        otp_login_submit = st.form_submit_button("Login With OTP")

                    if otp_login_submit:
                        if (
                            entered_login_otp
                            and entered_login_otp == st.session_state.otp_code
                            and st.session_state.otp_purpose == "login_otp"
                            and st.session_state.otp_user
                        ):
                            complete_login(st.session_state.otp_user)
                        else:
                            st.error("Invalid OTP Or Login Session")

            with register_tab:
                with st.form("register_form"):
                    new_username = st.text_input("Create Username")
                    new_email = st.text_input("Email Address")
                    new_phone = st.text_input("Phone Number")
                    new_password = st.text_input("Create Password", type="password")
                    register_submitted = st.form_submit_button("Create Account")

                if register_submitted:
                    if not new_username or not new_email or not new_phone or not new_password:
                        st.warning("Fill All Registration Fields")
                    elif create_user(new_username, new_email, new_phone, new_password):
                        st.success("Account Created. You Can Login Now.")
                    else:
                        st.error("Username, Email, Or Phone Number Already Exists")


def render_app() -> None:
    model, df, metrics, options = load_model_bundle()

    st.sidebar.title("Applicant Details")
    st.sidebar.caption(f"Logged In As {st.session_state.username.title()}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.user_id = None
        st.session_state.username = ""
        clear_otp_state()
        st.rerun()

    st.markdown(
        """
<div class="hero-card">
    <div class="hero-title">Credit Risk Analyzer</div>
    <p class="hero-subtitle">
        Predict Customer Credit Safety, Review Model Quality, And Store Decisions In A Database.
    </p>
</div>
""",
        unsafe_allow_html=True,
    )

    age = st.sidebar.slider("Age", 18, 75, 30)
    sex = st.sidebar.selectbox("Sex", options["Sex"])
    job = st.sidebar.selectbox("Job Level", [0, 1, 2, 3], index=1)
    housing = st.sidebar.selectbox("Housing", options["Housing"])
    saving = st.sidebar.selectbox("Saving Account", options["Saving accounts"])
    checking = st.sidebar.selectbox("Checking Account", options["Checking account"])
    purpose = st.sidebar.selectbox("Purpose", options["Purpose"])
    credit_amount = st.sidebar.number_input("Credit Amount", min_value=0, max_value=20000, value=3000)
    duration = st.sidebar.number_input("Duration (Months)", min_value=1, max_value=72, value=12)

    input_row = {
        "Age": age,
        "Sex": sex,
        "Job": job,
        "Housing": housing,
        "Saving accounts": saving,
        "Checking account": checking,
        "Credit amount": credit_amount,
        "Duration": duration,
        "Purpose": purpose,
    }

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Training Rows", metrics["train_shape"][0])
    metric_col2.metric("Test Rows", metrics["test_shape"][0])
    metric_col3.metric("ROC-AUC", f'{metrics["roc_auc"]:.3f}')

    if st.button("Predict Risk"):
        input_df = pd.DataFrame([input_row])
        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])

        result_label = "Safe Customer" if prediction == 1 else "Risky Customer"
        confidence = probability if prediction == 1 else 1 - probability
        risk_band = (
            "Very Low Risk"
            if probability >= 0.8
            else "Moderate Risk"
            if probability >= 0.5
            else "High Risk"
        )

        if prediction == 1:
            st.success(f"{result_label} ({confidence * 100:.2f}% Confidence)")
        else:
            st.error(f"{result_label} ({confidence * 100:.2f}% Confidence)")

        if probability >= 0.8:
            st.info("Risk Band: Very Low Risk")
        elif probability >= 0.5:
            st.warning("Risk Band: Moderate Risk")
        else:
            st.error("Risk Band: High Risk")

        st.markdown("### Prediction Summary")
        summary_col1, summary_col2 = st.columns(2)
        with summary_col1:
            st.write(f"Age: {age}")
            st.write(f"Credit Amount: {credit_amount}")
            st.write(f"Duration: {duration} Months")
        with summary_col2:
            st.write(f"Result: {result_label}")
            st.write(f"Confidence: {confidence * 100:.2f}%")
            st.write(f"Purpose: {purpose}")

        save_prediction(
            user_id=st.session_state.user_id,
            age=age,
            sex=sex,
            job=job,
            housing=housing,
            saving_accounts=saving,
            checking_account=checking,
            credit_amount=credit_amount,
            duration=duration,
            purpose=purpose,
            result=result_label,
            confidence=round(confidence, 4),
            risk_band=risk_band,
        )

    st.markdown("---")
    st.subheader("Prediction History")
    history_df = get_prediction_history(st.session_state.user_id)

    if history_df.empty:
        st.caption("No Prediction History Yet.")
    else:
        st.dataframe(history_df, width="stretch")
        csv_data = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download History CSV",
            data=csv_data,
            file_name="credit_prediction_history.csv",
            mime="text/csv",
        )

    st.markdown("---")
    st.subheader("Data Insights")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig1, ax1 = plt.subplots()
        ax1.hist(df["Credit amount"], bins=20, color="#00c389", edgecolor="#08111f")
        ax1.set_title("Credit Amount Distribution")
        ax1.set_xlabel("Credit Amount")
        ax1.set_ylabel("Count")
        st.pyplot(fig1)

    with chart_col2:
        fig2, ax2 = plt.subplots()
        df["Risk"].value_counts().sort_index().plot(
            kind="bar",
            ax=ax2,
            color=["#ef4444", "#10b981"],
        )
        ax2.set_title("Risk Class Balance")
        ax2.set_xlabel("Risk Label")
        ax2.set_ylabel("Count")
        st.pyplot(fig2)

    fig3, ax3 = plt.subplots()
    ax3.scatter(df["Age"], df["Credit amount"], alpha=0.7, color="#38bdf8")
    ax3.set_title("Age Vs Credit Amount")
    ax3.set_xlabel("Age")
    ax3.set_ylabel("Credit Amount")
    st.pyplot(fig3)

    with st.expander("Model Notes"):
        st.write(
            "This Demo Uses A Logistic Regression Pipeline With Imputation, Scaling, And "
            "One-Hot Encoding. The Dataset Does Not Contain An Official Target Label, "
            "So The Current Project Derives A Simple Demo Risk Flag From Credit Amount."
        )
        st.text(metrics["classification_report"])
        st.write("Confusion Matrix:")
        st.dataframe(pd.DataFrame(metrics["confusion_matrix"]))


def main() -> None:
    init_db()
    init_session_state()
    apply_custom_style()

    if st.session_state.logged_in:
        render_app()
    else:
        render_login_page()


if __name__ == "__main__":
    main()
