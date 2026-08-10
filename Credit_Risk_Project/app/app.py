import streamlit as st
import joblib
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.prediction import assess_credit_risk
from src.reporting import generate_credit_report

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Credit Risk Prediction Framework",
    page_icon="💳",
    layout="wide"
)


# --------------------------------------------------
# Load model
# --------------------------------------------------

@st.cache_resource
def load_model():
    model_path = PROJECT_ROOT / "models" / "credit_default_pipeline.pkl"
    return joblib.load(model_path)


pipeline = load_model()


# --------------------------------------------------
# Application title
# --------------------------------------------------

st.title("Credit Risk Prediction Framework")

st.write(
    "Interactive credit risk assessment based on the "
    "existing Machine Learning framework."
)


# --------------------------------------------------
# Applicant information
# --------------------------------------------------

st.header("Applicant Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30,
        step=1
    )

with col2:
    education = st.selectbox(
        "Education",
        [
            "Bachelor's",
            "High School",
            "Master's",
            "PhD"
        ]
    )

with col3:
    marital_status = st.selectbox(
        "Marital Status",
        [
            "Single",
            "Married",
            "Divorced"
        ]
    )


# --------------------------------------------------
# Employment & Financial Profile
# --------------------------------------------------

st.header("Employment & Financial Profile")

col1, col2, col3 = st.columns(3)

with col1:
    income = st.number_input(
        "Income",
        min_value=0.01,
        value=40000.0,
        step=1000.0
    )

with col2:
    months_employed = st.number_input(
        "Months Employed",
        min_value=0.0,
        value=24.0,
        step=1.0
    )

with col3:
    employment_type = st.selectbox(
        "Employment Type",
        [
            "Full-time",
            "Part-time",
            "Self-employed",
            "Unemployed"
        ]
    )


# --------------------------------------------------
# Credit Profile
# --------------------------------------------------

st.header("Credit Profile")

col1, col2, col3 = st.columns(3)

with col1:
    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=400,
        step=1
    )

with col2:
    num_credit_lines = st.number_input(
        "Number of Credit Lines",
        min_value=0.0,
        value=4.0,
        step=1.0
    )

with col3:
    has_mortgage = st.selectbox(
        "Has Mortgage",
        ["No", "Yes"]
    )

col1, col2 = st.columns(2)

with col1:
    has_dependents = st.selectbox(
        "Has Dependents",
        ["No", "Yes"]
    )

with col2:
    has_co_signer = st.selectbox(
        "Has Co-Signer",
        ["No", "Yes"]
    )


# --------------------------------------------------
# Loan Information
# --------------------------------------------------

st.header("Loan Information")

col1, col2, col3 = st.columns(3)

with col1:
    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0.01,
        value=15000.0,
        step=1000.0
    )

with col2:
    interest_rate = st.number_input(
        "Interest Rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=14.0,
        step=0.1
    )

with col3:
    loan_term = st.number_input(
        "Loan Term (months)",
        min_value=0.01,
        value=48.0,
        step=1.0
    )

col1, col2 = st.columns(2)

with col1:
    dti_ratio = st.number_input(
        "Debt-to-Income Ratio",
        min_value=0.0,
        max_value=1.0,
        value=0.48,
        step=0.01
    )

with col2:
    loan_purpose = st.selectbox(
        "Loan Purpose",
        [
            "Auto",
            "Business",
            "Education",
            "Home",
            "Other"
        ]
    )


# --------------------------------------------------
# Assessment
# --------------------------------------------------

st.divider()

if st.button(
    "Assess Credit Risk",
    type="primary"
):

    applicant = {
        "Age": age,
        "Income": income,
        "LoanAmount": loan_amount,
        "CreditScore": credit_score,
        "MonthsEmployed": months_employed,
        "NumCreditLines": num_credit_lines,
        "InterestRate": interest_rate,
        "LoanTerm": loan_term,
        "DTIRatio": dti_ratio,
        "Education": education,
        "EmploymentType": employment_type,
        "MaritalStatus": marital_status,
        "HasMortgage": 1 if has_mortgage == "Yes" else 0,
        "HasDependents": 1 if has_dependents == "Yes" else 0,
        "LoanPurpose": loan_purpose,
        "HasCoSigner": 1 if has_co_signer == "Yes" else 0
    }

    try:

        result = assess_credit_risk(
            applicant,
            pipeline
        )

        probability = result["probability"]
        prediction = result["prediction"]
        risk_level = result["risk_level"]

        report = generate_credit_report(
            applicant,
            pipeline
        )

        st.subheader("Credit Risk Assessment")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Probability of Default",
                f"{probability:.2%}"
            )

        with col2:
            st.metric(
                "Predicted Class",
                prediction
            )

        with col3:
            st.metric(
                "Risk Level",
                risk_level
            )

        st.subheader("Main Risk Factors")

        if report["risk_factors"]:

            for factor in report["risk_factors"]:
                st.write(factor)

        else:

            st.success("No major risk factors detected.")

        st.subheader("Recommendation")

        st.info(report["recommendation"])

    except ValueError as error:

        st.error(str(error))
