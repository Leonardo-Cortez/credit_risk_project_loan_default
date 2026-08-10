import pandas as pd

def identify_risk_factors(client):
    """
    Identify the main risk factors associated with a borrower.
    """

    factors = []

    # Age
    if client["Age"] < 30:
        factors.append("- Young age")

    # Income
    if client["Income"] < 50000:
        factors.append("- Low income")

    # Loan amount
    if client["LoanAmount"] > 150000:
        factors.append("- High loan amount")

    # Credit score
    if client["CreditScore"] < 600:
        factors.append("- Low credit score")

    # Months employed
    if client["MonthsEmployed"] < 24:
        factors.append("- Short employment history")

    # Number of credit lines
    if client["NumCreditLines"] >= 4:
        factors.append("- Many credit lines")

    # Interest rate
    if client["InterestRate"] >= 15:
        factors.append("- High interest rate")

    # DTI
    if client["DTIRatio"] >= 0.50:
        factors.append("- High debt-to-income ratio")

    # Employment type
    if client["EmploymentType"] == "Unemployed":
        factors.append("- Unemployed")

    # Co-signer
    if client["HasCoSigner"] in ["No", 0]:
        factors.append("- No co-signer")

    return factors


def generate_recommendation(probability):
    """
    Generate a business-oriented recommendation based
    on the estimated probability of default.
    """

    if probability >= 0.70:

        return (
            "The applicant presents a HIGH credit risk. "
            "A detailed credit review is recommended before loan approval."
        )

    elif probability >= 0.40:

        return (
            "The applicant presents a MODERATE credit risk. "
            "Additional documentation or guarantees are recommended."
        )

    else:

        return (
            "The applicant presents a LOW credit risk according to the model."
        )


def generate_credit_report(client, pipeline, threshold=0.50):
    """
    Generate a structured credit risk report.

    Parameters
    ----------
    client : dict
        Borrower information.

    pipeline : sklearn Pipeline
        Trained credit risk prediction pipeline.

    threshold : float, default=0.50
        Classification threshold used to convert
        probability into predicted class.

    Returns
    -------
    dict
        Structured credit risk report.
    """

    client = client.copy()

    client_df = pd.DataFrame([client])

    probability = pipeline.predict_proba(client_df)[0, 1]

    prediction = int(probability >= threshold)

    if probability < 0.30:
        risk = "LOW"

    elif probability < 0.60:
        risk = "MEDIUM"

    else:
        risk = "HIGH"

    factors = identify_risk_factors(client)

    recommendation = generate_recommendation(probability)

    return {
        "client": client,
        "probability": probability,
        "prediction": prediction,
        "risk_level": risk,
        "risk_factors": factors,
        "recommendation": recommendation,
    }
