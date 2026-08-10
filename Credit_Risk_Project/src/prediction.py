import pandas as pd
from src.validation import validate_client

def predict_default_risk(client_data, pipeline, threshold=0.50):
    """
    Predict default probability, class, and risk level
    for a new borrower using the existing ML pipeline.

    Parameters
    ----------
    client_data : dict
        Applicant information containing the 16 predictive variables.

    pipeline : sklearn.pipeline.Pipeline
        Trained credit default prediction pipeline.

    threshold : float, default=0.50
        Probability threshold used to determine predicted class.

    Returns
    -------
    dict
        Prediction results containing probability, predicted class,
        and risk level.
    """

    client_df = pd.DataFrame([client_data])

    probability = pipeline.predict_proba(client_df)[0, 1]

    prediction = int(probability >= threshold)

    if probability < 0.30:
        risk_level = "LOW"
    elif probability < 0.60:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    return {
        "probability": probability,
        "prediction": prediction,
        "risk_level": risk_level,
    }


def assess_credit_risk(client_data, pipeline, threshold=0.50):
    """
    Validate borrower information and assess credit risk.

    Parameters
    ----------
    client_data : dict
        Applicant information.

    pipeline : sklearn.pipeline.Pipeline
        Trained credit default prediction pipeline.

    threshold : float, default=0.50
        Probability threshold for predicted class.

    Returns
    -------
    dict
        Validated prediction results.
    """

    validated_client = validate_client(client_data)

    result = predict_default_risk(
        validated_client,
        pipeline,
        threshold=threshold
    )

    return result
