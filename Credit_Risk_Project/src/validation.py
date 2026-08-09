EXPECTED_COLUMNS = [
    "Age",
    "Income",
    "LoanAmount",
    "CreditScore",
    "MonthsEmployed",
    "NumCreditLines",
    "InterestRate",
    "LoanTerm",
    "DTIRatio",
    "Education",
    "EmploymentType",
    "MaritalStatus",
    "HasMortgage",
    "HasDependents",
    "LoanPurpose",
    "HasCoSigner",
]


VALID_CATEGORIES = {
    "Education": [
        "Bachelor's",
        "High School",
        "Master's",
        "PhD"
    ],

    "EmploymentType": [
        "Full-time",
        "Part-time",
        "Self-employed",
        "Unemployed"
    ],

    "MaritalStatus": [
        "Single",
        "Married",
        "Divorced"
    ],

    "LoanPurpose": [
        "Auto",
        "Business",
        "Education",
        "Home",
        "Other"
    ]
}


def validate_client(client):
    """
    Validate and normalize borrower information.

    Returns
    -------
    dict
        Validated borrower data with binary variables
        converted to 0/1.
    """

    client = client.copy()

    # Check missing variables
    missing = [
        col for col in EXPECTED_COLUMNS
        if col not in client
    ]

    if missing:
        raise ValueError(
            f"Missing variables: {missing}"
        )

    # Binary variables
    binary_mapping = {
        "Yes": 1,
        "No": 0,

        "yes": 1,
        "no": 0,

        "YES": 1,
        "NO": 0,

        True: 1,
        False: 0,

        1: 1,
        0: 0
    }

    for feature in [
        "HasMortgage",
        "HasDependents",
        "HasCoSigner"
    ]:

        value = client[feature]

        if value not in binary_mapping:
            raise ValueError(
                f"{feature} must be Yes/No or 1/0."
            )

        client[feature] = binary_mapping[value]

    # Categorical variables
    for feature, valid_values in VALID_CATEGORIES.items():

        if client[feature] not in valid_values:
            raise ValueError(
                f"{feature} must be one of: {valid_values}"
            )

    # Numeric variables
    numeric_variables = [
        "Age",
        "Income",
        "LoanAmount",
        "CreditScore",
        "MonthsEmployed",
        "NumCreditLines",
        "InterestRate",
        "LoanTerm",
        "DTIRatio"
    ]

    for feature in numeric_variables:

        try:
            client[feature] = float(client[feature])

        except (TypeError, ValueError):

            raise ValueError(
                f"{feature} must be numeric."
            )

    # Business rules

    if not (18 <= client["Age"] <= 100):
        raise ValueError(
            "Age must be between 18 and 100 years."
        )

    if client["Income"] <= 0:
        raise ValueError(
            "Income must be greater than 0."
        )

    if client["LoanAmount"] <= 0:
        raise ValueError(
            "LoanAmount must be greater than 0."
        )

    if not (300 <= client["CreditScore"] <= 850):
        raise ValueError(
            "CreditScore must be between 300 and 850."
        )

    if client["MonthsEmployed"] < 0:
        raise ValueError(
            "MonthsEmployed cannot be negative."
        )

    if client["NumCreditLines"] < 0:
        raise ValueError(
            "NumCreditLines cannot be negative."
        )

    if not (0 <= client["InterestRate"] <= 100):
        raise ValueError(
            "InterestRate must be between 0 and 100."
        )

    if client["LoanTerm"] <= 0:
        raise ValueError(
            "LoanTerm must be greater than 0."
        )

    if not (0 <= client["DTIRatio"] <= 1):
        raise ValueError(
            "DTIRatio must be between 0 and 1."
        )

    return client