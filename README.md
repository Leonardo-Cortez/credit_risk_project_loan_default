# Credit Risk Prediction Framework
An end-to-end Machine Learning solution for loan default prediction.

To access the notebooks directly and run them, click the button: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1a086kV0i9U3ieQ--zC-KeyqOihUXmIQs#scrollTo=j8dYmRW4p-v7)

## Project Overview
Credit risk assessment is one of the most important applications of Machine Learning in the financial industry. Financial institutions must evaluate the probability that a borrower will default on a loan before making lending decisions. An inaccurate assessment may lead to significant financial losses, while overly restrictive lending policies can reduce business opportunities.

This project presents an end-to-end Credit Risk Prediction Framework developed using supervised Machine Learning techniques. The complete workflow covers exploratory data analysis, feature engineering, data preprocessing, model development, model evaluation, threshold optimization, model interpretation and an inference pipeline capable of evaluating new loan applicants.

Several classification algorithms were compared to identify the most appropriate model according to business objectives. Instead of maximizing only overall accuracy, special attention was given to Recall, Precision, ROC-AUC and decision threshold optimization, since correctly identifying high-risk borrowers is considerably more valuable than maximizing the number of correctly classified low-risk customers.

Finally, the selected model was integrated into a prediction pipeline capable of estimating the probability of default for new applicants while automatically generating an interpretable credit risk report.

## Business Problem
Lending institutions face a fundamental challenge: approving as many profitable loans as possible while minimizing financial losses caused by borrower defaults.

Traditional credit assessment methods often rely on manual evaluations or fixed business rules, which may overlook complex relationships among applicant characteristics. As a result, financial institutions may approve high-risk applicants or reject reliable customers.

From a business perspective, the cost of these mistakes is not the same:

* **False Positives:**: Rejecting a reliable customer results in lost business opportunities and reduced revenue.
* **False Negatives:**: Approving a customer who later defaults can generate significant financial losses and increase the institution's credit risk exposure.

Therefore, the objective is not simply to maximize overall prediction accuracy. Instead, the model should prioritize identifying high-risk applicants while maintaining a reasonable balance between risk detection and business profitability.

This project addresses this challenge by developing a Machine Learning framework capable of estimating the probability of loan default before approval, providing decision support for credit analysts through interpretable predictions and risk-based recommendations.

## Project Objectives
The main objective of this project is to develop an end-to-end Machine Learning framework capable of predicting the probability that a loan applicant will default.

Specific objectives include:

* Perform Exploratory Data Analysis (EDA) to understand customer characteristics and identify potential risk patterns.
* Build a data preprocessing pipeline capable of handling numerical, categorical and binary variables.
* Compare multiple supervised Machine Learning algorithms for credit risk classification.
* Evaluate model performance using business-oriented metrics such as Precision, Recall, F1-Score and ROC-AUC.
* Optimize the decision threshold according to business requirements instead of relying solely on the default classification threshold.
* Interpret the selected model to identify the variables that most influence credit risk.
* Develop an inference pipeline capable of evaluating new applicants and automatically generating a credit risk assessment report.
* Demonstrate how Machine Learning can support credit approval decisions in financial institutions.

## Dataset

The project uses a public [**Loan Default Prediction**](https://www.kaggle.com/datasets/nikhil1e9/loan-default/data) dataset containing information about loan applicants and their repayment behavior.

### Dataset Summary

| Feature            |                 Value |
| ------------------ | --------------------: |
| Total observations |               255,347 |
| Input variables    |                    16 |
| Target variable    |               Default |
| Problem type       | Binary Classification |

The dataset includes applicant demographic information, financial indicators, employment history and loan characteristics, including:

* Applicant age
* Annual income
* Loan amount
* Credit score
* Debt-to-Income ratio (DTI)
* Employment history
* Number of credit lines
* Interest rate
* Loan purpose
* Education level
* Employment type
* Marital status
* Mortgage status
* Dependents
* Co-signer information

The target variable indicates whether the customer ultimately defaulted on the loan.

* **0** → Loan Paid
* **1** → Loan Default

