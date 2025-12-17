# Credit Card Fraud Detection  
**An End-to-End Machine Learning System with Explainability and API Deployment**

---

## Project Overview

This project presents a complete, production-oriented **Credit Card Fraud Detection system** built using machine learning techniques on highly imbalanced transactional data.

The work demonstrates the **full data science lifecycle**, including exploratory data analysis, advanced modeling, class-imbalance handling, explainable AI (XAI), and deployment through a RESTful API.

This project is suitable for:

- Data Scientist / Machine Learning Engineer roles  
- Master’s programs in Data Science, Artificial Intelligence, or Analytics  

---

## Problem Statement

Credit card fraud represents a critical challenge due to:

- Extreme class imbalance (fraud cases < 0.2%)  
- High cost of false negatives (missed fraud)  
- Need for model interpretability in financial systems  

The objective is to **accurately detect fraudulent transactions** while maintaining transparency in model decisions.

---

## Dataset Description

- **Source:** European cardholders transaction dataset  
- **Total transactions:** ~284,000  
- **Fraudulent transactions:** ~0.17%  

### Features
- `V1–V28`: PCA-transformed numerical features  
- `Amount`: Transaction amount  
- `Class`: Target variable  
  - `0` → Legitimate  
  - `1` → Fraud  

No personally identifiable information is included in the dataset.

---

## Exploratory Data Analysis (EDA)

Key insights derived during EDA:

- Severe class imbalance confirmed  
- Fraudulent transactions show distinct patterns in certain PCA components (e.g., `V14`, `V12`, `V10`)  
- Transaction amount alone is not sufficient for fraud detection  

EDA results guided model selection and evaluation strategy.

---

## Data Preprocessing

- Feature scaling using **StandardScaler**  
- Stratified train-test split to preserve class distribution  
- **SMOTE (Synthetic Minority Over-sampling Technique)** applied only on training data  
- Care taken to avoid data leakage  

---

## Model Development and Evaluation

### Models Implemented
- Logistic Regression (baseline)  
- Random Forest Classifier  
- XGBoost Classifier  

### Evaluation Metrics
- ROC-AUC  
- Precision, Recall, F1-Score  
- Precision-Recall AUC (critical for imbalanced data)  

### Performance Summary

| Model               | ROC-AUC | Fraud Recall |
|--------------------|--------|--------------|
| Logistic Regression | ~0.97  | High (post-SMOTE) |
| Random Forest       | ~0.97  | ~82% |
| XGBoost             | ~0.98  | ~87% |

**XGBoost** provided the best balance between precision and recall, while **Random Forest** offered strong stability and interpretability.

---

## Explainable AI (SHAP)

Explainability was treated as a first-class requirement.

- SHAP feature importance plots identify key drivers of fraud detection  
- SHAP force plots explain individual fraud predictions  
- SHAP dependence plots analyze feature interactions  

This is critical for financial and regulatory environments.

---

## Model Inference Example

Example prediction for a high-risk transaction:

```json
{
  "fraud_probability": 0.99,
  "fraud_prediction": 1
}

## Reproducibility Notes

Due to size constraints, the full dataset and trained model are not included.

To reproduce results:
1. Download the dataset from Kaggle (Credit Card Fraud Detection)
2. Place it in the project root or data folder
3. Run the notebook: Notebooks/01_fraud_detection.ipynb



All preprocessing steps, training logic, and evaluation are fully reproducible.


