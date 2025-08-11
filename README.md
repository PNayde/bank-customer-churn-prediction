# Bank Customer Churn Prediction

## Executive Summary
This project develops a predictive model to identify bank customers at high risk of churn, enabling targeted retention strategies.  
Using a 70/30 train-test split, multiple models were compared, with **XGBoost tuned using Optuna** achieving the best performance:

- **Accuracy:** 80.5%
- **Recall (Churn):** 75.3%
- **ROC-AUC:** 0.868

Key insight: Older, inactive customers with fewer products are most at risk.  
Explainable AI (SHAP) was applied to provide actionable business recommendations.

---

## Project Overview
The analysis supports the bank in understanding and predicting customer churn through data-driven insights and machine learning.  
This work was completed as part of a take-home data science assignment.

---

## Objectives
- Build and evaluate machine learning models for churn prediction.
- Apply hyperparameter optimization to improve predictive performance.
- Use explainability methods to provide business-relevant insights.

---

## Dataset
The dataset is available from Kaggle (https://www.kaggle.com/datasets/shubhammeshram579/bank-customer-churn-prediction/data) and contains demographic, account, and behavioural features.  
**Target variable:** `Churn` (1 = churned, 0 = retained).

---

## Methodology

### Data Preparation
- Exploratory Data Analysis (EDA) for feature understanding.
- Addressed class imbalance with `scale_pos_weight`.
- Train-test split: 70% training, 30% testing.

### Models Evaluated
1. Logistic Regression (baseline and tuned)
2. XGBoost Classifier (tuned with Optuna)

### Hyperparameter Optimization
- Used **Optuna** to tune parameters including:
  - `n_estimators`
  - `max_depth`
  - `learning_rate`
  - `subsample`
  - `colsample_bytree`

### Metrics
- Accuracy
- Precision, Recall, F1-score (focus on churn class)
- ROC-AUC
- Precision-Recall Curve

### Explainability
- SHAP analysis for feature impact and business interpretation.

---

## Results

| Model | Accuracy | Recall (Churn) | Precision (Churn) | F1 (Churn) | ROC-AUC |
|-------|----------|----------------|-------------------|------------|---------|
| Logistic Regression (Tuned) | 0.711 | 0.689 | 0.384 | 0.493 | 0.770 |
| XGBoost (Tuned) | 0.805 | 0.753 | 0.515 | 0.611 | 0.868 |

**Best model:** XGBoost tuned with Optuna.

---

## SHAP Insights
- **Age:** Higher churn risk for older customers.
- **Number of Products:** Customers with fewer products are more likely to churn.
- **Activity Status:** Inactive accounts show higher churn probability.
- **Balance/Country:** Moderate influence compared to top features.

**Recommendation:** Prioritise retention programs for older, inactive customers with few products.

---

## Technologies Used
- Python
- pandas, numpy
- scikit-learn
- XGBoost
- Optuna
- SHAP

---

## Author

Dr. Plamena Naydenova, PhD, BSc

Email: plamena.naydenova@gmail.com

