# -loan-approval-predictor

# Advanced Loan Approval Prediction System Using Machine Learning

A machine learning-based system to predict loan approvals, designed to enhance decision-making in financial institutions by improving accuracy, transparency, and fairness. Built as part of a data science graduate project at DePaul University.

---

## Overview

This project aims to automate and optimize the loan approval process using advanced ML techniques. We trained and compared multiple classification models on a structured dataset of 20,000 loan applications to predict loan approval decisions.

The final model (Gradient Boosting) achieved **99.2% accuracy** and **99.9% ROC-AUC**, delivering robust performance while maintaining high interpretability and generalizability.

---

## Tech Stack

- **Languages:** Python  
- **ML Libraries:** Scikit-learn, Pandas, NumPy, XGBoost, CatBoost  
- **Visualization:** Matplotlib, Seaborn  
- **Model Evaluation:** ROC-AUC, F1 Score, Precision, Recall, Confusion Matrices  
- **Tools:** Jupyter Notebook, Git, StandardScaler, OneHotEncoder

---

---

## Methodology

1. **Data Preprocessing**
   - Cleaned and normalized data (StandardScaler)
   - One-hot encoded categorical features
   - Engineered financial features: `DebtToIncomeRatio`, `RiskScore`, etc.
   - Removed multicollinearity (NetWorth, Experience)

2. **Models Evaluated**
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
   - SVC (Support Vector Classifier)
   - K-Nearest Neighbors

3. **Evaluation Metrics**
   - Accuracy, Precision, Recall, F1 Score, ROC-AUC
   - 5-Fold Cross-Validation for model generalization

---

## Results

| Model               | Accuracy | F1 Score | ROC-AUC |
|--------------------|----------|----------|---------|
| Gradient Boosting  | 99.20%   | 98.32%   | 99.92%  |
| Logistic Regression| 98.80%   | 97.48%   | 99.87%  |
| Random Forest      | 99.10%   | 98.11%   | 99.86%  |

- Gradient Boosting was selected as the final model.
- Feature importance: `DebtToIncomeRatio`, `RiskScore`, `TotalDebtToIncomeRatio`

---

## Visualizations

- Correlation Heatmap  
- Confusion Matrices  
- ROC Curves  
- Feature Importance Charts  

---
