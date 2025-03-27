# Transparent and Interpretable Credit Risk Assessment
![GitHub](https://img.shields.io/badge/Python-3.10.12-blue)
![GitHub](https://img.shields.io/badge/Libraries-scikit--learn%20%7C%20XGBoost%20%7C%20SHAP%20%7C%20LIME-orange)

## 📋 Project Overview
This repository contains code and models for a credit risk assessment system leveraging machine learning and Explainable AI (XAI) techniques. The project focuses on interpreting predictions for peer-to-peer (P2P) lending data from Lending Club (2013–2018) using SHAP and LIME.

## 📂 Data Processing Pipeline

### Data Sources
- **Raw Data**: `accepted_2007_to_2018q4.csv.zip` (Kaggle).
- **Processed Data**: Filtered to loans issued between 2013–2018 and refined to three classes: `Default`, `Fully Paid`, and `Charged Off`.
### Notebooks
| File | Purpose | Output |
|------|---------|--------|
| `1_Filter_data_2013_to_2018.ipynb` | Filters data by issue date | `accepted_2013_to_2018_filtered.csv` |
| `2_Filter_data_on_Loan_status.ipynb` | Refines loan status classes | `accepted_2013_to_2018_latest.csv` |
| `3_latest_filter_individual_loans_eda_and_cleaning.ipynb` | Performs EDA and cleaning | `FINAL_unbalancedData.csv` |
| `4_latest_Individual_loans_hypothesisTesting.ipynb` | Hypothesis testing and balancing | `FINAL_balancedData.csv` |

---
## 🤖 Model Training

### Feature Engineering & Encoding
- **Notebooks**: 
  - `Random_forest_model_feature_selection_and_encoding.ipynb`
  - `XGBoost_model_feature_selection_and_encoding.ipynb`
- **Output**: Final features stored in `x_data.csv` (predictors) and `y_data.csv` (target).

### Trained Models
| Model | Notebook | Saved File |
|-------|----------|------------|
| Logistic Regression | `Model_1_Logistic_regression.ipynb` | `Logistic_Regression.joblib` |
| Decision Tree | `Model_2_Decesion_Tree.ipynb` | `Decsion_tree_model.joblib` |
| Random Forest | `Model_3_Random_Forest.ipynb` | `Random_fores.joblib` |
| XGBoost | `Model_4_XGBoost.ipynb` | `XGBoost_model.joblib` |
| Neural Network (ANN) | `Model_5_ANN.ipynb` | `ANN_model.joblib` |

---
## 📊 Results & Explainability
1. **Performance Testing**: Run `test_Model_performances.ipynb` to evaluate models (uses `models_explainability.py`).
2. **XAI Interpretations**: 
   - `XAI_RandomForest.ipynb`: SHAP and LIME explanations for the best-performing Random Forest model.
   - **Note**: LIME results may vary due to its stochastic nature.

---

---

## 📜 Libraries & Versions
- Check `Library_Versions.ipynb` for Python and library dependencies (e.g., scikit-learn, SHAP, LIME).

---
