# 🫀 Heart Disease Prediction — End-to-End Machine Learning

![Python](https://img.shields.io/badge/Python-3.12-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-success)

**🌐 [Try the Live App](https://rishika-heart-predictor.streamlit.app)**

Complete ML pipeline for heart disease prediction using the Cleveland Heart Disease Dataset. Includes EDA, model comparison across 5 algorithms, hyperparameter tuning, SHAP explainability, and a live Streamlit web app.

## 🎯 Highlights

- **5 Models Compared**: Logistic Regression, Random Forest, XGBoost, SVM, MLP
- **Hyperparameter Tuning**: GridSearchCV with 5-fold cross-validation
- **Class Imbalance Handling**: SMOTE boosted Recall from 41% to 65%
- **Best Result**: 86.89% Accuracy, 94.12% ROC-AUC (Random Forest + SMOTE)
- **Explainable**: SHAP analysis for global and per-patient explanations
- **Deployed**: Interactive Streamlit web application

## 📊 Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 85.25% | 90.00% | 52.94% | 66.67% | 92.51% |
| Random Forest | 81.97% | 87.50% | 41.18% | 56.00% | 94.39% |
| XGBoost | 81.97% | 80.00% | 47.06% | 59.26% | 91.18% |
| SVM | 85.25% | 90.00% | 52.94% | 66.67% | 90.24% |
| **RF + SMOTE** ⭐ | **86.89%** | **84.62%** | **64.71%** | **73.33%** | **94.12%** |

## 🚀 Getting Started

```bash
git clone https://github.com/rishikaevenjalin-bit/heart-disease-prediction-ml.git
cd heart-disease-prediction-ml
pip install -r requirements.txt

# Run notebook
jupyter notebook notebooks/heart_disease_full_pipeline.ipynb

# Run web app locally
streamlit run app/streamlit_app.py
```

## 🔬 Methodology

1. **Preprocessing**: Categorical encoding, stratified 80/20 split, StandardScaler
2. **Training**: 4 traditional ML + 1 neural network with 5-fold cross-validation
3. **Tuning**: GridSearchCV on ROC-AUC metric
4. **Balancing**: SMOTE oversampling to address class imbalance
5. **Explainability**: SHAP values for interpretability

## 📈 Key Insights

- **Class imbalance handling matters most** — SMOTE improved Recall by 23%
- **Simple models compete well** — Logistic Regression matched complex ones
- **Top predictors** (SHAP-verified): `oldpeak`, `ca`, `thal`

## ⚠️ Limitations

- Small dataset (303 patients) limits generalizability
- Educational tool only — NOT for clinical diagnosis

## 🛠️ Tech Stack

Python 3.12 · scikit-learn · XGBoost · SHAP · Streamlit · pandas · matplotlib

## 👩‍💻 Author

**Rishika Evenjalin** — Final-year BSc Data Science, ESOFT (London Met)
[GitHub](https://github.com/rishikaevenjalin-bit) 

## 📄 License

MIT
