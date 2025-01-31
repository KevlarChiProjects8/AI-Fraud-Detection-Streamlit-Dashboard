# 🕵️ AI Fraud Detection Dashboard


A Streamlit-powered dashboard for detecting fraudulent transactions using **XGBoost** and **CNN** models, featuring advanced analytics, statistical tests, and interactive visualizations.

---

## 📌 Overview

This dashboard provides a comprehensive suite of tools to analyze credit card fraud detection models. It includes:
- **Model performance metrics** (ROC-AUC, Precision-Recall)
- **Interpretability** (SHAP values, Feature Importance)
- **Statistical testing** (T-Tests, Chi-Square, A/B Testing)
- **Interactive visualizations** (Heatmaps, Distribution Plots, ROC Curves)

**Key Achievements**:
- XGBoost model achieves **95.5% ROC-AUC**.
- CNN model achieves **93.6% ROC-AUC**.
- Improved fraud detection accuracy through ensemble insights and explainable AI.

---

## 🚀 Features

### 📊 **Dashboard Pages**
1. **Feature Importance**: Identify critical predictors of fraud.
2. **Confusion Matrix**: Visualize model performance (Precision, Recall, F1-Score).
3. **CNN vs. XGBoost Predictions**: Compare deep learning and gradient-boosting results.
4. **ROC Curve & Precision-Recall Curve**: Evaluate classification thresholds.
5. **Statistical Tests**:
   - T-Test: Fraud vs. Non-Fraud Transaction Amounts.
   - Chi-Square: Fraud Rate by Transaction Type.
   - A/B Testing: CNN vs. XGBoost Performance.
6. **Data Visualizations**: Box Plots, Violin Plots, Scatter Plots, Correlation Heatmaps.
7. **SHAP Values**: Explain model predictions at instance and global levels.

### 🎨 **Aesthetic Design**
- Custom CSS styling with **Google Orbitron font** and gradient animations.
- Themed metric boxes, buttons, and sidebar for enhanced UX.

---

## ⚙️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/fraud-detection-dashboard.git
   cd fraud-detection-dashboard

2. **Replace the X_train, X_test, y_train, and df in App.py with your own data, by dragging the .csv files into your Vscode folder directory**
3. **Install Dependencies**:
   pip install -r requirements.txt (Vscode terminal)
4. **Run the Streamlit App**:
    streamlit run app.py

  
