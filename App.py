import streamlit as st
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

import joblib
import shap 
import lime
import lime.lime_tabular

from transformers import pipeline 

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    average_precision_score,
)
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance 

import tensorflow as tf
import keras
from tensorflow.keras import layers, models

# T-Test: Fraud vs. Non-Fraud Transaction Amounts - is the amount statistically significant?
from scipy.stats import ttest_ind
# Chi-Square Test: Fraud Rate by Transaction Type - analyzes whether the fraud rate differs significantly across transaction types
from scipy.stats import chi2_contingency 
# A/B Testing: compares the performance (e.g., precision) of the CNN and XGBoost models to determine if one is significantly better
from scipy.stats import ttest_rel 

# Load the dataset for statistics
df = pd.read_csv('fraud_detection_dataset.csv', nrows=500000) 

# Clean the data 
df = df.dropna()

# Make df the 2.9 - 3rd million row, since the Google Colab CNN and XGBoost models were trained on the first million rows
df['count_nameOrig'] = df.groupby('nameOrig')['nameOrig'].transform('count')
df['count_nameDest'] = df.groupby('nameDest')['nameDest'].transform('count')

# Drop nameDest and nameOrig, since we have df['count_nameOrig'], df['count_nameDest']
df = df.drop(['nameDest', 'nameOrig'], axis=1)

# Load in X_train & X_test
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')

# Load in y_test
y_test = pd.read_csv('y_test.csv')


# CNN Model
cnn_model = tf.keras.models.load_model('cnn_fraud_model.keras')
xgb_model = joblib.load('xgboost_model.pkl')

# Streamlit Styling
st.markdown("""
    <style>
    /* Apply a futuristic font and global styling */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Orbitron', sans-serif;
        text-transform: uppercase;
        letter-spacing: 1px;
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: white;
    }

    /* Sidebar Gradient Background with Animation */
    section[data-testid="stSidebar"] {
        background: linear-gradient(135deg, #fc466b, #3f5efb);
        color: white;
        animation: gradientAnimation 10s ease infinite;
    }

    @keyframes gradientAnimation {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Header Gradient Background with Glow Effect */
    h1 {
        background: linear-gradient(135deg, #ff00cc, #333399);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0px 4px 20px rgba(255, 0, 204, 0.5);
        animation: glow 2s infinite alternate;
    }

    h2 {
        background: linear-gradient(135deg, #0cebeb, #20e3b2, #29ffc6);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0px 4px 20px rgba(32, 227, 178, 0.5);
        animation: glow 2s infinite alternate;
    }

    h3 {
        background: linear-gradient(135deg, #ff4b1f, #ff9068);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0px 4px 20px rgba(255, 75, 31, 0.5);
        animation: glow 2s infinite alternate;
    }

    @keyframes glow {
        0% { box-shadow: 0px 4px 20px rgba(255, 0, 204, 0.5); }
        100% { box-shadow: 0px 4px 30px rgba(255, 0, 204, 0.8); }
    }

    /* Metric Boxes with Neon Glow */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #8e2de2, #4a00e0);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0px 4px 20px rgba(138, 43, 226, 0.5);
        animation: glow 2s infinite alternate;
    }

    /* Buttons Styling with Hover Animation */
    div.stButton > button {
        background: linear-gradient(135deg, #f7971e, #ffd200);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-size: 16px;
        transition: background 0.3s, transform 0.2s, box-shadow 0.2s;
        animation: glow 2s infinite alternate;
    }

    div.stButton > button:hover {
        background: linear-gradient(135deg, #ffd200, #f7971e);
        transform: scale(1.07);
        box-shadow: 0px 4px 25px rgba(255, 210, 0, 0.6);
    }

    /* Radio Button Styling with Neon Effect */
    div.stRadio label {
        background: linear-gradient(135deg, #ff6a00, #ee0979);
        color: white;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
        display: block;
        text-align: center;
        box-shadow: 0px 4px 20px rgba(255, 106, 0, 0.5);
        animation: glow 2s infinite alternate;
    }

    /* Add a futuristic border to the main content */
    .main .block-container {
        border: 2px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        background: rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }

    /* Add a subtle animation to the sidebar */
    section[data-testid="stSidebar"] {
        animation: slideIn 1s ease-out;
    }

    @keyframes slideIn {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(0); }
    }
    </style>
""", unsafe_allow_html=True)


# Streamlit App
st.title("AI Fraud Detection Dashboard")
st.sidebar.header("Navigation")

page = st.sidebar.radio("Choose a Page", [
    "Feature Importance", "Confusion Matrix", "CNN Predictions", "ROC Curve", "Precision-Recall Curve",
    "Distribution Plots", "Correlation Heatmap", "Box Plots", "Violin Plots",
    "Scatter Plot", "SHAP Values", "T-Test (Fraud vs Non. Fraud Transaction Amounts)", 
    "Chi-Square Test: Fraud Rate by Transaction Type", "A/B Testing: Comparing CNN vs. XGBoost Fraud Detection",
    "Recall T-Test: CNN vs. XGBoost", 
])

# Feature Importance Plot
if page == "Feature Importance":
    st.header("Feature Importance Plot")
    fig, ax = plt.subplots()
    if isinstance(xgb_model, XGBClassifier):
        xgb.plot_importance(xgb_model, ax=ax)
    st.pyplot(fig) 

# Confusion Matrix
elif page == "Confusion Matrix":
    st.header("Confusion Matrix")
    y_pred = xgb_model.predict(X_test)

    # Calculate precision, recall, ROC-AUC, and F1-score
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    # Display Metrics in Colored Boxes
    st.metric("Precision Score:", f"{precision:.4f}")   
    st.metric("Recall Score:", f"{recall:.4f}")
    st.metric("F1 Score:", f"{f1:.4f}")
    st.metric("ROC-AUC Score:", f"{roc_auc:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# CNN Predictions
elif page == "CNN Predictions":
    st.header("CNN Model Predictions")

    # Get predicted probabilities for the entire test set
    y_pred_proba = cnn_model.predict(X_test)

    # Convert probabilities to class labels using a threshold (e.g., 0.5)
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Calculate accuracy metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)  # Use probabilities for ROC-AUC

    # Display Metrics in Colored Boxes
    st.metric("Precision Score:", f"{precision:.4f}")   
    st.metric("Recall Score:", f"{recall:.4f}")
    st.metric("F1 Score:", f"{f1:.4f}")
    st.metric("ROC-AUC Score:", f"{roc_auc:.4f}")

    # Confusion Matrix
    cnn_cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cnn_cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

# ROC Curve
elif page == "ROC Curve":
    st.header("ROC Curve")
    y_pred_proba = xgb_model.predict_proba(X_test)[:, 1] # Operates on the output of the predict_proba, selecting the second column - Probability that the Case is Fraud -  (Class 1)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba) # false positive rate, true positive rate, thresholds

    # Add a slider for threshold
    threshold = st.slider("Select Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

    # ROC_AUC score
    roc_auc = roc_auc_score(y_test, y_pred_proba)  # Use probabilities for ROC-AUC 

    # plt.subplots() is a function that returns a tuple containing a figure and axes object(s). Thus when using fig, ax = plt.subplots() you unpack this tuple into the variables fig and ax. 
    fig, ax = plt.subplots() 
    ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()

    # Highlight the selected threshold
    idx = np.argmin(np.abs(thresholds - threshold)) # Returns the indices of the minimum values in a numpy array
    ax.scatter(fpr[idx], tpr[idx], color="red", label=f"Threshold = {threshold:.2f}")
    ax.legend()
    st.pyplot(fig)

# Precision-Recall Curve
elif page == "Precision-Recall Curve":
    st.header("Precision-Recall Curve")
    y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    average_precision = average_precision_score(y_test, y_pred_proba)
    fig, ax = plt.subplots()
    ax.plot(recall, precision, label=f"Precision-Recall Curve (Average Precision = {average_precision:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    st.pyplot(fig)

# Distribution Plots
elif page == "Distribution Plots":
    st.header("Distribution Plots")
 
    df2 = df.drop(['type'], axis=1)

    # Remove the 'type' feature because it is a categorical column
    columns = df2.columns
    
    # Select a feature for the distribution plot
    feature = st.selectbox("Select a feature", columns)
    
    # Create the distribution plot
    fig, ax = plt.subplots()
    sns.histplot(data=df2, x=feature, hue="isFraud", kde=True, ax=ax)
    st.pyplot(fig)

# Correlation Heatmap
elif page == "Correlation Heatmap":
    st.header("Correlation Heatmap")

    df2 = df.drop(['type'], axis=1)

    fig, ax = plt.subplots()
    sns.heatmap(df2.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Box Plots
elif page == "Box Plots":
    st.header("Box Plots")
    df2 = df.drop(['type'], axis=1)
    feature = st.selectbox("Select a feature", df2.columns)
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="isFraud", y=feature, ax=ax)
    st.pyplot(fig)

# Violin Plots
elif page == "Violin Plots":
    st.header("Violin Plots")
    df2 = df.drop(['type'], axis=1)
    feature = st.selectbox("Select a feature", df2.columns)
    fig, ax = plt.subplots()
    sns.violinplot(data=df, x="isFraud", y=feature, ax=ax)
    st.pyplot(fig)

# Scatter Plot by feature
elif page == "Scatter Plot":
    st.header("Scatter Plot")

    # Select two numerical features for the scatter plot
    df2 = df.drop(['type'], axis=1)

    feature_x = st.selectbox("Select X-axis feature", df.columns)
    feature_y = st.selectbox("Select Y-axis feature", df.columns)

    # Ensure the selected features are numerical
    if df[feature_x].dtype not in ['float64', 'int64'] or df[feature_y].dtype not in ['float64', 'int64']:
        st.error("Please select numerical features for the scatter plot.")
    else:
        # Create the scatter plot
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=feature_x, y=feature_y, hue="isFraud", alpha=0.6, ax=ax)
        ax.set_title(f"Scatter Plot: {feature_x} vs {feature_y}")
        ax.set_xlabel(feature_x)
        ax.set_ylabel(feature_y)
        st.pyplot(fig)

# SHAP Values
elif page == "SHAP Values":
    st.header("SHAP Values")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_test, plot_type="bar")
    st.pyplot(fig)

# T-Test (Fraud vs Non. Fraud Transaction Amounts)"
elif page == "T-Test (Fraud vs Non. Fraud Transaction Amounts)":
    st.header("T-Test: Fraud vs. Non-Fraud Transaction Amounts")

    # Split the data into fraud and non-fraud transactions
    fraud_amounts = df[df['isFraud'] == 1]['amount']
    non_fraud_amounts = df[df['isFraud'] == 0]['amount']

    # Perform the T-Test
    t_stat, p_value = ttest_ind(fraud_amounts, non_fraud_amounts, equal_var=False)

    # Display results
    st.write(f"T-Statistic: {t_stat:.4f}")
    st.write(f"P-Value: {p_value:.4f}")

    # Interpret the results
    if p_value < 0.05:
        st.write("**Conclusion**: The difference in transaction amounts between fraud and non-fraud transactions is statistically significant (p < 0.05)")
    else:
        st.write("**Conclusion**: The difference in transaction amounts between fraud and non-fraud transactions is not statistically significant (p >= 0.05).")
    
# Chi-Square Test: Fraud Rate by Transaction Type
elif page == "Chi-Square Test: Fraud Rate by Transaction Type":
    st.header("Chi-Square Test: Fraud Rate by Transaction Type")

    # Create a contingency table of transaction type vs. fraud
    contingency_table = pd.crosstab(df['type'], df['isFraud'])

    # Perform the Chi-Square Test
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table) # dof = degrees of freedom (n. rows - 1) x (n. col - 1), p_value = probability of getting a
    # chi-square statistic as extreme or more extreme than the calculated "chi_stat" if the null hypothesis is true. A low p-value indicates a significant result -> there is sufficient evidence to reject the null hypothesis.
    # Expected values: Frequencies that would be expected in each cell of the contigency table if there was no association between the variables being analyzed

    # Display results
    if p_value < 0.05:
        st.write("*Conclusion**: The fraud rate differs significantly across transaction types (p < 0.05).")
    else:
        st.write("**Conclusion**: The fraud rate does not differ signiifcantly across transaction types (p >= 0.05).")

# A/B Testing: Comparing CNN vs. XGBoost Fraud Detection"
elif page == "A/B Testing: Comparing CNN vs. XGBoost Fraud Detection":
    st.header("A/B Testing: CNN vs. XGBoost Model Performance")

    # Get predictions from both models
    cnn_predictions = (cnn_model.predict(X_test) > 0.5).astype(int)
    xgb_predictions = xgb_model.predict(X_test)

    # Calculate precision for both models
    cnn_predictions = (cnn_model.predict(X_test) > 0.5).astype(int)
    xgb_predictions = xgb_model.predict(X_test)

    # Calculate precision for both models
    cnn_precision = precision_score(y_test, cnn_predictions)
    xgb_precision = precision_score(y_test, xgb_predictions)

    # Perform a paired T-Test on precision scores
    t_stat, p_value = ttest_rel([cnn_precision], [xgb_precision])

    # display results
    # Display Results in Colored Boxes
    st.metric("CNN Precision Score:", f"{cnn_precision:.4f}")   
    st.metric("XGB Precision Score:", f"{xgb_precision:.4f}")
    st.metric("T-Statistic Score:", f"{t_stat:.4f}")
    st.metric("P-Value Score:", f"{p_value:.4f}")

    # Display results
    if p_value < 0.05:
        st.write("**Conclusion**: There is a statistically significant difference in precision between the CNN and XGBoost models (p < 0.05).")
    else:
        st.write("**Conclusion**: There is no statistically significant difference in precision between the CNN and XGBoost models (p >= 0.05).")

elif page == "Recall T-Test: CNN vs. XGBoost":
    st.header("Recall T-Test: CNN vs. XGBoost")

    # Get predictions from both models
    cnn_predictions = (cnn_model.predict(X_test) > 0.5).astype(int)
    xgb_predictions = xgb_model.predict(X_test)

    # Calculate precision for both models
    cnn_recall = recall_score(y_test, cnn_predictions)
    xgb_recall = recall_score(y_test, xgb_predictions)

    # Perform a T-Test on precision scores
    t_stat, p_value = ttest_rel([cnn_recall], [xgb_recall])

    # display results
    st.metric("CNN Recall Score:", f"{cnn_recall:.4f}")   
    st.metric("XGB Recall Score:", f"{xgb_recall:.4f}")
    st.metric("T-Statistic Score:", f"{t_stat:.4f}")
    st.metric("P-Value Score:", f"{p_value:.4f}")
    # Interpret the results
    if p_value < 0.05:
        st.write("**Conclusion**: There is a statistically significant difference in Recall between the CNN and XGBoost models (p < 0.05).")
    else:
        st.write("**Conclusion**: There is no statistically significant difference in Recall between the CNN and XGBoost models (p >= 0.05).")

