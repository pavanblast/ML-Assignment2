import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    matthews_corrcoef
)

st.set_page_config(page_title="Fetal Health Classifier", layout="wide")

st.title("🩺 Fetal Health Prediction System")
st.markdown("Upload test data and evaluate trained ML models.")

# -------------------------------------------------
# Load trained models
# -------------------------------------------------
models = {
    "Logistic Regression": pickle.load(open("LR_model.pkl", "rb")),
    "Decision Tree": pickle.load(open("decision_tree_model.pkl", "rb")),
    "Random Forest": pickle.load(open("random_forest_model.pkl", "rb")),
    "Naive Bayes": pickle.load(open("gaussian_nb_model.pkl", "rb")),
    "K-Nearest Neighbors": pickle.load(open("knn_model.pkl", "rb")),
    #"XGBoost": pickle.load(open("xgboost_model.pkl", "rb"))
}

# -------------------------------------------------
# Upload CSV
# -------------------------------------------------
uploaded_file = st.file_uploader("Upload Test Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Dataset Preview")
    st.dataframe(data.head())

    # Split features and target
    X_test = data.drop("fetal_health", axis=1)
    y_test = data["fetal_health"]

    # -------------------------------------------------
    # Model selection
    # -------------------------------------------------
    model_name = st.selectbox("Select Model", list(models.keys()))
    model = models[model_name]

    
    # Predict
    # -------------------------------------------------
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # Fix label shift for XGBoost (0,1,2 → 1,2,3)
    if model_name == "XGBoost":
        y_pred = y_pred + 1

    # -------------------------------------------------
    # Metrics
    # -------------------------------------------------
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    mcc = matthews_corrcoef(y_test, y_pred)

    st.subheader("📊 Model Performance")

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    col1.metric("Accuracy", f"{acc:.3f}")
    col2.metric("ROC AUC", f"{auc:.3f}")
    col3.metric("Precision", f"{prec:.3f}")
    col4.metric("Recall", f"{rec:.3f}")
    col5.metric("F1 Score", f"{f1:.3f}")
    col6.metric("MCC", f"{mcc:.3f}")

    # -------------------------------------------------
    # Confusion Matrix
    # -------------------------------------------------
    st.subheader("🔍 Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Suspect", "Pathological"],
        yticklabels=["Normal", "Suspect", "Pathological"],
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # -------------------------------------------------
    # Classification Report
    # -------------------------------------------------
    st.subheader("📄 Classification Report")

    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=["Normal", "Suspect", "Pathological"],
        output_dict=True
    )

    report_df = pd.DataFrame(report_dict).transpose()
    report_df = report_df.round(3)

    # Custom CSS
    st.markdown("""
    <style>
    .report-table {
        border-collapse: collapse;
        width: 100%;
    }
    .report-table th {
        background-color: #0f4c81;
        color: white;
        padding: 10px;
        text-align: center;
    }
    .report-table td {
        padding: 8px;
        text-align: center;
        border-bottom: 1px solid #ddd;
    }
    .report-table tr:hover {
        background-color: #f1f1f1;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(report_df.to_html(classes="report-table", border=0), unsafe_allow_html=True)

