import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

import streamlit as st
from streamlit_autorefresh import st_autorefresh

# 30 minutes = 30 * 60 * 1000 milliseconds
st_autorefresh(interval=1800000, key="refresh_30min")

st.title("My App")
st.write("This app refreshes every 30 minutes.")


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
    "Logistic Regression": pickle.load(open("model/LR_model.pkl", "rb")),
    "Decision Tree": pickle.load(open("model/decision_tree_model.pkl", "rb")),
    "Random Forest": pickle.load(open("model/random_forest_model.pkl", "rb")),
    "Naive Bayes": pickle.load(open("model/gaussian_nb_model.pkl", "rb")),
    "K-Nearest Neighbors": pickle.load(open("model/knn_model.pkl", "rb")),
    "XGBoost": pickle.load(open("model/xgboost_model.pkl", "rb"))
}

# -------------------------------------------------
# Download Sample CSV File
# -------------------------------------------------
st.subheader("📥 Sample Test Data (CSV)")

try:
    with open("fetal_health - Test.csv", "rb") as f:
        st.download_button(
            label="Download Sample Test CSV",
            data=f,
            file_name="fetal_health - Test.csv",
            mime="text/csv"
        )
except:
    st.warning("Sample CSV file not found in project folder.")

# -------------------------------------------------
# Upload Dataset (CSV)
# -------------------------------------------------
st.subheader("📤 Upload Your Test Data")

uploaded_file = st.file_uploader(
    "Upload Test Dataset (CSV)",
    type=["csv"]
)

if uploaded_file is not None:

    # Read CSV file
    try:
        data = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    st.subheader("📄 Uploaded Dataset Preview")
    st.dataframe(data.head())

    # Check required column
    if "fetal_health" not in data.columns:
        st.error("Uploaded file must contain 'fetal_health' column as target label.")
        st.stop()

    # Split features and target
    X_test = data.drop("fetal_health", axis=1)
    y_test = data["fetal_health"]

    # -------------------------------------------------
    # Model selection
    # -------------------------------------------------
    model_name = st.selectbox("Select Model", list(models.keys()))
    model = models[model_name]

    # -------------------------------------------------
    # Predict
    # -------------------------------------------------
    try:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

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

    plt.close("all")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(4, 3))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=["Normal", "Suspect", "Pathological"],
        yticklabels=["Normal", "Suspect", "Pathological"],
        ax=ax,
    )

    ax.set_xlabel("Predicted", fontsize=6)
    ax.set_ylabel("Actual", fontsize=6)
    ax.set_title("Confusion Matrix", fontsize=8)

    ax.tick_params(axis='both', labelsize=8)

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
