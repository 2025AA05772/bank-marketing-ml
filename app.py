import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# ----------------------------
# Streamlit page configuration
# ----------------------------
st.set_page_config(page_title="Bank Marketing — ML Models", layout="centered")
st.title("Bank Marketing — Classification")
st.caption("Upload test_data.csv, pick a model, view metrics & confusion matrix. ('duration' is ignored to avoid leakage)")

# ----------------------------------------
# Discover and load available trained models
# ----------------------------------------
model_files = [f for f in os.listdir("model") if f.endswith(".pkl")]

# If no models are present, stop the app
if not model_files:
    st.error("No models found in ./model. Please train first.")
    st.stop()

# Build choices for the sidebar (strip '.pkl' and replace underscores)
choices = sorted([m.replace("_", " ")[:-4] for m in model_files])

# Sidebar control to select which trained model to evaluate
model_choice = st.sidebar.selectbox("Select a trained model", choices)

# Load the selected model/pipeline with joblib
pipe = joblib.load(f"model/{model_choice.replace(' ', '_')}.pkl")

# -------------------------------------------------
# Provide a link to a sample test CSV file
# -------------------------------------------------
st.sidebar.markdown("**Download example test file:**")
st.sidebar.write("https://github.com/2025AA05772/bank-marketing-ml/blob/main/test_data.csv")

# ------------------------
# File upload: test CSV(s)
# ------------------------
uploaded = st.file_uploader("Upload CSV (include 'y' for metrics).", type=['csv'])

if uploaded:
    #try to auto-detect delimiter ; or ,
    try:
        df = pd.read_csv(uploaded, sep=None, engine="python")
    except Exception:
        uploaded.seek(0)
        df = pd.read_csv(uploaded)

    # ----------------------------------------------------
    # Normalize target column 'y' if present and categorical
    # ----------------------------------------------------
    # Convert common "yes"/"no" labels to binary 1/0 (case/whitespace-insensitive)
    if 'y' in df.columns and df['y'].dtype == object:
        df['y'] = (
            df['y']
            .astype(str).str.strip().str.lower()
            .eq('yes')
            .astype(int)
        )

    # ------------------------------------------------------------
    # Drop 'duration' to avoid data leakage (per Bank Marketing UCI)
    # ------------------------------------------------------------
    if 'duration' in df.columns:
        df = df.drop(columns=['duration'])

    # ----------------------
    # Quick data preview UI
    # ----------------------
    st.subheader("Preview")
    st.dataframe(df.head())

    # --------------------------------------------------
    # Separate features and target, if 'y' is available
    # --------------------------------------------------
    if 'y' in df.columns:
        # Ensure y is integer 0/1
        y_true = df['y'].astype(int).values
        X_infer = df.drop(columns=['y'])
    else:
        # If no ground truth is provided, display only predictions/probabilities
        y_true = None
        X_infer = df

    # ----------------
    # Model inference
    # ----------------
    # Predict class labels
    y_pred = pipe.predict(X_infer)
    st.write(f"**Predictions (first 10):** {y_pred[:10].tolist()}")

    # Predict probabilities if the model supports it (e.g., logistic regression, tree-based)
    y_proba = pipe.predict_proba(X_infer)[:, 1] if hasattr(pipe, "predict_proba") else None
    if y_proba is not None:
        st.write(f"**Probabilities (first 10):** {np.round(y_proba[:10], 3).tolist()}")

    # -----------------------------
    # Metrics & Visualization block
    # -----------------------------
    if y_true is not None:
        # Core classification metrics
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)  # zero_division avoids warnings on degenerate cases
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)
        # AUC requires probabilities; if not available, show NA
        auc = roc_auc_score(y_true, y_proba) if y_proba is not None else float("nan")

        # Metric cards Layout
        st.subheader("Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{acc:.3f}")
        col2.metric("Precision", f"{prec:.3f}")
        col3.metric("Recall", f"{rec:.3f}")
        col1.metric("F1", f"{f1:.3f}")
        col2.metric("AUC", f"{auc:.3f}" if not np.isnan(auc) else "NA")
        col3.metric("MCC", f"{mcc:.3f}")

        # --------------------------------------
        # Confusion matrix for error breakdown
        # --------------------------------------
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap="Blues", ax=ax)
        plt.title(f"Confusion Matrix — {model_choice}")
        st.pyplot(fig)
    else:
        # If the dataset lacks 'y', can’t compute metrics
        st.info("No 'y' column found — showing predictions only.")
else:
    # Prompt the user to upload data to begin evaluation
    st.info("Upload a CSV to begin.")