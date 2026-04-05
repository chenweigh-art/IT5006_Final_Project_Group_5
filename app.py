import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- 1. PAGE CONFIG (Basic UI Requirement) ---
st.set_page_config(page_title="Chicago Crime POC", layout="centered")

# --- 2. LOAD ASSETS with Error Handling (Deployment Requirement) ---
@st.cache_resource
def load_assets():
    try:
        # We focus on the Decision Tree and Logistic Regression for the MVP
        models = {
            "Decision Tree (Best Model)": joblib.load('decision_tree_model.pkl'),
            "Logistic Regression": joblib.load('logistic_regression_model.pkl')
        }
        scaler = joblib.load('scaler.pkl')
        return models, scaler
    except Exception as e:
        st.error(f"Error loading model assets: {e}")
        return None, None

models, scaler = load_assets()

# --- 3. UI HEADER ---
st.title("🛡️ Chicago Crime Arrest Prediction (POC)")
st.markdown("""
**Project Goal:** Demonstrate a proof-of-concept for predicting arrest probability ($P(Arrest=1|X)$) 
based on spatial and temporal crime data from 2015-2024.
""")

if models is None:
    st.warning("Application is initializing. Please ensure model files are in the repository.")
else:
    # --- 4. BASIC USER INTERFACE (Input Validation Requirement) ---
    st.sidebar.header("Model Parameters")
    selected_model = st.sidebar.selectbox("Select Architecture", list(models.keys()))
    
    # Input Validation: We use sliders and min/max limits
    st.subheader("Input Crime Scenario Details")
    
    col1, col2 = st.columns(2)
    with col1:
        district = st.number_input("Police District (1-31)", min_value=1, max_value=31, value=11)
        month = st.slider("Month of Occurrence", 1, 12, 4)
    with col2:
        hour = st.slider("Hour of Day (24hr)", 0, 23, 12)
        is_domestic = st.selectbox("Domestic Violence?", ["No", "Yes"])

    # --- 5. CORE FUNCTIONALITY (Prediction Demonstration) ---
    if st.button("Generate Prediction Analysis"):
        try:
            # Prepare the 60-feature input vector
            # Mapping inputs to the exact indices used during Stage 3 training
            input_vector = np.zeros((1, 60))
            input_vector[0, 0] = month
            input_vector[0, 2] = hour
            input_vector[0, 10] = district
            input_vector[0, 13] = 1 if is_domestic == "Yes" else 0
            
            # Scale and Infer
            scaled_vector = scaler.transform(input_vector)
            prediction_prob = models[selected_model].predict_proba(scaled_vector)[0, 1]
            
            # Display Results
            st.divider()
            st.metric(label="Arrest Probability Score", value=f"{prediction_prob*100:.2f}%")
            
            # Operational Logic based on your 0.70 Threshold
            if prediction_prob >= 0.70:
                st.error("Result: HIGH PROBABILITY (Recommend immediate unit dispatch)")
            else:
                st.success("Result: LOW PROBABILITY (Standard processing recommended)")
                
        except Exception as e:
            st.error(f"Prediction Error: {e}. Please check input values.")

# --- 6. MODEL INSIGHTS (Validation Evidence) ---
st.divider()
with st.expander("View Technical Evaluation (ROC & Confusion Matrix)"):
    st.info("These static reports validate the model's performance on 2.7M rows.")
    st.image("roc_curves.png", caption="Model Comparison (AUC-ROC)")
    st.image("confusion_matrix_Decision_Tree.png", caption="Arrest Task Performance")