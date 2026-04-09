import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Chi-Town Sentinel | Intelligence Portal", layout="wide")

# --- 2. ASSET LOADING (RAM-Optimized & Lazy-Loaded) ---
@st.cache_resource
def load_assets():
    models = {}
    # Base Models (Essential & Light)
    models["Decision Tree (Best)"] = joblib.load('decision_tree_model.pkl')
    models["Logistic Regression"] = joblib.load('logistic_regression_model.pkl')
    
    # Advanced Models (Lazy Loading to prevent Cloud RAM crashes)
    # Note: Use your specific file names: 'random_forest.pkl' or 'random_forest_model.pkl'
    try:
        models["Random Forest (Ensemble)"] = joblib.load('random_forest_model.pkl')
    except Exception:
        st.sidebar.warning("Random Forest pkl not found or too large for Cloud RAM.")

    try:
        models["XGBoost (Gradient Boost)"] = joblib.load('xgboost_model.pkl')
    except Exception:
        st.sidebar.warning("XGBoost pkl not found.")
        
    scaler = joblib.load('scaler.pkl')
    return models, scaler

models, scaler = load_assets()

# --- 3. COORDINATE MAPPING (For Dynamic Map) ---
DISTRICT_COORDS = {
    1: [41.86, -87.62], 11: [41.88, -87.72], 12: [41.86, -87.67],
    24: [41.99, -87.67], 25: [41.92, -87.75]
}
DEFAULT_COORD = [41.8781, -87.6298] # Downtown Chicago

# --- 4. REFINED CATEGORIES ---
CRIME_TYPES = ["THEFT", "BATTERY", "CRIMINAL DAMAGE", "ASSAULT", "NARCOTICS", "WEAPONS VIOLATION"]
LOCATIONS = ["STREET", "RESIDENCE", "SIDEWALK", "APARTMENT", "GAS STATION", "PARKING LOT"]

# --- 5. SIDEBAR: INTELLIGENCE CONTROLS ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=80)
    st.title("Intelligence Portal")
    st.markdown("---")
    
    selected_model_name = st.selectbox("Intelligence Engine", list(models.keys()))
    current_model = models[selected_model_name]
    
    user_threshold = st.slider("Deployment Threshold", 0.50, 0.90, 0.70)
    
    st.markdown("---")
    st.caption("Operational Status: **LIVE**")
    st.caption(f"Active Engine: **{selected_model_name}**")

# --- 6. MAIN INTERFACE ---
st.title("🏙️ Chicago Crime Arrest Prediction MVP")
st.markdown("---")

col_input, col_status = st.columns([1.2, 1], gap="large")

with col_input:
    st.subheader("📝 Case Entry & Feature Engineering")
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            crime_type = st.selectbox("Primary Offense", CRIME_TYPES)
            location = st.selectbox("Location Description", LOCATIONS)
        with c2:
            district = st.number_input("Police District", 1, 31, 11)
            is_domestic = st.toggle("Domestic Incident Flag", value=True)

        st.markdown("**Temporal Parameters**")
        hour = st.select_slider("Time of Day", options=list(range(24)), value=22)
        month = st.select_slider("Month of Year", options=list(range(1, 13)), value=4)

        run_btn = st.button("🔥 Execute Strategic Analysis", use_container_width=True)

    # DYNAMIC MAP
    st.subheader("📍 Geospatial Context")
    map_lat, map_lon = DISTRICT_COORDS.get(district, DEFAULT_COORD)
    map_data = pd.DataFrame({'lat': [map_lat], 'lon': [map_lon]})
    st.map(map_data, zoom=11 if district in DISTRICT_COORDS else 10)

with col_status:
    if run_btn:
        # Prepare 60-feature vector matching your training pipeline
        input_data = np.zeros((1, 60)) 
        input_data[0, 0], input_data[0, 2] = month, hour
        input_data[0, 10], input_data[0, 13] = district, int(is_domestic)
        
        # One-Hot Encoding mapping
        if crime_type in CRIME_TYPES: input_data[0, 20 + CRIME_TYPES.index(crime_type)] = 1
        if location in LOCATIONS: input_data[0, 40 + LOCATIONS.index(location)] = 1
        
        # Scaling & Inference
        scaled_input = scaler.transform(input_data)
        prob = current_model.predict_proba(scaled_input)[0, 1]
        
        st.subheader("📡 Real-Time Intelligence")
        if prob >= user_threshold:
            st.error(f"### ARREST LIKELY ({prob*100:.1f}%)")
            st.markdown("> **Response Profile:** High Priority. Evidence suggests probable cause.")
        elif prob >= 0.40:
            st.warning(f"### ELEVATED RISK ({prob*100:.1f}%)")
            st.markdown("> **Response Profile:** Moderate Priority. Monitor for escalation.")
        else:
            st.success(f"### LOW PROBABILITY ({prob*100:.1f}%)")
            st.markdown("> **Response Profile:** Standard investigative response.")

        m1, m2 = st.columns(2)
        m1.metric("Model Architecture", selected_model_name.split(" ")[0])
        m2.metric("Threshold Delta", f"{prob - user_threshold:+.2f}")

# --- 7. TECHNICAL VERIFICATION SECTION ---
st.markdown("---")
st.subheader("🔬 Model Integrity & Explainability")
tab1, tab2, tab3 = st.tabs(["Performance Metrics", "Feature Drivers", "Confusion Matrix"])

with tab1:
    st.image("roc_curves.png", use_container_width=True)
    st.info("The AUC-ROC validates our classifier's ability to distinguish between events across 2.7M records.")
with tab2:
    st.image("feature_importance.png", use_container_width=True)
with tab3:
    # DYNAMIC MATRIX LOGIC
    if "Decision Tree" in selected_model_name:
        st.image("confusion_matrix_Decision_Tree.png", use_container_width=True)
    elif "Random Forest" in selected_model_name:
        st.image("confusion_matrix_Random_Forest.png", use_container_width=True)
    elif "XGBoost" in selected_model_name:
        st.image("confusion_matrix_XGBoost.png", use_container_width=True)
    else:
        st.image("confusion_matrix_Logistic_Regression.png", use_container_width=True)

st.divider()
st.caption(" Data Science for Crime Analytics | Final Project MVP")