import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Chi-Town Sentinel | Intelligence Portal", layout="wide")

# --- 2. ASSET LOADING ---
@st.cache_resource
def load_assets():
    models = {
        "Decision Tree (Best)": joblib.load('decision_tree_model.pkl'),
        "Logistic Regression": joblib.load('logistic_regression_model.pkl')
    }
    scaler = joblib.load('scaler.pkl')
    return models, scaler

models, scaler = load_assets()

# --- 3. COORDINATE MAPPING (For Dynamic Map) ---
# Approximate center coordinates for Chicago Police Districts
DISTRICT_COORDS = {
    1: [41.86, -87.62], 11: [41.88, -87.72], 12: [41.86, -87.67],
    24: [41.99, -87.67], 25: [41.92, -87.75]
}
DEFAULT_COORD = [41.8781, -87.6298] # Downtown Chicago

# --- 4. CATEGORIES ---
CRIME_TYPES = ["THEFT", "BATTERY", "CRIMINAL DAMAGE", "ASSAULT", "NARCOTICS", "WEAPONS VIOLATION"]
LOCATIONS = ["STREET", "RESIDENCE", "SIDEWALK", "APARTMENT", "GAS STATION", "PARKING LOT"]

# --- 5. SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=80)
    st.title("Intelligence Portal")
    selected_model_name = st.selectbox("Intelligence Engine", list(models.keys()))
    current_model = models[selected_model_name]
    user_threshold = st.slider("Deployment Threshold", 0.50, 0.90, 0.70)
    st.markdown("---")
    st.caption(f"Active Engine: **{selected_model_name}**")

# --- 6. MAIN INTERFACE ---
st.title("🏙️ Chicago Crime Arrest Prediction MVP")

col_input, col_status = st.columns([1, 1], gap="large")

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
        
        hour = st.select_slider("Time of Day", options=list(range(24)), value=22)
        month = st.select_slider("Month of Year", options=list(range(1, 13)), value=4)
        run_btn = st.button("🔥 Execute Strategic Analysis", use_container_width=True)

    # --- THE DYNAMIC MAP ---
    st.subheader("📍 Geospatial Context")
    # Get coordinates for the district or default to downtown
    map_lat, map_lon = DISTRICT_COORDS.get(district, DEFAULT_COORD)
    map_data = pd.DataFrame({'lat': [map_lat], 'lon': [map_lon]})
    
    st.map(map_data, zoom=11 if district in DISTRICT_COORDS else 10)
    st.caption(f"Visualizing predicted incident proximity for District {district}")

with col_status:
    if run_btn:
        # Prepare 60-feature vector
        input_data = np.zeros((1, 60)) 
        input_data[0, 0], input_data[0, 2] = month, hour
        input_data[0, 10], input_data[0, 13] = district, int(is_domestic)
        
        if crime_type in CRIME_TYPES: input_data[0, 20 + CRIME_TYPES.index(crime_type)] = 1
        if location in LOCATIONS: input_data[0, 40 + LOCATIONS.index(location)] = 1
        
        prob = current_model.predict_proba(scaler.transform(input_data))[0, 1]
        
        st.subheader("📡 Real-Time Intelligence")
        if prob >= user_threshold:
            st.error(f"### ARREST LIKELY ({prob*100:.1f}%)")
        elif prob >= 0.40:
            st.warning(f"### ELEVATED RISK ({prob*100:.1f}%)")
        else:
            st.success(f"### LOW PROBABILITY ({prob*100:.1f}%)")

        m1, m2 = st.columns(2)
        m1.metric("Model Confidence", f"{prob*100:.1f}%")
        m2.metric("Threshold Delta", f"{prob - user_threshold:+.2f}")

# --- 7. TECHNICAL VERIFICATION ---
st.markdown("---")
st.subheader("🔬 Model Integrity & Explainability")
tab1, tab2, tab3 = st.tabs(["Performance Metrics", "Feature Drivers", "Confusion Matrix"])
with tab1: st.image("roc_curves.png", use_container_width=True)
with tab2: st.image("feature_importance.png", use_container_width=True)
with tab3:
    img = "confusion_matrix_Decision_Tree.png" if "Decision Tree" in selected_model_name else "confusion_matrix_Logistic_Regression.png"
    st.image(img, use_container_width=True)

st.divider()
st.caption("Final Project MVP")