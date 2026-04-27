import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# =============================
# CONFIG
# =============================
st.set_page_config(page_title="Airline Predictor", layout="centered")

# =============================
# LOAD MODEL
# =============================
@st.cache_resource
def load_model():
    return joblib.load("xgb_pipeline.pkl")

model = load_model()

# =============================
# LOAD SHAP
# =============================
@st.cache_resource
def load_explainer():
    return shap.TreeExplainer(model.named_steps['clf'])

explainer = load_explainer()

# =============================
# SESSION STATE
# =============================
if "pred" not in st.session_state:
    st.session_state.pred = None
    st.session_state.prob = None
    st.session_state.input_df = None

# =============================
# FEATURE NAMES (FIXED)
# =============================
preprocessor = model.named_steps['pre']
feature_names = preprocessor.get_feature_names_out()
feature_names = [f.replace("num__", "").replace("cat__", "") for f in feature_names]

# =============================
# CACHE TRANSFORM
# =============================
@st.cache_data
def transform_input(df):
    return model.named_steps['pre'].transform(df)

# =============================
# UI
# =============================
st.title("✈️ Airline Satisfaction Predictor")
st.markdown("Predict passenger satisfaction using XGBoost model")

# =============================
# SIDEBAR
# =============================
st.sidebar.header("Passenger Info")

age = st.sidebar.slider("Age", 10, 80, 30)
distance = st.sidebar.number_input("Flight Distance", 100, 5000, 1000)

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
customer_type = st.sidebar.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
travel_type = st.sidebar.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
flight_class = st.sidebar.selectbox("Class", ["Business", "Eco", "Eco Plus"])

# =============================
# MAIN INPUT
# =============================
st.header("Flight Details")

col1, col2 = st.columns(2)

with col1:
    delay_dep = st.number_input("Departure Delay", 0, 500, 10)

with col2:
    delay_arr = st.number_input("Arrival Delay", 0, 500, 10)

# =============================
# SERVICES
# =============================
st.subheader("Service Ratings")

SERVICE_COLS = [
    'Inflight wifi service', 'Departure/Arrival time convenient',
    'Ease of Online booking', 'Gate location', 'Food and drink',
    'Online boarding', 'Seat comfort', 'Inflight entertainment',
    'On-board service', 'Leg room service', 'Baggage handling',
    'Checkin service', 'Inflight service', 'Cleanliness'
]

service_values = {}

with st.expander("Adjust service ratings"):
    for col in SERVICE_COLS:
        service_values[col] = st.slider(col, 0, 5, 3)

# =============================
# FEATURE ENGINEERING
# =============================
Total_delay = delay_dep + delay_arr
Log_Flight_Distance = np.log1p(distance)
Service_avg = np.mean(list(service_values.values()))

if age <= 25:
    Age_group = "Young"
elif age <= 60:
    Age_group = "Middle"
else:
    Age_group = "Senior"

# =============================
# INPUT DF (FIXED)
# =============================
input_dict = {
    'Age': age,
    'Flight Distance': distance,
    'Departure Delay in Minutes': delay_dep,
    'Arrival Delay in Minutes': delay_arr,
    'Total_delay': Total_delay,
    'Log_Flight_Distance': Log_Flight_Distance,
    'Service_avg': Service_avg,
    'Gender': gender,
    'Customer Type': customer_type,
    'Type of Travel': travel_type,
    'Class': flight_class,
    'Age_group': Age_group
}

input_dict.update(service_values)

input_df = pd.DataFrame([input_dict])

# 🔥 FIX: порядок колонок как в train
input_df = input_df[model.feature_names_in_]

# =============================
# PREDICT BUTTON
# =============================
st.divider()

if st.button("🚀 Predict"):
    with st.spinner("Analyzing..."):
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        st.session_state.pred = pred
        st.session_state.prob = prob
        st.session_state.input_df = input_df

# =============================
# SHOW RESULT
# =============================
if st.session_state.pred is not None:

    st.subheader("Result")
    st.metric("Satisfaction Probability", f"{st.session_state.prob:.2%}")

    if st.session_state.pred == 1:
        st.success("✅ Passenger is satisfied")
    else:
        st.error("❌ Passenger is NOT satisfied")

    # =============================
    # SHAP
    # =============================
    show_shap = st.checkbox("🔍 Show SHAP explanation")

    if show_shap:
        try:
            with st.spinner("Calculating SHAP..."):

                X_transformed = transform_input(st.session_state.input_df)

                shap_values = explainer.shap_values(X_transformed)

                if isinstance(shap_values, list):
                    shap_values = shap_values[1]

                base_value = explainer.expected_value
                if isinstance(base_value, list):
                    base_value = base_value[1]

                fig, ax = plt.subplots()

                shap.waterfall_plot(
                    shap.Explanation(
                        values=shap_values[0],
                        base_values=base_value,
                        data=X_transformed[0],
                        feature_names=feature_names
                    ),
                    show=False
                )

                st.pyplot(fig)

                # =============================
                # TOP FEATURES
                # =============================
                st.subheader("Top factors")

                values = shap_values[0]
                top_idx = np.argsort(np.abs(values))[::-1][:5]

                for i in top_idx:
                    impact = "⬆️ increases" if values[i] > 0 else "⬇️ decreases"
                    st.write(f"{feature_names[i]} {impact} satisfaction ({values[i]:.3f})")

        except Exception as e:
            st.error("SHAP error")
            st.text(str(e))
