import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Заголовок
st.title("✈️ Airline Passenger Satisfaction")
st.write("Predict whether a passenger is satisfied or not")

# Загрузка модели
model = joblib.load("xgb_pipeline.pkl")

# ---- INPUT FORM ----
st.header("Enter Passenger Data")

age = st.slider("Age", 0, 100, 30)
flight_distance = st.number_input("Flight Distance", 0, 10000, 1000)

wifi = st.slider("Inflight wifi service", 0, 5, 3)
online_boarding = st.slider("Online boarding", 0, 5, 3)
seat_comfort = st.slider("Seat comfort", 0, 5, 3)
food = st.slider("Food and drink", 0, 5, 3)

delay_dep = st.number_input("Departure Delay (minutes)", 0, 1000, 0)
delay_arr = st.number_input("Arrival Delay (minutes)", 0, 1000, 0)

gender = st.selectbox("Gender", ["Male", "Female"])
customer_type = st.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
travel_type = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
flight_class = st.selectbox("Class", ["Business", "Eco", "Eco Plus"])

# ---- FEATURE ENGINEERING (как в твоём проекте) ----
total_delay = delay_dep + delay_arr
log_distance = np.log1p(flight_distance)
service_avg = (wifi + online_boarding + seat_comfort + food) / 4

if age < 25:
    age_group = "Young"
elif age < 60:
    age_group = "Middle"
else:
    age_group = "Senior"

# ---- DATAFRAME ----
input_df = pd.DataFrame([{
    "Age": age,
    "Flight Distance": flight_distance,
    "Inflight wifi service": wifi,
    "Online boarding": online_boarding,
    "Seat comfort": seat_comfort,
    "Food and drink": food,
    "Total_delay": total_delay,
    "Log_Flight_Distance": log_distance,
    "Service_avg": service_avg,
    "Gender": gender,
    "Customer Type": customer_type,
    "Type of Travel": travel_type,
    "Class": flight_class,
    "Age_group": age_group
}])

# ---- PREDICT ----
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"✅ Satisfied (probability: {prob:.2f})")
    else:
        st.error(f"❌ Not satisfied (probability: {prob:.2f})")