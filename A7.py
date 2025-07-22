import streamlit as st
import numpy as np
import joblib

model = joblib.load("glass_model.pkl")
scaler = joblib.load("scaler.pkl")

glass_types = {
    1: "Building Windows Float Processed",
    2: "Building Windows Non-Float Processed",
    3: "Vehicle Windows Float Processed",
    4: "Vehicle Windows Non-Float Processed",
    5: "Containers",
    6: "Tableware",
    7: "Headlamps"
}
st.title("🔮 Glass Type Predictor")
st.markdown("Enter the chemical properties below to predict the type of glass.")

RI = st.number_input("Refractive Index (RI)", value=1.52)
Na = st.number_input("Sodium (Na)", value=13.0)
Mg = st.number_input("Magnesium (Mg)", value=2.0)
Al = st.number_input("Aluminum (Al)", value=1.0)
Si = st.number_input("Silicon (Si)", value=72.0)
K  = st.number_input("Potassium (K)", value=0.5)
Ca = st.number_input("Calcium (Ca)", value=8.0)
Ba = st.number_input("Barium (Ba)", value=0.1)
Fe = st.number_input("Iron (Fe)", value=0.1)

if st.button("Predict Glass Type"):
    user_input = np.array([[RI, Na, Mg, Al, Si, K, Ca, Ba, Fe]])
    user_scaled = scaler.transform(user_input)
    prediction = model.predict(user_scaled)[0]
    st.success(f"🏷 Predicted Glass Type: **{glass_types.get(prediction, 'Unknown')}**")
