import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load your model (make sure the file is in the same folder)
model = joblib.load("random_forest_pipeline_v2.pkl")

st.set_page_config(page_title="Sydney Airbnb Predictor", layout="centered")
st.title("Sydney Airbnb Price Prediction")
st.markdown("Predict nightly price for Airbnb listings in Sydney")

st.sidebar.header("Input Features")

room_type = st.sidebar.selectbox("Room Type", 
    ["Entire home/apt", "Private room", "Shared room", "Hotel room"])

neighbourhood = st.sidebar.selectbox("Neighbourhood", 
    sorted(['Bondi Beach', 'Sydney', 'Manly', 'Surry Hills', 'Newtown', 
            'Paddington', 'Coogee', 'Darlinghurst', 'Parramatta', 'Randwick']))

availability_365 = st.sidebar.slider("Availability (days/year)", 0, 365, 180)
reviews_per_month = st.sidebar.slider("Reviews per month", 0.0, 50.0, 2.0, 0.1)
price_per_bedroom = st.sidebar.slider("Price per bedroom ($)", 50.0, 1000.0, 200.0)

data = {
    'room_type': room_type,
    'neighbourhood_cleansed': neighbourhood,
    'availability_365': availability_365,
    'reviews_per_month': reviews_per_month,
    'price_per_bedroom': price_per_bedroom
}
input_df = pd.DataFrame(data, index=[0])

st.subheader("Your Input")
st.write(input_df)

if st.button("Predict Price"):
    with st.spinner("Predicting..."):
        pred = model.predict(input_df)[0]
        price = np.expm1(pred) if pred < 20 else pred   # auto-detect log scale
        st.metric("Predicted Nightly Price", f"${price:,.2f} AUD")
