import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("Tesla.csv")

# Train model if not already trained
@st.cache_resource
def train_model():
    df = load_data()
    df = df.dropna()
    X = df[['Open', 'High', 'Low']]
    y = df['Close']
    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, "model.pkl")
    return model

# Load or train model
try:
    model = joblib.load("model.pkl")
except:
    model = train_model()

st.title("Tesla Stock Price Predictor")

st.markdown("#### Enter stock features below")

open_price = st.number_input("Open Price", min_value=0.0)
high_price = st.number_input("High Price", min_value=0.0)
low_price = st.number_input("Low Price", min_value=0.0)

if st.button("Predict Closing Price"):
    input_data = np.array([[open_price, high_price, low_price]])
    prediction = model.predict(input_data)
    st.success(f"Predicted Close Price: ${prediction[0]:.2f}")
