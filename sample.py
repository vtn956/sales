import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# Streamlit App Title
st.title("Sales Forecasting App")

# Upload Dataset
uploaded_file = st.file_uploader("Upload a CSV file with 'date' and 'sales' columns", type="csv")

if uploaded_file is not None:
    # Load the data
    df = pd.read_csv(uploaded_file)

    # Ensure correct column names
    if "date" not in df.columns or "sales" not in df.columns:
        st.error("CSV must contain 'date' and 'sales' columns!")
    else:
        # Convert date column to datetime
        df["date"] = pd.to_datetime(df["date"])

        # Rename columns for Prophet
        df.rename(columns={"date": "ds", "sales": "y"}, inplace=True)

        # Display the data
        st.write("### Uploaded Sales Data")
        st.write(df.tail())

        # Train Prophet Model
        model = Prophet()
        model.fit(df)

        # Forecast Future Sales
        future = model.make_future_dataframe(periods=90)  # Predict next 90 days
        forecast = model.predict(future)

        # Plot the forecast
        fig = model.plot(forecast)
        st.pyplot(fig)

        # Show forecast data
        st.write("### Forecasted Data")
        st.write(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(10))
