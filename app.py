import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt  
import dill  
import yfinance as yf
with open("predict.pkl", "rb") as f:
    predict = dill.load(f) 
dt=pd.read_csv("dt.csv") 
start_date = "2023-06-01"
end_date = "2025-02-02"
ticker = "RELIANCE.NS"

# Download data from Yahoo Finance
data = yf.download(ticker, start=start_date, end=end_date)
# Reset the index of the downloaded data to have a single-level index
df = data.reset_index()[["Date", "Close"]].rename(columns={"Date": "date", "Close": "y"})
df['date']=pd.to_datetime(df['date'])
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0) 
dt['date']=pd.to_datetime(dt['date']) 
df = df.merge(dt, on='date', how='left') 
dt=df

st.title("SARIMAX Forecast with Weighted Sentiment Regressor")
user_date = st.text_input("Enter date in format YYYY-MM-DD (put date after 2025-03-03)")   
if user_date:
    try:
        # Get predictions 
        dt=dt.set_index("date")
        a, b = predict(user_date) 
        a.set_index("date", inplace=True)
        b.set_index("date", inplace=True)

        # Plot the graph
        fig, ax = plt.subplots(figsize=(10, 6))  # Fix figure size

        ax.plot(dt.index, dt['y'], label='Observed', color='blue')
        ax.plot(a.index, a['predicted_mean'], label='Forecast', color='red')
        ax.fill_between(b.index, b.iloc[:, 0], b.iloc[:, 1], color='pink', alpha=0.3)

        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.set_title("SARIMAX Forecast with Weighted Sentiment Regressor")
        ax.legend()

        # Display the graph in Streamlit
        st.pyplot(fig)
        plt.close(fig)  # Fix multiple figure issue

    except ValueError:
        st.error("⚠ Please enter a valid date in YYYY-MM-DD format.")
