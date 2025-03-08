# 📈 Reliance Industries Stock Price Forecasting using Sentiment Analysis

This project leverages sentiment analysis and time series forecasting to predict the stock price of **Reliance Industries** using data extracted from Reddit posts and comments. The project integrates advanced data cleaning techniques, sentiment scoring, and forecasting models to achieve accurate predictions.

---

## 🚀 Project Overview

This project combines:
- 📊 **Stock Price Data** — Extracted from Yahoo Finance for Reliance Industries.
- 🗣️ **Sentiment Analysis** — Collected through Reddit API using targeted keywords and analyzed using the FinBERT model.
- 🔄 **Rolling Weighted Sentiment Calculation** — Assigning dynamic weights to past sentiment values to capture trends effectively.
- 🧠 **Forecasting Model** — Utilizes SARIMAX for time series forecasting, enhanced with sentiment data as a regressor.
- 🔮 **Future Sentiment Prediction** — An LSTM model predicts future sentiment scores to extend the forecasting period by generating regressor values beyond available data.

---

## 🗂️ Data Collection

The project extracts sentiment data from Reddit using the `praw` library. Key steps include:
- 📡 **Subreddit Scraping** — Data is collected from finance-specific subreddits like `r/IndianStockMarket`, `r/Finance`, and others.
- 🔎 **Keyword Filtering** — Filters content using Reliance-specific keywords such as *Reliance*, *Mukesh Ambani*, *Reliance Jio*, etc.
- 📰 **News & Comments Extraction** — Collects both top-scoring news posts and relevant comments for sentiment analysis.

---

## 🧹 Data Cleaning and Preprocessing

Data cleaning steps include:
- 🔍 Removing duplicates and irrelevant posts.
- ✂️ Tokenization, stemming, and **lemmatization** for improved text clarity.
- 🚫 Stopword removal using combined `nltk` and `sklearn` stopwords.
- 📈 Generating **sentiment scores** using the FinBERT transformer model.

---

## 🧮 Weighted Sentiment Calculation

The weighted sentiment values are calculated to improve correlation with stock prices. The weighted sum formula is applied for both news and comments sentiment scores:

```python
def weighted_sentiment(series):
    return (
        series.shift(1) * 0.4 +
        series.shift(2) * 0.25 +
        series.shift(3) * 0.15 +
        series.shift(4) * 0.10 +
        series.shift(5) * 0.10
    ).fillna(0)
```

The two weighted sentiment scores are then averaged to obtain the **final weighted sentiment** column used for forecasting.

---

## 🔮 Forecasting Model

The project uses the **SARIMAX** model for forecasting stock prices with `weighted_sentiment` as a regressor. 

### Model Parameters for SARIMAX
- **Order**: (1,1,1) — Captures ARIMA structure.
- **Seasonal Order**: (1,1,1,12) — Captures yearly seasonality in stock trends.
- **Training Period**: From `2023-06-01` to `2025-02-01`.
- **Testing Period**: From `2025-02-02` to `2025-03-03`.

### Future Sentiment Prediction with LSTM
Since future sentiment data is unavailable beyond `2025-03-03`, an **LSTM** model was trained using previous 14 days' sentiment values to predict the next day's sentiment score. This predicted sentiment acts as the regressor value for future stock price forecasting.

---

## 📊 Results

- ✅ **Mean Absolute Error (MAE):** 41
- 📈 **Mean Predicted Value:** 1267
- 🟠 **Prediction Error Rate:** ~3.27%

The model demonstrated strong performance with minimal prediction error, showcasing the effectiveness of integrating sentiment analysis with stock price forecasting.

--- 

## 📹Video Demo

- **[Click here to watch the demo video](https://drive.google.com/file/d/17eRfHadpHCOLz0GtXHlYiB8AIyqEtCe_/view?usp=sharing)** 

--- 


## 🖥️ App Deployment

The project includes a `streamlit` app (`app.py`) that allows users to:
1. Select a future date beyond `2025-03-03`.
2. Predict the future stock price for Reliance Industries using forecasted sentiment scores.

---

## 📋 How to Run
1. Clone the repository.
2. Install dependencies using:
```bash
pip install -r requirements.txt
```
3. Run the Streamlit app:
```bash
streamlit run app.py
```
4. Select your desired date to view the forecasted stock price.

---

## 🏆 Key Achievements
- ✅ Effective use of **Reddit data scraping** for sentiment analysis.
- ✅ Applied **FinBERT** for improved sentiment scoring accuracy.
- ✅ Utilized **weighted sentiment** as a powerful feature for boosting prediction performance.
- ✅ Achieved **low MAE** with optimized SARIMAX parameters.

---

## 📧 Contact
For questions, suggestions, or collaborations, feel free to reach out. 😊 

mail-b23cy1021@iitj.ac.in

