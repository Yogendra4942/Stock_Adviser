import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta

# Function to fetch stock data
def fetch_stock_data(symbol, start_date, end_date):
    stock = yf.Ticker(symbol)
    data = stock.history(start=start_date, end=end_date)
    return data, stock

# Function to prepare data for prediction
def prepare_data(df):
    df = df[['Close']].copy()
    df['Prediction'] = df['Close'].shift(-1)
    df.dropna(inplace=True)
    
    X = np.array(df['Close']).reshape(-1, 1)
    y = np.array(df['Prediction'])
    return X, y

# Function to get fundamentals and analyze stock
def analyze_stock(symbol, exchange):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)

    print(f"\nFetching data for {symbol} ({exchange})...")
    data, stock = fetch_stock_data(symbol, start_date, end_date)

    if data.empty:
        print(f"No data available for {symbol} on {exchange}")
        return None, None, None, None

    X, y = prepare_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    
    score = model.score(X_test, y_test)
    last_price = np.array([data['Close'][-1]]).reshape(-1, 1)
    predicted_price = model.predict(last_price)[0]

    fundamentals = stock.info
    pe_ratio = fundamentals.get('trailingPE', None)
    eps = fundamentals.get('trailingEps', None)
    dividend_yield = fundamentals.get('dividendYield', None) or 0
    debt_to_equity = fundamentals.get('debtToEquity', None)
    market_cap = fundamentals.get('marketCap', None)

    print(f"\nFundamentals for {symbol} ({exchange}):")
    print(f"P/E Ratio: {pe_ratio if pe_ratio else 'N/A'}")
    print(f"Earnings Per Share (EPS): {eps if eps else 'N/A'}")
    print(f"Dividend Yield: {dividend_yield * 100:.2f}%")
    print(f"Debt-to-Equity Ratio: {debt_to_equity if debt_to_equity else 'N/A'}")
    print(f"Market Cap: ₹{market_cap:,} (if available)")
    print(f"Predicted next day price: ₹{predicted_price:.2f}")
    print(f"Model accuracy (R² score): {score:.4f}")

    recommendation = "Neutral"
    reasons = []

    if pe_ratio and pe_ratio < 20:
        reasons.append("Low P/E ratio (potentially undervalued)")
    elif pe_ratio and pe_ratio > 30:
        reasons.append("High P/E ratio (potentially overvalued)")

    if eps and eps > 0:
        reasons.append("Positive EPS (profitable company)")
    elif eps and eps < 0:
        reasons.append("Negative EPS (company losing money)")

    if dividend_yield > 0:
        reasons.append("Pays dividends (stable income)")
    
    if debt_to_equity and debt_to_equity < 1:
        reasons.append("Low debt-to-equity (financially healthy)")
    elif debt_to_equity and debt_to_equity > 2:
        reasons.append("High debt-to-equity (risky finances)")

    if predicted_price > last_price[0][0]:
        reasons.append("Upward price trend predicted")
    else:
        reasons.append("Downward or flat price trend predicted")

    positive_factors = sum(1 for r in reasons if "Low P/E" in r or "Positive EPS" in r or "Pays dividends" in r or "Low debt" in r or "Upward" in r)
    negative_factors = len(reasons) - positive_factors

    if positive_factors > negative_factors + 1:
        recommendation = "Good to Buy"
    elif negative_factors > positive_factors + 1:
        recommendation = "Not Good to Buy"

    print(f"\nRecommendation for {symbol} ({exchange}): {recommendation}")
    print("Reasons:")
    for reason in reasons:
        print(f"- {reason}")

    return data, predicted_price, score, recommendation

# Main function to handle input and display analysis
def predict_stock():
    stock_name = input("Enter the stock name (e.g., RELIANCE, TCS, TATAMOTORS): ").upper()

    nse_symbol = f"{stock_name}.NS"
    bse_symbol = f"{stock_name}.BO"

    analyze_stock(nse_symbol, "NSE")
    analyze_stock(bse_symbol, "BSE")

if __name__ == "__main__":
    predict_stock()
