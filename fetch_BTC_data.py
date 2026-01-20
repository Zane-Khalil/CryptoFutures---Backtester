import yfinance as yf
import pandas as pd

# Download 30 days of 1-hour BTC-USD data
print("⏳ Fetching BTC-USD data from Yahoo Finance...")
df = yf.download("BTC-USD", interval="1h", period="30d")

# Save to CSV
df.to_csv("btc_data.csv")
print("✅ Saved to btc_data.csv")
