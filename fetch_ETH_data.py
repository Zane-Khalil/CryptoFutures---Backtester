import yfinance as yf
import pandas as pd

# Download 30 days of 1-hour ETH-USD data
print("⏳ Fetching ETH-USD data from Yahoo Finance...")
df = yf.download("ETH-USD", interval="1h", period="30d")

# Save to CSV
df.to_csv("eth_data.csv")
print("✅ Saved to eth_data.csv")
