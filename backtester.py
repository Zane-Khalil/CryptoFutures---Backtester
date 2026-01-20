import os
import time
import random
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import argparse


class FuturesBacktester:
    def __init__(self, data, initial_balance=1000, leverage=10, fee_rate=0.0004,
                slippage=0.0005, funding_rate=0.0001):
        if isinstance(data, pd.DataFrame):
            data = FuturesBacktester.df_to_list(data)
        self.data = data

        self.balance = initial_balance
        self.equity = initial_balance
        self.leverage = leverage
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.funding_rate = funding_rate
        self.position = None
        self.history = []
        self.equity_curve = []
        self.daily_returns = []

    @staticmethod
    def df_to_list(df):
        df = df.dropna()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(1)

        formatted_data = []
        for idx, row in df.iterrows():
            timestamp = idx.strftime('%Y-%m-%d %H:%M:%S')

            open_price = row['Open']
            high_price = row['High']
            low_price = row['Low']
            close_price = row['Close']

            if isinstance(open_price, pd.Series): open_price = open_price.iloc[0]
            if isinstance(high_price, pd.Series): high_price = high_price.iloc[0]
            if isinstance(low_price, pd.Series): low_price = low_price.iloc[0]
            if isinstance(close_price, pd.Series): close_price = close_price.iloc[0]

            formatted_data.append([timestamp, open_price, high_price, low_price, close_price])
        return formatted_data

    @staticmethod
    def fetch_yfinance_data(symbol="BTC-USD", interval="1h", period="60d", retries=3, delay=10):
        interval_map = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "60m", "1d": "1d"}
        yf_interval = interval_map.get(interval, "60m")

        for attempt in range(retries):
            try:
                # Add delay between attempts to avoid rate limiting
                if attempt > 0:
                    time.sleep(delay)
                    
                df = yf.download(tickers=symbol, interval=yf_interval, period=period, progress=False)
                if df.empty:
                    raise ValueError("Downloaded data is empty.")
                print(f"📥 Fetched {len(df)} rows for {symbol} at {interval} interval.")
                return FuturesBacktester.df_to_list(df)
            except Exception as e:
                print(f"[Attempt {attempt+1}] Error fetching data: {e}")
                if attempt < retries - 1:
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
        print("All retry attempts failed.")
        return []

    def apply_slippage(self, price, side):
        slippage_amount = price * self.slippage
        return price + slippage_amount if side == 'long' else price - slippage_amount

    def enter_position(self, side, price, timestamp, stop_loss=None, take_profit=None):
        price = self.apply_slippage(price, side)
        size = (self.balance * self.leverage) / price
        fee = self.balance * self.fee_rate
        self.position = {
            'side': side,
            'entry_price': price,
            'size': size,
            'timestamp': timestamp,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }
        self.balance -= fee

    def exit_position(self, price, timestamp):
        if not self.position:
            return
        entry_price = self.position['entry_price']
        size = self.position['size']
        pnl = (price - entry_price) * size if self.position['side'] == 'long' else (entry_price - price) * size
        fee = self.balance * self.fee_rate
        funding_cost = entry_price * size * self.funding_rate
        self.balance += pnl - fee - funding_cost
        self.equity = self.balance
        self.history.append({
            'entry_time': self.position['timestamp'],
            'exit_time': timestamp,
            'entry_price': entry_price,
            'exit_price': price,
            'pnl': pnl - fee - funding_cost
        })
        self.position = None

    def run_backtest(self, strategy_func):
        print(f"🚀 Running backtest on {len(self.data)} data points")
        prev_equity = self.equity
        start_index = 50 if len(self.data) > 50 else 0
        for i in range(start_index, len(self.data)):
            row = self.data[i]
            timestamp, _, high, low, close = row

            self.equity_curve.append((timestamp, self.equity))
            if i > start_index:
                daily_return = (self.equity - prev_equity) / prev_equity
                self.daily_returns.append(daily_return)
                prev_equity = self.equity

            signal = strategy_func(self.data[:i])
            if signal and not self.position:
                self.enter_position(
                    signal['action'], close, timestamp,
                    stop_loss=signal.get('stop_loss'),
                    take_profit=signal.get('take_profit')
                )
            elif self.position:
                triggered_exit = False
                if self.position['side'] == 'long':
                    if self.position['stop_loss'] and low <= self.position['stop_loss']:
                        self.exit_position(self.position['stop_loss'], timestamp)
                        triggered_exit = True
                    elif self.position['take_profit'] and high >= self.position['take_profit']:
                        self.exit_position(self.position['take_profit'], timestamp)
                        triggered_exit = True
                elif self.position['side'] == 'short':
                    if self.position['stop_loss'] and high >= self.position['stop_loss']:
                        self.exit_position(self.position['stop_loss'], timestamp)
                        triggered_exit = True
                    elif self.position['take_profit'] and low <= self.position['take_profit']:
                        self.exit_position(self.position['take_profit'], timestamp)
                        triggered_exit = True

                if not triggered_exit:
                    signal = strategy_func(self.data[:i])
                    if signal == 'close':
                        self.exit_position(close, timestamp)

        if self.position:
            final_price = self.data[-1][4]
            self.exit_position(final_price, self.data[-1][0])

    def plot_equity_curve(self):
        if not self.equity_curve:
            print("No equity curve data to plot.")
            return
        times, equity = zip(*self.equity_curve)
        plt.figure(figsize=(12, 6))
        plt.plot(times, equity, label='Equity Curve')
        plt.xlabel('Time')
        plt.ylabel('Equity')
        plt.title('Futures Strategy Equity Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def calculate_sharpe_ratio(self, risk_free_rate=0.0):
        if not self.daily_returns:
            return None
        excess_returns = [r - risk_free_rate for r in self.daily_returns]
        avg_excess_return = np.mean(excess_returns)
        std_dev = np.std(excess_returns)
        if std_dev == 0:
            return 0
        return (avg_excess_return / std_dev) * np.sqrt(252)

    def summary(self):
        sharpe = self.calculate_sharpe_ratio()
        if sharpe is None:
            sharpe = 0.0
        return {
            'final_balance': round(self.balance, 2),
            'num_trades': len(self.history),
            'sharpe_ratio': round(sharpe, 3),
            'trades': self.history
        }


# ---------------------- STRATEGY ----------------------
def ma_crossover_strategy(data, short_window=10, long_window=30, sl_pct=0.01, tp_pct=0.02):
    if len(data) < long_window:
        return None
    closes = [row[4] for row in data]
    short_ma = sum(closes[-short_window:]) / short_window
    long_ma = sum(closes[-long_window:]) / long_window
    last_price = closes[-1]
    if short_ma > long_ma:
        return {
            'action': 'long',
            'stop_loss': last_price * (1 - sl_pct),
            'take_profit': last_price * (1 + tp_pct)
        }
    elif short_ma < long_ma:
        return {
            'action': 'short',
            'stop_loss': last_price * (1 + sl_pct),
            'take_profit': last_price * (1 - tp_pct)
        }
    else:
        return None


# ---------------------- DATA LOADER ----------------------
def load_or_fetch_data(
    symbol="BTC-USD",
    interval="1h",
    period="30d",
    local_csv="btc_data.csv",
    use_cache=True,
    save_after_fetch=True
):
    if use_cache and os.path.exists(local_csv):
        try:
            print(f"📂 Loading cached data from {local_csv}")
            df = pd.read_csv(local_csv, index_col=0)
            df.index = pd.to_datetime(df.index)
            return FuturesBacktester.df_to_list(df)
        except Exception as e:
            print(f"⚠️ Failed to load cached data: {e}")

    try:
        data = FuturesBacktester.fetch_yfinance_data(symbol=symbol, interval=interval, period=period)
        if data and save_after_fetch:
            # Convert data back to DataFrame for saving
            df_data = []
            for row in data:
                timestamp = pd.to_datetime(row[0])
                df_data.append({
                    'Open': row[1],
                    'High': row[2], 
                    'Low': row[3],
                    'Close': row[4]
                })
            df = pd.DataFrame(df_data, index=[row[0] for row in data])
            df.index = pd.to_datetime(df.index)
            df.to_csv(local_csv)
            print(f"💾 Saved fetched data to {local_csv}")
        return data
    except Exception as e:
        print(f"❌ Error fetching data from Yahoo Finance: {e}")
        return []



# ---------------------- MAIN ----------------------
if __name__ == "__main__":
    # 🔧 CLI argument parser
    parser = argparse.ArgumentParser(description="Run futures backtest on BTC or ETH")
    parser.add_argument(
        "--symbol", choices=["BTC-USD", "ETH-USD"], default="BTC-USD",
        help="Choose the crypto symbol to backtest"
    )
    args = parser.parse_args()

    csv_file = "btc_data.csv" if args.symbol == "BTC-USD" else "eth_data.csv"

    # 📂 Load CSV with proper error handling
    try:
        # First try to read as single-index CSV (normal format)
        df = pd.read_csv(csv_file, index_col=0)
        df.index = pd.to_datetime(df.index, errors='coerce')
        df = df.dropna()
        
        # Check if we have actual data (not just headers)
        if len(df) == 0 or df.empty:
            raise ValueError("CSV file is empty or contains no data")
            
        data = FuturesBacktester.df_to_list(df)
        print(f"✅ Loaded {len(data)} data points for {args.symbol} from {csv_file}")
    except Exception as e:
        print(f"❌ Failed to load {csv_file}: {e}")
        print("🔄 Attempting to fetch fresh data...")
        # Try to fetch fresh data if CSV loading fails
        data = load_or_fetch_data(
            symbol=args.symbol,
            interval="1h",
            period="30d",
            local_csv=csv_file,
            use_cache=False,
            save_after_fetch=True
        ) 

    if not data:
        print("Exiting: No market data available.")
        exit()

    print("Sample of data:")
    for row in data[:5]:
        print(row)

    bt = FuturesBacktester(data)
    bt.run_backtest(ma_crossover_strategy)

    print(bt.summary())
    bt.plot_equity_curve()
