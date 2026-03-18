import yfinance as yf
import pandas as pd

try:
    symbol = "^NSEI"
    print(f"Testing yfinance download for {symbol}...")
    df = yf.download(tickers=symbol, period="1d", interval="5m", progress=False)
    print("Download result:")
    print(df)
    print(f"Empty: {df.empty}")
except Exception as e:
    print(f"Caught exception: {type(e).__name__} - {e}")

try:
    symbol = "NIFTY_FIN_SERVICE.NS"
    print(f"\nTesting yfinance download for {symbol}...")
    df = yf.download(tickers=symbol, period="1d", interval="5m", progress=False)
    print("Download result:")
    print(df)
    print(f"Empty: {df.empty}")
except Exception as e:
    print(f"Caught exception: {type(e).__name__} - {e}")
