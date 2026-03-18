
import yfinance as yf
import pandas as pd
import numpy as np

def test_fetch(symbol, name):
    print(f"\n--- Testing {name} ({symbol}) ---")
    
    # Method 1: yf.download (used in health_check)
    print("Method 1: yf.download...")
    try:
        df1 = yf.download(
            tickers=symbol,
            period="5d",
            interval="5m",
            auto_adjust=False,
            progress=False,
            threads=False
        )
        print(f"  Empty: {df1.empty}")
        print(f"  Columns: {df1.columns.tolist()}")
        if not df1.empty:
            print(f"  Shape: {df1.shape}")
            print(f"  Sample:\n{df1.head(2)}")
    except Exception as e:
        print(f"  Error: {e}")

    # Method 2: yf.Ticker (used in fetch_data)
    print("\nMethod 2: yf.Ticker.history...")
    try:
        ticker = yf.Ticker(symbol)
        df2 = ticker.history(period="5d", interval="5m")
        print(f"  Empty: {df2.empty}")
        print(f"  Columns: {df2.columns.tolist()}")
        if not df2.empty:
            print(f"  Shape: {df2.shape}")
            print(f"  Sample:\n{df2.head(2)}")
    except Exception as e:
        print(f"  Error: {e}")

if __name__ == "__main__":
    test_fetch("^NSEI", "NIFTY")
    test_fetch("^NSEBANK", "BANKNIFTY")
    test_fetch("NIFTY_FIN_SERVICE.NS", "FINNIFTY")
