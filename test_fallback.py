from dhan_trading.broker.data_feed import DataFeed
from dhan_trading.core.config import Config
import datetime
import pandas as pd

def test_fallback():
    # Mock dhan client as None to force fallback
    feed = DataFeed(None)
    
    today = datetime.date.today().strftime('%Y-%m-%d')
    yesterday = (datetime.date.today() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Test NIFTY (ID 13)
    print(f"Testing NIFTY (13) fallback...")
    df = feed.fetch_historical_candles("13", "NSE", "5", yesterday, today)
    
    if not df.empty:
        print("SUCCESS: Data received!")
        print(df.head())
        print(f"Columns: {df.columns.tolist()}")
    else:
        print("FAILURE: No data received.")

    # Test BANKNIFTY (ID 25)
    print(f"\nTesting BANKNIFTY (25) fallback...")
    df = feed.fetch_historical_candles("25", "NSE", "5", yesterday, today)
    if not df.empty:
        print("SUCCESS: Data received!")
    else:
        print("FAILURE: No data received.")

if __name__ == "__main__":
    test_fallback()
