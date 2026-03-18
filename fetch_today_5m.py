import yfinance as yf
import pandas as pd

pd.set_option('display.max_rows', 100)

indexes = {'NIFTY': '^NSEI', 'BANKNIFTY': '^NSEBANK', 'FINNIFTY': 'NIFTY_FIN_SERVICE.NS'}
for name, ticker in indexes.items():
    print(f"\n{'='*40}\n  {name} (Today's 5m Data) \n{'='*40}")
    try:
        df = yf.download(ticker, period='1d', interval='5m', progress=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            
            # Print the whole day to see the trend
            print(df[['Open', 'High', 'Low', 'Close']].to_string())
            
            high_time = df['High'].idxmax()
            low_time = df['Low'].idxmin()
            print(f"\nDaily High: {df['High'].max():.2f} at {high_time}")
            print(f"Daily Low:  {df['Low'].min():.2f} at {low_time}")
            move_pct = (df['High'].max() - df['Low'].min()) / df['Low'].min() * 100
            print(f"Max Intraday Move: {move_pct:.2f}%")
        else:
            print("No data fetched.")
    except Exception as e:
        print(f"Error fetching {name}: {e}")
