
import sqlite3
import pandas as pd
from datetime import datetime, timedelta

def get_last_week_trades():
    try:
        conn = sqlite3.connect('trading_performance.db')
        
        # Check if 'trades' table exists
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        if not tables:
            print("No tables found in trading_performance.db")
            return
            
        table_name = tables[0][0]
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        conn.close()
        
        if df.empty:
            print("No trades found in DB.")
            return

        # Find date column
        date_col = next((col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()), None)
        if not date_col:
            print(f"No date column found. Columns: {df.columns.tolist()}")
            print(df.head())
            return
            
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Filter for last week (since 2026-03-08)
        last_week_start = datetime(2026, 3, 8)
        last_week_trades = df[df[date_col] >= last_week_start].copy()
        
        if last_week_trades.empty:
            print("No trades found from the last week.")
            # Print latest trades to see what's available
            print("Latest 5 trades in DB:")
            print(df.sort_values(date_col, ascending=False).head(5))
            return
            
        print(f"Found {len(last_week_trades)} trades from the last week.")
        print(last_week_trades.to_string())
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    get_last_week_trades()
