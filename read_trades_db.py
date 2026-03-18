
import sqlite3
import pandas as pd
from datetime import datetime

try:
    conn = sqlite3.connect('trading_performance.db')
    # List tables
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("Tables:", tables)

    # Assuming a 'trades' table exists, let's query it
    if tables:
        # Check columns in trades table
        cursor.execute(f"PRAGMA table_info({tables[0][0]});")
        columns = cursor.fetchall()
        print(f"Columns in {tables[0][0]}:", [col[1] for col in columns])
        
        # Query trades for today
        today = datetime.now().strftime('%Y-%m-%d')
        print(f"Querying for date: {today}")
        
        # Adjust query execution based on likely column names
        # Assuming 'timestamp' or 'entry_time' exists
        df = pd.read_sql_query(f"SELECT * FROM {tables[0][0]}", conn)
        
        # Filter for today if date column exists
        # Print first few rows to inspect date format
        if not df.empty:
            print("First 5 rows:")
            print(df.head())
            
            # Simple filtering in python
            # Look for date in 'entry_time' or similar
            date_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
            if date_cols:
                print("Date columns found:", date_cols)
                # Try to filter
                # df['entry_time'] = pd.to_datetime(df['entry_time'])
                # today_trades = df[df['entry_time'].dt.date == datetime.now().date()]
                # print("Today's trades:")
                # print(today_trades)
        else:
            print("No trades found in DB.")

    conn.close()

except Exception as e:
    print(f"Error: {e}")
