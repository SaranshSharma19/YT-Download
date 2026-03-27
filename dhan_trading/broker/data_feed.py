import pandas as pd
import yfinance as yf
from datetime import datetime
from dhanhq import dhanhq
from dhan_trading.core.config import Config
from dhan_trading.core.logger import system_logger, error_logger

class DataFeed:
    """
    Handles fetching historical market data via Dhan HQ API.
    Falls back to yfinance if Dhan is unavailable or unauthorized.
    """
    def __init__(self, dhan_client: dhanhq):
        self.dhan = dhan_client
        
    def fetch_historical_candles(self, security_id: str, exchange: str, interval: str, from_date: str, to_date: str) -> pd.DataFrame:
        """
        Fetch historical minute-level data.
        interval: e.g. "5" for 5 minutes
        from_date/to_date format: 'YYYY-MM-DD'
        """
        # 1. Try Dhan HQ first
        if self.dhan:
            try:
                # Assuming security_id is the numeric ID for indices (e.g. 13 for NIFTY)
                res = self.dhan.historical_minute_charts(
                    symbol=security_id, 
                    exchange_segment=exchange, 
                    instrument_type='INDEX', 
                    expiry_code=0, 
                    from_date=from_date, 
                    to_date=to_date
                )
                
                if res.get('status') == 'success':
                    data = res['data']
                    df = pd.DataFrame({
                        'timestamp': pd.to_datetime(data['start_Time']),
                        'open': data['open'],
                        'high': data['high'],
                        'low': data['low'],
                        'close': data['close'],
                        'volume': data['volume']
                    })
                    df.set_index('timestamp', inplace=True)
                    return df
                else:
                    error_logger.warning(f"Dhan Data Fetch Failed: {res.get('remarks')}")
            except Exception as e:
                error_logger.error(f"Error fetching from Dhan: {e}")

        # 2. Fallback to yfinance (Crucial for Paper Trading)
        # Map security_id/symbol to yfinance ticker
        # If security_id is '13' or 'NIFTY', we use the YFINANCE_TICKERS map
        yf_ticker = Config.YFINANCE_TICKERS.get(security_id)
        if not yf_ticker:
            # Try to find by reverse lookup or common names if security_id is e.g. "BANKNIFTY"
            yf_ticker = Config.YFINANCE_TICKERS.get(str(security_id).upper())

        if yf_ticker:
            system_logger.info(f"Using yfinance fallback for {security_id} ({yf_ticker})")
            try:
                # yfinance expects interval in '5m' format, Dhan uses '5'
                yf_interval = f"{interval}m" if interval.isdigit() else interval
                
                # If from_date == to_date, yfinance download(start=..., end=...) might return empty
                # We adjust to include the current day's data
                start_dt = datetime.strptime(from_date, '%Y-%m-%d')
                end_dt = datetime.strptime(to_date, '%Y-%m-%d')
                
                if start_dt == end_dt:
                    # Fetching for "today" -> use period='1d' or adjust end date to tomorrow
                    data = yf.download(
                        tickers=yf_ticker,
                        period='1d',
                        interval=yf_interval,
                        progress=False
                    )
                else:
                    # Adjust end_dt to be inclusive for yfinance by adding 1 day
                    end_dt_adj = (end_dt + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
                    data = yf.download(
                        tickers=yf_ticker,
                        start=from_date,
                        end=end_dt_adj,
                        interval=yf_interval,
                        progress=False
                    )
                
                if not data.empty:
                    # Clean up multi-index columns if they exist
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)
                        
                    # Standardize columns to lowercase
                    df = pd.DataFrame({
                        'open': data['Open'],
                        'high': data['High'],
                        'low': data['Low'],
                        'close': data['Close'],
                        'volume': data['Volume']
                    })
                    return df
            except Exception as e:
                error_logger.error(f"yfinance fallback failed for {yf_ticker}: {e}")
        
        return pd.DataFrame()
