import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Broker Credentials
    DHAN_CLIENT_ID = os.getenv("DHAN_CLIENT_ID", "")
    DHAN_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN", "")
    
    # Account & Risk Settings
    ACCOUNT_BALANCE = 100000.0
    RISK_PER_TRADE = 0.015
    MAX_DAILY_LOSS = 0.10
    DISABLE_DAILY_LOSS_LIMIT = True # Set to True to bypass the 3% safety stop during development
    MAX_POSITION_SIZE = 0.10
    
    # Strategy
    TIMEFRAME = "5m"
    CORE_INDICES = {
        'NIFTY': 'NIFTY',
        'BANKNIFTY': 'BANKNIFTY',
        'FINNIFTY': 'FINNIFTY'
    }
    
    # yfinance Tickers for fallbacks
    YFINANCE_TICKERS = {
        'NIFTY': '^NSEI',
        'BANKNIFTY': '^NSEBANK',
        'FINNIFTY': 'NIFTY_FIN_SERVICE.NS',
        '13': '^NSEI',
        '25': '^NSEBANK',
        '27': 'NIFTY_FIN_SERVICE.NS'
    }
    
    # Market Hours
    MARKET_START = "09:15"
    SQUARE_OFF_TIME = "15:10"
    MARKET_END = "15:30"
    
    # Model Configuration
    MIN_SCORE = 60
    
    # Bridge Configuration
    USE_MONOLITH_BRIDGE = True # Set to True to use internal brains from trading.py
