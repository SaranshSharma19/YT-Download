import yfinance as yf
from datetime import datetime
import pandas as pd
from trading import TradingBot, Config

def test_cooldown():
    bot = TradingBot()
    bot.last_signal_bar['NIFTY'] = pd.Timestamp.now()
    res = bot.execute_trade_lifecycle('NIFTY', '^NSEI')
    print("Cooldown check result:", res)
    assert res['status'] == 'Cooldown', "Cooldown not properly triggering"
    print("Cooldown logic passed!")

try:
    test_cooldown()
except Exception as e:
    print(f"Test failed: {e}")
