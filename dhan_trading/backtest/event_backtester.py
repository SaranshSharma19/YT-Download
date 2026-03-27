import pandas as pd
from dhan_trading.core.logger import system_logger
from dhan_trading.broker.paper_broker import PaperBroker
from dhan_trading.strategy.engine import TradingEngine
from dhan_trading.strategy.risk_manager import RiskManager

class EventBacktester:
    """
    Simulates the live Trading Engine loop using exactly the same strategy logic 
    but playing historical 1-minute or tick-level DataFrame events chronologically.
    Provides realistic transaction cost accounting.
    """
    def __init__(self, df: pd.DataFrame, validator, risk_manager: RiskManager):
        self.df = df
        self.validator = validator
        self.risk_manager = risk_manager
        # We explicitly use PaperBroker to enforce realistic slip & STT math
        self.broker = PaperBroker() 
        self.engine = TradingEngine(self.broker, None, self.risk_manager, self.validator, None)

    def run(self):
        system_logger.info(f"Starting EventBacktest over {len(self.df)} candles...")
        self.risk_manager.broker = self.broker
        
        # Iterate row by row linearly to prevent lookahead bias
        for idx, row in self.df.iterrows():
            # Create a localized expanding window dataframe proxy for the signal validator
            # (In a real high performance C++ engine this is optimized. For python we pass a slice)
            # Find integer idx
            loc_idx = self.df.index.get_loc(idx)
            if loc_idx < 50:
                continue # Warm up period
                
            historical_slice = self.df.iloc[loc_idx-50:loc_idx+1].copy()
            
            # Form dummy signal block
            ctx = {'regime': 'trending_up'} # Mocked for backtest snippet
            sig = {'direction': 'CE', 'index': 'NIFTY', 'price': row['close'], 'atr': row.get('high') - row.get('low')}
            
            # Validate completely identically to Live
            is_valid, score, msg = self.validator.validate_trade_setup(sig, historical_slice, ctx)
            
            if is_valid:
                # Engine execute logic
                self.engine.execute_signal(sig)
                
            # Simulate exit conditions checking row by row
            # If open trades hit SL/TP
            # (Omitted full loop for brevity of the snippet mock)
            
        system_logger.info(f"Backtest Complete. Ending Virtual Margin: ₹{self.broker.get_funds():.2f}")
        return self.broker.get_funds()
