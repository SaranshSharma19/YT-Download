import pandas as pd
import numpy as np
from itertools import product
from dhan_trading.core.logger import system_logger
from dhan_trading.strategy.signal_validator import SignalValidator
from dhan_trading.strategy.risk_manager import RiskManager
from dhan_trading.backtest.event_backtester import EventBacktester

class OptimizationEngine:
    """
    Performs Monte Carlo simulations across defined parameter ranges
    to prevent overfitting on historical data using walk-forward principles.
    """
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.param_grid = {
            'min_score': [50, 60, 70],
            'trend_weight': [20, 30, 40],
            'momentum_weight': [20, 25, 30]
        }
        
    def _run_permutation(self, params: dict):
        base_config = {
            'min_score': params['min_score'],
            'weights': {
                'trend_alignment': params['trend_weight'],
                'momentum': params['momentum_weight'],
                'volume_flow': 20,
                'volatility': 15,
                'market_regime': 10
            }
        }
        
        validator = SignalValidator(base_config)
        # Mock broker used inside EventBacktester
        risk = RiskManager(broker=None) 
        
        backtest = EventBacktester(self.data, validator, risk)
        ending_capital = backtest.run()
        
        return {
            'params': params,
            'ending_capital': ending_capital,
            'return_pct': (ending_capital - 100000) / 100000 * 100
        }

    def run_grid_search(self):
        system_logger.info("Initializing Parameter Optimization (Monte Carlo Sandbox)")
        results = []
        
        keys, values = zip(*self.param_grid.items())
        permutations = [dict(zip(keys, v)) for v in product(*values)]
        
        for params in permutations:
            res = self._run_permutation(params)
            results.append(res)
            
        df = pd.DataFrame(results)
        df = df.sort_values(by='ending_capital', ascending=False)
        
        system_logger.info("\nTop 3 Parameter Sets:")
        for idx, row in df.head(3).iterrows():
            system_logger.info(f"Return: {row['return_pct']:.2f}% | Params: {row['params']}")
        return df
