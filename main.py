import argparse
import sys
import pandas as pd
from dhan_trading.core.logger import system_logger
from dhan_trading.core.config import Config
from dhan_trading.broker.dhan_adapter import DhanBrokerAdapter
from dhan_trading.broker.paper_broker import PaperBroker
from dhan_trading.broker.instrument_mapper import InstrumentMapper
from dhan_trading.broker.data_feed import DataFeed
from dhan_trading.strategy.risk_manager import RiskManager
from dhan_trading.strategy.signal_validator import SignalValidator
from dhan_trading.strategy.engine import TradingEngine
from dhan_trading.strategy.regime_detector import RegimeDetector
from dhan_trading.monitoring.health_monitor import HealthMonitor
from dhan_trading.backtest.event_backtester import EventBacktester
from dhan_trading.backtest.optimizer import OptimizationEngine

def setup_components(mode: str):
    mapper = InstrumentMapper()
    live_broker = DhanBrokerAdapter()
    
    if mode == "paper":
        system_logger.info("Starting Paper Trading Engine with Realistic Fills")
        broker = PaperBroker()
    elif mode == "live":
        system_logger.info("DANGER: Starting LIVE API Broker Execution Engine")
        broker = live_broker
    else:
        broker = None

    data_feed = DataFeed(live_broker.dhan) if live_broker.dhan else None
    regime_detector = RegimeDetector()
    validator = SignalValidator({'min_score': Config.MIN_SCORE})
    risk = RiskManager(broker)
    engine = TradingEngine(broker, data_feed, risk, validator, mapper, regime_detector)
    health = HealthMonitor(broker)
    
    return engine, health, risk, validator

def run_paper_mode():
    engine, health, risk, validator = setup_components("paper")
    if not health.run_diagnostics():
        system_logger.critical("Health diagnostics failed preventing startup.")
        sys.exit(1)
    
    engine.run()

def run_backtest_mode():
    # Generate mock dataframe for example since DataFeed API returns historical
    # Normally we load huge CSVS for backtest. Here's a dummy for structural execution:
    system_logger.info("Loading Historical Data for Event-Driven Backtest...")
    df = pd.DataFrame({'close': [22000] * 100, 'volume': [1000] * 100})
    
    _, _, risk, validator = setup_components("backtest")
    tester = EventBacktester(df, validator, risk)
    tester.run()
    
def run_optimize_mode():
    system_logger.info("Initializing Walk-Forward Optimizer...")
    df = pd.DataFrame({'close': [22000] * 100, 'volume': [1000] * 100})
    optimizer = OptimizationEngine(df)
    optimizer.run_grid_search()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Institutional Dhan Trading Bot")
    parser.add_argument('mode', choices=['paper', 'backtest', 'optimize', 'live'], help="Mode of execution")
    args = parser.parse_args()
    
    if args.mode == "paper":
        run_paper_mode()
    elif args.mode == "backtest":
        run_backtest_mode()
    elif args.mode == "optimize":
        run_optimize_mode()
    elif args.mode == "live":
        system_logger.critical("Live trading is explicitly disabled in this iteration until paper completes 2 months.")
        sys.exit(1)