
import logging
import pandas as pd
import numpy as np
from trading import TradingBot, Config
from regime_detector import RegimeDetector
import json
import os

def test_system():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("TestSystem")
    
    logger.info("Starting System Verification Test...")
    
    # 1. Initialize Bot
    # Note: We might need to mock some parts if we don't want it to actually fetch data
    # or wait for market hours.
    bot = TradingBot()
    
    # 2. Test Regime Detector
    logger.info("Testing Regime Detector...")
    detector = RegimeDetector()
    # Create fake trending data
    df = pd.DataFrame({
        'Open': np.linspace(100, 110, 100),
        'High': np.linspace(101, 111, 100),
        'Low': np.linspace(99, 109, 100),
        'Close': np.linspace(100, 110, 100),
        'Volume': [1000] * 100
    }, index=pd.date_range(start='2023-01-01', periods=100, freq='5min'))
    
    regime = detector.detect_regime(df)
    logger.info(f"Detected Regime for trending data: {regime}")
    
    # 3. Test State Persistence
    logger.info("Testing State Persistence...")
    symbol = "TEST_INDEX"
    test_pos = {
        "trade_id": "test_123",
        "entry_price": 100.0,
        "direction": "CE",
        "size": 10,
        "sl": 95.0,
        "tp": 110.0,
        "entry_time": "2023-01-01T10:00:00",
        "regime": "trending_up"
    }
    bot.state.add_position(symbol, test_pos)
    
    # Re-init state to check load
    from trading import TradingState
    new_state = TradingState()
    loaded_pos = new_state.get_position(symbol)
    if loaded_pos and loaded_pos['trade_id'] == "test_123":
        logger.info("✅ State Persistence Check: PASSED")
    else:
        logger.error("❌ State Persistence Check: FAILED")
        
    # Clean up test position
    bot.state.remove_position(symbol)

    # 4. Dry Run Logic (Partial)
    logger.info("Testing dynamic thresholds logic...")
    params = bot.get_dynamic_thresholds("trending_up")
    logger.info(f"Dynamic params for trending_up: {params}")
    
    if params['confidence_threshold'] == 0.55:
        logger.info("✅ Dynamic Threshold Check: PASSED")
    else:
        logger.error(f"❌ Dynamic Threshold Check: FAILED (Got {params['confidence_threshold']}, expected 0.55)")

    logger.info("Verification Test Completed.")

if __name__ == "__main__":
    test_system()
