#!/usr/bin/env python3
"""
Run NIFTY Only
Modified runner to trade only NIFTY index to avoid waiting for other models
"""

import logging
import sys
import os

# Ensure we can import trading
sys.path.append(os.getcwd())

from trading import TradingBot, Config

# Modify configuration in memory
print("⚡ Configuring for NIFTY ONLY trading...")
Config.INDICES = {"NIFTY": "^NSEI"}
Config.FORCE_RETRAIN = False
Config.USE_SIMPLIFIED_FILTERS = True
Config.CONFIDENCE_THRESHOLD = 0.50 # Recommended for paper trading
Config.MIN_VOLUME_RATIO = 0.5

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nifty_trading.log'),
        logging.StreamHandler()
    ]
)

if __name__ == "__main__":
    try:
        print("🤖 Starting TradingBot for NIFTY...")
        bot = TradingBot()
        
        # Inject enhanced signal generation if not already in trading.py
        try:
            from generate_signal_enhanced import generate_signal_enhanced
            import types
            print("✨ Injecting enhanced signal generation method...")
            bot.generate_signal = types.MethodType(generate_signal_enhanced, bot)
        except ImportError:
            print("⚠️  Could not import generate_signal_enhanced. Using default.")
            
        # Explicitly load models as this is not done in __init__
        print("📥 Loading models...")
        bot.load_or_train_models()
            
        # Run with default poll interval (300s)
        print("🚀 Starting paper trading session...")
        bot.run()
    except KeyboardInterrupt:
        print("🛑 Stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
