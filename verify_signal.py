
import logging
import sys
import os
import joblib
from trading import TradingBot, Config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def verify():
    print("Initializing TradingBot for verification...")
    bot = TradingBot()
    
    # Ensure validators are set up (they should be in __init__ now, but double check)
    if not bot.signal_validator:
        print("WARNING: SignalValidator not initialized in bot!")
    
    # Load NIFTY model manually to ensure it's there
    model_path = os.path.join(Config.MODEL_DIR, "NIFTY_model.joblib")
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        bot.models['NIFTY'] = joblib.load(model_path)
    else:
        print("ERROR: NIFTY model not found!")
        return

    print("Generating signal for NIFTY...")
    signal = bot.generate_signal('NIFTY', '^NSEI')
    
    if signal:
        print("\n=== SIGNAL GENERATED ===")
        print(signal)
    else:
        print("\n=== NO SIGNAL GENERATED ===")
        print("Check logs for details (Regime filter, weak prediction, or validator rejection).")

if __name__ == "__main__":
    verify()
