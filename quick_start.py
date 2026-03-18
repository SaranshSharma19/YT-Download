#!/usr/bin/env python3
"""
Quick Start Script - Run Enhanced Trading System
This script demonstrates how to use all the new modules together
"""

import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_trading.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('QuickStart')

print("="*70)
print("ENHANCED TRADING SYSTEM - QUICK START")
print("="*70)
print()

# Import modules
try:
    from risk_manager import RiskManager
    from performance_tracker import PerformanceTracker
    from regime_detector import RegimeDetector
    from tradingview_validator import TradingViewValidator
    from adaptive_optimizer import AdaptiveOptimizer
    
    logger.info("✅ All enhanced modules imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import modules: {e}")
    print("\nPlease ensure all module files are in the same directory:")
    print("- risk_manager.py")
    print("- performance_tracker.py")
    print("- regime_detector.py")
    print("- tradingview_validator.py")
    print("- adaptive_optimizer.py")
    exit(1)

# Initialize modules
print("\n📦 Initializing modules...")

# 1. Risk Manager
risk_manager = RiskManager(account_balance=100000.0)
logger.info(f"Risk Manager: ₹{risk_manager.account_balance:,.2f} capital")

# 2. Performance Tracker
tracker = PerformanceTracker("trading_performance.db")
logger.info("Performance Tracker: Database ready")

# 3. Regime Detector
detector = RegimeDetector()
logger.info("Regime Detector: Ready")

# 4. TradingView Validator
validator = TradingViewValidator(auto_open=False)  # Set to True to open charts
logger.info("TradingView Validator: Ready")

# 5. Adaptive Optimizer
optimizer = AdaptiveOptimizer(performance_tracker=tracker)
logger.info("Adaptive Optimizer: Ready")

print("\n✅ All modules initialized successfully!")
print()

# Example workflow
print("="*70)
print("EXAMPLE WORKFLOW")
print("="*70)
print()

# Step 1: Detect regime
print("1️⃣  Detecting market regime...")
try:
    import yfinance as yf
    df = yf.download("^NSEI", period="60d", interval="5m", progress=False)
    
    if not df.empty:
        regime_result = detector.get_regime_with_params(df)
        print(f"   Regime: {regime_result['regime']}")
        print(f"   Threshold: {regime_result['parameters']['confidence_threshold']}")
        print(f"   Description: {regime_result['parameters']['description']}")
    else:
        print("   ⚠️  No data available")
except Exception as e:
    print(f"   ❌ Error: {e}")

print()

# Step 2: Calculate position size
print("2️⃣  Calculating position size...")
signal = {
    'atr': 87.5,
    'price': 19500.0,
    'confidence': 0.68
}

# Get recent stats
recent_stats = tracker.get_recent_stats('NIFTY', lookback_days=30)
position_info = risk_manager.calculate_position_size(signal, recent_stats)

print(f"   Position Size: {position_info['position_size']:.2f} units")
print(f"   Risk Amount: ₹{position_info['risk_amount']:.2f}")
print(f"   Kelly Adjusted: {position_info.get('kelly_adjusted', False)}")

print()

# Step 3: Calculate exits
print("3️⃣  Calculating stop-loss and targets...")
exits = risk_manager.calculate_exits(19500.0, 'CE', 87.5)

print(f"   Entry: ₹19,500.00")
print(f"   Stop Loss: ₹{exits['stop_loss']:,.2f} ({((exits['stop_loss']/19500-1)*100):.2f}%)")
print(f"   Take Profit 1: ₹{exits['tp1']:,.2f} ({((exits['tp1']/19500-1)*100):.2f}%)")
print(f"   Take Profit 2: ₹{exits['tp2']:,.2f} ({((exits['tp2']/19500-1)*100):.2f}%)")
print(f"   Risk:Reward: 1:{exits['risk_reward_ratio']:.1f}")

print()

# Step 4: Adaptive threshold
print("4️⃣  Checking adaptive threshold...")
threshold = optimizer.adaptive_params['NIFTY']['confidence_threshold']
print(f"   Current threshold: {threshold:.3f}")

# Check if retrain needed
retrain = optimizer.check_retrain_trigger('NIFTY', lookback_days=14)
print(f"   Retrain needed: {retrain['should_retrain']}")
if retrain['reasons']:
    print(f"   Reasons: {', '.join(retrain['reasons'])}")

print()

# Step 5: Performance summary
print("5️⃣  Performance summary...")
try:
    print(tracker.get_summary_report(lookback_days=30))
except Exception as e:
    print(f"   No trades logged yet")

print()

# Next steps
print("="*70)
print("NEXT STEPS")
print("="*70)
print()
print("1. Run the main trading system:")
print("   python trading.py")
print()
print("2. Monitor the logs:")
print("   tail -f enhanced_trading.log")
print()
print("3. Check the database:")
print("   sqlite3 trading_performance.db")
print()
print("4. Review TradingView charts:")
print("   ls chart_analysis/")
print()
print("5. Check adaptive parameters:")
print("   cat adaptive_config.json")
print()
print("="*70)
print("System ready for paper trading! 🚀")
print("="*70)
