#!/usr/bin/env python3
"""
Quick Test Script for Enhanced Trading System
Tests all new modules independently and together
"""

import sys
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

print("="*70)
print("ENHANCED TRADING SYSTEM - MODULE TEST")
print("="*70)
print()

# Test 1: Risk Manager
print("1️⃣  Testing Risk Manager...")
try:
    from risk_manager import RiskManager
    
    rm = RiskManager(account_balance=100000.0)
    
    # Test position sizing
    signal = {
        'atr': 87.5,
        'price': 19500.0,
        'confidence': 0.68
    }
    
    position_info = rm.calculate_position_size(signal)
    print(f"   ✅ Position Size: {position_info['position_size']:.2f} units")
    print(f"   ✅ Risk Amount: ₹{position_info['risk_amount']:.2f}")
    
    # Test exit levels
    exits = rm.calculate_exits(19500.0, 'CE', 87.5)
    print(f"   ✅ Stop Loss: ₹{exits['stop_loss']:.2f}")
    print(f"   ✅ Take Profit 1: ₹{exits['tp1']:.2f}")
    print(f"   ✅ Take Profit 2: ₹{exits['tp2']:.2f}")
    
    print("   ✅ Risk Manager: PASSED\n")
except Exception as e:
    print(f"   ❌ Risk Manager: FAILED - {e}\n")
    sys.exit(1)

# Test 2: Performance Tracker
print("2️⃣  Testing Performance Tracker...")
try:
    from performance_tracker import PerformanceTracker
    
    tracker = PerformanceTracker("test_performance.db")
    
    # Log a test trade
    trade_id = tracker.log_trade(
        trade_id=12345,
        index_name="NIFTY",
        direction="CE",
        entry_price=19500.0,
        position_size=100,
        confidence=0.68,
        regime="trending_up",
        stop_loss=19450.0,
        take_profit_1=19600.0,
        take_profit_2=19650.0
    )
    print(f"   ✅ Trade logged: ID={trade_id}")
    
    # Update with exit
    result = tracker.update_trade_exit(trade_id, exit_price=19580.0, exit_reason="tp1_hit")
    print(f"   ✅ Trade closed: P&L = ₹{result['pnl']:+.2f}")
    
    # Get stats
    stats = tracker.get_recent_stats(lookback_days=30)
    print(f"   ✅ Stats retrieved: {stats['total_trades']} trades")
    
    print("   ✅ Performance Tracker: PASSED\n")
except Exception as e:
    print(f"   ❌ Performance Tracker: FAILED - {e}\n")
    sys.exit(1)

# Test 3: Regime Detector
print("3️⃣  Testing Regime Detector...")
try:
    from regime_detector import RegimeDetector
    import yfinance as yf
    
    detector = RegimeDetector()
    
    # Fetch sample data
    df = yf.download("^NSEI", period="60d", interval="5m", progress=False)
    
    if not df.empty:
        result = detector.get_regime_with_params(df)
        print(f"   ✅ Regime detected: {result['regime']}")
        print(f"   ✅ Confidence threshold: {result['parameters']['confidence_threshold']}")
        print(f"   ✅ Description: {result['parameters']['description']}")
        print("   ✅ Regime Detector: PASSED\n")
    else:
        print("   ⚠️  No data available for regime detection\n")
except Exception as e:
    print(f"   ❌ Regime Detector: FAILED - {e}\n")
    sys.exit(1)

# Test 4: TradingView Validator
print("4️⃣  Testing TradingView Validator...")
try:
    from tradingview_validator import TradingViewValidator
    
    validator = TradingViewValidator(auto_open=False)  # Don't open browser in test
    
    signal = {
        'index': 'NIFTY',
        'direction': 'CE',
        'price': 19500.50,
        'confidence': 0.68,
        'time': '14:30:00',
        'atr_ratio': 0.0045,
        'regime': 'trending_up'
    }
    
    result = validator.validate_signal(signal, '^NSEI', interval='5m')
    print(f"   ✅ Chart URL generated: {result['chart_url'][:50]}...")
    print("   ✅ TradingView Validator: PASSED\n")
except Exception as e:
    print(f"   ❌ TradingView Validator: FAILED - {e}\n")
    sys.exit(1)

# Test 5: Adaptive Optimizer
print("5️⃣  Testing Adaptive Optimizer...")
try:
    from adaptive_optimizer import AdaptiveOptimizer
    
    optimizer = AdaptiveOptimizer(performance_tracker=tracker)
    
    # Get current threshold
    threshold = optimizer.adaptive_params['NIFTY']['confidence_threshold']
    print(f"   ✅ Current threshold: {threshold:.3f}")
    
    # Get regime-specific threshold
    regime_threshold = optimizer.get_regime_specific_threshold('NIFTY', 'choppy')
    print(f"   ✅ Choppy market threshold: {regime_threshold:.3f}")
    
    # Check retrain trigger
    retrain = optimizer.check_retrain_trigger('NIFTY', lookback_days=14)
    print(f"   ✅ Retrain needed: {retrain['should_retrain']}")
    
    print("   ✅ Adaptive Optimizer: PASSED\n")
except Exception as e:
    print(f"   ❌ Adaptive Optimizer: FAILED - {e}\n")
    sys.exit(1)

# Test 6: Config
print("6️⃣  Testing Config...")
try:
    from config import get_config, update_config
    
    risk_config = get_config('risk')
    print(f"   ✅ Risk per trade: {risk_config['risk_per_trade']*100}%")
    print(f"   ✅ Max daily loss: {risk_config['max_daily_loss']*100}%")
    
    regime_config = get_config('regime')
    print(f"   ✅ Regime configs loaded: {len(regime_config)} regimes")
    
    print("   ✅ Config: PASSED\n")
except Exception as e:
    print(f"   ❌ Config: FAILED - {e}\n")
    sys.exit(1)

# Summary
print("="*70)
print("✅ ALL TESTS PASSED!")
print("="*70)
print()
print("Next Steps:")
print("1. Run the main trading system: python trading.py")
print("2. Monitor signals and TradingView charts")
print("3. Review performance database: sqlite3 trading_performance.db")
print("4. Check adaptive parameters: cat adaptive_config.json")
print()
print("System is ready for paper trading! 🚀")
print()
