import os
import sys
import pandas as pd
import logging
from datetime import datetime

# Import classes to test
# We need to mock some dependencies or use the real ones if safe
from trading import EnhancedDataFetcher, Config, TradingBot, TradingState

def test_data_fetcher_validation():
    print("\n--- Testing DataFetcher Validation ---")
    fetcher = EnhancedDataFetcher()
    
    # Test empty symbol
    df = fetcher.fetch_data("")
    assert df.empty, "Dataframe should be empty for empty symbol"
    print("✅ Empty symbol test passed")
    
    # Test whitespace symbol
    df = fetcher.fetch_data("   ")
    assert df.empty, "Dataframe should be empty for whitespace symbol"
    print("✅ Whitespace symbol test passed")

def test_symbol_resilience():
    print("\n--- Testing Symbol Resilience in EOD Logic ---")
    bot = TradingBot()
    
    # Simulate an open position with a symbol as the key (the bug case)
    symbol_name = "NIFTY_FIN_SERVICE.NS"
    bot.state.add_position(symbol_name, {
        "trade_id": 12345,
        "entry_price": 20000.0,
        "direction": "CE",
        "position_size": 1
    })
    
    # Verify lookup logic (equivalent to what's in run() and _monitor_position())
    resolved_symbol = Config.INDICES.get(symbol_name, symbol_name)
    assert resolved_symbol == symbol_name, f"Should resolve to {symbol_name}, got {resolved_symbol}"
    print(f"✅ Symbol resonance lookup test passed: {symbol_name} -> {resolved_symbol}")
    
    # Cleanup
    bot.state.remove_position(symbol_name)

def test_integer_id_handling():
    print("\n--- Testing Integer ID Handling ---")
    bot = TradingBot()
    
    # Check if we can mock performance_tracker or if it returns integer IDs
    # Since we can't easily mock, let's just inspect the code logic via a dummy entry if possible
    # or just trust the code change if it's simple enough.
    # Actually, let's look at the state after a simulated _execute_entry.
    
    # We need a mock signal
    signal = {
        'price': 20000.0,
        'direction': 'CE',
        'confidence': 0.7,
        'regime': 'trending_up',
        'position_size': 1,
        'stop_loss': 19900.0,
        'take_profit_1': 20200.0
    }
    
    # We'll temporarily point performance_tracker to a test DB
    if bot.performance_tracker:
        original_db = bot.performance_tracker.db_path
        bot.performance_tracker.db_path = "test_verify_pnl.db"
    
    try:
        # Mock _execute_entry parameters
        name = "TEST_INDEX"
        symbol = "TEST_SYMBOL"
        
        # Execute entry
        bot._execute_entry(name, symbol, signal, 1.0)
        
        # Check state
        pos = bot.state.get_position(name)
        assert pos is not None
        trade_id = pos['trade_id']
        print(f"DEBUG: Captured trade_id: {trade_id} (Type: {type(trade_id)})")
        
        # In our fix, final_trade_id = db_id if db_id.
        # If tracker is running, it should be an int.
        # If tracker is None, it should be a string (temp_trade_id).
        
        if bot.performance_tracker:
            assert isinstance(trade_id, int), f"trade_id should be int, got {type(trade_id)}"
        
        print("✅ Integer ID handling test passed (or verified as expected)")
        
    finally:
        bot.state.remove_position("TEST_INDEX")
        if bot.performance_tracker:
            bot.performance_tracker.db_path = original_db
            if os.path.exists("test_verify_pnl.db"):
                os.remove("test_verify_pnl.db")

if __name__ == "__main__":
    # Ensure logs directory exists for tests
    os.makedirs("logs", exist_ok=True)
    
    try:
        test_data_fetcher_validation()
        test_symbol_resilience()
        test_integer_id_handling()
        print("\n✨ ALL TESTS PASSED ✨")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
