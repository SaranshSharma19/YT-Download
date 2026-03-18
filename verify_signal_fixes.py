from signal_validator import SignalValidator
import pandas as pd
import numpy as np

validator = SignalValidator({'min_score': 70})

# Create a mock dataframe representing an early reversal (down)
# Price has dropped significantly (so Trend EMA50 favors shorts),
# but lagging indicators like MTF (15min) and VWAP are still bullish
df = pd.DataFrame(index=pd.date_range("2026-03-05 10:00", periods=50, freq="5min"))
df['Close'] = np.linspace(25000, 24600, 50)  # Dropping price
df['Open'] = df['Close'] + 10
df['High'] = df['Close'] + 20
df['Low'] = df['Close'] - 10
df['Volume'] = np.random.randint(1000, 5000, size=50)

# Simulate momentum pointing down
df['momentum_rsi'] = np.linspace(60, 30, 50)
df['trend_macd_diff'] = np.linspace(5, -10, 50)

# Simulate VWAP lagging behind (VWAP > Price means bearish, but we need to trick the evaluator
# into giving a bad VWAP score to test the bypass. If VWAP is FAR below price, score=20 for PE)
df['Volume'] = 1000 # constant vol
# Make typical price artificially low early on so VWAP is low
tp = (df['High'] + df['Low'] + df['Close']) / 3
vwap = tp.expanding().mean()
# Override VWAP to force a low score for PE (Price < VWAP gives high score, so let's make Price > VWAP artificially inside the dummy)
df['Close'].iloc[-1] = 25000 # spike to ruin VWAP

# Actually, the easiest way to test the exact bypass logic:
# if trend_score == 50 and mtf_score <= 20 and vwap_score <= 20:
# and momentum_score >= 40 and volume_score >= 50

class MockValidator(SignalValidator):
    def _evaluate_trend(self, df, d): return 50.0  # Decent trend, not perfect
    def _evaluate_higher_tf(self, df, d): return 10.0 # Lagging MTF
    def _evaluate_vwap(self, df, d): return 15.0 # Lagging VWAP
    def _evaluate_momentum(self, df, d): return 45.0 # Good momentum
    def _evaluate_volume(self, df, d): return 60.0 # Good volume
    def _evaluate_volatility(self, df, c): return 50.0
    def _evaluate_gap(self, df, d): return 50.0
    def _evaluate_regime(self, c, d): return 50.0

print("Testing Early Reversal Scenario...")
mock_val = MockValidator({'min_score': 70})
# The base score without boost:
# Trend (20%): 50 * 0.2 = 10
# Mom (15%): 45 * 0.15 = 6.75
# Vol (10%): 60 * 0.1 = 6.0
# Volat (8%): 50 * 0.08 = 4.0
# Regime (7%): 50 * 0.07 = 3.5
# Gap (15%): 50 * 0.15 = 7.5
# MTF (15%): 10 * 0.15 = 1.5
# VWAP (10%): 15 * 0.1 = 1.5
# Total base = 10+6.75+6.0+4.0+3.5+7.5+1.5+1.5 = 40.75
# With +15 boost: 55.75
# Note: Regime 'trending_down' combined with early_reversal_boost lowers threshold by 5 (to 65)

context = {'regime': 'trending_down'}
signal = {'direction': 'PE', 'price': 24000}
is_valid, score, reason = mock_val.validate_trade_setup(signal, df, context)

print(f"Valid: {is_valid}, Score: {score}, Reason: {reason}")


class MockExhausted(SignalValidator):
    def _evaluate_trend(self, df, d): return 100.0  # Perfect trend
    def _evaluate_higher_tf(self, df, d): return 90.0 # Perfect MTF
    def _evaluate_vwap(self, df, d): return 10.0 # Price heavily overextended from VWAP
    def _evaluate_momentum(self, df, d): return 30.0 # Momentum dying
    def _evaluate_volume(self, df, d): return 50.0
    def _evaluate_volatility(self, df, c): return 50.0
    def _evaluate_gap(self, df, d): return 50.0
    def _evaluate_regime(self, c, d): return 100.0

print("\nTesting Mature Exhaustion Scenario...")
mock_ex = MockExhausted({'min_score': 70})
is_valid_ex, score_ex, reason_ex = mock_ex.validate_trade_setup(signal, df, context)
print(f"Valid: {is_valid_ex}, Score: {score_ex}, Reason: {reason_ex}")
