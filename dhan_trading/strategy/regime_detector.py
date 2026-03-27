"""
Market Regime Detection System
Classifies market conditions to optimize strategy parameters
Preserved original logic per user request.
"""
import logging
import pandas as pd
from typing import Dict
try:
    from ta.trend import ADXIndicator, EMAIndicator
    from ta.volatility import AverageTrueRange
except ImportError:
    pass

class RegimeDetector:
    def __init__(self, adx_threshold: float = 25.0, lookback: int = 100):
        self.adx_threshold = adx_threshold
        self.lookback = lookback
        self.logger = logging.getLogger('RegimeDetector')

    def detect_regime(self, df: pd.DataFrame) -> str:
        """Detect current market regime"""
        if len(df) < self.lookback:
            self.logger.warning("Not enough data to detect regime")
            return 'unknown'
            
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            adx_ind = ADXIndicator(high=high, low=low, close=close, window=14)
            adx = adx_ind.adx().iloc[-1]
            di_pos = adx_ind.adx_pos().iloc[-1]
            di_neg = adx_ind.adx_neg().iloc[-1]
            
            ema20 = EMAIndicator(close=close, window=20).ema_indicator().iloc[-1]
            ema50 = EMAIndicator(close=close, window=50).ema_indicator().iloc[-1]
            ema200 = EMAIndicator(close=close, window=200).ema_indicator().iloc[-1]
            
            current_price = close.iloc[-1]
            
            # Simple Choppiness Logic
            atr = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
            atr_pct = atr.iloc[-1] / current_price * 100
            
            return self._classify_regime(adx, di_pos, di_neg, current_price, ema20, ema50, ema200, atr_pct)
        except Exception as e:
            self.logger.error(f"Error in regime detection: {e}")
            return 'unknown'

    def _classify_regime(self, adx, di_pos, di_neg, price, e20, e50, e200, atr_pct) -> str:
        if adx > self.adx_threshold:
            if di_pos > di_neg and price > e20 > e50 > e200:
                return 'trending_up'
            elif di_neg > di_pos and price < e20 < e50 < e200:
                return 'trending_down'
        
        if atr_pct > 0.3: # Volatile threshold
            return 'volatile'
            
        if adx < 20:
            return 'choppy' if atr_pct > 0.15 else 'sideways'
            
        return 'unknown'

    def get_regime_parameters(self, regime: str) -> Dict:
        """Get optimal param scaling"""
        params = {
            'trending_up': {'confidence_threshold': 0.50, 'adx_min': 25, 'description': "Strong uptrend"},
            'trending_down': {'confidence_threshold': 0.50, 'adx_min': 25, 'description': "Strong downtrend"},
            'sideways': {'confidence_threshold': 0.65, 'adx_min': 0, 'description': "Sideways market"},
            'choppy': {'confidence_threshold': 0.65, 'adx_min': 0, 'description': "Choppy market"},
            'volatile': {'confidence_threshold': 0.55, 'adx_min': 15, 'description': "High volatility"},
            'unknown': {'confidence_threshold': 0.55, 'adx_min': 12, 'description': "Unknown regime"}
        }
        return params.get(regime, params['unknown'])

    def get_regime_with_params(self, df: pd.DataFrame) -> Dict:
        regime = self.detect_regime(df)
        return {
            'regime': regime,
            'parameters': self.get_regime_parameters(regime)
        }
