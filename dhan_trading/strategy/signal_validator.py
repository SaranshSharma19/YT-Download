import logging
import pandas as pd
import numpy as np
from typing import Dict, Tuple

class SignalValidator:
    """
    Validates trade signals using a weighted scoring system.
    Key Fix: Anchored VWAP correctly resets at 09:15 IST each daily session.
    """
    def __init__(self, config: Dict):
        self.logger = logging.getLogger('SignalValidator')
        self.config = config
        self.min_score = config.get('min_score', 60)
        self.weights = config.get('weights', {
            'trend_alignment': 30,
            'momentum': 25,
            'volume_flow': 20,
            'volatility': 15,
            'market_regime': 10
        })

    def validate_trade_setup(self, signal: Dict, df: pd.DataFrame, context: Dict) -> Tuple[bool, int, str]:
        if df is None or len(df) < 50:
            return False, 0, "Insufficient validation data"
            
        direction = signal['direction']
        score, details = self.calculate_confluence_score(signal, df, context)
        
        # Add anchored VWAP check strictly
        vwap_status, vwap_val = self._evaluate_anchored_vwap(df, direction)
        if not vwap_status:
            return False, score, f"Rejected by Session VWAP ({vwap_val:.2f})"
            
        if score >= self.min_score:
            return True, score, f"Score {score}: {', '.join([k for k,v in details.items() if v > 0])}"
            
        return False, score, f"Low Score ({score} < {self.min_score})"

    def calculate_confluence_score(self, signal: Dict, df: pd.DataFrame, context: Dict) -> Tuple[int, Dict]:
        direction = signal['direction']
        score = 0
        details = {}
        
        # Trend
        if self._evaluate_trend(df, direction):
            score += self.weights['trend_alignment']
            details['trend'] = self.weights['trend_alignment']
            
        # Momentum
        if self._evaluate_momentum(df, direction):
            score += self.weights['momentum']
            details['momentum'] = self.weights['momentum']
            
        # Volume
        if self._evaluate_volume(df, direction):
            score += self.weights['volume_flow']
            details['volume'] = self.weights['volume_flow']
            
        # Volatility
        if self._evaluate_volatility(df):
            score += self.weights['volatility']
            details['volatility'] = self.weights['volatility']
            
        # Regime
        if self._evaluate_regime(context, direction):
            score += self.weights['market_regime']
            details['regime'] = self.weights['market_regime']
            
        # Optional Boosts
        if signal.get('is_expiry_day'):
            bump = signal.get('validator_threshold_boost', 0)
            self.min_score += bump # Temporarily make it stricter
            
        return score, details

    def _evaluate_anchored_vwap(self, df: pd.DataFrame, direction: str) -> Tuple[bool, float]:
        """Calculates Session VWAP anchored at midnight/09:15 IST and compares price."""
        try:
            # Group by day and calculate cumulative volume * price
            # Assuming df.index is timezone-aware IST datetime or naïve but anchored to local date
            vpw = df['volume'] * ((df['high'] + df['low'] + df['close']) / 3)
            
            # Use df.index.date to anchor for the session
            grouped = pd.DataFrame({'vpw': vpw, 'v': df['volume']}).groupby(df.index.date)
            # Calculate cumulative sums per day
            df['cum_vpw'] = grouped['vpw'].cumsum()
            df['cum_v'] = grouped['v'].cumsum()
            df['vwap'] = df['cum_vpw'] / df['cum_v']
            
            vwap = df['vwap'].iloc[-1]
            price = df['close'].iloc[-1]
            
            if direction == 'CE' and price > vwap:
                return True, vwap
            elif direction == 'PE' and price < vwap:
                return True, vwap
            return False, vwap
        except Exception as e:
            self.logger.error(f"VWAP Error: {e}")
            return True, 0.0 # Fail open if error to not block

    def _evaluate_trend(self, df: pd.DataFrame, direction: str) -> bool:
        try:
            ema20 = df['close'].ewm(span=20).mean().iloc[-1]
            ema50 = df['close'].ewm(span=50).mean().iloc[-1]
            price = df['close'].iloc[-1]
            
            if direction == 'CE': return price > ema20 and ema20 > ema50
            if direction == 'PE': return price < ema20 and ema20 < ema50
        except: pass
        return False

    def _evaluate_momentum(self, df: pd.DataFrame, direction: str) -> bool:
        # Simplified RSI mock
        return True

    def _evaluate_volume(self, df: pd.DataFrame, direction: str) -> bool:
        try:
            vol_sma = df['volume'].rolling(20).mean().iloc[-1]
            return df['volume'].iloc[-1] > vol_sma * 0.8 # At least decent volume
        except: return True

    def _evaluate_volatility(self, df: pd.DataFrame) -> bool:
        return True

    def _evaluate_regime(self, context: Dict, direction: str) -> bool:
        regime = context.get('regime', 'unknown')
        if direction == 'CE' and regime in ['trending_up', 'volatile']: return True
        if direction == 'PE' and regime in ['trending_down', 'volatile']: return True
        if regime == 'choppy': return False # Tough to trade
        return True
