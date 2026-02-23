"""
Market Regime Detection System
Classifies market conditions to optimize strategy parameters
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional

try:
    from ta.trend import ADXIndicator, EMAIndicator
    from ta.volatility import AverageTrueRange
except ImportError:
    print("Warning: 'ta' library not found. Install with: pip install ta")
    ADXIndicator = None
    EMAIndicator = None
    AverageTrueRange = None


class RegimeDetector:
    """
    Detects market regime using multiple indicators:
    - Trend strength (ADX)
    - Choppiness Index (CHOP)
    - Directional bias (EMA alignment)
    - Volatility (ATR percentile)
    
    Regimes:
    - trending_up: Strong uptrend
    - trending_down: Strong downtrend
    - choppy: Low volatility, no clear trend
    - volatile: High volatility, unclear direction
    """
    
    def __init__(self, adx_threshold: float = 25.0, lookback: int = 100):
        self.logger = logging.getLogger('RegimeDetector')
        self.adx_threshold = adx_threshold
        self.lookback = lookback
    
    @staticmethod
    def calculate_choppiness_index(df: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate Choppiness Index (0-100)
        100++ = Very choppy
        < 38.2 = Trending
        """
        try:
            # True Range Calculation
            df = df.copy()
            df['tr0'] = abs(df['High'] - df['Low'])
            df['tr1'] = abs(df['High'] - df['Close'].shift(1))
            df['tr2'] = abs(df['Low'] - df['Close'].shift(1))
            df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
            
            atr_sum = df['tr'].rolling(window=period).sum()
            
            max_high = df['High'].rolling(window=period).max()
            min_low = df['Low'].rolling(window=period).min()
            
            denom = max_high - min_low
            # Avoid division by zero
            denom = denom.replace(0, np.nan).fillna(1e-9)
            
            # Crypto/Stock standard formula: 100 * LOG10( SUM(ATR, n) / ( MaxHi(n) - MinLo(n) ) ) / LOG10(n)
            chop = 100.0 * np.log10(atr_sum / denom) / np.log10(period)
            
            return float(chop.iloc[-1])
        except Exception as e:
            logging.getLogger('RegimeDetector').warning(f"Choppiness calculation failed: {e}")
            return 50.0

    def detect_regime(self, df: pd.DataFrame) -> str:
        """
        Detect current market regime
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            Regime string: 'trending_up', 'trending_down', 'choppy', 'volatile', 'sideways'
        """
        if df.empty or len(df) < 50:
            self.logger.warning("Insufficient data for regime detection")
            return 'unknown'
        
        if ADXIndicator is None:
            self.logger.warning("ta library not available, returning unknown regime")
            return 'unknown'
        
        try:
            # Calculate indicators
            close = df['Close']
            high = df['High']
            low = df['Low']
            
            # Trend strength (ADX)
            adx_indicator = ADXIndicator(high, low, close, window=14, fillna=True)
            adx = adx_indicator.adx().iloc[-1]
            
            # Directional indicators
            di_pos = adx_indicator.adx_pos().iloc[-1]
            di_neg = adx_indicator.adx_neg().iloc[-1]
            
            # EMAs for trend direction
            ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
            ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
            ema200 = close.ewm(span=200, adjust=False).mean().iloc[-1]
            
            current_price = close.iloc[-1]
            
            # Slope of EMA 50 (to detect flat markets)
            ema50_series = close.ewm(span=50, adjust=False).mean()
            ema50_slope = (ema50_series.iloc[-1] - ema50_series.iloc[-5]) / ema50_series.iloc[-5] * 100
            
            # Volatility (ATR percentile)
            atr_indicator = AverageTrueRange(high, low, close, window=14, fillna=True)
            atr = atr_indicator.average_true_range()
            
            # Calculate ATR percentile
            atr_current = atr.iloc[-1]
            atr_lookback = atr.iloc[-self.lookback:] if len(atr) >= self.lookback else atr
            atr_percentile = (atr_lookback < atr_current).sum() / len(atr_lookback) * 100
            
            # Choppiness Index
            choppiness = self.calculate_choppiness_index(df)
            
            # Regime classification logic
            regime = self._classify_regime(
                adx=adx,
                choppiness=choppiness,
                di_pos=di_pos,
                di_neg=di_neg,
                current_price=current_price,
                ema20=ema20,
                ema50=ema50,
                ema200=ema200,
                ema50_slope=ema50_slope,
                atr_percentile=atr_percentile
            )
            
            self.logger.debug(
                f"Regime: {regime} | ADX: {adx:.1f} | CHOP: {choppiness:.1f} | "
                f"EMA50 Slope: {ema50_slope:.4f}%"
            )
            
            return regime
            
        except Exception as e:
            self.logger.error(f"Regime detection failed: {e}")
            return 'unknown'
    
    def _classify_regime(
        self,
        adx: float,
        choppiness: float,
        di_pos: float,
        di_neg: float,
        current_price: float,
        ema20: float,
        ema50: float,
        ema200: float,
        ema50_slope: float,
        atr_percentile: float,
    ) -> str:
        """
        Classify regime based on multiple conditions.
        Prioritizes: 
        1. ADX for Trend Strength
        2. EMA alignment for Direction
        3. Choppiness Index for Ranging
        4. Volatility for specific 'volatile' state
        """
        
        # 1. Strong Trend Definition
        # ADX > 25 AND Choppiness < 50
        is_trending = (adx > 25) and (choppiness < 50)
        
        # 2. Sideways / Flat Definition
        # Low ADX OR High Choppiness OR Flat EMA Slope
        is_flat_slope = abs(ema50_slope) < 0.05 # Very flat slope
        is_sideways = (adx < 20) or (choppiness > 60) or is_flat_slope
        
        # 3. Directional Bias
        current_ema9 = df['Close'].ewm(span=9, adjust=False).mean().iloc[-1] if 'df' in locals() else ema20 # Optional: compute EMA9 if not passed, but we'll stick to mostly existing. We'll use EMA20 as proxy for short term.
        bullish_structure = (current_price > ema20) and (ema20 > ema50)
        bearish_structure = (current_price < ema20) and (ema20 < ema50)

        # 4. Crash/Dump detection
        is_dumping = (current_price < ema20) and (ema20 < ema50) and (adx > 30)

        # Classification Tree
        
        if is_dumping:
            return 'trending_down'
        
        if is_trending:
            if bullish_structure and di_pos > di_neg:
                return 'trending_up'
            elif bearish_structure and di_neg > di_pos:
                return 'trending_down'
            else:
                # Strong ADX but mixed EMAs -> Volatile Transition?
                return 'volatile'
                
        if atr_percentile > 85:
            return 'volatile' # High volatility regime regardless of trend
            
        if is_sideways:
            # If price is bouncing around EMA200
            dist_to_200 = abs(current_price - ema200) / ema200
            if dist_to_200 < 0.01: # Close to EMA200 usually means chop/accumulation
                return 'sideways'
            return 'choppy'

        # Default fallback
        return 'choppy'
    
    def get_regime_parameters(self, regime: str) -> Dict:
        """
        Get optimal parameters for each regime
        
        Returns:
            Dict with threshold, adx_min, num_filters
        """
        regime_params = {
            'trending_up': {
                'confidence_threshold': 0.55, # Slightly higher confidence required
                'adx_min': 20,
                'num_filters': 2,
                'description': 'Strong uptrend - Trend following'
            },
            'trending_down': {
                'confidence_threshold': 0.55,
                'adx_min': 20,
                'num_filters': 2,
                'description': 'Strong downtrend - Trend following'
            },
            'choppy': {
                'confidence_threshold': 0.70, # High confidence for choppy markets
                'adx_min': 0,
                'num_filters': 4,
                'description': 'Choppy market - High strictness'
            },
            'volatile': {
                'confidence_threshold': 0.65,
                'adx_min': 15,
                'num_filters': 3,
                'description': 'High volatility - Moderate strictness'
            },
            'unknown': {
                'confidence_threshold': 0.60,
                'adx_min': 15,
                'num_filters': 3,
                'description': 'Unknown regime - Conservative'
            }
        }
        
        return regime_params.get(regime, regime_params['unknown'])
    
    def get_regime_with_params(self, df: pd.DataFrame) -> Dict:
        """
        Detect regime and return with optimal parameters
        
        Returns:
            Dict with regime, parameters, and description
        """
        regime = self.detect_regime(df)
        params = self.get_regime_parameters(regime)
        
        return {
            'regime': regime,
            'parameters': params,
            'timestamp': pd.Timestamp.now()
        }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    import yfinance as yf
    
    detector = RegimeDetector()
    
    # Fetch sample data
    print("Fetching data for test...")
    df = yf.download("^NSEI", period="60d", interval="5m", progress=False)
    
    if not df.empty:
        # Fix for MultiIndex columns if present (yfinance update)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
            
        result = detector.get_regime_with_params(df)
        print(f"\nRegime Detection Result:")
        print(f"  Regime: {result['regime']}")
        print(f"  Description: {result['parameters']['description']}")
        print(f"  Confidence Threshold: {result['parameters']['confidence_threshold']}")
