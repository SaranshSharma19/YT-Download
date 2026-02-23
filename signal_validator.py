"""
Signal Validator / Decision Engine
Evaluates trade signals based on a multi-factor scoring system.
prioritizes precision over frequency.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

class SignalValidator:
    """
    Validates trade signals using a weighted scoring system.
    Factors:
    1. Trend Alignment (MTF)
    2. Momentum confluence (RSI, MACD)
    3. Volume/Price Action (VPA)
    4. Key Level Interaction (S/R)
    5. Volatility Context
    """
    
    def __init__(self, config: Dict = None):
        self.logger = logging.getLogger('SignalValidator')
        self.config = config or {}
        
        # Default scoring weights if not provided in config
        self.weights = self.config.get('weights', {
            'trend_alignment': 25,
            'momentum': 20,
            'volume_flow': 15,
            'volatility': 10,
            'market_regime': 10,
            'gap_analysis': 20
        })
        
        self.min_score = self.config.get('min_score', 70)  # Minimum score to take a trade

    def calculate_confluence_score(self, signal: Dict, df: pd.DataFrame, context: Dict) -> Tuple[float, Dict]:
        """
        Calculate a confidence score (0-100) for the trade.
        
        Args:
            signal: The base signal dictionary (direction, price, etc.)
            df: The dataframe with OHLCV and indicators
            context: Market regime/context dictionary
            
        Returns:
            Tuple(score, details_dict)
        """
        if df.empty:
            return 0.0, {'error': 'empty_data'}
            
        direction = signal.get('direction')
        current_price = signal.get('price')
        
        if not direction or not current_price:
            return 0.0, {'error': 'invalid_signal'}

        score = 0.0
        details = {}
        
        # 1. Trend Alignment (30 points)
        trend_score = self._evaluate_trend(df, direction)
        score += trend_score * (self.weights['trend_alignment'] / 100)
        details['trend_score'] = trend_score

        # 2. Momentum (25 points)
        momentum_score = self._evaluate_momentum(df, direction)
        score += momentum_score * (self.weights['momentum'] / 100)
        details['momentum_score'] = momentum_score

        # 3. Volume Flow (20 points)
        volume_score = self._evaluate_volume(df, direction)
        score += volume_score * (self.weights['volume_flow'] / 100)
        details['volume_score'] = volume_score

        # 4. Volatility / ATR (15 points)
        volatility_score = self._evaluate_volatility(df, context)
        score += volatility_score * (self.weights['volatility'] / 100)
        details['volatility_score'] = volatility_score
        
        # 5. Regime/Context Bonus (10 points)
        regime_score = self._evaluate_regime(context, direction)
        score += regime_score * (self.weights['market_regime'] / 100)
        details['regime_score'] = regime_score

        # 6. Gap Analysis (20 points)
        gap_score = self._evaluate_gap(df, direction)
        score += gap_score * (self.weights['gap_analysis'] / 100)
        details['gap_score'] = gap_score

        # Normalize score to 0-100 (sum of weights should be 100, but ensuring)
        total_weight = sum(self.weights.values())
        final_score = (score / (total_weight / 100)) if total_weight > 0 else 0
        
        details['final_score'] = round(final_score, 2)
        
        self.logger.info(f"Signal Validation Score: {final_score:.1f} | Details: {details}")
        
        return final_score, details

    def _evaluate_trend(self, df: pd.DataFrame, direction: str) -> float:
        """Evaluate alignment with longer-term Moving Averages."""
        score = 0.0
        try:
            close = df['Close']
            ema50 = close.ewm(span=50, adjust=False).mean()
            ema200 = close.ewm(span=200, adjust=False).mean()
            
            curr_close = close.iloc[-1]
            curr_ema50 = ema50.iloc[-1]
            curr_ema200 = ema200.iloc[-1]

            # Dynamic Intraday VWAP check (approximation if actual VWAP not strictly present directly)
            # If we don't have VWAP as series, we approximate with ema9/20 to be safe
            ema9 = close.ewm(span=9, adjust=False).mean()
            curr_ema9 = ema9.iloc[-1]
            
            # Strong Trend Alignment
            if direction == 'CE':
                if curr_close > curr_ema50: score += 50
                if curr_close > curr_ema200: score += 30
                if curr_ema50 > curr_ema200: score += 20 # Golden cross alignment
                
                # Circuit Breaker: Crash filter
                if curr_close < curr_ema9 * 0.998: # 0.2% below short-term EMA
                     score -= 80 # Heavy penalty for catching falling knives
            elif direction == 'PE':
                if curr_close < curr_ema50: score += 50
                if curr_close < curr_ema200: score += 30
                if curr_ema50 < curr_ema200: score += 20 # Death cross alignment
                
                # Circuit Breaker: Spike filter
                if curr_close > curr_ema9 * 1.002: # 0.2% above short-term EMA
                     score -= 80 # Heavy penalty for fighting spikes
                
        except Exception as e:
            self.logger.warning(f"Trend evaluation failed: {e}")
            
        return min(max(score, 0.0), 100.0)

    def _evaluate_momentum(self, df: pd.DataFrame, direction: str) -> float:
        """Evaluate RSI and MACD momentum."""
        score = 0.0
        try:
            # RSI
            if 'momentum_rsi' in df.columns:
                rsi = df['momentum_rsi'].iloc[-1]
            else:
                # Calculate RSI if missing
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs)).iloc[-1]

            # MACD
            if 'trend_macd_diff' not in df.columns:
                close = df['Close']
                ema12 = close.ewm(span=12, adjust=False).mean()
                ema26 = close.ewm(span=26, adjust=False).mean()
                macd = ema12 - ema26
                signal = macd.ewm(span=9, adjust=False).mean()
                macd_hist = macd - signal
                curr_hist = macd_hist.iloc[-1]
                prev_hist = macd_hist.iloc[-2]
            else:
                curr_hist = df['trend_macd_diff'].iloc[-1]
                prev_hist = df['trend_macd_diff'].iloc[-2]

            if direction == 'CE':
                if 40 <= rsi <= 70: score += 40  # Healthy bullish RSI
                if rsi > 50: score += 20
                if curr_hist > 0: score += 20
                if curr_hist > prev_hist: score += 20 # Growing momentum
            elif direction == 'PE':
                if 30 <= rsi <= 60: score += 40 # Healthy bearish RSI
                if rsi < 50: score += 20
                if curr_hist < 0: score += 20
                if curr_hist < prev_hist: score += 20 # Growing bearish momentum

        except Exception as e:
            self.logger.warning(f"Momentum evaluation failed: {e}")
        
        return min(score, 100.0)

    def _evaluate_volume(self, df: pd.DataFrame, direction: str) -> float:
        """Evaluate Volume Price Analysis."""
        score = 50.0 # Start neutral
        try:
            # Check last 3 bars for volume trend
            recent_vol = df['Volume'].iloc[-3:]
            recent_close = df['Close'].iloc[-3:]
            vol_ma = df['Volume'].rolling(20).mean().iloc[-1]
            
            current_vol = recent_vol.iloc[-1]
            
            if current_vol > vol_ma:
                # High volume supports the move
                if direction == 'CE' and recent_close.iloc[-1] > recent_close.iloc[-2]:
                    score += 30
                elif direction == 'PE' and recent_close.iloc[-1] < recent_close.iloc[-2]:
                    score += 30
            elif current_vol < vol_ma * 0.5:
                # Low volume weakens the signal
                score -= 20
                
        except Exception as e:
            self.logger.warning(f"Volume evaluation failed: {e}")
            
        return min(max(score, 0.0), 100.0)

    def _evaluate_volatility(self, df: pd.DataFrame, context: Dict) -> float:
        """Evaluate if volatility is sufficient for a trade."""
        score = 50.0
        try:
            # Prefer expansion, penalize contraction if too low
            bb_width = 0
            if 'volatility_bbw' in df.columns:
                 bb_width = df['volatility_bbw'].iloc[-1]
            
            # Use ATR if available
            atr = 0
            if 'volatility_atr' in df.columns:
                 atr = df['volatility_atr'].iloc[-1]
            
            # If volatility is "Healthy" (not too low, not extreme)
            start_score = 50
            
            # Context-based adjustment
            regime = context.get('regime', 'unknown')
            if regime == 'trending_up' or regime == 'trending_down':
                score += 20 # Trends are good
            elif regime == 'choppy':
                score -= 30 # Warning
            
            # ATR threshold check
            price = df['Close'].iloc[-1]
            if atr > 0:
                atr_pct = (atr / price) * 100
                if atr_pct > 0.15: # Min 0.15% movement per bar (5m)
                    score += 20
                elif atr_pct < 0.05: # Dead market
                    score = 0
                    
        except Exception as e:
             self.logger.warning(f"Volatility evaluation failed: {e}")
             
        return min(max(score, 0.0), 100.0)

    def _evaluate_regime(self, context: Dict, direction: str) -> float:
        """Evaluate alignment with overall market regime."""
        score = 50.0
        regime = context.get('regime', 'unknown')
        
        if regime == 'trending_up':
            if direction == 'CE': score = 100
            elif direction == 'PE': score = 20 # Counter-trend
        elif regime == 'trending_down':
            if direction == 'PE': score = 100
            elif direction == 'CE': score = 20 # Counter-trend
        elif regime == 'volatile':
            score = 40 # Caution
        elif regime == 'choppy':
            score = 10 # Danger
            
        return score

    def _evaluate_gap(self, df: pd.DataFrame, direction: str) -> float:
        """Evaluate alignment with the day's opening gap."""
        score = 50.0  # Default neutral
        try:
            # We need daily data or compare current day open vs prev day close
            # Assuming df has 'Open' and we can infer day boundaries or just use recent large moves
            # For simplicity in this intraday context, let's look at the open of the current session
            # If we don't have day breaks easily, we'll check the biggest candle in the last N bars or simply
            # check the relation to the first bar of the dataset if it represents the day.
            
            # Robust approach: Calculate the gap between the latest bar's current price and the 'Open' of the day
            # If we can't determine Day Open easily, use the relative change from the start of the dataframe 
            # (assuming df is fetched for recent period)
            
            if df.empty: return 50.0
            
            # Heuristic: Compare current close to the rolling mean of the last 50 bars
            # If we are effectively "Gapped Down" (Current << SMA50), bias PE
            
            curr_price = df['Close'].iloc[-1]
            sma50 = df['Close'].rolling(50).mean().iloc[-1]
            
            # If we are > 0.5% away from SMA50, consider it a trend/gap bias
            dist_pct = (curr_price - sma50) / sma50 * 100
            
            if dist_pct < -0.5: # Significant downside deviation (Gap down / Crash)
                if direction == 'PE': score = 100
                elif direction == 'CE': score = 10 # Fighting the drop
            elif dist_pct > 0.5: # Significant upside
                if direction == 'CE': score = 100
                elif direction == 'PE': score = 10
            else:
                score = 50 # Neutral
                
        except Exception as e:
            self.logger.warning(f"Gap/Deviation evaluation failed: {e}")
            
        return score

    def validate_trade_setup(self, signal: Dict, df: pd.DataFrame, context: Dict) -> Tuple[bool, float, str]:
        """
        Main entry point to validate a trade.
        Returns: (is_valid, score, reason)
        """
        score, details = self.calculate_confluence_score(signal, df, context)
        
        # Dynamic threshold based on regime
        threshold = self.min_score
        regime = context.get('regime', 'unknown')
        if regime == 'choppy':
            threshold += 15 # Required higher score in chop
        elif regime in ['trending_up', 'trending_down']:
            threshold -= 5 # Slightly lower threshold in strong trends (follow the flow)
            
        is_valid = score >= threshold
        
        reason = f"Score {score:.1f}/{threshold} | " + \
                 f"Trend:{details.get('trend_score')} Mom:{details.get('momentum_score')} " + \
                 f"Vol:{details.get('volume_score')}"
                 
        return is_valid, score, reason
