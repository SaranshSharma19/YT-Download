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
            'trend_alignment': 20,
            'momentum': 15,
            'volume_flow': 10,
            'volatility': 8,
            'market_regime': 7,
            'gap_analysis': 15,
            'mtf_confirmation': 15,
            'vwap_filter': 10
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

        # 6. Gap Analysis
        gap_score = self._evaluate_gap(df, direction)
        score += gap_score * (self.weights['gap_analysis'] / 100)
        details['gap_score'] = gap_score

        # 7. Multi-Timeframe Confirmation (15-min trend)
        mtf_score = self._evaluate_higher_tf(df, direction)
        score += mtf_score * (self.weights['mtf_confirmation'] / 100)
        details['mtf_score'] = mtf_score

        # 8. VWAP Directional Filter
        vwap_score = self._evaluate_vwap(df, direction)
        score += vwap_score * (self.weights['vwap_filter'] / 100)
        details['vwap_score'] = vwap_score

        # Normalize score to 0-100 (sum of weights should be 100, but ensuring)
        total_weight = sum(self.weights.values())
        final_score = (score / (total_weight / 100)) if total_weight > 0 else 0
        
        # EARLY REVERSAL BOOST:
        # If trend alignment is decent (e.g., crossing 50EMA but not yet 200EMA) 
        # but lagging MTF and VWAP are dragging the score down, boost it.
        # This prevents missing the most profitable early moves of a new trend.
        if trend_score == 50 and mtf_score <= 30 and vwap_score <= 30:
            if momentum_score >= 40 and volume_score >= 50:
                self.logger.info("Early Reversal Detected: Boosting score by +15 points")
                final_score += 15
                details['early_reversal_boost'] = True

        # MATURE TREND EXHAUSTION PENALTY:
        # If all lagging indicators (Trend, MTF) are perfect, but price has pulled
        # drastically away from VWAP and Momentum is dying, it's likely a trap (exhaustion).
        if trend_score >= 80 and mtf_score >= 80 and vwap_score <= 20:
            if momentum_score <= 40:
                self.logger.info("Mature Trend Exhaustion Detected: Penalizing score by -15 points")
                final_score -= 15
                details['exhaustion_penalty'] = True

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

            # Dynamic Intraday VWAP/StdDev check
            ema9 = close.ewm(span=9, adjust=False).mean()
            curr_ema9 = ema9.iloc[-1]
            std9 = close.rolling(9).std().iloc[-1]
            
            # Strong Trend Alignment
            if direction == 'CE':
                if curr_close > curr_ema50: score += 50
                if curr_close > curr_ema200: score += 30
                if curr_ema50 > curr_ema200: score += 20 # Golden cross alignment
                
                # Circuit Breaker: Crash filter
                if curr_close < curr_ema9 - std9: # Dropping past standard deviation
                     score -= 80 # Heavy penalty for catching falling knives
            elif direction == 'PE':
                if curr_close < curr_ema50: score += 50
                if curr_close < curr_ema200: score += 30
                if curr_ema50 < curr_ema200: score += 20 # Death cross alignment
                
                # Circuit Breaker: Spike filter
                if curr_close > curr_ema9 + std9: # Spiking rapidly above standard deviation
                     score -= 80 # Heavy penalty for fighting spikes
                
        except Exception as e:
            self.logger.warning(f"Trend evaluation failed: {e}")
            
        return min(max(score, 0.0), 100.0)

    def _evaluate_momentum(self, df: pd.DataFrame, direction: str) -> float:
        """Evaluate RSI and MACD momentum, including exhaustion/divergence detection."""
        score = 0.0
        try:
            # RSI
            if 'momentum_rsi' in df.columns:
                rsi = df['momentum_rsi'].iloc[-1]
                rsi_prev = df['momentum_rsi'].iloc[-6] if len(df) >= 6 else rsi
            else:
                # Calculate RSI if missing
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi_series = 100 - (100 / (1 + rs))
                rsi = rsi_series.iloc[-1]
                rsi_prev = rsi_series.iloc[-6] if len(rsi_series) >= 6 else rsi

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

            # Fix 7: RSI Divergence / Exhaustion Detection
            # Detects when price trend is dying — prevents entering at trend exhaustion
            if len(df) >= 6:
                price_now = df['Close'].iloc[-1]
                price_prev = df['Close'].iloc[-6]
                
                if direction == 'PE':
                    # Price making new lows but RSI making higher lows → bearish exhaustion
                    if price_now < price_prev and rsi > rsi_prev:
                        score -= 30  # Penalty for shorting into exhaustion
                        self.logger.debug("Bearish exhaustion detected (RSI divergence)")
                elif direction == 'CE':
                    # Price making new highs but RSI making lower highs → bullish exhaustion
                    if price_now > price_prev and rsi < rsi_prev:
                        score -= 30  # Penalty for buying into exhaustion
                        self.logger.debug("Bullish exhaustion detected (RSI divergence)")

            if direction == 'CE':
                if 40 <= rsi <= 70: score += 40  # Healthy bullish RSI
                if rsi > 50: score += 20
                if curr_hist > 0: score += 20
                if curr_hist > prev_hist: score += 20 # Growing momentum
                if rsi > 75: score -= 50 # Overbought
                
            elif direction == 'PE':
                if 30 <= rsi <= 60: score += 40 # Healthy bearish RSI
                if rsi < 50: score += 20
                if curr_hist < 0: score += 20
                if curr_hist < prev_hist: score += 20 # Growing bearish momentum
                if rsi < 25: score -= 80 # Severely oversold: Do not sell the bottom!

        except Exception as e:
            self.logger.warning(f"Momentum evaluation failed: {e}")
        
        return min(max(score, 0.0), 100.0)

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
            
            # If we are > 0.5% away from SMA50, consider it over-extended (Mean Reversion)
            dist_pct = (curr_price - sma50) / sma50 * 100
            
            if dist_pct < -0.4: # Significant downside deviation (Oversold/Crash)
                if direction == 'CE': score = 80 # Support bounce
                elif direction == 'PE': score -= 80 # Fighting extension, heavily dangerous
            elif dist_pct > 0.4: # Significant upside (Overbought)
                if direction == 'PE': score = 80 # Resistance rejection
                elif direction == 'CE': score -= 80 # Fighting extension, heavily dangerous
            elif dist_pct < -0.1: # Moderate downtrend
                if direction == 'PE': score = 70
                elif direction == 'CE': score = 30
            elif dist_pct > 0.1: # Moderate uptrend
                if direction == 'CE': score = 70
                elif direction == 'PE': score = 30
            else:
                score = 50 # Neutral
                
        except Exception as e:
            self.logger.warning(f"Gap/Deviation evaluation failed: {e}")
            
        return score

    def _evaluate_higher_tf(self, df: pd.DataFrame, direction: str) -> float:
        """Evaluate alignment with 15-minute trend (Multi-Timeframe Confirmation)."""
        score = 50.0  # Neutral default
        try:
            if len(df) < 15:
                return score

            # Resample 5-min bars into 15-min bars
            df_15m = df[['Open', 'High', 'Low', 'Close', 'Volume']].resample('15min').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()

            if len(df_15m) < 10:
                return score

            # 15-min EMA9 slope (last 3 bars)
            ema9_15m = df_15m['Close'].ewm(span=9, adjust=False).mean()
            slope = (ema9_15m.iloc[-1] - ema9_15m.iloc[-3]) / ema9_15m.iloc[-3] * 100

            if direction == 'CE':
                if slope > 0.05: score = 90  # 15-min uptrend confirms CE
                elif slope < -0.05: score = 40  # Softened from 20 to prevent over-rejection
            elif direction == 'PE':
                if slope < -0.05: score = 90  # 15-min downtrend confirms PE
                elif slope > 0.05: score = 40  # Softened from 20 to prevent over-rejection

        except Exception as e:
            self.logger.warning(f"MTF evaluation failed: {e}")

        return min(max(score, 0.0), 100.0)

    def _evaluate_vwap(self, df: pd.DataFrame, direction: str) -> float:
        """A-4: Evaluate alignment with intraday VWAP (today's session only).

        Previous implementation computed VWAP over the entire DataFrame (up to 30 days),
        which produced a meaningless long-term average instead of the intraday VWAP.
        Now correctly filters to the current IST trading day before cumulating.
        """
        score = 50.0  # Neutral default
        try:
            import pytz as _pytz
            ist = _pytz.timezone('Asia/Kolkata')
            today = pd.Timestamp.now(tz=ist).date()

            # A-4: Filter to today's intraday bars only
            if df.index.tz is None:
                df_today = df[df.index.tz_localize(ist).date == today].copy()
            else:
                df_today = df[df.index.tz_convert(ist).map(lambda t: t.date()) == today].copy()

            if len(df_today) < 5:
                self.logger.debug("[VWAP] Fewer than 5 intraday bars — returning neutral score 50")
                return 50.0

            # Calculate intraday VWAP
            typical_price = (df_today['High'] + df_today['Low'] + df_today['Close']) / 3
            cum_tp_vol = (typical_price * df_today['Volume']).cumsum()
            cum_vol = df_today['Volume'].cumsum()
            vwap = cum_tp_vol / cum_vol

            curr_price = df_today['Close'].iloc[-1]
            curr_vwap  = vwap.iloc[-1]

            if direction == 'CE':
                if curr_price > curr_vwap:
                    score = 85  # Price above VWAP — bullish confirmation
                else:
                    score = 50  # Price below VWAP — fighting institutional flow (neutralized)
            elif direction == 'PE':
                if curr_price < curr_vwap:
                    score = 85  # Price below VWAP — bearish confirmation
                else:
                    score = 50  # Price above VWAP — fighting institutional flow (neutralized)

        except Exception as e:
            self.logger.warning(f"VWAP evaluation failed: {e}")

        return min(max(score, 0.0), 100.0)

    def validate_trade_setup(self, signal: Dict, df: pd.DataFrame, context: Dict) -> Tuple[bool, float, str]:
        """
        Main entry point to validate a trade.
        Returns: (is_valid, score, reason)
        """
        score, details = self.calculate_confluence_score(signal, df, context)
        
        # Dynamic threshold based on regime (Optimized for participation)
        threshold = self.min_score
        regime = context.get('regime', 'unknown')
        if regime == 'choppy':
            threshold -= 5 # Increased discount
        elif regime == 'sideways':
            threshold -= 10 # Increased discount
        elif regime in ['trending_up', 'trending_down']:
            threshold -= 15 # Sharply lowered to capture early moves
        
        # Fix 8: VWAP Hard Block in sideways/choppy regimes (relaxed)
        # When price is far from VWAP in a range-bound market, mean-reversion dominates.
        # Only hard-block if VWAP score is VERY low (<15). Otherwise, apply heavy penalty.
        vwap_score = details.get('vwap_score', 50)
        if vwap_score <= 15 and regime in ['sideways', 'choppy']:
            # Hard block only for extreme VWAP divergence
            reason = f"VWAP divergence BLOCKED in {regime} regime | VWAP:{vwap_score} | Score:{score:.1f}"
            self.logger.info(f"Hard block: {reason}")
            return False, score, reason
        elif vwap_score <= 20 and regime in ['sideways', 'choppy']:
            # Apply heavy penalty instead of hard block (allows borderline cases if other factors strong)
            score -= 15
            self.logger.info(f"VWAP penalty applied: -15 points in {regime} regime (VWAP:{vwap_score})")

        # Dynamically lower threshold slightly for trending regimes to catch early moves
        if regime in ['trending_up', 'trending_down'] and details.get('early_reversal_boost'):
            threshold -= 15  # Drop threshold further because early reversals are mathematically capped by trailing indicators
            self.logger.info(f"Threshold sharply relaxed to {threshold} due to valid early reversal in trending regime")
            
        # Buffer Zone Logic: Approve if within 3 points and 200-EMA trend is aligned
        is_valid = score >= threshold
        
        if not is_valid and (threshold - score) <= 3.0:
            # Check 200-EMA alignment for buffer approval
            # Simple check: is price above 200-EMA for CE, or below for PE?
            # We use trend_score as a proxy (100 means full alignment)
            if details.get('trend_score', 0) >= 80:
                is_valid = True
                self.logger.info(f"Buffer Zone Approval: {score:.1f} is near threshold {threshold} and trend-aligned.")
        
        reason = f"Score {score:.1f}/{threshold} | " + \
                 f"Trend:{details.get('trend_score')} Mom:{details.get('momentum_score')} " + \
                 f"Vol:{details.get('volume_score')} MTF:{details.get('mtf_score')} VWAP:{details.get('vwap_score')}"
                 
        return is_valid, score, reason
