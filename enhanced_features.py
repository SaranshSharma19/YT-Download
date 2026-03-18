"""
Enhanced Feature Engineering for Intraday Options Trading.

Adds high-impact features to improve ML model signal quality.
These features capture intraday microstructure, mean-reversion, 
momentum expansion, and volume dynamics that standard TA libraries miss.

Usage:
    from enhanced_features import add_enhanced_features
    df = add_enhanced_features(df)
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger('EnhancedFeatures')


def add_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add high-impact features to an OHLCV DataFrame.
    
    Expected columns: Open, High, Low, Close, Volume
    All features are computed in-place to avoid copies.
    
    Returns:
        DataFrame with additional feature columns
    """
    if df.empty or len(df) < 20:
        return df
    
    # Defragment the dataframe to avoid PerformanceWarning
    df = df.copy()
    
    try:
        # ============================================================
        # 1. Distance from Day High/Low (intraday reversion signal)
        # ============================================================
        # How far is the current price from the session extremes?
        # Near high → overextended (bearish signal), near low → oversold (bullish)
        session_high = df['High'].expanding().max()
        session_low = df['Low'].expanding().min()
        session_range = session_high - session_low
        
        df['dist_from_high_pct'] = np.where(
            session_range > 0,
            (session_high - df['Close']) / session_range * 100,
            50.0
        )
        df['dist_from_low_pct'] = np.where(
            session_range > 0,
            (df['Close'] - session_low) / session_range * 100,
            50.0
        )
        
        # ============================================================
        # 2. VWAP Deviation % (institutional flow alignment)
        # ============================================================
        # Distance from VWAP as percentage — strong mean-reversion anchor
        cum_vol = df['Volume'].cumsum()
        cum_vwap = (df['Close'] * df['Volume']).cumsum()
        vwap = cum_vwap / cum_vol
        vwap = vwap.replace([np.inf, -np.inf], np.nan).fillna(df['Close'])
        
        df['vwap_deviation_pct'] = ((df['Close'] - vwap) / vwap * 100).fillna(0)
        df['vwap_distance_abs'] = abs(df['vwap_deviation_pct'])
        
        # ============================================================
        # 3. Opening Range Breakout Context
        # ============================================================
        # The first 3 bars (15 min) define the opening range.
        # Breakout above = bullish, below = bearish, inside = choppy.
        if len(df) >= 3:
            opening_high = df['High'].iloc[:3].max()
            opening_low = df['Low'].iloc[:3].min()
            opening_range = opening_high - opening_low
            
            df['orb_position'] = np.where(
                df['Close'] > opening_high, 1.0,      # Above opening range
                np.where(
                    df['Close'] < opening_low, -1.0,   # Below opening range
                    0.0                                  # Inside opening range
                )
            )
            df['orb_distance_pct'] = np.where(
                opening_range > 0,
                (df['Close'] - (opening_high + opening_low) / 2) / opening_range * 100,
                0.0
            )
        
        # ============================================================
        # 4. Volatility Contraction → Expansion (squeeze detection)
        # ============================================================
        # Bollinger Band width contraction precedes breakout moves.
        # Narrow BB = coiled spring, expansion = breakout in progress.
        bb_period = 20
        sma = df['Close'].rolling(bb_period).mean()
        std = df['Close'].rolling(bb_period).std()
        bb_upper = sma + 2 * std
        bb_lower = sma - 2 * std
        
        bb_width = ((bb_upper - bb_lower) / sma * 100).fillna(0)
        df['bb_width'] = bb_width
        df['bb_width_percentile'] = bb_width.rolling(50, min_periods=10).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        ).fillna(0.5)
        
        # Squeeze: BB width in bottom 20th percentile
        df['volatility_squeeze'] = (df['bb_width_percentile'] < 0.20).astype(int)
        
        # ============================================================
        # 5. Volume Surge vs Average (smart money footprint)
        # ============================================================
        vol_ma20 = df['Volume'].rolling(20).mean()
        df['volume_surge_ratio'] = (df['Volume'] / vol_ma20).fillna(1.0)
        df['volume_surge_flag'] = (df['volume_surge_ratio'] > 2.0).astype(int)
        
        # Volume-price divergence: high volume but small close change = absorption
        close_change_pct = df['Close'].pct_change().abs() * 100
        df['volume_price_divergence'] = np.where(
            close_change_pct > 0,
            df['volume_surge_ratio'] / (close_change_pct + 0.01),
            0.0
        )
        
        # ============================================================
        # 6. Candle Range Expansion (breakout momentum)
        # ============================================================
        candle_range = df['High'] - df['Low']
        avg_range = candle_range.rolling(10).mean()
        df['range_expansion'] = (candle_range / avg_range).fillna(1.0)
        
        # Wide range + close near high = strong bullish bar
        df['candle_strength'] = np.where(
            candle_range > 0,
            (df['Close'] - df['Low']) / candle_range,
            0.5
        )
        
        # ============================================================
        # 7. Intraday Momentum Score (composite)
        # ============================================================
        # Combines multiple signals into a single momentum reading
        df['intraday_momentum'] = (
            df['vwap_deviation_pct'].clip(-3, 3) / 3 * 0.3 +
            (df['dist_from_low_pct'] / 100 - 0.5) * 0.3 +
            df.get('orb_position', pd.Series(0, index=df.index)) * 0.2 +
            (df['candle_strength'] - 0.5) * 0.2
        ).fillna(0)
        
        # ============================================================
        # 8. Time-of-Day Features (session dynamics)
        # ============================================================
        if hasattr(df.index, 'hour'):
            df['hour_of_day'] = df.index.hour
            df['minutes_since_open'] = (df.index.hour - 9) * 60 + df.index.minute - 15
            df['minutes_since_open'] = df['minutes_since_open'].clip(lower=0)
            # Normalized session progress (0 = open, 1 = close)
            df['session_progress'] = (df['minutes_since_open'] / 375).clip(0, 1)
        
        logger.info(f"Enhanced features added. New columns: {len([c for c in df.columns if c.startswith(('dist_', 'vwap_d', 'orb_', 'bb_', 'volatility_sq', 'volume_su', 'volume_pr', 'range_', 'candle_', 'intraday_', 'hour_', 'minutes_', 'session_'))])}")
        
    except Exception as e:
        logger.warning(f"Enhanced feature creation failed: {e}")
    
    return df
