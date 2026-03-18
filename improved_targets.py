"""
Improved Target Labeling for Intraday Options Trading ML Model.

Current problem:
    Simple forward returns with 0.15% threshold → binary label.
    This is noisy because:
    1. A 0.15% up move followed by a 2% crash is labeled "bullish"
    2. No distinction between tradeable moves and noise
    3. No neutral/no-trade class

Solution:
    ATR-based triple-barrier labeling that simulates real trade outcomes.
    Labels reflect which exit would be hit first: TP, SL, or timeout.
    
Usage:
    from improved_targets import calculate_improved_targets
    targets = calculate_improved_targets(df, atr_tp_mult=1.5, atr_sl_mult=1.0, horizon=12)
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger('ImprovedTargets')


def calculate_improved_targets(
    df: pd.DataFrame,
    atr_tp_mult: float = 1.5,
    atr_sl_mult: float = 1.0,
    horizon: int = 12,
    atr_period: int = 14,
    mode: str = 'binary'  # 'binary' or 'three_class'
) -> pd.Series:
    """
    ATR-based triple-barrier target labeling.
    
    For each bar, simulates a hypothetical long trade:
    - TP = entry + atr_tp_mult × ATR  
    - SL = entry - atr_sl_mult × ATR
    - If TP hit first within horizon → label 1 (bullish)
    - If SL hit first within horizon → label 0 (bearish)
    - If neither hit → label NaN (binary) or 2 (three_class = neutral)
    
    Also evaluates the reverse (short) to determine true bearish signals.
    
    Args:
        df: DataFrame with OHLCV columns
        atr_tp_mult: ATR multiplier for take-profit
        atr_sl_mult: ATR multiplier for stop-loss  
        horizon: Number of bars to look ahead
        atr_period: ATR calculation period
        mode: 'binary' (1/0/NaN) or 'three_class' (1/0/2)
    
    Returns:
        Series with labels aligned to df.index
    """
    if df.empty or len(df) < atr_period + horizon + 5:
        logger.warning("Insufficient data for improved target calculation.")
        return pd.Series(dtype=float)
    
    # Calculate ATR
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()
    
    labels = pd.Series(np.nan, index=df.index, name='target')
    
    close_vals = close.values
    high_vals = high.values
    low_vals = low.values
    atr_vals = atr.values
    
    for i in range(atr_period, len(df) - horizon):
        entry = close_vals[i]
        current_atr = atr_vals[i]
        
        if np.isnan(current_atr) or current_atr <= 0:
            continue
        
        tp_long = entry + atr_tp_mult * current_atr
        sl_long = entry - atr_sl_mult * current_atr
        
        tp_short = entry - atr_tp_mult * current_atr
        sl_short = entry + atr_sl_mult * current_atr
        
        long_result = 0  # 0 = neither, 1 = TP hit, -1 = SL hit
        short_result = 0
        
        # Simulate forward bars
        for j in range(1, horizon + 1):
            idx = i + j
            if idx >= len(df):
                break
            
            bar_high = high_vals[idx]
            bar_low = low_vals[idx]
            
            # Long trade simulation
            if long_result == 0:
                if bar_high >= tp_long:
                    long_result = 1  # TP hit
                elif bar_low <= sl_long:
                    long_result = -1  # SL hit
            
            # Short trade simulation
            if short_result == 0:
                if bar_low <= tp_short:
                    short_result = 1  # TP hit (short)
                elif bar_high >= sl_short:
                    short_result = -1  # SL hit (short)
            
            # Both resolved
            if long_result != 0 and short_result != 0:
                break
        
        # Label assignment
        if long_result == 1 and short_result != 1:
            # Long TP hit, short didn't → bullish
            labels.iloc[i] = 1.0
        elif short_result == 1 and long_result != 1:
            # Short TP hit, long didn't → bearish
            labels.iloc[i] = 0.0
        elif long_result == 1 and short_result == 1:
            # Both TPs hit → choppy, label as neutral
            if mode == 'three_class':
                labels.iloc[i] = 2.0
            # else: NaN (skip in binary mode)
        elif long_result == -1 and short_result == -1:
            # Both SLs hit → extremely choppy
            if mode == 'three_class':
                labels.iloc[i] = 2.0
        else:
            # Neither TP hit → no clear move
            if mode == 'three_class':
                labels.iloc[i] = 2.0
            # else: NaN (skip in binary mode)
    
    # Stats
    valid = labels.dropna()
    if len(valid) > 0:
        if mode == 'three_class':
            bull_pct = (valid == 1.0).mean() * 100
            bear_pct = (valid == 0.0).mean() * 100
            neutral_pct = (valid == 2.0).mean() * 100
            logger.info(
                f"Improved targets (3-class): {len(valid)} samples | "
                f"Bull: {bull_pct:.1f}% | Bear: {bear_pct:.1f}% | Neutral: {neutral_pct:.1f}%"
            )
        else:
            bull_pct = (valid == 1.0).mean() * 100
            bear_pct = (valid == 0.0).mean() * 100
            logger.info(
                f"Improved targets (binary): {len(valid)} samples | "
                f"Bull: {bull_pct:.1f}% | Bear: {bear_pct:.1f}% | "
                f"Dropped (neutral): {len(labels) - len(valid)}"
            )
    
    return labels


def calculate_regime_aware_targets(
    df: pd.DataFrame,
    regime_series: pd.Series,
    horizon: int = 12,
    atr_period: int = 14
) -> pd.Series:
    """
    Regime-aware target labeling with adaptive ATR multipliers.
    
    In trending regimes: wider TP (let winners run), normal SL
    In sideways regimes: tight TP (scalp), tight SL
    In volatile regimes: wider both
    
    Args:
        df: DataFrame with OHLCV
        regime_series: Series with regime labels aligned to df.index
        horizon: Bars to look ahead
        atr_period: ATR period
    
    Returns:
        Series with labels
    """
    regime_params = {
        'trending_up':    {'tp_mult': 2.0, 'sl_mult': 1.0},
        'trending_down':  {'tp_mult': 2.0, 'sl_mult': 1.0},
        'sideways':       {'tp_mult': 1.0, 'sl_mult': 0.8},
        'choppy':         {'tp_mult': 1.0, 'sl_mult': 0.8},
        'volatile':       {'tp_mult': 2.5, 'sl_mult': 1.5},
        'unknown':        {'tp_mult': 1.5, 'sl_mult': 1.0},
    }
    
    # Calculate ATR
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - df['Close'].shift(1)).abs(),
        (df['Low'] - df['Close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()
    
    labels = pd.Series(np.nan, index=df.index, name='target')
    
    for i in range(atr_period, len(df) - horizon):
        regime = regime_series.iloc[i] if i < len(regime_series) else 'unknown'
        params = regime_params.get(regime, regime_params['unknown'])
        
        entry = df['Close'].iloc[i]
        current_atr = atr.iloc[i]
        
        if np.isnan(current_atr) or current_atr <= 0:
            continue
        
        tp_dist = params['tp_mult'] * current_atr
        sl_dist = params['sl_mult'] * current_atr
        
        long_hit = short_hit = None
        
        for j in range(1, horizon + 1):
            idx = i + j
            if idx >= len(df):
                break
            
            if long_hit is None:
                if df['High'].iloc[idx] >= entry + tp_dist:
                    long_hit = 'tp'
                elif df['Low'].iloc[idx] <= entry - sl_dist:
                    long_hit = 'sl'
            
            if short_hit is None:
                if df['Low'].iloc[idx] <= entry - tp_dist:
                    short_hit = 'tp'
                elif df['High'].iloc[idx] >= entry + sl_dist:
                    short_hit = 'sl'
        
        if long_hit == 'tp' and short_hit != 'tp':
            labels.iloc[i] = 1.0
        elif short_hit == 'tp' and long_hit != 'tp':
            labels.iloc[i] = 0.0
        # else: NaN (no clear signal)
    
    return labels
