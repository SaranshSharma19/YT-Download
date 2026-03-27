"""
Configuration file for enhanced trading system
Centralized parameter management with regime-specific settings
"""

# Trading Configuration
TRADING_CONFIG = {
    "indices": {
        "NIFTY": "^NSEI",
        "BANKNIFTY": "^NSEBANK",
        "FINNIFTY": "NIFTY_FIN_SERVICE.NS"
    },
    
    "timeframe": "5m",
    "prediction_horizon": 12,  # bars (~1 hour for 5m)
    "data_period_training": "60d",
    "data_period_signal": "30d",
}

# Risk Management Configuration
RISK_CONFIG = {
    "account_balance": 100000.0,
    "risk_per_trade": 0.015,  # 1.5% of capital
    "max_daily_loss": 0.10,   # 3% daily loss limit
    "max_position_size": 0.10,  # 10% max per trade
    "stop_loss_atr_multiple": 1.5,
    "take_profit_ratios": [2.0, 3.0],  # 2:1 and 3:1 R:R
    "kelly_fraction": 0.25,  # Use 25% of Kelly for safety
}

# Model Configuration
MODEL_CONFIG = {
    "confidence_threshold": 0.52,
    "retrain_frequency_days": 7,
    "min_training_samples": 1000,
    "cv_splits": 5,
    "force_retrain": False,  # Set to True to retrain on startup
}

# Regime-Specific Parameters (Updated for Precision)
REGIME_CONFIG = {
    "trending_up": {
        "confidence_threshold": 0.50,
        "adx_min": 25,
        "num_filters": 2,
        "description": "Strong uptrend - Trend following preferred"
    },
    "trending_down": {
        "confidence_threshold": 0.50,
        "adx_min": 25,
        "num_filters": 2,
        "description": "Strong downtrend - Trend following preferred"
    },
    "sideways": {
        "confidence_threshold": 0.65, # Reduced from 0.80 for agility
        "adx_min": 0,
        "num_filters": 5, 
        "description": "Sideways market - Sit out or scalp extremes"
    },
    "choppy": {
        "confidence_threshold": 0.65, # Reduced from 0.75
        "adx_min": 0,
        "num_filters": 4,
        "description": "Choppy market - High strictness"
    },
    "volatile": {
        "confidence_threshold": 0.55, # Reduced from 0.60
        "adx_min": 15,
        "num_filters": 3,
        "description": "High volatility - Reduced position sizing"
    },
    "unknown": {
        "confidence_threshold": 0.55, # Reduced from 0.60
        "adx_min": 12,
        "num_filters": 3,
        "description": "Unknown regime - Conservative"
    }
}

# Trade Scoring Configuration (New)
SCORING_CONFIG = {
    "min_score": 60, # Minimum score to enter a trade
    "weights": {
        "trend_alignment": 30,
        "momentum": 25,
        "volume_flow": 20,
        "volatility": 15,
        "market_regime": 10
    }
}

# TradingView Integration
TRADINGVIEW_CONFIG = {
    "enabled": True,
    "auto_open_charts": True,
    "min_validation_score": 60,
    "chart_directory": "./chart_analysis",
}

# Performance Tracking
PERFORMANCE_CONFIG = {
    "enabled": True,
    "database_path": "./trading_performance.db",
    "lookback_days_stats": 30,
    "lookback_days_retrain": 14,
}

# Adaptive Optimization
ADAPTIVE_CONFIG = {
    "enabled": True,
    "adjustment_frequency_days": 7,
    "min_trades_for_adjustment": 10,
    "threshold_min": 0.48,
    "threshold_max": 0.70,
    "threshold_step": 0.01,
}

# Feature Engineering
FEATURE_CONFIG = {
    "lookback_windows": [5, 10, 20, 50, 100],
    "price_movement_threshold": 0.0015,  # 0.15% for intraday
    "min_volume_ratio": 0.5,
    "min_atr_ratio": 0.002,
}

# Confirmation Filters (Simplified)
FILTER_CONFIG = {
    "use_simplified_filters": True,
    "core_filters": [
        "ml_confidence",
        "trend_alignment",
        "volatility_check"
    ],
    # Legacy filters (not used if simplified=True)
    "enable_vwap_confirmation": False,
    "enable_macd_confirmation": False,
    "enable_mtf_confirmation": False,
}

# Market Hours (IST)
MARKET_CONFIG = {
    "timezone": "Asia/Kolkata",
    "market_start": "09:15",
    "market_end": "15:30",
    "use_eod_signals": False,
}

# Logging
LOGGING_CONFIG = {
    "log_level": "INFO",
    "log_directory": "./logs",
    "log_file": "trading_bot.log",
}


def get_config(section: str = None):
    """
    Get configuration section
    
    Args:
        section: Config section name (e.g., 'risk', 'model', 'regime')
                If None, returns all configs
    
    Returns:
        Dict with configuration parameters
    """
    all_configs = {
        'trading': TRADING_CONFIG,
        'risk': RISK_CONFIG,
        'model': MODEL_CONFIG,
        'regime': REGIME_CONFIG,
        'tradingview': TRADINGVIEW_CONFIG,
        'performance': PERFORMANCE_CONFIG,
        'adaptive': ADAPTIVE_CONFIG,
        'feature': FEATURE_CONFIG,
        'filter': FILTER_CONFIG,
        'market': MARKET_CONFIG,
        'logging': LOGGING_CONFIG,
    }
    
    if section:
        return all_configs.get(section.lower(), {})
    return all_configs


def update_config(section: str, key: str, value):
    """
    Update a configuration parameter
    
    Example:
        update_config('risk', 'risk_per_trade', 0.02)
    """
    section_map = {
        'trading': TRADING_CONFIG,
        'risk': RISK_CONFIG,
        'model': MODEL_CONFIG,
        'tradingview': TRADINGVIEW_CONFIG,
        'performance': PERFORMANCE_CONFIG,
        'adaptive': ADAPTIVE_CONFIG,
        'feature': FEATURE_CONFIG,
        'filter': FILTER_CONFIG,
        'market': MARKET_CONFIG,
        'logging': LOGGING_CONFIG,
    }
    
    if section.lower() in section_map:
        section_map[section.lower()][key] = value
        print(f"Updated {section}.{key} = {value}")
    else:
        print(f"Unknown section: {section}")


if __name__ == "__main__":
    # Example usage
    import json
    
    # Get all configs
    all_configs = get_config()
    print(json.dumps(all_configs, indent=2))
    
    # Get specific section
    risk_config = get_config('risk')
    print(f"\nRisk Config: {risk_config}")
    
    # Update parameter
    update_config('risk', 'risk_per_trade', 0.02)
