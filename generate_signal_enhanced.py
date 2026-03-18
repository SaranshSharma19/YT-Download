import logging
import pandas as pd
import numpy as np
from typing import Optional, Dict
from datetime import datetime, timedelta
# Attempt to import Config, handling both potential locations
try:
    from config import Config
except ImportError:
    try:
        from trading import Config
    except ImportError:
        pass  # Will likely fail at runtime if Config is used but not found

# Enhanced Signal Generation Method for TradingBot

try:
    from regime_detector import RegimeDetector
    from tradingview_validator import TradingViewValidator
    from signal_validator import SignalValidator
except ImportError as e:
    print(f"Warning: Could not import enhanced modules: {e}")
    print("Some features may be limited. Ensure all module files are in the same directory.")
    RiskManager = None
    PerformanceTracker = None
    RegimeDetector = None
    TradingViewValidator = None
    SignalValidator = None

def generate_signal_enhanced(self, name: str, symbol: str) -> Optional[Dict]:
    """
    Generate trading signal with enhanced features:
    - Regime detection and adaptive parameters
    - Simplified confirmation filters (3 core filters)
    - Risk management integration
    - TradingView chart validation
    - Performance tracking
    """
    try:
        model_pack = self.models.get(name)
        if not model_pack:
            self.logger.warning(f"No trained model available for {name}. Cannot generate signal.")
            return None

        # 1. Fetch latest data
        fetch_period = Config.DATA_PERIOD_SIGNAL
        self.logger.debug(f"Fetching data for signal generation for {name} ({symbol})")
        raw_df = self.data_fetcher.fetch_data(symbol, period=fetch_period)

        if raw_df.empty or len(raw_df) < max(Config.LOOKBACK_WINDOWS) + 10:
            self.logger.warning(f"Insufficient data ({len(raw_df)} bars) for {name}. Skipping signal.")
            self.error_counts[symbol] = self.error_counts.get(symbol, 0) + 1
            return None

        # Reset error count on successful fetch
        self.error_counts[symbol] = 0

        # 2. Detect Market Regime
        regime = 'unknown'
        regime_params = {}
        if self.regime_detector:
            regime_result = self.regime_detector.get_regime_with_params(raw_df)
            regime = regime_result['regime']
            regime_params = regime_result['parameters']
            self.logger.info(f"Detected regime for {name}: {regime} - {regime_params['description']}")
        else:
            # Default parameters if regime detector not available
            regime_params = {
                'confidence_threshold': Config.CONFIDENCE_THRESHOLD,
                'adx_min': Config.ADX_MIN,
                'num_filters': 3
            }

        # 3. Create Features
        self.logger.debug(f"Creating features for {name}")
        feature_df = self.feature_engine.create_features(raw_df.copy())

        if feature_df.empty:
            self.logger.warning(f"Feature creation failed for {name}. Skipping signal.")
            return None

        latest_date = feature_df.index[-1]
        current_features_row = feature_df.iloc[[-1]].copy()
        raw_df.index = pd.to_datetime(raw_df.index)
        current_price = raw_df.loc[latest_date, 'Close']

        # Get ATR for risk calculations
        atr = current_features_row.get('atr', pd.Series([0])).iloc[0]
        if atr == 0 and 'atr_ratio' in current_features_row.columns:
            atr = current_features_row['atr_ratio'].iloc[0] * current_price

        # 4. Prepare features for model
        required_features = model_pack.get("features", [])
        if not required_features:
            self.logger.error(f"Model pack for {name} missing feature list.")
            return None

        current_features_row = current_features_row.reindex(columns=required_features, fill_value=0.0)
        X_pred = current_features_row[required_features]

        # Clean features
        if X_pred.isna().any().any() or np.isinf(X_pred.values).any():
            X_pred = X_pred.fillna(0).replace([np.inf, -np.inf], 0)

        # 5. Generate ML predictions
        base_models = model_pack.get("base_models")
        if not base_models:
            self.logger.error(f"Model pack for {name} missing base models.")
            return None

        meta_features = np.zeros((1, len(base_models)))
        model_names = list(base_models.keys())

        for i, mname in enumerate(model_names):
            model = base_models[mname]
            try:
                if hasattr(model, 'predict_proba'):
                    pred_proba = model.predict_proba(X_pred)
                    if pred_proba.shape == (1, 2):
                        meta_features[0, i] = pred_proba[0, 1]
                    else:
                        meta_features[0, i] = 0.5
                else:
                    return None
            except Exception as e:
                self.logger.warning(f"Prediction failed for {mname}: {e}")
                meta_features[0, i] = 0.5

        # 6. Meta prediction and calibration
        calibrator = model_pack.get("calibrator") or model_pack.get("meta_clf")
        if not calibrator:
            self.logger.error(f"Model pack for {name} missing calibrator.")
            return None

        try:
            probability = calibrator.predict_proba(meta_features)[0, 1]
        except Exception as e:
            self.logger.error(f"Calibrated prediction failed for {name}: {e}")
            return None

        # 7. PROBABILITY & SCORING VALIDATION (Decision Engine)
        if getattr(self, 'signal_validator', None) is None:
             # Initialize if not already done (lazy init)
             self.signal_validator = SignalValidator(config=getattr(Config, 'SCORING_CONFIG', {})) if SignalValidator else None

        # Initial Core Filter: ML Direction
        # Determine potential direction based on raw probability first
        confidence_threshold = regime_params.get('confidence_threshold', Config.CONFIDENCE_THRESHOLD)
        
        direction = None
        if probability >= confidence_threshold:
            direction = "CE"
        elif probability <= (1.0 - confidence_threshold):
            direction = "PE"
        else:
             self.logger.debug(f"Signal rejected: Low ML Confidence {probability:.2f} (Threshold: {confidence_threshold})")
             return None

        # Enhanced Validation
        validation_passed = False
        validation_score = 0
        validation_reason = ""

        if self.signal_validator:
            # Prepare context
            context = {
                'regime': regime,
                'regime_params': regime_params
            }
            
            # Temporary signal dict for validation
            temp_signal = {
                'direction': direction,
                'price': current_price,
                'confidence': probability
            }
            
            # Validate
            is_valid, score, reason = self.signal_validator.validate_trade_setup(
                signal=temp_signal,
                df=feature_df, # Pass feature DF which has indicators
                context=context
            )
            
            validation_passed = is_valid
            validation_score = score
            validation_reason = reason
            
            if not is_valid:
                self.logger.info(f"Signal REJECTED by Validator: {name} {direction} | {reason}")
                return None
            else:
                self.logger.info(f"Signal APPROVED by Validator: {name} {direction} | {reason}")
        
        else:
             # Fallback to simple checks if validator missing
             self.logger.warning("SignalValidator not available. Using fallback logic.")
             validation_passed = True
             validation_score = 50.0

        # 8. Calculate Risk Management Parameters
        position_info = None
        exits = None
        
        if self.risk_manager:
            # Get recent performance stats for Kelly Criterion
            recent_stats = None
            if self.performance_tracker:
                recent_stats = self.performance_tracker.get_recent_stats(
                    index_name=name,
                    lookback_days=30
                )
            
            # Calculate position size
            signal_for_risk = {
                'atr': atr,
                'price': current_price,
                'confidence': probability
            }
            position_info = self.risk_manager.calculate_position_size(
                signal_for_risk,
                recent_stats=recent_stats
            )
            
            if position_info['position_size'] == 0:
                self.logger.warning(f"Risk manager blocked trade: {position_info.get('reason', 'unknown')}")
                return None
            
            # Calculate stop-loss and take-profit levels
            # Pass raw_df for structural stop calculation
            exits = self.risk_manager.calculate_exits(
                entry_price=current_price,
                direction=direction,
                atr=atr,
                df=raw_df # NEW: Structural stops
            )

        # 9. TradingView Chart Validation
        validation_result = None
        if self.tradingview_validator:
            validation_result = self.tradingview_validator.validate_signal(
                signal={
                    'index': name,
                    'direction': direction,
                    'price': current_price,
                    'confidence': probability,
                    'time': datetime.now(Config.TIMEZONE).strftime("%H:%M:%S"),
                    'atr_ratio': atr / current_price if current_price > 0 else 0,
                    'regime': regime
                },
                symbol=symbol,
                interval=Config.INTERVAL
            )

        # 10. Construct Enhanced Signal
        signal = {
            'index': name,
            'direction': direction,
            'price': current_price,
            'confidence': probability,
            'time': datetime.now(Config.TIMEZONE).strftime("%H:%M:%S"),
            'regime': regime,
            'regime_description': regime_params.get('description', 'Unknown'),
            'atr': atr,
            'atr_ratio': atr / current_price if current_price > 0 else 0,
        }

        # Add risk management data
        if position_info:
            signal.update({
                'position_size': position_info['position_size'],
                'risk_amount': position_info['risk_amount'],
                'kelly_adjusted': position_info.get('kelly_adjusted', False)
            })

        if exits:
            signal.update({
                'stop_loss': exits['stop_loss'],
                'take_profit_1': exits['tp1'],
                'take_profit_2': exits['tp2'],
                'risk_reward_ratio': exits['risk_reward_ratio']
            })

        # Add validation data
        if validation_result:
            signal['chart_url'] = validation_result.get('chart_url')
            signal['validation_score'] = validation_result.get('validation_score')

        # 11. Log trade to performance tracker
        if self.performance_tracker and position_info and exits:
            try:
                trade_id = self.performance_tracker.log_trade(
                    index_name=name,
                    direction=direction,
                    entry_price=current_price,
                    position_size=position_info['position_size'],
                    confidence=probability,
                    regime=regime,
                    stop_loss=exits['stop_loss'],
                    take_profit_1=exits['tp1'],
                    take_profit_2=exits['tp2'],
                    validation_score=validation_result.get('validation_score') if validation_result else None,
                    features={
                        'adx': float(current_features_row.get('adx', pd.Series([0])).iloc[0]),
                        'rsi': float(current_features_row.get('momentum_rsi', pd.Series([50])).iloc[0]),
                        'atr_ratio': signal['atr_ratio']
                    }
                )
                signal['trade_id'] = trade_id
            except Exception as e:
                self.logger.warning(f"Failed to log trade: {e}")

        self.logger.info(
            f"✅ SIGNAL GENERATED: {name} {direction} @ ₹{current_price:.2f} | "
            f"Confidence: {probability:.3f} | Regime: {regime} | "
            f"SL: {exits['stop_loss']:.2f} | TP1: {exits['tp1']:.2f}" if exits else ""
        )

        return signal

    except Exception as e:
        self.logger.error(f"Signal generation failed for {name} ({symbol}): {str(e)}")
        import traceback
        self.logger.error(traceback.format_exc())
        return None
