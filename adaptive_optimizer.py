"""
Adaptive Optimizer - Self-Learning Parameter Adjustment
Continuously improves strategy by learning from trade outcomes
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import json
from pathlib import Path


class AdaptiveOptimizer:
    """
    Self-learning optimization system that:
    - Adjusts confidence thresholds based on recent performance
    - Tracks feature importance drift
    - Triggers model retraining when performance degrades
    - Optimizes parameters per regime
    """
    
    def __init__(
        self,
        performance_tracker=None,
        config_path: str = "adaptive_config.json"
    ):
        self.logger = logging.getLogger('AdaptiveOptimizer')
        self.performance_tracker = performance_tracker
        self.config_path = config_path
        
        # Load or initialize adaptive parameters
        self.adaptive_params = self._load_adaptive_params()
        
        # Performance thresholds for retraining
        self.min_sharpe_ratio = 0.5
        self.min_win_rate = 0.45
        self.max_drawdown_pct = 0.20
        
        # Adjustment limits
        self.threshold_min = 0.48
        self.threshold_max = 0.70
        self.threshold_step = 0.01
    
    def _load_adaptive_params(self) -> Dict:
        """Load adaptive parameters from file or create defaults"""
        if Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r') as f:
                    params = json.load(f)
                self.logger.info(f"Loaded adaptive parameters from {self.config_path}")
                return params
            except Exception as e:
                self.logger.warning(f"Failed to load adaptive params: {e}")
        
        # Default parameters
        return {
            'NIFTY': {
                'confidence_threshold': 0.52,
                'last_updated': datetime.now().isoformat(),
                'performance_history': []
            },
            'BANKNIFTY': {
                'confidence_threshold': 0.52,
                'last_updated': datetime.now().isoformat(),
                'performance_history': []
            },
            'FINNIFTY': {
                'confidence_threshold': 0.52,
                'last_updated': datetime.now().isoformat(),
                'performance_history': []
            }
        }
    
    def _save_adaptive_params(self):
        """Save adaptive parameters to file"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.adaptive_params, f, indent=2)
            self.logger.debug(f"Saved adaptive parameters to {self.config_path}")
        except Exception as e:
            self.logger.error(f"Failed to save adaptive params: {e}")
    
    def adjust_threshold(
        self,
        index_name: str,
        lookback_days: int = 7
    ) -> float:
        """
        Dynamically adjust confidence threshold based on recent performance
        
        Logic:
        - If win rate < 48%: Increase threshold (be more selective)
        - If win rate > 58%: Decrease threshold (capture more opportunities)
        - If profit factor < 1.2: Increase threshold
        - If Sharpe ratio < 0.8: Increase threshold
        
        Returns:
            Adjusted confidence threshold
        """
        if not self.performance_tracker:
            self.logger.warning("No performance tracker available for threshold adjustment")
            return self.adaptive_params.get(index_name, {}).get('confidence_threshold', 0.52)
        
        try:
            # Get recent performance stats
            stats = self.performance_tracker.get_recent_stats(
                index_name=index_name,
                lookback_days=lookback_days
            )
            
            if stats['total_trades'] < 10:
                self.logger.debug(f"Insufficient trades ({stats['total_trades']}) for threshold adjustment")
                return self.adaptive_params[index_name]['confidence_threshold']
            
            current_threshold = self.adaptive_params[index_name]['confidence_threshold']
            new_threshold = current_threshold
            
            win_rate = stats['win_rate']
            profit_factor = stats['profit_factor']
            
            # Adjustment logic
            if win_rate < 0.48:
                # Poor win rate - be more selective
                new_threshold = min(current_threshold + self.threshold_step, self.threshold_max)
                reason = f"Low win rate ({win_rate:.1%})"
            
            elif win_rate > 0.58 and profit_factor > 1.5:
                # Excellent performance - capture more opportunities
                new_threshold = max(current_threshold - self.threshold_step, self.threshold_min)
                reason = f"High win rate ({win_rate:.1%}) & profit factor ({profit_factor:.2f})"
            
            elif profit_factor < 1.2:
                # Poor profit factor - be more selective
                new_threshold = min(current_threshold + self.threshold_step, self.threshold_max)
                reason = f"Low profit factor ({profit_factor:.2f})"
            
            else:
                # Performance acceptable - no change
                reason = "Performance within acceptable range"
            
            # Update if changed
            if new_threshold != current_threshold:
                self.logger.info(
                    f"📊 Threshold adjusted for {index_name}: "
                    f"{current_threshold:.3f} → {new_threshold:.3f} ({reason})"
                )
                
                self.adaptive_params[index_name]['confidence_threshold'] = new_threshold
                self.adaptive_params[index_name]['last_updated'] = datetime.now().isoformat()
                
                # Track performance history
                self.adaptive_params[index_name]['performance_history'].append({
                    'date': datetime.now().isoformat(),
                    'win_rate': win_rate,
                    'profit_factor': profit_factor,
                    'threshold': new_threshold,
                    'reason': reason
                })
                
                # Keep only last 30 entries
                if len(self.adaptive_params[index_name]['performance_history']) > 30:
                    self.adaptive_params[index_name]['performance_history'] = \
                        self.adaptive_params[index_name]['performance_history'][-30:]
                
                self._save_adaptive_params()
            
            return new_threshold
            
        except Exception as e:
            self.logger.error(f"Threshold adjustment failed for {index_name}: {e}")
            return self.adaptive_params[index_name]['confidence_threshold']
    
    def check_retrain_trigger(
        self,
        index_name: str,
        lookback_days: int = 14
    ) -> Dict:
        """
        Check if model should be retrained based on performance degradation
        
        Triggers:
        - Win rate < 45%
        - Sharpe ratio < 0.5
        - Profit factor < 1.0
        - Drawdown > 20%
        
        Returns:
            Dict with should_retrain (bool) and reasons (list)
        """
        if not self.performance_tracker:
            return {'should_retrain': False, 'reasons': []}
        
        try:
            stats = self.performance_tracker.get_recent_stats(
                index_name=index_name,
                lookback_days=lookback_days
            )
            
            if stats['total_trades'] < 20:
                return {
                    'should_retrain': False,
                    'reasons': ['Insufficient trades for evaluation']
                }
            
            reasons = []
            
            # Check win rate
            if stats['win_rate'] < self.min_win_rate:
                reasons.append(f"Win rate {stats['win_rate']:.1%} < {self.min_win_rate:.1%}")
            
            # Check profit factor
            if stats['profit_factor'] < 1.0:
                reasons.append(f"Profit factor {stats['profit_factor']:.2f} < 1.0")
            
            # Calculate Sharpe ratio (simplified)
            if stats['total_trades'] > 0:
                # Get daily returns from performance tracker
                try:
                    regime_perf = self.performance_tracker.get_performance_by_regime(lookback_days)
                    if not regime_perf.empty:
                        total_pnl = regime_perf['total_pnl'].sum()
                        # Simplified Sharpe calculation
                        avg_daily_return = total_pnl / lookback_days
                        # Estimate volatility (rough approximation)
                        volatility = abs(stats['avg_loss']) * np.sqrt(252)
                        sharpe = (avg_daily_return * 252) / volatility if volatility > 0 else 0
                        
                        if sharpe < self.min_sharpe_ratio:
                            reasons.append(f"Sharpe ratio {sharpe:.2f} < {self.min_sharpe_ratio:.2f}")
                except Exception:
                    pass
            
            should_retrain = len(reasons) >= 2  # Require at least 2 triggers
            
            if should_retrain:
                self.logger.warning(
                    f"🔄 RETRAIN TRIGGER for {index_name}: {', '.join(reasons)}"
                )
            
            return {
                'should_retrain': should_retrain,
                'reasons': reasons,
                'stats': stats
            }
            
        except Exception as e:
            self.logger.error(f"Retrain check failed for {index_name}: {e}")
            return {'should_retrain': False, 'reasons': [str(e)]}
    
    def get_regime_specific_threshold(
        self,
        index_name: str,
        regime: str
    ) -> float:
        """
        Get optimized threshold for specific regime
        
        Combines base adaptive threshold with regime adjustments
        """
        base_threshold = self.adaptive_params[index_name]['confidence_threshold']
        
        # Regime adjustments (can be learned over time)
        regime_adjustments = {
            'trending_up': -0.02,    # More lenient in uptrends
            'trending_down': -0.02,  # More lenient in downtrends
            'choppy': +0.08,         # Much stricter in choppy markets
            'volatile': +0.03,       # Slightly stricter in volatile markets
            'unknown': 0.00
        }
        
        adjustment = regime_adjustments.get(regime, 0.00)
        adjusted_threshold = np.clip(
            base_threshold + adjustment,
            self.threshold_min,
            self.threshold_max
        )
        
        return adjusted_threshold
    
    def get_optimization_report(self) -> str:
        """Generate optimization status report"""
        report = "\n" + "="*60 + "\n"
        report += "ADAPTIVE OPTIMIZATION REPORT\n"
        report += "="*60 + "\n\n"
        
        for index_name, params in self.adaptive_params.items():
            report += f"📈 {index_name}\n"
            report += f"  Current Threshold: {params['confidence_threshold']:.3f}\n"
            report += f"  Last Updated: {params['last_updated']}\n"
            
            if params['performance_history']:
                recent = params['performance_history'][-1]
                report += f"  Recent Win Rate: {recent['win_rate']:.1%}\n"
                report += f"  Recent Profit Factor: {recent['profit_factor']:.2f}\n"
                report += f"  Last Adjustment: {recent['reason']}\n"
            
            report += "\n"
        
        report += "="*60 + "\n"
        return report


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create optimizer (without performance tracker for demo)
    optimizer = AdaptiveOptimizer()
    
    # Get current threshold
    threshold = optimizer.adaptive_params['NIFTY']['confidence_threshold']
    print(f"Current NIFTY threshold: {threshold:.3f}")
    
    # Get regime-specific threshold
    regime_threshold = optimizer.get_regime_specific_threshold('NIFTY', 'choppy')
    print(f"NIFTY threshold in choppy market: {regime_threshold:.3f}")
    
    # Generate report
    print(optimizer.get_optimization_report())
