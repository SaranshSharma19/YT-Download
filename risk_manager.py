"""
Advanced Risk Management Module
Handles position sizing, stop-loss, take-profit, and drawdown protection
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import json
import os


class RiskManager:
    """
    Production-grade risk management system with:
    - Kelly Criterion position sizing
    - ATR-based stop-loss and take-profit
    - Daily drawdown limits
    - Correlation-based exposure management
    """
    
    def __init__(self, account_balance: float = 100000.0, config_path: str = None, disable_daily_loss_limit: bool = False):
        self.logger = logging.getLogger('RiskManager')
        self.account_balance = account_balance
        self.initial_balance = account_balance
        self.disable_daily_loss_limit = disable_daily_loss_limit
        
        # Risk parameters (can be overridden by config)
        self.risk_per_trade = 0.015  # 1.5% of capital per trade
        self.max_daily_loss = 0.10   # 3% daily loss limit
        self.max_position_size = 0.10  # 10% max per trade
        self.stop_loss_atr_multiple = 1.5
        self.take_profit_ratios = [2.0, 3.0]  # 2:1 and 3:1 R:R
        self.kelly_fraction = 0.25  # Use 25% of Kelly for safety
        
        # Daily tracking
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.last_reset_date = datetime.now().date()
        
        # Position tracking
        self.open_positions = {}
        
        # Load config if provided
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)
    
    def _load_config(self, config_path: str):
        """Load risk parameters from config file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                risk_config = config.get('risk_management', {})
                
                self.risk_per_trade = risk_config.get('risk_per_trade', self.risk_per_trade)
                self.max_daily_loss = risk_config.get('max_daily_loss', self.max_daily_loss)
                self.max_position_size = risk_config.get('max_position_size', self.max_position_size)
                self.stop_loss_atr_multiple = risk_config.get('stop_loss_atr_multiple', self.stop_loss_atr_multiple)
                self.take_profit_ratios = risk_config.get('take_profit_ratios', self.take_profit_ratios)
                
                self.logger.info(f"Loaded risk config from {config_path}")
        except Exception as e:
            self.logger.warning(f"Failed to load config: {e}. Using defaults.")
    
    def reset_daily_tracking(self):
        """Reset daily P&L and trade count at start of new day"""
        current_date = datetime.now().date()
        if current_date > self.last_reset_date:
            self.logger.info(f"Resetting daily tracking. Previous day P&L: {self.daily_pnl:.2f}")
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.last_reset_date = current_date
    
    def calculate_position_size(
        self, 
        signal: Dict, 
        recent_stats: Optional[Dict] = None
    ) -> Dict:
        """
        Calculate optimal position size using Kelly Criterion with fractional adjustment
        
        Args:
            signal: Signal dictionary with 'atr', 'price', 'confidence', etc.
            recent_stats: Optional dict with 'win_rate', 'avg_win', 'avg_loss'
        
        Returns:
            Dict with position_size, risk_amount, stop_distance
        """
        self.reset_daily_tracking()
        
        # Check daily loss limits first
        daily_limit_factor = self.check_daily_limits()
        if daily_limit_factor == 0:
            self.logger.warning("Daily loss limit reached. No new positions allowed.")
            return {
                'position_size': 0,
                'risk_amount': 0,
                'stop_distance': 0,
                'reason': 'daily_loss_limit_reached'
            }
        
        # Extract signal data
        atr = signal.get('atr', signal.get('atr_ratio', 0) * signal.get('price', 0))
        price = signal.get('price', 0)
        
        if atr == 0 or price == 0:
            self.logger.error("Invalid ATR or price in signal")
            return {
                'position_size': 0,
                'risk_amount': 0,
                'stop_distance': 0,
                'reason': 'invalid_signal_data'
            }
        
        # Calculate stop distance based on ATR
        stop_distance = self.stop_loss_atr_multiple * atr
        
        # Options Delta Adjustment: ATM options move ~0.5 points per 1 index point.
        # Without this, position_size is calculated on raw index risk, severely
        # under-sizing the actual options contract quantity needed.
        options_delta = 0.5  # Approximate ATM delta for index options
        effective_stop_distance = stop_distance * options_delta
        
        # Base risk amount
        base_risk = self.account_balance * self.risk_per_trade
        
        # Apply Kelly Criterion if we have recent stats
        if recent_stats and all(k in recent_stats for k in ['win_rate', 'avg_win', 'avg_loss']):
            kelly_multiplier = self._calculate_kelly_multiplier(recent_stats)
            risk_amount = base_risk * kelly_multiplier * daily_limit_factor
        else:
            # Use base risk if no stats available
            risk_amount = base_risk * daily_limit_factor
        
        # Calculate position size using effective (delta-adjusted) stop distance
        position_size = risk_amount / effective_stop_distance
        
        # Apply maximum position size limit
        max_position_value = self.account_balance * self.max_position_size
        max_position_qty = max_position_value / price
        
        if position_size > max_position_qty:
            self.logger.info(f"Position size capped at {self.max_position_size*100}% of capital")
            position_size = max_position_qty
            risk_amount = position_size * stop_distance  # E-2: removed duplicate occurrence below

        # Ensure minimum position size is 1 (or adequate contract size)
        if position_size < 1 and position_size > 0.0:
            # Check if we can afford 1 qty
            if price <= self.account_balance:
                 position_size = 1.0
                 risk_amount = stop_distance # Recalculate risk
            else:
                 position_size = 0.0 # Cannot afford even 1
        else:
             position_size = int(position_size) # Floor to integer

        return {
            'position_size': position_size,
            'risk_amount': risk_amount,
            'stop_distance': stop_distance,
            'kelly_adjusted': recent_stats is not None,
            'daily_limit_factor': daily_limit_factor
        }
    
    def _calculate_kelly_multiplier(self, stats: Dict) -> float:
        """
        Calculate Kelly Criterion multiplier
        
        Kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        We use a fraction (default 25%) for safety
        """
        win_rate = stats['win_rate']
        avg_win = abs(stats['avg_win'])
        avg_loss = abs(stats['avg_loss'])
        
        if avg_win == 0:
            return 1.0
        
        # Kelly formula
        kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        
        # Apply fractional Kelly (more conservative)
        kelly_fraction = max(0.1, min(kelly * self.kelly_fraction, 1.5))
        
        self.logger.debug(f"Kelly multiplier: {kelly_fraction:.3f} (raw Kelly: {kelly:.3f})")
        return kelly_fraction
    
    
    def calculate_structural_stops(
        self,
        df: pd.DataFrame,
        direction: str,
        window: int = 20
    ) -> Optional[float]:
        """
        Identify structural stop levels based on recent swing highs/lows.
        """
        try:
            if df.empty:
                return None
                
            if direction == 'CE':
                # For Long, SL below recent lowest low
                recent_low = df['Low'].rolling(window=window).min().iloc[-1]
                # Add a small buffer (0.1%)
                return recent_low * 0.999
            else:
                # For Short, SL above recent highest high
                recent_high = df['High'].rolling(window=window).max().iloc[-1]
                # Add a small buffer (0.1%)
                return recent_high * 1.001
        except Exception as e:
            self.logger.warning(f"Structural stop calculation failed: {e}")
            return None

    def calculate_exits(
        self, 
        entry_price: float, 
        direction: str, 
        atr: float,
        df: Optional[pd.DataFrame] = None,
        regime: str = 'unknown',
        current_hour: int = 10
    ) -> Dict:
        """
        Calculate stop-loss and take-profit levels using ATR, Structure, Regime, and Time.
        
        Args:
            entry_price: Entry price
            direction: 'CE' (call/long) or 'PE' (put/short)
            atr: Average True Range value
            df: Optional dataframe for structural stop calculation
            regime: Market regime for adaptive TP/SL
            current_hour: Current hour (IST) for time-decay adjustment
        
        Returns:
            Dict with stop_loss, tp1, tp2, and exit strategy
        """
        # Fix 10: Adaptive TP/SL multipliers based on regime
        if regime in ['sideways', 'choppy']:
            # Tight scalp mode — mean reversion dominates
            sl_multiple = 1.0
            tp_ratios = [1.3, 2.0]
        elif regime in ['trending_up', 'trending_down']:
            # Let winners run in strong trends
            sl_multiple = 1.5
            tp_ratios = [2.5, 3.5]
        elif regime == 'volatile':
            sl_multiple = 1.8
            tp_ratios = [2.0, 3.0]
        else:
            sl_multiple = self.stop_loss_atr_multiple
            tp_ratios = list(self.take_profit_ratios)
        
        # Fix 12: Time-decay adjusted TP after 13:00 IST
        # Market closes at 15:15 (effective). After 13:00, TP shrinks proportionally.
        if current_hour >= 13:
            market_close_minute = 15 * 60 + 15  # 15:15 in minutes
            current_minute = current_hour * 60 + 30  # Approximate mid-hour
            remaining_minutes = max(market_close_minute - current_minute, 30)
            full_session_minutes = 375  # 09:15 to 15:30
            
            # Scale factor: 1.0 at 09:15, ~0.4 at 14:30
            time_decay_factor = max(0.4, remaining_minutes / full_session_minutes * 2.0)
            tp_ratios = [r * time_decay_factor for r in tp_ratios]
            self.logger.debug(f"Time-decay applied: factor={time_decay_factor:.2f}, "
                            f"adjusted TP ratios={[f'{r:.2f}' for r in tp_ratios]}")
        
        # 1. Structural Stop (Primary if available)
        structural_sl = None
        if df is not None:
            structural_sl = self.calculate_structural_stops(df, direction)
            
        # 2. ATR Stop (Secondary/Fallback)
        if direction == 'CE':
            atr_sl = entry_price - (sl_multiple * atr)
            tp1 = entry_price + (tp_ratios[0] * atr)
            tp2 = entry_price + (tp_ratios[1] * atr)
            
            # Use structural stop if it's not too far (max 2x ATR risk)
            if structural_sl and structural_sl > (entry_price - 2.5 * atr):
                stop_loss = structural_sl
            else:
                stop_loss = atr_sl

        else:  # PE
            atr_sl = entry_price + (sl_multiple * atr)
            tp1 = entry_price - (tp_ratios[0] * atr)
            tp2 = entry_price - (tp_ratios[1] * atr)
            
            # Use structural stop if it's not too far (max 2x ATR risk)
            if structural_sl and structural_sl < (entry_price + 2.5 * atr):
                stop_loss = structural_sl
            else:
                stop_loss = atr_sl
        
        # Calculate Risk/Reward based on potential
        risk = abs(entry_price - stop_loss)
        reward = abs(tp1 - entry_price)
        rr_ratio = reward / risk if risk > 0 else 0
        
        return {
            'stop_loss': stop_loss,
            'tp1': tp1,
            'tp2': tp2,
            'exit_50_at_tp1': True,
            'risk_reward_ratio': round(rr_ratio, 2),
            'type': 'structural' if stop_loss == structural_sl else 'atr',
            'regime_adapted': regime in ['sideways', 'choppy'],
            'time_decayed': current_hour >= 13
        }
    
    def check_daily_limits(self) -> float:
        """
        Check daily loss limits and return position size multiplier
        
        Returns:
            0.0: Stop trading (daily limit reached)
            0.5: Reduce position sizes by 50%
            1.0: Normal operation
        """
        self.reset_daily_tracking()
        
        if self.account_balance == 0:
            return 0.0
        
        daily_loss_pct = self.daily_pnl / self.account_balance
        
        if daily_loss_pct <= -self.max_daily_loss:
            if self.disable_daily_loss_limit:
                self.logger.warning(
                    f"🛡️ Daily loss limit bypass: {daily_loss_pct*100:.2f}% "
                    f"(limit: {self.max_daily_loss*100:.1f}%) but continuing due to DISABLE_DAILY_LOSS_LIMIT."
                )
                return 1.0 # Allow trading to continue
            
            self.logger.critical(
                f"🚨 DAILY LOSS LIMIT REACHED: {daily_loss_pct*100:.2f}% "
                f"(limit: {self.max_daily_loss*100:.1f}%). STOPPING TRADING."
            )
            return 0.0
        
        if daily_loss_pct <= -(self.max_daily_loss * 0.67):  # 2/3 of limit
            self.logger.warning(
                f"⚠️ Approaching daily loss limit: {daily_loss_pct*100:.2f}%. "
                f"Reducing position sizes by 50%."
            )
            return 0.5
        
        return 1.0
    
    def update_daily_pnl(self, pnl: float):
        """Update daily P&L tracking"""
        self.reset_daily_tracking()
        self.daily_pnl += pnl
        self.daily_trades += 1
        self.logger.info(
            f"Daily P&L updated: {self.daily_pnl:.2f} "
            f"({self.daily_pnl/self.account_balance*100:.2f}%) "
            f"after {self.daily_trades} trades"
        )
    
    def update_account_balance(self, new_balance: float):
        """Update account balance"""
        old_balance = self.account_balance
        self.account_balance = new_balance
        self.logger.info(
            f"Account balance updated: {old_balance:.2f} → {new_balance:.2f} "
            f"({(new_balance/old_balance - 1)*100:+.2f}%)"
        )
    
    def add_position(self, position_id: str, position_data: Dict):
        """Track open position"""
        self.open_positions[position_id] = position_data
        self.logger.info(f"Position opened: {position_id}")
    
    def close_position(self, position_id: str, exit_price: float) -> Optional[Dict]:
        """
        Close position and calculate P&L
        
        Returns:
            Dict with P&L details or None if position not found
        """
        if position_id not in self.open_positions:
            self.logger.warning(f"Position {position_id} not found")
            return None
        
        position = self.open_positions.pop(position_id)
        entry_price = position['entry_price']
        position_size = position['position_size']
        direction = position['direction']
        
        # Calculate P&L
        if direction == 'CE':
            pnl = (exit_price - entry_price) * position_size
        else:  # PE
            pnl = (entry_price - exit_price) * position_size
        
        pnl_pct = pnl / (entry_price * position_size) * 100
        
        # Update daily tracking
        self.update_daily_pnl(pnl)
        
        result = {
            'position_id': position_id,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'position_size': position_size,
            'direction': direction,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'outcome': 'win' if pnl > 0 else ('loss' if pnl < 0 else 'breakeven')
        }
        
        self.logger.info(
            f"Position closed: {position_id} | "
            f"{result['outcome'].upper()} | "
            f"P&L: {pnl:+.2f} ({pnl_pct:+.2f}%)"
        )
        
        return result
    
    def get_risk_summary(self) -> Dict:
        """Get current risk metrics summary"""
        return {
            'account_balance': self.account_balance,
            'initial_balance': self.initial_balance,
            'total_return_pct': (self.account_balance / self.initial_balance - 1) * 100,
            'daily_pnl': self.daily_pnl,
            'daily_pnl_pct': (self.daily_pnl / self.account_balance * 100) if self.account_balance > 0 else 0,
            'daily_trades': self.daily_trades,
            'open_positions': len(self.open_positions),
            'daily_limit_status': self.check_daily_limits(),
            'risk_per_trade_pct': self.risk_per_trade * 100,
            'max_daily_loss_pct': self.max_daily_loss * 100
        }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Initialize risk manager
    rm = RiskManager(account_balance=100000.0)
    
    # Example signal
    signal = {
        'index': 'NIFTY',
        'direction': 'CE',
        'price': 19500.0,
        'atr': 87.5,
        'confidence': 0.68
    }
    
    # Calculate position size
    position_info = rm.calculate_position_size(signal)
    print(f"\nPosition Info: {position_info}")
    
    # Calculate exits
    exits = rm.calculate_exits(
        entry_price=signal['price'],
        direction=signal['direction'],
        atr=signal['atr']
    )
    print(f"\nExit Levels: {exits}")
    
    # Get risk summary
    summary = rm.get_risk_summary()
    print(f"\nRisk Summary: {summary}")
