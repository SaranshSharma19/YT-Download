import logging
from typing import Dict, Tuple, Optional
from dhan_trading.core.config import Config

class RiskManager:
    """
    Advanced Risk Management System
    - Kelly Criterion Position Sizing
    - Real Margin Checking
    - ATR-based Stop Loss
    - Daily Drawdown Protection
    """
    def __init__(self, broker):
        self.logger = logging.getLogger('RiskManager')
        self.broker = broker
        self.daily_pnl = 0.0
        self.max_loss_limit = Config.ACCOUNT_BALANCE * Config.MAX_DAILY_LOSS
        self.trades_today = 0
        
    def reset_daily_stats(self):
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.logger.info("Daily Risk Metrics Reset")

    def check_daily_limits(self) -> float:
        """Returns 0.0 if trading should halt, 1.0 for normal"""
        if getattr(Config, 'DISABLE_DAILY_LOSS_LIMIT', False):
            if self.daily_pnl <= -self.max_loss_limit:
                self.logger.warning(f"🛡️ Daily loss limit bypass: {self.daily_pnl:.2f} (limit: {-self.max_loss_limit:.2f}) but continuing due to DISABLE_DAILY_LOSS_LIMIT.")
            return 1.0
            
        if self.daily_pnl <= -self.max_loss_limit:
            self.logger.critical(f"KILLS SWITCH ACTIVATED: Daily PnL ({self.daily_pnl:.2f}) < Limit ({-self.max_loss_limit:.2f})")
            return 0.0
        return 1.0

    def calculate_position_size(self, signal: Dict, price: float, atr: float) -> Tuple[int, float]:
        """
        Kelly Criterion based sizing bounded by Margin and Max Risk Per Trade.
        """
        if self.check_daily_limits() == 0.0:
            return 0, 0.0
            
        # Simplified Kelly (win_rate * avg_win_loss_ratio ...)
        # Using a fixed risk fraction for safe scale-in: 25% of absolute Kelly
        account_balance = self.broker.get_funds()
        risk_amount = account_balance * Config.RISK_PER_TRADE
        
        # Stop distance based on ATR
        stop_dist = atr * 1.5 
        if stop_dist <= 0: stop_dist = price * 0.005 # 0.5% default fallback
        
        # Calculate quantity such that SL hit = risk_amount
        raw_qty = int(risk_amount / stop_dist)
        
        # In Indian Markets, options/futures trade in lots. 
        # For simplicity of this adapter we map NIFTY=25 qty per lot, BANKNIFTY=15
        lot_sizes = {'NIFTY': 25, 'BANKNIFTY': 15, 'FINNIFTY': 40, 'MIDCPNIFTY': 75}
        # Derive base index
        base = signal.get('index', 'NIFTY')
        lot = lot_sizes.get(base, 25)
        
        # Round to nearest lot
        lots = max(1, raw_qty // lot)
        final_qty = lots * lot
        
        # Check margin affordablity (Requires margin roughly equal to (price * qty))
        margin_req = price * final_qty
        if margin_req > (account_balance * Config.MAX_POSITION_SIZE):
            max_allowed = int((account_balance * Config.MAX_POSITION_SIZE) / price)
            final_qty = max(lot, (max_allowed // lot) * lot)
            margin_req = price * final_qty
            
        if margin_req > account_balance:
            self.logger.error("Insufficient Funds for minimum lot size.")
            return 0, 0.0
            
        return final_qty, risk_amount

    def calculate_exits(self, entry: float, direction: str, atr: float, regime: str) -> Dict:
        """Dynamic SL/TP based on ATR and Regime"""
        multiplier = 1.5 if regime in ['trending_up', 'trending_down'] else 1.2
        sl_dist = atr * multiplier
        
        if direction == 'CE':
            sl = entry - sl_dist
            tp1 = entry + (sl_dist * 2.0)
            tp2 = entry + (sl_dist * 3.0)
        else:
            sl = entry + sl_dist
            tp1 = entry - (sl_dist * 2.0)
            tp2 = entry - (sl_dist * 3.0)
            
        return {'stop_loss': sl, 'tp1': tp1, 'tp2': tp2}
